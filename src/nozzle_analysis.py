"""ADR 0009's numbers: push axis, nozzle recurrences, and the two-wave split.

Compute-intensive companion to ``main.py``: run with ``make nozzle``, not part
of ``make run`` / ``make all``. Reproduces, from the Earth-phased direct-flyby
optimum (ADR 0008's "REAL" solution):

1. The aim-direction result that retired push-past-escape: the push axis (the
   arriving PuffSats' Earth-frame velocity) is ~148 deg from the required
   departure aim, the two Earth hyperbolas close only ~18 deg, and no push
   magnitude along the axis reaches Jupiter at nonzero delivered mass.
2. The honestly-priced head-on nozzle growth recurrences and their mass
   fractions: the one-wave parked architecture (a quadratic in the per-cycle
   growth, because push -> depart -> return spans two cycles) and the
   same-cycle variant (requires the two-wave split).
3. The two-wave split: unpowered Earth-hit roots are ~1 yr apart, so the
   10-30 d split must be bought with a perijove burn on the growth-wave
   portion.

See ``docs/adr/0009-push-axis-retires-push-past-escape.md``. (The original
scratch derivations lived in gitignored ``todos/`` files; this module and the
ADR are self-sufficient without them.)
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.bodies import Earth
from boinor.core.iod import izzo
from scipy.optimize import brentq, least_squares

from src import conic_kernel
from src.astro_constants import PUFFSAT_CYCLE_ORBIT_PERIOD, STD_FUDGE_FACTOR
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.propulsion import payload_mass_ratio
from src.retrograde_return_legs import (
    _assist_chain_params,
    _AssistChainParams,
    _body_state,
    _earth_phase_mismatch,
    _jupiter_assist_body,
    _mean_motion,
    _phased_jovian_flyby,
    _phased_ladder_burn,
)

_YEAR_S = 365.25 * 86400.0
_MU_EARTH = float(Earth.k.to_value(u.km**3 / u.s**2))

#: The Earth-phased direct-flyby optimum recorded in ADR 0008 (the "REAL" row:
#: departure 5.3751 km/s, v_b 59.77, doubling 3.6320 yr before the reversal
#: charge). ``jupiter_lon0`` is a seed; it is re-zeroed against the Earth-phase
#: mismatch at runtime because the recorded value is rounded.
#:
#: Search provenance (recorded per the ADR 0007 lesson -- results carry their
#: bounds): scipy ``differential_evolution`` maximizing the growth rate with a
#: graded Earth-mismatch penalty, over ``jupiter_lon0`` in [0, 2*pi] rad,
#: leg time in [0.6, 6.0] yr, log10-perijove in [0, 2.5] over the floor, and
#: Jovian burn in [0, 12] km/s (the optimum's burn is 0); both bend signs,
#: popsize 60, maxiter 2000, seeds {1, 42, 777}, penalty weights
#: {5, 20, 80, 300}, followed by a brentq projection of ``jupiter_lon0`` onto
#: the mismatch-zero surface. ``phased_geometry()`` replays and verifies the
#: point; re-deriving it means re-running that search over this box.
PHASED_JUPITER_LON0 = 1.637804  # rad
PHASED_LEG_TIME = 1.1374 * u.year
PHASED_LOG_PERIJOVE = 0.6116  # log10 multiples of the perijove floor
PHASED_BEND_SIGN = -1.0


@dataclass(frozen=True)
class ReturnArc:
    """One bend of the Jovian flyby, scored at Earth.

    Attributes:
        mismatch: Signed Earth-phase mismatch at the 1 AU crossing (rad).
        collision_speed: ``v_b`` of the arriving wave (km/s).
        arrival_time: Heliocentric time of the 1 AU crossing (s from epoch).
    """

    mismatch: float
    collision_speed: float
    arrival_time: float


@dataclass(frozen=True)
class PhasedGeometry:
    """The phased optimum's outbound leg and its Earth-phased reference arc.

    Attributes:
        params: The assist-chain parameter block used throughout.
        departure_burn: Oberth departure burn at 200 km (km/s).
        departure_excess: Departure hyperbolic excess in Earth's local
            (tangential, radial) basis (km/s).
        jupiter_lon0: Re-zeroed Jupiter longitude at epoch (rad).
        reference: The Earth-phased unpowered return arc (the nozzle wave).
        window: Earth-Jupiter synodic period (yr).
        cycle: Windowed departure-to-departure cycle (yr).
    """

    params: _AssistChainParams
    departure_burn: float
    departure_excess: Tuple[float, float]
    jupiter_lon0: float
    reference: ReturnArc
    window: float
    cycle: float


@lru_cache(maxsize=1)
def _chain_params() -> _AssistChainParams:
    """Build the standard assist-chain parameter block for the growth loop.

    Returns:
        The parameter block, with the departure state set to the cycle orbit's
        periapsis speed (the closed-cycle convention of ADR 0008).
    """
    v_rf = puffsat_cycle_periapsis_speed()
    return _assist_chain_params(
        target_collision_speed=float(v_rf.to_value(u.km / u.s)) + 0.5,
        cycle_periapsis_speed=v_rf,
    )


def _return_arc(
    params: _AssistChainParams,
    jupiter_lon0: float,
    log_perijove: float,
    burn: float,
    sign: float,
) -> Optional[ReturnArc]:
    """Fly one phased Jovian flyby off the fixed outbound leg and score it.

    Args:
        params: The assist-chain parameter block.
        jupiter_lon0: Jupiter's heliocentric longitude at epoch (rad).
        log_perijove: Perijove radius as log10 multiples of the floor.
        burn: Impulsive perijove burn (km/s, >= 0).
        sign: Bend sign (+1 or -1).

    Returns:
        The scored arc, or None if the flyby or return is infeasible.
    """
    earth = next(b for b in params.bodies if b.symbol == "E")
    jupiter = _jupiter_assist_body(params)
    t_leg = float(PHASED_LEG_TIME.to_value(u.s))
    priced = _phased_ladder_burn(
        0.0,
        [t_leg],
        (earth, jupiter),
        [0.0, jupiter_lon0],
        params,
        powered_nodes=True,
    )
    if priced is None or priced.arrival_excess < 1e-6:
        return None
    perijove = params.flyby.periapsis_floor * float(10.0**log_perijove)
    leg = _phased_jovian_flyby(
        priced.arrival_excess_vector,
        priced.arrival_longitude,
        perijove,
        burn,
        sign,
        params,
    )
    if leg is None:
        return None
    mismatch = _earth_phase_mismatch(
        leg, priced.arrival_longitude, priced.arrival_time, 0.0, params.flyby
    )
    return ReturnArc(
        mismatch=mismatch,
        collision_speed=leg.collision_speed,
        arrival_time=priced.arrival_time + leg.tof,
    )


def phased_geometry() -> PhasedGeometry:
    """Reconstruct the Earth-phased optimum and its departure geometry.

    Re-zeroes ``jupiter_lon0`` against the Earth-phase mismatch (the recorded
    seed is rounded), then replicates the outbound Lambert leg to recover the
    departure excess vector, which ``_LadderPricing`` keeps only as a speed.

    Returns:
        The phased geometry block.

    Raises:
        RuntimeError: If the recorded optimum cannot be reproduced.
    """
    params = _chain_params()

    def mismatch_at(lon: float) -> float:
        arc = _return_arc(params, lon, PHASED_LOG_PERIJOVE, 0.0, PHASED_BEND_SIGN)
        return arc.mismatch if arc is not None else float("nan")

    lon0 = float(
        brentq(
            mismatch_at,
            PHASED_JUPITER_LON0 - 0.05,
            PHASED_JUPITER_LON0 + 0.05,
            xtol=1e-13,
        )
    )
    reference = _return_arc(params, lon0, PHASED_LOG_PERIJOVE, 0.0, PHASED_BEND_SIGN)
    if reference is None or abs(reference.mismatch) > 1e-9:
        raise RuntimeError("could not reproduce the ADR 0008 phased optimum")

    earth = next(b for b in params.bodies if b.symbol == "E")
    jupiter = _jupiter_assist_body(params)
    t_leg = float(PHASED_LEG_TIME.to_value(u.s))
    priced = _phased_ladder_burn(
        0.0, [t_leg], (earth, jupiter), [0.0, lon0], params, powered_nodes=True
    )
    assert priced is not None
    lon_arrival = lon0 + _mean_motion(jupiter) * t_leg
    r0, v_earth = _body_state(earth.orbit_radius, 0.0, earth.v_circ)
    r1, _ = _body_state(jupiter.orbit_radius, lon_arrival, jupiter.v_circ)
    v_depart, _ = izzo(params.flyby.mu_sun, r0, r1, t_leg, 0, True, True, 35, 1e-8)
    excess: npt.NDArray[np.float64] = v_depart - v_earth
    # Earth sits at longitude 0 at epoch: t_hat = (0, 1, 0), r_hat = (1, 0, 0).
    dep_t, dep_r = float(excess[1]), float(excess[0])

    earth_n = params.flyby.v_earth_orbit / params.flyby.r_earth_orbit
    jup_n = params.flyby.v_jupiter_orbit / params.flyby.r_jupiter_orbit
    window = 2.0 * np.pi / (earth_n - jup_n) / _YEAR_S
    trip = reference.arrival_time / _YEAR_S
    coast = float(PUFFSAT_CYCLE_ORBIT_PERIOD.to_value(u.year))
    cycle = float(np.ceil((trip + coast) / window - 1e-9) * window)
    return PhasedGeometry(
        params=params,
        departure_burn=priced.departure_burn,
        departure_excess=(dep_t, dep_r),
        jupiter_lon0=lon0,
        reference=reference,
        window=window,
        cycle=cycle,
    )


@dataclass(frozen=True)
class AimGeometry:
    """The push-axis result that retired push-past-escape (ADR 0009).

    Attributes:
        required_aim_deg: Departure-excess direction from local prograde (deg).
        push_axis_deg: Arriving wave's Earth-frame direction at the crossing,
            from local prograde (deg).
        separation_deg: Angle between the two (deg).
        closeable_deg: Total bend the two Earth hyperbolas can contribute (deg).
        shortfall_deg: ``separation - closeable`` (deg).
        max_aphelion_au: Best aphelion reachable pushing along the push axis at
            the ``v_b`` excess cap, where delivered mass is zero (AU).
    """

    required_aim_deg: float
    push_axis_deg: float
    separation_deg: float
    closeable_deg: float
    shortfall_deg: float
    max_aphelion_au: float


def aim_geometry(geometry: PhasedGeometry) -> AimGeometry:
    """Measure the aim gap between the push axis and the required departure.

    Reconstructs the post-flyby heliocentric state exactly as
    ``_phased_jovian_flyby`` does, decomposes the 1 AU arrival velocity into
    Earth's local (tangential, radial) basis, and compares it with the
    departure excess direction.

    Args:
        geometry: The phased geometry block.

    Returns:
        The aim-geometry summary.
    """
    fl = geometry.params.flyby
    earth = next(b for b in geometry.params.bodies if b.symbol == "E")
    jupiter = _jupiter_assist_body(geometry.params)
    t_leg = float(PHASED_LEG_TIME.to_value(u.s))
    priced = _phased_ladder_burn(
        0.0,
        [t_leg],
        (earth, jupiter),
        [0.0, geometry.jupiter_lon0],
        geometry.params,
        powered_nodes=True,
    )
    assert priced is not None
    perijove = fl.periapsis_floor * float(10.0**PHASED_LOG_PERIJOVE)
    w_in_vec = priced.arrival_excess_vector
    w_in = float(np.linalg.norm(w_in_vec))
    v_peri = float(np.sqrt(w_in * w_in + 2.0 * fl.mu_jupiter / perijove))
    w_out = float(np.sqrt(v_peri**2 - 2.0 * fl.mu_jupiter / perijove))
    ecc_in = conic_kernel.hyperbolic_eccentricity(fl.mu_jupiter, perijove, w_in)
    ecc_out = conic_kernel.hyperbolic_eccentricity(fl.mu_jupiter, perijove, w_out)
    turn = PHASED_BEND_SIGN * conic_kernel.powered_bend_angle(ecc_in, ecc_out)
    ct, st = float(np.cos(turn)), float(np.sin(turn))
    scale = w_out / w_in
    w_out_vec = scale * np.array(
        [w_in_vec[0] * ct - w_in_vec[1] * st, w_in_vec[0] * st + w_in_vec[1] * ct]
    )
    lon = priced.arrival_longitude
    t_hat = np.array([-np.sin(lon), np.cos(lon)])
    r_hat = np.array([np.cos(lon), np.sin(lon)])
    v_t_out = fl.v_jupiter_orbit + float(np.dot(w_out_vec, t_hat))
    v_r_out = float(np.dot(w_out_vec, r_hat))

    # Mirrored prograde twin down to 1 AU, as in _flyby_return_leg.
    v_t_m = -v_t_out
    state = conic_kernel.conic_state_at_radius(
        fl.mu_sun, fl.r_jupiter_orbit, v_t_m, v_r_out
    )
    v_t1, v_r1_mag = conic_kernel.speed_components_at_radius(
        state, fl.mu_sun, fl.r_earth_orbit
    )
    v_r1 = -v_r1_mag
    arr_t, arr_r = -v_t1 - fl.v_earth_orbit, v_r1

    dep_t, dep_r = geometry.departure_excess
    required = float(np.degrees(np.arctan2(dep_r, dep_t)))
    push = float(np.degrees(np.arctan2(arr_r, arr_t)))
    sep = abs(((required - push) + 180.0) % 360.0 - 180.0)

    r_leo = 2.0 * _MU_EARTH / fl.v_esc_leo**2
    w_arr = float(np.hypot(arr_t, arr_r))
    e_arr = conic_kernel.hyperbolic_eccentricity(_MU_EARTH, r_leo, w_arr)
    bend_arr = float(np.degrees(conic_kernel.half_turn_angle(e_arr)))
    w_dep = float(np.hypot(dep_t, dep_r))
    e_dep = conic_kernel.hyperbolic_eccentricity(_MU_EARTH, r_leo, w_dep)
    bend_dep = float(np.degrees(conic_kernel.half_turn_angle(e_dep)))

    # Aphelion of a departure along the push axis at the excess cap (the
    # overtaking stream can never push the payload past v_b).
    v_b = geometry.reference.collision_speed
    s_cap = float(np.sqrt(max(0.0, v_b**2 - fl.v_esc_leo**2)))
    u_t, u_r = arr_t / w_arr, arr_r / w_arr
    vt = fl.v_earth_orbit + s_cap * u_t
    vr = s_cap * u_r
    cap_energy = (vt * vt + vr * vr) / 2.0 - fl.mu_sun / fl.r_earth_orbit
    if cap_energy >= 0.0:
        aph_au = float("inf")
    else:
        a = conic_kernel.semimajor_axis_from_energy(fl.mu_sun, cap_energy)
        h1 = fl.r_earth_orbit * vt
        ecc = conic_kernel.eccentricity_from_energy_and_h(fl.mu_sun, cap_energy, h1)
        aph_au = a * (1.0 + ecc) / fl.r_earth_orbit
    return AimGeometry(
        required_aim_deg=required,
        push_axis_deg=push,
        separation_deg=sep,
        closeable_deg=bend_arr + bend_dep,
        shortfall_deg=sep - bend_arr - bend_dep,
        max_aphelion_au=aph_au,
    )


def apoapsis_reversal_dv(period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD) -> u.Quantity:
    """Delta-v of the 180-degree apoapsis reversal for a parking-orbit period.

    At the near-escape apoapsis the payload crawls, so burning ``-2 v_apo``
    reverses the ellipse and the payload re-arrives at the same periapsis
    moving the opposite way -- the aim reversal that reconciles the push axis
    with the required prograde departure (ADR 0009).

    Args:
        period: Parking-orbit period (astropy Quantity, time).

    Returns:
        The reversal delta-v (astropy Quantity, km/s).
    """
    params = _chain_params()
    r_p = 2.0 * _MU_EARTH / params.flyby.v_esc_leo**2
    t_s = float(period.to_value(u.s))
    a = (_MU_EARTH * (t_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    r_a = 2.0 * a - r_p
    v_apo = float(np.sqrt(_MU_EARTH * (2.0 / r_a - 1.0 / a)))
    return 2.0 * v_apo * u.km / u.s


@lru_cache(maxsize=1)
def _cycle_v_rf() -> float:
    """Closed-cycle 200 km periapsis speed (km/s), cached for the k searches.

    Both nozzle-pricing entry points call this once per evaluation and the
    two-leg sweep evaluates them millions of times, so the orbit construction
    behind it is memoised rather than repeated.

    Returns:
        The periapsis speed in km/s.
    """
    return float(puffsat_cycle_periapsis_speed().to_value(u.km / u.s))


@dataclass(frozen=True)
class NozzlePricing:
    """A head-on-nozzle growth solution with its mass fractions.

    The ``wave_to_*`` pair means: for the parked architecture, the split of
    each *arriving wave* (growth push vs nozzle projectiles); for the
    same-cycle architecture, the split of each *departing batch at Jupiter*
    (powered growth bend vs unpowered projectile bend).

    Attributes:
        slug_ratio: Slug mass per projectile mass, ``k``.
        sigma: Slug consumed per unit final craft mass.
        growth: Net per-cycle growth factor.
        doubling: Doubling time (yr).
        wave_to_growth: Fraction feeding the growth collision.
        wave_to_projectiles: Fraction feeding the nozzle.
        parked_to_craft: Fraction of a parked payload departing as craft.
        parked_to_slug: Fraction consumed as slug.
        parked_to_reversal: Fraction burned as reversal methalox.
        delivered_fraction: ``craft / (craft + slug)`` for the departure burn.
        growth_slug_ratio: ``k`` of the *growth push*, zero when that leg is a
            pusher plate (which carries no slug).
        growth_sigma: Growth-push slug consumed per unit parked mass, zero for
            a pusher plate.
    """

    slug_ratio: float
    sigma: float
    growth: float
    doubling: float
    wave_to_growth: float
    wave_to_projectiles: float
    parked_to_craft: float
    parked_to_slug: float
    parked_to_reversal: float
    delivered_fraction: float
    growth_slug_ratio: float = 0.0
    growth_sigma: float = 0.0

    @property
    def mass_multiplier(self) -> float:
        """Mass at the intercept point per unit mass departing as craft.

        ``(1 + growth_sigma) (1 + sigma)`` -- everything the ground had to loft
        for each kilogram that survives both legs.  A pusher plate consumes
        nothing on the growth push, so its first factor is 1.
        """
        return (1.0 + self.growth_sigma) * (1.0 + self.sigma)


def _sigma(
    k: float,
    recovery: float,
    v_b: float,
    v_rf1: float,
    v_rf2: float,
    jet_efficiency: float = 1.0,
) -> float:
    """Slug consumed per unit final craft mass for the head-on nozzle burn.

    Integrates ``dv/(v_b + v) = recovery * beta(k) * (-dm/m)`` with
    ``beta = (eta_jet sqrt(1+k) - 1)/k`` (Appendix D's impulse per projectile
    kg, divided by the slug spent with it).

    **The two efficiencies act at different places and are not
    interchangeable.**  ``jet_efficiency`` is the paper's ``eta_jet``: it
    scales the *gross* jet, inside ``sqrt(1+k)``, because that is where
    ``eq:ve_general`` puts it -- effective exhaust velocity is what remains
    *after* debiting the incoming momentum.  ``recovery`` scales the net
    impulse after the debit, which is what ``e_2`` was defined to absorb.  The
    ``-1`` is the merged blob's bulk drift, pure momentum conservation, and no
    efficiency of either kind may touch it (Seth, 2026-08-25).

    That distinction has a consequence the outer form cannot express: the
    head-on leg has a **forward-thrust floor** at ``eta_jet = 1/sqrt(1+k)``,
    0.324 at ``k = 8.5``.  Below it the burn produces net backward impulse and
    no amount of slug buys the delta-v, so this returns infinity rather than a
    finite (and meaningless) mass ratio.  ``recovery`` has no such floor,
    which is exactly why the chemistry may not be charged through it.

    Args:
        k: Slug mass per projectile mass.
        recovery: Fraction of the *net* impulse recovered after the momentum
            debit (1.0 = ideal).
        v_b: Nozzle wave's collision speed (km/s).
        v_rf1: Periapsis speed at the start of the burn (km/s).
        v_rf2: Periapsis speed at the end of the burn (km/s).
        jet_efficiency: Fraction of the ideal one-directional *gross* jet the
            nozzle delivers, ``eta_jet``.  The default 1.0 reproduces the
            published ADR 0013/0014/0015 arithmetic exactly; pass
            ``eta_chem * eta_geom`` to charge the frozen-dissociation toll.

    Returns:
        Slug mass per unit final craft mass, or infinity below the
        forward-thrust floor.
    """
    length = float(np.log((v_b + v_rf2) / (v_b + v_rf1)))
    beta = (jet_efficiency * np.sqrt(1.0 + k) - 1.0) / k
    if beta <= 0.0:
        return float("inf")
    return float(np.exp(length / (recovery * beta)) - 1.0)


def _sigma_overtaking(
    k: float,
    recovery: float,
    v_b: float,
    v_ri: float,
    v_rf: float,
    jet_efficiency: float = 1.0,
) -> float:
    """Slug consumed per unit final craft mass for the overtaking growth push.

    The same integration as :func:`_sigma`, at the other end of ADR 0012's
    impulse law: the impactor overtakes, so its bulk drift *adds* and
    ``beta = (1 + sqrt(1+k))/k`` per slug kilogram rather than
    ``(sqrt(1+k) - 1)/k``.  The closing speed falls through the burn instead of
    climbing, because the vehicle is running away from the stream.

    As ``k -> 0`` this reproduces the pusher plate exactly: the parked mass per
    impactor kilogram ``k/sigma`` tends to ``2 * recovery / Lambda``, which is
    :func:`payload_mass_ratio` with ``recovery`` in the role of ``f``.  That
    identity is formal, not physical -- a magnetic nozzle has no plasma to grip
    at vanishing slug (``src/plume_thermal.py``).

    Args:
        k: Slug mass per impactor mass.
        recovery: Fraction of the *net* impulse recovered after the momentum
            bonus (1.0 = ideal).
        v_b: Growth wave's collision speed (km/s).
        v_ri: Vehicle speed at the start of the push (km/s).
        v_rf: Vehicle speed at the end of the push (km/s).
        jet_efficiency: Fraction of the ideal one-directional *gross* jet,
            ``eta_jet``.  Scales ``sqrt(1+k)`` only; the ``+1`` is the
            arriving blob's own momentum and is untouchable.  Default 1.0
            reproduces the published arithmetic.  Unlike the head-on leg this
            has no floor -- the bonus is already forward -- which is why the
            toll levers the two legs by different amounts.

    Returns:
        Slug mass per unit final craft mass.
    """
    length = float(np.log((v_b - v_ri) / (v_b - v_rf)))
    beta = (1.0 + jet_efficiency * np.sqrt(1.0 + k)) / k
    return float(np.exp(length / (recovery * beta)) - 1.0)


def _priced(
    k: float,
    sigma: float,
    growth: float,
    cycle: float,
    wave_to_growth: float,
    reversal_factor: float,
    growth_slug_ratio: float = 0.0,
    growth_sigma: float = 0.0,
) -> NozzlePricing:
    """Assemble a NozzlePricing from a solved recurrence point."""
    doubling = float("inf") if growth <= 1.0 else cycle * np.log(2.0) / np.log(growth)
    return NozzlePricing(
        slug_ratio=k,
        sigma=sigma,
        growth=growth,
        doubling=float(doubling),
        wave_to_growth=wave_to_growth,
        wave_to_projectiles=1.0 - wave_to_growth,
        parked_to_craft=reversal_factor / (1.0 + sigma),
        parked_to_slug=reversal_factor * sigma / (1.0 + sigma),
        parked_to_reversal=1.0 - reversal_factor,
        delivered_fraction=1.0 / (1.0 + sigma),
        growth_slug_ratio=growth_slug_ratio,
        growth_sigma=growth_sigma,
    )


def parked_nozzle(
    collision_speed: float,
    departure_dv: float,
    cycle: float,
    exhaust_speed: float,
    recovery: float = 0.8,
    slug_ratio: Optional[float] = None,
    jet_efficiency: float = 1.0,
) -> NozzlePricing:
    """Price the one-wave parked nozzle architecture.

    Wave N pushes the new payload; wave N+1 departs it. Push -> depart ->
    return spans two cycles, so the steady per-cycle growth ``g`` solves
    ``g^2 (1+sigma)/(r M) + (sigma/k) g = 1``.

    Args:
        collision_speed: ``v_b`` of the (single) Earth-phased wave (km/s).
        departure_dv: Periapsis delta-v of the departure burn (km/s).
        cycle: Departure-to-departure cycle (yr).
        exhaust_speed: Methalox exhaust speed for the reversal charge (km/s).
        recovery: Net impulse recovery fraction, after the momentum debit
            (1.0 = ideal ceiling).
        slug_ratio: Fix ``k``; None optimizes it.
        jet_efficiency: ``eta_jet``, scaling the gross jet inside
            ``sqrt(1+k)``.  Default 1.0 reproduces the published arithmetic.

    Returns:
        The pricing at the given (or optimal) slug ratio.
    """
    v_rf1 = _cycle_v_rf()
    v_rf2 = v_rf1 + departure_dv
    # payload_mass_ratio can hand back a dimensionless Quantity despite its
    # float annotation; coerce at the boundary.
    mass_ratio = float(
        payload_mass_ratio(v_rf=v_rf1 * u.km / u.s, v_b=collision_speed * u.km / u.s)
    )
    rev = float(np.exp(-apoapsis_reversal_dv().to_value(u.km / u.s) / exhaust_speed))

    def solve(k: float) -> Tuple[float, float]:
        sigma = _sigma(k, recovery, collision_speed, v_rf1, v_rf2, jet_efficiency)
        quad = (1.0 + sigma) / (rev * mass_ratio)
        lin = sigma / k
        g = float((-lin + np.sqrt(lin * lin + 4.0 * quad)) / (2.0 * quad))
        return g, sigma

    if slug_ratio is None:
        ks = np.logspace(np.log10(0.05), np.log10(80.0), 3000)
        slug_ratio = float(ks[int(np.argmax([solve(k)[0] for k in ks]))])
    g, sigma = solve(slug_ratio)
    alpha = 1.0 - sigma * g / slug_ratio
    return _priced(slug_ratio, sigma, g, cycle, alpha, rev)


def same_cycle_nozzle(
    growth_collision_speed: float,
    growth_wave_burn: float,
    nozzle_collision_speed: float,
    departure_dv: float,
    cycle: float,
    exhaust_speed: float,
    recovery: float = 0.8,
    slug_ratio: Optional[float] = None,
    fudge: float = STD_FUDGE_FACTOR,
    reversal_period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD,
    growth_slug_ratio: Optional[float] = None,
    growth_recovery: Optional[float] = None,
    jet_efficiency: float = 1.0,
    growth_jet_efficiency: float = 1.0,
) -> NozzlePricing:
    """Price the two-wave same-cycle nozzle architecture.

    The departing batch splits at Jupiter: fraction ``phi`` takes the powered
    bend (the growth wave, arriving ``dt`` early and paying its perijove burn
    as methalox), the rest the unpowered bend (the nozzle wave). The payload a
    growth wave pushes departs the *same* cycle, so ``g`` is linear:
    ``g [ (1+sigma)/(r M d1) + sigma/k ] = 1``.

    Args:
        growth_collision_speed: Wave 1's ``v_b`` after the powered bend (km/s).
        growth_wave_burn: Wave 1's perijove burn (km/s).
        nozzle_collision_speed: Wave 2's (unpowered) ``v_b`` (km/s).
        departure_dv: Periapsis delta-v of the departure burn (km/s).
        cycle: Departure-to-departure cycle (yr).
        exhaust_speed: Methalox exhaust speed (km/s).
        recovery: Nozzle impulse recovery fraction.
        slug_ratio: Fix ``k``; None optimizes it.
        fudge: Elasticity of the growth-push collision, ``f`` in
            ``eq:PuffSat_ratio``.
        reversal_period: Period of the parking orbit the pushed payload coasts
            through, which sizes the apoapsis reversal.  It is also the gap by
            which the growth wave must lead the nozzle wave, so a caller
            varying one must vary the other.
        growth_slug_ratio: Put a magnetic nozzle on the growth push too, at
            this slug ratio, instead of the paper's pusher plate.  Requires
            ``growth_recovery``; ``fudge`` is then unused.
        growth_recovery: Impulse recovery of that growth-push nozzle.
        jet_efficiency: ``eta_jet`` on the head-on departure burn, scaling the
            gross jet inside ``sqrt(1+k)``.  Default 1.0 reproduces the
            published arithmetic; pass ``eta_chem * eta_geom`` to charge the
            frozen-dissociation toll.
        growth_jet_efficiency: The same, for the overtaking growth push.  The
            two legs are different devices at different closing speeds, so
            they carry different tolls -- and a *plate* on the growth push
            owes no chemistry at all, so this is unused when
            ``growth_slug_ratio`` is None.

    Returns:
        The pricing; ``wave_to_growth`` is the batch fraction on the powered
        bend.

    Raises:
        ValueError: If only one of the growth-push nozzle arguments is given.
    """
    if (growth_slug_ratio is None) != (growth_recovery is None):
        raise ValueError("growth_slug_ratio and growth_recovery must be given together")
    v_rf1 = _cycle_v_rf()
    v_rf2 = v_rf1 + departure_dv
    if growth_slug_ratio is None:
        # Pusher plate: nothing is consumed on the growth push, so the parked
        # payload is the whole of what the wave lifted.
        growth_sigma = 0.0
        parked_per_impactor = float(
            payload_mass_ratio(
                v_rf=v_rf1 * u.km / u.s,
                v_b=growth_collision_speed * u.km / u.s,
                fudge_factor=fudge,
            )
        )
    else:
        assert growth_recovery is not None  # narrowed by the pairing check
        growth_sigma = _sigma_overtaking(
            growth_slug_ratio,
            growth_recovery,
            growth_collision_speed,
            0.0,
            v_rf1,
            growth_jet_efficiency,
        )
        parked_per_impactor = growth_slug_ratio / growth_sigma
    mass_ratio = parked_per_impactor
    rev = float(
        np.exp(
            -apoapsis_reversal_dv(reversal_period).to_value(u.km / u.s) / exhaust_speed
        )
    )
    delivered1 = float(np.exp(-growth_wave_burn / exhaust_speed))

    def solve(k: float) -> Tuple[float, float]:
        sigma = _sigma(
            k, recovery, nozzle_collision_speed, v_rf1, v_rf2, jet_efficiency
        )
        g = 1.0 / ((1.0 + sigma) / (rev * mass_ratio * delivered1) + sigma / k)
        return float(g), sigma

    if slug_ratio is None:
        ks = np.logspace(np.log10(0.05), np.log10(80.0), 3000)
        slug_ratio = float(ks[int(np.argmax([solve(k)[0] for k in ks]))])
    g, sigma = solve(slug_ratio)
    phi = 1.0 - (sigma / slug_ratio) * g
    return _priced(
        slug_ratio,
        sigma,
        g,
        cycle,
        phi,
        rev,
        growth_slug_ratio=0.0 if growth_slug_ratio is None else growth_slug_ratio,
        growth_sigma=growth_sigma,
    )


def corrected_incumbent(
    collision_speed: float, departure_dv: float, cycle: float, exhaust_speed: float
) -> Tuple[float, float]:
    """The methalox incumbent with the apoapsis reversal charged.

    Args:
        collision_speed: The Earth-phased wave's ``v_b`` (km/s).
        departure_dv: Methalox departure burn (km/s).
        cycle: Departure-to-departure cycle (yr).
        exhaust_speed: Methalox exhaust speed (km/s).

    Returns:
        ``(growth, doubling_years)``.
    """
    v_rf1 = _cycle_v_rf()
    mass_ratio = float(
        payload_mass_ratio(v_rf=v_rf1 * u.km / u.s, v_b=collision_speed * u.km / u.s)
    )
    rev = float(np.exp(-apoapsis_reversal_dv().to_value(u.km / u.s) / exhaust_speed))
    growth = mass_ratio * float(np.exp(-departure_dv / exhaust_speed)) * rev
    doubling = float("inf") if growth <= 1.0 else cycle * np.log(2.0) / np.log(growth)
    return growth, float(doubling)


def two_wave_unpowered_roots(
    geometry: PhasedGeometry, grid_points: int = 6001
) -> List[Tuple[float, float, float]]:
    """Enumerate every unpowered Earth-hit root of the bend sweep.

    Args:
        geometry: The phased geometry block.
        grid_points: Sweep resolution over log-perijove in [0, 2.5].

    Returns:
        ``(bend_sign, collision_speed, offset_days)`` per root, where the
        offset is relative to the reference (nozzle-wave) arrival.
    """
    roots: List[Tuple[float, float, float]] = []
    for sign in (PHASED_BEND_SIGN, -PHASED_BEND_SIGN):

        def mismatch_at(log_rp: float) -> float:
            arc = _return_arc(geometry.params, geometry.jupiter_lon0, log_rp, 0.0, sign)
            return arc.mismatch if arc is not None else float("nan")

        grid = np.linspace(0.0, 2.5, grid_points)
        vals = [mismatch_at(g) for g in grid]
        for i in range(len(grid) - 1):
            a, b = vals[i], vals[i + 1]
            if np.isnan(a) or np.isnan(b) or a * b > 0.0:
                continue
            if abs(a) > 0.5 or abs(b) > 0.5:  # the +pi/-pi wrap, not a root
                continue
            root = float(brentq(mismatch_at, grid[i], grid[i + 1], xtol=1e-12))
            arc = _return_arc(geometry.params, geometry.jupiter_lon0, root, 0.0, sign)
            if arc is None:
                continue
            offset = (arc.arrival_time - geometry.reference.arrival_time) / 86400.0
            roots.append((sign, arc.collision_speed, offset))
    return roots


def powered_split(
    geometry: PhasedGeometry, dt_days: float, starts: int = 8
) -> Optional[Tuple[float, float]]:
    """Solve the powered bend placing the growth wave ``dt_days`` early.

    A perijove burn can only speed the return up, so the powered portion is
    always the earlier (growth) wave; the nozzle wave stays on the unpowered
    reference bend. Solves ``(log_perijove, burn)`` for Earth-hit AND the
    target offset simultaneously.

    Args:
        geometry: The phased geometry block.
        dt_days: Target arrival-time gap before the reference wave (days).
        starts: Number of log-perijove starting points per bend sign.

    Returns:
        ``(burn_km_s, collision_speed)`` of the cheapest solution, or None.
    """
    best: Optional[Tuple[float, float]] = None
    for sign in (PHASED_BEND_SIGN, -PHASED_BEND_SIGN):
        for log_rp0 in np.linspace(0.1, 2.2, starts):

            def residuals(x: npt.NDArray[np.float64]) -> List[float]:
                arc = _return_arc(
                    geometry.params,
                    geometry.jupiter_lon0,
                    float(x[0]),
                    abs(float(x[1])),
                    sign,
                )
                if arc is None:
                    return [10.0, 10.0]
                dt = (geometry.reference.arrival_time - arc.arrival_time) / 86400.0
                return [arc.mismatch, (dt - dt_days) / 100.0]

            try:
                sol = least_squares(
                    residuals,
                    [float(log_rp0), 0.5],
                    bounds=([0.0, 0.0], [2.5, 8.0]),
                    xtol=1e-14,
                    ftol=1e-14,
                )
            except ValueError:
                continue
            if sol.cost > 1e-16:
                continue
            arc = _return_arc(
                geometry.params,
                geometry.jupiter_lon0,
                float(sol.x[0]),
                abs(float(sol.x[1])),
                sign,
            )
            if arc is None:
                continue
            burn = abs(float(sol.x[1]))
            if best is None or burn < best[0]:
                best = (burn, arc.collision_speed)
    return best


def _print_pricing(label: str, pricing: NozzlePricing, wave_names: str) -> None:
    """Print one pricing row with its mass fractions."""
    print(f"  {label}")
    print(
        f"    k* {pricing.slug_ratio:.2f}  sigma {pricing.sigma:.4f}  "
        f"growth {pricing.growth:.4f}  doubling {pricing.doubling:.4f} yr"
    )
    print(
        f"    {wave_names}: {pricing.wave_to_growth:.4f} growth | "
        f"{pricing.wave_to_projectiles:.4f} projectiles"
    )
    print(
        f"    parked payload: craft {pricing.parked_to_craft:.4f} | "
        f"slug {pricing.parked_to_slug:.4f} | "
        f"reversal {pricing.parked_to_reversal:.4f}   "
        f"(delivered {pricing.delivered_fraction:.4f})"
    )


def main() -> None:
    """Print every ADR 0009 number: aim, recurrences, fractions, split."""
    geometry = phased_geometry()
    fl = geometry.params.flyby
    v_b = geometry.reference.collision_speed
    dv = geometry.departure_burn
    cycle = geometry.cycle
    ve = fl.exhaust_speed
    print("=== Phased optimum (ADR 0008 'REAL' row, reproduced) ===")
    print(
        f"  v_b {v_b:.4f} km/s  departure {dv:.4f} km/s  "
        f"cycle {cycle:.4f} yr (window {geometry.window:.4f})  "
        f"mismatch {geometry.reference.mismatch:.1e} rad"
    )
    rev_dv = apoapsis_reversal_dv()
    print(f"  apoapsis reversal (20 d orbit): {rev_dv.to_value(u.m/u.s):.1f} m/s\n")

    aim = aim_geometry(geometry)
    print("=== 1. Aim: why push-past-escape is retired ===")
    print(f"  required departure aim {aim.required_aim_deg:+.1f} deg from prograde")
    print(f"  push axis              {aim.push_axis_deg:+.1f} deg from prograde")
    print(
        f"  separation {aim.separation_deg:.1f} deg, hyperbolas close "
        f"{aim.closeable_deg:.1f} deg -> shortfall {aim.shortfall_deg:.1f} deg"
    )
    print(
        f"  aphelion along the push axis at the v_b cap (delivered mass -> 0): "
        f"{aim.max_aphelion_au:.3f} AU\n"
    )

    growth_inc, doubling_inc = corrected_incumbent(v_b, dv, cycle, ve)
    print("=== 2. Recurrences and mass fractions ===")
    print(
        f"  incumbent (methalox, reversal charged): growth {growth_inc:.4f}  "
        f"doubling {doubling_inc:.4f} yr"
    )
    for recovery, tag in ((1.0, "ideal"), (0.8, "derated f=0.8")):
        pricing = parked_nozzle(v_b, dv, cycle, ve, recovery=recovery)
        _print_pricing(f"parked one-wave, {tag}:", pricing, "arriving wave")
    print()

    print("=== 3. Two-wave split ===")
    roots = two_wave_unpowered_roots(geometry)
    for sign, root_vb, offset in roots:
        print(
            f"  unpowered root: sign {sign:+.0f}  v_b {root_vb:8.4f}  "
            f"offset {offset:+9.2f} d"
        )
    close = [r for r in roots if 5.0 <= -r[2] <= 90.0]
    print(
        f"  -> unpowered pairs 5-90 d apart: {len(close)} "
        "(roots are ~1 yr apart; the split must be bought)\n"
    )
    for dt in (10.0, 20.0, 30.0):
        found = powered_split(geometry, dt)
        if found is None:
            print(f"  dt {dt:.0f} d: no powered solution found")
            continue
        burn, vb1 = found
        print(
            f"  dt {dt:.0f} d: perijove burn {burn:.4f} km/s, "
            f"growth-wave v_b {vb1:.4f}"
        )
        for recovery, tag in ((1.0, "ideal"), (0.8, "derated f=0.8")):
            pricing = same_cycle_nozzle(
                vb1, burn, v_b, dv, cycle, ve, recovery=recovery
            )
            _print_pricing(f"same-cycle, {tag}:", pricing, "batch split at Jupiter")
        print()


if __name__ == "__main__":
    main()
