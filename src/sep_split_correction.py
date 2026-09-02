"""Who pays for the 20-day split, and whether argon SEP can pay it instead.

Three questions the two-wave split left open, answered against the same real
ephemeris ``two_wave_growth`` flies (ADR 0013's chain, ADR 0011's adaptive
two-or-three-synodic cadence):

1. **What does the split actually cost, cycle by cycle, at the paper's 20-day
   parking orbit?**  ADR 0013 published a chain mean of 0.4641 km/s and left
   the distribution behind it unreported.  It is not a spread around that mean:
   eight of eleven cycles buy their early arrival for under 2 m/s and three pay
   1.4-2.1 km/s.  The expensive ones are **bend-limited**, not speed-limited --
   their growth wave is pinned at the 4,000 km perijove floor and the flyby
   still cannot turn far enough, so the residue is an *angle*, and no perijove
   change buys it.

2. **Should the cadence avoid them?**  ``two_wave_growth``'s policy picks a
   two-synodic return whenever the *nozzle* wave's maneuver proxy stays under
   50 m/s and never looks at what the growth wave will then pay.  Widening it
   to gate on both waves removes the kilometre-per-second tail completely --
   and costs 9% in doubling time, because a 2S->3S fallback spends 1.09 years
   of clock *and* lands a colder wave (``v_b`` 52.6-55.4 km/s against
   60.1-63.3).  Paying the tail is the right call.

3. **Can argon SEP pay it instead of methalox?**  Yes, and it is the largest
   of the three effects: at the repo's existing 2000 s
   (:data:`~src.astro_constants.ARGON_SEP_ISP`) the worst cycle's 2.108 km/s
   costs 10.2% of the wave rather than 43.2%.  Power is not the binding
   constraint -- see :func:`leg_impulse_per_tonne`.  Array *mass* is, and
   :func:`break_even_wave_mass` says where the trade turns over.

Two burns are charged per cycle and both are SEP candidates:

* the **nozzle wave's DSM**, the returning mass's own correction, confined to
  ``r > SEP_DEPLOY_RADIUS`` because the wave needs the inner part of its leg to
  deploy into a spread-out PuffSat stream;
* the **20-day separation**, the growth wave's cost of arriving early,
  unconstrained in place, so its whole trajectory counts -- outbound leg
  included, where the array is nearest the Sun and strongest.

The apoapsis reversal is deliberately *not* moved to argon.  It is a
180-degree turn at the apoapsis of a 20-day Earth parking orbit, not a
multi-year cruise, and low thrust does not do that maneuver; it stays methalox
in every row here.  ``same_cycle_nozzle`` charges one exhaust speed for both,
so :func:`correction_cycles_to_two_wave` scales the interplanetary correction
into the equivalent methalox delta-v ``dv * v_e_methalox / v_e_argon``, which
reproduces argon exactly on the ledger's ``delivered1`` and leaves ``rev``
alone.

Run it with ``make sep-split``; it is not imported by ``src.main`` and is not
part of ``make all``.  See ADR 0026.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time
from boinor.core.iod import izzo
from scipy.integrate import solve_ivp

from src.astro_constants import ARGON_SEP_ISP
from src.propulsion import exhaust_velocity_from_isp
from src.real_orbit_resonance import (
    _DAYS_PER_YEAR,
    _FIXED_TWO_SYNODIC_DAYS,
    _MU_SUN,
    _SECONDS_PER_DAY,
    _heliocentric_state,
    _ManeuverSolution,
    _resonance_epochs,
    _window_maneuver_solutions,
)
from src.two_wave_growth import (
    VE_METHALOX,
    ChainGrowth,
    TwoWaveCycle,
    price_chain,
)

#: Split gap this module reports at: the paper's parking-orbit period
#: (``PUFFSAT_CYCLE_ORBIT_PERIOD``).  ADR 0013 decided 10 days on a 0.7%
#: margin; the constant the rest of the repo carries is 20, and the two waves
#: must be one parking orbit apart, so 20 is what the split is priced at here.
DEFAULT_SPLIT_DAYS = 20.0
#: Study horizon, matching ADR 0013 so the chains are the same chains.
DEFAULT_START = "2026-08-11"
DEFAULT_HORIZON_YEARS = 30.0
#: Maneuver proxy above which a cycle falls back to a three-synodic return.
DEFAULT_THRESHOLD_M_S = 50.0
#: Maneuver architectures priced for each wave.  ``two_wave_growth`` asks only
#: for ``dsm_only``; the powered-flyby cases are included here because their
#: absence looked like an omission, and measuring it is how ADR 0026 retires
#: the suspicion (it is worth 0.06%).
MANEUVER_CASES = ("perijove_only", "dsm_only", "hybrid_50mps")

#: Argon exhaust speed (km/s) from the repo's existing SEP constant.
VE_ARGON = float(exhaust_velocity_from_isp(ARGON_SEP_ISP).to_value(u.km / u.s))
#: Thruster efficiency.  Argon Hall runs 0.45-0.55 at 2000 s, below xenon
#: because argon's higher ionisation cost per unit mass is paid twice over at
#: the same Isp.  0.5 is the middle of that, and every impulse figure below
#: scales linearly in it.
SEP_THRUSTER_EFFICIENCY = 0.5
#: Array power at 1 AU (kW), falling as 1/r^2 -- so 4.00 kW at 5 AU.
SEP_ARRAY_POWER_1AU_KW = 100.0
#: Heliocentric radius (AU) inside which the returning wave must have finished
#: correcting, because it needs that stretch to deploy into a spread stream.
SEP_DEPLOY_RADIUS_AU = 2.0
#: SEP system specific mass (kg per kW at 1 AU), swept.  Covers array, power
#: processing, thrusters and gimbals -- **not** propellant tankage, which is
#: charged separately below because it scales with the burn rather than the
#: power.  The span runs from a flexible roll-out array (~150 W/kg, so ~7
#: kg/kW) with a light PPU, to a conservative rigid-panel system.  These are
#: asserted engineering ranges, not figures calibrated against a flight
#: system, and every wave-mass floor below scales linearly in them.
SEP_SPECIFIC_MASS_KG_PER_KW = (10.0, 15.0, 20.0)
#: Tank and feed-system mass per kg of argon carried.  Argon stores at roughly
#: a third the density of xenon at comparable conditions, so its tankage is the
#: heavier of the two by a wide margin; 0.15 is a mid-range supercritical
#: composite-overwrap figure.
ARGON_TANK_FRACTION = 0.15
#: The design's quoted SEP operating point: characteristic acceleration at
#: 1 AU, and the value quoted at Jupiter.  **The two do not agree.**  Under
#: ``1/r^2`` the first falls to 7.40e-7 at 5.2 AU, not 1e-6; the pair implies
#: 4.47 AU.  Nor does the first agree with a 100 kW array on a 500 t wave,
#: which gives 1.02e-5 -- it implies 196 kW at that wave mass.  Both are kept
#: as stated, because the point of the requirement table is to compare what was
#: claimed against what is needed, and pinned in the tests.
STATED_ACCELERATION_1AU = 2.0e-5
STATED_ACCELERATION_JUPITER = 1.0e-6
#: Thrust of one flight-proven high-power Hall thruster (N), for scale in the
#: requirement table -- a 12.5 kW class unit.  Only a yardstick; nothing
#: computed depends on it.
REFERENCE_THRUSTER_NEWTONS = 0.64
REFERENCE_THRUSTER_KW = 12.5
#: Wave mass the absolute power and array figures are quoted at.  The
#: architecture wants a large payload -- the nozzle has a minimum mass, so a
#: small wave is mostly dry mass -- and 500 t is the scale the design is aiming
#: at.  Nothing dimensionless depends on it; only the megawatt and tonne
#: columns rescale.
REFERENCE_WAVE_TONNES = 500.0
#: The same, per kg of methalox.  Dense propellants in a cryogenic stage, so
#: much lighter per kg carried -- but not zero, and charging argon's tankage
#: while ignoring methalox's would flatter the comparison.
METHALOX_TANK_FRACTION = 0.08

_AU_KM = 149_597_870.7
#: ``(5.2 AU)^2``: how far the array's output falls between Earth and Jupiter.
#: Named because both the stated operating point and every Jupiter-side figure
#: below divide by it, and the design's own quoted pair does not.
_JUPITER_INVERSE_SQUARE = 5.2**2
_LAMBERT_ITERATIONS = 350
_LAMBERT_RTOL = 1e-8
#: Samples along a propagated leg for the impulse quadrature.  The integrand
#: ``1/r^2`` is smooth and monotone over each leg, so this is far more than the
#: trapezoid needs; it costs nothing next to the propagation.
_LEG_SAMPLES = 20_001


@dataclass(frozen=True)
class CorrectionCycle:
    """One flown cycle with both of its correction burns resolved.

    Attributes:
        index: Position in the flown chain, counting from zero.
        departure_jd: TDB Julian date the batch leaves Earth.
        return_jd: TDB Julian date the nozzle wave reaches Earth, which is also
            the next cycle's departure.
        synodic_multiple: Two or three, whichever the policy selected.
        period_years: Departure-to-departure cycle length.
        forced_by_split: Whether this cycle fell back to three synodic because
            of the *split* cost rather than the nozzle wave's own maneuver.
            Always False under the baseline policy.
        nozzle: The nozzle wave's selected maneuver solution.
        growth: The growth wave's cheapest maneuver solution.
        growth_dsm_only: The growth wave priced on ``dsm_only`` alone, which is
            what ``two_wave_growth`` charges; kept so the powered-flyby cases
            can be scored against it.
        split_days: Gap by which the growth wave leads the nozzle wave.
    """

    index: int
    departure_jd: float
    return_jd: float
    synodic_multiple: int
    period_years: float
    forced_by_split: bool
    nozzle: _ManeuverSolution
    growth: _ManeuverSolution
    growth_dsm_only: _ManeuverSolution
    split_days: float

    @property
    def correction_total(self) -> float:
        """Both correction burns summed (km/s), the ledger's ``delivered1``."""
        return self.nozzle.dsm + self.growth.total_dv

    @property
    def bend_limited(self) -> bool:
        """Whether no powered-flyby solution exists for the growth wave.

        True when the required turn exceeds what even a floor perijove can
        bend, so the residue is an angle and the deep-space burn is the only
        way to pay it.
        """
        return self.growth.case == "dsm_only" and self.growth.total_dv > 0.05


def _cheapest(
    departure_jd: float, arrival_jd: float, cases: Sequence[str] = MANEUVER_CASES
) -> Dict[str, _ManeuverSolution]:
    """Every maneuver architecture that joins two fixed Earth epochs.

    Args:
        departure_jd: TDB Julian date of Earth departure.
        arrival_jd: TDB Julian date the wave must reach Earth.
        cases: Maneuver architectures to price.

    Returns:
        Solutions keyed by case name; a case absent from the mapping admits no
        solution for this window.

    Raises:
        RuntimeError: If the window admits no solution at all.
    """
    solutions = _window_maneuver_solutions(departure_jd, arrival_jd, None, cases=cases)
    if not solutions:
        raise RuntimeError(f"no solution for window {departure_jd}->{arrival_jd}")
    return {solution.case: solution for solution in solutions}


#: Cadence policies this module flies.
#:
#: ``adaptive`` is ``two_wave_growth``'s own: three-synodic whenever the
#: *nozzle* wave's maneuver proxy exceeds the threshold.  ``split_aware`` also
#: gates on what the growth wave will pay for its early arrival.  ``always_2s``
#: never falls back at all -- which only becomes thinkable once the correction
#: is bought in argon, because the whole point of the fallback was to dodge a
#: kilometre-per-second burn that methalox could not afford.
CADENCE_POLICIES = ("adaptive", "split_aware", "always_2s")


def fly_correction_chain(
    policy: str = "adaptive",
    start: str = DEFAULT_START,
    years: float = DEFAULT_HORIZON_YEARS,
    threshold_m_s: float = DEFAULT_THRESHOLD_M_S,
    split_days: float = DEFAULT_SPLIT_DAYS,
) -> List[CorrectionCycle]:
    """Fly one cadence policy, resolving both waves' corrections per cycle.

    ``adaptive`` reproduces ``two_wave_growth``'s chain exactly.  The other two
    are the policies ADR 0026 scores against it: ``split_aware`` avoids the
    expensive splits and ``always_2s`` refuses to slow down for anything.

    Args:
        policy: One of :data:`CADENCE_POLICIES`.
        start: First date a departure may occur (ISO date).
        years: Length of the study horizon (Julian years).
        threshold_m_s: Proxy above which a cycle falls back to three synodic.
            Unused by ``always_2s``.
        split_days: Gap by which the growth wave must lead the nozzle wave.

    Returns:
        The flown cycles, in order, whose returns fit inside the horizon.

    Raises:
        ValueError: If the policy is unknown, or the horizon, threshold or
            split gap is not positive.
    """
    if policy not in CADENCE_POLICIES:
        raise ValueError(f"unknown cadence policy: {policy}")
    if years <= 0.0:
        raise ValueError("years must be positive")
    if threshold_m_s < 0.0:
        raise ValueError("threshold_m_s must be nonnegative")
    if split_days <= 0.0:
        raise ValueError("split_days must be positive")
    start_epoch = Time(start, scale="tdb")
    departure_jd = float(_resonance_epochs(start_epoch, 5.0)[0])
    horizon_end_jd = float(start_epoch.tdb.jd) + years * _DAYS_PER_YEAR
    one_synodic_days = _FIXED_TWO_SYNODIC_DAYS / 2.0
    cycles: List[CorrectionCycle] = []
    while departure_jd + 2.0 * one_synodic_days <= horizon_end_jd:
        two_return_jd = departure_jd + 2.0 * one_synodic_days
        nozzle = _cheapest(departure_jd, two_return_jd)
        growth = _cheapest(departure_jd, two_return_jd - split_days)
        over_nozzle = 1000.0 * nozzle["dsm_only"].dsm > threshold_m_s
        over_split = policy == "split_aware" and (
            1000.0 * min(s.total_dv for s in growth.values()) > threshold_m_s
        )
        multiple, return_jd, forced = 2, two_return_jd, False
        if policy != "always_2s" and (over_nozzle or over_split):
            forced = over_split and not over_nozzle
            multiple = 3
            return_jd = departure_jd + 3.0 * one_synodic_days
            if return_jd > horizon_end_jd:
                break
            nozzle = _cheapest(departure_jd, return_jd)
            growth = _cheapest(departure_jd, return_jd - split_days)
        cycles.append(
            CorrectionCycle(
                index=len(cycles),
                departure_jd=departure_jd,
                return_jd=return_jd,
                synodic_multiple=multiple,
                period_years=(return_jd - departure_jd) / _DAYS_PER_YEAR,
                forced_by_split=forced,
                nozzle=nozzle["dsm_only"],
                growth=min(growth.values(), key=lambda s: s.total_dv),
                growth_dsm_only=growth["dsm_only"],
                split_days=split_days,
            )
        )
        departure_jd = return_jd
    if not cycles:
        raise RuntimeError("study horizon contains no complete cycle")
    return cycles


def correction_cycles_to_two_wave(
    cycles: Sequence[CorrectionCycle],
    propellant: str = "methalox",
    growth_case: str = "cheapest",
    array_fraction: float = 0.0,
) -> List[TwoWaveCycle]:
    """Convert to the ledger's cycle type, charging corrections in a propellant.

    ``same_cycle_nozzle`` charges one exhaust speed for the interplanetary
    correction and the Earth-orbit apoapsis reversal alike, but only the first
    is an SEP candidate.  Charging argon therefore scales the correction into
    the methalox delta-v that costs the same mass fraction,
    ``dv * v_e_methalox / v_e_argon``, so ``delivered1`` becomes exactly the
    argon figure while the reversal stays methalox.  The scaled burns are not
    physical delta-v and must not be read back as such.

    The SEP array is hardware, not propellant, so the rocket equation has no
    place to put it -- but the wave is destroyed on arrival, so it is consumed
    just the same.  ``array_fraction`` charges it as the extra equivalent
    delta-v that costs the same mass, ``-ln(1 - array_fraction) * v_e``, on the
    growth wave.  The *nozzle* wave's array is still uncharged, so a chain
    priced this way remains optimistic; see ADR 0026's open items.

    Args:
        cycles: The flown cycles.
        propellant: ``"methalox"`` or ``"argon"``.
        growth_case: ``"cheapest"`` to charge the growth wave its best
            maneuver, or ``"dsm_only"`` for what ``two_wave_growth`` charges.
        array_fraction: Fraction of the growth wave that is SEP hardware.
            Zero is the ledger's own accounting, propellant only.

    Returns:
        Ledger cycles ready for :func:`~src.two_wave_growth.price_chain`.

    Raises:
        ValueError: If ``propellant`` or ``growth_case`` is unrecognised, or
            the array fraction is not in [0, 1).
    """
    if propellant not in ("methalox", "argon"):
        raise ValueError(f"unknown propellant: {propellant}")
    if growth_case not in ("cheapest", "dsm_only"):
        raise ValueError(f"unknown growth_case: {growth_case}")
    if not 0.0 <= array_fraction < 1.0:
        raise ValueError("array_fraction must be in [0, 1)")
    scale = 1.0 if propellant == "methalox" else VE_METHALOX / VE_ARGON
    array_dv = -float(np.log1p(-array_fraction)) * VE_METHALOX
    out: List[TwoWaveCycle] = []
    for cycle in cycles:
        growth = cycle.growth if growth_case == "cheapest" else cycle.growth_dsm_only
        out.append(
            TwoWaveCycle(
                index=cycle.index,
                departure_jd=cycle.departure_jd,
                return_jd=cycle.return_jd,
                synodic_multiple=cycle.synodic_multiple,
                period_years=cycle.period_years,
                departure_burn=(
                    cycle.nozzle.earth_departure_periapsis_speed
                    - _cycle_periapsis_speed()
                ),
                nozzle_wave_v_b=cycle.nozzle.earth_return_collision_speed,
                nozzle_wave_dsm=scale * cycle.nozzle.dsm,
                split_days=cycle.split_days,
                growth_wave_arrival_jd=cycle.return_jd - cycle.split_days,
                growth_wave_v_b=growth.earth_return_collision_speed,
                growth_wave_burn=scale * growth.total_dv + array_dv,
            )
        )
    return out


def _cycle_periapsis_speed() -> float:
    """Closed-cycle 200 km periapsis speed (km/s), the departure burn's start."""
    from src.two_wave_growth import _cycle_periapsis_speed as speed

    return speed()


def price_correction_chain(
    cycles: Sequence[CorrectionCycle],
    propellant: str = "methalox",
    growth_case: str = "cheapest",
    recovery: float = 0.6,
    fudge: float = 0.8,
    array_fraction: float = 0.0,
) -> ChainGrowth:
    """Price a flown chain on ADR 0013's two-wave ledger.

    The parking orbit is resized to the chain's own split gap, since the pushed
    payload coasts exactly one of its periods between the two waves.

    Args:
        cycles: The flown cycles.
        propellant: Currency for both correction burns.
        growth_case: Which growth-wave maneuver to charge.
        recovery: Nozzle impulse recovery fraction, ``e``.
        fudge: Growth-push elasticity, ``f``.
        array_fraction: Fraction of the growth wave that is SEP hardware.

    Returns:
        The chain's compounded growth and its per-year rate.
    """
    ledger = correction_cycles_to_two_wave(
        cycles, propellant, growth_case, array_fraction
    )
    return price_chain(
        ledger,
        recovery=recovery,
        fudge=fudge,
        reversal_period=cycles[0].split_days * u.day,
    )


@dataclass(frozen=True)
class LegImpulse:
    """What the array can deliver along one propagated leg.

    Attributes:
        dv_per_tonne: Velocity change a one-tonne wave could accumulate over
            the counted stretch (m/s), from ``F = 2 eta P / v_e`` integrated in
            time.  Scales as the reciprocal of wave mass.
        days_counted: Duration of the counted stretch (days).
        mean_power_kw: Mean array power while counting (kW).
        min_radius_au: Closest heliocentric approach sampled on the leg (AU).
    """

    dv_per_tonne: float
    days_counted: float
    mean_power_kw: float
    min_radius_au: float


def _two_body(_t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Heliocentric two-body derivative for ``solve_ivp``."""
    position = y[:3]
    radius = float(np.linalg.norm(position))
    return np.concatenate([y[3:], -_MU_SUN * position / radius**3])


def leg_impulse_per_tonne(
    position: npt.NDArray[np.float64],
    velocity: npt.NDArray[np.float64],
    duration_s: float,
    beyond_au: Optional[float] = None,
    power_1au_kw: float = SEP_ARRAY_POWER_1AU_KW,
    efficiency: float = SEP_THRUSTER_EFFICIENCY,
) -> LegImpulse:
    """Integrate deliverable SEP impulse along one heliocentric leg.

    The leg is propagated as an unperturbed conic -- the thrust that would
    perturb it is precisely what is being sized, and at these levels it moves
    the radius history far less than the ephemeris model already does.  The
    array follows ``1/r^2`` and the thruster converts at a fixed efficiency, so
    the integrand is ``2 eta P(r(t)) / v_e``.

    This is a budget, not a trajectory solution.  It says how many
    thrust-seconds exist along the leg; whether a distributed burn achieves the
    same endpoint change as the impulsive proxy it is being compared against is
    a low-thrust optimisation this does not perform.

    Args:
        position: Heliocentric position at the start of the leg (km).
        velocity: Heliocentric velocity at the start of the leg (km/s).
        duration_s: Leg duration (s).
        beyond_au: Count only the stretch outside this radius; None counts all.
        power_1au_kw: Array power at 1 AU (kW).
        efficiency: Thruster efficiency.

    Returns:
        The leg's impulse budget.

    Raises:
        ValueError: If the duration is not positive.
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    solution = solve_ivp(
        _two_body,
        (0.0, duration_s),
        np.concatenate([position, velocity]),
        rtol=1e-10,
        atol=1e-3,
        dense_output=True,
    )
    times = np.linspace(0.0, duration_s, _LEG_SAMPLES)
    radii = np.linalg.norm(solution.sol(times)[:3], axis=0) / _AU_KM
    power_kw = power_1au_kw / radii**2
    thrust_n = 2.0 * efficiency * (power_kw * 1000.0) / (VE_ARGON * 1000.0)
    counted = (
        np.ones_like(radii, dtype=bool) if beyond_au is None else radii > beyond_au
    )
    return LegImpulse(
        # N.s on one tonne is mm/s; /1000 twice would be wrong, so: N.s / 1000 kg
        dv_per_tonne=float(np.trapezoid(np.where(counted, thrust_n, 0.0), times))
        / 1000.0,
        days_counted=float(np.trapezoid(counted.astype(float), times))
        / _SECONDS_PER_DAY,
        mean_power_kw=float(power_kw[counted].mean()) if counted.any() else 0.0,
        min_radius_au=float(radii.min()),
    )


def _lambert_velocity(
    origin: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    duration_s: float,
    prograde: bool,
) -> npt.NDArray[np.float64]:
    """Departure velocity of the zero-revolution arc joining two points.

    Args:
        origin: Heliocentric position left (km).
        target: Heliocentric position reached (km).
        duration_s: Time of flight (s).
        prograde: Whether the arc is prograde.

    Returns:
        The heliocentric departure velocity (km/s).

    Raises:
        RuntimeError: If neither branch converges.
    """
    for low_path in (True, False):
        try:
            velocity, _ = izzo(
                _MU_SUN,
                origin,
                target,
                duration_s,
                0,
                prograde,
                low_path,
                _LAMBERT_ITERATIONS,
                _LAMBERT_RTOL,
            )
        except (AssertionError, RuntimeError, ValueError):
            continue
        return np.asarray(velocity, dtype=np.float64)
    raise RuntimeError("no Lambert arc joins the requested endpoints")


@dataclass(frozen=True)
class CycleFeasibility:
    """Whether the array can pay one cycle's two corrections.

    Attributes:
        cycle: The flown cycle priced.
        dsm_leg: Impulse available to the nozzle wave outside the deploy
            radius, which is the only stretch its correction may use.
        split_legs: Impulse available to the growth wave over its whole
            trajectory, outbound leg included, since its burn is unconstrained
            in place.
        wave_tonnes: Heaviest wave both burns could drive, the smaller of the
            two per-burn limits.  Infinite when neither burn is needed.
    """

    cycle: CorrectionCycle
    dsm_leg: LegImpulse
    split_legs: Tuple[LegImpulse, LegImpulse]
    wave_tonnes: float


def cycle_feasibility(
    cycle: CorrectionCycle,
    deploy_radius_au: float = SEP_DEPLOY_RADIUS_AU,
    power_1au_kw: float = SEP_ARRAY_POWER_1AU_KW,
) -> CycleFeasibility:
    """Integrate the impulse each of a cycle's two corrections can draw on.

    Args:
        cycle: The flown cycle.
        deploy_radius_au: Radius inside which the returning wave must have
            finished correcting, to leave time for stream deployment.
        power_1au_kw: Array power at 1 AU (kW).

    Returns:
        The cycle's impulse budgets and the wave mass they support.
    """
    earth_departure, _ = _heliocentric_state("earth", cycle.departure_jd)
    nozzle_jupiter, _ = _heliocentric_state("jupiter", cycle.nozzle.encounter_jd)
    nozzle_earth, _ = _heliocentric_state("earth", cycle.return_jd)
    nozzle_tof = (cycle.return_jd - cycle.nozzle.encounter_jd) * _SECONDS_PER_DAY
    dsm_leg = leg_impulse_per_tonne(
        nozzle_jupiter,
        _lambert_velocity(nozzle_jupiter, nozzle_earth, nozzle_tof, False),
        nozzle_tof,
        beyond_au=deploy_radius_au,
        power_1au_kw=power_1au_kw,
    )

    growth_arrival_jd = cycle.return_jd - cycle.split_days
    growth_jupiter, _ = _heliocentric_state("jupiter", cycle.growth.encounter_jd)
    growth_earth, _ = _heliocentric_state("earth", growth_arrival_jd)
    outbound_tof = (cycle.growth.encounter_jd - cycle.departure_jd) * _SECONDS_PER_DAY
    return_tof = (growth_arrival_jd - cycle.growth.encounter_jd) * _SECONDS_PER_DAY
    outbound = leg_impulse_per_tonne(
        earth_departure,
        _lambert_velocity(earth_departure, growth_jupiter, outbound_tof, True),
        outbound_tof,
        power_1au_kw=power_1au_kw,
    )
    inbound = leg_impulse_per_tonne(
        growth_jupiter,
        _lambert_velocity(growth_jupiter, growth_earth, return_tof, False),
        return_tof,
        power_1au_kw=power_1au_kw,
    )

    limits = []
    if cycle.nozzle.dsm > 0.0:
        limits.append(dsm_leg.dv_per_tonne / (1000.0 * cycle.nozzle.dsm))
    if cycle.growth.total_dv > 0.0:
        limits.append(
            (outbound.dv_per_tonne + inbound.dv_per_tonne)
            / (1000.0 * cycle.growth.total_dv)
        )
    return CycleFeasibility(
        cycle=cycle,
        dsm_leg=dsm_leg,
        split_legs=(outbound, inbound),
        wave_tonnes=min(limits) if limits else float("inf"),
    )


def delivered_fraction(
    correction_dv: float,
    propellant: str,
    wave_tonnes: Optional[float] = None,
    specific_mass_kg_per_kw: float = 15.0,
    power_1au_kw: float = SEP_ARRAY_POWER_1AU_KW,
) -> float:
    """Fraction of a wave that survives its correction as impactor.

    The growth wave is destroyed on arrival, so *everything* it carries to buy
    the correction is consumed every cycle -- propellant, tank and, for SEP,
    the array.  The comparison between propellants is therefore between mass
    fractions of the same wave, not between propellant masses:

        ``delivered = exp(-dv/v_e) - tau * (1 - exp(-dv/v_e)) - m_array/m_wave``

    with ``tau`` the tank mass per kg of propellant and the array term present
    only for SEP.  Note this is **not** what the ledger charges: ADR 0013's
    ``delivered1`` is the bare rocket equation, so the priced chains are
    optimistic for argon by exactly the tank and array terms here.

    Args:
        correction_dv: Total correction the wave must buy (km/s).
        propellant: ``"methalox"`` or ``"argon"``.
        wave_tonnes: Wave mass, needed only for argon's array charge.  None
            omits the array, which is the ledger's own accounting.
        specific_mass_kg_per_kw: SEP system mass per kW at 1 AU.
        power_1au_kw: Array power at 1 AU (kW).

    Returns:
        The surviving impactor fraction, which may be negative when the
        hardware outweighs the wave.

    Raises:
        ValueError: If the propellant is unknown or the correction negative.
    """
    if correction_dv < 0.0:
        raise ValueError("correction_dv must be nonnegative")
    if propellant == "argon":
        exhaust, tank = VE_ARGON, ARGON_TANK_FRACTION
    elif propellant == "methalox":
        exhaust, tank = VE_METHALOX, METHALOX_TANK_FRACTION
    else:
        raise ValueError(f"unknown propellant: {propellant}")
    kept = float(np.exp(-correction_dv / exhaust))
    delivered = kept - tank * (1.0 - kept)
    if propellant == "argon" and wave_tonnes is not None:
        delivered -= power_1au_kw * specific_mass_kg_per_kw / (wave_tonnes * 1000.0)
    return delivered


def characteristic_acceleration(
    power_1au_kw: float,
    wave_tonnes: float,
    efficiency: float = SEP_THRUSTER_EFFICIENCY,
) -> float:
    """Thrust acceleration at 1 AU (m/s^2), the SEP designer's own unit.

    ``a = 2 eta P / (m v_e)``.  It falls as ``1/r^2`` with the array, so the
    value at Jupiter is this divided by about 27.

    Args:
        power_1au_kw: Array power at 1 AU (kW).
        wave_tonnes: Mass being accelerated (tonnes).
        efficiency: Thruster efficiency.

    Returns:
        Characteristic acceleration at 1 AU (m/s^2).

    Raises:
        ValueError: If the wave mass is not positive.
    """
    if wave_tonnes <= 0.0:
        raise ValueError("wave_tonnes must be positive")
    return (
        2.0
        * efficiency
        * (power_1au_kw * 1000.0)
        / (wave_tonnes * 1000.0 * VE_ARGON * 1000.0)
    )


def required_characteristic_acceleration(feasible: "CycleFeasibility") -> float:
    """Acceleration at 1 AU the binding burn demands (m/s^2), at any wave mass.

    Scale-invariant for the same reason :func:`array_mass_fraction` is: power
    and mass rise together, so their ratio -- which is what acceleration is --
    does not move.

    Args:
        feasible: The cycle's integrated impulse budgets.

    Returns:
        Required characteristic acceleration at 1 AU (m/s^2); zero when the
        cycle needs no correction.
    """
    if feasible.wave_tonnes == float("inf"):
        return 0.0
    return characteristic_acceleration(required_power_kw(feasible, 1.0), 1.0)


def required_power_kw(
    feasible: "CycleFeasibility",
    wave_tonnes: float,
) -> float:
    """Array power at 1 AU needed to buy one cycle's corrections (kW).

    :func:`cycle_feasibility` measures how heavy a wave the reference array
    could drive; a wave ``n`` times heavier needs ``n`` times the power, since
    the leg durations are fixed by the trajectory and thrust is linear in
    power.  So this is a simple rescale of that measurement.

    Args:
        feasible: The cycle's integrated impulse budgets.
        wave_tonnes: Mass of the wave to be corrected.

    Returns:
        Required array power at 1 AU (kW), for the binding one of the two
        burns.  Zero when the cycle needs no correction at all.

    Raises:
        ValueError: If the wave mass is not positive.
    """
    if wave_tonnes <= 0.0:
        raise ValueError("wave_tonnes must be positive")
    if feasible.wave_tonnes == float("inf"):
        return 0.0
    return SEP_ARRAY_POWER_1AU_KW * wave_tonnes / feasible.wave_tonnes


def deliverable_dv(
    feasible: "CycleFeasibility",
    acceleration_1au: float = STATED_ACCELERATION_1AU,
) -> Tuple[float, float]:
    """Delta-v a given characteristic acceleration actually buys (m/s).

    The integrated budgets in ``feasible`` were measured at the reference
    array's own acceleration; thrust is linear in power, so rescaling to any
    other acceleration is a single multiplication.

    Args:
        feasible: The cycle's integrated impulse budgets.
        acceleration_1au: Characteristic acceleration at 1 AU (m/s^2).

    Returns:
        ``(over the whole trajectory, outside the deploy radius)`` in m/s --
        the budgets available to the separation burn and to the returning
        wave's DSM respectively.
    """
    reference = characteristic_acceleration(SEP_ARRAY_POWER_1AU_KW, 1.0)
    scale = acceleration_1au / reference
    outbound, inbound = feasible.split_legs
    return (
        (outbound.dv_per_tonne + inbound.dv_per_tonne) * scale,
        feasible.dsm_leg.dv_per_tonne * scale,
    )


def array_mass_fraction(
    feasible: "CycleFeasibility",
    specific_mass_kg_per_kw: float = 15.0,
) -> float:
    """Fraction of the wave that must be SEP hardware, at any wave mass.

    **This is scale-invariant, and that is the whole point.**  A wave ``n``
    times heavier needs ``n`` times the power to buy the same delta-v in the
    same time, hence ``n`` times the array -- so the array is the same
    *percentage* of a five-tonne wave and a five-hundred-tonne one.  Sizing the
    wave therefore cannot make the array cheap; only a smaller correction, a
    longer burn window, or a lighter array can.

    That also retires the question of a break-even wave mass, which only looked
    meaningful while the array was held at a fixed 100 kW independent of what
    it had to push.

    Args:
        feasible: The cycle's integrated impulse budgets.
        specific_mass_kg_per_kw: SEP system mass per kW at 1 AU.

    Returns:
        The hardware fraction of the wave, dimensionless.
    """
    if feasible.wave_tonnes == float("inf"):
        return 0.0
    return (
        specific_mass_kg_per_kw * SEP_ARRAY_POWER_1AU_KW / feasible.wave_tonnes / 1000.0
    )


@dataclass(frozen=True)
class SepSplitAnalysis:
    """Tables behind ADR 0026.

    Attributes:
        cycles: One row per flown cycle of each cadence, with both corrections.
        chains: One row per (cadence, growth case, propellant) priced chain.
        feasibility: One row per flown cycle of each cadence, with the impulse
            each of its corrections can draw on and the wave mass that
            supports.
        requirements: One row per cadence: what its worst cycle demands of the
            SEP system, and how that compares with the design's stated
            operating point.  This is the table behind the decision.
        break_even: One row per (cadence, SEP specific mass): the power per
            tonne the cadence's worst cycle demands, the scale-invariant array
            fraction that implies, and both quoted in absolute terms at
            :data:`REFERENCE_WAVE_TONNES`.
    """

    cycles: pd.DataFrame
    chains: pd.DataFrame
    feasibility: pd.DataFrame
    requirements: pd.DataFrame
    break_even: pd.DataFrame


def analyze_sep_split(
    start: str = DEFAULT_START,
    years: float = DEFAULT_HORIZON_YEARS,
    split_days: float = DEFAULT_SPLIT_DAYS,
    threshold_m_s: float = DEFAULT_THRESHOLD_M_S,
    specific_masses: Sequence[float] = SEP_SPECIFIC_MASS_KG_PER_KW,
) -> SepSplitAnalysis:
    """Fly both cadences, price them in both propellants, and size the array.

    Args:
        start: First date a departure may occur (ISO date).
        years: Length of the study horizon (Julian years).
        split_days: Gap by which the growth wave leads the nozzle wave.
        threshold_m_s: Proxy above which a cycle falls back to three synodic.
        specific_masses: SEP system masses per kW at 1 AU to report.

    Returns:
        The per-cycle, chain, feasibility and break-even tables.
    """
    chains = {
        policy: fly_correction_chain(policy, start, years, threshold_m_s, split_days)
        for policy in CADENCE_POLICIES
    }

    cycle_rows = []
    for tag, cycles in chains.items():
        for cycle in cycles:
            cycle_rows.append(
                {
                    "cadence": tag,
                    "index": cycle.index,
                    "departure": Time(cycle.departure_jd, format="jd", scale="tdb").iso[
                        :10
                    ],
                    "synodic_multiple": cycle.synodic_multiple,
                    "forced_by_split": cycle.forced_by_split,
                    "nozzle_dsm_m_s": 1000.0 * cycle.nozzle.dsm,
                    "split_dsm_only_m_s": 1000.0 * cycle.growth_dsm_only.total_dv,
                    "split_best_m_s": 1000.0 * cycle.growth.total_dv,
                    "split_best_case": cycle.growth.case,
                    "correction_total_m_s": 1000.0 * cycle.correction_total,
                    "bend_limited": cycle.bend_limited,
                    "growth_perijove_r_j": cycle.growth.perijove_radius / 71_492.0,
                    "nozzle_v_b": cycle.nozzle.earth_return_collision_speed,
                    "growth_v_b": cycle.growth.earth_return_collision_speed,
                }
            )

    chain_rows = []
    for tag, cycles in chains.items():
        for growth_case in ("dsm_only", "cheapest"):
            for propellant in ("methalox", "argon"):
                growth = price_correction_chain(
                    cycles, propellant=propellant, growth_case=growth_case
                )
                chain_rows.append(
                    {
                        "cadence": tag,
                        "growth_case": growth_case,
                        "propellant": propellant,
                        "specific_mass_kg_per_kw": float("nan"),
                        "array_fraction": 0.0,
                        "cycles": growth.cycles,
                        "two_synodic": growth.two_synodic_cycles,
                        "three_synodic": growth.three_synodic_cycles,
                        "horizon_years": growth.horizon_years,
                        "slug_ratio": growth.slug_ratio,
                        "total_growth": growth.total_growth,
                        "rate": growth.rate,
                        "doubling": growth.doubling,
                    }
                )

    # What the array costs the ledger.  The chains above charge propellant
    # only -- the rocket equation has nowhere to put hardware -- so these rows
    # re-price argon with the array carried as mass on the growth wave.  The
    # fraction is set by the chain's worst cycle and does *not* depend on wave
    # mass (see :func:`array_mass_fraction`), so the sweep is over how heavy
    # the hardware is per kW, not over how big the wave is.
    worst_feasible = {
        tag: max(
            (cycle_feasibility(c) for c in cycles),
            key=lambda f: array_mass_fraction(f),
        )
        for tag, cycles in chains.items()
    }
    for tag, cycles in chains.items():
        for alpha in specific_masses:
            fraction = array_mass_fraction(worst_feasible[tag], alpha)
            if fraction >= 1.0:
                continue
            growth = price_correction_chain(
                cycles, propellant="argon", array_fraction=fraction
            )
            chain_rows.append(
                {
                    "cadence": tag,
                    "growth_case": "cheapest",
                    "propellant": "argon+array",
                    "specific_mass_kg_per_kw": alpha,
                    "array_fraction": fraction,
                    "cycles": growth.cycles,
                    "two_synodic": growth.two_synodic_cycles,
                    "three_synodic": growth.three_synodic_cycles,
                    "horizon_years": growth.horizon_years,
                    "slug_ratio": growth.slug_ratio,
                    "total_growth": growth.total_growth,
                    "rate": growth.rate,
                    "doubling": growth.doubling,
                }
            )

    feasibility_rows = []
    for tag, cycles in chains.items():
        for cycle in cycles:
            feasible = cycle_feasibility(cycle)
            outbound, inbound = feasible.split_legs
            feasibility_rows.append(
                {
                    "cadence": tag,
                    "departure": Time(cycle.departure_jd, format="jd", scale="tdb").iso[
                        :10
                    ],
                    "synodic_multiple": cycle.synodic_multiple,
                    "nozzle_dsm_m_s": 1000.0 * cycle.nozzle.dsm,
                    "split_m_s": 1000.0 * cycle.growth.total_dv,
                    "correction_total_m_s": 1000.0 * cycle.correction_total,
                    "dsm_dv_per_tonne": feasible.dsm_leg.dv_per_tonne,
                    "dsm_days_beyond_deploy": feasible.dsm_leg.days_counted,
                    "dsm_mean_kw": feasible.dsm_leg.mean_power_kw,
                    "split_dv_per_tonne": outbound.dv_per_tonne + inbound.dv_per_tonne,
                    "wave_tonnes": feasible.wave_tonnes,
                }
            )
    feasibility = pd.DataFrame(feasibility_rows)

    # What the array must be, per cadence.  Power scales with wave mass and
    # the array scales with power, so the *fraction* is fixed and the absolute
    # power is quoted at a reference wave the architecture actually wants.
    hardware_rows = []
    for tag in chains:
        feasible = worst_feasible[tag]
        for alpha in specific_masses:
            fraction = array_mass_fraction(feasible, alpha)
            hardware_rows.append(
                {
                    "cadence": tag,
                    "specific_mass_kg_per_kw": alpha,
                    "worst_correction_m_s": 1000.0 * feasible.cycle.correction_total,
                    "kw_per_tonne": required_power_kw(feasible, 1.0),
                    "required_accel_1au": required_characteristic_acceleration(
                        feasible
                    ),
                    "required_accel_jupiter": required_characteristic_acceleration(
                        feasible
                    )
                    / 27.04,
                    "array_fraction": fraction,
                    "power_mw_at_reference": required_power_kw(
                        feasible, REFERENCE_WAVE_TONNES
                    )
                    / 1000.0,
                    "array_tonnes_at_reference": fraction * REFERENCE_WAVE_TONNES,
                    "argon_delivered": delivered_fraction(
                        feasible.cycle.correction_total, "argon"
                    )
                    - fraction,
                    "methalox_delivered": delivered_fraction(
                        feasible.cycle.correction_total, "methalox"
                    ),
                }
            )

    # What each cadence demands of the SEP system, against what was claimed.
    thrust_n_per_kw = 2.0 * SEP_THRUSTER_EFFICIENCY / VE_ARGON
    requirement_rows = []
    for tag in chains:
        feasible = worst_feasible[tag]
        needed = required_characteristic_acceleration(feasible)
        power_kw = required_power_kw(feasible, REFERENCE_WAVE_TONNES)
        whole, beyond = deliverable_dv(feasible)
        split_needed = 1000.0 * feasible.cycle.growth.total_dv
        dsm_needed = 1000.0 * feasible.cycle.nozzle.dsm
        requirement_rows.append(
            {
                "cadence": tag,
                "worst_departure": Time(
                    feasible.cycle.departure_jd, format="jd", scale="tdb"
                ).iso[:10],
                "split_needed_m_s": split_needed,
                "dsm_needed_m_s": dsm_needed,
                "split_available_m_s": whole,
                "dsm_available_m_s": beyond,
                "split_shortfall": split_needed / whole if whole > 0 else np.inf,
                "dsm_shortfall": dsm_needed / beyond if beyond > 0 else np.inf,
                "required_kw_per_tonne": required_power_kw(feasible, 1.0),
                "required_accel_1au": needed,
                "required_accel_jupiter": needed / _JUPITER_INVERSE_SQUARE,
                "times_stated_accel": needed / STATED_ACCELERATION_1AU,
                "power_mw_at_reference": power_kw / 1000.0,
                "thrust_n_at_1au": thrust_n_per_kw * power_kw,
                "thrust_n_at_jupiter": thrust_n_per_kw
                * power_kw
                / _JUPITER_INVERSE_SQUARE,
                "reference_thrusters": power_kw / REFERENCE_THRUSTER_KW,
            }
        )

    return SepSplitAnalysis(
        cycles=pd.DataFrame(cycle_rows),
        chains=pd.DataFrame(chain_rows),
        feasibility=feasibility,
        requirements=pd.DataFrame(requirement_rows),
        break_even=pd.DataFrame(hardware_rows),
    )


def main() -> None:
    """Print the ADR 0026 tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--years", type=float, default=DEFAULT_HORIZON_YEARS)
    parser.add_argument("--split-days", type=float, default=DEFAULT_SPLIT_DAYS)
    parser.add_argument("--csv", default=None, help="write the cycle table here")
    args = parser.parse_args()

    analysis = analyze_sep_split(
        start=args.start, years=args.years, split_days=args.split_days
    )
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(
        f"argon v_e {VE_ARGON:.3f} km/s (Isp {ARGON_SEP_ISP}); "
        f"methalox v_e {VE_METHALOX:.4f} km/s; ratio {VE_ARGON / VE_METHALOX:.3f}"
    )
    print(
        f"array {SEP_ARRAY_POWER_1AU_KW:.0f} kW at 1 AU -> "
        f"{SEP_ARRAY_POWER_1AU_KW / 25.0:.2f} kW at 5 AU, "
        f"{2.0 * SEP_THRUSTER_EFFICIENCY / VE_ARGON * 1e3:.1f} mN per electrical kW\n"
    )
    print("--- flown cycles ---")
    print(analysis.cycles.to_string(index=False))
    print("\n--- priced chains ---")
    print(analysis.chains.to_string(index=False))
    print("\n--- SEP impulse available ---")
    print(analysis.feasibility.to_string(index=False))
    print("\n--- what the SEP system must deliver, against what was stated ---")
    print(
        f"stated operating point: {STATED_ACCELERATION_1AU:.1e} m/s^2 at 1 AU "
        f"({STATED_ACCELERATION_JUPITER:.1e} quoted at Jupiter; 1/r^2 from the "
        f"first gives {STATED_ACCELERATION_1AU / _JUPITER_INVERSE_SQUARE:.2e})"
    )
    print(analysis.requirements.to_string(index=False))
    print("\n--- SEP hardware each cadence demands ---")
    print(analysis.break_even.to_string(index=False))
    if args.csv:
        analysis.cycles.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
