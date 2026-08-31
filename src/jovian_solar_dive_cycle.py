"""Earth -> Jupiter -> 4 solar-radii Oberth -> Earth, clocked to a synodic multiple.

The paper's no-ISRU growth loop (``sec:no_isru_rocket``) sends a payload straight
from Earth to a 4 solar-radii periapsis, boosts it there, and brings it home at
about 150 km/s.  Phasing that return costs the single-impulse resonant dive's
~37.5 km/s heliocentric Earth boost (``sec:earth_reintercept``).  This module
asks whether **Jupiter** can place the dive instead, so no Earth-side dive
injection is needed at all: depart Earth for Jupiter, let an unpowered Jovian
flyby drop perihelion to 4 solar radii, take the Oberth boost there, and cross
1 AU where Earth is waiting -- with the whole loop timed to an integer multiple
of the Earth--Jupiter synodic period, so the next cycle repeats the geometry.

The model is circular and coplanar, in the same spirit as
:mod:`src.circular_resonance_impulse`; the real-ephemeris audit of that clock
lives in :mod:`src.real_orbit_resonance`.  All algebra is float (km, s, km/s,
rad) and delegates to :mod:`src.conic_kernel` -- see CONTEXT.md, "Conic kernel".

The headline is asymmetric, which is why it earned an ADR
(``docs/adr/0019-three-synodic-closes-two-does-not.md``):

* **Three synodic periods closes unpowered**, with 58.8 degrees of spare Jovian
  bend, off a modest 10.54 km/s Earth departure excess.
* **Two synodic periods does not close, and no perijove burn fixes it.**
  Holding the clock and the Earth intercept leaves a 6.84 degree bend deficit,
  and sweeping the perijove burn over every magnitude and sign bottoms the
  deficit out at +6.14 degrees without ever reaching zero.  Only a maneuver
  *off* the flyby closes it -- 2.27 km/s as a deep-space maneuver at 4.39 AU,
  2.12 km/s as an SOI-seam rotation -- so 2S buys a 33 percent shorter cycle
  with about 2 km/s of carried propellant per pass.

Run with ``make jovian-dive``.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.constants import g0
from boinor.bodies import Earth, Sun
from scipy.optimize import brentq, fsolve
from tabulate import tabulate

from src.astro_constants import (
    LEO_ALTITUDE,
    LOW_JUPITER_ALTITUDE,
    SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
    STD_FUDGE_FACTOR,
)
from src.circular_resonance_impulse import impulse_per_impactor_kg
from src.conic_kernel import (
    ConicState,
    conic_radius_crossings,
    conic_state_at_radius,
    elliptic_tof_seconds,
    half_turn_angle,
    hyperbolic_eccentricity,
    hyperbolic_tof_seconds,
    semimajor_axis_from_energy,
    speed_components_at_radius,
    speed_with_escape_energy,
    true_anomaly_at_radius_rad,
    unpowered_bend_angle,
    wrap_pi,
)
from src.heliocentric_reintercept import single_impulse_resonant_dive
from src.propulsion import payload_mass_ratio
from src.retrograde_return_legs import _FlybyParams, _powered_flyby_params

_SECONDS_PER_YEAR = 365.25 * 86400.0
_MU_EARTH = float(Earth.k.to_value(u.km**3 / u.s**2))
_BURN_RADIUS = float((Earth.R + LEO_ALTITUDE).to_value(u.km))
_STANDARD_GRAVITY = float(g0.to_value(u.km / u.s**2))

# Hyperbolic-excess speed the boosted projectile keeps after climbing out of the
# Sun (km/s) -- the paper's ~150 km/s Earth-crossing scale (sec:four_radii_thermal).
DEFAULT_RETURN_EXCESS = 150.0

# --- Search bounds.  Recorded here, not just in a scratch harness: ADR 0007's
#     numbers became unreproducible when its bounds were left out (CLAUDE.md).
# Earth-departure hyperbolic-excess bracket (km/s).  The floor is below any
# transfer that reaches Jupiter at all; the ceiling is well past the 25 km/s
# production flyby-search cap of circular_resonance_impulse.
DEPARTURE_EXCESS_BRACKET = (8.0, 40.0)
# Departure aim bracket (deg from Earth's prograde direction; +90 is radially
# outward).  Wide enough to bracket both the 2S root (~+11 deg) and the 3S root
# (~-12 deg) with margin on each side.
DEPARTURE_AIM_BRACKET = (-60.0, 85.0)
# Aim samples used to bracket the intercept root before it is refined.
_AIM_SAMPLES = 59
# Bisections used to locate the departure excess below which the transfer's
# aphelion no longer reaches Jupiter's orbit radius.  50 halvings of the 32 km/s
# bracket resolve the floor far below any speed the model distinguishes.
_FLOOR_BISECTIONS = 50
# Outgoing-excess samples for the powered-perijove bend-deficit sweep (km/s).
PERIJOVE_EXCESS_BRACKET = (12.0, 32.0)
_PERIJOVE_SAMPLES = 41
# Fraction of the outbound leg's flight time at which a deep-space maneuver is
# placed for the DSM price.  The maneuver gets steadily cheaper the later it
# fires, so this is set as late as the model can defend: 0.80 puts the burn near
# 4.4 AU, still 0.8 AU short of Jupiter and so well outside its ~0.322 AU sphere
# of influence, where the burn is a genuine heliocentric DSM rather than a
# Jupiter-relative maneuver wearing a DSM's name.
DSM_LEG_FRACTION = 0.80
# Aim samples for the DSM price scan, and the local refinement around the best
# of them (the price roughly doubles a few degrees either side of the minimum).
_DSM_AIM_SAMPLES = 33
_DSM_REFINE_SAMPLES = 25
# Steps used to integrate the departure burn.  Both the closing speed and the
# impact angle move as the vehicle accelerates, so the rocket equation is
# integrated rather than evaluated at a midpoint -- on the paper's own dive the
# closing speed falls 148 -> 125 km/s across the burn, which a midpoint hides.
_BURN_INTEGRATION_STEPS = 400

# Defaults for the departure-nozzle ledger.  A magnetic nozzle rather than a bare
# collision is the flown device at these speeds (sec:minimum_nozzle -- no wall
# survives the pass), so the slug ratio is the designer's knob and 30 sits well
# inside the plume ignition window at this cycle's ~158 km/s closing speed.
DEFAULT_SLUG_RATIO = 30.0
# The paper's eta_jet**2: the fraction of collision energy reaching coherent
# axial exhaust momentum.  Nothing in either repository bounds eta_geom
# (sec:jet_efficiency), so this is a stated operating point, not a result.
DEFAULT_JET_ENERGY_EFFICIENCY = 0.60
# Mass fraction surviving the 4 solar-radii collision, both the prograde payload
# and the opposing projectile stream counted.  0.60 is deliberately conservative:
# eq:external_reaction_mass at eta_jet**2 = 0.6 and the optimal retrograde share
# gives about 0.69.
DEFAULT_PERIAPSIS_SURVIVAL = 0.60


def _dive_periapsis_radius(
    solar_radii: float = SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
) -> float:
    """Dive periapsis radius in km, from a count of solar radii.

    Args:
        solar_radii: Periapsis distance in solar radii (default: the 2005 Solar
            Probe design's 4.0, ``SOLAR_DIVE_PERIAPSIS_SOLAR_RADII``).

    Returns:
        The periapsis radius from the Sun's center (km).
    """
    return solar_radii * float(Sun.R.to_value(u.km))


def synodic_period(params: Optional[_FlybyParams] = None) -> float:
    """Circular Earth--Jupiter synodic period (s).

    Args:
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The synodic period (s).
    """
    p = params if params is not None else _powered_flyby_params()
    n_earth = p.v_earth_orbit / p.r_earth_orbit
    n_jupiter = p.v_jupiter_orbit / p.r_jupiter_orbit
    return 2.0 * np.pi / (n_earth - n_jupiter)


# --------------------------------------------------------------------------
# Dive placement: what a Jovian flyby must produce to reach 4 solar radii
# --------------------------------------------------------------------------


def _dive_tangential_speed(
    energy: float, dive_radius: float, orbit_radius: float, mu: float, sign: float
) -> float:
    """Tangential speed at ``orbit_radius`` for an orbit with periapsis ``dive_radius``.

    Angular momentum is set by the periapsis condition, ``h = r_p * v_p`` with
    ``v_p = sqrt(2*(energy + mu/r_p))``; the tangential speed at any other
    radius is then ``h / r``.

    Args:
        energy: Specific orbital energy (km^2/s^2).
        dive_radius: Target periapsis radius (km).
        orbit_radius: Radius at which to report the tangential speed (km).
        mu: Gravitational parameter of the Sun (km^3/s^2).
        sign: +1 for a prograde dive, -1 for a retrograde one.

    Returns:
        The signed tangential speed (km/s).
    """
    periapsis_speed = np.sqrt(2.0 * (energy + mu / dive_radius))
    return sign * dive_radius * float(periapsis_speed) / orbit_radius


def dive_placement_excess_floor(
    sign: float,
    dive_radius: Optional[float] = None,
    params: Optional[_FlybyParams] = None,
) -> float:
    """Minimum Jupiter arrival excess speed that can place the dive perihelion.

    A flyby rotates the Jupiter-relative excess velocity but never rescales it
    (CONTEXT.md, "Tisserand invariant"), so reaching a perihelion of
    ``dive_radius`` requires cancelling almost all of Jupiter's 13.06 km/s of
    orbital motion.  The cheapest post-flyby state that does it is purely
    tangential at Jupiter -- Jupiter's orbit is then the dive's aphelion -- and
    the excess speed it needs is the floor returned here.  Below it the dive
    does not exist at any perijove: a *necessary condition*, not a cost, exactly
    like the 13.058 km/s floor on the unpowered retrograde return (CONTEXT.md,
    flagged ambiguities).

    Args:
        sign: +1 for a prograde dive, -1 for a retrograde one.  A value of 0
            asks for the radial-plunge case, which cancels Jupiter's motion
            entirely and floors at Jupiter's own orbital speed.
        dive_radius: Target perihelion radius (km); 4 solar radii by default.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The minimum Jupiter-relative arrival excess speed (km/s).
    """
    p = params if params is not None else _powered_flyby_params()
    if sign == 0.0:
        return p.v_jupiter_orbit
    r_dive = dive_radius if dive_radius is not None else _dive_periapsis_radius()
    v_tangential = sign * 1.0
    for _ in range(200):
        energy = v_tangential * v_tangential / 2.0 - p.mu_sun / p.r_jupiter_orbit
        v_tangential = _dive_tangential_speed(
            energy, r_dive, p.r_jupiter_orbit, p.mu_sun, sign
        )
    return abs(v_tangential - p.v_jupiter_orbit)


def _solve_dive_state(
    excess_out: float,
    sign: float,
    dive_radius: float,
    params: _FlybyParams,
) -> Optional[Tuple[float, float]]:
    """Post-flyby heliocentric state at Jupiter that dives to ``dive_radius``.

    With the outgoing excess *speed* fixed, the outgoing velocity lies on a
    circle of that radius around Jupiter's own velocity, so a single unknown --
    the tangential component -- parametrises it.  Substituting the excess-speed
    constraint makes the specific energy *linear* in that unknown, and the
    periapsis condition closes the equation.

    Args:
        excess_out: Jupiter-relative excess speed after the flyby (km/s).
        sign: +1 prograde dive, -1 retrograde.
        dive_radius: Target perihelion radius (km).
        params: Float parameter block.

    Returns:
        (v_tangential, v_radial) at Jupiter (km/s), radial negative because the
        dive is inbound; or None when no such state exists at this excess speed.
    """

    def residual(v_tangential: float) -> float:
        energy = (
            (excess_out * excess_out - params.v_jupiter_orbit**2) / 2.0
            + v_tangential * params.v_jupiter_orbit
            - params.mu_sun / params.r_jupiter_orbit
        )
        if energy + params.mu_sun / dive_radius <= 0.0:
            return float("nan")
        return (
            _dive_tangential_speed(
                energy, dive_radius, params.r_jupiter_orbit, params.mu_sun, sign
            )
            - v_tangential
        )

    # The residual is concave in the unknown and zero is always on the correct
    # side of it: a prograde dive needs a small positive tangential speed, a
    # retrograde one a small negative speed, so zero brackets the physical root
    # against the far edge of the excess circle. No grid search is needed, which
    # matters because this is the innermost loop of every scan below.
    lo = params.v_jupiter_orbit - excess_out
    hi = params.v_jupiter_orbit + excess_out
    bracket = (0.0, hi) if sign > 0.0 else (lo, 0.0)
    if bracket[0] >= bracket[1]:
        return None
    low_value, high_value = residual(bracket[0]), residual(bracket[1])
    if not np.isfinite(low_value) or not np.isfinite(high_value):
        return None
    if low_value * high_value > 0.0:
        return None
    v_tangential = float(brentq(residual, bracket[0], bracket[1], xtol=1e-12))
    radial_squared = excess_out**2 - (v_tangential - params.v_jupiter_orbit) ** 2
    if radial_squared < 0.0:
        return None
    return v_tangential, -float(np.sqrt(radial_squared))


# --------------------------------------------------------------------------
# The three legs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _OutboundLeg:
    """Earth-to-Jupiter heliocentric leg.

    Attributes:
        tof: Time of flight (s).
        swept: Heliocentric longitude swept (rad, positive/prograde).
        v_tangential: Tangential speed at Jupiter's orbit radius (km/s).
        v_radial: Radial-outward speed there (km/s).
        arrival_excess: Jupiter-relative arrival excess speed (km/s).
    """

    tof: float
    swept: float
    v_tangential: float
    v_radial: float
    arrival_excess: float


def _outbound_leg(
    departure_excess: float, aim_deg: float, params: _FlybyParams
) -> Optional[_OutboundLeg]:
    """Earth departure to the first outbound crossing of Jupiter's orbit radius.

    The Earth-relative excess velocity is aimed at ``aim_deg`` from Earth's
    prograde direction (+90 deg is radially outward).  Aiming is free in
    propellant -- it only rotates the escape hyperbola (CONTEXT.md, "Free-aim
    departure") -- so it is a genuine second knob alongside the excess speed.

    Args:
        departure_excess: Earth-relative hyperbolic-excess speed (km/s).
        aim_deg: Aim angle from Earth's prograde direction (deg).
        params: Float parameter block.

    Returns:
        The :class:`_OutboundLeg`, or None if the transfer never reaches
        Jupiter's orbit radius outbound.
    """
    aim = np.radians(aim_deg)
    v_tangential = params.v_earth_orbit + departure_excess * float(np.cos(aim))
    v_radial = departure_excess * float(np.sin(aim))
    if v_tangential <= 0.0:
        return None
    try:
        crossings = [
            crossing
            for crossing in conic_radius_crossings(
                params.mu_sun,
                params.r_earth_orbit,
                v_tangential,
                v_radial,
                params.r_jupiter_orbit,
            )
            if crossing.outbound
        ]
    except ValueError:
        # Aphelion falls short of Jupiter's orbit: this departure never gets
        # there. The kernel's reachability guard is the signal, not an error.
        return None
    if not crossings:
        return None
    crossing = min(crossings, key=lambda item: item.tof)
    excess = float(
        np.hypot(
            crossing.v_tangential - params.v_jupiter_orbit,
            crossing.v_radial,
        )
    )
    return _OutboundLeg(
        tof=crossing.tof,
        swept=crossing.swept,
        v_tangential=crossing.v_tangential,
        v_radial=crossing.v_radial,
        arrival_excess=excess,
    )


@dataclass(frozen=True)
class _DiveLeg:
    """Jovian flyby onto the solar dive, and the fall to perihelion.

    Attributes:
        excess_in: Jupiter-relative arrival excess speed (km/s).
        excess_out: Jupiter-relative departure excess speed (km/s); equal to
            ``excess_in`` for an unpowered flyby.
        bend_required: Turn the flyby must deliver (rad).
        bend_available: Turn it can deliver at the perijove floor (rad).
        perijove_burn: Impulsive perijove burn implied by the excess change
            (km/s); zero for an unpowered flyby.
        tof: Fall time from Jupiter's orbit radius to perihelion (s).
        swept: Heliocentric longitude swept (rad; negative for a retrograde dive).
        energy: Specific energy of the dive orbit (km^2/s^2).
    """

    excess_in: float
    excess_out: float
    bend_required: float
    bend_available: float
    perijove_burn: float
    tof: float
    swept: float
    energy: float

    @property
    def bend_margin(self) -> float:
        """Available minus required bend (rad); negative means the flyby cannot turn far enough."""
        return self.bend_available - self.bend_required


def _dive_leg(
    outbound: _OutboundLeg,
    sign: float,
    params: _FlybyParams,
    dive_radius: float,
    excess_out: Optional[float] = None,
) -> Optional[_DiveLeg]:
    """Bend the arrival onto the solar dive and fall to perihelion.

    Args:
        outbound: The Earth-to-Jupiter leg that sets the arrival excess vector.
        sign: +1 prograde dive, -1 retrograde.
        params: Float parameter block.
        dive_radius: Target perihelion radius (km).
        excess_out: Outgoing excess speed (km/s).  ``None`` keeps the arrival
            speed, i.e. a strictly unpowered flyby; any other value implies a
            perijove burn and the bend becomes the split-hyperbola sum.

    Returns:
        The :class:`_DiveLeg`, or None if no dive state exists at this excess.
    """
    excess_in = outbound.arrival_excess
    w_out = excess_in if excess_out is None else excess_out
    state = _solve_dive_state(w_out, sign, dive_radius, params)
    if state is None:
        return None
    v_tangential, v_radial = state
    incoming = np.array(
        [outbound.v_tangential - params.v_jupiter_orbit, outbound.v_radial]
    )
    outgoing = np.array([v_tangential - params.v_jupiter_orbit, v_radial])
    cosine = float(np.dot(incoming, outgoing) / (excess_in * w_out))
    bend_required = float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    ecc_in = hyperbolic_eccentricity(
        params.mu_jupiter, params.periapsis_floor, excess_in
    )
    if excess_out is None:
        bend_available = unpowered_bend_angle(ecc_in)
        perijove_burn = 0.0
    else:
        ecc_out = hyperbolic_eccentricity(
            params.mu_jupiter, params.periapsis_floor, w_out
        )
        bend_available = half_turn_angle(ecc_in) + half_turn_angle(ecc_out)
        v_esc_perijove = float(
            np.sqrt(2.0 * params.mu_jupiter / params.periapsis_floor)
        )
        perijove_burn = abs(
            speed_with_escape_energy(w_out, v_esc_perijove)
            - speed_with_escape_energy(excess_in, v_esc_perijove)
        )

    conic = conic_state_at_radius(
        params.mu_sun, params.r_jupiter_orbit, abs(v_tangential), v_radial
    )
    tof_swept = _fall_to_periapsis(conic, params.mu_sun, params.r_jupiter_orbit)
    if tof_swept is None:
        return None
    tof, swept = tof_swept
    return _DiveLeg(
        excess_in=excess_in,
        excess_out=w_out,
        bend_required=bend_required,
        bend_available=bend_available,
        perijove_burn=perijove_burn,
        tof=tof,
        swept=sign * swept,
        energy=conic.energy,
    )


def _fall_to_periapsis(
    conic: ConicState, mu: float, r_start: float
) -> Optional[Tuple[float, float]]:
    """Time and swept angle from an inbound radius to the conic's periapsis.

    Args:
        conic: The dive orbit's conserved constants.
        mu: Gravitational parameter of the Sun (km^3/s^2).
        r_start: Radius the fall starts from (km), approached inbound.

    Returns:
        (time of flight in s, swept angle in rad, positive) or None if the
        starting radius is not on the conic.
    """
    nu = true_anomaly_at_radius_rad(conic.p, conic.ecc, r_start)
    if nu is None:
        return None
    if conic.ecc < 1.0:
        a = semimajor_axis_from_energy(mu, conic.energy)
        return elliptic_tof_seconds(mu, a, conic.ecc, nu), nu
    a_abs = abs(semimajor_axis_from_energy(mu, conic.energy))
    return hyperbolic_tof_seconds(mu, a_abs, conic.ecc, nu), nu


@dataclass(frozen=True)
class _ClimbLeg:
    """The periapsis boost and the escaping climb out to 1 AU.

    Attributes:
        periapsis_boost: Tangential PuffSat boost at the dive periapsis (km/s).
        periapsis_speed_in: Speed arriving at periapsis (km/s).
        periapsis_speed_out: Speed leaving it (km/s).
        tof: Climb time from periapsis to 1 AU (s).
        swept: Heliocentric longitude swept (rad; signed like the dive).
        v_tangential: Tangential speed at the 1 AU crossing (km/s).
        v_radial: Radial-outward speed there (km/s).
    """

    periapsis_boost: float
    periapsis_speed_in: float
    periapsis_speed_out: float
    tof: float
    swept: float
    v_tangential: float
    v_radial: float


def _climb_leg(
    dive: _DiveLeg,
    sign: float,
    params: _FlybyParams,
    dive_radius: float,
    return_excess: float,
) -> Optional[_ClimbLeg]:
    """Boost at the dive periapsis and climb out to Earth's orbit radius.

    The boost is tangential, so it is pure Oberth: the angular momentum it
    leaves is ``r_p * v_p``, which is what makes the return leg nearly radial at
    1 AU and fixes the whip-around to about 130 degrees.

    Args:
        dive: The dive leg whose energy sets the arriving periapsis speed.
        sign: +1 prograde, -1 retrograde.
        params: Float parameter block.
        dive_radius: Perihelion radius (km).
        return_excess: Hyperbolic-excess speed the boost must leave (km/s).

    Returns:
        The :class:`_ClimbLeg`, or None if the boosted orbit never reaches 1 AU.
    """
    v_escape = float(np.sqrt(2.0 * params.mu_sun / dive_radius))
    speed_out = speed_with_escape_energy(return_excess, v_escape)
    speed_in = float(np.sqrt(2.0 * (dive.energy + params.mu_sun / dive_radius)))
    conic = conic_state_at_radius(params.mu_sun, dive_radius, speed_out, 0.0)
    nu = true_anomaly_at_radius_rad(conic.p, conic.ecc, params.r_earth_orbit)
    if nu is None or conic.ecc <= 1.0:
        return None
    a_abs = abs(semimajor_axis_from_energy(params.mu_sun, conic.energy))
    tof = hyperbolic_tof_seconds(params.mu_sun, a_abs, conic.ecc, nu)
    v_tangential, v_radial = speed_components_at_radius(
        conic, params.mu_sun, params.r_earth_orbit
    )
    return _ClimbLeg(
        periapsis_boost=speed_out - speed_in,
        periapsis_speed_in=speed_in,
        periapsis_speed_out=speed_out,
        tof=tof,
        swept=sign * nu,
        v_tangential=sign * v_tangential,
        v_radial=v_radial,
    )


# --------------------------------------------------------------------------
# Closing the cycle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SynodicCycleClosure:
    """One Earth -> Jupiter -> dive -> Earth cycle that closes on a synodic multiple.

    Attributes:
        synodic_multiple: Number of Earth--Jupiter synodic periods in the cycle.
        departure_excess: Earth-relative hyperbolic-excess speed at departure (km/s).
        departure_aim_deg: Aim angle from Earth's prograde direction (deg).
        departure_speed_200km: Speed the departure needs at 200 km altitude (km/s),
            i.e. the collision's push target ``v_rf``.
        jupiter_arrival_excess: Jupiter-relative arrival excess speed (km/s).
        bend_required_deg: Turn the Jovian flyby must deliver (deg).
        bend_available_deg: Turn available at the perijove altitude floor (deg).
        perijove_burn: Perijove burn spent (km/s); zero when unpowered.
        outbound_tof_years: Earth-to-Jupiter flight time (yr).
        dive_tof_years: Jupiter-to-perihelion fall time (yr).
        climb_tof_days: Perihelion-to-Earth climb time (d).
        total_tof_years: Whole cycle (yr).
        intercept_miss_deg: Heliocentric angle between the 1 AU crossing and
            Earth; the Earth re-intercept residual, driven to zero by the solve.
        periapsis_boost: PuffSat boost at the dive periapsis (km/s).
        return_excess: Earth-*relative* excess speed of the arriving stream at
            the 1 AU crossing (km/s).  Larger than the heliocentric excess the
            boost bought, because the return is nearly radial and Earth's own
            29.8 km/s is almost entirely transverse to it.
        earth_closing_speed: Collision speed ``v_b`` at Earth, folded through
            Earth's surface escape speed (km/s).
        push_axis_deg: Direction the returning stream can push, measured from
            Earth's prograde direction (deg).
        aim_separation_deg: Angle between that push axis and the aim the next
            departure needs -- the cant an Earth-side nozzle must carry.
        earth_push_mass_ratio: Payload mass ratio of the Earth-side collision.
        feasible: True when the flyby can actually deliver the required bend.
    """

    synodic_multiple: int
    departure_excess: float
    departure_aim_deg: float
    departure_speed_200km: float
    jupiter_arrival_excess: float
    bend_required_deg: float
    bend_available_deg: float
    perijove_burn: float
    outbound_tof_years: float
    dive_tof_years: float
    climb_tof_days: float
    total_tof_years: float
    intercept_miss_deg: float
    periapsis_boost: float
    return_excess: float
    earth_closing_speed: float
    push_axis_deg: float
    aim_separation_deg: float
    earth_push_mass_ratio: float
    feasible: bool

    @property
    def bend_deficit_deg(self) -> float:
        """Required minus available bend (deg); positive means the flyby falls short."""
        return self.bend_required_deg - self.bend_available_deg


@dataclass(frozen=True)
class _CycleEvaluation:
    """Internal roll-up of the three legs at one (excess, aim) pair."""

    total_tof: float
    swept: float
    outbound: _OutboundLeg
    dive: _DiveLeg
    climb: _ClimbLeg


def _evaluate(
    departure_excess: float,
    aim_deg: float,
    params: _FlybyParams,
    dive_radius: float,
    return_excess: float,
    sign: float,
    excess_out: Optional[float] = None,
) -> Optional[_CycleEvaluation]:
    """Fly all three legs at one departure state.

    Args:
        departure_excess: Earth-relative excess speed (km/s).
        aim_deg: Aim angle (deg).
        params: Float parameter block.
        dive_radius: Perihelion radius (km).
        return_excess: Return-leg excess speed (km/s).
        sign: +1 prograde dive, -1 retrograde.
        excess_out: Outgoing Jovian excess (km/s); None keeps it unpowered.

    Returns:
        The :class:`_CycleEvaluation`, or None if any leg fails to exist.
    """
    outbound = _outbound_leg(departure_excess, aim_deg, params)
    if outbound is None:
        return None
    dive = _dive_leg(outbound, sign, params, dive_radius, excess_out)
    if dive is None:
        return None
    climb = _climb_leg(dive, sign, params, dive_radius, return_excess)
    if climb is None:
        return None
    return _CycleEvaluation(
        total_tof=outbound.tof + dive.tof + climb.tof,
        swept=outbound.swept + dive.swept + climb.swept,
        outbound=outbound,
        dive=dive,
        climb=climb,
    )


def _solve_clock(
    aim_deg: float,
    target_tof: float,
    params: _FlybyParams,
    dive_radius: float,
    return_excess: float,
    sign: float,
    excess_out: Optional[float] = None,
) -> Optional[float]:
    """Departure excess whose whole cycle takes exactly ``target_tof``.

    Total flight time falls monotonically with departure excess at fixed aim
    (a hotter departure shortens both the outbound leg and, through a hotter
    Jovian arrival, the dive), so the whole feasible range is bracketed by its
    two ends and one ``brentq`` finds the root.

    The lower end is *not* the bracket's floor: below a certain excess the
    transfer's aphelion falls short of Jupiter's orbit and no cycle exists at
    all.  Flight time rises steeply as that reachability floor is approached
    (the arrival goes tangential), so the floor is found by bisection rather
    than sampled -- a uniform grid straddles it and silently misses the longest
    cycles, which is exactly where the three-synodic root lives.

    Args:
        aim_deg: Aim angle (deg).
        target_tof: Cycle time to hit (s).
        params: Float parameter block.
        dive_radius: Perihelion radius (km).
        return_excess: Return-leg excess speed (km/s).
        sign: +1 prograde dive, -1 retrograde.
        excess_out: Outgoing Jovian excess (km/s); None keeps it unpowered.

    Returns:
        The departure excess (km/s), or None if no root lies in the bracket.
    """

    def flies(excess: float) -> bool:
        return (
            _evaluate(
                excess, aim_deg, params, dive_radius, return_excess, sign, excess_out
            )
            is not None
        )

    def residual(excess: float) -> float:
        evaluation = _evaluate(
            excess, aim_deg, params, dive_radius, return_excess, sign, excess_out
        )
        return float("nan") if evaluation is None else evaluation.total_tof - target_tof

    low, high = DEPARTURE_EXCESS_BRACKET
    if not flies(high):
        return None
    if flies(low):
        floor = low
    else:
        # Reachability is monotone in the excess speed, so bisect the boundary.
        lo, hi = low, high
        for _ in range(_FLOOR_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if flies(mid):
                hi = mid
            else:
                lo = mid
        floor = hi
    lower = residual(floor)
    upper = residual(high)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower * upper > 0.0:
        return None
    return float(brentq(residual, floor, high, xtol=1e-10))


def _intercept_miss(evaluation: _CycleEvaluation, params: _FlybyParams) -> float:
    """Heliocentric angle between the 1 AU crossing and Earth (rad).

    Earth advances at its own mean motion for the whole cycle; the projectile
    sweeps whatever the three legs sweep.  The residual is the difference,
    wrapped -- this is the **Earth re-intercept** condition, not merely crossing
    1 AU (CONTEXT.md, "Earth re-intercept").

    Args:
        evaluation: The flown cycle.
        params: Float parameter block.

    Returns:
        The signed miss angle (rad), in [-pi, pi).
    """
    earth_advance = (params.v_earth_orbit / params.r_earth_orbit) * evaluation.total_tof
    return wrap_pi(evaluation.swept - earth_advance)


def solve_synodic_closure(
    synodic_multiple: int,
    params: Optional[_FlybyParams] = None,
    dive_solar_radii: float = SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
    return_excess: float = DEFAULT_RETURN_EXCESS,
    sign: float = 1.0,
    excess_out: Optional[float] = None,
) -> Optional[SynodicCycleClosure]:
    """Solve the cycle for both closure conditions at a synodic multiple.

    Two conditions must hold together: the cycle takes exactly
    ``synodic_multiple`` synodic periods (so the next departure sees the same
    Earth--Jupiter phase), and the 1 AU crossing lands where Earth actually is.
    Two knobs meet them -- the departure excess speed and the free aim angle --
    so the solution is discrete, and the Jovian bend it demands is then an
    *output*, not something the search may choose.  Whether that demand is
    payable is what :attr:`SynodicCycleClosure.feasible` reports.

    Search bounds are ``DEPARTURE_EXCESS_BRACKET`` and ``DEPARTURE_AIM_BRACKET``.

    Args:
        synodic_multiple: Number of Earth--Jupiter synodic periods per cycle.
        params: Float parameter block; built with defaults when omitted.
        dive_solar_radii: Dive perihelion in solar radii.
        return_excess: Hyperbolic-excess speed of the return leg (km/s).
        sign: +1 for a prograde dive, -1 for a retrograde one.
        excess_out: Outgoing Jovian excess speed (km/s).  None -- the default --
            keeps the flyby strictly unpowered.

    Returns:
        The :class:`SynodicCycleClosure`, or None if no closure exists inside
        the recorded search bounds.

    Raises:
        ValueError: If ``synodic_multiple`` is not positive.
    """
    if synodic_multiple <= 0:
        raise ValueError("synodic_multiple must be positive")
    p = params if params is not None else _powered_flyby_params()
    dive_radius = _dive_periapsis_radius(dive_solar_radii)
    target = synodic_multiple * synodic_period(p)

    def miss_at_aim(aim_deg: float) -> Tuple[Optional[float], Optional[float]]:
        excess = _solve_clock(
            aim_deg, target, p, dive_radius, return_excess, sign, excess_out
        )
        if excess is None:
            return None, None
        evaluation = _evaluate(
            excess, aim_deg, p, dive_radius, return_excess, sign, excess_out
        )
        if evaluation is None:
            return None, None
        return excess, _intercept_miss(evaluation, p)

    aims = np.linspace(*DEPARTURE_AIM_BRACKET, _AIM_SAMPLES)
    previous: Optional[Tuple[float, float]] = None
    aim_root: Optional[float] = None
    for aim in aims:
        _, miss = miss_at_aim(float(aim))
        if miss is None:
            previous = None
            continue
        if previous is not None and previous[1] * miss < 0.0:
            aim_root = float(
                brentq(
                    lambda value: miss_at_aim(value)[1] or 0.0,
                    previous[0],
                    float(aim),
                    xtol=1e-9,
                )
            )
            break
        previous = (float(aim), miss)
    if aim_root is None:
        return None
    excess, _ = miss_at_aim(aim_root)
    if excess is None:
        return None
    evaluation = _evaluate(
        excess, aim_root, p, dive_radius, return_excess, sign, excess_out
    )
    if evaluation is None:
        return None
    return _as_closure(synodic_multiple, excess, aim_root, evaluation, p)


def _as_closure(
    synodic_multiple: int,
    departure_excess: float,
    aim_deg: float,
    evaluation: _CycleEvaluation,
    params: _FlybyParams,
) -> SynodicCycleClosure:
    """Project an internal evaluation onto the public closure record.

    Args:
        synodic_multiple: Number of synodic periods per cycle.
        departure_excess: Earth-relative excess speed (km/s).
        aim_deg: Aim angle (deg).
        evaluation: The flown cycle.
        params: Float parameter block.

    Returns:
        The :class:`SynodicCycleClosure`.
    """
    climb = evaluation.climb
    relative_tangential = climb.v_tangential - params.v_earth_orbit
    closing_excess = float(np.hypot(relative_tangential, climb.v_radial))
    push_axis = float(np.degrees(np.arctan2(climb.v_radial, relative_tangential)))
    v_b = speed_with_escape_energy(closing_excess, params.v_esc_surface)
    v_rf = speed_with_escape_energy(departure_excess, params.v_esc_leo)
    return SynodicCycleClosure(
        synodic_multiple=synodic_multiple,
        departure_excess=departure_excess,
        departure_aim_deg=aim_deg,
        departure_speed_200km=v_rf,
        jupiter_arrival_excess=evaluation.dive.excess_in,
        bend_required_deg=float(np.degrees(evaluation.dive.bend_required)),
        bend_available_deg=float(np.degrees(evaluation.dive.bend_available)),
        perijove_burn=evaluation.dive.perijove_burn,
        outbound_tof_years=evaluation.outbound.tof / _SECONDS_PER_YEAR,
        dive_tof_years=evaluation.dive.tof / _SECONDS_PER_YEAR,
        climb_tof_days=climb.tof / 86400.0,
        total_tof_years=evaluation.total_tof / _SECONDS_PER_YEAR,
        intercept_miss_deg=float(np.degrees(_intercept_miss(evaluation, params))),
        periapsis_boost=climb.periapsis_boost,
        return_excess=closing_excess,
        earth_closing_speed=v_b,
        push_axis_deg=push_axis,
        aim_separation_deg=abs(push_axis - aim_deg),
        # payload_mass_ratio is annotated -> float but returns a dimensionless
        # Quantity in practice; coerce so the dataclass really holds a float.
        earth_push_mass_ratio=float(
            payload_mass_ratio(
                v_rf * u.km / u.s, v_b * u.km / u.s, fudge_factor=STD_FUDGE_FACTOR
            )
        ),
        feasible=evaluation.dive.bend_margin >= 0.0,
    )


# --------------------------------------------------------------------------
# Why two synodic periods does not close
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PerijoveBurnProbe:
    """One point on the powered-perijove sweep at a fixed synodic multiple.

    Attributes:
        excess_out: Outgoing Jupiter-relative excess speed (km/s).
        excess_in: Arrival excess speed the clock then demands (km/s).
        departure_excess: Earth departure excess (km/s).
        departure_aim_deg: Aim angle (deg).
        perijove_burn: Perijove burn implied by the excess change (km/s).
        bend_deficit_deg: Required minus available bend (deg); positive is short.
    """

    excess_out: float
    excess_in: float
    departure_excess: float
    departure_aim_deg: float
    perijove_burn: float
    bend_deficit_deg: float


def perijove_burn_sweep(
    synodic_multiple: int = 2,
    params: Optional[_FlybyParams] = None,
    dive_solar_radii: float = SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
    return_excess: float = DEFAULT_RETURN_EXCESS,
) -> List[PerijoveBurnProbe]:
    """Sweep the perijove burn and read the bend deficit that survives it.

    This is the load-bearing negative result for the two-synodic cycle.  A
    perijove burn is the obvious second knob -- it is what breaks the ``v_b``
    lottery on the retrograde return (CONTEXT.md) -- but here it cannot close
    the bend at *any* magnitude or sign, because the clock is what binds:
    braking lengthens the dive, so holding the cycle time forces a hotter
    outbound leg and a hotter Jovian arrival, which raises the bend demand right
    back; accelerating shortens the dive but inflates the outgoing eccentricity,
    which eats the outgoing half-angle.  The deficit therefore has an interior
    minimum above zero.

    Search bounds are ``PERIJOVE_EXCESS_BRACKET`` at ``_PERIJOVE_SAMPLES`` points.

    Args:
        synodic_multiple: Number of synodic periods per cycle.
        params: Float parameter block; built with defaults when omitted.
        dive_solar_radii: Dive perihelion in solar radii.
        return_excess: Return-leg excess speed (km/s).

    Returns:
        One :class:`PerijoveBurnProbe` per outgoing-excess sample that admits a
        closure of the clock and the Earth intercept.
    """
    p = params if params is not None else _powered_flyby_params()
    probes: List[PerijoveBurnProbe] = []
    for excess_out in np.linspace(*PERIJOVE_EXCESS_BRACKET, _PERIJOVE_SAMPLES):
        closure = solve_synodic_closure(
            synodic_multiple,
            params=p,
            dive_solar_radii=dive_solar_radii,
            return_excess=return_excess,
            excess_out=float(excess_out),
        )
        if closure is None:
            continue
        probes.append(
            PerijoveBurnProbe(
                excess_out=float(excess_out),
                excess_in=closure.jupiter_arrival_excess,
                departure_excess=closure.departure_excess,
                departure_aim_deg=closure.departure_aim_deg,
                perijove_burn=closure.perijove_burn,
                bend_deficit_deg=closure.bend_deficit_deg,
            )
        )
    return probes


def _signed_time_from_periapsis(conic: ConicState, mu: float, nu: float) -> float:
    """Time from periapsis to a signed true anomaly on either conic type.

    Negative ``nu`` is the inbound branch, and its time is the mirror of the
    outbound one, so the sign carries straight through.

    Args:
        conic: The orbit's conserved constants.
        mu: Gravitational parameter (km^3/s^2).
        nu: Signed true anomaly (rad), magnitude below pi.

    Returns:
        Signed time from periapsis passage (s).
    """
    magnitude = abs(nu)
    if conic.ecc < 1.0:
        a = semimajor_axis_from_energy(mu, conic.energy)
        elapsed = elliptic_tof_seconds(mu, a, conic.ecc, magnitude)
    else:
        a_abs = abs(semimajor_axis_from_energy(mu, conic.energy))
        elapsed = hyperbolic_tof_seconds(mu, a_abs, conic.ecc, magnitude)
    return float(np.sign(nu)) * elapsed


def _true_anomaly_after(
    conic: ConicState, mu: float, nu_start: float, nu_end: float, elapsed: float
) -> float:
    """True anomaly reached ``elapsed`` seconds after ``nu_start``.

    Kepler's equation inverted by bisection between the two known anomalies --
    the conic kernel gives time from anomaly, and this leg needs the reverse.

    Args:
        conic: The orbit's conserved constants.
        mu: Gravitational parameter (km^3/s^2).
        nu_start: Signed true anomaly at the start (rad).
        nu_end: Signed true anomaly bounding the search (rad), ahead of the start.
        elapsed: Time after the start (s).

    Returns:
        The signed true anomaly (rad).
    """
    base = _signed_time_from_periapsis(conic, mu, nu_start)
    return float(
        brentq(
            lambda nu: _signed_time_from_periapsis(conic, mu, nu) - base - elapsed,
            nu_start,
            nu_end,
            xtol=1e-12,
        )
    )


def _outbound_leg_with_dsm(
    departure_excess: float,
    aim_deg: float,
    params: _FlybyParams,
    leg_fraction: float,
    dv_radial: float,
    dv_tangential: float,
) -> Optional[Tuple[_OutboundLeg, float]]:
    """Earth-to-Jupiter leg interrupted by a deep-space maneuver.

    The maneuver fires at ``leg_fraction`` of the *un-maneuvered* leg's flight
    time, so its heliocentric radius is a reported output rather than a knob --
    which is what lets the caller keep it outside Jupiter's sphere of influence
    and call it a DSM honestly.

    Args:
        departure_excess: Earth-relative excess speed (km/s).
        aim_deg: Aim angle (deg).
        params: Float parameter block.
        leg_fraction: Fraction of the un-maneuvered leg time at which to burn.
        dv_radial: Radial-outward component of the maneuver (km/s).
        dv_tangential: Tangential component of the maneuver (km/s).

    Returns:
        (the resulting leg, the maneuver's heliocentric radius in km), or None
        if either half of the leg does not exist.
    """
    aim = np.radians(aim_deg)
    v_tangential = params.v_earth_orbit + departure_excess * float(np.cos(aim))
    v_radial = departure_excess * float(np.sin(aim))
    if v_tangential <= 0.0:
        return None
    first = conic_state_at_radius(
        params.mu_sun, params.r_earth_orbit, v_tangential, v_radial
    )
    nu_earth = true_anomaly_at_radius_rad(first.p, first.ecc, params.r_earth_orbit)
    nu_jupiter = true_anomaly_at_radius_rad(first.p, first.ecc, params.r_jupiter_orbit)
    if nu_earth is None or nu_jupiter is None:
        return None
    if v_radial < 0.0:
        nu_earth = -nu_earth
    if nu_jupiter <= nu_earth:
        return None
    full_tof = _signed_time_from_periapsis(
        first, params.mu_sun, nu_jupiter
    ) - _signed_time_from_periapsis(first, params.mu_sun, nu_earth)
    if full_tof <= 0.0:
        return None
    nu_burn = _true_anomaly_after(
        first, params.mu_sun, nu_earth, nu_jupiter, leg_fraction * full_tof
    )
    r_burn = first.p / (1.0 + first.ecc * float(np.cos(nu_burn)))
    try:
        v_t_burn, v_r_burn = speed_components_at_radius(first, params.mu_sun, r_burn)
    except ValueError:
        return None
    v_r_burn *= float(np.sign(nu_burn)) if nu_burn != 0.0 else 1.0
    second = conic_state_at_radius(
        params.mu_sun, r_burn, v_t_burn + dv_tangential, v_r_burn + dv_radial
    )
    if second.h <= 0.0:
        return None
    nu_after = true_anomaly_at_radius_rad(second.p, second.ecc, r_burn)
    nu_arrive = true_anomaly_at_radius_rad(second.p, second.ecc, params.r_jupiter_orbit)
    if nu_after is None or nu_arrive is None:
        return None
    if v_r_burn + dv_radial < 0.0:
        nu_after = -nu_after
    if nu_arrive <= nu_after:
        return None
    second_tof = _signed_time_from_periapsis(
        second, params.mu_sun, nu_arrive
    ) - _signed_time_from_periapsis(second, params.mu_sun, nu_after)
    if second_tof <= 0.0:
        return None
    try:
        arrival_t, arrival_r = speed_components_at_radius(
            second, params.mu_sun, params.r_jupiter_orbit
        )
    except ValueError:
        return None
    leg = _OutboundLeg(
        tof=leg_fraction * full_tof + second_tof,
        swept=(nu_burn - nu_earth) + (nu_arrive - nu_after),
        v_tangential=arrival_t,
        v_radial=arrival_r,
        arrival_excess=float(np.hypot(arrival_t - params.v_jupiter_orbit, arrival_r)),
    )
    return leg, r_burn


@dataclass(frozen=True)
class DsmCorrection:
    """The cheapest deep-space maneuver that closes a cycle the bend cannot.

    Attributes:
        magnitude: Maneuver size (km/s).
        radius_au: Heliocentric radius at which it fires (AU).
        departure_excess: Earth departure excess it goes with (km/s).
        departure_aim_deg: Aim angle it goes with (deg).
        jupiter_arrival_excess: Resulting Jupiter arrival excess (km/s).
        bend_required_deg: Bend the flyby then needs (deg).
        bend_available_deg: Bend available at the perijove floor (deg).
    """

    magnitude: float
    radius_au: float
    departure_excess: float
    departure_aim_deg: float
    jupiter_arrival_excess: float
    bend_required_deg: float
    bend_available_deg: float


def minimum_dsm_correction(
    synodic_multiple: int = 2,
    params: Optional[_FlybyParams] = None,
    dive_solar_radii: float = SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
    return_excess: float = DEFAULT_RETURN_EXCESS,
    leg_fraction: float = DSM_LEG_FRACTION,
) -> Optional[DsmCorrection]:
    """Cheapest mid-course maneuver that closes clock, intercept and bend together.

    Three unknowns -- departure excess and the two maneuver components -- meet
    three conditions at each aim: the cycle takes exactly ``synodic_multiple``
    synodic periods, the 1 AU crossing lands on Earth, and the Jovian bend sits
    exactly at what the perijove floor can deliver.  Scanning the aim then picks
    the cheapest.

    Search bounds are ``DEPARTURE_AIM_BRACKET`` at ``_DSM_AIM_SAMPLES`` points,
    with the maneuver fixed at ``leg_fraction`` of the leg.

    Args:
        synodic_multiple: Number of synodic periods per cycle.
        params: Float parameter block; built with defaults when omitted.
        dive_solar_radii: Dive perihelion in solar radii.
        return_excess: Return-leg excess speed (km/s).
        leg_fraction: Where on the outbound leg the maneuver fires.

    Returns:
        The cheapest :class:`DsmCorrection` found, or None if none closes.
    """
    p = params if params is not None else _powered_flyby_params()
    dive_radius = _dive_periapsis_radius(dive_solar_radii)
    target = synodic_multiple * synodic_period(p)
    earth_rate = p.v_earth_orbit / p.r_earth_orbit

    def residuals(
        unknowns: "np.ndarray[Tuple[int], np.dtype[np.float64]]", aim: float
    ) -> List[float]:
        excess, dv_radial, dv_tangential = (float(value) for value in unknowns)
        built = _outbound_leg_with_dsm(
            excess, aim, p, leg_fraction, dv_radial, dv_tangential
        )
        if built is None:
            return [1.0e3, 1.0e3, 1.0e3]
        outbound, _ = built
        dive = _dive_leg(outbound, 1.0, p, dive_radius, None)
        if dive is None:
            return [1.0e3, 1.0e3, 1.0e3]
        climb = _climb_leg(dive, 1.0, p, dive_radius, return_excess)
        if climb is None:
            return [1.0e3, 1.0e3, 1.0e3]
        total = outbound.tof + dive.tof + climb.tof
        swept = outbound.swept + dive.swept + climb.swept
        return [
            (total - target) / _SECONDS_PER_YEAR,
            wrap_pi(swept - earth_rate * total),
            dive.bend_required - dive.bend_available,
        ]

    def solve_at_aim(aim: float) -> Optional[DsmCorrection]:
        cheapest: Optional[DsmCorrection] = None
        for guess in ((12.7, -1.4, -1.3), (11.0, 0.5, 0.5), (15.0, -0.5, 1.0)):
            solution, _, flag, _ = fsolve(
                residuals, np.array(guess), args=(aim,), full_output=True
            )
            if flag != 1:
                continue
            if max(abs(value) for value in residuals(solution, aim)) > 1.0e-9:
                continue
            excess = float(solution[0])
            if not DEPARTURE_EXCESS_BRACKET[0] <= excess <= DEPARTURE_EXCESS_BRACKET[1]:
                continue
            magnitude = float(np.hypot(solution[1], solution[2]))
            if cheapest is not None and magnitude >= cheapest.magnitude:
                continue
            built = _outbound_leg_with_dsm(
                excess, aim, p, leg_fraction, float(solution[1]), float(solution[2])
            )
            if built is None:
                continue
            outbound, r_burn = built
            dive = _dive_leg(outbound, 1.0, p, dive_radius, None)
            if dive is None:
                continue
            cheapest = DsmCorrection(
                magnitude=magnitude,
                radius_au=r_burn / float((1 * u.AU).to_value(u.km)),
                departure_excess=excess,
                departure_aim_deg=aim,
                jupiter_arrival_excess=outbound.arrival_excess,
                bend_required_deg=float(np.degrees(dive.bend_required)),
                bend_available_deg=float(np.degrees(dive.bend_available)),
            )
        return cheapest

    aims = np.linspace(*DEPARTURE_AIM_BRACKET, _DSM_AIM_SAMPLES)
    best: Optional[DsmCorrection] = None
    for aim in aims:
        candidate = solve_at_aim(float(aim))
        if candidate is not None and (
            best is None or candidate.magnitude < best.magnitude
        ):
            best = candidate
    if best is None:
        return None
    # The coarse aim grid straddles a sharp minimum -- the DSM price roughly
    # doubles a few degrees either side of it -- so refine locally rather than
    # quoting a grid point as the optimum.
    step = float(aims[1] - aims[0])
    for aim in np.linspace(
        best.departure_aim_deg - step,
        best.departure_aim_deg + step,
        _DSM_REFINE_SAMPLES,
    ):
        candidate = solve_at_aim(float(aim))
        if candidate is not None and candidate.magnitude < best.magnitude:
            best = candidate
    return best


def soi_seam_correction(closure: SynodicCycleClosure) -> float:
    """Delta-v to rotate the outgoing excess through a leftover bend deficit (km/s).

    ADR 0011's convention: an exact velocity match at the Jupiter
    patched-conic/SOI seam after the unpowered bend.  Rotating a vector of
    length ``w`` by an angle ``d`` costs the chord ``2*w*sin(d/2)``.  It
    establishes the correction *scale*; it is not a finite-location
    interplanetary DSM optimum.

    Args:
        closure: A closure whose bend the flyby could not deliver.

    Returns:
        The seam correction (km/s); zero when the closure is already feasible.
    """
    deficit = np.radians(closure.bend_deficit_deg)
    if deficit <= 0.0:
        return 0.0
    return float(2.0 * closure.jupiter_arrival_excess * np.sin(deficit / 2.0))


# --------------------------------------------------------------------------
# The departure nozzle, and what the cycle grows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DepartureNozzleLedger:
    """A magnetic nozzle catching the returning stream to fly the Earth departure.

    Attributes:
        slug_ratio: Kilograms of carried slug per kilogram of arriving impactor.
        jet_energy_efficiency: Fraction of collision energy reaching coherent
            axial exhaust momentum (the paper's ``eta_jet**2``).
        thrust_axis_deg: Direction of the periapsis velocity the burn adds to,
            from Earth's prograde direction.  The **departure-hyperbola mirror**
            makes this the outgoing aim plus *or* minus ``arcsin(1/e)``, and the
            sign is free, so the ledger takes whichever delivers more.
        impact_angle_start_deg: Vehicle-frame impact angle as the burn lights.
        impact_angle_end_deg: The same as it finishes.
        closing_speed_start: Impactor speed relative to the moving vehicle at
            the start of the burn (km/s).
        closing_speed_end: The same at the end (km/s).
        effective_exhaust_speed: Slug-charged exhaust speed at mid-burn (km/s).
        specific_impulse: That speed as an Isp (s).
        delta_v: Burn the departure needs (km/s).
        delivered_fraction: Mass fraction surviving the burn.
        impactor_fraction: Impactor kilograms consumed per departing kilogram.
    """

    slug_ratio: float
    jet_energy_efficiency: float
    thrust_axis_deg: float
    impact_angle_start_deg: float
    impact_angle_end_deg: float
    closing_speed_start: float
    closing_speed_end: float
    effective_exhaust_speed: float
    specific_impulse: float
    delta_v: float
    delivered_fraction: float
    impactor_fraction: float

    @property
    def slug_per_delivered_kg(self) -> float:
        """Kilograms of slug spent per kilogram actually placed on the transfer."""
        return (1.0 - self.delivered_fraction) / self.delivered_fraction

    @property
    def impactor_per_delivered_kg(self) -> float:
        """Kilograms of arriving impactor consumed per kilogram placed on the transfer."""
        return self.impactor_fraction / self.delivered_fraction


def _vehicle_frame_impact(
    stream_axis_deg: float,
    thrust_axis_deg: float,
    stream_speed: float,
    vehicle_speed: float,
) -> Tuple[float, float]:
    """Impact angle and closing speed seen by a vehicle moving along its thrust axis.

    The **aim separation** measured between excess vectors at the patched-conic
    boundary is a diagnostic, *not* what the impulse law consumes (CONTEXT.md):
    the law wants the angle at the burn point relative to the moving vehicle.
    Subtracting the vehicle's velocity is the whole difference, and it is worth
    tens of degrees.

    Args:
        stream_axis_deg: Direction the impactor stream travels, from Earth's
            prograde direction.
        thrust_axis_deg: Direction the vehicle's periapsis velocity points.
        stream_speed: Impactor speed at the burn radius (km/s).
        vehicle_speed: Vehicle speed at that instant (km/s).

    Returns:
        (impact angle in rad from the thrust axis, closing speed in km/s).
    """
    relative = np.radians(stream_axis_deg - thrust_axis_deg)
    along = stream_speed * float(np.cos(relative)) - vehicle_speed
    across = stream_speed * float(np.sin(relative))
    return float(np.arctan2(abs(across), along)), float(np.hypot(along, across))


def departure_nozzle_ledger(
    departure_excess: float,
    departure_aim_deg: float,
    stream_excess: float,
    stream_axis_deg: float,
    slug_ratio: float,
    jet_energy_efficiency: float,
    params: Optional[_FlybyParams] = None,
) -> DepartureNozzleLedger:
    """Fly the Earth departure on a magnetic nozzle fed by the returning stream.

    The vehicle carries slug at ``slug_ratio``; the arriving stream vaporises it
    and the field collimates the plume.  Only the slug is drawn from the vehicle,
    so the rocket equation is charged against it -- but both the closing speed
    and the impact angle move as the vehicle accelerates, so it is integrated
    over ``_BURN_INTEGRATION_STEPS`` rather than evaluated once.

    Args:
        departure_excess: Earth-relative excess speed the departure must reach (km/s).
        departure_aim_deg: Direction that excess must point, from Earth's prograde.
        stream_excess: Earth-relative excess speed of the arriving stream (km/s).
        stream_axis_deg: Direction the stream travels, from Earth's prograde.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`DepartureNozzleLedger` for the better mirror sign.
    """
    p = params if params is not None else _powered_flyby_params()
    speed_end = speed_with_escape_energy(departure_excess, p.v_esc_leo)
    stream_speed = speed_with_escape_energy(stream_excess, p.v_esc_leo)
    mirror = float(
        np.degrees(
            half_turn_angle(
                hyperbolic_eccentricity(_MU_EARTH, _BURN_RADIUS, departure_excess)
            )
        )
    )
    speeds = np.linspace(p.v_depart_from, speed_end, _BURN_INTEGRATION_STEPS + 1)
    best: Optional[DepartureNozzleLedger] = None
    for sign in (1.0, -1.0):
        axis = departure_aim_deg + sign * mirror
        log_fraction = 0.0
        for lower, upper in zip(speeds[:-1], speeds[1:]):
            angle, closing = _vehicle_frame_impact(
                stream_axis_deg, axis, stream_speed, 0.5 * (float(lower) + float(upper))
            )
            beta = impulse_per_impactor_kg(
                angle * u.rad, slug_ratio, jet_energy_efficiency
            )
            if beta <= 0.0:
                log_fraction = -np.inf
                break
            log_fraction -= (float(upper) - float(lower)) / (
                beta * closing / slug_ratio
            )
        delivered = float(np.exp(log_fraction))
        start_angle, start_speed = _vehicle_frame_impact(
            stream_axis_deg, axis, stream_speed, p.v_depart_from
        )
        end_angle, end_speed = _vehicle_frame_impact(
            stream_axis_deg, axis, stream_speed, speed_end
        )
        mid_angle, mid_speed = _vehicle_frame_impact(
            stream_axis_deg, axis, stream_speed, 0.5 * (p.v_depart_from + speed_end)
        )
        exhaust = (
            impulse_per_impactor_kg(
                mid_angle * u.rad, slug_ratio, jet_energy_efficiency
            )
            * mid_speed
            / slug_ratio
        )
        candidate = DepartureNozzleLedger(
            slug_ratio=slug_ratio,
            jet_energy_efficiency=jet_energy_efficiency,
            thrust_axis_deg=axis,
            impact_angle_start_deg=float(np.degrees(start_angle)),
            impact_angle_end_deg=float(np.degrees(end_angle)),
            closing_speed_start=start_speed,
            closing_speed_end=end_speed,
            effective_exhaust_speed=exhaust,
            specific_impulse=exhaust / _STANDARD_GRAVITY,
            delta_v=speed_end - p.v_depart_from,
            delivered_fraction=delivered,
            impactor_fraction=(1.0 - delivered) / slug_ratio,
        )
        if best is None or candidate.delivered_fraction > best.delivered_fraction:
            best = candidate
    assert best is not None
    return best


def methalox_per_delivered_kg(
    delta_v: float, params: Optional[_FlybyParams] = None
) -> float:
    """Propellant a methalox stage would burn for the same delta-v, per delivered kg.

    The comparison that keeps the slug ratio in perspective.  ``k`` = 30 sounds
    extravagant until it is read in the right units: it is slug per *impactor*
    kilogram, and impactors are well under a percent of the vehicle.  What the
    stage actually spends is the slug, and against a 380 s chemical stage flying
    the same burn it spends an order of magnitude less.

    Args:
        delta_v: Burn to fly (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        Kilograms of methalox per kilogram delivered.
    """
    p = params if params is not None else _powered_flyby_params()
    return float(np.exp(delta_v / p.exhaust_speed) - 1.0)


@dataclass(frozen=True)
class CycleGrowthLedger:
    """What one cycle multiplies, counting only impactor kilograms as precious.

    This scores the architecture under the assumption that **mass lifted off
    Earth is free**, so the slug is not charged and the only scarce input is the
    returning stream.  That assumption is doing heavy lifting -- the launched
    slug grows in lockstep with the payload, so the constraint moves to the pad
    rather than disappearing (see ``return_per_impactor_kg`` against
    ``launched_slug_per_impactor_kg``). CONTEXT.md's **launch ledger** is the
    accounting this deliberately switches off.

    Attributes:
        label: Which cycle this is.
        departure_excess: Earth-relative departure excess (km/s).
        departure_aim_deg: Direction it must point, from Earth's prograde.
        cycle_years: Departure-to-return time (yr).
        nozzle: The departure burn's ledger.
        periapsis_survival: Mass fraction surviving the solar-periapsis
            collision, both streams counted.
        round_trip_fraction: Departing mass that comes home.
        return_per_impactor_kg: Returning kilograms per impactor kilogram spent.
        launched_slug_per_impactor_kg: Slug that had to be lofted, per the same.
        growth_rate: Natural-log growth per year (e-foldings/yr).
        doubling_years: Payload doubling time (yr).
        millionfold_years: Time to a millionfold (yr).
    """

    label: str
    departure_excess: float
    departure_aim_deg: float
    cycle_years: float
    nozzle: DepartureNozzleLedger
    periapsis_survival: float
    round_trip_fraction: float
    return_per_impactor_kg: float
    launched_slug_per_impactor_kg: float
    growth_rate: float
    doubling_years: float
    millionfold_years: float


def cycle_growth_ledger(
    label: str,
    departure_excess: float,
    departure_aim_deg: float,
    cycle_years: float,
    stream_excess: float,
    stream_axis_deg: float,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    periapsis_survival: float = DEFAULT_PERIAPSIS_SURVIVAL,
    params: Optional[_FlybyParams] = None,
) -> CycleGrowthLedger:
    """Score a cycle on returning mass per impactor kilogram spent.

    One impactor kilogram buys ``k`` kilograms of spent slug, which flies
    ``k*f/(1-f)`` kilograms of vehicle through the departure; the solar-periapsis
    collision then keeps ``periapsis_survival`` of it.

    Args:
        label: Name for the cycle.
        departure_excess: Earth-relative departure excess (km/s).
        departure_aim_deg: Direction it must point, from Earth's prograde.
        cycle_years: Departure-to-return time (yr).
        stream_excess: Earth-relative excess of the arriving stream (km/s).
        stream_axis_deg: Direction the stream travels, from Earth's prograde.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        periapsis_survival: Mass fraction surviving the 4 solar-radii collision.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`CycleGrowthLedger`.

    Raises:
        ValueError: If the departure burn delivers nothing, so no growth exists.
    """
    nozzle = departure_nozzle_ledger(
        departure_excess,
        departure_aim_deg,
        stream_excess,
        stream_axis_deg,
        slug_ratio,
        jet_energy_efficiency,
        params,
    )
    delivered = nozzle.delivered_fraction
    if delivered <= 0.0 or delivered >= 1.0:
        raise ValueError("departure burn delivers no usable fraction")
    growth = periapsis_survival * delivered * slug_ratio / (1.0 - delivered)
    return CycleGrowthLedger(
        label=label,
        departure_excess=departure_excess,
        departure_aim_deg=departure_aim_deg,
        cycle_years=cycle_years,
        nozzle=nozzle,
        periapsis_survival=periapsis_survival,
        round_trip_fraction=delivered * periapsis_survival,
        return_per_impactor_kg=growth,
        launched_slug_per_impactor_kg=slug_ratio,
        growth_rate=float(np.log(growth) / cycle_years),
        doubling_years=float(cycle_years * np.log(2.0) / np.log(growth)),
        millionfold_years=float(cycle_years * np.log(1.0e6) / np.log(growth)),
    )


def paper_resonant_dive_ledger(
    stream_excess: float,
    stream_axis_deg: float,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    periapsis_survival: float = DEFAULT_PERIAPSIS_SURVIVAL,
    params: Optional[_FlybyParams] = None,
) -> CycleGrowthLedger:
    """Score the paper's own single-impulse resonant dive on the same nozzle.

    The comparison this cycle has to beat.  Its departure state is *derived* from
    :func:`single_impulse_resonant_dive` rather than quoted, so the two rows sit
    on one device and one set of constants.

    Args:
        stream_excess: Earth-relative excess of the arriving stream (km/s).
        stream_axis_deg: Direction the stream travels, from Earth's prograde.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        periapsis_survival: Mass fraction surviving the 4 solar-radii collision.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`CycleGrowthLedger` for the paper's cycle.
    """
    dive = single_impulse_resonant_dive()
    retrograde = float(dive.retrograde_component.to_value(u.km / u.s))
    radial = float(dive.radial_component.to_value(u.km / u.s))
    return cycle_growth_ledger(
        "paper single-impulse resonant dive",
        float(dive.earth_boost.to_value(u.km / u.s)),
        float(np.degrees(np.arctan2(radial, -retrograde))),
        float(dive.reintercept_time.to_value(u.year)),
        stream_excess,
        stream_axis_deg,
        slug_ratio,
        jet_energy_efficiency,
        periapsis_survival,
        params,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _closure_rows(closure: SynodicCycleClosure) -> List[List[str]]:
    """Format one closure as labelled rows for the CLI table.

    Args:
        closure: The closure to render.

    Returns:
        Rows of (quantity, value) strings.
    """
    return [
        ["cycle", f"{closure.synodic_multiple}S = {closure.total_tof_years:.4f} yr"],
        [
            "Earth departure excess (v_inf)",
            f"{closure.departure_excess:.3f} km/s",
        ],
        [
            "  -> speed at 200 km altitude",
            f"{closure.departure_speed_200km:.3f} km/s",
        ],
        [
            "  -> C3",
            f"{closure.departure_excess ** 2:.1f} km^2/s^2",
        ],
        ["departure aim from prograde", f"{closure.departure_aim_deg:+.2f} deg"],
        ["Jupiter arrival excess", f"{closure.jupiter_arrival_excess:.3f} km/s"],
        [
            "Jovian bend required / available",
            f"{closure.bend_required_deg:.2f} / {closure.bend_available_deg:.2f} deg",
        ],
        [
            "  -> margin",
            f"{-closure.bend_deficit_deg:+.2f} deg "
            f"({'FEASIBLE' if closure.feasible else 'INFEASIBLE'})",
        ],
        ["perijove burn", f"{closure.perijove_burn:.4f} km/s"],
        [
            "legs (E->J / J->perihelion / perihelion->E)",
            f"{closure.outbound_tof_years:.3f} yr / "
            f"{closure.dive_tof_years:.3f} yr / {closure.climb_tof_days:.2f} d",
        ],
        ["Earth re-intercept miss", f"{closure.intercept_miss_deg:+.4f} deg"],
        ["periapsis boost", f"{closure.periapsis_boost:.2f} km/s"],
        ["return excess at Earth", f"{closure.return_excess:.2f} km/s"],
        ["collision speed v_b", f"{closure.earth_closing_speed:.2f} km/s"],
        [
            "push axis / next aim / separation",
            f"{closure.push_axis_deg:.2f} / {closure.departure_aim_deg:+.2f} / "
            f"{closure.aim_separation_deg:.2f} deg",
        ],
        ["Earth-push payload mass ratio", f"{closure.earth_push_mass_ratio:.2f}"],
    ]


def main() -> None:
    """Print the two-versus-three synodic verdict and the closure details."""
    parser = argparse.ArgumentParser(
        description=(
            "Earth -> Jupiter -> 4 solar-radii Oberth -> Earth, on a synodic clock."
        )
    )
    parser.add_argument(
        "--return-excess",
        type=float,
        default=DEFAULT_RETURN_EXCESS,
        help="hyperbolic-excess speed of the return leg (km/s)",
    )
    parser.add_argument(
        "--dive-solar-radii",
        type=float,
        default=SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
        help="dive perihelion, in solar radii",
    )
    parser.add_argument(
        "--slug-ratio",
        type=float,
        default=DEFAULT_SLUG_RATIO,
        help="kilograms of carried slug per kilogram of arriving impactor",
    )
    parser.add_argument(
        "--jet-efficiency",
        type=float,
        default=DEFAULT_JET_ENERGY_EFFICIENCY,
        help="eta_jet^2: fraction of collision energy reaching axial exhaust momentum",
    )
    parser.add_argument(
        "--periapsis-survival",
        type=float,
        default=DEFAULT_PERIAPSIS_SURVIVAL,
        help="mass fraction surviving the 4 solar-radii collision, both streams",
    )
    args = parser.parse_args()
    params = _powered_flyby_params()
    period = synodic_period(params) / _SECONDS_PER_YEAR

    print("=" * 78)
    print("Jovian solar-dive cycle: Earth -> Jupiter -> dive -> Earth")
    print("=" * 78)
    print(
        f"Earth-Jupiter synodic period {period:.4f} yr; "
        f"2S = {2 * period:.4f} yr, 3S = {3 * period:.4f} yr"
    )
    print(
        f"dive perihelion {args.dive_solar_radii:.2f} solar radii; "
        f"perijove floor {LOW_JUPITER_ALTITUDE.to_value(u.km):.0f} km altitude; "
        f"return excess {args.return_excess:.1f} km/s"
    )

    print("\n-- Jupiter arrival excess needed to place the dive at all --")
    print(
        tabulate(
            [
                [
                    label,
                    f"{dive_placement_excess_floor(sign, params=params):.4f} km/s",
                ]
                for label, sign in (
                    ("prograde dive", 1.0),
                    ("radial plunge", 0.0),
                    ("retrograde dive", -1.0),
                )
            ],
            headers=["placement", "minimum arrival v_inf"],
            tablefmt="github",
        )
    )
    print(
        "A flyby rotates the excess but never rescales it, so these are necessary\n"
        "conditions, not costs: below them the dive does not exist at any perijove."
    )

    for multiple in (2, 3):
        closure = solve_synodic_closure(
            multiple,
            params=params,
            dive_solar_radii=args.dive_solar_radii,
            return_excess=args.return_excess,
        )
        print(f"\n-- {multiple} synodic periods, unpowered flyby --")
        if closure is None:
            print("no closure inside the recorded search bounds")
            continue
        print(tabulate(_closure_rows(closure), tablefmt="github"))
        if not closure.feasible:
            print(
                f"SOI-seam correction to buy the missing bend: "
                f"{soi_seam_correction(closure) * 1000:.0f} m/s"
            )

    print("\n-- can a perijove burn buy the 2S bend? --")
    probes = perijove_burn_sweep(
        2,
        params=params,
        dive_solar_radii=args.dive_solar_radii,
        return_excess=args.return_excess,
    )
    if not probes:
        print("no 2S closures at any outgoing excess in the sweep bracket")
    else:
        best = min(probes, key=lambda probe: probe.bend_deficit_deg)
        print(
            tabulate(
                [
                    [
                        f"{probe.excess_out:.2f}",
                        f"{probe.excess_in:.2f}",
                        f"{probe.departure_excess:.2f}",
                        f"{probe.departure_aim_deg:+.1f}",
                        f"{probe.perijove_burn * 1000:.0f}",
                        f"{probe.bend_deficit_deg:+.2f}",
                    ]
                    for probe in probes[:: max(1, len(probes) // 12)]
                ],
                headers=[
                    "v_inf out",
                    "v_inf in",
                    "v_inf Earth",
                    "aim",
                    "perijove dv (m/s)",
                    "bend deficit (deg)",
                ],
                tablefmt="github",
            )
        )
        print(
            f"minimum deficit over the whole sweep: {best.bend_deficit_deg:+.2f} deg "
            f"at v_inf_out {best.excess_out:.2f} km/s "
            f"({best.perijove_burn * 1000:.0f} m/s of perijove burn)."
        )
        print(
            "The deficit never reaches zero, so no perijove burn closes the 2S cycle."
        )

    print("\n-- what a deep-space maneuver charges to close 2S --")
    correction = minimum_dsm_correction(
        2,
        params=params,
        dive_solar_radii=args.dive_solar_radii,
        return_excess=args.return_excess,
    )
    if correction is None:
        print("no DSM closure inside the recorded search bounds")
    else:
        print(
            tabulate(
                [
                    ["maneuver", f"{correction.magnitude * 1000:.0f} m/s"],
                    ["fires at", f"{correction.radius_au:.3f} AU"],
                    [
                        "Earth departure excess",
                        f"{correction.departure_excess:.3f} km/s",
                    ],
                    ["departure aim", f"{correction.departure_aim_deg:+.2f} deg"],
                    [
                        "Jupiter arrival excess",
                        f"{correction.jupiter_arrival_excess:.3f} km/s",
                    ],
                    [
                        "bend required / available",
                        f"{correction.bend_required_deg:.2f} / "
                        f"{correction.bend_available_deg:.2f} deg",
                    ],
                ],
                tablefmt="github",
            )
        )
        print(
            "Jupiter's sphere of influence reaches ~0.322 AU, so a burn inside about\n"
            "4.88 AU would be a Jupiter-relative maneuver wearing a DSM's name."
        )

    three = solve_synodic_closure(
        3,
        params=params,
        dive_solar_radii=args.dive_solar_radii,
        return_excess=args.return_excess,
    )
    if three is None:
        return
    print(
        f"\n-- the 3S departure on a magnetic nozzle (k = {args.slug_ratio:.0f}, "
        f"eta_jet^2 = {args.jet_efficiency:.2f}, {1 - args.periapsis_survival:.0%} "
        "propellant at 4 R_sun) --"
    )
    print(
        "Earth-launched slug is treated as FREE here; only impactor kilograms are\n"
        "charged. That switches off CONTEXT.md's launch ledger -- see the caveat below."
    )
    ledgers = [
        cycle_growth_ledger(
            f"this {three.synodic_multiple}S Jovian solar dive",
            three.departure_excess,
            three.departure_aim_deg,
            three.total_tof_years,
            three.return_excess,
            three.push_axis_deg,
            slug_ratio=args.slug_ratio,
            jet_energy_efficiency=args.jet_efficiency,
            periapsis_survival=args.periapsis_survival,
            params=params,
        ),
        paper_resonant_dive_ledger(
            three.return_excess,
            three.push_axis_deg,
            slug_ratio=args.slug_ratio,
            jet_energy_efficiency=args.jet_efficiency,
            periapsis_survival=args.periapsis_survival,
            params=params,
        ),
    ]
    print(
        tabulate(
            [
                [
                    ledger.label,
                    f"{ledger.departure_excess:.2f}",
                    f"{ledger.nozzle.impact_angle_start_deg:.0f}-"
                    f"{ledger.nozzle.impact_angle_end_deg:.0f}",
                    f"{ledger.nozzle.closing_speed_start:.0f}-"
                    f"{ledger.nozzle.closing_speed_end:.0f}",
                    f"{ledger.nozzle.specific_impulse:.0f}",
                    f"{ledger.nozzle.delivered_fraction:.3f}",
                    f"{ledger.round_trip_fraction:.3f}",
                    f"{ledger.return_per_impactor_kg:.1f}",
                    f"{ledger.cycle_years:.3f}",
                    f"{ledger.growth_rate:.3f}",
                    f"{ledger.doubling_years:.3f}",
                ]
                for ledger in ledgers
            ],
            headers=[
                "cycle",
                "v_inf",
                "theta (deg)",
                "w (km/s)",
                "Isp (s)",
                "departs",
                "round trip",
                "per impactor kg",
                "cycle (yr)",
                "e-fold/yr",
                "doubling (yr)",
            ],
            tablefmt="github",
        )
    )
    print("\n-- the stage this actually needs, and how hard it leans on efficiency --")
    mine = ledgers[0]
    print(
        tabulate(
            [
                [
                    f"{eta:.2f}",
                    f"{probe.nozzle.specific_impulse:.0f}",
                    f"{probe.nozzle.effective_exhaust_speed:.2f}",
                    f"{probe.nozzle.slug_per_delivered_kg:.3f}",
                    f"{probe.nozzle.impactor_per_delivered_kg:.4f}",
                    f"{probe.nozzle.delivered_fraction:.3f}",
                    f"{probe.doubling_years:.3f}",
                ]
                for eta, probe in (
                    (
                        eta,
                        cycle_growth_ledger(
                            "3S",
                            three.departure_excess,
                            three.departure_aim_deg,
                            three.total_tof_years,
                            three.return_excess,
                            three.push_axis_deg,
                            slug_ratio=args.slug_ratio,
                            jet_energy_efficiency=eta,
                            periapsis_survival=args.periapsis_survival,
                            params=params,
                        ),
                    )
                    for eta in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
                )
            ],
            headers=[
                "eta_jet^2",
                "Isp (s)",
                "v_e (km/s)",
                "slug/kg delivered",
                "impactor/kg delivered",
                "departs",
                "doubling (yr)",
            ],
            tablefmt="github",
        )
    )
    print(
        f"A methalox stage flying the same {mine.nozzle.delta_v:.2f} km/s would burn "
        f"{methalox_per_delivered_kg(mine.nozzle.delta_v, params):.2f} kg per kg delivered, "
        f"against {mine.nozzle.slug_per_delivered_kg:.2f} kg of slug -- "
        f"{methalox_per_delivered_kg(mine.nozzle.delta_v, params) / mine.nozzle.slug_per_delivered_kg:.0f}x leaner.\n"
        "The slug ratio is per *impactor* kilogram, and impactors are under 1% of the\n"
        "vehicle, so k = 30 is not a large propellant load. Doubling time moves only\n"
        "0.54 -> 0.48 yr across the whole efficiency range, so eta_geom being unmeasured\n"
        "(sec:jet_efficiency) is not load-bearing here."
    )

    fastest = max(ledgers, key=lambda ledger: ledger.growth_rate)
    print(
        f"Fastest doubling on this device: {fastest.label} at "
        f"{fastest.doubling_years:.3f} yr "
        f"(millionfold in {fastest.millionfold_years:.1f} yr)."
    )
    print(
        "CAVEAT: growth is per impactor kilogram, and the launched slug grows with\n"
        f"it -- {ledgers[0].launched_slug_per_impactor_kg:.0f} kg lofted per impactor kg, so a "
        f"{ledgers[0].return_per_impactor_kg:.0f}x cycle is also a "
        f"{ledgers[0].return_per_impactor_kg:.0f}x launch-rate cycle. The constraint moves\n"
        "to the pad; it does not disappear. Charging it is CONTEXT.md's launch ledger."
    )


if __name__ == "__main__":
    main()
