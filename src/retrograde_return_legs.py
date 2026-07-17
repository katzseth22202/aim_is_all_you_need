"""Float-valued substrate shared by the powered Jovian flyby and the unpowered
assist chain (CONTEXT.md "Retrograde-return legs").

Both paths onto the retrograde Earth-crossing return -- a powered gravity
assist at Jupiter (jovian_flyby.py) and an unpowered Venus/Earth/Mars chain
(assist_chain.py) -- assemble the same leg-by-leg body/radius/velocity/TOF
state and share the same Jovian-terminal return-leg algebra. nozzle_analysis.py
is a third caller of this substrate, reusing it to score an alternative
(nozzle-push) departure. All values are plain floats (km, s, km/s, rad), like
conic_kernel.py: this sits below orbit_utils.py in the module hierarchy and is
evaluated many thousands of times per optimizer run.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.bodies import Earth, Jupiter, Mars, Sun, Venus
from boinor.core.iod import izzo
from scipy.optimize import brentq

from src import conic_kernel
from src.astro_constants import (
    ASSIST_CHAIN_MAX_FLYBYS,
    ASSIST_CHAIN_MAX_TRIP_TIME,
    EARTH_A,
    JUPITER_A,
    JUPITER_FLYBY_MAX_TOF,
    LEO_ALTITUDE,
    LOW_ASSIST_FLYBY_ALTITUDE,
    LOW_JUPITER_ALTITUDE,
    MARS_A,
    METHALOX_VACUUM_ISP,
    STD_FUDGE_FACTOR,
    VENUS_A,
)
from src.orbit_utils import escape_velocity
from src.propulsion import exhaust_velocity_from_isp
from src.scenario_catalog import lunar_transfer_periapsis_speed

# Rooted with brentq over the bend-limited rotation, the same scan-for-sign-
# change-then-root pattern used throughout the assist-chain search.
_ASSIST_ROTATION_SAMPLES = 31  # unpowered-bend samples per inner-planet flyby
_ASSIST_MIN_LEG_TIME = 86400.0  # s; ignore degenerate sub-day legs


@dataclass(frozen=True)
class _FlybyParams:
    """Float-valued physical inputs of the powered-flyby search (km, s, km/s).

    Attributes:
        mu_sun: Sun gravitational parameter (km^3/s^2).
        mu_jupiter: Jupiter gravitational parameter (km^3/s^2).
        r_earth_orbit: Earth heliocentric orbit radius, 1 AU (km).
        r_jupiter_orbit: Jupiter heliocentric orbit radius (km).
        v_earth_orbit: Earth circular heliocentric speed (km/s).
        v_jupiter_orbit: Jupiter circular heliocentric speed (km/s).
        v_esc_leo: Earth escape speed at the 200 km burn altitude (km/s). Sets
            the Oberth leverage: leaving with excess ``w`` needs periapsis speed
            ``hypot(w, v_esc_leo)``.
        v_depart_from: Speed the mass already carries at that periapsis when the
            departure burn lights, so the burn is
            ``hypot(w, v_esc_leo) - v_depart_from``. Equals ``v_esc_leo`` under
            the legacy C3 = 0 convention, or the cycle orbit's periapsis speed
            when the growth loop is closed (see
            :func:`puffsat_cycle_periapsis_speed`).
        v_esc_surface: Earth surface escape speed -- the collision-speed
            convention of retrograde_jovian_hohmann_transfer (km/s).
        periapsis_floor: Minimum Jovian flyby periapsis radius, center-based (km).
        exhaust_speed: Methalox vacuum effective exhaust speed (km/s).
        v_rf: Push target of the growth row the mass ratio is scored on (km/s).
        max_tof: Cap on outbound + return heliocentric time of flight (s).
    """

    mu_sun: float
    mu_jupiter: float
    r_earth_orbit: float
    r_jupiter_orbit: float
    v_earth_orbit: float
    v_jupiter_orbit: float
    v_esc_leo: float
    v_depart_from: float
    v_esc_surface: float
    periapsis_floor: float
    exhaust_speed: float
    v_rf: float
    max_tof: float


def _powered_flyby_params(
    max_total_tof: u.Quantity = JUPITER_FLYBY_MAX_TOF,
    periapsis_floor_altitude: u.Quantity = LOW_JUPITER_ALTITUDE,
    cycle_periapsis_speed: Optional[u.Quantity] = None,
) -> _FlybyParams:
    """Build the float parameter block for the powered-flyby search.

    Args:
        max_total_tof: Cap on outbound + return time of flight (astropy Quantity).
        periapsis_floor_altitude: Minimum flyby altitude above Jupiter's 1-bar
            level (astropy Quantity).
        cycle_periapsis_speed: Speed the departure burn starts from, and the
            collision's push target -- see :func:`puffsat_cycle_periapsis_speed`.
            When None the legacy convention is kept: depart from escape at 200 km
            (C3 = 0) and score the mass ratio against the lunar-transfer periapsis
            speed. Those are two different states, so the legacy pair is not a
            closed cycle; pass the cycle speed to close it.

    Returns:
        The :class:`_FlybyParams` with everything converted to km / s / km/s.
    """
    mu_sun = float(Sun.k.to_value(u.km**3 / u.s**2))
    r_earth_orbit = float(EARTH_A.to_value(u.km))
    r_jupiter_orbit = float(JUPITER_A.to_value(u.km))
    v_esc_leo = float(escape_velocity(Earth, LEO_ALTITUDE).to_value(u.km / u.s))
    if cycle_periapsis_speed is None:
        v_depart_from = v_esc_leo
        v_rf = float(lunar_transfer_periapsis_speed().to_value(u.km / u.s))
    else:
        v_depart_from = float(cycle_periapsis_speed.to_value(u.km / u.s))
        v_rf = v_depart_from
    return _FlybyParams(
        mu_sun=mu_sun,
        mu_jupiter=float(Jupiter.k.to_value(u.km**3 / u.s**2)),
        r_earth_orbit=r_earth_orbit,
        r_jupiter_orbit=r_jupiter_orbit,
        v_earth_orbit=float(np.sqrt(mu_sun / r_earth_orbit)),
        v_jupiter_orbit=float(np.sqrt(mu_sun / r_jupiter_orbit)),
        v_esc_leo=v_esc_leo,
        v_depart_from=v_depart_from,
        v_esc_surface=float(escape_velocity(Earth).to_value(u.km / u.s)),
        periapsis_floor=float((Jupiter.R + periapsis_floor_altitude).to_value(u.km)),
        exhaust_speed=float(
            exhaust_velocity_from_isp(METHALOX_VACUUM_ISP).to_value(u.km / u.s)
        ),
        v_rf=v_rf,
        max_tof=float(max_total_tof.to_value(u.s)),
    )


@dataclass(frozen=True)
class _ReturnLeg:
    """Float summary of the retrograde Jupiter-to-1 AU return leg.

    Attributes:
        perihelion: Perihelion radius of the return orbit (km).
        tof: Time of flight from the flyby to the first inbound 1 AU crossing (s).
        closing_speed: Earth-relative speed at the 1 AU crossing (km/s).
        collision_speed: Closing speed folded through Earth's gravity well to the
            surface-escape convention of retrograde_jovian_hohmann_transfer (km/s).
        sweep_angle: Heliocentric longitude swept from the flyby to the 1 AU
            crossing (rad, always positive). The return is *retrograde*, so
            longitude decreases: the crossing happens at
            ``jupiter_longitude - sweep_angle``. This is what makes it possible to
            ask whether Earth is actually at the crossing; the leg itself does not
            check -- see `_earth_phase_mismatch`.
    """

    perihelion: float
    tof: float
    closing_speed: float
    collision_speed: float
    sweep_angle: float


def _flyby_return_leg(
    v_tangential: float, v_radial: float, params: _FlybyParams
) -> Optional[_ReturnLeg]:
    """Score the heliocentric return from Jupiter's orbit radius to 1 AU.

    Takes the post-flyby heliocentric velocity at ``r_jupiter_orbit`` in the
    (tangential, radial-outward) basis. The orbit must be retrograde
    (``v_tangential < 0``) and cross 1 AU; the leg ends at the first *inbound*
    1 AU crossing. Internally the orbit is mirrored to a prograde twin (radii
    and times are unchanged) so the conic formulas keep a positive angular
    momentum.

    Args:
        v_tangential: Post-flyby tangential heliocentric speed (km/s, negative
            for retrograde).
        v_radial: Post-flyby radial-outward heliocentric speed (km/s).
        params: The float parameter block.

    Returns:
        The :class:`_ReturnLeg`, or None if the state is not retrograde, never
        crosses 1 AU, or never comes back inward.
    """
    if v_tangential >= 0.0:
        return None
    mu = params.mu_sun
    r_j = params.r_jupiter_orbit
    r_e = params.r_earth_orbit
    # Mirrored prograde twin: same radii and times, positive angular momentum.
    v_t = -v_tangential
    v_r = v_radial
    state = conic_kernel.conic_state_at_radius(mu, r_j, v_t, v_r)
    ecc = state.ecc
    if abs(ecc - 1.0) < 1e-9 or ecc < 1e-9:
        return None
    perihelion = conic_kernel.periapsis_radius_of_conic(state.p, ecc)
    if perihelion > r_e:
        return None
    nu_jupiter = conic_kernel.true_anomaly_at_radius_rad(state.p, ecc, r_j)
    nu_earth = conic_kernel.true_anomaly_at_radius_rad(state.p, ecc, r_e)
    if nu_jupiter is None or nu_earth is None:
        return None
    # The swept angle is derived in the same branches as the time of flight, from
    # the same true anomalies, so the two cannot disagree about which arc is
    # being flown. In the mirrored prograde twin an outbound state sits at true
    # anomaly +nu and an inbound one at 2*pi - nu.
    if ecc > 1.0:
        # Hyperbolic return: only an already-inbound state ever re-crosses 1 AU.
        if v_r > 0.0:
            return None
        a_abs = state.p / (ecc * ecc - 1.0)
        tof = conic_kernel.hyperbolic_tof_seconds(mu, a_abs, ecc, nu_jupiter)
        tof -= conic_kernel.hyperbolic_tof_seconds(mu, a_abs, ecc, nu_earth)
        sweep = nu_jupiter - nu_earth
    else:
        a = state.p / (1.0 - ecc * ecc)
        period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
        t_jupiter = conic_kernel.elliptic_tof_seconds(mu, a, ecc, nu_jupiter)
        t_earth = conic_kernel.elliptic_tof_seconds(mu, a, ecc, nu_earth)
        if v_r > 0.0:
            # Out to aphelion first, then back in through 1 AU.
            tof = (period - t_earth) - t_jupiter
            sweep = (2.0 * np.pi - nu_earth) - nu_jupiter
        else:
            tof = t_jupiter - t_earth
            sweep = nu_jupiter - nu_earth
    if tof <= 0.0:
        return None
    v_t1, v_r1 = conic_kernel.speed_components_at_radius(state, mu, r_e)
    # Un-mirrored the tangential speed is -v_t1, so the Earth-relative closing
    # speed is hypot(v_t1 + v_earth, v_r1).
    closing = float(np.hypot(v_t1 + params.v_earth_orbit, v_r1))
    collision = conic_kernel.speed_with_escape_energy(closing, params.v_esc_surface)
    return _ReturnLeg(
        perihelion=perihelion,
        tof=tof,
        closing_speed=closing,
        collision_speed=collision,
        sweep_angle=float(sweep),
    )


def _earth_phase_mismatch(
    leg: _ReturnLeg,
    jupiter_longitude: float,
    flyby_time: float,
    earth_longitude_0: float,
    params: _FlybyParams,
) -> float:
    """Signed angle between the 1 AU crossing and where Earth actually is (rad).

    The return leg scores the closing speed of a geometry that may arrive at
    empty space: nothing in :func:`_flyby_return_leg` checks Earth's position. For
    a *growth loop* that is not a detail -- the collision with Earth's mass IS the
    cycle, so a crossing Earth misses does not merely score badly, it does not
    close the loop at all.

    The grill's terms make this binding rather than fixable: PuffSats are deployed
    far out and coast ballistically, with no deep-space maneuver permitted below
    3 AU, so a phase error cannot be trimmed out on approach. It has to be flown
    correctly from Jupiter.

    Args:
        leg: The scored return leg (supplies ``sweep_angle`` and ``tof``).
        jupiter_longitude: Jupiter's heliocentric longitude at the flyby (rad).
        flyby_time: Time of the flyby, measured from the same t=0 as
            ``earth_longitude_0`` (s).
        earth_longitude_0: Earth's heliocentric longitude at t=0 (rad).
        params: The float parameter block.

    Returns:
        The mismatch wrapped to [-pi, pi). Zero means Earth is exactly at the
        crossing; the loop closes only where this vanishes.
    """
    arrival_time = flyby_time + leg.tof
    # The return is retrograde, so the crossing sits *behind* Jupiter in longitude.
    crossing = jupiter_longitude - leg.sweep_angle
    n_earth = params.v_earth_orbit / params.r_earth_orbit
    earth = earth_longitude_0 + n_earth * arrival_time
    return conic_kernel.wrap_pi(crossing - earth)


@dataclass(frozen=True)
class _FlybyLeg:
    """Float summary of one full Earth-to-1-AU powered-flyby trajectory.

    Attributes:
        departure_burn: Oberth burn above escape at 200 km, delta-v1 (km/s).
        flyby_burn: Impulsive burn at Jovian periapsis, delta-v2 (km/s).
        v_infinity_earth: Hyperbolic excess speed leaving Earth (km/s).
        aim_angle: Angle of the departure excess velocity off Earth's orbital
            velocity (rad; positive tilts radially outward).
        periapsis_radius: Jovian flyby periapsis radius, center-based (km).
        v_infinity_in: Jupiter-relative excess speed before the burn (km/s).
        v_infinity_out: Jupiter-relative excess speed after the burn (km/s).
        turn_angle: Total split-hyperbola bend asin(1/e_in) + asin(1/e_out) (rad).
        return_leg: The scored retrograde return.
        outbound_tof: Earth-to-Jupiter heliocentric time of flight (s).
        delivered_fraction: Mass fraction surviving both burns.
        mass_ratio: Payload/PuffSat mass ratio at the achieved collision speed.
        end_to_end: delivered_fraction x mass_ratio, the objective.
    """

    departure_burn: float
    flyby_burn: float
    v_infinity_earth: float
    aim_angle: float
    periapsis_radius: float
    v_infinity_in: float
    v_infinity_out: float
    turn_angle: float
    return_leg: _ReturnLeg
    outbound_tof: float
    delivered_fraction: float
    mass_ratio: float
    end_to_end: float


def _powered_flyby_leg(
    v_infinity_earth: float,
    aim_angle: float,
    periapsis_radius: float,
    flyby_burn: float,
    descending_arrival: bool,
    bend_sign: float,
    params: _FlybyParams,
) -> Optional[_FlybyLeg]:
    """Evaluate one powered-flyby trajectory from its four knobs.

    Free-aim departure: the burn cost depends only on ``v_infinity_earth``;
    ``aim_angle`` orients the excess velocity for free. The flyby is a split
    hyperbola -- an impulsive tangential burn at periapsis joins an inbound and
    an outbound hyperbola of different eccentricities, so the total bend is
    ``asin(1/e_in) + asin(1/e_out)``, rotated to the side ``bend_sign`` picks.

    Args:
        v_infinity_earth: Hyperbolic excess speed leaving Earth (km/s).
        aim_angle: Angle of the excess velocity off Earth's orbital velocity
            (rad; positive tilts radially outward).
        periapsis_radius: Jovian flyby periapsis radius, center-based (km).
        flyby_burn: Impulsive burn at Jovian periapsis, >= 0 (km/s).
        descending_arrival: Arrive at Jupiter's orbit radius past aphelion
            (inward-moving) instead of on the outbound branch.
        bend_sign: +1 rotates the excess velocity from tangential toward
            radial-outward, -1 the other way (which side Jupiter is passed on).
        params: The float parameter block.

    Returns:
        The :class:`_FlybyLeg`, or None if infeasible (cannot reach Jupiter,
        non-retrograde return, no 1 AU crossing, or over the time-of-flight cap).
    """
    mu = params.mu_sun
    r_e = params.r_earth_orbit
    r_j = params.r_jupiter_orbit
    if v_infinity_earth <= 0.0 or flyby_burn < 0.0:
        return None
    if periapsis_radius < params.periapsis_floor * (1.0 - 1e-12):
        return None
    departure_burn = float(
        np.hypot(v_infinity_earth, params.v_esc_leo) - params.v_depart_from
    )

    # Outbound heliocentric transfer from the free-aimed departure state.
    v_t0 = params.v_earth_orbit + v_infinity_earth * float(np.cos(aim_angle))
    v_r0 = v_infinity_earth * float(np.sin(aim_angle))
    if v_t0 <= 0.0:
        return None
    state = conic_kernel.conic_state_at_radius(mu, r_e, v_t0, v_r0)
    ecc = state.ecc
    if abs(ecc - 1.0) < 1e-9 or ecc < 1e-9:
        return None
    if ecc < 1.0 and state.p / (1.0 - ecc) < r_j:
        return None
    if ecc > 1.0 and descending_arrival:
        return None
    nu_depart = conic_kernel.true_anomaly_at_radius_rad(state.p, ecc, r_e)
    nu_arrive = conic_kernel.true_anomaly_at_radius_rad(state.p, ecc, r_j)
    if nu_depart is None or nu_arrive is None:
        return None
    if ecc < 1.0:
        a = state.p / (1.0 - ecc * ecc)
        period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
        t_depart = conic_kernel.elliptic_tof_seconds(mu, a, ecc, nu_depart)
        t_arrive = conic_kernel.elliptic_tof_seconds(mu, a, ecc, nu_arrive)
        if descending_arrival:
            t_arrive = period - t_arrive
    else:
        a_abs = state.p / (ecc * ecc - 1.0)
        t_depart = conic_kernel.hyperbolic_tof_seconds(mu, a_abs, ecc, nu_depart)
        t_arrive = conic_kernel.hyperbolic_tof_seconds(mu, a_abs, ecc, nu_arrive)
    if v_r0 < 0.0:
        t_depart = -t_depart
    outbound_tof = t_arrive - t_depart
    if outbound_tof <= 0.0:
        return None

    # Jupiter-relative arrival state.
    v_t_arr, v_r_arr = conic_kernel.speed_components_at_radius(state, mu, r_j)
    if descending_arrival:
        v_r_arr = -v_r_arr
    rel_t = v_t_arr - params.v_jupiter_orbit
    rel_r = v_r_arr
    w_in = float(np.hypot(rel_t, rel_r))
    if w_in < 1e-6:
        return None

    # Split-hyperbola powered flyby at periapsis_radius.
    mu_j = params.mu_jupiter
    v_peri_in = float(np.sqrt(w_in * w_in + 2.0 * mu_j / periapsis_radius))
    v_peri_out = v_peri_in + flyby_burn
    w_out_sq = v_peri_out * v_peri_out - 2.0 * mu_j / periapsis_radius
    if w_out_sq <= 0.0:
        return None
    w_out = float(np.sqrt(w_out_sq))
    ecc_in = conic_kernel.hyperbolic_eccentricity(mu_j, periapsis_radius, w_in)
    ecc_out = conic_kernel.hyperbolic_eccentricity(mu_j, periapsis_radius, w_out)
    turn = conic_kernel.powered_bend_angle(ecc_in, ecc_out)
    cos_turn = float(np.cos(bend_sign * turn))
    sin_turn = float(np.sin(bend_sign * turn))
    unit_t = rel_t / w_in
    unit_r = rel_r / w_in
    out_t = cos_turn * unit_t - sin_turn * unit_r
    out_r = sin_turn * unit_t + cos_turn * unit_r
    v_t_out = params.v_jupiter_orbit + w_out * out_t
    v_r_out = w_out * out_r

    return_leg = _flyby_return_leg(v_t_out, v_r_out, params)
    if return_leg is None:
        return None
    if outbound_tof + return_leg.tof > params.max_tof:
        return None
    if return_leg.collision_speed <= params.v_rf:
        return None

    delivered = float(np.exp(-(departure_burn + flyby_burn) / params.exhaust_speed))
    # payload_mass_ratio with v_ri = 0, inlined for the float hot loop.
    mass_ratio = float(
        2.0
        * STD_FUDGE_FACTOR
        / np.log(
            return_leg.collision_speed / (return_leg.collision_speed - params.v_rf)
        )
    )
    return _FlybyLeg(
        departure_burn=departure_burn,
        flyby_burn=flyby_burn,
        v_infinity_earth=v_infinity_earth,
        aim_angle=aim_angle,
        periapsis_radius=periapsis_radius,
        v_infinity_in=w_in,
        v_infinity_out=w_out,
        turn_angle=turn,
        return_leg=return_leg,
        outbound_tof=outbound_tof,
        delivered_fraction=delivered,
        mass_ratio=mass_ratio,
        end_to_end=delivered * mass_ratio,
    )


@dataclass(frozen=True)
class _AssistBody:
    """One inner-planet flyby body of the assist chain (km, s, km/s floats).

    Attributes:
        symbol: One-letter tag used in sequence strings ("V", "E", "M").
        name: Full body name for the public result.
        orbit_radius: Heliocentric orbit radius (km).
        mu: Gravitational parameter (km^3/s^2).
        min_periapsis: Minimum flyby periapsis radius, center-based (km).
        v_circ: Circular heliocentric speed at orbit_radius (km/s).
    """

    symbol: str
    name: str
    orbit_radius: float
    mu: float
    min_periapsis: float
    v_circ: float


@dataclass(frozen=True)
class _AssistChainParams:
    """Inputs of the assist-chain beam search.

    Attributes:
        flyby: The powered-flyby parameter block, reused for the Sun/Jupiter
            constants, the Jovian periapsis floor, the exhaust speed, and the
            v_rf push target (its max_tof field is unused here).
        bodies: The flyby bodies, in (Venus, Earth, Mars) order.
        earth_index: Index of Earth in ``bodies`` (the departure body).
        target_collision_speed: Minimum acceptable v_b of the return (km/s).
        max_trip_time: Cap on departure-to-1 AU-crossing time (s).
        max_flybys: Cap on the number of inner-planet flybys.
    """

    flyby: _FlybyParams
    bodies: Tuple[_AssistBody, ...]
    earth_index: int
    target_collision_speed: float
    max_trip_time: float
    max_flybys: int


def _assist_chain_params(
    target_collision_speed: float,
    max_trip_time: u.Quantity = ASSIST_CHAIN_MAX_TRIP_TIME,
    max_flybys: int = ASSIST_CHAIN_MAX_FLYBYS,
    cycle_periapsis_speed: Optional[u.Quantity] = None,
) -> _AssistChainParams:
    """Build the float parameter block for the assist-chain search.

    Args:
        target_collision_speed: Minimum acceptable collision speed v_b (km/s).
        max_trip_time: Cap on total trip time (astropy Quantity).
        max_flybys: Cap on the number of inner-planet flybys.
        cycle_periapsis_speed: Speed the departure burn starts from; see
            :func:`_powered_flyby_params`. None keeps the legacy C3 = 0
            convention.

    Returns:
        The :class:`_AssistChainParams` with everything in km / s / km/s.
    """
    flyby = _powered_flyby_params(
        max_total_tof=max_trip_time, cycle_periapsis_speed=cycle_periapsis_speed
    )
    altitude = float(LOW_ASSIST_FLYBY_ALTITUDE.to_value(u.km))
    bodies: List[_AssistBody] = []
    for symbol, body, semi_major_axis in (
        ("V", Venus, VENUS_A),
        ("E", Earth, EARTH_A),
        ("M", Mars, MARS_A),
    ):
        orbit_radius = float(semi_major_axis.to_value(u.km))
        bodies.append(
            _AssistBody(
                symbol=symbol,
                name=str(body.name),
                orbit_radius=orbit_radius,
                mu=float(body.k.to_value(u.km**3 / u.s**2)),
                min_periapsis=float(body.R.to_value(u.km)) + altitude,
                v_circ=float(np.sqrt(flyby.mu_sun / orbit_radius)),
            )
        )
    return _AssistChainParams(
        flyby=flyby,
        bodies=tuple(bodies),
        earth_index=1,
        target_collision_speed=target_collision_speed,
        max_trip_time=float(max_trip_time.to_value(u.s)),
        max_flybys=max_flybys,
    )


def _mean_motion(body: _AssistBody) -> float:
    """Mean motion of a chain body on its circular orbit (rad/s).

    Args:
        body: The chain body.

    Returns:
        Angular rate v_circ / r, positive (prograde).
    """
    return conic_kernel.mean_motion(body.v_circ, body.orbit_radius)


def _phased_leg_rotations(
    body: _AssistBody,
    body_longitude: float,
    excess_t: float,
    excess_r: float,
    target: _AssistBody,
    target_longitude: float,
    outbound: bool,
    params: _AssistChainParams,
    samples: int = _ASSIST_ROTATION_SAMPLES,
    rotation_limit: Optional[float] = None,
) -> List[Tuple[float, float, float, float]]:
    """Flyby rotations at ``body`` that arrive where ``target`` actually is.

    This is what turns the phasing-free ladder into a phased one. The
    phasing-free search samples the rotation freely and lets the target planet
    be wherever the leg lands; here the rotation is *solved* so the leg lands on
    the planet. Rotating by ``theta`` fixes both the swept longitude and the
    flight time, so "arrive where the target will be" is one equation in one
    unknown -- rooted with brentq over the bend-limited rotation, the same
    scan-for-sign-change-then-root pattern as
    :func:`apoapsis_raise_reintercept`.

    Wrapped residuals jump by 2*pi where they cross the branch cut, which looks
    exactly like a sign change; brackets spanning more than pi of residual are
    rejected so those phantom roots are never rooted.

    Args:
        body: Body the flyby happens at.
        body_longitude: Its heliocentric longitude at the flyby (rad).
        excess_t: Tangential component of the body-relative excess (km/s).
        excess_r: Radial-outward component of the body-relative excess (km/s).
        target: Body the leg must arrive at.
        target_longitude: Target's heliocentric longitude at the *same* instant
            as ``body_longitude`` -- both are stated at the flyby, so the
            absolute epoch never enters and the caller keeps that bookkeeping.
        outbound: Which arrival branch to solve on.
        params: The assist-chain parameter block.
        samples: Rotations scanned for sign changes before rooting.
        rotation_limit: Override for the reachable rotation half-range (rad).
            Defaults to the body's unpowered bend limit. Pass ``pi`` for the
            *departure*, where there is no flyby and the excess may be aimed
            anywhere for free -- a bend limit there would be fictitious.

    Returns:
        One (rotation rad, leg time s, tangential speed at arrival km/s,
        signed radial speed at arrival km/s) per solved rotation, possibly
        empty. Several roots can coexist: the equation is transcendental.
    """
    w = float(np.hypot(excess_t, excess_r))
    if w < 1e-9:
        return []
    phi = float(np.arctan2(excess_r, excess_t))
    if rotation_limit is None:
        ecc = conic_kernel.hyperbolic_eccentricity(body.mu, body.min_periapsis, w)
        bend_limit = conic_kernel.unpowered_bend_angle(ecc)
    else:
        bend_limit = rotation_limit
    n_target = _mean_motion(target)

    def evaluate(theta: float) -> Optional[Tuple[float, float, float, float]]:
        angle = phi + theta
        v_t0 = body.v_circ + w * float(np.cos(angle))
        v_r0 = w * float(np.sin(angle))
        for leg_tof, v_t1, v_r1, out, swept in conic_kernel.conic_radius_crossings(
            params.flyby.mu_sun,
            body.orbit_radius,
            v_t0,
            v_r0,
            target.orbit_radius,
            min_leg_time=_ASSIST_MIN_LEG_TIME,
        ):
            if out != outbound:
                continue
            # Both longitudes are stated at `epoch`, so the target advances by
            # the leg time only -- advancing it from zero would double-count
            # the epoch and manufacture a spurious window cadence.
            arrival = body_longitude + swept
            where_target_is = target_longitude + n_target * leg_tof
            return (
                conic_kernel.wrap_pi(arrival - where_target_is),
                leg_tof,
                v_t1,
                v_r1,
            )
        return None

    def residual(theta: float) -> float:
        found = evaluate(theta)
        return float("nan") if found is None else found[0]

    grid = np.linspace(-bend_limit, bend_limit, samples)
    values = [residual(float(t)) for t in grid]
    solved: List[Tuple[float, float, float, float]] = []
    for i in range(len(grid) - 1):
        lo, hi = values[i], values[i + 1]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if lo == 0.0:
            root = float(grid[i])
        elif lo * hi > 0.0:
            continue
        elif abs(lo - hi) > np.pi:
            # Branch-cut jump, not a crossing: the residual wrapped rather than
            # passed through zero.
            continue
        else:
            try:
                root = float(brentq(residual, grid[i], grid[i + 1], xtol=1e-12))
            except (ValueError, RuntimeError):
                continue
        found = evaluate(root)
        if found is None:
            continue
        solved.append((root, found[1], found[2], found[3]))
    return solved


_PHASED_LADDER_FAILED = 1e3  # km/s; above any real ladder cost


@dataclass(frozen=True)
class _LadderPricing:
    """What it costs to fly an inner ladder against real planet positions.

    Attributes:
        departure_burn: Oberth burn at 200 km buying the first leg's excess
            (km/s).
        node_total: Sum of the flyby node burns (km/s).
        node_burns: Per-node burns in ladder order, one per intermediate body
            (km/s).
        arrival_excess: Speed of the planet-relative excess at the final ladder
            body, where the Jovian leg begins (km/s).
        arrival_excess_vector: That excess as a heliocentric 3-vector (km/s).
        arrival_time: Seconds from the longitudes0 reference at the final body.
        arrival_longitude: Heliocentric longitude of the final body then (rad).
    """

    departure_burn: float
    node_total: float
    node_burns: List[float]
    arrival_excess: float
    arrival_excess_vector: npt.NDArray[np.float64]
    arrival_time: float
    arrival_longitude: float


def _phased_ladder_burn(
    epoch: float,
    leg_times: Sequence[float],
    ladder: Tuple[_AssistBody, ...],
    longitudes0: Sequence[float],
    params: _AssistChainParams,
    powered_nodes: bool = False,
) -> Optional[_LadderPricing]:
    """Burn to fly ``ladder`` from ``epoch`` against real planet positions.

    The phasing-free model puts each planet wherever the trajectory needs it;
    this puts the planets where they actually are and charges for the
    difference. Each leg becomes a Lambert arc between the two bodies at their
    true positions, and each node is charged for whatever the flyby cannot
    supply. That total is what ``ASSIST_CHAIN_PHASING_BUDGET`` is a stand-in
    for.

    Args:
        epoch: Launch time, seconds from the longitudes0 reference.
        leg_times: Duration of each leg (s), one per ladder hop.
        ladder: Bodies in order, departure body first; the Jovian leg is not
            included (this prices the inner ladder only).
        longitudes0: Heliocentric longitude of each ladder body at time zero
            (rad), parallel to ``ladder``.
        params: The assist-chain parameter block.
        powered_nodes: Charge each node with :func:`_powered_node_burn`, the
            honest split-hyperbola model, instead of ADR 0007's retired
            :func:`_flyby_mismatch_burn`. Defaults to False, which reproduces
            ADR 0007's numbers.

    Returns:
        The :class:`_LadderPricing`, or None if any leg's Lambert fails.
    """
    mu = params.flyby.mu_sun
    times = [float(t) for t in leg_times]
    node_burns: List[float] = []
    excess_in: Optional[npt.NDArray[np.float64]] = None
    departure_burn = 0.0
    clock = epoch
    lon_there = 0.0
    for index in range(len(ladder) - 1):
        body, target = ladder[index], ladder[index + 1]
        tof = times[index]
        if tof <= _ASSIST_MIN_LEG_TIME:
            return None
        lon_here = longitudes0[index] + _mean_motion(body) * clock
        lon_there = longitudes0[index + 1] + _mean_motion(target) * (clock + tof)
        r0, v_body = _body_state(body.orbit_radius, lon_here, body.v_circ)
        r1, v_target = _body_state(target.orbit_radius, lon_there, target.v_circ)
        separation = float(np.linalg.norm(np.cross(r0, r1)))
        if separation < 1e-6 * body.orbit_radius * target.orbit_radius:
            # Collinear endpoints leave the transfer plane undefined.
            return None
        try:
            v_depart, v_arrive = izzo(mu, r0, r1, tof, 0, True, True, 35, 1e-8)
        except (ValueError, RuntimeError):
            return None
        excess_out = v_depart - v_body
        if excess_in is None:
            # Departure: no flyby to turn anything, so the whole excess is
            # bought with the Oberth burn at 200 km, starting from whatever the
            # mass already carries there (escape, or the cycle orbit's periapsis
            # speed once the growth loop is closed).
            speed = float(np.linalg.norm(excess_out))
            departure_burn = (
                float(np.sqrt(params.flyby.v_esc_leo**2 + speed * speed))
                - params.flyby.v_depart_from
            )
        elif powered_nodes:
            node_burns.append(_powered_node_burn(excess_in, excess_out, body)[0])
        else:
            node_burns.append(_flyby_mismatch_burn(excess_in, excess_out, body))
        excess_in = v_arrive - v_target
        clock += tof
    if excess_in is None:
        return None
    return _LadderPricing(
        departure_burn=departure_burn,
        node_total=float(sum(node_burns)),
        node_burns=node_burns,
        arrival_excess=float(np.linalg.norm(excess_in)),
        arrival_excess_vector=excess_in,
        arrival_time=clock,
        arrival_longitude=lon_there,
    )


def _body_state(
    radius: float, longitude: float, v_circ: float
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cartesian position and velocity of a body on its circular orbit.

    The chain's own model is planar and circular, so the third component is
    always zero; Lambert wants 3-vectors regardless.

    Args:
        radius: Heliocentric orbit radius (km).
        longitude: Heliocentric longitude (rad).
        v_circ: Circular speed (km/s).

    Returns:
        (position km, velocity km/s) as 3-vectors.
    """
    c, s = float(np.cos(longitude)), float(np.sin(longitude))
    position = np.array([radius * c, radius * s, 0.0])
    velocity = np.array([-v_circ * s, v_circ * c, 0.0])
    return position, velocity


def _powered_node_burn(
    excess_in: npt.NDArray[np.float64],
    excess_out: npt.NDArray[np.float64],
    body: _AssistBody,
) -> Tuple[float, float]:
    """Burn at a *powered* gravity assist supplying the required turn (km/s).

    The honest node model, replacing :func:`_flyby_mismatch_burn`. An impulsive
    tangential burn at periapsis joins an inbound hyperbola of eccentricity
    ``e_in`` to an outbound one of ``e_out``, so the total bend is
    ``asin(1/e_in) + asin(1/e_out)`` -- the same split-hyperbola geometry
    :func:`_powered_flyby_leg` uses at Jupiter. The unpowered formula
    ``sin(delta/2) = 1/e`` does **not** apply across a burn.

    Periapsis radius and burn magnitude are two knobs against two demands (the
    outgoing excess *speed* and the *turn*), so the node is solved, not
    estimated. Both eccentricities rise with ``r_p``, so the bend falls
    monotonically in it and the solve is a 1-D root find: the turn alone fixes
    the geometry, and the burn follows from it.

    The consequence is that **a node that must turn hard is cheap**: a bigger
    turn forces a lower periapsis, which deepens the well and so deepens the
    Oberth discount on the speed change. Cost is therefore *non-monotonic* in
    the turn. A 5 -> 7 km/s node at Venus costs the full 2.0 km/s at zero turn
    (nothing to bend around, so no Oberth), falls to ~1.17 km/s by 60 degrees,
    and jumps to ~3.10 km/s at 90 degrees, past the 72.9-degree bend limit.
    Against :func:`_flyby_mismatch_burn` the comparison cuts both ways: this is
    cheaper in the middle of the range, equal at zero turn, and dearer past the
    limit, because the retired model grants ``2*asin(1/e_in)`` of bend -- both
    halves at the *incoming* eccentricity -- and so overstates the limit itself
    (84.4 vs 72.9 degrees for that node).

    Args:
        excess_in: Incoming body-relative excess velocity (km/s, 3-vector).
        excess_out: Excess velocity the next leg needs (km/s, 3-vector).
        body: The flyby body.

    Returns:
        (burn magnitude km/s, periapsis radius km). When even the periapsis
        floor cannot supply the turn, the flyby bends as far as it can and the
        residual angle is bought with a deep-space kick, which makes the result
        an upper bound in that regime only.
    """
    speed_in = float(np.linalg.norm(excess_in))
    speed_out = float(np.linalg.norm(excess_out))
    if speed_in < 1e-9 or speed_out < 1e-9:
        return abs(speed_out - speed_in), body.min_periapsis

    def bend(periapsis: float) -> float:
        ecc_in = conic_kernel.hyperbolic_eccentricity(body.mu, periapsis, speed_in)
        ecc_out = conic_kernel.hyperbolic_eccentricity(body.mu, periapsis, speed_out)
        return conic_kernel.powered_bend_angle(ecc_in, ecc_out)

    def burn(periapsis: float) -> float:
        v_in = float(np.sqrt(speed_in * speed_in + 2.0 * body.mu / periapsis))
        v_out = float(np.sqrt(speed_out * speed_out + 2.0 * body.mu / periapsis))
        return abs(v_out - v_in)

    cos_turn = float(
        np.clip(np.dot(excess_in, excess_out) / (speed_in * speed_out), -1.0, 1.0)
    )
    turn = float(np.arccos(cos_turn))
    floor = body.min_periapsis
    bend_max = bend(floor)
    if turn > bend_max:
        # Even a grazing pass cannot point the excess where it must go. Bend as
        # far as the floor allows, fix the speed there (with Oberth), then buy
        # the leftover angle with a deep-space kick.
        residual = turn - bend_max
        kick = 2.0 * speed_out * float(np.sin(0.5 * residual))
        return burn(floor) + kick, floor
    # Bend falls monotonically in periapsis radius; bracket upward and root it.
    high = floor
    for _ in range(200):
        if bend(high) <= turn:
            break
        high *= 2.0
    else:
        # The bend is strictly positive at every finite periapsis, so a turn at
        # or near zero exhausts the bracket. There is no flyby geometry that
        # bends this little: the pass is irrelevant and the speed change is
        # bought out at infinity, with no Oberth discount. Falling back to
        # burn(floor) here would hand the caller the *deepest* discount for the
        # turn that has earned none of it.
        return abs(speed_out - speed_in), high
    if high == floor:
        return burn(floor), floor
    periapsis = float(brentq(lambda r: bend(r) - turn, floor, high, xtol=1e-6))
    return burn(periapsis), periapsis


def _flyby_mismatch_burn(
    excess_in: npt.NDArray[np.float64],
    excess_out: npt.NDArray[np.float64],
    body: _AssistBody,
) -> float:
    """Burn needed at a flyby that cannot supply the turn unaided (km/s).

    The **retired** node model, kept to reproduce ADR 0007's numbers and to
    document the overcharge. It charges the excess-velocity change *at
    infinity*, so it collects no Oberth leverage, and it bounds the bend with
    the unpowered ``sin(delta/2) = 1/e`` -- which CONTEXT.md warns does not hold
    across a burn. Prefer :func:`_powered_node_burn`.


    An unpowered flyby rotates the excess velocity but never rescales it, and
    can only rotate it so far. Phasing the *next* leg generally demands both a
    different magnitude and a turn beyond the bend limit, and the gap is what a
    deep-space maneuver has to pay for.

    The bend limit is evaluated at the incoming speed. Charging the shortfall
    this way is the standard MGA node model: turn as far as the flyby allows
    for free, then buy the rest. It is an estimate, not an optimum -- burning
    and bending together (the burn changes the speed, which changes the limit)
    can beat it -- so treat the result as an upper bound on the node's cost.

    Args:
        excess_in: Incoming body-relative excess velocity (km/s, 3-vector).
        excess_out: Excess velocity the next leg needs (km/s, 3-vector).
        body: The flyby body.

    Returns:
        Burn magnitude, zero when the flyby can do the whole job unaided.
    """
    speed_in = float(np.linalg.norm(excess_in))
    speed_out = float(np.linalg.norm(excess_out))
    if speed_in < 1e-9 or speed_out < 1e-9:
        return abs(speed_out - speed_in)
    ecc = conic_kernel.hyperbolic_eccentricity(body.mu, body.min_periapsis, speed_in)
    bend_limit = conic_kernel.unpowered_bend_angle(ecc)
    cos_turn = float(
        np.clip(np.dot(excess_in, excess_out) / (speed_in * speed_out), -1.0, 1.0)
    )
    turn = float(np.arccos(cos_turn))
    if turn <= bend_limit:
        # The flyby can point the excess where it must go; only the speed is
        # wrong, and a colinear burn is the cheapest way to fix that.
        return abs(speed_out - speed_in)
    # Bend as far as allowed, then close the remaining angle by burning.
    residual_turn = turn - bend_limit
    return float(
        np.sqrt(
            max(
                0.0,
                speed_in * speed_in
                + speed_out * speed_out
                - 2.0 * speed_in * speed_out * np.cos(residual_turn),
            )
        )
    )


def _jupiter_assist_body(params: _AssistChainParams) -> _AssistBody:
    """Jupiter as a ladder body, so the Jovian leg can be phased like any other.

    ``_jovian_terminal`` and ``_powered_jovian_terminal`` reach Jupiter through
    :func:`conic_kernel.conic_radius_crossings`, which finds crossings of
    Jupiter's orbit *radius* and assumes Jupiter is there. That is the model's
    largest hole: it
    phases Venus and Earth while letting Jupiter be anywhere, so every chain
    result built on it is a lower bound.

    Handing Jupiter to :func:`_phased_ladder_burn` as an ordinary body closes it.
    The Jovian leg becomes a Lambert arc to where Jupiter actually is, and the
    last Earth flyby stops being a free rotation: it must supply the excess that
    arc demands, and is charged when it cannot.

    Args:
        params: The assist-chain parameter block (for the Sun's mu and Jupiter's
            periapsis floor).

    Returns:
        The :class:`_AssistBody` for Jupiter.
    """
    orbit_radius = float(JUPITER_A.to_value(u.km))
    return _AssistBody(
        symbol="J",
        name=str(Jupiter.name),
        orbit_radius=orbit_radius,
        mu=float(Jupiter.k.to_value(u.km**3 / u.s**2)),
        min_periapsis=params.flyby.periapsis_floor,
        v_circ=float(np.sqrt(params.flyby.mu_sun / orbit_radius)),
    )


def _phased_jovian_flyby(
    excess_in: npt.NDArray[np.float64],
    jupiter_longitude: float,
    periapsis_radius: float,
    flyby_burn: float,
    bend_sign: float,
    params: _AssistChainParams,
) -> Optional[_ReturnLeg]:
    """Powered Jovian flyby from a *phased* arrival, into the retrograde return.

    The counterpart to :func:`_jupiter_assist_body`: once the Jovian leg is a
    Lambert arc onto Jupiter's true position, the arrival excess is a vector in
    the heliocentric frame and Jupiter's longitude is known, so the flyby needs
    no radius-crossing search. It only bends and rescales that excess.

    The burn moves the turn as well as the speed -- ``e_out`` comes from the
    post-burn excess, so the bend is ``asin(1/e_in) + asin(1/e_out)`` and
    speeding up costs bend.

    Args:
        excess_in: Jupiter-relative arrival excess (km/s, heliocentric 3-vector).
        jupiter_longitude: Jupiter's heliocentric longitude at arrival (rad).
        periapsis_radius: Perijove radius, center-based (km).
        flyby_burn: Impulsive perijove burn, >= 0 (km/s).
        bend_sign: Which side Jupiter is passed on (+1 or -1).
        params: The assist-chain parameter block.

    Returns:
        The scored :class:`_ReturnLeg`, or None if the flyby is infeasible (below
        the perijove floor, captured by the burn, or no retrograde 1 AU return).
    """
    flyby = params.flyby
    if periapsis_radius < flyby.periapsis_floor * (1.0 - 1e-12) or flyby_burn < 0.0:
        return None
    w_in = float(np.linalg.norm(excess_in))
    if w_in < 1e-6:
        return None
    mu_j = flyby.mu_jupiter
    v_peri_out = (
        float(np.sqrt(w_in * w_in + 2.0 * mu_j / periapsis_radius)) + flyby_burn
    )
    w_out_sq = v_peri_out * v_peri_out - 2.0 * mu_j / periapsis_radius
    if w_out_sq <= 0.0:
        # A retro burn past the escape margin captures the craft.
        return None
    w_out = float(np.sqrt(w_out_sq))
    ecc_in = conic_kernel.hyperbolic_eccentricity(mu_j, periapsis_radius, w_in)
    ecc_out = conic_kernel.hyperbolic_eccentricity(mu_j, periapsis_radius, w_out)
    turn = bend_sign * conic_kernel.powered_bend_angle(ecc_in, ecc_out)
    cos_t, sin_t = float(np.cos(turn)), float(np.sin(turn))
    scale = w_out / w_in
    excess_out = scale * np.array(
        [
            excess_in[0] * cos_t - excess_in[1] * sin_t,
            excess_in[0] * sin_t + excess_in[1] * cos_t,
            0.0,
        ]
    )
    lon = jupiter_longitude
    t_hat = np.array([-np.sin(lon), np.cos(lon), 0.0])
    r_hat = np.array([np.cos(lon), np.sin(lon), 0.0])
    v_helio = flyby.v_jupiter_orbit * t_hat + excess_out
    return _flyby_return_leg(
        float(np.dot(v_helio, t_hat)), float(np.dot(v_helio, r_hat)), flyby
    )
