"""Earth occultation and finite-burn turning on the circular 3S return.

The circular three-synodic Jupiter return in
:mod:`src.circular_resonance_impulse` has two apparent departure-hyperbola
mirrors.  The impulse ledger chooses the mirror that makes the returning
projectiles less nearly head-on.  That choice is only useful if the projectile
stream can reach the burn point without first crossing Earth.

This module resolves both pieces of geometry that an impulsive burn hides:

* It constructs both departure mirrors at the 600 km burn radius and solves
  the two Kepler hyperbolae by which the prescribed incoming excess vector can
  reach each mirror.  A route is visible when it is still inbound at the
  intercept, or when its already-passed periapsis stays above Earth.
* It integrates a 20-minute, approximately constant-thrust burn on the 20-day
  parking orbit.  The burn placement is optimized against final orbital energy
  and the impact angle is recomputed as Earth turns the vehicle velocity.
* It separately optimizes an uncanted 3000 K axial nozzle with pulse-dependent
  hydrogen loading, retains the projectile's full momentum vector and requires
  the final speed and direction to close on the exact 3S departure asymptote.

The reference trajectory is the minimum-departure 3S circular closure from
ADR 0012 (91 x 121 verification grid followed by continuous refinement):
``v_inf,out = 11.5642794307 km/s``, ``v_inf,in = 55.1783413091 km/s`` and
``aim separation = 144.899950401 deg``.  These inputs are arguments rather
than hidden search results so the finite-burn calculation remains cheap and
reproducible.

Run ``make orbit-turn`` for the compact report.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.bodies import Earth
from scipy.integrate import OdeSolution, solve_ivp
from scipy.optimize import brentq, minimize, minimize_scalar

from src.astro_constants import PUFFSAT_CYCLE_ORBIT_PERIOD
from src.chamber_isp import HYDROGEN_5500K, REFERENCE_CLOSING_SPEED
from src.finite_burn_loss import DEPARTURE_ALTITUDE

THREE_SYNODIC_DEPARTURE_VINF = 11.564279430721319 * u.km / u.s
THREE_SYNODIC_RETURN_VINF = 55.17834130911211 * u.km / u.s
THREE_SYNODIC_AIM_SEPARATION = 144.89995040100516 * u.deg
REFERENCE_BURN_TIME = 20.0 * u.min
REFERENCE_EXHAUST_SPEED = 11.0 * u.km / u.s
REFERENCE_SLUG_RATIO = 3.0
REFERENCE_COLLIMATION = 0.8
UNCANTED_CHAMBER_TEMPERATURE = 3000.0 * u.K
UNCANTED_EXHAUST_VELOCITY_EFFICIENCY = 0.85

# The 5500 K chamber's reference loading pins mixed hot-gas energy without
# inventing a new hydrogen heat capacity.  Scale that energy linearly with
# temperature for the requested 3000 K sensitivity.  The new case has no plug:
# the reference plug appears here only because it shared the 5500 K energy.
_REFERENCE_MIXED_SPECIFIC_ENERGY = (
    0.5
    * REFERENCE_CLOSING_SPEED**2
    / (HYDROGEN_5500K.reference_slug_ratio + 1.0 + HYDROGEN_5500K.plug_ratio)
)
UNCANTED_MIXED_SPECIFIC_ENERGY = _REFERENCE_MIXED_SPECIFIC_ENERGY * (
    UNCANTED_CHAMBER_TEMPERATURE / HYDROGEN_5500K.temperature
)
UNCANTED_IDEAL_EXHAUST_SPEED = np.sqrt(2.0 * UNCANTED_MIXED_SPECIFIC_ENERGY)
UNCANTED_EXHAUST_SPEED = (
    UNCANTED_EXHAUST_VELOCITY_EFFICIENCY * UNCANTED_IDEAL_EXHAUST_SPEED
)

_RTOL = 1.0e-10
_ATOL = 1.0e-9
_PROFILE_SAMPLES = 1201


@dataclass(frozen=True)
class IncomingRoute:
    """One Earth hyperbola connecting an incoming asymptote to a point.

    Attributes:
        impact_parameter: Signed dimensionless impact parameter
            ``h v_inf / mu``.
        velocity: Projectile velocity at the intercept, km/s.
        radial_velocity: Outward radial component at the intercept, km/s.
        periapsis_radius: Radius the projectile would reach if it continued,
            km.  It may be below Earth when the intercept is still inbound.
        visible: Whether the projectile reaches the intercept before Earth.
    """

    impact_parameter: float
    velocity: npt.NDArray[np.float64]
    radial_velocity: float
    periapsis_radius: float
    visible: bool


@dataclass(frozen=True)
class PeriapsisMirror:
    """One sign of the 3S departure hyperbola at the burn point.

    Attributes:
        angular_momentum_sign: ``+1`` for counter-clockwise departure and
            ``-1`` for its mirror.
        departure_half_turn: Angle from periapsis velocity to the outgoing
            excess vector, degrees.
        projectile_periapsis_altitude: Closest altitude of the least-blocked
            incoming route, km.
        projectile_radial_velocity: Its radial velocity at intercept, km/s.
        blocked: Whether Earth lies on both possible incoming routes.
        impact_angle: Vehicle-frame angle from thrust to incoming relative
            velocity, degrees; 180 degrees is exactly head-on.
        exhaust_cant: Ideal plume cant needed to remove transverse momentum at
            the reference slug ratio, degrees.
    """

    angular_momentum_sign: int
    departure_half_turn: u.Quantity
    projectile_periapsis_altitude: u.Quantity
    projectile_radial_velocity: u.Quantity
    blocked: bool
    impact_angle: u.Quantity
    exhaust_cant: u.Quantity


@dataclass(frozen=True)
class FiniteBurnTurn:
    """Optimized placement and angle history of one finite departure burn.

    Attributes:
        steered: Thrust follows the velocity when true; otherwise it stays in
            the periapsis-tangent direction.
        burn_time: Total burn duration.
        seconds_before_reference_periapsis: Burn time before the periapsis of
            the unpowered parking orbit.
        seconds_after_reference_periapsis: Burn time after that reference.
        seconds_before_closest_approach: Burn time before the powered path's
            actual closest approach.
        seconds_after_closest_approach: Burn time after closest approach.
        closest_altitude: Minimum altitude on the powered path.
        impulsive_burn: Instantaneous burn needed at reference periapsis.
        finite_burn: Ideal velocity integral needed to reach the same energy.
        finite_burn_loss: Difference between those two burns.
        timing_gain: Ideal burn saved by optimizing timing against a half-and-half
            split about the reference periapsis.
        off_head_on_start: Collision angle away from exactly head-on at light.
        off_head_on_end: The same angle at cutoff.
        off_head_on_maximum: Largest angle away from head-on during the burn.
        time_above_22_degrees: Fraction of burn time more than 22 degrees away
            from head-on.
        exhaust_cant_minimum: Smallest ideal plume-steering angle.
        exhaust_cant_maximum: Largest ideal plume-steering angle.
        exhaust_cant_mean: Time-mean plume-steering angle.
        exhaust_cant_impulse_mean: Mean weighted by delivered ideal impulse.
        angle_gain_over_head_on: Delivered mass relative to an exactly head-on
            stream at the same projectile speed and acceleration history.
    """

    steered: bool
    burn_time: u.Quantity
    seconds_before_reference_periapsis: u.Quantity
    seconds_after_reference_periapsis: u.Quantity
    seconds_before_closest_approach: u.Quantity
    seconds_after_closest_approach: u.Quantity
    closest_altitude: u.Quantity
    impulsive_burn: u.Quantity
    finite_burn: u.Quantity
    finite_burn_loss: u.Quantity
    timing_gain: u.Quantity
    off_head_on_start: u.Quantity
    off_head_on_end: u.Quantity
    off_head_on_maximum: u.Quantity
    time_above_22_degrees: float
    exhaust_cant_minimum: u.Quantity
    exhaust_cant_maximum: u.Quantity
    exhaust_cant_mean: u.Quantity
    exhaust_cant_impulse_mean: u.Quantity
    angle_gain_over_head_on: float


@dataclass(frozen=True)
class UncantedThermalBurn:
    """A 3000 K axial-nozzle burn with the projectile momentum retained.

    Attributes:
        front_side: Whether this is the less-head-on mirror hidden at 600 km.
        reference_periapsis_altitude: Periapsis altitude of the unpowered
            20-day parking orbit.
        periapsis_position_angle: Position angle of that periapsis relative to
            the required outgoing excess vector.
        seconds_before_reference_periapsis: Burn time before that periapsis.
        seconds_after_reference_periapsis: Burn time after it.
        hydrogen_spent_fraction: Onboard hydrogen spent per unit initial mass.
        delivered_fraction: Final vehicle mass per unit initial mass.
        initial_to_delivered_mass_ratio: Initial over final vehicle mass.
        hydrogen_per_delivered: Hydrogen spent per unit delivered vehicle.
        impactor_mass_fraction: External projectile mass used per unit initial
            vehicle mass; it is not carried at ignition.
        closest_altitude: Minimum vehicle altitude during the powered arc.
        minimum_projectile_clearance: Smallest route clearance above Earth.
        closing_speed_minimum: Smallest vehicle-relative projectile speed.
        closing_speed_maximum: Largest vehicle-relative projectile speed.
        slug_ratio_minimum: Smallest hydrogen/projectile mass ratio.
        slug_ratio_maximum: Largest hydrogen/projectile mass ratio.
        off_head_on_start: Collision direction away from head-on at ignition.
        off_head_on_end: Collision direction away from head-on at cutoff.
        off_head_on_maximum: Largest departure from head-on during the burn.
        integrated_axial_delta_v: Integral of axial chamber acceleration.
        integrated_lateral_delta_v: Signed integral of uncancelled lateral
            acceleration in the instantaneous velocity frame.
        integrated_thrust_delta_v: Integral of the thrust-acceleration
            vector's magnitude.
        impulsive_delta_v: Tangential impulse at the reference periapsis for
            the same outgoing excess speed.
        finite_burn_penalty: Integrated thrust delta-v minus that impulse.
        missed_projectile_periapsis_altitude: Lowest eventual periapsis of a
            projectile that misses instead of being consumed at interception.
        missed_projectile_earth_impact_fraction: Projectile-mass fraction whose
            unconsumed continuation intersects Earth.
        outgoing_vinf: Solved outgoing excess speed.
        outgoing_angle_error: Solved excess-vector angle from the 3S target.
    """

    front_side: bool
    reference_periapsis_altitude: u.Quantity
    periapsis_position_angle: u.Quantity
    seconds_before_reference_periapsis: u.Quantity
    seconds_after_reference_periapsis: u.Quantity
    hydrogen_spent_fraction: float
    delivered_fraction: float
    initial_to_delivered_mass_ratio: float
    hydrogen_per_delivered: float
    impactor_mass_fraction: float
    closest_altitude: u.Quantity
    minimum_projectile_clearance: u.Quantity
    closing_speed_minimum: u.Quantity
    closing_speed_maximum: u.Quantity
    slug_ratio_minimum: float
    slug_ratio_maximum: float
    off_head_on_start: u.Quantity
    off_head_on_end: u.Quantity
    off_head_on_maximum: u.Quantity
    integrated_axial_delta_v: u.Quantity
    integrated_lateral_delta_v: u.Quantity
    integrated_thrust_delta_v: u.Quantity
    impulsive_delta_v: u.Quantity
    finite_burn_penalty: u.Quantity
    missed_projectile_periapsis_altitude: u.Quantity
    missed_projectile_earth_impact_fraction: float
    outgoing_vinf: u.Quantity
    outgoing_angle_error: u.Quantity


def _rotation(angle: float) -> npt.NDArray[np.float64]:
    """Return a planar rotation matrix."""

    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _unit(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a vector normalized to unit length."""

    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)


def _orbit(
    altitude: u.Quantity = DEPARTURE_ALTITUDE,
    period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD,
) -> Tuple[float, float, float, float]:
    """Return ``mu``, Earth radius, periapsis radius and parking speed."""

    mu = float(Earth.k.to_value(u.km**3 / u.s**2))
    earth_radius = float(Earth.R.to_value(u.km))
    radius = float((Earth.R + altitude).to_value(u.km))
    seconds = float(period.to_value(u.s))
    semimajor = (mu * (seconds / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    speed = float(np.sqrt(mu * (2.0 / radius - 1.0 / semimajor)))
    return mu, earth_radius, radius, speed


def _coast_before_periapsis(
    mu: float, radius: float, speed: float, seconds: float
) -> npt.NDArray[np.float64]:
    """Propagate the unpowered parking orbit backward from periapsis."""

    def rhs(_: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        position = state[:2]
        return np.concatenate(
            (state[2:], -mu * position / np.linalg.norm(position) ** 3)
        )

    periapsis = np.array([radius, 0.0, 0.0, speed], dtype=np.float64)
    solution = solve_ivp(rhs, (0.0, -seconds), periapsis, rtol=_RTOL, atol=_ATOL)
    return np.asarray(solution.y[:, -1], dtype=np.float64)


def _incoming_routes(
    mu: float,
    earth_radius: float,
    position: npt.NDArray[np.float64],
    excess_speed: float,
    incoming_direction: npt.NDArray[np.float64],
) -> List[IncomingRoute]:
    """Solve the two hyperbolae from one incoming asymptote to ``position``.

    The signed dimensionless impact parameter is ``q = h v_inf / mu``.  At
    incoming infinity the eccentricity vector is
    ``e = u + q * rot_cw(u)``.  Substitution in ``p/r = 1 + e dot r_hat`` gives
    one quadratic in ``q`` and therefore the two routes around Earth.
    """

    radius = float(np.linalg.norm(position))
    radial = _unit(position)
    transverse = _rotation(0.5 * np.pi) @ radial
    incoming = _unit(incoming_direction)
    clockwise = np.array([incoming[1], -incoming[0]], dtype=np.float64)
    coefficient = mu / (radius * excess_speed**2)
    linear = float(np.dot(clockwise, radial))
    constant = 1.0 + float(np.dot(incoming, radial))
    roots = np.roots([coefficient, -linear, -constant])
    routes: List[IncomingRoute] = []
    for root in roots:
        q = float(np.real(root))
        angular_momentum = q * mu / excess_speed
        eccentricity_vector = incoming + q * clockwise
        radial_speed = float(
            -mu / angular_momentum * np.dot(eccentricity_vector, transverse)
        )
        transverse_speed = angular_momentum / radius
        velocity = radial_speed * radial + transverse_speed * transverse
        eccentricity = float(np.linalg.norm(eccentricity_vector))
        periapsis = angular_momentum**2 / mu / (1.0 + eccentricity)
        # An inbound projectile is intercepted before reaching its mathematical
        # periapsis.  An outbound one must already have cleared Earth there.
        visible = radial_speed <= 0.0 or periapsis >= earth_radius
        routes.append(
            IncomingRoute(
                impact_parameter=q,
                velocity=np.asarray(velocity, dtype=np.float64),
                radial_velocity=radial_speed,
                periapsis_radius=periapsis,
                visible=visible,
            )
        )
    return routes


def occultation_half_angle(
    altitude: u.Quantity = DEPARTURE_ALTITUDE,
) -> u.Quantity:
    """Angular radius of Earth as seen from the departure burn altitude.

    An upstream projectile direction inside this cone around nadir is hidden by
    Earth in the straight-line limit.  Hyperbolic route checks in this module
    use periapsis rather than this approximation.
    """

    ratio = float((Earth.R / (Earth.R + altitude)).to_value(u.one))
    return np.arcsin(ratio) * u.rad


def _cant(impact_angle: float, slug_ratio: float) -> float:
    """Ideal exhaust cant that cancels transverse projectile momentum."""

    projectile_fraction = 1.0 / (1.0 + slug_ratio)
    return float(
        np.arcsin(
            np.clip(
                np.sqrt(projectile_fraction) * abs(np.sin(impact_angle)),
                0.0,
                1.0,
            )
        )
    )


def three_synodic_periapsis_mirrors(
    departure_vinf: u.Quantity = THREE_SYNODIC_DEPARTURE_VINF,
    return_vinf: u.Quantity = THREE_SYNODIC_RETURN_VINF,
    aim_separation: u.Quantity = THREE_SYNODIC_AIM_SEPARATION,
    slug_ratio: float = REFERENCE_SLUG_RATIO,
    altitude: u.Quantity = DEPARTURE_ALTITUDE,
    period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD,
) -> Tuple[PeriapsisMirror, PeriapsisMirror]:
    """Classify both departure mirrors by incoming-stream visibility.

    Returns the counter-clockwise and clockwise departure mirrors, in that
    order.  If a mirror is blocked, its reported route is the one whose
    mathematical periapsis comes closest to clearing Earth.
    """

    if slug_ratio <= 0.0:
        raise ValueError("slug_ratio must be positive")
    mu, earth_radius, burn_radius, parking_speed = _orbit(altitude, period)
    departure = float(departure_vinf.to_value(u.km / u.s))
    incoming_speed = float(return_vinf.to_value(u.km / u.s))
    separation = float(aim_separation.to_value(u.rad))
    outgoing = np.array([1.0, 0.0], dtype=np.float64)
    incoming = _rotation(separation) @ outgoing
    eccentricity = 1.0 + burn_radius * departure**2 / mu
    half_turn = float(np.arcsin(1.0 / eccentricity))
    mirrors: List[PeriapsisMirror] = []
    for sign in (1, -1):
        tangent = _rotation(-sign * half_turn) @ outgoing
        radial = _rotation(-sign * 0.5 * np.pi) @ tangent
        position = burn_radius * radial
        routes = _incoming_routes(mu, earth_radius, position, incoming_speed, incoming)
        visible = [route for route in routes if route.visible]
        route = (
            max(visible, key=lambda item: item.periapsis_radius)
            if visible
            else max(routes, key=lambda item: item.periapsis_radius)
        )
        vehicle_velocity = parking_speed * tangent
        relative = route.velocity - vehicle_velocity
        impact_angle = float(
            np.arccos(
                np.clip(np.dot(relative, tangent) / np.linalg.norm(relative), -1.0, 1.0)
            )
        )
        mirrors.append(
            PeriapsisMirror(
                angular_momentum_sign=sign,
                departure_half_turn=(half_turn * u.rad).to(u.deg),
                projectile_periapsis_altitude=(route.periapsis_radius - earth_radius)
                * u.km,
                projectile_radial_velocity=route.radial_velocity * u.km / u.s,
                blocked=not bool(visible),
                impact_angle=(impact_angle * u.rad).to(u.deg),
                exhaust_cant=(_cant(impact_angle, slug_ratio) * u.rad).to(u.deg),
            )
        )
    return mirrors[0], mirrors[1]


def blocked_mirror_clearance_angle(
    departure_vinf: u.Quantity = THREE_SYNODIC_DEPARTURE_VINF,
    return_vinf: u.Quantity = THREE_SYNODIC_RETURN_VINF,
    aim_separation: u.Quantity = THREE_SYNODIC_AIM_SEPARATION,
    altitude: u.Quantity = DEPARTURE_ALTITUDE,
) -> u.Quantity:
    """Smallest rotation of the incoming asymptote that clears blocked mirror.

    The favorable direction is rooted from the reference aim to 90 degrees
    farther around Earth.  This is a geometry diagnostic, not a proposed
    maneuver: changing the return asymptote breaks the synodic closure.
    """

    mu, earth_radius, radius, _ = _orbit(altitude)
    departure = float(departure_vinf.to_value(u.km / u.s))
    incoming_speed = float(return_vinf.to_value(u.km / u.s))
    separation = float(aim_separation.to_value(u.rad))
    half_turn = float(np.arcsin(1.0 / (1.0 + radius * departure**2 / mu)))
    tangent = _rotation(half_turn) @ np.array([1.0, 0.0])
    radial = _rotation(0.5 * np.pi) @ tangent
    position = radius * radial

    def margin(change: float) -> float:
        incoming = _rotation(separation + change) @ np.array([1.0, 0.0])
        routes = _incoming_routes(mu, earth_radius, position, incoming_speed, incoming)
        if any(route.radial_velocity <= 0.0 for route in routes):
            return radius - earth_radius
        return max(route.periapsis_radius for route in routes) - earth_radius

    change = float(brentq(margin, 0.0, 0.5 * np.pi, xtol=1.0e-12))
    return (change * u.rad).to(u.deg)


def _physical_stream_geometry(
    mu: float,
    earth_radius: float,
    radius: float,
    departure_vinf: float,
    return_vinf: float,
    aim_separation: float,
) -> Tuple[npt.NDArray[np.float64], float]:
    """Incoming direction and anchor impact parameter in local burn axes."""

    half_turn = float(np.arcsin(1.0 / (1.0 + radius * departure_vinf**2 / mu)))
    # The visible +h departure mirror has, in axes where outgoing v_inf is +x,
    # tangent angle -alpha and radial angle -pi/2-alpha.  Rotate the latter to
    # local +x, leaving the parking velocity on local +y.
    incoming_angle = aim_separation + 0.5 * np.pi + half_turn
    incoming = np.array(
        [np.cos(incoming_angle), np.sin(incoming_angle)], dtype=np.float64
    )
    routes = _incoming_routes(
        mu,
        earth_radius,
        np.array([radius, 0.0], dtype=np.float64),
        return_vinf,
        incoming,
    )
    visible = [route for route in routes if route.visible]
    if not visible:
        raise RuntimeError("the reference 3S mirror has no visible incoming route")
    anchor = max(visible, key=lambda route: abs(route.impact_parameter))
    return incoming, anchor.impact_parameter


def _select_visible_route(
    routes: List[IncomingRoute], anchor_impact_parameter: float
) -> IncomingRoute:
    """Select the visible route continuous with the periapsis intercept."""

    visible = [route for route in routes if route.visible]
    if not visible:
        raise RuntimeError("Earth occults the incoming stream during the burn")
    return min(
        visible,
        key=lambda route: abs(route.impact_parameter - anchor_impact_parameter),
    )


def uncanted_slug_ratio(
    closing_speed: u.Quantity,
    temperature: u.Quantity = UNCANTED_CHAMBER_TEMPERATURE,
) -> float:
    """Hydrogen mass per projectile mass that holds one chamber temperature.

    The hot-mixture specific energy is anchored to the repository's 5500 K
    hydrogen chamber and scaled linearly with temperature.  With no carried
    plug, energy conservation gives ``k = w^2 / (2 u_hot) - 1``.

    Args:
        closing_speed: Projectile speed relative to the vehicle.
        temperature: Common hydrogen/projectile temperature after impact.

    Returns:
        Hydrogen/projectile mass ratio ``k``.

    Raises:
        ValueError: If the temperature or resulting loading is nonphysical.
    """

    if temperature <= 0.0 * u.K:
        raise ValueError("temperature must be positive")
    hot_energy = _REFERENCE_MIXED_SPECIFIC_ENERGY * (
        temperature / HYDROGEN_5500K.temperature
    )
    ratio = float((closing_speed**2 / (2.0 * hot_energy)).to_value(u.one)) - 1.0
    if ratio <= 0.0:
        raise ValueError("closing speed cannot heat the projectile to temperature")
    return ratio


@dataclass(frozen=True)
class _UncantedEvaluation:
    """One endpoint and sampled path used by the uncanted optimizer."""

    residual: npt.NDArray[np.float64]
    clearance: float
    missed_periapsis: float
    times: npt.NDArray[np.float64]
    states: npt.NDArray[np.float64]


def _uncanted_route(routes: List[IncomingRoute], front_side: bool) -> IncomingRoute:
    """Select the impact-parameter branch continuous with one 3S mirror."""

    chooser = max if front_side else min
    return chooser(routes, key=lambda item: item.impact_parameter)


def _coast_with_sense(
    mu: float,
    radius: float,
    speed: float,
    seconds: float,
    sense: int,
) -> npt.NDArray[np.float64]:
    """Propagate either parking-orbit sense backward from periapsis."""

    def rhs(_: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        position = state[:2]
        return np.concatenate(
            (state[2:], -mu * position / np.linalg.norm(position) ** 3)
        )

    periapsis = np.array([radius, 0.0, 0.0, sense * speed], dtype=np.float64)
    solution = solve_ivp(rhs, (0.0, -seconds), periapsis, rtol=_RTOL, atol=_ATOL)
    return np.asarray(solution.y[:, -1], dtype=np.float64)


def _outgoing_asymptote(
    mu: float, state: npt.NDArray[np.float64]
) -> Tuple[float, float]:
    """Return hyperbolic excess speed and outgoing direction angle."""

    position = state[:2]
    velocity = state[2:4]
    radius = float(np.linalg.norm(position))
    energy = 0.5 * float(velocity @ velocity) - mu / radius
    if energy <= 0.0:
        return 0.0, np.pi
    excess = float(np.sqrt(2.0 * energy))
    angular_momentum = float(position[0] * velocity[1] - position[1] * velocity[0])
    eccentricity_vector = (
        (float(velocity @ velocity) - mu / radius) * position
        - float(position @ velocity) * velocity
    ) / mu
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    if eccentricity <= 1.0:
        return excess, np.pi
    true_anomaly = float(np.arccos(-1.0 / eccentricity))
    outgoing = (
        _rotation(np.sign(angular_momentum) * true_anomaly)
        @ eccentricity_vector
        / eccentricity
    )
    return excess, float(np.arctan2(outgoing[1], outgoing[0]))


def _uncanted_evaluation(
    parameters: npt.NDArray[np.float64],
    front_side: bool,
    variable_altitude: bool,
    duration: float,
    hot_energy: float,
    exhaust_speed: float,
    fine: bool,
) -> _UncantedEvaluation:
    """Propagate one constant-hydrogen-flow axial-nozzle candidate."""

    orientation, before_fraction, spent = (float(value) for value in parameters[:3])
    altitude = (
        float(parameters[3]) * 10000.0
        if variable_altitude
        else float(DEPARTURE_ALTITUDE.to_value(u.km))
    )
    sense = -1 if front_side else 1
    mu, earth_radius, radius, parking_speed = _orbit(altitude * u.km)
    start = _coast_with_sense(
        mu,
        radius,
        parking_speed,
        before_fraction * duration,
        sense,
    )
    rotation = _rotation(orientation)
    start[:2] = rotation @ start[:2]
    start[2:] = rotation @ start[2:]
    initial = np.concatenate((start, np.array([1.0], dtype=np.float64)))
    hydrogen_flow = spent / duration
    incoming = _rotation(
        float(THREE_SYNODIC_AIM_SEPARATION.to_value(u.rad))
    ) @ np.array([1.0, 0.0], dtype=np.float64)
    projectile_excess = float(THREE_SYNODIC_RETURN_VINF.to_value(u.km / u.s))

    def rhs(_: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        position = state[:2]
        velocity = state[2:4]
        mass = float(state[4])
        route = _uncanted_route(
            _incoming_routes(mu, earth_radius, position, projectile_excess, incoming),
            front_side,
        )
        relative = route.velocity - velocity
        closing = float(np.linalg.norm(relative))
        slug_ratio = closing**2 / (2.0 * hot_energy) - 1.0
        axis = _unit(velocity)
        # Per kilogram of onboard hydrogen, the exhaust carries (k + 1) / k
        # kilograms backward while the external projectile brings w / k of
        # vehicle-relative momentum.  The latter is deliberately not canted
        # away or efficiency-derated.
        impulse = (
            1.0 + 1.0 / slug_ratio
        ) * exhaust_speed * axis + relative / slug_ratio
        gravity = -mu * position / np.linalg.norm(position) ** 3
        return np.concatenate(
            (
                velocity,
                gravity + hydrogen_flow / mass * impulse,
                np.array([-hydrogen_flow], dtype=np.float64),
            )
        )

    sample_count = _PROFILE_SAMPLES if fine else 121
    times = np.linspace(0.0, duration, sample_count)
    solution = solve_ivp(
        rhs,
        (0.0, duration),
        initial,
        t_eval=times,
        rtol=_RTOL if fine else 3.0e-8,
        atol=_ATOL if fine else 3.0e-8,
        max_step=2.0 if fine else 20.0,
    )
    if not solution.success:
        raise RuntimeError("uncanted burn propagation failed")
    states = np.asarray(solution.y, dtype=np.float64)
    outgoing_speed, outgoing_angle = _outgoing_asymptote(mu, states[:, -1])
    target_speed = float(THREE_SYNODIC_DEPARTURE_VINF.to_value(u.km / u.s))
    residual = np.array(
        [(outgoing_speed - target_speed) / target_speed, outgoing_angle],
        dtype=np.float64,
    )
    clearance = float("inf")
    missed_periapsis = float("inf")
    for position in states[:2].T:
        route = _uncanted_route(
            _incoming_routes(mu, earth_radius, position, projectile_excess, incoming),
            front_side,
        )
        state_altitude = float(np.linalg.norm(position)) - earth_radius
        route_clearance = (
            state_altitude
            if route.radial_velocity <= 0.0
            else route.periapsis_radius - earth_radius
        )
        clearance = min(clearance, state_altitude, route_clearance)
        missed_periapsis = min(missed_periapsis, route.periapsis_radius - earth_radius)
    return _UncantedEvaluation(
        residual=residual,
        clearance=clearance,
        missed_periapsis=missed_periapsis,
        times=times,
        states=states,
    )


def uncanted_thermal_burn(
    front_side: bool,
    burn_time: u.Quantity = REFERENCE_BURN_TIME,
    temperature: u.Quantity = UNCANTED_CHAMBER_TEMPERATURE,
    exhaust_velocity_efficiency: float = UNCANTED_EXHAUST_VELOCITY_EFFICIENCY,
    missed_periapsis_floor: u.Quantity | None = None,
) -> UncantedThermalBurn:
    """Optimize a 20-minute axial-nozzle burn onto the exact 3S asymptote.

    Hydrogen flow is constant, the projectile is externally supplied, and the
    hydrogen/projectile loading changes pulse by pulse to hold ``temperature``.
    Exhaust goes opposite instantaneous velocity.  Its speed is
    ``exhaust_velocity_efficiency`` times the ideal speed from the hot-mixture
    energy; the projectile's complete incoming momentum vector is retained.

    Args:
        front_side: Search the less-head-on mirror hidden at 600 km.  False
            searches the naturally visible mirror at the 600 km altitude floor.
        burn_time: Total powered duration.
        temperature: Common hot-mixture temperature.
        exhaust_velocity_efficiency: Actual over ideal exhaust velocity.
        missed_periapsis_floor: If set, every unconsumed projectile must retain
            at least this eventual Earth-periapsis altitude.  The optimizer may
            raise the parking periapsis to satisfy it.

    Returns:
        Optimized burn, including closure, loading and clearance diagnostics.

    Raises:
        ValueError: If an input is nonphysical.
        RuntimeError: If the constrained optimizer or final closure fails.
    """

    duration = float(burn_time.to_value(u.s))
    if duration <= 0.0:
        raise ValueError("burn_time must be positive")
    if temperature <= 0.0 * u.K:
        raise ValueError("temperature must be positive")
    if not 0.0 < exhaust_velocity_efficiency <= 1.0:
        raise ValueError("exhaust_velocity_efficiency must be in (0, 1]")
    if missed_periapsis_floor is not None and missed_periapsis_floor < 0.0 * u.km:
        raise ValueError("missed_periapsis_floor must be nonnegative")
    hot_energy_quantity = _REFERENCE_MIXED_SPECIFIC_ENERGY * (
        temperature / HYDROGEN_5500K.temperature
    )
    hot_energy = float(hot_energy_quantity.to_value(u.km**2 / u.s**2))
    exhaust_speed = exhaust_velocity_efficiency * float(np.sqrt(2.0 * hot_energy))
    variable_altitude = front_side or missed_periapsis_floor is not None
    seed_altitude = (
        5000.0 if variable_altitude else float(DEPARTURE_ALTITUDE.to_value(u.km))
    )
    mu, _, radius, _ = _orbit(seed_altitude * u.km)
    departure = float(THREE_SYNODIC_DEPARTURE_VINF.to_value(u.km / u.s))
    half_turn = float(np.arcsin(1.0 / (1.0 + radius * departure**2 / mu)))
    sense = -1 if front_side else 1
    orientation = -sense * (0.5 * np.pi + half_turn)
    parameters = [orientation, 0.5, 0.53 if front_side else 0.50]
    bounds = [(-np.pi, np.pi), (0.0, 1.0), (0.05, 0.85)]
    if variable_altitude:
        parameters.append(seed_altitude / 10000.0)
        bounds.append((0.06, 2.0))
    initial = np.asarray(parameters, dtype=np.float64)
    cache: dict[Tuple[float, ...], _UncantedEvaluation] = {}

    def evaluate(values: npt.NDArray[np.float64]) -> _UncantedEvaluation:
        key = tuple(float(value) for value in values)
        if key not in cache:
            cache.clear()
            cache[key] = _uncanted_evaluation(
                values,
                front_side,
                variable_altitude,
                duration,
                hot_energy,
                exhaust_speed,
                fine=False,
            )
        return cache[key]

    def objective(values: npt.NDArray[np.float64]) -> float:
        return float(values[2])

    def closure(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return evaluate(values).residual

    def visibility(values: npt.NDArray[np.float64]) -> float:
        return evaluate(values).clearance / 1000.0

    constraints = [
        {"type": "eq", "fun": closure},
        {"type": "ineq", "fun": visibility},
    ]
    if missed_periapsis_floor is not None:
        missed_floor = float(missed_periapsis_floor.to_value(u.km))

        def miss_safety(values: npt.NDArray[np.float64]) -> float:
            return (evaluate(values).missed_periapsis - missed_floor) / 1000.0

        constraints.append({"type": "ineq", "fun": miss_safety})

    optimum = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1.0e-10, "maxiter": 120},
    )
    if not optimum.success:
        raise RuntimeError(f"uncanted burn optimization failed: {optimum.message}")
    solved = np.asarray(optimum.x, dtype=np.float64)
    evaluation = _uncanted_evaluation(
        solved,
        front_side,
        variable_altitude,
        duration,
        hot_energy,
        exhaust_speed,
        fine=True,
    )
    if float(np.linalg.norm(evaluation.residual)) > 1.0e-6:
        raise RuntimeError("uncanted burn did not close on the 3S asymptote")
    if evaluation.clearance < -1.0e-3:
        raise RuntimeError("uncanted burn's projectile route intersects Earth")
    if (
        missed_periapsis_floor is not None
        and evaluation.missed_periapsis
        < float(missed_periapsis_floor.to_value(u.km)) - 1.0e-3
    ):
        raise RuntimeError("uncanted burn is below the missed-projectile floor")

    altitude = (
        float(solved[3]) * 10000.0
        if variable_altitude
        else float(DEPARTURE_ALTITUDE.to_value(u.km))
    )
    mu, earth_radius, burn_radius, parking_speed = _orbit(altitude * u.km)
    incoming = _rotation(
        float(THREE_SYNODIC_AIM_SEPARATION.to_value(u.rad))
    ) @ np.array([1.0, 0.0], dtype=np.float64)
    projectile_excess = float(THREE_SYNODIC_RETURN_VINF.to_value(u.km / u.s))
    closing_speeds = np.zeros(_PROFILE_SAMPLES)
    slug_ratios = np.zeros(_PROFILE_SAMPLES)
    off_head_on = np.zeros(_PROFILE_SAMPLES)
    axial_impulse = np.zeros(_PROFILE_SAMPLES)
    lateral_impulse = np.zeros(_PROFILE_SAMPLES)
    route_clearances = np.zeros(_PROFILE_SAMPLES)
    missed_periapsis_altitudes = np.zeros(_PROFILE_SAMPLES)
    for index, (position, velocity) in enumerate(
        zip(evaluation.states[:2].T, evaluation.states[2:4].T)
    ):
        route = _uncanted_route(
            _incoming_routes(mu, earth_radius, position, projectile_excess, incoming),
            front_side,
        )
        relative = route.velocity - velocity
        closing = float(np.linalg.norm(relative))
        slug_ratio = closing**2 / (2.0 * hot_energy) - 1.0
        axis = _unit(velocity)
        cross = float(axis[0] * relative[1] - axis[1] * relative[0])
        angle = float(np.arctan2(cross, float(axis @ relative)))
        impulse = (
            1.0 + 1.0 / slug_ratio
        ) * exhaust_speed * axis + relative / slug_ratio
        state_altitude = float(np.linalg.norm(position)) - earth_radius
        route_clearance = (
            state_altitude
            if route.radial_velocity <= 0.0
            else route.periapsis_radius - earth_radius
        )
        closing_speeds[index] = closing
        slug_ratios[index] = slug_ratio
        off_head_on[index] = np.pi - abs(angle)
        axial_impulse[index] = float(axis @ impulse)
        lateral_impulse[index] = float(axis[0] * impulse[1] - axis[1] * impulse[0])
        route_clearances[index] = min(state_altitude, route_clearance)
        missed_periapsis_altitudes[index] = route.periapsis_radius - earth_radius

    spent = float(solved[2])
    hydrogen_flow = spent / duration
    masses = evaluation.states[4]
    acceleration_scale = hydrogen_flow / masses
    integrated_axial = float(
        np.trapezoid(acceleration_scale * axial_impulse, evaluation.times)
    )
    integrated_lateral = float(
        np.trapezoid(acceleration_scale * lateral_impulse, evaluation.times)
    )
    integrated_thrust = float(
        np.trapezoid(
            acceleration_scale * np.hypot(axial_impulse, lateral_impulse),
            evaluation.times,
        )
    )
    impactor_mass = float(np.trapezoid(hydrogen_flow / slug_ratios, evaluation.times))
    impacting_impactor_mass = float(
        np.trapezoid(
            hydrogen_flow / slug_ratios * (missed_periapsis_altitudes < 0.0),
            evaluation.times,
        )
    )
    radii = np.linalg.norm(evaluation.states[:2].T, axis=1)
    outgoing_speed, outgoing_angle = _outgoing_asymptote(mu, evaluation.states[:, -1])
    target_excess = float(THREE_SYNODIC_DEPARTURE_VINF.to_value(u.km / u.s))
    final_impulsive_speed = float(np.sqrt(target_excess**2 + 2.0 * mu / burn_radius))
    impulsive_delta_v = final_impulsive_speed - parking_speed
    before = float(solved[1]) * duration
    return UncantedThermalBurn(
        front_side=front_side,
        reference_periapsis_altitude=altitude * u.km,
        periapsis_position_angle=(float(solved[0]) * u.rad).to(u.deg),
        seconds_before_reference_periapsis=before * u.s,
        seconds_after_reference_periapsis=(duration - before) * u.s,
        hydrogen_spent_fraction=spent,
        delivered_fraction=1.0 - spent,
        initial_to_delivered_mass_ratio=1.0 / (1.0 - spent),
        hydrogen_per_delivered=spent / (1.0 - spent),
        impactor_mass_fraction=impactor_mass,
        closest_altitude=(float(np.min(radii)) - earth_radius) * u.km,
        minimum_projectile_clearance=float(np.min(route_clearances)) * u.km,
        closing_speed_minimum=float(np.min(closing_speeds)) * u.km / u.s,
        closing_speed_maximum=float(np.max(closing_speeds)) * u.km / u.s,
        slug_ratio_minimum=float(np.min(slug_ratios)),
        slug_ratio_maximum=float(np.max(slug_ratios)),
        off_head_on_start=(off_head_on[0] * u.rad).to(u.deg),
        off_head_on_end=(off_head_on[-1] * u.rad).to(u.deg),
        off_head_on_maximum=(float(np.max(off_head_on)) * u.rad).to(u.deg),
        integrated_axial_delta_v=integrated_axial * u.km / u.s,
        integrated_lateral_delta_v=integrated_lateral * u.km / u.s,
        integrated_thrust_delta_v=integrated_thrust * u.km / u.s,
        impulsive_delta_v=impulsive_delta_v * u.km / u.s,
        finite_burn_penalty=(integrated_thrust - impulsive_delta_v) * u.km / u.s,
        missed_projectile_periapsis_altitude=float(np.min(missed_periapsis_altitudes))
        * u.km,
        missed_projectile_earth_impact_fraction=(
            impacting_impactor_mass / impactor_mass
        ),
        outgoing_vinf=outgoing_speed * u.km / u.s,
        outgoing_angle_error=(outgoing_angle * u.rad).to(u.deg),
    )


def _propagate_burn(
    mu: float,
    radius: float,
    parking_speed: float,
    seconds_before: float,
    duration: float,
    ideal_burn: float,
    exhaust_speed: float,
    steered: bool,
    dense: bool = False,
) -> Tuple[npt.NDArray[np.float64], OdeSolution | None]:
    """Propagate a constant-mass-flow burn from the parking orbit."""

    start = _coast_before_periapsis(mu, radius, parking_speed, seconds_before)
    spent = 1.0 - np.exp(-ideal_burn / exhaust_speed)
    fixed_axis = np.array([0.0, 1.0], dtype=np.float64)

    def rhs(time: float, state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        position = state[:2]
        velocity = state[2:]
        acceleration = (
            exhaust_speed * spent / duration / (1.0 - spent * time / duration)
        )
        axis = _unit(velocity) if steered else fixed_axis
        gravity = -mu * position / np.linalg.norm(position) ** 3
        return np.concatenate((velocity, gravity + acceleration * axis))

    solution = solve_ivp(
        rhs,
        (0.0, duration),
        start,
        rtol=_RTOL,
        atol=_ATOL,
        dense_output=dense,
        max_step=2.0 if dense else np.inf,
    )
    dense_solution = solution.sol if dense else None
    return np.asarray(solution.y[:, -1], dtype=np.float64), dense_solution


def finite_burn_turn(
    steered: bool,
    burn_time: u.Quantity = REFERENCE_BURN_TIME,
    departure_vinf: u.Quantity = THREE_SYNODIC_DEPARTURE_VINF,
    return_vinf: u.Quantity = THREE_SYNODIC_RETURN_VINF,
    aim_separation: u.Quantity = THREE_SYNODIC_AIM_SEPARATION,
    reference_exhaust_speed: u.Quantity = REFERENCE_EXHAUST_SPEED,
    slug_ratio: float = REFERENCE_SLUG_RATIO,
    collimation: float = REFERENCE_COLLIMATION,
    altitude: u.Quantity = DEPARTURE_ALTITUDE,
    period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD,
) -> FiniteBurnTurn:
    """Integrate and optimize a finite burn on the visible 3S mirror.

    ``steered=True`` follows the instantaneous velocity and is the energy-best
    sensitivity.  ``False`` holds thrust on the inertial periapsis tangent, the
    fixed-direction constraint used for a head-on chamber.
    """

    if slug_ratio <= 0.0:
        raise ValueError("slug_ratio must be positive")
    if collimation <= 0.0:
        raise ValueError("collimation must be positive")
    duration = float(burn_time.to_value(u.s))
    if duration <= 0.0:
        raise ValueError("burn_time must be positive")
    mu, earth_radius, radius, parking_speed = _orbit(altitude, period)
    departure = float(departure_vinf.to_value(u.km / u.s))
    incoming_speed = float(return_vinf.to_value(u.km / u.s))
    separation = float(aim_separation.to_value(u.rad))
    reference_exhaust = float(reference_exhaust_speed.to_value(u.km / u.s))
    escape = float(np.sqrt(2.0 * mu / radius))
    final_impulsive_speed = float(np.hypot(departure, escape))
    impulsive_burn = final_impulsive_speed - parking_speed
    target_energy = 0.5 * final_impulsive_speed**2 - mu / radius

    def final_energy(seconds_before: float, ideal_burn: float) -> float:
        final, _ = _propagate_burn(
            mu,
            radius,
            parking_speed,
            seconds_before,
            duration,
            ideal_burn,
            reference_exhaust,
            steered,
        )
        return float(0.5 * final[2:] @ final[2:] - mu / np.linalg.norm(final[:2]))

    def required_burn(seconds_before: float) -> float:
        return float(
            brentq(
                lambda ideal: final_energy(seconds_before, ideal) - target_energy,
                impulsive_burn,
                2.0 * impulsive_burn + 1.0,
                xtol=1.0e-9,
            )
        )

    optimum = minimize_scalar(
        required_burn,
        bounds=(0.0, duration),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    seconds_before = float(optimum.x)
    ideal_burn = float(optimum.fun)
    centered_burn = required_burn(0.5 * duration)
    _, dense_solution = _propagate_burn(
        mu,
        radius,
        parking_speed,
        seconds_before,
        duration,
        ideal_burn,
        reference_exhaust,
        steered,
        dense=True,
    )
    if dense_solution is None:
        raise RuntimeError("burn propagation did not return a dense solution")
    times = np.linspace(0.0, duration, _PROFILE_SAMPLES)
    states = np.asarray(dense_solution(times), dtype=np.float64)
    positions = states[:2].T
    velocities = states[2:].T
    fixed_axis = np.array([0.0, 1.0], dtype=np.float64)
    axes = (
        velocities / np.linalg.norm(velocities, axis=1)[:, np.newaxis]
        if steered
        else np.tile(fixed_axis, (_PROFILE_SAMPLES, 1))
    )
    incoming, anchor = _physical_stream_geometry(
        mu,
        earth_radius,
        radius,
        departure,
        incoming_speed,
        separation,
    )
    impact_angles = np.zeros(_PROFILE_SAMPLES)
    exhaust_cants = np.zeros(_PROFILE_SAMPLES)
    effective_exhaust = np.zeros(_PROFILE_SAMPLES)
    head_on_exhaust = np.zeros(_PROFILE_SAMPLES)
    for index, (position, velocity, axis) in enumerate(
        zip(positions, velocities, axes)
    ):
        routes = _incoming_routes(mu, earth_radius, position, incoming_speed, incoming)
        route = _select_visible_route(routes, anchor)
        relative = route.velocity - velocity
        closing = float(np.linalg.norm(relative))
        impact_angle = float(
            np.arccos(np.clip(np.dot(relative, axis) / closing, -1.0, 1.0))
        )
        beta = float(
            np.sqrt(max(1.0 + slug_ratio - np.sin(impact_angle) ** 2, 0.0))
            + np.cos(impact_angle)
        )
        impact_angles[index] = impact_angle
        exhaust_cants[index] = _cant(impact_angle, slug_ratio)
        effective_exhaust[index] = collimation * beta * closing / slug_ratio
        projectile_speed = float(np.linalg.norm(route.velocity))
        head_on_closing = projectile_speed + float(np.dot(velocity, axis))
        head_on_exhaust[index] = (
            collimation
            * (np.sqrt(1.0 + slug_ratio) - 1.0)
            * head_on_closing
            / slug_ratio
        )

    spent = 1.0 - np.exp(-ideal_burn / reference_exhaust)
    acceleration = (
        reference_exhaust * spent / duration / (1.0 - spent * times / duration)
    )
    log_mass_ratio = float(np.trapezoid(acceleration / effective_exhaust, times))
    head_on_log_mass_ratio = float(np.trapezoid(acceleration / head_on_exhaust, times))
    radii = np.linalg.norm(positions, axis=1)
    closest_index = int(np.argmin(radii))
    off_head_on = np.pi - impact_angles
    impulse = float(np.trapezoid(acceleration, times))
    return FiniteBurnTurn(
        steered=steered,
        burn_time=duration * u.s,
        seconds_before_reference_periapsis=seconds_before * u.s,
        seconds_after_reference_periapsis=(duration - seconds_before) * u.s,
        seconds_before_closest_approach=times[closest_index] * u.s,
        seconds_after_closest_approach=(duration - times[closest_index]) * u.s,
        closest_altitude=(radii[closest_index] - earth_radius) * u.km,
        impulsive_burn=impulsive_burn * u.km / u.s,
        finite_burn=ideal_burn * u.km / u.s,
        finite_burn_loss=(ideal_burn - impulsive_burn) * u.km / u.s,
        timing_gain=(centered_burn - ideal_burn) * u.km / u.s,
        off_head_on_start=(off_head_on[0] * u.rad).to(u.deg),
        off_head_on_end=(off_head_on[-1] * u.rad).to(u.deg),
        off_head_on_maximum=(float(np.max(off_head_on)) * u.rad).to(u.deg),
        time_above_22_degrees=float(
            np.mean(off_head_on > (22.0 * u.deg).to_value(u.rad))
        ),
        exhaust_cant_minimum=(float(np.min(exhaust_cants)) * u.rad).to(u.deg),
        exhaust_cant_maximum=(float(np.max(exhaust_cants)) * u.rad).to(u.deg),
        exhaust_cant_mean=(
            float(np.trapezoid(exhaust_cants, times) / duration) * u.rad
        ).to(u.deg),
        exhaust_cant_impulse_mean=(
            float(np.trapezoid(exhaust_cants * acceleration, times) / impulse) * u.rad
        ).to(u.deg),
        angle_gain_over_head_on=float(np.exp(head_on_log_mass_ratio - log_mass_ratio)),
    )


def _print_report() -> None:
    """Print the reference mirror and finite-burn results."""

    visible, blocked = three_synodic_periapsis_mirrors()
    print("3S circular return at the 600 km departure burn")
    print(
        f"Earth angular radius from the vehicle: "
        f"{occultation_half_angle().to_value(u.deg):.2f} deg"
    )
    for label, mirror in (("visible", visible), ("other", blocked)):
        print(
            f"{label:>7} mirror: blocked={mirror.blocked}, "
            f"projectile periapsis altitude "
            f"{mirror.projectile_periapsis_altitude.to_value(u.km):.1f} km, "
            f"impact {mirror.impact_angle.to_value(u.deg):.2f} deg, "
            f"plume cant {mirror.exhaust_cant.to_value(u.deg):.2f} deg"
        )
    print(
        "blocked mirror needs incoming-asymptote rotation: "
        f"{blocked_mirror_clearance_angle().to_value(u.deg):.2f} deg"
    )
    for steered in (False, True):
        result = finite_burn_turn(steered)
        label = "velocity-steered" if steered else "fixed direction"
        print(f"\n20 min, {label}")
        print(
            "  reference-periapsis split: "
            f"{result.seconds_before_reference_periapsis.to_value(u.s):.1f} / "
            f"{result.seconds_after_reference_periapsis.to_value(u.s):.1f} s"
        )
        print(
            "  finite-burn loss: "
            f"{result.finite_burn_loss.to_value(u.m / u.s):.1f} m/s; "
            f"timing gain {result.timing_gain.to_value(u.m / u.s):.2f} m/s"
        )
        print(
            "  off head-on: "
            f"{result.off_head_on_start.to_value(u.deg):.2f} -> "
            f"{result.off_head_on_end.to_value(u.deg):.2f} deg, "
            f"max {result.off_head_on_maximum.to_value(u.deg):.2f}, "
            f">22 deg for {100.0 * result.time_above_22_degrees:.1f}%"
        )
        print(
            "  plume cant: "
            f"{result.exhaust_cant_minimum.to_value(u.deg):.2f} to "
            f"{result.exhaust_cant_maximum.to_value(u.deg):.2f} deg, "
            f"impulse mean "
            f"{result.exhaust_cant_impulse_mean.to_value(u.deg):.2f} deg"
        )
        print(
            "  delivered-mass gain over exactly head-on: "
            f"{100.0 * (result.angle_gain_over_head_on - 1.0):.3f}%"
        )

    print("\n3000 K uncanted axial nozzle, exact 3S vector closure")
    print(
        "  hot-mixture energy / ideal / 85%-velocity exhaust: "
        f"{UNCANTED_MIXED_SPECIFIC_ENERGY.to_value(u.MJ / u.kg):.2f} MJ/kg / "
        f"{UNCANTED_IDEAL_EXHAUST_SPEED.to_value(u.km / u.s):.2f} / "
        f"{UNCANTED_EXHAUST_SPEED.to_value(u.km / u.s):.2f} km/s"
    )

    def print_thermal(label: str, thermal: UncantedThermalBurn) -> None:
        """Print one vector-closed uncanted solution."""

        print(f"\n  {label}")
        print(
            "    parking periapsis / burn split: "
            f"{thermal.reference_periapsis_altitude.to_value(u.km):.1f} km / "
            f"{thermal.seconds_before_reference_periapsis.to_value(u.s):.1f} + "
            f"{thermal.seconds_after_reference_periapsis.to_value(u.s):.1f} s"
        )
        print(
            "    closing speed / hydrogen ratio k: "
            f"{thermal.closing_speed_minimum.to_value(u.km / u.s):.2f}-"
            f"{thermal.closing_speed_maximum.to_value(u.km / u.s):.2f} km/s / "
            f"{thermal.slug_ratio_minimum:.2f}-{thermal.slug_ratio_maximum:.2f}"
        )
        print(
            "    off head-on start -> end / maximum: "
            f"{thermal.off_head_on_start.to_value(u.deg):.2f} -> "
            f"{thermal.off_head_on_end.to_value(u.deg):.2f} / "
            f"{thermal.off_head_on_maximum.to_value(u.deg):.2f} deg"
        )
        print(
            "    H2 propellant / delivered / m0/mf / external impactor: "
            f"{100.0 * thermal.hydrogen_spent_fraction:.2f}% / "
            f"{100.0 * thermal.delivered_fraction:.2f}% / "
            f"{thermal.initial_to_delivered_mass_ratio:.3f} / "
            f"{100.0 * thermal.impactor_mass_fraction:.2f}% of initial mass"
        )
        print(
            "    required dv total / impulse / finite-burn penalty: "
            f"{thermal.integrated_thrust_delta_v.to_value(u.km / u.s):.3f} / "
            f"{thermal.impulsive_delta_v.to_value(u.km / u.s):.3f} km/s / "
            f"{thermal.finite_burn_penalty.to_value(u.m / u.s):.1f} m/s"
        )
        print(
            "    axial / signed lateral dv: "
            f"{thermal.integrated_axial_delta_v.to_value(u.km / u.s):.3f} / "
            f"{thermal.integrated_lateral_delta_v.to_value(u.km / u.s):.3f} km/s"
        )
        print(
            "    intercept clearance / missed-shot periapsis / Earth-impact share: "
            f"{thermal.minimum_projectile_clearance.to_value(u.km):.1f} / "
            f"{thermal.missed_projectile_periapsis_altitude.to_value(u.km):.1f} km / "
            f"{100.0 * thermal.missed_projectile_earth_impact_fraction:.1f}%"
        )

    print("\n  mass-optimal, visibility required only to interception")
    for front_side in (False, True):
        thermal = uncanted_thermal_burn(front_side)
        label = "front-side" if front_side else "visible-side"
        print_thermal(label, thermal)

    print("\n  fail-safe: every missed projectile keeps a 600 km periapsis")
    for front_side in (False, True):
        thermal = uncanted_thermal_burn(front_side, missed_periapsis_floor=600.0 * u.km)
        label = "front-side safe" if front_side else "visible-side safe"
        print_thermal(label, thermal)


if __name__ == "__main__":
    _print_report()
