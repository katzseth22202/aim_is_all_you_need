"""Ideal circular 2S/3S resonance search scored on departure-burn delivered mass.

The real-ephemeris resonance audit in :mod:`src.real_orbit_resonance` fixes the
Earth--Jupiter phase inherited from one circular reference solution.  This
module asks the complementary design question: in the circular, coplanar model,
how much can that phase be changed to make a returning PuffSat less opposed to
the next Jupiter departure -- and is the answer worth anything?

For each integer synodic multiple, the search joins a prograde Earth-to-Jupiter
zero-revolution Lambert arc to a retrograde Jupiter-to-Earth zero-revolution
Lambert arc.  It roots the Jupiter-relative incoming/outgoing excess-speed
mismatch to zero, so every retained candidate is a strictly unpowered Jovian
flyby, and rejects candidates whose required perijove falls below the
repository's 4,000 km altitude floor.

Every closure is then scored in the frame that decides the collision: the
**vehicle** frame at the 200 km departure burn.  Both quantities the head-on
nozzle depends on are taken relative to the moving vehicle -- the closing speed
``w`` and the momentum-bias angle -- and the departure hyperbola's periapsis
velocity is rotated off its outgoing excess vector by half the Earth turn,
whose mirror sign is free.  The patched-conic aim angle between excess vectors
survives only as a diagnostic; it is neither of the two things the impulse law
uses.  See ``docs/adr/0012-head-on-impact-penalty-is-bounded.md``.

Impulse per arriving impactor kg, for slug ratio ``k`` and impact angle
``theta`` measured from the thrust axis, with the exhaust canted to leave the
vehicle no transverse impulse:

    beta(theta, k) = sqrt(1 + k - sin^2 theta) + cos theta

which is ADR 0009's head-on nozzle ``sqrt(1+k) - 1`` at 180 degrees and the
growth push's ``1 + sqrt(1+k)`` at 0 degrees -- one law at two angles.  The
departure nozzle's effective exhaust speed is ``eta * beta * w / k``.

Run the committed search with ``make resonance-impulse``.
"""

import argparse
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.bodies import Earth, Jupiter
from boinor.core.iod import izzo
from scipy.optimize import brentq, minimize
from tabulate import tabulate

from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.retrograde_return_legs import (
    _body_state,
    _FlybyParams,
    _powered_flyby_params,
)

_SECONDS_PER_YEAR = 365.25 * 86400.0
_ENCOUNTER_MIN_YEARS = 0.25
_RETURN_MIN_YEARS = 0.10
_DEFAULT_PHASE_SAMPLES = 361
_DEFAULT_ENCOUNTER_SAMPLES = 301
_DEFAULT_MAX_DEPARTURE_VINF = 25.0  # km/s; production flyby-search cap
_DEFAULT_SLUG_RATIO = 3.0  # Appendix-D k; ADR 0009's two-currency k* is 6-12
_DEFAULT_COLLIMATION = 0.8  # eta == STD_FUDGE_FACTOR, the handoff's D1
_BURN_STEPS = 401
_CEILING_SAMPLES = 1801
_LAMBERT_ITERATIONS = 80
_LAMBERT_RTOL = 1.0e-10
_ROOT_XTOL_SECONDS = 1.0e-5
_REFINEMENT_STARTS = 8


@dataclass(frozen=True)
class DepartureImpulseLedger:
    """Vehicle-frame score of one closure's head-on-nozzle departure burn.

    Attributes:
        slug_ratio: Kilograms of carried slug per kilogram of arriving impactor.
        collimation: Uniform plume-collimation recovery factor applied to the
            ideal impulse.
        mirror_rotation: Signed rotation from the departure excess vector to the
            periapsis velocity, half the Earth departure hyperbola's turn.  Both
            signs are physical; the reported one is whichever delivers more mass.
        aim_at_burn: Angle from the thrust axis to the impactor's inertial
            velocity at the 200 km burn point.
        relative_aim_start: Angle from the thrust axis to the impactor's velocity
            *relative to the vehicle* when the burn lights.
        relative_aim_end: The same angle when the burn ends.
        closing_speed_start: Impactor speed relative to the vehicle at burn start.
        closing_speed_end: The same speed at burn end.
        impulse_per_impactor_start: ``beta`` at burn start, in impactor-kg units.
        exhaust_cant: Cant of the final exhaust from straight backward needed to
            carry the transverse momentum away at burn start.
        effective_exhaust_speed: Burn-averaged exhaust speed implied by the
            integrated mass ledger, ``dv / ln(mass ratio)``.
        delivered_fraction: Payload fraction surviving the departure burn.
        head_on_delivered_fraction: The same fraction for an exactly head-on
            impact at the same speeds -- what ADR 0009 assumes.
    """

    slug_ratio: float
    collimation: float
    mirror_rotation: u.Quantity
    aim_at_burn: u.Quantity
    relative_aim_start: u.Quantity
    relative_aim_end: u.Quantity
    closing_speed_start: u.Quantity
    closing_speed_end: u.Quantity
    impulse_per_impactor_start: float
    exhaust_cant: u.Quantity
    effective_exhaust_speed: u.Quantity
    delivered_fraction: float
    head_on_delivered_fraction: float

    @property
    def gain_over_head_on(self) -> float:
        """Delivered mass relative to assuming an exactly head-on impact."""

        return self.delivered_fraction / self.head_on_delivered_fraction


@dataclass(frozen=True)
class CircularResonanceCandidate:
    """One exact, unpowered circular resonance closure.

    Attributes:
        synodic_multiple: Number of Earth--Jupiter synodic periods in the cycle.
        period: Exact departure-to-departure period.
        earth_minus_jupiter_phase: Earth longitude minus Jupiter longitude at
            departure.
        outbound_time: Earth-to-Jupiter time of flight.
        earth_departure_vinf: Earth-relative departure excess speed.
        earth_departure_burn: Departure speed increment above the closed-cycle
            200 km periapsis speed.
        earth_return_vinf: Earth-relative return excess speed.
        collision_speed_200km: Return speed folded through Earth's gravity well
            to 200 km altitude.
        aim_separation: Diagnostic only -- the included angle between the
            incoming excess vector and the next cycle's desired departure-excess
            vector, at the patched-conic boundary.  180 degrees is exactly
            backward.  The impulse law uses ``ledger`` instead, because neither
            the closing speed nor the bias angle is measured at the SOI.
        cant_from_backward: ``180 degrees - aim_separation``.
        backward_projection_fraction: Fraction of the incoming speed projected
            backward along the desired thrust axis, clamped at zero when the
            impact instead has a forward component.
        transverse_projection_fraction: Magnitude of the incoming transverse
            projection relative to its speed.
        ledger: Vehicle-frame departure-burn score.
        jupiter_vinf: Conserved Jupiter-relative excess speed.
        jupiter_turn: Required unpowered Jovian turn angle.
        perijove_altitude: Required altitude above Jupiter's reference radius.
        jupiter_vinf_mismatch: Incoming minus outgoing Jupiter-relative excess
            speed; approximately zero for every returned candidate.
    """

    synodic_multiple: int
    period: u.Quantity
    earth_minus_jupiter_phase: u.Quantity
    outbound_time: u.Quantity
    earth_departure_vinf: u.Quantity
    earth_departure_burn: u.Quantity
    earth_return_vinf: u.Quantity
    collision_speed_200km: u.Quantity
    aim_separation: u.Quantity
    cant_from_backward: u.Quantity
    backward_projection_fraction: float
    transverse_projection_fraction: float
    ledger: DepartureImpulseLedger
    jupiter_vinf: u.Quantity
    jupiter_turn: u.Quantity
    perijove_altitude: u.Quantity
    jupiter_vinf_mismatch: u.Quantity


@dataclass(frozen=True)
class CircularResonanceFamily:
    """Three views of one integer-synodic circular resonance family.

    Attributes:
        synodic_multiple: Number of synodic periods in every repeated cycle.
        feasible_grid_closures: Feasible closures found by the reproducible
            coarse grid before continuous refinement.
        minimum_departure: Closure minimizing Earth departure excess speed.
        least_backward: Closure minimizing the diagnostic incoming/departure
            included angle, and therefore its backward projection.
        maximum_delivered_mass: Closure maximizing the payload fraction
            surviving the departure burn.
        free_aim_ceiling: Delivered-mass gain the minimum-departure closure
            would show if its impact angle were free of any delta-v charge --
            the upper bound on every aim-steering scheme at this slug ratio.
        free_aim_ceiling_angle: The angle attaining that ceiling.
        head_on_crossover_slug_ratio: Slug ratio above which an exactly head-on
            impact is itself the optimum, so canting can only lose.
    """

    synodic_multiple: int
    feasible_grid_closures: int
    minimum_departure: CircularResonanceCandidate
    least_backward: CircularResonanceCandidate
    maximum_delivered_mass: CircularResonanceCandidate
    free_aim_ceiling: float
    free_aim_ceiling_angle: u.Quantity
    head_on_crossover_slug_ratio: float


@dataclass(frozen=True)
class CircularResonanceAnalysis:
    """Complete idealized 2S/3S directional-impact comparison.

    Attributes:
        two_synodic: Repeating two-synodic family.
        three_synodic: Repeating three-synodic family.
        synodic_period: Mean circular Earth--Jupiter synodic period.
        slug_ratio: Slug ratio the ledgers are scored at.
        collimation: Plume-collimation recovery factor.
        maximum_departure_vinf: Hard search cap on departure excess speed.
        phase_samples: Committed departure-phase grid size.
        encounter_samples: Committed Jupiter-encounter-time grid size.
    """

    two_synodic: CircularResonanceFamily
    three_synodic: CircularResonanceFamily
    synodic_period: u.Quantity
    slug_ratio: float
    collimation: float
    maximum_departure_vinf: u.Quantity
    phase_samples: int
    encounter_samples: int


@dataclass(frozen=True)
class _Ledger:
    """Internal float-valued departure-burn ledger (km/s, rad, fractions)."""

    mirror_rotation: float
    aim_at_burn: float
    relative_aim_start: float
    relative_aim_end: float
    closing_speed_start: float
    closing_speed_end: float
    beta_start: float
    effective_exhaust_speed: float
    delivered_fraction: float
    head_on_delivered_fraction: float


@dataclass(frozen=True)
class _Geometry:
    """Internal float-valued Lambert-pair geometry (km, s, km/s, rad)."""

    synodic_multiple: int
    period: float
    phase: float
    encounter_time: float
    departure_vinf: float
    departure_burn: float
    return_vinf: float
    collision_speed_200km: float
    aim_separation: float
    ledger: _Ledger
    jupiter_vinf_in: float
    jupiter_vinf_out: float
    jupiter_turn: float
    perijove_radius: float

    @property
    def jupiter_vinf_mismatch(self) -> float:
        """Incoming minus outgoing Jupiter-relative excess speed (km/s)."""

        return self.jupiter_vinf_in - self.jupiter_vinf_out


def impulse_per_impactor_kg(
    impact_angle: u.Quantity, slug_ratio: float = _DEFAULT_SLUG_RATIO
) -> float:
    """Ideal collimated impulse per arriving impactor kilogram.

    The exhaust is canted just enough to carry the incoming transverse momentum,
    so the vehicle takes no sideways kick and the axial impulse is what is left:
    ``sqrt(1 + k - sin^2 theta) + cos theta``, in units of the closing speed.
    At 180 degrees this is ADR 0009's head-on nozzle ``sqrt(1+k) - 1``; at zero
    it is the growth push's ``1 + sqrt(1+k)``.

    Args:
        impact_angle: Angle from the thrust axis to the impactor's velocity
            relative to the vehicle.  180 degrees is exactly head-on.
        slug_ratio: Kilograms of carried slug per kilogram of arriving impactor.

    Returns:
        Impulse per impactor kilogram, in units of the closing speed.

    Raises:
        ValueError: If the slug ratio is not positive.
    """

    if slug_ratio <= 0.0:
        raise ValueError("slug_ratio must be positive")
    theta = float(impact_angle.to_value(u.rad))
    return float(
        np.sqrt(max(0.0, 1.0 + slug_ratio - np.sin(theta) ** 2)) + np.cos(theta)
    )


def _mu_earth() -> float:
    """Earth gravitational parameter (km^3/s^2)."""

    return float(Earth.k.to_value(u.km**3 / u.s**2))


def _burn_radius(params: _FlybyParams) -> float:
    """Radius of the 200 km departure burn implied by the escape speed (km)."""

    return 2.0 * _mu_earth() / params.v_esc_leo**2


def _departure_half_turn(departure_vinf: float, params: _FlybyParams) -> float:
    """Half the Earth departure hyperbola's turn, ``arcsin(1/e)`` (rad).

    The periapsis velocity bisects the hyperbola's bend, so the thrust axis at
    the burn is rotated off the outgoing excess vector by this angle.  Both
    mirror images of the hyperbola give the same outgoing excess direction, so
    the sign is a free design choice.
    """

    eccentricity = 1.0 + _burn_radius(params) * departure_vinf**2 / _mu_earth()
    return float(np.arcsin(1.0 / eccentricity))


def _integrate_burn(
    aim_at_burn: float,
    return_vinf: float,
    departure_vinf: float,
    slug_ratio: float,
    collimation: float,
    params: _FlybyParams,
) -> Tuple[float, float, float, float, float, float, float]:
    """Integrate the departure burn's variable-exhaust-speed mass ledger."""

    impactor_speed = float(np.hypot(return_vinf, params.v_esc_leo))
    final_speed = float(np.hypot(departure_vinf, params.v_esc_leo))
    if final_speed <= params.v_depart_from:
        return (aim_at_burn, aim_at_burn, impactor_speed, impactor_speed, 0.0, 0.0, 1.0)
    speeds = np.linspace(params.v_depart_from, final_speed, _BURN_STEPS)
    transverse = impactor_speed * float(np.sin(aim_at_burn))
    axial = impactor_speed * float(np.cos(aim_at_burn)) - speeds
    closing = np.hypot(axial, transverse)
    relative = np.arctan2(np.full_like(speeds, transverse), axial)
    beta = np.sqrt(np.maximum(1.0 + slug_ratio - np.sin(relative) ** 2, 0.0)) + np.cos(
        relative
    )
    exhaust = collimation * beta * closing / slug_ratio
    integrand = 1.0 / exhaust
    log_ratio = float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(speeds)))
    return (
        float(relative[0]),
        float(relative[-1]),
        float(closing[0]),
        float(closing[-1]),
        float(beta[0]),
        (final_speed - params.v_depart_from) / log_ratio,
        float(np.exp(-log_ratio)),
    )


def _burn_ledger(
    aim_separation: float,
    return_vinf: float,
    departure_vinf: float,
    slug_ratio: float,
    collimation: float,
    params: _FlybyParams,
) -> _Ledger:
    """Score one closure's departure burn, taking the better hyperbola mirror."""

    half_turn = _departure_half_turn(departure_vinf, params)
    best: Optional[Tuple[float, Tuple[float, ...]]] = None
    for rotation in (-half_turn, half_turn):
        integrated = _integrate_burn(
            aim_separation + rotation,
            return_vinf,
            departure_vinf,
            slug_ratio,
            collimation,
            params,
        )
        if best is None or integrated[6] > best[1][6]:
            best = (rotation, integrated)
    assert best is not None
    rotation, (
        relative_start,
        relative_end,
        closing_start,
        closing_end,
        beta_start,
        exhaust,
        delivered,
    ) = best
    head_on = _integrate_burn(
        np.pi, return_vinf, departure_vinf, slug_ratio, collimation, params
    )[6]
    return _Ledger(
        mirror_rotation=rotation,
        aim_at_burn=aim_separation + rotation,
        relative_aim_start=relative_start,
        relative_aim_end=relative_end,
        closing_speed_start=closing_start,
        closing_speed_end=closing_end,
        beta_start=beta_start,
        effective_exhaust_speed=exhaust,
        delivered_fraction=delivered,
        head_on_delivered_fraction=head_on,
    )


def free_aim_ceiling(
    return_vinf: u.Quantity,
    departure_vinf: u.Quantity,
    slug_ratio: float = _DEFAULT_SLUG_RATIO,
    collimation: float = _DEFAULT_COLLIMATION,
) -> Tuple[float, u.Quantity]:
    """Best delivered-mass gain if the impact angle carried no delta-v charge.

    Scans the impact angle at fixed speeds, so the result bounds *every* scheme
    that steers the impact direction: no trajectory, resonance or phasing can
    beat an angle handed over for free.  The objective is maximized at an
    endpoint -- a perfectly overtaking or an exactly head-on impact, never
    between -- because the closing speed and the momentum bias pull opposite
    ways across the interval.

    Args:
        return_vinf: Earth-relative excess speed of the arriving impactors.
        departure_vinf: Earth-relative excess speed the departure must reach.
        slug_ratio: Kilograms of carried slug per kilogram of arriving impactor.
        collimation: Plume-collimation recovery factor.

    Returns:
        The delivered-mass ratio against an exactly head-on impact, and the
        angle from the thrust axis attaining it.

    Raises:
        ValueError: If the slug ratio or collimation factor is not positive.
    """

    if slug_ratio <= 0.0:
        raise ValueError("slug_ratio must be positive")
    if collimation <= 0.0:
        raise ValueError("collimation must be positive")
    params = _powered_flyby_params(
        cycle_periapsis_speed=puffsat_cycle_periapsis_speed()
    )
    ret = float(return_vinf.to_value(u.km / u.s))
    dep = float(departure_vinf.to_value(u.km / u.s))
    angles = np.linspace(0.0, np.pi, _CEILING_SAMPLES)
    delivered = np.array(
        [
            _integrate_burn(float(angle), ret, dep, slug_ratio, collimation, params)[6]
            for angle in angles
        ]
    )
    best = int(np.argmax(delivered))
    head_on = _integrate_burn(np.pi, ret, dep, slug_ratio, collimation, params)[6]
    return float(delivered[best] / head_on), np.degrees(float(angles[best])) * u.deg


def head_on_crossover_slug_ratio(
    return_vinf: u.Quantity,
    departure_vinf: u.Quantity,
    collimation: float = _DEFAULT_COLLIMATION,
) -> float:
    """Slug ratio above which the exactly head-on impact is itself the optimum.

    The departure nozzle's exhaust speed goes as ``beta * w``.  Canting raises
    ``beta`` but lowers the closing speed ``w``, because a head-on impact adds
    the vehicle's own speed to the impactor's.  The bias term ``cos theta`` is
    independent of the slug ratio while ``sqrt(1+k)`` grows with it, so past
    some ``k`` the closing-speed loss wins and head-on stops being a penalty.

    Args:
        return_vinf: Earth-relative excess speed of the arriving impactors.
        departure_vinf: Earth-relative excess speed the departure must reach.
        collimation: Plume-collimation recovery factor.

    Returns:
        The slug ratio at which a perfectly overtaking and an exactly head-on
        impact deliver the same mass.

    Raises:
        ValueError: If the collimation factor is not positive.
        RuntimeError: If no crossover exists in the bracketed range.
    """

    if collimation <= 0.0:
        raise ValueError("collimation must be positive")
    params = _powered_flyby_params(
        cycle_periapsis_speed=puffsat_cycle_periapsis_speed()
    )
    ret = float(return_vinf.to_value(u.km / u.s))
    dep = float(departure_vinf.to_value(u.km / u.s))

    def endpoint_gap(slug_ratio: float) -> float:
        overtaking = _integrate_burn(0.0, ret, dep, slug_ratio, collimation, params)[6]
        head_on = _integrate_burn(np.pi, ret, dep, slug_ratio, collimation, params)[6]
        return overtaking - head_on

    try:
        return float(brentq(endpoint_gap, 1.0, 200.0, xtol=1.0e-8))
    except ValueError as error:
        raise RuntimeError(
            "no head-on crossover slug ratio in 1 <= k <= 200"
        ) from error


def _rotation(angle: float) -> npt.NDArray[np.float64]:
    """Return a three-dimensional rotation about the ecliptic pole."""

    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _synodic_period(params: _FlybyParams) -> float:
    """Circular Earth--Jupiter synodic period (s)."""

    n_earth = params.v_earth_orbit / params.r_earth_orbit
    n_jupiter = params.v_jupiter_orbit / params.r_jupiter_orbit
    return 2.0 * np.pi / (n_earth - n_jupiter)


def _evaluate_geometry(
    synodic_multiple: int,
    phase: float,
    encounter_time: float,
    params: _FlybyParams,
    slug_ratio: float = _DEFAULT_SLUG_RATIO,
    collimation: float = _DEFAULT_COLLIMATION,
) -> Optional[_Geometry]:
    """Evaluate one circular Lambert/Lambert pair before closure filtering."""

    synodic = _synodic_period(params)
    period = synodic_multiple * synodic
    if not (
        _ENCOUNTER_MIN_YEARS * _SECONDS_PER_YEAR
        <= encounter_time
        <= period - _RETURN_MIN_YEARS * _SECONDS_PER_YEAR
    ):
        return None

    n_earth = params.v_earth_orbit / params.r_earth_orbit
    n_jupiter = params.v_jupiter_orbit / params.r_jupiter_orbit
    earth_departure_position, earth_departure_velocity = _body_state(
        params.r_earth_orbit, 0.0, params.v_earth_orbit
    )
    jupiter_position, jupiter_velocity = _body_state(
        params.r_jupiter_orbit,
        phase + n_jupiter * encounter_time,
        params.v_jupiter_orbit,
    )
    earth_return_position, earth_return_velocity = _body_state(
        params.r_earth_orbit, n_earth * period, params.v_earth_orbit
    )
    try:
        departure_velocity, jupiter_arrival_velocity = izzo(
            params.mu_sun,
            earth_departure_position,
            jupiter_position,
            encounter_time,
            0,
            True,
            True,
            _LAMBERT_ITERATIONS,
            _LAMBERT_RTOL,
        )
        jupiter_departure_velocity, earth_arrival_velocity = izzo(
            params.mu_sun,
            jupiter_position,
            earth_return_position,
            period - encounter_time,
            0,
            False,
            True,
            _LAMBERT_ITERATIONS,
            _LAMBERT_RTOL,
        )
    except (AssertionError, RuntimeError, ValueError):
        return None

    # The return must be heliocentrically retrograde, not merely an Earth hit.
    if (
        float(
            np.dot(
                np.cross(jupiter_position, jupiter_departure_velocity),
                np.cross(jupiter_position, jupiter_velocity),
            )
        )
        >= 0.0
    ):
        return None

    incoming_jupiter = np.asarray(
        jupiter_arrival_velocity - jupiter_velocity, dtype=np.float64
    )
    outgoing_jupiter = np.asarray(
        jupiter_departure_velocity - jupiter_velocity, dtype=np.float64
    )
    jupiter_vinf_in = float(np.linalg.norm(incoming_jupiter))
    jupiter_vinf_out = float(np.linalg.norm(outgoing_jupiter))
    if jupiter_vinf_in <= 1.0e-9 or jupiter_vinf_out <= 1.0e-9:
        return None
    jupiter_turn = float(
        np.arccos(
            np.clip(
                np.dot(incoming_jupiter, outgoing_jupiter)
                / (jupiter_vinf_in * jupiter_vinf_out),
                -1.0,
                1.0,
            )
        )
    )
    if jupiter_turn <= 1.0e-12:
        perijove = np.inf
    else:
        mean_jupiter_vinf = 0.5 * (jupiter_vinf_in + jupiter_vinf_out)
        perijove = (
            params.mu_jupiter
            * (1.0 / float(np.sin(0.5 * jupiter_turn)) - 1.0)
            / mean_jupiter_vinf**2
        )

    departure_excess = np.asarray(
        departure_velocity - earth_departure_velocity, dtype=np.float64
    )
    return_excess = np.asarray(
        earth_arrival_velocity - earth_return_velocity, dtype=np.float64
    )
    departure_vinf = float(np.linalg.norm(departure_excess))
    return_vinf = float(np.linalg.norm(return_excess))
    if departure_vinf <= 1.0e-9 or return_vinf <= 1.0e-9:
        return None
    next_departure_excess = _rotation(n_earth * period) @ departure_excess
    aim_separation = float(
        np.arccos(
            np.clip(
                np.dot(return_excess, next_departure_excess)
                / (return_vinf * departure_vinf),
                -1.0,
                1.0,
            )
        )
    )
    return _Geometry(
        synodic_multiple=synodic_multiple,
        period=period,
        phase=phase,
        encounter_time=encounter_time,
        departure_vinf=departure_vinf,
        departure_burn=float(
            np.hypot(departure_vinf, params.v_esc_leo) - params.v_depart_from
        ),
        return_vinf=return_vinf,
        collision_speed_200km=float(np.hypot(return_vinf, params.v_esc_leo)),
        aim_separation=aim_separation,
        ledger=_burn_ledger(
            aim_separation,
            return_vinf,
            departure_vinf,
            slug_ratio,
            collimation,
            params,
        ),
        jupiter_vinf_in=jupiter_vinf_in,
        jupiter_vinf_out=jupiter_vinf_out,
        jupiter_turn=jupiter_turn,
        perijove_radius=perijove,
    )


def _is_feasible(
    geometry: _Geometry, params: _FlybyParams, maximum_departure_vinf: float
) -> bool:
    """Whether a rooted geometry satisfies the committed physical/search caps."""

    return (
        abs(geometry.jupiter_vinf_mismatch) <= 1.0e-5
        and geometry.perijove_radius >= params.periapsis_floor - 1.0e-5
        and geometry.departure_vinf <= maximum_departure_vinf + 1.0e-8
    )


def _coarse_candidates(
    synodic_multiple: int,
    params: _FlybyParams,
    maximum_departure_vinf: float,
    phase_samples: int,
    encounter_samples: int,
    slug_ratio: float = _DEFAULT_SLUG_RATIO,
    collimation: float = _DEFAULT_COLLIMATION,
) -> List[_Geometry]:
    """Root every sampled phase's exact unpowered encounter-time closure."""

    period = synodic_multiple * _synodic_period(params)
    phases = np.linspace(-np.pi, np.pi, phase_samples, endpoint=False)
    encounter_grid = np.linspace(
        _ENCOUNTER_MIN_YEARS * _SECONDS_PER_YEAR,
        period - _RETURN_MIN_YEARS * _SECONDS_PER_YEAR,
        encounter_samples,
    )
    found: List[_Geometry] = []
    for phase_value in phases:
        phase = float(phase_value)
        previous: Optional[Tuple[float, float]] = None
        phase_roots: List[float] = []
        for encounter_value in encounter_grid:
            encounter = float(encounter_value)
            geometry = _evaluate_geometry(
                synodic_multiple, phase, encounter, params, slug_ratio, collimation
            )
            if geometry is None:
                previous = None
                continue
            mismatch = geometry.jupiter_vinf_mismatch
            bracket: Optional[Tuple[float, float]] = None
            if abs(mismatch) <= 1.0e-10:
                bracket = (encounter, encounter)
            elif previous is not None and previous[1] * mismatch < 0.0:
                bracket = (previous[0], encounter)
            if bracket is not None:
                if bracket[0] == bracket[1]:
                    root = bracket[0]
                else:

                    def residual(time: float) -> float:
                        evaluated = _evaluate_geometry(
                            synodic_multiple,
                            phase,
                            time,
                            params,
                            slug_ratio,
                            collimation,
                        )
                        if evaluated is None:
                            return float("nan")
                        return evaluated.jupiter_vinf_mismatch

                    try:
                        root = float(
                            brentq(
                                residual,
                                bracket[0],
                                bracket[1],
                                xtol=_ROOT_XTOL_SECONDS,
                            )
                        )
                    except (RuntimeError, ValueError):
                        previous = (encounter, mismatch)
                        continue
                if all(abs(root - old_root) > 1.0 for old_root in phase_roots):
                    rooted = _evaluate_geometry(
                        synodic_multiple, phase, root, params, slug_ratio, collimation
                    )
                    if rooted is not None and _is_feasible(
                        rooted, params, maximum_departure_vinf
                    ):
                        found.append(rooted)
                        phase_roots.append(root)
            previous = (encounter, mismatch)
    return found


def _refine_candidates(
    synodic_multiple: int,
    candidates: Sequence[_Geometry],
    score: Callable[[_Geometry], float],
    params: _FlybyParams,
    maximum_departure_vinf: float,
    slug_ratio: float = _DEFAULT_SLUG_RATIO,
    collimation: float = _DEFAULT_COLLIMATION,
) -> _Geometry:
    """Continuously refine the best coarse starts under exact closure constraints."""

    ranked = sorted(candidates, key=score)[:_REFINEMENT_STARTS]
    refined: List[_Geometry] = list(ranked)
    period_years = synodic_multiple * _synodic_period(params) / _SECONDS_PER_YEAR

    for start in ranked:

        def evaluate(x: npt.NDArray[np.float64]) -> Optional[_Geometry]:
            return _evaluate_geometry(
                synodic_multiple,
                float(x[0]),
                float(x[1]) * _SECONDS_PER_YEAR,
                params,
                slug_ratio,
                collimation,
            )

        def objective(x: npt.NDArray[np.float64]) -> float:
            geometry = evaluate(x)
            return 1.0e9 if geometry is None else score(geometry)

        def equality(x: npt.NDArray[np.float64]) -> float:
            geometry = evaluate(x)
            return 1.0e3 if geometry is None else geometry.jupiter_vinf_mismatch

        def perijove_constraint(x: npt.NDArray[np.float64]) -> float:
            geometry = evaluate(x)
            if geometry is None:
                return -1.0e9
            return geometry.perijove_radius - params.periapsis_floor

        def departure_constraint(x: npt.NDArray[np.float64]) -> float:
            geometry = evaluate(x)
            if geometry is None:
                return -1.0e9
            return maximum_departure_vinf - geometry.departure_vinf

        optimized = minimize(
            objective,
            np.array(
                [start.phase, start.encounter_time / _SECONDS_PER_YEAR],
                dtype=np.float64,
            ),
            method="SLSQP",
            bounds=(
                (-np.pi, np.pi),
                (_ENCOUNTER_MIN_YEARS, period_years - _RETURN_MIN_YEARS),
            ),
            constraints=(
                {"type": "eq", "fun": equality},
                {"type": "ineq", "fun": perijove_constraint},
                {"type": "ineq", "fun": departure_constraint},
            ),
            options={"ftol": 1.0e-12, "maxiter": 500},
        )
        geometry = evaluate(np.asarray(optimized.x, dtype=np.float64))
        if geometry is not None and _is_feasible(
            geometry, params, maximum_departure_vinf
        ):
            refined.append(geometry)
    return min(refined, key=score)


def _as_public(
    geometry: _Geometry, slug_ratio: float, collimation: float
) -> CircularResonanceCandidate:
    """Convert the internal float geometry to the Quantity-valued interface."""

    angle = geometry.aim_separation
    backward = max(0.0, -float(np.cos(angle)))
    ledger = geometry.ledger
    projectile_fraction = 1.0 / (1.0 + slug_ratio)
    exhaust_cant = float(
        np.arcsin(
            np.clip(
                np.sqrt(projectile_fraction) * abs(np.sin(ledger.relative_aim_start)),
                0.0,
                1.0,
            )
        )
    )
    jupiter_radius = float(Jupiter.R.to_value(u.km))
    return CircularResonanceCandidate(
        synodic_multiple=geometry.synodic_multiple,
        period=(geometry.period * u.s).to(u.year),
        earth_minus_jupiter_phase=(-np.degrees(geometry.phase)) * u.deg,
        outbound_time=(geometry.encounter_time * u.s).to(u.year),
        earth_departure_vinf=geometry.departure_vinf * u.km / u.s,
        earth_departure_burn=geometry.departure_burn * u.km / u.s,
        earth_return_vinf=geometry.return_vinf * u.km / u.s,
        collision_speed_200km=geometry.collision_speed_200km * u.km / u.s,
        aim_separation=np.degrees(angle) * u.deg,
        cant_from_backward=(180.0 - np.degrees(angle)) * u.deg,
        backward_projection_fraction=backward,
        transverse_projection_fraction=abs(float(np.sin(angle))),
        ledger=DepartureImpulseLedger(
            slug_ratio=slug_ratio,
            collimation=collimation,
            mirror_rotation=np.degrees(ledger.mirror_rotation) * u.deg,
            aim_at_burn=np.degrees(ledger.aim_at_burn) * u.deg,
            relative_aim_start=np.degrees(ledger.relative_aim_start) * u.deg,
            relative_aim_end=np.degrees(ledger.relative_aim_end) * u.deg,
            closing_speed_start=ledger.closing_speed_start * u.km / u.s,
            closing_speed_end=ledger.closing_speed_end * u.km / u.s,
            impulse_per_impactor_start=ledger.beta_start,
            exhaust_cant=np.degrees(exhaust_cant) * u.deg,
            effective_exhaust_speed=ledger.effective_exhaust_speed * u.km / u.s,
            delivered_fraction=ledger.delivered_fraction,
            head_on_delivered_fraction=ledger.head_on_delivered_fraction,
        ),
        jupiter_vinf=(0.5 * (geometry.jupiter_vinf_in + geometry.jupiter_vinf_out))
        * u.km
        / u.s,
        jupiter_turn=np.degrees(geometry.jupiter_turn) * u.deg,
        perijove_altitude=(geometry.perijove_radius - jupiter_radius) * u.km,
        jupiter_vinf_mismatch=geometry.jupiter_vinf_mismatch * u.km / u.s,
    )


def _analyze_family(
    synodic_multiple: int,
    params: _FlybyParams,
    maximum_departure_vinf: float,
    phase_samples: int,
    encounter_samples: int,
    slug_ratio: float,
    collimation: float,
) -> CircularResonanceFamily:
    """Search and summarize one fixed integer-synodic family."""

    candidates = _coarse_candidates(
        synodic_multiple,
        params,
        maximum_departure_vinf,
        phase_samples,
        encounter_samples,
        slug_ratio,
        collimation,
    )
    if not candidates:
        raise RuntimeError(
            f"no feasible {synodic_multiple}S circular resonance closures found"
        )

    def refine(score: Callable[[_Geometry], float]) -> _Geometry:
        return _refine_candidates(
            synodic_multiple,
            candidates,
            score,
            params,
            maximum_departure_vinf,
            slug_ratio,
            collimation,
        )

    minimum_departure = refine(lambda geometry: geometry.departure_vinf)
    least_backward = refine(lambda geometry: geometry.aim_separation)
    best_delivered = refine(lambda geometry: -geometry.ledger.delivered_fraction)
    ceiling, ceiling_angle = free_aim_ceiling(
        minimum_departure.return_vinf * u.km / u.s,
        minimum_departure.departure_vinf * u.km / u.s,
        slug_ratio,
        collimation,
    )
    return CircularResonanceFamily(
        synodic_multiple=synodic_multiple,
        feasible_grid_closures=len(candidates),
        minimum_departure=_as_public(minimum_departure, slug_ratio, collimation),
        least_backward=_as_public(least_backward, slug_ratio, collimation),
        maximum_delivered_mass=_as_public(best_delivered, slug_ratio, collimation),
        free_aim_ceiling=ceiling,
        free_aim_ceiling_angle=ceiling_angle,
        head_on_crossover_slug_ratio=head_on_crossover_slug_ratio(
            minimum_departure.return_vinf * u.km / u.s,
            minimum_departure.departure_vinf * u.km / u.s,
            collimation,
        ),
    )


def analyze_circular_resonant_impacts(
    maximum_departure_vinf: u.Quantity = _DEFAULT_MAX_DEPARTURE_VINF * u.km / u.s,
    phase_samples: int = _DEFAULT_PHASE_SAMPLES,
    encounter_samples: int = _DEFAULT_ENCOUNTER_SAMPLES,
    slug_ratio: float = _DEFAULT_SLUG_RATIO,
    collimation: float = _DEFAULT_COLLIMATION,
) -> CircularResonanceAnalysis:
    """Optimize ideal circular 2S/3S returns for departure-burn delivered mass.

    Args:
        maximum_departure_vinf: Maximum permitted Earth departure excess speed.
            The default is the production powered-flyby search's 25 km/s cap;
            without a cap, aim-angle objectives run toward arbitrarily costly
            high-energy departures.
        phase_samples: Number of Earth--Jupiter departure phases in the
            reproducible coarse grid. Must be at least 9.
        encounter_samples: Number of Jupiter encounter times in each phase's
            root-bracketing grid. Must be at least 9.
        slug_ratio: Kilograms of carried slug per kilogram of arriving impactor.
            The default 3.0 is Appendix D's; ADR 0009's two-currency optimum is
            6-12, where the aim question is worth materially less.
        collimation: Plume-collimation recovery factor applied to the ideal
            impulse.

    Returns:
        The two fixed-synodic family summaries and search assumptions.

    Raises:
        ValueError: If a search bound or resolution is invalid.
        RuntimeError: If the grid finds no feasible exact closure.
    """

    max_vinf = float(maximum_departure_vinf.to_value(u.km / u.s))
    if max_vinf <= 0.0:
        raise ValueError("maximum_departure_vinf must be positive")
    if phase_samples < 9:
        raise ValueError("phase_samples must be at least 9")
    if encounter_samples < 9:
        raise ValueError("encounter_samples must be at least 9")
    if slug_ratio <= 0.0:
        raise ValueError("slug_ratio must be positive")
    if collimation <= 0.0:
        raise ValueError("collimation must be positive")
    params = _powered_flyby_params(
        cycle_periapsis_speed=puffsat_cycle_periapsis_speed()
    )
    synodic = _synodic_period(params)
    return CircularResonanceAnalysis(
        two_synodic=_analyze_family(
            2,
            params,
            max_vinf,
            phase_samples,
            encounter_samples,
            slug_ratio,
            collimation,
        ),
        three_synodic=_analyze_family(
            3,
            params,
            max_vinf,
            phase_samples,
            encounter_samples,
            slug_ratio,
            collimation,
        ),
        synodic_period=(synodic * u.s).to(u.year),
        slug_ratio=slug_ratio,
        collimation=collimation,
        maximum_departure_vinf=max_vinf * u.km / u.s,
        phase_samples=phase_samples,
        encounter_samples=encounter_samples,
    )


def _row(
    family: CircularResonanceFamily,
    label: str,
    candidate: CircularResonanceCandidate,
) -> List[object]:
    """Render one compact CLI table row."""

    ledger = candidate.ledger
    return [
        f"{family.synodic_multiple}S",
        label,
        candidate.earth_minus_jupiter_phase.to_value(u.deg),
        candidate.earth_departure_vinf.to_value(u.km / u.s),
        candidate.earth_departure_burn.to_value(u.km / u.s),
        candidate.earth_return_vinf.to_value(u.km / u.s),
        candidate.aim_separation.to_value(u.deg),
        ledger.mirror_rotation.to_value(u.deg),
        ledger.relative_aim_start.to_value(u.deg),
        ledger.closing_speed_start.to_value(u.km / u.s),
        ledger.closing_speed_end.to_value(u.km / u.s),
        ledger.impulse_per_impactor_start,
        ledger.effective_exhaust_speed.to_value(u.km / u.s),
        ledger.delivered_fraction,
        ledger.gain_over_head_on,
        candidate.perijove_altitude.to_value(u.km),
    ]


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Score ideal circular 2S/3S returns on departure delivered mass"
    )
    parser.add_argument(
        "--max-departure-vinf",
        type=float,
        default=_DEFAULT_MAX_DEPARTURE_VINF,
        help="maximum Earth departure excess speed in km/s (default: 25)",
    )
    parser.add_argument("--phase-samples", type=int, default=_DEFAULT_PHASE_SAMPLES)
    parser.add_argument(
        "--encounter-samples", type=int, default=_DEFAULT_ENCOUNTER_SAMPLES
    )
    parser.add_argument(
        "--slug-ratio",
        type=float,
        default=_DEFAULT_SLUG_RATIO,
        help="slug kg per impactor kg (default: 3.0, Appendix D)",
    )
    parser.add_argument(
        "--collimation",
        type=float,
        default=_DEFAULT_COLLIMATION,
        help="plume collimation recovery factor (default: 0.8)",
    )
    return parser


def main() -> None:
    """Run and print the committed circular resonance-impact analysis."""

    args = _parser().parse_args()
    analysis = analyze_circular_resonant_impacts(
        maximum_departure_vinf=args.max_departure_vinf * u.km / u.s,
        phase_samples=args.phase_samples,
        encounter_samples=args.encounter_samples,
        slug_ratio=args.slug_ratio,
        collimation=args.collimation,
    )
    rows: List[List[object]] = []
    for family in (analysis.two_synodic, analysis.three_synodic):
        rows.extend(
            [
                _row(family, "minimum departure", family.minimum_departure),
                _row(family, "least backward", family.least_backward),
                _row(family, "max delivered", family.maximum_delivered_mass),
            ]
        )
    print("Ideal circular Earth-Jupiter resonant-impact audit")
    print(
        f"S = {analysis.synodic_period.to_value(u.year):.9f} yr; "
        f"slug ratio k = {analysis.slug_ratio:.3f}; "
        f"collimation eta = {analysis.collimation:.2f}; "
        "strictly unpowered Jupiter flyby"
    )
    print(
        f"search: {analysis.phase_samples} phases x "
        f"{analysis.encounter_samples} encounter times; departure v_inf <= "
        f"{analysis.maximum_departure_vinf.to_value(u.km / u.s):.1f} km/s"
    )
    print(
        tabulate(
            rows,
            headers=[
                "family",
                "selection",
                "E-J phase\ndeg",
                "dep v_inf\nkm/s",
                "dep burn\nkm/s",
                "ret v_inf\nkm/s",
                "aim @SOI\ndeg",
                "mirror\ndeg",
                "aim @burn\ndeg (rel)",
                "w start\nkm/s",
                "w end\nkm/s",
                "beta\nper kg",
                "v_e eff\nkm/s",
                "delivered",
                "vs head-on",
                "perijove alt\nkm",
            ],
            floatfmt=".3f",
        )
    )
    print(
        "\nThe aim @SOI column is a diagnostic: 180 deg is exactly backward at "
        "the patched-conic boundary.  The impulse law uses aim @burn, which is "
        "the angle to the impactor's velocity *relative to the moving vehicle*, "
        "after the free mirror choice of the departure hyperbola."
    )
    print("\nCeiling if the impact angle were free of any delta-v charge:")
    for family in (analysis.two_synodic, analysis.three_synodic):
        print(
            f"  {family.synodic_multiple}S: x{family.free_aim_ceiling:.4f} at "
            f"{family.free_aim_ceiling_angle.to_value(u.deg):.1f} deg; "
            "head-on is itself the optimum above k = "
            f"{family.head_on_crossover_slug_ratio:.2f}"
        )


if __name__ == "__main__":
    main()
