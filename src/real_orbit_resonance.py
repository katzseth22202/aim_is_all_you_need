"""Real-orbit audit of the two-synodic Earth-Jupiter return resonance.

The production growth model is circular and coplanar.  In that model a complete
Earth -> Jupiter -> retrograde Earth trajectory can be repeated after twice the
Earth-Jupiter synodic period because only the planets' relative longitude
matters.  This opt-in analysis replaces the circular planet states with
Astropy's built-in analytical ephemeris and asks whether the same sequence of
relative-phase epochs still admits an *unpowered* Jovian flyby.

Each window is a three-point patched-conic solve: a zero-revolution prograde
Lambert arc from the real Earth state to the real Jupiter state, followed by a
zero-revolution retrograde Lambert arc to Earth's real state at the next
two-synodic epoch.  Jupiter encounter time is free.  The encounter is unpowered
only where the incoming and outgoing Jupiter-relative hyperbolic-excess speeds
are equal; the required turn then uniquely determines perijove radius.  A row
is physically feasible when that radius is at least 4,000 km above Jupiter's
reference radius.

For each selected closure the table also reports a local powered-flyby
diagnostic at the 4,000 km-altitude floor: the signed tangential perijove burn
that would make the selected turn geometrically available.  Positive is a
forward acceleration and negative is a backward braking burn.  This diagnostic
changes the outgoing Jupiter-relative excess speed, so it is not itself a
re-optimized Lambert return to Earth; that powered comparison is deliberately
kept separate from this unpowered baseline.

This is deliberately not imported by :mod:`src.main`.  Run it explicitly with
``python -m src.real_orbit_resonance`` or ``make resonance``.  Astropy's
``builtin`` ephemeris is a fast analytical planetary theory, not a
navigation-grade JPL DE kernel; the latter remains the appropriate final check
for a mission design, particularly after 2100.

The module also audits a chained fixed-mean-cadence fallback: try an exact
two-synodic return from every actual departure, but use three synodic periods
when its DSM proxy exceeds a configurable threshold.  The selected return is
the next departure, preserving integer-synodic timing without treating the
three-synodic alternatives as independent rows.
"""

import argparse
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
from boinor.bodies import Earth, Jupiter, Sun
from boinor.core.iod import izzo
from erfa import ErfaWarning
from scipy.optimize import brentq, minimize_scalar

from src.astro_constants import EARTH_A, JUPITER_A, LEO_ALTITUDE, LOW_JUPITER_ALTITUDE
from src.orbit_utils import escape_velocity

_DEFAULT_START = "2026-08-11"
_DEFAULT_YEARS = 200.0
_EPHEMERIS = "builtin"

# The circular-coplanar zero-revolution solution that minimizes outbound Earth
# v_infinity while respecting the repository's 4,000 km Jovian altitude floor
# has Jupiter 87.4115 degrees ahead of Earth.  Real-orbit epochs repeat this
# *relative phase*, rather than conjunction, every two synodic periods.
_REFERENCE_EARTH_MINUS_JUPITER_DEG = -87.41151408463821

_EPOCH_GRID_STEP_DAYS = 2.0
_ENCOUNTER_TOF_MIN_YEARS = 0.60
_ENCOUNTER_TOF_MAX_YEARS = 1.40
_ENCOUNTER_SAMPLES = 161
_MANEUVER_ENCOUNTER_TOF_MIN_YEARS = 0.25
_MANEUVER_RETURN_TOF_MIN_YEARS = 0.10
_MANEUVER_ENCOUNTER_SAMPLES = 201
_MANEUVER_MAX_EARTH_DEPARTURE_VINF = 25.0  # km/s; production flyby search cap
_HYBRID_PERIJOVE_BURN_CAP = 0.050  # km/s
_HYBRID_PERIJOVE_SAMPLES = 21
_HYBRID_PERIJOVE_RATIO_SAMPLES = 33
_HYBRID_PERIJOVE_RATIO_MAX = 1.0e4
_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.25
_LAMBERT_ITERATIONS = 80
_LAMBERT_RTOL = 1e-10

_MU_SUN = float(Sun.k.to_value(u.km**3 / u.s**2))
_MU_JUPITER = float(Jupiter.k.to_value(u.km**3 / u.s**2))
_JUPITER_RADIUS = float(Jupiter.R.to_value(u.km))
_PERIJOVE_FLOOR = float((Jupiter.R + LOW_JUPITER_ALTITUDE).to_value(u.km))
_EARTH_ESCAPE_SURFACE = float(escape_velocity(Earth).to_value(u.km / u.s))
_EARTH_ESCAPE_LEO = float(escape_velocity(Earth, LEO_ALTITUDE).to_value(u.km / u.s))
_EARTH_ORBIT_RADIUS = float(EARTH_A.to_value(u.km))
_JUPITER_ORBIT_RADIUS = float(JUPITER_A.to_value(u.km))
_EARTH_CIRCULAR_PERIOD_DAYS = (
    2.0 * np.pi * np.sqrt(_EARTH_ORBIT_RADIUS**3 / _MU_SUN) / _SECONDS_PER_DAY
)
_JUPITER_CIRCULAR_PERIOD_DAYS = (
    2.0 * np.pi * np.sqrt(_JUPITER_ORBIT_RADIUS**3 / _MU_SUN) / _SECONDS_PER_DAY
)
_FIXED_TWO_SYNODIC_DAYS = 2.0 / (
    1.0 / _EARTH_CIRCULAR_PERIOD_DAYS - 1.0 / _JUPITER_CIRCULAR_PERIOD_DAYS
)

# Astropy builtin states are in the ICRS-equatorial frame.  Any fixed inertial
# rotation leaves the Lambert solution unchanged, but an ecliptic frame makes
# the repeated relative-longitude phase meaningful and easy to inspect.
_OBLIQUITY = np.deg2rad(23.4392911)
_EQUATORIAL_TO_ECLIPTIC = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(_OBLIQUITY), np.sin(_OBLIQUITY)],
        [0.0, -np.sin(_OBLIQUITY), np.cos(_OBLIQUITY)],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ResonanceAnalysis:
    """Tables and headline result of a real-orbit resonance audit.

    Attributes:
        windows: One row per successive two-synodic trajectory window.
        statistics: Min/max/variance summary for timing and velocity metrics.
        maneuvers: Exact-endpoint per-window maneuver comparison.
        maneuver_summary: Best/worst and feasibility summary by maneuver case.
        all_unpowered_feasible: Whether every window clears the perijove floor.
        all_corrected_feasible: Whether every window has at least one modeled
            exact-endpoint correction.
        ephemeris: Ephemeris provider used to generate the planet states.
        reference_phase: Repeated Earth-minus-Jupiter longitude (degrees).
    """

    windows: pd.DataFrame
    statistics: pd.DataFrame
    maneuvers: pd.DataFrame
    maneuver_summary: pd.DataFrame
    all_unpowered_feasible: bool
    all_corrected_feasible: bool
    ephemeris: str
    reference_phase: float


@dataclass(frozen=True)
class AdaptiveCadenceAnalysis:
    """Conditional two-or-three-synodic fixed-cadence DSM audit.

    Attributes:
        cycles: One row per flown cycle in the chained conditional schedule.
        summary: Headline selection counts and DSM statistics.
        threshold_m_s: Two-synodic DSM threshold that selects a three-synodic
            cycle when exceeded.
        ephemeris: Ephemeris provider used to generate the planet states.
    """

    cycles: pd.DataFrame
    summary: pd.DataFrame
    threshold_m_s: float
    ephemeris: str


@dataclass(frozen=True)
class _Candidate:
    """One exact unpowered Lambert/Lambert closure for a window."""

    encounter_jd: float
    outbound_tof_days: float
    return_tof_days: float
    earth_departure_vinf: float
    earth_departure_periapsis_speed: float
    jupiter_vinf: float
    jupiter_vinf_mismatch: float
    turn_angle_deg: float
    required_perijove: float
    earth_return_vinf: float
    earth_return_collision_speed: float

    @property
    def feasible(self) -> bool:
        """Whether the required perijove clears the configured altitude floor."""
        return self.required_perijove >= _PERIJOVE_FLOOR


@dataclass(frozen=True)
class _EncounterGeometry:
    """Lambert states and scalar flyby geometry at one encounter epoch."""

    encounter_jd: float
    earth_departure_vinf: float
    earth_departure_periapsis_speed: float
    earth_return_vinf: float
    earth_return_collision_speed: float
    incoming_vinf: float
    required_outgoing_vinf: float
    turn_angle: float


@dataclass(frozen=True)
class _ManeuverSolution:
    """One maneuver architecture that returns at the fixed resonance endpoint."""

    case: str
    encounter_jd: float
    perijove_burn: float
    dsm: float
    total_dv: float
    perijove_radius: float
    earth_departure_vinf: float
    earth_departure_periapsis_speed: float
    earth_return_vinf: float
    earth_return_collision_speed: float
    incoming_vinf: float
    outgoing_vinf: float
    turn_angle: float


def _heliocentric_state(
    body: str, jd: float | npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a body's heliocentric J2000-ecliptic position and velocity.

    Args:
        body: Astropy body name (``"earth"`` or ``"jupiter"`` here).
        jd: Scalar or array of TDB Julian dates.

    Returns:
        Position (km) and velocity (km/s), each with a final axis of length 3.
    """
    epoch = Time(jd, format="jd", scale="tdb")
    with warnings.catch_warnings():
        # ERFA warns outside its best-tested date interval.  The limitation is
        # surfaced in this module's CLI and documentation instead of repeated
        # once per state evaluation.
        warnings.simplefilter("ignore", ErfaWarning)
        with solar_system_ephemeris.set(_EPHEMERIS):
            body_position, body_velocity = get_body_barycentric_posvel(body, epoch)
            sun_position, sun_velocity = get_body_barycentric_posvel("sun", epoch)
    position = np.stack(
        [
            (body_position.x - sun_position.x).to_value(u.km),
            (body_position.y - sun_position.y).to_value(u.km),
            (body_position.z - sun_position.z).to_value(u.km),
        ],
        axis=-1,
    )
    velocity = np.stack(
        [
            (body_velocity.x - sun_velocity.x).to_value(u.km / u.s),
            (body_velocity.y - sun_velocity.y).to_value(u.km / u.s),
            (body_velocity.z - sun_velocity.z).to_value(u.km / u.s),
        ],
        axis=-1,
    )
    return (
        np.asarray(position @ _EQUATORIAL_TO_ECLIPTIC.T, dtype=np.float64),
        np.asarray(velocity @ _EQUATORIAL_TO_ECLIPTIC.T, dtype=np.float64),
    )


def _wrapped_relative_phase(
    jd: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Earth-minus-Jupiter heliocentric ecliptic longitude in ``[-pi, pi]``."""
    earth, _ = _heliocentric_state("earth", jd)
    jupiter, _ = _heliocentric_state("jupiter", jd)
    relative = np.arctan2(earth[..., 1], earth[..., 0]) - np.arctan2(
        jupiter[..., 1], jupiter[..., 0]
    )
    return np.asarray(np.angle(np.exp(1j * relative)), dtype=np.float64)


def _resonance_epochs(start: Time, years: float) -> npt.NDArray[np.float64]:
    """Successive epochs at one fixed phase, separated by two synodic cycles.

    Only trajectories whose departure and return both fall within ``years`` of
    ``start`` are included.

    Args:
        start: Beginning of the departure-epoch interval.
        years: Length of that interval (Julian years).

    Returns:
        TDB Julian dates of the repeated-phase epochs.
    """
    if years <= 0.0:
        raise ValueError("years must be positive")
    start_jd = float(start.tdb.jd)
    departure_end = start_jd + years * _DAYS_PER_YEAR
    grid_end = departure_end + 3.0 * _DAYS_PER_YEAR
    grid = np.arange(start_jd, grid_end + _EPOCH_GRID_STEP_DAYS, _EPOCH_GRID_STEP_DAYS)
    phase = np.unwrap(_wrapped_relative_phase(grid))
    reference = np.deg2rad(_REFERENCE_EARTH_MINUS_JUPITER_DEG)
    first_level = reference + np.ceil((phase[0] - reference) / (2.0 * np.pi)) * (
        2.0 * np.pi
    )
    # Four pi advances of unwrapped relative phase are two synodic periods.
    levels = np.arange(first_level, phase[-1] + 1e-12, 4.0 * np.pi)
    guesses = np.interp(levels, phase, grid)
    roots: List[float] = []
    for level, guess in zip(levels, guesses):

        def mismatch(jd: float) -> float:
            wrapped = float(_wrapped_relative_phase(jd))
            lifted = wrapped + 2.0 * np.pi * round(
                (float(level) - wrapped) / (2.0 * np.pi)
            )
            return lifted - float(level)

        roots.append(float(brentq(mismatch, guess - 3.0, guess + 3.0, xtol=1e-9)))
    epochs = np.asarray(roots, dtype=np.float64)
    contained = epochs[epochs <= departure_end]
    if len(contained) < 2:
        raise RuntimeError("study interval contains no two-synodic departure epoch")
    return contained


def _fixed_cadence_epochs(start: Time, years: float) -> npt.NDArray[np.float64]:
    """Exact circular-model two-synodic cadence chained over the horizon.

    The first endpoint is the same real relative-phase epoch used by
    :func:`_resonance_epochs`. Every later endpoint advances by the single
    constant circular-model period, so endpoint timing cannot accumulate drift.

    Args:
        start: Beginning of the study interval.
        years: Length of the endpoint interval (Julian years).

    Returns:
        TDB Julian dates separated by exactly ``_FIXED_TWO_SYNODIC_DAYS``.
    """
    # A two-synodic recurrence is about 2.18 years.  Five years guarantees that
    # _resonance_epochs sees two complete phase epochs even when ``start`` falls
    # immediately after one of them.
    first = float(_resonance_epochs(start, 5.0)[0])
    end = float(start.tdb.jd) + years * _DAYS_PER_YEAR
    count = int(np.floor((end - first) / _FIXED_TWO_SYNODIC_DAYS)) + 1
    epochs = np.asarray(
        first + np.arange(count, dtype=np.float64) * _FIXED_TWO_SYNODIC_DAYS,
        dtype=np.float64,
    )
    if len(epochs) < 2:
        raise RuntimeError("study interval contains no complete fixed-cadence cycle")
    return epochs


def _lambert_pair(
    departure_jd: float,
    return_jd: float,
    encounter_jd: float,
    outbound_low_path: bool,
    return_low_path: bool,
    earth_departure_position: npt.NDArray[np.float64],
    earth_return_position: npt.NDArray[np.float64],
    jupiter_position: Optional[npt.NDArray[np.float64]] = None,
    jupiter_velocity: Optional[npt.NDArray[np.float64]] = None,
) -> Optional[
    Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]
]:
    """Solve the prograde outbound and retrograde return Lambert arcs."""
    if jupiter_position is None or jupiter_velocity is None:
        jupiter_position, jupiter_velocity = _heliocentric_state(
            "jupiter", encounter_jd
        )
    try:
        departure_velocity, jupiter_arrival_velocity = izzo(
            _MU_SUN,
            earth_departure_position,
            jupiter_position,
            (encounter_jd - departure_jd) * _SECONDS_PER_DAY,
            0,
            True,
            outbound_low_path,
            _LAMBERT_ITERATIONS,
            _LAMBERT_RTOL,
        )
        jupiter_departure_velocity, earth_arrival_velocity = izzo(
            _MU_SUN,
            jupiter_position,
            earth_return_position,
            (return_jd - encounter_jd) * _SECONDS_PER_DAY,
            0,
            False,
            return_low_path,
            _LAMBERT_ITERATIONS,
            _LAMBERT_RTOL,
        )
    except (AssertionError, RuntimeError, ValueError):
        return None
    incoming_excess = jupiter_arrival_velocity - jupiter_velocity
    outgoing_excess = jupiter_departure_velocity - jupiter_velocity
    return (
        np.asarray(departure_velocity, dtype=np.float64),
        np.asarray(jupiter_departure_velocity, dtype=np.float64),
        np.asarray(earth_arrival_velocity, dtype=np.float64),
        np.asarray(jupiter_position, dtype=np.float64),
        np.asarray(jupiter_velocity, dtype=np.float64),
        np.asarray(incoming_excess, dtype=np.float64),
        np.asarray(outgoing_excess, dtype=np.float64),
    )


def _candidate_at_encounter(
    departure_jd: float,
    return_jd: float,
    encounter_jd: float,
    outbound_low_path: bool,
    return_low_path: bool,
    earth_departure_position: npt.NDArray[np.float64],
    earth_departure_velocity: npt.NDArray[np.float64],
    earth_return_position: npt.NDArray[np.float64],
    earth_return_velocity: npt.NDArray[np.float64],
) -> Optional[_Candidate]:
    """Turn an equal-v-infinity encounter root into a scored candidate."""
    pair = _lambert_pair(
        departure_jd,
        return_jd,
        encounter_jd,
        outbound_low_path,
        return_low_path,
        earth_departure_position,
        earth_return_position,
    )
    if pair is None:
        return None
    (
        departure_velocity,
        jupiter_departure_velocity,
        earth_arrival_velocity,
        jupiter_position,
        jupiter_velocity,
        incoming_excess,
        outgoing_excess,
    ) = pair
    # The post-Jupiter heliocentric angular momentum must oppose Jupiter's local
    # orbital angular momentum: this is an orbit-level retrograde check that
    # remains meaningful for an inclined real orbit.
    return_h = np.cross(jupiter_position, jupiter_departure_velocity)
    jupiter_h = np.cross(jupiter_position, jupiter_velocity)
    if float(np.dot(return_h, jupiter_h)) >= 0.0:
        return None
    incoming_speed = float(np.linalg.norm(incoming_excess))
    outgoing_speed = float(np.linalg.norm(outgoing_excess))
    if incoming_speed <= 0.0 or outgoing_speed <= 0.0:
        return None
    turn = float(
        np.arccos(
            np.clip(
                np.dot(incoming_excess, outgoing_excess)
                / (incoming_speed * outgoing_speed),
                -1.0,
                1.0,
            )
        )
    )
    if turn <= 1e-12:
        required_perijove = np.inf
    else:
        mean_excess = 0.5 * (incoming_speed + outgoing_speed)
        required_perijove = (
            _MU_JUPITER
            * (1.0 / float(np.sin(0.5 * turn)) - 1.0)
            / (mean_excess * mean_excess)
        )
    departure_vinf = float(
        np.linalg.norm(departure_velocity - earth_departure_velocity)
    )
    return_vinf = float(np.linalg.norm(earth_arrival_velocity - earth_return_velocity))
    return _Candidate(
        encounter_jd=encounter_jd,
        outbound_tof_days=encounter_jd - departure_jd,
        return_tof_days=return_jd - encounter_jd,
        earth_departure_vinf=departure_vinf,
        earth_departure_periapsis_speed=float(
            np.hypot(departure_vinf, _EARTH_ESCAPE_LEO)
        ),
        jupiter_vinf=0.5 * (incoming_speed + outgoing_speed),
        jupiter_vinf_mismatch=incoming_speed - outgoing_speed,
        turn_angle_deg=float(np.rad2deg(turn)),
        required_perijove=required_perijove,
        earth_return_vinf=return_vinf,
        earth_return_collision_speed=float(
            np.hypot(return_vinf, _EARTH_ESCAPE_SURFACE)
        ),
    )


def _window_candidates(departure_jd: float, return_jd: float) -> List[_Candidate]:
    """Enumerate exact unpowered closures in one two-synodic window."""
    earth_departure_position, earth_departure_velocity = _heliocentric_state(
        "earth", departure_jd
    )
    earth_return_position, earth_return_velocity = _heliocentric_state(
        "earth", return_jd
    )
    encounter_min = departure_jd + _ENCOUNTER_TOF_MIN_YEARS * _DAYS_PER_YEAR
    encounter_max = min(
        departure_jd + _ENCOUNTER_TOF_MAX_YEARS * _DAYS_PER_YEAR,
        return_jd - 0.10 * _DAYS_PER_YEAR,
    )
    encounter_grid = np.linspace(encounter_min, encounter_max, _ENCOUNTER_SAMPLES)
    jupiter_positions, jupiter_velocities = _heliocentric_state(
        "jupiter", encounter_grid
    )
    candidates: List[_Candidate] = []
    for outbound_low_path in (True, False):
        for return_low_path in (True, False):
            mismatch: List[float] = []
            for encounter, jupiter_position, jupiter_velocity in zip(
                encounter_grid, jupiter_positions, jupiter_velocities
            ):
                pair = _lambert_pair(
                    departure_jd,
                    return_jd,
                    float(encounter),
                    outbound_low_path,
                    return_low_path,
                    earth_departure_position,
                    earth_return_position,
                    jupiter_position,
                    jupiter_velocity,
                )
                if pair is None:
                    mismatch.append(np.nan)
                else:
                    incoming_excess, outgoing_excess = pair[-2:]
                    mismatch.append(
                        float(
                            np.linalg.norm(incoming_excess)
                            - np.linalg.norm(outgoing_excess)
                        )
                    )
            for left, right, f_left, f_right in zip(
                encounter_grid[:-1],
                encounter_grid[1:],
                mismatch[:-1],
                mismatch[1:],
            ):
                if not np.isfinite(f_left + f_right) or f_left * f_right > 0.0:
                    continue

                def speed_mismatch(encounter_jd: float) -> float:
                    pair = _lambert_pair(
                        departure_jd,
                        return_jd,
                        encounter_jd,
                        outbound_low_path,
                        return_low_path,
                        earth_departure_position,
                        earth_return_position,
                    )
                    if pair is None:
                        raise ValueError("Lambert branch vanished inside root bracket")
                    incoming_excess, outgoing_excess = pair[-2:]
                    return float(
                        np.linalg.norm(incoming_excess)
                        - np.linalg.norm(outgoing_excess)
                    )

                try:
                    root = float(
                        brentq(speed_mismatch, left, right, xtol=1e-7, rtol=1e-12)
                    )
                except (RuntimeError, ValueError):
                    continue
                candidate = _candidate_at_encounter(
                    departure_jd,
                    return_jd,
                    root,
                    outbound_low_path,
                    return_low_path,
                    earth_departure_position,
                    earth_departure_velocity,
                    earth_return_position,
                    earth_return_velocity,
                )
                if candidate is None:
                    continue
                duplicate = any(
                    abs(candidate.encounter_jd - old.encounter_jd) < 1e-6
                    and abs(candidate.earth_departure_vinf - old.earth_departure_vinf)
                    < 1e-6
                    for old in candidates
                )
                if not duplicate:
                    candidates.append(candidate)
    return candidates


def _select_candidate(candidates: Sequence[_Candidate]) -> _Candidate:
    """Choose minimum-departure feasible closure, or the closest failure."""
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if feasible:
        return min(feasible, key=lambda candidate: candidate.earth_departure_vinf)
    if not candidates:
        raise RuntimeError("no exact unpowered Lambert closure found in window")
    # If the flyby cannot clear the floor, retain the solution requiring the
    # highest perijove.  Its floor shortfall is the most useful failure metric.
    return max(
        candidates,
        key=lambda candidate: (
            candidate.required_perijove,
            -candidate.earth_departure_vinf,
        ),
    )


def _perijove_turn_trim(candidate: _Candidate) -> float:
    """Signed tangential burn making a selected turn available at the floor.

    The incoming half of a powered flyby is fixed by the selected Lambert
    arrival.  At the perijove floor, the desired total turn then fixes the
    outgoing hyperbola's half-angle and therefore its excess speed.  The burn is
    the outgoing minus incoming *perijove* speed, where Jupiter's deep gravity
    well supplies the Oberth leverage.

    A candidate that already clears the floor returns exactly zero.  Otherwise
    a positive result accelerates along the instantaneous perijove velocity
    (forward/prograde through the encounter), while a negative result brakes
    against it (backward/retrograde through the encounter).

    This is a local turn-authority diagnostic.  Changing the outgoing excess
    speed also changes the heliocentric return arc, so the value does not claim
    that the original Lambert return still intercepts Earth without retargeting.

    Args:
        candidate: Selected exact-unpowered closure or closest floor failure.

    Returns:
        Signed tangential perijove delta-v (km/s).
    """
    if candidate.feasible:
        return 0.0
    incoming_vinf = candidate.jupiter_vinf + 0.5 * candidate.jupiter_vinf_mismatch
    incoming_ecc = 1.0 + (_PERIJOVE_FLOOR * incoming_vinf * incoming_vinf / _MU_JUPITER)
    incoming_half_angle = float(np.arcsin(min(1.0, 1.0 / incoming_ecc)))
    outgoing_half_angle = np.deg2rad(candidate.turn_angle_deg) - incoming_half_angle
    if outgoing_half_angle <= 0.0 or outgoing_half_angle > 0.5 * np.pi:
        raise RuntimeError("selected turn has no tangential powered-flyby solution")
    outgoing_ecc = 1.0 / float(np.sin(outgoing_half_angle))
    outgoing_vinf = float(np.sqrt(_MU_JUPITER * (outgoing_ecc - 1.0) / _PERIJOVE_FLOOR))
    escape_energy = 2.0 * _MU_JUPITER / _PERIJOVE_FLOOR
    incoming_perijove_speed = float(
        np.sqrt(incoming_vinf * incoming_vinf + escape_energy)
    )
    outgoing_perijove_speed = float(
        np.sqrt(outgoing_vinf * outgoing_vinf + escape_energy)
    )
    return outgoing_perijove_speed - incoming_perijove_speed


def _encounter_geometry(
    departure_jd: float,
    return_jd: float,
    encounter_jd: float,
    outbound_low_path: bool,
    return_low_path: bool,
    earth_departure_position: npt.NDArray[np.float64],
    earth_departure_velocity: npt.NDArray[np.float64],
    earth_return_position: npt.NDArray[np.float64],
    earth_return_velocity: npt.NDArray[np.float64],
    jupiter_position: Optional[npt.NDArray[np.float64]] = None,
    jupiter_velocity: Optional[npt.NDArray[np.float64]] = None,
) -> Optional[_EncounterGeometry]:
    """Lambert geometry joining two fixed successive resonance epochs."""
    pair = _lambert_pair(
        departure_jd,
        return_jd,
        encounter_jd,
        outbound_low_path,
        return_low_path,
        earth_departure_position,
        earth_return_position,
        jupiter_position,
        jupiter_velocity,
    )
    if pair is None:
        return None
    (
        departure_velocity,
        jupiter_departure_velocity,
        earth_arrival_velocity,
        actual_jupiter_position,
        actual_jupiter_velocity,
        incoming_excess,
        outgoing_excess,
    ) = pair
    return_h = np.cross(actual_jupiter_position, jupiter_departure_velocity)
    jupiter_h = np.cross(actual_jupiter_position, actual_jupiter_velocity)
    if float(np.dot(return_h, jupiter_h)) >= 0.0:
        return None
    departure_vinf = float(
        np.linalg.norm(departure_velocity - earth_departure_velocity)
    )
    if departure_vinf > _MANEUVER_MAX_EARTH_DEPARTURE_VINF:
        return None
    incoming_vinf = float(np.linalg.norm(incoming_excess))
    outgoing_vinf = float(np.linalg.norm(outgoing_excess))
    if incoming_vinf <= 0.0 or outgoing_vinf <= 0.0:
        return None
    turn = float(
        np.arccos(
            np.clip(
                np.dot(incoming_excess, outgoing_excess)
                / (incoming_vinf * outgoing_vinf),
                -1.0,
                1.0,
            )
        )
    )
    return_vinf = float(np.linalg.norm(earth_arrival_velocity - earth_return_velocity))
    return _EncounterGeometry(
        encounter_jd=encounter_jd,
        earth_departure_vinf=departure_vinf,
        earth_departure_periapsis_speed=float(
            np.hypot(departure_vinf, _EARTH_ESCAPE_LEO)
        ),
        earth_return_vinf=return_vinf,
        earth_return_collision_speed=float(
            np.hypot(return_vinf, _EARTH_ESCAPE_SURFACE)
        ),
        incoming_vinf=incoming_vinf,
        required_outgoing_vinf=outgoing_vinf,
        turn_angle=turn,
    )


def _split_hyperbola_bend(
    periapsis: float, incoming_vinf: float, outgoing_vinf: float
) -> float:
    """Powered-flyby bend for two excess speeds sharing one periapsis."""
    incoming_ecc = 1.0 + periapsis * incoming_vinf**2 / _MU_JUPITER
    outgoing_ecc = 1.0 + periapsis * outgoing_vinf**2 / _MU_JUPITER
    return float(
        np.arcsin(min(1.0, 1.0 / incoming_ecc))
        + np.arcsin(min(1.0, 1.0 / outgoing_ecc))
    )


def _solution_from_geometry(
    case: str,
    geometry: _EncounterGeometry,
    perijove_burn: float,
    dsm: float,
    periapsis: float,
    outgoing_vinf: float,
) -> _ManeuverSolution:
    """Pack a maneuver solution while preserving its fixed Lambert endpoints."""
    return _ManeuverSolution(
        case=case,
        encounter_jd=geometry.encounter_jd,
        perijove_burn=perijove_burn,
        dsm=dsm,
        total_dv=abs(perijove_burn) + dsm,
        perijove_radius=periapsis,
        earth_departure_vinf=geometry.earth_departure_vinf,
        earth_departure_periapsis_speed=geometry.earth_departure_periapsis_speed,
        earth_return_vinf=geometry.earth_return_vinf,
        earth_return_collision_speed=geometry.earth_return_collision_speed,
        incoming_vinf=geometry.incoming_vinf,
        outgoing_vinf=outgoing_vinf,
        turn_angle=geometry.turn_angle,
    )


def _perijove_only_solution(
    geometry: _EncounterGeometry,
) -> Optional[_ManeuverSolution]:
    """Exact Earth-closing powered-flyby solution, if its bend is available."""
    bend_at_floor = _split_hyperbola_bend(
        _PERIJOVE_FLOOR,
        geometry.incoming_vinf,
        geometry.required_outgoing_vinf,
    )
    if bend_at_floor + 1e-11 < geometry.turn_angle:
        return None
    high = _PERIJOVE_FLOOR
    while (
        _split_hyperbola_bend(
            high, geometry.incoming_vinf, geometry.required_outgoing_vinf
        )
        > geometry.turn_angle
        and high < _PERIJOVE_FLOOR * _HYBRID_PERIJOVE_RATIO_MAX
    ):
        high *= 2.0
    if high >= _PERIJOVE_FLOOR * _HYBRID_PERIJOVE_RATIO_MAX:
        return None
    if high == _PERIJOVE_FLOOR:
        periapsis = high
    else:
        periapsis = float(
            brentq(
                lambda radius: _split_hyperbola_bend(
                    radius,
                    geometry.incoming_vinf,
                    geometry.required_outgoing_vinf,
                )
                - geometry.turn_angle,
                _PERIJOVE_FLOOR,
                high,
                xtol=1e-6,
            )
        )
    escape_energy = 2.0 * _MU_JUPITER / periapsis
    incoming_perijove_speed = float(np.sqrt(geometry.incoming_vinf**2 + escape_energy))
    outgoing_perijove_speed = float(
        np.sqrt(geometry.required_outgoing_vinf**2 + escape_energy)
    )
    return _solution_from_geometry(
        "perijove_only",
        geometry,
        outgoing_perijove_speed - incoming_perijove_speed,
        0.0,
        periapsis,
        geometry.required_outgoing_vinf,
    )


def _dsm_only_solution(geometry: _EncounterGeometry) -> _ManeuverSolution:
    """Unpowered Jovian bend plus a resonance-preserving SOI DSM proxy."""
    max_bend = _split_hyperbola_bend(
        _PERIJOVE_FLOOR, geometry.incoming_vinf, geometry.incoming_vinf
    )
    if geometry.turn_angle <= max_bend:
        high = _PERIJOVE_FLOOR
        while (
            _split_hyperbola_bend(high, geometry.incoming_vinf, geometry.incoming_vinf)
            > geometry.turn_angle
        ):
            high *= 2.0
        periapsis = float(
            brentq(
                lambda radius: _split_hyperbola_bend(
                    radius, geometry.incoming_vinf, geometry.incoming_vinf
                )
                - geometry.turn_angle,
                _PERIJOVE_FLOOR,
                high,
                xtol=1e-6,
            )
        )
        residual_angle = 0.0
    else:
        periapsis = _PERIJOVE_FLOOR
        residual_angle = geometry.turn_angle - max_bend
    dsm = float(
        np.sqrt(
            max(
                0.0,
                geometry.incoming_vinf**2
                + geometry.required_outgoing_vinf**2
                - 2.0
                * geometry.incoming_vinf
                * geometry.required_outgoing_vinf
                * np.cos(residual_angle),
            )
        )
    )
    return _solution_from_geometry(
        "dsm_only",
        geometry,
        0.0,
        dsm,
        periapsis,
        geometry.incoming_vinf,
    )


def _hybrid_state(
    geometry: _EncounterGeometry, perijove_burn: float, log10_ratio: float
) -> Optional[_ManeuverSolution]:
    """A bounded perijove trim followed by the exact-return SOI DSM proxy."""
    periapsis = _PERIJOVE_FLOOR * 10.0**log10_ratio
    escape_energy = 2.0 * _MU_JUPITER / periapsis
    incoming_perijove_speed = float(np.sqrt(geometry.incoming_vinf**2 + escape_energy))
    outgoing_perijove_speed = incoming_perijove_speed + perijove_burn
    if outgoing_perijove_speed**2 <= escape_energy:
        return None
    outgoing_vinf = float(np.sqrt(outgoing_perijove_speed**2 - escape_energy))
    achieved_bend = _split_hyperbola_bend(
        periapsis, geometry.incoming_vinf, outgoing_vinf
    )
    residual_angle = abs(geometry.turn_angle - achieved_bend)
    dsm = float(
        np.sqrt(
            max(
                0.0,
                outgoing_vinf**2
                + geometry.required_outgoing_vinf**2
                - 2.0
                * outgoing_vinf
                * geometry.required_outgoing_vinf
                * np.cos(residual_angle),
            )
        )
    )
    return _solution_from_geometry(
        "hybrid_50mps",
        geometry,
        perijove_burn,
        dsm,
        periapsis,
        outgoing_vinf,
    )


def _hybrid_solution(geometry: _EncounterGeometry) -> _ManeuverSolution:
    """Minimum total correction with a bounded +/-50 m/s perijove trim."""
    # This is the inner loop of the 200-year encounter-time search.  A committed
    # rectangular grid makes its resolution reproducible without nesting two
    # scalar optimizers inside every encounter sample.  Zero and both +/-50 m/s
    # bounds are exact grid points.
    burn_grid = np.linspace(
        -_HYBRID_PERIJOVE_BURN_CAP,
        _HYBRID_PERIJOVE_BURN_CAP,
        _HYBRID_PERIJOVE_SAMPLES,
    )
    log_grid = np.linspace(
        0.0,
        np.log10(_HYBRID_PERIJOVE_RATIO_MAX),
        _HYBRID_PERIJOVE_RATIO_SAMPLES,
    )
    burns = burn_grid[:, np.newaxis]
    periapses = _PERIJOVE_FLOOR * 10.0 ** log_grid[np.newaxis, :]
    escape_energy = 2.0 * _MU_JUPITER / periapses
    incoming_perijove = np.sqrt(geometry.incoming_vinf**2 + escape_energy)
    outgoing_perijove = incoming_perijove + burns
    outgoing_energy = outgoing_perijove**2 - escape_energy
    valid = outgoing_energy > 0.0
    outgoing_vinf = np.sqrt(np.maximum(0.0, outgoing_energy))
    incoming_ecc = 1.0 + periapses * geometry.incoming_vinf**2 / _MU_JUPITER
    outgoing_ecc = 1.0 + periapses * outgoing_vinf**2 / _MU_JUPITER
    achieved_bend = np.arcsin(np.minimum(1.0, 1.0 / incoming_ecc)) + np.arcsin(
        np.minimum(1.0, 1.0 / outgoing_ecc)
    )
    residual_angle = np.abs(geometry.turn_angle - achieved_bend)
    dsm = np.sqrt(
        np.maximum(
            0.0,
            outgoing_vinf**2
            + geometry.required_outgoing_vinf**2
            - 2.0
            * outgoing_vinf
            * geometry.required_outgoing_vinf
            * np.cos(residual_angle),
        )
    )
    total = np.abs(burns) + dsm
    total = np.where(valid, total, np.inf)
    if not np.isfinite(total).any():
        raise RuntimeError("hybrid perijove grid found no hyperbolic departure")
    burn_index, ratio_index = np.unravel_index(int(np.argmin(total)), total.shape)
    best_grid = _hybrid_state(
        geometry,
        float(burn_grid[burn_index]),
        float(log_grid[ratio_index]),
    )
    if best_grid is None:
        raise RuntimeError("best hybrid grid cell did not reconstruct")
    solutions = [best_grid]
    # The hybrid architecture includes the zero-burn DSM-only solution exactly.
    solutions.append(_dsm_only_solution(geometry))
    best = min(solutions, key=lambda solution: solution.total_dv)
    if best.case == "dsm_only":
        return _solution_from_geometry(
            "hybrid_50mps",
            geometry,
            0.0,
            best.dsm,
            best.perijove_radius,
            best.outgoing_vinf,
        )
    return best


def _zero_maneuver_solutions(candidate: _Candidate) -> List[_ManeuverSolution]:
    """Represent an exact unpowered closure in all three comparison cases."""
    geometry = _EncounterGeometry(
        encounter_jd=candidate.encounter_jd,
        earth_departure_vinf=candidate.earth_departure_vinf,
        earth_departure_periapsis_speed=candidate.earth_departure_periapsis_speed,
        earth_return_vinf=candidate.earth_return_vinf,
        earth_return_collision_speed=candidate.earth_return_collision_speed,
        incoming_vinf=candidate.jupiter_vinf,
        required_outgoing_vinf=candidate.jupiter_vinf,
        turn_angle=np.deg2rad(candidate.turn_angle_deg),
    )
    return [
        _solution_from_geometry(
            case,
            geometry,
            0.0,
            0.0,
            candidate.required_perijove,
            candidate.jupiter_vinf,
        )
        for case in ("perijove_only", "dsm_only", "hybrid_50mps")
    ]


def _window_maneuver_solutions(
    departure_jd: float,
    return_jd: float,
    selected_unpowered: Optional[_Candidate],
    cases: Sequence[str] = ("perijove_only", "dsm_only", "hybrid_50mps"),
) -> List[_ManeuverSolution]:
    """Optimize exact-endpoint maneuver cases inside one resonance window."""
    if selected_unpowered is not None and selected_unpowered.feasible:
        return [
            solution
            for solution in _zero_maneuver_solutions(selected_unpowered)
            if solution.case in cases
        ]
    earth_departure_position, earth_departure_velocity = _heliocentric_state(
        "earth", departure_jd
    )
    earth_return_position, earth_return_velocity = _heliocentric_state(
        "earth", return_jd
    )
    encounter_min = departure_jd + _MANEUVER_ENCOUNTER_TOF_MIN_YEARS * _DAYS_PER_YEAR
    encounter_max = return_jd - _MANEUVER_RETURN_TOF_MIN_YEARS * _DAYS_PER_YEAR
    encounter_grid = np.linspace(
        encounter_min, encounter_max, _MANEUVER_ENCOUNTER_SAMPLES
    )
    jupiter_positions, jupiter_velocities = _heliocentric_state(
        "jupiter", encounter_grid
    )
    all_evaluators = {
        "perijove_only": _perijove_only_solution,
        "dsm_only": _dsm_only_solution,
        "hybrid_50mps": _hybrid_solution,
    }
    unknown_cases = set(cases) - set(all_evaluators)
    if unknown_cases:
        raise ValueError(f"unknown maneuver cases: {sorted(unknown_cases)}")
    evaluators = {case: all_evaluators[case] for case in cases}
    best: dict[str, _ManeuverSolution] = {}
    refinement_brackets: dict[str, List[Tuple[float, float, bool, bool]]] = {
        case: [] for case in evaluators
    }
    for outbound_low_path in (True, False):
        for return_low_path in (True, False):
            geometries = [
                _encounter_geometry(
                    departure_jd,
                    return_jd,
                    float(encounter),
                    outbound_low_path,
                    return_low_path,
                    earth_departure_position,
                    earth_departure_velocity,
                    earth_return_position,
                    earth_return_velocity,
                    jupiter_position,
                    jupiter_velocity,
                )
                for encounter, jupiter_position, jupiter_velocity in zip(
                    encounter_grid, jupiter_positions, jupiter_velocities
                )
            ]
            for case, evaluator in evaluators.items():
                sampled: List[Optional[_ManeuverSolution]] = []
                for geometry in geometries:
                    solution = None if geometry is None else evaluator(geometry)
                    sampled.append(solution)
                    if solution is not None and (
                        case not in best or solution.total_dv < best[case].total_dv
                    ):
                        best[case] = solution
                ranked = sorted(
                    (
                        (solution.total_dv, index)
                        for index, solution in enumerate(sampled)
                        if solution is not None
                    ),
                    key=lambda item: item[0],
                )
                for _, index in ranked[:4]:
                    left = float(encounter_grid[max(0, index - 1)])
                    right = float(
                        encounter_grid[min(len(encounter_grid) - 1, index + 1)]
                    )
                    if right > left:
                        refinement_brackets[case].append(
                            (left, right, outbound_low_path, return_low_path)
                        )
    for case, evaluator in evaluators.items():
        for left, right, outbound_low_path, return_low_path in refinement_brackets[
            case
        ]:

            def objective(encounter_jd: float) -> float:
                geometry = _encounter_geometry(
                    departure_jd,
                    return_jd,
                    encounter_jd,
                    outbound_low_path,
                    return_low_path,
                    earth_departure_position,
                    earth_departure_velocity,
                    earth_return_position,
                    earth_return_velocity,
                )
                solution = None if geometry is None else evaluator(geometry)
                return 1.0e6 if solution is None else solution.total_dv

            optimized = minimize_scalar(
                objective,
                bounds=(left, right),
                method="bounded",
                options={"xatol": 1e-7},
            )
            geometry = _encounter_geometry(
                departure_jd,
                return_jd,
                float(optimized.x),
                outbound_low_path,
                return_low_path,
                earth_departure_position,
                earth_departure_velocity,
                earth_return_position,
                earth_return_velocity,
            )
            solution = None if geometry is None else evaluator(geometry)
            if solution is not None and (
                case not in best or solution.total_dv < best[case].total_dv
            ):
                best[case] = solution
    # The hybrid architecture contains the zero-perijove-burn DSM architecture.
    # Independent encounter-time refinements can otherwise leave the hybrid at
    # a slightly worse local numerical minimum even though its feasible set is
    # a strict superset.  Enforce that physical dominance before reporting.
    if (
        "dsm_only" in evaluators
        and "hybrid_50mps" in evaluators
        and "dsm_only" in best
        and (
            "hybrid_50mps" not in best
            or best["dsm_only"].total_dv < best["hybrid_50mps"].total_dv
        )
    ):
        dsm = best["dsm_only"]
        best["hybrid_50mps"] = _ManeuverSolution(
            case="hybrid_50mps",
            encounter_jd=dsm.encounter_jd,
            perijove_burn=0.0,
            dsm=dsm.dsm,
            total_dv=dsm.total_dv,
            perijove_radius=dsm.perijove_radius,
            earth_departure_vinf=dsm.earth_departure_vinf,
            earth_departure_periapsis_speed=dsm.earth_departure_periapsis_speed,
            earth_return_vinf=dsm.earth_return_vinf,
            earth_return_collision_speed=dsm.earth_return_collision_speed,
            incoming_vinf=dsm.incoming_vinf,
            outgoing_vinf=dsm.outgoing_vinf,
            turn_angle=dsm.turn_angle,
        )
    return [best[case] for case in cases if case in best]


def _maneuver_rows(
    schedule: str,
    window: int,
    departure_jd: float,
    return_jd: float,
    solutions: Sequence[_ManeuverSolution],
) -> List[dict[str, object]]:
    """Render all modeled maneuver cases, including explicit infeasible rows."""
    by_case = {solution.case: solution for solution in solutions}
    rows: List[dict[str, object]] = []
    for case in ("perijove_only", "dsm_only", "hybrid_50mps"):
        solution = by_case.get(case)
        row: dict[str, object] = {
            "schedule": schedule,
            "window": window,
            "departure_tdb": Time(departure_jd, format="jd", scale="tdb").isot[:10],
            "return_tdb": Time(return_jd, format="jd", scale="tdb").isot[:10],
            "period_days": return_jd - departure_jd,
            "case": case,
            "feasible": solution is not None,
            "resonance_endpoint_error_s": 0.0,
        }
        if solution is None:
            row.update(
                {
                    "jupiter_tdb": None,
                    "outbound_days": np.nan,
                    "perijove_burn_m_s": np.nan,
                    "dsm_m_s": np.nan,
                    "total_correction_m_s": np.nan,
                    "perijove_altitude_km": np.nan,
                    "earth_departure_vinf_km_s": np.nan,
                    "earth_departure_200km_speed_km_s": np.nan,
                    "earth_return_vinf_km_s": np.nan,
                    "earth_collision_speed_km_s": np.nan,
                    "jupiter_vinf_in_km_s": np.nan,
                    "jupiter_vinf_out_km_s": np.nan,
                    "jupiter_turn_deg": np.nan,
                }
            )
        else:
            row.update(
                {
                    "jupiter_tdb": Time(
                        solution.encounter_jd, format="jd", scale="tdb"
                    ).isot[:10],
                    "outbound_days": solution.encounter_jd - departure_jd,
                    "perijove_burn_m_s": 1000.0 * solution.perijove_burn,
                    "dsm_m_s": 1000.0 * solution.dsm,
                    "total_correction_m_s": 1000.0 * solution.total_dv,
                    "perijove_altitude_km": (
                        solution.perijove_radius - _JUPITER_RADIUS
                    ),
                    "earth_departure_vinf_km_s": solution.earth_departure_vinf,
                    "earth_departure_200km_speed_km_s": (
                        solution.earth_departure_periapsis_speed
                    ),
                    "earth_return_vinf_km_s": solution.earth_return_vinf,
                    "earth_collision_speed_km_s": (
                        solution.earth_return_collision_speed
                    ),
                    "jupiter_vinf_in_km_s": solution.incoming_vinf,
                    "jupiter_vinf_out_km_s": solution.outgoing_vinf,
                    "jupiter_turn_deg": np.rad2deg(solution.turn_angle),
                }
            )
        rows.append(row)
    return rows


def _maneuver_summary(maneuvers: pd.DataFrame) -> pd.DataFrame:
    """Best/worst exact-resonance correction statistics for each architecture."""
    rows = []
    for (schedule, case), group in maneuvers.groupby(["schedule", "case"], sort=False):
        total_windows = int(group["window"].nunique())
        feasible = group.loc[group["feasible"]]
        if feasible.empty:
            rows.append(
                {
                    "schedule": schedule,
                    "case": case,
                    "feasible_windows": 0,
                    "total_windows": total_windows,
                    "all_200yr_repeatable": False,
                }
            )
            continue
        best_index = feasible["total_correction_m_s"].idxmin()
        worst_index = feasible["total_correction_m_s"].idxmax()
        rows.append(
            {
                "schedule": schedule,
                "case": case,
                "feasible_windows": len(feasible),
                "total_windows": total_windows,
                "all_200yr_repeatable": len(feasible) == total_windows,
                "best_total_m_s": float(feasible["total_correction_m_s"].min()),
                "best_departure": feasible.loc[best_index, "departure_tdb"],
                "worst_total_m_s": float(feasible["total_correction_m_s"].max()),
                "worst_departure": feasible.loc[worst_index, "departure_tdb"],
                "mean_total_m_s": float(feasible["total_correction_m_s"].mean()),
                "median_total_m_s": float(feasible["total_correction_m_s"].median()),
                "min_departure_vinf_km_s": float(
                    feasible["earth_departure_vinf_km_s"].min()
                ),
                "max_departure_vinf_km_s": float(
                    feasible["earth_departure_vinf_km_s"].max()
                ),
                "max_endpoint_error_s": float(
                    feasible["resonance_endpoint_error_s"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _maneuver_correction_table(maneuvers: pd.DataFrame) -> pd.DataFrame:
    """Build a compact per-window table of total correction by architecture."""
    identifiers = [
        "schedule",
        "window",
        "departure_tdb",
        "return_tdb",
        "period_days",
    ]
    table = maneuvers.pivot(
        index=identifiers,
        columns="case",
        values="total_correction_m_s",
    ).reset_index()
    table.columns.name = None
    return table.rename(
        columns={
            "perijove_only": "perijove_only_m_s",
            "dsm_only": "dsm_proxy_m_s",
            "hybrid_50mps": "hybrid_50mps_m_s",
        }
    )


def _statistics(windows: pd.DataFrame) -> pd.DataFrame:
    """Build min/max/variance statistics for the requested headline metrics."""
    metrics = [
        ("two-synodic period", "period_days", "day"),
        ("Earth departure v_inf", "earth_departure_vinf_km_s", "km/s"),
        (
            "Earth departure speed at 200 km",
            "earth_departure_200km_speed_km_s",
            "km/s",
        ),
        ("Earth return v_inf", "earth_return_vinf_km_s", "km/s"),
        ("Earth surface collision speed", "earth_collision_speed_km_s", "km/s"),
        ("Jupiter encounter v_inf", "jupiter_vinf_km_s", "km/s"),
        (
            "perijove turn-trim magnitude",
            "perijove_turn_trim_abs_m_s",
            "m/s",
        ),
        ("required perijove altitude", "required_perijove_altitude_km", "km"),
    ]
    rows = []
    for label, column, unit in metrics:
        values = windows[column].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if abs(mean) < 1e-15:
            std_dev_pct = 0.0
            peak_to_peak_pct = 0.0
        else:
            std_dev_pct = 100.0 * float(np.std(values)) / mean
            peak_to_peak_pct = 100.0 * (maximum - minimum) / mean
        rows.append(
            {
                "metric": label,
                "unit": unit,
                "min": minimum,
                "max": maximum,
                "mean": mean,
                "variance": float(np.var(values)),
                "std_dev": float(np.std(values)),
                "std_dev_pct": std_dev_pct,
                "peak_to_peak_pct": peak_to_peak_pct,
            }
        )
    return pd.DataFrame(rows)


def analyze_two_synodic_resonance(
    start: str = _DEFAULT_START, years: float = _DEFAULT_YEARS
) -> ResonanceAnalysis:
    """Audit successive real-orbit two-synodic windows.

    Args:
        start: First date on which a departure epoch may occur (ISO date).
        years: Span of departure epochs to include (Julian years).

    Returns:
        Per-window and summary tables plus the all-windows feasibility result.
    """
    start_epoch = Time(start, scale="tdb")
    epochs = _resonance_epochs(start_epoch, years)
    rows = []
    maneuver_rows: List[dict[str, object]] = []
    for index, (departure_jd, return_jd) in enumerate(zip(epochs[:-1], epochs[1:])):
        candidates = _window_candidates(float(departure_jd), float(return_jd))
        selected = _select_candidate(candidates)
        feasible_count = sum(candidate.feasible for candidate in candidates)
        perijove_altitude = selected.required_perijove - _JUPITER_RADIUS
        turn_trim_m_s = 1000.0 * _perijove_turn_trim(selected)
        if turn_trim_m_s > 1e-9:
            turn_trim_direction = "forward"
        elif turn_trim_m_s < -1e-9:
            turn_trim_direction = "backward"
        else:
            turn_trim_direction = "none"
        rows.append(
            {
                "window": index,
                "departure_tdb": Time(departure_jd, format="jd", scale="tdb").isot[:10],
                "jupiter_tdb": Time(
                    selected.encounter_jd, format="jd", scale="tdb"
                ).isot[:10],
                "return_tdb": Time(return_jd, format="jd", scale="tdb").isot[:10],
                "period_days": return_jd - departure_jd,
                "outbound_days": selected.outbound_tof_days,
                "return_days": selected.return_tof_days,
                "earth_departure_vinf_km_s": selected.earth_departure_vinf,
                "earth_departure_200km_speed_km_s": (
                    selected.earth_departure_periapsis_speed
                ),
                "jupiter_vinf_km_s": selected.jupiter_vinf,
                "jupiter_vinf_mismatch_m_s": (1000.0 * selected.jupiter_vinf_mismatch),
                "jupiter_turn_deg": selected.turn_angle_deg,
                "required_perijove_altitude_km": perijove_altitude,
                "perijove_floor_margin_km": (
                    selected.required_perijove - _PERIJOVE_FLOOR
                ),
                "perijove_turn_trim_m_s": turn_trim_m_s,
                "perijove_turn_trim_abs_m_s": abs(turn_trim_m_s),
                "perijove_turn_trim_direction": turn_trim_direction,
                "earth_return_vinf_km_s": selected.earth_return_vinf,
                "earth_collision_speed_km_s": (selected.earth_return_collision_speed),
                "unpowered_feasible": selected.feasible,
                "exact_closures": len(candidates),
                "feasible_closures": feasible_count,
            }
        )
        solutions = _window_maneuver_solutions(
            float(departure_jd), float(return_jd), selected
        )
        maneuver_rows.extend(
            _maneuver_rows(
                "real_phase_2S",
                index,
                float(departure_jd),
                float(return_jd),
                solutions,
            )
        )
    windows = pd.DataFrame(rows)
    fixed_epochs = _fixed_cadence_epochs(start_epoch, years)
    for index, (departure_jd, return_jd) in enumerate(
        zip(fixed_epochs[:-1], fixed_epochs[1:])
    ):
        # Only the first fixed endpoint is a real-phase epoch; subsequent fixed
        # endpoints intentionally test whether maneuver authority can replace
        # the natural +/- timing variation without accumulating cycle drift.
        solutions = _window_maneuver_solutions(
            float(departure_jd), float(return_jd), None
        )
        maneuver_rows.extend(
            _maneuver_rows(
                "fixed_mean_2S",
                index,
                float(departure_jd),
                float(return_jd),
                solutions,
            )
        )
    maneuvers = pd.DataFrame(maneuver_rows)
    maneuver_summary = _maneuver_summary(maneuvers)
    return ResonanceAnalysis(
        windows=windows,
        statistics=_statistics(windows),
        maneuvers=maneuvers,
        maneuver_summary=maneuver_summary,
        all_unpowered_feasible=bool(windows["unpowered_feasible"].all()),
        all_corrected_feasible=bool(
            maneuver_summary.loc[
                maneuver_summary["case"].isin(["dsm_only", "hybrid_50mps"]),
                "all_200yr_repeatable",
            ].all()
        ),
        ephemeris=_EPHEMERIS,
        reference_phase=_REFERENCE_EARTH_MINUS_JUPITER_DEG,
    )


def analyze_adaptive_synodic_cadence(
    start: str = _DEFAULT_START,
    years: float = _DEFAULT_YEARS,
    threshold_m_s: float = 50.0,
) -> AdaptiveCadenceAnalysis:
    """Audit a chained 2S/3S policy on the fixed mean synodic lattice.

    Each cycle first optimizes the DSM proxy for a return exactly two mean
    Earth-Jupiter synodic periods after its actual departure. If that DSM is
    strictly greater than ``threshold_m_s``, the flown return is instead
    optimized at exactly three synodic periods. The selected return becomes
    the next departure, so a 3S choice correctly shifts the later phase lattice.
    Only cycles whose selected return fits inside the requested horizon are
    reported.

    Args:
        start: First date on which a departure epoch may occur (ISO date).
        years: Length of the endpoint study interval (Julian years).
        threshold_m_s: DSM threshold above which the fallback changes from a
            two-synodic to a three-synodic cycle.

    Returns:
        Per-cycle policy choices and a one-row headline summary.
    """
    if years <= 0.0:
        raise ValueError("years must be positive")
    if threshold_m_s < 0.0:
        raise ValueError("threshold_m_s must be nonnegative")
    start_epoch = Time(start, scale="tdb")
    departure_jd = float(_resonance_epochs(start_epoch, 5.0)[0])
    horizon_end_jd = float(start_epoch.tdb.jd) + years * _DAYS_PER_YEAR
    one_synodic_days = _FIXED_TWO_SYNODIC_DAYS / 2.0
    rows: List[dict[str, object]] = []
    while departure_jd + 2.0 * one_synodic_days <= horizon_end_jd:
        two_return_jd = departure_jd + 2.0 * one_synodic_days
        two_solutions = _window_maneuver_solutions(
            departure_jd, two_return_jd, None, cases=("dsm_only",)
        )
        if not two_solutions:
            raise RuntimeError("two-synodic DSM search found no solution")
        two_solution = two_solutions[0]
        two_dsm_m_s = 1000.0 * two_solution.dsm
        selected_solution = two_solution
        selected_multiple = 2
        selected_return_jd = two_return_jd
        if two_dsm_m_s > threshold_m_s:
            selected_multiple = 3
            selected_return_jd = departure_jd + 3.0 * one_synodic_days
            if selected_return_jd > horizon_end_jd:
                break
            three_solutions = _window_maneuver_solutions(
                departure_jd,
                selected_return_jd,
                None,
                cases=("dsm_only",),
            )
            if not three_solutions:
                raise RuntimeError("three-synodic DSM search found no solution")
            selected_solution = three_solutions[0]
        selected_dsm_m_s = 1000.0 * selected_solution.dsm
        rows.append(
            {
                "cycle": len(rows),
                "departure_tdb": Time(departure_jd, format="jd", scale="tdb").isot[:10],
                "return_tdb": Time(selected_return_jd, format="jd", scale="tdb").isot[
                    :10
                ],
                "selected_synodic_periods": selected_multiple,
                "selected_period_days": selected_return_jd - departure_jd,
                "two_synodic_dsm_m_s": two_dsm_m_s,
                "selected_dsm_m_s": selected_dsm_m_s,
                "three_synodic_selected": selected_multiple == 3,
                "jupiter_tdb": Time(
                    selected_solution.encounter_jd, format="jd", scale="tdb"
                ).isot[:10],
                "outbound_days": selected_solution.encounter_jd - departure_jd,
                "perijove_altitude_km": (
                    selected_solution.perijove_radius - _JUPITER_RADIUS
                ),
                "earth_departure_vinf_km_s": (selected_solution.earth_departure_vinf),
                "earth_departure_200km_speed_km_s": (
                    selected_solution.earth_departure_periapsis_speed
                ),
                "earth_return_vinf_km_s": selected_solution.earth_return_vinf,
                "earth_collision_speed_km_s": (
                    selected_solution.earth_return_collision_speed
                ),
                "jupiter_vinf_in_km_s": selected_solution.incoming_vinf,
                "jupiter_vinf_out_km_s": selected_solution.outgoing_vinf,
                "jupiter_turn_deg": np.rad2deg(selected_solution.turn_angle),
                "resonance_endpoint_error_s": 0.0,
            }
        )
        departure_jd = selected_return_jd
    cycles = pd.DataFrame(rows)
    if cycles.empty:
        raise RuntimeError("study interval contains no complete adaptive cycle")
    three = cycles.loc[cycles["three_synodic_selected"], "selected_dsm_m_s"]
    selected = cycles["selected_dsm_m_s"]
    worst_three_departure: Optional[str]
    if three.empty:
        three_mean = np.nan
        three_median = np.nan
        three_max = np.nan
        worst_three_departure = None
    else:
        three_mean = float(three.mean())
        three_median = float(three.median())
        three_max = float(three.max())
        worst_three_departure = str(cycles.loc[three.idxmax(), "departure_tdb"])
    three_count = int(cycles["three_synodic_selected"].sum())
    summary = pd.DataFrame(
        [
            {
                "total_cycles": len(cycles),
                "two_synodic_cycles": len(cycles) - three_count,
                "three_synodic_cycles": three_count,
                "three_synodic_fraction": three_count / len(cycles),
                "three_synodic_cycles_per_century": 100.0 * three_count / years,
                "three_synodic_mean_dsm_m_s": three_mean,
                "three_synodic_median_dsm_m_s": three_median,
                "three_synodic_max_dsm_m_s": three_max,
                "worst_three_synodic_departure": worst_three_departure,
                "selected_max_dsm_m_s": float(selected.max()),
                "selected_mean_dsm_m_s": float(selected.mean()),
                "selected_median_dsm_m_s": float(selected.median()),
                "selected_dsm_over_threshold_cycles": int(
                    (selected > threshold_m_s).sum()
                ),
            }
        ]
    )
    return AdaptiveCadenceAnalysis(
        cycles=cycles,
        summary=summary,
        threshold_m_s=threshold_m_s,
        ephemeris=_EPHEMERIS,
    )


def _parser() -> argparse.ArgumentParser:
    """Command-line parser for the opt-in analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", default=_DEFAULT_START, help="first ISO departure date"
    )
    parser.add_argument(
        "--years",
        default=_DEFAULT_YEARS,
        type=float,
        help="span of departure epochs in Julian years (default: 200)",
    )
    parser.add_argument(
        "--csv",
        help="optional path to write the full per-window table as CSV",
    )
    parser.add_argument(
        "--maneuver-csv",
        help="optional path to write exact-endpoint maneuver cases as CSV",
    )
    parser.add_argument(
        "--adaptive-threshold-m-s",
        default=50.0,
        type=float,
        help="2S DSM threshold for the chained 3S fallback (default: 50)",
    )
    parser.add_argument(
        "--adaptive-csv",
        help="optional path to write the chained 2S/3S policy cycles as CSV",
    )
    return parser


def main() -> None:
    """Run and print the real-orbit two-synodic audit."""
    args = _parser().parse_args()
    analysis = analyze_two_synodic_resonance(start=args.start, years=args.years)
    adaptive = analyze_adaptive_synodic_cadence(
        start=args.start,
        years=args.years,
        threshold_m_s=args.adaptive_threshold_m_s,
    )
    if args.csv:
        analysis.windows.to_csv(args.csv, index=False)
    if args.maneuver_csv:
        analysis.maneuvers.to_csv(args.maneuver_csv, index=False)
    if args.adaptive_csv:
        adaptive.cycles.to_csv(args.adaptive_csv, index=False)
    print(
        "Real-orbit two-synodic resonance audit\n"
        f"ephemeris: Astropy {analysis.ephemeris!r} analytical ephemeris "
        "(not a navigation-grade JPL DE kernel)\n"
        f"reference Earth-minus-Jupiter phase: {analysis.reference_phase:.6f} deg\n"
        f"perijove floor: {LOW_JUPITER_ALTITUDE.to_value(u.km):.0f} km altitude\n"
        "flyby baseline: strictly unpowered; no Jovian trim burn applied\n"
        "turn-trim diagnostic: signed tangential burn at the perijove floor; "
        "negative = backward/braking, positive = forward/accelerating; this "
        "local diagnostic requires powered-trajectory retargeting to preserve "
        "the Earth intercept\n"
    )
    with pd.option_context("display.max_rows", None, "display.width", 240):
        print(
            analysis.windows.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
    feasible = int(analysis.windows["unpowered_feasible"].sum())
    total = len(analysis.windows)
    print("\nSummary statistics (population variance/std. dev.):")
    print(analysis.statistics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(
        "\nExact-endpoint maneuver comparison. Every return epoch is exactly "
        "the following cycle's departure epoch; endpoint error is zero by "
        "construction. The DSM is an SOI velocity-correction proxy after the "
        "Jovian bend, not an optimized finite-location interplanetary burn. "
        "Blank perijove-only cells are infeasible:"
    )
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(
            _maneuver_correction_table(analysis.maneuvers).to_string(
                index=False, float_format=lambda x: f"{x:.3f}"
            )
        )
    print("\nManeuver best/worst summary:")
    print(
        analysis.maneuver_summary.to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print(
        "\nCorrected exact-endpoint repeatability across both schedules: "
        f"{analysis.all_corrected_feasible}."
    )
    print(
        "\nConditional fixed-cadence policy. Try 2S from every actual "
        f"departure; select 3S when 2S DSM > {adaptive.threshold_m_s:.3f} m/s. "
        "The selected return becomes the next departure:"
    )
    print(adaptive.summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nUnpowered feasibility: {feasible}/{total} windows "
        f"({100.0 * feasible / total:.1f}%); "
        f"all windows feasible = {analysis.all_unpowered_feasible}."
    )
    if not analysis.all_unpowered_feasible:
        failures = analysis.windows.loc[~analysis.windows["unpowered_feasible"]]
        print(
            "The exact two-synodic resonance is therefore NOT an indefinitely "
            "repeatable unpowered trajectory at this reference phase. "
            f"Worst perijove-floor shortfall: "
            f"{-float(failures['perijove_floor_margin_km'].min()):.0f} km."
        )


if __name__ == "__main__":
    main()
