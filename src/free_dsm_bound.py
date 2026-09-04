"""How cheap could the 20-day split be if the burn went anywhere it liked?

ADR 0026 priced the split's correction as ADR 0011's **seam proxy**: an exact
velocity match at the Jupiter patched-conic boundary, taken instantaneously.
That is the most constrained place a burn could possibly go, and ADR 0011 said
so at the time -- "an upper-bound-like architecture comparison".  Three cycles
pay 1.4-2.1 km/s there, and ADR 0026's verdict against solar-electric
propulsion rests on those being real.

This module tests that, and it does so **without building a low-thrust
transcription**, by way of one observation:

    The minimum-delta-v *impulsive* trajectory, with the burns free to occur
    anywhere, is a **lower bound** on the minimum-delta-v *finite-thrust*
    trajectory between the same boundary conditions.

The impulsive problem is the finite-thrust problem with the thrust-magnitude
constraint deleted -- a relaxation -- so its optimum cannot be worse.  Hence:

* if the free-burn optimum still exceeds what the array can deliver, no
  finite-thrust solution exists either, and ADR 0026's verdict holds rigorously
  rather than by assumption;
* only if the free-burn optimum drops near the deliverable budget does a
  Sims-Flanagan transcription become worth building, because only then could
  the answer go either way.

**The trajectory posed here.** The growth wave leaves Earth at the cycle's
departure epoch and must reach Earth ``split_days`` before the nozzle wave --
the same boundary conditions ADR 0013 imposes.  The default gap is the 20 days
ADR 0013 and ADR 0026 price; ``--split-days 10`` flies the gap the paper flies,
where the same three cycles are expensive and pay roughly half as much.

Between them it flies an MGA-nDSM shape: up to
:data:`DEFAULT_MANEUVERS_PER_LEG` maneuvers on the outbound leg, an unpowered
Jupiter flyby at a free encounter epoch with a free aim point, and up to the
same number on the return.  Every maneuver is free in time and direction, and
unbounded in magnitude.

The flyby is left **unpowered** deliberately.  A perijove burn is an Oberth
device and a solar-electric stage cannot fire one -- it has no thrust to spend
in the few hours that matter -- so allowing one would relax a constraint the
hardware really has, and the bound would stop bounding the question asked.

**The answer, and why.** Nothing helps.  On all three expensive cycles the
optimum is a single burn just after Jupiter, at the perijove floor, keeping the
seam solution's own encounter epoch to within a tenth of a day, with every
additional maneuver left at zero magnitude and nothing at all spent outbound.
It comes back **0.015-0.09% above** the seam charge -- that residual is this
module rebuilding the seam trajectory in its own flyby model rather than
importing it, and it is the wrong sign to be a saving.  The reason is geometric
rather than about placement: these cycles need a **109-degree turn and Jupiter
can only supply 104-107** even scraping the 4,000 km floor, so the correction
is paying off a 1.0-4.6 degree *bend deficit* at 20-21 km/s, which runs about
0.36 km/s per degree and accounts for 82-86% of the charge.  The one alternative -- slowing down before
the flyby to buy turn authority, since ``e = 1 + r_p v_inf^2 / mu`` falls with
speed -- costs more than the turn returns, and the optimiser rejects it by
leaving the outbound maneuvers at zero.

**Why the search is trusted this far, and no further.**  The optimum reported
here is an upper bound on the relaxation's own optimum -- a heuristic global
search cannot certify that nothing cheaper exists -- so the argument above is
only as strong as the search.  Two things make it strong enough to act on and
neither is a proof: the cycle that sizes the hardware clears what the array
delivers by a *factor* -- 8.2 at the 20-day gap, 4.0 at the paper's 10-day one
-- rather than by a few percent, and blind sampling of the box
(:func:`blind_sample_floor`, the ADR 0008 harness rule) finds nothing within
fifty times the seam, so the seeded basin is not one of a crowd.

Run it with ``make dsm-bound``; it is not imported by ``src.main`` and is not
part of ``make all``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.time import Time
from boinor.core.iod import izzo
from boinor.core.propagation import farnocchia as _propagate_rv
from scipy.optimize import differential_evolution, minimize

from src.real_orbit_resonance import (
    _DAYS_PER_YEAR,
    _FIXED_TWO_SYNODIC_DAYS,
    _MU_JUPITER,
    _MU_SUN,
    _PERIJOVE_FLOOR,
    _SECONDS_PER_DAY,
    _heliocentric_state,
    _resonance_epochs,
    _window_maneuver_solutions,
)

#: Study window, matching ADR 0013 and ADR 0026 so the chains are the same.
DEFAULT_START = "2026-08-11"
DEFAULT_HORIZON_YEARS = 30.0
DEFAULT_THRESHOLD_M_S = 50.0
DEFAULT_SPLIT_DAYS = 20.0
#: Seam charge (km/s) above which a cycle is worth bounding.  Set at what the
#: array actually delivers (ADR 0026's 254.5 m/s over a whole trajectory)
#: rather than at a round number, because that is the threshold the question
#: is about: below it low thrust already pays, and there is nothing to test.
#: The split's cost is bimodal with nothing between 2 m/s and 0.4 km/s, so the
#: same three cycles are selected at both the 10-day gap the paper flies and
#: the 20-day gap ADR 0013 and ADR 0026 price.
DEFAULT_MINIMUM_DV = 0.2545

#: Bounds on the free departure excess speed (km/s).  The growth wave's
#: departure is provided by the head-on nozzle, not by propellant, so it is not
#: charged in the objective -- but it cannot be unbounded either, or the
#: optimiser buys the whole correction at Earth for free.  The box spans the
#: departure excess ADR 0013's own solutions use, plus room either side.
DEPARTURE_VINF_BOUNDS = (8.0, 20.0)
#: Fraction of each leg's flight time before its maneuver.  The lower end must
#: reach essentially zero: the seam proxy burns *at* Jupiter, so a floor of
#: 0.05 -- some 36 days into a two-year return leg -- cannot express the very
#: trajectory this bound has to be seeded with, and the search came back worse
#: than the thing it was bounding.  1e-4 of a leg is a few hours.
DSM_FRACTION_BOUNDS = (1.0e-4, 0.98)
#: Perijove radii the flyby may use, as a multiple of the 4,000 km altitude
#: floor the repository enforces everywhere.
PERIJOVE_RATIO_BOUNDS = (1.0, 60.0)
#: How far the Jupiter encounter may move from the seam solution's own
#: encounter epoch (days).  Wide enough to be a free variable, bounded so the
#: legs stay zero-revolution.
ENCOUNTER_WINDOW_DAYS = 400.0

#: Maneuvers allowed on each leg.  One per leg reproduces ADR 0011's seam
#: architecture; more is what makes the result a bound rather than a sample,
#: because the minimum-delta-v *unbounded-thrust* solution is impulsive and
#: primer-vector theory says it needs only a few impulses -- generically no
#: more than about four for a transfer of this shape.
DEFAULT_MANEUVERS_PER_LEG = 5
#: Largest free maneuver (km/s).  Wide enough to contain the manoeuvre that
#: would slow the wave before Jupiter to buy turn authority, which is the one
#: trade a multi-impulse solution could exploit and a single-impulse one
#: cannot.
#:
#: Each free maneuver is carried as **(magnitude, right ascension,
#: declination)**, not as three Cartesian components.  That is not cosmetic.
#: The objective sums ``norm(dv)``, and in Cartesian coordinates that norm has
#: a kink at the origin: the finite-difference gradient there is about 10 in
#: normalised units, which swamps the small smooth gradients in the timing and
#: B-plane directions, fails the line search, and stops L-BFGS-B before it can
#: make the improvement the single-maneuver case finds trivially.  In polar
#: form the cost is *linear* in magnitude, so the kink is gone -- and
#: ``d(cost)/d(magnitude)`` evaluated at zero is precisely the primer-vector
#: condition for whether an impulse there is worth adding.
MANEUVER_MAGNITUDE_BOUND = 5.0

#: Search settings, recorded per the ADR 0007 lesson.
#:
#: The engine is **monotonic basin hopping**, not differential evolution.  DE
#: was tried first and visibly failed: unseeded it returned 3,367-4,432 m/s on
#: a cycle whose seeded value is 2,127, and at 40 parameters it is worse still.
#: MGA-nDSM landscapes are many narrow basins rather than one broad one, which
#: is the structure MBH is built for and DE is not.  DE is kept behind a flag
#: as an independent cross-check, not as the primary search.
MBH_ITERATIONS = 60
#: Perturbation scale, as a fraction of the (normalised) box width.
MBH_PERTURBATION = 0.15
#: Chance each coordinate is perturbed on a given hop.  Perturbing all of them
#: at once mostly lands outside every basin.
MBH_PERTURB_PROBABILITY = 0.35
#: Local method.  Nelder-Mead was the first choice and is wrong here: it is
#: derivative-free and degrades badly above about ten dimensions, and this
#: local step is load-bearing -- refining from the seeded single-maneuver
#: solution is exactly what tests whether inserting a maneuver anywhere helps.
LOCAL_METHOD = "L-BFGS-B"
#: Finite-difference step in normalised coordinates.  Parameters are rescaled
#: to the unit box before optimising because their natural units span eleven
#: orders (a 1e-4 leg fraction against a 400-day encounter offset), which no
#: gradient method survives.
LOCAL_FD_STEP = 1.0e-7
LOCAL_MAXITER = 400
#: Seeds started slightly *off* zero maneuver magnitude.  Seeding an extra
#: maneuver at exactly zero puts the optimiser on the kink of ``norm(dv)``,
#: where every direction raises the cost to first order while any benefit is
#: second order -- L-BFGS-B stops immediately and the richer budget scores
#: worse than the poorer one, which is impossible for a correct search.  These
#: start each free maneuver at a small random magnitude instead, spread across
#: the leg, so the search begins on a differentiable part of the objective.
OFFSET_SEEDS = 6
#: Magnitude range for those nudges (km/s).
OFFSET_SEED_RANGE = (0.005, 0.100)
#: Blind samples of the box drawn per cycle, as the ADR 0008 harness check:
#: an optimiser result is only worth reading once you know what the box looks
#: like without one.  Here the check runs the other way round from ADR 0008's
#: -- the question is not whether random points fly at all, but whether any of
#: them come near the seeded basin.  2,000 costs about three seconds.
BLIND_SAMPLES = 2000
DE_POPSIZE = 40
DE_MAXITER = 320
DE_TOLERANCE = 1e-8
DE_SEED = 20260902
#: Penalty (km/s) added per unit constraint violation, so the global search sees
#: a continuous landscape instead of a wall.
INFEASIBLE_PENALTY = 1.0e3

_LAMBERT_ITERATIONS = 350
_LAMBERT_RTOL = 1e-8


@dataclass(frozen=True)
class FreeDsmSolution:
    """One free-maneuver trajectory for a cycle's growth wave.

    Attributes:
        departure_jd: TDB Julian date the wave leaves Earth.
        arrival_jd: TDB Julian date it must reach Earth.
        encounter_jd: TDB Julian date of the Jupiter flyby it chose.
        departure_vinf: Earth departure excess speed (km/s).
        first_dsm: Maneuver on the outbound leg (km/s).
        second_dsm: Maneuver on the return leg (km/s).
        perijove_radius: Flyby perijove (km).
        total_dv: ``first_dsm + second_dsm`` (km/s) -- the bound.
        seam_dv: What ADR 0026 charges for the same cycle (km/s).
        feasible: Whether both Lambert arcs solved and the flyby cleared the
            perijove floor.
    """

    departure_jd: float
    arrival_jd: float
    encounter_jd: float
    departure_vinf: float
    first_dsm: float
    second_dsm: float
    perijove_radius: float
    total_dv: float
    seam_dv: float
    feasible: bool

    @property
    def improvement(self) -> float:
        """How many times cheaper the free-burn optimum is than the seam."""
        return self.seam_dv / self.total_dv if self.total_dv > 0.0 else float("inf")


def _unit_from_angles(
    right_ascension: float, declination: float
) -> npt.NDArray[np.float64]:
    """Unit vector from a right ascension and declination (radians).

    Args:
        right_ascension: Angle in the reference plane.
        declination: Angle out of it.

    Returns:
        A unit 3-vector.
    """
    return np.array(
        [
            np.cos(declination) * np.cos(right_ascension),
            np.cos(declination) * np.sin(right_ascension),
            np.sin(declination),
        ]
    )


def _rotate(
    vector: npt.NDArray[np.float64],
    axis: npt.NDArray[np.float64],
    angle: float,
) -> npt.NDArray[np.float64]:
    """Rotate a vector about an axis by an angle (Rodrigues).

    Args:
        vector: The vector to rotate.
        axis: Rotation axis; need not be normalised.
        angle: Rotation angle (radians).

    Returns:
        The rotated vector.
    """
    unit = axis / np.linalg.norm(axis)
    rotated = (
        vector * np.cos(angle)
        + np.cross(unit, vector) * np.sin(angle)
        + unit * float(np.dot(unit, vector)) * (1.0 - np.cos(angle))
    )
    return np.asarray(rotated, dtype=np.float64)


def _flyby_exit(
    incoming_excess: npt.NDArray[np.float64],
    perijove_radius: float,
    plane_angle: float,
) -> npt.NDArray[np.float64]:
    """Rotate the hyperbolic excess through an unpowered Jupiter flyby.

    The magnitude is preserved -- that is what "unpowered" means -- and the
    turn angle follows from the perijove radius through
    ``e = 1 + r_p v_inf^2 / mu`` and ``sin(delta/2) = 1/e``.  The ``plane_angle``
    selects which way the turn is taken, which is the B-plane freedom an
    approach aim point buys for nothing.

    Args:
        incoming_excess: Jupiter-relative arrival velocity (km/s).
        perijove_radius: Flyby perijove (km).
        plane_angle: Orientation of the turn about the incoming asymptote.

    Returns:
        The Jupiter-relative departure velocity (km/s).
    """
    speed = float(np.linalg.norm(incoming_excess))
    eccentricity = 1.0 + perijove_radius * speed**2 / _MU_JUPITER
    turn = 2.0 * np.arcsin(1.0 / eccentricity)
    direction = incoming_excess / speed
    # any vector not parallel to the asymptote gives a starting normal; the
    # plane angle then sweeps it around, covering every admissible B-plane.
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, direction))) > 0.95:
        seed = np.array([1.0, 0.0, 0.0])
    normal = np.cross(direction, seed)
    normal = _rotate(normal, direction, plane_angle)
    return _rotate(incoming_excess, normal, turn)


def _lambert(
    origin: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    duration_s: float,
    prograde: bool,
) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Zero-revolution Lambert arc, taking whichever branch converges.

    Args:
        origin: Heliocentric position left (km).
        target: Heliocentric position reached (km).
        duration_s: Time of flight (s).
        prograde: Whether the arc is prograde.

    Returns:
        ``(departure velocity, arrival velocity)`` in km/s, or None.
    """
    best: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    for low_path in (True, False):
        try:
            v1, v2 = izzo(
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
        except (AssertionError, RuntimeError, ValueError, ZeroDivisionError):
            continue
        candidate = (
            np.asarray(v1, dtype=np.float64),
            np.asarray(v2, dtype=np.float64),
        )
        if best is None or np.linalg.norm(candidate[0]) < np.linalg.norm(best[0]):
            best = candidate
    return best


def _safe_propagate(
    position: npt.NDArray[np.float64],
    velocity: npt.NDArray[np.float64],
    duration_s: float,
) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Kepler-propagate a heliocentric state, or report that it cannot be done.

    Farnocchia asserts rather than raising a typed error on the near-parabolic
    and out-of-range-anomaly states an optimiser will inevitably probe, and an
    ``AssertionError`` escaping a cost function kills the whole search.  Those
    states are simply infeasible trajectories, so they belong in the penalty,
    not in a traceback.

    Args:
        position: Heliocentric position (km).
        velocity: Heliocentric velocity (km/s).
        duration_s: Propagation time (s).

    Returns:
        The propagated state, or None if the propagation is not admissible.
    """
    try:
        propagated = _propagate_rv(_MU_SUN, position, velocity, duration_s)
    except (AssertionError, RuntimeError, ValueError, ZeroDivisionError):
        return None
    result = (
        np.asarray(propagated[0], dtype=np.float64),
        np.asarray(propagated[1], dtype=np.float64),
    )
    if not np.all(np.isfinite(result[0])) or not np.all(np.isfinite(result[1])):
        return None
    return result


def _fly_leg(
    position: npt.NDArray[np.float64],
    velocity: npt.NDArray[np.float64],
    target_position: npt.NDArray[np.float64],
    duration_s: float,
    fractions: npt.NDArray[np.float64],
    burns: npt.NDArray[np.float64],
    prograde: bool,
) -> Optional[Tuple[float, npt.NDArray[np.float64]]]:
    """Coast, burn, coast, ... then a Lambert arc onto the terminus.

    The last maneuver is not a free variable: it is whatever the closing
    Lambert arc demands, which is how the endpoint constraint is met exactly
    instead of being driven to zero by a penalty.  Every earlier maneuver is
    free in all three components.

    Args:
        position: Heliocentric position at the start of the leg (km).
        velocity: Heliocentric velocity at the start of the leg (km/s).
        target_position: Where the leg must end (km).
        duration_s: Total leg duration (s).
        fractions: Ascending maneuver epochs as fractions of the leg, one per
            maneuver; the last is where the closing arc begins.
        burns: The free maneuvers, one fewer than ``fractions``, shaped
            ``(n - 1, 3)`` in km/s.
        prograde: Whether the closing arc is prograde.

    Returns:
        ``(total maneuver delta-v in km/s, arrival velocity)``, or None if the
        closing arc will not solve.
    """
    elapsed = 0.0
    total = 0.0
    for fraction, burn in zip(fractions[:-1], burns):
        step = (fraction - elapsed) * duration_s
        if step > 0.0:
            propagated = _safe_propagate(position, velocity, step)
            if propagated is None:
                return None
            position, velocity = propagated
        velocity = velocity + burn
        total += float(np.linalg.norm(burn))
        elapsed = fraction
    step = (fractions[-1] - elapsed) * duration_s
    if step > 0.0:
        propagated = _safe_propagate(position, velocity, step)
        if propagated is None:
            return None
        position, velocity = propagated
    arc = _lambert(
        position, target_position, (1.0 - fractions[-1]) * duration_s, prograde
    )
    if arc is None:
        return None
    total += float(np.linalg.norm(arc[0] - velocity))
    return total, arc[1]


def _unpack_leg(
    raw: npt.NDArray[np.float64], maneuvers: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Split one leg's raw parameters into maneuver epochs and burns.

    The epochs are *sorted* rather than constrained to be ascending, so the
    optimiser sees a plain box instead of an ordering constraint it would spend
    its budget rediscovering.

    Args:
        raw: ``maneuvers`` epoch parameters followed by one
            ``(magnitude, right ascension, declination)`` triple per free
            maneuver.
        maneuvers: How many maneuvers this leg carries.

    Returns:
        ``(ascending fractions, Cartesian burns shaped (maneuvers - 1, 3))``.
    """
    fractions = np.sort(np.asarray(raw[:maneuvers], dtype=np.float64))
    polar = np.asarray(raw[maneuvers:], dtype=np.float64).reshape(maneuvers - 1, 3)
    burns = np.array(
        [
            magnitude * _unit_from_angles(right_ascension, declination)
            for magnitude, right_ascension, declination in polar
        ]
    ).reshape(maneuvers - 1, 3)
    return fractions, burns


def _leg_parameter_count(maneuvers: int) -> int:
    """How many parameters one leg contributes.

    Args:
        maneuvers: Maneuvers on the leg.

    Returns:
        ``maneuvers`` epochs plus three components for each free maneuver.
    """
    return maneuvers + 3 * (maneuvers - 1)


def _evaluate(
    parameters: Sequence[float] | npt.NDArray[np.float64],
    departure_jd: float,
    arrival_jd: float,
    encounter_centre_jd: float,
    outbound_maneuvers: int = 1,
    inbound_maneuvers: int = 1,
) -> Tuple[float, Optional[FreeDsmSolution]]:
    """Score one parameter vector for the growth wave's trajectory.

    Layout: departure excess (speed, right ascension, declination), encounter
    offset in days, the outbound leg's parameters, the flyby's perijove ratio
    and B-plane angle, then the return leg's parameters.

    Args:
        parameters: The packed vector.
        departure_jd: Fixed Earth departure epoch.
        arrival_jd: Fixed Earth arrival epoch.
        encounter_centre_jd: Epoch the encounter window is centred on.
        outbound_maneuvers: Maneuvers allowed on the Earth-to-Jupiter leg.
        inbound_maneuvers: Maneuvers allowed on the Jupiter-to-Earth leg.

    Returns:
        ``(cost in km/s, solution or None)``.  Infeasible vectors return a
        large penalised cost so the global search sees a slope, not a cliff.

    Note:
        The coasts use Farnocchia rather than Markley because a departure
        excess of 14 km/s on top of Earth's 29.8 puts the heliocentric arc
        above the 42.1 km/s solar escape speed at 1 AU -- the outbound leg can
        be hyperbolic, and an elliptic-only Kepler solver returns NaN there.
    """
    vector = np.asarray(parameters, dtype=np.float64)
    outbound_width = _leg_parameter_count(outbound_maneuvers)
    vinf, right_ascension, declination, encounter_offset = vector[:4]
    outbound_raw = vector[4 : 4 + outbound_width]
    perijove_ratio, plane_angle = vector[4 + outbound_width : 6 + outbound_width]
    inbound_raw = vector[6 + outbound_width :]

    encounter_jd = encounter_centre_jd + encounter_offset
    outbound_days = encounter_jd - departure_jd
    return_days = arrival_jd - encounter_jd
    if outbound_days < 200.0 or return_days < 200.0:
        return INFEASIBLE_PENALTY, None

    earth_position, earth_velocity = _heliocentric_state("earth", departure_jd)
    jupiter_position, jupiter_velocity = _heliocentric_state("jupiter", encounter_jd)
    arrival_position, _ = _heliocentric_state("earth", arrival_jd)

    departure_velocity = earth_velocity + vinf * _unit_from_angles(
        right_ascension, declination
    )
    fractions, burns = _unpack_leg(outbound_raw, outbound_maneuvers)
    outbound = _fly_leg(
        earth_position,
        departure_velocity,
        jupiter_position,
        outbound_days * _SECONDS_PER_DAY,
        fractions,
        burns,
        True,
    )
    if outbound is None:
        return INFEASIBLE_PENALTY, None
    first_dsm, jupiter_arrival = outbound
    incoming_excess = jupiter_arrival - jupiter_velocity

    perijove_radius = perijove_ratio * _PERIJOVE_FLOOR
    outgoing_excess = _flyby_exit(incoming_excess, perijove_radius, plane_angle)

    fractions, burns = _unpack_leg(inbound_raw, inbound_maneuvers)
    inbound = _fly_leg(
        jupiter_position,
        jupiter_velocity + outgoing_excess,
        arrival_position,
        return_days * _SECONDS_PER_DAY,
        fractions,
        burns,
        False,
    )
    if inbound is None:
        return INFEASIBLE_PENALTY, None
    second_dsm, _ = inbound

    total = first_dsm + second_dsm
    return total, FreeDsmSolution(
        departure_jd=departure_jd,
        arrival_jd=arrival_jd,
        encounter_jd=encounter_jd,
        departure_vinf=float(vinf),
        first_dsm=first_dsm,
        second_dsm=second_dsm,
        perijove_radius=perijove_radius,
        total_dv=total,
        seam_dv=float("nan"),
        feasible=True,
    )


def _bounds_for(
    outbound_maneuvers: int, inbound_maneuvers: int
) -> List[Tuple[float, float]]:
    """Search box for a given maneuver budget.

    Args:
        outbound_maneuvers: Maneuvers on the Earth-to-Jupiter leg.
        inbound_maneuvers: Maneuvers on the Jupiter-to-Earth leg.

    Returns:
        One ``(low, high)`` pair per parameter, in packed order.
    """
    polar = [
        (0.0, MANEUVER_MAGNITUDE_BOUND),
        (0.0, 2.0 * np.pi),
        (-0.5 * np.pi, 0.5 * np.pi),
    ]

    def leg(maneuvers: int) -> List[Tuple[float, float]]:
        return [DSM_FRACTION_BOUNDS] * maneuvers + polar * (maneuvers - 1)

    return (
        [
            DEPARTURE_VINF_BOUNDS,
            (0.0, 2.0 * np.pi),
            (-0.5 * np.pi, 0.5 * np.pi),
            (-ENCOUNTER_WINDOW_DAYS, ENCOUNTER_WINDOW_DAYS),
        ]
        + leg(outbound_maneuvers)
        + [PERIJOVE_RATIO_BOUNDS, (0.0, 2.0 * np.pi)]
        + leg(inbound_maneuvers)
    )


def _seam_guess(
    departure_jd: float,
    arrival_jd: float,
    encounter_jd: float,
    perijove_radius: float,
    outbound_maneuvers: int = 1,
    inbound_maneuvers: int = 1,
    plane_angles: int = 24,
) -> Optional[npt.NDArray[np.float64]]:
    """Parameter vector reproducing the seam solution's own trajectory.

    Without this the global search can return something *worse* than the seam
    proxy, which would bound nothing.  Seeding with the trajectory ADR 0026
    already charges guarantees the answer is at least as good.

    The seam takes its whole correction instantaneously at Jupiter, so the
    equivalent here is a direct Lambert outbound with no maneuver and the
    return maneuver as early as the parameterisation allows.  With more than
    one maneuver per leg the extras are seeded to zero at the same epoch, which
    reproduces the single-maneuver trajectory exactly -- so a richer budget can
    only improve on it, never do worse.  The B-plane angle has no counterpart
    in the seam formulation, so it is swept.

    Args:
        departure_jd: Fixed Earth departure epoch.
        arrival_jd: Fixed Earth arrival epoch.
        encounter_jd: The seam solution's Jupiter encounter epoch.
        perijove_radius: The seam solution's perijove (km).
        outbound_maneuvers: Maneuvers on the Earth-to-Jupiter leg.
        inbound_maneuvers: Maneuvers on the Jupiter-to-Earth leg.
        plane_angles: How many B-plane orientations to try.

    Returns:
        The best seeding vector found, or None if the direct arc will not
        solve.
    """
    earth_position, earth_velocity = _heliocentric_state("earth", departure_jd)
    jupiter_position, _ = _heliocentric_state("jupiter", encounter_jd)
    outbound_s = (encounter_jd - departure_jd) * _SECONDS_PER_DAY
    direct = _lambert(earth_position, jupiter_position, outbound_s, True)
    if direct is None:
        return None
    excess = direct[0] - earth_velocity
    magnitude = float(np.linalg.norm(excess))
    speed = float(np.clip(magnitude, *DEPARTURE_VINF_BOUNDS))
    right_ascension = float(np.arctan2(excess[1], excess[0])) % (2.0 * np.pi)
    declination = float(np.arcsin(excess[2] / magnitude))
    ratio = float(np.clip(perijove_radius / _PERIJOVE_FLOOR, *PERIJOVE_RATIO_BOUNDS))
    floor = DSM_FRACTION_BOUNDS[0]

    def leg(maneuvers: int) -> List[float]:
        return [floor] * maneuvers + [0.0] * (3 * (maneuvers - 1))

    best: Optional[npt.NDArray[np.float64]] = None
    best_cost = np.inf
    for index in range(plane_angles):
        candidate = np.array(
            [speed, right_ascension, declination, 0.0]
            + leg(outbound_maneuvers)
            + [ratio, 2.0 * np.pi * index / plane_angles]
            + leg(inbound_maneuvers)
        )
        candidate_cost, _ = _evaluate(
            candidate,
            departure_jd,
            arrival_jd,
            encounter_jd,
            outbound_maneuvers,
            inbound_maneuvers,
        )
        if candidate_cost < best_cost:
            best_cost, best = candidate_cost, candidate
    return best


def _to_unit(
    vector: npt.NDArray[np.float64], bounds: Sequence[Tuple[float, float]]
) -> npt.NDArray[np.float64]:
    """Map a parameter vector into the unit box.

    Args:
        vector: Parameters in their natural units.
        bounds: One ``(low, high)`` pair per parameter.

    Returns:
        The same point with every coordinate in [0, 1].
    """
    low = np.array([bound[0] for bound in bounds])
    high = np.array([bound[1] for bound in bounds])
    return np.asarray(
        np.clip((vector - low) / (high - low), 0.0, 1.0), dtype=np.float64
    )


def _from_unit(
    unit: npt.NDArray[np.float64], bounds: Sequence[Tuple[float, float]]
) -> npt.NDArray[np.float64]:
    """Map a point in the unit box back to natural units.

    Args:
        unit: Coordinates in [0, 1].
        bounds: One ``(low, high)`` pair per parameter.

    Returns:
        The parameter vector.
    """
    low = np.array([bound[0] for bound in bounds])
    high = np.array([bound[1] for bound in bounds])
    return np.asarray(low + unit * (high - low), dtype=np.float64)


def _basin_hop(
    cost: "Callable[[npt.NDArray[np.float64]], float]",
    start: npt.NDArray[np.float64],
    dimension: int,
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Monotonic basin hopping on the unit box.

    Perturb the incumbent, solve locally, keep the result only if it improved.
    The perturbation is Cauchy rather than Gaussian: its heavy tail produces
    the occasional long jump that escapes a basin, which is the whole reason
    the method works on trajectory problems where the basins are narrow and
    numerous.

    Args:
        cost: Objective on the unit box.
        start: Incumbent to begin from, in unit coordinates.
        dimension: Number of parameters.
        iterations: How many hops to attempt.
        rng: Random source, seeded by the caller for reproducibility.

    Returns:
        ``(best cost, best point in unit coordinates)``.
    """
    unit_bounds = [(0.0, 1.0)] * dimension

    def refine(point: npt.NDArray[np.float64]) -> Tuple[float, npt.NDArray[np.float64]]:
        result = minimize(
            cost,
            point,
            method=LOCAL_METHOD,
            bounds=unit_bounds,
            options={"maxiter": LOCAL_MAXITER, "eps": LOCAL_FD_STEP},
        )
        return float(result.fun), np.clip(np.asarray(result.x), 0.0, 1.0)

    best_cost, best_point = refine(start)
    if cost(start) < best_cost:
        best_cost, best_point = cost(start), start
    for _ in range(iterations):
        step = MBH_PERTURBATION * rng.standard_cauchy(dimension)
        step *= rng.random(dimension) < MBH_PERTURB_PROBABILITY
        candidate = np.clip(best_point + step, 0.0, 1.0)
        candidate_cost, candidate_point = refine(candidate)
        if candidate_cost < best_cost:
            best_cost, best_point = candidate_cost, candidate_point
    return best_cost, best_point


def _offset_seeds(
    guess: npt.NDArray[np.float64],
    outbound_maneuvers: int,
    inbound_maneuvers: int,
    count: int,
    rng: np.random.Generator,
) -> List[npt.NDArray[np.float64]]:
    """Variants of the seed with maneuvers nudged off zero and spread in time.

    See :data:`OFFSET_SEEDS`.  The zero-magnitude seed is a non-differentiable
    point of the objective, so a gradient method cannot leave it; these give
    the search somewhere differentiable to start from.  The exact seed is still
    evaluated separately, so the bound can never come back worse.

    Args:
        guess: The seam-derived seed vector.
        outbound_maneuvers: Maneuvers on the Earth-to-Jupiter leg.
        inbound_maneuvers: Maneuvers on the Jupiter-to-Earth leg.
        count: How many variants to build.
        rng: Random source.

    Returns:
        The variants, which may be empty when neither leg has a free maneuver.
    """
    outbound_width = _leg_parameter_count(outbound_maneuvers)
    if outbound_maneuvers < 2 and inbound_maneuvers < 2:
        return []
    seeds: List[npt.NDArray[np.float64]] = []
    low, high = OFFSET_SEED_RANGE
    for _ in range(count):
        candidate = guess.copy()
        for offset, maneuvers in (
            (4, outbound_maneuvers),
            (6 + outbound_width, inbound_maneuvers),
        ):
            # spread the maneuver epochs across the leg instead of stacking
            # them all at its start, where they are interchangeable and the
            # sort that orders them makes every one of them a flat direction
            candidate[offset : offset + maneuvers] = np.sort(
                rng.uniform(DSM_FRACTION_BOUNDS[0], 0.9, maneuvers)
            )
            burns = candidate[
                offset + maneuvers : offset + _leg_parameter_count(maneuvers)
            ]
            if burns.size:
                triples = np.empty((burns.size // 3, 3))
                triples[:, 0] = rng.uniform(low, high, triples.shape[0])
                triples[:, 1] = rng.uniform(0.0, 2.0 * np.pi, triples.shape[0])
                triples[:, 2] = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, triples.shape[0])
                candidate[
                    offset + maneuvers : offset + _leg_parameter_count(maneuvers)
                ] = triples.ravel()
        seeds.append(candidate)
    return seeds


def _structured_seeds(
    guess: npt.NDArray[np.float64],
    departure_jd: float,
    arrival_jd: float,
    encounter_centre_jd: float,
    outbound_maneuvers: int,
    inbound_maneuvers: int,
) -> List[npt.NDArray[np.float64]]:
    """Seeds that keep the seam's big burn but free the arc after it.

    The closing Lambert arc is the *last* maneuver on a leg, so every other
    maneuver must precede it.  Reproducing the seam puts that closing arc at
    the very start of the return leg -- which squeezes every additional
    maneuver into the first few hours of a 440-day leg, where it can do
    nothing.  That, not the seeding magnitude, is why a richer maneuver budget
    scored no better than a poorer one.

    These seeds restructure it: the seam's own correction becomes an explicit
    free burn just after the flyby, and the closing arc moves to a range of
    later points, which is the arrangement that lets a second maneuver
    actually trade against the first.

    Args:
        guess: The seam-derived seed vector.
        departure_jd: Fixed Earth departure epoch.
        arrival_jd: Fixed Earth arrival epoch.
        encounter_centre_jd: Epoch the encounter window is centred on.
        outbound_maneuvers: Maneuvers on the Earth-to-Jupiter leg.
        inbound_maneuvers: Maneuvers on the Jupiter-to-Earth leg.

    Returns:
        One seed per closing-arc placement; empty when the return leg carries
        only one maneuver and there is nothing to restructure.
    """
    if inbound_maneuvers < 2:
        return []
    outbound_width = _leg_parameter_count(outbound_maneuvers)
    vinf, right_ascension, declination, encounter_offset = guess[:4]
    perijove_ratio, plane_angle = guess[4 + outbound_width : 6 + outbound_width]
    encounter_jd = encounter_centre_jd + encounter_offset

    earth_position, earth_velocity = _heliocentric_state("earth", departure_jd)
    jupiter_position, jupiter_velocity = _heliocentric_state("jupiter", encounter_jd)
    arrival_position, _ = _heliocentric_state("earth", arrival_jd)

    departure_velocity = earth_velocity + vinf * _unit_from_angles(
        right_ascension, declination
    )
    outbound = _lambert(
        earth_position,
        jupiter_position,
        (encounter_jd - departure_jd) * _SECONDS_PER_DAY,
        True,
    )
    if outbound is None:
        return []
    del departure_velocity
    incoming_excess = outbound[1] - jupiter_velocity
    outgoing_excess = _flyby_exit(
        incoming_excess, perijove_ratio * _PERIJOVE_FLOOR, plane_angle
    )
    return_s = (arrival_jd - encounter_jd) * _SECONDS_PER_DAY
    closing = _lambert(jupiter_position, arrival_position, return_s, False)
    if closing is None:
        return []
    correction = closing[0] - (jupiter_velocity + outgoing_excess)

    seeds: List[npt.NDArray[np.float64]] = []
    floor = DSM_FRACTION_BOUNDS[0]
    for placement in (0.05, 0.15, 0.30, 0.50, 0.70):
        candidate = guess.copy()
        offset = 6 + outbound_width
        fractions = [floor] * (inbound_maneuvers - 1) + [placement]
        candidate[offset : offset + inbound_maneuvers] = fractions
        magnitude = float(np.linalg.norm(correction))
        burns = np.zeros((inbound_maneuvers - 1, 3))
        burns[0] = [
            magnitude,
            float(np.arctan2(correction[1], correction[0])) % (2.0 * np.pi),
            float(np.arcsin(correction[2] / magnitude)),
        ]
        candidate[
            offset
            + inbound_maneuvers : offset
            + _leg_parameter_count(inbound_maneuvers)
        ] = burns.ravel()
        seeds.append(candidate)
    return seeds


def solve_free_dsm(
    departure_jd: float,
    arrival_jd: float,
    encounter_centre_jd: float,
    seam_dv: float,
    seam_perijove_radius: float = _PERIJOVE_FLOOR,
    outbound_maneuvers: int = 1,
    inbound_maneuvers: int = 1,
    basin_hops: int = MBH_ITERATIONS,
    random_starts: int = 4,
    use_differential_evolution: bool = False,
    seed: int = DE_SEED,
    maxiter: int = DE_MAXITER,
) -> FreeDsmSolution:
    """Minimise total maneuver delta-v with every burn free in place and time.

    Monotonic basin hopping on the unit box: a bounded L-BFGS-B solve from the
    seam-derived seed, then repeated Cauchy-perturbed restarts keeping only
    improvements, plus independent random starts.  See :data:`MBH_ITERATIONS`
    for why this rather than differential evolution.

    Args:
        departure_jd: Fixed Earth departure epoch (TDB Julian date).
        arrival_jd: Fixed Earth arrival epoch.
        encounter_centre_jd: Epoch the encounter search window is centred on,
            normally the seam solution's own encounter.
        seam_dv: What ADR 0026 charges for this cycle (km/s), carried through
            for comparison.
        seam_perijove_radius: The seam solution's own perijove (km), used to
            seed the search so the answer cannot come back worse than the
            trajectory it is meant to bound.
        outbound_maneuvers: Maneuvers allowed on the Earth-to-Jupiter leg.
        inbound_maneuvers: Maneuvers allowed on the Jupiter-to-Earth leg.
        basin_hops: Basin-hopping iterations from the seeded incumbent.
        random_starts: Independent random incumbents to hop from as well.
        use_differential_evolution: Also run a DE pass as a cross-check.  Off
            by default; it costs more than the hops and finds less.
        seed: Random seed, recorded for reproducibility.
        maxiter: Differential-evolution iteration cap, when it is enabled.

    Returns:
        The cheapest trajectory found.  ``feasible`` is False if the search
        never found one at all.
    """
    bounds = _bounds_for(outbound_maneuvers, inbound_maneuvers)
    dimension = len(bounds)

    def cost(parameters: npt.NDArray[np.float64]) -> float:
        return _evaluate(
            parameters,
            departure_jd,
            arrival_jd,
            encounter_centre_jd,
            outbound_maneuvers,
            inbound_maneuvers,
        )[0]

    def unit_cost(unit: npt.NDArray[np.float64]) -> float:
        return cost(_from_unit(unit, bounds))

    rng = np.random.default_rng(seed)
    candidates: List[Tuple[float, npt.NDArray[np.float64]]] = []

    guess = _seam_guess(
        departure_jd,
        arrival_jd,
        encounter_centre_jd,
        seam_perijove_radius,
        outbound_maneuvers,
        inbound_maneuvers,
    )
    if guess is not None:
        candidates.append((cost(guess), guess))
        hop_cost, hop_unit = _basin_hop(
            unit_cost, _to_unit(guess, bounds), dimension, basin_hops, rng
        )
        candidates.append((hop_cost, _from_unit(hop_unit, bounds)))
        restructured = _structured_seeds(
            guess,
            departure_jd,
            arrival_jd,
            encounter_centre_jd,
            outbound_maneuvers,
            inbound_maneuvers,
        )
        for nudged in restructured + _offset_seeds(
            guess, outbound_maneuvers, inbound_maneuvers, OFFSET_SEEDS, rng
        ):
            hop_cost, hop_unit = _basin_hop(
                unit_cost,
                _to_unit(nudged, bounds),
                dimension,
                max(1, basin_hops // 3),
                rng,
            )
            candidates.append((hop_cost, _from_unit(hop_unit, bounds)))

    # Independent starts, so the answer does not rest on the seam seed alone.
    # Each gets a shorter hop budget than the seeded run, which is where the
    # good basin is already known to be.
    for _ in range(random_starts):
        hop_cost, hop_unit = _basin_hop(
            unit_cost,
            rng.random(dimension),
            dimension,
            max(1, basin_hops // 4),
            rng,
        )
        candidates.append((hop_cost, _from_unit(hop_unit, bounds)))

    if use_differential_evolution:
        result = differential_evolution(
            cost,
            bounds,
            popsize=DE_POPSIZE,
            maxiter=maxiter,
            tol=DE_TOLERANCE,
            seed=seed,
            polish=False,
            init="sobol",
            x0=guess,
        )
        candidates.append((float(result.fun), np.asarray(result.x)))

    _, best_x = min(candidates, key=lambda item: item[0])
    _, solution = _evaluate(
        best_x,
        departure_jd,
        arrival_jd,
        encounter_centre_jd,
        outbound_maneuvers,
        inbound_maneuvers,
    )
    if solution is None:
        return FreeDsmSolution(
            departure_jd=departure_jd,
            arrival_jd=arrival_jd,
            encounter_jd=encounter_centre_jd,
            departure_vinf=float("nan"),
            first_dsm=float("nan"),
            second_dsm=float("nan"),
            perijove_radius=float("nan"),
            total_dv=float("inf"),
            seam_dv=seam_dv,
            feasible=False,
        )
    return FreeDsmSolution(
        departure_jd=solution.departure_jd,
        arrival_jd=solution.arrival_jd,
        encounter_jd=solution.encounter_jd,
        departure_vinf=solution.departure_vinf,
        first_dsm=solution.first_dsm,
        second_dsm=solution.second_dsm,
        perijove_radius=solution.perijove_radius,
        total_dv=solution.total_dv,
        seam_dv=seam_dv,
        feasible=True,
    )


@dataclass(frozen=True)
class ExpensiveCycle:
    """A cycle whose split the seam proxy prices in kilometres per second."""

    departure_jd: float
    arrival_jd: float
    encounter_jd: float
    synodic_multiple: int
    seam_dv: float
    seam_perijove_radius: float
    incoming_vinf: float
    required_turn: float


def expensive_cycles(
    start: str = DEFAULT_START,
    years: float = DEFAULT_HORIZON_YEARS,
    threshold_m_s: float = DEFAULT_THRESHOLD_M_S,
    split_days: float = DEFAULT_SPLIT_DAYS,
    minimum_dv: float = DEFAULT_MINIMUM_DV,
) -> List[ExpensiveCycle]:
    """Fly ADR 0013's cadence and return the cycles whose split is dear.

    Only these matter: the other eight already cost under 2 m/s at the 20-day
    gap and under 0.3 at the 10-day one, which any plausible propulsion buys,
    so nothing about them is in question.

    Args:
        start: First date a departure may occur (ISO date).
        years: Length of the study horizon (Julian years).
        threshold_m_s: Proxy above which a cycle falls back to three synodic.
        split_days: Gap by which the growth wave leads the nozzle wave.
        minimum_dv: Seam cost (km/s) above which a cycle counts as expensive;
            see :data:`DEFAULT_MINIMUM_DV` for why it is the array's own budget.

    Returns:
        The expensive cycles, in order.
    """
    start_epoch = Time(start, scale="tdb")
    departure_jd = float(_resonance_epochs(start_epoch, 5.0)[0])
    horizon_end_jd = float(start_epoch.tdb.jd) + years * _DAYS_PER_YEAR
    one_synodic_days = _FIXED_TWO_SYNODIC_DAYS / 2.0
    found: List[ExpensiveCycle] = []
    while departure_jd + 2.0 * one_synodic_days <= horizon_end_jd:
        two_return_jd = departure_jd + 2.0 * one_synodic_days
        nozzle = _window_maneuver_solutions(
            departure_jd, two_return_jd, None, cases=("dsm_only",)
        )[0]
        multiple, return_jd = 2, two_return_jd
        if 1000.0 * nozzle.dsm > threshold_m_s:
            multiple = 3
            return_jd = departure_jd + 3.0 * one_synodic_days
            if return_jd > horizon_end_jd:
                break
        arrival_jd = return_jd - split_days
        growth = _window_maneuver_solutions(
            departure_jd, arrival_jd, None, cases=("dsm_only",)
        )[0]
        if growth.total_dv >= minimum_dv:
            found.append(
                ExpensiveCycle(
                    departure_jd=departure_jd,
                    arrival_jd=arrival_jd,
                    encounter_jd=growth.encounter_jd,
                    synodic_multiple=multiple,
                    seam_dv=growth.total_dv,
                    seam_perijove_radius=growth.perijove_radius,
                    incoming_vinf=growth.incoming_vinf,
                    required_turn=growth.turn_angle,
                )
            )
        departure_jd = return_jd
    return found


def bend_deficit_degrees(
    incoming_vinf: float,
    required_turn: float,
    perijove_radius: float = _PERIJOVE_FLOOR,
) -> Tuple[float, float]:
    """How far short of the required turn the flyby falls, in degrees.

    ``e = 1 + r_p v_inf^2 / mu`` and ``delta = 2 arcsin(1/e)``, so turn
    authority falls as the wave arrives faster.  This is the quantity that
    actually decides these cycles: the correction is paying off the shortfall,
    at roughly ``2 v_inf sin(deficit/2)``.

    Args:
        incoming_vinf: Jupiter-relative arrival speed (km/s).
        required_turn: Turn the trajectory demands (radians).
        perijove_radius: Flyby perijove (km); the floor by default, which is
            the most turn Jupiter can give.

    Returns:
        ``(available turn, deficit)`` in degrees; the deficit is negative when
        the flyby has authority to spare.
    """
    eccentricity = 1.0 + perijove_radius * incoming_vinf**2 / _MU_JUPITER
    available = float(np.degrees(2.0 * np.arcsin(1.0 / eccentricity)))
    return available, float(np.degrees(required_turn)) - available


def blind_sample_floor(
    cycle: "ExpensiveCycle",
    samples: int = BLIND_SAMPLES,
    maneuvers_per_leg: int = DEFAULT_MANEUVERS_PER_LEG,
    seed: int = DE_SEED,
) -> Tuple[int, float]:
    """Cheapest trajectory found by sampling the box uniformly at random.

    ADR 0008's rule is that an optimiser table means nothing until the box has
    been sampled blind, because "no solution" and "no search" look identical.
    The same check answers a different question here: the optimiser is seeded
    with the very trajectory it is meant to bound, so the risk is not that it
    fails to find anything but that it never leaves the basin it started in.
    If blind sampling turned up trajectories anywhere near the seam, that basin
    would be one of many and the search would have to be widened.  It does not
    -- see ADR 0028.

    Args:
        cycle: The cycle to sample around.
        samples: How many points to draw.
        maneuvers_per_leg: Maneuver budget, which sets the box's dimension.
        seed: Random seed, recorded for reproducibility.

    Returns:
        ``(how many samples flew a complete trajectory, the cheapest of them in
        km/s)``.  The cost is infinite when none did.
    """
    bounds = _bounds_for(maneuvers_per_leg, maneuvers_per_leg)
    low = np.array([bound[0] for bound in bounds])
    high = np.array([bound[1] for bound in bounds])
    rng = np.random.default_rng(seed)
    feasible = 0
    best = float("inf")
    for _ in range(samples):
        cost, _solution = _evaluate(
            low + rng.random(len(bounds)) * (high - low),
            cycle.departure_jd,
            cycle.arrival_jd,
            cycle.encounter_jd,
            maneuvers_per_leg,
            maneuvers_per_leg,
        )
        if cost < INFEASIBLE_PENALTY:
            feasible += 1
            best = min(best, cost)
    return feasible, best


def bound_the_expensive_cycles(
    deliverable_m_s: float = 254.5,
    maneuvers_per_leg: int = DEFAULT_MANEUVERS_PER_LEG,
    basin_hops: int = MBH_ITERATIONS,
    random_starts: int = 4,
    blind_samples: int = BLIND_SAMPLES,
    **kwargs: object,
) -> pd.DataFrame:
    """Run the free-burn bound on every expensive cycle and score the verdict.

    Args:
        deliverable_m_s: What the array can actually deliver over a whole
            trajectory at the design's stated acceleration (m/s); ADR 0026
            measures 254.5 for the adaptive cadence's worst cycle.
        maneuvers_per_leg: Maneuver budget on each leg.
        basin_hops: Basin-hopping iterations per cycle.  Exposed so a test can
            buy the table at a fraction of the CLI's cost; the published
            figures are the defaults.
        random_starts: Independent random incumbents per cycle, likewise.
        blind_samples: Uniform samples of the box per cycle, reported alongside
            the optimum so the table carries its own harness check.
        **kwargs: Passed to :func:`expensive_cycles`.

    Returns:
        One row per expensive cycle: the seam price, the free-burn bound, how
        much the bound improves on it, and whether the bound is still out of
        reach of the array.
    """
    rows = []
    for cycle in expensive_cycles(**kwargs):  # type: ignore[arg-type]
        solution = solve_free_dsm(
            cycle.departure_jd,
            cycle.arrival_jd,
            cycle.encounter_jd,
            cycle.seam_dv,
            seam_perijove_radius=cycle.seam_perijove_radius,
            outbound_maneuvers=maneuvers_per_leg,
            inbound_maneuvers=maneuvers_per_leg,
            basin_hops=basin_hops,
            random_starts=random_starts,
        )
        available, deficit = bend_deficit_degrees(
            cycle.incoming_vinf, cycle.required_turn
        )
        flew, blind_best = blind_sample_floor(
            cycle, samples=blind_samples, maneuvers_per_leg=maneuvers_per_leg
        )
        rows.append(
            {
                "departure": Time(cycle.departure_jd, format="jd", scale="tdb").iso[
                    :10
                ],
                "synodic_multiple": cycle.synodic_multiple,
                "seam_m_s": 1000.0 * cycle.seam_dv,
                "free_burn_m_s": 1000.0 * solution.total_dv,
                "improvement": solution.improvement,
                "first_dsm_m_s": 1000.0 * solution.first_dsm,
                "second_dsm_m_s": 1000.0 * solution.second_dsm,
                "departure_vinf": solution.departure_vinf,
                "perijove_r_j": solution.perijove_radius / 71_492.0,
                "encounter_shift_days": solution.encounter_jd - cycle.encounter_jd,
                "incoming_vinf": cycle.incoming_vinf,
                "turn_needed_deg": float(np.degrees(cycle.required_turn)),
                "turn_available_deg": available,
                "bend_deficit_deg": deficit,
                "blind_feasible": flew,
                "blind_best_km_s": blind_best,
                "deliverable_m_s": deliverable_m_s,
                "still_out_of_reach": 1000.0 * solution.total_dv > deliverable_m_s,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Print the free-burn lower bound against the seam proxy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--years", type=float, default=DEFAULT_HORIZON_YEARS)
    parser.add_argument("--split-days", type=float, default=DEFAULT_SPLIT_DAYS)
    parser.add_argument("--minimum-dv", type=float, default=DEFAULT_MINIMUM_DV)
    parser.add_argument("--deliverable-m-s", type=float, default=254.5)
    args = parser.parse_args()

    frame = bound_the_expensive_cycles(
        deliverable_m_s=args.deliverable_m_s,
        start=args.start,
        years=args.years,
        split_days=args.split_days,
        minimum_dv=args.minimum_dv,
    )
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(
        "Free-burn impulsive optimum: a LOWER BOUND on any finite-thrust\n"
        "solution between the same endpoints, because the impulsive problem is\n"
        "the finite-thrust problem with the thrust bound deleted.\n"
    )
    print(frame.to_string(index=False))
    if not frame.empty:
        print(
            f"\nverdict: {int(frame.still_out_of_reach.sum())} of {len(frame)} "
            f"expensive cycles remain beyond the array even with the burn free "
            f"to go anywhere."
        )


if __name__ == "__main__":
    main()
