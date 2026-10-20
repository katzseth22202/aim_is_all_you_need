"""The left push plate's disposal nudge, propagated with the Moon (the parent's S43 item 5).

    python -m src.plate_disposal

With two springs (``solar_dive_growth.Design.shared_spring = False``) the
push plate is left in the 10-day parking orbit and nudged at its apoapsis so
that its next perigee is 100 km, where it re-enters.  ADR 0074 priced that
nudge two-body: 4.2 m/s from a 400 km perigee.  The orbit's apoapsis is
385 000 km, the Moon's distance, where the Moon pulls about as hard as the
Earth when it is near.  So the next perigee depends on where the Moon is.

Here the plate flies from its apoapsis to its next perigee under the Earth,
the Moon on a circular orbit (384 400 km, 27.32 days) inclined to the
parking orbit, and the Sun's tide (1 AU, coplanar with the Moon).  The Moon's
phase at the apoapsis is swept; at each, the nudge along the velocity that
brings the next perigee to 100 km is solved.  The plate is not followed past
that perigee: at 100 km it re-enters.
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from tabulate import tabulate

_MU_EARTH = 398600.4418  # km^3/s^2
_MU_MOON = 4902.800
_MU_SUN = 1.32712440018e11
_R_EARTH = 6378.137  # km
_R_MOON = 1737.4  # km
_MOON_DISTANCE = 384400.0  # km
_MOON_PERIOD = 27.321661 * 86400.0  # s
_AU = 1.495978707e8  # km
_YEAR = 365.25 * 86400.0  # s
_DAY = 86400.0

PUSH_PERIGEE = 400.0  # km, the push's altitude
TARGET_PERIGEE = 100.0  # km
#: Lunar phases at the apoapsis (deg from the apoapsis direction) and the
#: Moon's inclination to the parking orbit: coplanar and the most the Moon's
#: orbit tilts against Earth's equator (28.6 deg).
PHASES = tuple(float(p) for p in range(0, 360, 15))
INCLINATIONS = (0.0, 28.6)
#: The Sun's direction at the apoapsis (deg): its tide is ~0.6% of Earth's
#: pull there, so two directions bracket it.
SUN_PHASES = (0.0, 90.0)
#: The nudge searched along and against the velocity (km/s).
NUDGE_BOUND = 0.25
#: A Moon this close in phase to the apoapsis meets the parking orbit itself
#: (under ~100 000 km at 15 deg): the ship's orbit too, so it is a constraint
#: on the departure's timing, not the plate's disposal.
ENCOUNTER_PHASE = 20.0  # deg


def _orbit_a(days: float) -> float:
    return float((_MU_EARTH * (days * _DAY / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0))


def two_body_nudge(
    days: float, perigee: float = PUSH_PERIGEE, target: float = TARGET_PERIGEE
) -> float:
    """The apoapsis nudge to ``target`` perigee with no Moon (km/s), ADR 0074's."""
    a = _orbit_a(days)
    low = _R_EARTH + perigee
    apo = 2.0 * a - low
    before = math.sqrt(_MU_EARTH * (2.0 / apo - 1.0 / a))
    after = math.sqrt(_MU_EARTH * (2.0 / apo - 2.0 / (apo + _R_EARTH + target)))
    return before - after


@dataclass(frozen=True)
class Pass:
    """The plate from its apoapsis to its next perigee.

    Attributes:
        perigee: The next perigee's altitude (km); 0 strikes Earth, NaN the Moon.
        moon_closest: The plate's closest approach to the Moon's centre (km);
            0 if it strikes the Moon.
        days: Time from the apoapsis to the perigee.
    """

    perigee: float
    moon_closest: float
    days: float


def _moon(t: float, phase: float, inclination: float) -> np.ndarray:
    angle = phase + 2.0 * math.pi * t / _MOON_PERIOD
    return _MOON_DISTANCE * np.array(
        [
            math.cos(angle),
            math.sin(angle) * math.cos(inclination),
            math.sin(angle) * math.sin(inclination),
        ]
    )


def _sun(t: float, phase: float) -> np.ndarray:
    angle = phase + 2.0 * math.pi * t / _YEAR
    return _AU * np.array([math.cos(angle), math.sin(angle), 0.0])


def next_perigee(
    days: float,
    nudge: float,
    moon_phase_deg: float,
    inclination_deg: float,
    sun_phase_deg: float = 0.0,
    perigee: float = PUSH_PERIGEE,
) -> Pass:
    """Fly the plate from the apoapsis, nudged ``nudge`` km/s against its motion.

    The parking orbit is two-body at the apoapsis (its apoapsis on +x, moving
    along +y); the Moon's phase is measured from +x in its own plane, tilted
    about +x.

    Args:
        days: The parking orbit's period.
        nudge: The retrograde nudge (km/s; negative is prograde).
        moon_phase_deg: The Moon's angle from the apoapsis direction.
        inclination_deg: The Moon's orbit against the parking orbit.
        sun_phase_deg: The Sun's direction.
        perigee: The parking orbit's perigee altitude (km).

    Returns:
        The pass.
    """
    a = _orbit_a(days)
    apo = 2.0 * a - (_R_EARTH + perigee)
    speed = math.sqrt(_MU_EARTH * (2.0 / apo - 1.0 / a)) - nudge
    phase = math.radians(moon_phase_deg)
    tilt = math.radians(inclination_deg)
    sun_phase = math.radians(sun_phase_deg)

    def rates(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        moon = _moon(t, phase, tilt)
        sun = _sun(t, sun_phase)
        to_moon = r - moon
        to_sun = r - sun
        acc = (
            -_MU_EARTH * r / np.linalg.norm(r) ** 3
            - _MU_MOON
            * (to_moon / np.linalg.norm(to_moon) ** 3 + moon / _MOON_DISTANCE**3)
            - _MU_SUN * (to_sun / np.linalg.norm(to_sun) ** 3 + sun / _AU**3)
        )
        return np.concatenate([y[3:], acc])

    def at_perigee(t: float, y: np.ndarray) -> float:
        # Only inside half the apoapsis, so a lunar swing-by is not a perigee.
        if np.linalg.norm(y[:3]) > 0.5 * apo:
            return -1.0
        return float(np.dot(y[:3], y[3:]))

    at_perigee.terminal = True  # type: ignore[attr-defined]
    at_perigee.direction = 1.0  # type: ignore[attr-defined]

    def struck(t: float, y: np.ndarray) -> float:
        return float(np.linalg.norm(y[:3]) - _R_EARTH)

    struck.terminal = True  # type: ignore[attr-defined]
    struck.direction = -1.0  # type: ignore[attr-defined]

    def lunar(t: float, y: np.ndarray) -> float:
        return float(np.linalg.norm(y[:3] - _moon(t, phase, tilt)) - _R_MOON)

    lunar.terminal = True  # type: ignore[attr-defined]
    lunar.direction = -1.0  # type: ignore[attr-defined]

    run = solve_ivp(
        rates,
        (0.0, 3.0 * days * _DAY),
        [apo, 0.0, 0.0, 0.0, speed, 0.0],
        method="DOP853",
        rtol=1e-10,
        atol=1e-9,
        events=(at_perigee, struck, lunar),
        dense_output=True,
    )
    end = run.t[-1]
    radius = float(np.linalg.norm(run.y[:3, -1]))
    if run.t_events[2].size:
        return Pass(math.nan, 0.0, end / _DAY)
    samples = np.linspace(0.0, end, 4000)
    states = run.sol(samples)
    angle = phase + 2.0 * math.pi * samples / _MOON_PERIOD
    moons = _MOON_DISTANCE * np.array(
        [np.cos(angle), np.sin(angle) * math.cos(tilt), np.sin(angle) * math.sin(tilt)]
    )
    closest = float(np.min(np.linalg.norm(states[:3] - moons, axis=0)))
    return Pass(radius - _R_EARTH, closest, end / _DAY)


def targeted_nudge(
    days: float,
    moon_phase_deg: float,
    inclination_deg: float,
    sun_phase_deg: float = 0.0,
    target: float = TARGET_PERIGEE,
) -> Optional[float]:
    """The retrograde nudge that brings the next perigee to ``target`` (km/s).

    Zero if the Moon alone brings it lower, or into the Moon; None if no
    nudge within :data:`NUDGE_BOUND` does.
    """

    def miss(nudge: float) -> float:
        flown = next_perigee(
            days, nudge, moon_phase_deg, inclination_deg, sun_phase_deg
        )
        return -1.0 if math.isnan(flown.perigee) else flown.perigee - target

    if miss(0.0) <= 0.0:
        return 0.0
    if miss(NUDGE_BOUND) > 0.0:
        return None
    return float(brentq(miss, 0.0, NUDGE_BOUND, xtol=1e-6))


@dataclass(frozen=True)
class DisposalRow:
    """One lunar geometry.

    Attributes:
        phase: The Moon's angle from the apoapsis (deg).
        inclination: The Moon's orbit against the parking orbit (deg).
        sun_phase: The Sun's direction (deg).
        unnudged: The next perigee with no nudge (km).
        two_body: The next perigee after ADR 0074's two-body nudge (km).
        nudge: The nudge that reaches 100 km (m/s); None if over the bound.
        moon_closest: The closest approach to the Moon on that pass (km).
    """

    phase: float
    inclination: float
    sun_phase: float
    unnudged: float
    two_body: float
    nudge: Optional[float]
    moon_closest: float


def _row(geometry: Tuple[float, float, float, float]) -> DisposalRow:
    days, phase, inclination, sun_phase = geometry
    nudge = targeted_nudge(days, phase, inclination, sun_phase)
    flown = next_perigee(
        days, nudge if nudge is not None else 0.0, phase, inclination, sun_phase
    )
    return DisposalRow(
        phase,
        inclination,
        sun_phase,
        next_perigee(days, 0.0, phase, inclination, sun_phase).perigee,
        next_perigee(days, two_body_nudge(days), phase, inclination, sun_phase).perigee,
        None if nudge is None else nudge * 1e3,
        flown.moon_closest,
    )


def disposal_rows(
    days: float = 10.0,
    phases: Sequence[float] = PHASES,
    inclinations: Sequence[float] = INCLINATIONS,
    sun_phases: Sequence[float] = SUN_PHASES,
    workers: int = 4,
) -> List[DisposalRow]:
    """Every geometry, with its targeted nudge, in ``workers`` processes."""
    from multiprocessing import Pool

    geometries = [
        (days, phase, inclination, sun_phase)
        for inclination in inclinations
        for sun_phase in sun_phases
        for phase in phases
    ]
    with Pool(workers) as pool:
        return pool.map(_row, geometries)


def encounter(phase: float) -> bool:
    """Whether the Moon is within :data:`ENCOUNTER_PHASE` of the apoapsis."""
    return min(phase % 360.0, 360.0 - phase % 360.0) < ENCOUNTER_PHASE


def report(days: float = 10.0) -> str:
    """The sweep, by inclination and Sun direction, and its summary."""
    blind = two_body_nudge(days) * 1e3
    rows = disposal_rows(days)
    table = tabulate(
        [
            [
                f"{r.inclination:g}",
                f"{r.sun_phase:g}",
                f"{r.phase:g}",
                f"{r.unnudged:.0f}",
                f"{r.two_body:.0f}",
                "> bound" if r.nudge is None else f"{r.nudge:.1f}",
                f"{r.moon_closest:.0f}",
            ]
            for r in rows
        ],
        headers=[
            "Moon incl.",
            "Sun",
            "Moon phase",
            "perigee, no nudge (km)",
            f"perigee after {blind:.1f} m/s",
            "nudge to 100 km (m/s)",
            "closest to Moon (km)",
        ],
    )
    clear = [r for r in rows if not encounter(r.phase)]
    nudges = sorted(r.nudge for r in clear if r.nudge is not None)
    alone = sum(1 for r in clear if r.nudge == 0.0)
    near = sum(1 for r in clear if abs(r.two_body - TARGET_PERIGEE) <= 50.0)
    high = max(r.two_body for r in clear)
    met = [r for r in rows if encounter(r.phase)]
    return (
        f"{days:g}-day orbit, {PUSH_PERIGEE:g} km perigee; two-body nudge {blind:.1f} m/s\n"
        + table
        + f"\n\nMoon more than {ENCOUNTER_PHASE:g} deg from the apoapsis"
        f" ({len(clear)} geometries): targeted nudge median {np.median(nudges):.1f} m/s,"
        f" largest {nudges[-1]:.1f}; the Moon alone re-enters the plate in {alone};"
        f" the two-body nudge lands within 50 km of 100 km in {near}, and leaves it"
        f" as high as {high:.0f} km.\n"
        f"Lunar encounters (within {ENCOUNTER_PHASE:g} deg, {len(met)} geometries): the"
        f" parking orbit meets the Moon; nudges "
        + ", ".join("> bound" if r.nudge is None else f"{r.nudge:.0f}" for r in met)
        + " m/s."
    )


#: The ship's own parking orbit (the parent's S47 item 7): pushed at 400 km,
#: periapsis raised to 600 km at the two-body apoapsis time, for the departure.
DEPARTURE_PERIGEE = 600.0  # km
#: The Moon's Hill sphere (~61 500 km), inside which its pull rivals Earth's.
HILL_RADIUS = _MOON_DISTANCE * (_MU_MOON / (3.0 * _MU_EARTH)) ** (1.0 / 3.0)


@dataclass(frozen=True)
class Drift:
    """The ship's departure perigee flown through Earth, Moon and Sun, uncorrected.

    Attributes:
        altitude: The perigee's altitude (km), negative below the surface
            (Earth is a point mass here); NaN if the ship strikes the Moon.
        hours: Its time less the two-body time (h).
        apse_deg: Its direction less the two-body one, in the orbit's plane.
        moon_closest: The closest approach to the Moon's centre (km).
    """

    altitude: float
    hours: float
    apse_deg: float
    moon_closest: float

    @property
    def captured(self) -> bool:
        """Inside the Moon's Hill sphere: the Moon, not Earth, sets the orbit."""
        return bool(self.moon_closest < HILL_RADIUS)


def parking_drift(
    days: float,
    moon_phase_deg: float,
    inclination_deg: float,
    sun_phase_deg: float = 0.0,
    moon: bool = True,
    sun: bool = True,
) -> Drift:
    """Fly the ship from the push perigee, with the two-body raise at apoapsis.

    The push perigee is on -x at ``t = 0``, so the two-body apoapsis is on +x
    at half the period; the Moon's phase and the Sun's direction are at that
    time, in :func:`next_perigee`'s convention.  The raise to 600 km is the
    two-body tangential burn, applied at the two-body apoapsis time.  What
    remains at the departure perigee is what a navigated orbit must correct.

    Args:
        days: The parking orbit's period: the growth-to-departure gap.
        moon_phase_deg: The Moon's angle from the apoapsis at the apoapsis time.
        inclination_deg: The Moon's orbit against the parking orbit.
        sun_phase_deg: The Sun's direction then.
        moon: Include the Moon.
        sun: Include the Sun's tide.

    Returns:
        The drift.
    """
    a = _orbit_a(days)
    low = _R_EARTH + PUSH_PERIGEE
    high = _R_EARTH + DEPARTURE_PERIGEE
    apo = 2.0 * a - low
    raised = 0.5 * (apo + high)
    t_apo = 0.5 * days * _DAY
    t_peri = t_apo + math.pi * math.sqrt(raised**3 / _MU_EARTH)
    raise_dv = math.sqrt(_MU_EARTH * (2.0 / apo - 1.0 / raised)) - math.sqrt(
        _MU_EARTH * (2.0 / apo - 1.0 / a)
    )
    phase = math.radians(moon_phase_deg)
    tilt = math.radians(inclination_deg)
    sun_phase = math.radians(sun_phase_deg)

    def rates(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        acc = -_MU_EARTH * r / np.linalg.norm(r) ** 3
        if moon:
            m = _moon(t - t_apo, phase, tilt)
            acc = acc - _MU_MOON * (
                (r - m) / np.linalg.norm(r - m) ** 3 + m / _MOON_DISTANCE**3
            )
        if sun:
            s = _sun(t - t_apo, sun_phase)
            acc = acc - _MU_SUN * ((r - s) / np.linalg.norm(r - s) ** 3 + s / _AU**3)
        return np.concatenate([y[3:], acc])

    def struck(t: float, y: np.ndarray) -> float:
        lunar = np.linalg.norm(y[:3] - _moon(t - t_apo, phase, tilt)) - _R_MOON
        return float(lunar if moon else 1.0)

    struck.terminal = True  # type: ignore[attr-defined]

    def closest(run: object) -> float:
        times = np.linspace(run.t[0], run.t[-1], 2000)  # type: ignore[attr-defined]
        states = run.sol(times)  # type: ignore[attr-defined]
        moons = np.array([_moon(t - t_apo, phase, tilt) for t in times]).T
        return float(np.min(np.linalg.norm(states[:3] - moons, axis=0)))

    start = [-low, 0.0, 0.0, 0.0, -math.sqrt(_MU_EARTH * (2.0 / low - 1.0 / a)), 0.0]
    out = solve_ivp(
        rates,
        (0.0, t_apo),
        start,
        method="DOP853",
        rtol=1e-11,
        atol=1e-9,
        events=(struck,),
        dense_output=True,
    )
    near = closest(out) if moon else math.inf
    if out.t_events[0].size:
        return Drift(math.nan, math.nan, math.nan, 0.0)
    state = out.y[:, -1].copy()
    state[4] += raise_dv

    def at_perigee(t: float, y: np.ndarray) -> float:
        if np.linalg.norm(y[:3]) > 0.5 * apo:
            return -1.0
        return float(np.dot(y[:3], y[3:]))

    at_perigee.terminal = True  # type: ignore[attr-defined]
    at_perigee.direction = 1.0  # type: ignore[attr-defined]
    back = solve_ivp(
        rates,
        (t_apo, t_apo + 3.0 * days * _DAY),
        state,
        method="DOP853",
        rtol=1e-11,
        atol=1e-9,
        events=(at_perigee, struck),
        dense_output=True,
    )
    if moon:
        near = min(near, closest(back))
    if back.t_events[1].size or not back.t_events[0].size:
        return Drift(
            math.nan, math.nan, math.nan, 0.0 if back.t_events[1].size else near
        )
    y = back.y_events[0][0]
    return Drift(
        float(np.linalg.norm(y[:3]) - _R_EARTH),
        float((back.t_events[0][0] - t_peri) / 3600.0),
        math.degrees(math.atan2(y[1], -y[0])),
        near,
    )


def drift_report(days: float = 10.0) -> str:
    """The departure perigee's drift, by Moon phase and Sun direction."""
    rows = []
    for sun_phase in SUN_PHASES:
        rows.append(
            [f"Sun only, Sun {sun_phase:g}"]
            + _drift_cells(parking_drift(days, 0.0, 0.0, sun_phase, moon=False))
        )
    for phase in (60.0, 90.0, 180.0, 270.0, 300.0):
        rows.append(
            [f"Moon only, phase {phase:g}"]
            + _drift_cells(parking_drift(days, phase, 5.1, sun=False))
        )
    for sun_phase in SUN_PHASES:
        for phase in (60.0, 90.0, 180.0, 270.0, 300.0):
            rows.append(
                [f"both, phase {phase:g}, Sun {sun_phase:g}"]
                + _drift_cells(parking_drift(days, phase, 5.1, sun_phase))
            )
    summary = []
    for inclination in (0.0, 5.1, 28.6):
        flown = [
            parking_drift(days, phase, inclination, sun_phase)
            for sun_phase in SUN_PHASES
            for phase in range(0, 360, 5)
        ]
        clear = [d for d in flown if not d.captured and not math.isnan(d.altitude)]
        summary.append(
            f"Moon at {inclination:g} deg, 72 phases x 2 Sun directions:"
            f" {1.0 - len(clear) / len(flown):.0%} pass inside the Moon's"
            f" {HILL_RADIUS / 1e3:.0f} 000 km Hill sphere"
            + (f" (phases {_phase_runs(flown)});" if len(clear) < len(flown) else ";")
            + f" the rest reach perigee at {min(d.altitude for d in clear):.0f} to"
            f" {max(d.altitude for d in clear):.0f} km,"
            f" up to {max(abs(d.hours) for d in clear):.1f} h off,"
            f" apse up to {max(abs(d.apse_deg) for d in clear):.1f} deg"
        )
    return (
        f"The ship's {days:g}-day parking orbit, uncorrected (Moon at 5.1 deg):"
        f" the departure perigee against the two-body {DEPARTURE_PERIGEE:g} km\n"
        + tabulate(
            rows,
            headers=[
                "forces",
                "perigee (km)",
                "time (h)",
                "apse (deg)",
                "Moon (1000 km)",
            ],
        )
        + "\n"
        + "\n".join(summary)
    )


def _phase_runs(flown: Sequence[Drift]) -> str:
    """The phases (5 deg grid, either Sun direction) a pass is captured at."""
    hit = {
        (i % 72) * 5
        for i, d in enumerate(flown)
        if d.captured or math.isnan(d.altitude)
    }
    runs, start = [], None
    for p in range(0, 365, 5):
        inside = p in hit
        if inside and start is None:
            start = p
        if not inside and start is not None:
            runs.append(f"{start}-{p - 5}")
            start = None
    return ", ".join(runs) + " deg"


def _drift_cells(d: Drift) -> List[str]:
    if math.isnan(d.altitude):
        return ["strikes the Moon", "-", "-", "0"]
    return [
        f"{d.altitude:.0f}",
        f"{d.hours:+.2f}",
        f"{d.apse_deg:+.2f}",
        f"{d.moon_closest / 1e3:.0f}" + (" (inside Hill)" if d.captured else ""),
    ]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the disposal sweep.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=float, default=10.0, help="the parking orbit")
    parser.add_argument(
        "--drift",
        action="store_true",
        help="the ship's own uncorrected parking orbit instead (S47 item 7)",
    )
    args = parser.parse_args(argv)
    print(drift_report(args.days) if args.drift else report(args.days))


if __name__ == "__main__":
    main()
