"""Keep the departure stream in view: a visibility-constrained finite-burn loss.

ADR 0045 showed Earth does not occult the stream during a 20-minute burn on the
circular 3S closure.  The ledger's burns are longer (25-38 min for methane,
41-60 for hydrogen), and the flown chain's streams arrive more nearly head-on
than that closure's 144.9 degrees.  Centred on periapsis, such a burn loses the
stream on its approach leg: the PuffSats would have to pass through Earth to
reach the vehicle (ADR 0066).

Three pieces:

* :func:`onward_streams` rebuilds, for each flown cycle, the geometry its
  departure meets: the onward departure's excess speed, the returning nozzle
  wave's excess speed, and the angle between the two excess vectors (the *aim
  separation*).  It re-flies each window with the chain's own policy and
  re-solves the selected Lambert legs, matching the stored speeds.
* :func:`visible_burn` integrates a fixed-direction burn from the 20-day,
  600 km parking orbit (:mod:`src.orbit_turn_analysis`'s integrator, on its
  visible departure mirror) and finds the placement with the least finite-burn
  loss among those that keep every incoming route visible.  Blocking always
  falls at the start of a burn, on the approach leg, so the latest admissible
  start is found by bisection and the loss is minimised below it.
* :func:`stored_loss_models` reads the flown chain's tables from
  :data:`VISIBILITY_CSV` (the flown and three-synodic-only chains on the 10-
  and 20-day orbits), which ``make departure-visibility`` regenerates.
* :func:`visible_loss_model` tabulates that loss over burn length for one
  stream (:func:`visible_loss_table`) and returns a :data:`src.chamber_departure.LossModel`.  A burn with
  no admissible placement raises ``ValueError``, which the ledger's chamber
  search treats as "add a chamber".

Like :mod:`src.finite_burn_loss`, the loss is computed at one representative
exhaust speed, :data:`src.finite_burn_loss.TABLE_EXHAUST`; the visibility
check uses the same trajectory.  Planar and two-body; the parking orbit's plane
is taken to contain both excess vectors, which its apoapsis burns can arrange.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar

from src.chamber_departure import LossModel
from src.finite_burn_loss import TABLE_EXHAUST
from src.orbit_turn_analysis import (
    _incoming_routes,
    _orbit,
    _physical_stream_geometry,
    _propagate_burn,
)
from src.real_orbit_resonance import _heliocentric_state, _lambert_pair
from src.two_wave_growth import (
    _DEFAULT_THRESHOLD_M_S,
    DEFAULT_SPLIT_MODE,
    TwoWaveCycle,
    window_waves,
)

#: The departure burn's periapsis altitude (``sec:jovian_meeting_altitudes``).
DEPARTURE_ALTITUDE = 600.0 * u.km
#: Samples along a burn at which every incoming route is checked.
VISIBILITY_SAMPLES = 241
#: Burn lengths tabulated; past the finite-burn loss table's 3,600 s, since
#: the slower chambers burn that long.
TABLE_TIMES = np.arange(0.0, 7200.01, 400.0)
#: Bisection steps on the share of the burn flown before periapsis.
_BISECTION_STEPS = 12
#: Parking orbits whose chains are stored (days).
STORED_PERIODS = (20.0, 10.0)
#: The chains' tables, stored so the reports need not rebuild them
#: (about 6 min a process).  Regenerate with ``make departure-visibility``.
VISIBILITY_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "departure_visibility.csv"
)


@dataclass(frozen=True)
class OnwardStream:
    """The stream one flown cycle's onward departure meets.

    Attributes:
        index: The cycle's index in the flown chain.
        synodic_multiple: The cycle's synodic multiple.
        departure_vinf: The onward departure's excess speed (km/s).
        return_vinf: The returning nozzle wave's excess speed (km/s).
        aim_separation: Angle between the departure's excess vector and the
            stream's arriving excess velocity (degrees); 180 is exactly
            head-on at infinity.
    """

    index: int
    synodic_multiple: int
    departure_vinf: float
    return_vinf: float
    aim_separation: float


@dataclass(frozen=True)
class VisibleBurn:
    """The best fully visible placement of one fixed-direction burn.

    Attributes:
        burn_time: Burn length (s).
        share_before: Share of the burn flown before the reference periapsis.
        unconstrained_share_before: The same without the visibility constraint.
        loss: Finite-burn loss at the visible placement (km/s).
        unconstrained_loss: The least loss with the stream ignored (km/s).
    """

    burn_time: float
    share_before: float
    unconstrained_share_before: float
    loss: float
    unconstrained_loss: float


def _excess_vectors(
    departure_jd: float, return_jd: float, solution: object
) -> Tuple[np.ndarray, np.ndarray]:
    """Departure and arrival excess vectors of a window's selected return.

    The stored solution keeps only speeds, so each Lambert path pairing is
    re-solved and the one whose speeds match is taken.
    """
    departure_position, departure_velocity = _heliocentric_state("earth", departure_jd)
    return_position, return_velocity = _heliocentric_state("earth", return_jd)
    best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    for outbound_low in (True, False):
        for return_low in (True, False):
            pair = _lambert_pair(
                departure_jd,
                return_jd,
                float(getattr(solution, "encounter_jd")),
                outbound_low,
                return_low,
                departure_position,
                return_position,
            )
            if pair is None:
                continue
            departure = pair[0] - departure_velocity
            arrival = pair[2] - return_velocity
            error = abs(
                float(np.linalg.norm(departure))
                - float(getattr(solution, "earth_departure_vinf"))
            ) + abs(
                float(np.linalg.norm(arrival))
                - float(getattr(solution, "earth_return_vinf"))
            )
            if best is None or error < best[0]:
                best = (error, departure, arrival)
    if best is None or best[0] > 1.0e-6:
        raise RuntimeError("the selected return's Lambert legs did not re-solve")
    return best[1], best[2]


def onward_streams(
    cycles: Sequence[TwoWaveCycle],
    threshold_m_s: float = _DEFAULT_THRESHOLD_M_S,
    split: str = DEFAULT_SPLIT_MODE,
) -> List[OnwardStream]:
    """The stream each flown cycle's onward departure meets.

    Cycle ``i``'s nozzle wave arrives at its return, which is cycle ``i + 1``'s
    departure; the last cycle's departure is the window after the chain.  Each
    window is re-flown as the chain flew it (its waves from
    :func:`src.two_wave_growth.window_waves`), so the departure is the one the
    split shares.

    Args:
        cycles: Consecutive flown cycles.
        threshold_m_s: The chain's 2S/3S policy threshold.
        split: How the chain's batches divide.

    Returns:
        One stream per cycle.
    """
    gap = cycles[0].split_days
    selected = [window_waves(c.departure_jd, threshold_m_s, gap, split) for c in cycles]
    after = window_waves(cycles[-1].return_jd, threshold_m_s, gap, split)
    streams = []
    for i, cycle in enumerate(cycles):
        _, arrival = _excess_vectors(
            cycle.departure_jd, cycle.return_jd, selected[i][2]
        )
        if i + 1 < len(cycles):
            nxt = cycles[i + 1]
            window = (nxt.departure_jd, nxt.return_jd, selected[i + 1][2])
        else:
            window = (cycle.return_jd, after[1], after[2])
        departure, _ = _excess_vectors(*window)
        cosine = float(
            departure @ arrival / np.linalg.norm(departure) / np.linalg.norm(arrival)
        )
        streams.append(
            OnwardStream(
                index=cycle.index,
                synodic_multiple=cycle.synodic_multiple,
                departure_vinf=float(np.linalg.norm(departure)),
                return_vinf=float(np.linalg.norm(arrival)),
                aim_separation=float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
            )
        )
    return streams


@dataclass(frozen=True)
class BurnTrajectory:
    """One fixed-direction departure burn, sampled.

    Attributes:
        loss: Finite-burn loss against the impulse (km/s).
        ideal_burn: The burn's ideal velocity integral (km/s).
        times: Sample times from ignition (s).
        states: Position and velocity at each sample (km, km/s), shape (4, n).
        acceleration: Thrust acceleration at each sample (km/s^2).
        incoming: The stream's incoming asymptote direction, burn axes.
        anchor: Impact parameter of the stream's route at periapsis.
        mu: Earth's gravitational parameter (km^3/s^2).
        earth_radius: Earth's radius (km).
    """

    loss: float
    ideal_burn: float
    times: np.ndarray
    states: np.ndarray
    acceleration: np.ndarray
    incoming: np.ndarray
    anchor: float
    mu: float
    earth_radius: float


def burn_trajectory(
    stream: OnwardStream,
    burn_time: float,
    share_before: float,
    period_days: float = 20.0,
    exhaust_speed: float = float(TABLE_EXHAUST.to_value(u.km / u.s)),
    samples: int = VISIBILITY_SAMPLES,
) -> BurnTrajectory:
    """Integrate a fixed-direction burn that reaches the stream's departure.

    Args:
        stream: The stream the departure meets.
        burn_time: Burn length (s).
        share_before: Share of the burn flown before the reference periapsis.
        period_days: Parking-orbit period.
        exhaust_speed: Constant exhaust speed of the mass-flow profile (km/s).
        samples: Samples along the burn.

    Returns:
        The trajectory.
    """
    mu, earth_radius, radius, parking = _orbit(DEPARTURE_ALTITUDE, period_days * u.day)
    exhaust = exhaust_speed
    final = float(np.hypot(stream.departure_vinf, np.sqrt(2.0 * mu / radius)))
    impulsive = final - parking
    target = 0.5 * final**2 - mu / radius
    before = share_before * burn_time

    def gap(ideal: float) -> float:
        state, _ = _propagate_burn(
            mu, radius, parking, before, burn_time, ideal, exhaust, False
        )
        speed_sq = float(state[2:] @ state[2:])
        return 0.5 * speed_sq - mu / float(np.linalg.norm(state[:2])) - target

    ideal = float(brentq(gap, impulsive, 2.0 * impulsive + 1.0, xtol=1.0e-7))
    _, dense = _propagate_burn(
        mu, radius, parking, before, burn_time, ideal, exhaust, False, dense=True
    )
    assert dense is not None
    times = np.linspace(0.0, burn_time, samples)
    incoming, anchor = _physical_stream_geometry(
        mu,
        earth_radius,
        radius,
        stream.departure_vinf,
        stream.return_vinf,
        float(np.radians(stream.aim_separation)),
    )
    spent = 1.0 - np.exp(-ideal / exhaust)
    acceleration = exhaust * spent / burn_time / (1.0 - spent * times / burn_time)
    return BurnTrajectory(
        loss=ideal - impulsive,
        ideal_burn=ideal,
        times=times,
        states=np.asarray(dense(times)),
        acceleration=np.asarray(acceleration),
        incoming=np.asarray(incoming),
        anchor=float(anchor),
        mu=mu,
        earth_radius=earth_radius,
    )


def _fly(
    stream: OnwardStream, period_days: float, burn_time: float, share_before: float
) -> Tuple[float, float]:
    """Loss (km/s) and share of the burn the stream is blocked for."""
    flown = burn_trajectory(stream, burn_time, share_before, period_days)
    blocked = sum(
        not _in_view(
            flown.mu, flown.earth_radius, position, stream.return_vinf, flown.incoming
        )
        for position in flown.states[:2].T
    )
    return flown.loss, blocked / VISIBILITY_SAMPLES


def _in_view(
    mu: float,
    earth_radius: float,
    position: np.ndarray,
    return_vinf: float,
    incoming: np.ndarray,
) -> bool:
    """Whether any incoming route reaches ``position`` before Earth.

    A vehicle exactly on the stream's asymptote line, upstream of Earth, is
    reached by the radial route of zero impact parameter, which the conic
    solver cannot represent; that route falls inbound to the vehicle first.
    """
    try:
        routes = _incoming_routes(mu, earth_radius, position, return_vinf, incoming)
    except ZeroDivisionError:
        return True
    return any(route.visible for route in routes)


def visible_burn(
    stream: OnwardStream, burn_time: float, period_days: float = 20.0
) -> Optional[VisibleBurn]:
    """The least-loss placement of a burn that keeps the stream in view.

    Args:
        stream: The stream the departure meets.
        burn_time: Burn length (s).
        period_days: Parking-orbit period.

    Returns:
        The placement, or None if no placement keeps the stream in view.
    """
    if burn_time <= 0.0:
        return VisibleBurn(0.0, 0.5, 0.5, 0.0, 0.0)

    def loss(share: float) -> float:
        return _fly(stream, period_days, burn_time, share)[0]

    free = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    free_share, free_loss = float(free.x), float(free.fun)
    if _fly(stream, period_days, burn_time, free_share)[1] == 0.0:
        return VisibleBurn(burn_time, free_share, free_share, free_loss, free_loss)
    # Blocking falls on the approach leg, so start later; if it ever fell at
    # the end, start earlier.  Either way the loss falls toward the free
    # optimum, so the visible placement nearest it wins.
    for edge in (0.0, 1.0):
        if _fly(stream, period_days, burn_time, edge)[1] > 0.0:
            continue
        visible, blocked = edge, free_share
        for _ in range(_BISECTION_STEPS):
            middle = 0.5 * (visible + blocked)
            if _fly(stream, period_days, burn_time, middle)[1] == 0.0:
                visible = middle
            else:
                blocked = middle
        return VisibleBurn(burn_time, visible, free_share, loss(visible), free_loss)
    return None


@lru_cache(maxsize=64)
def visible_loss_table(
    stream: OnwardStream, period_days: float = 20.0
) -> Tuple[float, ...]:
    """Visible loss (km/s) at each of :data:`TABLE_TIMES`; inf where none.

    About 2 s a node.  Picklable, so a report can build every cycle's table
    once and hand them to worker processes.

    Args:
        stream: The stream the departure meets.
        period_days: Parking-orbit period.

    Returns:
        The losses.
    """
    values = []
    for seconds in TABLE_TIMES:
        burn = visible_burn(stream, float(seconds), period_days)
        values.append(np.inf if burn is None else burn.loss)
    return tuple(values)


def loss_model_from_table(table: Sequence[float]) -> LossModel:
    """A finite-burn loss read off one :func:`visible_loss_table`.

    The stream fixes the departure's excess speed, so the impulsive burn passed
    in is not used; the table is over burn length only.  Past the longest
    burn that keeps the stream in view the model raises ``ValueError``.

    Args:
        table: Losses (km/s) at :data:`TABLE_TIMES`.

    Returns:
        The loss model.

    Raises:
        RuntimeError: If not even the shortest tabulated burns stay in view.
    """
    values = np.asarray(table, dtype=np.float64)
    finite = np.isfinite(values)
    last = int(np.argmin(finite)) if not finite.all() else values.size
    if last < 2:
        raise RuntimeError("no burn length keeps this stream in view")
    curve = PchipInterpolator(TABLE_TIMES[:last], values[:last])
    longest = float(TABLE_TIMES[last - 1])

    def model(burn: u.Quantity, burn_time: u.Quantity) -> u.Quantity:
        seconds = float(burn_time.to_value(u.s))
        if not 0.0 <= seconds <= longest:
            raise ValueError(
                f"burn time {burn_time} loses the stream (longest visible "
                f"{longest:.0f} s)"
            )
        return (float(curve(seconds)) * u.km / u.s).to(u.m / u.s)

    return model


def visible_loss_model(stream: OnwardStream, period_days: float = 20.0) -> LossModel:
    """A finite-burn loss that keeps the stream in view, for one cycle.

    Args:
        stream: The stream the departure meets.
        period_days: Parking-orbit period.

    Returns:
        The loss model.
    """
    return loss_model_from_table(visible_loss_table(stream, period_days))


_Key = Tuple[float, float, float, float]


def _key(
    departure_jd: float, return_jd: float, onward_burn: float, period_days: float
) -> _Key:
    """A cycle's identity: its dates, the onward window it feeds and its orbit.

    Two chains can share a cycle's dates yet fly different onward windows, and
    the onward departure sets the stream's geometry.
    """
    return (
        round(departure_jd, 6),
        round(return_jd, 6),
        round(onward_burn, 6),
        round(period_days, 6),
    )


@lru_cache(maxsize=1)
def _stored() -> Dict[_Key, Tuple[OnwardStream, Tuple[float, ...]]]:
    """The stored tables, keyed by departure date and parking period."""
    return _read(VISIBILITY_CSV)


def _read(
    path: Path,
) -> Dict[_Key, Tuple[OnwardStream, Tuple[float, ...]]]:
    out: Dict[_Key, Tuple[OnwardStream, Tuple[float, ...]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            stream = OnwardStream(
                int(row["index"]),
                int(row["synodic_multiple"]),
                float(row["departure_vinf"]),
                float(row["return_vinf"]),
                float(row["aim_separation"]),
            )
            table = tuple(float(row[f"loss_{t:.0f}s"]) for t in TABLE_TIMES)
            key = _key(
                float(row["departure_jd"]),
                float(row["return_jd"]),
                float(row["onward_burn"]),
                float(row["period_days"]),
            )
            out[key] = (
                stream,
                table,
            )
    return out


def stored_tables(
    cycles: Sequence[TwoWaveCycle], period_days: float = 20.0
) -> List[Tuple[OnwardStream, Tuple[float, ...]]]:
    """Each cycle's stream and visible-loss table, read from :data:`VISIBILITY_CSV`.

    Args:
        cycles: Flown cycles.
        period_days: Parking-orbit period.

    Returns:
        One (stream, table) per cycle.

    Raises:
        KeyError: If a cycle is not stored; pass a loss model explicitly or
            regenerate the file for that chain.
    """
    stored = _stored()
    out = []
    for cycle in cycles:
        key = _key(cycle.departure_jd, cycle.return_jd, cycle.onward_burn, period_days)
        if key not in stored:
            raise KeyError(
                f"no stored departure visibility for the cycle leaving JD "
                f"{cycle.departure_jd} on a {period_days:g}-day orbit; run "
                "`make departure-visibility` for its chain"
            )
        out.append(stored[key])
    return out


def stored_loss_models(
    cycles: Sequence[TwoWaveCycle], period_days: float = 20.0
) -> List[LossModel]:
    """Each cycle's visible loss, from :func:`stored_tables`.

    Args:
        cycles: Flown cycles.
        period_days: Parking-orbit period.

    Returns:
        One loss model per cycle.
    """
    return [
        loss_model_from_table(table) for _, table in stored_tables(cycles, period_days)
    ]


def write_tables(
    chains: Sequence[Tuple[Sequence[TwoWaveCycle], float, float]],
    path: Path,
    workers: int,
) -> None:
    """Build every cycle's table and write them to ``path``.

    Args:
        chains: (cycles, parking-orbit period in days, the chain's 2S/3S
            threshold in m/s) triples; a cycle two chains share is written
            once.
        path: The CSV to write.
        workers: Worker processes.
    """
    rows: Dict[_Key, Tuple[TwoWaveCycle, float, OnwardStream]] = {}
    for cycles, period_days, threshold in chains:
        for cycle, stream in zip(cycles, onward_streams(cycles, threshold)):
            rows.setdefault(
                _key(
                    cycle.departure_jd, cycle.return_jd, cycle.onward_burn, period_days
                ),
                (cycle, period_days, stream),
            )
    entries = list(rows.values())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tables = list(
            pool.map(
                visible_loss_table,
                [stream for _, _, stream in entries],
                [period for _, period, _ in entries],
            )
        )
    columns = [
        "index",
        "synodic_multiple",
        "departure_jd",
        "return_jd",
        "onward_burn",
        "period_days",
        "aim_separation",
        "departure_vinf",
        "return_vinf",
    ] + [f"loss_{t:.0f}s" for t in TABLE_TIMES]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for (cycle, period_days, stream), table in zip(entries, tables):
            writer.writerow(
                [
                    stream.index,
                    stream.synodic_multiple,
                    f"{cycle.departure_jd:.6f}",
                    f"{cycle.return_jd:.6f}",
                    f"{cycle.onward_burn:.9f}",
                    f"{period_days:g}",
                    f"{stream.aim_separation:.6f}",
                    f"{stream.departure_vinf:.9f}",
                    f"{stream.return_vinf:.9f}",
                ]
                + [f"{value:.9f}" for value in table]
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Regenerate :data:`VISIBILITY_CSV`: the flown and three-synodic-only
    chains on the 20-day orbit and on the 10-day one the growth ledger compares.

    Args:
        argv: ``--workers``.
    """
    # Imported here: seed_cost sits above this module in the import graph.
    from src.seed_cost import chains

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    pairs: List[Tuple[Sequence[TwoWaveCycle], float, float]] = []
    for period_days in STORED_PERIODS:
        flown, three = chains(period_days)
        pairs += [
            (flown, period_days, _DEFAULT_THRESHOLD_M_S),
            (three, period_days, 0.0),
        ]
    write_tables(pairs, VISIBILITY_CSV, args.workers)
    print(f"wrote {len(_read(VISIBILITY_CSV))} cycles to {VISIBILITY_CSV}")


if __name__ == "__main__":
    main()
