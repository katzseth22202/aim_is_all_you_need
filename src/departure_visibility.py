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
* :func:`visible_loss_model` tabulates that loss over burn length for one
  stream (:func:`visible_loss_table`) and returns a :data:`src.chamber_departure.LossModel`.  A burn with
  no admissible placement raises ``ValueError``, which the ledger's chamber
  search treats as "add a chamber".

Like :mod:`src.finite_burn_loss`, the loss is computed at one representative
exhaust speed, :data:`src.finite_burn_loss.TABLE_EXHAUST`; the visibility
check uses the same trajectory.  Planar and two-body; the parking orbit's plane
is taken to contain both excess vectors, which its apoapsis burns can arrange.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

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
    TwoWaveCycle,
    _fly_window,
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
    cycles: Sequence[TwoWaveCycle], threshold_m_s: float = _DEFAULT_THRESHOLD_M_S
) -> List[OnwardStream]:
    """The stream each flown cycle's onward departure meets.

    Cycle ``i``'s nozzle wave arrives at its return, which is cycle ``i + 1``'s
    departure; the last cycle's departure is the window after the chain.

    Args:
        cycles: Consecutive flown cycles.
        threshold_m_s: The chain's 2S/3S policy threshold.

    Returns:
        One stream per cycle.
    """
    selected = [_fly_window(c.departure_jd, threshold_m_s) for c in cycles]
    after = _fly_window(cycles[-1].return_jd, threshold_m_s)
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


def _fly(
    stream: OnwardStream, period_days: float, burn_time: float, share_before: float
) -> Tuple[float, float]:
    """Loss (km/s) and share of the burn the stream is blocked for."""
    mu, earth_radius, radius, parking = _orbit(DEPARTURE_ALTITUDE, period_days * u.day)
    exhaust = float(TABLE_EXHAUST.to_value(u.km / u.s))
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
    states = dense(np.linspace(0.0, burn_time, VISIBILITY_SAMPLES))
    incoming, _ = _physical_stream_geometry(
        mu,
        earth_radius,
        radius,
        stream.departure_vinf,
        stream.return_vinf,
        float(np.radians(stream.aim_separation)),
    )
    blocked = sum(
        not any(
            route.visible
            for route in _incoming_routes(
                mu, earth_radius, position, stream.return_vinf, incoming
            )
        )
        for position in states[:2].T
    )
    return ideal - impulsive, blocked / VISIBILITY_SAMPLES


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
