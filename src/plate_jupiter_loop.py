"""The Jupiter-only chain on ADR 0073's plates, with no Sun dive and no nozzle.

The parent's S41 (2026-10-19): fly the child's flown chain
(:func:`src.two_wave_growth.adaptive_two_wave_cycles`, real ephemerides,
the 2S/3S cadence, split by role on the 10-day orbit) with the solar dive's
two Earth plates (:func:`src.solar_dive_growth.fly_earth`):

1. **Push.** The growth wave overtakes the lofted unit at 400 km and pushes
   it onto the parking orbit on the flat push plate, at 2 Hz and 1 g, from the
   2.43 km/s lob.  The push plate is nudged down and left at apoapsis.
2. **Apoapsis.** Raise periapsis to 600 km and turn the orbit from the growth
   wave's direction onto the departure's thrust axis (nearly a reversal here).
3. **Departure.** The nozzle wave's stream meets the ship nearly head-on and
   threads the holed bowl at 1 Hz and a 0.25 g mean.  Its finite-burn loss is
   the chain's own, with every incoming route kept in view (ADR 0066).
4. **Jupiter and home.** The ship rides the chain's flyby and returns as the
   next wave: the departing stack less 1% avionics, less the methalox its
   wave burns on the way home (the growth wave's split burn, the nozzle
   wave's correction).

There is no Sun stage, so no disks, carrier, shade or Sun plate.  Each window
is its own ledger, and the program is valued on the solar dive's gated rNPV
with the Sun gate removed, the chain repeated past its stored horizon.

Run with ``make plate-jupiter-loop``.
"""

import argparse
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from scipy.optimize import minimize
from tabulate import tabulate

from src.departure_visibility import _excess_vectors, stored_loss_models
from src.jovian_solar_dive_cycle import synodic_period
from src.real_orbit_resonance import (
    _MANEUVER_ENCOUNTER_SAMPLES,
    _MANEUVER_ENCOUNTER_TOF_MIN_YEARS,
    _MANEUVER_RETURN_TOF_MIN_YEARS,
    _dsm_only_solution,
    _encounter_geometry,
    _heliocentric_state,
)
from src.retrograde_return_legs import _powered_flyby_params
from src.solar_dive_growth import (
    AVIONICS_FRACTION,
    DEPARTURE_ALTITUDE,
    ESTIMATE,
    GROWTH_SYNODIC,
    HORIZON_YEARS,
    K_EARTH,
    K_PUSH,
    MISS_FLOOR,
    PARKING_DAYS,
    PUSH_ALTITUDE,
    BurnLoss,
    CycleGrowth,
    Design,
    Gate,
    GatedSchedule,
    SolarDivePrices,
    StageLaws,
    UnitLedger,
    Valuation,
    _money,
    _unit_dollars,
    balance,
    delivery_unit,
    fly_earth,
    miss_fraction,
    visible_sign,
    wave_speed,
)
from src.two_wave_growth import (
    _DAYS_PER_YEAR,
    _DEFAULT_THRESHOLD_M_S,
    DEFAULT_SPLIT_MODE,
    VE_METHALOX,
    adaptive_two_wave_cycles,
    window_waves,
)

#: The chains flown: the adaptive 2S/3S cadence, and three-synodic only.
CHAINS = {"flown": _DEFAULT_THRESHOLD_M_S, "3S only": 0.0}
#: The departure accelerations traded (S41 item 3).
DEPARTURE_GS = (0.25, 0.5, 1.0)
#: The optimizer's knobs and bounds; there is no Sun stage to tune.
_KNOBS = ("k_push", "k_earth", "k_delivery")
_BOUNDS = {"k_push": K_PUSH, "k_earth": K_EARTH, "k_delivery": K_PUSH}
#: Where the optimizer starts (bounds ``_BOUNDS``); ADR 0075 records the
#: optima it reaches: k 6.5-6.8 push, 12.3-12.7 departure, 6.0-6.6 delivery.
START = Design(k_push=6.5, k_earth=10.0, k_delivery=5.0)
#: ``(k_push, k_earth)`` "relatively high" loadings, for comparison (item 2).
HIGH_K = Design(k_push=10.0, k_earth=20.0, k_delivery=10.0)
#: Methalox return correction a return-family member may carry (item 5).
FAMILY_DSM_CAP = 0.5  # km/s


@dataclass(frozen=True)
class Carrier:
    """One carrier role's onward departure and return.

    Attributes:
        excess: Earth departure excess (km/s).
        aim_deg: Its direction in the ecliptic (deg).
        return_burn: Methalox its wave burns on the way home (km/s).
        loss: The departure's finite-burn loss, stream kept in view.
    """

    excess: float
    aim_deg: float
    return_burn: float
    loss: BurnLoss


@dataclass(frozen=True)
class LoopWindow:
    """The launch units one flown cycle's returning waves push, and their flight.

    Cycle ``k``'s two waves arrive at its return, push units that depart on
    window ``k + 1``'s growth and nozzle trajectories, and those units return
    at window ``k + 1``'s end.

    Attributes:
        index: The arriving cycle's index.
        multiple: The onward window's synodic multiple.
        period_years: The onward window's length.
        jupiter_years: Onward departure to the Jovian encounter.
        push_stream: The growth wave's speed at 400 km (km/s).
        push_axis_deg: The growth wave's direction (deg, ecliptic).
        stream_excess: The nozzle wave's Earth excess (km/s).
        stream_axis_deg: The nozzle wave's direction (deg, ecliptic).
        growth: The growth carrier.
        rods: The rod carrier (the chain's name; on plates it carries no rods).
        arrival_multiple: The arriving cycle's own synodic multiple.
        arrival_years: The arriving cycle's own length.
        arrival_jupiter_years: Its departure to its Jovian encounter.
        arrival_departure_excess: Its nozzle trajectory's departure excess
            (km/s): what a seed flying that cycle leaves with.
    """

    index: int
    multiple: int
    period_years: float
    jupiter_years: float
    push_stream: float
    push_axis_deg: float
    stream_excess: float
    stream_axis_deg: float
    growth: Carrier
    rods: Carrier
    arrival_multiple: int
    arrival_years: float
    arrival_jupiter_years: float
    arrival_departure_excess: float


def _planar_deg(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _float_loss(model: object) -> BurnLoss:
    def loss(burn: float, seconds: float) -> float:
        value = model(burn * u.km / u.s, seconds * u.s)  # type: ignore[operator]
        return float(value.to_value(u.km / u.s))

    return loss


@lru_cache(maxsize=None)
def loop_windows(chain: str = "flown") -> Tuple[LoopWindow, ...]:
    """Every flown cycle's onward units, from the chain's own legs.

    The excess vectors are re-solved from each window's selected Lambert legs
    (:func:`src.departure_visibility._excess_vectors`) and projected onto the
    ecliptic for the departure's planar geometry; magnitudes are the full
    vectors'.

    Args:
        chain: A key of :data:`CHAINS`.

    Returns:
        One window per flown cycle.
    """
    threshold = CHAINS[chain]
    gap = PARKING_DAYS
    cycles = adaptive_two_wave_cycles(threshold_m_s=threshold, split_days=gap)
    waves = [
        window_waves(c.departure_jd, threshold, gap, DEFAULT_SPLIT_MODE) for c in cycles
    ]
    after_jd = cycles[-1].return_jd
    waves.append(window_waves(after_jd, threshold, gap, DEFAULT_SPLIT_MODE))
    starts = [c.departure_jd for c in cycles] + [after_jd]
    losses = {
        role: stored_loss_models(cycles, gap, role) for role in ("rods", "growth")
    }
    out = []
    for k, cycle in enumerate(cycles):
        multiple, return_jd, nozzle, growth = waves[k]
        nozzle_out, nozzle_in = _excess_vectors(starts[k], return_jd, nozzle)
        _, growth_in = _excess_vectors(starts[k], return_jd - gap, growth)
        on_multiple, on_return, on_nozzle, on_growth = waves[k + 1]
        start = starts[k + 1]
        rods_out, _ = _excess_vectors(start, on_return, on_nozzle)
        growth_out, _ = _excess_vectors(start, on_return - gap, on_growth)
        out.append(
            LoopWindow(
                index=cycle.index,
                multiple=on_multiple,
                period_years=(on_return - start) / _DAYS_PER_YEAR,
                jupiter_years=(on_nozzle.encounter_jd - start) / _DAYS_PER_YEAR,
                push_stream=wave_speed(float(np.linalg.norm(growth_in)), PUSH_ALTITUDE),
                push_axis_deg=_planar_deg(growth_in),
                stream_excess=float(np.linalg.norm(nozzle_in)),
                stream_axis_deg=_planar_deg(nozzle_in),
                growth=Carrier(
                    float(np.linalg.norm(growth_out)),
                    _planar_deg(growth_out),
                    on_growth.total_dv,
                    _float_loss(losses["growth"][k]),
                ),
                rods=Carrier(
                    float(np.linalg.norm(rods_out)),
                    _planar_deg(rods_out),
                    on_nozzle.total_dv,
                    _float_loss(losses["rods"][k]),
                ),
                arrival_multiple=multiple,
                arrival_years=(return_jd - starts[k]) / _DAYS_PER_YEAR,
                arrival_jupiter_years=(nozzle.encounter_jd - starts[k])
                / _DAYS_PER_YEAR,
                arrival_departure_excess=float(np.linalg.norm(nozzle_out)),
            )
        )
    return tuple(out)


#: Sun-stage lines the cost book reads, all zero on this loop.
_NO_SUN = (
    "sun_plate",
    "bore_collar",
    "film_sun",
    "sun_charge",
    "sun_mounts",
    "disks",
    "carrier_dry",
    "carrier_propellant",
    "shield",
)


def fly_carrier(
    window: LoopWindow,
    carrier: Carrier,
    design: Design,
    law: Optional[StageLaws] = None,
) -> UnitLedger:
    """One launch unit pushed, departed on the bowl, and returned as sheets.

    Args:
        window: The window.
        carrier: Which carrier (``window.growth`` or ``window.rods``).
        design: The knobs (the Sun's are unused).
        law: The stage laws; the vendored tables by default.

    Returns:
        The ledger.
    """
    law = law or StageLaws.load(bonds=design.bonds)
    geometry = visible_sign(
        carrier.excess, carrier.aim_deg, window.stream_excess, window.stream_axis_deg
    )
    earth = fly_earth(
        window.push_stream,
        window.push_axis_deg,
        geometry,
        carrier.excess,
        PARKING_DAYS,
        design,
        law,
        carrier.loss,
    )
    lines = dict(earth.lines)
    avionics = AVIONICS_FRACTION * earth.departing
    ship = earth.departing - avionics
    arrives = math.exp(-carrier.return_burn / VE_METHALOX)
    returned = ship * arrives
    lines.update({key: 0.0 for key in _NO_SUN})
    lines.update(
        avionics=avionics,
        return_methalox=ship - returned,
        apoapsis_methalox=lines["methalox"],
        sheets=returned,
    )
    lines["methalox"] += ship - returned
    return UnitLedger(earth.push_sheets, earth.departure_sheets, returned, 0.0, lines)


def window_growth(
    window: LoopWindow, design: Design, arrives: float = 1.0 - MISS_FLOOR
) -> CycleGrowth:
    """The window's growth, split by role (ADR 0069, 0071)."""
    law = StageLaws.load(bonds=design.bonds)
    return balance(
        fly_carrier(window, window.growth, design, law),
        fly_carrier(window, window.rods, design, law),
        arrives,
    )


@dataclass(frozen=True)
class ChainRate:
    """The chain's growth, reported as the solar dive's is.

    Attributes:
        rate_per_year: ``sum ln g / sum years``.
        doubling: ``ln 2 / rate`` (years; inf if it does not grow).
        per_three_synodic: Growth over three synodic periods at that rate.
        growths: Each window's growth.
    """

    rate_per_year: float
    doubling: float
    per_three_synodic: float
    growths: Tuple[float, ...]


#: Three synodic periods (years), the solar dive's cycle, for comparison.
_THREE_SYNODIC_YEARS = (
    3.0 * synodic_period(_powered_flyby_params()) / (365.25 * 86400.0)
)


def chain_rate(windows: Sequence[LoopWindow], design: Design) -> ChainRate:
    """Growth over the chain at the floor miss rate."""
    growths = tuple(window_growth(w, design).growth for w in windows)
    if min(growths) <= 0.0:
        return ChainRate(-math.inf, math.inf, 0.0, growths)
    rate = sum(math.log(g) for g in growths) / sum(w.period_years for w in windows)
    doubling = math.log(2.0) / rate if rate > 0.0 else math.inf
    return ChainRate(rate, doubling, math.exp(rate * _THREE_SYNODIC_YEARS), growths)


def loop_gates(window: LoopWindow) -> Tuple[Gate, ...]:
    """The solar dive's milestones with the Sun gate removed (S41).

    The seed flies the chain's first cycle (``window``'s arriving one).
    """
    return (
        Gate("seed departs and flies by Jupiter", window.arrival_jupiter_years, 0.95),
        Gate(
            "sheets unfurl and arrive face-on",
            window.arrival_years - PARKING_DAYS / 365.25 - 1e-6,
            0.85,
        ),
        Gate("first 1 g push and hole-bowl departure", window.arrival_years, 0.90),
    )


def seed_excess(window: LoopWindow) -> float:
    """The seed's departure excess: the chain's first departure, as the child's (km/s)."""
    return window.arrival_departure_excess


def seed_price(window: LoopWindow, dear: bool = False) -> float:
    """Seed dollars per kilogram at the seed's excess (ADR 0057's ends)."""
    from src.seed_cost import seed_price_range

    prices = seed_price_range(seed_excess(window) * u.km / u.s)
    return prices.high if dear else prices.low


def value_chain(
    windows: Sequence[LoopWindow],
    design: Design,
    prices: SolarDivePrices = ESTIMATE,
    seed_price_per_kg: float = 1367.0,
    guidance: float = 0.05,
    rate: float = 0.10,
    gates: Optional[Tuple[Gate, ...]] = None,
    growth_synodic: int = GROWTH_SYNODIC,
    delay_years: float = 0.0,
) -> Valuation:
    """Grow from a seed for ``growth_synodic`` synodic periods, then hold level.

    :func:`src.solar_dive_growth.value_design` over a chain of unequal
    windows.  The seed is one launch unit's worth of departing ships, priced
    as window 0's in its steady split, built new and flown on the chain's
    first cycle at :func:`seed_excess`, as the child's seed is (ADR 0060).
    Its return is window 0's arrival; units pushed by window ``k``'s waves
    return at window ``k + 1``'s end; the chain repeats past its stored
    horizon.  ``delay_years`` shifts every
    flow after the seed's outlay (a seed route that arrives later).

    Returns:
        The valuation; ``growth`` is the chain's growth per three synodic
        periods.
    """
    laws = StageLaws.load(bonds=design.bonds)
    schedule = GatedSchedule(rate, loop_gates(windows[0]) if gates is None else gates)
    units = [
        (
            fly_carrier(w, w.growth, design, laws),
            fly_carrier(w, w.rods, design, laws),
        )
        for w in windows
    ]
    n_windows = len(windows)
    lob = prices.lob * prices.lob_climb
    gap_years = PARKING_DAYS / 365.25
    plates = {"push": 0.0, "bowl": 0.0, "sun": 0.0}
    packages = [0.0]
    p0, r0 = units[0]
    share = balance(p0, r0, 1.0 - MISS_FLOOR).push_share
    seed_mass = share * p0.lines["departing"] + (1 - share) * r0.lines["departing"]
    seed_contents = 0.0
    # The loop's streams are 57-69 km/s, so its sheets price as Kapton; the
    # stream is read anyway so a faster window would be charged its fabric.
    body = prices.sheet_price(wave_speed(windows[0].stream_excess, PUSH_ALTITUDE))
    for unit, part in ((p0, share), (r0, 1 - share)):
        lines = _unit_dollars(unit, prices, plates, packages, guidance, 0.0, body)
        seed_contents += part * sum(lines.values())
    seed_outlay = seed_price_per_kg * seed_mass + seed_contents
    first_return = share * p0.returned + (1 - share) * r0.returned
    if first_return <= 0.0:
        raise ValueError("the launch unit returns no sheets")
    pv_cost, pv_cargo = seed_outlay, 0.0
    count, cumulative, fleet = 1.0, 0.0, 0.0
    years, synodics = 0.0, 0
    harvest: Optional[int] = None
    tally: Dict[str, float] = {}
    tally_cargo = 0.0
    n = 1
    while True:
        if n == 1:
            years, synodics = windows[0].arrival_years, windows[0].arrival_multiple
            p_back, r_back = p0, r0
        else:
            flying = (n - 2) % n_windows
            years += windows[flying].period_years
            synodics += windows[flying].multiple
            p_back, r_back = units[flying]
        when = years - gap_years + delay_years
        if when > HORIZON_YEARS:
            break
        returned = count * (share * p_back.returned + (1 - share) * r_back.returned)
        cumulative += returned
        arrives = 1.0 - miss_fraction(cumulative, first_return)
        now = (n - 1) % n_windows
        p_unit, r_unit = units[now]
        growth = balance(p_unit, r_unit, arrives)
        caught = returned * arrives
        if harvest is None and synodics >= growth_synodic:
            harvest, fleet = n, caught
        harvesting = harvest is not None
        new = count * (min(growth.growth, 1.0) if harvesting else growth.growth)
        surplus = max(caught - new * growth.sheets_per_unit, 0.0) if harvesting else 0.0
        cost: Dict[str, float] = {}
        body = prices.sheet_price(wave_speed(windows[now].stream_excess, PUSH_ALTITUDE))
        for unit, part in (
            (p_unit, growth.push_share),
            (r_unit, 1 - growth.push_share),
        ):
            for key, value in _unit_dollars(
                unit, prices, plates, packages, guidance, lob, body
            ).items():
                cost[key] = cost.get(key, 0.0) + part * value * new
        delivery = delivery_unit(windows[now].stream_excess, design, laws)
        deliveries = surplus / delivery.sheets
        cost["delivery_lob"] = lob * delivery.climb * 1500.0e3 * deliveries
        cost["delivery_plate"] = prices.plate_curve(
            prices.earth_plate_share
        ).batch_cost(plates["push"], deliveries) * (delivery.plate + delivery.bag)
        cost["delivery_charge"] = prices.polyethylene * delivery.charge * deliveries
        plates["push"] += deliveries
        cargo = deliveries * delivery.cargo
        factor = schedule.factor(when)
        pv_cost += factor * sum(cost.values())
        pv_cargo += factor * cargo
        if harvest is not None and n == harvest + 6:
            tally, tally_cargo = cost, cargo
        count, share = new, growth.push_share
        n += 1
    lines = (
        {key: value / tally_cargo for key, value in tally.items()}
        if tally_cargo
        else {}
    )
    chain = chain_rate(windows, design)
    return Valuation(
        growth=chain.per_three_synodic,
        doubling=chain.doubling,
        steady_cost=float(sum(lines.values())),
        steady_lines=lines,
        break_even=pv_cost / pv_cargo if pv_cargo > 0 else None,
        seed_per_fleet_kg=seed_outlay / fleet if fleet > 0 else math.inf,
        fleet_multiple=fleet / first_return if first_return > 0 else 0.0,
    )


@dataclass(frozen=True)
class LoopOptimum:
    """The cheapest slug ratios on one chain.

    Attributes:
        design: The knobs at the optimum.
        valuation: Its valuation on the cheap seed.
        dear_break_even: Break-even on the dear seed (the objective).
    """

    design: Design
    valuation: Valuation
    dear_break_even: Optional[float]


def optimize_loop(
    windows: Sequence[LoopWindow],
    start: Design = START,
    prices: SolarDivePrices = ESTIMATE,
    seed_factor: float = 1.0,
    delay_years: float = 0.0,
) -> LoopOptimum:
    """Choose the push, departure and delivery slug ratios for the dear break-even.

    Nelder-Mead in log space inside :data:`_BOUNDS`, as
    :func:`src.solar_dive_growth.optimize_design` does.  ``seed_factor``
    scales both seed prices (a seed route's ratio, item 4).
    """
    low, high = (seed_factor * seed_price(windows[0], d) for d in (False, True))
    lows = np.log([_BOUNDS[k][0] for k in _KNOBS])
    highs = np.log([_BOUNDS[k][1] for k in _KNOBS])

    def design_at(x: np.ndarray) -> Design:
        clipped = np.exp(np.clip(x, lows, highs))
        return replace(start, **dict(zip(_KNOBS, (float(v) for v in clipped))))  # type: ignore[arg-type]

    def score(x: np.ndarray) -> float:
        penalty = float(np.sum(np.maximum(x - highs, 0.0) + np.maximum(lows - x, 0.0)))
        try:
            v = value_chain(
                windows, design_at(x), prices, high, delay_years=delay_years
            )
        except ValueError:
            return 1e6
        return (v.break_even or 1e5) * (1.0 + penalty)

    x0 = np.log([getattr(start, k) for k in _KNOBS])
    result = minimize(
        score,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400},
    )
    best = design_at(result.x)
    cheap = value_chain(windows, best, prices, low, delay_years=delay_years)
    dear = value_chain(windows, best, prices, high, delay_years=delay_years)
    return LoopOptimum(best, cheap, dear.break_even)


# --------------------------------------------------------------------------
# Item 5: the return family at a window's fixed endpoints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyMember:
    """One unpowered-plus-correction return between a window's fixed epochs.

    Attributes:
        encounter_jd: Jovian encounter (TDB JD).
        outbound_low: The outbound Lambert path.
        return_low: The return Lambert path.
        departure_excess: Earth departure excess (km/s).
        return_excess: Earth return excess (km/s).
        dsm: The correction it needs (km/s).
    """

    encounter_jd: float
    outbound_low: bool
    return_low: bool
    departure_excess: float
    return_excess: float
    dsm: float


def return_family(departure_jd: float, return_jd: float) -> List[FamilyMember]:
    """Every sampled encounter and path pair joining two fixed Earth epochs.

    The chain keeps the cheapest correction of these
    (:func:`src.two_wave_growth._cheapest_return`); here the whole sampled
    family is kept, so a plate ledger can choose among them.
    """
    dep_r, dep_v = _heliocentric_state("earth", departure_jd)
    ret_r, ret_v = _heliocentric_state("earth", return_jd)
    grid = np.linspace(
        departure_jd + _MANEUVER_ENCOUNTER_TOF_MIN_YEARS * _DAYS_PER_YEAR,
        return_jd - _MANEUVER_RETURN_TOF_MIN_YEARS * _DAYS_PER_YEAR,
        _MANEUVER_ENCOUNTER_SAMPLES,
    )
    jup_r, jup_v = _heliocentric_state("jupiter", grid)
    out = []
    for outbound_low in (True, False):
        for return_low in (True, False):
            for jd, jr, jv in zip(grid, jup_r, jup_v):
                geometry = _encounter_geometry(
                    departure_jd,
                    return_jd,
                    float(jd),
                    outbound_low,
                    return_low,
                    dep_r,
                    dep_v,
                    ret_r,
                    ret_v,
                    jr,
                    jv,
                )
                if geometry is None:
                    continue
                solution = _dsm_only_solution(geometry)
                out.append(
                    FamilyMember(
                        float(jd),
                        outbound_low,
                        return_low,
                        solution.earth_departure_vinf,
                        solution.earth_return_vinf,
                        solution.dsm,
                    )
                )
    return out


def one_batch_growth(
    window: LoopWindow,
    departure_excess: float,
    return_excess: float,
    dsm: float,
    design: Design,
) -> float:
    """Growth if both carriers flew one family member and its own wave pushed.

    A screening measure (item 5): one batch on one trajectory, departed by a
    stream at the member's own return excess along the window's stream
    direction, pushed by it at 400 km, paying ``dsm`` in methalox home.
    """
    law = StageLaws.load(bonds=design.bonds)
    flown = replace(
        window,
        push_stream=wave_speed(return_excess, PUSH_ALTITUDE),
        stream_excess=return_excess,
    )
    carrier = replace(window.rods, excess=departure_excess, return_burn=dsm)
    unit = fly_carrier(flown, carrier, design, law)
    return float(
        balance(unit, unit, 1.0 - MISS_FLOOR).growth if unit.returned > 0 else 0.0
    )


# --------------------------------------------------------------------------
# The report (make plate-jupiter-loop)
# --------------------------------------------------------------------------


def window_report(windows: Sequence[LoopWindow], design: Design) -> str:
    """Each window's streams, departure geometry and growth."""
    rows = []
    for w in windows:
        g = window_growth(w, design)
        rods = g.departure_unit.lines
        grow = g.push_unit.lines
        rows.append(
            [
                w.index,
                f"{w.multiple}S",
                f"{w.push_stream:.1f}",
                f"{wave_speed(w.stream_excess, DEPARTURE_ALTITUDE):.1f}",
                f"{w.growth.excess:.2f} / {w.rods.excess:.2f}",
                f"{rods['impact_angle']:.1f} / {rods['departure_closing']:.1f}",
                f"{rods['methalox_dv']:.3f}",
                f"{grow['departure_dv']:.2f} / {rods['departure_dv']:.2f}",
                f"{rods['departure_loss']:.2f} / {rods['departure_minutes']:.0f}",
                f"{w.growth.return_burn:.3f} / {w.rods.return_burn:.3f}",
                f"{g.growth:.2f}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "cycle",
            "onward",
            "growth stream (400 km)",
            "nozzle stream (600 km)",
            "excess G / R",
            "bowl angle / closing (R)",
            "apoapsis turn",
            "burn G / R",
            "loss / min (R)",
            "return burn G / R",
            "growth",
        ],
    )


def _row(label: str, optimum: LoopOptimum) -> List[str]:
    v, d = optimum.valuation, optimum.design
    return [
        label,
        f"{d.k_push:.2f} / {d.k_earth:.2f} / {d.k_delivery:.2f}",
        f"{v.growth:.2f}",
        f"{v.doubling:.2f}",
        f"{v.steady_cost:.0f}" if v.steady_lines else "no deliveries",
        f"{_money(v.break_even)} / {_money(optimum.dear_break_even)}",
        f"{v.seed_per_fleet_kg:.0f} at x{v.fleet_multiple:.0f}",
    ]


_HEADERS = [
    "case",
    "k push / dep / delivery",
    "growth per 3S",
    "doubling (yr)",
    "steady $/kg",
    "break-even cheap / dear",
    "seed $/fleet kg",
]


def headline_rows() -> Tuple[List[List[str]], Dict[str, LoopOptimum]]:
    """Each chain at its dollar optimum, at each departure rate."""
    rows, optima = [], {}
    for chain in CHAINS:
        windows = loop_windows(chain)
        start = START
        for g in DEPARTURE_GS:
            found = optimize_loop(windows, replace(start, departure_g=g))
            start = found.design
            optima[f"{chain}, {g:g} g"] = found
            rows.append(_row(f"{chain}, departure {g:g} g", found))
        high = HIGH_K
        rows.append(
            _row(
                f"{chain}, k 10 / 20 / 10 at 0.25 g",
                LoopOptimum(
                    high,
                    value_chain(
                        windows, high, seed_price_per_kg=seed_price(windows[0])
                    ),
                    value_chain(
                        windows, high, seed_price_per_kg=seed_price(windows[0], True)
                    ).break_even,
                ),
            )
        )
    return rows, optima


def seed_report(optima: Dict[str, LoopOptimum]) -> str:
    """The seed at the chain's 12-14 km/s, direct and on ADR 0059's loop (item 4)."""
    from src.growth_cost_inputs import LOOP_2YR

    rows = []
    for chain in CHAINS:
        windows = loop_windows(chain)
        w = windows[0]
        base = optima[f"{chain}, 0.25 g"]
        looped = optimize_loop(
            windows,
            base.design,
            seed_factor=LOOP_2YR.cheap_ratio,
            delay_years=LOOP_2YR.delay_years,
        )
        rows.append(
            [
                chain,
                f"{seed_excess(w):.2f}",
                f"{seed_price(w):.0f} / {seed_price(w, True):.0f}",
                f"{_money(base.valuation.break_even)} / {_money(base.dear_break_even)}",
                f"{_money(looped.valuation.break_even)} / {_money(looped.dear_break_even)}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "chain",
            "seed excess (km/s)",
            "seed $/kg cheap / dear",
            "direct: break-even",
            f"2-yr loop (x{LOOP_2YR.cheap_ratio:g}, +{LOOP_2YR.delay_years:g} yr)",
        ],
    )


def family_report(design: Design, chain: str = "flown") -> str:
    """Each window's selected return against the family's best on the plate ledger."""
    windows = loop_windows(chain)
    cycles = adaptive_two_wave_cycles(
        threshold_m_s=CHAINS[chain], split_days=PARKING_DAYS
    )
    rows = []
    seen = set()
    for w, cycle in zip(windows, cycles):
        if cycle.synodic_multiple in seen:
            continue
        seen.add(cycle.synodic_multiple)
        family = [
            m
            for m in return_family(cycle.departure_jd, cycle.return_jd)
            if m.dsm <= FAMILY_DSM_CAP
        ]
        scored = [
            (one_batch_growth(w, m.departure_excess, m.return_excess, m.dsm, design), m)
            for m in family
        ]
        selected = min(family, key=lambda m: m.dsm)
        best_growth, best = max(scored, key=lambda pair: pair[0])
        for label, g, m in (
            (
                "chain's (least correction)",
                one_batch_growth(
                    w,
                    selected.departure_excess,
                    selected.return_excess,
                    selected.dsm,
                    design,
                ),
                selected,
            ),
            ("best on the plate ledger", best_growth, best),
            ("fastest return", *max(scored, key=lambda pair: pair[1].return_excess)),
        ):
            rows.append(
                [
                    f"{cycle.synodic_multiple}S from cycle {cycle.index}",
                    label,
                    f"{m.departure_excess:.2f}",
                    f"{m.return_excess:.2f}",
                    f"{m.dsm:.3f}",
                    f"{g:.2f}",
                ]
            )
    return tabulate(
        rows,
        headers=[
            "window",
            "return",
            "departure excess",
            "return excess",
            "correction (km/s)",
            "one-batch growth",
        ],
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the Jupiter-only plate loop (the parent's S41).

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--family", action="store_true", help="the return family, item 5 (~5 min)"
    )
    args = parser.parse_args(argv)
    rows, optima = headline_rows()
    print(
        "The Jupiter-only chain on plates: gated rNPV at 10%/yr (95/85/90, no Sun gate),\n"
        "15 synodic periods of growth, dear-seed optimum\n"
        + tabulate(rows, headers=_HEADERS)
    )
    for chain in CHAINS:
        best = optima[f"{chain}, 0.25 g"]
        print(
            f"\n{chain} chain at its 0.25 g optimum\n"
            + window_report(loop_windows(chain), best.design)
        )
        print(
            "  $/kg at L1 by line: "
            + ", ".join(
                f"{k} {x:.1f}"
                for k, x in sorted(
                    best.valuation.steady_lines.items(), key=lambda kv: -kv[1]
                )
                if x >= 0.05
            )
        )
    print("\nThe seed (item 4)\n" + seed_report(optima))
    if args.family:
        print(
            "\nThe return family at each window's fixed epochs (item 5), corrections"
            f" <= {FAMILY_DSM_CAP:g} km/s\n"
            + family_report(optima["flown, 0.25 g"].design)
        )


if __name__ == "__main__":
    main()
