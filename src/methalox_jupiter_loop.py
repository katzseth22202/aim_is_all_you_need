"""Milestone 1: the Jupiter-only plate loop leaving Earth on methalox (S63).

The parent's S63 (2026-10-29), from the child's three milestones: fly ADR
0075's Jupiter-only chain (:mod:`src.plate_jupiter_loop`) with the departure
bowl replaced by the growth ledger's methalox incumbent
(:func:`src.growth_ledger.best_methalox_departure`: Raptor 3 at 380 s, 0.018
kg of tank per kg of propellant, the engine count optimized, engines and
tanks expended).

1. **Push.** As ADR 0075: Kapton sheets overtake the lofted unit at 400 km
   and push it onto the 10-day parking orbit on the flat push plate, on its
   gas spring, at 2 Hz and 1 g.  There is no bowl to share the plate with,
   so it is nudged down and left at apoapsis.
2. **Apoapsis.** Raise periapsis to 600 km and turn the orbit so the
   periapsis velocity points along the methalox burn's axis, the departure
   asymptote turned back by the hyperbola's half-turn at 600 km.
3. **Departure.** Raptor 3s, steered along the velocity, from 600 km; the
   steered finite-burn loss on the 10-day orbit.
4. **Jupiter and home.** No departure wave means no role split: every ship
   flies the chain's growth trajectory, which returns ``PARKING_DAYS`` before
   the next departure, pays its correction in methalox and comes home as the
   next push wave.

Item 2 sweeps the departure excess on each window's fixed epochs
(:func:`src.plate_jupiter_loop.return_family`) and on a five-synodic clock.

Run with ``make methalox-loop``.
"""

import argparse
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from astropy import units as u
from tabulate import tabulate

from src.conic_kernel import half_turn_angle, hyperbolic_eccentricity
from src.finite_burn_loss import steered_loss
from src.growth_ledger import (
    METHALOX_TANK_FRACTION,
    RAPTOR3_MASS,
    best_methalox_departure,
)
from src.jovian_solar_dive_cycle import synodic_period
from src.plate_jupiter_loop import (
    _NO_SUN,
    START,
    LoopOptimum,
    LoopWindow,
    loop_windows,
    optimize_loop,
    return_family,
    seed_excess,
    window_growth,
)
from src.retrograde_return_legs import _powered_flyby_params
from src.seed_cost import STRIPPED_SHIPS, seed_payload
from src.solar_dive_growth import (
    _MU_EARTH,
    _R_EARTH,
    AVIONICS_FRACTION,
    CHARGE_MOUNT_FRACTION,
    DEPARTURE_ALTITUDE,
    FILM_PER_IMPULSE,
    MISS_FLOOR,
    PARKING_DAYS,
    PUSH_ALTITUDE,
    Design,
    Gate,
    StageLaws,
    UnitLedger,
    _money,
    balance,
    departure_burn,
    push_and_turn,
    wave_speed,
    wrap_pi,
)
from src.two_wave_growth import VE_METHALOX, adaptive_two_wave_cycles

#: The chains flown, as :data:`src.plate_jupiter_loop.CHAINS`.
CHAINS = ("3S only", "flown")
#: No bowl: only the push and delivery slug ratios are knobs.
KNOBS = ("k_push", "k_delivery")
#: Where the optimizer starts: ADR 0075's push, the plate left at apoapsis.
METHALOX_START = replace(START, shared_spring=False)
#: Synodic periods the slow-departure check flies (item 2).
SLOW_MULTIPLES = (3, 4, 5)
#: A Hohmann transfer's Earth departure excess to Jupiter (km/s), the
#: parent's 8.8 km/s.
HOHMANN_EXCESS = 8.8


def stream_excess(stream_at_400: float) -> float:
    """A wave's Earth excess from its speed at 400 km (km/s)."""
    escape = 2.0 * _MU_EARTH / (_R_EARTH + PUSH_ALTITUDE)
    return math.sqrt(max(stream_at_400**2 - escape, 0.0))


def thrust_axis(excess: float, aim_deg: float, push_axis_deg: float) -> float:
    """The periapsis burn axis that leaves along ``aim_deg``, nearer the push's.

    The asymptote lies a half-turn past the periapsis velocity, on either
    side; the side that turns the orbit least at apoapsis is kept.
    """
    mirror = math.degrees(
        half_turn_angle(
            hyperbolic_eccentricity(_MU_EARTH, _R_EARTH + DEPARTURE_ALTITUDE, excess)
        )
    )
    return min(
        (aim_deg + sign * mirror for sign in (1.0, -1.0)),
        key=lambda axis: abs(wrap_pi(math.radians(push_axis_deg - axis))),
    )


def _steered(burn: u.Quantity, seconds: u.Quantity) -> u.Quantity:
    return steered_loss(burn, seconds, PARKING_DAYS * u.day)


def fly_methalox(window: LoopWindow, design: Design, law: StageLaws) -> UnitLedger:
    """One launch unit pushed, departed on Raptor 3s, and returned as sheets.

    The ship flies ``window.growth`` (its excess, aim and correction).

    Args:
        window: The window.
        design: The knobs (the bowl's and the Sun's are unused).
        law: The stage laws.

    Returns:
        The ledger; ``departure_sheets`` is 0.

    Raises:
        ValueError: With ``Design.shared_spring``: there is no bowl.
    """
    if design.shared_spring:
        raise ValueError("a methalox departure has no bowl to share the push plate")
    carrier = window.growth
    axis = thrust_axis(carrier.excess, carrier.aim_deg, window.push_axis_deg)
    leg = push_and_turn(
        window.push_stream, window.push_axis_deg, axis, PARKING_DAYS, design, law
    )
    impulsive = departure_burn(carrier.excess, PARKING_DAYS)
    flown = best_methalox_departure(leg.stack * u.kg, impulsive * u.km / u.s, _steered)
    loss = float(flown.finite_burn_loss.to_value(u.km / u.s))
    spent = leg.stack * (1.0 - math.exp(-(impulsive + loss) / VE_METHALOX))
    film = FILM_PER_IMPULSE * leg.impulse
    mounts = CHARGE_MOUNT_FRACTION * leg.charge
    departing = flown.delivered_net * leg.stack - film - mounts
    avionics = AVIONICS_FRACTION * departing
    ship = departing - avionics
    returned = ship * math.exp(-carrier.return_burn / VE_METHALOX)
    lines: Dict[str, float] = dict(leg.lines)
    lines.update({key: 0.0 for key in _NO_SUN})
    lines.update(
        bowl=0.0,
        departure_charge=0.0,
        film_earth=film,
        earth_mounts=mounts,
        apoapsis_methalox=lines["methalox"],
        departure_methalox=spent,
        departure_tanks=METHALOX_TANK_FRACTION * spent,
        departure_engines=float(flown.engines),
        departure_engine_mass=flown.engines * float(RAPTOR3_MASS.to_value(u.kg)),
        departure_dv=impulsive + loss,
        departure_loss=loss,
        departure_minutes=float(flown.burn_time.to_value(u.min)),
        departing=departing,
        avionics=avionics,
        return_methalox=ship - returned,
        sheets=returned,
    )
    lines["methalox"] += spent + ship - returned
    return UnitLedger(leg.sheets, 0.0, returned, 0.0, lines)


def methalox_pair(
    window: LoopWindow, design: Design, law: StageLaws
) -> Tuple[UnitLedger, UnitLedger]:
    """Both roles are the one methalox ship (``plate_jupiter_loop.Fly``)."""
    unit = fly_methalox(window, design, law)
    return unit, unit


@lru_cache(maxsize=None)
def methalox_windows(chain: str = "3S only") -> Tuple[LoopWindow, ...]:
    """The chain's windows with every ship on the growth trajectory.

    The returning stream is the growth wave's, so the delivery and the
    sheets' fabric are read from it.
    """
    return tuple(
        replace(
            w,
            stream_excess=stream_excess(w.push_stream),
            stream_axis_deg=w.push_axis_deg,
            rods=w.growth,
        )
        for w in loop_windows(chain)
    )


def methalox_gates(window: LoopWindow) -> Tuple[Gate, ...]:
    """ADR 0075's no-Sun gates, the last one now the push and the Raptor burn.

    The parent puts the first push in its 90% gate, shared with the bowl; with
    no bowl to retire the gate is kept whole (conservative).
    """
    return (
        Gate("seed departs and flies by Jupiter", window.arrival_jupiter_years, 0.95),
        Gate(
            "sheets unfurl and arrive face-on",
            window.arrival_years - PARKING_DAYS / 365.25 - 1e-6,
            0.85,
        ),
        Gate("first 1 g push and methalox departure", window.arrival_years, 0.90),
    )


def optimize_methalox(chain: str = "3S only") -> LoopOptimum:
    """Milestone 1's dear-seed optimum on a chain."""
    windows = methalox_windows(chain)
    return optimize_loop(
        windows,
        METHALOX_START,
        fly=methalox_pair,
        knobs=KNOBS,
        gates=methalox_gates(windows[0]),
    )


@dataclass(frozen=True)
class SeedShips:
    """The seed in tonnes and stripped Starships.

    Attributes:
        tonnes: One launch unit's departing ship.
        excess: The seed's departure excess (km/s).
        ships: Stripped Starships at that excess, cheap (40 t) and dear (60 t).
    """

    tonnes: float
    excess: float
    ships: Tuple[float, float]


def seed_ships(optimum: LoopOptimum, window: LoopWindow) -> SeedShips:
    """The optimum's seed over what one stripped Starship sends."""
    excess = seed_excess(window)
    tonnes = optimum.valuation.seed_mass / 1e3
    carried = [
        float(seed_payload(ship, excess * u.km / u.s).payload.to_value(u.t))
        for ship in STRIPPED_SHIPS
    ]
    return SeedShips(tonnes, excess, (tonnes / carried[0], tonnes / carried[1]))


# --------------------------------------------------------------------------
# Item 2: slower departures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlowRow:
    """The best methalox ship in one departure-excess band.

    Attributes:
        multiple: Synodic periods between departure and return.
        band: The band's lower edge (km/s).
        departure_excess: The member's excess (km/s).
        return_excess: Its Earth return excess (km/s).
        stream: That return's speed at 400 km (km/s).
        dsm: Its correction (km/s).
        growth: One batch's growth flying it every cycle.
    """

    multiple: int
    band: float
    departure_excess: float
    return_excess: float
    stream: float
    dsm: float
    growth: float


def one_batch_methalox(
    window: LoopWindow,
    departure_excess: float,
    return_excess: float,
    dsm: float,
    design: Design,
) -> float:
    """Growth if every cycle flew one family member, pushed by its own return.

    The screening measure of
    :func:`src.plate_jupiter_loop.one_batch_growth`: the window's directions
    are kept, the speeds are the member's.
    """
    law = StageLaws.load(bonds=design.bonds)
    flown = replace(
        window,
        push_stream=wave_speed(return_excess, PUSH_ALTITUDE),
        growth=replace(window.growth, excess=departure_excess, return_burn=dsm),
    )
    try:
        unit = fly_methalox(flown, design, law)
    except (RuntimeError, ValueError):
        return 0.0
    return float(balance(unit, unit, 1.0 - MISS_FLOOR).growth)


def slow_departures(
    design: Design, band: float = 0.5, dsm_cap: float = 2.0
) -> List[SlowRow]:
    """The best member per excess band, on the first 3S window and longer clocks.

    The departure epoch is the first three-synodic cycle's; the return is
    ``multiple`` synodic periods later, less the parking gap.
    """
    cycle = adaptive_two_wave_cycles(threshold_m_s=0.0, split_days=PARKING_DAYS)[0]
    window = methalox_windows("3S only")[0]
    synodic_days = synodic_period(_powered_flyby_params()) / 86400.0
    rows = []
    for multiple in SLOW_MULTIPLES:
        back = (
            cycle.return_jd
            if multiple == cycle.synodic_multiple
            else cycle.departure_jd + multiple * synodic_days
        ) - PARKING_DAYS
        best: Dict[int, SlowRow] = {}
        for m in return_family(cycle.departure_jd, back):
            if m.dsm > dsm_cap:
                continue
            g = one_batch_methalox(
                window, m.departure_excess, m.return_excess, m.dsm, design
            )
            key = int(m.departure_excess // band)
            if key not in best or g > best[key].growth:
                best[key] = SlowRow(
                    multiple,
                    key * band,
                    m.departure_excess,
                    m.return_excess,
                    wave_speed(m.return_excess, PUSH_ALTITUDE),
                    m.dsm,
                    g,
                )
        rows += [best[k] for k in sorted(best)]
    return rows


# --------------------------------------------------------------------------
# The report (make methalox-loop)
# --------------------------------------------------------------------------


def window_report(chain: str, design: Design) -> str:
    """Each window's stream, burn, engines and growth."""
    rows = []
    for w in methalox_windows(chain):
        g = window_growth(w, design, fly=methalox_pair)
        m = g.push_unit.lines
        rows.append(
            [
                w.index,
                f"{w.multiple}S",
                f"{w.push_stream:.1f}",
                f"{w.growth.excess:.2f}",
                f"{m['methalox_dv']:.3f}",
                f"{m['departure_dv']:.2f} / {m['departure_loss'] * 1e3:.0f}",
                f"{m['departure_engines']:.0f} / {m['departure_minutes']:.0f}",
                f"{m['departure_methalox'] / 1e3:.0f}",
                f"{m['departing'] / 1e3:.0f}",
                f"{w.growth.return_burn:.3f}",
                f"{g.growth:.2f}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "cycle",
            "onward",
            "push stream (400 km)",
            "excess",
            "apoapsis turn",
            "burn km/s / loss m/s",
            "Raptors / min",
            "methalox (t)",
            "sent on (t)",
            "return burn",
            "growth",
        ],
    )


def headline(optima: Dict[str, LoopOptimum]) -> str:
    """The chains at their dear-seed optima, with the seed."""
    rows = []
    for chain, best in optima.items():
        v, d = best.valuation, best.design
        seed = seed_ships(best, methalox_windows(chain)[0])
        rows.append(
            [
                chain,
                f"{d.k_push:.2f} / {d.k_delivery:.2f}",
                f"{v.growth:.2f}",
                f"{v.doubling:.2f}",
                f"{v.steady_cost:.0f}" if v.steady_lines else "no deliveries",
                f"{_money(v.break_even)} / {_money(best.dear_break_even)}",
                f"{seed.tonnes:.0f} at {seed.excess:.2f}",
                f"{seed.ships[0]:.2f} / {seed.ships[1]:.2f}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "chain",
            "k push / delivery",
            "growth per 3S",
            "doubling (yr)",
            "steady $/kg",
            "break-even cheap / dear",
            "seed t at km/s",
            "stripped ships 40 / 60 t",
        ],
    )


def slow_report(design: Design) -> str:
    """Item 2's table."""
    return tabulate(
        [
            [
                f"{r.multiple}S",
                f"{r.band:.1f}",
                f"{r.departure_excess:.2f}",
                f"{r.return_excess:.2f}",
                f"{r.stream:.1f}",
                f"{r.dsm:.3f}",
                f"{r.growth:.2f}",
            ]
            for r in slow_departures(design)
        ],
        headers=[
            "clock",
            "band",
            "departure excess",
            "return excess",
            "stream at 400 km",
            "correction",
            "one-batch growth",
        ],
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print milestone 1 (the parent's S63 items 1 and 2).

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--no-slow", action="store_true", help="skip item 2's departure sweep"
    )
    args = parser.parse_args(argv)
    optima = {chain: optimize_methalox(chain) for chain in CHAINS}
    print(
        "Milestone 1: the Jupiter-only plate loop on a Raptor 3 departure, gated\n"
        "rNPV at 10%/yr (95/85/90, no Sun gate), 15 synodic periods of growth,\n"
        "dear-seed optimum\n" + headline(optima)
    )
    for chain, best in optima.items():
        print(f"\n{chain} chain at its optimum\n" + window_report(chain, best.design))
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
    if not args.no_slow:
        print(
            "\nSlower departures (item 2): the best ship per 0.5 km/s band of"
            " departure excess,\nfrom the first 3S window's departure, every"
            " cycle flying it, corrections <= 2 km/s\n"
            + slow_report(optima["3S only"].design)
        )


if __name__ == "__main__":
    main()
