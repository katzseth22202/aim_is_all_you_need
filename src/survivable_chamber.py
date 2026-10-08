"""The survivable methane chamber through the growth ledger and cost book (ADR 0044).

The parent's S17 (``docs/survivable_chamber_asks_for_aim_repo.md`` there,
applying the impact sim @ ``69d1f40``): the 20 m^3 sphere's bonded overwrap
delaminates under the stopped rod's line blast, so the departure chamber is
bulged to 212 m^3 and wrapped dry.  It fires a 5 kg rod at 2 Hz behind a 20 kg
frozen-methane plug, on a 48.2 t wall (68.8 t maraging fallback), with a 4.9 t
(A/A* = 100) or 14.7 t (A/A* = 300) extension.  Its pitch is 3.5-24 kg a pulse.
:func:`src.chamber_isp.survivable_methane` builds the pairing, and its ``eta``
is the one that reproduces the impact sim's 906 / 971 kN s per pulse at 75 km/s
with the gate charged (:func:`src.chamber_isp.efficiency_for_impulse`): 0.488 at
A/A* = 100 and 0.539 at 300, the latter the old solved 0.538.

Everything flies behind the spray cup (ADR 0041/0042, film left to the cost
book), on the 20-day orbit, with the integrated fixed-direction finite-burn
loss charged and solved with the burn, as every chamber row since ADR 0032.
The impact sim's +3.1-4.8% assumed thrust held along the velocity; that is the
``steered`` sensitivity here, which a head-on chamber cannot fly.

Each chain takes about four minutes; the cases run in parallel worker
processes.  ``--ledger`` is the growth ledger (~20 min on four workers),
``--cost`` the cost book and seed (~25 min).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from astropy import units as u
from tabulate import tabulate

from src.chamber_departure import chamber_unit_mass
from src.chamber_isp import (
    HYDROGEN_5500K,
    SURVIVABLE_FALLBACK_WALL_MASS,
    SURVIVABLE_IMPULSE,
    SURVIVABLE_WALL_MASS,
    ChamberPairing,
    absolute_efficiency,
    efficiency_for_impulse,
    survivable_methane,
)
from src.finite_burn_loss import steered_loss
from src.growth_cost import ESTIMATE, PESSIMISTIC, SCENARIOS, DesignInputs, PriceBook
from src.growth_cost_inputs import design_inputs, seed_prices
from src.growth_cost_report import climbing_lob, hardware, headline
from src.growth_ledger import (
    DEFAULT_PARKING_DAYS,
    DEPARTURE_ALTITUDE,
    LAUNCH_UNIT,
    SOLVED_EFFICIENCY,
    SPRAY_CUP,
    CycleGrowth,
    PlateDesign,
    _orbit_loss,
    chain_growth,
    departure_at_altitude,
    parking_period,
    summarize_chain,
)
from src.lob_rise import OPERATING_RISE_SPEED, booster_growth
from src.seed_cost import (
    DESIGNS,
    STRIPPED_SHIPS,
    Design,
    DesignChain,
    _amortization,
    chains,
    design_chain,
    seed_excess_speed,
    seed_price_range,
)
from src.two_wave_growth import TwoWaveCycle

#: Pitch a pulse spends, between the wall-heat edges (parent S17).
SURVIVABLE_PITCH_RANGE = (3.5 * u.kg, 24.0 * u.kg)
#: The heavy end, which the matrix carries as it did 5.6 kg for the sphere.
SURVIVABLE_PITCH = SURVIVABLE_PITCH_RANGE[1]
AREA_RATIOS = (100, 300)
#: Shares of the A/A* = 300 chemistry ceiling for the matrix's other rows.
#: A/A* = 100 returns fewer bonds, and its ceiling is not sized.
CEILING_SHARES = (0.50, 0.70, 0.90, 1.00)
#: Worker processes; each holds a few hundred MB.
WORKERS = 4


def survivable_efficiency(area_ratio: int) -> float:
    """The ``eta`` that gives the impact sim's net impulse per pulse.

    Args:
        area_ratio: 100 or 300.

    Returns:
        ``eta`` at 75 km/s, gate charged.
    """
    return efficiency_for_impulse(
        SURVIVABLE_IMPULSE[area_ratio], survivable_methane(area_ratio)
    )


def survivable_design(
    area_ratio: int,
    pitch: u.Quantity = SURVIVABLE_PITCH,
    wall_mass: u.Quantity = SURVIVABLE_WALL_MASS,
    rods_split: int = 1,
    chambers: Optional[int] = None,
    efficiency: Optional[float] = None,
) -> Design:
    """The survivable chamber as a design the ledger and cost book fly.

    Args:
        area_ratio: 100 or 300.
        pitch: Pitch each 5 kg pulse spends; split chambers spend their share.
        wall_mass: Wall of one 5 kg chamber, or of all the split ones.
        rods_split: As in :func:`src.chamber_isp.survivable_methane`.
        chambers: Fly exactly this many chambers; None picks the best count.
        efficiency: ``eta``; the impact sim's by default.

    Returns:
        The design.
    """
    pairing = survivable_methane(area_ratio, wall_mass, rods_split)
    eta = survivable_efficiency(area_ratio) if efficiency is None else efficiency
    label = (
        f"CH4 AR{area_ratio}, {pitch.to_value(u.kg):g} kg"
        + (
            f", wall {wall_mass.to_value(u.t):g} t"
            if wall_mass != SURVIVABLE_WALL_MASS
            else ""
        )
        + (f", {rods_split}x{5 / rods_split:g} kg" if rods_split > 1 else "")
        + (f", n={chambers}" if chambers is not None else "")
    )
    return Design(label, pairing, eta, pitch / rods_split, chambers)


@dataclass(frozen=True)
class Case:
    """One ledger run.

    Attributes:
        group: The table it prints in.
        label: Row label.
        design: What departs.
        steered: Charge the steered loss instead of the fixed-direction one.
    """

    group: str
    label: str
    design: Design
    steered: bool = False


def _fly(case: Case, plate: PlateDesign = SPRAY_CUP) -> List[CycleGrowth]:
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    pairing = case.design.pairing
    assert pairing is not None and case.design.efficiency is not None
    loss = _orbit_loss(steered_loss, parking_period(flown[0])) if case.steered else None
    return chain_growth(
        flown,
        plate.efficiency,
        pairing,
        case.design.efficiency,
        case.design.pitch_ratio,
        loss_model=loss,
        max_slug_ratio=plate.max_slug_ratio,
        impactor_bond_energy=plate.impactor_bond_energy,
        film_per_impulse=plate.film_per_impulse,
        chambers=case.design.chambers,
    )


def _span(values: Sequence[float], fmt: str = ".0f") -> str:
    low, high = min(values), max(values)
    return (
        f"{low:{fmt}}"
        if f"{low:{fmt}}" == f"{high:{fmt}}"
        else f"{low:{fmt}}-{high:{fmt}}"
    )


def ledger_row(case: Case) -> Dict[str, object]:
    """Fly one case over the chain and summarise it (runs in a worker).

    Args:
        case: The case.

    Returns:
        The row.
    """
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    try:
        grown = _fly(case)
    except (RuntimeError, ValueError) as error:
        # A forced chamber count can burn past the loss table's hour.
        return {"group": case.group, "departure": case.label, "doubling yr": str(error)}
    pairing = case.design.pairing
    assert pairing is not None
    summary = summarize_chain(
        [c.period_years for c in flown], [g.growth for g in grown]
    )
    stacks = [g.departing_stack.to_value(u.t) for g in grown]
    deps = [g.departure for g in grown]
    shares = [
        (d.finite_burn_loss / _burn(c)).to_value(u.one) for c, d in zip(flown, deps)
    ]
    return {
        "group": case.group,
        "departure": case.label,
        "eta": case.design.efficiency,
        "doubling yr": summary.doubling_years,
        "annual": summary.annual_growth,
        "10yr step": summary.ten_year_stepwise,
        "G_0": grown[0].growth,
        "stack t": _span(stacks),
        "n": _span([d.chambers for d in deps]),
        "hw t": _span(
            [d.chambers * chamber_unit_mass(pairing).to_value(u.t) for d in deps], ".1f"
        ),
        "pulses": _span([d.pulses for d in deps]),
        "burn min": _span([d.burn_time.to_value(u.min) for d in deps]),
        "loss m/s": _span([d.finite_burn_loss.to_value(u.m / u.s) for d in deps]),
        "loss %": _span([100.0 * b for b in shares], ".1f"),
        "payload t": float(
            np.mean([d.delivered_net * s for d, s in zip(deps, stacks)])
        ),
    }


def _burn(cycle: TwoWaveCycle) -> u.Quantity:
    """The impulsive departure burn a cycle's unit flies from 600 km."""
    return departure_at_altitude(
        cycle.onward_burn * u.km / u.s, DEPARTURE_ALTITUDE, parking_period(cycle)
    ).burn


def _ledger_row_at(index: int) -> Dict[str, object]:
    """:func:`ledger_row` of ``ledger_cases()[index]`` (in a worker)."""
    return ledger_row(ledger_cases()[index])


def ledger_cases() -> List[Case]:
    """Every S17 ledger run: items 1-3, and the ceiling shares.

    Returns:
        The cases, grouped by table.
    """
    cases = [
        Case(
            "1. Survivable methane, solved",
            survivable_design(ar, p).label,
            survivable_design(ar, p),
        )
        for ar in AREA_RATIOS
        for p in SURVIVABLE_PITCH_RANGE
    ]
    pairing = survivable_methane(300)
    for share in CEILING_SHARES:
        eta = absolute_efficiency(pairing, share)
        cases.append(
            Case(
                "1b. A/A* = 300 at shares of the chemistry ceiling, 24 kg pitch",
                f"CH4 AR300 {share:.0%}",
                survivable_design(300, efficiency=eta),
            )
        )
    for ar in AREA_RATIOS:
        cases += [
            Case(
                "3. Redundancy, 24 kg pitch (n = count flown)",
                f"AR{ar}: 5 kg chambers, best count",
                survivable_design(ar),
            ),
            Case(
                "3. Redundancy, 24 kg pitch (n = count flown)",
                f"AR{ar}: 2.5 kg chambers (24.9 t each), best count",
                survivable_design(ar, wall_mass=49.8 * u.t, rods_split=2),
            ),
            Case(
                "4. Sensitivities, 24 kg pitch",
                f"AR{ar}: maraging fallback wall",
                survivable_design(ar, wall_mass=SURVIVABLE_FALLBACK_WALL_MASS),
            ),
            Case(
                "4. Sensitivities, 24 kg pitch",
                f"AR{ar}: steered loss (cannot fly)",
                survivable_design(ar),
                steered=True,
            ),
        ]
    return cases


_In = TypeVar("_In")
_Out = TypeVar("_Out")


def _in_workers(function: Callable[[_In], _Out], items: Sequence[_In]) -> List[_Out]:
    """``map`` over worker processes, in order.

    Pass indices, not designs: a pickled pairing is a copy, and the ledger
    tells hydrogen from methane by identity (``pairing is HYDROGEN_5500K``).
    """
    workers = min(WORKERS, os.cpu_count() or 1, len(items))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


def ledger_report() -> str:
    """The S17 growth-ledger tables.

    Returns:
        The tables.
    """
    rows = _in_workers(_ledger_row_at, range(len(ledger_cases())))
    out = []
    for group in dict.fromkeys(r["group"] for r in rows):
        picked = [
            {k: v for k, v in r.items() if k != "group"}
            for r in rows
            if r["group"] == group
        ]
        out.append(f"{group}\n" + tabulate(picked, headers="keys", floatfmt=".4g"))
    return "\n\n".join(out)


#: The sphere's chamber, wall plus extension, which the methane price is for.
_SPHERE_UNIT = 19.0 + 2.19


def cost_designs() -> List[Design]:
    """The designs the cost book prices: methalox, both survivable area
    ratios at both pitch edges, and solved hydrogen (wall unverified).

    Returns:
        The designs.
    """
    return (
        [DESIGNS[0]]
        + [
            survivable_design(ar, p)
            for ar in AREA_RATIOS
            for p in SURVIVABLE_PITCH_RANGE
        ]
        + [DESIGNS[4]]
    )


def _priced(index: int) -> Tuple[DesignChain, DesignInputs]:
    """The chain and cost inputs of ``cost_designs()[index]`` (in a worker)."""
    design = cost_designs()[index]
    return design_chain(design, SPRAY_CUP), design_inputs(design, SPRAY_CUP)


def _by_mass(prices: PriceBook, design: Design) -> PriceBook:
    """The book with the methane chamber priced in proportion to its mass."""
    assert design.pairing is not None
    scale = chamber_unit_mass(design.pairing).to_value(u.t) / _SPHERE_UNIT
    curve = prices.methane_chamber
    return replace(
        prices,
        label=f"{prices.label}, chamber x{scale:.2f}",
        methane_chamber=replace(
            curve, first=curve.first * scale, floor=curve.floor * scale
        ),
    )


def cost_report() -> str:
    """The cost book and seed tables behind the spray cup, survivable methane.

    Returns:
        The report.
    """
    designs = cost_designs()
    priced = _in_workers(_priced, range(len(designs)))
    inputs = [i for _, i in priced]
    books = [climbing_lob(b, OPERATING_RISE_SPEED) for b in SCENARIOS]
    low, high = seed_prices()
    out = [
        f"Spray cup, film in the book; lob x{booster_growth(OPERATING_RISE_SPEED):.4f} "
        f"(1.1 km/s climb, brake held back).  Seed ${low:.0f} / ${high:.0f} per kg; "
        "'a / b' is cheap / dear seed.  Hydrogen's wall is unverified (S18).",
        "Hardware per launch unit, cycle 0\n" + hardware(inputs),
    ]
    for prices in books:
        out.append(
            f"{prices.label}: 10% a year, 50% odds the cycle works (ADR 0040)\n"
            + headline(inputs, prices)
        )
    estimate = next(b for b in books if b.label == ESTIMATE.label)
    pessimistic = next(b for b in books if b.label == PESSIMISTIC.label)
    for book in (estimate, pessimistic):
        scaled = [
            (d, i)
            for d, i in zip(designs, inputs)
            if d.pairing not in (None, HYDROGEN_5500K)
        ]
        out.append(
            f"{book.label} with each methane chamber priced by its mass against the "
            f"sphere's {_SPHERE_UNIT:.1f} t\n"
            + "\n".join(
                headline([i], _by_mass(book, d)).splitlines()[-1] for d, i in scaled
            )
        )
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    out.append(
        "tab:seed_amortization behind the spray cup, stripped baseline\n"
        + _amortization(
            [c for c, _ in priced],
            STRIPPED_SHIPS[0],
            seed_price_range(seed_excess_speed(flown[0])),
        )
    )
    return "\n\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the S17 recompute.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ledger", action="store_true", help="the growth ledger")
    parser.add_argument("--cost", action="store_true", help="the cost book and seed")
    args = parser.parse_args(argv)
    both = not (args.ledger or args.cost)
    print(
        f"Survivable methane chamber (ADR 0044, parent S17) behind {SPRAY_CUP.label}, "
        f"{LAUNCH_UNIT:g} unit, {DEFAULT_PARKING_DAYS:g}-day orbit, gate charged, "
        "fixed-direction loss integrated and solved with the burn.  "
        + ", ".join(
            f"AR{ar}: eta {survivable_efficiency(ar):.3f}" for ar in AREA_RATIOS
        )
        + f" (sphere's solved {SOLVED_EFFICIENCY['CH4 7000 K']:g})."
    )
    if args.ledger or both:
        print()
        print(ledger_report())
    if args.cost or both:
        print()
        print(cost_report())


if __name__ == "__main__":
    main()
