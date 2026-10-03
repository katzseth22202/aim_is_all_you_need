"""Report for the growth cost model (ADR 0037): ``make growth-cost``.

Prints the outputs the parent asked for (``docs/growth_cost_for_parent.md``,
asks G1-G4): the hardware each launch unit carries, the steady-state cost per
kilogram at L1 by line, break-even sale prices and value per seed dollar
under the stepped rate and the flat comparisons, the chamber and package
prices at which a chamber matches methalox, plate learning, and one-line
sensitivities.  Building the chains takes several minutes.
"""

import argparse
from dataclasses import replace
from typing import Callable, List, Optional, Sequence, Tuple

from scipy.optimize import brentq
from tabulate import tabulate

from src.growth_cost import (
    ESTIMATE,
    PESSIMISTIC,
    SCENARIOS,
    Departure,
    DesignInputs,
    DiscountSchedule,
    HalvingPrice,
    PriceBook,
    RouteSeed,
    break_even_price,
    legacy_prices,
    route_break_even,
    run_program,
    steady_state_cost,
    value_per_seed_dollar,
)
from src.growth_cost_inputs import design_inputs, seed_prices
from src.harvest import PLATE_PRICES, solve_rate
from src.learning_curve import LearningCurve
from src.seed_cost import DESIGNS

#: The late rate after the first growth return (ASKS G2), and the two higher
#: ones the companion sweeps because one measured cycle is a thin proof.
LATE_RATES = (0.10, 0.15, 0.20)
EARLY_RATE = 0.30
FLAT_RATES = (0.076, 0.30)
LINES = (
    "lob",
    "plate",
    "plate_spray",
    "argon",
    "fleet",
    "departure_hw",
    "chamber_spray",
    "tanks",
    "propellant",
    "cryostats",
    "pulse_consumables",
    "film",
)

#: Seed routes (ADR 0039), each relative to the direct route on its own ship.
#: Empty: methalox routes never pay; SEP wins are unverified (ADR 0039).
SEED_ROUTES: Tuple[RouteSeed, ...] = ()


def stepped(inputs: DesignInputs, late: float = LATE_RATES[0]) -> DiscountSchedule:
    """30% until the design's first growth return, ``late`` after (ask G2).

    Args:
        inputs: The design.
        late: The rate once the cycle is proven.

    Returns:
        The schedule.
    """
    return DiscountSchedule.stepped(EARLY_RATE, late, inputs.proof_years)


def _fmt(price: Optional[float]) -> str:
    return ">3000" if price is None else f"{price:.0f}"


def _pair(values: Sequence[Optional[float]]) -> str:
    return " / ".join(_fmt(v) for v in values)


def _break_evens(
    inputs: DesignInputs,
    prices: PriceBook,
    schedule: DiscountSchedule,
    steady: bool = True,
) -> str:
    return _pair(
        [break_even_price(inputs, prices, s, schedule, steady) for s in seed_prices()]
    )


def validation(designs: Sequence[DesignInputs]) -> str:
    """Growth uncharged, plate flat at $114/kg: ``tab:seed_return`` at $500."""
    prices = legacy_prices(PLATE_PRICES["learned"])
    rows = []
    for inputs in designs:
        row: List[str] = [inputs.label]
        for steady in (False, True):
            irrs = []
            for seed in seed_prices():

                def value(rate: float, s: float = seed, held: bool = steady) -> float:
                    flat = DiscountSchedule.flat(rate)
                    return value_per_seed_dollar(inputs, prices, s, 500.0, flat, held)

                irr = solve_rate(value, low=1.0e-6)
                irrs.append("none" if irr is None else f"{100 * irr:.0f}")
            row.append(" / ".join(irrs))
        rows.append(row)
    return tabulate(rows, ["Design", "Liquidation IRR %", "Steady IRR %"])


def hardware(designs: Sequence[DesignInputs]) -> str:
    """Per launch unit on cycle 0, and the rods and packages on every cycle."""
    rows = []
    for inputs in designs:
        c = inputs.cycles[0]
        rows.append(
            [
                inputs.label,
                c.departure_units,
                f"{c.tanks / 1e3:.1f}",
                f"{c.gas / 1e3:.0f}",
                f"{c.argon / 1e3:.0f}",
                f"{(c.onward_puffsats + c.onward_rods) / 1e3:.0f}",
                f"{c.consumed / 1e3:.1f}",
                f"{c.growth:.2f}",
                " ".join(f"{cyc.pulses:.0f}" for cyc in inputs.cycles),
            ]
        )
    return tabulate(
        rows,
        [
            "Design",
            "Units",
            "Tanks t",
            "Propellant t",
            "Argon t",
            "Onward t",
            "Consumed t",
            "G_0",
            "Pulses (= rods) per cycle",
        ],
    )


def headline(designs: Sequence[DesignInputs], prices: PriceBook) -> str:
    """Steady $/kg by line; stepped break-evens and value per seed dollar."""
    rows = []
    for inputs in designs:
        total, lines = steady_state_cost(inputs, prices)
        schedule = stepped(inputs)
        values = [
            " / ".join(
                f"{value_per_seed_dollar(inputs, prices, s, p, schedule):.1f}"
                for s in seed_prices()
            )
            for p in (500.0, 200.0)
        ]
        rows.append(
            [inputs.label, f"{total:.0f}"]
            + [f"{lines.get(k, 0.0):.1f}" for k in LINES]
            + [
                _break_evens(inputs, prices, schedule, steady=False),
                _break_evens(inputs, prices, schedule),
            ]
            + values
        )
    return tabulate(
        rows,
        ["Design", "Steady $/kg"]
        + list(LINES)
        + ["BE liquid.", "BE steady", "Value @500", "Value @200"],
    )


def rates(designs: Sequence[DesignInputs]) -> str:
    """Steady break-even under the stepped rates and the flat comparisons."""
    rows = []
    for inputs in designs:
        row = [inputs.label]
        row += [
            _break_evens(inputs, ESTIMATE, stepped(inputs, late)) for late in LATE_RATES
        ]
        row += [
            _break_evens(inputs, ESTIMATE, DiscountSchedule.flat(r)) for r in FLAT_RATES
        ]
        row += [
            _break_evens(
                inputs, legacy_prices(PLATE_PRICES["learned"]), stepped(inputs)
            )
        ]
        rows.append(row)
    return tabulate(
        rows,
        ["Design"]
        + [f"30%->{late:.0%}" for late in LATE_RATES]
        + [f"flat {r:.1%}" for r in FLAT_RATES]
        + ["30%->10%, growth uncharged"],
    )


def _matching(
    designs: Sequence[DesignInputs], vary: Callable[[float], PriceBook], top: float
) -> List[Tuple[str, Optional[float]]]:
    """The price at which each chamber design's steady $/kg equals methalox's."""
    methalox = next(d for d in designs if d.departure is Departure.METHALOX)
    out = []
    for inputs in designs:
        if inputs is methalox:
            continue

        def gap(price: float, chamber: DesignInputs = inputs) -> float:
            book = vary(price)
            return (
                steady_state_cost(chamber, book)[0]
                - steady_state_cost(methalox, book)[0]
            )

        found = brentq(gap, 0.0, top) if gap(0.0) < 0.0 < gap(top) else None
        out.append((inputs.label, found))
    return out


def matching(designs: Sequence[DesignInputs]) -> str:
    """Flat chamber price and package price at which a chamber matches methalox."""

    def chamber(price: float) -> PriceBook:
        flat = LearningCurve.flat(price)
        return replace(ESTIMATE, methane_chamber=flat, hydrogen_chamber=flat)

    def package(price: float) -> PriceBook:
        return replace(ESTIMATE, package=LearningCurve.flat(price))

    chambers = dict(_matching(designs, chamber, 1.0e9))
    packages = dict(_matching(designs, package, 1.0e5))
    rows = [
        [label, _money(chambers[label]), _money(packages[label], unit=1.0)]
        for label in chambers
    ]
    return tabulate(rows, ["Design", "Chamber, flat", "Package, flat"])


def _money(value: Optional[float], unit: float = 1.0e6) -> str:
    if value is None:
        return "never"
    return f"${value / 1e6:.1f}M" if unit == 1.0e6 else f"${value:,.0f}"


def plate_learning(designs: Sequence[DesignInputs]) -> str:
    """Plates built before a plate costs under $10M; plates built by liquidation."""
    firsts, learning = (20.0e6, 30.0e6, 50.0e6, 78.0e6), (0.85, 0.80, 0.75, 0.70)
    grid = [
        [f"${first / 1e6:.0f}M"]
        + [f"{LearningCurve(first, 6.0e6, r).units_to(10.0e6):,.0f}" for r in learning]
        for first in firsts
    ]
    built = [
        [
            inputs.label,
            f"{run_program(inputs, ESTIMATE, 337.0, 500.0, False).plates:.0f}",
        ]
        for inputs in designs
    ]
    return (
        tabulate(grid, ["First plate"] + [f"{r:.0%}" for r in learning])
        + "\n\n"
        + tabulate(built, ["Design", "Plates built by the harvest"])
    )


def packages(designs: Sequence[DesignInputs]) -> str:
    """Package price over volume or time, against the flat $100."""
    cases = [
        ("Flat $100", replace(ESTIMATE, package=LearningCurve.flat(100.0))),
        ("Wright 80%, $10 floor (Estimate)", ESTIMATE),
        (
            "Wright 70%, $10 floor",
            replace(ESTIMATE, package=LearningCurve(100.0, 10.0, 0.7, 1.0e5)),
        ),
        (
            "Halves every 3 yr, $10 floor",
            replace(ESTIMATE, package=HalvingPrice(100.0, 10.0, 3.0)),
        ),
        ("Free packages (bound)", replace(ESTIMATE, package=LearningCurve.flat(0.0))),
    ]
    return _sweep(designs, cases)


def sensitivities(designs: Sequence[DesignInputs]) -> str:
    """One line at a time from the Estimate (ask output 10), and the lob band."""
    cases = [
        ("Estimate", ESTIMATE),
        ("Lob $5/kg", replace(ESTIMATE, lob=5.0)),
        ("Lob $10/kg", replace(ESTIMATE, lob=10.0)),
        ("Lob $15/kg", replace(ESTIMATE, lob=15.0)),
        ("Lob $50/kg", replace(ESTIMATE, lob=50.0)),
        ("Argon $5/kg", replace(ESTIMATE, argon=5.0)),
        ("Package $30", replace(ESTIMATE, package=LearningCurve.flat(30.0))),
        ("Package $300", replace(ESTIMATE, package=LearningCurve.flat(300.0))),
        ("Package $1000", replace(ESTIMATE, package=LearningCurve.flat(1000.0))),
        ("Package $3000", replace(ESTIMATE, package=LearningCurve.flat(3000.0))),
        ("Plate spray $8M", replace(ESTIMATE, plate_spray=8.0e6)),
        ("Orion film (4%)", replace(ESTIMATE, film_fraction=0.04)),
        ("Chamber spray $5M", replace(ESTIMATE, chamber_spray=5.0e6)),
        ("Cryostats $1000/kg", replace(ESTIMATE, cryostats=1000.0)),
        ("Plugs $10, pitch $5", replace(ESTIMATE, plugs=10.0, pitch=5.0)),
        ("Paper's $20/kg fleet", replace(ESTIMATE, fleet_flat=20.0)),
        ("Pessimistic", PESSIMISTIC),
    ]
    return _sweep(designs, cases)


def _sweep(
    designs: Sequence[DesignInputs], cases: Sequence[Tuple[str, PriceBook]]
) -> str:
    rows = []
    for name, prices in cases:
        rows.append(
            [name]
            + [f"{steady_state_cost(d, prices)[0]:.0f}" for d in designs]
            + [_break_evens(d, prices, stepped(d)) for d in designs]
        )
    labels = [d.label for d in designs]
    return tabulate(rows, ["Case"] + labels + [f"BE {label}" for label in labels])


def seed_routes(
    designs: Sequence[DesignInputs], routes: Sequence[RouteSeed] = SEED_ROUTES
) -> str:
    """Steady break-even, stepped rate, with the seed flown on each route.

    Args:
        designs: The designs.
        routes: The routes; each applies to its own ship's seed price.

    Returns:
        A table: direct, then one column per route.
    """
    if not routes:
        return (
            "(none: methalox assists never pay and SEP routes are unverified; "
            "the seed flies direct, ADR 0039)"
        )
    prices = seed_prices()
    rows = []
    for inputs in designs:
        row = [inputs.label, _break_evens(inputs, ESTIMATE, stepped(inputs))]
        for route in routes:
            row.append(
                _fmt(
                    route_break_even(
                        inputs,
                        ESTIMATE,
                        prices[1 if route.dear else 0],
                        route,
                        stepped,
                    )
                )
            )
        rows.append(row)
    headers = ["Design", "direct (cheap / dear)"]
    headers += [f"{r.label} ({'dear' if r.dear else 'cheap'})" for r in routes]
    return tabulate(rows, headers)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the growth cost model's report.

    Args:
        argv: Command-line arguments; ``--quick`` skips the sweeps.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="headline tables only")
    args = parser.parse_args(argv)
    designs = [design_inputs(d) for d in DESIGNS]
    low, high = seed_prices()
    print(
        f"Seed ${low:.0f} / ${high:.0f} per kg. Entries 'a / b' are cheap / dear seed."
    )
    print("Break-even (BE) is the L1 sale price, $/kg, that repays the seed and every")
    print(
        "growth cost. Steady $/kg is undiscounted, seed excluded, a lap into steady state.\n"
    )
    print("1. Validation: growth uncharged, plate $114/kg flat (tab:seed_return)")
    print(validation(designs))
    print("\n2. Hardware per launch unit, cycle 0")
    print(hardware(designs))
    for prices in SCENARIOS:
        print(
            f"\n3. {prices.label}: stepped rate 30% -> 10% at the first growth return"
        )
        print(headline(designs, prices))
    if args.quick:
        return
    pick = [designs[0], designs[2], designs[4]]
    print("\n4. Steady-state break-even by cost of capital (Estimate)")
    print(rates(designs))
    print(
        "\n5. Flat price at which a chamber's steady $/kg equals methalox's (Estimate)"
    )
    print(matching(designs))
    print("\n6. Plates built before the price falls below $10M ($6M floor)")
    print(plate_learning(designs))
    print("\n7. Package price over volume or time: steady $/kg and stepped BE")
    print(packages(pick))
    print("\n8. One line at a time from the Estimate: steady $/kg and stepped BE")
    print(sensitivities(pick))
    print("\n9. Seed routes (ADR 0039): steady BE, stepped 30% -> 10%, Estimate")
    print(seed_routes(designs))


if __name__ == "__main__":
    main()
