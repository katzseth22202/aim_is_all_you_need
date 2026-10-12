"""Slush hydrogen for the hydrogen chamber's charge (ADR 0051).

Normal-boiling-point liquid hydrogen is 70.8 kg/m^3 and boils 0.3% a day over
the 20-day hold, about 6% of what is launched.  Slush at 50% solid, made on the
ground by freeze-thaw, is 15% denser and carries 18% more heat sink (NASA Lewis
K-Site program, 1990-94).  Melting its solid and warming the liquid from the
13.8 K triple point absorbs what would otherwise boil it, so the same
insulation loses nothing over a 20-day hold (good for about 60 days).  The tank
scales with volume and shrinks by the density ratio.  The chamber, the burn and
everything else are unchanged.

The 3000 K chamber (:data:`src.chamber_isp.HYDROGEN_3000K`, ADR 0052) is a
study point: NTR-tested temperature, equilibrium energy, wall scaled by peak
pressure, efficiency guessed.

The report flies the hydrogen chambers behind the spray cup, with the lob
climbing and the Estimate book, as the survivable-chamber and hardware-recovery
reports do.  It sets slush against liquid with and without recovered tanks
(ADR 0050).  Two sensitivities: cryostats at 5% instead of 3%, and $2/kg more
for slush hydrogen.  Both are guesses: no flight-scale slush system exists.

Run ``make slush-hydrogen`` (a few minutes).
"""

from dataclasses import replace
from typing import List, Tuple

from tabulate import tabulate

from src.chamber_isp import (
    HYDROGEN_3000K,
    HYDROGEN_5500K,
    SLUSH_HYDROGEN,
    stored_as,
)
from src.growth_cost import (
    ESTIMATE,
    DesignInputs,
    PriceBook,
    recovered,
    steady_state_cost,
)
from src.growth_cost_inputs import design_inputs, seed_prices
from src.growth_cost_report import _break_evens, climbing_lob, risked
from src.growth_ledger import SOLVED_EFFICIENCY
from src.hardware_recovery import HYDROGEN_RECOVERY
from src.lob_rise import OPERATING_RISE_SPEED
from src.seed_cost import DESIGNS, Design
from src.survivable_chamber import PLATE

#: Slush priced this much above liquid hydrogen, $/kg, in the sensitivity.
SLUSH_PREMIUM = 2.0
#: The 3000 K chamber's energy efficiency: 85% of ideal exhaust velocity, the
#: convention of ADRs 0045 and 0049.  A guess; nothing has solved it.
EFFICIENCY_3000K = 0.85**2


def designs() -> List[Design]:
    """The 5500 K chamber as liquid, slush, and slush with 5% cryostats, at its
    solved efficiency; the 3000 K chamber as liquid and slush at 85% of ideal
    velocity, and as slush at the 5500 K chamber's solved efficiency."""
    eta = SOLVED_EFFICIENCY[HYDROGEN_5500K.name]
    slush = stored_as(HYDROGEN_5500K, SLUSH_HYDROGEN)
    heavy = stored_as(
        HYDROGEN_5500K,
        replace(SLUSH_HYDROGEN, name="slush 50%, 5% cryostats", cryostat_fraction=0.05),
    )
    cool = stored_as(HYDROGEN_3000K, SLUSH_HYDROGEN)
    return [
        DESIGNS[4],
        Design("Hydrogen, slush", slush, eta),
        Design("Hydrogen, slush, 5% cryostats", heavy, eta),
        Design("H2 3000 K", HYDROGEN_3000K, EFFICIENCY_3000K),
        Design("H2 3000 K, slush", cool, EFFICIENCY_3000K),
        Design(f"H2 3000 K, slush, eta {eta:g}", cool, eta),
    ]


def _growth(inputs: DesignInputs) -> float:
    product = 1.0
    for cycle in inputs.cycles:
        product *= cycle.growth
    return float(product ** (1.0 / len(inputs.cycles)))


def _row(label: str, inputs: DesignInputs, prices: PriceBook) -> List[str]:
    cycle = inputs.cycles[0]
    weakest = min(c.growth for c in inputs.cycles)
    # A fleet held level reinvests 1/G of each return; at G <= 1 there is
    # nothing left to deliver, and the steady-state arithmetic means nothing.
    if weakest > 1.0:
        cost, _ = steady_state_cost(inputs, prices)
        priced = [f"{cost:.1f}", _break_evens(inputs, prices, risked(inputs))]
    else:
        priced = ["does not grow", "-"]
    return [
        label,
        f"{cycle.tanks / 1e3:.1f}",
        f"{cycle.gas / 1e3:.1f}",
        f"{cycle.cryostats / 1e3:.1f}",
        f"{_growth(inputs):.3f}",
        f"{weakest:.3f}",
    ] + priced


def rows(book: PriceBook) -> List[List[str]]:
    """One row per storage, with and without recovered tanks.

    Args:
        book: The price book.

    Returns:
        The rows.
    """
    out = []
    priced: List[Tuple[Design, DesignInputs]] = [
        (d, design_inputs(d, PLATE)) for d in designs()
    ]
    for design, inputs in priced:
        for label, chain in (
            (design.label, inputs),
            (f"{design.label}, tanks recovered", recovered(inputs, HYDROGEN_RECOVERY)),
        ):
            out.append(_row(label, chain, book))
    slush = priced[1][1]
    dear = replace(book, hydrogen=book.hydrogen + SLUSH_PREMIUM)
    out.append(_row(f"Hydrogen, slush, +${SLUSH_PREMIUM:g}/kg", slush, dear))
    return out


def report() -> str:
    """The slush comparison, Estimate book, spray cup, lob climbing.

    Returns:
        The report.
    """
    book = climbing_lob(ESTIMATE, OPERATING_RISE_SPEED)
    low, high = seed_prices()
    return (
        f"{book.label}, spray cup, lob climbing; 10% a year, 50% odds. Seed "
        f"${low:.0f} / ${high:.0f} per kg; BE 'a / b' is cheap / dear seed.\n"
        "Tank, gas launched and cryostats are cycle 0, per launch unit.\n\n"
        + tabulate(
            rows(book),
            [
                "Design",
                "Tank t",
                "H2 launched t",
                "Cryostat t",
                "Growth (geo.)",
                "Weakest cycle",
                "Steady $/kg",
                "BE steady",
            ],
        )
    )


def main() -> None:
    """Print the slush-hydrogen report."""
    print(report())


if __name__ == "__main__":
    main()
