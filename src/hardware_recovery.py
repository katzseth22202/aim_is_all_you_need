"""Spent departure hardware cut up in flight and flown back as PuffSat mass (ADR 0050).

After the departure burn the empty tanks and the expended chamber ride on toward
Jupiter with the payload anyway.  A processor carried in the stack cuts and
packages them during the three-year cruise, and they come back as part of the
next return.  :func:`src.growth_cost.recovered` is the arithmetic; this module
runs it on the paper's chamber designs behind the spray cup, the book the
survivable-chamber report uses (ADR 0044), and sets the recycled mass against
the atmosphere's natural input.

**Where the recovered mass can go.**  Growth PuffSats are water that disperses
into gas before the plate, so solid pieces cannot simply join them.  The
default (``rods only``) lets recovered hardware stand in only for departure
rods, which hit the chamber at 60-75 km/s and vaporize.  ``rods + plate`` also
lets it be growth-wave impactors that spread water PuffSats over a plate met at
44 km/s or faster.  That is a design option, not yet a modeled one.

**Materials (ADR 0050).**  Tanks are carbon fibre in PEEK with a thin steel
liner; aluminium is out, because recycled aluminium reaches the meteoric
background at a handful of departures a year.  The methane chamber is Cr-Mo
steel and is recycled.  The hydrogen chamber's liner is copper, which nature
supplies at 2 t a year, so the hydrogen chamber is **not** recycled; its tanks
alone cover its rods in most cycles.

**Exhaust.**  Every recycled kilogram is eventually fired, and most departure
exhaust is slower than orbital speed and falls back (ADR 0050 brackets the
falling share at 0.6-1).  The report gives recycled tonnes per year per seed
and how many seeds' fleets reach each benchmark if all of it falls back:

* chamber steel against meteoric iron, 2,477 t a year injected (Schulz et al.
  2025, arXiv 2510.21328, Table 5);
* tank carbon against today's rocket soot, about 1,000 t a year (Maloney et
  al. 2022).  That bound treats every carbon atom as soot, which overstates it;
  the design's methane and polyethylene rods already put far more carbon in
  the exhaust than the tanks add.

Run ``make hardware-recovery`` (several minutes: the survivable chain is flown
in worker processes).
"""

from typing import List, Optional, Sequence, Tuple

from tabulate import tabulate

from src.growth_cost import (
    ESTIMATE,
    DesignInputs,
    HardwareRecovery,
    PriceBook,
    recovered,
    steady_exhaust,
    steady_state_cost,
)
from src.growth_cost_inputs import seed_prices
from src.growth_cost_report import _break_evens, climbing_lob, risked
from src.lob_rise import OPERATING_RISE_SPEED
from src.seed_cost import DESIGNS, Design
from src.survivable_chamber import (
    HYDROGEN_BULGED_DESIGN,
    SURVIVABLE_PITCH,
    _by_mass,
    _in_workers,
    _priced,
    cost_designs,
    survivable_design,
)

#: ADR 0046's scenario, kept: 90% of the tanks, 60% of the chamber, a 5 t
#: processor bought for $1M on every growth launch.  All are assumptions.
METHANE_RECOVERY = HardwareRecovery(tank_share=0.9, chamber_share=0.6)
#: Hydrogen recycles its tanks only: its chamber liner is copper.
HYDROGEN_RECOVERY = HardwareRecovery(tank_share=0.9, chamber_share=0.0)
#: Natural meteoric iron injected into the atmosphere, t a year.
METEORIC_IRON = 2477.0
#: Iron in the Cr-Mo chamber steel.
STEEL_IRON = 0.97
#: Rocket soot emitted today, t a year (Maloney et al. 2022).
ROCKET_SOOT = 1000.0
#: Carbon in carbon fibre in PEEK: about 95% in the fibre, 79% in the resin.
COMPOSITE_CARBON = 0.88


def _designs() -> List[Design]:
    """The survivable methane chamber at A/A* = 100 (ADR 0055's lead; within
    0.1 yr of the other nozzles on the finite-rate impulse, ADR 0056), solved
    hydrogen in the sphere, and the bulged hydrogen chamber."""
    return [
        survivable_design(100, SURVIVABLE_PITCH),
        DESIGNS[4],
        HYDROGEN_BULGED_DESIGN,
    ]


def _geometric_growth(inputs: DesignInputs) -> float:
    product = 1.0
    for cycle in inputs.cycles:
        product *= cycle.growth
    return float(product ** (1.0 / len(inputs.cycles)))


def _seeds(budget: float, tonnes: float) -> str:
    return "-" if tonnes <= 0.0 else f"{budget / tonnes:.0f}"


def recovery_rows(
    designs: Sequence[Tuple[Design, DesignInputs]], book: PriceBook
) -> List[List[str]]:
    """One row per design and recovery case.

    Args:
        designs: Each design with its chain, as flown without recovery.
        book: The price book, before the methane chamber is priced by mass.

    Returns:
        The rows.
    """
    rows = []
    for design, inputs in designs:
        methane = design.pairing is not None and not design.pairing.hydrogen
        prices = _by_mass(book, design) if methane else book
        base = METHANE_RECOVERY if methane else HYDROGEN_RECOVERY
        cases: List[Tuple[str, Optional[HardwareRecovery]]] = [
            ("none", None),
            ("rods only", base),
            (
                "rods + plate",
                HardwareRecovery(
                    base.tank_share, base.chamber_share, base.processor_mass, True
                ),
            ),
        ]
        for name, recovery in cases:
            chain = inputs if recovery is None else recovered(inputs, recovery)
            cost, _ = steady_state_cost(chain, prices)
            flow = steady_exhaust(chain, prices)
            steel = flow.recycled_chambers / 1.0e3 * STEEL_IRON
            carbon = flow.recycled_tanks / 1.0e3 * COMPOSITE_CARBON
            rows.append(
                [
                    design.label,
                    name,
                    f"{_geometric_growth(chain):.3f}",
                    f"{cost:.1f}",
                    _break_evens(chain, prices, risked(chain)),
                    f"{flow.launches:.2f}",
                    f"{flow.recycled_tanks / 1.0e3:.1f}",
                    f"{flow.recycled_chambers / 1.0e3:.1f}",
                    _seeds(METEORIC_IRON, steel),
                    _seeds(ROCKET_SOOT, carbon),
                ]
            )
    return rows


def report() -> str:
    """The recovery table behind the spray cup, lob climbing, Estimate book.

    Returns:
        The report.
    """
    designs = _designs()
    names = [d.label for d in cost_designs()]
    priced = _in_workers(_priced, [names.index(d.label) for d in designs])
    book = climbing_lob(ESTIMATE, OPERATING_RISE_SPEED)
    low, high = seed_prices()
    table = tabulate(
        recovery_rows(list(zip(designs, (i for _, i in priced))), book),
        [
            "Design",
            "Recovery",
            "Growth (geo.)",
            "Steady $/kg",
            "BE steady",
            "Launches/yr",
            "Tank t/yr",
            "Chamber t/yr",
            "Seeds to meteoric Fe",
            "Seeds to rocket soot",
        ],
    )
    return (
        f"{book.label}, spray cup, lob climbing; 10% a year, 50% odds. Seed "
        f"${low:.0f} / ${high:.0f} per kg; BE 'a / b' is cheap / dear seed.\n"
        "Methane recovers 90% of tanks and 60% of chamber; hydrogen its tanks only.\n"
        "Rates are per seed's held-level fleet; the last two columns assume all "
        "recycled mass falls back\n(carbon all as soot, an upper bound).\n\n" + table
    )


def main() -> None:
    """Print the hardware-recovery report."""
    print(report())


if __name__ == "__main__":
    main()
