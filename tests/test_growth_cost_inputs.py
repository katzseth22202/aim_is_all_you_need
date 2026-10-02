"""Tests for src/growth_cost_inputs.py: the real chain, as the cost model reads it.

All slow: each builds the growth ledger's chains (minutes, cached per process).
"""

import pytest
from astropy import units as u

from src.chamber_isp import ROD_MASS
from src.growth_cost import (
    ESTIMATE,
    PLATE_AND_ABSORBER,
)
from src.growth_cost import ROD_MASS as COST_ROD_MASS
from src.growth_cost import (
    Departure,
    DiscountSchedule,
    break_even_price,
    legacy_prices,
    steady_state_cost,
    value_per_seed_dollar,
)
from src.growth_cost_inputs import design_inputs, seed_prices
from src.growth_ledger import LAUNCH_UNIT, PLATE_MASS
from src.harvest import ABSORBER_MASS, PLATE_PRICES, solve_rate
from src.seed_cost import DESIGNS, design_chain

pytestmark = pytest.mark.slow


def test_the_pure_models_masses_are_the_ledgers() -> None:
    assert PLATE_AND_ABSORBER == pytest.approx(
        (PLATE_MASS + ABSORBER_MASS).to_value(u.kg)
    )
    assert COST_ROD_MASS == pytest.approx(ROD_MASS.to_value(u.kg))


@pytest.mark.parametrize("design", DESIGNS, ids=lambda d: d.label)
def test_the_inputs_carry_the_chains_seed_and_growth(design) -> None:
    chain = design_chain(design)
    inputs = design_inputs(design)
    assert inputs.seed == pytest.approx(chain.seed.to_value(u.kg))
    assert [c.growth for c in inputs.cycles] == pytest.approx(list(chain.growths))
    assert inputs.launch_unit == pytest.approx(LAUNCH_UNIT.to_value(u.kg))
    # The seed is one launch unit's first-cycle consumption.
    assert inputs.cycles[0].consumed == pytest.approx(inputs.seed)


def test_methalox_flies_no_rods_and_chambers_do() -> None:
    methalox = design_inputs(DESIGNS[0])
    hydrogen = design_inputs(DESIGNS[4])
    assert methalox.departure is Departure.METHALOX
    assert methalox.seed_rods == 0.0
    assert all(c.onward_rods == 0.0 for c in methalox.cycles)
    assert hydrogen.departure is Departure.HYDROGEN
    # Rods are 20% of solved hydrogen's first-cycle consumption (17% in ASKS H6,
    # before ADR 0038's onward burn).
    assert hydrogen.seed_rods / hydrogen.seed == pytest.approx(0.196, abs=0.005)
    assert hydrogen.cycles[0].cryostats > 0.0


def test_a_launch_unit_spends_what_the_ledger_launched() -> None:
    # Solved methane, cycle 0: one chamber, 13.5 t of tanks, 468 t of departure
    # mass (gas, plugs and pitch) and 580 t of argon.  The ASKS v1 table's
    # 11.3 t, 392 t and 577 t predate ADR 0038: cycle 0's payload now flies
    # cycle 1's 2S burn (7.17 km/s), not its own 3S burn (5.33).
    cycle = design_inputs(DESIGNS[2]).cycles[0]
    assert cycle.departure_units == 1
    assert cycle.tanks / 1e3 == pytest.approx(13.54, abs=0.05)
    assert (cycle.gas + cycle.plugs + cycle.pitch) / 1e3 == pytest.approx(
        467.7, abs=0.5
    )
    assert cycle.argon / 1e3 == pytest.approx(580.4, abs=0.5)


@pytest.mark.parametrize(
    "design, liquidation, steady",
    # ADR 0038 (onward burn); tab:seed_return printed 45 / 3, 39 / 10 and
    # 87 / 33, 84 / 36.
    [(DESIGNS[0], (42, 2), (38, 9)), (DESIGNS[4], (81, 29), (80, 33))],
    ids=["methalox", "hydrogen solved"],
)
def test_with_growth_uncharged_the_seed_returns_tab_seed_return(
    design, liquidation, steady
) -> None:
    # Ask G1's validation: no growth charge and the plate flat at $114/kg give
    # back the parent's tab:seed_return at $500 (IRR %, cheap / dear seed).
    inputs = design_inputs(design)
    prices = legacy_prices(plate_per_kg=PLATE_PRICES["learned"])
    for held, expected in ((False, liquidation), (True, steady)):
        for seed_price, irr in zip(seed_prices(), expected):

            def value(rate: float) -> float:
                schedule = DiscountSchedule.flat(rate)
                return value_per_seed_dollar(
                    inputs, prices, seed_price, 500.0, schedule, held
                )

            assert 100.0 * solve_rate(value, low=1.0e-6) == pytest.approx(irr, abs=0.5)


@pytest.mark.parametrize(
    "design, steady_cost, break_even",
    [
        (DESIGNS[0], 169, (360, 1919)),
        (DESIGNS[2], 100, (135, 240)),
        (DESIGNS[4], 100, (132, 206)),
    ],
    ids=["methalox", "methane solved", "hydrogen solved"],
)
def test_the_estimates_headline_figures(design, steady_cost, break_even) -> None:
    # ADR 0037's headline: steady-state $/kg at L1 and the stepped-rate
    # steady-state break-even, cheap / dear seed.  Pinned to the dollar.
    inputs = design_inputs(design)
    assert steady_state_cost(inputs, ESTIMATE)[0] == pytest.approx(steady_cost, abs=1.0)
    schedule = DiscountSchedule.stepped(0.30, 0.10, inputs.proof_years)
    for seed_price, expected in zip(seed_prices(), break_even):
        found = break_even_price(inputs, ESTIMATE, seed_price, schedule, steady=True)
        assert found == pytest.approx(expected, abs=1.0)
