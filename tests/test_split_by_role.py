"""Tests for the split comparison (ADRs 0068, 0069)."""

import pytest

from src.chamber_departure import growth_per_cycle
from src.growth_ledger import summarize_chain
from src.split_by_role import chain_by_role, role_growth, split_gap_row
from src.survivable_chamber import PLATE
from src.two_wave_growth import adaptive_two_wave_cycles


def test_with_equal_burns_the_role_split_is_the_batch_ledger():
    puffsats, stack, onward, rods = 100.0, 805.0, 178.0, 22.3
    growth_arrives, rods_arrive = 0.95, 0.99
    batch = growth_per_cycle(
        stack / puffsats * growth_arrives,
        rods / stack / rods_arrive,
        onward / stack,
    )
    split = role_growth(
        puffsats, onward, onward, rods, rods, growth_arrives, rods_arrive
    )
    assert split == pytest.approx(batch, rel=1e-9)


def test_a_harder_growth_departure_costs_growth():
    equal = role_growth(100.0, 178.0, 178.0, 22.3, 22.3, 1.0, 1.0)
    harder = role_growth(100.0, 160.0, 178.0, 24.0, 22.3, 1.0, 1.0)
    assert harder < equal


@pytest.mark.slow
def test_the_role_split_reproduces_adr_0069():
    cycles = adaptive_two_wave_cycles(split_days=20.0, split="independent")
    growths = chain_by_role("methane")
    doubling = summarize_chain([c.period_years for c in cycles], growths).doubling_years
    assert doubling == pytest.approx(3.91, abs=0.005)


@pytest.mark.slow
def test_the_ledgers_role_split_reproduces_adr_0069():
    # growth_ledger prices both carriers itself (ADR 0071); centred on the
    # 20-day orbit it is split_by_role's 3.91 yr.
    from src.finite_burn_loss import fixed_direction_loss
    from src.growth_ledger import _orbit_loss, chain_growth, parking_period
    from src.split_by_role import DESIGNS

    cycles = adaptive_two_wave_cycles(split_days=20.0)
    design = DESIGNS["methane"]
    grown = chain_growth(
        cycles,
        PLATE.efficiency,
        design.pairing,
        design.efficiency,
        design.pitch_ratio,
        loss_model=_orbit_loss(fixed_direction_loss, parking_period(cycles[0])),
        max_slug_ratio=PLATE.max_slug_ratio,
        impactor_bond_energy=PLATE.impactor_bond_energy,
        film_per_impulse=PLATE.film_per_impulse,
    )
    doubling = summarize_chain(
        [c.period_years for c in cycles], [g.growth for g in grown]
    ).doubling_years
    assert doubling == pytest.approx(3.91, abs=0.005)
    assert all(0.0 < g.growth_share < 1.0 for g in grown)


@pytest.mark.slow
def test_ten_days_is_the_best_split_gap_for_methane():
    # ADR 0071's sweep, stream kept in view: 4.14 / 3.98 / 4.00 yr at 7 / 10 /
    # 15 days.
    rows = {gap: split_gap_row(("methane", gap)) for gap in (7.0, 10.0, 15.0)}
    assert rows[10.0]["doubling yr"] == pytest.approx(3.98, abs=0.01)
    assert rows[10.0]["doubling yr"] < min(
        rows[7.0]["doubling yr"], rows[15.0]["doubling yr"]
    )


def test_a_unit_that_cannot_depart_grows_nothing():
    assert role_growth(100.0, 178.0, -5.0, 22.3, 22.3, 1.0, 1.0) == 0.0
    assert role_growth(100.0, -5.0, 178.0, 22.3, 22.3, 1.0, 1.0) == 0.0
