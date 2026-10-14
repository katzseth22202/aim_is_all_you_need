"""Tests for the split comparison (ADRs 0068, 0069)."""

import pytest

from src.chamber_departure import growth_per_cycle
from src.growth_ledger import summarize_chain
from src.split_by_role import chain_by_role, role_growth
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
