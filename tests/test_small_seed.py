"""Tests for the seed's tonnage and the smaller seed (src/small_seed.py)."""

import pytest

from src import solar_dive_growth as sdg
from src.small_seed import CLOSURES, headline, scale_audit, seed_mass, small_seed


@pytest.mark.slow
def test_the_adopted_headline_seeds_567_and_674_tonnes() -> None:
    shallow, deep = (headline(c) for c in CLOSURES)
    assert seed_mass(shallow) / 1e3 == pytest.approx(567.5, abs=0.5)
    assert seed_mass(deep) / 1e3 == pytest.approx(673.9, abs=0.5)


@pytest.mark.slow
def test_a_one_unit_seed_values_as_before() -> None:
    h = headline(CLOSURES[0])
    plain = sdg.value_design(h.departure, h.push, h.design)
    same = sdg.value_design(h.departure, h.push, h.design, seed_units=1.0)
    assert same.break_even == plain.break_even
    assert plain.seed_mass == pytest.approx(seed_mass(h))


@pytest.mark.slow
def test_the_sun_plate_does_not_shrink_with_the_unit() -> None:
    h = headline(CLOSURES[0])
    moved = {key: (a, b) for key, a, b in scale_audit(h, 0.5)}
    a, b = moved["sun_plate"]
    assert b == pytest.approx(2.0 * a, rel=0.02)
    small = sdg.sub_unit(h.departure, h.push, h.design, 0.25)
    assert small[1].lines["pulse_disk"] == pytest.approx(1.0, rel=1e-4)


@pytest.mark.slow
def test_a_smaller_seed_costs_dollars_and_years_not_the_loop() -> None:
    h = headline(CLOSURES[0])
    full = sdg.value_design(
        h.departure, h.push, h.design, seed_price=sdg.seed_price(h.departure)
    )
    one, two = (small_seed(h, n, dear=False) for n in (1, 2))
    assert one.units < two.units < 1.0
    assert full.break_even < two.valuation.break_even < one.valuation.break_even
    assert one.valuation.seed_outlay < two.valuation.seed_outlay < full.seed_outlay
    assert 0.0 < two.lag_years < one.lag_years < 8.0
