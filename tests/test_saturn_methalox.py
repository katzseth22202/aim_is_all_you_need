"""Tests for the reactor-plus-methalox comparison (src/saturn_methalox.py)."""

from dataclasses import replace

import pytest

from src.saturn_methalox import (
    CARBON_PER_METHALOX,
    CERIA,
    CERIA_PESSIMISTIC,
    CHEM,
    CO2_PER_METHALOX,
    ELECTRIC,
    IAPETUS,
    IAPETUS_DOGLEG,
    KITS,
    PHOEBE,
    PHOEBE_LEG,
    methalox_per_mw_year,
    stage_unit,
)


def test_the_legs_reproduce_the_parents_round_times() -> None:
    """1.17 yr dog-leg each way at Iapetus, 0.27 yr fall at Phoebe: 9.5 / 7.7 yr."""
    assert IAPETUS_DOGLEG == pytest.approx(1.17, abs=0.01)
    assert PHOEBE_LEG == pytest.approx(0.27, abs=0.01)
    assert IAPETUS.round_years() == pytest.approx(9.5, abs=0.02)
    assert PHOEBE.round_years() == pytest.approx(7.7, abs=0.02)


def test_the_sites_reproduce_the_parents_delta_v() -> None:
    """Moon to departure: 3.70 (Iapetus) and 3.97 (Phoebe); Phoebe out ~4.1."""
    assert IAPETUS.home_dv == pytest.approx(3.70, abs=0.01)
    assert PHOEBE.home_dv == pytest.approx(3.97, abs=0.01)
    assert PHOEBE.outbound_dv == pytest.approx(4.15, abs=0.05)


def test_the_stage_reproduces_the_parents_methalox() -> None:
    """1,620-1,860 t per 826 t at Iapetus; 1,850-2,160 t at Phoebe."""
    shipped = replace(CHEM, tanks_shipped=True)
    assert 1620 <= stage_unit(IAPETUS, shipped).methalox * 826 <= 1860
    assert 1850 <= stage_unit(PHOEBE, shipped).methalox * 826 <= 2160


def test_carbon_needs_a_quarter_of_the_co2_mass() -> None:
    assert CARBON_PER_METHALOX == pytest.approx(CO2_PER_METHALOX * 12 / 44)
    carbon = stage_unit(IAPETUS, CHEM, KITS)
    co2 = stage_unit(IAPETUS, CHEM, replace(KITS, carbon_form="CO2"))
    assert carbon.regolith < 0.3 * co2.regolith


def test_a_mw_of_fuel_makes_2300_t_a_year_before_its_other_work() -> None:
    """1 MW for a year over 13.7 MJ/kg, with shipped tanks and CO2 ice at 10%."""
    shipped = replace(CHEM, tanks_shipped=True)
    kits = replace(KITS, carbon_form="CO2", carbon_grade=0.10)
    unit = stage_unit(IAPETUS, shipped, kits)
    assert methalox_per_mw_year(CERIA, unit, kits) / 1e3 == pytest.approx(
        2303.0, rel=0.01
    )


def test_plant_prices_and_masses_scale_with_heat() -> None:
    assert CERIA_PESSIMISTIC.mass_per_mw == pytest.approx(2.0 * CERIA.mass_per_mw)
    assert CERIA.price_per_mw == pytest.approx(200.0e6)
    assert ELECTRIC.unit_fuel_mw == pytest.approx(2.0)


@pytest.mark.slow
def test_the_plant_cycle_grows_and_shipped_tanks_stall_it() -> None:
    from src.saturn_methalox import build_plant_cycle, value_plant_cycle

    c = build_plant_cycle(PHOEBE)
    v = value_plant_cycle(c)
    assert c.growth() > 3.0
    assert v.break_even is not None and v.steady_cost < v.break_even
    stalled = build_plant_cycle(PHOEBE, stage=replace(CHEM, tanks_shipped=True))
    assert stalled.growth() < c.growth() / 2.0
