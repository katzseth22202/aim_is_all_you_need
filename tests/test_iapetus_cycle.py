"""Tests for the Iapetus water cycle (src/iapetus_cycle.py, S51, ADR 0078)."""

import math
from dataclasses import replace

import pytest

from src.iapetus_cycle import (
    BASE_INPUTS,
    CLOSING_SPEED,
    ESTIMATE,
    LEDGER_A,
    LEDGER_B,
    METHALOX_DEPARTURE,
    OUTBOUND_BOOK,
    PARENT_TABLE_INPUTS,
    ROUND_YEARS,
    WATER_1300K,
    WATER_1800K,
    PushLaw,
    build_cycle,
    dogleg_dvs,
    engine_mass,
    fastest_arrival,
    fitted_b,
    impulsive_escape,
    outbound_at,
    overtake_leverage,
    periapsis_burn,
    rate_rows,
    return_stage,
    sic_recession,
    value_cycle,
)
from src.solar_dive_growth import (
    HALO_INSERTION,
    PUSH_ALTITUDE,
    delivery_speed,
    l1_transfer_speed,
    periapsis_speed,
)
from src.two_wave_growth import VE_METHALOX


def test_the_dogleg_conics_reproduce_the_parents_legs() -> None:
    """0.59 raise/drop, 0.99 match, 2.37 departure, ~2.5 capture (S51's table)."""
    d = dogleg_dvs()
    assert d["raise"] == pytest.approx(0.589, abs=1e-3)
    assert d["match"] == pytest.approx(0.989, abs=1e-3)
    assert d["departure"] == pytest.approx(2.372, abs=1e-3)
    assert d["capture"] == pytest.approx(2.55, abs=0.01)


def test_the_loading_orbit_is_800_km_radius() -> None:
    """The parent's 0.74 km/s impulsive escape comes from an 800 km radius."""
    assert impulsive_escape(dogleg_dvs()["escape"]) == pytest.approx(0.74, abs=0.005)


def test_the_engine_rows_reproduce_the_parents_thrust() -> None:
    """205 kN and 197 kN per 560 MW reactor (1.85x and 1.78x NERVA)."""
    assert WATER_1300K.thrust == pytest.approx(205.0, abs=1.0)
    assert WATER_1800K.thrust == pytest.approx(197.0, abs=1.0)


def test_engine_mass_follows_the_ge711_pair() -> None:
    """2.1x the power for 1.5x the mass."""
    assert engine_mass(2675.0, 564.0e6) / engine_mass(2675.0, 266.0e6) == pytest.approx(
        1.5, rel=1e-6
    )


def test_the_recession_law_reproduces_the_parents_coat_table() -> None:
    """1400 K, 10 atm: 10-19 um/h at 30-100 m/s; 1950 K at 50 atm is mm/h."""
    assert sic_recession(1400.0, 10.0, 30.0) == pytest.approx(10.3, abs=0.2)
    assert sic_recession(1400.0, 10.0, 100.0) == pytest.approx(18.8, abs=0.3)
    assert sic_recession(1950.0, 50.0, 30.0) == pytest.approx(1574.0, rel=0.01)


def test_the_overtake_law_reproduces_the_parents_placeholder() -> None:
    """10.3x at k = 8.5 and eta_jet 0.775, and B round-trips."""
    b = 1.0 + 0.775 * math.sqrt(9.5)
    assert overtake_leverage(8.5, b) == pytest.approx(10.3, abs=0.05)
    assert fitted_b(4.0, 8.12) == pytest.approx(
        fitted_b(4.0, overtake_leverage(4.0, fitted_b(4.0, 8.12))), rel=1e-9
    )


def test_the_push_law_reproduces_result_20_at_the_tabled_k() -> None:
    """With no miss and no liner, the law gives the sim's table back."""
    law = PushLaw(miss=0.0, liner_carried=False)
    assert law.leverage(4.0) == pytest.approx(8.12, rel=1e-6)
    assert law.leverage(8.5) == pytest.approx(7.20, rel=1e-6)
    assert replace(law, frozen=True).leverage(8.5) == pytest.approx(6.21, rel=1e-6)
    assert replace(law, column="shaped").leverage(8.5) == pytest.approx(8.83, rel=1e-6)


def test_a_backup_pulse_costs_leverage() -> None:
    """Misses at J/mv 1.74 pull the leverage down; the 60 km/s gain lifts it."""
    assert PushLaw(miss=0.2).leverage(4.0) < PushLaw(miss=0.05).leverage(4.0)
    assert PushLaw(closing=60.0).leverage(8.5) > PushLaw().leverage(8.5)
    with pytest.raises(ValueError):
        PushLaw().leverage(2.0)  # the compact column starts at 4


def test_the_plate_is_about_4_percent_of_a_push() -> None:
    """26.4 t on a 625 t ship (Result 20 item 4), spring bags added."""
    law = PushLaw()
    assert law.plate_share(4.0) == pytest.approx(0.044, abs=0.002)
    assert law.pushed_per_push(4.0) == pytest.approx(654.0e3, rel=0.01)


def test_the_return_reaches_51_km_s_from_10_4_at_saturn() -> None:
    """Coplanar circles: 10.4 km/s at Saturn buys ~51 km/s at 200 km in 3.7 yr."""
    assert fastest_arrival(10.4) == pytest.approx(CLOSING_SPEED, abs=0.3)


def test_the_round_is_9_5_years() -> None:
    assert ROUND_YEARS == pytest.approx(9.5, abs=0.01)


def test_rate_rows_reproduce_the_parents_method_check() -> None:
    """G 4.4 at 9.5 yr: doubling 4.4 yr, x19 in 19 yr; at 7.7 yr, x39."""
    r = rate_rows(4.4, 9.5)
    assert r["doubling"] == pytest.approx(4.44, abs=0.05)
    assert r["pipelined_19"] == pytest.approx(19.4, abs=0.2)
    assert rate_rows(4.4, 7.7)["pipelined_19"] == pytest.approx(38.7, abs=0.5)
    assert rate_rows(4.4, 7.7)["whole_rounds"] == 2.0
    assert rate_rows(4.4, 6.6)["whole_rounds"] == 3.0


def test_the_engine_price_fixes_only_the_u235() -> None:
    fixed = ESTIMATE.u235 * ESTIMATE.u235_inventory
    assert ESTIMATE.engine_price(560.0e6) == pytest.approx(60.0e6)
    assert ESTIMATE.engine_price(0.0) == pytest.approx(fixed)


def test_the_solar_dive_can_deliver_to_the_eccentric_orbit() -> None:
    """L1 unchanged; the eccentric orbit is the 10-day orbit at 400 km."""
    speed, kept = delivery_speed("l1")
    assert speed == l1_transfer_speed()
    assert kept == pytest.approx(math.exp(-HALO_INSERTION / VE_METHALOX))
    speed, kept = delivery_speed("eccentric")
    assert speed == pytest.approx(periapsis_speed(10.0, PUSH_ALTITUDE))
    assert kept == 1.0
    with pytest.raises(ValueError):
        delivery_speed("moon")


@pytest.mark.slow
def test_the_periapsis_burn_reproduces_the_parents_oberth_losses() -> None:
    """1.3 / 2.8 / 8.5 / 20.3 / 27.2% at 2 / 3 / 6 / 12 / 16 h."""
    for hours, loss in (
        (2.0, 0.013),
        (3.0, 0.028),
        (6.0, 0.085),
        (12.0, 0.203),
        (16.0, 0.272),
    ):
        assert periapsis_burn(hours, WATER_1300K.ve, 10.4)[1] == pytest.approx(
            loss, abs=0.001
        )


@pytest.mark.slow
def test_the_stage_reproduces_the_parents_puffsat_table() -> None:
    """214 / 314 / 554 / 826 / 919 t at the parent's 1% tank; 32.5 h run at 12 h."""
    for hours, puff in (
        (2.0, 214.0),
        (3.0, 314.0),
        (6.0, 554.0),
        (12.0, 826.0),
        (16.0, 919.0),
    ):
        stage = return_stage(hours, WATER_1300K, PARENT_TABLE_INPUTS)
        assert stage.puffsat / 1e3 == pytest.approx(puff, rel=0.025)
    stage = return_stage(12.0, WATER_1300K, PARENT_TABLE_INPUTS)
    assert stage.run_hours == pytest.approx(32.5, abs=0.2)
    assert stage.escape_dv == pytest.approx(0.99, abs=0.01)
    assert stage.puffsat_per_earth_kg == pytest.approx(4.8, abs=0.2)


@pytest.mark.slow
def test_the_new_tank_base_carries_half_again_the_puffsat() -> None:
    """0.2% tank walls instead of 1%: ~1270 t, not 826, on the 12 h policy."""
    stage = return_stage(12.0, WATER_1300K, BASE_INPUTS)
    assert stage.puffsat / 1e3 == pytest.approx(1270.0, rel=0.02)


@pytest.mark.slow
def test_the_outbound_stack_and_its_cheapest_loss() -> None:
    """The hydrogen stage's 5% loss is dear in engines; methalox needs no engines."""
    five = outbound_at(False, OUTBOUND_BOOK, 0.05)
    fifteen = outbound_at(False, OUTBOUND_BOOK, 0.15)
    assert fifteen.ntp_engines < five.ntp_engines
    assert fifteen.ratio < five.ratio
    assert outbound_at(False, METHALOX_DEPARTURE).ntp_engines == 0.0


@pytest.mark.slow
def test_growth_is_leverage_times_puffsat_over_outbound() -> None:
    """G = bags x (1 - miss) x useful leverage / outbound mass per engine."""
    c = build_cycle(LEDGER_A, 4.0)
    expected = c.stage.puffsat * 0.97 * c.useful / (c.stage.earth_kg * c.out.ratio)
    assert c.growth() == pytest.approx(expected)
    assert c.growth() > 1.0


@pytest.mark.slow
def test_the_valuation_is_scale_free_and_the_loan_is_cheaper() -> None:
    c = build_cycle(LEDGER_B, 4.0)
    tenth = value_cycle(c, 0.10)
    loan = value_cycle(c, 0.04)
    assert tenth.break_even is not None and loan.break_even is not None
    assert loan.break_even < tenth.break_even
    assert tenth.steady_cost < tenth.break_even
