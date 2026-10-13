"""Tests for src/seed_sep_tug.py: the argon tug that lifts the seed ship (ADR 0058)."""

import numpy as np
import pytest

from src.seed_sep_tug import (
    _MU,
    START_RADIUS,
    SUNLIT,
    SYNODIC_YEARS,
    best_tugs,
    circular_spiral,
    l1_credit,
    mean_sunlit,
    perigee_pump,
    tug_seed,
    window_delay,
    worth,
)


def test_the_circular_spiral_gives_up_the_oberth_effect() -> None:
    """6.98 km/s out and 0.69 back down: almost the 8.50 km/s it replaces."""
    spiral, drop = circular_spiral()
    assert spiral == pytest.approx(6.980, abs=0.005)
    assert drop == pytest.approx(0.688, abs=0.005)


def test_the_sunlit_share_is_the_spirals() -> None:
    assert mean_sunlit() == pytest.approx(SUNLIT, abs=0.005)


def test_a_narrow_arc_is_nearly_an_impulsive_apogee_raise() -> None:
    """Thrust within 30 deg of perigee costs 2% above the impulsive burn."""
    pump = perigee_pump(30.0, 160.0e3)
    impulsive = np.sqrt(_MU * (2.0 / START_RADIUS - 1.0 / 160.0e3)) - np.sqrt(
        _MU / START_RADIUS
    )
    assert pump.delta_v == pytest.approx(impulsive, rel=0.03)
    assert pump.delta_v > impulsive
    assert pump.perigee - START_RADIUS < 150.0


def test_wider_arcs_buy_duty_with_delta_v_and_perigee() -> None:
    narrow, wide = perigee_pump(30.0, 40.0e3), perigee_pump(90.0, 40.0e3)
    assert wide.duty > 4.0 * narrow.duty
    assert wide.delta_v > narrow.delta_v
    assert wide.departure_burn > narrow.departure_burn


def test_a_climb_waits_for_the_next_jupiter_window() -> None:
    assert window_delay(0.5) == pytest.approx(SYNODIC_YEARS)
    assert window_delay(SYNODIC_YEARS) == pytest.approx(SYNODIC_YEARS)
    assert window_delay(2.0) == pytest.approx(2.0 * SYNODIC_YEARS)


def test_the_array_sells_for_less_than_the_tug_cost() -> None:
    """Half the price, derated, plus shipping, discounted: under the bill."""
    assert 0.0 < l1_credit(1.0e6, 10.0, 1.0) < 10.0e6


def test_boil_off_eats_the_seed() -> None:
    pump = perigee_pump(90.0, 25.0e3)
    dry, wet = tug_seed(pump, 2.0, 0.0, 10.0), tug_seed(pump, 2.0, 0.0005, 10.0)
    assert wet.seed < 0.8 * dry.seed
    assert dry.seed == pytest.approx(203.5e3, rel=0.01)


def test_the_best_case_only_matches_the_earth_loop() -> None:
    """$10/W, no boil-off, array sold at L1: 1.15x, against EEJ's free 1.16x."""
    pump = perigee_pump(90.0, 25.0e3)
    k, value = worth(tug_seed(pump, 2.0, 0.0, 10.0), credited=True)
    assert k == pytest.approx(1.42, abs=0.01)
    assert value == pytest.approx(1.153, abs=0.005)


@pytest.mark.slow
def test_above_ten_dollars_a_watt_or_with_boil_off_the_tug_loses() -> None:
    for row in best_tugs():
        if row["$/W"] > 10.0 or row["boil %/d"] > 0.0:
            assert row["worth"] < 1.0
