"""Tests for the water-battery argon tug (ADR 0061)."""

import numpy as np
import pytest

from src.seed_battery_tug import (
    LOOP_FLIGHTS,
    WATER_SINKS,
    array_credit,
    climb,
    loop_seed,
    price_tug,
    radiator_mass,
    ship_seed,
)
from src.seed_sep_tug import _MU, START_RADIUS


def test_the_plain_routes_match_adr_0059_within_the_departure_loss():
    # ADR 0059 charged a full 1200 t burn's loss; this charges what is burned.
    assert loop_seed(0, "2-yr loop") == pytest.approx(128.66e3, rel=0.02)
    assert loop_seed(1, "2-yr loop") == pytest.approx(110.55e3, rel=0.02)
    assert loop_seed(0) == pytest.approx(86.03e3, rel=0.01)


def test_a_higher_perigee_speed_sends_more_seed():
    axis = 15.0e3
    speed = float(np.sqrt(_MU * (2.0 / START_RADIUS - 1.0 / axis)))
    assert ship_seed(START_RADIUS, speed, axis, 0) > 1.5 * loop_seed(0)


def test_the_battery_lets_a_small_array_feed_a_large_peak():
    flown = climb(90.0, 15.0e3, 20.0, 5.0e6, 30.0e6, 1.6e6)
    assert flown.axis >= 15.0e3
    assert flown.peak > 1.5 * 5.0e6
    assert flown.hydrogen > 0.0
    assert flown.fuel_cell > 0.0


def test_ice_holds_more_heat_per_kilogram_but_needs_a_bigger_radiator():
    liquid, ice = WATER_SINKS["liquid"], WATER_SINKS["ice"]
    assert ice[0] > liquid[0]
    assert radiator_mass(1.0e6, ice[1]) > radiator_mass(1.0e6, liquid[1])


def test_the_array_credit_includes_the_shipping_a_buyer_saves():
    assert array_credit(1.0e6, 0.0) == pytest.approx(3.5e6 / 1.1**0.25)
    assert array_credit(1.0e6, 10.0) > 8.0e6 * 0.97


def test_the_loop_flights_are_adr_0059s_champions():
    assert LOOP_FLIGHTS[0][0] == pytest.approx(5.21552)
    assert sum(LOOP_FLIGHTS[0][1]) == pytest.approx(3.269, abs=1e-3)


def test_a_low_handover_tug_prices():
    tug = price_tug((90.0, 15.0e3, 20.0, 5.0, 6.0), 25.0, 1.0)
    assert tug is not None
    assert tug.seed > 1.5 * loop_seed(0)
    assert tug.water_needed <= tug.seed


def test_the_tug_loses_on_dear_burst_hardware():
    # ADR 0061's direct-route champion at $25/W array and $3/W burst: 0.978.
    x = (64.11553515986981, 11802.744761299888, 12.036676513563076, 1.8564002128542363)
    for ratio in (1.5, 2.7, 4.0):
        tug = price_tug((*x, ratio), 25.0, 3.0)
        assert tug is None or tug.worth < 1.0
