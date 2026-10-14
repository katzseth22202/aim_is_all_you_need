"""Tests for the lighter plate flown against gravity (ADR 0062)."""

import numpy as np
import pytest
from astropy import units as u

from src.growth_ledger import LAUNCH_UNIT
from src.light_plate_push import (
    DELIVERY_CAP_G,
    FIXED_PLATE_SHARE,
    FLOOR_RADIUS,
    REFERENCE_PLATE,
    REFERENCE_PULSE,
    SPRING_RIPPLE,
    WAVE_SPEEDS_400,
    delivered,
    excess_speed,
    fly_push,
    lofted,
    max_aim,
    plate_mass,
)
from src.lob_rise import BASELINE_RISE_SPEED, OPERATING_RISE_SPEED

SLOW = WAVE_SPEEDS_400["3S slowest"]
UP = (1.0, 1.0, 0.5, 0.0)


def test_the_plate_scales_with_its_pulse():
    assert plate_mass(REFERENCE_PULSE) == pytest.approx(REFERENCE_PLATE)
    assert plate_mass(REFERENCE_PULSE / 2) == pytest.approx(REFERENCE_PLATE / 2)
    assert plate_mass(REFERENCE_PULSE, FIXED_PLATE_SHARE) == pytest.approx(
        REFERENCE_PLATE
    )
    assert plate_mass(0.0, FIXED_PLATE_SHARE) == pytest.approx(FIXED_PLATE_SHARE)


def test_the_booster_lofts_the_launch_unit_on_the_parents_lob():
    assert lofted(BASELINE_RISE_SPEED) == pytest.approx(LAUNCH_UNIT.to_value(u.kg))
    # x1.065 dearer at 1.1 km/s is about 6% less lofted.
    assert lofted(OPERATING_RISE_SPEED) == pytest.approx(1409e3, rel=0.01)


def test_aim_is_flat_at_the_floor_and_about_five_degrees_at_430_km():
    excess = excess_speed(SLOW)
    assert max_aim(FLOOR_RADIUS, excess) == 0.0
    assert np.degrees(max_aim(FLOOR_RADIUS + 30e3, excess)) == pytest.approx(
        5.4, abs=0.3
    )


def test_todays_plate_holds_the_floor_and_the_cap_limits_the_mean():
    free = fly_push(SLOW, REFERENCE_PULSE, OPERATING_RISE_SPEED, UP, None)
    capped = fly_push(SLOW, REFERENCE_PULSE, OPERATING_RISE_SPEED, UP, DELIVERY_CAP_G)
    assert free.feasible and capped.feasible
    assert free.peak_g > 6.0
    assert capped.peak_g == pytest.approx(
        DELIVERY_CAP_G / (1 + SPRING_RIPPLE), rel=1e-3
    )
    # The cap stretches the push but barely moves the mass delivered.
    assert capped.duration > free.duration
    assert delivered(capped, REFERENCE_PULSE) == pytest.approx(
        delivered(free, REFERENCE_PULSE), rel=0.01
    )
    # The capped pulses taper along the wave.
    assert capped.last_puffsat < 0.7 * capped.first_puffsat


def test_aiming_down_drops_a_weak_push_through_the_floor():
    push = fly_push(SLOW, 5.8e6, OPERATING_RISE_SPEED, (-1.0,) * 4, DELIVERY_CAP_G)
    assert not push.feasible


def test_a_lighter_plate_delivers_more_on_the_same_lob():
    heavy = fly_push(SLOW, REFERENCE_PULSE, OPERATING_RISE_SPEED, UP, DELIVERY_CAP_G)
    light = fly_push(SLOW, 5.8e6, OPERATING_RISE_SPEED, (1.0,) * 4, DELIVERY_CAP_G)
    assert light.feasible
    assert delivered(light, 5.8e6) > delivered(heavy, REFERENCE_PULSE) + 50e3


def test_the_growth_loading_cannot_hold_the_floor_on_the_delivery_optimums_plate():
    # ADR 0062: the ledger's lighter loading keeps the stack heavy, so the
    # 8.3 MN s plate on the parent's lob falls back to 400 km at any price.
    for price in (0.01, 0.05, 0.2):
        push = fly_push(
            SLOW, 8.28e6, BASELINE_RISE_SPEED, (1.0,) * 4, None, None, price
        )
        assert not push.feasible
    push = fly_push(
        SLOW, 10.0e6, BASELINE_RISE_SPEED, (1.0, 1.0, 0.5, 0.0), None, None, 0.05
    )
    assert push.feasible
