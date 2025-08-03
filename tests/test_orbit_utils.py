"""Tests for orbital mechanics functions in orbit_utils.py."""

from typing import Tuple

import pytest
from astropy import units as u
from poliastro.bodies import Earth, Sun
from poliastro.maneuver import Maneuver

from src.astro_constants import EARTH_A, JUPITER_A, LEO_ALTITUDE
from src.orbit_utils import (
    body_speed,
    distance_to_center,
    escape_velocity,
    find_periapsis_radius_from_apoapsis_and_velocity,
    get_period,
    get_semimajor_axis,
    speed_around_attractor,
    velocity_at_distance,
)
from tests.test_helpers import is_nearly_equal

TEST_VP = 200 * u.km / u.s
EXPECTED_TEST_RP = 6364822 * u.km


def test_body_speed_earth_low_orbit() -> None:
    # 400 km altitude (typical for ISS)
    altitude = 400 * u.km
    speed = body_speed(Earth, altitude)
    # The expected speed is about 7.67 km/s for a 400 km LEO
    expected = 7.67 * u.km / u.s
    assert is_nearly_equal(speed, expected, percent=0.01)


def test_body_speed_earth_around_sun() -> None:
    speed: u.Quantity = speed_around_attractor(a=1 * u.AU)
    assert is_nearly_equal(speed, 29.78 * u.km / u.s)


def test_escape_velocity_earth_200km() -> None:
    # 200 km altitude above Earth's surface
    altitude = 200 * u.km
    v_esc = escape_velocity(Earth, altitude)
    # The expected escape velocity at 200 km altitude is about 11.0 km/s
    expected = 11.0 * u.km / u.s
    assert is_nearly_equal(v_esc, expected)


def test_period() -> None:
    T = get_period(Sun, EARTH_A)
    assert is_nearly_equal(T, 1 * u.year)


def test_semi_major_axis() -> None:
    a = get_semimajor_axis(Sun, 1 * u.year)
    assert is_nearly_equal(a, EARTH_A)


def test_distance_to_center() -> None:
    d = distance_to_center(LEO_ALTITUDE, Earth)
    assert is_nearly_equal(d, Earth.R + LEO_ALTITUDE)


def test_find_periapsis_radius_from_apoapsis_and_velocity() -> None:
    v_p = find_periapsis_radius_from_apoapsis_and_velocity(
        apoapsis_radius=EARTH_A, periapsis_velocity=TEST_VP
    )
    assert is_nearly_equal(v_p, EXPECTED_TEST_RP)


def test_velocity_at_distance() -> None:
    speed_at_earth = velocity_at_distance(
        radius_periapsis=EXPECTED_TEST_RP, velocity_periapsis=TEST_VP, distance=EARTH_A
    )
    after_burn = TEST_VP + 50 * u.km / u.s
    speed_at_earth = velocity_at_distance(
        radius_periapsis=EXPECTED_TEST_RP,
        velocity_periapsis=after_burn,
        distance=EARTH_A,
    )
    assert is_nearly_equal(speed_at_earth, 150.24114202 * u.km / u.s)
