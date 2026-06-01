"""Tests for orbital mechanics functions in orbit_utils.py."""

from typing import Tuple

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Earth, Moon, Sun
from boinor.maneuver import Maneuver

from src.astro_constants import EARTH_A, JUPITER_A, LEO_ALTITUDE
from src.orbit_utils import (
    body_speed,
    distance_to_center,
    escape_velocity,
    find_periapsis_radius_from_apoapsis_and_speed,
    get_period,
    get_semimajor_axis,
    orbit_from_periapsis_speed_and_apoapsis_radius,
    speed_around_attractor,
    speed_at_distance,
    speed_with_escape_energy,
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


def test_speed_with_escape_energy_matches_quadrature() -> None:
    # The primitive must equal the explicit sqrt(v_inf**2 + v_esc**2) it replaced.
    v_infinity = 10 * u.km / u.s
    altitude = 200 * u.km
    v_esc = escape_velocity(Earth, altitude)
    expected = np.sqrt(v_infinity**2 + v_esc**2)
    result = speed_with_escape_energy(v_infinity, Earth, altitude)
    assert is_nearly_equal(result, expected)


def test_speed_with_escape_energy_zero_excess_is_escape_velocity() -> None:
    # With no hyperbolic excess, the local speed is exactly escape velocity.
    result = speed_with_escape_energy(0 * u.km / u.s, Moon)
    assert is_nearly_equal(result, escape_velocity(Moon))


def test_period() -> None:
    T = get_period(Sun, EARTH_A)
    assert is_nearly_equal(T, 1 * u.year)


def test_semi_major_axis() -> None:
    a = get_semimajor_axis(Sun, 1 * u.year)
    assert is_nearly_equal(a, EARTH_A)


def test_distance_to_center() -> None:
    d = distance_to_center(LEO_ALTITUDE, Earth)
    assert is_nearly_equal(d, Earth.R + LEO_ALTITUDE)


def test_find_periapsis_radius_from_apoapsis_and_speed() -> None:
    v_p = find_periapsis_radius_from_apoapsis_and_speed(
        apoapsis_radius=EARTH_A, periapsis_speed=TEST_VP
    )
    assert is_nearly_equal(v_p, EXPECTED_TEST_RP)


def test_orbit_from_periapsis_speed_and_apoapsis_radius() -> None:
    # The shipped constructor solves r_p through the single periapsis-radius
    # primitive and builds the orbit; pin its periapsis/apoapsis directly
    # (previously this path was only exercised transitively).
    orbit = orbit_from_periapsis_speed_and_apoapsis_radius(
        periapsis_speed=TEST_VP, apoapsis_radius=EARTH_A, attractor_body=Sun
    )
    assert is_nearly_equal(orbit.r_p, EXPECTED_TEST_RP)
    assert is_nearly_equal(orbit.r_a, EARTH_A)
    # The constructor's periapsis radius is exactly the primitive's output --
    # the quadratic now lives in one place, so the two cannot drift apart.
    assert is_nearly_equal(
        orbit.r_p,
        find_periapsis_radius_from_apoapsis_and_speed(
            apoapsis_radius=EARTH_A, periapsis_speed=TEST_VP
        ),
    )


def test_speed_at_distance() -> None:
    speed_at_earth = speed_at_distance(
        radius_periapsis=EXPECTED_TEST_RP, periapsis_speed=TEST_VP, distance=EARTH_A
    )
    after_burn = TEST_VP + 50 * u.km / u.s
    speed_at_earth = speed_at_distance(
        radius_periapsis=EXPECTED_TEST_RP,
        periapsis_speed=after_burn,
        distance=EARTH_A,
    )
    assert is_nearly_equal(speed_at_earth, 150.24114202 * u.km / u.s)
