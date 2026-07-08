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
    hyperbolic_eccentricity,
    hyperbolic_time_of_flight,
    orbit_from_periapsis_speed_and_apoapsis_radius,
    speed_around_attractor,
    speed_at_distance,
    speed_with_escape_energy,
    true_anomaly_at_radius,
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


def test_hyperbolic_eccentricity_from_periapsis_and_vinf() -> None:
    # A 4 solar-radii periapsis with ~233 km/s hyperbolic-excess speed gives the
    # escaping-hyperbola eccentricity of the boosted solar-dive return
    # (paper Appendix sec:earth_reintercept): e = 1 + r_p * v_inf**2 / mu.
    periapsis = 4 * Sun.R
    e = hyperbolic_eccentricity(periapsis, 233 * u.km / u.s, Sun)
    assert e > 1.0  # genuinely hyperbolic
    assert e == pytest.approx(2.14, abs=0.05)


def test_true_anomaly_at_radius_ellipse_apoapsis_is_180() -> None:
    # On any ellipse, apoapsis (r = r_a) sits at true anomaly 180 deg. Take a
    # (r_p, r_a) = (1, 3) ellipse: e = (3-1)/(3+1) = 0.5.
    nu = true_anomaly_at_radius(1 * u.AU, 0.5, 3 * u.AU)
    assert is_nearly_equal(nu, 180 * u.deg, percent=0.001)


def test_true_anomaly_at_radius_hyperbola_climbout() -> None:
    # The boosted solar-dive hyperbola (4 R_sun periapsis, e ~ 2.14) re-crosses
    # 1 AU at a true anomaly of ~116 deg -- the climb-out half of the ~295 deg
    # whip-around (paper Appendix sec:earth_reintercept).
    nu = true_anomaly_at_radius(4 * Sun.R, 2.14, EARTH_A)
    assert is_nearly_equal(nu, 116 * u.deg, percent=0.03)


def test_true_anomaly_at_radius_rejects_unreachable_radius() -> None:
    # A radius inside periapsis is unreachable (|cos nu| > 1); the function
    # raises rather than returning a nonsense angle.
    with pytest.raises(ValueError, match="not reachable"):
        true_anomaly_at_radius(1 * u.AU, 0.5, 0.5 * u.AU)


def test_hyperbolic_time_of_flight_matches_climb_out_week() -> None:
    # Climbing the boosted solar-dive hyperbola from periapsis out to 1 AU takes
    # ~1 week (paper Appendix sec:earth_reintercept), which -- added to the ~66 d
    # fall -- makes the round trip ~0.2 yr.
    tof = hyperbolic_time_of_flight(4 * Sun.R, 233 * u.km / u.s, 116 * u.deg, Sun)
    assert is_nearly_equal(tof, 7.0 * u.day, percent=0.1)


def test_hyperbolic_time_of_flight_rejects_nonhyperbolic_orbit() -> None:
    # With zero excess speed the orbit is parabolic (e = 1), not hyperbolic, so
    # the hyperbolic time-of-flight is undefined and the function raises rather
    # than dividing by v_infinity = 0.
    with pytest.raises(ValueError, match="hyperbolic"):
        hyperbolic_time_of_flight(1 * u.AU, 0 * u.km / u.s, 90 * u.deg, Sun)
