"""Tests for scenario-related functions in scenario.py."""

import pytest
from astropy import units as u
from poliastro.bodies import Moon

from src.scenario import (
    burn_for_v_infinity,
)
from tests.test_helpers import is_nearly_equal


def test_burn_for_v_infinity_earth_leo() -> None:
    """Test burn_for_v_infinity with default parameters (Earth, LEO altitude)."""
    # Test case: achieve 10 km/s v_infinity from Earth LEO
    v_infinity = 10 * u.km / u.s
    burn = burn_for_v_infinity(v_infinity)

    # Expected: sqrt(v_escape^2 + v_infinity^2) - v_initial
    # v_escape at 200 km altitude ≈ 11.0 km/s
    # v_total = sqrt(11.0^2 + 10^2) = sqrt(121 + 100) = sqrt(221) ≈ 14.87 km/s
    # burn = 14.87 - 0 = 14.87 km/s
    expected_burn = 14.87 * u.km / u.s
    assert is_nearly_equal(burn, expected_burn, percent=0.01)


def test_burn_for_v_infinity_with_initial_velocity() -> None:
    """Test burn_for_v_infinity with non-zero initial velocity."""
    # Test case: achieve 5 km/s v_infinity from Earth LEO with 7 km/s initial velocity
    v_infinity = 5 * u.km / u.s
    initial_velocity = 7 * u.km / u.s
    burn = burn_for_v_infinity(v_infinity, initial_velocity=initial_velocity)

    # Expected: sqrt(v_escape^2 + v_infinity^2) - v_initial
    # v_escape at 200 km altitude ≈ 11.0 km/s
    # v_total = sqrt(11.0^2 + 5^2) = sqrt(121 + 25) = sqrt(146) ≈ 12.08 km/s
    # burn = 12.08 - 7 = 5.08 km/s
    expected_burn = 5.08 * u.km / u.s
    assert is_nearly_equal(burn, expected_burn, percent=0.01)


def test_burn_for_v_infinity_moon() -> None:
    """Test burn_for_v_infinity with Moon as the body."""
    # Test case: achieve 2 km/s v_infinity from Moon at 1 km altitude
    v_infinity = 2 * u.km / u.s
    altitude = 1 * u.km
    burn = burn_for_v_infinity(v_infinity, body=Moon, altitude=altitude)

    # Expected: sqrt(v_escape^2 + v_infinity^2) - v_initial
    # v_escape at 1 km altitude above Moon ≈ 2.37 km/s
    # v_total = sqrt(2.37^2 + 2^2) = sqrt(5.62 + 4) = sqrt(9.62) ≈ 3.10 km/s
    # burn = 3.10 - 0 = 3.10 km/s
    expected_burn = 3.10 * u.km / u.s
    assert is_nearly_equal(burn, expected_burn, percent=0.01)


def test_burn_for_v_infinity_zero_v_infinity() -> None:
    """Test burn_for_v_infinity with zero v_infinity (should raise ValueError)."""
    with pytest.raises(ValueError, match="v_infinity must be positive"):
        burn_for_v_infinity(0 * u.km / u.s)


def test_burn_for_v_infinity_negative_v_infinity() -> None:
    """Test burn_for_v_infinity with negative v_infinity (should raise ValueError)."""
    with pytest.raises(ValueError, match="v_infinity must be positive"):
        burn_for_v_infinity(-5 * u.km / u.s)
