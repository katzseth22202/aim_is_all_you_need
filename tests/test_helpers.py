"""Shared test helper functions."""

import numpy as np
import pytest
from astropy import units as u


def is_nearly_equal(
    actual: u.Quantity, expected: u.Quantity, percent: float = 0.001
) -> bool:
    """Return True if the ratio of actual to expected is within the specified percentage.
    If expected is zero, use absolute difference.
    """
    if expected.value == 0 * expected.unit:
        return bool(np.isclose(actual, expected, atol=percent))
    # Check if the ratio is within the percentage tolerance
    ratio = actual / expected
    return bool(np.isclose(ratio, 1.0, rtol=percent))


def test_is_nearly_equal_non_zero_cases() -> None:
    """Test is_nearly_equal with non-zero expected values."""
    # Test exact match
    assert is_nearly_equal(100 * u.km, 100 * u.km, percent=0.001)

    # Test within 1% tolerance
    assert is_nearly_equal(101 * u.km, 100 * u.km, percent=0.01)  # 1% higher
    assert is_nearly_equal(99 * u.km, 100 * u.km, percent=0.01)  # 1% lower

    # Test at the boundary of 1% tolerance
    assert is_nearly_equal(100.99 * u.km, 100 * u.km, percent=0.01)  # Just within
    assert is_nearly_equal(99.01 * u.km, 100 * u.km, percent=0.01)  # Just within

    # Test outside tolerance
    assert not is_nearly_equal(102 * u.km, 100 * u.km, percent=0.01)  # 2% higher
    assert not is_nearly_equal(98 * u.km, 100 * u.km, percent=0.01)  # 2% lower

    # Test with different units (should convert automatically)
    assert is_nearly_equal(1000 * u.m, 1 * u.km, percent=0.001)

    # Test with 5% tolerance
    assert is_nearly_equal(105 * u.km, 100 * u.km, percent=0.05)  # 5% higher
    assert is_nearly_equal(95 * u.km, 100 * u.km, percent=0.05)  # 5% lower


def test_is_nearly_equal_zero_cases() -> None:
    """Test is_nearly_equal with zero expected values."""
    # Test exact zero match
    assert is_nearly_equal(0 * u.km, 0 * u.km, percent=0.001)

    # Test small values within absolute tolerance
    assert is_nearly_equal(0.0005 * u.km, 0 * u.km, percent=0.001)  # Within 0.1%
    assert is_nearly_equal(-0.0005 * u.km, 0 * u.km, percent=0.001)  # Within 0.1%

    # Test values at the boundary of absolute tolerance
    assert is_nearly_equal(0.001 * u.km, 0 * u.km, percent=0.001)  # At boundary
    assert is_nearly_equal(-0.001 * u.km, 0 * u.km, percent=0.001)  # At boundary

    # Test values outside absolute tolerance
    assert not is_nearly_equal(0.002 * u.km, 0 * u.km, percent=0.001)  # Outside
    assert not is_nearly_equal(-0.002 * u.km, 0 * u.km, percent=0.001)  # Outside

    # Test with different units for zero
    assert is_nearly_equal(0 * u.m, 0 * u.km, percent=0.001)
    assert is_nearly_equal(0 * u.mm, 0 * u.km, percent=0.001)


def test_is_nearly_equal_edge_cases() -> None:
    """Test is_nearly_equal with edge cases."""
    # Test with very small non-zero values
    assert is_nearly_equal(1e-10 * u.km, 1e-10 * u.km, percent=0.001)

    # Test with very large values
    assert is_nearly_equal(1e10 * u.km, 1e10 * u.km, percent=0.001)

    # Test with negative values
    assert is_nearly_equal(-100 * u.km, -100 * u.km, percent=0.001)
    assert is_nearly_equal(-101 * u.km, -100 * u.km, percent=0.01)  # Within 1%

    # Test with different physical quantities
    assert is_nearly_equal(100 * u.km / u.s, 100 * u.km / u.s, percent=0.001)
    assert is_nearly_equal(100 * u.kg, 100 * u.kg, percent=0.001)
