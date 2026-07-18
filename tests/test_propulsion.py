"""Tests for propulsion-related functions in propulsion.py."""

import pytest
from astropy import units as u
from boinor.bodies import Earth, Moon

from src.astro_constants import STD_FUDGE_FACTOR
from src.propulsion import (
    burn_for_v_infinity,
    exhaust_velocity_from_isp,
    payload_mass_ratio,
    retrograde_jovian_hohmann_transfer,
    rocket_equation,
)
from tests.test_helpers import is_nearly_equal


def test_payload_mass_ratio() -> None:
    """Test payload_mass_ratio function using the PuffSat mass test."""

    # This is just formula for velocity after elastic collisions from Wikipedia
    def rocket_new_velocity(
        m_r: u.Quantity, m_b: u.Quantity, v_b: u.Quantity, v_ri: u.Quantity
    ) -> u.Quantity:
        denom = m_r + m_b
        term1 = 2 * m_b * v_b / denom
        term2 = (m_r - m_b) / denom * v_ri
        v2 = term1 + term2
        return v2.to(u.km / u.s)

    def puffsat_mass_test(
        v_rf: u.Quantity,
        v_b: u.Quantity,
        v_ri: u.Quantity = 0 * u.km / u.s,
        mass_factor: int = 50000,
        fudge_factor: float = STD_FUDGE_FACTOR,
    ) -> float:
        m_r = mass_factor * u.kg
        m_b = 1 * u.kg
        current_v = v_ri
        mass = 0 * u.kg
        while True:
            current_v = rocket_new_velocity(m_r=m_r, m_b=m_b, v_b=v_b, v_ri=current_v)
            if current_v >= v_rf:
                break
            mass += m_b
        return float((m_r * STD_FUDGE_FACTOR / mass).value)

    v_b = 11 * u.km / u.s
    v_rf = 8 * u.km / u.s
    v_ri = 2 * u.km / u.s
    assert is_nearly_equal(
        puffsat_mass_test(v_rf=v_rf, v_b=v_b, v_ri=v_ri),
        payload_mass_ratio(v_rf=v_rf, v_ri=v_ri, v_b=v_b),
    )


def test_rocket_equation() -> None:
    """Test rocket_equation function."""
    print(rocket_equation(9.8 * u.km / u.s, 4.5 * u.km / u.s))
    assert is_nearly_equal(
        u.Quantity(0.88670699, u.dimensionless_unscaled),
        rocket_equation(9.8 * u.km / u.s, 4.5 * u.km / u.s),
    )


def test_exhaust_velocity_from_isp() -> None:
    """Test exhaust_velocity_from_isp function (v_e = Isp * g0)."""
    # Methalox sea-level Isp of 310 s -> 310 * 9.80665 = 3040.06 m/s.
    ve = exhaust_velocity_from_isp(310 * u.s)
    assert is_nearly_equal(ve, 3.04006 * u.km / u.s, percent=0.001)


def test_retrograde_jovian_hohmann_transfer() -> None:
    """Test retrograde_jovian_hohmann_transfer function.

    69.2702 km/s, not the previously-computed 69.2730: the old implementation
    built its initial circular orbit via boinor's Orbit.circular(attractor,
    alt), whose second argument is altitude above the attractor's *surface*,
    not absolute orbital radius -- so passing JUPITER_A there silently
    inflated Jupiter's orbital radius by Sun.R (~695,700 km, ~0.09% of
    Jupiter's orbital radius). Confirmed independently via a closed-form
    vis-viva calculation.
    """
    speed = retrograde_jovian_hohmann_transfer()
    expected = 69.2702 * u.km / u.s
    assert is_nearly_equal(speed, expected, percent=1e-4)


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
