from typing import Tuple

import numpy as np
import pytest
from astropy import units as u
from poliastro.bodies import Earth, Sun
from poliastro.maneuver import Maneuver

from src.astro_constants import EARTH_A, JUPITER_A, LEO_ALTITUDE
from src.compute_utils import (
    STD_FUDGE_FACTOR,
    body_speed,
    distance_to_center,
    escape_velocity,
    get_hohmann_burns,
    get_period,
    get_semimajor_axis,
    hohmann_transfer,
    payload_mass_ratio,
    retrograde_jovian_hohmann_transfer,
    speed_around_attractor,
)


def is_nearly_equal(
    actual: u.Quantity, expected: u.Quantity, percent: float = 0.001
) -> bool:
    """Return True if two astropy Quantity values are within a certain percentage of each other.
    If expected is zero, use absolute difference.
    """
    actual = actual.to(expected.unit)
    if expected.value == 0:
        return float(abs(actual.value)) <= percent
    rel_diff = float(abs(actual.value - expected.value) / abs(expected.value))
    return rel_diff <= percent


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


def test_hohmann_transfer() -> None:
    transfer: Maneuver = hohmann_transfer(r_i=EARTH_A, r_f=JUPITER_A)
    initial_burn, final_burn = get_hohmann_burns(transfer)
    assert is_nearly_equal(initial_burn, 8757 * u.m / u.s)
    assert is_nearly_equal(final_burn, 5628 * u.m / u.s)


def test_balloon_mass() -> None:
    # This is just formula for velocity after elastic colliions from Wikpeida
    def rocket_new_velocity(
        m_r: u.Quantity, m_b: u.Quantity, v_b: u.Quantity, v_ri: u.Quantity
    ) -> u.Quantity:
        denom = m_r + m_b
        term1 = 2 * m_b * v_b / denom
        term2 = (m_r - m_b) / denom * v_ri
        v2 = term1 + term2
        return v2.to(u.km / u.s)

    def balloon_mass_test(
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
        return m_r * STD_FUDGE_FACTOR / mass

    v_b = 11 * u.km / u.s
    v_rf = 8 * u.km / u.s
    v_ri = 2 * u.km / u.s
    assert is_nearly_equal(
        balloon_mass_test(v_rf=v_rf, v_b=v_b, v_ri=v_ri),
        payload_mass_ratio(v_rf=v_rf, v_ri=v_ri, v_b=v_b),
    )


def test_escape_velocity_earth_200km() -> None:
    # 200 km altitude above Earth's surface
    altitude = 200 * u.km
    v_esc = escape_velocity(Earth, altitude)
    # The expected escape velocity at 200 km altitude is about 11.0 km/s
    expected = 11.0 * u.km / u.s
    assert is_nearly_equal(v_esc, expected)


def test_retrograde_jovian_hohmann_transfer() -> None:
    speed = retrograde_jovian_hohmann_transfer()
    expected = 69.272 * u.km / u.s
    assert is_nearly_equal(speed, expected)


def test_period() -> None:
    T = get_period(Sun, EARTH_A)
    assert is_nearly_equal(T, 1 * u.year)


def test_semi_major_axis() -> None:
    a = get_semimajor_axis(Sun, 1 * u.year)
    assert is_nearly_equal(a, EARTH_A)


def test_distance_to_center() -> None:
    d = distance_to_center(LEO_ALTITUDE, Earth)
    assert is_nearly_equal(d, Earth.R + LEO_ALTITUDE)
