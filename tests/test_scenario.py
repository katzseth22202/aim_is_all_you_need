"""Tests for scenario-related functions in scenario.py."""

from astropy import units as u

from src.astro_constants import CERES_A, EARTH_A, SATURN_A
from src.scenario import solar_impact_dv
from tests.test_helpers import is_nearly_equal


def test_solar_impact_dv() -> None:
    # solar_impact_dv reproduces the paper's solar-impact delta-v figures, which
    # equal the heliocentric circular orbital velocity at each body's distance:
    # ~30 km/s from Earth, ~10 km/s from Saturn, ~18 km/s from Ceres
    # (citation audit L1047 Earth/Saturn, L1107 Ceres).
    assert is_nearly_equal(solar_impact_dv(EARTH_A), 29.78 * u.km / u.s, percent=0.01)
    assert is_nearly_equal(solar_impact_dv(SATURN_A), 9.64 * u.km / u.s, percent=0.01)
    assert is_nearly_equal(solar_impact_dv(CERES_A), 17.91 * u.km / u.s, percent=0.01)
