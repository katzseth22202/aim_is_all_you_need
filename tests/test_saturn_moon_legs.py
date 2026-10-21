"""Tests for the Saturn-side legs on real planes (src/saturn_moon_legs.py)."""

import math

import numpy as np
import pytest

from src.saturn_moon_legs import (
    CAPTURE_PERIAPSIS,
    DOGLEG_APOAPSIS,
    IAPETUS_A,
    IAPETUS_NORMAL,
    PHOEBE_NORMAL,
    SATURN_POLE,
    apse_burn,
    apse_options,
    asymptote_angle,
    plane_normal,
)


def test_plane_normals_follow_inclination() -> None:
    assert plane_normal(0.0, 0.0) == pytest.approx([0.0, 0.0, 1.0])
    assert IAPETUS_NORMAL[2] == pytest.approx(math.cos(math.radians(16.778)))
    assert PHOEBE_NORMAL[2] < 0.0  # retrograde
    # Saturn's equator sits ~26.7 deg from its orbit, ~28 from the ecliptic.
    assert math.degrees(math.acos(SATURN_POLE[2])) == pytest.approx(28.0, abs=1.0)


def test_apse_options_sit_on_the_cone_and_the_plane() -> None:
    axis = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    for apse in apse_options(axis, math.radians(40.0), normal):
        assert float(apse @ normal) == pytest.approx(0.0, abs=1e-12)
        assert math.degrees(math.acos(float(apse @ axis))) == pytest.approx(40.0)
    # An axis 80 deg out of the plane cannot meet a 40 deg cone in it.
    steep = np.array([math.cos(math.radians(80.0)), 0.0, math.sin(math.radians(80.0))])
    assert apse_options(steep, math.radians(40.0), normal) == []


def test_a_coplanar_arrival_pays_only_the_periapsis_raise() -> None:
    """Asymptote in Iapetus's plane: the dog-leg raise is the coplanar 0.589."""
    e1 = np.cross(IAPETUS_NORMAL, [0.0, 0.0, 1.0])
    e1 = e1 / np.linalg.norm(e1)
    burn = apse_burn(
        e1,
        10.7,
        IAPETUS_NORMAL,
        DOGLEG_APOAPSIS,
        (0.5 * (IAPETUS_A + DOGLEG_APOAPSIS), DOGLEG_APOAPSIS),
        True,
    )
    assert burn is not None
    assert burn.dv == pytest.approx(0.589, abs=0.002)
    assert burn.turn_deg == pytest.approx(0.0, abs=0.5)
    assert burn.node_radii[0] >= CAPTURE_PERIAPSIS - 1.0


def test_the_asymptote_angle_is_supplementary_to_nu_inf() -> None:
    """At 10.7 km/s and 150 000 km the apoapsis sits ~46.5 deg from -s."""
    assert math.degrees(asymptote_angle(10.7)) == pytest.approx(46.5, abs=0.2)


@pytest.mark.slow
def test_the_real_planes_cost_little_turn_and_a_faster_capture() -> None:
    """Medians over 2035-2048: turns under 0.02 km/s; capture above 2.55."""
    from src.saturn_moon_legs import site_summary

    s = site_summary("Iapetus")
    assert s["apse"] - s["apse_coplanar"] < 0.02
    assert s["capture"] > 2.55
    assert s["node_lowest"] >= CAPTURE_PERIAPSIS - 1.0
    p = site_summary("Phoebe")
    assert p["apse"] == pytest.approx(1.453, abs=0.01)
