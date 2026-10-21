"""Tests for aerocapture at Saturn (src/saturn_aerocapture.py)."""

import math

import numpy as np
import pytest

from src.iapetus_cycle import OUTBOUND_BOOK, outbound_at
from src.saturn_aerocapture import (
    IAPETUS_NORMAL,
    R_EQ,
    _coplanar_leg,
    density,
    moon_leg,
    ring_zone,
    tps_share,
)
from src.saturn_moon_legs import DOGLEG_APOAPSIS, SATURN_POLE, _unit


def test_the_atmosphere_holds_one_bar_and_falls_off() -> None:
    """~0.2 kg/m^3 at 1 bar (AMAT's 0.198), and a ~50 km scale height above."""
    assert density(0.0) == pytest.approx(0.198, rel=0.02)
    assert density(300.0) / density(350.0) == pytest.approx(math.exp(50 / 48), rel=0.2)
    assert density(5000.0) == 0.0


def test_the_heat_shield_fits() -> None:
    """HEEET: 5% at 1 kJ/cm^2; carbon phenolic: Galileo's ~50% at ~200 kJ/cm^2."""
    assert tps_share(1.0e3) == pytest.approx(0.05)
    assert tps_share(2.0e5, "carbon phenolic") == pytest.approx(0.50, abs=0.03)
    assert tps_share(4.0e4) < tps_share(4.0e4, "carbon phenolic")


def test_ring_zones() -> None:
    assert ring_zone(62_000.0) == "clear"
    assert ring_zone(100_000.0) == "blocked"
    assert ring_zone(150_000.0) == "dusty"
    assert ring_zone(170_000.0) == "blocked"
    assert ring_zone(500_000.0) == "clear"


def test_a_leg_with_the_apse_in_the_moons_plane_is_the_tangent_dog_leg() -> None:
    """The sweep finds the parent's raise + match when the apse line allows it."""
    apse = _unit(np.cross(IAPETUS_NORMAL, SATURN_POLE))
    rp = R_EQ + 400.0
    leg = moon_leg(apse, IAPETUS_NORMAL, rp, DOGLEG_APOAPSIS, "Iapetus")
    assert leg is not None
    assert leg.dv == pytest.approx(_coplanar_leg("Iapetus", rp), abs=0.01)
    assert leg.meet_angle < 2.0
    assert leg.years == pytest.approx(1.17, abs=0.03)


def test_the_aeroshell_rides_on_top_of_the_saturn_stage() -> None:
    base = outbound_at(False, OUTBOUND_BOOK, saturn_dv=2.4)
    shelled = outbound_at(False, OUTBOUND_BOOK, saturn_dv=2.4, aeroshell=0.25)
    post = 1.0 + base.methalox + base.methalox_dry
    assert base.aeroshell == 0.0
    assert shelled.aeroshell == pytest.approx(post / 0.75 - post)
    assert shelled.ratio > base.ratio


@pytest.mark.slow
def test_aerocapture_closes_ring_clear_and_halves_the_outbound() -> None:
    """Every window clears the rings; the water NTR's rNPV falls but stays above $143."""
    from src.iapetus_cycle import LEDGER_A, best_k, build_cycle, value_cycle
    from src.saturn_aerocapture import aero_site, summary

    m = summary("Iapetus")
    assert m["flyable"] == m["windows"] == m["clear"]
    assert 0.2 < m["aeroshell"] < 0.3
    site = aero_site("Iapetus", True)
    assert site is not None
    k, _ = best_k(LEDGER_A, real_planes=True)
    burn = build_cycle(LEDGER_A, k, real_planes=True)
    aero = build_cycle(
        LEDGER_A, k, real_planes=True, aero=(site.outbound_dv, site.aeroshell)
    )
    assert aero.out.ratio < 0.6 * burn.out.ratio
    v_burn, v_aero = value_cycle(burn).break_even, value_cycle(aero).break_even
    assert v_burn is not None and v_aero is not None
    assert 143.0 < v_aero < 0.5 * v_burn
