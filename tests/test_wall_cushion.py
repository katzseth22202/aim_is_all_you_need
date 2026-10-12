"""Tests for src/wall_cushion.py: a tile on a gas cushion (ADR 0054)."""

import pytest
from astropy import units as u

from src.wall_cushion import cushion_response

TILE = 80.0 * u.kg / u.m**2


def test_a_precharge_above_the_peak_never_lifts_the_tile_without_a_spike() -> None:
    r = cushion_response(TILE, 6.0 * u.cm, 120.0 * u.bar, 0.0 * u.kPa * u.s)
    assert r.travel.to_value(u.cm) == pytest.approx(0.0, abs=1e-9)
    assert r.backwall_ratio == pytest.approx(120.0 / 94.0)


def test_the_cushion_turns_a_5_kpa_s_spike_into_a_mild_backwall_load() -> None:
    """The ADR 0054 row: 6 cm at 80 bar, travel ~1.9 cm, backwall 1.43 x."""
    r = cushion_response(TILE, 6.0 * u.cm, 80.0 * u.bar, 5.0 * u.kPa * u.s)
    assert r.travel.to_value(u.cm) == pytest.approx(1.87, abs=0.05)
    assert r.backwall_ratio == pytest.approx(1.43, abs=0.03)
    assert r.backwall_ratio < 1.8  # the lowest solved hoop overshoot D


def test_damping_shortens_the_travel() -> None:
    free = cushion_response(TILE, 6.0 * u.cm, 80.0 * u.bar, 2.0 * u.kPa * u.s)
    damped = cushion_response(
        TILE, 6.0 * u.cm, 80.0 * u.bar, 2.0 * u.kPa * u.s, 1.0e5 * u.Pa * u.s / u.m
    )
    assert damped.travel < free.travel
    assert damped.backwall_ratio < free.backwall_ratio


def test_a_thin_soft_cushion_slams_the_tile_into_the_backwall() -> None:
    """1 cm at 20 bar uses 99% of its travel and passes 200 times the steady
    load: the cushion has to be sized, not just present."""
    r = cushion_response(TILE, 1.0 * u.cm, 20.0 * u.bar, 5.0 * u.kPa * u.s)
    assert r.travel.to_value(u.cm) > 0.99
    assert r.backwall_ratio > 100.0
