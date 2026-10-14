"""Tests for the departure-lever ranking (ADR 0063)."""

import pytest
from astropy import units as u

from src.chamber_isp import HYDROGEN_BULGED
from src.departure_levers import (
    HOT_PULSE_RATES,
    HOT_TEMPERATURES,
    RECOVERED,
    cases,
    hot_hydrogen,
)


def test_a_hotter_charge_is_smaller():
    cool, _ = hot_hydrogen(7000.0, 0.94)
    hot, _ = hot_hydrogen(10000.0, 0.94)
    assert HYDROGEN_BULGED.reference_slug_ratio == 24.0
    assert cool.reference_slug_ratio == pytest.approx(9.51, abs=0.01)
    assert hot.reference_slug_ratio == pytest.approx(5.49, abs=0.01)


def test_lost_bonds_lower_the_ceiling_and_the_efficiency():
    full, eta_full = hot_hydrogen(10000.0, 0.94)
    half, eta_half = hot_hydrogen(10000.0, 0.50)
    assert full.chemistry_ceiling == pytest.approx(1 - 0.6 * 0.06)
    assert half.chemistry_ceiling == pytest.approx(0.7)
    assert eta_half < eta_full < 0.773


def test_the_hot_rows_keep_the_bulged_chambers_wall_and_tanks():
    pairing, _ = hot_hydrogen(7000.0, 0.70)
    assert pairing.wall_mass == HYDROGEN_BULGED.wall_mass
    assert pairing.tank_fraction == HYDROGEN_BULGED.tank_fraction
    assert pairing.temperature == 7000.0 * u.K


def test_every_lever_is_listed_once():
    labels = [label for label, _ in cases()]
    assert len(labels) == len(set(labels))
    hot = [label for label in labels if "unsolved" in label]
    assert len(hot) == len(HOT_TEMPERATURES) * len(RECOVERED) * len(HOT_PULSE_RATES)
    rates = {d.pairing.pulse_rate.to_value(1 / u.s) for _, d in cases() if d.pairing}
    assert {3.0, 4.0} <= rates
