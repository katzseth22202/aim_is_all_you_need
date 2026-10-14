"""Tests for the impact-angle ledger (ADR 0066)."""

import pytest
from astropy import units as u

from src.departure_visibility import OnwardStream
from src.impact_angle import angle_ledger
from src.orbit_turn_analysis import (
    THREE_SYNODIC_AIM_SEPARATION,
    THREE_SYNODIC_DEPARTURE_VINF,
    THREE_SYNODIC_RETURN_VINF,
)
from src.survivable_chamber import HYDROGEN_BULGED_DESIGN

REFERENCE = OnwardStream(
    0,
    3,
    float(THREE_SYNODIC_DEPARTURE_VINF.to_value(u.km / u.s)),
    float(THREE_SYNODIC_RETURN_VINF.to_value(u.km / u.s)),
    float(THREE_SYNODIC_AIM_SEPARATION.to_value(u.deg)),
)


@pytest.mark.slow
def test_adr_0045s_reference_burn_swings_19_8_to_11_7_degrees():
    angles = angle_ledger(REFERENCE, 1200.0, 11.0, HYDROGEN_BULGED_DESIGN)
    assert angles.off_head_on_start == pytest.approx(19.8, abs=0.2)
    assert angles.off_head_on_end == pytest.approx(11.7, abs=0.2)
    # Off head-on helps, but only by about a percent.
    assert 1.0 < angles.mass_gain < 1.02
    assert angles.head_on_isp < angles.bent_isp < angles.gross_isp
