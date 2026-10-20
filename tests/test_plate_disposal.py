"""Tests for the left push plate's disposal with the Moon (src/plate_disposal.py)."""

import pytest

from src.plate_disposal import (
    encounter,
    next_perigee,
    parking_drift,
    targeted_nudge,
    two_body_nudge,
)


def test_with_the_moon_far_and_tiny_the_two_body_nudge_lands_at_100_km() -> None:
    """With the Moon's pull switched off, the propagation is two-body."""
    import src.plate_disposal as module

    saved = module._MU_MOON, module._MU_SUN
    module._MU_MOON, module._MU_SUN = 0.0, 0.0
    try:
        flown = next_perigee(10.0, two_body_nudge(10.0), 180.0, 0.0)
        assert flown.perigee == pytest.approx(100.0, abs=0.5)
        assert flown.days == pytest.approx(5.0, abs=0.01)
    finally:
        module._MU_MOON, module._MU_SUN = saved


@pytest.mark.slow
def test_the_moon_moves_the_next_perigee_by_hundreds_of_km() -> None:
    """ADR 0076: Moon behind the apoapsis raises it; ahead, it drops it to Earth."""
    behind = next_perigee(10.0, 0.0, 225.0, 0.0)
    assert 700.0 < behind.perigee < 950.0
    assert next_perigee(10.0, 0.0, 90.0, 0.0).perigee < 1.0
    assert targeted_nudge(10.0, 90.0, 0.0) == 0.0
    assert targeted_nudge(10.0, 225.0, 0.0) == pytest.approx(0.0099, abs=5e-4)


def test_encounter_phases_wrap() -> None:
    assert encounter(0.0) and encounter(345.0) and encounter(15.0)
    assert not encounter(30.0) and not encounter(330.0)


def test_without_moon_or_sun_the_ship_departs_from_600_km_on_time() -> None:
    """S47 item 7: the two-body parking orbit is the drift's zero."""
    for days in (10.0, 5.0):
        flown = parking_drift(days, 0.0, 0.0, moon=False, sun=False)
        assert flown.altitude == pytest.approx(600.0, abs=0.5)
        assert abs(flown.hours) < 1e-3 and abs(flown.apse_deg) < 1e-3


def test_the_suns_tide_alone_moves_the_10_day_departure_by_hours() -> None:
    """Even with the Moon far, the 10-day orbit must be navigated."""
    flown = parking_drift(10.0, 0.0, 0.0, 0.0, moon=False)
    assert flown.hours == pytest.approx(3.13, abs=0.05)
    assert not flown.captured


def test_the_moon_captures_the_10_day_pass_near_apoapsis_but_never_the_5_day() -> None:
    """S47 item 7: 335-30 deg blocks the 10-day gap; the 5-day apoapsis is clear."""
    assert parking_drift(10.0, 15.0, 5.1).captured
    assert not parking_drift(10.0, 180.0, 5.1).captured
    for phase in (0.0, 15.0, 345.0):
        assert not parking_drift(5.0, phase, 5.1).captured
