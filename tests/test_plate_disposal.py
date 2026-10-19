"""Tests for the left push plate's disposal with the Moon (src/plate_disposal.py)."""

import pytest

from src.plate_disposal import encounter, next_perigee, targeted_nudge, two_body_nudge


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
