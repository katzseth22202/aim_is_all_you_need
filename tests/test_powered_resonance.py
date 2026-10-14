"""Tests for the powered 3S flyby scan (ADR 0070)."""

import numpy as np
import pytest

from src.powered_resonance import (
    CHAMBER_EXHAUSTS,
    JUPITER_RADIUS_KM,
    best_for,
    frontier,
    perijove_burn,
    perijove_for_turn,
    powered_closures,
    split_bend,
)
from src.retrograde_return_legs import _powered_flyby_params

PARAMS = _powered_flyby_params()
MU, FLOOR = PARAMS.mu_jupiter, PARAMS.periapsis_floor


def test_the_legs_turn_pins_the_perijove_where_leverage_is_moderate():
    """ADR 0070's worked example: 13.77 in, 15.70 out, a 69.86 degree turn."""
    perijove = perijove_for_turn(13.77, 15.70, np.radians(69.86), MU, FLOOR)
    assert perijove / JUPITER_RADIUS_KM == pytest.approx(6.12, abs=0.02)
    assert perijove_burn(13.77, 15.70, perijove, MU) == pytest.approx(1.0, abs=0.02)
    # Falling to the floor would halve the burn but over-turn by ~54 degrees.
    assert perijove_burn(13.77, 15.70, FLOOR, MU) == pytest.approx(0.48, abs=0.02)
    assert np.degrees(split_bend(FLOOR, 13.77, 15.70, MU)) == pytest.approx(
        124.5, abs=1.0
    )


def test_a_turn_no_perijove_can_give_is_refused():
    assert perijove_for_turn(15.0, 15.0, np.radians(179.0), MU, FLOOR) is None


@pytest.mark.slow
def test_the_unpowered_closure_wins_for_every_chamber():
    closures = powered_closures(3)
    for exhaust in CHAMBER_EXHAUSTS.values():
        best = best_for(closures, exhaust)
        assert abs(best.jupiter_burn) < 0.01
        assert best.departure_vinf == pytest.approx(11.60, abs=0.05)
    cheap = [c for c in frontier(closures) if 10.5 <= c.departure_vinf < 11.0]
    assert cheap[0].jupiter_burn == pytest.approx(1.006, abs=0.01)
