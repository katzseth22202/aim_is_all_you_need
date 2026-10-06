"""Tests for src/lob_rise.py (ADR 0041)."""

import pytest

from src.lob_rise import ISP_PESSIMISTIC, booster_growth, lofted_mass, push_track


def test_lofted_mass_scales_with_the_booster() -> None:
    """At fixed thrust-to-weight and propellant fraction (ADR 0037)."""
    assert lofted_mass(0.0, scale=1.2) == pytest.approx(
        1.2 * lofted_mass(0.0), rel=1e-4
    )


@pytest.mark.parametrize("isp", [380.0, ISP_PESSIMISTIC])
def test_climbing_at_1_to_1_2_km_s_costs_7_to_11_percent_of_the_lob(isp: float) -> None:
    assert booster_growth(0.0, isp) == pytest.approx(1.0)
    assert 1.06 < booster_growth(1.0e3, isp) < 1.08
    assert 1.10 < booster_growth(1.2e3, isp) < 1.12


def test_an_unsupported_push_falls_off_the_stream_and_a_climbing_one_tracks_it() -> (
    None
):
    """The stream at ~60 km/s rises ~185 km over a ~300 s push; a craft that
    arrives at an apex ends ~400 km under it, one climbing at 1.1 km/s stays
    within ~70 km of it."""
    flat = push_track(0.0, 60.0, 10.79, 0.80)
    climbing = push_track(1.1, 60.0, 10.79, 0.80)
    assert 250.0 < flat.duration < 350.0
    assert 150.0 < flat.stream_rise < 220.0
    assert flat.below > 350.0
    assert max(climbing.below, climbing.above) < 80.0
    assert climbing.final_periapsis > 400.0
