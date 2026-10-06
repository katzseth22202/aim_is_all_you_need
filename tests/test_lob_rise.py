"""Tests for src/lob_rise.py (ADR 0041-0043)."""

import pytest

from src.lob_rise import (
    BASELINE_RISE_SPEED,
    ISP_PESSIMISTIC,
    OPERATING_RISE_SPEED,
    booster_growth,
    braked_lob,
    lofted_mass,
    push_track,
)


def test_lofted_mass_scales_with_the_booster() -> None:
    """At fixed thrust-to-weight and propellant fraction (ADR 0037)."""
    assert lofted_mass(0.0, scale=1.2) == pytest.approx(
        1.2 * lofted_mass(0.0), rel=1e-4
    )


@pytest.mark.parametrize("isp", [380.0, ISP_PESSIMISTIC])
def test_from_an_apex_climbing_at_1_to_1_2_km_s_costs_7_to_11_percent(
    isp: float,
) -> None:
    """ADR 0041's first measure, kept as ``baseline=0`` with no brake."""

    def growth(v: float) -> float:
        return booster_growth(v, isp, baseline=0.0, brake=False)

    assert growth(0.0) == pytest.approx(1.0)
    assert 1.06 < growth(1.0e3) < 1.08
    assert 1.10 < growth(1.2e3) < 1.12


@pytest.mark.parametrize("isp", [380.0, ISP_PESSIMISTIC])
def test_only_the_climb_above_the_priced_lob_is_charged(isp: float) -> None:
    """ADR 0042: the parent's lob already passes 400 km at ~0.75 km/s.  The
    unbraked charge, kept as ``brake=False``."""

    def growth(v: float) -> float:
        return booster_growth(v, isp, brake=False)

    assert growth(BASELINE_RISE_SPEED) == pytest.approx(1.0)
    assert 1.02 < growth(1.0e3) < 1.04
    assert 1.04 < growth(OPERATING_RISE_SPEED) < 1.05
    assert 1.05 < growth(1.2e3) < 1.07


@pytest.mark.parametrize("isp", [380.0, ISP_PESSIMISTIC])
def test_the_priced_lobs_brake_reproduces_the_parents(isp: float) -> None:
    """``sec:vertical_lob``: 1.33-1.38 km/s, 114-185 t with the landing burn,
    and the unbraked booster crossing 60 km at ~2.6 km/s."""
    lob = braked_lob(BASELINE_RISE_SPEED, isp=isp)
    assert 1.33e3 < lob.brake < 1.40e3
    assert 114.0e3 < lob.reserve < 185.0e3
    assert lob.unbraked_entry == pytest.approx(2.6e3, abs=0.05e3)
    # Skipping the brake lofts 9-19% more.
    assert 1.09 < lofted_mass(BASELINE_RISE_SPEED, isp=isp) / lob.lofted < 1.19


@pytest.mark.parametrize("isp", [380.0, ISP_PESSIMISTIC])
def test_a_faster_climb_brakes_harder_and_charges_more(isp: float) -> None:
    """ADR 0043: separating higher and faster, the booster brakes 1.53-1.58 km/s
    at 1.1 km/s, and the charge rises from x1.044-1.048 to x1.065-1.074."""
    slow, fast = braked_lob(BASELINE_RISE_SPEED, isp=isp), braked_lob(
        OPERATING_RISE_SPEED, isp=isp
    )
    assert fast.separation_altitude > slow.separation_altitude
    assert 1.50e3 < fast.brake < 1.60e3
    assert fast.reserve > slow.reserve
    assert 1.06 < booster_growth(OPERATING_RISE_SPEED, isp) < 1.08
    assert booster_growth(OPERATING_RISE_SPEED, isp) > booster_growth(
        OPERATING_RISE_SPEED, isp, brake=False
    )


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
