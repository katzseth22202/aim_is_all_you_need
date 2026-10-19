"""Tests for the Jupiter-only chain on plates (src/plate_jupiter_loop.py)."""

import math

import pytest

from src.plate_jupiter_loop import (
    START,
    Carrier,
    LoopWindow,
    chain_rate,
    fly_carrier,
    loop_gates,
    loop_windows,
    value_chain,
    window_growth,
)
from src.solar_dive_growth import AVIONICS_FRACTION, VE_METHALOX


def _no_loss(burn: float, seconds: float) -> float:
    del burn, seconds
    return 0.0


def _window(return_burn: float = 0.0) -> LoopWindow:
    """A synthetic 3S window: ~12.3 km/s out, a ~61 km/s stream nearly head-on."""
    carrier = Carrier(12.3, 0.0, return_burn, _no_loss)
    return LoopWindow(
        index=0,
        multiple=3,
        period_years=3.276,
        jupiter_years=1.2,
        push_stream=63.0,
        push_axis_deg=167.0,
        stream_excess=60.0,
        stream_axis_deg=167.0,
        growth=carrier,
        rods=carrier,
        arrival_multiple=3,
        arrival_years=3.276,
        arrival_jupiter_years=1.2,
        arrival_departure_excess=12.3,
    )


def test_the_ship_returns_its_departing_stack_less_avionics_and_its_burn() -> None:
    plain = fly_carrier(_window(), _window().rods, START)
    lines = plain.lines
    assert plain.returned == pytest.approx(lines["departing"] * (1 - AVIONICS_FRACTION))
    assert plain.disks == 0.0 and lines["sun_plate"] == 0.0 and lines["shield"] == 0.0
    burned = fly_carrier(_window(0.2), _window(0.2).rods, START)
    assert burned.returned == pytest.approx(
        plain.returned * math.exp(-0.2 / VE_METHALOX)
    )


def test_a_nearly_head_on_stream_is_met_on_the_visible_sign_and_reversed() -> None:
    lines = fly_carrier(_window(), _window().rods, START).lines
    assert 150.0 < lines["impact_angle"] <= 180.0
    # A turn of ~180 degrees at apoapsis is the child's reversal plus the raise.
    assert lines["methalox_dv"] == pytest.approx(0.38, abs=0.02)


def test_the_gates_have_no_sun_and_step_at_the_seeds_return() -> None:
    gates = loop_gates(_window())
    assert [g.odds for g in gates] == [0.95, 0.85, 0.90]
    assert gates[-1].years == pytest.approx(3.276)


def test_a_synthetic_chain_grows_and_breaks_even() -> None:
    windows = [_window()] * 3
    rate = chain_rate(windows, START)
    assert rate.growths[0] == pytest.approx(window_growth(windows[0], START).growth)
    assert rate.doubling == pytest.approx(
        3.276 * math.log(2) / math.log(rate.growths[0])
    )
    value = value_chain(windows, START, seed_price_per_kg=2000.0)
    assert value.break_even is not None and value.break_even > value.steady_cost > 0


@pytest.mark.slow
def test_the_flown_chain_is_the_childs_and_its_seed_leaves_at_12_km_s() -> None:
    windows = loop_windows("flown")
    assert len(windows) == 11
    assert windows[0].arrival_multiple == 3
    assert windows[0].arrival_departure_excess == pytest.approx(12.0, abs=0.3)
    # The departure streams arrive nearly head-on, at 52-63 km/s excess.
    for w in windows:
        assert 50.0 < w.stream_excess < 66.0
        assert w.rods.excess > 11.5
