"""Tests for milestone 1, the plate loop on a methalox departure (src/methalox_jupiter_loop.py)."""

import math
from dataclasses import replace

import pytest

from src.chamber_departure import role_growth
from src.methalox_jupiter_loop import (
    METHALOX_START,
    fly_methalox,
    methalox_pair,
    methalox_windows,
    optimize_methalox,
    slow_departures,
    stream_excess,
    thrust_axis,
)
from src.plate_jupiter_loop import Carrier, LoopWindow, value_chain, window_growth
from src.solar_dive_growth import (
    AVIONICS_FRACTION,
    PUSH_ALTITUDE,
    VE_METHALOX,
    StageLaws,
    wave_speed,
)


def _no_loss(burn: float, seconds: float) -> float:
    del burn, seconds
    return 0.0


def _window(return_burn: float = 0.0) -> LoopWindow:
    """A synthetic 3S window: ~12.3 km/s out, pushed by a ~56 km/s stream."""
    carrier = Carrier(12.3, 0.0, return_burn, _no_loss)
    return LoopWindow(
        index=0,
        multiple=3,
        period_years=3.276,
        jupiter_years=1.2,
        push_stream=56.0,
        push_axis_deg=167.0,
        stream_excess=stream_excess(56.0),
        stream_axis_deg=167.0,
        growth=carrier,
        rods=carrier,
        arrival_multiple=3,
        arrival_years=3.276,
        arrival_jupiter_years=1.2,
        arrival_departure_excess=12.3,
    )


def _law() -> StageLaws:
    return StageLaws.load(bonds=METHALOX_START.bonds)


def test_with_no_rods_spent_every_unit_pushes() -> None:
    assert role_growth(100.0, 130.0, 130.0, 0.0, 0.0, 0.9, 0.9) == pytest.approx(
        130.0 * 0.9 / 100.0
    )


def test_the_stream_excess_inverts_the_wave_speed() -> None:
    assert wave_speed(stream_excess(56.0), PUSH_ALTITUDE) == pytest.approx(56.0)


def test_the_burn_axis_is_the_side_nearer_the_push() -> None:
    near = thrust_axis(12.3, 0.0, 167.0)
    far = thrust_axis(12.3, 0.0, -167.0)
    assert near == pytest.approx(-far)
    assert near > 0.0


def test_a_methalox_unit_spends_no_rods_and_leaves_its_push_plate() -> None:
    unit = fly_methalox(_window(), METHALOX_START, _law())
    m = unit.lines
    assert unit.departure_sheets == 0.0 and m["bowl"] == 0.0
    assert m["push_plate"] > 0.0 and m["plate_disposal"] > 0.0
    assert m["departure_engines"] >= 1.0
    # The Raptor burn dominates the stack: 5.9 km/s at 3.73 km/s exhaust.
    assert (
        0.6 < m["departure_methalox"] / (m["departure_methalox"] + m["departing"]) < 0.9
    )
    assert unit.returned == pytest.approx(m["departing"] * (1 - AVIONICS_FRACTION))
    burned = fly_methalox(_window(0.2), METHALOX_START, _law())
    assert burned.returned == pytest.approx(
        unit.returned * math.exp(-0.2 / VE_METHALOX)
    )


def test_a_shared_spring_has_no_bowl_to_share() -> None:
    with pytest.raises(ValueError):
        fly_methalox(_window(), replace(METHALOX_START, shared_spring=True), _law())


def test_the_growth_is_the_return_over_the_push_sheets() -> None:
    w = _window()
    g = window_growth(w, METHALOX_START, arrives=0.95, fly=methalox_pair)
    unit = g.push_unit
    assert g.push_share == pytest.approx(1.0)
    assert g.growth == pytest.approx(unit.returned * 0.95 / unit.push_sheets)


def test_a_synthetic_methalox_chain_is_valued() -> None:
    value = value_chain(
        [_window()] * 3, METHALOX_START, seed_price_per_kg=2000.0, fly=methalox_pair
    )
    assert value.seed_mass > 0.0 and value.seed_outlay > 0.0
    assert value.break_even is None or value.break_even > value.steady_cost


@pytest.mark.slow
def test_milestone_one_barely_grows_on_three_synodic_windows() -> None:
    best = optimize_methalox("3S only")
    assert 1.0 < best.valuation.growth < 1.2
    assert best.design.k_push == pytest.approx(5.0, abs=0.05)  # the table's floor
    assert best.dear_break_even is not None and best.dear_break_even > 5000.0
    # Two-synodic windows shrink it: the flown chain does not grow.
    flown = optimize_methalox("flown")
    assert flown.valuation.growth < 1.0


@pytest.mark.slow
def test_no_slower_departure_closes_on_the_3s_to_5s_clocks() -> None:
    windows = methalox_windows("3S only")
    assert len(windows) == 9
    rows = slow_departures(METHALOX_START)
    assert min(r.departure_excess for r in rows) > 11.0
    assert min(r.stream for r in rows) > 45.0
