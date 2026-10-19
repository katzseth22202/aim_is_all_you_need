"""Tests for the solar-dive charge-pusher cycle (src/solar_dive_growth.py)."""

import math

import pytest

from src.solar_dive_growth import (
    MISS_FIRST,
    MISS_FLOOR,
    Design,
    Gate,
    GatedSchedule,
    StageLaws,
    apoapsis_methalox,
    arc_loss,
    carrier_burn,
    carrier_perijove,
    cycle_growth,
    departure_burn,
    departure_route,
    doubling_years,
    lob_climb_factor,
    miss_fraction,
    push_climb,
    push_route,
    ship_perijove_rj,
    value_design,
    visible_departure,
)


def test_stage_laws_reproduce_the_simulations_optimum_inputs() -> None:
    """growth_cycle.csv's 9.86 R_sun row: v_e 5.74 km/s at k_E 24, 29.9 at k_S 30."""
    laws = StageLaws.load()
    assert laws.earth_ve(24.0, 103.2, 101.15) == pytest.approx(5.74, rel=0.01)
    assert laws.sun_ve(29.907, 391.692) == pytest.approx(30.07, rel=1e-3)
    # The overtake handoff: mass ratio ~1.49 for 11 km/s at k = 10.
    assert math.exp(11.0 / laws.push_ve(10.0, 100.0)) == pytest.approx(1.493, rel=1e-3)


def test_the_bowl_law_reproduces_the_simulations_hole_ve() -> None:
    """departure_hole_law.csv's visible routes, within the k-interpolation."""
    laws = StageLaws.load()
    for k, v in ((5.0, 13.601), (10.0, 10.716), (20.0, 7.658), (30.0, 5.774)):
        assert laws.hole_ve(k, 84.0, 129.9) == pytest.approx(v, rel=0.006)
    for k, v in ((5.0, 20.310), (20.0, 11.272)):
        assert laws.hole_ve(k, 107.9, 117.0) == pytest.approx(v, rel=0.006)
    condensed = StageLaws.load(bonds="bonds returned")
    assert condensed.hole_ve(30.0, 84.0, 129.9) == pytest.approx(7.907, rel=0.006)


def test_stage_laws_refuse_to_extrapolate_the_slug_ratio() -> None:
    with pytest.raises(ValueError):
        StageLaws.load().sun_ve(10.0, 391.7)


def test_the_10_day_orbit_departure_is_dearer_than_the_simulations_4_15() -> None:
    """4.15 km/s starts from parabolic speed at 200 km; from 600 km it is 4.34."""
    assert departure_burn(10.428) == pytest.approx(4.340, abs=2e-3)
    assert departure_burn(10.539) == pytest.approx(4.418, abs=2e-3)


def test_the_apoapsis_turn_is_the_childs_reversal_at_180_degrees() -> None:
    # ADR 0071's 10-day reversal is 372 m/s; the raise to 600 km adds a little.
    # (The ledger reverses the 400 km orbit; this reverses the raised one.)
    assert apoapsis_methalox(180.0) == pytest.approx(0.372 + 0.004, abs=0.02)
    assert apoapsis_methalox(0.0) < 0.01


def test_the_sun_arc_loss_is_second_order() -> None:
    class _Route:
        arc_time_scale = 9.73

    assert arc_loss(4.0, _Route()) == pytest.approx((4 / 9.73) ** 2 * (1 / 96 + 1 / 24))  # type: ignore[arg-type]


def test_gated_schedule_weights_each_flow_by_the_gates_before_it() -> None:
    schedule = GatedSchedule(0.10, (Gate("a", 1.0, 0.5), Gate("b", 2.0, 0.8)))
    assert schedule.factor(0.5) == pytest.approx(1.1**-0.5)
    assert schedule.factor(1.5) == pytest.approx(0.5 * 1.1**-1.5)
    assert schedule.factor(3.0) == pytest.approx(0.4 * 1.1**-3.0)
    assert schedule.cumulative == pytest.approx(0.4)


def test_miss_fraction_learns_to_its_floor() -> None:
    assert miss_fraction(1.0, 1.0) == pytest.approx(MISS_FIRST)
    assert miss_fraction(2.0, 1.0) == pytest.approx(0.8 * MISS_FIRST)
    assert miss_fraction(1e9, 1.0) == pytest.approx(MISS_FLOOR)


def test_doubling_years() -> None:
    assert doubling_years(2.0, 3.0) == pytest.approx(3.0)
    assert math.isinf(doubling_years(0.9, 3.0))


@pytest.mark.slow
def test_the_closure_is_the_one_the_simulation_copied() -> None:
    route = departure_route(9.86, 90.0)
    assert route is not None
    assert route.departure_excess == pytest.approx(10.428, abs=1e-3)
    assert route.periapsis_boost == pytest.approx(20.451, abs=1e-3)
    assert route.earth_excess == pytest.approx(100.85, abs=0.01)
    assert route.sun_closing == pytest.approx(391.7, abs=0.1)


@pytest.mark.slow
def test_the_disks_need_the_true_mirror_excess_and_a_lower_perijove() -> None:
    """G10: 15.07 km/s, not the 14.78 floor; their turn fails above ~6.7 R_J."""
    route = departure_route(9.86, 90.0)
    assert route is not None
    assert route.mirror_excess == pytest.approx(15.067, abs=2e-3)
    assert ship_perijove_rj(route) == pytest.approx(7.86, abs=0.01)
    assert carrier_burn(route, 10.0).bend_available_deg < route.disk_bend_deg
    assert carrier_perijove(route, 10.0) == pytest.approx(6.72, abs=0.02)
    assert carrier_burn(route, 6.5).burn == pytest.approx(1.672, abs=2e-3)


@pytest.mark.slow
def test_the_push_wave_returns_ten_days_early_on_a_faster_climb() -> None:
    departure = departure_route(9.86, 90.0)
    push = push_route(9.86, 90.0)
    assert departure is not None and push is not None
    assert departure.cycle_years - push.cycle_years == pytest.approx(
        10 / 365.25, abs=1e-6
    )
    assert push.return_excess > departure.return_excess


@pytest.mark.slow
def test_stripped_of_the_push_the_ledger_is_the_simulations_growth() -> None:
    """With the simulation's own inputs (4.15 km/s, no push) G is 8.4 / 7.0."""
    route = departure_route(9.86, 90.0)
    assert route is not None
    laws = StageLaws.load()
    ve_e = laws.earth_ve(24.0, route.departure_closing, route.impact_angle_deg)
    ve_s = laws.sun_ve(30.0, route.sun_closing)
    d = 24.0 / (math.exp(4.15 / ve_e) - 1.0)
    mr_s = math.exp(route.periapsis_boost / ve_s)
    for dead, expected in ((0.0, 8.4), (0.1599, 7.0)):
        growth = d * (1 - dead) / mr_s / (1 + d * (1 - 1 / mr_s) / 30.0)
        assert growth == pytest.approx(expected, abs=0.05)


@pytest.mark.slow
def test_everything_charged_the_cycle_still_grows_and_breaks_even() -> None:
    departure = departure_route(9.86, 90.0)
    push = push_route(9.86, 90.0)
    assert departure is not None and push is not None
    growth = cycle_growth(departure, push, Design(), 0.97)
    assert 1.0 < growth.growth < 8.4
    assert growth.per_total_impactor < growth.growth
    value = value_design(departure, push, Design())
    assert value.break_even is not None and value.break_even > value.steady_cost > 0.0


@pytest.mark.slow
def test_the_visible_sign_is_the_parents_s38_geometry() -> None:
    """ADR 0072's sign hides the stream behind Earth; the other is S38's."""
    departure = departure_route(9.86, 60.0)
    assert departure is not None
    assert departure.impact_angle_deg == pytest.approx(89.2, abs=0.1)
    seen = visible_departure(departure, departure)
    assert seen.impact_angle_deg == pytest.approx(129.9, abs=0.1)
    assert seen.closing == pytest.approx(84.0, abs=0.1)
    deep = departure_route(4.0, 90.0)
    assert deep is not None
    seen = visible_departure(deep, deep)
    assert seen.impact_angle_deg == pytest.approx(117.0, abs=0.1)
    assert seen.closing == pytest.approx(107.9, abs=0.1)


@pytest.mark.slow
def test_the_1_g_push_climbs_faster_and_the_childs_push_does_not() -> None:
    assert 2.25 < push_climb(1.0) < 2.5
    assert push_climb(3.26) == pytest.approx(1.1)
    assert 1.4 < lob_climb_factor(push_climb(1.0)) < 1.6
    assert lob_climb_factor(1.1) == 1.0
