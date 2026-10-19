"""Tests for the solar-dive charge-pusher cycle (src/solar_dive_growth.py)."""

import math

import numpy as np
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
    bag_mass,
    carrier_burn,
    carrier_perijove,
    cycle_growth,
    delivery_unit,
    departure_burn,
    departure_route,
    doubling_years,
    gas_spring,
    lob_climb_factor,
    miss_fraction,
    peak_to_mean,
    periapsis_speed,
    plate_disposal_dv,
    push_climb,
    push_route,
    sheet_material,
    ship_perijove_rj,
    spring_plate_stroke,
    value_design,
    visible_departure,
)


def test_stage_laws_reproduce_the_simulations_optimum_inputs() -> None:
    """growth_cycle.csv's 9.86 R_sun row: v_e 5.74 km/s at k_E 24, 29.9 at k_S 30."""
    laws = StageLaws.load()
    assert laws.earth_ve(24.0, 103.2, 101.15) == pytest.approx(5.74, rel=0.01)
    assert laws.sun_ve(29.907, 391.692) == pytest.approx(30.07, rel=1e-3)
    # The overtake handoff: mass ratio ~1.49 for 11 km/s at k = 10.
    assert math.exp(11.0 / laws.push_ve(10.0, 100.0, law="linear")) == pytest.approx(
        1.493, rel=1e-3
    )


def test_the_fast_wave_law_reproduces_the_simulations_closures() -> None:
    """S43 item 1: fast_wave_law.csv's re-priced v_e, from the vendored grid."""
    laws = StageLaws.load()
    for k, w, impactor, ve in (
        (7.07, 99.2, "Kapton", 35.540),
        (6.67, 137.1, "graphite", 51.340),
        (7.93, 132.1, "graphite", 44.089),
        (8.21, 207.8, "graphite", 62.286),
    ):
        assert laws.push_ve(k, w, impactor) == pytest.approx(ve, rel=0.002)
    # 1-4% below the linear law at 99-137 km/s, ~9% at 208.
    assert laws.push_ve(8.21, 207.8, "graphite") / laws.push_ve(
        8.21, 207.8, law="linear"
    ) == pytest.approx(0.911, abs=0.006)
    assert sheet_material(132.0) == "graphite" and sheet_material(120.0) == "Kapton"


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


def test_the_left_push_plates_nudge_is_charged_for_the_moon() -> None:
    """S43 item 5: 4.2 m/s two-body, 0-15 m/s with the Moon; 9.2 m/s at 3 days."""
    from src.plate_disposal import two_body_nudge

    assert two_body_nudge(10.0) == pytest.approx(0.0042, abs=2e-4)
    assert plate_disposal_dv(10.0) == 0.015
    assert plate_disposal_dv(1.0) == pytest.approx(two_body_nudge(1.0))


def test_the_spring_law_matches_a_direct_integration() -> None:
    """S45: peak / mean and the stroke, integrated, on a craft only 6x the plate."""
    from scipy.integrate import solve_ivp

    m, craft, rate, theta, force = 63.0e3, 380.0e3, 2.0, 2.0 * math.pi / 3.0, 14.7e6
    reduced = m * craft / (m + craft)
    k = reduced * (theta * rate) ** 2
    omega = theta * rate
    pulse = force / rate
    # The paper's start: spring latched at its steady compression, then a
    # half-size first pulse.  State: compression, its rate, craft velocity.
    state = [pulse / (2.0 * m * omega) / math.tan(theta / 2.0), 0.0, 0.0]
    forces, compressions = [], []
    for n in range(12):
        state[1] += (pulse / 2.0 if n == 0 else pulse) / m
        run = solve_ivp(
            lambda t, y: [y[1], -k * y[0] / reduced, k * y[0] / craft],
            (0.0, 1.0 / rate),
            state,
            rtol=1e-10,
            atol=1e-12,
            dense_output=True,
        )
        state = list(run.y[:, -1])
        if n >= 2:
            x = run.sol(np.linspace(0.0, 1.0 / rate, 2001))[0]
            forces.append(k * x)
            compressions.append(x)
    f = np.concatenate(forces)
    x = np.concatenate(compressions)
    assert f.min() > 0.0  # never leaves the spring below theta = pi
    assert f.mean() == pytest.approx(force * craft / (m + craft), rel=1e-3)
    assert f.max() / f.mean() == pytest.approx(peak_to_mean(theta), rel=1e-3)
    assert peak_to_mean(theta) == pytest.approx(1.209, abs=1e-3)
    assert x.max() - x.min() == pytest.approx(
        spring_plate_stroke(force, rate, theta) / m, rel=1e-3
    )


def test_the_parents_s45_ballpark() -> None:
    """507 t m at 2 Hz, 1 g; 272 t m for the bowl's 1.97 MN at 1 Hz."""
    theta = 2.0 * math.pi / 3.0
    assert spring_plate_stroke(14.71e6, 2.0, theta) == pytest.approx(507e3, rel=0.01)
    assert spring_plate_stroke(1.97e6, 1.0, theta) == pytest.approx(272e3, rel=0.01)
    spring = gas_spring(14.71e6, 2.0, 1200.0e3, Design())
    assert spring.plate == pytest.approx(63.4e3, rel=0.01) and spring.stroke == 8.0
    # As one 63 t assembly the departure needs 4.3 m of the same stroke.
    shared = gas_spring(1.97e6, 1.0, 600.0e3, Design(), plate=spring.plate)
    assert shared.stroke == pytest.approx(4.3, abs=0.05)
    # A survival floor holds the plate; it then swings less.
    held = gas_spring(1.97e6, 1.0, 600.0e3, Design(), floor=50.0e3)
    assert held.plate == 50.0e3 and held.stroke < 8.0


def test_the_bags_are_the_fibre_wall_and_the_helium() -> None:
    """3 PV rho / sigma at half of 2.5 GPa, 1600 kg/m^3, plus helium at 293 K."""
    pv = 1.7e8
    wall = 3.0 * peak_to_mean(2.0 * math.pi / 3.0) * pv * 1600.0 / 1.25e9
    helium = pv / (2077.1 * 293.0)
    assert bag_mass(pv, 2.0 * math.pi / 3.0) == pytest.approx(wall + helium)
    assert 1.0e3 < bag_mass(pv, 2.0 * math.pi / 3.0) < 1.2e3
    assert bag_mass(pv, 2.0 * math.pi / 3.0, 5.0e3 / 1.7e8) == pytest.approx(
        wall + helium + 5.0e3
    )


def test_the_delivery_plate_rides_the_spring() -> None:
    """S45 item 4: the cargo feels ~1.2x the mean thrust, whatever the stroke."""
    laws = StageLaws.load()
    slow = delivery_unit(75.5, Design(k_delivery=5.6), laws)
    assert slow.pulse == pytest.approx(7.355e6, rel=1e-3)
    assert slow.plate == pytest.approx(63.4e3, rel=0.01) and slow.stroke == 8.0
    assert 1.0e3 < slow.bag < 5.0e3
    assert 1.2 < slow.start_g < slow.peak_g < 2.0
    longer = delivery_unit(75.5, Design(k_delivery=5.6, spring_stroke=16.0), laws)
    assert longer.plate == pytest.approx(slow.plate / 2.0)
    assert longer.peak_g == pytest.approx(slow.peak_g, rel=0.02)
    child = delivery_unit(75.5, Design(k_delivery=5.6, slow_delivery=False), laws)
    assert child.climb == 1.0
    # 4 Hz at 3.26 g: m x stroke scales as F / nu^2, so 3.26 / 4 of the 2 Hz plate.
    assert child.plate == pytest.approx(slow.plate * 3.26 / 4.0, rel=1e-3)


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
    assert push.early_days == 10.0 and departure.early_days == 0.0
    with pytest.raises(ValueError):
        push_route(9.86, 90.0, 0.0)


@pytest.mark.slow
def test_a_shorter_gap_slows_the_growth_stream_and_widens_the_turn() -> None:
    """S39 item 8: the gap reaches every use (the paper's 3/5/10-day rows)."""
    from src.solar_dive_growth import PUSH_ALTITUDE, fly_unit, wave_speed

    departure = departure_route(9.86, 60.0)
    assert departure is not None
    streams, turns = [], []
    for days in (3.0, 5.0, 10.0):
        push = push_route(9.86, 60.0, days)
        assert push is not None
        streams.append(wave_speed(push.earth_excess, PUSH_ALTITUDE))
        turns.append(
            fly_unit(departure, push, departure, Design()).lines["methalox_dv"]
        )
    assert streams == pytest.approx([82.6, 87.1, 99.2], abs=0.1)
    assert turns == pytest.approx([0.76, 0.54, 0.33], abs=0.01)


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
    end = periapsis_speed(10.0, 400.0)
    assert 2.25 < push_climb(1.0, end) < 2.5
    assert push_climb(3.26, end) == pytest.approx(1.1)
    assert 1.4 < lob_climb_factor(push_climb(1.0, end)) < 1.6
    assert lob_climb_factor(1.1) == 1.0


def test_sheets_are_kapton_below_130_km_s_and_carbon_fabric_above() -> None:
    """S43 item 2: $100/kg fabric above 130 km/s; the 100-130 band a switch."""
    from dataclasses import replace

    from src.solar_dive_growth import ESTIMATE

    assert ESTIMATE.sheet_price(99.2) == ESTIMATE.sheet_body == 3.0
    assert ESTIMATE.sheet_price(121.8) == 3.0
    assert ESTIMATE.sheet_price(137.1) == ESTIMATE.carbon_fabric == 100.0
    band = replace(ESTIMATE, band_carbon=True)
    assert band.sheet_price(121.8) == 100.0 and band.sheet_price(99.2) == 3.0


@pytest.mark.slow
def test_one_shared_spring_leaves_nothing_in_the_parking_orbit() -> None:
    """S45 item 2: the shared assembly is the heavier leg's plate, flown on."""
    from dataclasses import replace

    from src.solar_dive_growth import fly_unit

    departure = departure_route(9.86, 60.0)
    push = push_route(9.86, 60.0)
    assert departure is not None and push is not None
    shared = fly_unit(departure, push, departure, Design()).lines
    two = fly_unit(
        departure, push, departure, replace(Design(), shared_spring=False)
    ).lines
    assert shared["push_plate"] == shared["plate_disposal"] == 0.0
    assert shared["bowl"] == pytest.approx(
        63.4e3 + shared["bowl_bag"] + 1.0e3, rel=0.01
    )
    assert 3.0 < shared["bowl_stroke"] < 8.0  # the departure needs less stroke
    assert two["push_plate"] > 63.4e3 and two["bowl_stroke"] == pytest.approx(8.0)
    assert two["bowl_floor"] < two["bowl"] < shared["bowl"]
