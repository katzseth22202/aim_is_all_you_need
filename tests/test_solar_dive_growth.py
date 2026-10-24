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
    assert sheet_material(137.1) == "Kapton" and sheet_material(141.0) == "graphite"
    assert sheet_material(132.0, limit=130.0) == "graphite"


def test_the_bowl_law_reproduces_the_simulations_hole_ve() -> None:
    """departure_hole_law.csv's visible routes, within the k-interpolation."""
    laws = StageLaws.load()
    for k, v in ((5.0, 13.601), (10.0, 10.716), (20.0, 7.658), (30.0, 5.774)):
        assert laws.hole_ve(k, 84.0, 129.9) == pytest.approx(v, rel=0.006)
    for k, v in ((5.0, 20.310), (20.0, 11.272)):
        assert laws.hole_ve(k, 107.9, 117.0, far_focus=False) == pytest.approx(
            v, rel=0.006
        )
    condensed = StageLaws.load(bonds="bonds returned")
    assert condensed.hole_ve(30.0, 84.0, 129.9) == pytest.approx(7.907, rel=0.006)


def test_below_the_rim_line_the_bowl_is_the_simulations_2p5_f_run() -> None:
    """S47 item 3: band_far_bowl.csv's hole_ve_25 at 117 deg; 129.9 deg stays 1.5 F."""
    laws = StageLaws.load()
    for k, v in ((10.0, 13.954), (20.0, 10.318), (30.0, 8.299)):
        assert laws.hole_ve(k, 107.9, 117.0) == pytest.approx(v, rel=0.006)
    assert laws.far_focus(20.0) == pytest.approx(2.619036 / 2.861043, rel=1e-6)
    assert laws.far_focus(5.0) == laws.far_focus(10.0)  # held flat outside 10-30
    assert laws.hole_ve(20.0, 84.0, 129.9) == laws.hole_ve(
        20.0, 84.0, 129.9, far_focus=False
    )


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


def test_the_bags_are_the_fibre_wall_the_gas_and_the_bladder() -> None:
    """3 PV rho / sigma at half of 2.5 GPa, 1600 kg/m^3, gas at 293 K, bladder."""
    theta = 2.0 * math.pi / 3.0
    pv = 1.7e8
    wall = 3.0 * peak_to_mean(theta) * pv * 1600.0 / 1.25e9
    helium = pv / (2077.1 * 293.0)
    nitrogen = pv / (296.8 * 293.0)
    bladder = pv * 1.3e3 / 259.6e6
    assert bag_mass(pv, theta, gas="helium", bladder=0.0) == pytest.approx(
        wall + helium
    )
    assert bag_mass(pv, theta) == pytest.approx(wall + nitrogen + bladder)
    assert bag_mass(pv, theta, 5.0e3 / 1.7e8) == pytest.approx(
        wall + nitrogen + bladder + 5.0e3
    )


def test_the_nitrogen_bags_are_the_simulations_result_17() -> None:
    """63 t on 8 m: the sim's 2.9 t of nitrogen and 1.3 t of bladder, 5-6 t in all."""
    spring = gas_spring(1500.0e3 * 9.80665, 2.0, 1300.0e3, Design())
    assert spring.plate == pytest.approx(63.4e3, rel=0.01)
    assert spring.pv / (296.8 * 293.0) == pytest.approx(2.9e3, rel=0.1)
    assert 5.0e3 < spring.bag < 6.0e3
    helium = gas_spring(
        1500.0e3 * 9.80665, 2.0, 1300.0e3, Design(bag_gas="helium", bladder=0.0)
    )
    assert 1.5e3 < helium.bag < 2.5e3  # ADR 0076's 1.9 t


def test_the_delivery_plate_rides_the_spring() -> None:
    """S45 item 4: the cargo feels ~1.2x the mean thrust, whatever the stroke."""
    laws = StageLaws.load()
    slow = delivery_unit(75.5, Design(k_delivery=5.6), laws)
    assert slow.pulse == pytest.approx(7.355e6, rel=1e-3)
    assert slow.plate == pytest.approx(63.4e3, rel=0.01) and slow.stroke == 8.0
    assert 5.0e3 < slow.bag < 6.0e3  # nitrogen and its bladder
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


def test_sheets_are_kapton_below_140_km_s_and_carbon_fabric_above() -> None:
    """S47 item 1: floor-wide Kapton through 137.1 km/s (Result 18), fabric above."""
    from src.solar_dive_growth import ESTIMATE

    assert ESTIMATE.sheet_price(99.2) == ESTIMATE.sheet_body == 3.0
    assert ESTIMATE.sheet_price(137.1) == 3.0
    assert ESTIMATE.sheet_price(207.8) == ESTIMATE.carbon_fabric == 100.0
    assert ESTIMATE.sheet_price(137.1, limit=130.0) == 100.0


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


def test_a_reused_delivery_plate_buys_a_tenth_of_a_plate_a_flight() -> None:
    """S53 item 2: each use buys 1 / uses of a plate, on the shared learning count."""
    from dataclasses import replace

    from src.solar_dive_growth import ESTIMATE, DeliveryUnit, delivery_dollars

    delivery = DeliveryUnit(
        sheets=8.0e4,
        cargo=1.0e6,
        charge=4.0e5,
        plate=6.3e4,
        bag=5.0e3,
        pulse=7.4e6,
        stroke=8.0,
        peak_g=1.7,
        start_g=1.2,
        climb=1.49,
    )
    once, plates = {"push": 50.0}, {"push": 50.0}
    single = delivery_dollars(delivery, 3.0, ESTIMATE, once, 25.0)
    reused = replace(ESTIMATE, delivery_plate_uses=10.0)
    tenth = delivery_dollars(delivery, 3.0, reused, plates, 25.0)
    assert once["push"] == 53.0 and plates["push"] == pytest.approx(50.3)
    assert tenth["delivery_lob"] == single["delivery_lob"]
    assert tenth["delivery_charge"] == single["delivery_charge"]
    # Fewer plates built means less learning, so a use costs a little over a tenth.
    assert 0.1 < tenth["delivery_plate"] / single["delivery_plate"] < 0.11


@pytest.mark.slow
def test_the_booster_burns_about_half_a_dollar_of_methalox_per_kg_lofted() -> None:
    """S53 item 3: 2.46 kg of propellant a kg on ADR 0037's lob, 4.3 at 2.43 km/s."""
    from src.solar_dive_growth import lob_floor, lob_propellant

    assert lob_propellant(0.75) == pytest.approx(2.46, abs=0.01)
    assert lob_propellant(1.1) == pytest.approx(2.69, abs=0.01)
    assert lob_propellant(2.43) == pytest.approx(4.31, abs=0.01)
    assert lob_floor() == pytest.approx(0.49, abs=0.01)


@pytest.mark.slow
def test_the_mature_book_in_the_ledger_matches_the_parents_scratch() -> None:
    """S53 item 1: the paper's book reproduces; reuse in the ledger is the parent's /10."""
    from dataclasses import replace

    from src.solar_dive_growth import ESTIMATE, MATURE, UNFLOORED_STARTS, unfloored

    expected = {(9.86, 60.0): (94.15, 85.29), (4.0, 90.0): (73.52, 67.81)}
    for key, (paper, scratch) in expected.items():
        departure = departure_route(*key)
        push = push_route(*key)
        assert departure is not None and push is not None
        held = unfloored(UNFLOORED_STARTS[key])  # ADR 0082 predates the floor
        estimate = value_design(departure, push, held, ESTIMATE)
        assert estimate.steady_cost == pytest.approx(paper, abs=0.01)
        mature = value_design(departure, push, held, MATURE)
        assert mature.steady_cost == pytest.approx(scratch, abs=0.1)
        # Shared, the push plate is the bowl: there is no push plate to reuse.
        assert mature.steady_lines["push_plate"] == 0.0
        once = replace(MATURE, push_plate_uses=1.0, delivery_plate_uses=1.0)
        two = replace(held, shared_spring=False)
        flown_once = value_design(departure, push, two, once).steady_lines
        flown_ten = value_design(departure, push, two, MATURE).steady_lines
        # Over a tenth: both plates share one learning count, now ten times slower.
        assert 0.1 < flown_ten["push_plate"] / flown_once["push_plate"] < 0.15
        # A delivery lofts 1500 t for its cargo; the growth lobs come on top.
        lofted = mature.steady_lofted
        own = sum(v for k, v in lofted.items() if k != "growth_lob")
        assert lofted["cargo"] == 1.0 and 1.3 < own < 1.5
        assert 0.2 < lofted["growth_lob"] < 0.6


@pytest.mark.slow
def test_the_apoapsis_stage_and_reserve_are_dropped_mass() -> None:
    """S27, off by default: one Raptor and its tanks, and a reserve before Jupiter."""
    from dataclasses import replace

    from src.solar_dive_growth import APOAPSIS_ENGINE, fly_unit

    departure = departure_route(4.0, 90.0)
    push = push_route(4.0, 90.0)
    assert departure is not None and push is not None
    plain = fly_unit(departure, push, departure, Design())
    staged = fly_unit(
        departure,
        push,
        departure,
        replace(Design(), apoapsis_stage=True, deep_space_reserve=0.04),
    )
    assert plain.lines["apoapsis_engine"] == plain.lines["deep_space_reserve"] == 0.0
    assert staged.lines["apoapsis_engine"] == APOAPSIS_ENGINE
    assert staged.lines["apoapsis_tanks"] == pytest.approx(
        0.018 * staged.lines["methalox"]
    )
    assert staged.lines["deep_space_reserve"] == pytest.approx(
        plain.lines["departing"] * 0.0111, rel=0.05
    )
    assert 0.98 < staged.returned / plain.returned < 0.99


def test_snow_foam_prices_the_earth_charges_and_not_the_suns() -> None:
    """The author's snow foam replaces polyethylene only where snow survives."""
    from dataclasses import replace

    from src.solar_dive_growth import ESTIMATE, SNOW_FOAM

    assert ESTIMATE.earth_charge_price == ESTIMATE.polyethylene
    foam = replace(ESTIMATE, earth_charge=SNOW_FOAM)
    assert foam.earth_charge_price == pytest.approx(0.0619, abs=1e-4)
    assert foam.polyethylene == ESTIMATE.polyethylene


def test_the_floor_arc_solves_a_pulse_that_falls_as_the_arc() -> None:
    """S59: the secant finds the arc at which the lightest pulse is the floor."""
    from src.solar_dive_growth import arc_on_floor

    def pulse(arc: float) -> float:  # 0.2 kg at 4.97 h, a little arc loss
        return 0.2 * 4.97 / arc * (1.0 + 0.01 * (arc / 4.97) ** 2)

    arc = arc_on_floor(pulse, 1.0, 4.97)
    assert pulse(arc) == pytest.approx(1.0, rel=1e-6)
    assert 0.98 < arc < 1.0


@pytest.mark.slow
def test_the_disk_floor_holds_every_ship_and_unfloored_is_adr_0077() -> None:
    """S59 items 1 and 4: the floor binds the departure-wave ship at 9.86 R_sun."""
    from dataclasses import replace

    from src.solar_dive_growth import (
        HEADLINE_STARTS,
        UNFLOORED_STARTS,
        fly_unit,
        optimize_design,
        pulse_disks,
        unfloored,
    )

    key = (9.86, 60.0)
    departure = departure_route(*key)
    push = push_route(*key)
    assert departure is not None and push is not None
    free = optimize_design(departure, push, unfloored(UNFLOORED_STARTS[key]))
    assert free.dear_break_even == pytest.approx(145.66, abs=0.05)
    assert pulse_disks(departure, push, free.design) == pytest.approx(0.200, abs=1e-3)
    floored = optimize_design(departure, push, HEADLINE_STARTS[key])
    d = floored.design
    assert pulse_disks(departure, push, d) == pytest.approx(1.0, abs=1e-4)
    assert d.arc_hours == pytest.approx(0.99, abs=0.01)
    assert floored.dear_break_even == pytest.approx(158.2, abs=0.2)
    unit = fly_unit(departure, push, departure, d).lines
    pushed = fly_unit(push, push, departure, d).lines
    assert unit["pulse_disk"] < pushed["pulse_disk"]  # the departure ship binds
    # 25 g of gas a pulse, carried and turned by the carrier, never struck.
    pulses = 2.0 * d.arc_hours * 3600.0
    assert unit["disk_gas"] == pytest.approx(0.025 * pulses)
    no_gas = fly_unit(departure, push, departure, replace(d, disk_gas=0.0)).lines
    assert unit["carrier_dry"] == pytest.approx(
        no_gas["carrier_dry"] * (1 + unit["disk_gas"] / unit["disks"]), rel=0.01
    )
    # The Sun plate at a 1 kg pulse sits at its 1 kg, 400 km/s rating.
    assert unit["sun_survival"] == pytest.approx(
        30.0 * 533.041 * (departure.sun_closing / 400.0) ** (4.0 / 3.0), rel=1e-3
    )


@pytest.mark.slow
def test_a_stacked_sun_ship_shares_one_plate_and_collar() -> None:
    """S59 item 5: five units' pulses on one plate, a fifth of it each."""
    from dataclasses import replace

    from src.solar_dive_growth import UNFLOORED_STARTS, fly_unit

    key = (9.86, 60.0)
    departure = departure_route(*key)
    push = push_route(*key)
    assert departure is not None and push is not None
    one = UNFLOORED_STARTS[key]
    five = replace(one, sun_stack=5)
    single = fly_unit(departure, push, departure, one).lines
    stacked = fly_unit(departure, push, departure, five).lines
    assert stacked["pulse_disk"] == pytest.approx(5.0 * single["pulse_disk"], rel=0.01)
    assert stacked["bore_collar"] == pytest.approx(200.0)
    # The survival floor goes as the pulse to the 2/3: each unit carries less.
    assert stacked["sun_survival"] == pytest.approx(
        single["sun_survival"] * 5.0 ** (2.0 / 3.0), rel=0.02
    )
    assert stacked["sun_plate"] < 0.62 * single["sun_plate"]
    assert stacked["disk_gas"] == pytest.approx(single["disk_gas"] / 5.0)
