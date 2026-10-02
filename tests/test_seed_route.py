"""Tests for src/seed_route.py: cheaper seeds on gravity assists and SEP (ADR 0039).

The SEP stage and the dollar test are pure arithmetic and run fast. The route
searches need pygmo and pykep and are slow.
"""

import math

import pytest

from src import conic_kernel
from src.seed_route import (
    RETURN_FLOOR,
    SEED_WINDOW_OPENS,
    Propulsion,
    SepStage,
    best_route,
    discounted_seed_per_dollar,
    fly_route,
    planet_longitudes,
    seed_route_worth,
    ship_cost,
    shortfall,
)

AU_KM = 1.495978707e8


def test_a_route_pays_when_its_discount_outweighs_its_delay() -> None:
    # k * (1 + r)^-dt: 2.9 times cheaper, 2.3 years later, at 30%.
    assert seed_route_worth(2.9, 2.3, 0.30) == pytest.approx(2.9 * 1.3**-2.3)
    assert seed_route_worth(2.0, 3.0, 0.30) < 1.0 < seed_route_worth(2.0, 2.0, 0.30)


YEAR_S = 365.25 * 86400.0


def test_a_kilowatt_per_tonne_delivers_about_1_6_km_s_a_year_at_1_au() -> None:
    # 2 eta P / (m v_e): 2 x 0.5 x 1 W/kg / 19 613 m/s = 5.10e-5 m/s^2.
    stage = SepStage(specific_power=1.0)
    assert stage.acceleration_1au() == pytest.approx(5.099e-5, rel=1e-3)
    assert stage.capacity([1.0], YEAR_S) == pytest.approx(1.609, rel=1e-3)


def test_thrust_falls_as_the_inverse_square_of_sun_distance() -> None:
    stage = SepStage(specific_power=1.0)
    at_one = stage.capacity([1.0, 1.0], YEAR_S)
    assert stage.capacity([2.0, 2.0], YEAR_S) == pytest.approx(at_one / 4.0)
    # Averaged over the arc's samples, not its endpoints.
    assert stage.capacity([1.0, 2.0, 5.2], YEAR_S) == pytest.approx(
        at_one * (1.0 + 0.25 + 5.2**-2) / 3.0
    )


def test_the_stage_and_its_argon_displace_puffsats() -> None:
    # 1 W/kg at 15 kg/kW is 1.5% of the stack; 3 km/s at 19.6 km/s burns
    # 1 - exp(-3/19.613) of it as argon, with 0.15 kg of tank per kg.
    stage = SepStage(specific_power=1.0, specific_mass=15.0)
    split = stage.split(stack=600.0e3, burn=3.0)
    argon = 600.0e3 * (1.0 - math.exp(-3.0 / 19.6133))
    assert split.hardware == pytest.approx(9.0e3)
    assert split.argon == pytest.approx(argon, rel=1e-4)
    assert split.tanks == pytest.approx(0.15 * argon, rel=1e-4)
    assert split.puffsats == pytest.approx(600.0e3 - 9.0e3 - 1.15 * argon, rel=1e-4)


def test_the_array_is_bought_by_the_watt() -> None:
    stage = SepStage(specific_power=1.2, price_per_watt=200.0)
    # 1.2 W/kg on a 580 t stack is 696 kW at 1 AU.
    assert stage.price(stack=580.0e3) == pytest.approx(696.0e3 * 200.0)


def test_no_burn_and_no_array_leaves_the_whole_stack_as_puffsats() -> None:
    split = SepStage(specific_power=0.0).split(stack=72.0e3, burn=0.0)
    assert split.puffsats == pytest.approx(72.0e3)


def test_each_node_must_be_paid_by_thrusting_before_it() -> None:
    # Capacities 1.0 and 2.0 km/s on the two legs before nodes 1 and 2.
    capacities = [1.0, 2.0]
    assert shortfall([0.8, 2.0], capacities) == pytest.approx(0.0)
    # 1.5 at the first node is more than its leg can deliver, even though the
    # total (1.5 + 0.5 = 2.0) fits within 3.0.
    assert shortfall([1.5, 0.5], capacities) == pytest.approx(0.5)
    assert shortfall([0.5, 3.0], capacities) == pytest.approx(0.5)


@pytest.mark.slow
def test_flying_the_direct_route_reproduces_adr_0008s_phased_optimum() -> None:
    from src.nozzle_analysis import (
        PHASED_BEND_SIGN,
        PHASED_LEG_TIME,
        PHASED_LOG_PERIJOVE,
        phased_geometry,
    )

    reference = phased_geometry()
    flight = fly_route(
        "EJ",
        longitudes=(0.0, reference.jupiter_lon0),
        leg_years=(float(PHASED_LEG_TIME.to_value("yr")),),
        log_perijove=PHASED_LOG_PERIJOVE,
        bend_sign=PHASED_BEND_SIGN,
    )
    assert flight is not None
    assert flight.departure_burn == pytest.approx(reference.departure_burn, rel=1e-9)
    assert flight.node_burns == ()
    assert flight.collision_speed == pytest.approx(
        reference.reference.collision_speed, rel=1e-9
    )
    assert flight.mismatch == pytest.approx(0.0, abs=1e-8)
    assert len(flight.leg_distances_au) == 1
    # The Earth-Jupiter arc starts at 1 AU and ends at 5.2.
    assert flight.leg_distances_au[0][0] == pytest.approx(1.0, abs=1e-3)
    assert flight.leg_distances_au[0][-1] == pytest.approx(5.2, abs=0.05)


def test_earth_sits_opposite_the_sun_at_the_march_equinox() -> None:
    # 2026-03-20 14:46 UTC: the Sun crosses the equinox *of date*, so Earth's
    # heliocentric longitude is 180 degrees in that frame. The longitudes are
    # in the fixed J2000 ecliptic, which the equinox has precessed past by
    # 50.29 arcsec/yr x 26.2 yr = 0.366 degrees. A day later Earth has moved
    # ~0.9856 degrees.
    equinox = 2461120.115
    precession = 50.29 / 3600.0 * (equinox - 2451545.0) / 365.25
    (earth,) = planet_longitudes("E", equinox)
    (later,) = planet_longitudes("E", equinox + 1.0)
    assert math.degrees(earth) % 360.0 == pytest.approx(180.0 - precession, abs=0.02)
    step = conic_kernel.wrap_pi(later - earth)
    assert math.degrees(step) == pytest.approx(0.9856, abs=0.03)


@pytest.mark.slow
def test_the_best_direct_seed_closes_on_earth_at_the_return_floor() -> None:
    pytest.importorskip("pygmo")
    best = best_route("EJ", propulsion=Propulsion.METHALOX, seed=11)
    assert best is not None
    flight = best.flight
    assert abs(flight.mismatch) < 1e-8
    assert flight.collision_speed >= RETURN_FLOOR - 1e-6
    assert flight.node_burns == ()
    assert best.launch_jd >= SEED_WINDOW_OPENS
    assert best.puffsats_per_ship > 0.0


def test_routes_are_scored_on_seed_per_dollar_discounted_to_its_return() -> None:
    # 200 t back after 5 years on a $670M ship plus a $100M array, against
    # 72 t back after 3.3 years on the ship alone, at 30%.
    slow = discounted_seed_per_dollar(200.0e3, 5.0, 670.0e6 + 100.0e6, 0.30)
    fast = discounted_seed_per_dollar(72.0e3, 3.3, 670.0e6, 0.30)
    assert slow == pytest.approx(200.0e3 * 1.3**-5.0 / 770.0e6)
    assert slow / fast == pytest.approx(
        seed_route_worth((200.0e3 / 770.0e6) / (72.0e3 / 670.0e6), 1.7, 0.30)
    )


def test_a_seed_ship_costs_its_thirteen_launches_and_its_hull() -> None:
    assert ship_cost(flight_price=50.0e6, hull=20.0e6) == pytest.approx(670.0e6)


@pytest.mark.slow
def test_thruster_nodes_are_charged_at_infinity_not_at_periapsis() -> None:
    # A thruster pays its node in deep space: an unpowered flyby plus the
    # excess-velocity change bought far from the planet, with no Oberth
    # leverage (_flyby_mismatch_burn). Methalox burns at periapsis
    # (_powered_node_burn). Neither always costs less, so pin the wiring.
    from src.retrograde_return_legs import _phased_ladder_burn
    from src.seed_route import _bodies, route_params

    params = route_params()
    longitudes = planet_longitudes("EVEJ", SEED_WINDOW_OPENS)
    checked = 0
    # Feasible geometries are rare; (0.6, 0.9, 1.5) yr at 10^1.5 perijove floors
    # on the -1 side closes from the window's opening date.
    for legs in ((0.6, 0.9, 1.5),):
        for log_perijove, sign in ((1.5, -1.0),):
            for powered in (True, False):
                flight = fly_route(
                    "EVEJ", longitudes, legs, log_perijove, sign, powered_nodes=powered
                )
                if flight is None:
                    continue
                ladder = _phased_ladder_burn(
                    0.0,
                    [y * 365.25 * 86400.0 for y in legs],
                    _bodies("EVEJ", params),
                    list(longitudes),
                    params,
                    powered_nodes=powered,
                )
                assert ladder is not None
                assert flight.node_burns == pytest.approx(tuple(ladder.node_burns))
                checked += 1
    assert checked == 2


def test_island_workers_are_capped_by_free_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import seed_route

    monkeypatch.setattr(seed_route.os, "cpu_count", lambda: 8)
    gib = 2**30
    monkeypatch.setattr(seed_route, "_available_bytes", lambda: 3 * gib)
    assert seed_route._island_workers(8) == 2 * gib // seed_route._ISLAND_BYTES
    monkeypatch.setattr(seed_route, "_available_bytes", lambda: 0)
    assert seed_route._island_workers(8) == 1
    monkeypatch.setattr(seed_route, "_available_bytes", lambda: None)
    assert seed_route._island_workers(8) == 8
    assert seed_route._island_workers(3) == 3
