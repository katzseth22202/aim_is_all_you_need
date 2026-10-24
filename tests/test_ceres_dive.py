"""Tests for the solar dive grown at Ceres (S56, ADR 0083)."""

import math

import pytest

from src.ceres_dive import (
    AU,
    HEADLINE,
    PAPER_DEPARTURE,
    PAPER_DEPARTURE_VE,
    POLYETHYLENE_BOND,
    CeresDesign,
    best_push,
    ceres_ve,
    closures,
    earth_excess_for,
    law_one_ve,
    pulse_disks,
    rings,
    steady_loop,
    sun_route,
    tug_dv,
    water_chemistry,
)
from src.solar_dive_growth import Design, unfloored


def test_law_one_reproduces_the_parents_departure_check():
    w, theta, k = PAPER_DEPARTURE
    assert law_one_ve(k, w, theta, POLYETHYLENE_BOND) == pytest.approx(8.07, abs=0.01)
    assert PAPER_DEPARTURE_VE == pytest.approx(8.0, abs=0.01)


def test_the_hole_law_pays_waters_toll_and_falls_as_it_rises():
    assert water_chemistry(15.0, 95.0) < water_chemistry(15.0, 95.0, recovered=0.31)
    assert ceres_ve(15.0, 95.0, 80.0) < ceres_ve(15.0, 95.0, 80.0, recovered=0.31)


@pytest.mark.parametrize(
    "depth, excess, retrograde, dv, miss",
    [
        (9.86, 60.0, False, 14.67, -115.0),
        (9.86, 90.0, True, 21.14, -28.0),
        (4.0, 90.0, False, 15.84, -107.0),
        (4.0, 90.0, True, 19.97, -42.0),
    ],
)
def test_direct_dives_reproduce_the_parents_misses(depth, excess, retrograde, dv, miss):
    route = sun_route(depth, excess, retrograde)
    assert route.departure_dv == pytest.approx(dv, abs=0.02)
    assert route.miss_deg == pytest.approx(miss, abs=1.0)


def test_the_parents_retrograde_inbound_closure():
    found = closures(9.86, 90.0, True, revolutions=(0,))[0]
    assert found.departure_dv == pytest.approx(23.3, abs=0.05)
    assert found.aphelion / AU == pytest.approx(3.28, abs=0.01)
    assert found.inbound


def test_a_three_depot_ring_closes_almost_free_at_4_rsun_x150():
    ring = rings(4.0, 150.0, revolutions=(0,))[0]
    assert (ring.depots, ring.step) == (3, 1)
    assert ring.route.departure_dv == pytest.approx(15.84, abs=0.05)
    assert ring.route.miss_deg == pytest.approx(-120.0, abs=0.1)


def test_earth_stream_excess_holds_the_papers_speeds():
    assert earth_excess_for(4.0, 137.0) == pytest.approx(128.0, abs=0.5)
    assert earth_excess_for(9.86, 99.0) == pytest.approx(87.0, abs=0.5)


def test_depot_drift_matches_the_parents_formula():
    assert tug_dv(100.0, 10.0) == pytest.approx(1.52, abs=0.01)
    assert tug_dv(100.0, 20.0) == pytest.approx(0.76, abs=0.01)


def test_the_fewest_pellet_push_beats_its_neighbours():
    push = best_push(15.84, 124.0, 82.7)
    for k in (push.k * 0.8, push.k * 1.2):
        other = best_push(15.84, 124.0, 82.7, k=k)
        assert other.stream >= push.stream - 1e-9


@pytest.mark.slow
def test_the_headline_reproduces_and_the_delivery_lob_is_its_floor():
    loop = steady_loop(HEADLINE[4.0])
    assert pulse_disks(HEADLINE[4.0]) >= 1.0  # the solar dive's floor (ADR 0085)
    assert loop.price == pytest.approx(1.488, abs=0.002)
    assert loop.lines["delivery_lob"] == pytest.approx(1.006, abs=0.005)
    adr_0083 = CeresDesign(
        **{
            **HEADLINE[4.0].__dict__,
            "k_sun": 29.907,
            "arc_hours": 3.14,
            "k_delivery": 5.62,
            "spring": unfloored(Design()),
        }
    )
    assert steady_loop(adr_0083).price == pytest.approx(1.463, abs=0.002)
    assert pulse_disks(adr_0083) == pytest.approx(0.60, abs=0.01)
    assert loop.growth > 1.0
    child = steady_loop(
        CeresDesign(**{**HEADLINE[4.0].__dict__, "child_delivery": True})
    )
    assert child.price < loop.price - 0.3
    assert math.isfinite(child.c_earth)
