"""Tests for src/seed_loop.py: the Lucy-style Earth loop for the seed (ADR 0059)."""

import numpy as np
import pytest
from astropy import units as u

from src.retrograde_return_legs import _body_state
from src.seed_cost import STRIPPED_SHIPS, chains, seed_payload
from src.seed_loop import (
    CHAIN_FLOOR,
    STORED_ISP,
    VE_STORED,
    _LoopProblem,
    loop_flyby,
    seed_after_burns,
    seed_carried,
)


def test_the_stored_burns_fly_at_370_s() -> None:
    assert STORED_ISP == 370.0
    assert VE_STORED == pytest.approx(3.628, abs=1e-3)


def test_the_burns_and_their_stage_come_out_of_the_stack() -> None:
    """1 km/s at 370 s spends 24.0% of the stack, plus 8% of that in stage."""
    spent = 1.0 - np.exp(-1.0 / VE_STORED)
    assert seed_after_burns(100.0, [0.4, 0.6]) == pytest.approx(
        100.0 * (1.0 - 1.08 * spent)
    )
    assert seed_after_burns(100.0, []) == pytest.approx(100.0)


def _problem() -> _LoopProblem:
    return _LoopProblem("2 yr", 1.0, STRIPPED_SHIPS[0])


def test_a_prograde_launch_adds_its_excess_to_earths_speed() -> None:
    """alpha = 0 points the excess along Earth's velocity."""
    udp = _problem()
    flown = loop_flyby(udp.longitudes(0.0), 5.0, 0.0, 0.5, 2.0, 2.0, udp.params)
    assert flown is not None
    priced, _ = flown
    r0, v_launch, _ = priced.legs[0]
    earth = udp.params.bodies[udp.params.earth_index]
    _, v_earth = _body_state(earth.orbit_radius, udp.longitudes(0.0)[0], earth.v_circ)
    assert np.linalg.norm(v_launch) == pytest.approx(np.linalg.norm(v_earth) + 5.0)
    fl = udp.params.flyby
    assert priced.departure_burn == pytest.approx(
        np.sqrt(fl.v_esc_leo**2 + 25.0) - fl.v_depart_from
    )


def test_the_loop_has_two_nodes_and_the_flyby_points_at_jupiter() -> None:
    """The deep-space burn, then the flyby that must deliver the Jupiter arc."""
    udp = _problem()
    flown = loop_flyby(udp.longitudes(0.0), 5.0, 0.0, 0.5, 2.0, 2.0, udp.params)
    assert flown is not None
    priced, flyby = flown
    assert len(priced.node_burns) == 2
    assert priced.node_total == pytest.approx(sum(priced.node_burns))
    assert flyby.periapsis_altitude >= 300.0 - 1e-6
    # The arc to Jupiter leaves Earth with the excess the flyby delivers.
    r1, v_depart, cruise = priced.legs[2]
    assert np.linalg.norm(r1) == pytest.approx(
        udp.params.bodies[udp.params.earth_index].orbit_radius
    )
    assert cruise == pytest.approx(2.0 * 365.25 * 86400.0)
    assert flyby.excess_out > 0.0


def test_a_ship_that_burns_nothing_more_sends_the_direct_seed() -> None:
    """With no later burns the carried model is ``seed_payload``'s."""
    ship = STRIPPED_SHIPS[0]
    flown = seed_payload(ship, 11.99276933 * u.km / u.s)
    assert seed_carried(ship, flown.mass_ratio, []) == pytest.approx(
        flown.payload.to_value(u.kg)
    )


def test_carrying_the_ship_costs_more_than_staging_it() -> None:
    """The 40 t hull riding through a 2 km/s burn: less seed than a light stage."""
    ship = STRIPPED_SHIPS[0]
    flown = seed_payload(ship, 5.0 * u.km / u.s)
    staged = seed_after_burns(flown.payload.to_value(u.kg), [2.0])
    assert seed_carried(ship, flown.mass_ratio, [2.0]) < staged


@pytest.mark.slow
def test_the_floor_is_the_chains_first_growth_wave() -> None:
    flown, _ = chains()
    assert flown[0].growth_wave_v_b == pytest.approx(CHAIN_FLOOR, abs=0.005)
