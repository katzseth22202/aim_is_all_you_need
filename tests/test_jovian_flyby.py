"""Tests for src/jovian_flyby.py."""

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Earth, Jupiter

from src.astro_constants import (
    EARTH_A,
    JUPITER_FLYBY_MAX_TOF,
    LEO_ALTITUDE,
    LOW_JUPITER_ALTITUDE,
)
from src.jovian_flyby import (
    jupiter_flyby_vb_trade_curve,
    puffsat_cycle_growth,
    puffsat_cycle_periapsis_speed,
)
from src.orbit_utils import escape_velocity, speed_with_escape_energy
from src.propulsion import payload_mass_ratio, retrograde_jovian_hohmann_transfer
from src.retrograde_return_legs import _powered_flyby_leg
from src.scenario_catalog import (
    lunar_transfer_periapsis_speed,
    parker_injection_burns,
    parker_rows_rescored_at,
)
from tests.test_helpers import is_nearly_equal


def test_powered_jovian_flyby_return_optimum(flyby_optimum, flyby_params):
    # Constraints hold at the optimum.
    assert flyby_optimum.total_time <= JUPITER_FLYBY_MAX_TOF * 1.0001
    assert (
        flyby_optimum.flyby_periapsis_radius
        >= (Jupiter.R + LOW_JUPITER_ALTITUDE) * 0.9999
    )
    assert flyby_optimum.return_perihelion <= EARTH_A * 1.0001
    # The achieved v_b lands strictly between the barely-retrograde plunge
    # (~49.5 km/s) and the retrograde Hohmann the catalog rows assume
    # (~69.27 km/s) -- the interior optimum of ADR 0002.
    assert 45.0 * u.km / u.s < flyby_optimum.collision_speed
    assert flyby_optimum.collision_speed < retrograde_jovian_hohmann_transfer()
    # End-to-end objective is internally consistent and beats a hand-checked
    # feasible powered trajectory (v_inf=12, tangential, 2x floor, dv2=2:
    # end-to-end ~1.33).
    assert flyby_optimum.end_to_end_mass_ratio == pytest.approx(
        flyby_optimum.delivered_fraction * flyby_optimum.payload_puffsat_mass_ratio,
        rel=1e-9,
    )
    hand_case = _powered_flyby_leg(
        12.0, 0.0, 2.0 * flyby_params.periapsis_floor, 2.0, False, 1.0, flyby_params
    )
    assert hand_case is not None
    assert flyby_optimum.end_to_end_mass_ratio > hand_case.end_to_end
    assert flyby_optimum.end_to_end_mass_ratio > 1.5
    # Mass ratio is scored against the sec:jupiter_only_growth push.
    assert flyby_optimum.payload_puffsat_mass_ratio == pytest.approx(
        payload_mass_ratio(
            v_rf=lunar_transfer_periapsis_speed(), v_b=flyby_optimum.collision_speed
        ),
        rel=1e-9,
    )
    # v_b uses the retrograde_jovian_hohmann_transfer convention: the 1 AU
    # closing speed folded through Earth's well to the surface.
    assert is_nearly_equal(
        flyby_optimum.collision_speed,
        speed_with_escape_energy(flyby_optimum.closing_speed_1au, Earth),
        percent=0.001,
    )
    # Departure burn is the Oberth increment above escape at 200 km.
    v_esc_leo = flyby_params.v_esc_leo * u.km / u.s
    assert is_nearly_equal(
        flyby_optimum.departure_burn,
        np.sqrt(flyby_optimum.v_infinity_earth**2 + v_esc_leo**2) - v_esc_leo,
        percent=0.001,
    )


def test_parker_rows_rescoreable_at_flyby_v_b(flyby_optimum):
    # The grill decision: the two Parker rows are reported at the achieved v_b,
    # not optimized for. Both injection burns must sit below the achieved v_b
    # so their mass ratios are defined there.
    prograde_burn, retrograde_burn = parker_injection_burns()
    assert prograde_burn < flyby_optimum.collision_speed
    assert retrograde_burn < flyby_optimum.collision_speed
    prograde_ratio, retrograde_ratio = parker_rows_rescored_at(
        flyby_optimum.collision_speed
    )
    assert prograde_ratio > 0
    assert retrograde_ratio > 0
    # Matches calling payload_mass_ratio directly against the same burns.
    assert prograde_ratio == pytest.approx(
        payload_mass_ratio(v_rf=prograde_burn, v_b=flyby_optimum.collision_speed)
    )
    assert retrograde_ratio == pytest.approx(
        payload_mass_ratio(v_rf=retrograde_burn, v_b=flyby_optimum.collision_speed)
    )


@pytest.mark.slow
def test_jupiter_flyby_vb_trade_curve():
    # The ADR 0002 trade curve: min total burn is nondecreasing in the v_b
    # floor, every feasible point meets its floor, and the mid-range targets
    # spanning plunge-to-Hohmann are all reachable within the 7 yr cap.
    points = jupiter_flyby_vb_trade_curve()
    assert len(points) == 6
    kms = u.km / u.s
    feasible = [p for p in points if p.feasible]
    # 50-65 km/s must all be feasible (45 and 70 may sit at the edges).
    for point in points:
        if 50.0 * kms <= point.target_collision_speed <= 65.0 * kms:
            assert point.feasible
    previous_burn = -np.inf * kms
    for point in feasible:
        # Floors are met (up to optimizer tolerance).
        assert point.achieved_collision_speed >= point.target_collision_speed * 0.9999
        assert point.total_time <= JUPITER_FLYBY_MAX_TOF * 1.0001
        assert is_nearly_equal(
            point.total_burn, point.departure_burn + point.flyby_burn, percent=0.001
        )
        # Monotone within a small optimizer tolerance.
        assert point.total_burn >= previous_burn - 0.1 * kms
        previous_burn = point.total_burn


def test_puffsat_cycle_periapsis_speed_is_just_under_escape() -> None:
    # The collision must leave the mass BOUND -- push it to escape and the next
    # cycle's payload drifts away instead of falling back to the burn point.
    # So the cycle orbit's periapsis speed sits just under escape at 200 km.
    v_esc = escape_velocity(Earth, LEO_ALTITUDE)
    v_rf = puffsat_cycle_periapsis_speed()
    assert v_rf < v_esc
    assert is_nearly_equal(v_rf, 10.9503 * u.km / u.s, percent=0.01)
    assert (v_esc - v_rf) < 0.06 * u.km / u.s
    # The period is a nearly free choice: 5 to 60 days spans ~120 m/s, because
    # the mass ratio is blind to v_rf while v_b >> v_rf. Nothing downstream
    # should hinge on the 20-day value.
    speeds = [
        puffsat_cycle_periapsis_speed(period=days * u.day) for days in (5, 20, 60)
    ]
    assert all(s < v_esc for s in speeds)
    spread = max(speeds) - min(speeds)
    assert spread < 0.13 * u.km / u.s
    # Longer period -> higher apoapsis -> closer to escape, monotonically.
    assert speeds[0] < speeds[1] < speeds[2]


def test_puffsat_cycle_growth_trades_delta_v_against_cycle_time() -> None:
    # The point of scoring on doubling time: a chain that is cheaper in delta-v
    # can still be the worse machine if it takes longer to fly. Pin that the
    # comparison actually inverts, so nobody "optimizes" back to min-dv.
    cheap_slow = puffsat_cycle_growth(
        total_dv=1.5 * u.km / u.s,
        collision_speed=51.134 * u.km / u.s,
        trip_time=12.0 * u.year,
    )
    dear_fast = puffsat_cycle_growth(
        total_dv=5.3392 * u.km / u.s,
        collision_speed=60.0 * u.km / u.s,
        trip_time=3.26 * u.year,
    )
    # The cheap chain grows far more per cycle ...
    assert cheap_slow.net_growth > dear_fast.net_growth
    # ... and still doubles slower, because the cycle is what gets paid.
    assert cheap_slow.doubling_time > dear_fast.doubling_time

    # The direct powered flyby's own numbers, as the reference point.
    assert dear_fast.mass_ratio == pytest.approx(7.9400, rel=1e-3)
    assert dear_fast.delivered_fraction == pytest.approx(0.23866, rel=1e-3)
    assert dear_fast.net_growth == pytest.approx(1.8949, rel=1e-3)
    # Cycle is the trip plus the 20-day coast back to periapsis.
    assert dear_fast.cycle_time.to_value(u.year) == pytest.approx(3.3148, rel=1e-3)
    assert dear_fast.doubling_time.to_value(u.year) == pytest.approx(3.595, rel=1e-3)


def test_puffsat_cycle_growth_reports_no_doubling_when_the_cycle_shrinks() -> None:
    # A cycle that loses more to propellant than the collision returns never
    # doubles. Report that as infinite time rather than a negative one, which is
    # what cycle * ln2 / ln(growth) yields for growth < 1 and would sort as the
    # *best* answer in a minimizer.
    shrinking = puffsat_cycle_growth(
        total_dv=12.0 * u.km / u.s,
        collision_speed=51.134 * u.km / u.s,
        trip_time=5.0 * u.year,
    )
    assert shrinking.net_growth < 1.0
    assert np.isinf(shrinking.doubling_time.to_value(u.year))


def test_puffsat_cycle_growth_rejects_a_collision_below_the_push_target() -> None:
    # v_b <= v_rf makes ln((v_b - v_ri)/(v_b - v_rf)) undefined or negative --
    # the PuffSat simply cannot drive the mass to the cycle orbit. Fail loudly
    # rather than return a nonsense ratio.
    with pytest.raises(ValueError, match="does not exceed the push target"):
        puffsat_cycle_growth(
            total_dv=1.0 * u.km / u.s,
            collision_speed=5.0 * u.km / u.s,
            trip_time=3.0 * u.year,
        )
