"""Tests for src/heliocentric_reintercept.py."""

import numpy as np
from astropy import units as u
from boinor.bodies import Sun

from src.astro_constants import SOLAR_DIVE_PERIAPSIS_BURN
from src.heliocentric_reintercept import (
    SOLAR_DIVE_PERIAPSIS,
    boosted_solar_dive_v_infinity,
    earth_reintercept_cycle_floor,
    launch_capacity_time,
    millionfold_scaling_time,
    min_energy_solar_dive_time,
    periapsis_reaim_cost_per_degree,
    single_impulse_resonant_dive,
    solar_dive_periapsis_speed,
    solar_dive_reintercept_gap,
    solar_dive_whip_around_angle,
    two_impulse_phasing_loop,
)
from src.orbit_utils import escape_velocity
from tests.test_helpers import is_nearly_equal


def test_solar_dive_periapsis_speed_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: a dive from 1 AU to 4 solar radii reaches
    # ~309 km/s at periapsis (the paper rounds the ~306 km/s dive speed up to the
    # local escape speed). Pin the value from the repo's primitives.
    v = solar_dive_periapsis_speed()
    assert is_nearly_equal(v, 306.0 * u.km / u.s, percent=0.01)


def test_boosted_solar_dive_v_infinity_uses_actual_incoming_speed() -> None:
    # The incoming minimum-energy ellipse reaches ~306.0 km/s, not the local
    # ~308.8 km/s escape speed. The burn is tuned to the resonant dive (see
    # test_single_impulse_resonant_dive_matches_main_text_scale), so the shallower
    # min-energy default leaves ~146.9 km/s -- computed from the actual incoming
    # speed, not by injecting energy the ellipse lacks.
    v_inf = boosted_solar_dive_v_infinity()
    assert is_nearly_equal(v_inf, 146.9 * u.km / u.s, percent=0.01)


def test_single_impulse_resonant_dive_matches_main_text_scale() -> None:
    # The burn is tuned so the resonant dive -- the no-ISRU cycle's actual return
    # leg -- carries the main text's ~150 km/s Earth-crossing excess.
    dive = single_impulse_resonant_dive()
    v_inf = boosted_solar_dive_v_infinity(apoapsis_radius=dive.closing_aphelion)
    assert is_nearly_equal(v_inf, 150 * u.km / u.s, percent=0.01)


def test_boosted_solar_dive_v_infinity_conserves_incoming_orbit_energy() -> None:
    incoming = solar_dive_periapsis_speed()
    escape = escape_velocity(Sun, SOLAR_DIVE_PERIAPSIS - Sun.R)
    expected = np.sqrt((incoming + SOLAR_DIVE_PERIAPSIS_BURN) ** 2 - escape**2)

    assert is_nearly_equal(
        boosted_solar_dive_v_infinity(), expected.to(u.km / u.s), percent=1e-9
    )


def test_boosted_solar_dive_v_infinity_tracks_incoming_aphelion() -> None:
    # A higher incoming aphelion carries more periapsis energy, so the same burn
    # must leave more hyperbolic excess instead of reusing the 1 AU result.
    one_au = boosted_solar_dive_v_infinity(apoapsis_radius=1 * u.AU)
    two_au = boosted_solar_dive_v_infinity(apoapsis_radius=2 * u.AU)
    assert two_au > one_au


def test_boosted_solar_dive_requires_an_escaping_post_burn_speed() -> None:
    with np.testing.assert_raises_regex(ValueError, "does not put.*hyperbola"):
        boosted_solar_dive_v_infinity(periapsis_burn=0 * u.km / u.s)


def test_min_energy_solar_dive_time_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: the minimum-energy fall from 1 AU to 4 solar
    # radii (half of a ~0.509 AU ellipse) takes ~66 days.
    t = min_energy_solar_dive_time()
    assert is_nearly_equal(t, 66 * u.day, percent=0.02)


def test_solar_dive_whip_around_angle_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: 180 deg falling to periapsis plus ~130 deg
    # of hyperbolic climb-out gives a ~310 deg whip-around.
    whip = solar_dive_whip_around_angle()
    assert is_nearly_equal(whip, 310 * u.deg, percent=0.02)


def test_solar_dive_reintercept_gap_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: over the ~0.21 yr round trip Earth advances
    # only ~76 deg while the projectile whips ~310 deg and re-crosses ~50 deg
    # behind launch, an unphased miss of ~125 deg. Crossing 1 AU is not reaching
    # Earth.
    gap = solar_dive_reintercept_gap()
    assert is_nearly_equal(gap, 125 * u.deg, percent=0.03)


def test_periapsis_reaim_cost_per_degree_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: re-aiming at the ~309 km/s periapsis costs
    # ~5.4 km/s per degree, prohibitive against a ~24 km/s dive boost -- so the
    # miss is fixed by phasing, not re-aiming.
    cost = periapsis_reaim_cost_per_degree()
    assert is_nearly_equal(cost, 5.4 * u.km / u.s, percent=0.02)


def test_two_impulse_phasing_loop_is_free_in_total_impulse() -> None:
    # Appendix sec:earth_reintercept: the boost sequence 29.78 / 24.4 / 5.7 km/s
    # (Earth, dip-aphelion, deep-dive-aphelion) has two colinear retrograde legs
    # summing to a direct dive's ~24 km/s, and the dip returns after ~0.65 yr.
    loop = two_impulse_phasing_loop()
    assert is_nearly_equal(loop.earth_speed, 29.78 * u.km / u.s, percent=0.01)
    assert is_nearly_equal(loop.dip_aphelion_speed, 24.4 * u.km / u.s, percent=0.02)
    assert is_nearly_equal(
        loop.deep_dive_aphelion_speed, 5.7 * u.km / u.s, percent=0.02
    )
    assert is_nearly_equal(loop.total_boost, 24 * u.km / u.s, percent=0.02)
    assert is_nearly_equal(loop.dip_return_time, 0.65 * u.year, percent=0.02)
    # The two legs are colinear and retrograde, so they add to the direct boost.
    leg_one = loop.earth_speed - loop.dip_aphelion_speed
    leg_two = loop.dip_aphelion_speed - loop.deep_dive_aphelion_speed
    assert is_nearly_equal(leg_one + leg_two, loop.total_boost, percent=1e-6)


def test_single_impulse_resonant_dive_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: folding the phasing into one Earth boost
    # aims the projectile outbound to a ~1.9 AU aphelion; it re-intercepts Earth
    # after ~0.89 yr, and the boost grows to ~37.5 km/s (a ~24 km/s retrograde
    # component plus a ~29 km/s outbound radial one).
    dive = single_impulse_resonant_dive()
    assert is_nearly_equal(dive.closing_aphelion, 1.9 * u.AU, percent=0.02)
    assert is_nearly_equal(dive.reintercept_time, 0.89 * u.year, percent=0.02)
    assert is_nearly_equal(dive.earth_boost, 37.5 * u.km / u.s, percent=0.02)
    assert is_nearly_equal(dive.retrograde_component, 24 * u.km / u.s, percent=0.02)
    assert is_nearly_equal(dive.radial_component, 29 * u.km / u.s, percent=0.03)


def test_single_impulse_resonant_dive_couples_climb_energy_to_aphelion() -> None:
    # This high-precision root pins the energy-coupled solve. Computing climb-out
    # once from the default 1 AU ellipse instead gives the old ~1.92593 AU root.
    dive = single_impulse_resonant_dive()
    assert is_nearly_equal(dive.closing_aphelion, 1.925931893 * u.AU, percent=1e-6)
    assert is_nearly_equal(dive.reintercept_time, 0.892533677 * u.year, percent=1e-6)


def test_single_impulse_resonant_dive_boost_decomposition() -> None:
    # The single Earth boost is the vector sum of a retrograde leg and an outbound
    # radial leg, and the retrograde leg is the same ~24 km/s a direct dive (and
    # the free two-impulse loop) spends to reach the solar periapsis. The radial
    # leg -- what "buys" the phasing coast -- is the whole reason the boost grows.
    dive = single_impulse_resonant_dive()
    combined = np.sqrt(dive.retrograde_component**2 + dive.radial_component**2).to(
        u.km / u.s
    )
    assert is_nearly_equal(dive.earth_boost, combined, percent=1e-9)
    assert is_nearly_equal(
        dive.retrograde_component, two_impulse_phasing_loop().total_boost, percent=0.01
    )
    assert dive.radial_component > 0 * u.km / u.s


def test_single_impulse_resonant_dive_costs_more_than_free_phasing() -> None:
    # Appendix sec:earth_reintercept: the single-impulse route needs only the Earth
    # node but is not free -- its ~37.5 km/s boost exceeds the two-impulse loop's free
    # ~24 km/s total, which is why its doubling factor falls below two. Its ~0.89 yr
    # cycle is also longer than the ~0.86 yr two-impulse floor.
    dive = single_impulse_resonant_dive()
    assert dive.earth_boost > two_impulse_phasing_loop().total_boost
    assert dive.reintercept_time > earth_reintercept_cycle_floor()


def test_earth_reintercept_cycle_floor_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: Earth reaches the fixed 1 AU crossing
    # longitude after sweeping the whip-around fraction of a year, ~0.86 yr. This
    # supersedes the paper's earlier ~0.5 yr ("6 month") cycle.
    floor = earth_reintercept_cycle_floor()
    assert is_nearly_equal(floor, 0.86 * u.year, percent=0.02)
    # The floor is strictly longer than the retired 6-month cycle.
    assert floor > 0.5 * u.year


def test_millionfold_scaling_time_is_about_17_years() -> None:
    # Appendix sec:earth_reintercept: ~20 doublings at ~0.86 yr each reach a
    # millionfold in ~17 years -- not the "under a decade" a ~0.5 yr cycle implies.
    t = millionfold_scaling_time()
    assert is_nearly_equal(t, 17 * u.year, percent=0.03)
    # The correction is conservative: the old 6-month cycle gave under a decade.
    old = launch_capacity_time(2, 0.5 * u.year)
    assert t > old
    assert old < 10 * u.year


def test_millionfold_scaling_time_uses_derived_floor() -> None:
    # millionfold_scaling_time defaults to the derived re-intercept floor, so it
    # equals launch_capacity_time evaluated at that same floor (no magic 0.82).
    floor = earth_reintercept_cycle_floor()
    assert is_nearly_equal(
        millionfold_scaling_time(), launch_capacity_time(2, floor), percent=1e-9
    )
