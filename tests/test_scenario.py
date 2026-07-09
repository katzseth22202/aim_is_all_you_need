"""Tests for scenario-related functions in scenario.py."""

import numpy as np
import pytest
from astropy import units as u

from src.astro_constants import CERES_A, EARTH_A, MARS_A, SATURN_A, VENUS_A
from src.propulsion import payload_mass_ratio
from src.scenario import (
    SCENARIO_COLUMNS,
    PuffSatScenario,
    boosted_solar_dive_v_infinity,
    earth_reintercept_cycle_floor,
    earth_reintercept_scenarios,
    find_best_lunar_return,
    launch_capacity_time,
    lunar_return_transfer_dv,
    millionfold_scaling_time,
    min_energy_solar_dive_time,
    paper_scenarios,
    periapsis_reaim_cost_per_degree,
    scenarios_to_dataframe,
    single_impulse_resonant_dive,
    solar_dive_periapsis_speed,
    solar_dive_reintercept_gap,
    solar_dive_whip_around_angle,
    solar_impact_dv,
    suborbital_200km_propellant_fraction,
    two_impulse_phasing_loop,
)
from tests.test_helpers import is_nearly_equal


def test_solar_impact_dv() -> None:
    # solar_impact_dv reproduces the paper's solar-impact delta-v figures, which
    # equal the heliocentric circular orbital velocity at each body's distance:
    # ~30 km/s from Earth, ~10 km/s from Saturn, ~18 km/s from Ceres
    # (citation audit L1047 Earth/Saturn, L1107 Ceres).
    assert is_nearly_equal(solar_impact_dv(EARTH_A), 29.78 * u.km / u.s, percent=0.01)
    assert is_nearly_equal(solar_impact_dv(SATURN_A), 9.64 * u.km / u.s, percent=0.01)
    assert is_nearly_equal(solar_impact_dv(CERES_A), 17.91 * u.km / u.s, percent=0.01)


def test_suborbital_200km_propellant_fraction() -> None:
    # Paper L199 (citation audit L206): "a suborbital rocket that merely reaches
    # 200 km in altitude might have a propellant mass fraction under 60 percent".
    # With a ~2.5 km/s suborbital delta-v budget (2.0 km/s ideal coast + 0.5 km/s
    # gravity drag) and a methalox sea-level Isp of 310 s, the Tsiolkovsky rocket
    # equation gives ~56%, backing the under-60% claim.
    frac = suborbital_200km_propellant_fraction()
    # The headline claim: comfortably below 60 percent.
    assert frac < 0.6 * u.dimensionless_unscaled
    # Within 1% of the value computed from the repo's primitives.
    assert is_nearly_equal(frac, 0.5606 * u.dimensionless_unscaled, percent=0.01)


def test_lunar_return_transfer_dv() -> None:
    # Paper L691: from a lunar-return orbit, firing at a 200 km perigee, only
    # 300-600 m/s beyond lunar escape reaches a Venus or Mars Hohmann transfer.
    venus_dv = lunar_return_transfer_dv(VENUS_A)
    mars_dv = lunar_return_transfer_dv(MARS_A)
    # Both fall inside the paper's stated 300-600 m/s band.
    assert 300 * u.m / u.s < venus_dv < 600 * u.m / u.s
    assert 300 * u.m / u.s < mars_dv < 600 * u.m / u.s
    # Within 1% of the values computed from the repo's primitives.
    assert is_nearly_equal(venus_dv, 372.36 * u.m / u.s, percent=0.01)
    assert is_nearly_equal(mars_dv, 480.06 * u.m / u.s, percent=0.01)


def test_puffsat_scenario_knows_its_mass_ratio() -> None:
    # A scenario computes its own mass ratio from its three velocities, binding
    # to the payload_mass_ratio primitive rather than living in a DataFrame cell.
    scenario = PuffSatScenario(
        v_rf=8 * u.km / u.s,
        v_b=11 * u.km / u.s,
        v_ri=2 * u.km / u.s,
        desc="hand case",
    )
    expected = payload_mass_ratio(
        v_rf=8 * u.km / u.s, v_b=11 * u.km / u.s, v_ri=2 * u.km / u.s
    )
    assert float(scenario.mass_ratio) == pytest.approx(float(expected))


def test_puffsat_scenario_v_ri_defaults_to_zero_kms() -> None:
    # The initial velocity defaults to a units-carrying 0 km/s, not a bare 0.
    scenario = PuffSatScenario(v_rf=8 * u.km / u.s, v_b=11 * u.km / u.s, desc="default")
    assert scenario.v_ri == 0 * u.km / u.s
    assert scenario.v_ri.unit.is_equivalent(u.km / u.s)


def test_paper_scenarios_is_uniform_catalog() -> None:
    # The catalog is a uniform list of single-collision scenarios -- the
    # lunar-return optimum (a blended result) is no longer shoehorned in.
    catalog = paper_scenarios()
    assert len(catalog) == 8
    assert all(isinstance(s, PuffSatScenario) for s in catalog)
    # Every scenario yields a finite, real mass ratio (no tuple-valued v_rf row).
    for scenario in catalog:
        assert np.isfinite(float(scenario.mass_ratio))


def test_paper_scenarios_mass_ratios_regression() -> None:
    # Pin the paper's headline mass ratios so no future refactor can silently
    # shift them. Values captured from the repo's primitives.
    expected = [
        1.28130083,
        2.30831207,
        2.97219375,
        3.82951318,
        1.84518438,
        9.33111762,
        2.30831207,
        1.29699573,
    ]
    actual = [float(s.mass_ratio) for s in paper_scenarios()]
    assert actual == pytest.approx(expected, rel=1e-6)


def test_find_best_lunar_return_reproduces_paper_lunar_cycle() -> None:
    # Paper (Appendix D context, p.13-14): a ~1.9 km/s perigee burn reaches the
    # Moon at ~7.2 km/s, and the lunar cycle launches "about 1.455 times the
    # starting mass" per loop. Pin the optimum so this headline figure can't drift.
    best = find_best_lunar_return()
    # Headline claim: ~1.455x per loop.
    assert is_nearly_equal(
        best.combined_mass_ratio, 1.455 * u.dimensionless_unscaled, percent=0.01
    )
    # Exact values from the repo's primitives.
    assert is_nearly_equal(best.burn, 1.946162 * u.km / u.s, percent=0.001)
    assert is_nearly_equal(
        best.combined_mass_ratio, 1.455765 * u.dimensionless_unscaled, percent=0.001
    )
    assert is_nearly_equal(best.incoming_v, 7.207867 * u.km / u.s, percent=0.001)


def test_find_best_lunar_return_raises_when_no_feasible_burn() -> None:
    # The documented error mode: if no candidate burn yields an incoming speed
    # above the (here, impossibly high) retrograde requirement, every burn is
    # filtered out and the function raises rather than returning None.
    with pytest.raises(ValueError, match="No valid burn"):
        find_best_lunar_return(retrograde_dv_required=1000 * u.km / u.s)


def test_solar_dive_periapsis_speed_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: a dive from 1 AU to 4 solar radii reaches
    # ~309 km/s at periapsis (the paper rounds the ~306 km/s dive speed up to the
    # local escape speed). Pin the value from the repo's primitives.
    v = solar_dive_periapsis_speed()
    assert is_nearly_equal(v, 306.0 * u.km / u.s, percent=0.01)


def test_boosted_solar_dive_v_infinity_matches_appendix() -> None:
    # Appendix sec:earth_reintercept: a ~34.5 km/s PuffSat boost lifts the
    # ~309 km/s escape speed to ~343 km/s, leaving ~150 km/s of hyperbolic-excess
    # speed to spare, so the return genuinely escapes on a hyperbola.
    v_inf = boosted_solar_dive_v_infinity()
    assert is_nearly_equal(v_inf, 150 * u.km / u.s, percent=0.01)


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


def test_earth_reintercept_scenarios_phases_for_return() -> None:
    # Appendix sec:earth_reintercept: the phased single-impulse resonant dive is
    # the row that actually re-intercepts Earth. Off the same ~69 km/s Jovian
    # PuffSat, its boost is the resonant dive's ~37.5 km/s Earth boost (not the
    # prograde Parker row's minimum-energy ~23.7 km/s injection).
    catalog = earth_reintercept_scenarios()
    assert len(catalog) == 1
    phased = catalog[0]
    assert is_nearly_equal(
        phased.v_rf, single_impulse_resonant_dive().earth_boost, percent=1e-9
    )
    # Pin the phased mass ratio (captured from the repo's primitives).
    assert float(phased.mass_ratio) == pytest.approx(2.05041932, rel=1e-6)


def test_earth_reintercept_phasing_lowers_mass_ratio() -> None:
    # The whole point: folding Earth-return phasing into the boost costs delta-v,
    # so the phased dive's mass ratio sits below the prograde Parker injection's
    # ~3.83 (paper_scenarios index 3) -- it does not come for free.
    phased_ratio = float(earth_reintercept_scenarios()[0].mass_ratio)
    prograde_parker_ratio = float(paper_scenarios()[3].mass_ratio)
    assert phased_ratio < prograde_parker_ratio


def test_scenarios_to_dataframe_projects_catalog() -> None:
    # The table is a pure projection: right schema, one row per scenario, and
    # each row's ratio equals the scenario's own mass_ratio.
    catalog = paper_scenarios()
    df = scenarios_to_dataframe(catalog)
    assert list(df.columns) == SCENARIO_COLUMNS
    assert len(df) == len(catalog)
    for i, scenario in enumerate(catalog):
        assert float(df.iloc[i]["payload_puffsat_mass_ratio"]) == pytest.approx(
            float(scenario.mass_ratio)
        )
        assert df.iloc[i]["desc"] == scenario.desc
