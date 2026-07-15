"""Tests for scenario-related functions in scenario.py."""

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Earth, Jupiter
from boinor.core.iod import izzo
from boinor.threebody.flybys import compute_flyby

from src.astro_constants import (
    ASSIST_CHAIN_MAX_FLYBYS,
    ASSIST_CHAIN_MAX_TRIP_TIME,
    ASSIST_CHAIN_PHASING_BUDGET,
    CERES_A,
    EARTH_A,
    JUPITER_A,
    JUPITER_FLYBY_MAX_TOF,
    LEO_ALTITUDE,
    LOW_JUPITER_ALTITUDE,
    MARS_A,
    SATURN_A,
    VENUS_A,
)
from src.orbit_utils import (
    apoapsis_speed,
    escape_velocity,
    hyperbolic_eccentricity,
    orbit_from_rp_ra,
    periapsis_speed,
    speed_with_escape_energy,
)
from src.propulsion import payload_mass_ratio, retrograde_jovian_hohmann_transfer
from src.scenario import (
    SCENARIO_COLUMNS,
    PuffSatScenario,
    _assist_chain_params,
    _body_state,
    _conic_radius_crossings,
    _earth_phase_mismatch,
    _flyby_mismatch_burn,
    _flyby_return_leg,
    _jupiter_assist_body,
    _mean_motion,
    _phased_jovian_flyby,
    _phased_ladder_burn,
    _phased_leg_rotations,
    _powered_flyby_leg,
    _powered_flyby_params,
    _powered_jovian_terminal,
    _powered_node_burn,
    _wrap_pi,
    apoapsis_raise_economics,
    apoapsis_raise_finite_burn,
    apoapsis_raise_reintercept,
    assist_chain_return,
    assist_chain_window_cadence,
    boosted_solar_dive_v_infinity,
    earth_reintercept_cycle_floor,
    earth_reintercept_scenarios,
    find_best_lunar_return,
    hohmann_v_infinity,
    jovian_return_phasing_envelope,
    jupiter_flyby_vb_trade_curve,
    launch_capacity_time,
    lunar_return_transfer_dv,
    lunar_transfer_periapsis_speed,
    millionfold_scaling_time,
    min_energy_solar_dive_time,
    minimum_departure_burn_assist_chain,
    paper_scenarios,
    parker_injection_burns,
    periapsis_reaim_cost_per_degree,
    powered_jovian_flyby_return,
    puffsat_cycle_growth,
    puffsat_cycle_periapsis_speed,
    scenarios_to_dataframe,
    single_impulse_resonant_dive,
    solar_dive_periapsis_speed,
    solar_dive_reintercept_gap,
    solar_dive_whip_around_angle,
    solar_impact_dv,
    suborbital_200km_propellant_fraction,
    two_impulse_phasing_loop,
    venus_reach_departure_floor,
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
    # Every paper row cites the section that develops it, mirroring the
    # \autoref each row of the paper's tab:mass_scenarios carries.
    for scenario in catalog + earth_reintercept_scenarios():
        assert scenario.paper_ref.startswith("sec:")


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
    # prograde Parker row's minimum-energy ~23.7 km/s injection). The apoapsis-raise
    # re-intercept is the second phased row.
    catalog = earth_reintercept_scenarios()
    assert len(catalog) == 2
    phased = catalog[0]
    assert is_nearly_equal(
        phased.v_rf, single_impulse_resonant_dive().earth_boost, percent=1e-9
    )
    # Pin the phased mass ratio (captured from the repo's primitives).
    assert float(phased.mass_ratio) == pytest.approx(2.05041932, rel=1e-6)
    # The apoapsis-raise row is scored on its own collision: v_rf = 200 km Earth
    # escape speed, v_b = the ~24 km/s closing speed, ratio ~2.62.
    apoapsis_row = catalog[1]
    assert is_nearly_equal(
        apoapsis_row.v_b, apoapsis_raise_reintercept().closing_speed, percent=1e-9
    )
    assert float(apoapsis_row.mass_ratio) == pytest.approx(2.61570, rel=1e-4)


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
        assert df.iloc[i]["paper_ref"] == scenario.paper_ref


def test_apoapsis_raise_reintercept_matches_design_point() -> None:
    # Apoapsis-raise design doc Sec. 4 (locked design point): raising heliocentric
    # aphelion to a phasing-exact Q ~ 2.26 AU with a methalox Oberth burn, then a
    # 4 km/s retrograde argon-SEP burn at apoapsis, lets the craft fall back to
    # intercept Earth at a ~24 km/s closing speed after ~1.69 yr. Every value is
    # pinned from the repo's primitives against the design doc's table.
    r = apoapsis_raise_reintercept()
    # The single free knob Q is root-solved to phasing-exact (residual ~ 0).
    assert is_nearly_equal(r.aphelion, 2.2579 * u.AU, percent=0.001)  # doc 2.258 AU
    assert is_nearly_equal(
        r.leg1_semimajor_axis, 1.6289 * u.AU, percent=0.001
    )  # doc 1.629 AU
    assert abs(r.phasing_residual) < 1e-6 * u.deg  # doc 0.036 deg; root-solved to ~0
    # Burns: methalox Oberth departure and the argon SEP apoapsis kick.
    assert is_nearly_equal(
        r.departure_burn, 1.2014 * u.km / u.s, percent=0.001
    )  # doc 1.202 km/s
    assert is_nearly_equal(r.sep_burn, 4.0 * u.km / u.s, percent=1e-9)  # doc 4.000
    # Rocket-equation mass fractions retained and their product.
    assert r.methalox_mass_fraction == pytest.approx(0.7244, rel=1e-3)  # doc 0.724
    assert r.sep_mass_fraction == pytest.approx(0.8155, rel=1e-3)  # doc 0.816
    assert r.combined_dry_fraction == pytest.approx(0.5908, rel=1e-3)  # doc 0.591
    # Trajectory: truncated perihelion (never reached), SOI excess, closing speed.
    assert is_nearly_equal(
        r.truncated_perihelion, 0.4598 * u.AU, percent=0.002
    )  # doc 0.460 AU
    assert is_nearly_equal(
        r.v_infinity_arrival, 21.393 * u.km / u.s, percent=0.001
    )  # doc 21.394 km/s
    assert is_nearly_equal(
        r.closing_speed, 24.059 * u.km / u.s, percent=0.001
    )  # doc 24.060 km/s
    assert is_nearly_equal(
        r.transit_time, 1.69196 * u.year, percent=0.001
    )  # doc 1.692 yr
    # The 2x escape-velocity design bar is met with the doc's ~9% margin.
    assert is_nearly_equal(
        r.twice_escape_target, 22.017 * u.km / u.s, percent=0.001
    )  # doc 22.017 km/s
    assert r.closing_speed > r.twice_escape_target
    assert float((r.closing_speed / r.twice_escape_target).value) == pytest.approx(
        1.093, rel=1e-2
    )  # doc margin 9.3%


def test_apoapsis_raise_economics_matches_design_point() -> None:
    # Apoapsis-raise design doc Sec. 5 (economics): eq:PuffSat_ratio at f=0.8 with
    # v_rf = 200 km Earth escape and v_p = the closing speed gives m_r/m_p ~ 2.62,
    # a ~+54.7% net growth per 1.69 yr cycle, and ~54 yr to a millionfold. (The
    # doc's 2.619 uses v_rf=11.0 flat; the repo uses the self-consistent 11.009 km/s
    # Earth escape speed, which rounds to the same 2.62.)
    econ = apoapsis_raise_economics()
    assert econ.payload_puffsat_mass_ratio == pytest.approx(2.6157, rel=1e-3)  # ~2.62
    assert econ.net_growth_per_cycle == pytest.approx(1.5453, rel=1e-3)  # doc 1.5469
    assert is_nearly_equal(econ.cycle_time, 1.69196 * u.year, percent=0.001)  # 1.692
    assert econ.cycles_to_millionfold == pytest.approx(31.75, rel=1e-2)  # doc 31.7
    assert is_nearly_equal(
        econ.time_to_millionfold, 53.71 * u.year, percent=0.01
    )  # doc ~53.6 yr
    assert econ.doublings_per_year == pytest.approx(0.371, rel=1e-2)  # doc 0.372
    # The whole point: the gentle 24 km/s cycle grows, but ~3x slower than the
    # ~17 yr solar-dive loop -- the price of the far lower closing speed.
    assert econ.net_growth_per_cycle > 1.0
    assert econ.time_to_millionfold > millionfold_scaling_time()


def test_apoapsis_raise_finite_burn_confirms_impulsive() -> None:
    # Apoapsis-raise design doc Sec. 3: a finite-thrust SEP burn centered on
    # apoapsis reproduces the impulsive kick -- a 60-90 day burn to within ~1% in
    # closing speed (0.3-0.7% in the doc), a 180-day burn within 2.8%. Apoapsis is
    # the slowest part of the orbit, so the thrust stays near-tangential throughout.
    r = apoapsis_raise_reintercept()
    finite_60 = apoapsis_raise_finite_burn(burn_duration=60 * u.day, reintercept=r)
    finite_90 = apoapsis_raise_finite_burn(burn_duration=90 * u.day, reintercept=r)
    finite_180 = apoapsis_raise_finite_burn(burn_duration=180 * u.day, reintercept=r)
    # Closing speed reproduced to within the doc's stated windows.
    assert finite_60.closing_speed_error < 0.005  # doc ~0.3%
    assert finite_90.closing_speed_error < 0.01  # doc ~0.7%
    assert finite_180.closing_speed_error < 0.03  # doc 2.8%
    # For the 60-90 day window the doc claims ~1 day in transit, ~0.005 AU in
    # perihelion, and under 1 deg in phasing -- all hold here.
    assert abs(finite_90.transit_time - r.transit_time) < 0.01 * u.year
    assert abs(finite_90.truncated_perihelion - r.truncated_perihelion) < 0.01 * u.AU
    assert abs(finite_90.phasing_residual) < 1.0 * u.deg
    # Longer burns drift monotonically further from the impulsive kick.
    assert (
        finite_60.closing_speed_error
        < finite_90.closing_speed_error
        < finite_180.closing_speed_error
    )


# ---------------------------------------------------------------------------
# Powered Jovian flyby retrograde return (ADR 0002-jupiter-flyby-objective).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flyby_params():
    """Float parameter block for the powered-flyby helpers."""
    return _powered_flyby_params()


@pytest.fixture(scope="module")
def flyby_optimum():
    """The end-to-end optimum, computed once for the module (seeded search)."""
    return powered_jovian_flyby_return()


@pytest.fixture(scope="module")
def chain_at_300(flyby_optimum):
    """The 300 m/s assist chain, computed once (the beam search is slow)."""
    chain = assist_chain_return(
        departure_burn=0.300 * u.km / u.s,
        target_collision_speed=flyby_optimum.collision_speed,
    )
    assert chain is not None
    return chain


def test_flyby_return_leg_reproduces_retrograde_hohmann(flyby_params):
    # The v_b convention pin: fed the retrograde Jupiter->Earth Hohmann state
    # (tangential retrograde at Jupiter's orbit radius, apoapsis speed of the
    # 1 AU x 5.2028 AU transfer ellipse), the float return leg must reproduce
    # retrograde_jovian_hohmann_transfer() -- the ~69.27 km/s the catalog's
    # three Jovian rows assume.
    v_apo = apoapsis_speed(
        orbit_from_rp_ra(apoapsis_radius=JUPITER_A, periapsis_radius=EARTH_A)
    ).to_value(u.km / u.s)
    leg = _flyby_return_leg(-v_apo, 0.0, flyby_params)
    assert leg is not None
    reference = retrograde_jovian_hohmann_transfer()
    assert is_nearly_equal(leg.collision_speed * u.km / u.s, reference, percent=0.001)
    # Tangential arrival: the return perihelion is 1 AU itself.
    assert is_nearly_equal(leg.perihelion * u.km, EARTH_A.to(u.km), percent=0.001)
    # Half the transfer ellipse period, ~2.73 yr.
    assert is_nearly_equal((leg.tof * u.s).to(u.year), 2.731 * u.year, percent=0.01)


def test_flyby_return_leg_rejects_prograde(flyby_params):
    # Retrograde is orbit-level: a prograde tangential state must be rejected,
    # not scored (the retrograde-plunge degeneracy guard, ADR 0002).
    assert _flyby_return_leg(+5.0, 0.0, flyby_params) is None
    assert _flyby_return_leg(0.0, -10.0, flyby_params) is None


def test_powered_flyby_unpowered_limit(flyby_params):
    # With dv2 = 0 the split hyperbola collapses to a single one: the excess
    # speed is conserved and the turn equals the unpowered 2*asin(1/e), with e
    # from the cross-module hyperbolic_eccentricity. Pin point verified
    # feasible: v_inf = 11 km/s, tangential, periapsis 6x the floor.
    periapsis_radius = 6.0 * flyby_params.periapsis_floor
    leg = _powered_flyby_leg(
        v_infinity_earth=11.0,
        aim_angle=0.0,
        periapsis_radius=periapsis_radius,
        flyby_burn=0.0,
        descending_arrival=False,
        bend_sign=1.0,
        params=flyby_params,
    )
    assert leg is not None
    assert leg.v_infinity_out == pytest.approx(leg.v_infinity_in, rel=1e-9)
    ecc = hyperbolic_eccentricity(
        periapsis_radius * u.km, leg.v_infinity_in * u.km / u.s, Jupiter
    )
    assert leg.turn_angle == pytest.approx(2.0 * np.arcsin(1.0 / ecc), rel=1e-9)


def test_powered_flyby_burn_grows_v_infinity(flyby_params):
    # A periapsis burn pumps the outgoing excess speed above the incoming one
    # (Oberth) while shrinking the outbound bend contribution: the powered
    # trajectory's turn must be smaller than the unpowered one at the same
    # periapsis, since the faster outbound hyperbola bends less.
    periapsis_radius = 2.0 * flyby_params.periapsis_floor
    powered = _powered_flyby_leg(
        12.0, 0.0, periapsis_radius, 2.0, False, 1.0, flyby_params
    )
    assert powered is not None
    assert powered.v_infinity_out > powered.v_infinity_in
    ecc_in = 1.0 + periapsis_radius * powered.v_infinity_in**2 / (
        flyby_params.mu_jupiter
    )
    unpowered_turn = 2.0 * np.arcsin(1.0 / ecc_in)
    assert powered.turn_angle < unpowered_turn


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
    assert payload_mass_ratio(v_rf=prograde_burn, v_b=flyby_optimum.collision_speed) > 0
    assert (
        payload_mass_ratio(v_rf=retrograde_burn, v_b=flyby_optimum.collision_speed) > 0
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


def test_venus_reach_departure_floor() -> None:
    # The analytic floor of the assist chain: below ~279.4 m/s even the best
    # (anti-tangential) aim leaves the transfer perihelion above Venus's orbit.
    # In particular the hoped-for 250 m/s cannot reach Venus at all.
    floor = venus_reach_departure_floor()
    assert 0.279 * u.km / u.s < floor < 0.280 * u.km / u.s
    assert floor > 0.250 * u.km / u.s


def test_conic_radius_crossings_reproduces_hohmann_leg(flyby_params):
    # Fed the periapsis state of the Earth->Jupiter Hohmann ellipse, the leg
    # machinery must reproduce the textbook transfer: the Jupiter crossing at
    # aphelion after the half period (~2.731 yr) at the apoapsis speed, and the
    # Mars-radius crossings as an outbound/inbound pair sharing h/r.
    transfer = orbit_from_rp_ra(periapsis_radius=EARTH_A, apoapsis_radius=JUPITER_A)
    v_peri = periapsis_speed(transfer).to_value(u.km / u.s)
    jupiter_crossings = _conic_radius_crossings(
        flyby_params.mu_sun,
        flyby_params.r_earth_orbit,
        v_peri,
        0.0,
        flyby_params.r_jupiter_orbit,
    )
    assert jupiter_crossings
    v_apo = apoapsis_speed(transfer).to_value(u.km / u.s)
    leg_time, v_t1, v_r1, _, swept = jupiter_crossings[0]
    assert is_nearly_equal((leg_time * u.s).to(u.year), 2.731 * u.year, percent=0.1)
    assert v_t1 == pytest.approx(v_apo, rel=1e-6)
    assert v_r1 == pytest.approx(0.0, abs=1e-3)
    # A Hohmann runs perihelion to aphelion, so it sweeps exactly half a turn.
    # This is the anchor for the swept angle a phased chain steers on: paired
    # with the leg time it says *where* the craft arrives, not merely when.
    # Tolerance is 1e-4 deg (0.36 arcsec), not tighter: recovering the true
    # anomaly at an apsis inverts a cosine through arccos(-1), which is
    # ill-conditioned there and lands ~1e-6 deg out.
    assert np.degrees(swept) == pytest.approx(180.0, abs=1e-4)
    mars_crossings = _conic_radius_crossings(
        flyby_params.mu_sun,
        flyby_params.r_earth_orbit,
        v_peri,
        0.0,
        float(MARS_A.to_value(u.km)),
    )
    assert len(mars_crossings) == 2
    outbound = [c for c in mars_crossings if c[3]]
    inbound = [c for c in mars_crossings if not c[3]]
    assert len(outbound) == 1 and len(inbound) == 1
    assert outbound[0][0] < inbound[0][0]  # outward crossing comes first
    assert outbound[0][2] > 0.0 > inbound[0][2]  # radial speed signs
    h = flyby_params.r_earth_orbit * v_peri
    assert outbound[0][1] == pytest.approx(h / float(MARS_A.to_value(u.km)), rel=1e-9)


def test_venus_only_pump_ceiling_blocks_vve_at_300_mps(flyby_params) -> None:
    # ADR 0004: a V->V leg preserves the Venus-relative excess speed (the
    # Tisserand invariant), so any number of Venus-only interior flybys caps
    # the Earth-return excess at the single-Venus-pump ceiling. At a 300 m/s
    # departure that ceiling (~5.24 km/s) sits below even the Hohmann
    # Jupiter-reach excess (~8.79), so Cassini-style E-V..V-E-J chains cannot
    # close at low burn -- the pump ladder's V<->E alternations are mandatory.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies

    def max_excess_at(body_from, excess, body_to):
        best = 0.0
        for aim in np.linspace(-np.pi, np.pi, 721):
            v_t = body_from.v_circ + excess * np.cos(aim)
            v_r = excess * np.sin(aim)
            for _, v_t1, v_r1, _, _ in _conic_radius_crossings(
                params.flyby.mu_sun,
                body_from.orbit_radius,
                float(v_t),
                float(v_r),
                body_to.orbit_radius,
            ):
                best = max(best, float(np.hypot(v_t1 - body_to.v_circ, v_r1)))
        return best

    v_esc = flyby_params.v_esc_leo
    launch_excess = np.sqrt((v_esc + 0.300) ** 2 - v_esc**2)
    max_w_venus = max_excess_at(earth, launch_excess, venus)
    ceiling = max_excess_at(venus, max_w_venus, earth)
    assert 5.0 < ceiling < 5.5  # the ~5.24 km/s single-Venus-pump ceiling
    jupiter_reach = hohmann_v_infinity(JUPITER_A).to_value(u.km / u.s)
    assert ceiling < jupiter_reach
    # The retrograde hard floor is stricter still: even a Hohmann arrival's
    # excess at Jupiter is far below Jupiter's orbital speed, so no unpowered
    # bend of any size can turn it retrograde.
    assert jupiter_reach < flyby_params.v_jupiter_orbit


def test_assist_chain_window_cadence() -> None:
    # ADR 0005: the synodic scaffolding behind "how often can the chain fly".
    # Pins the textbook periods -- Earth-Venus synodic ~1.60 yr (583.9 d),
    # Earth-Jupiter ~1.09 yr, the 8-yr Earth-Venus near-cycle, the ~24 yr full
    # V-E-J realignment (also ~2 Jupiter years) -- and the growth arithmetic
    # at the calibrated 300 m/s chain's trip time and end-to-end ratio.
    cadence = assist_chain_window_cadence(
        total_time=3.46 * u.year, end_to_end_mass_ratio=5.72
    )
    assert is_nearly_equal(cadence.venus_window, 1.599 * u.year, percent=0.1)
    assert is_nearly_equal(cadence.jupiter_window, 1.092 * u.year, percent=0.1)
    assert is_nearly_equal(cadence.earth_venus_cycle, 7.99 * u.year, percent=0.1)
    assert 23.5 * u.year < cadence.triple_realignment < 24.5 * u.year
    # The realignment is also ~22 Earth-Jupiter synodics and ~2 Jupiter years.
    assert is_nearly_equal(
        cadence.triple_realignment, 22.0 * cadence.jupiter_window, percent=0.3
    )
    # Effective cycle spans trip time to trip time plus one Venus window.
    assert cadence.effective_cycle_floor == 3.46 * u.year
    assert is_nearly_equal(
        cadence.effective_cycle_ceiling - cadence.effective_cycle_floor,
        cadence.venus_window,
        percent=0.001,
    )
    # Doubling every ~1.4-2.0 yr at ~5.7x per cycle.
    assert 1.3 * u.year < cadence.doubling_time_floor < 1.5 * u.year
    assert 1.9 * u.year < cadence.doubling_time_ceiling < 2.1 * u.year
    # No growth cycle exists at ratio <= 1.
    with pytest.raises(ValueError, match="must exceed 1"):
        assist_chain_window_cadence(3.46 * u.year, 0.9)


@pytest.mark.slow
def test_assist_chain_return_at_300_mps(flyby_optimum, flyby_params):
    # The replay-validated headline chain: 300 m/s of departure burn (vs the
    # powered flyby's 4.45 km/s) reaches the same-target retrograde return via
    # an E-V pump ladder in well under the 10 yr cap.
    result = assist_chain_return(
        departure_burn=0.300 * u.km / u.s,
        target_collision_speed=flyby_optimum.collision_speed,
    )
    assert result is not None
    assert result.collision_speed >= flyby_optimum.collision_speed * 0.9999
    assert result.total_time <= ASSIST_CHAIN_MAX_TRIP_TIME
    assert result.total_time < 4.0 * u.year  # calibrated chain: ~3.46 yr
    assert result.sequence.startswith("E-V") and result.sequence.endswith("-J")
    assert 1 <= result.flyby_count <= ASSIST_CHAIN_MAX_FLYBYS
    # Departure is a coast state, not a flyby: its rotation must be zero.
    assert result.steps[0].rotation_angle == 0.0 * u.deg
    # Bookkeeping consistency: legs sum to the chain time, chain + return to
    # the total, and elapsed is monotone.
    legs_total = sum((step.leg_time for step in result.steps), start=0.0 * u.year)
    assert is_nearly_equal(legs_total, result.chain_time, percent=0.001)
    assert is_nearly_equal(
        result.chain_time + result.return_time, result.total_time, percent=0.001
    )
    elapsed = [step.elapsed.to_value(u.year) for step in result.steps]
    assert elapsed == sorted(elapsed)
    # Same v_b convention as the powered flyby: 1 AU closing speed folded
    # through Earth's well to the surface.
    assert is_nearly_equal(
        result.collision_speed,
        speed_with_escape_energy(result.closing_speed_1au, Earth),
        percent=0.001,
    )
    # Mass accounting: burn + phasing budget through the rocket equation, times
    # the mass ratio at the achieved v_b against the same push as the flyby.
    assert result.phasing_budget == ASSIST_CHAIN_PHASING_BUDGET
    assert result.end_to_end_mass_ratio == pytest.approx(
        result.delivered_fraction * result.payload_puffsat_mass_ratio, rel=1e-9
    )
    assert result.payload_puffsat_mass_ratio == pytest.approx(
        payload_mass_ratio(
            v_rf=lunar_transfer_periapsis_speed(), v_b=result.collision_speed
        ),
        rel=1e-6,
    )
    # The point of the chain: with propellant nearly free, the end-to-end mass
    # ratio beats the powered flyby's optimum.
    assert result.end_to_end_mass_ratio > flyby_optimum.end_to_end_mass_ratio
    # ADR 0004 bend margin: retrograde requires the Jovian arrival excess to
    # exceed Jupiter's own orbital speed (the hard floor VVE cannot clear at
    # low burn), and the bend actually used must fit inside the unpowered
    # turning limit at the ADR 0002 perijove floor -- with room to spare
    # (~96 of ~122 deg at the ~15.4 km/s arrival).
    w_jupiter = result.v_infinity_jupiter.to_value(u.km / u.s)
    assert w_jupiter > flyby_params.v_jupiter_orbit
    ecc = 1.0 + flyby_params.periapsis_floor * w_jupiter**2 / flyby_params.mu_jupiter
    bend_limit_deg = 2.0 * np.degrees(np.arcsin(1.0 / ecc))
    bend_used_deg = abs(result.jovian_bend_angle.to_value(u.deg))
    assert bend_used_deg <= bend_limit_deg
    assert bend_limit_deg - bend_used_deg > 10.0  # not scraping the limit


def test_bend_limit_matches_boinor_flyby(flyby_params) -> None:
    # The turning formula (e = 1 + r_p*w^2/mu, delta = 2*arcsin(1/e)) carries a
    # lot of weight: _jovian_terminal bounds its bend with it, the phased leg
    # solver bounds its root search with it, _flyby_mismatch_burn prices nodes
    # with it, and ADR 0006's 56.27 km/s Tisserand ceiling and 122.5 deg bend
    # limit both follow from it. Check it against an implementation we did not
    # write, at the excesses the chain actually reaches.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    cases = [
        (venus, 3.349),
        (venus, 11.114),
        (earth, 5.210),
        (earth, 17.842),
    ]
    for body, excess in cases:
        v_body = np.array([0.0, body.v_circ, 0.0]) * (u.km / u.s)
        v_sc = np.array([excess, body.v_circ, 0.0]) * (u.km / u.s)
        v_out, delta = compute_flyby(
            v_sc, v_body, body.mu * u.km**3 / u.s**2, body.min_periapsis * u.km
        )
        mine = 2.0 * np.degrees(
            np.arcsin(1.0 / (1.0 + body.min_periapsis * excess**2 / body.mu))
        )
        assert delta.to_value(u.deg) == pytest.approx(mine, abs=1e-9)
        # And the flyby is unpowered: it rotates the excess without rescaling
        # it. That is the Tisserand invariant the whole assist chain rests on.
        excess_out = np.linalg.norm((v_out - v_body).to_value(u.km / u.s))
        assert excess_out == pytest.approx(excess, abs=1e-9)
    # The same formula at the Jovian arrival gives ADR 0006's bend limit.
    jovian = 2.0 * np.degrees(
        np.arcsin(
            1.0
            / (
                1.0
                + flyby_params.periapsis_floor * 15.3685**2 / flyby_params.mu_jupiter
            )
        )
    )
    v_body = np.array([0.0, flyby_params.v_jupiter_orbit, 0.0]) * (u.km / u.s)
    v_sc = np.array([15.3685, flyby_params.v_jupiter_orbit, 0.0]) * (u.km / u.s)
    _, delta = compute_flyby(
        v_sc,
        v_body,
        flyby_params.mu_jupiter * u.km**3 / u.s**2,
        flyby_params.periapsis_floor * u.km,
    )
    assert delta.to_value(u.deg) == pytest.approx(jovian, abs=1e-9)
    assert jovian == pytest.approx(122.48, abs=0.01)


def test_lambert_reproduces_the_conic_propagator(flyby_params) -> None:
    # The phased ladder prices legs with Lambert while the phasing-free chain
    # propagates them as conics. Both must describe the same arc, or the price
    # is of a different trajectory than the one being priced. Generate a leg by
    # propagation, then Lambert between its endpoints: the velocities must
    # match. This also pins the swept angle and the (tangential, radial-out)
    # to Cartesian convention that _body_state encodes.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    excess = 2.5875  # the 300 m/s chain's departure
    v_t0, v_r0 = earth.v_circ - excess, 0.0  # aim -180: brake to fall inward
    crossings = _conic_radius_crossings(
        params.flyby.mu_sun, earth.orbit_radius, v_t0, v_r0, venus.orbit_radius
    )
    assert crossings
    for leg_tof, v_t1, v_r1, _, swept in crossings:
        r0, _ = _body_state(earth.orbit_radius, 0.0, earth.v_circ)
        r1, _ = _body_state(venus.orbit_radius, swept, venus.v_circ)
        depart = np.array([-v_t0 * np.sin(0.0), v_t0 * np.cos(0.0), 0.0]) + np.array(
            [v_r0 * np.cos(0.0), v_r0 * np.sin(0.0), 0.0]
        )
        arrive = np.array(
            [-v_t1 * np.sin(swept), v_t1 * np.cos(swept), 0.0]
        ) + np.array([v_r1 * np.cos(swept), v_r1 * np.sin(swept), 0.0])
        lambert_depart, lambert_arrive = izzo(
            params.flyby.mu_sun, r0, r1, leg_tof, 0, True, True, 35, 1e-8
        )
        assert np.linalg.norm(lambert_depart - depart) < 1e-6
        assert np.linalg.norm(lambert_arrive - arrive) < 1e-6


def test_phased_ladder_burn_is_free_when_the_planets_cooperate() -> None:
    # The pricing model's calibration: hand it planets placed exactly where the
    # phasing-free chain imagines them, and it must charge nothing beyond the
    # departure -- that chain is perfectly phased for its own invented
    # ephemeris, and every turn it uses is inside the bend limit. Any node cost
    # here would mean the model charges for phasing that is already satisfied.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    excess = 2.5875
    leg1 = [
        c
        for c in _conic_radius_crossings(
            params.flyby.mu_sun,
            earth.orbit_radius,
            earth.v_circ - excess,
            0.0,
            venus.orbit_radius,
        )
        if not c[3]
    ]
    assert leg1
    tof1, v_t1, v_r1, _, swept1 = leg1[0]
    # Rotate at Venus by the real chain's recorded bend, then run on to Earth.
    ex_t, ex_r = v_t1 - venus.v_circ, v_r1
    w2 = float(np.hypot(ex_t, ex_r))
    angle = float(np.arctan2(ex_r, ex_t)) + np.radians(44.08122687)
    leg2 = [
        c
        for c in _conic_radius_crossings(
            params.flyby.mu_sun,
            venus.orbit_radius,
            venus.v_circ + w2 * np.cos(angle),
            w2 * np.sin(angle),
            earth.orbit_radius,
        )
        if c[3]
    ]
    assert leg2
    tof2, _, _, _, swept2 = leg2[0]
    # Place each planet exactly at its leg's arrival point.
    longitudes = [
        0.0,
        swept1 - _mean_motion(venus) * tof1,
        (swept1 + swept2) - _mean_motion(earth) * (tof1 + tof2),
    ]
    priced = _phased_ladder_burn(
        0.0, [tof1, tof2], (earth, venus, earth), longitudes, params
    )
    assert priced is not None
    # The departure is the real chain's 300 m/s, recovered through Lambert.
    assert priced.departure_burn == pytest.approx(0.300, abs=1e-4)
    # And the flyby needs no help at all.
    assert priced.node_total < 1e-6
    assert all(burn < 1e-6 for burn in priced.node_burns)


def test_powered_and_retired_nodes_agree_on_a_pure_rotation() -> None:
    # The two node models disagree about what a speed change costs, but a node
    # that only *rotates* the excess within the bend limit is free under both.
    # Any daylight here would mean one of them charges for geometry the flyby
    # supplies for nothing, which would make the ADR 0007 re-price unreadable.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, _, _ = params.bodies
    excess_in = np.array([4.0, 0.0, 0.0])
    for turn_deg in (0.0, 10.0, 30.0):
        turn = np.radians(turn_deg)
        excess_out = 4.0 * np.array([np.cos(turn), np.sin(turn), 0.0])
        powered, _ = _powered_node_burn(excess_in, excess_out, venus)
        retired = _flyby_mismatch_burn(excess_in, excess_out, venus)
        assert powered < 1e-9
        assert retired < 1e-9


def test_powered_node_burn_gets_no_oberth_discount_for_a_zero_turn() -> None:
    # A node that needs no turn has no reason to go anywhere near the planet:
    # the bend is strictly positive at every finite periapsis, so zero turn is
    # only approached as r_p -> infinity, where the Oberth discount vanishes and
    # the speed change costs its full value at infinity. Charging burn(floor)
    # here would award the *deepest* discount to the turn that earned none --
    # a free 2x that an optimizer will drive the whole chain into.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    for body in (venus, earth):
        excess_in = np.array([5.0, 0.0, 0.0])
        excess_out = np.array([7.0, 0.0, 0.0])
        powered, periapsis = _powered_node_burn(excess_in, excess_out, body)
        # The whole 2 km/s, not the ~1 km/s a grazing pass would cost.
        assert powered == pytest.approx(2.0, abs=1e-6)
        assert periapsis > 1e6 * body.min_periapsis
    # And the discount must arrive smoothly as the turn grows, not in a jump:
    # more turn forces a lower periapsis, which deepens the well.
    previous = 2.0
    for turn_deg in (10.0, 20.0, 40.0, 60.0):
        turn = np.radians(turn_deg)
        excess_out = 7.0 * np.array([np.cos(turn), np.sin(turn), 0.0])
        burn, periapsis = _powered_node_burn(
            np.array([5.0, 0.0, 0.0]), excess_out, venus
        )
        assert burn < previous
        previous = burn


def test_phased_leg_solves_onto_the_moving_target() -> None:
    # The phased ladder's primitive: solve the flyby rotation so the leg lands
    # where the planet actually is, rather than sampling the rotation and
    # letting the planet be wherever the leg lands (the phasing-free model).
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    n_venus = _mean_motion(venus)
    # A departure aims its excess for free -- no flyby, so no bend limit.
    solutions = []
    for outbound in (False, True):
        solutions += _phased_leg_rotations(
            earth,
            0.0,
            2.5875,
            0.0,
            venus,
            np.radians(300.0),
            outbound,
            params,
            samples=721,
            rotation_limit=np.pi,
        )
    assert solutions
    # Every solved rotation must actually put the craft on Venus: recompute the
    # arrival longitude independently and check Venus is there at that time.
    for rotation, leg_tof, _, _ in solutions:
        assert abs(rotation) <= np.pi
        assert leg_tof > 0.0
        arrival_is = np.radians(300.0) + n_venus * leg_tof
        v_t0 = earth.v_circ + 2.5875 * np.cos(rotation)
        v_r0 = 2.5875 * np.sin(rotation)
        swept = [
            c[4]
            for c in _conic_radius_crossings(
                params.flyby.mu_sun, earth.orbit_radius, v_t0, v_r0, venus.orbit_radius
            )
            if abs(c[0] - leg_tof) < 1.0
        ]
        assert swept, "solved leg did not replay"
        assert _wrap_pi(swept[0] - arrival_is) == pytest.approx(0.0, abs=1e-6)


def test_phased_leg_windows_recur_at_the_earth_venus_synodic() -> None:
    # The check that separates a working phasing model from a plausible one:
    # the launch-window cadence must *emerge*, never be supplied. Advancing the
    # launch epoch by one Earth-Venus synodic period restores the same relative
    # geometry (Venus gains exactly one lap on Earth), so the identical
    # rotations must solve. An earlier draft advanced the target from t=0
    # instead of from the flyby, double-counting the epoch and inventing a
    # spurious 162 d cadence -- this pins the real one.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    n_earth, n_venus = _mean_motion(earth), _mean_motion(venus)
    synodic = 2.0 * np.pi / abs(n_venus - n_earth)
    assert is_nearly_equal((synodic * u.s).to(u.day), 583.9 * u.day, percent=0.1)

    def solve_at(epoch: float):
        out = []
        for outbound in (False, True):
            out += _phased_leg_rotations(
                earth,
                n_earth * epoch,
                2.5875,
                0.0,
                venus,
                n_venus * epoch,
                outbound,
                params,
                samples=721,
                rotation_limit=np.pi,
            )
        return sorted(r for r, _, _, _ in out)

    inside_window = 496.0 * 86400.0
    now = solve_at(inside_window)
    assert now, "the measured E-V window near day 496 should admit a phased leg"
    later = solve_at(inside_window + synodic)
    assert len(later) == len(now)
    assert later == pytest.approx(now, abs=1e-6)
    # And the geometry is genuinely a window, not a permanent state: half a
    # synodic period later the relative geometry is opposite and it shuts.
    assert not solve_at(inside_window + synodic / 2.0)


def test_jovian_return_phasing_envelope_closes(chain_at_300):
    # ADR 0006: retuning the unpowered Jovian bend walks the Earth arrival
    # across more than a full Earth orbit, so every launch phase admits a
    # phased intercept with no propellant and no powered flyby.
    env = jovian_return_phasing_envelope(chain_at_300)
    assert env.closes
    assert env.wraps > 3.0  # ~3.09: coverage ~1113 deg
    assert env.earth_phase_coverage > 360 * u.deg
    # It still closes under the powered flyby's own 7 yr cap, and under a 5 yr
    # one -- the cap trims the slow outbound branch but not past one wrap.
    assert jovian_return_phasing_envelope(
        chain_at_300, max_total_time=JUPITER_FLYBY_MAX_TOF
    ).closes
    capped = jovian_return_phasing_envelope(chain_at_300, max_total_time=5.0 * u.year)
    assert capped.closes
    assert capped.wraps > 1.0  # ~1.58
    # The chain's own return is inside the reachable span, and the span is a
    # superset of it -- the search just keeps the earliest arrival.
    assert env.return_time_min <= chain_at_300.return_time <= env.return_time_max


def test_jovian_return_phasing_span_is_connected(chain_at_300):
    # The span is only authority if it has no holes. Quoting max-minus-min over
    # a disconnected set would count an unreachable hole as coverage, so pin
    # the evidence of connectivity: the largest step between sampled arrivals
    # is resolution-limited and falls as 1/samples. A real hole would plateau.
    gaps = [
        jovian_return_phasing_envelope(chain_at_300, samples=n).largest_gap.to_value(
            u.day
        )
        for n in (1000, 4000, 16000)
    ]
    assert gaps[0] > gaps[1] > gaps[2]
    # Each 4x refinement should shrink the gap ~4x; allow slack for where on
    # the curve the widest step lands.
    for coarse, fine in zip(gaps, gaps[1:]):
        assert 2.0 < coarse / fine < 8.0
    assert gaps[-1] < 1.0  # sub-day at 16000 samples


def test_jovian_return_phasing_cannot_reach_the_catalog_69_km_s(
    chain_at_300, flyby_params
):
    # Phasing is free but v_b is not a choice: the bend that phases Earth
    # dictates v_b, and its ceiling is full reversal of the Tisserand-fixed
    # excess -- no timing or periapsis reaches the catalog's 69.27 km/s.
    env = jovian_return_phasing_envelope(chain_at_300)
    ceiling = env.collision_speed_max
    assert ceiling < retrograde_jovian_hohmann_transfer()
    assert is_nearly_equal(ceiling, 56.27 * u.km / u.s, percent=0.1)
    # The ceiling IS the full-reversal state: excess exactly anti-parallel to
    # Jupiter's motion, i.e. v_r = 0 and v_t = v_jupiter - v_inf.
    w = env.v_infinity.to_value(u.km / u.s)
    reversed_leg = _flyby_return_leg(
        flyby_params.v_jupiter_orbit - w, 0.0, flyby_params
    )
    assert reversed_leg is not None
    assert is_nearly_equal(
        reversed_leg.collision_speed * u.km / u.s, ceiling, percent=0.01
    )
    # Every v_b phasing can force on you is an acceptable loop, which is the
    # only reason the lottery is benign.
    assert env.collision_speed_min > lunar_transfer_periapsis_speed()


@pytest.mark.slow
def test_minimum_departure_burn_assist_chain(flyby_optimum):
    # The default probe grid brackets the Venus-reach floor: 250 and 280 m/s
    # find no chain (280 clears the floor but is Tisserand-locked too tightly
    # for the beam), while 290 m/s closes -- an order of magnitude below the
    # powered flyby's departure burn.
    scan = minimum_departure_burn_assist_chain(
        target_collision_speed=flyby_optimum.collision_speed
    )
    assert scan.minimum is not None
    assert is_nearly_equal(
        scan.minimum.departure_burn, 0.290 * u.km / u.s, percent=0.001
    )
    infeasible = [burn.to_value(u.km / u.s) for burn in scan.infeasible_burns]
    assert infeasible == [0.250, 0.280]
    assert scan.minimum.total_time <= ASSIST_CHAIN_MAX_TRIP_TIME
    assert scan.minimum.collision_speed >= scan.target_collision_speed * 0.9999
    assert scan.minimum.departure_burn < flyby_optimum.departure_burn / 10.0


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


def test_powered_jovian_flyby_burn_changes_the_turn_not_just_the_speed() -> None:
    # The property that makes a powered flyby two knobs instead of one: the burn
    # fires at perijove, so the OUTGOING hyperbola has its own eccentricity and
    # the total bend is asin(1/e_in) + asin(1/e_out). Speeding up raises e_out
    # and so COSTS bend; slowing down buys it. Charging the burn against the
    # incoming eccentricity twice -- 2*asin(1/e_in), as the retired node model
    # did -- overstates what the flyby can point at.
    params = _assist_chain_params(target_collision_speed=51.134)
    flyby = params.flyby
    r_p = flyby.periapsis_floor
    w_in = 8.0
    mu_j = flyby.mu_jupiter
    ecc_in = 1.0 + r_p * w_in**2 / mu_j
    unpowered_turn = 2.0 * np.arcsin(1.0 / ecc_in)

    def turn_for(burn: float) -> float:
        v_peri_in = np.sqrt(w_in**2 + 2.0 * mu_j / r_p)
        w_out = np.sqrt((v_peri_in + burn) ** 2 - 2.0 * mu_j / r_p)
        ecc_out = 1.0 + r_p * w_out**2 / mu_j
        return float(np.arcsin(1.0 / ecc_in) + np.arcsin(1.0 / ecc_out))

    # Zero burn recovers the unpowered bend exactly.
    assert turn_for(0.0) == pytest.approx(unpowered_turn, rel=1e-12)
    # Speeding up costs bend, monotonically -- the turn is NOT independent of dv.
    speeding = [turn_for(b) for b in (0.0, 1.0, 2.0, 4.0)]
    assert speeding == sorted(speeding, reverse=True)
    assert speeding[-1] < speeding[0]
    # Slowing down buys bend back -- but only just. Perijove speed clears
    # Jupiter escape (57.94 km/s at the floor) by 0.55 km/s at this w_in, so a
    # retro burn past that CAPTURES the craft rather than bending it. "Slow down
    # for more turn" is barely available at Jupiter, which is why the useful
    # powered flyby here speeds up and pays for it in bend.
    assert turn_for(-0.3) > unpowered_turn
    v_esc_jupiter = float(np.sqrt(2.0 * mu_j / r_p))
    v_peri_in = float(np.sqrt(w_in**2 + 2.0 * mu_j / r_p))
    assert v_peri_in - v_esc_jupiter < 0.6
    assert np.isnan(turn_for(-1.0))


def test_powered_jovian_terminal_breaks_the_vb_lottery() -> None:
    # An unpowered Jupiter has ONE knob: perijove radius IS the bend. Demanding
    # both a v_b and an arrival time over-determines it. A powered flyby has two
    # (r_p, dv), so it can hold v_b while the geometry varies. Show the second
    # knob doing real work: at a FIXED perijove, the burn moves v_b.
    params = _assist_chain_params(target_collision_speed=51.134)
    earth = params.bodies[params.earth_index]
    v_t0, v_r0 = earth.v_circ + 10.0, 0.0
    r_p = params.flyby.periapsis_floor

    def reach(burn: float):
        return _powered_jovian_terminal(
            v_t0, v_r0, earth.orbit_radius, 0.0, r_p, burn, 1.0, params
        )

    # The lottery: unpowered, this arrival cannot make the target v_b AT ALL.
    # One knob (perijove radius is the bend) against two demands.
    assert reach(0.0) is None
    # Powered, the same arrival makes it comfortably -- the burn is the second
    # knob, and it is what buys the return, not a marginal saving.
    hot = reach(4.0)
    hotter = reach(6.0)
    assert hot is not None and hotter is not None
    assert hot.return_leg.collision_speed == pytest.approx(55.872, rel=1e-3)
    assert hotter.return_leg.collision_speed == pytest.approx(69.424, rel=1e-3)
    # Perijove Oberth is why: ~7 km/s of v_b per km/s of burn.
    leverage = (
        hotter.return_leg.collision_speed - hot.return_leg.collision_speed
    ) / 2.0
    assert leverage > 5.0
    # Speeding up costs bend, so the turn must shrink as the burn grows.
    assert hotter.turn_angle < hot.turn_angle
    assert hotter.v_infinity_out > hot.v_infinity_out > hot.v_infinity_in


def test_phased_ladder_lands_the_jovian_leg_on_jupiter_itself() -> None:
    # The model's largest hole: _jovian_terminal reaches Jupiter through
    # _conic_radius_crossings, which finds crossings of Jupiter's orbit RADIUS
    # and assumes Jupiter is there. Handing Jupiter to _phased_ladder_burn as an
    # ordinary body closes it -- the Jovian leg becomes a Lambert arc onto
    # Jupiter's true position. Verify Jupiter is actually THERE on arrival.
    params = _assist_chain_params(target_collision_speed=51.134)
    venus, earth, _ = params.bodies
    jupiter = _jupiter_assist_body(params)
    assert jupiter.symbol == "J"
    assert jupiter.min_periapsis == params.flyby.periapsis_floor

    year = 365.25 * 86400.0
    ladder = (earth, venus, earth, jupiter)
    # Jupiter placed so it IS where a 2.0 yr final leg arrives.
    longitudes0 = [0.0, 1.0, 2.0, 2.7]
    legs = [0.4 * year, 0.9 * year, 2.0 * year]
    priced = _phased_ladder_burn(0.0, legs, ladder, longitudes0, params, True)
    assert priced is not None

    # Jupiter's longitude at arrival, computed independently of the pricing.
    arrival = sum(legs)
    expected_lon = longitudes0[3] + _mean_motion(jupiter) * arrival
    assert priced.arrival_longitude == pytest.approx(expected_lon, rel=1e-12)
    # And the arrival excess is Jupiter-relative, so it must be modest -- if the
    # leg were merely crossing 5.2 AU with Jupiter elsewhere, this would be the
    # raw heliocentric speed instead.
    assert 0.0 < priced.arrival_excess < 40.0
    # The last Earth is now a charged node, not a free rotation: two nodes for a
    # four-body ladder (Venus and the returning Earth).
    assert len(priced.node_burns) == 2


def test_phased_jovian_flyby_bends_a_known_arrival_into_a_return() -> None:
    # With the leg phased, the flyby needs no radius search: it bends and
    # rescales a known arrival excess. A zero burn must leave the excess SPEED
    # untouched (an unpowered flyby only rotates), and the burn must raise it.
    params = _assist_chain_params(target_collision_speed=20.0)
    r_p = params.flyby.periapsis_floor
    lon = 0.7
    # A retrograde return requires the post-flyby heliocentric tangential speed
    # to go NEGATIVE, i.e. v_jupiter + w_out*cos(angle) < 0. No bend angle can
    # do that unless w_out exceeds Jupiter's own orbital speed -- you cannot
    # rotate a 6 km/s excess into cancelling 13.058 km/s of Jupiter's motion.
    # So arrival v_inf > 13.058 km/s is a NECESSARY condition for an unpowered
    # Jovian flyby, and the only way round it is to burn.
    assert params.flyby.v_jupiter_orbit == pytest.approx(13.058, abs=0.01)
    cold = np.array([-5.0, 4.0, 0.0])  # w = 6.40, far under the threshold
    assert float(np.linalg.norm(cold)) < params.flyby.v_jupiter_orbit
    assert _phased_jovian_flyby(cold, lon, r_p, 0.0, 1.0, params) is None
    # ... but a burn lifts w_out over the threshold and the return appears.
    assert _phased_jovian_flyby(cold, lon, r_p, 3.0, 1.0, params) is not None

    # A hot enough arrival returns unpowered, as ADR 0002's direct flyby does
    # (it reaches Jupiter at w_in ~ 15.1 km/s and spends no Jovian burn).
    excess_in = np.array([-14.0, 6.0, 0.0])
    assert float(np.linalg.norm(excess_in)) > params.flyby.v_jupiter_orbit
    unpowered = _phased_jovian_flyby(excess_in, lon, r_p, 0.0, 1.0, params)
    powered = _phased_jovian_flyby(excess_in, lon, r_p, 3.0, 1.0, params)
    assert unpowered is not None and powered is not None
    # Powering the flyby buys a hotter return.
    assert powered.collision_speed > unpowered.collision_speed
    # Below the perijove floor is refused outright.
    assert _phased_jovian_flyby(excess_in, lon, r_p * 0.5, 0.0, 1.0, params) is None
    # A retro burn past the escape margin captures rather than bends.
    w_in = float(np.linalg.norm(excess_in))
    margin = np.sqrt(w_in**2 + 2 * params.flyby.mu_jupiter / r_p) - np.sqrt(
        2 * params.flyby.mu_jupiter / r_p
    )
    assert (
        _phased_jovian_flyby(excess_in, lon, r_p, -(margin + 0.5), 1.0, params) is None
    )


def test_return_leg_sweep_angle_approaches_a_hohmann_half_turn() -> None:
    # The retrograde Jupiter->Earth Hohmann is the one return whose geometry is
    # known in closed form: it leaves Jupiter's radius at aphelion and arrives at
    # 1 AU at perihelion, so it sweeps exactly half a turn in half a period.
    #
    # The exact Hohmann cannot be evaluated here, and that is a property of the
    # problem rather than a defect: its perihelion lands 4 ulp ABOVE 1 AU (0.12 mm
    # on 1.5e8 km), so the leg correctly reports no crossing. The tangent case has
    # measure zero. Approach the limit from inside instead, which pins the branch,
    # the sign convention AND continuity into the closed-form answer.
    params = _powered_flyby_params(max_total_tof=10 * u.year)
    mu = params.mu_sun
    r_j, r_e = params.r_jupiter_orbit, params.r_earth_orbit

    def leg_with_perihelion(q: float) -> object:
        # Aphelion at Jupiter's radius, perihelion at q, flown retrograde. An
        # aphelion state is purely tangential by definition, so v_r = 0.
        a = 0.5 * (r_j + q)
        v_aph = float(np.sqrt(mu * (2.0 / r_j - 1.0 / a)))
        return _flyby_return_leg(-v_aph, 0.0, params)

    sweeps = []
    for shortfall in (1e-4, 1e-6, 1e-8):
        leg = leg_with_perihelion(r_e * (1.0 - shortfall))
        assert leg is not None
        # Crossing 1 AU strictly before perihelion, so strictly under a half turn.
        assert leg.sweep_angle < np.pi
        sweeps.append(leg.sweep_angle)
    # Monotone, and converging on the half turn.
    assert sweeps[0] < sweeps[1] < sweeps[2]
    assert sweeps[2] == pytest.approx(np.pi, abs=1e-3)

    # At the closest approach to the limit, tof must match half the transfer
    # ellipse's period -- the independent check that sweep and tof describe the
    # SAME arc, since they are derived in the same branch from the same anomalies.
    q = r_e * (1.0 - 1e-8)
    leg = leg_with_perihelion(q)
    assert leg is not None
    a = 0.5 * (r_j + q)
    period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
    # Strictly under half a period, and for the same reason the sweep is strictly
    # under a half turn: 1 AU is crossed just BEFORE perihelion. Converging to the
    # half-period limit from below is the check; landing exactly on it would mean
    # the branch had silently run past perihelion.
    assert leg.tof < 0.5 * period
    assert leg.tof == pytest.approx(0.5 * period, rel=1e-4)
    assert leg.perihelion == pytest.approx(q, rel=1e-9)


def test_return_leg_sweeps_further_when_it_climbs_before_falling() -> None:
    # An outbound state (v_r > 0) must go out to aphelion before coming back in,
    # so it sweeps MORE than the same-energy inbound state, and takes longer. The
    # two branches are separately derived in _flyby_return_leg; this pins that
    # they stay ordered.
    params = _powered_flyby_params(max_total_tof=10 * u.year)
    inbound = _flyby_return_leg(-6.0, -2.0, params)
    outbound = _flyby_return_leg(-6.0, 2.0, params)
    assert inbound is not None and outbound is not None
    assert outbound.sweep_angle > inbound.sweep_angle
    assert outbound.tof > inbound.tof
    # Same speed and radius, so same energy -> same conic, hence same perihelion
    # and the same arrival speed. Only the arc flown differs.
    assert outbound.perihelion == pytest.approx(inbound.perihelion, rel=1e-9)
    assert outbound.collision_speed == pytest.approx(inbound.collision_speed, rel=1e-9)
    # Every sweep is a real angle in (0, 2*pi).
    for leg in (inbound, outbound):
        assert 0.0 < leg.sweep_angle < 2.0 * np.pi


def test_earth_phase_mismatch_vanishes_exactly_where_earth_is() -> None:
    # The mismatch is the constraint the whole "real doubling time" question
    # turns on, so it must be zero ONLY when Earth is truly at the crossing.
    params = _powered_flyby_params(max_total_tof=10 * u.year)
    leg = _flyby_return_leg(-6.0, -2.0, params)
    assert leg is not None
    n_earth = params.v_earth_orbit / params.r_earth_orbit
    jupiter_lon, flyby_time = 1.3, 0.0

    # Place Earth at t=0 exactly where the crossing will be at arrival, running
    # Earth's motion backwards over the flight. By construction the mismatch is 0.
    crossing = jupiter_lon - leg.sweep_angle
    earth_0 = crossing - n_earth * (flyby_time + leg.tof)
    assert _earth_phase_mismatch(
        leg, jupiter_lon, flyby_time, earth_0, params
    ) == pytest.approx(0.0, abs=1e-12)

    # Nudge Earth and the mismatch tracks it one-for-one...
    assert _earth_phase_mismatch(
        leg, jupiter_lon, flyby_time, earth_0 + 0.25, params
    ) == pytest.approx(-0.25, abs=1e-12)
    # ... and is wrapped, so a half-turn error never reports as a near miss.
    assert _earth_phase_mismatch(
        leg, jupiter_lon, flyby_time, earth_0 + np.pi, params
    ) == pytest.approx(-np.pi, abs=1e-9)
    # A whole-turn offset IS the same place: Earth is a body on a ring, not a
    # point on a line.
    assert _earth_phase_mismatch(
        leg, jupiter_lon, flyby_time, earth_0 + 2.0 * np.pi, params
    ) == pytest.approx(0.0, abs=1e-9)
