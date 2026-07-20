"""Tests for src/scenario_catalog.py."""

import numpy as np
import pytest
from astropy import units as u

from src.apoapsis_raise_reintercept import apoapsis_raise_reintercept
from src.astro_constants import CERES_A, EARTH_A, MARS_A, SATURN_A, VENUS_A
from src.heliocentric_reintercept import single_impulse_resonant_dive
from src.propulsion import payload_mass_ratio
from src.scenario_catalog import (
    SCENARIO_COLUMNS,
    PuffSatScenario,
    earth_reentry_disposal_dv,
    earth_reintercept_scenarios,
    find_best_lunar_return,
    lunar_impact_disposal_dv,
    lunar_return_transfer_dv,
    near_escape_disposal_dv,
    paper_scenarios,
    parker_injection_burns,
    parker_rows_rescored_at,
    scenarios_to_dataframe,
    solar_impact_dv,
    suborbital_200km_propellant_fraction,
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


def test_near_escape_disposal_reentry_vs_lunar() -> None:
    # Paper sec:coordinator_node_dry_mass_disposal: from a just-past-escape
    # turnaround near Earth's gravitational reach (~900,000 km, where the package
    # crawls at ~0.94 km/s), the retrograde burn to graze the atmosphere is
    # ~0.86 km/s while the burn to fall only to lunar distance is ~0.43 km/s.
    reentry_dv = earth_reentry_disposal_dv()
    lunar_dv = lunar_impact_disposal_dv()
    # The headline claim: hitting the Moon costs less delta-v than reentering.
    assert lunar_dv < reentry_dv
    # Within 1% of the values computed from the repo's primitives.
    assert is_nearly_equal(reentry_dv, 0.8615 * u.km / u.s, percent=0.01)
    assert is_nearly_equal(lunar_dv, 0.4263 * u.km / u.s, percent=0.01)


def test_lunar_impact_redirect_stays_below_reentry() -> None:
    # Folding a 20-30 deg plane-change redirect into the retrograde burn (to match
    # the Moon's orbital plane) raises the lunar-impact cost to ~0.49-0.56 km/s,
    # still below the ~0.86 km/s reentry burn. Zero redirect recovers the coplanar
    # perigee-lowering delta-v exactly.
    coplanar = lunar_impact_disposal_dv()
    assert lunar_impact_disposal_dv(0 * u.deg) == coplanar
    dv_20 = lunar_impact_disposal_dv(20 * u.deg)
    dv_30 = lunar_impact_disposal_dv(30 * u.deg)
    assert coplanar < dv_20 < dv_30 < earth_reentry_disposal_dv()
    assert is_nearly_equal(dv_20, 0.4901 * u.km / u.s, percent=0.01)
    assert is_nearly_equal(dv_30, 0.5582 * u.km / u.s, percent=0.01)


def test_near_escape_disposal_deeper_perigee_costs_more() -> None:
    # near_escape_disposal_dv is monotone in how far it drops perigee: a lower
    # target perigee means a slower apoapsis speed on the transfer ellipse, so a
    # larger delta-v. This is why reentry (atmosphere perigee) beats the Moon.
    from src.astro_constants import MOON_A, REENTRY_PERIGEE_RADIUS

    assert near_escape_disposal_dv(REENTRY_PERIGEE_RADIUS) > near_escape_disposal_dv(
        MOON_A
    )
    # earth_reentry_disposal_dv delegates straight to near_escape_disposal_dv.
    assert (
        near_escape_disposal_dv(REENTRY_PERIGEE_RADIUS) == earth_reentry_disposal_dv()
    )
    # The zero-redirect lunar burn matches the plain perigee-lowering delta-v (its
    # vector-difference form reduces to the same magnitude, up to rounding).
    assert is_nearly_equal(
        near_escape_disposal_dv(MOON_A), lunar_impact_disposal_dv(), percent=1e-9
    )


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
    #
    # Indices 3-5 (v_b = retrograde_jovian_hohmann_transfer()) were corrected
    # here: the old value baked in a boinor Orbit.circular altitude/radius
    # bug that inflated Jupiter's orbital radius by Sun.R (~695,700 km), which
    # overstated v_b by ~2.7 m/s (~0.004%) -- see
    # test_propulsion.test_retrograde_jovian_hohmann_transfer.
    expected = [
        1.28130083,
        2.30831207,
        2.97219375,
        3.82932510,
        1.84506818,
        9.33071486,
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


def test_parker_rows_rescored_at_matches_payload_mass_ratio() -> None:
    # parker_rows_rescored_at is the named home for a derivation main.py used
    # to compute inline: re-score the two Parker injection burns' mass ratios
    # against an arbitrary v_b, e.g. the powered Jovian flyby's achieved
    # collision speed rather than the retrograde-Hohmann one paper_scenarios
    # uses.
    prograde_burn, retrograde_burn = parker_injection_burns()
    v_b = 50 * u.km / u.s
    prograde_ratio, retrograde_ratio = parker_rows_rescored_at(v_b)
    assert prograde_ratio == pytest.approx(
        payload_mass_ratio(v_rf=prograde_burn, v_b=v_b)
    )
    assert retrograde_ratio == pytest.approx(
        payload_mass_ratio(v_rf=retrograde_burn, v_b=v_b)
    )
    # A higher v_b (more collision energy to work with) gives a better ratio.
    hotter_prograde, hotter_retrograde = parker_rows_rescored_at(60 * u.km / u.s)
    assert hotter_prograde > prograde_ratio
    assert hotter_retrograde > retrograde_ratio


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
    # Pin the phased mass ratio (captured from the repo's primitives). The
    # energy-coupled solar-dive solve raises the required Earth boost slightly
    # and therefore lowers this ratio from the former 2.05029641.
    assert float(phased.mass_ratio) == pytest.approx(2.04924969, rel=1e-6)
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
