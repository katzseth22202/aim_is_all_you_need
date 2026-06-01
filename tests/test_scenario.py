"""Tests for scenario-related functions in scenario.py."""

import numpy as np
import pytest
from astropy import units as u

from src.astro_constants import CERES_A, EARTH_A, MARS_A, SATURN_A, VENUS_A
from src.propulsion import payload_mass_ratio
from src.scenario import (
    SCENARIO_COLUMNS,
    PuffSatScenario,
    lunar_return_transfer_dv,
    paper_scenarios,
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
