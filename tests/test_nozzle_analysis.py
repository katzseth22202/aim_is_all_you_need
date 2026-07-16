"""Pinned tests for src/nozzle_analysis.py (ADR 0009).

The pricing algebra is pinned with the phased optimum's values passed
explicitly, so those tests stay fast; the orbit-geometry pipeline (Lambert leg,
bend sweeps, powered split) is exercised in tests marked ``slow``.
"""

import numpy as np
import pytest
from astropy import units as u

from src.nozzle_analysis import (
    aim_geometry,
    apoapsis_reversal_dv,
    corrected_incumbent,
    parked_nozzle,
    phased_geometry,
    powered_split,
    same_cycle_nozzle,
    two_wave_unpowered_roots,
)

# The Earth-phased optimum (ADR 0008 "REAL" row), pinned as explicit inputs.
V_B = 59.7649
DV_DEP = 5.3751
CYCLE = 3.276
VE_METHALOX = 3.7265


def test_apoapsis_reversal_dv_20_day_orbit():
    dv = apoapsis_reversal_dv(20 * u.day).to_value(u.m / u.s)
    assert dv == pytest.approx(233.9, abs=0.5)


def test_corrected_incumbent_pins_adr_0009():
    growth, doubling = corrected_incumbent(V_B, DV_DEP, CYCLE, VE_METHALOX)
    assert growth == pytest.approx(1.7549, abs=2e-3)
    assert doubling == pytest.approx(4.038, abs=5e-3)


def test_parked_nozzle_derated_pins_adr_0009():
    pricing = parked_nozzle(V_B, DV_DEP, CYCLE, VE_METHALOX, recovery=0.8)
    assert pricing.slug_ratio == pytest.approx(6.0, abs=0.5)
    assert pricing.growth == pytest.approx(2.137, abs=5e-3)
    assert pricing.doubling == pytest.approx(2.990, abs=0.01)
    # Mass fractions quoted in the ADR discussion.
    assert pricing.wave_to_growth == pytest.approx(0.859, abs=5e-3)
    assert pricing.parked_to_craft == pytest.approx(0.6725, abs=5e-3)
    assert pricing.parked_to_slug == pytest.approx(0.2667, abs=5e-3)
    assert pricing.parked_to_reversal == pytest.approx(0.0608, abs=1e-3)
    assert pricing.delivered_fraction == pytest.approx(0.716, abs=5e-3)


def test_parked_nozzle_ideal_beats_derated_and_both_beat_incumbent():
    ideal = parked_nozzle(V_B, DV_DEP, CYCLE, VE_METHALOX, recovery=1.0)
    derated = parked_nozzle(V_B, DV_DEP, CYCLE, VE_METHALOX, recovery=0.8)
    _, doubling_inc = corrected_incumbent(V_B, DV_DEP, CYCLE, VE_METHALOX)
    assert ideal.doubling < derated.doubling < doubling_inc
    assert ideal.doubling == pytest.approx(2.810, abs=0.01)


def test_parked_nozzle_free_burn_limit_is_sqrt_law():
    # recovery -> huge makes the burn free (sigma -> 0): g -> sqrt(r*M).
    pricing = parked_nozzle(V_B, DV_DEP, CYCLE, VE_METHALOX, recovery=1e9)
    from src.propulsion import payload_mass_ratio

    mass_ratio = float(
        payload_mass_ratio(v_rf=10.9503 * u.km / u.s, v_b=V_B * u.km / u.s)
    )
    rev = np.exp(-apoapsis_reversal_dv().to_value(u.km / u.s) / VE_METHALOX)
    assert pricing.growth == pytest.approx(np.sqrt(rev * mass_ratio), rel=1e-3)


def test_same_cycle_nozzle_derated_pins_adr_0009():
    # The dt = 10 d powered split found by the slow solve: burn 0.3262 km/s,
    # growth-wave v_b 61.4647.
    pricing = same_cycle_nozzle(
        61.4647, 0.3262, V_B, DV_DEP, CYCLE, VE_METHALOX, recovery=0.8
    )
    assert pricing.growth == pytest.approx(3.845, abs=5e-3)
    assert pricing.doubling == pytest.approx(1.686, abs=5e-3)
    assert pricing.wave_to_growth == pytest.approx(0.8105, abs=5e-3)
    assert pricing.parked_to_craft == pytest.approx(0.635, abs=5e-3)


@pytest.mark.slow
def test_phased_geometry_reproduces_adr_0008_real_row():
    geometry = phased_geometry()
    assert geometry.reference.collision_speed == pytest.approx(59.765, abs=0.01)
    assert geometry.departure_burn == pytest.approx(5.3751, abs=5e-3)
    assert abs(geometry.reference.mismatch) < 1e-9
    assert geometry.cycle == pytest.approx(3.276, abs=2e-3)


@pytest.mark.slow
def test_aim_geometry_pins_the_shortfall():
    geometry = phased_geometry()
    aim = aim_geometry(geometry)
    assert aim.separation_deg == pytest.approx(148.0, abs=0.3)
    assert aim.closeable_deg == pytest.approx(18.1, abs=0.3)
    assert aim.shortfall_deg > 125.0
    # Even at the v_b cap (zero delivered mass) the push axis barely grazes
    # Jupiter's orbit radius.
    assert aim.max_aphelion_au == pytest.approx(5.24, abs=0.05)


@pytest.mark.slow
def test_unpowered_roots_are_about_a_year_apart():
    geometry = phased_geometry()
    roots = two_wave_unpowered_roots(geometry, grid_points=1501)
    offsets = sorted(offset for _, _, offset in roots)
    assert len(offsets) >= 4
    gaps = np.diff(offsets)
    assert gaps.min() > 300.0  # days -- no unpowered 5-90 d pair exists
    close = [o for o in offsets if 5.0 <= -o <= 90.0]
    assert not close


@pytest.mark.slow
def test_powered_split_10_day_burn():
    geometry = phased_geometry()
    found = powered_split(geometry, 10.0)
    assert found is not None
    burn, vb1 = found
    assert burn == pytest.approx(0.326, abs=0.02)
    assert vb1 == pytest.approx(61.46, abs=0.1)
