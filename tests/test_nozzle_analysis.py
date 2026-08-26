"""Pinned tests for src/nozzle_analysis.py (ADR 0009).

The pricing algebra is pinned with the phased optimum's values passed
explicitly, so those tests stay fast; the orbit-geometry pipeline (Lambert leg,
bend sweeps, powered split) is exercised in tests marked ``slow``.
"""

import numpy as np
import pytest
from astropy import units as u

from src.nozzle_analysis import (
    _sigma,
    _sigma_overtaking,
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


def test_growth_nozzle_reduces_to_the_plate_as_slug_vanishes() -> None:
    """``k1 -> 0`` reproduces ``payload_mass_ratio`` with recovery as ``f``.

    Formal, not physical: the ignition window's lower root sits well above
    zero, so a real nozzle cannot be run there.
    """
    common = dict(
        growth_collision_speed=59.29,
        growth_wave_burn=0.0,
        nozzle_collision_speed=55.45,
        departure_dv=5.329,
        cycle=3.276,
        exhaust_speed=VE_METHALOX,
        recovery=0.6,
        slug_ratio=8.0,
    )
    plate = same_cycle_nozzle(fudge=0.7, **common)
    nozzle = same_cycle_nozzle(growth_slug_ratio=1e-8, growth_recovery=0.7, **common)
    assert np.isclose(plate.growth, nozzle.growth, rtol=1e-6)
    assert np.isclose(plate.mass_multiplier, nozzle.mass_multiplier, rtol=1e-6)


def test_growth_nozzle_costs_launch_mass_that_the_plate_does_not() -> None:
    """A plate consumes nothing on the push; a nozzle spends slug for it."""
    common = dict(
        growth_collision_speed=59.29,
        growth_wave_burn=0.0,
        nozzle_collision_speed=55.45,
        departure_dv=5.329,
        cycle=3.276,
        exhaust_speed=VE_METHALOX,
        recovery=0.6,
        slug_ratio=8.0,
    )
    plate = same_cycle_nozzle(fudge=0.8, **common)
    nozzle = same_cycle_nozzle(growth_slug_ratio=6.0, growth_recovery=0.8, **common)
    assert plate.growth_sigma == 0.0
    assert nozzle.growth_sigma > 0.0
    assert nozzle.mass_multiplier > plate.mass_multiplier
    assert nozzle.growth > plate.growth


def test_pairing_of_the_growth_nozzle_arguments_is_enforced() -> None:
    """A slug ratio without a recovery is not a specification."""
    with pytest.raises(ValueError):
        same_cycle_nozzle(
            growth_collision_speed=59.29,
            growth_wave_burn=0.0,
            nozzle_collision_speed=55.45,
            departure_dv=5.329,
            cycle=3.276,
            exhaust_speed=VE_METHALOX,
            growth_slug_ratio=4.0,
        )


def test_jet_efficiency_defaults_reproduce_the_published_arithmetic() -> None:
    """``eta_jet = 1`` must leave ADR 0013/0014/0015's numbers bit-identical."""
    assert _sigma(8.5, 0.6, 60.0, 10.0, 15.0) == _sigma(
        8.5, 0.6, 60.0, 10.0, 15.0, jet_efficiency=1.0
    )
    assert _sigma_overtaking(8.5, 0.6, 56.53, 0.0, 10.9) == _sigma_overtaking(
        8.5, 0.6, 56.53, 0.0, 10.9, jet_efficiency=1.0
    )


def test_the_two_efficiencies_are_not_interchangeable() -> None:
    """Inside and outside the momentum debit are different places to charge.

    This is the whole point of the split: ``recovery`` scales the net impulse
    after the ``-1`` bulk-drift debit, ``jet_efficiency`` scales the gross jet
    before it.  If they were interchangeable the frozen-dissociation toll could
    have been charged through ``recovery`` and no change would be owed.
    """
    outside = _sigma(8.5, 0.8, 60.0, 10.0, 15.0)
    inside = _sigma(8.5, 1.0, 60.0, 10.0, 15.0, jet_efficiency=0.8)
    assert not np.isclose(outside, inside)
    assert inside > outside


def test_head_on_leg_has_a_forward_thrust_floor_the_overtake_lacks() -> None:
    """Forward thrust vanishes at ``eta_jet = 1/sqrt(1+k)``, 0.324 at k = 8.5.

    ``recovery`` has no such floor, which is exactly why the chemistry may not
    be charged through it.  The overtake has none either -- its ``+1`` is a
    bonus, not a debit -- so the toll levers the two legs differently.
    """
    floor = 1.0 / np.sqrt(9.5)
    assert np.isclose(floor, 0.3244, rtol=1e-3)
    assert _sigma(8.5, 1.0, 60.0, 10.0, 15.0, jet_efficiency=floor * 0.99) == float(
        "inf"
    )
    assert np.isfinite(_sigma(8.5, 1.0, 60.0, 10.0, 15.0, jet_efficiency=floor * 1.10))
    # No floor on the overtake at any efficiency, however poor.
    assert np.isfinite(
        _sigma_overtaking(8.5, 1.0, 56.53, 0.0, 10.9, jet_efficiency=0.01)
    )


def test_efficiency_never_scales_the_bulk_drift_term() -> None:
    """The ``-1`` and ``+1`` are momentum conservation; no chemistry touches them.

    Checked by construction: at ``k -> 0`` the overtake's impulse per slug kg
    must still carry the arriving momentum whatever the jet does, so ``sigma``
    stays finite as ``jet_efficiency -> 0``.
    """
    starved = _sigma_overtaking(0.5, 1.0, 56.53, 0.0, 10.9, jet_efficiency=1e-9)
    ideal = _sigma_overtaking(0.5, 1.0, 56.53, 0.0, 10.9, jet_efficiency=1.0)
    assert np.isfinite(starved) and starved > ideal
