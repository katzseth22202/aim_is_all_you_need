"""Pinned tests for src/plume_thermal.py (ADR 0014).

The module is pure algebra over a caloric model, so every test here is fast and
closed-form; nothing touches an ephemeris.
"""

import numpy as np
import pytest
from astropy import units as u

from src.plume_thermal import (
    NOZZLE_FLOOR_TEMPERATURE,
    NOZZLE_GATE_TEMPERATURE,
    WATER_ATOMISATION_ENTHALPY,
    chemistry_efficiency,
    plume_ignition_energy,
    plume_temperature,
    slug_ratio_window,
    specific_thermal_energy,
)


def test_ignition_energy_is_dominated_by_atomisation() -> None:
    """Breaking H2O apart, not ionising it, is where the energy goes."""
    total = plume_ignition_energy().to_value(u.MJ / u.kg)
    atomisation = 0.99 * WATER_ATOMISATION_ENTHALPY.to_value(u.MJ / u.mol) / 0.018015
    assert 80.0 < total < 90.0
    assert atomisation / total > 0.55


def test_ignition_energy_rises_with_temperature() -> None:
    """Only the translational term depends on temperature, and it is linear."""
    cold = plume_ignition_energy(0.0 * u.K).to_value(u.J / u.kg)
    warm = plume_ignition_energy(15000.0 * u.K).to_value(u.J / u.kg)
    hot = plume_ignition_energy(30000.0 * u.K).to_value(u.J / u.kg)
    assert warm > cold
    assert np.isclose(hot - warm, warm - cold, rtol=1e-9)


def test_thermal_energy_peaks_at_unit_slug_ratio() -> None:
    """``k/(1+k)^2`` maxes at ``k = 1``, so the plume is hottest there."""
    speed = 50.0 * u.km / u.s
    peak = specific_thermal_energy(speed, 1.0).to_value(u.J / u.kg)
    for k in (0.1, 0.5, 2.0, 10.0):
        assert specific_thermal_energy(speed, k).to_value(u.J / u.kg) < peak
    assert np.isclose(peak, 0.125 * (50_000.0**2), rtol=1e-12)


def test_thermal_energy_rejects_negative_slug_ratio() -> None:
    """A negative slug ratio is not a physical loading."""
    with pytest.raises(ValueError):
        specific_thermal_energy(50.0 * u.km / u.s, -1.0)


def test_window_is_two_sided() -> None:
    """Too little slug is as cold as too much: the admissible set is closed."""
    window = slug_ratio_window(48.82 * u.km / u.s)
    assert window is not None
    low, high = window
    assert 0.0 < low < 1.0 < high
    for k in (low * 1.001, 1.0, high * 0.999):
        assert plume_temperature(48.82 * u.km / u.s, k) >= NOZZLE_FLOOR_TEMPERATURE
    for k in (low * 0.9, high * 1.1):
        assert plume_temperature(48.82 * u.km / u.s, k) < NOZZLE_FLOOR_TEMPERATURE


def test_window_endpoints_sit_exactly_on_the_floor() -> None:
    """Both roots solve the same quadratic, so both land on the floor."""
    window = slug_ratio_window(48.82 * u.km / u.s)
    assert window is not None
    for k in window:
        reached = plume_temperature(48.82 * u.km / u.s, k).to_value(u.K)
        assert np.isclose(reached, NOZZLE_FLOOR_TEMPERATURE.to_value(u.K), rtol=1e-6)


def test_window_widens_with_closing_speed() -> None:
    """A faster impact dissipates more, so it tolerates more slug and less."""
    slow = slug_ratio_window(48.82 * u.km / u.s)
    fast = slug_ratio_window(70.72 * u.km / u.s)
    assert slow is not None and fast is not None
    assert fast[1] > slow[1]
    assert fast[0] < slow[0]


def test_no_window_below_the_ignition_speed() -> None:
    """Peak thermal energy is ``w^2/8``; below that nothing ignites at any k."""
    assert slug_ratio_window(5.0 * u.km / u.s) is None


def test_pinned_windows_at_the_chain_extremes() -> None:
    """Pin the two closing speeds that bound the flown chain's cold ends."""
    leg1 = slug_ratio_window(47.49 * u.km / u.s)
    leg2 = slug_ratio_window(65.44 * u.km / u.s)
    assert leg1 is not None and leg2 is not None
    assert np.isclose(leg1[1], 11.27, rtol=2e-3)
    assert np.isclose(leg2[1], 23.32, rtol=2e-3)


def test_temperature_reports_zero_when_the_blob_cannot_dissociate() -> None:
    """Below the chemical bill the caloric model no longer applies."""
    assert plume_temperature(48.82 * u.km / u.s, 1000.0).to_value(u.K) == 0.0


def test_chemistry_efficiency_reproduces_the_solved_toll_surface() -> None:
    """Cross-repo pin against ``puffsat_impact_simulation``'s ``eta_chem.csv``.

    That surface is solved on ``eos_water`` through the expansion and the
    fireball freeze; this is a closed form.  They must agree, or one of the two
    repositories is charging the atomisation store differently.
    """
    assert np.isclose(
        chemistry_efficiency(45.58 * u.km / u.s, 8.5, 0.926955), 0.753759, rtol=2e-3
    )
    assert np.isclose(
        chemistry_efficiency(75.0 * u.km / u.s, 8.5, 0.999979), 0.909910, rtol=2e-3
    )


def test_unit_bond_fraction_is_a_floor_on_the_solved_value() -> None:
    """``phi <= 1`` and ``eta_chem`` falls with ``phi``, so ``phi = 1`` cannot overstate."""
    for w, k, phi in ((45.58, 8.5, 0.926955), (56.53, 8.5, 1.0), (75.0, 20.0, 0.99)):
        floor = chemistry_efficiency(w * u.km / u.s, k)
        assert floor <= chemistry_efficiency(w * u.km / u.s, k, phi) + 1e-12


def test_the_toll_is_worse_on_the_cold_leg_and_at_high_slug_ratio() -> None:
    """Both monotonicities, since the design's whole exposure follows from them."""
    cold = chemistry_efficiency(45.58 * u.km / u.s, 8.5)
    hot = chemistry_efficiency(75.0 * u.km / u.s, 8.5)
    assert cold < hot
    assert chemistry_efficiency(56.53 * u.km / u.s, 16.0) < chemistry_efficiency(
        56.53 * u.km / u.s, 4.0
    )


def test_no_jet_survives_a_toll_larger_than_the_one_axis_budget() -> None:
    """At 45.58 km/s the chemistry forbids ``k`` above ~19.4 at ``phi = 1``."""
    assert chemistry_efficiency(45.58 * u.km / u.s, 20.0) == 0.0
    assert chemistry_efficiency(45.58 * u.km / u.s, 19.0) > 0.0


def test_the_gate_is_looser_than_the_design_floor() -> None:
    """10 000 K admits more slug than 15 000 K, which reads backwards until named.

    The gate asks where the plume stops dissociating, not where the nozzle
    wants to start; the seed keeps conductivity up well below it.
    """
    gate = slug_ratio_window(45.58 * u.km / u.s, NOZZLE_GATE_TEMPERATURE)
    floor = slug_ratio_window(45.58 * u.km / u.s, NOZZLE_FLOOR_TEMPERATURE)
    assert gate is not None and floor is not None
    assert gate[1] > floor[1]
    assert gate[0] < floor[0]
    # Pinned against the impact sim's solved [0.081, 12.29] and the paper's
    # own [0.098, 10.21] at its 85.1 MJ/kg bill.
    assert np.isclose(gate[1], 11.94, rtol=5e-3)
    assert np.isclose(floor[1], 10.21, rtol=5e-3)
