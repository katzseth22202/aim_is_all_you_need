"""Pinned tests for src/plume_state.py (ledger items 1 and 3).

Item 1's solve lives in ``puffsat_impact_simulation``; what is pinned here is
the burn envelope this repository owns, the bag consequence that follows, and
the fact that the vendored table is consumed correctly.
"""

import astropy.units as u
import numpy as np
import pytest

from src.plume_state import (
    CLIFF_TEMPERATURE,
    FLOWN_BAG_DENSITY,
    QUOTED_SPEEDS,
    SOLVED_V_L,
    _interpolated_cliff,
    burn_envelope,
    burn_stored_energy,
    cliff_temperature,
    field_ratio,
    implied_v_l,
    leak_fraction,
    magnetic_reynolds,
    plume_state,
    pressure_ratio,
    seed_window,
)


def test_dissipated_energy_reproduces_the_published_burn_exactly() -> None:
    """``eps = w^2 k/(2(1+k)^2)`` is this repository's own, so it must be exact."""
    published = {75.0: 265.0, 65.0: 199.0, 56.53: 150.0, 45.58: 98.0}
    for speed, _dissipated, *_ in burn_envelope():
        pass
    for speed, dissipated, *_ in burn_envelope():
        assert np.isclose(dissipated.to_value(u.MJ / u.kg), published[speed], rtol=1e-2)


def test_solved_state_agrees_with_the_papers_hand_figures_to_three_percent() -> None:
    """And it runs *warmer* at every anchor, which is explained rather than noise.

    The hand solve charged 54 MJ/kg for vaporisation plus dissociation while
    ``eos_water``'s bond energy is 50.9, so ~3 MJ/kg stays in the thermal pool.
    The gap widens toward the cold end because there is least energy there for
    it to hide in.
    """
    hand = {75.0: 26200, 65.0: 22400, 56.53: 19400, 45.58: 14700}
    for speed, expected in hand.items():
        temperature, _ = plume_state(speed)
        assert np.isclose(temperature.to_value(u.K), expected, rtol=3.2e-2)
        assert temperature.to_value(u.K) > expected


def test_pressure_ratio_counts_particles_not_mass() -> None:
    """``P/P0 = (1+f) T/T0``: the temperature rise and the extra particles multiply."""
    published = {75.0: 2.75, 65.0: 2.05, 56.53: 1.58, 45.58: 1.03}
    for speed, expected in published.items():
        assert np.isclose(pressure_ratio(speed), expected, rtol=5e-2)
        assert np.isclose(
            field_ratio(speed), np.sqrt(pressure_ratio(speed)), rtol=1e-12
        )


def test_stored_energy_spans_the_published_band() -> None:
    """4.6 to 12.2 GJ across the burn, which is what sizes the nozzle."""
    assert np.isclose(burn_stored_energy(75.0).to_value(u.GJ), 12.2, rtol=5e-2)
    assert np.isclose(burn_stored_energy(45.58).to_value(u.GJ), 4.6, rtol=5e-2)


def test_density_matters_and_the_table_is_two_dimensional_for_that_reason() -> None:
    """Dissipated energy ignores density; Saha does not.

    At 56.53 km/s the solve runs 16 857 K at 0.05 kg/m^3 and 21 795 at 1.0 --
    denser pushes recombination, spends less of the budget stripping electrons,
    and leaves more as heat.  Since the bag volume is a live design variable, a
    one-dimensional table would not have served.
    """
    cold, _ = plume_state(56.53, 0.05)
    dense, _ = plume_state(56.53, 1.0)
    assert dense > cold
    assert np.isclose(cold.to_value(u.K), 16857, rtol=2e-2)


def test_refuses_to_extrapolate_the_saha_solve() -> None:
    """Outside 44-76 km/s and 0.05-2.0 kg/m^3 there is no answer to give."""
    for speed, density in ((30.0, 0.323), (90.0, 0.323), (56.53, 0.001), (56.53, 5.0)):
        with pytest.raises(ValueError):
            plume_state(speed, density)


def test_seed_window_leak_is_one_over_rm_and_capped() -> None:
    """Below the conductivity cliff the field is gone, not leaking by 1000%."""
    assert np.isclose(magnetic_reynolds(15000), 649.7, rtol=1e-2)
    assert np.isclose(leak_fraction(15000), 1.0 / magnetic_reynolds(15000), rtol=1e-9)
    assert leak_fraction(2000) == 1.0


def test_the_papers_rm_column_is_not_one_expansion() -> None:
    """Item 3's finding, and it is why the table wants regenerating.

    If ``tab:seed_window`` were a single flow sampled at several temperatures,
    every row would imply the same ``v L``.  The spread is a factor of 13 and
    is non-monotone -- rising to 6000 K then collapsing at 15 000 -- so no
    single ``v L`` reproduces the column.
    """
    implied = [implied_v_l(row[0]) for row in seed_window()]
    assert max(implied) / min(implied) > 10.0
    # Non-monotone: 6000 K implies the largest, 15 000 K nearly the smallest.
    assert implied_v_l(6000) > implied_v_l(15000)


def test_conductivity_rises_monotonically_through_the_window() -> None:
    """The seed keeps supplying electrons where the water has re-formed."""
    conductivities = [row[2] for row in seed_window()]
    assert conductivities == sorted(conductivities)
    assert conductivities[-1] / conductivities[0] > 1000.0


def test_the_cliff_is_solved_not_interpolated() -> None:
    """D9: the crossing is an output of the conductivity model, not of its table.

    ``_SEED_WINDOW_SIGMA`` samples every 1000 K and ``sigma`` climbs 60x
    between its first two points, so interpolating the crossing out of it is
    guessing at the shape of the steepest part of the curve.  It guesses high
    by about 120 K, which is what put 2568 K into the paper's handoff where the
    companion solves 2450 K.
    """
    assert cliff_temperature() == 2450.0
    error = _interpolated_cliff() - cliff_temperature()
    assert 100.0 < error < 140.0


def test_a_faster_expansion_lowers_the_cliff() -> None:
    """``Rm = mu0 sigma v L``, so more ``v L`` holds the field to colder plume."""
    solved = [CLIFF_TEMPERATURE[v_l] for v_l in sorted(CLIFF_TEMPERATURE)]
    assert solved == sorted(solved, reverse=True)
    assert cliff_temperature(1.81e4) > cliff_temperature(SOLVED_V_L)


def test_the_cliff_is_not_what_binds_the_seed_window() -> None:
    """The leak floor the paper states, 3800 K, sits well above the cliff.

    The gap is the argument: the slug runs out of capacity to absorb leaked
    heat long before the field loses its grip, so the window's floor is a
    thermal limit rather than a conductivity one.
    """
    assert 1300.0 < 3800.0 - cliff_temperature() < 1400.0


def test_no_cliff_is_offered_for_an_unsolved_expansion() -> None:
    """These are recorded solves, not a model; interpolating them is the bug."""
    with pytest.raises(KeyError):
        cliff_temperature(6.0e4)
