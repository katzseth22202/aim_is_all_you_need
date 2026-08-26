"""Pinned tests for src/bag_state.py (ledger items 5-8 and 10).

``tab:bag_state`` is a chain of six dependent steps, so the reproduction test
is the load-bearing one: if any link drifts the printed table stops matching.
"""

import astropy.units as u
import numpy as np
import pytest

from src.bag_state import (
    PAPER_LEAK_FRACTION,
    SLUG_MASS,
    SOLVED_LEAK_FRACTIONS,
    BagState,
    film_mass_fraction,
    paper_bag_state,
    saturation_density,
    saturation_pressure,
    saturation_temperature,
    stored_field_energy,
)


def test_reproduces_every_printed_cell_of_tab_bag_state() -> None:
    """Both columns, all eight rows, to the digits the paper prints."""
    jupiter, earth = paper_bag_state("jupiter"), paper_bag_state("earth")
    for state, boil, x, temp, press, film in (
        (jupiter, 0.26, 0.11, 306.0, 4.9, 2.8),
        (earth, 0.78, 0.33, 328.0, 16.0, 8.9),
    ):
        assert np.isclose(state.waste_heat.to_value(u.MJ / u.kg), 0.99, atol=5e-3)
        assert np.isclose(state.boiling.to_value(u.MJ / u.kg), boil, atol=5e-3)
        assert np.isclose(state.vapour_fraction, x, atol=5e-3)
        assert np.isclose(state.mist_temperature.to_value(u.K), temp, atol=1.0)
        assert np.isclose(state.mist_pressure.to_value(u.kPa), press, rtol=4e-2)
        assert np.isclose(state.film_mass.to_value(u.kg), film, atol=0.15)


def test_stored_energy_reproduces_both_published_values() -> None:
    """4.43 GJ dissociated-neutral, 12.2 GJ at the hottest pulse's ionisation."""
    assert np.isclose(stored_field_energy().to_value(u.GJ), 4.43, rtol=3e-3)
    assert np.isclose(
        stored_field_energy(26200 * u.K, 0.573).to_value(u.GJ), 12.2, rtol=5e-3
    )


def test_stored_energy_is_invariant_across_the_bag_sizing_radius_sweep() -> None:
    """Item 5: ``E_B = n R_g T`` has no radius in it, over a 15x range.

    This is why a bigger bag does not buy a lighter nozzle -- the nozzle is
    sized by the energy it contains, and that is fixed by slug and temperature.
    """
    reference = stored_field_energy().to_value(u.J)
    for _ in range(1):  # radius does not enter the call at all, which is the point
        assert np.isclose(stored_field_energy().to_value(u.J), reference, rtol=1e-12)


def test_saturation_curve_matches_the_papers_two_printed_points() -> None:
    """Magnus-Tetens against 4.9 kPa at 306 K and 16 kPa at 328 K."""
    assert np.isclose(saturation_pressure(306 * u.K).to_value(u.kPa), 4.9, rtol=4e-2)
    assert np.isclose(saturation_pressure(328 * u.K).to_value(u.kPa), 16.0, rtol=4e-2)


def test_saturation_inversion_round_trips() -> None:
    """The root-find must invert its own forward curve."""
    for kelvin in (280.0, 306.0, 328.0, 400.0):
        density = saturation_density(kelvin * u.K)
        assert np.isclose(
            saturation_temperature(density).to_value(u.K), kelvin, rtol=1e-6
        )


def test_saturation_inversion_refuses_to_leave_its_domain() -> None:
    """Past the bracket the model is not applicable and must say so."""
    with pytest.raises(ValueError):
        saturation_temperature(1e-9 * u.kg / u.m**3)
    with pytest.raises(ValueError):
        saturation_temperature(1e4 * u.kg / u.m**3)


def test_film_mass_is_linear_in_vapour_and_temperature_and_free_of_radius() -> None:
    """Item 8: every factor of bag radius cancels out of ``eq:bag_film_mass``."""
    base = film_mass_fraction(0.11, 306 * u.K)
    assert np.isclose(film_mass_fraction(0.22, 306 * u.K), 2.0 * base, rtol=1e-12)
    assert np.isclose(film_mass_fraction(0.11, 612 * u.K), 2.0 * base, rtol=1e-12)
    assert np.isclose(
        film_mass_fraction(0.11, 306 * u.K, 2.0) / base, 2.0 / 1.5, rtol=1e-12
    )


def test_the_leak_bracket_closes_dry_from_cold_storage() -> None:
    """Item 10, and it is the answer to "3.7 kg against 31 kg": neither.

    At the solved residence-weighted leak the waste heat does not even finish
    melting the slug on any leg, so nothing boils, the bag holds no pressure,
    and the film is not a mass item at all.  The feared case -- the paper's
    4.4% held while ``E_B`` rises to 12.2 GJ -- is 23 kg on the same leg.
    """
    legs = {75.0: (26200, 0.573), 45.58: (14700, 0.053)}
    for speed, (temp, ionised) in legs.items():
        energy = stored_field_energy(temp * u.K, ionised)
        solved = BagState(SOLVED_LEAK_FRACTIONS["equilibrium"][speed], energy)
        feared = BagState(PAPER_LEAK_FRACTION, energy)
        assert solved.vapour_fraction == 0.0
        assert solved.film_mass.to_value(u.kg) == 0.0
        assert feared.film_mass.to_value(u.kg) > 3.0
    assert (
        BagState(
            PAPER_LEAK_FRACTION, stored_field_energy(26200 * u.K, 0.573)
        ).film_mass.to_value(u.kg)
        > 20.0
    )


def test_the_cold_leg_has_the_least_margin_against_boiling() -> None:
    """The hot legs clear it 10x over; the cold leg only 1.2x.

    ``E_B`` falls faster with closing speed than the leak fraction rises, so
    the *product* is smallest on the hot legs -- but the cold leg's 2.54% leak
    against a 2.93% onset is the one worth watching, not the 12.2 GJ store.
    """
    onsets = {}
    for speed, (temp, ionised) in {
        75.0: (26200, 0.573),
        45.58: (14700, 0.053),
    }.items():
        energy = stored_field_energy(temp * u.K, ionised)
        state = BagState(0.0, energy)
        deficit = state.warming - state.intercept
        onsets[speed] = float(deficit * SLUG_MASS / energy)
    assert onsets[75.0] / SOLVED_LEAK_FRACTIONS["equilibrium"][75.0] > 8.0
    assert 1.0 < onsets[45.58] / SOLVED_LEAK_FRACTIONS["equilibrium"][45.58] < 1.5


def test_warm_storage_is_what_makes_the_bag_a_pressure_vessel() -> None:
    """Cold storage absorbs 0.73 MJ/kg that never becomes pressure.

    From Earth's 278 K only 0.21 MJ/kg is absorbed, so the cold leg does boil
    and does need a film -- 4.9 kg rather than nothing.
    """
    energy = stored_field_energy(14700 * u.K, 0.053)
    leak = SOLVED_LEAK_FRACTIONS["equilibrium"][45.58]
    assert BagState(leak, energy, "jupiter").film_mass.to_value(u.kg) == 0.0
    warm = BagState(leak, energy, "earth")
    assert 4.0 < warm.film_mass.to_value(u.kg) < 6.0
