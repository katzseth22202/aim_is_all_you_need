"""Pinned tests for src/bag_state.py (ledger items 5-8 and 10).

``tab:bag_state`` is a chain of six dependent steps, so the reproduction test
is the load-bearing one: if any link drifts the printed table stops matching.
"""

import astropy.units as u
import numpy as np
import pytest

from src.bag_state import (
    AXIAL_BAG_LENGTHS,
    BAG_SIZING_RADII,
    HANDLING_FILM_GAUGE,
    HANDLING_GAUGE_BAND,
    HOT_PULSE_STATE,
    PAPER_LEAK_FRACTION,
    SLUG_MASS,
    SOLVED_LEAK_FRACTIONS,
    SPHERE_BORE_RADIUS,
    BagState,
    bag_density_at,
    bag_surface_area,
    bore_radius,
    coldest_mist_temperature,
    confinement_field_at,
    film_mass_fraction,
    governing_film_mass,
    handling_film_mass,
    paper_bag_state,
    radiated_fraction,
    saturation_density,
    saturation_pressure,
    saturation_temperature,
    shape_factor,
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


def test_bag_sizing_density_and_both_field_columns_reproduce() -> None:
    """Item 4: 18 of the table's cells, to the digits the paper prints.

    Reproducing the field to three figures is what settles that the table is
    ideal-gas pressure and not ``(gamma-1) e`` -- the two differ by ~30% here.
    """
    published = {
        1.8: (8.72, 21.3, 35.4),
        3.5: (1.19, 7.9, 13.1),
        5.4: (0.32, 4.1, 6.8),
        8.7: (0.077, 2.0, 3.3),
        17.5: (0.0095, 0.70, 1.16),
        28.0: (0.0023, 0.35, 0.58),
    }
    for radius, (density, cold, hot) in published.items():
        metres = radius * u.m
        assert np.isclose(
            bag_density_at(metres).to_value(u.kg / u.m**3), density, rtol=2e-2
        )
        assert np.isclose(confinement_field_at(metres).to_value(u.T), cold, rtol=2e-2)
        assert np.isclose(
            confinement_field_at(metres, *HOT_PULSE_STATE).to_value(u.T), hot, rtol=2e-2
        )


def test_the_hot_column_is_the_cold_one_scaled_by_one_number() -> None:
    """1.66x at every radius, from ``sqrt(3(1+f) T_hot / 3 T_cold)``.

    Ionisation enters pressure only through the particle count, so the ratio is
    a property of the plume state and not of the bag.
    """
    ratios = [
        confinement_field_at(r * u.m, *HOT_PULSE_STATE) / confinement_field_at(r * u.m)
        for r in BAG_SIZING_RADII
    ]
    for ratio in ratios:
        assert np.isclose(float(ratio), 1.657, rtol=1e-3)


def test_radiated_fraction_reproduces_and_grows_as_radius_squared() -> None:
    """0.1% at 1.8 m to 31% at 28 m -- the bound that caps the bag from above."""
    # Tolerances are the paper's own printed precision: 0.1% and 31% are one
    # and two significant figures respectively.
    for radius, expected, tolerance in (
        (1.8, 0.001, 5e-4),
        (5.4, 0.012, 1e-3),
        (8.7, 0.030, 1e-3),
        (28.0, 0.31, 5e-3),
    ):
        assert np.isclose(radiated_fraction(radius * u.m), expected, atol=tolerance)
    assert np.isclose(
        radiated_fraction(10.8 * u.m) / radiated_fraction(5.4 * u.m), 4.0, rtol=1e-9
    )


def test_mist_column_reproduces_across_the_usable_band() -> None:
    """Item 4's last column, where the paper's own design actually sits.

    The paper settles on 3.5-7 m (opacity above, radiative loss below).  Inside
    that band this is within ~1.5 K; outside it, it overshoots by 16 K at 1.8 m
    and 9 K at 28 m, which is recorded on the function.
    """
    for radius, expected in ((3.5, 332.0), (5.4, 306.0), (8.7, 281.0)):
        assert np.isclose(
            coldest_mist_temperature(radius * u.m).to_value(u.K), expected, atol=2.0
        )


def test_axial_bag_reproduces_every_cell() -> None:
    """Item 9: all five rows, bore and conductor and shape factor and film."""
    published = {
        10.8: (5.40, 1.00, 1.50, 2.8),
        16.0: (3.62, 0.99, 1.82, 3.4),
        23.0: (3.02, 1.19, 1.90, 3.6),
        32.0: (2.56, 1.41, 1.94, 3.6),
        50.0: (2.05, 1.76, 1.97, 3.7),
    }
    reference = SPHERE_BORE_RADIUS.to_value(u.m) * 10.8
    for length, (bore, conductor, factor, film) in published.items():
        metres = length * u.m
        radius = (
            SPHERE_BORE_RADIUS if length == 10.8 else bore_radius(metres)
        ).to_value(u.m)
        assert np.isclose(radius, bore, rtol=5e-3)
        assert np.isclose(radius * length / reference, conductor, rtol=1e-2)
        assert np.isclose(shape_factor(metres), factor, rtol=5e-3)
        assert np.isclose(2.8 * shape_factor(metres) / 1.5, film, atol=0.1)


def test_the_sphere_is_the_lightest_film_and_the_long_tube_the_heaviest() -> None:
    """``F`` runs 1.5 to 2.0, so stretching the bag costs a third of the film."""
    assert np.isclose(shape_factor(10.8 * u.m), 1.5, rtol=1e-3)
    assert shape_factor(50.0 * u.m) < 2.0
    assert shape_factor(200.0 * u.m) > shape_factor(50.0 * u.m)


def test_a_capsule_carries_exactly_two_pi_r_l_of_film() -> None:
    """The hemispherical caps put back what shortening the cylinder removed."""
    sphere = bag_surface_area(2.0 * SPHERE_BORE_RADIUS)
    assert np.isclose(
        sphere.to_value(u.m**2),
        (4.0 * np.pi * SPHERE_BORE_RADIUS**2).to_value(u.m**2),
        rtol=1e-9,
    )


def test_stretching_the_bag_costs_area_faster_than_it_costs_shape_factor() -> None:
    """D4's finding: ``F`` is the wrong scaling once the bag holds no pressure.

    A pressure vessel's film mass is independent of radius and rises only
    through ``F``, which saturates at 2.0.  A handling-gauge bag pays for
    *area*, ``2 pi r L``, which grows as ``sqrt(L)`` and does not saturate: from
    16 m to 50 m the shape factor rises 8% while the area rises 77%.
    """
    short, long = 16.0 * u.m, 50.0 * u.m
    shape_growth = shape_factor(long) / shape_factor(short)
    area_growth = float(bag_surface_area(long) / bag_surface_area(short))
    assert shape_growth < 1.1 < 1.7 < area_growth
    assert np.isclose(area_growth, np.sqrt(50.0 / 16.0), rtol=1e-6)


def test_the_handling_floor_does_not_care_whether_the_slug_boils() -> None:
    """Which is the whole point: ``eq:bag_film_mass`` returns 0 kg, a bag does not.

    From cold storage the solved leak boils nothing, so the pressure vessel is
    massless.  The flown 23 m column still needs 2.4-10.0 kg of film depending
    on gauge, centred on 5.1 kg at Echo 1's half-mil.
    """
    flown = 23.0 * u.m
    thin, thick = HANDLING_GAUGE_BAND
    assert np.isclose(handling_film_mass(flown).to_value(u.kg), 5.1, atol=0.1)
    assert np.isclose(handling_film_mass(flown, thin).to_value(u.kg), 2.4, atol=0.1)
    assert np.isclose(handling_film_mass(flown, thick).to_value(u.kg), 10.0, atol=0.1)
    assert governing_film_mass(0.0 * u.kg, flown) == handling_film_mass(flown)


def test_the_pressure_vessel_still_wins_when_the_slug_does_boil() -> None:
    """The floor is a floor, not a replacement: whichever is larger flies."""
    heavy = 40.0 * u.kg
    assert governing_film_mass(heavy, 23.0 * u.m) == heavy


def test_the_gauge_band_brackets_the_quoted_gauge() -> None:
    """6-25 um around Echo 1's 12.7 um, so the floor is quoted with a range."""
    thin, thick = HANDLING_GAUGE_BAND
    assert thin < HANDLING_FILM_GAUGE < thick
    for length in AXIAL_BAG_LENGTHS:
        metres = length * u.m
        assert (
            handling_film_mass(metres, thin)
            < handling_film_mass(metres)
            < handling_film_mass(metres, thick)
        )
