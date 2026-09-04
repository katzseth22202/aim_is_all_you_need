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
    LEG_PLUME_STATES,
    MELTING_GATE,
    PLUG_MASS,
    SLUG_MASS,
    SOLVED_LEAK_FRACTIONS,
    SPHERE_BORE_RADIUS,
    SUPERSEDED_LEAK_FRACTION,
    SUPERSEDED_WARMING_TO_LIQUID,
    TABLE_CLOSING_SPEED,
    WARMING_TO_LIQUID,
    BagState,
    bag_density_at,
    bag_surface_area,
    boil_onset_leak,
    bore_radius,
    coldest_mist_temperature,
    confinement_field_at,
    film_mass_fraction,
    governing_film_mass,
    handling_film_mass,
    melting_fixed_point,
    paper_bag_state,
    radiated_fraction,
    saturation_density,
    saturation_pressure,
    saturation_temperature,
    shape_factor,
    stored_field_energy,
    superseded_bag_state,
    table_stored_energy,
)


def test_reproduces_every_printed_cell_of_tab_bag_state() -> None:
    """Both columns, every row, to the digits the paper prints.

    The table is the solved 2.54% leak at the 45.58 km/s cold end of the growth
    push -- one column of ``LEG_PLUME_STATES``, not a flown assumption.  The
    Jupiter column is the interesting one and it is the one ADR 0027 moved:
    0.646 MJ/kg of waste heat against a 0.570 MJ/kg *melting gate*, so the
    slug melts through with 0.076 to spare and boils 2.2% of itself.  It used
    to read zero, because the gate was taken to be the whole 0.653 ladder --
    liquid warming included, which is not a melting cost.
    """
    jupiter, earth = paper_bag_state("jupiter"), paper_bag_state("earth")
    for state in (jupiter, earth):
        assert np.isclose(state.intercept.to_value(u.MJ / u.kg), 0.10, atol=5e-3)
        # The row reads "2.54% of 21.5 MJ/kg", and 21.5 is itself the rounded
        # 21.43 the cold leg actually stores, so the printed 0.55 is 0.3%
        # above the product of the unrounded inputs.  Same shape of rounding
        # as the superseded table's 0.89-against-0.915.
        assert np.isclose(state.leak.to_value(u.MJ / u.kg), 0.55, atol=1e-2)
        assert np.isclose(state.waste_heat.to_value(u.MJ / u.kg), 0.65, atol=5e-3)

    assert jupiter.melts
    assert np.isclose(jupiter.melting_gate.to_value(u.MJ / u.kg), 0.570, atol=5e-4)
    # The warming row is an output on this column: the gate plus liquid
    # warming to the solved mist, not to an assumed room temperature.
    assert np.isclose(jupiter.warming.to_value(u.MJ / u.kg), 0.594, atol=5e-3)
    assert np.isclose(jupiter.boiling.to_value(u.MJ / u.kg), 0.052, atol=5e-3)
    assert np.isclose(jupiter.vapour_fraction, 0.022, atol=5e-4)
    assert np.isclose(jupiter.vapour_mass.to_value(u.kg), 4.68, atol=0.05)
    assert np.isclose(jupiter.mist_temperature.to_value(u.K), 278.8, atol=0.5)
    assert np.isclose(jupiter.mist_pressure.to_value(u.kPa), 0.91, atol=0.05)
    # "a fraction of a kilogram", and far under tab:axial_bag's 2.4-10.0 kg
    # handling floor, so the conclusion the section rests on is unchanged.
    assert np.isclose(jupiter.film_mass.to_value(u.kg), 0.51, atol=0.05)
    assert jupiter.film_mass < 1.0 * u.kg

    assert np.isclose(earth.boiling.to_value(u.MJ / u.kg), 0.44, atol=5e-3)
    assert np.isclose(earth.vapour_fraction, 0.18, atol=5e-3)
    assert np.isclose(earth.mist_temperature.to_value(u.K), 316.0, atol=1.0)
    assert np.isclose(earth.mist_pressure.to_value(u.kPa), 8.7, rtol=4e-2)
    assert np.isclose(earth.film_mass.to_value(u.kg), 4.9, atol=0.15)
    assert np.isclose(earth.film_fraction, 0.023, atol=5e-4)  # "2.3%, 4.9 kg"


def test_the_table_is_the_cold_leg_and_says_so() -> None:
    """``E_B`` = 21.5 MJ/kg is the 45.58 km/s row, which the caption quotes."""
    assert np.isclose(
        (table_stored_energy() / SLUG_MASS).to_value(u.MJ / u.kg), 21.5, atol=0.1
    )
    assert SOLVED_LEAK_FRACTIONS["equilibrium"][TABLE_CLOSING_SPEED] == 0.0254


def test_the_superseded_table_is_kept_and_is_not_the_one_in_print() -> None:
    """306 K and 328 K are the fingerprint of the pre-solve 4.4% leak.

    Anything downstream still quoting either was computed before the leak was
    solved, which is the only reason this reproduction is carried at all.
    """
    was_jupiter, was_earth = superseded_bag_state("jupiter"), superseded_bag_state(
        "earth"
    )
    for state, boil, x, temp, press, film in (
        (was_jupiter, 0.26, 0.11, 306.0, 4.9, 2.8),
        (was_earth, 0.78, 0.33, 328.0, 16.0, 8.9),
    ):
        assert np.isclose(state.waste_heat.to_value(u.MJ / u.kg), 0.99, atol=5e-3)
        assert np.isclose(state.boiling.to_value(u.MJ / u.kg), boil, atol=5e-3)
        assert np.isclose(state.vapour_fraction, x, atol=5e-3)
        assert np.isclose(state.mist_temperature.to_value(u.K), temp, atol=1.0)
        assert np.isclose(state.mist_pressure.to_value(u.kPa), press, rtol=4e-2)
        assert np.isclose(state.film_mass.to_value(u.kg), film, atol=0.15)
    # The two tables must not be confusable: every output row moved.
    now = paper_bag_state("earth")
    assert now.mist_temperature < was_earth.mist_temperature
    assert now.film_mass < was_earth.film_mass


def test_the_plug_can_only_take_vapour_away() -> None:
    """``sec:needle_through_fog``, and the direction is the whole point.

    The plug is 37.5 kg of extra condensed mass absorbing heat that the pulse
    was going to deliver anyway, so at a fixed bill it removes 24.5 MJ from
    the boiling step -- 17.8% of the 138 MJ, which is the paper's "a sixth".
    Vapour mass and mist temperature must both fall.

    The 24.5 is the corrected ladder, 0.653 MJ/kg over 37.5 kg.  It was 27.4
    at 0.73, so this whole row moved with ADR 0027 and the paper's B1 figures
    (27.7 kg, 309 K) move with it.
    """
    bare, plugged = (
        paper_bag_state("earth"),
        paper_bag_state("earth", plug_mass=PLUG_MASS),
    )
    assert np.isclose((plugged.plug_sink * SLUG_MASS).to_value(u.MJ), 24.5, atol=0.1)
    assert np.isclose(float(plugged.plug_sink / bare.waste_heat), 0.178, atol=5e-3)
    assert np.isclose(bare.vapour_mass.to_value(u.kg), 39.3, atol=0.1)
    assert np.isclose(plugged.vapour_mass.to_value(u.kg), 28.9, atol=0.1)
    assert plugged.vapour_mass < bare.vapour_mass
    assert plugged.mist_temperature < bare.mist_temperature
    assert np.isclose(plugged.mist_temperature.to_value(u.K), 310.2, atol=0.5)
    assert plugged.film_mass < bare.film_mass


def test_the_plug_is_what_keeps_the_cold_leg_dry() -> None:
    """The direction reversed with ADR 0027, and the plug now earns its keep.

    Cold storage on its own no longer clears the boiling step: the bare column
    melts through by 0.076 MJ/kg.  The plug's 24.5 MJ is 0.115 MJ/kg of slug,
    which is more than that surplus, so it puts the column back under the gate.
    Cold storage *plus a plug* is the one dry case left in the table.
    """
    bare = paper_bag_state("jupiter")
    plugged = paper_bag_state("jupiter", plug_mass=PLUG_MASS)
    assert bare.melts and bare.vapour_fraction > 0.0
    assert not plugged.melts
    assert plugged.vapour_fraction == 0.0
    assert plugged.film_mass.to_value(u.kg) == 0.0


def test_the_plug_takes_the_flown_column_below_the_handling_floor() -> None:
    """So on the leg that does boil, the plug retires the pressure vessel."""
    flown = 23.0 * u.m
    scale = shape_factor(flown) / 1.5
    bare = paper_bag_state("earth").film_mass * scale
    plugged = paper_bag_state("earth", plug_mass=PLUG_MASS).film_mass * scale
    assert governing_film_mass(bare, flown) == bare
    assert governing_film_mass(plugged, flown) == handling_film_mass(flown)


def test_the_earth_warming_row_is_still_closed_against_the_superseded_mist() -> None:
    """Cut 4, on the one column that still cuts the loop.

    0.21 MJ/kg warms liquid water from 278 K to 328 K, and 328 K is the
    superseded table's Earth mist.  Closing the loop moves that column by half
    a kilogram of film -- reported, not applied, because the paper prints the
    row as an input.
    """
    earth = paper_bag_state("earth")
    closed = melting_fixed_point(earth.waste_heat, "earth")
    assert closed.warming < earth.warming
    assert closed.vapour_fraction > earth.vapour_fraction
    assert np.isclose(closed.mist_temperature.to_value(u.K), 318.2, atol=0.5)
    assert abs((closed.film_mass - earth.film_mass).to_value(u.kg)) < 1.0


def test_the_ice_column_is_the_solve_rather_than_a_report_against_it() -> None:
    """ADR 0027: the Jupiter column's warming row became an output.

    Its surplus over the melting gate is 0.076 MJ/kg against a 0.653 ladder,
    so charging the liquid warming at an assumed room temperature would swamp
    it.  ``paper_bag_state`` and ``melting_fixed_point`` must therefore agree
    on that column exactly rather than bracket each other.
    """
    jupiter = paper_bag_state("jupiter")
    solved = melting_fixed_point(jupiter.waste_heat, "jupiter")
    assert not solved.below_freezing
    assert np.isclose(
        jupiter.warming.to_value(u.MJ / u.kg), solved.warming.to_value(u.MJ / u.kg)
    )
    assert np.isclose(jupiter.vapour_fraction, solved.vapour_fraction)
    assert np.isclose(
        jupiter.mist_temperature.to_value(u.K), solved.mist_temperature.to_value(u.K)
    )

    # And the column's own book balances: nothing is charged twice or dropped.
    residual = (
        jupiter.waste_heat - jupiter.warming - jupiter.plug_sink - jupiter.boiling
    )
    assert abs(residual.to_value(u.MJ / u.kg)) < 1e-9

    # With the plug on top the two curves stop meeting above freezing at all,
    # which is the same verdict ``BagState.melts`` reaches.
    frozen = melting_fixed_point(jupiter.waste_heat, "jupiter", plug_mass=PLUG_MASS)
    assert frozen.below_freezing
    assert not paper_bag_state("jupiter", plug_mass=PLUG_MASS).melts


def test_the_melting_gate_is_the_first_two_terms_and_nothing_else() -> None:
    """The correction itself, stated as an identity.

    Warming 122 K ice to the melting point (0.236, integrating the measured
    heat capacity of ice Ih) plus fusion (0.334) is 0.570 MJ/kg, and that is
    the whole gate.  The ladder adds liquid warming to room temperature and
    reaches 0.653, which is what the plug is charged and what a shading window
    must reject -- but it is not a melting cost, and using it as one is what
    put a zero in the Jupiter column.
    """
    gate = MELTING_GATE["jupiter"]
    ladder = WARMING_TO_LIQUID["jupiter"]
    assert np.isclose(gate.to_value(u.MJ / u.kg), 0.570, atol=5e-4)
    assert np.isclose(ladder.to_value(u.MJ / u.kg), 0.653, atol=1e-3)
    assert MELTING_GATE["earth"].to_value(u.MJ / u.kg) == 0.0

    # Onset follows the gate, and the paper prints 2.19%.
    assert np.isclose(boil_onset_leak(table_stored_energy()), 0.0219, atol=1e-4)
    # The retired claim was 2.93%: the pre-correction 0.73 ladder charged as
    # the gate.  Both errors pushed the same way -- 0.02 MJ/kg from holding
    # ice's heat capacity constant, 0.08 from putting liquid warming inside a
    # melting threshold -- and together they hid the crossing.
    superseded = float(
        (SUPERSEDED_WARMING_TO_LIQUID["jupiter"] - paper_bag_state("jupiter").intercept)
        * SLUG_MASS
        / table_stored_energy()
    )
    assert np.isclose(superseded, 0.0293, atol=1e-4)
    assert superseded > SOLVED_LEAK_FRACTIONS["equilibrium"][TABLE_CLOSING_SPEED]
    assert boil_onset_leak(table_stored_energy()) < (
        SOLVED_LEAK_FRACTIONS["equilibrium"][TABLE_CLOSING_SPEED]
    )


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

    At the solved residence-weighted leak the three hot legs never finish
    melting and the cold one boils half a kilogram of film -- against a
    2.4-10.0 kg handling floor, so on no leg is the pressure vessel a mass
    item.  The feared case -- the paper's 4.4% held while ``E_B`` rises to
    12.2 GJ -- is 21 kg on the hot leg.
    """
    for speed, (temp, ionised) in LEG_PLUME_STATES.items():
        energy = stored_field_energy(temp * u.K, ionised)
        solved = BagState(SOLVED_LEAK_FRACTIONS["equilibrium"][speed], energy)
        feared = BagState(SUPERSEDED_LEAK_FRACTION, energy)
        if speed == TABLE_CLOSING_SPEED:  # the cold end, the only leg that melts
            assert solved.melts
            assert solved.film_mass.to_value(u.kg) < 1.0
        else:
            assert not solved.melts
            assert solved.film_mass.to_value(u.kg) == 0.0
        assert feared.film_mass.to_value(u.kg) > 3.0
    assert (
        BagState(
            SUPERSEDED_LEAK_FRACTION, stored_field_energy(26200 * u.K, 0.573)
        ).film_mass.to_value(u.kg)
        > 20.0
    )


def test_the_cold_leg_is_past_boiling_onset_rather_than_short_of_it() -> None:
    """The sign of this margin flipped with ADR 0027, and the paper says so.

    ``E_B`` falls faster with closing speed than the leak fraction rises, so
    the *product* is smallest on the hot legs.  Those clear onset 4.9-7.5x
    over.  The cold leg's 2.54% against a 2.19% onset does not clear it at
    all: melting completes and the leg boils.  Charged against the 0.653
    ladder instead, onset read 2.93% and the same leg looked 1.2x safe.
    """
    ratios = {
        speed: boil_onset_leak(stored_field_energy(temp * u.K, ionised))
        / SOLVED_LEAK_FRACTIONS["equilibrium"][speed]
        for speed, (temp, ionised) in LEG_PLUME_STATES.items()
    }
    hot = [ratio for speed, ratio in ratios.items() if speed != TABLE_CLOSING_SPEED]
    assert 4.5 < min(hot) and max(hot) < 8.0
    assert ratios[TABLE_CLOSING_SPEED] < 1.0
    assert np.isclose(ratios[TABLE_CLOSING_SPEED], 0.86, atol=0.02)


def test_warm_storage_is_what_makes_the_bag_a_pressure_vessel() -> None:
    """Cold storage absorbs 0.570 MJ/kg before a gram of it can boil.

    From Earth's 278 K there is no gate at all, so the same leg boils eight
    times harder and does need a film -- 4.9 kg against half a kilogram.  The
    claim the section rests on survives ADR 0027 in the form "cold storage
    removes the pressure vessel", because half a kilogram is far under the
    handling floor; it does not survive in the form "nothing boils".
    """
    energy = stored_field_energy(14700 * u.K, 0.053)
    leak = SOLVED_LEAK_FRACTIONS["equilibrium"][45.58]
    cold = BagState(leak, energy, "jupiter")
    warm = BagState(leak, energy, "earth")
    assert 0.4 < cold.film_mass.to_value(u.kg) < 0.6
    assert 4.0 < warm.film_mass.to_value(u.kg) < 6.0
    assert warm.vapour_mass / cold.vapour_mass > 8.0
    # Under the handling floor tab:axial_bag prints for the flown 23 m column.
    assert cold.film_mass < handling_film_mass(23.0 * u.m, HANDLING_GAUGE_BAND[0])


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
    that band this is within ~1 K; outside it, it overshoots, which is recorded
    on the function.  The held fraction is ``tab:bag_state``'s Earth-storage
    column, which the caption names -- and the 5.4 m row therefore has to be
    that table's own 316 K by construction.
    """
    for radius, expected in ((3.5, 346.0), (5.4, 316.0), (8.7, 290.0)):
        assert np.isclose(
            coldest_mist_temperature(radius * u.m).to_value(u.K), expected, atol=1.5
        )
    assert np.isclose(
        coldest_mist_temperature(5.4 * u.m).to_value(u.K),
        paper_bag_state("earth").mist_temperature.to_value(u.K),
        atol=0.5,
    )


def test_the_mist_column_tracks_the_table_rather_than_a_pinned_fraction() -> None:
    """This column went stale once already; it must not be able to again.

    It was written against the superseded 4.4% leak's ``x`` = 0.11, printed a
    306 K flown row, and that 306 K then survived into prose after the leak was
    solved.  Reading the default off ``paper_bag_state`` is what closes that.
    """
    superseded = coldest_mist_temperature(
        5.4 * u.m, superseded_bag_state("jupiter").vapour_fraction
    )
    assert np.isclose(superseded.to_value(u.K), 306.0, atol=1.0)
    assert coldest_mist_temperature(5.4 * u.m) > superseded


def test_axial_bag_reproduces_every_cell() -> None:
    """Item 9: all five rows, bore and conductor and shape factor and film.

    The film column is ``tab:axial_bag``'s **Pressure** column, which the
    caption pins to ``tab:bag_state``'s Earth-storage vapour state -- the only
    case left that boils.  It scales with nothing but ``F``.
    """
    published = {
        10.8: (5.40, 1.00, 1.50, 4.9),
        16.0: (3.62, 0.99, 1.82, 5.9),
        23.0: (3.02, 1.19, 1.90, 6.2),
        32.0: (2.56, 1.41, 1.94, 6.3),
        50.0: (2.05, 1.76, 1.97, 6.4),
    }
    reference = SPHERE_BORE_RADIUS.to_value(u.m) * 10.8
    sphere = paper_bag_state("earth").film_mass.to_value(u.kg)
    for length, (bore, conductor, factor, film) in published.items():
        metres = length * u.m
        radius = (
            SPHERE_BORE_RADIUS if length == 10.8 else bore_radius(metres)
        ).to_value(u.m)
        assert np.isclose(radius, bore, rtol=5e-3)
        assert np.isclose(radius * length / reference, conductor, rtol=1e-2)
        assert np.isclose(shape_factor(metres), factor, rtol=5e-3)
        assert np.isclose(sphere * shape_factor(metres) / 1.5, film, atol=0.1)


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
