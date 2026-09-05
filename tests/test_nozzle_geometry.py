"""Pinned tests for src/nozzle_geometry.py (ledger item 11).

The arrival radius is an *input* nobody had written down and ``k`` is its
output, so these pin the relation in both directions and the cross-check
against ``puffsat_impact_simulation``'s own integration.
"""

import numpy as np
import pytest

from src.nozzle_geometry import (
    BAG_SLUG_MASS,
    BAG_VOLUME,
    BORE_RADIUS,
    DESIGN_ARRIVAL_FRACTION,
    IMPACTOR_MASS,
    PLUG_MASS,
    SIM_SOLVE_DENSITY,
    arrival_fraction_for,
    bag_density,
    conductor_mass,
    confinement_field,
    full_bore_slug_ratio,
    mirror_stagnation,
    self_consistent_slug_ratio,
    shocked_sound_speed,
    swept_slug_ratio,
    two_term_nozzle_mass,
)


def test_full_bore_sweep_reproduces_the_papers_slug_ratio() -> None:
    """`rho A L / m` = 8.52, which is where the paper's 8.5 comes from.

    Exactly ``BAG_SLUG_MASS / IMPACTOR_MASS``, because the density and the
    swept volume are now the same bag: ADR 0029 derives ``BAG_VOLUME`` from the
    bore and the column instead of carrying the 5.4 m sphere's 659.6, which had
    the sweep returning 8.69 for a quantity whose definition is 213/25.
    """
    assert np.isclose(full_bore_slug_ratio(), 8.52, rtol=1e-9)
    assert np.isclose(full_bore_slug_ratio(), BAG_SLUG_MASS / IMPACTOR_MASS, rtol=1e-9)
    assert np.isclose(bag_density(), 0.3165, rtol=1e-3)
    # The companion sim's runs were solved at the old density and stay reachable.
    assert np.isclose(full_bore_slug_ratio(SIM_SOLVE_DENSITY), 8.692, rtol=1e-3)


def test_rigid_front_is_exactly_quadratic_in_arrival_radius() -> None:
    """The whole sensitivity: half the radius is a quarter of the slug."""
    for fraction in (0.5, 0.6, 0.8, 0.9, 1.0):
        assert np.isclose(
            swept_slug_ratio(fraction), full_bore_slug_ratio() * fraction**2, rtol=1e-9
        )
    assert np.isclose(swept_slug_ratio(0.5), 0.25 * swept_slug_ratio(1.0), rtol=1e-9)


def test_the_design_arrival_is_the_compact_aperture_not_a_wide_front() -> None:
    """``sec:needle_through_fog`` keeps the projectile compact, on purpose.

    A head-on nozzle is a mirror with a hole where the projectile entered, so a
    wide entry stops being a mirror.  0.15 m into 28 m^2 is 0.25% open; 0.8 of
    the bore would be 64% open.
    """
    assert np.isclose(DESIGN_ARRIVAL_FRACTION * BORE_RADIUS, 0.15, rtol=1e-9)
    open_fraction = (DESIGN_ARRIVAL_FRACTION * BORE_RADIUS) ** 2 / BORE_RADIUS**2
    assert np.isclose(open_fraction, 0.0025, rtol=1e-9)
    assert np.isclose(0.8**2, 0.64, rtol=1e-9)


def test_the_compact_needle_beats_the_wide_front_on_k_as_well_as_aperture() -> None:
    """It is not a compromise: 7.10 is inside ADR 0016's 6.75-7.77 optimum band.

    The wide front's 8.43 overshoots it, so the compact projectile wins on
    aperture, on slug ratio, and on chain growth at once.  Both figures fell
    2% with the adopted bag density (ADR 0029) and the verdict is unmoved --
    the compact arrival had 0.46 of band to spare and now has 0.67.
    """
    compact = self_consistent_slug_ratio(DESIGN_ARRIVAL_FRACTION)
    wide = self_consistent_slug_ratio(0.8)
    assert 6.75 <= compact <= 7.77
    assert wide > 7.77
    assert np.isclose(compact, 7.100, rtol=5e-3)


def test_wide_front_numbers_stay_reachable_for_comparison() -> None:
    """The rigid-front bound at 0.8/0.9, quoted in ADR 0016 addendum 1.

    The addendum's own 5.563 / 8.277 / 7.041 were computed at the 659.6 m^3
    density, and are still reachable through ``slug_density``; the flown bag
    delivers 2% less of each (ADR 0029).
    """
    assert np.isclose(swept_slug_ratio(0.8), 5.453, rtol=1e-3)
    assert np.isclose(swept_slug_ratio(0.8, 3.0), 8.111, rtol=2e-3)
    assert np.isclose(swept_slug_ratio(0.9), 6.901, rtol=1e-3)
    for fraction, expansion, published in ((0.8, 0.0, 5.563), (0.9, 0.0, 7.041)):
        assert np.isclose(
            swept_slug_ratio(fraction, expansion, slug_density=SIM_SOLVE_DENSITY),
            published,
            rtol=1e-3,
        )


def test_a_compact_ice_rod_falls_out_the_bottom_of_the_window() -> None:
    """The failure direction that is easy to get backwards.

    A compact projectile does not overload the plume, it fails to make one:
    ``k`` = 0.034 is below the ignition window's *lower* root (0.084 at
    45.58 km/s), not above its ceiling.
    """
    radius = (3.0 * (IMPACTOR_MASS / 917.0) / (4.0 * np.pi)) ** (1.0 / 3.0)
    assert np.isclose(radius, 0.1867, rtol=1e-3)
    delivered = swept_slug_ratio(radius / BORE_RADIUS)
    assert np.isclose(delivered, 0.0330, rtol=1e-2)
    assert delivered < 0.084


def test_spreading_makes_the_arrival_radius_nearly_irrelevant() -> None:
    """23.8 m of column is a long way to close a 1.2 m gap.

    This is why the parameter is only load-bearing for a *rigid* front, and why
    ``c_exp`` rather than the arrival radius is the thing neither repo solves.
    """
    rigid = swept_slug_ratio(0.6)
    spread = swept_slug_ratio(0.6, 5.0)
    assert rigid < 3.2
    assert spread > 7.5
    # Faster projectiles have less time to spread, so k falls with closing speed.
    assert swept_slug_ratio(0.6, 5.0, 75.0) < swept_slug_ratio(0.6, 5.0, 45.58)


def test_arrival_fraction_inverts_the_sweep() -> None:
    """The optimiser picks k; this says whether the geometry can supply it."""
    for k in (3.0, 5.453, 6.75, 8.52):
        assert np.isclose(swept_slug_ratio(arrival_fraction_for(k)), k, rtol=1e-9)
    # The tolled optimum needs 0.89, not the 0.97 that k = 8.5 demanded.
    assert np.isclose(arrival_fraction_for(6.75), 0.890, rtol=2e-3)


def test_arrival_fraction_reports_the_infeasible_case_rather_than_clipping() -> None:
    """Above 1.0 a rigid front cannot supply the ratio at all."""
    assert arrival_fraction_for(12.0) > 1.0


def test_rejects_a_nonsense_arrival_fraction() -> None:
    """Zero radius sweeps nothing; above the bore is not a bore."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            swept_slug_ratio(bad)


def test_shocked_front_is_hot_enough_to_widen_at_a_quarter_of_its_own_speed() -> None:
    """The number that retires the rigid front.

    A 45.58 km/s impact shocks the plume to ~94 600 K, whose sound speed is
    21.1 km/s -- very nearly half the closing speed, a 25 degree half-angle.
    A front that widens that fast reaches the bore wall almost immediately.
    """
    assert np.isclose(shocked_sound_speed(45.58), 21.1, rtol=1e-2)
    assert shocked_sound_speed(45.58) / 45.58 > 0.4


def test_sound_speed_rises_with_front_speed_and_clamps_outside_the_table() -> None:
    """Monotone, and it must not extrapolate off the ends of the provenance table."""
    speeds = [shocked_sound_speed(v) for v in (5.0, 15.0, 30.0, 45.0)]
    assert speeds == sorted(speeds)
    assert shocked_sound_speed(0.5) == shocked_sound_speed(2.0)
    assert shocked_sound_speed(500.0) == shocked_sound_speed(80.0)


def test_self_consistent_front_makes_the_arrival_radius_nearly_irrelevant() -> None:
    """The finding: 0.03-8.5 rigid collapses to 7.1-8.5 once the front may widen.

    Pinned against a direct run of ``puffsat_impact_simulation``'s ``eos_water``
    at commit ``0216a09``, which gave 7.240 and 8.601 for these two cases; the
    in-repo interpolation must track that solve, not merely be self-consistent.
    **That run was solved at the 659.6 m^3 bag**, so the pin is evaluated at
    :data:`SIM_SOLVE_DENSITY` rather than at the flown density ADR 0029
    adopted; re-solving it at 0.3165 is an ask on the companion repository.
    """
    compact = (3.0 * (IMPACTOR_MASS / 917.0) / (4.0 * np.pi)) ** (1.0 / 3.0)
    for fraction, published in ((compact / BORE_RADIUS, 7.240), (0.8, 8.601)):
        assert np.isclose(
            self_consistent_slug_ratio(fraction, slug_density=SIM_SOLVE_DENSITY),
            published,
            rtol=5e-3,
        )
    # At the flown density every figure falls by the density ratio, near enough.
    assert np.isclose(
        self_consistent_slug_ratio(compact / BORE_RADIUS), 7.100, rtol=5e-3
    )
    assert np.isclose(self_consistent_slug_ratio(0.8), 8.432, rtol=5e-3)
    # The whole spread across every plausible arrival radius is now under 1.5 in k.
    ratios = [self_consistent_slug_ratio(f) for f in (0.3, 0.5, 0.8, 1.0)]
    assert max(ratios) - min(ratios) < 1.5


def test_self_consistent_front_is_bounded_by_the_rigid_and_full_bore_cases() -> None:
    """It must beat a rigid front and never exceed sweeping the whole bag."""
    for fraction in (0.2, 0.5, 0.8):
        assert (
            swept_slug_ratio(fraction)
            < self_consistent_slug_ratio(fraction)
            <= full_bore_slug_ratio()
        )
    assert np.isclose(
        self_consistent_slug_ratio(1.0), full_bore_slug_ratio(), rtol=1e-9
    )


def test_the_delivered_ratio_overshoots_the_tolled_optimum() -> None:
    """The design consequence, and it points at the bag rather than the projectile.

    ADR 0016 puts the tolled chain optimum at k = 6.75-7.77.  A self-widening
    front at the design 0.8 delivers 8.60, so the flown bag carries *more* slug
    than the chain wants, and the fix is a smaller bag.
    """
    assert self_consistent_slug_ratio(0.8) > 7.77


def test_virial_floor_reproduces_both_of_the_papers_bands() -> None:
    """3.7-11 t at 4.43 GJ and 10-30 t at 12.2 GJ, which the paper quotes."""
    for energy, low, high in ((4.427e9, 3.69, 11.07), (12.15e9, 10.13, 30.38)):
        lo, hi, _ = two_term_nozzle_mass(energy)
        assert np.isclose(lo / 1000.0, low, rtol=1e-2)
        assert np.isclose(hi / 1000.0, high, rtol=1e-2)


def test_confinement_field_matches_the_bag_sizing_row() -> None:
    """``E_B = B^2 V / 2 mu0`` inverted, for the two bags that differ by 2%.

    ``tab:bag_sizing``'s 4.11 T is its 5.4 m *sphere*, 659.6 m^3.  The flown
    column holds 672.9 and therefore stands the same energy off with 4.07 T:
    ``B`` goes as ``V^-1/2``, so the 2% in volume is 1% in field.  ADR 0029
    made the module default the column, so the sphere is now asked for by name.
    """
    assert np.isclose(confinement_field(4.427e9, 659.6), 4.11, rtol=5e-3)
    assert np.isclose(confinement_field(4.427e9), 4.07, rtol=5e-3)


def test_conductor_term_is_not_negligible_against_the_structure_floor() -> None:
    """The finding of item 13: the paper quotes structure only.

    At 500 A the tape is 4.4 t against a 3.7 t optimistic structure floor, so
    the paper's low end is not reachable -- the two-term floor is ~8 t, not
    3.7 t.  It stays significant across the whole plausible tape-current band.
    """
    lo, _, conductor = two_term_nozzle_mass(4.427e9)
    assert conductor > lo
    assert np.isclose((lo + conductor) / 1000.0, 8.05, rtol=2e-2)
    for current, expected in ((300.0, 7.3), (1000.0, 2.2)):
        _, _, tape = two_term_nozzle_mass(4.427e9, tape_current=current)
        assert np.isclose(tape / 1000.0, expected, rtol=3e-2)


def test_conductor_grows_as_the_square_root_of_column_length() -> None:
    """Bore and conductor trade inversely, one for one.

    ``m_tape ~ B sqrt(V l / pi)`` while ``r ~ 1/sqrt(l)``, so lengthening the
    column 4.6x buys 2.15x the tape for 0.46x the bore -- the exact reciprocal
    pair the paper asserts.
    """
    field = confinement_field(4.427e9)
    short, long = conductor_mass(field, 10.8), conductor_mass(field, 50.0)
    assert np.isclose(long / short, np.sqrt(50.0 / 10.8), rtol=1e-9)
    assert np.isclose(long / short, 2.152, rtol=1e-2)


def test_structure_mass_is_indifferent_to_shape() -> None:
    """The virial floor tracks contained energy, which shape leaves alone."""
    for length in (10.8, 23.8, 50.0):
        lo, hi, _ = two_term_nozzle_mass(4.427e9, column_length=length)
        assert np.isclose(lo, two_term_nozzle_mass(4.427e9)[0], rtol=1e-12)
        assert np.isclose(hi, two_term_nozzle_mass(4.427e9)[1], rtol=1e-12)


def test_mirror_stagnation_reproduces_both_plug_positions() -> None:
    """6.7 / 1.26 GPa / 56 T at the ship end, 1.17 / 23 MPa / 7.6 T at the throat."""
    ratio, pressure, field = mirror_stagnation(PLUG_MASS, 28.27)
    assert np.isclose(ratio, 6.7, rtol=1e-2)
    assert np.isclose(pressure / 1e9, 1.26, rtol=2e-2)
    assert np.isclose(field, 56.0, rtol=2e-2)

    ratio, pressure, field = mirror_stagnation(213.0, BAG_VOLUME)
    assert np.isclose(ratio, 1.17, rtol=1e-2)
    assert np.isclose(pressure / 1e6, 23.0, rtol=2e-2)
    assert np.isclose(field, 7.6, rtol=2e-2)


def test_sweeping_more_mass_dissipates_more_energy_but_lowers_the_wall() -> None:
    """The result in one line, and it is not obvious.

    A throat-end plug lets the fireball snowplow the whole column first, which
    dissipates *more* energy -- but leaves it moving slower and spread through
    a far larger volume, and the ram term falls faster than the static term
    rises.  Seven times less field for more dissipated energy.
    """
    near = mirror_stagnation(PLUG_MASS, 28.27)
    far = mirror_stagnation(213.0, BAG_VOLUME)
    assert far[0] < near[0]
    assert far[2] < near[2] / 7.0 * 1.02


def test_momentum_is_conserved_through_the_merge() -> None:
    """``M v = m_p w`` is what sets the post-merge drift, at any swept mass."""
    for added in (10.0, 37.5, 213.0):
        merged = IMPACTOR_MASS + added
        drift = IMPACTOR_MASS * 56000.0 / merged
        assert np.isclose(merged * drift, IMPACTOR_MASS * 56000.0, rtol=1e-12)
