"""Tests for the ideal circular 2S/3S departure-impact audit."""

import numpy as np
import pytest
from astropy import units as u

from src.circular_resonance_impulse import (
    analyze_circular_resonant_impacts,
    free_aim_ceiling,
    head_on_crossover_slug_ratio,
    impulse_per_impactor_kg,
)

TWO_S_RETURN_VINF = 62.305352 * u.km / u.s
TWO_S_DEPARTURE_VINF = 13.822848 * u.km / u.s


@pytest.fixture(scope="module")
def circular_analysis():
    """Use a smaller bracketing grid; continuous refinement recovers the optima."""

    return analyze_circular_resonant_impacts(phase_samples=91, encounter_samples=121)


@pytest.mark.parametrize(
    ("angle_deg", "slug_ratio", "expected"),
    [
        (180.0, 3.0, np.sqrt(4.0) - 1.0),  # ADR 0009's head-on nozzle
        (0.0, 3.0, np.sqrt(4.0) + 1.0),  # the growth push's along-axis slug
        (90.0, 3.0, np.sqrt(3.0)),  # broadside: pure energy, no bias
        (180.0, 7.057, np.sqrt(8.057) - 1.0),
        (0.0, 10.0, np.sqrt(11.0) + 1.0),
    ],
)
def test_impulse_law_reproduces_both_named_geometries(angle_deg, slug_ratio, expected):
    """One law must collapse to the head-on and along-axis laws at its endpoints."""

    assert impulse_per_impactor_kg(angle_deg * u.deg, slug_ratio) == pytest.approx(
        expected, abs=1e-12
    )


def test_impulse_gain_is_second_order_in_the_cant():
    """Canting slightly off head-on must buy essentially nothing."""

    head_on = impulse_per_impactor_kg(180.0 * u.deg, 3.0)
    small = impulse_per_impactor_kg(175.0 * u.deg, 3.0)
    large = impulse_per_impactor_kg(150.0 * u.deg, 3.0)
    # 5 degrees of cant is worth 0.2%, and the gain grows as the square of the
    # cant: 30 degrees is 6x the angle, so ~36x the gain, not 6x.
    assert small / head_on == pytest.approx(1.0019, abs=1e-4)
    assert (small / head_on - 1.0) * 36.0 == pytest.approx(
        large / head_on - 1.0, rel=0.05
    )
    assert large / head_on < 1.08


def test_free_aim_ceiling_is_attained_at_a_perfectly_overtaking_impact():
    """The optimum is bang-bang; nothing between the endpoints wins."""

    ceiling, angle = free_aim_ceiling(TWO_S_RETURN_VINF, TWO_S_DEPARTURE_VINF, 3.0)
    assert angle.to_value(u.deg) == pytest.approx(0.0, abs=1e-9)
    assert ceiling == pytest.approx(1.1656, abs=2e-4)


@pytest.mark.parametrize(
    ("slug_ratio", "expected"),
    [(3.0, 1.1656), (7.057, 1.1051), (10.0, 1.0721)],
)
def test_the_free_aim_ceiling_shrinks_with_the_slug_ratio(slug_ratio, expected):
    """More slug dilutes the momentum bias, so the whole prize shrinks."""

    ceiling, _ = free_aim_ceiling(TWO_S_RETURN_VINF, TWO_S_DEPARTURE_VINF, slug_ratio)
    assert ceiling == pytest.approx(expected, abs=2e-4)


def test_head_on_becomes_optimal_above_the_crossover_slug_ratio():
    """Past the crossover the closing-speed loss beats the momentum-bias gain."""

    crossover = head_on_crossover_slug_ratio(TWO_S_RETURN_VINF, TWO_S_DEPARTURE_VINF)
    assert crossover == pytest.approx(18.47, abs=2e-2)
    below, angle_below = free_aim_ceiling(
        TWO_S_RETURN_VINF, TWO_S_DEPARTURE_VINF, crossover - 2.0
    )
    above, angle_above = free_aim_ceiling(
        TWO_S_RETURN_VINF, TWO_S_DEPARTURE_VINF, crossover + 2.0
    )
    assert angle_below.to_value(u.deg) == pytest.approx(0.0, abs=1e-9)
    assert below > 1.0
    assert angle_above.to_value(u.deg) == pytest.approx(180.0, abs=1e-9)
    assert above == pytest.approx(1.0, abs=1e-12)


@pytest.mark.slow
def test_circular_families_are_exact_repeating_unpowered_closures(
    circular_analysis,
):
    """Every reported selection must close its integer-synodic Jovian flyby."""

    for family in (
        circular_analysis.two_synodic,
        circular_analysis.three_synodic,
    ):
        for candidate in (
            family.minimum_departure,
            family.least_backward,
            family.maximum_delivered_mass,
        ):
            assert candidate.period.to_value(u.year) == pytest.approx(
                family.synodic_multiple
                * circular_analysis.synodic_period.to_value(u.year),
                abs=1e-12,
            )
            assert abs(candidate.jupiter_vinf_mismatch.to_value(u.km / u.s)) < 1e-8
            assert candidate.perijove_altitude.to_value(u.km) >= 4000.0 - 1e-5


@pytest.mark.slow
def test_phase_freedom_moves_the_3s_aim_but_not_the_2s_aim(circular_analysis):
    """3S can trade a hot departure for a forward-biased impact; 2S cannot."""

    two = circular_analysis.two_synodic
    three = circular_analysis.three_synodic
    assert two.minimum_departure.earth_return_vinf.to_value(
        u.km / u.s
    ) == pytest.approx(62.30535, abs=2e-4)
    assert three.minimum_departure.earth_return_vinf.to_value(
        u.km / u.s
    ) == pytest.approx(55.17834, abs=2e-4)
    # The whole 2S family stays pinned near head-on even at the 25 km/s cap.
    assert two.least_backward.aim_separation.to_value(u.deg) == pytest.approx(
        149.1777, abs=2e-3
    )
    assert two.least_backward.backward_projection_fraction > 0.85
    # 3S reaches a forward-biased impact, but only by paying the cap.
    assert three.least_backward.aim_separation.to_value(u.deg) == pytest.approx(
        55.4509, abs=2e-3
    )
    assert three.least_backward.backward_projection_fraction == 0.0
    assert three.least_backward.earth_departure_vinf.to_value(
        u.km / u.s
    ) == pytest.approx(
        circular_analysis.maximum_departure_vinf.to_value(u.km / u.s), abs=1e-8
    )


@pytest.mark.slow
def test_the_delivered_mass_optimum_is_a_near_baseline_departure(circular_analysis):
    """Scored on mass, the best buy is a few hundred m/s, not the cap."""

    for family, expected_gain, expected_burn in (
        (circular_analysis.two_synodic, 1.0140, 6.7325),
        (circular_analysis.three_synodic, 1.0293, 5.1548),
    ):
        best = family.maximum_delivered_mass
        assert best.earth_departure_burn.to_value(u.km / u.s) == pytest.approx(
            expected_burn, abs=2e-3
        )
        assert best.ledger.gain_over_head_on == pytest.approx(expected_gain, abs=5e-4)
        # The metric it replaced would have picked the far end of the trade curve.
        assert (
            best.earth_departure_vinf < family.least_backward.earth_departure_vinf
        ) or family.synodic_multiple == 2


@pytest.mark.slow
def test_the_realized_gain_stays_far_under_the_free_aim_ceiling(circular_analysis):
    """The searched optimum must not approach the no-charge upper bound."""

    for family in (
        circular_analysis.two_synodic,
        circular_analysis.three_synodic,
    ):
        realized = family.maximum_delivered_mass.ledger.gain_over_head_on
        assert 1.0 < realized < family.free_aim_ceiling
        assert realized - 1.0 < 0.30 * (family.free_aim_ceiling - 1.0)
    assert circular_analysis.two_synodic.free_aim_ceiling == pytest.approx(
        1.1656, abs=2e-4
    )
    assert circular_analysis.three_synodic.free_aim_ceiling == pytest.approx(
        1.1312, abs=2e-4
    )
    assert circular_analysis.two_synodic.head_on_crossover_slug_ratio == pytest.approx(
        18.47, abs=2e-2
    )
    assert (
        circular_analysis.three_synodic.head_on_crossover_slug_ratio
        == pytest.approx(16.43, abs=2e-2)
    )


@pytest.mark.slow
def test_the_free_mirror_is_worth_more_than_the_searched_angle(circular_analysis):
    """Half the Earth turn is free and larger than what delta-v buys."""

    for family in (
        circular_analysis.two_synodic,
        circular_analysis.three_synodic,
    ):
        baseline = family.minimum_departure
        rotation = abs(baseline.ledger.mirror_rotation.to_value(u.deg))
        assert 13.0 < rotation < 19.0
        # The rotation must reduce the head-on bias, not add to it.
        assert baseline.ledger.aim_at_burn.to_value(
            u.deg
        ) < baseline.aim_separation.to_value(u.deg)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maximum_departure_vinf": 0.0 * u.km / u.s}, "must be positive"),
        ({"phase_samples": 8}, "phase_samples must be at least 9"),
        ({"encounter_samples": 8}, "encounter_samples must be at least 9"),
        ({"slug_ratio": 0.0}, "slug_ratio must be positive"),
        ({"collimation": -1.0}, "collimation must be positive"),
    ],
)
def test_circular_resonance_rejects_invalid_search_inputs(kwargs, message):
    """Invalid physical bounds and grids should fail before starting a search."""

    with pytest.raises(ValueError, match=message):
        analyze_circular_resonant_impacts(**kwargs)
