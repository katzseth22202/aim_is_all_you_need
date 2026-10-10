"""Tests for Earth occultation and the finite 3S departure burn."""

import pytest
from astropy import units as u

from src.orbit_turn_analysis import (
    UNCANTED_EXHAUST_SPEED,
    UNCANTED_IDEAL_EXHAUST_SPEED,
    UNCANTED_MIXED_SPECIFIC_ENERGY,
    blocked_mirror_clearance_angle,
    finite_burn_turn,
    occultation_half_angle,
    three_synodic_periapsis_mirrors,
    uncanted_slug_ratio,
    uncanted_thermal_burn,
)


@pytest.fixture(scope="module")
def turn_results():
    """Integrate each 20-minute thrust policy once for the module."""

    return finite_burn_turn(steered=False), finite_burn_turn(steered=True)


@pytest.fixture(scope="module")
def uncanted_results():
    """Optimize each 3000 K axial-nozzle mirror once for the module."""

    return uncanted_thermal_burn(front_side=False), uncanted_thermal_burn(
        front_side=True
    )


@pytest.fixture(scope="module")
def miss_safe_results():
    """Optimize both mirrors with a 600 km missed-projectile floor."""

    return uncanted_thermal_burn(
        front_side=False, missed_periapsis_floor=600.0 * u.km
    ), uncanted_thermal_burn(front_side=True, missed_periapsis_floor=600.0 * u.km)


@pytest.fixture(scope="module")
def long_miss_safe_results():
    """Optimize both miss-safe mirrors for a lower-thrust 2,500 s burn."""

    return uncanted_thermal_burn(
        front_side=False,
        burn_time=2500.0 * u.s,
        missed_periapsis_floor=600.0 * u.km,
    ), uncanted_thermal_burn(
        front_side=True,
        burn_time=2500.0 * u.s,
        missed_periapsis_floor=600.0 * u.km,
    )


def test_earth_occults_the_apparently_favorable_departure_mirror():
    """The impulse-only ledger's canted mirror passes through Earth."""

    visible, blocked = three_synodic_periapsis_mirrors()
    assert visible.angular_momentum_sign == 1
    assert not visible.blocked
    assert visible.projectile_periapsis_altitude.to_value(u.km) == pytest.approx(
        234.9, abs=0.1
    )
    assert visible.impact_angle.to_value(u.deg) == pytest.approx(164.46, abs=0.01)
    assert visible.exhaust_cant.to_value(u.deg) == pytest.approx(7.70, abs=0.01)

    assert blocked.angular_momentum_sign == -1
    assert blocked.blocked
    assert blocked.projectile_periapsis_altitude.to_value(u.km) == pytest.approx(
        -1904.6, abs=0.1
    )
    assert blocked.exhaust_cant.to_value(u.deg) == pytest.approx(19.68, abs=0.01)


def test_the_block_is_not_a_limb_rounding_error():
    """The rejected stream needs a 27-degree asymptote change to graze Earth."""

    assert occultation_half_angle().to_value(u.deg) == pytest.approx(66.07, abs=0.01)
    assert blocked_mirror_clearance_angle().to_value(u.deg) == pytest.approx(
        27.19, abs=0.01
    )


def test_fixed_direction_burn_is_centered_and_never_reaches_22_degrees(
    turn_results,
):
    """The chamber-compatible burn is symmetric on reference periapsis."""

    fixed, _ = turn_results
    assert fixed.seconds_before_reference_periapsis.to_value(u.s) == pytest.approx(
        599.7, abs=0.2
    )
    assert fixed.seconds_after_reference_periapsis.to_value(u.s) == pytest.approx(
        600.3, abs=0.2
    )
    assert fixed.finite_burn_loss.to_value(u.m / u.s) == pytest.approx(273.3, abs=0.2)
    assert fixed.timing_gain.to_value(u.m / u.s) == pytest.approx(0.0, abs=0.001)
    assert fixed.off_head_on_maximum.to_value(u.deg) == pytest.approx(19.81, abs=0.02)
    assert fixed.time_above_22_degrees == 0.0
    assert fixed.exhaust_cant_maximum.to_value(u.deg) == pytest.approx(9.76, abs=0.02)
    assert fixed.angle_gain_over_head_on == pytest.approx(1.00298, abs=2e-5)


def test_steering_shifts_the_optimum_seconds_not_minutes(turn_results):
    """Velocity steering moves reference centering 18 s and saves 0.2 m/s."""

    _, steered = turn_results
    assert steered.seconds_before_reference_periapsis.to_value(u.s) == pytest.approx(
        582.5, abs=0.2
    )
    assert steered.seconds_after_reference_periapsis.to_value(u.s) == pytest.approx(
        617.5, abs=0.2
    )
    assert steered.seconds_before_closest_approach.to_value(u.s) == pytest.approx(
        474.0, abs=1.1
    )
    assert steered.seconds_after_closest_approach.to_value(u.s) == pytest.approx(
        726.0, abs=1.1
    )
    assert steered.finite_burn_loss.to_value(u.m / u.s) == pytest.approx(211.2, abs=0.2)
    assert steered.timing_gain.to_value(u.m / u.s) == pytest.approx(0.16, abs=0.01)


def test_even_steered_thrust_is_above_22_degrees_only_at_the_end(turn_results):
    """Orbit turning is real but leaves the plume-cant and mass effect small."""

    _, steered = turn_results
    assert steered.off_head_on_start.to_value(u.deg) == pytest.approx(3.32, abs=0.02)
    assert steered.off_head_on_end.to_value(u.deg) == pytest.approx(23.16, abs=0.02)
    assert steered.time_above_22_degrees == pytest.approx(0.119, abs=0.002)
    assert steered.exhaust_cant_maximum.to_value(u.deg) == pytest.approx(
        11.34, abs=0.02
    )
    assert steered.exhaust_cant_impulse_mean.to_value(u.deg) == pytest.approx(
        6.74, abs=0.02
    )
    assert steered.angle_gain_over_head_on == pytest.approx(1.00274, abs=2e-5)


def test_3000_k_loading_uses_85_percent_of_ideal_exhaust_velocity():
    """The requested efficiency is a velocity multiplier, not an energy share."""

    assert UNCANTED_MIXED_SPECIFIC_ENERGY.to_value(u.MJ / u.kg) == pytest.approx(
        57.3278, abs=0.0001
    )
    assert UNCANTED_IDEAL_EXHAUST_SPEED.to_value(u.km / u.s) == pytest.approx(
        10.7077, abs=0.0001
    )
    assert UNCANTED_EXHAUST_SPEED.to_value(u.km / u.s) == pytest.approx(
        9.1016, abs=0.0001
    )
    assert uncanted_slug_ratio(45.0 * u.km / u.s) == pytest.approx(16.6616)
    assert uncanted_slug_ratio(73.0 * u.km / u.s) == pytest.approx(45.4784, abs=0.0001)


@pytest.mark.slow
def test_uncanted_momentum_closes_the_visible_3s_departure(uncanted_results):
    """The small projectile share gives a visible, vector-closed 600 km burn."""

    visible, _ = uncanted_results
    assert visible.reference_periapsis_altitude.to_value(u.km) == pytest.approx(600.0)
    assert visible.seconds_before_reference_periapsis.to_value(u.s) == pytest.approx(
        583.0, abs=0.2
    )
    assert visible.slug_ratio_minimum == pytest.approx(36.79, abs=0.01)
    assert visible.slug_ratio_maximum == pytest.approx(40.25, abs=0.01)
    assert visible.hydrogen_spent_fraction == pytest.approx(0.50083, abs=1e-5)
    assert visible.delivered_fraction == pytest.approx(0.49917, abs=1e-5)
    assert visible.initial_to_delivered_mass_ratio == pytest.approx(2.00331, abs=1e-5)
    assert visible.impactor_mass_fraction == pytest.approx(0.01275, abs=1e-5)
    assert visible.closest_altitude.to_value(u.km) == pytest.approx(706.5, abs=0.2)
    assert visible.integrated_lateral_delta_v.to_value(u.km / u.s) == pytest.approx(
        0.2743, abs=0.0002
    )
    assert visible.integrated_thrust_delta_v.to_value(u.km / u.s) == pytest.approx(
        5.33935, abs=0.00002
    )
    assert visible.impulsive_delta_v.to_value(u.km / u.s) == pytest.approx(
        5.11887, abs=0.00002
    )
    assert visible.finite_burn_penalty.to_value(u.m / u.s) == pytest.approx(
        220.48, abs=0.05
    )
    assert visible.missed_projectile_periapsis_altitude.to_value(u.km) == pytest.approx(
        -3103.7, abs=0.2
    )
    assert visible.missed_projectile_earth_impact_fraction == pytest.approx(
        0.4456, abs=0.0002
    )
    assert visible.outgoing_vinf.to_value(u.km / u.s) == pytest.approx(
        11.56427943, abs=1e-7
    )
    assert visible.outgoing_angle_error.to_value(u.deg) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.slow
def test_raising_periapsis_to_open_the_front_side_loses_mass(uncanted_results):
    """The less-head-on mirror clears only after giving away too much Oberth gain."""

    visible, front = uncanted_results
    assert front.reference_periapsis_altitude.to_value(u.km) == pytest.approx(
        3866.3, abs=0.2
    )
    assert front.seconds_before_reference_periapsis.to_value(u.s) == pytest.approx(
        0.0, abs=0.01
    )
    assert front.minimum_projectile_clearance.to_value(u.km) == pytest.approx(
        0.0, abs=0.01
    )
    assert front.slug_ratio_minimum == pytest.approx(32.25, abs=0.01)
    assert front.slug_ratio_maximum == pytest.approx(37.97, abs=0.01)
    assert front.off_head_on_maximum.to_value(u.deg) == pytest.approx(44.82, abs=0.02)
    assert front.integrated_lateral_delta_v.to_value(u.km / u.s) == pytest.approx(
        0.8094, abs=0.0002
    )
    assert front.delivered_fraction / visible.delivered_fraction == pytest.approx(
        0.9354, abs=0.0002
    )


@pytest.mark.slow
def test_a_600_km_missed_projectile_floor_moves_the_optimum(miss_safe_results):
    """Fail-safe targeting costs mass but still favors the original visible side."""

    visible, front = miss_safe_results
    assert visible.reference_periapsis_altitude.to_value(u.km) == pytest.approx(
        2670.0, abs=0.2
    )
    assert visible.seconds_before_reference_periapsis.to_value(u.s) == pytest.approx(
        1200.0, abs=0.01
    )
    assert visible.closest_altitude.to_value(u.km) == pytest.approx(3082.2, abs=0.2)
    assert visible.missed_projectile_periapsis_altitude.to_value(u.km) == pytest.approx(
        600.0, abs=0.01
    )
    assert visible.missed_projectile_earth_impact_fraction == pytest.approx(0.0)
    assert visible.integrated_thrust_delta_v.to_value(u.km / u.s) == pytest.approx(
        5.79806, abs=0.00002
    )
    assert visible.finite_burn_penalty.to_value(u.m / u.s) == pytest.approx(
        221.83, abs=0.05
    )
    assert visible.hydrogen_spent_fraction == pytest.approx(0.53136, abs=1e-5)
    assert visible.delivered_fraction == pytest.approx(0.46864, abs=1e-5)
    assert visible.initial_to_delivered_mass_ratio == pytest.approx(2.13383, abs=1e-5)

    assert front.reference_periapsis_altitude.to_value(u.km) == pytest.approx(
        4535.4, abs=0.2
    )
    assert front.missed_projectile_periapsis_altitude.to_value(u.km) == pytest.approx(
        600.0, abs=0.01
    )
    assert front.delivered_fraction == pytest.approx(0.46151, abs=1e-5)
    assert visible.delivered_fraction > front.delivered_fraction


@pytest.mark.slow
def test_lower_thrust_reverses_the_miss_safe_mirror_choice(long_miss_safe_results):
    """At 2,500 s the safe front side narrowly beats the visible-side family."""

    visible, front = long_miss_safe_results
    assert visible.reference_periapsis_altitude.to_value(u.km) == pytest.approx(
        4640.0, abs=0.2
    )
    assert visible.delivered_fraction == pytest.approx(0.43481, abs=1e-5)

    assert front.reference_periapsis_altitude.to_value(u.km) == pytest.approx(
        5566.2, abs=0.2
    )
    assert front.seconds_before_reference_periapsis.to_value(u.s) == pytest.approx(
        0.0, abs=0.01
    )
    assert front.missed_projectile_periapsis_altitude.to_value(u.km) == pytest.approx(
        600.0, abs=0.01
    )
    assert front.hydrogen_spent_fraction == pytest.approx(0.56140, abs=1e-5)
    assert front.impactor_mass_fraction == pytest.approx(0.01639, abs=1e-5)
    assert (
        front.impactor_mass_fraction / front.hydrogen_spent_fraction
        == pytest.approx(0.02919, abs=1e-5)
    )
    assert front.integrated_thrust_delta_v.to_value(u.km / u.s) == pytest.approx(
        6.52635, abs=0.00002
    )
    assert front.finite_burn_penalty.to_value(u.m / u.s) == pytest.approx(
        458.29, abs=0.05
    )
    assert front.delivered_fraction == pytest.approx(0.43860, abs=1e-5)
    assert front.delivered_fraction > visible.delivered_fraction


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"burn_time": 0.0 * u.s}, "burn_time must be positive"),
        ({"slug_ratio": 0.0}, "slug_ratio must be positive"),
        ({"collimation": 0.0}, "collimation must be positive"),
    ],
)
def test_finite_burn_turn_rejects_nonphysical_inputs(kwargs, message):
    """Reject invalid inputs before starting the integration."""

    with pytest.raises(ValueError, match=message):
        finite_burn_turn(steered=False, **kwargs)
