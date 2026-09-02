"""Tests for src/sep_split_correction.py.

The fast tests protect structure and the closed-form pieces -- the propellant
and array accounting, the acceleration law, and the impulse quadrature, which
has an exact answer on a circular orbit.  The slow ones fly the chains, which
is where the ADR 0026 figures come from; they cost about 40 s, not the minutes
the module's CLI takes, because the fixture flies each cadence once.
"""

import numpy as np
import pytest

from src.real_orbit_resonance import _MU_SUN, _ManeuverSolution
from src.sep_split_correction import (
    ARGON_TANK_FRACTION,
    CADENCE_POLICIES,
    DEFAULT_SPLIT_DAYS,
    METHALOX_TANK_FRACTION,
    SEP_ARRAY_POWER_1AU_KW,
    SEP_SPECIFIC_MASS_KG_PER_KW,
    SEP_THRUSTER_EFFICIENCY,
    STATED_ACCELERATION_1AU,
    STATED_ACCELERATION_JUPITER,
    VE_ARGON,
    CorrectionCycle,
    CycleFeasibility,
    LegImpulse,
    analyze_sep_split,
    array_mass_fraction,
    characteristic_acceleration,
    correction_cycles_to_two_wave,
    cycle_feasibility,
    deliverable_dv,
    delivered_fraction,
    fly_correction_chain,
    leg_impulse_per_tonne,
    price_correction_chain,
    required_characteristic_acceleration,
    required_power_kw,
)
from src.two_wave_growth import VE_METHALOX

_AU_KM = 149_597_870.7


def _solution(case="dsm_only", dsm=0.0, total=0.0):
    """A minimal maneuver solution; only the charged fields matter here."""
    return _ManeuverSolution(
        case=case,
        encounter_jd=2_461_000.0,
        perijove_burn=0.0,
        dsm=dsm,
        total_dv=total,
        perijove_radius=75_492.0,
        earth_departure_vinf=14.0,
        earth_departure_periapsis_speed=18.0,
        earth_return_vinf=60.0,
        earth_return_collision_speed=62.0,
        incoming_vinf=15.0,
        outgoing_vinf=15.0,
        turn_angle=1.0,
    )


def _feasibility(split_dv=2.089377, dsm_dv=0.018479, whole=64884.2, beyond=14313.2):
    """A cycle whose leg budgets are pinned, so the arithmetic is checkable."""
    cycle = CorrectionCycle(
        index=0,
        departure_jd=2_461_000.0,
        return_jd=2_461_800.0,
        synodic_multiple=2,
        period_years=2.184,
        forced_by_split=False,
        nozzle=_solution(dsm=dsm_dv, total=dsm_dv),
        growth=_solution(total=split_dv),
        growth_dsm_only=_solution(total=split_dv),
        split_days=20.0,
    )
    leg = lambda dv: LegImpulse(dv, 365.0, 8.4, 1.0)
    limits = [beyond / (1000.0 * dsm_dv), whole / (1000.0 * split_dv)]
    return CycleFeasibility(
        cycle=cycle,
        dsm_leg=leg(beyond),
        split_legs=(leg(whole / 2.0), leg(whole / 2.0)),
        wave_tonnes=min(limits),
    )


def _circular_state(radius_au: float):
    """Position and velocity of a circular heliocentric orbit."""
    radius = radius_au * _AU_KM
    speed = float(np.sqrt(_MU_SUN / radius))
    return (
        np.array([radius, 0.0, 0.0]),
        np.array([0.0, speed, 0.0]),
    )


# --------------------------------------------------------------- fast tests


def test_argon_is_the_repo_constant_not_a_new_one():
    """The SEP Isp comes from ARGON_SEP_ISP, so it cannot drift from Leg 2."""
    assert VE_ARGON == pytest.approx(19.6133, abs=1e-3)
    assert VE_ARGON / VE_METHALOX == pytest.approx(5.263, abs=1e-3)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown cadence policy"):
        fly_correction_chain("hopeful")


def test_unknown_propellant_and_growth_case_are_rejected():
    with pytest.raises(ValueError, match="unknown propellant"):
        correction_cycles_to_two_wave([], propellant="hydrazine")
    with pytest.raises(ValueError, match="unknown growth_case"):
        correction_cycles_to_two_wave([], growth_case="wishful")


def test_impulse_on_a_circular_orbit_matches_the_closed_form():
    """At fixed radius the array is constant, so the integral is F * t.

    This is the one geometry where the quadrature has an exact answer, which
    makes it the check on the ``2 eta P / v_e`` conversion itself.
    """
    position, velocity = _circular_state(1.0)
    duration = 100.0 * 86_400.0
    leg = leg_impulse_per_tonne(position, velocity, duration)
    thrust_n = (
        2.0
        * SEP_THRUSTER_EFFICIENCY
        * SEP_ARRAY_POWER_1AU_KW
        * 1000.0
        / (VE_ARGON * 1000.0)
    )
    assert leg.dv_per_tonne == pytest.approx(thrust_n * duration / 1000.0, rel=1e-4)
    assert leg.mean_power_kw == pytest.approx(SEP_ARRAY_POWER_1AU_KW, rel=1e-4)
    assert leg.min_radius_au == pytest.approx(1.0, rel=1e-4)
    assert leg.days_counted == pytest.approx(100.0, rel=1e-6)


def test_impulse_scales_as_inverse_square_with_radius():
    """Twice the radius is a quarter the power, hence a quarter the impulse."""
    duration = 100.0 * 86_400.0
    near = leg_impulse_per_tonne(*_circular_state(1.0), duration)
    far = leg_impulse_per_tonne(*_circular_state(2.0), duration)
    assert far.dv_per_tonne == pytest.approx(near.dv_per_tonne / 4.0, rel=1e-3)


def test_the_deploy_radius_excludes_the_inner_stretch():
    """A wave inside the deploy radius may not correct, so nothing is counted."""
    position, velocity = _circular_state(1.0)
    leg = leg_impulse_per_tonne(position, velocity, 100.0 * 86_400.0, beyond_au=2.0)
    assert leg.dv_per_tonne == 0.0
    assert leg.days_counted == 0.0
    assert leg.mean_power_kw == 0.0


def test_a_zero_length_leg_is_rejected():
    position, velocity = _circular_state(1.0)
    with pytest.raises(ValueError, match="duration_s must be positive"):
        leg_impulse_per_tonne(position, velocity, 0.0)


def test_tankage_is_charged_against_both_propellants():
    """Charging argon's tanks while ignoring methalox's would flatter argon."""
    correction = 2.10786
    for propellant, exhaust, tank in (
        ("argon", VE_ARGON, ARGON_TANK_FRACTION),
        ("methalox", VE_METHALOX, METHALOX_TANK_FRACTION),
    ):
        kept = float(np.exp(-correction / exhaust))
        assert delivered_fraction(correction, propellant) == pytest.approx(
            kept - tank * (1.0 - kept)
        )
        assert delivered_fraction(correction, propellant) < kept


def test_the_array_shrinks_as_a_fraction_of_a_bigger_wave():
    """The array is a fixed mass, so a heavier wave hides it better."""
    light = delivered_fraction(2.10786, "argon", wave_tonnes=5.0)
    heavy = delivered_fraction(2.10786, "argon", wave_tonnes=20.0)
    assert heavy > light
    assert light < delivered_fraction(2.10786, "argon")


def test_a_wave_lighter_than_its_own_array_delivers_nothing():
    """The accounting is allowed to go negative rather than silently clamp."""
    assert delivered_fraction(2.10786, "argon", wave_tonnes=0.5) < 0.0


def test_delivered_fraction_rejects_an_unknown_propellant():
    with pytest.raises(ValueError, match="unknown propellant"):
        delivered_fraction(1.0, "hydrazine")


def test_a_negative_correction_is_rejected():
    with pytest.raises(ValueError, match="correction_dv must be nonnegative"):
        delivered_fraction(-1.0, "argon")


def test_characteristic_acceleration_matches_the_thrust_law():
    """``a = 2 eta P / (m v_e)`` -- the check on the whole power conversion."""
    accel = characteristic_acceleration(100.0, 500.0)
    assert accel == pytest.approx(
        2.0 * SEP_THRUSTER_EFFICIENCY * 100.0e3 / (500.0e3 * VE_ARGON * 1e3)
    )
    assert accel == pytest.approx(1.020e-5, rel=1e-3)


def test_the_stated_operating_point_needs_twice_the_stated_power():
    """2e-5 m/s^2 on a 500 t wave is a 196 kW array, not a 100 kW one.

    Pinned because the design's power figure and its acceleration figure were
    quoted independently and do not agree; see ADR 0026.
    """
    assert characteristic_acceleration(196.0, 500.0) == pytest.approx(2e-5, rel=2e-3)
    assert characteristic_acceleration(100.0, 500.0) < 1.1e-5


def test_acceleration_is_linear_in_power_and_inverse_in_mass():
    base = characteristic_acceleration(100.0, 500.0)
    assert characteristic_acceleration(200.0, 500.0) == pytest.approx(2.0 * base)
    assert characteristic_acceleration(100.0, 1000.0) == pytest.approx(base / 2.0)


def test_a_massless_wave_is_rejected():
    with pytest.raises(ValueError, match="wave_tonnes must be positive"):
        characteristic_acceleration(100.0, 0.0)


# --------------------------------------------------------------- slow tests


@pytest.fixture(scope="module")
def chains():
    """The three flown cadences at the 20-day split."""
    return {policy: fly_correction_chain(policy) for policy in CADENCE_POLICIES}


@pytest.mark.slow
def test_the_adaptive_cadence_reproduces_adr_0013(chains):
    """The baseline chain is ADR 0013's: 11 cycles, 7 x 2S + 4 x 3S."""
    cycles = chains["adaptive"]
    assert len(cycles) == 11
    assert sum(1 for c in cycles if c.synodic_multiple == 2) == 7
    assert sum(1 for c in cycles if c.synodic_multiple == 3) == 4
    assert sum(c.period_years for c in cycles) == pytest.approx(28.3930, abs=1e-3)
    burns = [c.growth_dsm_only.total_dv for c in cycles]
    assert float(np.mean(burns)) == pytest.approx(0.46406, abs=1e-4)


@pytest.mark.slow
def test_the_expensive_splits_are_bend_limited_at_the_perijove_floor(chains):
    """Three cycles pay km/s, and each is pinned at the 4,000 km floor.

    The residue is an *angle*, not a speed mismatch: no perijove change can buy
    it, which is why the powered-flyby cases return nothing for these windows.
    """
    dear = [c for c in chains["adaptive"] if c.growth.total_dv > 1.0]
    assert len(dear) == 3
    for cycle in dear:
        assert cycle.bend_limited
        assert cycle.growth.case == "dsm_only"
        assert cycle.growth.perijove_radius / 71_492.0 == pytest.approx(
            1.05595, abs=1e-4
        )


@pytest.mark.slow
def test_the_worst_total_correction_is_two_point_one_km_s(chains):
    """The headline delta-v: both burns, worst cycle, adaptive cadence."""
    worst = max(c.correction_total for c in chains["adaptive"])
    assert worst == pytest.approx(2.10786, abs=1e-3)


@pytest.mark.slow
def test_the_split_aware_cadence_removes_the_tail_and_costs_the_clock(chains):
    """It buys sub-m/s corrections with two fewer cycles -- a losing trade."""
    aware = chains["split_aware"]
    assert max(c.correction_total for c in aware) < 0.002
    assert len(aware) < len(chains["adaptive"])
    methalox = {
        policy: price_correction_chain(cycles).doubling
        for policy, cycles in chains.items()
    }
    assert methalox["split_aware"] > methalox["adaptive"]


@pytest.mark.slow
def test_the_powered_flyby_is_worth_almost_nothing(chains):
    """Enabling perijove_only cuts the median burn but not the growth.

    This is the measurement that retires the suspicion that
    ``two_wave_growth`` asking only for ``dsm_only`` left growth on the table.
    """
    cycles = chains["adaptive"]
    dsm = price_correction_chain(cycles, growth_case="dsm_only")
    best = price_correction_chain(cycles, growth_case="cheapest")
    assert best.total_growth > dsm.total_growth
    assert best.total_growth / dsm.total_growth < 1.002


@pytest.mark.slow
def test_argon_pays_the_tail_that_methalox_cannot(chains):
    """The correction's currency is the largest of the three effects."""
    cycles = chains["adaptive"]
    methalox = price_correction_chain(cycles, propellant="methalox")
    argon = price_correction_chain(cycles, propellant="argon")
    assert argon.doubling < methalox.doubling
    assert argon.total_growth / methalox.total_growth > 2.0


@pytest.mark.slow
def test_argon_does_nothing_for_a_chain_that_had_no_tail(chains):
    """The split-aware chain's burns are already sub-m/s, so there is no gain."""
    cycles = chains["split_aware"]
    methalox = price_correction_chain(cycles, propellant="methalox")
    argon = price_correction_chain(cycles, propellant="argon")
    assert argon.doubling == pytest.approx(methalox.doubling, rel=1e-3)


@pytest.mark.slow
def test_the_array_is_not_the_binding_constraint(chains):
    """Every cycle can drive a wave far heavier than the array itself."""
    for cycle in chains["adaptive"]:
        feasible = cycle_feasibility(cycle)
        assert feasible.wave_tonnes > 25.0
        assert feasible.dsm_leg.days_counted > 300.0


@pytest.mark.slow
def test_the_split_burn_binds_before_the_deploy_constrained_dsm(chains):
    """The 2 AU deployment window costs nothing: the DSM it confines is tiny."""
    worst = max(chains["adaptive"], key=lambda c: c.correction_total)
    feasible = cycle_feasibility(worst)
    dsm_limit = feasible.dsm_leg.dv_per_tonne / (1000.0 * worst.nozzle.dsm)
    assert dsm_limit > 10.0 * feasible.wave_tonnes


@pytest.mark.slow
def test_the_analysis_tables_line_up_with_the_flown_chains():
    """`analyze_sep_split` reports every policy and both propellants."""
    analysis = analyze_sep_split()
    assert set(analysis.cycles.cadence) == set(CADENCE_POLICIES)
    assert set(analysis.feasibility.cadence) == set(CADENCE_POLICIES)
    # two growth cases x two propellants, plus one array-charged row per
    # specific mass -- the array rows are argon only, since methalox carries
    # no hardware to charge.
    assert set(analysis.chains.propellant) == {"methalox", "argon", "argon+array"}
    assert len(analysis.chains) == len(CADENCE_POLICIES) * (
        4 + len(SEP_SPECIFIC_MASS_KG_PER_KW)
    )
    assert (analysis.break_even.array_fraction > 0).all()

    # the requirement table is the one the decision rests on
    requirements = analysis.requirements.set_index("cadence")
    assert set(requirements.index) == set(CADENCE_POLICIES)
    assert (requirements.split_available_m_s > 0).all()
    assert requirements.loc["adaptive", "split_shortfall"] > 8.0
    assert requirements.loc["always_2s", "dsm_shortfall"] > 30.0
    assert requirements.loc["split_aware", "dsm_shortfall"] < 0.05
    # power is the array's rating at 1 AU, so the Jupiter-side thrust is the
    # 1 AU thrust after the 1/r^2 falloff and nothing else
    for row in analysis.requirements.itertuples():
        assert row.thrust_n_at_jupiter == pytest.approx(row.thrust_n_at_1au / 5.2**2)


@pytest.mark.slow
def test_the_split_gap_is_the_parking_orbit_period(chains):
    """Both waves are one parking orbit apart, and that is the paper's 20 days."""
    for cycle in chains["adaptive"]:
        assert cycle.split_days == DEFAULT_SPLIT_DAYS
        assert cycle.return_jd - (cycle.return_jd - cycle.split_days) == pytest.approx(
            20.0
        )


@pytest.mark.slow
def test_the_array_fraction_does_not_depend_on_wave_mass(chains):
    """Power scales with mass, so the hardware is a fixed *percentage*.

    This is what retires the earlier "break-even wave mass" framing: sizing the
    wave up cannot make the array cheap, because the array grows with it.
    """
    worst = max(
        (cycle_feasibility(c) for c in chains["adaptive"]),
        key=array_mass_fraction,
    )
    fraction = array_mass_fraction(worst)
    for wave_tonnes in (5.0, 50.0, 500.0):
        power = required_power_kw(worst, wave_tonnes)
        assert power * 15.0 / (wave_tonnes * 1000.0) == pytest.approx(fraction)


@pytest.mark.slow
def test_the_stated_acceleration_cannot_buy_the_expensive_corrections(chains):
    """At 2e-5 m/s^2 the km/s burns are out of reach by nearly an order.

    The cheap corrections are comfortably affordable, which is why the
    split-aware cadence is the only SEP-compatible one at that power.
    """
    stated = 2e-5
    worst = max(
        (cycle_feasibility(c) for c in chains["adaptive"]),
        key=required_characteristic_acceleration,
    )
    assert required_characteristic_acceleration(worst) > 8.0 * stated
    aware = max(
        (cycle_feasibility(c) for c in chains["split_aware"]),
        key=required_characteristic_acceleration,
    )
    assert required_characteristic_acceleration(aware) < stated / 10.0


def test_deliverable_dv_rescales_linearly_with_acceleration():
    """Thrust is linear in power, so the budget is linear in acceleration."""
    feasible = _feasibility()
    reference = characteristic_acceleration(SEP_ARRAY_POWER_1AU_KW, 1.0)
    whole, beyond = deliverable_dv(feasible, reference)
    assert whole == pytest.approx(64884.2, rel=1e-6)
    assert beyond == pytest.approx(14313.2, rel=1e-6)
    half_whole, half_beyond = deliverable_dv(feasible, reference / 2.0)
    assert half_whole == pytest.approx(whole / 2.0)
    assert half_beyond == pytest.approx(beyond / 2.0)


def test_the_stated_point_is_short_by_about_eight_on_the_worst_split():
    """The headline shortfall, pinned: 2.089 km/s wanted, ~255 m/s available."""
    whole, beyond = deliverable_dv(_feasibility(), STATED_ACCELERATION_1AU)
    assert whole == pytest.approx(254.5, rel=2e-3)
    assert beyond == pytest.approx(56.1, rel=2e-3)
    assert 2089.377 / whole == pytest.approx(8.2, rel=0.02)


def test_the_stated_jupiter_acceleration_is_not_inverse_square_consistent():
    """2e-5 at 1 AU is 7.4e-7 at 5.2 AU, not the 1e-6 quoted alongside it."""
    implied = STATED_ACCELERATION_1AU / 5.2**2
    assert implied == pytest.approx(7.40e-7, rel=1e-2)
    assert STATED_ACCELERATION_JUPITER > implied
    # the quoted pair is a 20x drop, which is 4.47 AU, not Jupiter's 5.2
    ratio = STATED_ACCELERATION_1AU / STATED_ACCELERATION_JUPITER
    assert ratio**0.5 == pytest.approx(4.47, rel=1e-2)


def test_the_required_acceleration_and_array_are_consistent():
    """kW/tonne, acceleration and array fraction are one number three ways."""
    feasible = _feasibility()
    kw_per_tonne = required_power_kw(feasible, 1.0)
    assert required_characteristic_acceleration(feasible) == pytest.approx(
        characteristic_acceleration(kw_per_tonne, 1.0)
    )
    assert array_mass_fraction(feasible, 15.0) == pytest.approx(
        15.0 * kw_per_tonne / 1000.0
    )


def test_a_cycle_needing_no_correction_demands_no_hardware():
    feasible = CycleFeasibility(
        cycle=_feasibility().cycle,
        dsm_leg=LegImpulse(1.0, 1.0, 1.0, 1.0),
        split_legs=(LegImpulse(1.0, 1.0, 1.0, 1.0),) * 2,
        wave_tonnes=float("inf"),
    )
    assert required_power_kw(feasible, 500.0) == 0.0
    assert required_characteristic_acceleration(feasible) == 0.0
    assert array_mass_fraction(feasible) == 0.0
