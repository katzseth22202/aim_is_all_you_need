"""Tests for src/free_dsm_bound.py.

The fast tests protect the closed-form pieces and the parameterisation: the
flyby rotation (whose whole content is that an unpowered pass preserves the
excess speed), the bend-deficit algebra the ADR 0028 verdict actually turns on,
the pack/unpack of a leg's parameters, and the guards that keep an optimiser's
wilder probes out of a traceback.

The slow ones run the bound itself.  They are the only thing that checks the
claim, so they cannot be dropped -- but the module's own CLI budget is minutes
per cycle, so they run a reduced hop count and assert *bracketing* facts that a
cheaper search still has to reproduce: that the bound never beats the seam by
more than a hair, and that it stays far above what the array can deliver.
"""

import numpy as np
import pytest

from src.free_dsm_bound import (
    DEFAULT_MINIMUM_DV,
    DEPARTURE_VINF_BOUNDS,
    DSM_FRACTION_BOUNDS,
    INFEASIBLE_PENALTY,
    PERIJOVE_RATIO_BOUNDS,
    ExpensiveCycle,
    FreeDsmSolution,
    _bounds_for,
    _evaluate,
    _fly_leg,
    _flyby_exit,
    _from_unit,
    _lambert,
    _leg_parameter_count,
    _rotate,
    _safe_propagate,
    _seam_guess,
    _to_unit,
    _unit_from_angles,
    _unpack_leg,
    bend_deficit_degrees,
    blind_sample_floor,
    bound_the_expensive_cycles,
    expensive_cycles,
    solve_free_dsm,
)
from src.real_orbit_resonance import _MU_JUPITER, _MU_SUN, _PERIJOVE_FLOOR

_AU_KM = 149_597_870.7
#: The three cycles ADR 0026 prices in kilometres per second, in order.
_SEAM_KM_S = (1.584955, 2.089377, 1.425619)
#: What the array delivers over a whole trajectory at the stated acceleration.
_DELIVERABLE_M_S = 254.5
#: The same three cycles at the ten-day gap the paper flies (m/s), in order.
_SEAM_TEN_DAY_M_S = (556.0, 1011.3, 398.5)


# --------------------------------------------------------------------------
# Geometry and parameterisation
# --------------------------------------------------------------------------


def test_angles_make_a_unit_vector_and_come_back_out():
    """The polar maneuver form has to round-trip, or the seeds mean nothing."""
    for right_ascension, declination in ((0.3, 0.7), (5.9, -1.1), (2.0, 0.0)):
        vector = _unit_from_angles(right_ascension, declination)
        assert np.isclose(np.linalg.norm(vector), 1.0)
        assert np.isclose(
            np.arctan2(vector[1], vector[0]) % (2.0 * np.pi),
            right_ascension % (2.0 * np.pi),
        )
        assert np.isclose(np.arcsin(vector[2]), declination)


def test_rotation_preserves_length_and_fixes_its_own_axis():
    """Rodrigues, as used by both the flyby turn and the B-plane sweep."""
    vector = np.array([3.0, -1.0, 2.0])
    axis = np.array([0.0, 0.0, 1.0])
    rotated = _rotate(vector, axis, 0.9)
    assert np.isclose(np.linalg.norm(rotated), np.linalg.norm(vector))
    assert np.allclose(_rotate(vector, vector, 1.3), vector)
    assert np.allclose(_rotate(vector, axis, 0.0), vector)


def test_the_flyby_is_unpowered_so_the_excess_speed_survives_it():
    """That invariant is the entire modelling content of the flyby."""
    incoming = np.array([12.0, -8.0, 3.0])
    for plane_angle in (0.0, 1.2, 4.7):
        outgoing = _flyby_exit(incoming, _PERIJOVE_FLOOR, plane_angle)
        assert np.isclose(np.linalg.norm(outgoing), np.linalg.norm(incoming))


def test_the_flyby_turns_by_the_conic_turn_angle():
    """``e = 1 + r_p v_inf^2 / mu`` and ``sin(delta/2) = 1/e``."""
    incoming = np.array([20.0, 0.0, 0.0])
    speed = float(np.linalg.norm(incoming))
    for radius in (_PERIJOVE_FLOOR, 5.0 * _PERIJOVE_FLOOR):
        eccentricity = 1.0 + radius * speed**2 / _MU_JUPITER
        expected = 2.0 * np.arcsin(1.0 / eccentricity)
        outgoing = _flyby_exit(incoming, radius, 0.4)
        cosine = float(np.dot(incoming, outgoing)) / speed**2
        assert np.isclose(np.arccos(np.clip(cosine, -1.0, 1.0)), expected)


def test_turn_authority_falls_with_both_perijove_and_speed():
    """Why the expensive cycles are stuck: they arrive fast and cannot slow."""
    available_at_floor, _ = bend_deficit_degrees(20.6, np.radians(109.0))
    available_higher, _ = bend_deficit_degrees(
        20.6, np.radians(109.0), perijove_radius=2.0 * _PERIJOVE_FLOOR
    )
    available_slower, _ = bend_deficit_degrees(15.0, np.radians(109.0))
    assert available_at_floor > available_higher
    assert available_slower > available_at_floor


def test_the_bend_deficit_is_the_shortfall_in_degrees_and_signs_correctly():
    """Positive when the flyby cannot turn far enough, negative when it can."""
    available, deficit = bend_deficit_degrees(20.622, np.radians(109.56))
    assert np.isclose(deficit, 109.56 - available)
    assert 3.0 < deficit < 5.0
    _, spare = bend_deficit_degrees(10.0, np.radians(40.0))
    assert spare < 0.0


def test_a_leg_carries_one_epoch_per_maneuver_and_three_numbers_per_free_burn():
    """The last maneuver is the closing Lambert arc, so it costs no parameters."""
    assert _leg_parameter_count(1) == 1
    assert _leg_parameter_count(5) == 5 + 3 * 4


def test_maneuver_epochs_are_sorted_rather_than_constrained():
    """The optimiser sees a box; ordering is imposed on the way out."""
    raw = np.array([0.6, 0.1, 0.4, 2.0, 0.0, 0.0, 3.0, np.pi, 0.0])
    fractions, burns = _unpack_leg(raw, 3)
    assert np.allclose(fractions, [0.1, 0.4, 0.6])
    assert burns.shape == (2, 3)
    assert np.isclose(np.linalg.norm(burns[0]), 2.0)
    assert np.isclose(np.linalg.norm(burns[1]), 3.0)


def test_a_zero_magnitude_maneuver_is_a_zero_vector_whatever_its_angles():
    """Seeding the extras at zero has to reproduce the poorer budget exactly."""
    _, burns = _unpack_leg(np.array([0.5, 0.5, 0.0, 1.7, -0.9]), 2)
    assert np.allclose(burns, 0.0)


def test_the_search_box_has_one_entry_per_parameter():
    """Four shared parameters, two for the flyby, and both legs' own."""
    bounds = _bounds_for(5, 5)
    assert len(bounds) == 4 + _leg_parameter_count(5) + 2 + _leg_parameter_count(5)
    assert bounds[0] == DEPARTURE_VINF_BOUNDS
    assert bounds[4 + _leg_parameter_count(5)] == PERIJOVE_RATIO_BOUNDS


def test_expensive_means_beyond_what_the_array_delivers():
    """The threshold is the question, not a round number."""
    assert np.isclose(1000.0 * DEFAULT_MINIMUM_DV, _DELIVERABLE_M_S)


def test_the_maneuver_epoch_floor_reaches_the_flyby_itself():
    """The seam burns *at* Jupiter; a floor that cannot express it bounds nothing."""
    assert DSM_FRACTION_BOUNDS[0] <= 1.0e-4


def test_normalising_the_box_round_trips():
    """Eleven orders of magnitude separate the parameters; the solve is on [0, 1]."""
    bounds = _bounds_for(2, 3)
    point = np.array([low + 0.37 * (high - low) for low, high in bounds])
    assert np.allclose(_from_unit(_to_unit(point, bounds), bounds), point)


# --------------------------------------------------------------------------
# Guards: an optimiser probes states a physics library will not accept
# --------------------------------------------------------------------------


def test_a_hopeless_propagation_is_reported_rather_than_raised():
    """Farnocchia asserts on states the search will certainly reach."""
    assert _safe_propagate(np.zeros(3), np.zeros(3), 1.0e7) is None


def test_a_hyperbolic_heliocentric_arc_still_propagates():
    """A 14 km/s departure excess puts the outbound leg above solar escape."""
    position = np.array([_AU_KM, 0.0, 0.0])
    speed = 1.3 * np.sqrt(2.0 * _MU_SUN / _AU_KM)
    propagated = _safe_propagate(position, np.array([0.0, speed, 0.0]), 5.0e6)
    assert propagated is not None
    assert np.linalg.norm(propagated[0]) > _AU_KM


def test_an_unsolvable_lambert_arc_is_reported_rather_than_raised():
    """Zero time of flight between distinct points has no arc."""
    assert (
        _lambert(np.array([_AU_KM, 0.0, 0.0]), np.array([0.0, _AU_KM, 0.0]), 0.0, True)
        is None
    )


def test_lambert_reproduces_a_circular_quarter_orbit():
    """The arc solver is used raw, so pin it against a case with an answer."""
    radius = _AU_KM
    speed = np.sqrt(_MU_SUN / radius)
    quarter = 0.25 * 2.0 * np.pi * radius / speed
    arc = _lambert(
        np.array([radius, 0.0, 0.0]), np.array([0.0, radius, 0.0]), quarter, True
    )
    assert arc is not None
    assert np.allclose(arc[0], [0.0, speed, 0.0], atol=1.0e-6)


def test_a_leg_with_one_maneuver_costs_exactly_its_closing_arc():
    """With no free burn the whole cost is the Lambert match, by construction."""
    radius = _AU_KM
    speed = np.sqrt(_MU_SUN / radius)
    position = np.array([radius, 0.0, 0.0])
    velocity = np.array([0.0, 0.9 * speed, 0.0])
    duration = 0.25 * 2.0 * np.pi * radius / speed
    flown = _fly_leg(
        position,
        velocity,
        np.array([0.0, radius, 0.0]),
        duration,
        np.array([DSM_FRACTION_BOUNDS[0]]),
        np.zeros((0, 3)),
        True,
    )
    assert flown is not None
    arc = _lambert(position, np.array([0.0, radius, 0.0]), duration, True)
    assert arc is not None
    assert np.isclose(flown[0], float(np.linalg.norm(arc[0] - velocity)), rtol=1e-3)


def test_a_leg_too_short_to_fly_is_penalised_not_flown():
    """The encounter epoch is free, so the search will push it past both ends."""
    departure, arrival = 2_461_000.0, 2_461_800.0
    parameters = np.zeros(len(_bounds_for(1, 1)))
    parameters[3] = -790.0
    cost, solution = _evaluate(parameters, departure, arrival, departure + 400.0)
    assert cost == INFEASIBLE_PENALTY
    assert solution is None


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def test_the_improvement_is_the_ratio_to_the_seam():
    """One is the answer this ADR expects; the field exists to show it."""
    solution = FreeDsmSolution(
        departure_jd=2_461_000.0,
        arrival_jd=2_461_800.0,
        encounter_jd=2_461_400.0,
        departure_vinf=14.0,
        first_dsm=0.0,
        second_dsm=1.6,
        perijove_radius=75_492.0,
        total_dv=1.6,
        seam_dv=3.2,
        feasible=True,
    )
    assert np.isclose(solution.improvement, 2.0)


def test_a_free_trajectory_is_infinitely_better_than_nothing():
    """Guards the division; a zero-cost bound would be the interesting answer."""
    solution = FreeDsmSolution(
        departure_jd=0.0,
        arrival_jd=1.0,
        encounter_jd=0.5,
        departure_vinf=0.0,
        first_dsm=0.0,
        second_dsm=0.0,
        perijove_radius=75_492.0,
        total_dv=0.0,
        seam_dv=1.0,
        feasible=True,
    )
    assert solution.improvement == float("inf")


# --------------------------------------------------------------------------
# The cycles the bound is run on
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cycles():
    """ADR 0026's three expensive cycles, flown from the same cadence."""
    return expensive_cycles()


@pytest.mark.slow
def test_the_expensive_cycles_are_adr_0026s_three(cycles):
    """Same chain, same three cycles, same seam prices."""
    assert len(cycles) == 3
    assert [c.synodic_multiple for c in cycles] == [2, 2, 2]
    assert np.allclose([c.seam_dv for c in cycles], _SEAM_KM_S, atol=1.0e-3)


@pytest.mark.slow
def test_the_papers_ten_day_gap_selects_the_same_three_cycles():
    """Halving the head start halves the bill and changes nothing else.

    The paper flies a ten-day split and quotes 398.5, 556.0 and 1011.3 m/s;
    ADR 0013 and ADR 0026 price twenty. Same three departures either way,
    because the split's cost is bimodal at both.
    """
    ten_day = expensive_cycles(split_days=10.0)
    assert len(ten_day) == 3
    assert np.allclose(
        [1000.0 * c.seam_dv for c in ten_day], _SEAM_TEN_DAY_M_S, atol=0.1
    )
    assert all(np.isclose(c.seam_perijove_radius, _PERIJOVE_FLOOR) for c in ten_day)
    for cycle in ten_day:
        _available, deficit = bend_deficit_degrees(
            cycle.incoming_vinf, cycle.required_turn
        )
        assert 0.9 < deficit < 2.5


@pytest.mark.slow
def test_every_expensive_cycle_is_already_scraping_jupiter(cycles):
    """Pinned at the 4,000 km floor, which is why no perijove change buys them."""
    assert all(np.isclose(c.seam_perijove_radius, _PERIJOVE_FLOOR) for c in cycles)


@pytest.mark.slow
def test_the_residue_is_a_bend_deficit_of_a_few_degrees(cycles):
    """109 degrees demanded, 104-106 available: the shortfall is an angle."""
    for cycle in cycles:
        available, deficit = bend_deficit_degrees(
            cycle.incoming_vinf, cycle.required_turn
        )
        assert 108.0 < np.degrees(cycle.required_turn) < 110.0
        assert 104.0 < available < 106.0
        assert 3.0 < deficit < 5.0


@pytest.mark.slow
def test_the_deficit_explains_the_bulk_of_the_seam_charge(cycles):
    """``2 v_inf sin(deficit/2)`` is 82-85% of it, at ~0.36 km/s per degree.

    Not all of it: the seam charge is an exact velocity match, so it also pays
    whatever speed error the arc carries, while the chord prices the rotation
    alone.  The point of the check is that the rotation is the dominant term --
    which is why no amount of moving the burn helps.
    """
    for cycle in cycles:
        _, deficit = bend_deficit_degrees(cycle.incoming_vinf, cycle.required_turn)
        chord = 2.0 * cycle.incoming_vinf * np.sin(0.5 * np.radians(deficit))
        assert 0.80 < chord / cycle.seam_dv < 0.88
        assert 0.35 < chord / deficit < 0.38


# --------------------------------------------------------------------------
# The bound itself
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bounded(cycles):
    """The worst cycle's bound, on a hop budget a test can afford.

    Bracketed around the answer rather than searched from scratch, per the
    CLAUDE.md rule: the module's own budget is minutes per cycle and this is
    seconds, which is enough to assert that the optimum sits at the seam and
    not somewhere far below it.
    """
    cycle = max(cycles, key=lambda c: c.seam_dv)
    return cycle, solve_free_dsm(
        cycle.departure_jd,
        cycle.arrival_jd,
        cycle.encounter_jd,
        cycle.seam_dv,
        seam_perijove_radius=cycle.seam_perijove_radius,
        outbound_maneuvers=2,
        inbound_maneuvers=2,
        basin_hops=3,
        random_starts=0,
    )


@pytest.mark.slow
def test_the_bound_lands_on_the_trajectory_it_bounds(bounded):
    """The optimum is the seam architecture, not something else entirely.

    Within a percent or two rather than exactly, because this module rebuilds
    the seam trajectory in its own flyby model -- a swept B-plane and a
    maneuver placed 1e-4 of a leg after the encounter -- rather than importing
    it.  That reconstruction gap is what the tolerance covers; the direction
    that carries the finding is the other one, and it is
    ``test_freeing_the_burn_buys_essentially_nothing``.
    """
    cycle, solution = bounded
    assert solution.feasible
    assert solution.total_dv <= cycle.seam_dv * 1.02


@pytest.mark.slow
def test_freeing_the_burn_buys_essentially_nothing(bounded):
    """The finding: the optimum is the seam value, to well under a percent."""
    cycle, solution = bounded
    assert solution.improvement < 1.01


@pytest.mark.slow
def test_the_bound_stays_far_above_what_the_array_can_deliver(bounded):
    """Hence ADR 0026's verdict holds against *any* finite-thrust solution."""
    _, solution = bounded
    assert 1000.0 * solution.total_dv > 5.0 * _DELIVERABLE_M_S


@pytest.mark.slow
def test_the_optimum_spends_nothing_before_jupiter(bounded):
    """Slowing down to buy turn authority costs more than the turn returns."""
    _, solution = bounded
    assert 1000.0 * solution.first_dsm < 1.0


@pytest.mark.slow
def test_the_optimum_stays_at_the_perijove_floor(bounded):
    """Every kilometre of extra perijove is turn authority given away."""
    _, solution = bounded
    assert solution.perijove_radius < 1.05 * _PERIJOVE_FLOOR


@pytest.mark.slow
def test_the_seam_seed_reproduces_the_seam_price(cycles):
    """The seed is the bound's floor guarantee, so pin it on its own."""
    cycle = cycles[0]
    guess = _seam_guess(
        cycle.departure_jd,
        cycle.arrival_jd,
        cycle.encounter_jd,
        cycle.seam_perijove_radius,
    )
    assert guess is not None
    cost, solution = _evaluate(
        guess, cycle.departure_jd, cycle.arrival_jd, cycle.encounter_jd
    )
    assert solution is not None
    assert np.isclose(cost, cycle.seam_dv, rtol=0.02)


@pytest.mark.slow
def test_every_maneuver_budget_starts_from_the_same_trajectory(cycles):
    """What makes a richer budget a relaxation and not a different problem.

    The extras are seeded to zero magnitude at the same epoch, so the seed
    *is* the single-maneuver trajectory however many maneuvers are allowed.
    Pinned because it is the guarantee the bound rests on, and because it is
    invisible at the level of the reported answer: a richer search converges
    more slowly from that seed, so on a small hop budget it can still finish
    above a leaner one without anything being wrong.
    """
    cycle = cycles[0]
    costs = []
    for maneuvers in (1, 3, 5):
        guess = _seam_guess(
            cycle.departure_jd,
            cycle.arrival_jd,
            cycle.encounter_jd,
            cycle.seam_perijove_radius,
            maneuvers,
            maneuvers,
        )
        assert guess is not None
        cost, _ = _evaluate(
            guess,
            cycle.departure_jd,
            cycle.arrival_jd,
            cycle.encounter_jd,
            maneuvers,
            maneuvers,
        )
        costs.append(cost)
    assert np.allclose(costs, costs[0])


@pytest.mark.slow
def test_the_search_never_returns_more_than_its_own_seed(cycles):
    """The invariant that caught the zero-magnitude seeding bug.

    Seeding an extra maneuver at exactly zero puts L-BFGS-B on the kink of
    ``norm(dv)``, where every direction raises the cost to first order: it
    stopped there, and a five-maneuver search reported *more* than its own
    seed, which no correct search can do.
    """
    cycle = cycles[0]
    guess = _seam_guess(
        cycle.departure_jd,
        cycle.arrival_jd,
        cycle.encounter_jd,
        cycle.seam_perijove_radius,
        3,
        3,
    )
    assert guess is not None
    seed_cost, _ = _evaluate(
        guess, cycle.departure_jd, cycle.arrival_jd, cycle.encounter_jd, 3, 3
    )
    rich = solve_free_dsm(
        cycle.departure_jd,
        cycle.arrival_jd,
        cycle.encounter_jd,
        cycle.seam_dv,
        seam_perijove_radius=cycle.seam_perijove_radius,
        outbound_maneuvers=3,
        inbound_maneuvers=3,
        basin_hops=2,
        random_starts=0,
    )
    assert rich.total_dv <= seed_cost + 1.0e-9


@pytest.mark.slow
def test_blind_sampling_flies_a_tenth_of_the_box(cycles):
    """ADR 0008's rule: know what the box looks like before reading a search.

    The failure this guards against is the opposite of ADR 0008's own. There a
    table of failures hid a working search; here a seeded search could hide a
    box it never explores. A tenth of blind draws flying a complete
    Earth-Jupiter-Earth trajectory says the feasible set is amply populated, so
    "nothing cheaper was found" is a statement about cost and not about
    feasibility.
    """
    for cycle in cycles:
        flew, _best = blind_sample_floor(cycle, samples=400)
        assert 0.03 * 400 < flew < 0.40 * 400


@pytest.mark.slow
def test_blind_sampling_finds_nothing_near_the_seam(cycles):
    """The cheapest of 2,000 random trajectories is 54-69x the seam charge.

    That is what makes the seeded basin worth trusting: it is not one of a
    crowd of comparable ones the hops might have missed.
    """
    for cycle in cycles:
        _flew, best = blind_sample_floor(cycle)
        assert best > 50.0 * cycle.seam_dv


@pytest.mark.slow
def test_the_table_scores_every_cycle_out_of_reach():
    """The CLI's verdict row, on a reduced budget: 3 of 3 beyond the array."""
    frame = bound_the_expensive_cycles(
        maneuvers_per_leg=1, basin_hops=4, random_starts=0, blind_samples=200
    )
    assert len(frame) == 3
    assert bool(frame.still_out_of_reach.all())
    assert (frame.free_burn_m_s > _DELIVERABLE_M_S).all()
    assert (frame.improvement < 1.01).all()
    assert (frame.blind_best_km_s > frame.seam_m_s / 1000.0).all()


def test_an_expensive_cycle_carries_what_the_bound_needs_to_reproduce_it():
    """The record is the whole interface between the cadence and the search."""
    cycle = ExpensiveCycle(
        departure_jd=2_462_550.7,
        arrival_jd=2_463_328.4,
        encounter_jd=2_462_960.0,
        synodic_multiple=2,
        seam_dv=1.585,
        seam_perijove_radius=_PERIJOVE_FLOOR,
        incoming_vinf=20.622,
        required_turn=np.radians(109.56),
    )
    assert cycle.arrival_jd - cycle.departure_jd > 0.0
    assert cycle.seam_perijove_radius == _PERIJOVE_FLOOR
