"""Tests for the dive node's second arrival (ADR 0024)."""

import numpy as np
import pytest

from src.jovian_solar_dive_cycle import (
    _dive_leg,
    _dive_periapsis_radius,
    _outbound_leg,
    _powered_flyby_params,
    solve_synodic_closure,
)
from src.opposing_stream_ledger import (
    COLOCATION_BRACKET,
    TANGENCY_NUDGE,
    bielliptic_coplacement,
    direct_dive_charge,
    jovian_cycle_charge,
    opposing_placement_route,
)
from src.solar_dive_depth_trade import (
    CONSERVATIVE_RETURN_EXCESS,
    opposing_stream_placement,
)

_PARAMS = _powered_flyby_params()


def _closure(depth):
    return solve_synodic_closure(3, _PARAMS, depth, CONSERVATIVE_RETURN_EXCESS, 1.0)


# --------------------------------------------------------------------------
# The route exists
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [4.0, 8.0, 16.0, 23.0, 32.0])
def test_the_opposing_stream_can_be_placed_at_every_depth(depth):
    """A co-locating retrograde placement exists across the whole depth dial."""
    route = opposing_placement_route(depth, _closure(depth), _PARAMS)
    assert route is not None
    assert route.feasible


@pytest.mark.parametrize("depth", [4.0, 23.0, 32.0])
def test_the_unpowered_flyby_has_ample_bend_for_the_placement(depth):
    """The turn needed is well inside what a perijove-floor flyby delivers."""
    route = opposing_placement_route(depth, _closure(depth), _PARAMS)
    assert route.bend_required_deg < route.bend_available_deg
    # Never tighter than 1.5x the required turn, so this is not a knife-edge.
    assert route.bend_available_deg / route.bend_required_deg > 1.5


def test_co_location_is_bought_for_a_third_of_a_percent_or_less():
    """Meeting the payload in longitude *and* time costs tens of m/s."""
    premiums = {}
    for depth in (4.0, 8.0, 16.0, 23.0, 32.0):
        route = opposing_placement_route(depth, _closure(depth), _PARAMS)
        premiums[depth] = route.colocation_premium - 1.0
    assert premiums[4.0] == pytest.approx(0.0012, abs=2e-4)
    assert premiums[32.0] == pytest.approx(0.0032, abs=2e-4)
    # Monotone in depth, and never more than a third of a percent.
    assert all(0.0 < v < 0.0035 for v in premiums.values())
    assert sorted(premiums, key=lambda d: premiums[d]) == [4.0, 8.0, 16.0, 23.0, 32.0]


def test_the_placement_costs_about_what_the_payload_departure_costs():
    """Per kilogram the second arrival is no dearer than the first, via Jupiter."""
    for depth in (4.0, 23.0, 32.0):
        route = opposing_placement_route(depth, _closure(depth), _PARAMS)
        ratio = route.colocation_departure / route.payload_departure_excess
        assert 0.9 < ratio < 1.15


def test_the_placement_floor_is_a_tangency_that_defeats_the_dive_solver():
    """The bug the nudge exists for: at exactly the floor, no dive state.

    ``retrograde_tangential_departure`` is solved to land *on* the floor, and
    the floor is where the post-flyby state is purely tangential, so
    ``_solve_dive_state`` sits on its bracket edge.  At 23 solar radii that
    returns None outright; the nudge is what makes the leg buildable.
    """
    depth = 23.0
    placement = opposing_stream_placement(depth, _closure(depth), _PARAMS)
    radius = _dive_periapsis_radius(depth)
    at_floor = _outbound_leg(placement.retrograde_tangential_departure, 0.0, _PARAMS)
    assert _dive_leg(at_floor, -1.0, _PARAMS, radius) is None
    nudged = _outbound_leg(
        placement.retrograde_tangential_departure + TANGENCY_NUDGE, 0.0, _PARAMS
    )
    assert _dive_leg(nudged, -1.0, _PARAMS, radius) is not None


def test_the_dive_leaves_jupiter_almost_tangentially_at_the_floor():
    """At the floor Jupiter's orbit is the dive's aphelion, so it sweeps 180 deg."""
    depth = 23.0
    placement = opposing_stream_placement(depth, _closure(depth), _PARAMS)
    outbound = _outbound_leg(
        placement.retrograde_tangential_departure + TANGENCY_NUDGE, 0.0, _PARAMS
    )
    leg = _dive_leg(outbound, -1.0, _PARAMS, _dive_periapsis_radius(depth))
    assert np.degrees(leg.swept) == pytest.approx(-180.0, abs=0.5)


# --------------------------------------------------------------------------
# The charge is a rounding error, in both currencies
# --------------------------------------------------------------------------


def test_charging_the_jovian_cycle_costs_under_one_percent_of_doubling():
    """The architecture that has Jupiter barely notices the second arrival."""
    deep = jovian_cycle_charge(4.0)
    shallow = jovian_cycle_charge(32.0)
    assert deep.doubling_penalty == pytest.approx(1.0014, abs=5e-4)
    assert shallow.doubling_penalty == pytest.approx(1.0075, abs=5e-4)
    assert deep.doubling_penalty < shallow.doubling_penalty


def test_charging_the_paper_dive_costs_under_one_and_a_half_percent():
    """The architecture without Jupiter pays 3x the impulse and still barely moves."""
    deep = direct_dive_charge(4.0)
    shallow = direct_dive_charge(32.0)
    assert deep.placement_excess == pytest.approx(35.48, abs=0.02)
    assert shallow.placement_excess == pytest.approx(44.94, abs=0.02)
    assert deep.doubling_penalty == pytest.approx(1.0063, abs=5e-4)
    assert shallow.doubling_penalty == pytest.approx(1.0112, abs=5e-4)


def test_neither_ledger_dominates_and_the_ratio_is_exact():
    """pad / growth is exactly (1 - delivered) / (1 - placed), either side of 1."""
    from src.solar_dive_depth_trade import depth_trade_row

    deep = jovian_cycle_charge(4.0)
    shallow = jovian_cycle_charge(32.0)
    # The deep Jovian row's own departure wastes less than its placement does,
    # so there the growth reading is the harsher one -- the ordering is not
    # a property of the ledgers, it is a property of the two burns.
    assert deep.pad_charge < deep.growth_charge
    assert shallow.pad_charge > shallow.growth_charge
    for depth in (4.0, 23.0, 32.0):
        row = depth_trade_row(dive_solar_radii=depth)
        charge = jovian_cycle_charge(depth)
        expected = (1.0 - row.growth.nozzle.delivered_fraction) / (
            1.0 - charge.placement_delivered_fraction
        )
        assert charge.pad_charge / charge.growth_charge == pytest.approx(
            expected, rel=1e-9
        )


def test_charging_flips_no_pad_verdict_anywhere():
    """Rows that cleared ADR 0021's 1/15 floor still clear it; failures stay failures."""
    for depth in (4.0, 8.0, 16.0, 23.0, 32.0):
        assert not jovian_cycle_charge(depth).pad_verdict_flips
        assert not direct_dive_charge(depth).pad_verdict_flips


def test_the_pad_charge_never_exceeds_two_percent():
    """The bound the ADR quotes, on the reading that could have broken it."""
    charges = [jovian_cycle_charge(d).pad_charge for d in (4.0, 8.0, 16.0, 23.0, 32.0)]
    charges += [direct_dive_charge(d).pad_charge for d in (4.0, 23.0, 32.0)]
    assert max(charges) < 0.021
    assert min(charges) > 0.004
    # As a fraction of returned mass lost, rather than of pad mass added.
    assert max(c / (1.0 + c) for c in charges) < 0.02


def test_the_charge_is_small_because_of_mass_not_impulse():
    """The node's slug ratio, not a cheap burn, is what makes this negligible."""
    paper = direct_dive_charge(32.0)
    # 44.94 km/s is nearly 4x the Jovian placement, and the nozzle delivers
    # only a fifth of the vehicle through it...
    assert paper.placement_delivered_fraction < 0.21
    # ...yet the mass wanted is well under a kilogram per impactor kilogram.
    assert paper.opposing_kg_per_impactor_kg < 0.2


# --------------------------------------------------------------------------
# The bi-elliptic node phases itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [4.0, 23.0, 32.0])
@pytest.mark.parametrize("node", [1.9649, 3.0])
def test_both_arrivals_share_one_ellipse_so_the_clock_matches_exactly(depth, node):
    """Same semi-major axis, so the same half-period: no residual to tune."""
    row = bielliptic_coplacement(depth, node)
    expected = (
        np.pi
        * np.sqrt(
            (0.5 * (node * 1.495978707e8 + depth * 695700.0)) ** 3 / 1.32712440018e11
        )
        / (365.25 * 86400.0)
    )
    assert row.dive_tof_years == pytest.approx(expected, rel=1e-6)
    assert row.head_on_degrees == 180.0


@pytest.mark.parametrize("depth", [4.0, 23.0, 32.0])
def test_neither_leg_ever_goes_deeper_than_the_node(depth):
    """The shallow dive's whole reason for existing survives the placement."""
    row = bielliptic_coplacement(depth)
    assert row.closest_approach_solar_radii == depth


def test_the_flip_costs_more_than_the_cut_and_the_gap_widens_with_depth():
    """The two-leg asymmetry, in the one architecture that pays both at once."""
    deep = bielliptic_coplacement(4.0)
    shallow = bielliptic_coplacement(32.0)
    assert deep.opposing_flip > deep.payload_cut
    assert shallow.opposing_flip > shallow.payload_cut
    # The payload's leg gets cheaper as the dive backs out, the opposing
    # stream's gets dearer -- they move opposite ways.
    assert shallow.payload_cut < deep.payload_cut
    assert shallow.opposing_flip > deep.opposing_flip


def test_a_farther_node_makes_both_burns_cheaper_and_the_coast_longer():
    """The bi-elliptic trade, stated on both legs at once."""
    near = bielliptic_coplacement(32.0, 1.9649)
    far = bielliptic_coplacement(32.0, 3.0)
    assert far.payload_cut < near.payload_cut
    assert far.opposing_flip < near.opposing_flip
    assert far.dive_tof_years > near.dive_tof_years


def test_the_colocation_bracket_is_recorded_and_narrow():
    """The search box is pinned, per the ADR 0007 lesson."""
    assert COLOCATION_BRACKET == (1.0e-4, 0.05)
    assert TANGENCY_NUDGE == 1.0e-4
