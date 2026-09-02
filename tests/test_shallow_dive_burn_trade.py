"""Tests for the shallow-dive burn trade (ADR 0025)."""

import pytest
from astropy import units as u

from src.bielliptic_dive_split import (
    _PERIAPSIS_BURN,
    PAD_ADMISSIBLE_DEPTH,
    direct_departure_conduction_depth,
    resonant_dive_at_depth,
)
from src.heliocentric_reintercept import single_impulse_resonant_dive
from src.shallow_dive_burn_trade import (
    BURN_BRACKET,
    SPLIT_PAD_BRACKET,
    SPLIT_REUSE,
    conducting_burn,
    shallow_dive_row,
    split_pad_crossing,
    split_pad_row,
)

# --------------------------------------------------------------------------
# The burn now reaches the closure
# --------------------------------------------------------------------------


def test_the_closure_honours_the_perihelion_burn():
    """A larger burn climbs out faster, so the loop closes slightly shorter."""
    base = single_impulse_resonant_dive()
    hotter = single_impulse_resonant_dive(periapsis_burn=43.172 * u.km / u.s)
    assert hotter.reintercept_time < base.reintercept_time
    assert hotter.closing_aphelion < base.closing_aphelion
    # Second order: the climb is ~10 d of a ~326 d cycle, so 20% more burn
    # moves the clock about 1%.
    ratio = float(hotter.reintercept_time / base.reintercept_time)
    assert 0.985 < ratio < 0.995


def test_passing_the_burn_through_moves_no_published_crossing():
    """The latent bug was real but below the precision anything is quoted at."""
    # At the paper's own tuning nothing moves at all, since that was the default.
    assert direct_departure_conduction_depth() == pytest.approx(19.804, abs=5e-3)
    # Away from it the shift stays under a tenth of a solar radius.
    assert direct_departure_conduction_depth(periapsis_burn=38.148) == pytest.approx(
        23.013, abs=0.02
    )
    assert direct_departure_conduction_depth(periapsis_burn=45.0) == pytest.approx(
        35.969, abs=0.05
    )


def test_resonant_dive_at_depth_actually_uses_its_burn():
    """It accepted the argument and dropped it; that is fixed."""
    cold = resonant_dive_at_depth(23.0, 30.0)
    hot = resonant_dive_at_depth(23.0, 44.0)
    assert cold[0] != hot[0]  # closing aphelion
    assert cold[3] != hot[3]  # cycle time


# --------------------------------------------------------------------------
# The direct route can fly shallow
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_six_percent_larger_burn_reaches_the_pad_floor():
    """The claim S2 had to decide, and it decides for the weaker reading."""
    burn = conducting_burn(PAD_ADMISSIBLE_DEPTH)
    assert burn is not None
    assert burn == pytest.approx(38.10, abs=0.05)
    assert burn / _PERIAPSIS_BURN == pytest.approx(1.059, abs=0.005)


@pytest.mark.slow
def test_the_direct_dive_still_grows_at_the_pad_floor():
    """Flying shallow degrades it; it does not stop it, so the split is not required."""
    row = shallow_dive_row(PAD_ADMISSIBLE_DEPTH)
    assert row is not None
    assert row.conducts
    assert row.growth.return_per_impactor_kg > 1.0
    assert row.growth.return_per_impactor_kg == pytest.approx(2.505, abs=0.01)
    assert row.growth.doubling_years == pytest.approx(0.657, abs=0.005)


@pytest.mark.slow
def test_flying_shallow_costs_roughly_double_the_clock():
    """The real price of depth, once the node is charged for its own boost."""
    deep = shallow_dive_row(4.0, periapsis_burn=_PERIAPSIS_BURN)
    shallow = shallow_dive_row(PAD_ADMISSIBLE_DEPTH)
    assert shallow.growth.doubling_years / deep.growth.doubling_years == pytest.approx(
        2.14, abs=0.05
    )


@pytest.mark.slow
def test_the_burn_needed_rises_with_depth_and_survival_falls():
    """One-sided trade, which is why the minimum conducting burn is the right one."""
    rows = [shallow_dive_row(d) for d in (19.8, 26.0, 32.0, 36.0)]
    assert all(r is not None for r in rows)
    burns = [r.periapsis_burn for r in rows]
    survivals = [r.node.survival for r in rows]
    doublings = [r.growth.doubling_years for r in rows]
    assert burns == sorted(burns)
    assert survivals == sorted(survivals, reverse=True)
    assert doublings == sorted(doublings)


def test_no_conducting_burn_exists_far_past_the_useful_band():
    """The bracket runs out before the physics does, and that is recorded."""
    assert conducting_burn(44.0) is None
    assert BURN_BRACKET == (18.0, 46.0)


def test_the_constraint_does_not_bind_at_the_papers_own_depth():
    """At 4 solar radii every burn in the bracket conducts, so there is no minimum.

    ``conducting_burn`` returns None for "the constraint is slack here" as well
    as for "no burn reaches this depth"; the two are told apart by which side of
    the bracket the crossing falls on, and callers wanting 4 solar radii should
    pass the burn they mean.
    """
    assert conducting_burn(4.0) is None
    assert shallow_dive_row(4.0) is None
    assert shallow_dive_row(4.0, periapsis_burn=_PERIAPSIS_BURN) is not None


@pytest.mark.slow
def test_the_node_exhaust_speed_collapses_as_the_dive_backs_out():
    """Why a gentler node is a less efficient one, in one number."""
    deep = shallow_dive_row(4.0, periapsis_burn=_PERIAPSIS_BURN)
    shallow = shallow_dive_row(32.0)
    assert deep.node.exhaust_speed == pytest.approx(68.09, abs=0.1)
    assert shallow.node.exhaust_speed == pytest.approx(23.78, abs=0.1)


# --------------------------------------------------------------------------
# The stated survival is the bigger error
# --------------------------------------------------------------------------


def test_the_stated_survival_is_right_at_four_solar_radii():
    """Which is why nobody caught it: the paper's headline depth is fine."""
    row = shallow_dive_row(4.0, periapsis_burn=_PERIAPSIS_BURN)
    assert row.node.survival == pytest.approx(0.5895, abs=1e-3)
    assert row.survival_ratio == pytest.approx(1.018, abs=5e-3)
    assert row.flattery == pytest.approx(1.009, abs=5e-3)


@pytest.mark.slow
def test_the_stated_survival_flatters_the_shallow_rows_two_to_threefold():
    """The result that matters more than the burn question S2 was framed on."""
    at_pad = shallow_dive_row(PAD_ADMISSIBLE_DEPTH)
    at_32 = shallow_dive_row(32.0)
    assert at_pad.survival_ratio == pytest.approx(2.32, abs=0.02)
    assert at_32.survival_ratio == pytest.approx(3.68, abs=0.03)
    assert at_pad.flattery == pytest.approx(1.915, abs=0.01)
    assert at_32.flattery == pytest.approx(3.068, abs=0.02)


@pytest.mark.slow
def test_on_the_stated_survival_depth_looks_nearly_free():
    """The illusion, stated as the near-flat doubling time it produces."""
    stated = [
        shallow_dive_row(
            d, periapsis_burn=_PERIAPSIS_BURN if d == 4.0 else None
        ).doubling_stated
        for d in (4.0, PAD_ADMISSIBLE_DEPTH, 32.0)
    ]
    # Fixed survival: doubling barely moves across the whole dial -- 0.305 to
    # 0.343 to 0.309 yr, non-monotone and inside 13%...
    assert max(stated) / min(stated) < 1.15
    derived = [
        shallow_dive_row(
            d, periapsis_burn=_PERIAPSIS_BURN if d == 4.0 else None
        ).growth.doubling_years
        for d in (4.0, PAD_ADMISSIBLE_DEPTH, 32.0)
    ]
    # ...while the derived one triples.
    assert derived[-1] / derived[0] > 3.0


@pytest.mark.slow
def test_the_pad_margin_falls_but_never_cleared_anyway():
    """ADR 0021 already had the direct dive failing the floor at 4 solar radii."""
    for depth in (4.0, PAD_ADMISSIBLE_DEPTH, 32.0):
        row = shallow_dive_row(
            depth, periapsis_burn=_PERIAPSIS_BURN if depth == 4.0 else None
        )
        assert row.pad_margin < 1.0


# --------------------------------------------------------------------------
# The split dive on the pad, which is the only scoreboard it ever won
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_split_earns_its_launch_only_at_the_deepest_dive():
    """ADR 0023's 'buys the pad' claim, finally in ADR 0021's committed currency."""
    deep = split_pad_row(4.0)
    assert deep is not None
    assert deep.clears
    assert deep.pad_margin == pytest.approx(1.179, abs=0.01)
    # By 8 solar radii it is already failing.
    shallow = split_pad_row(8.0)
    assert shallow is not None
    assert not shallow.clears
    assert shallow.pad_margin == pytest.approx(0.808, abs=0.01)


@pytest.mark.slow
def test_the_split_stops_paying_for_its_launch_at_five_and_a_half():
    """The crossing, bisected on the constraint rather than sampled near it."""
    crossing = split_pad_crossing()
    assert crossing is not None
    assert crossing == pytest.approx(5.583, abs=0.02)
    # Nowhere near the shallow end ADR 0022 recommends starting at.
    assert crossing < PAD_ADMISSIBLE_DEPTH / 3.0


@pytest.mark.slow
def test_the_splits_pad_edge_shrinks_as_the_dive_backs_out():
    """Its advantage is concentrated exactly where the thermal case is worst."""
    edges = {}
    for depth in (4.0, 16.0, 32.0):
        split = split_pad_row(depth)
        direct = shallow_dive_row(
            depth, periapsis_burn=_PERIAPSIS_BURN if depth <= 19.81 else None
        )
        edges[depth] = split.pad_margin / direct.pad_margin
    assert edges[4.0] == pytest.approx(1.80, abs=0.03)
    assert edges[32.0] == pytest.approx(1.09, abs=0.03)
    assert edges[4.0] > edges[16.0] > edges[32.0]


@pytest.mark.slow
def test_neither_architecture_earns_its_launch_at_the_recommended_depth():
    """The result that matters for where the paper says to start."""
    split = split_pad_row(PAD_ADMISSIBLE_DEPTH)
    direct = shallow_dive_row(PAD_ADMISSIBLE_DEPTH)
    assert not split.clears
    assert split.pad_margin < 1.0
    assert direct.pad_margin < 1.0


@pytest.mark.slow
def test_the_phased_closure_still_solves_across_the_whole_depth_dial():
    """The pad verdict would be vacuous if the cycle could not be built there."""
    for depth in (4.0, 16.0, PAD_ADMISSIBLE_DEPTH, 32.0):
        row = split_pad_row(depth)
        assert row is not None and row.closes


def test_the_split_pad_search_boxes_are_recorded():
    """Both brackets pinned, per the ADR 0007 lesson."""
    assert SPLIT_PAD_BRACKET == (4.0, 8.0)
    assert SPLIT_REUSE == (5, 8)
