"""Tests for the split solar-dive injection in src/bielliptic_dive_split.py.

Two claims carry ADR 0023 and are pinned here.  The first is a *degeneracy*: at
an outbound perihelion equal to the dive perihelion, with the injection at
aphelion, the split loop must reduce exactly to the paper's single-impulse
resonant dive -- same closing aphelion, same clock, a zero outer burn.  That is
what makes every other row in the family comparable to the paper's.  The second
is that the three-condition phasing solution set is real and discrete: both
closures and a rational beam reuse, met at once.

Most other tests check structure rather than digits -- the direction a residual
moves, the sign of a gap, the ordering of two currencies -- so a re-tuned search
that tells the same physical story keeps passing.

Nothing here is marked ``slow``: the whole file runs in about three seconds, so
ADR 0023's figures sit in the *fast* suite.  That is deliberate, and it is the
lesson of ADR 0022, whose headline numbers all came from slow-marked paths and
so went unprotected by ``make test`` while they were wrong.
"""

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Sun

from src import bielliptic_dive_split as split
from src.heliocentric_reintercept import single_impulse_resonant_dive
from src.jovian_solar_dive_cycle import paper_resonant_dive_ledger

_AU = split._AU
_DIVE = split._DIVE_PERIAPSIS


# --------------------------------------------------------------- degeneracy ---


def test_the_split_reduces_to_the_papers_dive_when_nothing_is_split() -> None:
    # The load-bearing check: outbound perihelion at the dive perihelion means
    # the outbound ellipse IS the dive ellipse, so the outer burn is zero and the
    # closure must land on single_impulse_resonant_dive()'s own aphelion.
    reference = single_impulse_resonant_dive()
    aphelion = split.reintercept_closing_aphelion(_DIVE, revolutions=0)
    assert aphelion / _AU == pytest.approx(
        reference.closing_aphelion.to_value(u.AU), abs=1e-3
    )
    geometry = split.split_dive_geometry(_DIVE, aphelion, 180.0)
    assert geometry is not None
    assert geometry.cycle_years == pytest.approx(
        reference.reintercept_time.to_value(u.year), abs=1e-3
    )
    assert geometry.departure_excess == pytest.approx(
        reference.earth_boost.to_value(u.km / u.s), abs=0.05
    )
    # "Zero" outer burn up to the aphelion-tangency round-off of the impulse search.
    assert geometry.injection.delta_v < 0.02


def test_the_degenerate_row_matches_the_papers_growth_ledger() -> None:
    # Same device, same currency, so the degenerate row must reproduce ADR 0019's
    # published baseline for the paper's dive rather than merely resemble it.
    aphelion = split.reintercept_closing_aphelion(_DIVE, revolutions=0)
    geometry = split.split_dive_geometry(_DIVE, aphelion, 180.0)
    assert geometry is not None
    ledger = split.split_dive_ledger(geometry)
    paper = paper_resonant_dive_ledger(geometry.stream_excess, geometry.stream_axis)
    assert ledger.doubling_years == pytest.approx(paper.doubling_years, rel=0.02)
    assert ledger.return_per_impactor_kg == pytest.approx(
        paper.return_per_impactor_kg, rel=0.02
    )


# ------------------------------------------------------------ the beam is radial ---


def test_the_returning_beam_is_nearly_radial_and_gets_more_so() -> None:
    # This is why one beam can serve both nodes: its 1 AU crossing and its outer
    # crossing are the same ray. If this ever stopped holding, the whole
    # co-location idea would need a second launch instead.
    beam = split.returning_beam(0.5 * (_DIVE + 3.0 * _AU))
    angles = []
    for radius_au in (1.0, 2.0, 3.0):
        tangential, radial = split.speed_components_at_radius(
            beam, split._MU_SUN, radius_au * _AU
        )
        angles.append(float(np.degrees(np.arctan2(tangential, radial))))
    assert angles[0] < 3.0
    assert angles == sorted(angles, reverse=True)


def test_the_beam_crosses_1_au_and_3_au_at_almost_one_longitude() -> None:
    beam = split.returning_beam(0.5 * (_DIVE + 3.0 * _AU))
    seconds, sweep = split.beam_leg(beam, _AU, 3.0 * _AU)
    assert seconds / 86400.0 == pytest.approx(22.4, abs=1.5)
    assert sweep < 3.0


# -------------------------------------------------------- the injection impulse ---


def test_at_aphelion_the_injection_is_the_tangential_apsis_burn() -> None:
    # The general impulse search must collapse to the textbook answer where the
    # textbook applies, or the off-aphelion rows cannot be trusted.
    aphelion = 3.0 * _AU
    mu = split._MU_SUN

    def apsis_speed(perihelion: float, radius: float) -> float:
        return float(np.sqrt(mu * (2.0 / radius - 2.0 / (perihelion + aphelion))))

    arriving = apsis_speed(_AU, aphelion)
    expected = arriving - apsis_speed(_DIVE, aphelion)
    injection = split.dive_injection_impulse(aphelion, arriving, 0.0, _DIVE)
    assert injection.delta_v == pytest.approx(expected, rel=1e-6)
    assert injection.v_radial == pytest.approx(0.0, abs=1e-6)


def test_the_injection_refuses_a_burn_point_its_target_cannot_reach() -> None:
    with pytest.raises(ValueError, match="reaches"):
        split.dive_injection_impulse(3.0 * _AU, 0.1, 0.0, _DIVE)


# --------------------------------------------------------- the delta-v claim ---


def test_the_bielliptic_route_is_cheaper_and_falls_towards_the_escape_floor() -> None:
    # A 1 AU -> 4 R_sun transfer is a 54:1 radius ratio, far past the ~15.6:1
    # where bi-elliptic always wins, so the total must fall monotonically towards
    # (sqrt(2)-1) * v_earth and never reach it.
    v_earth = float(np.sqrt(split._MU_SUN / _AU))
    direct = v_earth - float(np.sqrt(split._MU_SUN * (2.0 / _AU - 2.0 / (_DIVE + _AU))))
    assert direct == pytest.approx(24.09, abs=0.02)
    totals = [
        split.bielliptic_injection_cost(a * _AU)[2] for a in (1.5, 2, 3, 5, 10, 30)
    ]
    assert all(b < a for a, b in zip(totals, totals[1:]))
    assert all(t < direct for t in totals)
    assert totals[-1] > v_earth * (np.sqrt(2.0) - 1.0)


def test_three_au_costs_about_seventy_percent_of_the_direct_dive() -> None:
    raise_dv, drop_dv, total = split.bielliptic_injection_cost(3.0 * _AU)
    assert raise_dv == pytest.approx(6.694, abs=0.01)
    assert drop_dv == pytest.approx(10.250, abs=0.01)
    assert total == pytest.approx(16.944, abs=0.02)


# ------------------------------------------------------- the closure curve ---


def test_raising_the_outbound_perihelion_lengthens_the_clock() -> None:
    # The whole reason the split does not buy growth: a small prograde push puts
    # 1 AU at the outbound ellipse's *perihelion*, so the raise leg becomes a half
    # orbit instead of the last 11 degrees of one.
    clocks = []
    for perihelion_au in (_DIVE / _AU, 0.3, 0.6, 0.9):
        perihelion = perihelion_au * _AU
        geometry = split.split_dive_geometry(
            perihelion,
            split.reintercept_closing_aphelion(perihelion, revolutions=0),
            180.0,
        )
        assert geometry is not None
        clocks.append(geometry.cycle_years)
    assert clocks == sorted(clocks)
    assert clocks[-1] > 1.3 * clocks[0]


def test_the_closing_aphelion_barely_moves_along_the_curve() -> None:
    aphelions = [
        split.reintercept_closing_aphelion(q * _AU, revolutions=0) / _AU
        for q in (_DIVE / _AU, 0.3, 0.6, 0.9)
    ]
    assert min(aphelions) > 1.85
    assert max(aphelions) < 2.0


def test_slug_per_delivered_kg_falls_monotonically_as_the_split_deepens() -> None:
    # The pad-side currency is the one the split actually buys, and unlike the
    # doubling time it improves all the way to the pure bi-elliptic end.
    costs = []
    for perihelion_au in (_DIVE / _AU, 0.3, 0.6, 0.9, 0.99):
        perihelion = perihelion_au * _AU
        geometry = split.split_dive_geometry(
            perihelion,
            split.reintercept_closing_aphelion(perihelion, revolutions=0),
            180.0,
        )
        assert geometry is not None
        costs.append(split.split_dive_ledger(geometry).slug_per_delivered_kg)
    assert costs == sorted(costs, reverse=True)
    assert costs[0] / costs[-1] > 2.0


def test_the_co_location_gap_never_closes_on_the_curve() -> None:
    # Two conditions, two knobs, no intersection: this is what forces the third
    # knob (an off-aphelion injection) in two_node_closure().
    for revolutions, expected_sign in ((0, +1.0), (1, -1.0)):
        gaps = []
        for perihelion_au in (_DIVE / _AU, 0.2, 0.4, 0.6, 0.8, 0.95):
            perihelion = perihelion_au * _AU
            geometry = split.split_dive_geometry(
                perihelion,
                split.reintercept_closing_aphelion(perihelion, revolutions),
                180.0,
            )
            assert geometry is not None
            gaps.append(geometry.colocation_residual)
        assert all(np.sign(g) == expected_sign for g in gaps)
        assert min(abs(g) for g in gaps) > 40.0


# ------------------------------------------------------ the Pareto optimum ---


def test_the_partial_split_beats_the_papers_dive_on_both_axes() -> None:
    # ADR 0023's only strictly dominating result. Interior optimum, so a search.
    optimum = split.partial_split_optimum(revolutions=0)
    paper = paper_resonant_dive_ledger(
        optimum.geometry.stream_excess, optimum.geometry.stream_axis
    )
    paper_slug = (
        1.0 - paper.nozzle.delivered_fraction
    ) / paper.nozzle.delivered_fraction
    assert optimum.doubling_years < paper.doubling_years
    assert optimum.slug_per_delivered_kg < paper_slug
    assert 0.4 < optimum.geometry.outbound_perihelion / _AU < 0.6
    assert optimum.doubling_years == pytest.approx(0.2971, abs=2e-3)
    assert optimum.slug_per_delivered_kg == pytest.approx(1.538, abs=0.02)


# ---------------------------------------------------- the phased two-node cycle ---


def test_the_five_eighths_closure_meets_all_three_conditions() -> None:
    closure = split.two_node_closure(5, 8, revolutions=1)
    assert closure is not None
    geometry = closure.geometry
    assert abs(geometry.reintercept_residual) < split.CLOSURE_RESIDUAL_TOLERANCE
    assert abs(geometry.colocation_residual) < split.CLOSURE_RESIDUAL_TOLERANCE
    assert geometry.reuse_fraction == pytest.approx(0.625, abs=1e-9)
    # ADR 0023's headline row.
    assert geometry.outbound_aphelion / _AU == pytest.approx(2.940, abs=2e-3)
    assert geometry.cycle_years == pytest.approx(2.2789, abs=2e-3)
    assert np.degrees(geometry.injection_anomaly) - 180.0 == pytest.approx(
        11.296, abs=0.02
    )
    assert closure.node_cadence_years == pytest.approx(0.2849, abs=1e-3)


def test_the_phased_cycle_buys_the_pad_and_not_the_clock() -> None:
    closure = split.two_node_closure(5, 8, revolutions=1)
    assert closure is not None
    paper = paper_resonant_dive_ledger(
        closure.geometry.stream_excess, closure.geometry.stream_axis
    )
    paper_slug = (
        1.0 - paper.nozzle.delivered_fraction
    ) / paper.nozzle.delivered_fraction
    # Slower on the clock ...
    assert closure.ledger.doubling_years > 1.6 * paper.doubling_years
    # ... and much cheaper on the pad.
    assert closure.ledger.slug_per_delivered_kg < paper_slug / 2.5
    # Most of a wave flies past Earth to feed the outer node.
    assert 0.2 < closure.ledger.earth_caught_fraction < 0.4


def test_the_injection_burn_sits_just_past_aphelion_on_every_closure() -> None:
    # The third knob is small: the vehicle is slow out there, so a degree of
    # burn-point shift moves the phasing several degrees. If a closure ever
    # needed a large shift the impulse cost would stop being near-tangential.
    for numerator, denominator in split.DEFAULT_REUSE_FRACTIONS:
        closure = split.two_node_closure(numerator, denominator, revolutions=1)
        assert closure is not None
        past_aphelion = np.degrees(closure.geometry.injection_anomaly) - 180.0
        assert 5.0 < past_aphelion < 15.0
        assert 2.9 < closure.geometry.outbound_aphelion / _AU < 3.1


def test_the_closure_rejects_a_reuse_fraction_outside_the_family() -> None:
    # 1/2 lies below the family's reachable reuse band (~0.61 to 0.63), so the
    # solve must return None rather than a root that fails a residual.
    assert split.two_node_closure(1, 2, revolutions=1) is None


# ------------------------------------------ the conduction floor by dive depth ---


def test_backing_the_dive_out_weakens_the_stream_and_the_boost_together() -> None:
    # Why the direct architecture does NOT simply fail on impulse at a shallow
    # dive: the boost it needs falls alongside the stream that has to deliver it.
    # Any argument that the shallow dive is unreachable has to survive this.
    boosts, streams = [], []
    for radii in (4.0, 8.0, 16.0, 32.0):
        _, boost, _, _ = split.resonant_dive_at_depth(radii)
        row = split.depth_conduction(radii)
        assert row is not None
        boosts.append(boost)
        streams.append(row.stream_excess)
    assert boosts == sorted(boosts, reverse=True)
    assert streams == sorted(streams, reverse=True)
    # Both fall, and neither collapses: 37.5 -> 28.4 km/s against 157 -> 97.
    assert boosts[0] / boosts[-1] < 1.5
    assert streams[0] / streams[-1] < 2.0


def test_the_earth_departure_cools_itself_and_the_far_node_cannot() -> None:
    # The asymmetry the whole depth argument rests on. At Earth the vehicle
    # accelerates roughly along the stream, so the closing speed falls through
    # the burn; at the far node the thrust is perpendicular to a radial stream,
    # so the burn cannot run away from what feeds it.
    row = split.depth_conduction(4.0)
    assert row is not None
    assert row.direct_closing_end < row.direct_closing_start - 20.0
    closure = split.two_node_closure(5, 8, revolutions=1)
    assert closure is not None
    ledger = closure.ledger
    beam = split.returning_beam(
        closure.geometry.injection.semi_major_axis,
        closure.geometry.dive_perihelion,
        closure.geometry.periapsis_burn,
    )
    tangential, radial = split.speed_components_at_radius(
        beam, split._MU_SUN, closure.geometry.node_radius
    )
    beam_speed = float(np.hypot(tangential, radial))
    # The far node's closing speed never strays far from the beam's own speed.
    assert abs(ledger.outer_closing_speed - beam_speed) < 1.0


def test_the_direct_departure_goes_cold_before_the_split_does() -> None:
    crossing = split.direct_departure_conduction_depth()
    assert crossing is not None
    assert 15.0 < crossing < 25.0
    for radii in (23.0, 32.0, 48.0):
        row = split.depth_conduction(radii)
        assert row is not None
        assert not row.direct_clears
        assert row.split_clears
        assert row.outer_clears


def test_the_direct_departure_still_conducts_at_the_papers_own_depth() -> None:
    # The crossing must not be so aggressive that it condemns 4 solar radii too;
    # that would mean the floor had been misapplied rather than a depth found.
    row = split.depth_conduction(4.0)
    assert row is not None
    assert row.direct_clears
    assert row.direct_closing_end > row.threshold + 40.0


def test_the_crossing_depth_moves_with_the_expansion_margin() -> None:
    # The result is margin-dependent and must be quoted with its margin, exactly
    # as ADR 0022 quotes the conduction bracket with its own.
    depths = [
        split.direct_departure_conduction_depth(expansion_margin=margin)
        for margin in (1.5, 1.25, 1.0)
    ]
    assert all(d is not None for d in depths)
    assert depths == sorted(depths)  # a looser margin permits a shallower dive


def test_the_split_closure_can_be_flown_at_a_shallower_depth() -> None:
    # The phasing must still close when the depth knob is moved, or the
    # conduction argument would be about a cycle that cannot be built.
    shallow = split.two_node_closure(
        5, 8, revolutions=1, dive_perihelion=16.0 * split._SOLAR_RADIUS
    )
    assert shallow is not None
    assert abs(shallow.geometry.reintercept_residual) < split.CLOSURE_RESIDUAL_TOLERANCE
    assert abs(shallow.geometry.colocation_residual) < split.CLOSURE_RESIDUAL_TOLERANCE
    assert shallow.geometry.dive_perihelion == pytest.approx(16.0 * split._SOLAR_RADIUS)
    # A shallower dive is a smaller drop, so the far-node burn is smaller too.
    deep = split.two_node_closure(5, 8, revolutions=1)
    assert deep is not None
    assert shallow.geometry.injection.delta_v < deep.geometry.injection.delta_v


def test_the_split_opens_the_top_of_the_admissible_depth_band() -> None:
    # The window claim, stated narrowly: ADR 0022's pad floor admits (4, 22.93]
    # R_sun, the direct departure conducts over (4, 19.80], so what the split
    # adds is the last ~3 R_sun -- which is the end ADR 0022 recommends starting
    # at. If the conduction crossing ever rose above the pad crossing, the split
    # would open nothing and this ADR's addendum would be void.
    crossing = split.direct_departure_conduction_depth()
    assert crossing is not None
    assert crossing < split.PAD_ADMISSIBLE_DEPTH
    assert split.PAD_ADMISSIBLE_DEPTH - crossing > 1.0
    # And the split must actually clear at the top of that band.
    row = split.depth_conduction(split.PAD_ADMISSIBLE_DEPTH)
    assert row is not None
    assert row.split_clears and row.outer_clears
    assert not row.direct_clears


# ------------------------------------------------ the opposing stream's leg ---


def test_the_two_legs_move_in_opposite_directions_under_the_depth_dial() -> None:
    # The correction that forced ADR 0023's addendum to be widened. The payload's
    # injection gets CHEAPER as the dive is backed out; the opposing stream's gets
    # DEARER, because a shallower perihelion keeps more angular momentum and
    # reversing its sign costs more. Quoting only the first leg understates the
    # shallow dive's difficulty.
    payload, opposing = [], []
    for radii in (4.0, 8.0, 16.0, 23.0, 32.0):
        _, boost, _, _ = split.resonant_dive_at_depth(radii)
        row = split.opposing_stream_placement_trade(radii)
        assert row is not None
        payload.append(boost)
        opposing.append(row.direct_excess)
    assert payload == sorted(payload, reverse=True)
    assert opposing == sorted(opposing)
    # And they cross: the opposing stream starts comparable and ends dominant.
    assert opposing[0] < payload[0]
    assert opposing[-1] > 1.5 * payload[-1]


def test_the_opposing_stream_conducts_at_every_depth_because_it_is_canted() -> None:
    # It aims retrograde against a radially outward stream, so it never runs
    # *along* what feeds it -- unlike the payload's 31-degree departure, its
    # closing speed does not collapse through the burn. Its problem is cost,
    # not conduction, and the two must not be conflated.
    for radii in (4.0, 16.0, 32.0):
        row = split.opposing_stream_placement_trade(radii)
        assert row is not None
        assert row.direct_clears
        assert row.direct_closing_end > 0.9 * row.direct_closing_start


def test_the_split_saving_on_the_opposing_stream_grows_with_depth() -> None:
    # The mirror image of the payload leg, and the reason the split's case at a
    # shallow dive rests mostly here: the raise costs the same either way, so
    # only the flip term differs, and the flip is what the depth dial inflates.
    savings = [
        split.opposing_stream_placement_trade(radii).saving  # type: ignore[union-attr]
        for radii in (4.0, 8.0, 16.0, 23.0, 32.0)
    ]
    assert savings == sorted(savings)
    assert savings[0] > 1.6
    assert savings[-1] > 1.8


def test_the_far_node_flip_conducts_even_reversing_tangential_motion() -> None:
    # The flip is larger than the payload's injection and passes through zero
    # tangential speed, but it is still perpendicular to a radial beam, so the
    # closing speed barely moves.
    for radii in (4.0, 23.0, 32.0):
        row = split.opposing_stream_placement_trade(radii)
        assert row is not None
        assert row.split_clears
        assert abs(row.split_flip_closing_end - row.split_flip_closing_start) < 1.5


def test_the_radial_plunge_is_flat_in_depth_and_sits_between_the_two() -> None:
    # The third option: kill all of Earth's tangential motion and fall in, taking
    # the node impact across the payload's path instead of into it. Costs Earth's
    # own orbital speed at every depth -- neither growing nor shrinking.
    excesses = {
        radii: split.opposing_stream_placement_trade(radii).radial_excess  # type: ignore[union-attr]
        for radii in (4.0, 16.0, 32.0)
    }
    assert len(set(round(v, 9) for v in excesses.values())) == 1
    deep = split.opposing_stream_placement_trade(32.0)
    assert deep is not None
    assert deep.radial_excess < deep.direct_excess


# ------------------------------------------------- the node's own geometry ---


def test_the_plunger_arrives_at_135_degrees_not_90() -> None:
    # The easy mistake: "it comes in sideways, so theta = 90". It does not.
    # Payload and plunger reach perihelion within a percent of the same speed, so
    # their relative velocity bisects the tangential and radial axes.
    for radii in (4.0, 16.0, 32.0):
        row = split.node_geometry_trade(radii)
        assert row.plunger_angle == pytest.approx(135.0, abs=0.5)
        assert row.plunger_speed == pytest.approx(row.arrival_speed, rel=0.02)
        assert row.plunger_closing / row.head_on_closing == pytest.approx(
            1.0 / np.sqrt(2.0), rel=0.01
        )


def test_beta_rises_as_the_closing_speed_falls() -> None:
    # The whole point: the impulse law's cos-theta debit is the impactor's own
    # momentum arriving backwards, and a 135-degree arrival only pays 0.707 of it.
    # So the geometry that loses closing speed gains impulse per impactor.
    for slug_ratio in (1.0, 3.0, 30.0):
        row = split.node_geometry_trade(4.0, slug_ratio)
        assert row.plunger_beta > row.head_on_beta
        assert row.plunger_closing < row.head_on_closing


def test_the_isp_kept_falls_as_the_slug_ratio_rises() -> None:
    # And this is what decides the trade. The debit is k-independent while the
    # useful term grows as sqrt(k), so the relief is large at a slug-poor node
    # and nearly absent at a slug-rich one. Quoting one k alone would mislead.
    ratios = [
        split.node_geometry_trade(4.0, k).exhaust_ratio for k in (1.0, 3.0, 8.5, 30.0)
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] > 0.95  # k = 1: the geometry is nearly free
    assert ratios[-1] < 0.80  # k = 30: it costs a fifth of the exhaust speed
    # It never recovers the whole loss: beta cannot gain the full root two.
    assert all(r < 1.0 for r in ratios)


def test_the_plunger_halves_the_collision_heat_at_every_slug_ratio() -> None:
    # Thermalised energy goes as w**2 and w falls by exactly 1/root 2, so this is
    # a clean one-half independent of k -- which is what makes the trade
    # attractive at all despite the Isp loss.
    for slug_ratio in (1.0, 3.0, 30.0):
        for radii in (4.0, 32.0):
            row = split.node_geometry_trade(radii, slug_ratio)
            assert row.thermal_ratio == pytest.approx(0.5, abs=0.01)


def test_halving_the_collision_heat_is_worth_about_a_factor_of_two_in_depth() -> None:
    # The reason the trade matters: depth is expensive (ADR 0020 prices 4 -> 32
    # solar radii at 2.51x the doubling time), so buying it with a fifth of the
    # node's Isp may be cheap. The equivalence is on the collision term only.
    for radii in (4.0, 8.0, 16.0):
        equivalent = split.plunger_equivalent_depth(radii)
        assert equivalent is not None
        assert 1.8 < equivalent / radii < 2.2


def test_the_node_exhaust_speed_peaks_well_below_the_repo_default() -> None:
    # Worth pinning because it frames the trade: the node's own exhaust speed
    # peaks near k = 3 and is falling by k = 30, so the slug ratio that makes the
    # plunger look worst is already past the node's Isp optimum.
    speeds = {
        k: split.node_geometry_trade(4.0, k).head_on_exhaust
        for k in (1.0, 3.0, 8.5, 30.0)
    }
    assert speeds[3.0] > speeds[1.0]
    assert speeds[3.0] > speeds[8.5] > speeds[30.0]
