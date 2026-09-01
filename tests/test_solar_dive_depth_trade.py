"""Tests for the solar-dive depth trade in src/solar_dive_depth_trade.py.

Three claims carry ADR 0020 and the tests are shaped around them rather than
around digits: the dive node priced from the impulse law agrees with ADR 0019's
*stated* survival at 4 solar radii (so the shallow rows are not scored on a
different device), backing the dive out is monotone in every ledger, and the
cap that binds the departure slug ratio is the launch ledger rather than the
plume ignition window.  Pinned numbers are the ones ADR 0020 quotes.

The last section tests ADR 0021, which re-scores the **launch ledger** on the
**split push**.  Where the two disagree the pad-charged figure is the current
one; the free-parking numbers earlier in this file are pinned as the superseded
reading, marked as such, so the size of the correction stays checkable.
"""

from typing import List

import numpy as np
import pytest
from astropy import units as u

from src import solar_dive_depth_trade as trade
from src.circular_resonance_impulse import impulse_per_impactor_kg
from src.jovian_solar_dive_cycle import (
    DEFAULT_JET_ENERGY_EFFICIENCY,
    DEFAULT_PERIAPSIS_SURVIVAL,
    DEFAULT_RETURN_EXCESS,
    DEFAULT_SLUG_RATIO,
    SynodicCycleClosure,
    solve_synodic_closure,
)
from src.plume_thermal import chemistry_efficiency, slug_ratio_window
from src.retrograde_return_legs import _powered_flyby_params
from src.two_leg_nozzle_sweep import RETURN_FLOOR

_PARAMS = _powered_flyby_params()


@pytest.fixture(scope="module")
def shallow() -> trade.DepthTradeRow:
    """The conservative 32 solar-radii cycle ADR 0020 is written about."""
    row = trade.depth_trade_row(params=_PARAMS)
    assert row is not None
    return row


@pytest.fixture(scope="module")
def deep() -> trade.DepthTradeRow:
    """ADR 0019's own 4 solar-radii reference cycle, scored on this module."""
    row = trade.depth_trade_row(
        4.0, DEFAULT_RETURN_EXCESS, DEFAULT_SLUG_RATIO, params=_PARAMS
    )
    assert row is not None
    return row


# --------------------------------------------------------------------------
# The dive node, priced rather than stated
# --------------------------------------------------------------------------


def test_perihelion_speed_is_just_under_the_local_escape_speed() -> None:
    # A fall from Jupiter's orbit is near-parabolic, so the perihelion speed sits
    # a little below sqrt(2 mu / r_p) -- by 0.18% at 4 solar radii and 1.4% at 32.
    for radii, shortfall in ((4.0, 0.003), (32.0, 0.02)):
        r_p = trade._dive_periapsis_radius(radii)
        escape = float(np.sqrt(2.0 * _PARAMS.mu_sun / r_p))
        speed = trade.near_parabolic_perihelion_speed(radii, _PARAMS)
        assert speed < escape
        assert speed == pytest.approx(escape, rel=shortfall)


def test_opposing_closing_speed_is_twice_the_perihelion_speed() -> None:
    for radii in (4.0, 32.0):
        assert trade.opposing_stream_closing_speed(radii, _PARAMS) == pytest.approx(
            2.0 * trade.near_parabolic_perihelion_speed(radii, _PARAMS)
        )


def test_derived_survival_reproduces_adr_0019s_stated_fraction(
    deep: trade.DepthTradeRow,
) -> None:
    # The load-bearing agreement. jovian_solar_dive_cycle states 0.60 at the
    # 4 R_sun node; the impulse law at the same k and efficiency derives 0.5977.
    # If these ever diverge, the shallow rows are being scored on a device the
    # deep one was not, and every comparison in ADR 0020 is void.
    assert deep.node.survival == pytest.approx(0.5977, abs=5e-4)
    assert deep.node.survival == pytest.approx(DEFAULT_PERIAPSIS_SURVIVAL, rel=0.01)


def test_shallow_node_is_less_efficient_despite_needing_a_smaller_boost(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # The mechanism behind the whole trade: v_e goes as beta*w/k and w is set by
    # the depth, so backing out cuts the exhaust speed (68.1 -> 23.8 km/s) faster
    # than it cuts the boost (35.0 -> 24.7), and survival falls even though less
    # delta-v is being bought.
    assert shallow.node.boost < deep.node.boost
    assert shallow.node.exhaust_speed < deep.node.exhaust_speed
    assert shallow.node.survival < deep.node.survival
    assert shallow.node.exhaust_speed == pytest.approx(23.78, abs=0.05)
    assert shallow.node.survival == pytest.approx(0.3537, abs=5e-4)


def test_shallow_node_is_the_gentler_place_to_be(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # Why anyone would back out at all: flux falls as 1/r^2 and the equilibrium
    # temperature as 1/sqrt(r), and the merge energy falls as w^2.
    assert deep.node.solar_flux / shallow.node.solar_flux == pytest.approx(
        64.0, rel=0.01
    )
    assert deep.node.equilibrium_temperature == pytest.approx(2041.0, abs=5.0)
    assert shallow.node.equilibrium_temperature == pytest.approx(722.0, abs=5.0)
    assert (
        deep.node.thermalised_energy / shallow.node.thermalised_energy
        == pytest.approx(8.2, rel=0.02)
    )
    assert trade.rendezvous_timing_tolerance(
        shallow.node
    ) > 2.8 * trade.rendezvous_timing_tolerance(deep.node)


# --------------------------------------------------------------------------
# The closure still holds at 32 solar radii
# --------------------------------------------------------------------------


def test_three_synodic_still_closes_unpowered_at_thirty_two_solar_radii(
    shallow: trade.DepthTradeRow,
) -> None:
    c = shallow.closure
    assert c.feasible
    assert c.perijove_burn == 0.0
    assert c.intercept_miss_deg == pytest.approx(0.0, abs=1e-6)
    assert -c.bend_deficit_deg == pytest.approx(54.48, abs=0.05)
    assert c.departure_excess == pytest.approx(12.499, abs=5e-3)
    assert c.departure_aim_deg == pytest.approx(-44.21, abs=0.05)


def test_the_shallow_closure_is_the_only_root_in_the_aim_bracket() -> None:
    # ADR 0020 quotes a single root; a second one would mean the reported cycle
    # is a search artefact rather than the closure. Scanned at 1 degree here,
    # at 0.5 degrees when the ADR was written.
    from src.jovian_solar_dive_cycle import (
        _evaluate,
        _intercept_miss,
        _solve_clock,
        synodic_period,
    )

    r_p = trade._dive_periapsis_radius(trade.CONSERVATIVE_DIVE_SOLAR_RADII)
    target = 3.0 * synodic_period(_PARAMS)
    previous = None
    roots = 0
    for aim in np.arange(-85.0, 85.0, 1.0):
        excess = _solve_clock(
            float(aim),
            target,
            _PARAMS,
            r_p,
            trade.CONSERVATIVE_RETURN_EXCESS,
            1.0,
            None,
        )
        evaluation = (
            None
            if excess is None
            else _evaluate(
                excess,
                float(aim),
                _PARAMS,
                r_p,
                trade.CONSERVATIVE_RETURN_EXCESS,
                1.0,
                None,
            )
        )
        if evaluation is None:
            previous = None
            continue
        miss = _intercept_miss(evaluation, _PARAMS)
        if previous is not None and previous * miss < 0.0:
            roots += 1
        previous = miss
    assert roots == 1


def test_the_cant_grows_as_the_dive_is_backed_out(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # The push axis barely moves (it is nearly radial either way) but the
    # departure aim swings from -12 to -44 degrees, so the Earth-side collision
    # gets more head-on -- past the k = 16.43 head-on crossover only if k is
    # large, which the shallow cycle's cap forbids.
    assert deep.closure.aim_separation_deg == pytest.approx(110.61, abs=0.05)
    assert shallow.closure.aim_separation_deg == pytest.approx(141.07, abs=0.05)
    assert shallow.growth.nozzle.impact_angle_start_deg > 130.0


def test_the_departure_burn_gets_hotter_not_colder(
    shallow: trade.DepthTradeRow,
) -> None:
    # The premise this module was asked to check: that the closing speed decays
    # towards ~60 km/s by the end of the burn and threatens the plasma. It does
    # the opposite -- the impact is past 90 degrees, so the vehicle's own
    # acceleration adds to the closing speed.
    nozzle = shallow.growth.nozzle
    assert nozzle.closing_speed_end > nozzle.closing_speed_start
    assert nozzle.closing_speed_start == pytest.approx(91.78, abs=0.05)
    assert nozzle.closing_speed_end == pytest.approx(95.55, abs=0.05)


# --------------------------------------------------------------------------
# Growth, and the two ceilings on the slug ratio
# --------------------------------------------------------------------------


def test_growth_at_the_conservative_operating_point(
    shallow: trade.DepthTradeRow,
) -> None:
    assert shallow.growth.nozzle.delivered_fraction == pytest.approx(0.7261, abs=5e-4)
    assert shallow.growth.round_trip_fraction == pytest.approx(0.2568, abs=5e-4)
    assert shallow.growth.return_per_impactor_kg == pytest.approx(7.97, abs=0.02)
    assert shallow.growth.doubling_years == pytest.approx(1.094, abs=5e-3)
    assert shallow.growth.cycle_years == pytest.approx(3.2761, abs=5e-4)


def test_backing_out_costs_about_ten_fold_growth(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    ratio = deep.growth.return_per_impactor_kg / shallow.growth.return_per_impactor_kg
    assert ratio == pytest.approx(10.5, rel=0.02)
    assert shallow.growth.doubling_years / deep.growth.doubling_years == pytest.approx(
        2.13, rel=0.02
    )


def test_the_plasma_cap_is_nowhere_near_binding(
    shallow: trade.DepthTradeRow,
) -> None:
    # The cap the conservative operating point was proposed to respect. At the
    # true closing speed it sits at 47.9, more than five times the 8.5 that was
    # feared, so plasma retention is not what limits the slug ratio.
    ceilings = shallow.ceilings
    assert ceilings.plasma_ceiling == pytest.approx(47.88, abs=0.05)
    assert (
        ceilings.plasma_floor < trade.CONSERVATIVE_SLUG_RATIO < ceilings.plasma_ceiling
    )
    # Even under the pessimistic 60 km/s reading the window still admits 8.5.
    window = slug_ratio_window(60.0 * u.km / u.s)
    assert window is not None
    assert window[0] < trade.CONSERVATIVE_SLUG_RATIO < window[1]


def test_the_launch_ledger_is_the_binding_cap_when_shallow(
    shallow: trade.DepthTradeRow,
) -> None:
    # SUPERSEDED READING, pinned so the correction stays visible. These are the
    # free-parking-orbit numbers: slug_ratio_ceilings() reads both caps off
    # cycle_growth_ledger, which starts the departure at Earth escape and never
    # charges the lob. Charged from the pad the shallow cycle clears the floor
    # at *no* slug ratio, so the 5.25 ceiling below is not the cap it looked
    # like -- see test_the_shallow_cycle_fails_the_pad_floor_at_every_slug_ratio.
    ceilings = shallow.ceilings
    assert ceilings.binding == "launch"
    assert ceilings.launch_ceiling is not None
    assert ceilings.launch_ceiling == pytest.approx(5.25, abs=0.02)
    assert ceilings.launch_ceiling < ceilings.plasma_ceiling
    assert ceilings.launch_peak_margin == pytest.approx(1.022, abs=5e-3)
    # It is the *committed* floor that bites; restated for a returned kilogram's
    # worth at 85 km/s, the whole ignition window clears.
    assert ceilings.rescaled_ceiling is None
    assert ceilings.rescaled_floor_reachable


def test_the_deep_cycle_has_launch_margin_the_shallow_one_spent(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # The deep cycle is capped by the expansion floor, never by the pad; the
    # shallow one runs out of launch margin first. Free-parking reading again --
    # the ordering survives the correction, the margins do not.
    assert deep.ceilings.binding == "expansion"
    assert deep.ceilings.launch_ceiling is None
    assert deep.ceilings.launch_peak_margin > 2.0
    assert deep.launch.stated_margin > 1.8
    assert shallow.launch.stated_margin < 1.0


def test_a_failing_floor_is_not_reported_as_a_missing_ceiling() -> None:
    # Both "cleared at every k" and "failed at every k" leave no crossing, and
    # conflating them would read a collapsed cycle as an unconstrained one.
    row = trade.depth_trade_row(32.0, 150.0, 8.5, params=_PARAMS)
    assert row is not None
    assert row.ceilings.launch_ceiling is None
    assert not row.ceilings.committed_floor_reachable
    assert row.ceilings.binding == "launch (fails at every k)"


def test_growth_rises_with_slug_ratio_while_pad_return_falls(
    shallow: trade.DepthTradeRow,
) -> None:
    # The opposition that makes the launch ledger a real cap: impactors are the
    # scarce input in one ledger and launched slug is scarce in the other. It
    # survives the pad correction unchanged -- see
    # test_growth_and_pad_return_still_pull_opposite_ways_after_the_correction.
    entries = trade.slug_ratio_table(shallow, params=_PARAMS)
    growth = [g.return_per_impactor_kg for _, g, _ in entries]
    pad = [v.returned_per_pad_kg for _, _, v in entries]
    assert growth == sorted(growth)
    assert pad == sorted(pad, reverse=True)


# --------------------------------------------------------------------------
# The depth and boost sweeps
# --------------------------------------------------------------------------


def test_every_depth_on_the_grid_still_closes_three_synodic() -> None:
    rows = trade.depth_trade_table(params=_PARAMS)
    assert len(rows) == len(trade.DEPTH_GRID)
    assert all(row.closure.feasible for row in rows)
    assert all(row.closure.perijove_burn == 0.0 for row in rows)


def test_the_depth_sweep_is_monotone_in_every_ledger() -> None:
    rows = trade.depth_trade_table(params=_PARAMS)
    for attribute, series in (
        ("survival", [r.node.survival for r in rows]),
        ("growth", [r.growth.return_per_impactor_kg for r in rows]),
        ("pad return", [r.launch.returned_per_pad_kg for r in rows]),
        ("bend margin", [-r.closure.bend_deficit_deg for r in rows]),
    ):
        assert series == sorted(series, reverse=True), attribute
    # ... and the cant, the one quantity that gets worse going the same way.
    cants = [r.closure.aim_separation_deg for r in rows]
    assert cants == sorted(cants)


def test_delivered_mass_is_the_one_ledger_that_is_not_monotone() -> None:
    # Deliberately excluded from the sweep above rather than overlooked. The cant
    # grows with depth and drags the impact angle through the impulse law's
    # optimum on the way, so delivered mass peaks a little *inside* the grid --
    # at 6 solar radii, not at 4. The effect is under half a point and it is
    # swamped by the node's survival, which is why the growth column stays
    # monotone anyway.
    rows = trade.depth_trade_table(params=_PARAMS)
    delivered = [r.growth.nozzle.delivered_fraction for r in rows]
    peak = delivered.index(max(delivered))
    assert rows[peak].node.dive_solar_radii == 6.0
    assert max(delivered) - delivered[0] < 0.005
    assert delivered[peak:] == sorted(delivered[peak:], reverse=True)


def test_the_boost_stops_paying_somewhere_between_the_two_depths() -> None:
    # Deep, the node's 68 km/s exhaust makes the Oberth boost nearly free and
    # the optimum sits high. Shallow, the node runs at 24 km/s and the optimum
    # falls off the bottom of the bracket: the boost is a pure cost.
    deep_optimum = trade.growth_optimal_return_excess(4.0, params=_PARAMS)
    assert deep_optimum is not None
    assert deep_optimum == pytest.approx(121.3, rel=0.02)
    assert trade.growth_optimal_return_excess(32.0, params=_PARAMS) is None
    rows = trade.boost_trade_table(params=_PARAMS)
    growth = [r.growth.return_per_impactor_kg for r in rows]
    assert growth == sorted(growth, reverse=True)


# --------------------------------------------------------------------------
# Placing the opposing stream, the one thing depth makes harder
# --------------------------------------------------------------------------


def test_backing_out_spreads_the_placement_floors_apart(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # A shallower perihelion needs less of Jupiter's motion cancelled prograde
    # and more of it reversed retrograde, so the two floors move opposite ways
    # about the unchanged radial plunge.
    assert shallow.placement.prograde_floor < deep.placement.prograde_floor
    assert shallow.placement.retrograde_floor > deep.placement.retrograde_floor
    assert shallow.placement.radial_floor == pytest.approx(deep.placement.radial_floor)
    assert shallow.placement.prograde_floor == pytest.approx(9.9785, abs=5e-4)
    assert shallow.placement.retrograde_floor == pytest.approx(16.1374, abs=5e-4)


def test_the_opposing_stream_cannot_ride_either_cycle(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # Already open in ADR 0019 and wider here: the 3S closure's arrival excess is
    # below the retrograde floor at both depths, so the two streams need two
    # departure energies.
    for row in (deep, shallow):
        assert not row.placement.retrograde_rides_the_cycle
        assert row.placement.cycle_arrival_excess >= row.placement.prograde_floor


def test_a_one_way_launch_places_the_opposing_stream_cheaply(
    shallow: trade.DepthTradeRow,
) -> None:
    # The reassuring half: the separate departure energy is barely above the
    # payload's own, so needing two is a scheduling cost rather than a wall.
    placement = shallow.placement
    assert placement.retrograde_tangential_departure == pytest.approx(11.833, abs=5e-3)
    assert placement.retrograde_tangential_departure < shallow.closure.departure_excess
    assert trade.tangential_arrival_excess(
        placement.retrograde_tangential_departure, _PARAMS
    ) == pytest.approx(placement.retrograde_floor, abs=1e-6)


def test_direct_placement_from_earth_is_the_expensive_alternative(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # Closed form, so worth checking against its own definition: 1 AU is the
    # aphelion of the cheapest retrograde dive, and Earth's motion opposes it.
    for row, expected in ((deep, 35.48), (shallow, 44.94)):
        assert row.placement.direct_from_earth_excess == pytest.approx(
            expected, abs=0.02
        )
        r_p = trade._dive_periapsis_radius(row.node.dive_solar_radii)
        aphelion_speed = float(
            np.sqrt(
                2.0
                * _PARAMS.mu_sun
                * r_p
                / (_PARAMS.r_earth_orbit * (_PARAMS.r_earth_orbit + r_p))
            )
        )
        assert row.placement.direct_from_earth_excess == pytest.approx(
            _PARAMS.v_earth_orbit + aphelion_speed
        )
    assert (
        shallow.placement.direct_from_earth_excess
        > deep.placement.direct_from_earth_excess
    )


def test_the_shallow_node_eats_more_opposing_projectile_mass(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # Uncharged in the growth ledger at both depths, and it moves the wrong way:
    # a bigger propellant fraction at the same k_p is more arriving impactor.
    assert deep.node.opposing_impactor_fraction == pytest.approx(0.0134, abs=5e-5)
    assert shallow.node.opposing_impactor_fraction == pytest.approx(0.0215, abs=5e-5)


# --------------------------------------------------------------------------
# The report renders
# --------------------------------------------------------------------------


def test_the_row_report_names_the_binding_ceiling(
    shallow: trade.DepthTradeRow,
) -> None:
    text = trade.describe_row(shallow)
    assert "BINDING CEILING" in text
    assert "launch" in text
    assert "32 R_sun" in text


def test_the_return_floor_constant_is_the_one_the_ceilings_use() -> None:
    assert RETURN_FLOOR == pytest.approx(1.0 / 15.0)


# --------------------------------------------------------------------------
# The expansion floor: what the plume can actually give back
# --------------------------------------------------------------------------


def test_the_frozen_bill_is_most_of_the_ignition_bill() -> None:
    # ADR 0016: vaporisation, atomisation and seed ionisation default rather
    # than recombining, so only the translational remainder can do work.
    assert trade.ignition_bill() == pytest.approx(84.41, abs=0.05)
    assert trade.frozen_ignition_energy() == pytest.approx(53.47, abs=0.05)
    assert trade.frozen_ignition_energy() / trade.ignition_bill() == pytest.approx(
        0.633, abs=0.005
    )


def test_the_jet_takes_more_than_its_stated_share_of_what_exists() -> None:
    # eta_jet**2 is defined against the ideal one-axis budget w**2/2, but the
    # merge only has w**2 k / (2(1+k)) to give. Reading the two as the same
    # quantity is lenient, not conservative, and it is what made the first
    # version of the expansion floor too generous by 18% at k = 8.5.
    assert trade.jet_energy_fraction(8.5) == pytest.approx(0.671, abs=1e-3)
    assert trade.jet_energy_fraction(30.0) == pytest.approx(0.620, abs=1e-3)
    assert trade.jet_energy_fraction(100.0) == pytest.approx(0.606, abs=1e-3)
    # Below k = eta/(1 - eta) the law asks for more than the collision gives.
    assert trade.jet_energy_fraction(1.0) == 1.0
    with pytest.raises(ValueError):
        trade.jet_energy_fraction(0.0)


def test_the_efficiency_ceiling_is_stricter_than_adr_0016s_not_new() -> None:
    # Recorded because it would be easy to present as a discovery: ADR 0016
    # already put a frozen-chemistry ceiling on eta_jet**2 and implemented it as
    # plume_thermal.chemistry_efficiency. This one asks the harder question --
    # is the plume still *conducting* at exit, not merely is the jet positive --
    # so it must be stricter everywhere, and agree on the verdict.
    for speed, ratio in ((158.17, 30.0), (91.78, 8.5), (66.09, 23.75)):
        strict = trade.maximum_jet_efficiency(speed, ratio)
        loose = float(chemistry_efficiency(speed * u.km / u.s, ratio)) ** 2
        assert strict < loose
    # And both condemn the point an ignition-only search picks.
    assert trade.maximum_jet_efficiency(66.09, 23.75) < 0.01
    assert float(chemistry_efficiency(66.09 * u.km / u.s, 23.75)) ** 2 < 0.60
    # The two reference cycles clear the stated 0.60 on the strict bound.
    assert trade.maximum_jet_efficiency(158.17, 30.0) == pytest.approx(0.759, abs=5e-3)
    assert trade.maximum_jet_efficiency(91.78, 8.5) == pytest.approx(0.704, abs=5e-3)


def test_the_residual_is_what_the_ignition_bill_is_compared_against() -> None:
    # Nothing double-charges: the frozen chemistry sits inside both the residual
    # and the bill, so "residual >= bill" is the same statement as "residual
    # translation >= the bill's translational term".
    residual = trade.expansion_residual(91.78, 8.5)
    assert residual == pytest.approx(130.7, abs=0.5)
    frozen = trade.frozen_ignition_energy()
    thermal_floor = trade.ignition_bill() - frozen
    assert residual - frozen == pytest.approx(130.7 - frozen, abs=0.5)
    assert (residual - frozen) > thermal_floor
    assert trade.expansion_residual(66.09, 23.75) == pytest.approx(31.7, abs=0.5)


def test_the_expansion_window_is_far_tighter_than_the_ignition_window() -> None:
    for speed, ignition_max, expansion in (
        (66.09, 23.83, (3.22, 5.13)),
        (91.78, 47.88, (1.93, 16.03)),
        (158.17, 146.19, (1.615, 55.66)),
    ):
        window = slug_ratio_window(speed * u.km / u.s)
        assert window is not None
        assert window[1] == pytest.approx(ignition_max, rel=1e-3)
        tighter = trade.expansion_limited_slug_ratio_window(speed)
        assert tighter is not None
        assert tighter[0] == pytest.approx(expansion[0], rel=2e-3)
        assert tighter[1] == pytest.approx(expansion[1], rel=2e-3)
        # Strictly inside on *both* sides -- the lower root is the jet asking
        # for more than the merge dissipates, which ignition cannot see.
        assert window[0] < tighter[0] and tighter[1] < window[1]
    # It vanishes below about 65 km/s, where ignition still permits k up to 22.
    assert trade.expansion_limited_slug_ratio_window(64.9) is None
    assert trade.expansion_limited_slug_ratio_window(65.1) is not None
    assert slug_ratio_window(60.0 * u.km / u.s) is not None
    with pytest.raises(ValueError):
        trade.expansion_limited_slug_ratio_window(90.0, 1.0)


def test_the_proposed_slug_ratio_clears_the_expansion_floor(
    deep: trade.DepthTradeRow, shallow: trade.DepthTradeRow
) -> None:
    # The point of the whole correction: k = 8.5 was proposed to protect the
    # plasma and turns out to be needed for the expansion instead -- and it
    # carries about the headroom the deep reference cycle does.
    assert shallow.ceilings.expansion_ceiling == pytest.approx(16.03, rel=2e-3)
    assert trade.CONSERVATIVE_SLUG_RATIO < shallow.ceilings.expansion_ceiling
    assert shallow.ceilings.expansion_margin == pytest.approx(1.55, abs=0.02)
    assert deep.ceilings.expansion_margin == pytest.approx(1.76, abs=0.02)
    assert shallow.ceilings.maximum_jet_efficiency > 0.60


def test_the_ignition_windows_own_upper_root_is_unphysical() -> None:
    # Scored honestly, the point a plasma-only search picks fails outright.
    row = trade.depth_trade_row(32.0, 45.0, 23.75, params=_PARAMS)
    assert row is not None
    assert row.ceilings.expansion_margin == pytest.approx(0.38, abs=0.02)
    assert row.ceilings.maximum_jet_efficiency < 0.01
    assert row.ceilings.binding == "expansion"
    assert row.ceilings.expansion_ceiling is not None
    assert 23.75 > row.ceilings.expansion_ceiling


def test_the_conduction_reserve_is_a_bracket_not_a_number() -> None:
    # The conservative end reserves the whole bill so the plume exits at
    # 15,000 K; the hard end reserves only what cannot come back. ADR 0016's
    # frozen-recombination finding argues for the hard end, so the ADR quotes the
    # conservative one and records both.
    assert trade.conduction_reserve() == pytest.approx(84.41, abs=0.05)
    assert trade.conduction_reserve(6000.0 * u.K) == pytest.approx(65.85, abs=0.05)
    hard = trade.conduction_reserve(reserve_thermal=False)
    assert hard == pytest.approx(trade.frozen_ignition_energy())
    assert hard < trade.conduction_reserve(6000.0 * u.K) < trade.conduction_reserve()


def test_both_ends_of_the_bracket_clear_what_the_cycles_are_scored_at() -> None:
    # The reason ADR 0020's own numbers do not depend on where in the bracket the
    # truth sits: the solar-dive closing speeds are 92-158 km/s, hot enough that
    # even the conservative reserve leaves eta well above the 0.60 scored.
    hard = trade.conduction_reserve(reserve_thermal=False)
    warm = trade.conduction_reserve(6000.0 * u.K)
    for speed, ratio, expected in (
        (158.17, 30.0, (0.759, 0.805, 0.835)),
        (91.78, 8.5, (0.704, 0.746, 0.774)),
    ):
        ceilings = (
            trade.maximum_jet_efficiency(speed, ratio),
            trade.maximum_jet_efficiency(speed, ratio, warm),
            trade.maximum_jet_efficiency(speed, ratio, hard),
        )
        for got, want in zip(ceilings, expected):
            assert got == pytest.approx(want, abs=5e-3)
        assert ceilings[0] < ceilings[1] < ceilings[2]
        assert min(ceilings) > DEFAULT_JET_ENERGY_EFFICIENCY


def test_a_looser_reserve_widens_the_window_both_ways() -> None:
    tight = trade.expansion_limited_slug_ratio_window(91.78)
    loose = trade.expansion_limited_slug_ratio_window(
        91.78, reserve=trade.conduction_reserve(reserve_thermal=False)
    )
    assert tight is not None and loose is not None
    assert loose[0] < tight[0] and loose[1] > tight[1]
    assert loose == pytest.approx((1.74, 27.77), abs=0.02)


@pytest.mark.slow
def test_the_constrained_optimum_lands_where_adr_0020_says() -> None:
    shallow = trade.constrained_growth_optimum(32.0, params=_PARAMS)
    assert shallow is not None
    assert shallow.limited_by == "expansion"
    assert shallow.return_excess == pytest.approx(72.5, abs=0.01)
    # The proposed cap was a guess at a plasma limit and lands within 4% of the
    # expansion-limited optimum, which is the ADR's headline coincidence.
    assert shallow.slug_ratio == pytest.approx(8.18, abs=0.01)
    assert shallow.slug_ratio == pytest.approx(trade.CONSERVATIVE_SLUG_RATIO, rel=0.05)
    assert shallow.row.growth.doubling_years == pytest.approx(1.0795, abs=5e-3)
    assert shallow.row.launch.stated_margin > 1.0

    deep = trade.constrained_growth_optimum(4.0, params=_PARAMS)
    assert deep is not None
    assert deep.limited_by == "expansion"
    assert deep.row.growth.doubling_years == pytest.approx(0.4961, abs=5e-3)
    assert deep.row.launch.stated_margin > 1.0
    # The whole trade in one number: a 64x cooler node costs 2.2x the clock.
    assert (
        shallow.row.growth.doubling_years / deep.row.growth.doubling_years
    ) == pytest.approx(2.18, rel=0.02)


# --------------------------------------------------------------------------
# Two pushes: charging the payload's trip from the pad
# --------------------------------------------------------------------------


def test_the_lob_does_not_deliver_anything_like_escape_speed() -> None:
    # The inconsistency the split-push ledger exists to fix. The launch ledger
    # assumes a 4.09 km/s lob; cycle_growth_ledger starts the burn at 11.0086.
    assert trade.lob_arrival_speed() == pytest.approx(3.60, abs=0.02)
    assert trade.lob_arrival_speed() < 0.4 * _PARAMS.v_esc_leo
    assert trade.lob_arrival_speed(1.0) == 0.0  # too slow to reach 200 km at all


def test_periapsis_speed_barely_notices_a_much_shorter_parking_orbit() -> None:
    # Why the phasing objection is weaker than it looks: v_p is dominated by
    # 2 mu / r_p, so a 6-hour ellipse still reaches 90% of escape speed.
    for days, speed in ((0.25, 9.870), (1.0, 10.571), (5.0, 10.861), (20.0, 10.950)):
        assert trade.parking_orbit_periapsis_speed(days) == pytest.approx(
            speed, abs=5e-3
        )
    assert trade.parking_orbit_periapsis_speed(0.25) > 0.89 * _PARAMS.v_esc_leo


def test_the_reaim_is_cheap_only_on_a_long_parking_orbit() -> None:
    # The split's enabling trick and its binding cost are the same term.
    assert trade.apoapsis_reaim_cost(20.0, 124.8) == pytest.approx(0.207, abs=2e-3)
    assert trade.apoapsis_reaim_cost(5.0, 124.8) == pytest.approx(0.527, abs=2e-3)
    assert trade.apoapsis_reaim_cost(0.25, 124.8) == pytest.approx(4.271, abs=5e-3)
    # It scales as sin(cant/2), so a squarer cant is cheaper to turn through.
    assert trade.apoapsis_reaim_cost(5.0, 89.9) < trade.apoapsis_reaim_cost(5.0, 124.8)


def test_the_overtaking_leg_is_twice_the_engine_the_canted_one_is() -> None:
    # The entire reason to split: at theta = 0 the impactor's own momentum adds.
    along = impulse_per_impactor_kg(0.0 * u.deg, 8.5, 0.60)
    canted = impulse_per_impactor_kg(124.8 * u.deg, 8.5, 0.60)
    square = impulse_per_impactor_kg(89.9 * u.deg, 8.5, 0.60)
    assert along == pytest.approx(3.387, abs=5e-3)
    assert canted == pytest.approx(1.671, abs=5e-3)
    assert along / canted == pytest.approx(2.03, rel=0.02)
    # 4 R_sun's cant lands at 90 degrees, where the bulk term vanishes entirely.
    assert square > canted


def test_the_split_push_ledger_at_both_reference_depths() -> None:
    shallow = trade.split_push_ledger(32.0, 75.0, 8.5, node_survival=1.0 / 3.0)
    assert shallow is not None
    assert shallow.cant_deg == pytest.approx(124.8, abs=0.1)
    assert shallow.overtaking_fraction == pytest.approx(0.7911, abs=1e-3)
    assert shallow.departure_fraction == pytest.approx(0.7201, abs=1e-3)
    assert shallow.split_growth == pytest.approx(3.49, abs=0.02)
    assert shallow.split_doubling_years == pytest.approx(1.816, abs=5e-3)

    deep = trade.split_push_ledger(4.0, 150.0, 30.0, node_survival=0.5)
    assert deep is not None
    assert deep.cant_deg == pytest.approx(89.9, abs=0.1)
    assert deep.split_growth == pytest.approx(23.00, abs=0.05)
    assert deep.split_doubling_years == pytest.approx(0.724, abs=5e-3)

    # The depth penalty, charged honestly on both sides.
    assert shallow.split_doubling_years / deep.split_doubling_years == pytest.approx(
        2.51, rel=0.02
    )


def test_the_split_beats_one_push_and_both_beat_the_committed_fiction() -> None:
    for depth, rex, k, s in ((32.0, 75.0, 8.5, 1.0 / 3.0), (4.0, 150.0, 30.0, 0.5)):
        row = trade.split_push_ledger(depth, rex, k, node_survival=s)
        assert row is not None
        assert row.single_growth < row.split_growth < row.free_parking_growth
        assert row.split_advantage > 1.0
        assert (
            row.free_parking_doubling_years
            < row.split_doubling_years
            < row.single_doubling_years
        )


def test_the_split_saturates_long_before_the_phasing_gets_awkward() -> None:
    rows = trade.split_push_period_sweep()
    growth = [r.split_growth for r in rows]
    assert growth == sorted(growth)  # monotone in the parking period
    by_period = {r.parking_period_days: r.split_growth for r in rows}
    # 5 days captures most of what 40 days offers, at a quarter of Earth's drift.
    assert by_period[5.0] / by_period[40.0] > 0.94
    # Below half a day the re-aim has eaten the whole advantage.
    short = trade.split_push_ledger(parking_period_days=0.125, node_survival=1.0 / 3.0)
    assert short is not None
    assert short.split_growth < short.single_growth


# --------------------------------------------------------------------------
# The launch ledger re-scored: what the cycle returns per kilogram off the pad
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deep_closure() -> SynodicCycleClosure:
    """The 4 solar-radii closure, solved once for the pad-ledger tests."""
    closure = solve_synodic_closure(3, _PARAMS, 4.0, 150.0, 1.0)
    assert closure is not None
    return closure


@pytest.fixture(scope="module")
def shallow_closure() -> SynodicCycleClosure:
    """The 32 solar-radii closure, solved once for the pad-ledger tests."""
    closure = solve_synodic_closure(3, _PARAMS, 32.0, 75.0, 1.0)
    assert closure is not None
    return closure


def _pad(depth: float, excess: float, k: float, survival: float) -> float:
    """Kilograms returned per pad kilogram on the split push at one point."""
    row = trade.split_push_ledger(depth, excess, k, node_survival=survival)
    assert row is not None
    return trade.split_push_launch_ledger(row, _PARAMS).returned_per_pad_kg


def test_the_two_ledgers_now_start_the_payload_in_the_same_place() -> None:
    # The whole content of the correction. PAYLOAD_FRACTION_AT_INTERCEPT is what
    # a 4.09 km/s lob delivers to the intercept point; lob_arrival_speed() is the
    # speed it delivers it at. The free-parking ledger charged the first and
    # ignored the second, which is how 7.4 km/s went unpaid.
    assert trade.PAYLOAD_FRACTION_AT_INTERCEPT == pytest.approx(0.25)
    assert trade.LOB_DELTA_V == pytest.approx(4.09, abs=1e-9)
    from src.two_leg_nozzle_sweep import LAUNCH_PROPELLANT_FRACTION

    # 2/3 of liftoff as propellant at 380 s *is* the 4.09 km/s lob, so the two
    # constants are one assumption and cannot drift apart.
    implied = -_PARAMS.exhaust_speed * np.log(1.0 - LAUNCH_PROPELLANT_FRACTION)
    assert implied == pytest.approx(trade.LOB_DELTA_V, abs=0.01)


def test_charging_the_lob_costs_about_a_third_of_the_pad_return() -> None:
    # The paper's own estimate of the size of this correction, checked: "charging
    # it changes the mass that survives to departure by about a third".
    for depth, excess, k, survival in (
        (4.0, 150.0, 30.0, 0.5),
        (32.0, 75.0, 8.5, 1.0 / 3.0),
    ):
        row = trade.split_push_ledger(depth, excess, k, node_survival=survival)
        assert row is not None
        launch = trade.split_push_launch_ledger(row, _PARAMS)
        assert launch.correction_factor == pytest.approx(0.68, abs=0.02)
        # The chain the free-parking reading set to 1 is three real charges.
        assert launch.chain_to_departure == pytest.approx(
            row.overtaking_fraction * row.reaim_fraction * row.departure_fraction
        )
        assert launch.free_parking_returned_per_pad_kg > launch.returned_per_pad_kg


def test_the_corrected_pad_return_at_both_reference_cycles() -> None:
    # The figures the paper says are owed. The deep cycle survives the
    # correction by 4 percent; the shallow one does not survive it at all.
    deep = _pad(4.0, 150.0, 30.0, 0.5)
    assert deep == pytest.approx(0.0695, abs=2e-4)
    assert deep / RETURN_FLOOR == pytest.approx(1.043, abs=5e-3)

    shallow = _pad(32.0, 75.0, 8.5, 1.0 / 3.0)
    assert shallow == pytest.approx(0.0412, abs=2e-4)
    assert shallow / RETURN_FLOOR == pytest.approx(0.618, abs=5e-3)


def test_the_shallow_cycle_loses_the_rescale_that_used_to_rescue_it() -> None:
    # ADR 0019's argument was that the 1/15 floor was calibrated on a ~60 km/s
    # return, so a faster one deserves a lower bar. It still holds and it is
    # still not enough here: the rescale is worth 1.46x at 85 km/s against 2.80x
    # at 158, and the correction is worth more than the rescale.
    shallow = trade.split_push_ledger(32.0, 75.0, 8.5, node_survival=1.0 / 3.0)
    deep = trade.split_push_ledger(4.0, 150.0, 30.0, node_survival=0.5)
    assert shallow is not None and deep is not None
    shallow_launch = trade.split_push_launch_ledger(shallow, _PARAMS)
    deep_launch = trade.split_push_launch_ledger(deep, _PARAMS)

    assert shallow_launch.return_value_ratio == pytest.approx(1.46, abs=0.02)
    assert deep_launch.return_value_ratio == pytest.approx(2.80, abs=0.02)
    # Before the correction the rescale carried the shallow cycle over; now it
    # does not, which is the verdict that actually changed.
    assert (
        shallow_launch.free_parking_returned_per_pad_kg > shallow_launch.rescaled_floor
    )
    assert not shallow_launch.clears_rescaled_floor
    assert not shallow_launch.clears_committed_floor
    assert deep_launch.clears_committed_floor and deep_launch.clears_rescaled_floor


def test_the_shallow_cycle_fails_the_pad_floor_at_every_slug_ratio(
    shallow_closure: SynodicCycleClosure,
) -> None:
    # Not "fails at k = 8.5" -- fails everywhere, which is a different statement
    # and the one that rules the operating point out rather than moving it.
    ceilings = trade.split_push_pad_ceilings(
        shallow_closure, 1.0 / 3.0, label="32 R_sun", params=_PARAMS
    )
    assert ceilings is not None
    assert ceilings.launch_window is None
    assert not ceilings.committed_floor_reachable
    assert ceilings.peak_slug_ratio == pytest.approx(1.93, abs=0.02)
    assert ceilings.peak_margin == pytest.approx(0.761, abs=5e-3)


def test_the_split_push_added_a_colder_leg_than_any_the_old_ledger_saw(
    shallow_closure: SynodicCycleClosure, shallow: trade.DepthTradeRow
) -> None:
    # The second thing the correction changed. At theta = 0 the vehicle's own
    # speed subtracts from the closing speed directly, so the overtaking leg
    # ends colder than the canted leg ever gets -- and the expansion floor was
    # never asked of it, because in the single-push ledger it did not exist.
    ceilings = trade.split_push_pad_ceilings(
        shallow_closure, 1.0 / 3.0, label="32 R_sun", params=_PARAMS
    )
    assert ceilings is not None
    assert ceilings.overtaking_coldest_closing == pytest.approx(74.21, abs=0.05)
    assert ceilings.departure_coldest_closing == pytest.approx(91.71, abs=0.05)
    assert ceilings.overtaking_coldest_closing < ceilings.departure_coldest_closing
    # The single-push ledger's coldest instant is the canted leg's, so it reads
    # the expansion floor off the warmer of the two.
    assert shallow.ceilings.coldest_closing_speed > ceilings.overtaking_coldest_closing
    # And at that colder speed no slug ratio leaves the plume conducting.
    assert ceilings.overtaking_expansion_window is None
    assert ceilings.departure_expansion_window is not None
    assert ceilings.expansion_window is None
    assert ceilings.admissible_window is None
    assert ceilings.binding == "launch and expansion (neither admits any k)"


def test_the_deep_cycle_is_admissible_and_its_k_sits_just_under_the_ceiling(
    deep_closure: SynodicCycleClosure,
) -> None:
    ceilings = trade.split_push_pad_ceilings(
        deep_closure, 0.5, label="4 R_sun", params=_PARAMS
    )
    assert ceilings is not None
    window = ceilings.admissible_window
    assert window is not None
    # ADR 0019's k = 30 is inside, with about 1 percent of headroom -- so it is
    # a defensible operating point and not a comfortable one.
    assert window[0] < DEFAULT_SLUG_RATIO < window[1]
    assert DEFAULT_SLUG_RATIO / window[1] > 0.98
    # The expansion floor binds before the pad floor does, but only just: the
    # committed floor would have allowed k up to about 35.
    assert ceilings.binding == "expansion"
    assert ceilings.launch_window is not None
    assert ceilings.launch_window[1] == pytest.approx(34.97, abs=0.05)


def test_growth_and_pad_return_still_pull_opposite_ways_after_the_correction(
    shallow_closure: SynodicCycleClosure,
) -> None:
    # The opposition that makes the launch ledger a real cap survives being
    # charged honestly: slug is free per impactor kilogram and not per pad one.
    rows = trade.split_push_slug_ratio_table(shallow_closure, 1.0 / 3.0, params=_PARAMS)
    assert len(rows) > 4
    growth = [row.split_growth for _, row, _ in rows]
    pad = [launch.returned_per_pad_kg for _, _, launch in rows]
    assert growth == sorted(growth)
    assert pad == sorted(pad, reverse=True)


def test_the_pad_floor_is_lost_partway_down_the_depth_dial() -> None:
    rows = trade.split_push_depth_table(params=_PARAMS)
    margins = [launch.stated_margin for _, _, launch in rows]
    assert margins == sorted(margins, reverse=True)  # monotone in depth
    assert margins[0] > 1.0 and margins[-1] < 1.0
    crossing = trade.pad_floor_depth(params=_PARAMS)
    assert crossing is not None
    assert crossing == pytest.approx(21.09, abs=0.05)
    # The reference shallow cycle sits well past it, and the deep one well short.
    assert 4.0 < crossing < 32.0


@pytest.mark.slow
def test_no_climb_out_rescues_the_shallow_cycle() -> None:
    # The squeeze, and the reason the verdict is "ruled out" rather than
    # "mis-tuned": the pad ledger wants a cheap node boost and the expansion
    # floor wants the hot closing speed a big one buys. At 32 solar radii the
    # two never overlap on a point that pays.
    rows = trade.pad_return_frontier(32.0, params=_PARAMS)
    assert rows, "no closure anywhere on the frontier grid"
    assert not any(row.clears_committed_floor for row in rows)
    admissible = [row for row in rows if row.expansion_window is not None]
    assert admissible, "the expansion floor should admit something somewhere"
    best = max(row.best_margin or 0.0 for row in admissible)
    assert best == pytest.approx(0.625, abs=5e-3)
    # Everything below an 85 km/s climb-out is barred by the overtaking leg, and
    # those are exactly the rows whose pad return would have been good enough.
    barred = [row for row in rows if row.expansion_window is None]
    assert barred and max(row.return_excess for row in barred) < 85.0


@pytest.mark.slow
def test_every_admissible_point_at_four_solar_radii_pays_for_its_launch() -> None:
    rows = [
        row
        for row in trade.pad_return_frontier(4.0, params=_PARAMS)
        if row.expansion_window is not None
    ]
    assert rows
    assert all(row.clears_committed_floor for row in rows)
    assert min(row.best_margin or 0.0 for row in rows) > 1.2


def test_the_paper_dive_carries_the_same_defect_and_the_ranking_survives_it(
    deep_closure: SynodicCycleClosure,
) -> None:
    # The comparison row was scored the same way and had to be re-derived before
    # either figure could be quoted. Charged from the pad it doubles in 0.377 yr
    # rather than 0.305 -- it has 35.5 km/s to fly from the lob, far more than
    # either Jovian cycle -- and it still fails the committed floor, as it did
    # before. So the correction moves both figures and reverses nothing.
    paper = trade.paper_resonant_dive_split_push(
        deep_closure.return_excess,
        deep_closure.push_axis_deg,
        slug_ratio=DEFAULT_SLUG_RATIO,
        params=_PARAMS,
    )
    assert paper is not None
    assert paper.departure_speed - paper.lob_speed == pytest.approx(35.51, abs=0.05)
    assert paper.split_doubling_years == pytest.approx(0.377, abs=2e-3)

    launch = trade.split_push_launch_ledger(paper, _PARAMS)
    assert launch.stated_margin == pytest.approx(0.486, abs=5e-3)
    assert not launch.clears_committed_floor

    jovian = trade.split_push_ledger(4.0, 150.0, DEFAULT_SLUG_RATIO, node_survival=0.6)
    assert jovian is not None
    # Unchanged for the sixth time: the paper's dive grows faster, the Jovian
    # cycle is the one that pays for its launch.
    assert paper.split_doubling_years < jovian.split_doubling_years
    assert trade.split_push_launch_ledger(jovian, _PARAMS).clears_committed_floor


def test_the_mirror_sign_is_chosen_not_assumed(
    shallow_closure: SynodicCycleClosure,
) -> None:
    # split_push_from_state flies both departure-hyperbola mirror signs and keeps
    # the better, exactly as departure_nozzle_ledger does. On both Jovian cycles
    # the plus sign wins, so the published numbers are unchanged by the choice --
    # but the paper's resonant dive aims somewhere else entirely and needs it.
    row = trade.split_push_from_state(
        "probe",
        shallow_closure.departure_excess,
        shallow_closure.departure_aim_deg,
        shallow_closure.total_tof_years,
        shallow_closure.return_excess,
        shallow_closure.push_axis_deg,
        1.0 / 3.0,
        8.5,
        params=_PARAMS,
    )
    assert row is not None
    assert row.cant_deg == pytest.approx(124.84, abs=0.02)  # the plus-sign cant


@pytest.mark.slow
def test_the_fastest_admissible_deep_cycle_is_expansion_limited_not_pad_limited() -> (
    None
):
    # The corrected replacement for ADR 0020's constrained-optimum row. Growth
    # per impactor kilogram rises with k throughout, so the fastest cycle sits on
    # whichever cap arrives first -- and at 4 solar radii that is still the
    # expansion floor, with the pad floor a comfortable 1.24x behind it.
    rows = [
        row
        for row in trade.pad_return_frontier(4.0, params=_PARAMS)
        if row.growth_doubling_years is not None
    ]
    assert rows
    fastest = min(rows, key=lambda row: row.growth_doubling_years or float("inf"))
    assert fastest.return_excess == pytest.approx(150.0)
    assert fastest.growth_slug_ratio == pytest.approx(30.40, abs=0.05)
    assert fastest.growth_doubling_years == pytest.approx(0.684, abs=3e-3)
    assert fastest.growth_margin == pytest.approx(1.24, abs=0.02)
    # Above 150 km/s the pad floor takes over as the binding cap, which is what
    # a margin pinned at exactly 1.0 means.
    pad_limited = [row for row in rows if row.return_excess > 150.0]
    assert pad_limited
    assert all(
        (row.growth_margin or 0.0) == pytest.approx(1.0, abs=1e-3)
        for row in pad_limited
    )


@pytest.mark.slow
def test_the_overtaking_leg_puts_a_floor_under_the_climb_out_at_every_depth() -> None:
    # New with the split push and it applies to both depths: below a threshold
    # closing speed the overtaking leg's plume stops conducting, so the
    # cheap-boost end of the dial -- exactly where the pad ledger wanted to go --
    # is unavailable whatever the depth.
    #
    # Asserted against the threshold rather than against a climb-out value,
    # because which grid points fall either side of it is a property of
    # PAD_FRONTIER_EXCESS_GRID and not of the physics.
    threshold = trade.conduction_threshold_closing_speed(
        0.60, trade.DEFAULT_EXPANSION_MARGIN
    )
    assert threshold is not None
    for depth in (4.0, 32.0):
        rows = trade.pad_return_frontier(depth, params=_PARAMS)
        barred = [row for row in rows if row.expansion_window is None]
        allowed = [row for row in rows if row.expansion_window is not None]
        assert barred and allowed
        # It is the overtaking leg that decides, and it decides on its coldest
        # instant against the threshold -- nothing else.
        assert all(row.overtaking_coldest_closing < threshold for row in barred)
        assert all(row.overtaking_coldest_closing >= threshold for row in allowed)
        # So the barred set is a prefix of the climb-out grid, not a scatter.
        assert max(row.return_excess for row in barred) < min(
            row.return_excess for row in allowed
        )


# --------------------------------------------------------------------------
# Does the pad verdict survive every reading of the plume physics?
# --------------------------------------------------------------------------


def test_the_reserve_moves_the_climb_out_floor_not_the_pad_return() -> None:
    # The coupling, isolated. The conduction reserve is a plume-thermodynamics
    # quantity and the pad return is a mass quantity, so the reserve cannot
    # change what a given k returns. What it changes is which k -- and therefore
    # which climb-out -- is admissible at all.
    row = trade.split_push_ledger(32.0, 75.0, 8.5, node_survival=1.0 / 3.0)
    assert row is not None
    before = trade.split_push_launch_ledger(row, _PARAMS).returned_per_pad_kg
    # Nothing in split_push_ledger takes a reserve, which is the point: the two
    # only meet through which slug ratios the expansion floor admits.
    for reserve in (84.41, 65.85, 53.47):
        window = trade.expansion_limited_slug_ratio_window(
            row.overtaking_coldest_closing, 0.60, 1.5, reserve
        )
        assert window is None or window[0] < window[1]
    assert trade.split_push_launch_ledger(row, _PARAMS).returned_per_pad_kg == before


def test_a_looser_reserve_lets_the_plume_survive_a_colder_burn() -> None:
    # Monotone, and it is the whole mechanism: reserving less leaves the plume
    # conducting further down, so a slower closing speed becomes flyable.
    thresholds = [
        trade.conduction_threshold_closing_speed(0.60, 1.5, reserve)
        for _, reserve in trade.CONDUCTION_RESERVE_BRACKET
    ]
    assert all(t is not None for t in thresholds)
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[0] == pytest.approx(79.56, abs=0.05)  # 15000 K, all reserved
    assert thresholds[-1] == pytest.approx(63.33, abs=0.05)  # frozen chemistry only
    # Dropping the margin to 1.0 moves it further than crossing the bracket does.
    assert trade.conduction_threshold_closing_speed(
        0.60, 1.0, trade.CONDUCTION_RESERVE_BRACKET[0][1]
    ) == pytest.approx(64.96, abs=0.05)


@pytest.fixture(scope="module")
def shallow_bracket() -> List[trade.ConductionBracketRow]:
    """The 32 solar-radii conduction sweep, run once for both claims about it.

    Six frontier scans, so it is the most expensive fixture in the file; the two
    halves of the verdict -- robust at 1.5x, not robust at unit margin -- read
    off the same rows rather than paying for them twice.
    """
    return trade.conduction_bracket_frontier(32.0, params=_PARAMS)


@pytest.mark.slow
def test_the_shallow_failure_is_robust_at_the_operating_margin(
    shallow_bracket: List[trade.ConductionBracketRow],
) -> None:
    # The claim that matters: at ADR 0020's 1.5x margin the shallow cycle fails
    # at every reading of the conduction reserve, so the verdict does not rest
    # on where in that bracket the truth sits.
    rows = [row for row in shallow_bracket if row.expansion_margin == 1.5]
    assert len(rows) == len(trade.CONDUCTION_RESERVE_BRACKET)
    assert not any(row.clears_committed_floor for row in rows)
    # The hard end gets close, which is worth pinning rather than rounding away.
    assert max(row.best_margin or 0.0 for row in rows) == pytest.approx(0.979, abs=0.01)


@pytest.mark.slow
def test_but_it_does_not_survive_unit_margin_plus_a_cooler_outlet(
    shallow_bracket: List[trade.ConductionBracketRow],
) -> None:
    # The honest limit of the verdict, and the reason the margin is load-bearing
    # rather than decorative. At unit margin -- sitting exactly on the expansion
    # boundary, which is what ADR 0020 introduced the margin to prevent -- a
    # 6000 K outlet or looser does let the shallow cycle pay.
    rows = {(row.reserve_label, row.expansion_margin): row for row in shallow_bracket}
    assert not rows[("15000 K, all reserved", 1.0)].clears_committed_floor
    for label in ("6000 K", "frozen chemistry only"):
        winner = rows[(label, 1.0)]
        assert winner.clears_committed_floor
        # It pays by moving to a much cooler climb-out than the reference 75.
        assert winner.best_return_excess is not None
        assert winner.best_return_excess <= 55.0
        # And what it buys is slower than the cycle the paper already has: the
        # Jupiter-only chain doubles in 1.74 yr.
        assert winner.best_doubling_years is not None
        assert winner.best_doubling_years > 1.74
