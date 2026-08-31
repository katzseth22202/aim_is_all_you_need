"""Tests for the Jovian solar-dive cycle in src/jovian_solar_dive_cycle.py.

The load-bearing claim under test is asymmetric: the three-synodic cycle closes
with a strictly unpowered Jovian flyby, and the two-synodic one does not close
at any perijove burn.  Most tests below therefore check *structure* -- floors,
monotonicity, sign of a margin, the shape of the deficit curve -- rather than
pinning digits, so a re-tuned search that still tells the same physical story
keeps passing.  The handful of pinned numbers are the ones ADR 0019 quotes.
"""

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Sun

from src import jovian_solar_dive_cycle as cycle
from src.circular_resonance_impulse import impulse_per_impactor_kg
from src.conic_kernel import hyperbolic_eccentricity, unpowered_bend_angle
from src.retrograde_return_legs import _powered_flyby_params

_PARAMS = _powered_flyby_params()
_DIVE_RADIUS = cycle._dive_periapsis_radius()


def test_synodic_period_matches_the_earth_jupiter_beat() -> None:
    # 1/T_syn = 1/T_earth - 1/T_jupiter with Jupiter at 5.2028 AU: 1.0920 yr.
    years = cycle.synodic_period(_PARAMS) / cycle._SECONDS_PER_YEAR
    assert years == pytest.approx(1.0920, abs=5e-4)


def test_dive_radius_is_four_solar_radii() -> None:
    assert cycle._dive_periapsis_radius() == pytest.approx(
        4.0 * float(Sun.R.to_value(u.km))
    )


def test_placement_floors_are_ordered_around_jupiters_orbital_speed() -> None:
    # A flyby rotates the excess but cannot rescale it, so placing the dive means
    # cancelling Jupiter's tangential motion. A prograde dive keeps a sliver of
    # prograde tangential speed and so needs slightly less than Jupiter's own
    # 13.06 km/s; a radial plunge needs exactly it; a retrograde dive needs the
    # same sliver on the far side.
    prograde = cycle.dive_placement_excess_floor(1.0, params=_PARAMS)
    radial = cycle.dive_placement_excess_floor(0.0, params=_PARAMS)
    retrograde = cycle.dive_placement_excess_floor(-1.0, params=_PARAMS)
    assert prograde < radial < retrograde
    assert radial == pytest.approx(_PARAMS.v_jupiter_orbit)
    # The two dive floors sit symmetrically about the plunge, because they differ
    # only in the sign of the same small residual tangential speed.
    assert radial - prograde == pytest.approx(retrograde - radial, rel=1e-9)


def test_placement_floors_match_the_quoted_values() -> None:
    assert cycle.dive_placement_excess_floor(1.0, params=_PARAMS) == pytest.approx(
        11.9557, abs=1e-3
    )
    assert cycle.dive_placement_excess_floor(-1.0, params=_PARAMS) == pytest.approx(
        14.1602, abs=1e-3
    )


def test_deeper_dive_needs_a_hotter_arrival() -> None:
    # Less angular momentum survives a deeper perihelion, so more of Jupiter's
    # motion has to be cancelled.
    shallow = cycle.dive_placement_excess_floor(
        1.0, dive_radius=cycle._dive_periapsis_radius(9.86), params=_PARAMS
    )
    deep = cycle.dive_placement_excess_floor(
        1.0, dive_radius=cycle._dive_periapsis_radius(2.0), params=_PARAMS
    )
    assert deep > shallow


def test_solved_dive_state_really_has_the_target_perihelion() -> None:
    # The whole placement solve exists to put perihelion at 4 solar radii; check
    # it against an independent vis-viva reconstruction of the periapsis radius.
    state = cycle._solve_dive_state(17.75, 1.0, _DIVE_RADIUS, _PARAMS)
    assert state is not None
    v_tangential, v_radial = state
    assert v_radial < 0.0  # the dive is inbound
    conic = cycle.conic_state_at_radius(
        _PARAMS.mu_sun, _PARAMS.r_jupiter_orbit, v_tangential, v_radial
    )
    periapsis = conic.p / (1.0 + conic.ecc)
    assert periapsis == pytest.approx(_DIVE_RADIUS, rel=1e-8)


def test_no_dive_state_below_the_placement_floor() -> None:
    floor = cycle.dive_placement_excess_floor(1.0, params=_PARAMS)
    assert cycle._solve_dive_state(floor - 0.5, 1.0, _DIVE_RADIUS, _PARAMS) is None
    assert cycle._solve_dive_state(floor + 0.5, 1.0, _DIVE_RADIUS, _PARAMS) is not None


def test_outbound_leg_flight_time_falls_with_departure_excess() -> None:
    # The clock solve assumes this monotonicity to bracket its root.
    times = []
    for excess in (11.0, 13.0, 16.0, 20.0, 26.0):
        leg = cycle._outbound_leg(excess, 0.0, _PARAMS)
        assert leg is not None
        times.append(leg.tof)
    assert all(later < earlier for earlier, later in zip(times, times[1:]))


def test_outbound_leg_returns_none_when_aphelion_falls_short() -> None:
    # 4 km/s of excess leaves an aphelion well inside Jupiter's orbit; the
    # kernel's reachability guard must surface as None, not an exception.
    assert cycle._outbound_leg(4.0, 0.0, _PARAMS) is None


def test_climb_out_sweeps_the_papers_whip_around_angle() -> None:
    # sec:earth_reintercept: the boosted climb-out from the dive covers about
    # 130 degrees of heliocentric longitude before it re-crosses 1 AU.
    outbound = cycle._outbound_leg(12.679, 10.71, _PARAMS)
    assert outbound is not None
    dive = cycle._dive_leg(outbound, 1.0, _PARAMS, _DIVE_RADIUS, None)
    assert dive is not None
    climb = cycle._climb_leg(dive, 1.0, _PARAMS, _DIVE_RADIUS, 150.0)
    assert climb is not None
    assert float(np.degrees(climb.swept)) == pytest.approx(130.4, abs=0.5)
    # And it is short: the paper's "the last leg would be brief".
    assert climb.tof / 86400.0 == pytest.approx(10.35, abs=0.2)


def test_periapsis_boost_is_near_the_papers_36_km_s() -> None:
    # sec:four_radii_thermal sizes a ~36 km/s boost at 4 solar radii to leave
    # ~150 km/s of excess. Arriving from Jupiter rather than 1 AU is a slightly
    # hotter arrival, so the boost comes out a little cheaper.
    outbound = cycle._outbound_leg(12.679, 10.71, _PARAMS)
    assert outbound is not None
    dive = cycle._dive_leg(outbound, 1.0, _PARAMS, _DIVE_RADIUS, None)
    assert dive is not None
    climb = cycle._climb_leg(dive, 1.0, _PARAMS, _DIVE_RADIUS, 150.0)
    assert climb is not None
    assert 33.0 < climb.periapsis_boost < 36.0


def test_unpowered_dive_leg_keeps_the_arrival_excess() -> None:
    outbound = cycle._outbound_leg(12.679, 10.71, _PARAMS)
    assert outbound is not None
    dive = cycle._dive_leg(outbound, 1.0, _PARAMS, _DIVE_RADIUS, None)
    assert dive is not None
    assert dive.excess_out == pytest.approx(dive.excess_in)
    assert dive.perijove_burn == 0.0
    # And the bend it can offer is exactly the kernel's unpowered turn.
    ecc = hyperbolic_eccentricity(
        _PARAMS.mu_jupiter, _PARAMS.periapsis_floor, dive.excess_in
    )
    assert dive.bend_available == pytest.approx(unpowered_bend_angle(ecc))


def test_three_synodic_closes_unpowered_with_room_to_spare() -> None:
    closure = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert closure is not None
    assert closure.feasible
    assert closure.perijove_burn == 0.0
    # Both closure conditions actually hold.
    assert closure.total_tof_years == pytest.approx(
        3.0 * cycle.synodic_period(_PARAMS) / cycle._SECONDS_PER_YEAR, rel=1e-9
    )
    assert closure.intercept_miss_deg == pytest.approx(0.0, abs=1e-6)
    # The numbers ADR 0019 quotes.
    assert closure.departure_excess == pytest.approx(10.54, abs=0.02)
    assert closure.departure_speed_200km == pytest.approx(15.24, abs=0.02)
    assert closure.bend_available_deg - closure.bend_required_deg == pytest.approx(
        58.8, abs=0.3
    )


def test_two_synodic_does_not_close_unpowered() -> None:
    closure = cycle.solve_synodic_closure(2, params=_PARAMS)
    assert closure is not None
    # A closure of the clock and the intercept exists -- the geometry is not the
    # problem, and in particular the return does not have to pass through the Sun.
    assert closure.total_tof_years == pytest.approx(
        2.0 * cycle.synodic_period(_PARAMS) / cycle._SECONDS_PER_YEAR, rel=1e-9
    )
    assert closure.intercept_miss_deg == pytest.approx(0.0, abs=1e-6)
    # What fails is the Jovian bend.
    assert not closure.feasible
    assert closure.bend_deficit_deg == pytest.approx(6.84, abs=0.1)


def test_two_synodic_departure_is_hotter_than_three_synodic() -> None:
    # The shorter clock has to be bought with departure energy, which is what
    # drives the arrival excess up and the available bend down.
    two = cycle.solve_synodic_closure(2, params=_PARAMS)
    three = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert two is not None and three is not None
    assert two.departure_excess > three.departure_excess
    assert two.jupiter_arrival_excess > three.jupiter_arrival_excess
    assert two.bend_available_deg < three.bend_available_deg
    assert two.bend_required_deg > three.bend_required_deg


def test_three_synodic_pays_for_its_clock_in_payload_mass_ratio() -> None:
    # The slower cycle departs more gently, so each Earth-side collision buys
    # more payload -- the trade the ADR weighs against the longer clock.
    two = cycle.solve_synodic_closure(2, params=_PARAMS)
    three = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert two is not None and three is not None
    assert three.earth_push_mass_ratio > two.earth_push_mass_ratio
    # Both are far above the paper's Earth-side dive injection, which pushes to
    # 39.11 km/s at 200 km instead of 15-17.
    assert three.earth_push_mass_ratio > 10.0


def test_both_closures_see_the_same_returning_stream() -> None:
    # The return leg is set by the dive radius and the boost, not by the clock,
    # so the collision speed and push axis are common to 2S and 3S.
    two = cycle.solve_synodic_closure(2, params=_PARAMS)
    three = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert two is not None and three is not None
    assert two.earth_closing_speed == pytest.approx(three.earth_closing_speed, rel=1e-6)
    assert two.push_axis_deg == pytest.approx(three.push_axis_deg, rel=1e-6)
    # The stream arrives all but radially outward, which is why the next
    # departure is a heavily canted collision in both cases.
    assert two.push_axis_deg == pytest.approx(98.55, abs=0.1)
    assert two.aim_separation_deg > 80.0
    assert three.aim_separation_deg > 80.0


def test_perijove_burn_never_closes_the_two_synodic_bend() -> None:
    # The negative result ADR 0019 rests on: sweeping the burn over every
    # magnitude and sign leaves an interior minimum strictly above zero.
    probes = cycle.perijove_burn_sweep(2, params=_PARAMS)
    assert len(probes) > 20
    deficits = [probe.bend_deficit_deg for probe in probes]
    assert min(deficits) > 0.0
    assert min(deficits) == pytest.approx(6.14, abs=0.15)
    # It is a genuine interior minimum, not a monotone trend running off the
    # edge of the sweep: both ends are worse than the middle.
    best = int(np.argmin(deficits))
    assert 0 < best < len(deficits) - 1


def test_perijove_burn_sweep_brackets_the_unpowered_point() -> None:
    # Where the sweep's outgoing excess happens to equal the arrival excess, the
    # burn vanishes and the deficit must match the unpowered closure's.
    probes = cycle.perijove_burn_sweep(2, params=_PARAMS)
    unpowered = min(probes, key=lambda probe: probe.perijove_burn)
    assert unpowered.perijove_burn < 0.2
    assert unpowered.bend_deficit_deg == pytest.approx(6.84, abs=0.3)


def test_soi_seam_correction_is_zero_for_a_feasible_closure() -> None:
    three = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert three is not None
    assert cycle.soi_seam_correction(three) == 0.0


def test_soi_seam_correction_prices_the_two_synodic_gap() -> None:
    two = cycle.solve_synodic_closure(2, params=_PARAMS)
    assert two is not None
    correction = cycle.soi_seam_correction(two)
    assert correction == pytest.approx(2.119, abs=0.02)
    # It is the chord of the leftover rotation, so it scales with the arrival
    # excess as well as the deficit.
    expected = (
        2.0
        * two.jupiter_arrival_excess
        * np.sin(np.radians(two.bend_deficit_deg) / 2.0)
    )
    assert correction == pytest.approx(float(expected))


@pytest.mark.slow
def test_deep_space_maneuver_closes_two_synodic_for_about_two_km_s() -> None:
    correction = cycle.minimum_dsm_correction(2, params=_PARAMS)
    assert correction is not None
    assert correction.magnitude == pytest.approx(2.27, abs=0.15)
    # It drives the bend exactly onto the perijove floor -- that is the third
    # equation the maneuver's extra freedom buys.
    assert correction.bend_required_deg == pytest.approx(
        correction.bend_available_deg, abs=1e-6
    )
    # And it fires outside Jupiter's ~0.322 AU sphere of influence, so it is a
    # heliocentric maneuver rather than a disguised Jupiter-relative burn.
    assert correction.radius_au < 4.88


@pytest.mark.slow
def test_deep_space_maneuver_beats_no_correction_but_not_by_much() -> None:
    # The point of pricing both: a mid-course DSM is the textbook cheap fix, and
    # here it is within ~10 percent of simply rotating the excess at the seam.
    # So ~2 km/s is the correction scale wherever the burn is placed.
    two = cycle.solve_synodic_closure(2, params=_PARAMS)
    correction = cycle.minimum_dsm_correction(2, params=_PARAMS)
    assert two is not None and correction is not None
    seam = cycle.soi_seam_correction(two)
    assert 0.8 < correction.magnitude / seam < 1.3


def test_solve_synodic_closure_rejects_a_nonpositive_multiple() -> None:
    with pytest.raises(ValueError):
        cycle.solve_synodic_closure(0, params=_PARAMS)


# --------------------------------------------------------------------------
# The departure nozzle and the growth ledger (ADR 0019 addendum)
# --------------------------------------------------------------------------


def _three_synodic_ledger(**kwargs: float) -> cycle.CycleGrowthLedger:
    closure = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert closure is not None
    return cycle.cycle_growth_ledger(
        "3S",
        closure.departure_excess,
        closure.departure_aim_deg,
        closure.total_tof_years,
        closure.return_excess,
        closure.push_axis_deg,
        params=_PARAMS,
        **kwargs,
    )


def test_impulse_law_efficiency_only_scales_the_exhaust_term() -> None:
    # The cos(theta) term is the impactor's own momentum arriving through the
    # nozzle; momentum conservation does not care how good the nozzle is. So at
    # 90 degrees, where cos vanishes, the whole law must scale as sqrt(eta).
    ideal = impulse_per_impactor_kg(90.0 * u.deg, 30.0, 1.0)
    lossy = impulse_per_impactor_kg(90.0 * u.deg, 30.0, 0.6)
    bare = np.sqrt(0.6 * 31.0 - 1.0) / np.sqrt(31.0 - 1.0)
    assert lossy / ideal == pytest.approx(float(bare), rel=1e-9)


def test_impulse_law_default_efficiency_is_the_loss_free_ceiling() -> None:
    assert impulse_per_impactor_kg(120.0 * u.deg, 9.0) == pytest.approx(
        impulse_per_impactor_kg(120.0 * u.deg, 9.0, 1.0)
    )


def test_vehicle_frame_angle_differs_from_the_soi_aim_separation() -> None:
    # CONTEXT.md: the patched-conic aim separation is a diagnostic and must not
    # be fed to the impulse law. The free departure-hyperbola mirror plus the
    # vehicle's own motion move it by tens of degrees, and in the helpful
    # direction here.
    closure = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert closure is not None
    ledger = _three_synodic_ledger()
    assert closure.aim_separation_deg == pytest.approx(110.6, abs=0.2)
    assert ledger.nozzle.impact_angle_start_deg == pytest.approx(93.9, abs=0.3)
    assert ledger.nozzle.impact_angle_start_deg < closure.aim_separation_deg


def test_departure_nozzle_picks_the_better_mirror_sign() -> None:
    # Both mirror images give the same outgoing excess, so the sign is free and
    # the ledger must take the one that delivers more.
    closure = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert closure is not None
    ledger = _three_synodic_ledger()
    mirror = abs(ledger.nozzle.thrust_axis_deg - closure.departure_aim_deg)
    assert mirror == pytest.approx(20.67, abs=0.1)
    # The chosen axis is the one rotated toward the incoming stream.
    assert ledger.nozzle.thrust_axis_deg > closure.departure_aim_deg


def test_three_synodic_departure_on_a_k30_nozzle_delivers_most_of_the_mass() -> None:
    ledger = _three_synodic_ledger()
    assert ledger.nozzle.effective_exhaust_speed == pytest.approx(21.7, abs=0.3)
    assert ledger.nozzle.specific_impulse == pytest.approx(2214, abs=30)
    assert ledger.nozzle.delivered_fraction == pytest.approx(0.823, abs=0.005)
    assert ledger.round_trip_fraction == pytest.approx(0.494, abs=0.005)
    # 1 kg of returning stream flies about 167 kg of vehicle.
    assert 1.0 / ledger.nozzle.impactor_fraction == pytest.approx(169.0, abs=4.0)


def test_the_cant_costs_only_a_few_points_at_high_slug_ratio() -> None:
    # The whole point of k = 30: the un-steerable bulk term stays at magnitude 1
    # while the steerable thermal term grows as sqrt(k), so the misaimed share
    # shrinks. Compare against the same burn flown straight down the axis.
    ledger = _three_synodic_ledger()
    along_axis = cycle.impulse_per_impactor_kg(
        0.0 * u.deg, ledger.nozzle.slug_ratio, ledger.nozzle.jet_energy_efficiency
    )
    axis_exhaust = (
        along_axis * ledger.nozzle.closing_speed_start / ledger.nozzle.slug_ratio
    )
    axis_delivered = float(np.exp(-ledger.nozzle.delta_v / axis_exhaust))
    assert axis_delivered - ledger.nozzle.delivered_fraction < 0.05


def test_cant_penalty_shrinks_as_the_slug_ratio_grows() -> None:
    kept = []
    for slug_ratio in (3.0, 9.0, 30.0, 100.0):
        canted = impulse_per_impactor_kg(110.6 * u.deg, slug_ratio)
        axis = impulse_per_impactor_kg(0.0 * u.deg, slug_ratio)
        kept.append(canted / axis)
    assert all(later > earlier for earlier, later in zip(kept, kept[1:]))
    assert kept[0] < 0.5 and kept[-1] > 0.85


def test_growth_is_slug_ratio_times_the_burn_and_the_solar_node() -> None:
    # One impactor kg buys k kg of spent slug, which flies k*f/(1-f) kg of
    # vehicle; the solar collision then keeps `periapsis_survival` of it.
    ledger = _three_synodic_ledger()
    delivered = ledger.nozzle.delivered_fraction
    expected = (
        ledger.periapsis_survival
        * delivered
        * ledger.nozzle.slug_ratio
        / (1.0 - delivered)
    )
    assert ledger.return_per_impactor_kg == pytest.approx(expected)
    assert ledger.doubling_years == pytest.approx(
        ledger.cycle_years * np.log(2.0) / np.log(ledger.return_per_impactor_kg)
    )


def test_three_synodic_growth_matches_the_quoted_ledger() -> None:
    ledger = _three_synodic_ledger()
    assert ledger.return_per_impactor_kg == pytest.approx(83.7, abs=1.5)
    assert ledger.doubling_years == pytest.approx(0.513, abs=0.01)


def test_paper_dive_still_doubles_faster_on_the_same_nozzle() -> None:
    # The comparison the ADR records: this cycle wins every per-pass number and
    # still loses on rate, because 3.7x the clock beats 11x the payload once the
    # logarithm is taken.
    closure = cycle.solve_synodic_closure(3, params=_PARAMS)
    assert closure is not None
    mine = _three_synodic_ledger()
    paper = cycle.paper_resonant_dive_ledger(
        closure.return_excess, closure.push_axis_deg, params=_PARAMS
    )
    assert mine.nozzle.delivered_fraction > paper.nozzle.delivered_fraction
    assert mine.return_per_impactor_kg > paper.return_per_impactor_kg
    assert mine.cycle_years > paper.cycle_years
    assert paper.growth_rate > mine.growth_rate
    assert paper.doubling_years == pytest.approx(0.305, abs=0.015)


def test_paper_dive_departure_state_is_derived_not_hardcoded() -> None:
    # Its excess and aim must come from single_impulse_resonant_dive(), so the
    # two rows sit on one set of constants.
    from src.heliocentric_reintercept import single_impulse_resonant_dive

    dive = single_impulse_resonant_dive()
    paper = cycle.paper_resonant_dive_ledger(157.42, 98.55, params=_PARAMS)
    assert paper.departure_excess == pytest.approx(
        float(dive.earth_boost.to_value(u.km / u.s))
    )
    assert paper.cycle_years == pytest.approx(
        float(dive.reintercept_time.to_value(u.year))
    )
    # Its closing speed FALLS through the burn (the vehicle accelerates away from
    # a mostly-crossing stream), where ours barely moves. A midpoint would hide it.
    assert paper.nozzle.closing_speed_end < paper.nozzle.closing_speed_start - 15.0


def test_growth_ledger_rejects_a_burn_that_delivers_nothing() -> None:
    with pytest.raises(ValueError):
        cycle.cycle_growth_ledger(
            "impossible",
            60.0,
            0.0,
            1.0,
            157.42,
            98.55,
            slug_ratio=0.05,
            params=_PARAMS,
        )


def test_slug_ratio_thirty_is_not_a_large_propellant_load() -> None:
    # k = 30 is slug per *impactor* kilogram, and impactors are well under a
    # percent of the vehicle. What the stage actually spends is ~0.2 kg of slug
    # per kg delivered. Reading k as a propellant fraction is the confusion this
    # test exists to prevent.
    ledger = _three_synodic_ledger(jet_energy_efficiency=0.70)
    assert ledger.nozzle.slug_ratio == 30.0
    assert ledger.nozzle.slug_per_delivered_kg == pytest.approx(0.196, abs=0.01)
    assert ledger.nozzle.impactor_per_delivered_kg == pytest.approx(0.0066, abs=0.0005)
    assert ledger.nozzle.impactor_per_delivered_kg < 0.01


def test_the_stage_is_an_order_of_magnitude_leaner_than_methalox() -> None:
    ledger = _three_synodic_ledger(jet_energy_efficiency=0.70)
    chemical = cycle.methalox_per_delivered_kg(ledger.nozzle.delta_v, _PARAMS)
    assert chemical == pytest.approx(2.11, abs=0.05)
    assert chemical / ledger.nozzle.slug_per_delivered_kg > 9.0


def test_specific_impulse_at_seventy_percent_efficiency() -> None:
    ledger = _three_synodic_ledger(jet_energy_efficiency=0.70)
    assert ledger.nozzle.specific_impulse == pytest.approx(2405, abs=30)
    assert ledger.nozzle.effective_exhaust_speed == pytest.approx(23.58, abs=0.3)


def test_the_cycle_barely_leans_on_nozzle_efficiency() -> None:
    # eta_geom is the largest unmeasured quantity in either repository
    # (sec:jet_efficiency). Because dv/v_e is small, the whole efficiency range
    # moves doubling time by only ~12 percent -- so it is not load-bearing here,
    # unlike in ADR 0014/0015 where it decided a device choice.
    doubling = [
        _three_synodic_ledger(jet_energy_efficiency=eta).doubling_years
        for eta in (0.4, 0.6, 0.8, 1.0)
    ]
    assert all(later < earlier for earlier, later in zip(doubling, doubling[1:]))
    assert doubling[0] / doubling[-1] < 1.15
    assert doubling[0] == pytest.approx(0.543, abs=0.01)
    assert doubling[-1] == pytest.approx(0.481, abs=0.01)


def test_specific_impulse_rises_only_as_the_square_root_of_efficiency() -> None:
    # beta goes as sqrt(eta*(1+k)) once the bulk term is small, so quadrupling
    # the energy efficiency roughly doubles Isp -- which is why a poor nozzle
    # costs so little here.
    low = _three_synodic_ledger(jet_energy_efficiency=0.25).nozzle.specific_impulse
    high = _three_synodic_ledger(jet_energy_efficiency=1.0).nozzle.specific_impulse
    assert high / low == pytest.approx(2.0, abs=0.15)
