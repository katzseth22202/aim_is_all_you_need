"""Tests for src/fly_and_park.py."""

import numpy as np
import pytest
from astropy import units as u

from src.astro_constants import PUFFSAT_CYCLE_ORBIT_PERIOD
from src.fly_and_park import (
    COAST_SYNODICS,
    EXCHANGE_BASELINE_SPEED,
    FIXED_POINT_TOLERANCE,
    METHALOX_ISP_SECONDS,
    MINIMUM_PARK,
    PERFECT_RETROGRADE_SPEED,
    SYNODIC_DAYS,
    THREE_SYNODIC_TOLERANCE,
    Cycle,
    departure_burn_span,
    doubling_ladder,
    enumerate_phase_grid,
    exchange_rate,
    exhaust_speed_from_isp,
    fixed_points,
    fly_and_park_comparison,
    growth_rate,
    hottest_reachable,
    payload_mass_ratio_float,
    perfect_retrograde_premium,
    phase_reaches,
    sustainable_chain_optimum,
    sweet_phase,
    synodic_lock,
    usable_phase_fraction,
    usable_phase_windows,
)
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.propulsion import payload_mass_ratio

_PUSH = float(puffsat_cycle_periapsis_speed().to_value(u.km / u.s))


@pytest.fixture(scope="module")
def grid():
    """The phase sweep, enumerated once for the whole module (~60 s)."""
    return enumerate_phase_grid()


def test_mass_ratio_float_matches_the_quantity_helper() -> None:
    # The float inlining must agree with propulsion.payload_mass_ratio, which is
    # the paper's eq:PuffSat_ratio.
    for v_b in (52.0, 60.0, 68.0):
        got = payload_mass_ratio_float(v_b, _PUSH)
        want = float(payload_mass_ratio(v_rf=_PUSH * u.km / u.s, v_b=v_b * u.km / u.s))
        assert got == pytest.approx(want, rel=1e-12)


def test_mass_ratio_is_zero_below_the_push_target() -> None:
    # A collision that cannot reach v_rf mints no payload; it must not return a
    # negative or a divide-by-zero.
    assert payload_mass_ratio_float(_PUSH - 1.0, _PUSH) == 0.0
    assert payload_mass_ratio_float(_PUSH, _PUSH) == 0.0


def test_exchange_rate_is_the_burn_that_exactly_cancels_the_gain() -> None:
    # The definition: spend exactly this much more burn and the hotter arrival
    # is worth precisely nothing.
    exhaust = exhaust_speed_from_isp(1200.0)
    budget = exchange_rate(68.0, exhaust, _PUSH)
    base = payload_mass_ratio_float(EXCHANGE_BASELINE_SPEED, _PUSH)
    hot = payload_mass_ratio_float(68.0, _PUSH) * float(np.exp(-budget / exhaust))
    assert hot == pytest.approx(base, rel=1e-9)


def test_exchange_rate_scales_with_exhaust_speed() -> None:
    # The budget is v_e * ln(gain), so doubling the exhaust doubles the budget.
    a = exchange_rate(68.0, exhaust_speed_from_isp(600.0), _PUSH)
    b = exchange_rate(68.0, exhaust_speed_from_isp(1200.0), _PUSH)
    assert b == pytest.approx(2.0 * a, rel=1e-9)


def test_exchange_rate_is_zero_when_the_arrival_is_no_hotter() -> None:
    exhaust = exhaust_speed_from_isp(1200.0)
    assert exchange_rate(EXCHANGE_BASELINE_SPEED, exhaust, _PUSH) == 0.0
    assert exchange_rate(EXCHANGE_BASELINE_SPEED - 5.0, exhaust, _PUSH) == 0.0


def test_methalox_budget_is_smaller_than_the_manoeuvre_costs() -> None:
    # ADR 0030's load-bearing sentence: methalox does not fail by a lot, it
    # fails by about 5%. The measured cost is ~1.04 km/s.
    budget = exchange_rate(68.0, exhaust_speed_from_isp(METHALOX_ISP_SECONDS), _PUSH)
    assert 0.9 < budget < 1.1
    assert budget < 1.036  # the measured median extra departure burn


def test_impactor_isp_clears_the_same_cost_with_room() -> None:
    for isp, floor in ((1200.0, 3.0), (2214.0, 5.5)):
        assert exchange_rate(68.0, exhaust_speed_from_isp(isp), _PUSH) > floor


def test_cycle_flight_time_excludes_coast_and_park() -> None:
    c = Cycle(
        departure_phase=0.5,
        outbound_years=1.0,
        return_years=2.0,
        departure_burn=5.0,
        flyby_burn=0.0,
        collision_speed=60.0,
    )
    assert c.flight_years == pytest.approx(3.0)
    assert c.growth(exhaust_speed_from_isp(1200.0), _PUSH) > 1.0


@pytest.mark.slow
def test_the_sweet_phase_is_where_reaching_jupiter_is_cheapest(grid) -> None:
    # The term the paper needs defined: the departure phase whose cheapest
    # outbound burn is smallest. Nothing else in the module asserts where it is.
    phase, burn = sweet_phase(grid)
    assert 0.0 <= phase < 1.0
    assert burn == pytest.approx(4.409, abs=0.05)
    # It really is the minimum over the grid.
    for cycles in grid.values():
        if cycles:
            assert min(c.departure_burn for c in cycles) >= burn - 1e-9


@pytest.mark.slow
def test_reaching_jupiter_swings_by_almost_an_order_of_magnitude(grid) -> None:
    # This swing is the whole mechanism: it is why a departure stage that cannot
    # afford the dear phases gets pinned to the sweet one, and hence to a 3S clock.
    low, high = departure_burn_span(grid)
    assert low == pytest.approx(4.409, abs=0.05)
    assert high == pytest.approx(38.485, abs=0.5)
    assert high / low > 8.0


@pytest.mark.slow
def test_usable_phase_fraction_rises_with_exhaust_speed(grid) -> None:
    # The launch-cadence result. Monotone in Isp, 18% at methalox, 100% by 1900 s.
    fractions = [
        usable_phase_fraction(grid, exhaust_speed_from_isp(isp), _PUSH)
        for isp in (380, 700, 1000, 1200, 1500, 1800, 1900, 2214)
    ]
    assert fractions == sorted(fractions)
    assert fractions[0] == pytest.approx(0.18, abs=0.02)
    assert fractions[-2] == pytest.approx(1.0)
    assert fractions[-1] == pytest.approx(1.0)


@pytest.mark.slow
def test_fly_and_park_loses_on_methalox_and_wins_on_impactor_isp(grid) -> None:
    # The architecture verdict. Counting phases where padding a shorter, hotter
    # cycle out to 3.00 S beats flying the whole window.
    wins = {}
    for isp in (METHALOX_ISP_SECONDS, 1200.0, 2214.0):
        comps = fly_and_park_comparison(grid, exhaust_speed_from_isp(isp), _PUSH)
        wins[isp] = (sum(1 for c in comps if c.gain > 1.0), len(comps))
    assert wins[METHALOX_ISP_SECONDS][0] <= 2, wins  # 1 of 11
    assert wins[1200.0][0] >= 12, wins  # 16 of 30
    assert wins[2214.0][0] >= 20, wins  # 25 of 30
    assert wins[2214.0][0] > wins[1200.0][0] > wins[METHALOX_ISP_SECONDS][0]


@pytest.mark.slow
def test_the_gain_is_smallest_at_the_sweet_phase_and_largest_away_from_it(grid) -> None:
    # It is a robustness mechanism, not a growth win: ADR 0030 insists the 27%
    # is never quoted without the 5%.
    comps = fly_and_park_comparison(grid, exhaust_speed_from_isp(2214.0), _PUSH)
    sweet, _ = sweet_phase(grid)
    near = [c.gain for c in comps if abs(c.phase - sweet) < 0.03]
    far = [c.gain for c in comps if abs(c.phase - sweet) > 0.10]
    assert near and far
    assert max(near) < 1.10
    assert max(far) > 1.20


@pytest.mark.slow
def test_a_hot_arrival_is_available_at_every_phase_that_works(grid) -> None:
    # Availability is never the constraint -- profitability is. Every phase with
    # a viable pure-3S cycle also offers a parkable cycle reaching 60 km/s.
    comps = fly_and_park_comparison(grid, exhaust_speed_from_isp(2214.0), _PUSH)
    assert comps
    assert all(phase_reaches(grid, c.phase, 60.0) for c in comps)


@pytest.mark.slow
def test_the_hottest_parkable_arrival_reaches_the_perfect_retrograde_boundary(
    grid,
) -> None:
    # CONTEXT.md, "Perfect-retrograde boundary": the minimum-energy purely
    # tangential retrograde arrival is 69.24 km/s, rising to 72.74 at solar
    # escape. A parkable cycle gets there.
    hottest = hottest_reachable(grid)
    assert 68.0 < hottest < 72.8


@pytest.mark.slow
def test_parking_never_makes_the_cycle_longer_than_three_synodics(grid) -> None:
    # The clock is the invariant: flight + park is exactly 3.00 S, so the next
    # departure lands on the same phase and nothing drifts.
    comps = fly_and_park_comparison(grid, exhaust_speed_from_isp(2214.0), _PUSH)
    for comp in comps:
        if comp.parked is None:
            continue
        assert comp.parked.flight_synodics <= 3.0 - MINIMUM_PARK
        assert comp.park_synodics >= MINIMUM_PARK
        assert comp.parked.flight_synodics + comp.park_synodics == pytest.approx(3.0)


@pytest.mark.slow
def test_the_perfect_retrograde_premium_is_what_adr_0030_quotes(grid) -> None:
    # ADR 0030 and the paper document both quote "+0.000 to +1.978 km/s, median
    # +1.036" as the cost of buying a perfect-retrograde arrival. That number was
    # a scratch figure until this harness landed; pin it so it stays reproducible.
    premiums = perfect_retrograde_premium(grid, exhaust_speed_from_isp(2214.0), _PUSH)
    assert premiums
    extras = [p.extra_burn for p in premiums]
    assert min(extras) == pytest.approx(0.0, abs=1e-6)
    assert max(extras) == pytest.approx(1.978, abs=0.02)
    assert float(np.median(extras)) == pytest.approx(1.036, abs=0.02)


@pytest.mark.slow
def test_the_premium_fails_the_methalox_budget_and_clears_the_impactor_one(
    grid,
) -> None:
    # The architecture verdict in one assertion: methalox cannot afford the
    # perfect-retrograde arrival and an impactor-driven departure can. Methalox
    # misses narrowly -- 0.99 against 1.036 -- which is the point.
    premiums = perfect_retrograde_premium(grid, exhaust_speed_from_isp(2214.0), _PUSH)
    cost = float(np.median([p.extra_burn for p in premiums]))
    methalox = exchange_rate(68.0, exhaust_speed_from_isp(METHALOX_ISP_SECONDS), _PUSH)
    assert methalox < cost
    assert cost - methalox < 0.10  # it fails by a little, not by a lot
    for isp in (1200.0, 2214.0):
        assert exchange_rate(68.0, exhaust_speed_from_isp(isp), _PUSH) > cost


@pytest.mark.slow
def test_every_premium_row_actually_buys_a_hotter_arrival(grid) -> None:
    # The premium is selected by arrival speed, so every row must clear the
    # threshold and none may be cheaper than the cycle it is measured against.
    premiums = perfect_retrograde_premium(grid, exhaust_speed_from_isp(2214.0), _PUSH)
    for item in premiums:
        assert item.hot.collision_speed >= PERFECT_RETROGRADE_SPEED
        assert item.hot.collision_speed >= item.pure.collision_speed
        assert item.extra_burn >= -1e-9
        assert item.hot.flight_synodics <= 3.0 - MINIMUM_PARK


@pytest.mark.slow
def test_a_two_synodic_fixed_point_exists(grid) -> None:
    # ADR 0011's resonance -- the paper's own baseline -- survives this model's
    # Earth-intercept constraint. ADR 0030 originally recorded its existence as
    # unresolved; it is not.
    twos = fixed_points(grid, multiple=2)
    assert twos, "no two-synodic fixed point found"
    best = twos[0]
    assert abs(best.drift) <= FIXED_POINT_TOLERANCE
    assert best.synodics == pytest.approx(2.0, abs=1e-3)
    # It really does repeat: the drift is small enough to fly for decades.
    assert best.repeats_before_drifting >= 100
    # Its identity, so a change to the grid is noticed rather than absorbed.
    assert best.cycle.collision_speed == pytest.approx(63.35, abs=0.2)
    assert best.cycle.departure_burn == pytest.approx(8.613, abs=0.05)


@pytest.mark.slow
def test_the_two_synodic_point_shrinks_on_methalox_and_wins_above_it(grid) -> None:
    # The resolution of ADR 0030's open question. The chain never selects the 2S
    # point not because the search fails but because on methalox it *shrinks* --
    # declining it is correct. Give the departure a real exhaust speed and it
    # beats the 3S cycle the chain actually flies.
    best2 = fixed_points(grid, multiple=2)[0].cycle
    three = fixed_points(grid, multiple=3, tolerance=THREE_SYNODIC_TOLERANCE)
    assert three

    methalox = exhaust_speed_from_isp(METHALOX_ISP_SECONDS)
    assert best2.growth(methalox, _PUSH) < 1.0
    assert growth_rate(best2, methalox, _PUSH) == float("-inf")
    # ...while 3S grows on the same propellant.
    assert max(growth_rate(f.cycle, methalox, _PUSH) for f in three) > 0.1

    for isp in (1200.0, 2214.0):
        exhaust = exhaust_speed_from_isp(isp)
        rate2 = growth_rate(best2, exhaust, _PUSH)
        rate3 = max(growth_rate(f.cycle, exhaust, _PUSH) for f in three)
        assert rate2 > rate3, (isp, rate2, rate3)
        assert rate2 / rate3 > 1.25  # 2S wins by 28% at 1200 and 40% at 2214


@pytest.mark.slow
def test_growth_rate_reports_minus_infinity_for_a_shrinking_cycle(grid) -> None:
    # A shrinking cycle is a negative gradient, not an error (CONTEXT.md,
    # "Growth rate"): it must be comparable, not raise.
    shrinking = Cycle(
        departure_phase=0.0,
        outbound_years=1.0,
        return_years=2.0,
        departure_burn=40.0,
        flyby_burn=0.0,
        collision_speed=55.0,
    )
    assert shrinking.growth(exhaust_speed_from_isp(380.0), _PUSH) < 1.0
    assert growth_rate(shrinking, exhaust_speed_from_isp(380.0), _PUSH) == float("-inf")


def test_the_minimum_park_is_the_mandatory_coast() -> None:
    # ADR 0031. Parking lengthens the bound near-escape orbit the payload
    # already coasts through between the push and the departure burn; it cannot
    # shorten it. So the floor under any park is that coast and nothing looser.
    assert MINIMUM_PARK == COAST_SYNODICS
    assert MINIMUM_PARK * SYNODIC_DAYS == pytest.approx(
        float(PUFFSAT_CYCLE_ORBIT_PERIOD.to_value(u.day)), rel=1e-9
    )
    assert MINIMUM_PARK > 0.02  # the value it replaced, which was 2.5x too small


def test_usable_phase_windows_measures_arcs_cyclically() -> None:
    # A window may wrap through phase 0, where phase 1.0 and phase 0.0 are the
    # same geometry. Counting the wrap as two windows would understate the
    # widest one, which is the number the launch-cadence sentence quotes.
    hot = Cycle(0.0, 1.0, 2.0, 5.0, 0.0, 60.0)
    cold = Cycle(0.0, 1.0, 2.0, 60.0, 0.0, 60.0)
    exhaust = exhaust_speed_from_isp(1200.0)
    assert hot.growth(exhaust, _PUSH) > 1.0 and cold.growth(exhaust, _PUSH) < 1.0
    wrapping = {0.0: [hot], 0.25: [cold], 0.5: [cold], 0.75: [hot]}
    windows = usable_phase_windows(wrapping, exhaust, _PUSH)
    assert len(windows) == 1
    assert windows[0].phases == 2
    assert windows[0].start_phase == pytest.approx(0.75)
    # And the degenerate ends behave.
    assert usable_phase_windows({p: [cold] for p in (0.0, 0.5)}, exhaust, _PUSH) == []
    full = usable_phase_windows({p: [hot] for p in (0.0, 0.5)}, exhaust, _PUSH)
    assert len(full) == 1 and full[0].days == pytest.approx(SYNODIC_DAYS)


@pytest.mark.slow
def test_the_usable_phases_form_one_window_at_every_exhaust_speed(grid) -> None:
    # S7 for the paper: the fraction says how many phases work, not whether they
    # sit together. They do -- one contiguous arc widening about the sweet phase
    # at every exhaust speed tested, so the claim is "one window, 3.4x wider at
    # Isp 1200", not "a set of windows".
    widths = []
    for isp in (380, 700, 1000, 1200, 1500, 1800, 1900, 2214):
        windows = usable_phase_windows(grid, exhaust_speed_from_isp(isp), _PUSH)
        assert len(windows) == 1, (isp, [w.phases for w in windows])
        widths.append(windows[0].days)
    assert widths == sorted(widths)
    assert widths[0] == pytest.approx(71.0, abs=6.0)  # methalox
    assert widths[3] == pytest.approx(240.5, abs=6.0)  # Isp 1200
    assert widths[-1] == pytest.approx(SYNODIC_DAYS, abs=1e-6)  # every phase
    assert widths[3] / widths[0] == pytest.approx(3.4, abs=0.2)


@pytest.mark.slow
def test_the_windows_account_for_every_usable_phase(grid) -> None:
    # The layout must not lose or double-count a phase against the fraction the
    # paper already publishes.
    for isp in (380, 1200, 2214):
        exhaust = exhaust_speed_from_isp(isp)
        windows = usable_phase_windows(grid, exhaust, _PUSH)
        counted = sum(w.phases for w in windows)
        assert counted / len(grid) == pytest.approx(
            usable_phase_fraction(grid, exhaust, _PUSH)
        )


@pytest.mark.slow
def test_the_nine_day_two_synodic_lock_cannot_be_flown(grid) -> None:
    # ADR 0031, and the reason the 0.84 yr doubling does not stand. Padding a
    # 1.978 S flight to 2.00 S leaves a 8.7-day park -- shorter than the 20-day
    # coast every cycle already owes, so it is arithmetic rather than a
    # trajectory.
    exhaust = exhaust_speed_from_isp(2214.0)
    loose = synodic_lock(grid, 2.0, exhaust, _PUSH, minimum_park=0.02)
    assert loose is not None
    assert loose.park_days < float(PUFFSAT_CYCLE_ORBIT_PERIOD.to_value(u.day))
    assert loose.doubling_years == pytest.approx(0.843, abs=0.01)


@pytest.mark.slow
def test_the_admissible_two_synodic_lock_is_the_resonance_already_reported(
    grid,
) -> None:
    # The lock the paper may quote. Charge the coast and the 2S lock collapses
    # onto ADR 0011's two-synodic resonance -- same phase, same burn, same
    # arrival -- so it is not a new cycle at all. Its park is the coast plus
    # about an hour, which is why fly-and-park adds nothing to it.
    exhaust = exhaust_speed_from_isp(2214.0)
    lock = synodic_lock(grid, 2.0, exhaust, _PUSH)
    assert lock is not None
    assert lock.park_days >= float(PUFFSAT_CYCLE_ORBIT_PERIOD.to_value(u.day))
    assert lock.park_days < 21.0
    assert lock.cycle.departure_phase == pytest.approx(0.8082, abs=1e-3)
    assert lock.cycle.departure_burn == pytest.approx(8.613, abs=0.05)
    assert lock.cycle.collision_speed == pytest.approx(63.35, abs=0.2)
    assert lock.doubling_years == pytest.approx(0.873, abs=0.01)
    # It is the same trajectory fixed_points() finds, not merely a similar one.
    resonance = fixed_points(grid, multiple=2)[0].cycle
    assert lock.cycle == resonance


@pytest.mark.slow
def test_charging_the_coast_costs_the_two_synodic_lock_a_few_percent(grid) -> None:
    # How much the correction is worth, so the paper can say it: the admissible
    # lock is slower than the inadmissible one, but only just.
    exhaust = exhaust_speed_from_isp(2214.0)
    loose = synodic_lock(grid, 2.0, exhaust, _PUSH, minimum_park=0.02)
    tight = synodic_lock(grid, 2.0, exhaust, _PUSH)
    assert loose is not None and tight is not None
    penalty = tight.doubling_years / loose.doubling_years - 1.0
    assert 0.0 < penalty < 0.06
    assert tight.phases_offering_one < loose.phases_offering_one


@pytest.mark.slow
def test_every_lock_pads_to_exactly_its_target(grid) -> None:
    # The invariant the whole construction rests on: flight plus park is the
    # target to machine precision, so the next departure lands on the phase this
    # one left and nothing drifts.
    for isp in (380.0, 1200.0, 2214.0):
        for target in (2.0, 3.0):
            lock = synodic_lock(grid, target, exhaust_speed_from_isp(isp), _PUSH)
            if lock is None:
                continue
            assert lock.cycle.flight_synodics + lock.park_synodics == pytest.approx(
                target
            )
            assert lock.park_synodics >= MINIMUM_PARK
            assert lock.growth > 1.0


@pytest.mark.slow
def test_methalox_admits_no_two_synodic_lock_at_all(grid) -> None:
    # Not "worse" -- absent. Every 2S-capable cycle either shrinks on methalox or
    # cannot leave the coast, which is the same verdict ADR 0030 reached for the
    # resonance and the reason the chain never sustained one.
    assert (
        synodic_lock(grid, 2.0, exhaust_speed_from_isp(METHALOX_ISP_SECONDS), _PUSH)
        is None
    )
    assert synodic_lock(grid, 3.0, exhaust_speed_from_isp(METHALOX_ISP_SECONDS), _PUSH)


@pytest.mark.slow
def test_the_chain_optimum_is_the_lock_itself(grid) -> None:
    # S5's chain check, and ADR 0030's N4.2 caveat discharged. A chain free to
    # park any length settles on a single cycle returning to its own phase --
    # the lock -- so the single-cycle optimum does survive the lookahead here.
    for isp in (380.0, 1200.0, 2214.0):
        exhaust = exhaust_speed_from_isp(isp)
        optimum = sustainable_chain_optimum(grid, exhaust, _PUSH)
        assert optimum is not None, isp
        assert optimum.is_synodic_lock, (isp, len(optimum.policy))
        target = round(optimum.total_synodics)
        assert optimum.total_synodics == pytest.approx(float(target), abs=1e-6)
        lock = synodic_lock(grid, float(target), exhaust, _PUSH)
        assert lock is not None
        assert optimum.doubling_years == pytest.approx(lock.doubling_years, rel=1e-4)
        assert optimum.policy[0].cycle == lock.cycle


@pytest.mark.slow
def test_the_chain_flies_three_synodics_on_methalox_and_two_above_it(grid) -> None:
    # The cadence verdict, chain-checked rather than argued. Methalox cannot pay
    # the 2S lock's 8.6 km/s departure burn, so the sustainable optimum is 3S;
    # give the departure a nozzle and the optimum moves to 2S and roughly
    # halves the clock.
    clocks = {}
    for isp in (380.0, 1200.0, 2214.0):
        optimum = sustainable_chain_optimum(grid, exhaust_speed_from_isp(isp), _PUSH)
        assert optimum is not None
        clocks[isp] = (round(optimum.total_synodics), optimum.doubling_years)
    assert clocks[380.0][0] == 3
    assert clocks[1200.0][0] == 2
    assert clocks[2214.0][0] == 2
    assert clocks[380.0][1] == pytest.approx(3.641, abs=0.02)
    assert clocks[1200.0][1] == pytest.approx(1.082, abs=0.01)
    assert clocks[2214.0][1] == pytest.approx(0.873, abs=0.01)


@pytest.mark.slow
def test_the_doubling_ladder_carries_its_scorer_on_every_rung(grid) -> None:
    # S6. Four doubling times already live in sec:jupiter_only_growth from three
    # different devices, so a rung without its provenance attached is a rung
    # that will be read as disagreeing with one it does not measure against.
    rungs = doubling_ladder(grid, _PUSH)
    assert rungs
    assert [r.doubling_years for r in rungs] == sorted(r.doubling_years for r in rungs)
    for rung in rungs:
        assert rung.scorer and rung.model and rung.scope and rung.efficiency
        assert "no nozzle impulse recovery" in rung.efficiency
        assert rung.rate > 0.0
        assert rung.doubling_years == pytest.approx(np.log(2.0) / rung.rate)
    fastest = rungs[0]
    assert "2.00 S" in fastest.label and "2214" in fastest.label
    assert fastest.doubling_years == pytest.approx(0.873, abs=0.01)
