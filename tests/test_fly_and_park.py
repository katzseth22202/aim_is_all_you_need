"""Tests for src/fly_and_park.py."""

import numpy as np
import pytest
from astropy import units as u

from src.fly_and_park import (
    EXCHANGE_BASELINE_SPEED,
    METHALOX_ISP_SECONDS,
    MINIMUM_PARK,
    PERFECT_RETROGRADE_SPEED,
    Cycle,
    departure_burn_span,
    enumerate_phase_grid,
    exchange_rate,
    exhaust_speed_from_isp,
    fly_and_park_comparison,
    hottest_reachable,
    payload_mass_ratio_float,
    perfect_retrograde_premium,
    phase_reaches,
    sweet_phase,
    usable_phase_fraction,
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
    assert wins[1200.0][0] >= 12, wins  # 17 of 30
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
