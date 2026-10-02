"""Tests for src/two_wave_growth.py.

The real-orbit chain (Astropy ephemeris + Lambert arcs + per-cycle split
solves) is exercised in ``slow``-marked tests; the pricing algebra is pinned
fast with explicit inputs.
"""

import math
from dataclasses import replace

import astropy.units as u
import numpy as np
import pytest

from src.nozzle_analysis import same_cycle_nozzle
from src.plume_thermal import NOZZLE_FLOOR_TEMPERATURE, NOZZLE_GATE_TEMPERATURE
from src.two_wave_growth import (
    VE_METHALOX,
    TwoWaveCycle,
    adaptive_two_wave_cycles,
    analyze_two_wave_growth,
    departure_burn_after,
    fleet_ignition_windows,
    headon_slug_ratio_bounds,
    link_onward_burns,
    price_chain,
    price_cycle,
)

HORIZON_YEARS = 30.0

# A representative flown 2S cycle, pinned as explicit inputs so the pricing
# tests stay fast (no ephemeris, no Lambert arcs).
CYCLE = TwoWaveCycle(
    index=0,
    departure_jd=2461354.0,
    return_jd=2462151.7,
    synodic_multiple=2,
    period_years=2.1841,
    departure_burn=7.1706,
    nozzle_wave_v_b=61.3778,
    nozzle_wave_dsm=0.0014,
    split_days=20.0,
    growth_wave_arrival_jd=2462131.7,
    growth_wave_v_b=62.5,
    growth_wave_burn=0.5,
    onward_burn=7.0580,
)


@pytest.mark.slow
def test_chain_rides_the_audited_adaptive_policy():
    """The flown cadence is ADR 0011's, not a new one invented here."""
    cycles = adaptive_two_wave_cycles(years=HORIZON_YEARS)

    assert len(cycles) == 11
    assert sum(1 for c in cycles if c.synodic_multiple == 2) == 7
    assert sum(1 for c in cycles if c.synodic_multiple == 3) == 4
    assert max(c.nozzle_wave_dsm for c in cycles) == pytest.approx(
        18.478978 / 1000.0, abs=1e-6
    )


def test_each_cycles_waves_push_a_payload_onto_the_next_departure() -> None:
    # A 3S return feeding a 2S departure: the payload flies the 2S burn.
    three = replace(CYCLE, index=0, synodic_multiple=3, departure_burn=5.329)
    two = replace(CYCLE, index=1, departure_burn=7.1706)
    linked = link_onward_burns([three, two], after_last=6.9)
    assert [c.onward_burn for c in linked] == [7.1706, 6.9]
    assert [c.departure_burn for c in linked] == [5.329, 7.1706]


@pytest.mark.slow
def test_the_flown_chains_onward_burns_are_its_next_departures() -> None:
    """The last cycle's payload flies the window after the horizon."""
    cycles = adaptive_two_wave_cycles(years=HORIZON_YEARS)
    for cycle, following in zip(cycles, cycles[1:]):
        assert cycle.onward_burn == following.departure_burn
    last = cycles[-1]
    assert last.onward_burn == pytest.approx(departure_burn_after(last.return_jd))
    # Cycle 0 is a 3S return feeding the 2S cycle 1: 5.33 out, 7.17 onward.
    assert cycles[0].departure_burn == pytest.approx(5.329, abs=1e-3)
    assert cycles[0].onward_burn == pytest.approx(7.171, abs=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("split_days", [10.0, 20.0])
def test_growth_wave_arrives_exactly_one_split_gap_early(split_days):
    """Both waves leave together; the growth wave lands ``split_days`` sooner."""
    cycles = adaptive_two_wave_cycles(years=HORIZON_YEARS, split_days=split_days)

    for cycle in cycles:
        assert cycle.growth_wave_arrival_jd == pytest.approx(
            cycle.return_jd - split_days, abs=1e-6
        )
        assert cycle.growth_wave_burn > 0.0
        assert cycle.growth_wave_v_b > cycle.nozzle_wave_v_b


def test_pricing_composes_with_the_committed_nozzle_ledger():
    """At f = 0.8 and a 20 d parking orbit, pricing is same_cycle_nozzle's default,
    with the payload departing on the next window's burn (ADR 0038)."""
    priced = price_cycle(CYCLE, recovery=0.6, fudge=0.8, slug_ratio=7.0)

    expected = same_cycle_nozzle(
        growth_collision_speed=CYCLE.growth_wave_v_b,
        growth_wave_burn=CYCLE.growth_wave_burn + CYCLE.nozzle_wave_dsm,
        nozzle_collision_speed=CYCLE.nozzle_wave_v_b,
        departure_dv=CYCLE.onward_burn,
        cycle=CYCLE.period_years,
        exhaust_speed=VE_METHALOX,
        recovery=0.6,
        slug_ratio=7.0,
    )
    assert priced.growth == pytest.approx(expected.growth, rel=1e-12)
    assert priced.sigma == pytest.approx(expected.sigma, rel=1e-12)


def test_a_shorter_parking_orbit_costs_more_reversal():
    """The split gap sizes the apoapsis reversal, so a tighter orbit prices worse."""
    tight = price_cycle(
        CYCLE, recovery=0.6, fudge=0.8, slug_ratio=7.0, reversal_period=10 * u.day
    )
    loose = price_cycle(
        CYCLE, recovery=0.6, fudge=0.8, slug_ratio=7.0, reversal_period=60 * u.day
    )
    assert tight.growth < loose.growth


def test_growth_rises_with_recovery_and_with_the_fudge_factor():
    """Better collimation and a more elastic push both mint more payload."""
    by_recovery = [
        price_cycle(CYCLE, recovery=e, fudge=0.8, slug_ratio=7.0).growth
        for e in (0.25, 0.5, 0.75, 0.9)
    ]
    by_fudge = [
        price_cycle(CYCLE, recovery=0.6, fudge=f, slug_ratio=7.0).growth
        for f in (0.5, 0.6, 0.7, 0.8)
    ]
    assert by_recovery == sorted(by_recovery)
    assert by_fudge == sorted(by_fudge)


def test_charging_the_deep_space_maneuver_can_only_lower_growth():
    """The policy's DSM proxy is real methalox, not a free correction."""
    charged = price_cycle(CYCLE, recovery=0.6, fudge=0.8, slug_ratio=7.0)
    uncharged = price_cycle(
        replace(CYCLE, nozzle_wave_dsm=0.0), recovery=0.6, fudge=0.8, slug_ratio=7.0
    )
    assert charged.growth < uncharged.growth


@pytest.mark.slow
def test_chain_growth_compounds_its_cycles_over_the_flown_span():
    """A chain is worth the product of its cycles, scored per flown year."""
    cycles = adaptive_two_wave_cycles(years=HORIZON_YEARS)
    chain = price_chain(cycles, recovery=0.6, fudge=0.8, slug_ratio=7.0)

    expected = math.prod(
        price_cycle(c, recovery=0.6, fudge=0.8, slug_ratio=7.0).growth for c in cycles
    )
    assert chain.total_growth == pytest.approx(expected, rel=1e-12)
    assert chain.horizon_years == pytest.approx(
        sum(c.period_years for c in cycles), rel=1e-9
    )
    assert chain.rate == pytest.approx(
        math.log(expected) / chain.horizon_years, rel=1e-12
    )
    assert chain.two_synodic_cycles == 7
    assert chain.three_synodic_cycles == 4


@pytest.mark.slow
def test_sweep_covers_the_grid_and_finds_interior_slug_ratios():
    """One row per (e, f), each with an optimum k off the search-box walls."""
    analysis = analyze_two_wave_growth(
        years=HORIZON_YEARS, recoveries=(0.25, 0.6, 0.9), fudges=(0.5, 0.8)
    )

    assert len(analysis.sweep) == 6
    assert set(analysis.sweep["recovery"]) == {0.25, 0.6, 0.9}
    assert set(analysis.sweep["fudge"]) == {0.5, 0.8}
    assert analysis.sweep["slug_ratio"].between(0.3, 79.0).all()
    assert len(analysis.cycles) == 11


def test_chain_reports_a_continuous_annual_rate_and_a_horizon_projection():
    """The rate is an exponent; the projection is that exponent run out."""
    chain = price_chain([CYCLE], recovery=0.6, fudge=0.8, slug_ratio=7.0)

    assert chain.annual_increase == pytest.approx(math.exp(chain.rate) - 1.0)
    assert chain.mass_after(30.0) == pytest.approx(math.exp(30.0 * chain.rate))
    # The projection is the continuous idealization of a lumpy chain: over the
    # span actually flown it must agree with the compounded cycles.
    assert chain.mass_after(chain.horizon_years) == pytest.approx(
        chain.total_growth, rel=1e-12
    )


def test_search_box_is_intersected_with_the_ignition_window() -> None:
    """The bare ceiling of 80 admitted slug ratios no plume can supply.

    ``k_max`` is 36.88 even at the 75 km/s head-on anchor and 26.9 across the
    flown fleet, so the old box ran six times past where the nozzle has a
    plasma to grip.  It never changed an answer -- the optimum sits near 8.5 --
    but a recorded search box that is not the admissible set is the ADR 0007
    failure mode.
    """
    cycles = [CYCLE]
    low, high = headon_slug_ratio_bounds(cycles)
    window = fleet_ignition_windows(cycles, NOZZLE_GATE_TEMPERATURE)[1]
    assert window is not None
    assert high == pytest.approx(min(window[1], 80.0))
    assert high < 80.0
    assert low == pytest.approx(max(window[0], 0.2))


def test_the_gate_admits_more_slug_than_the_design_floor() -> None:
    """10 000 K is the physics gate; 15 000 K is the design intent."""
    cycles = [CYCLE]
    gated = headon_slug_ratio_bounds(cycles, NOZZLE_GATE_TEMPERATURE)
    floored = headon_slug_ratio_bounds(cycles, NOZZLE_FLOOR_TEMPERATURE)
    assert gated[1] > floored[1]


def test_charging_the_toll_costs_growth_and_lowers_the_optimal_slug_ratio() -> None:
    """Both directions matter: less growth, and less slug is worth carrying.

    ``eta_chem`` falls with ``k`` -- more slug means more water to pull apart
    per unit of collision energy -- so the toll pulls the optimum down as well
    as pushing the growth down.
    """
    cycles = [CYCLE]
    plain = price_chain(cycles, 1.0, 0.8, geometric_efficiency=None)
    tolled = price_chain(cycles, 1.0, 0.8, geometric_efficiency=1.0)
    assert tolled.total_growth < plain.total_growth
    assert tolled.slug_ratio < plain.slug_ratio


def test_double_charging_the_toll_silently_shrinks_the_chain() -> None:
    """D3's trap: ``recovery`` derates from outside the debit, the toll inside.

    Passing the same number to both is not an error and returns no warning --
    it just returns a much smaller growth.  Pinned here so the two paths stay
    distinguishable, and so nobody re-derives the C1 plate column by charging
    ``eta_geom`` twice.
    """
    cycles = [CYCLE]
    once = price_chain(cycles, 1.0, 0.818, geometric_efficiency=0.8)
    twice = price_chain(cycles, 0.8, 0.818, geometric_efficiency=0.8)
    # One cycle loses ~19%, which is small enough to look like a rounding
    # difference; the flown chain compounds it over eleven cycles into a factor
    # of nine (6.289e4 against 7282 at the target's own f = 0.80).
    assert twice.total_growth < 0.85 * once.total_growth
    assert (twice.total_growth / once.total_growth) ** 11 < 0.2


@pytest.mark.slow
def test_the_plate_column_reproduces_at_the_measured_elasticity() -> None:
    """D3, resolved: the paper's plate column is quoted at ``f`` = 0.818.

    ``0.818`` is the plate's full measured elasticity; the target's default is
    ``0.800``, and the difference is the whole gap that made the column look
    unreproducible.  Both are defensible -- they answer different questions --
    so what is pinned is which one the published figures came from.

    ADR 0038 moved the column: each payload now departs on the next window's
    burn.  The parent printed 1.464e6, 4.244e5 and 7.486e4 before the fix.
    """
    cycles = adaptive_two_wave_cycles()
    published = {1.0: 1.199e6, 0.9: 3.364e5, 0.8: 5.662e4}
    for eta_geom, growth in published.items():
        chain = price_chain(cycles, 1.0, 0.818, geometric_efficiency=eta_geom)
        assert np.isclose(chain.total_growth, growth, rtol=1e-3)


def test_a_flown_cycle_reports_its_length_in_synodic_periods() -> None:
    """The cadence's own clock, read off the fixed 2S period it is built on."""
    assert CYCLE.period_synodics == pytest.approx(2.0, abs=1e-3)
    assert CYCLE.phase_drift == pytest.approx(0.0, abs=1e-3)
    stretched = replace(CYCLE, period_years=CYCLE.period_years * 1.05)
    assert stretched.period_synodics == pytest.approx(2.1, abs=1e-3)
    assert stretched.phase_drift == pytest.approx(0.1, abs=1e-3)


@pytest.mark.slow
def test_every_flown_cycle_is_already_an_exact_synodic_lock() -> None:
    """S8 for the paper: the 2S cycles it flies need no padding to lock.

    The **synodic lock** ``src/fly_and_park.py`` constructs -- flight plus park
    summing to a whole number of synodic periods, so the cycle returns to its
    own departure phase -- is not a new cycle here.  The adaptive cadence builds
    every return on an exact synodic multiple, so each flown cycle is already a
    fixed point and fly-and-park has nothing to add to it.  What the cadence
    does *not* guarantee is that the next window is flyable, which is why the
    chain still takes four 3S fallbacks (ADR 0011, ADR 0031).
    """
    cycles = adaptive_two_wave_cycles()
    assert cycles
    for cycle in cycles:
        assert cycle.period_synodics == pytest.approx(
            float(cycle.synodic_multiple), abs=1e-4
        ), cycle.index
        assert abs(cycle.phase_drift) < 1e-4
    twos = [c for c in cycles if c.synodic_multiple == 2]
    threes = [c for c in cycles if c.synodic_multiple == 3]
    assert len(twos) == 7 and len(threes) == 4


@pytest.mark.slow
def test_the_flown_two_synodic_cycles_bracket_the_circular_lock() -> None:
    """The circular 2S lock sits inside the family the chain already flies.

    ``src/fly_and_park.py``'s admissible 2.00 S lock comes out at a 8.61 km/s
    departure burn and a 63.35 km/s arrival, in a circular coplanar model.  The
    real-ephemeris chain's own 2S cycles run 6.84-7.17 km/s and 61.8-65.1 km/s,
    so the arrival lands inside the family and the burn sits about 20 percent
    above the dearest of them.  Different models, so this is a family
    resemblance and not an identity -- which is exactly why the paper is told to
    quote it as one (ADR 0031).
    """
    twos = [c for c in adaptive_two_wave_cycles() if c.synodic_multiple == 2]
    burns = [c.departure_burn for c in twos]
    arrivals = [c.growth_wave_v_b for c in twos]
    assert min(burns) == pytest.approx(6.84, abs=0.05)
    assert max(burns) == pytest.approx(7.17, abs=0.05)
    assert min(arrivals) == pytest.approx(61.83, abs=0.2)
    assert max(arrivals) == pytest.approx(65.13, abs=0.2)
    assert min(arrivals) < 63.35 < max(arrivals)
    assert max(burns) < 8.613


@pytest.mark.slow
def test_the_flown_chain_diverts_a_fifth_of_each_batch_to_projectiles() -> None:
    """What the fly-and-park scorer omits, sized so the ladder can say it.

    ``src/fly_and_park.py`` mints ``M(v_b)`` from the whole arriving wave and
    charges only the slug, so it never pays for delivering the head-on stream
    (CONTEXT.md, "departure-burn accounting seam").  This ledger does pay: the
    batch splits at Jupiter and the unpowered nozzle bend carries mass that
    mints no payload.  That gap is why the fly-and-park doubling times are
    systematically optimistic against this chain's, and ADR 0031 quotes these
    ranges rather than asserting a direction.
    """
    cycles = adaptive_two_wave_cycles()
    # ADR 0038 (onward burn): was 0.195-0.236 and 0.242-0.290.
    for recovery, low, high in ((0.8, 0.170, 0.242), (0.6, 0.213, 0.297)):
        bends = [
            1.0 - price_cycle(cycle, recovery, 0.8).wave_to_growth for cycle in cycles
        ]
        assert min(bends) == pytest.approx(low, abs=0.005), recovery
        assert max(bends) == pytest.approx(high, abs=0.005), recovery
