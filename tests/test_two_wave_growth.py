"""Tests for src/two_wave_growth.py.

The real-orbit chain (Astropy ephemeris + Lambert arcs + per-cycle split
solves) is exercised in ``slow``-marked tests; the pricing algebra is pinned
fast with explicit inputs.
"""

import math
from dataclasses import replace

import astropy.units as u
import pytest

from src.nozzle_analysis import same_cycle_nozzle
from src.two_wave_growth import (
    VE_METHALOX,
    TwoWaveCycle,
    adaptive_two_wave_cycles,
    analyze_two_wave_growth,
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
    """At f = 0.8 and a 20 d parking orbit, pricing is same_cycle_nozzle's default."""
    priced = price_cycle(CYCLE, recovery=0.6, fudge=0.8, slug_ratio=7.0)

    expected = same_cycle_nozzle(
        growth_collision_speed=CYCLE.growth_wave_v_b,
        growth_wave_burn=CYCLE.growth_wave_burn + CYCLE.nozzle_wave_dsm,
        nozzle_collision_speed=CYCLE.nozzle_wave_v_b,
        departure_dv=CYCLE.departure_burn,
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
