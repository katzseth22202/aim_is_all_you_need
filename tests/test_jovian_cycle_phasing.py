"""Tests for src/jovian_cycle_phasing.py."""

import numpy as np
import pytest
from astropy import units as u

import src.jovian_cycle_phasing as jcp
from src.astro_constants import METHALOX_VACUUM_ISP
from src.jovian_cycle_phasing import (
    _EARTH_JUPITER_SYNODIC_YEARS,
    ChainResult,
    _closing_returns,
    _cycle_branches,
    _net_growth,
    _outbound_arrival,
    optimize_jovian_cycle_chain,
)
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.propulsion import exhaust_velocity_from_isp, payload_mass_ratio
from src.retrograde_return_legs import _assist_chain_params, _earth_phase_mismatch

# Jupiter longitude that puts a cheap, growth-viable cycle at the epoch (found by
# sweeping the initial phase; the incumbent powered flyby's ~4.4 km/s departure).
_GOOD_JUPITER_LON = 1.702  # rad
_YEAR_S = float((1.0 * u.year).to_value(u.s))


@pytest.fixture
def coarse_search(monkeypatch):
    """Run the chain search on the pre-ADR-0030 coarse box, for speed.

    ADR 0030's converged settings (200 samples, 200-wide beam) cost ~32 s a call,
    which does not belong in the fast suite. The tests using this fixture protect
    *structure* -- that each departure is pinned to the previous arrival, and
    that a self-sustaining chain closes at all -- and neither reads a figure off
    the search, so a coarse box proves exactly as much. Anything that quotes a
    number runs converged and is marked ``slow`` (CLAUDE.md, "The slow split").
    """
    monkeypatch.setattr(jcp, "_OUTBOUND_TOF_SAMPLES", 26)
    monkeypatch.setattr(jcp, "_BEAM_WIDTH", 48)


def _params() -> object:
    cycle_speed = puffsat_cycle_periapsis_speed()
    return _assist_chain_params(
        target_collision_speed=float(cycle_speed.to_value(u.km / u.s)),
        cycle_periapsis_speed=cycle_speed,
    )


def test_net_growth_matches_the_growth_arithmetic() -> None:
    # _net_growth inlines delivered_fraction x payload_mass_ratio in floats; it
    # must match the Quantity-valued helpers it stands in for.
    params = _params()
    v_rf = params.flyby.v_rf  # type: ignore[attr-defined]
    departure_burn, flyby_burn, v_b = 4.0, 1.0, 55.0
    got = _net_growth(departure_burn, flyby_burn, v_b, params)  # type: ignore[arg-type]
    exhaust = exhaust_velocity_from_isp(METHALOX_VACUUM_ISP)
    delivered = float(np.exp(-((departure_burn + flyby_burn) * u.km / u.s) / exhaust))
    mass_ratio = payload_mass_ratio(v_rf=v_rf * u.km / u.s, v_b=v_b * u.km / u.s)
    assert got is not None
    assert got == pytest.approx(delivered * mass_ratio, rel=1e-9)


def test_net_growth_rejects_collision_below_push_target() -> None:
    # A collision that does not exceed the cycle-orbit push target mints no
    # payload, so no growth factor exists.
    params = _params()
    v_rf = params.flyby.v_rf  # type: ignore[attr-defined]
    assert _net_growth(1.0, 0.0, v_rf - 1.0, params) is None  # type: ignore[arg-type]


def test_closing_return_actually_intercepts_earth() -> None:
    # Every leg _closing_returns yields must drive the Earth-phase mismatch to
    # zero: the return crossing coincides with where Earth actually is.
    params = _params()
    outbound_tof = 1.34 * _YEAR_S
    arrival = _outbound_arrival(0.0, outbound_tof, 0.0, _GOOD_JUPITER_LON, params)  # type: ignore[arg-type]
    assert arrival is not None
    _departure_burn, excess_arrival, lon_jupiter = arrival
    found = False
    for bend_sign in (1.0, -1.0):
        for leg, _residual in _closing_returns(
            excess_arrival,
            lon_jupiter,
            outbound_tof,
            0.0,
            0.0,
            bend_sign,
            6.0 * _YEAR_S,
            params,  # type: ignore[arg-type]
        ):
            mismatch = _earth_phase_mismatch(
                leg, lon_jupiter, outbound_tof, 0.0, params.flyby  # type: ignore[attr-defined]
            )
            assert abs(mismatch) < 1e-3
            found = True
    assert found, "expected at least one closing retrograde return"


def test_cycle_branches_are_growth_viable_at_a_good_phase() -> None:
    # From a well-phased departure the enumerator finds closing trajectories, and
    # the best of them grows the payload (net_growth > 1).
    params = _params()
    coast = float((20.0 * u.day).to_value(u.s))
    branches = _cycle_branches(
        0.0, False, 0.0, _GOOD_JUPITER_LON, 7.0 * _YEAR_S, coast, params  # type: ignore[arg-type]
    )
    assert branches
    assert all(b.collision_speed > params.flyby.v_rf for b in branches)  # type: ignore[attr-defined]
    assert max(b.net_growth for b in branches) > 1.0
    # Every branch advances time by at least one outbound leg (~1.1 yr).
    assert all(b.next_departure > 1.1 * _YEAR_S for b in branches)


def test_chain_departures_are_pinned_to_the_previous_arrival(coarse_search) -> None:
    # The mass cannot wait: each cycle's launch is the previous cycle's launch
    # plus its full cycle time (outbound + return + 20-day coast).
    result = optimize_jovian_cycle_chain(years=12.0, powered=False)
    assert len(result.cycles) >= 2
    for earlier, later in zip(result.cycles, result.cycles[1:]):
        expected = earlier.launch_time + earlier.cycle_time
        assert float(later.launch_time.to_value(u.year)) == pytest.approx(
            float(expected.to_value(u.year)), abs=0.02
        )
    # cycle_time is exactly outbound + return + 20 days.
    coast = (20.0 * u.day).to(u.year)
    for cycle in result.cycles:
        total = cycle.outbound_time + cycle.return_time + coast
        assert float(cycle.cycle_time.to_value(u.year)) == pytest.approx(
            float(total.to_value(u.year)), rel=1e-9
        )


def test_unpowered_chain_self_sustains(coarse_search) -> None:
    # The headline: with no perijove burn the loop keeps closing at growth-viable
    # cost, so the launched mass compounds rather than stalling.
    result = optimize_jovian_cycle_chain(years=12.0, powered=False)
    assert isinstance(result, ChainResult)
    assert result.all_growth_positive
    assert result.mass_multiple_30yr > 1.0
    assert len(result.cycles) >= 3


@pytest.mark.slow
def test_the_perijove_burn_converges_to_zero_and_buys_nothing() -> None:
    # ADR 0030, retiring ADR 0010 decision 3. Handed a free perijove burn the
    # optimizer drives every cycle's burn to exactly zero and reproduces the
    # unpowered chain. The "+20% from a second steering knob" was an artifact of
    # the old 26-sample / 48-wide search box; at converged settings the two runs
    # are identical.
    #
    # Horizon is 8 yr, not 30 or 12. The powered branch set is ~5x the unpowered
    # one, so this test is dominated by the powered run, and the claim it makes
    # is *structural* -- every burn is zero, and the two chains coincide -- not a
    # statement about any horizon. Cost and cycle count by horizon, measured as
    # one sequential sweep so the ratios are comparable (every row gave
    # max|burn| = 0 and exact powered/unpowered equality):
    #
    #     6 yr    1 cycle    0.27x   <- too few: cannot show the chain chains
    #     7 yr    2 cycles   0.40x
    #     8 yr    2 cycles   0.52x   <- chosen; keeps a cycle of margin over 7
    #    10 yr    3 cycles   0.74x
    #    12 yr    3 cycles   1.00x   <- previous setting
    #
    # Cost is given relative to the old 12 yr setting rather than in seconds:
    # absolute timings on the machine that measured this varied by up to 2x
    # between otherwise identical runs, so only the ratios are meaningful.
    #
    # 8 yr keeps two cycles, so the equality is exercised across a chained
    # relaunch rather than a single solve, and roughly halves the slow suite's
    # largest single contribution. CLAUDE.md, "The slow split": bracket the
    # search around the answer you already know rather than dropping coverage.
    unpowered = optimize_jovian_cycle_chain(years=8.0, powered=False)
    powered = optimize_jovian_cycle_chain(years=8.0, powered=True)
    assert unpowered.all_growth_positive
    assert powered.all_growth_positive
    # Two cycles, so "the chain keeps closing" is actually under test. Without
    # this a future change that quietly drops to one cycle would still pass.
    assert len(unpowered.cycles) >= 2
    # mass_multiple_30yr is the compounded total over whatever horizon was run,
    # so at years=8 it is the 8-year figure (~3.64 across 2 cycles), not 233.
    assert unpowered.mass_multiple_30yr > 3.0

    # Every perijove burn on the powered chain is zero...
    for cycle in powered.cycles:
        assert float(cycle.flyby_burn.to_value(u.km / u.s)) == pytest.approx(
            0.0, abs=1e-9
        )
    # ...so the powered run cannot do better, and in fact matches exactly.
    assert powered.mass_multiple_30yr == pytest.approx(
        unpowered.mass_multiple_30yr, rel=1e-9
    )
    assert len(powered.cycles) == len(unpowered.cycles)
    for lhs, rhs in zip(unpowered.cycles, powered.cycles):
        assert float(lhs.collision_speed.to_value(u.km / u.s)) == pytest.approx(
            float(rhs.collision_speed.to_value(u.km / u.s)), rel=1e-9
        )

    # The milestones grow monotonically along each chain.
    for result in (unpowered, powered):
        assert (
            result.mass_multiple_10yr
            <= result.mass_multiple_20yr
            <= result.mass_multiple_30yr
        )


@pytest.mark.slow
def test_the_converged_chain_settles_onto_the_synodic_clock() -> None:
    # ADR 0030. Nothing in _cycle_branches knows what a synodic period is, and
    # the search may return any real cycle length -- yet at converged settings
    # every cycle but the horizon-truncated last one lands within a few percent
    # of an integer multiple of the 1.0923 yr Earth-Jupiter synodic. Outside a
    # narrow ~2.92-3.06 S window the next departure offers no growing cycle at
    # all, so the chain has nowhere else to stand.
    result = optimize_jovian_cycle_chain(years=30.0, powered=False)
    assert len(result.cycles) >= 8
    synodics = [
        float(c.cycle_time.to_value(u.year)) / _EARTH_JUPITER_SYNODIC_YEARS
        for c in result.cycles[:-1]  # the last cycle is cut short by the horizon
    ]
    for value in synodics:
        assert abs(value - round(value)) < 0.05, synodics
    # And the clock it picks is 3S, not the 2S the paper's resonance audit uses.
    assert round(synodics[0]) == 3
