"""Pinned tests for src/two_leg_nozzle_sweep.py (ADR 0014).

Everything that needs the real ephemeris is marked ``slow``; the ledger algebra
and the ground-launch bookkeeping are pinned against synthetic cycles so the
fast suite still covers the substitution that defines the module.
"""

import astropy.units as u
import numpy as np
import pytest

from src.nozzle_analysis import same_cycle_nozzle
from src.plume_thermal import slug_ratio_window
from src.two_leg_nozzle_sweep import DEFAULT_SPLIT_DAYS as SWEEP_SPLIT_DAYS
from src.two_leg_nozzle_sweep import (
    PAYLOAD_FRACTION_AT_INTERCEPT,
    RETURN_FLOOR,
    _score,
    coldest_closing_speeds,
    equivalent_plate_elasticity,
    fleet_ignition_windows,
    price_chain_two_leg,
    returned_launch_fraction,
)
from src.two_wave_growth import DEFAULT_SPLIT_DAYS as CHAIN_SPLIT_DAYS
from src.two_wave_growth import TwoWaveCycle, adaptive_two_wave_cycles, price_chain

VE_METHALOX = 3.7265


def _cycle(
    index: int = 0, growth_v_b: float = 59.29, nozzle_v_b: float = 55.45
) -> TwoWaveCycle:
    """Build a synthetic flown cycle with the chain's typical numbers."""
    return TwoWaveCycle(
        index=index,
        departure_jd=2_461_000.0,
        return_jd=2_462_200.0,
        synodic_multiple=3,
        period_years=3.276,
        departure_burn=5.329,
        nozzle_wave_v_b=nozzle_v_b,
        nozzle_wave_dsm=0.0,
        split_days=10.0,
        growth_wave_arrival_jd=2_462_190.0,
        growth_wave_v_b=growth_v_b,
        growth_wave_burn=0.0,
    )


def test_payload_fraction_matches_the_stated_ledger() -> None:
    """Two thirds propellant, a quarter of the rest dry, leaves a quarter."""
    assert np.isclose(PAYLOAD_FRACTION_AT_INTERCEPT, 0.25, rtol=1e-12)


def test_coldest_ends_are_opposite_ends_of_the_two_burns() -> None:
    """The push slows down; the departure burn speeds up into the stream."""
    push_end, burn_start = coldest_closing_speeds(_cycle())
    assert push_end < _cycle().growth_wave_v_b
    assert burn_start > _cycle().nozzle_wave_v_b


def test_returned_fraction_charges_every_loss() -> None:
    """Ground propellant, dry mass, both slugs, reversal and corrections."""
    cycle = _cycle()
    pricing = same_cycle_nozzle(
        growth_collision_speed=cycle.growth_wave_v_b,
        growth_wave_burn=0.0,
        nozzle_collision_speed=cycle.nozzle_wave_v_b,
        departure_dv=cycle.departure_burn,
        cycle=cycle.period_years,
        exhaust_speed=VE_METHALOX,
        recovery=0.6,
        slug_ratio=7.0,
        growth_slug_ratio=4.0,
        growth_recovery=0.6,
        reversal_period=cycle.split_days * u.day,
    )
    returned = returned_launch_fraction(cycle, pricing)
    assert (
        0.0 < returned < PAYLOAD_FRACTION_AT_INTERCEPT / pricing.mass_multiplier * 1.001
    )
    assert returned < PAYLOAD_FRACTION_AT_INTERCEPT


def test_fleet_window_is_the_intersection_of_the_cycles() -> None:
    """One nozzle flies every cycle, so the coldest cycle sets the ceiling."""
    cold = _cycle(0, growth_v_b=58.44, nozzle_v_b=54.49)
    warm = _cycle(1, growth_v_b=66.77, nozzle_v_b=63.31)
    leg1, leg2 = fleet_ignition_windows([cold, warm])
    assert leg1 is not None and leg2 is not None
    cold_alone = slug_ratio_window(coldest_closing_speeds(cold)[0] * u.km / u.s)
    assert cold_alone is not None
    assert np.isclose(leg1[1], cold_alone[1], rtol=1e-12)


@pytest.mark.slow
def test_plate_path_reproduces_adr_0013_where_the_floor_is_slack() -> None:
    """At e = 0.6, f = 0.8 the launch floor does not bind, so ADR 0013 stands."""
    cycles = adaptive_two_wave_cycles(split_days=10.0)
    incumbent = price_chain(cycles, 0.6, 0.8)
    constrained = price_chain_two_leg(cycles, 0.6, None, 0.8)
    assert constrained is not None
    assert np.isclose(constrained.rate, incumbent.rate, rtol=1e-3)
    assert np.isclose(constrained.rate, 0.3989, atol=5e-4)
    assert constrained.worst_return_fraction > RETURN_FLOOR


@pytest.mark.slow
def test_launch_floor_is_what_binds_across_most_of_the_grid() -> None:
    """The ground-launch ledger, not the plume, is the active constraint."""
    cycles = adaptive_two_wave_cycles(split_days=10.0)
    chain = price_chain_two_leg(cycles, 0.6, 0.6)
    assert chain is not None
    assert chain.launch_limited
    assert not chain.ignition_limited
    assert np.isclose(chain.worst_return_fraction, RETURN_FLOOR, rtol=2e-3)


@pytest.mark.slow
def test_crossover_against_the_paper_plate() -> None:
    """The overtaking nozzle must beat the plate's own recovery to draw level.

    At a good head-on leg (``e2 = 0.8``) the crossover against the paper's
    ``f = 0.8`` plate sits between ``e1 = 0.5`` and ``e1 = 0.6``, and beating a
    perfectly elastic plate takes ``e1`` around 0.8.
    """
    cycles = adaptive_two_wave_cycles(split_days=10.0)
    below = price_chain_two_leg(cycles, 0.8, 0.5)
    above = price_chain_two_leg(cycles, 0.8, 0.6)
    perfect = price_chain_two_leg(cycles, 0.8, 0.8)
    assert below is not None and above is not None and perfect is not None
    f_below = equivalent_plate_elasticity(cycles, below)
    f_above = equivalent_plate_elasticity(cycles, above)
    f_perfect = equivalent_plate_elasticity(cycles, perfect)
    assert f_below is not None and f_above is not None and f_perfect is not None
    assert f_below < 0.8 < f_above
    assert f_perfect > 1.0


def test_both_modules_fly_the_same_split_gap() -> None:
    """The sweep and a bare chain call must not silently fly different chains.

    ``adaptive_two_wave_cycles`` once defaulted to a 20-day gap while the sweep
    ran at 10, so calling the chain builder directly gave growth-wave closing
    speeds about 2 km/s hotter than any published table.  ADR 0013 decided the
    gap is 10 days; both modules now read one constant.
    """
    assert CHAIN_SPLIT_DAYS == SWEEP_SPLIT_DAYS == 10.0


@pytest.mark.slow
def test_matched_recovery_reverses_the_plate_verdict() -> None:
    """ADR 0015's headline: at ``e1 = e2 = f`` the two-leg nozzle wins.

    Pins the ends and the crossover of ADR 0015's matched-recovery table.  The
    nozzle loses at ``e = 0.25``, ties near 0.30, and is 8.5x the plate by 0.60.
    """
    cycles = adaptive_two_wave_cycles(split_days=CHAIN_SPLIT_DAYS)

    for recovery, nozzle_growth, plate_growth in (
        (0.25, 1.33e-4, 3.09e-4),
        (0.30, 2.95e-2, 2.91e-2),
        (0.60, 6.632e4, 7.827e3),
    ):
        nozzle = price_chain_two_leg(cycles, recovery, recovery)
        plate = price_chain_two_leg(cycles, recovery, None, recovery)
        assert nozzle is not None and plate is not None
        assert np.isclose(nozzle.total_growth, nozzle_growth, rtol=3e-3)
        assert np.isclose(plate.total_growth, plate_growth, rtol=3e-3)

    # The verdict flips between 0.25 and 0.30 and compounds above it.
    loses = price_chain_two_leg(cycles, 0.25, 0.25)
    loses_plate = price_chain_two_leg(cycles, 0.25, None, 0.25)
    wins = price_chain_two_leg(cycles, 0.60, 0.60)
    wins_plate = price_chain_two_leg(cycles, 0.60, None, 0.60)
    assert loses is not None and loses_plate is not None
    assert wins is not None and wins_plate is not None
    assert loses.total_growth < loses_plate.total_growth
    assert wins.total_growth > 8.0 * wins_plate.total_growth

    # As a rate the same gap is only ~24%, which is why ADR 0015 insists the
    # currency be named: 0.3910 e-foldings/yr against 0.3158.
    assert np.isclose(wins.rate, 0.3910, atol=5e-4)
    assert np.isclose(wins_plate.rate, 0.3158, atol=5e-4)


def test_the_toll_penalises_the_two_leg_nozzle_more_than_the_plate() -> None:
    """A plate owes no chemistry, so the toll is not symmetric between options.

    This is the finding that puts ADR 0014/0015 back in play: the frozen
    dissociation reaches ``tab:space_mortgage_growth`` only through the head-on
    leg, but reaches ``tab:two_leg_growth`` through both -- and the growth
    push's 45-56 km/s is where ``eta_chem`` is harshest.
    """
    cycles = [_cycle(0), _cycle(1, growth_v_b=57.43, nozzle_v_b=56.53)]
    plate_hit = (
        _score(cycles, None, 0.8, 1.0, 0.0, 8.5, 0.8)[0]
        - _score(cycles, None, 0.8, 0.8, 0.0, 8.5)[0]
    )
    nozzle_hit = (
        _score(cycles, 0.8, 0.8, 1.0, 8.5, 8.5, 0.8)[0]
        - _score(cycles, 0.8, 0.8, 0.8, 8.5, 8.5)[0]
    )
    assert nozzle_hit < plate_hit < 0.0


def test_a_plate_leg_is_never_charged_the_chemistry() -> None:
    """``growth_recovery=None`` means leg 1 is a plate, and a plate cannot freeze.

    Isolated by varying ``k1``, which a plate ignores: if the toll had leaked
    onto the plate leg it would have made ``k1`` load-bearing there.  Leg 2 is
    tolled in both calls, so only the plate leg is under test.
    """
    cycles = [_cycle()]
    quiet = _score(cycles, None, 0.8, 1.0, 0.5, 8.5, 0.8)[0]
    loud = _score(cycles, None, 0.8, 1.0, 40.0, 8.5, 0.8)[0]
    assert np.isclose(quiet, loud, rtol=1e-12)
