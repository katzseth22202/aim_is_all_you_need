"""Tests for src/harvest.py: delivery to L1, liquidation and the steady state."""

import numpy as np
import pytest
from astropy import units as u

from src.growth_ledger import LAUNCH_UNIT, PLATE_MASS, wave_speed_at_altitude
from src.harvest import (
    HALO_INSERTION,
    PLATE_PRICES,
    Returns,
    constant_k_delivery,
    fleet_irr,
    l1_transfer_speed,
    liquidation_value,
    optimal_delivery,
    solve_rate,
    steady_over_liquidation,
    steady_state_value,
    value_design,
)
from src.propulsion import payload_mass_ratio
from src.seed_cost import DESIGNS, design_chain
from src.two_wave_growth import VE_METHALOX
from src.water_plate import ARGON_SLUG, PLATE_MAX_SLUG_RATIO, plate_push

KM_S = u.km / u.s
#: Cycle 0's return (55.449 km/s at 200 km) moved to the 400 km intercept.
CYCLE0_WAVE = wave_speed_at_altitude(55.449482531 * KM_S, 400 * u.km)
CYCLE1_WAVE = wave_speed_at_altitude(61.377829636 * KM_S, 400 * u.km)


def test_the_l1_transfer_leaves_400_km_at_10_82() -> None:
    assert l1_transfer_speed().to_value(KM_S) == pytest.approx(10.82, abs=0.005)


def test_k0_cargo_is_the_unit_less_the_plate_and_the_halo_insertion() -> None:
    flown = constant_k_delivery(CYCLE0_WAVE, 0.0)
    halo = np.exp(-HALO_INSERTION.to_value(KM_S) / VE_METHALOX)
    expected = (LAUNCH_UNIT - PLATE_MASS).to_value(u.t) * halo
    assert flown.cargo.to_value(u.t) == pytest.approx(expected, rel=1e-9)
    assert flown.slug.to_value(u.t) == 0.0


@pytest.mark.parametrize(
    "efficiency, companion, parent", [(0.64, 8.86, 7.94), (1.0, 9.84, 9.93)]
)
def test_k0_against_the_parents_puffsat_ratio(
    efficiency: float, companion: float, parent: float
) -> None:
    """``eq:PuffSat_ratio`` has ``2f``; the companion derates only the rebound,
    ``beta = sqrt(eta) eta_chem + 1``.  At ``f = sqrt(eta) = 0.8`` it sits above;
    at ``eta = 1`` slightly below, since the ice's bonds leave ``eta_chem`` ~0.98."""
    push = plate_push(60 * KM_S, 10.95 * KM_S, efficiency, 0.0, ARGON_SLUG)
    assert push.delivered_per_puffsat == pytest.approx(companion, abs=0.005)
    ratio = payload_mass_ratio(
        10.95 * KM_S, 60 * KM_S, fudge_factor=np.sqrt(efficiency)
    )
    assert float(ratio) == pytest.approx(parent, abs=0.005)


def test_cycle0_matches_the_todo_scratch() -> None:
    """k = 0: P 7.48, $28.0/kg lob.  k = 10: P 10.15, $55.7/kg."""
    low = constant_k_delivery(CYCLE0_WAVE, 0.0)
    high = constant_k_delivery(CYCLE0_WAVE, PLATE_MAX_SLUG_RATIO)
    assert low.cargo_per_puffsat == pytest.approx(7.48, abs=0.005)
    assert low.lob_per_kg == pytest.approx(28.0, abs=0.05)
    assert high.cargo_per_puffsat == pytest.approx(10.15, abs=0.005)
    assert high.lob_per_kg == pytest.approx(55.7, abs=0.05)
    assert high.puffsats.to_value(u.t) == pytest.approx(66.4, abs=0.05)


def test_cycle1_at_constant_k10() -> None:
    """The todo's pin: cycle 1 (2S) at k = 10, P 11.98 and $51.4/kg."""
    flown = constant_k_delivery(CYCLE1_WAVE, PLATE_MAX_SLUG_RATIO)
    assert flown.cargo_per_puffsat == pytest.approx(11.98, abs=0.005)
    assert flown.lob_per_kg == pytest.approx(51.4, abs=0.05)


@pytest.mark.parametrize("price", [100.0, 200.0, 500.0])
def test_the_optimised_schedule_beats_both_constant_loadings(price: float) -> None:
    plate = PLATE_PRICES["learned"]
    best = optimal_delivery(CYCLE0_WAVE, price, plate)
    for k in (0.0, PLATE_MAX_SLUG_RATIO):
        held = constant_k_delivery(CYCLE0_WAVE, k)
        assert best.net_per_puffsat(price, plate) >= held.net_per_puffsat(price, plate)
    assert best.slug_ratio_start <= PLATE_MAX_SLUG_RATIO


def test_at_500_the_schedule_tapers_from_the_cap() -> None:
    best = optimal_delivery(CYCLE0_WAVE, 500.0, PLATE_PRICES["learned"])
    assert best.slug_ratio_start == pytest.approx(PLATE_MAX_SLUG_RATIO)
    assert 1.0 < best.slug_ratio_end < 4.0
    held = constant_k_delivery(CYCLE0_WAVE, PLATE_MAX_SLUG_RATIO)
    assert best.cargo > held.cargo
    assert best.cargo_per_puffsat > held.cargo_per_puffsat


def _constant_returns(growth: float, period: float, cycles: int = 4) -> Returns:
    times = tuple(period * (n + 1) for n in range(cycles))
    return Returns(
        growths=(growth,) * cycles,
        times=times,
        arrivals=(1.0,) * cycles,
        chain_years=times[-1],
        harvest=0,
        batch=1.0,
    )


def test_the_solver_returns_the_rate_that_repays_the_seed() -> None:
    returns = _constant_returns(3.0, 3.0)
    rate = solve_rate(lambda r: liquidation_value(returns, [100.0] * 4, r, 40.0))
    assert rate is not None
    assert liquidation_value(returns, [100.0] * 4, rate, 40.0) == pytest.approx(
        1.0, abs=1e-9
    )


def test_the_solver_returns_zero_at_break_even_and_none_below() -> None:
    returns = _constant_returns(3.0, 3.0)
    assert solve_rate(lambda r: liquidation_value(returns, [40.0] * 4, r, 40.0)) == 0.0
    assert solve_rate(lambda r: liquidation_value(returns, [30.0] * 4, r, 40.0)) is None


@pytest.mark.parametrize("growth, period", [(3.0, 3.276), (2.2, 2.184)])
def test_steady_beats_liquidation_exactly_when_g_beats_the_rate(
    growth: float, period: float
) -> None:
    """The flip sits at ``G = (1+r)^T``, from the formula and the chain sum alike."""
    flip = growth ** (1.0 / period) - 1.0
    returns = _constant_returns(growth, period)
    for rate, steady_wins in ((flip * 0.99, True), (flip * 1.01, False)):
        assert (steady_over_liquidation(growth, period, rate) > 1.0) is steady_wins
        steady = steady_state_value(returns, [1.0] * 4, rate, 1.0)
        liquid = liquidation_value(returns, [1.0] * 4, rate, 1.0)
        assert (steady > liquid) is steady_wins
        assert steady / liquid == pytest.approx(
            steady_over_liquidation(growth, period, rate), rel=1e-12
        )


def test_the_steady_state_needs_a_positive_rate() -> None:
    with pytest.raises(ValueError):
        steady_state_value(_constant_returns(2.0, 2.0), [1.0] * 4, 0.0, 1.0)


@pytest.mark.slow
def test_the_fleet_irr_is_the_growth_rate_less_the_stepwise_gap() -> None:
    """``M10^(1/t_h) - 1`` against ``annual_growth`` on the 3S chain (methalox)."""
    chain = design_chain(DESIGNS[0])
    valued = value_design(chain, 500.0, PLATE_PRICES["learned"])
    years = valued.returns.harvest_years
    assert years == pytest.approx(9.83, abs=0.005)
    irr = fleet_irr(chain.summary.ten_year_stepwise, years)
    # ADR 0038 (onward burn): was -0.0137.
    assert irr - chain.summary.annual_growth == pytest.approx(-0.0057, abs=5e-4)


@pytest.mark.slow
def test_the_harvest_batch_is_m10_over_the_last_growth() -> None:
    chain = design_chain(DESIGNS[4])
    valued = value_design(chain, 500.0, PLATE_PRICES["learned"])
    h = valued.returns.harvest
    assert h == chain.finished - 1
    assert valued.returns.batch * chain.growths[h] == pytest.approx(
        chain.summary.ten_year_stepwise, rel=1e-12
    )


@pytest.mark.slow
def test_the_irr_rises_with_the_sale_price_and_falls_with_the_seed_price() -> None:
    chain = design_chain(DESIGNS[0])
    plate = PLATE_PRICES["learned"]
    rates = [
        value_design(chain, p, plate).liquidation_irr(337.0) for p in (200, 300, 500)
    ]
    assert all(r is not None for r in rates)
    assert rates[0] < rates[1] < rates[2]  # type: ignore[operator]
    valued = value_design(chain, 500.0, plate)
    cheap, dear = valued.steady_irr(337.0), valued.steady_irr(3000.0)
    assert cheap is not None and dear is not None and cheap > dear
