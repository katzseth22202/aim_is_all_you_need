"""Tests for src/jovian_cycle_phasing.py."""

import numpy as np
import pytest
from astropy import units as u

from src.astro_constants import METHALOX_VACUUM_ISP
from src.jovian_cycle_phasing import (
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


def test_chain_departures_are_pinned_to_the_previous_arrival() -> None:
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


def test_unpowered_chain_self_sustains() -> None:
    # The headline: with no perijove burn the loop keeps closing at growth-viable
    # cost, so the launched mass compounds rather than stalling.
    result = optimize_jovian_cycle_chain(years=12.0, powered=False)
    assert isinstance(result, ChainResult)
    assert result.all_growth_positive
    assert result.mass_multiple_30yr > 1.0
    assert len(result.cycles) >= 3


@pytest.mark.slow
def test_powered_chain_beats_unpowered_over_30_years() -> None:
    # Over the full horizon both self-sustain, and the perijove burn -- a second
    # steering knob -- packs in more compounded mass than bending alone.
    unpowered = optimize_jovian_cycle_chain(years=30.0, powered=False)
    powered = optimize_jovian_cycle_chain(years=30.0, powered=True)
    assert unpowered.all_growth_positive
    assert powered.all_growth_positive
    assert unpowered.mass_multiple_30yr > 10.0
    assert powered.mass_multiple_30yr >= unpowered.mass_multiple_30yr
    # The milestones grow monotonically along each chain.
    for result in (unpowered, powered):
        assert (
            result.mass_multiple_10yr
            <= result.mass_multiple_20yr
            <= result.mass_multiple_30yr
        )
