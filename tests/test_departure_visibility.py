"""Tests for the visibility-constrained departure loss (ADR 0066)."""

import numpy as np
import pytest
from astropy import units as u

from src.departure_visibility import (
    TABLE_TIMES,
    OnwardStream,
    _stored,
    loss_model_from_table,
    onward_streams,
    stored_loss_models,
    stored_tables,
    visible_burn,
)
from src.growth_ledger import (
    DEFAULT_PARKING_DAYS,
    DEPARTURE_ALTITUDE,
    departure_at_altitude,
    parking_period,
)
from src.orbit_turn_analysis import (
    THREE_SYNODIC_AIM_SEPARATION,
    THREE_SYNODIC_DEPARTURE_VINF,
    THREE_SYNODIC_RETURN_VINF,
    finite_burn_turn,
)
from src.two_wave_growth import TwoWaveCycle

#: ADR 0045's circular 3S closure, which a 20-minute burn keeps in view.
REFERENCE = OnwardStream(
    0,
    3,
    float(THREE_SYNODIC_DEPARTURE_VINF.to_value(u.km / u.s)),
    float(THREE_SYNODIC_RETURN_VINF.to_value(u.km / u.s)),
    float(THREE_SYNODIC_AIM_SEPARATION.to_value(u.deg)),
)
#: The flown chain's most nearly head-on stream (cycle 1, 2S).
HEAD_ON = OnwardStream(1, 2, 14.2516, 60.3510, 175.513)


def test_a_reference_burn_in_view_takes_its_free_optimum():
    burn = visible_burn(REFERENCE, 1200.0)
    assert burn is not None
    assert burn.share_before == burn.unconstrained_share_before
    # ADR 0045's fixed-direction loss on the same closure and burn.
    expected = finite_burn_turn(False).finite_burn_loss.to_value(u.km / u.s)
    assert burn.loss == pytest.approx(expected, rel=0.01)


def test_a_head_on_stream_forces_the_burn_later_and_costs_loss():
    burn = visible_burn(HEAD_ON, 1200.0)
    assert burn is not None
    assert burn.share_before < 0.5 * burn.unconstrained_share_before
    assert burn.loss > 1.2 * burn.unconstrained_loss


def test_the_loss_model_refuses_a_burn_that_loses_the_stream():
    table = [0.0, 0.1, 0.3] + [np.inf] * (TABLE_TIMES.size - 3)
    model = loss_model_from_table(table)
    assert model(5.0 * u.km / u.s, 400.0 * u.s).to_value(u.m / u.s) == pytest.approx(
        100.0
    )
    with pytest.raises(ValueError, match="loses the stream"):
        model(5.0 * u.km / u.s, (TABLE_TIMES[2] + 1.0) * u.s)
    with pytest.raises(RuntimeError):
        loss_model_from_table([0.0] + [np.inf] * (TABLE_TIMES.size - 1))


@pytest.mark.slow
def test_the_onward_streams_match_the_chains_own_speeds():
    from src.seed_cost import chains

    flown, _ = chains(DEFAULT_PARKING_DAYS)
    streams = onward_streams(flown)
    mu = 398600.4418
    radius = 6378.1366 + DEPARTURE_ALTITUDE.to_value(u.km)
    for cycle, stream in zip(flown, streams):
        departure = departure_at_altitude(
            cycle.onward_burn * u.km / u.s, DEPARTURE_ALTITUDE, parking_period(cycle)
        )
        final = (departure.start_speed + departure.burn).to_value(u.km / u.s)
        assert stream.departure_vinf == pytest.approx(
            np.sqrt(final**2 - 2.0 * mu / radius), rel=1e-4
        )
        assert 150.0 < stream.aim_separation < 180.0


def test_the_stored_tables_are_head_on_and_rise_with_burn_length():
    stored = list(_stored().values())
    assert len(stored) > 11
    for stream, table in stored:
        assert 140.0 < stream.aim_separation < 180.0
        assert table[0] == 0.0
        assert np.all(np.diff(table) > 0.0)


def test_a_cycle_off_the_stored_chain_is_refused():
    synthetic = TwoWaveCycle(
        index=0, departure_jd=0.0, return_jd=0.0, synodic_multiple=3,
        period_years=3.276, departure_burn=5.33, nozzle_wave_v_b=55.4,
        nozzle_wave_dsm=0.0, split_days=20.0, growth_wave_arrival_jd=0.0,
        growth_wave_v_b=57.4, growth_wave_burn=0.0, onward_burn=5.33,
    )  # fmt: skip
    with pytest.raises(KeyError, match="make departure-visibility"):
        stored_loss_models([synthetic])


@pytest.mark.slow
def test_the_stored_tables_reproduce_for_the_flown_chain():
    from src.seed_cost import chains

    flown, three = chains(DEFAULT_PARKING_DAYS)
    for cycles, threshold in ((flown, 50.0), (three, 0.0)):
        stored = stored_tables(cycles, DEFAULT_PARKING_DAYS)
        fresh = onward_streams(cycles, threshold)
        for (stream, _), rebuilt in zip(stored, fresh):
            assert stream.aim_separation == pytest.approx(
                rebuilt.aim_separation, abs=1e-5
            )
    stored = stored_tables(flown, DEFAULT_PARKING_DAYS)
    # One node regenerated: the most head-on stream at a 40-minute burn.
    stream, table = stored[1]
    node = int(np.searchsorted(TABLE_TIMES, 2400.0))
    burn = visible_burn(stream, float(TABLE_TIMES[node]), DEFAULT_PARKING_DAYS)
    assert burn is not None
    assert burn.loss == pytest.approx(table[node], rel=1e-6)
