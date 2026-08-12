"""Tests for the opt-in real-orbit two-synodic analysis."""

import numpy as np
import pytest

from src.real_orbit_resonance import (
    analyze_adaptive_synodic_cadence,
    analyze_two_synodic_resonance,
)


def test_short_real_orbit_audit_returns_tables_and_exact_closures() -> None:
    """A short audit should expose timing, velocities, and honest feasibility."""
    result = analyze_two_synodic_resonance(start="2026-08-11", years=5.0)
    assert len(result.windows) >= 2
    assert set(
        [
            "period_days",
            "earth_departure_vinf_km_s",
            "earth_return_vinf_km_s",
            "earth_collision_speed_km_s",
            "required_perijove_altitude_km",
            "perijove_turn_trim_m_s",
            "perijove_turn_trim_direction",
            "unpowered_feasible",
        ]
    ).issubset(result.windows.columns)
    assert result.windows["period_days"].between(780.0, 815.0).all()
    assert (result.windows["exact_closures"] >= 1).all()
    assert np.isfinite(result.statistics["variance"]).all()
    assert (result.windows["perijove_turn_trim_m_s"] <= 0.0).all()
    assert set(result.windows["perijove_turn_trim_direction"]) <= {
        "none",
        "backward",
    }
    assert set(result.maneuvers["schedule"]) == {
        "real_phase_2S",
        "fixed_mean_2S",
    }
    assert set(result.maneuvers["case"]) == {
        "perijove_only",
        "dsm_only",
        "hybrid_50mps",
    }
    assert (result.maneuvers["resonance_endpoint_error_s"] == 0.0).all()
    assert (
        result.maneuvers.loc[result.maneuvers["feasible"], "total_correction_m_s"]
        >= 0.0
    ).all()
    totals = result.maneuvers.pivot(
        index=["schedule", "window"],
        columns="case",
        values="total_correction_m_s",
    )
    assert (totals["hybrid_50mps"] <= totals["dsm_only"] + 1e-9).all()


def test_real_orbit_audit_is_strictly_unpowered() -> None:
    """Every retained closure must conserve Jupiter-relative excess speed."""
    result = analyze_two_synodic_resonance(start="2035-01-01", years=3.0)
    # The rooted incoming versus outgoing v-infinity residual verifies that the
    # flown baseline only rotates. The turn-trim column is explicitly a local
    # powered diagnostic and is not applied to these Lambert arcs.
    assert (result.windows["jupiter_vinf_km_s"] > 0.0).all()
    assert result.windows["jupiter_vinf_mismatch_m_s"].abs().max() < 1e-3
    feasible = result.windows["unpowered_feasible"]
    assert (result.windows.loc[feasible, "perijove_turn_trim_m_s"] == 0.0).all()


def test_fixed_cadence_repeats_the_identical_period() -> None:
    """The simple schedule must not accumulate endpoint timing drift."""
    result = analyze_two_synodic_resonance(start="2026-08-11", years=7.0)
    fixed = result.maneuvers.loc[
        (result.maneuvers["schedule"] == "fixed_mean_2S")
        & (result.maneuvers["case"] == "dsm_only")
    ]
    departure = np.array(
        [np.datetime64(value) for value in fixed["departure_tdb"]],
        dtype="datetime64[D]",
    )
    returns = np.array(
        [np.datetime64(value) for value in fixed["return_tdb"]], dtype="datetime64[D]"
    )
    assert np.array_equal(returns[:-1], departure[1:])
    assert np.allclose(fixed["period_days"], fixed["period_days"].iloc[0], atol=1e-9)
    assert 797.0 < fixed["period_days"].iloc[0] < 798.0
    assert fixed["resonance_endpoint_error_s"].eq(0.0).all()


def test_adaptive_cadence_chains_exact_two_or_three_synodic_periods() -> None:
    """The selected return must become the next departure without timing drift."""
    result = analyze_adaptive_synodic_cadence(
        start="2026-08-11", years=8.0, threshold_m_s=50.0
    )
    cycles = result.cycles
    assert set(cycles["selected_synodic_periods"]) <= {2, 3}
    assert cycles["three_synodic_selected"].equals(
        cycles["selected_synodic_periods"].eq(3)
    )
    assert cycles["resonance_endpoint_error_s"].eq(0.0).all()
    assert np.array_equal(
        cycles["return_tdb"].to_numpy()[:-1],
        cycles["departure_tdb"].to_numpy()[1:],
    )
    selected_two = cycles.loc[~cycles["three_synodic_selected"]]
    selected_three = cycles.loc[cycles["three_synodic_selected"]]
    assert (selected_two["two_synodic_dsm_m_s"] <= result.threshold_m_s).all()
    assert (selected_three["two_synodic_dsm_m_s"] > result.threshold_m_s).all()


@pytest.mark.slow
def test_adaptive_cadence_pins_200_year_fallback_result() -> None:
    """Pin the conditional policy's selection frequency and DSM tail."""
    result = analyze_adaptive_synodic_cadence()
    summary = result.summary.iloc[0]
    assert summary["total_cycles"] == 76
    assert summary["two_synodic_cycles"] == 47
    assert summary["three_synodic_cycles"] == 29
    assert np.isclose(summary["three_synodic_fraction"], 29.0 / 76.0)
    assert np.isclose(summary["three_synodic_cycles_per_century"], 14.5)
    assert np.isclose(summary["three_synodic_mean_dsm_m_s"], 0.394707, atol=1e-6)
    assert np.isclose(summary["three_synodic_median_dsm_m_s"], 0.345221, atol=1e-6)
    assert np.isclose(summary["three_synodic_max_dsm_m_s"], 1.009817, atol=1e-6)
    assert summary["worst_three_synodic_departure"] == "2179-09-30"
    assert np.isclose(summary["selected_max_dsm_m_s"], 18.478978, atol=1e-6)
    assert summary["selected_dsm_over_threshold_cycles"] == 0


def test_adaptive_cadence_rejects_invalid_inputs() -> None:
    """The public policy analysis should reject invalid horizons and thresholds."""
    with pytest.raises(ValueError, match="years must be positive"):
        analyze_adaptive_synodic_cadence(years=0.0)
    with pytest.raises(ValueError, match="threshold_m_s must be nonnegative"):
        analyze_adaptive_synodic_cadence(years=5.0, threshold_m_s=-1.0)
