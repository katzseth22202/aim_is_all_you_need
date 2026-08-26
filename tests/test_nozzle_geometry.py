"""Pinned tests for src/nozzle_geometry.py (ledger item 11).

The arrival radius is an *input* nobody had written down and ``k`` is its
output, so these pin the relation in both directions and the cross-check
against ``puffsat_impact_simulation``'s own integration.
"""

import numpy as np
import pytest

from src.nozzle_geometry import (
    BORE_RADIUS,
    DESIGN_ARRIVAL_FRACTION,
    IMPACTOR_MASS,
    arrival_fraction_for,
    bag_density,
    full_bore_slug_ratio,
    swept_slug_ratio,
)


def test_full_bore_sweep_reproduces_the_papers_slug_ratio() -> None:
    """`rho A L / m` = 8.69, which is where the paper's 8.5 comes from."""
    assert np.isclose(full_bore_slug_ratio(), 8.692, rtol=1e-3)
    assert np.isclose(bag_density(), 0.3229, rtol=1e-3)


def test_rigid_front_is_exactly_quadratic_in_arrival_radius() -> None:
    """The whole sensitivity: half the radius is a quarter of the slug."""
    for fraction in (0.5, 0.6, 0.8, 0.9, 1.0):
        assert np.isclose(
            swept_slug_ratio(fraction), full_bore_slug_ratio() * fraction**2, rtol=1e-9
        )
    assert np.isclose(swept_slug_ratio(0.5), 0.25 * swept_slug_ratio(1.0), rtol=1e-9)


def test_the_design_assumption_delivers_the_slug_ratio_it_does() -> None:
    """Seth 2026-08-26: the front spans 0.8 of the bore.

    Rigid that is 5.56, and with any spreading at all it recovers to ~8.3.
    Both numbers are quoted downstream, so both are pinned.
    """
    assert np.isclose(swept_slug_ratio(DESIGN_ARRIVAL_FRACTION), 5.563, rtol=1e-3)
    assert np.isclose(swept_slug_ratio(DESIGN_ARRIVAL_FRACTION, 3.0), 8.277, rtol=2e-3)
    assert np.isclose(swept_slug_ratio(0.9), 7.041, rtol=1e-3)


def test_a_compact_ice_rod_falls_out_the_bottom_of_the_window() -> None:
    """The failure direction that is easy to get backwards.

    A compact projectile does not overload the plume, it fails to make one:
    ``k`` = 0.034 is below the ignition window's *lower* root (0.084 at
    45.58 km/s), not above its ceiling.
    """
    radius = (3.0 * (IMPACTOR_MASS / 917.0) / (4.0 * np.pi)) ** (1.0 / 3.0)
    assert np.isclose(radius, 0.1867, rtol=1e-3)
    delivered = swept_slug_ratio(radius / BORE_RADIUS)
    assert np.isclose(delivered, 0.0337, rtol=1e-2)
    assert delivered < 0.084


def test_spreading_makes_the_arrival_radius_nearly_irrelevant() -> None:
    """23.8 m of column is a long way to close a 1.2 m gap.

    This is why the parameter is only load-bearing for a *rigid* front, and why
    ``c_exp`` rather than the arrival radius is the thing neither repo solves.
    """
    rigid = swept_slug_ratio(0.6)
    spread = swept_slug_ratio(0.6, 5.0)
    assert rigid < 3.2
    assert spread > 7.5
    # Faster projectiles have less time to spread, so k falls with closing speed.
    assert swept_slug_ratio(0.6, 5.0, 75.0) < swept_slug_ratio(0.6, 5.0, 45.58)


def test_arrival_fraction_inverts_the_sweep() -> None:
    """The optimiser picks k; this says whether the geometry can supply it."""
    for k in (3.0, 5.563, 6.75, 8.692):
        assert np.isclose(swept_slug_ratio(arrival_fraction_for(k)), k, rtol=1e-9)
    # The tolled optimum needs 0.88, not the 0.97 that k = 8.5 demanded.
    assert np.isclose(arrival_fraction_for(6.75), 0.881, rtol=2e-3)


def test_arrival_fraction_reports_the_infeasible_case_rather_than_clipping() -> None:
    """Above 1.0 a rigid front cannot supply the ratio at all."""
    assert arrival_fraction_for(12.0) > 1.0


def test_rejects_a_nonsense_arrival_fraction() -> None:
    """Zero radius sweeps nothing; above the bore is not a bore."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            swept_slug_ratio(bad)
