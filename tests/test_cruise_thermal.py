"""Pinned tests for src/cruise_thermal.py (ledger item 14).

The sublimation term is exponential in temperature, so an error in the vapour
pressure correlation costs orders of magnitude rather than percent.  These pin
the correlation, the balance and the two published consequences.
"""

import astropy.units as u
import numpy as np

from src.cruise_thermal import (
    SHADED_TEMPERATURE,
    averaged_absorbed_flux,
    cruise_loss,
    equilibrium_temperature,
    rod_geometry,
    sublimation_flux,
    vapour_pressure,
)


def test_sphere_average_of_the_solar_constant() -> None:
    """Projected area over total area is exactly 1/4 for a convex body."""
    assert np.isclose(
        averaged_absorbed_flux().to_value(u.W / u.m**2), 340.25, rtol=1e-3
    )
    assert np.isclose(
        averaged_absorbed_flux(distance_au=5.2).to_value(u.W / u.m**2), 12.58, rtol=1e-2
    )


def test_marti_mauersberger_at_the_published_point() -> None:
    """0.064 Pa at 194 K is the paper's quoted vapour pressure."""
    assert np.isclose(vapour_pressure(194 * u.K).to_value(u.Pa), 0.0643, rtol=1e-2)


def test_equilibrium_lands_at_the_published_temperature() -> None:
    """194 K, and it is a balance rather than an assumption.

    Bare ice at 1 AU does not sit at the 278 K a rock would, because sublimation
    carries off latent heat far faster than radiation alone would shed it.
    """
    assert np.isclose(equilibrium_temperature().to_value(u.K), 194.0, atol=1.0)


def test_sublimation_dominates_radiation_at_the_balance() -> None:
    """Which is why the answer is 194 K and not the radiative 278 K."""
    temperature = equilibrium_temperature()
    from astropy import constants as const

    from src.cruise_thermal import SUBLIMATION_ENTHALPY

    radiated = (const.sigma_sb * temperature**4).to_value(u.W / u.m**2)
    carried = (SUBLIMATION_ENTHALPY * sublimation_flux(temperature)).to_value(
        u.W / u.m**2
    )
    assert carried > 2.0 * radiated


def test_rod_geometry_matches_the_published_length() -> None:
    """A 25 kg ice rod 0.1 m in radius is 0.87 m long."""
    length, area, areal = rod_geometry()
    assert np.isclose(length.to_value(u.m), 0.87, rtol=1e-2)
    assert np.isclose(area.to_value(u.m**2), 0.608, rtol=1e-2)
    # Total surface, caps included: a body in vacuum sublimates from everywhere.
    assert np.isclose(areal.to_value(u.kg / u.m**2), 41.1, rtol=1e-2)


def test_shaded_cruise_loss_reproduces_the_published_fraction() -> None:
    """0.58 kg/m^2 over two years, which is 1.4% of the rod.

    This is the load-bearing check on the areal density, because 1.4% only
    comes out if the rod's total surface is 0.608 m^2 -- the same area that
    makes the bare areal density 41.1 kg/m^2 rather than the 55 the paper
    prints two sentences earlier.
    """
    _, area, _ = rod_geometry()
    loss = cruise_loss()
    assert np.isclose(loss.to_value(u.kg / u.m**2), 0.58, rtol=2e-2)
    fraction = float(loss * area / (25.0 * u.kg))
    assert np.isclose(fraction, 0.014, rtol=5e-2)


def test_the_sunshade_is_what_makes_ice_viable() -> None:
    """Four orders of magnitude between 194 K and 150 K."""
    bare = sublimation_flux(equilibrium_temperature())
    shaded = sublimation_flux(SHADED_TEMPERATURE)
    assert float(bare / shaded) > 1e3
