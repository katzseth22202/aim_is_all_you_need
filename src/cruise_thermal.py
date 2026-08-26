"""Whether an ice projectile survives the cruise to its target.

Ice is the right material for the projectile on chemistry grounds: making it out
of the same substance as the slug leaves the merged blob a single material, so
the ignition budget describes all 238 kg rather than 213 kg plus a contaminant.
Polyethylene would break the recombination loan outright -- carbon scavenges
oxygen into carbon monoxide at 11.1 eV, the strongest bond in diatomic
chemistry, and that oxygen does not come back.

**Ice's weakness is the cruise, not the impact.**  A bare ice body near 1 AU
does not sit at the 278 K a rock would.  It cools until the latent heat carried
off by sublimation, plus what it radiates, balances the sunlight it absorbs:

    absorbed  =  sigma T^4  +  L * sublimation_flux(T)

That balance is a root-find, and the sublimation term is exponential in ``T``,
so the two sides cross sharply -- which is why the answer is a temperature
rather than a range, and why getting the vapour-pressure correlation wrong costs
orders of magnitude rather than percent.

See ``sec:needle_through_fog``, the ice-versus-polyethylene paragraph.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from astropy import constants as const
from astropy import units as u

from src.plume_thermal import WATER_MOLAR_MASS

#: Solar irradiance at 1 AU.
SOLAR_CONSTANT = 1361.0 * u.W / u.m**2
#: Latent heat of sublimation of water ice.
SUBLIMATION_ENTHALPY = 2.83 * u.MJ / u.kg
#: Density of solid water ice.
ICE_DENSITY = 917.0 * u.kg / u.m**3
#: Temperature a shaded projectile sits at behind the detachable multilayer
#: sunshade of ``sec:lox_puffsat``.
SHADED_TEMPERATURE = 150.0 * u.K
#: Cruise the projectile has to survive.
CRUISE_YEARS = 2.0


def averaged_absorbed_flux(
    distance_au: float = 1.0, absorptivity: float = 1.0
) -> u.Quantity:
    """Sunlight absorbed per unit surface area, averaged over a convex body.

    A convex body intercepts sunlight on its projected area and radiates from
    its whole surface, and for a sphere the ratio is exactly 1/4.  At 1 AU that
    turns 1361 W/m^2 into 340 W/m^2, which is the figure the paper balances
    against.

    Args:
        distance_au: Heliocentric distance in astronomical units.
        absorptivity: Fraction of incident sunlight absorbed.

    Returns:
        Averaged absorbed flux (astropy Quantity, W/m^2).
    """
    return absorptivity * SOLAR_CONSTANT / (4.0 * distance_au**2)


def vapour_pressure(temperature: u.Quantity) -> u.Quantity:
    """Sublimation vapour pressure of water ice, Marti and Mauersberger (1993).

    ``log10(P/Pa) = -2663.5/T + 12.537``, fitted over 170-250 K, which brackets
    both the bare equilibrium and the shaded case.

    Args:
        temperature: Ice temperature.

    Returns:
        Vapour pressure (astropy Quantity, Pa).
    """
    kelvin = temperature.to_value(u.K)
    return (10.0 ** (-2663.5 / kelvin + 12.537)) * u.Pa


def sublimation_flux(temperature: u.Quantity) -> u.Quantity:
    """Free-evaporation mass loss per unit area, Hertz-Knudsen.

    ``Phi = P sqrt(M / (2 pi R T))``, the kinetic-theory flux of molecules
    leaving a surface into vacuum with nothing returning.  An upper bound on the
    real rate, since it assumes unit sticking on the way out and no re-impact.

    Args:
        temperature: Ice temperature.

    Returns:
        Mass flux (astropy Quantity, kg/m^2/s).
    """
    return (
        vapour_pressure(temperature)
        * np.sqrt(WATER_MOLAR_MASS / (2.0 * np.pi * const.R * temperature))
    ).to(u.kg / u.m**2 / u.s)


def equilibrium_temperature(
    absorbed: u.Quantity | None = None,
    bounds: Tuple[float, float] = (80.0, 320.0),
) -> u.Quantity:
    """Temperature at which radiation plus sublimation balance the sunlight.

    Bisection, because the sublimation term is exponential in ``T`` and swings
    by orders of magnitude across the bracket while the radiative term moves by
    a factor of a few.  The sum is strictly increasing, which is all bisection
    needs.

    Args:
        absorbed: Absorbed flux; defaults to the 1 AU sphere average.
        bounds: Temperature bracket to search, in kelvin.

    Returns:
        Equilibrium temperature (astropy Quantity, K).
    """
    target = (averaged_absorbed_flux() if absorbed is None else absorbed).to_value(
        u.W / u.m**2
    )
    sigma = float(const.sigma_sb.to_value(u.W / u.m**2 / u.K**4))
    latent = float(SUBLIMATION_ENTHALPY.to_value(u.J / u.kg))

    def shed(kelvin: float) -> float:
        flux = float(sublimation_flux(kelvin * u.K).to_value(u.kg / u.m**2 / u.s))
        return sigma * kelvin**4 + latent * flux

    low, high = bounds
    for _ in range(200):
        mid = 0.5 * (low + high)
        if shed(mid) < target:
            low = mid
        else:
            high = mid
    return (0.5 * (low + high)) * u.K


def rod_geometry(
    mass: u.Quantity = 25.0 * u.kg, radius: u.Quantity = 0.1 * u.m
) -> Tuple[u.Quantity, u.Quantity, u.Quantity]:
    """Length, total surface area and areal density of a solid ice rod.

    **Total** surface, caps included, because a body in vacuum sublimates from
    everywhere rather than from its side alone.

    Args:
        mass: Rod mass.
        radius: Rod radius.

    Returns:
        ``(length, surface_area, areal_density)``.
    """
    length = (mass / (ICE_DENSITY * np.pi * radius**2)).to(u.m)
    area = (2.0 * np.pi * radius * length + 2.0 * np.pi * radius**2).to(u.m**2)
    return length, area, (mass / area).to(u.kg / u.m**2)


def cruise_loss(
    temperature: u.Quantity = SHADED_TEMPERATURE, years: float = CRUISE_YEARS
) -> u.Quantity:
    """Ice lost per unit area over a cruise at a held temperature.

    Args:
        temperature: Temperature the projectile is held at.
        years: Length of the cruise in Julian years.

    Returns:
        Areal mass loss (astropy Quantity, kg/m^2).
    """
    return (sublimation_flux(temperature) * (years * u.yr)).to(u.kg / u.m**2)


def main() -> None:
    """Print the cruise-survival balance and the shaded case."""
    temperature = equilibrium_temperature()
    daily = (sublimation_flux(temperature) * u.day).to(u.kg / u.m**2)
    print("=== Item 14: does a bare ice projectile survive the cruise? ===")
    print(
        f"{'absorbed at 1 AU (sphere-averaged)':<40}"
        f"{averaged_absorbed_flux().to_value(u.W / u.m**2):>10.1f} W/m^2"
    )
    print(f"{'equilibrium temperature':<40}{temperature.to_value(u.K):>10.1f} K")
    print(
        f"{'vapour pressure there':<40}"
        f"{vapour_pressure(temperature).to_value(u.Pa):>10.4f} Pa"
    )
    print(
        f"{'free-evaporation rate':<40}"
        f"{daily.to_value(u.kg / u.m**2):>10.2f} kg/m^2/day"
    )

    length, area, areal = rod_geometry()
    print(
        f"\n25 kg ice rod at r = 0.1 m: {length.to_value(u.m):.3f} m long, "
        f"{area.to_value(u.m**2):.4f} m^2 of surface"
    )
    print(f"{'  areal density':<40}{areal.to_value(u.kg / u.m**2):>10.1f} kg/m^2")
    print(f"{'  bare survival':<40}" f"{float(areal / daily):>10.1f} days")

    loss = cruise_loss()
    print(
        f"\nBehind the sunshade at {SHADED_TEMPERATURE.to_value(u.K):.0f} K, over "
        f"{CRUISE_YEARS:.0f} years:"
    )
    print(f"{'  areal loss':<40}{loss.to_value(u.kg / u.m**2):>10.3f} kg/m^2")
    print(
        f"{'  fraction of the rod':<40}"
        f"{100 * float(loss * area / (25.0 * u.kg)):>10.2f} %"
    )
    print(
        "\nThe sunshade is not optional: bare, the projectile is gone in under a\n"
        "week; shaded, it loses 1.4% over a two-year cruise."
    )


if __name__ == "__main__":
    main()
