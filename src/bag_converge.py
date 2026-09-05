"""Iterate the bag design to a fixed point, and report the gap as a number.

The paper's `sec:watering_it_down` runs a loop and cuts it in three places, on
purpose, so that every printed digit can be followed by hand:

    bag volume V -> slug density rho -> plume state (T, f) -> pressure
      -> field B -> stored energy E_B -> field leak -> waste heat
      -> vapour fraction x -> mist state -> film mass
                    ^                                    |
                    +--- V came from a radiative trade ---+
                         that depends on the T at the bottom

**Cut 1.** The plume was held at 15 000 K while the bag was sized; the solved
value runs 15 165-26 521 K.
**Cut 2.** The leak was held at 4.4% while ``E_B`` moved 4.43 -> 12.2 GJ.
**Cut 3.** The vapour fraction's effect on the ignition budget was held rather
than recomputed.
**Cut 4.** ``tab:bag_state``'s warming row warms liquid water up to the mist
temperature, so it is an output of the row four lines below it. The published
0.21 MJ/kg was closed against 328 K, which was the mist of the *superseded*
4.4% leak, and the table now prints 316 K instead. **The Jupiter column no
longer cuts here** -- ADR 0027 made its warming row the solve rather than a
report against it -- so this cut is now the Earth column alone.

Cutting a loop is the right way to write a paper -- a reader can check each link
-- and it is the wrong way to find out whether the design closes. This module
does the other half: it iterates the same chain to a fixed point and **reports
the gap against the published tables rather than replacing them**.

**The loop closes through radiation, and that is the whole reason it moves.**
Radiative loss goes as ``T^4 r^2``, so a plume at 26 521 K rather than 15 000 K
radiates almost ten times as hard. Holding the radiated share at the value the
bag was sized for therefore demands a smaller bag; a smaller bag is denser; a
denser plume is hotter still (Saha spends less of the budget on electrons); and
round it goes.

Run it with ``make bag-converge``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from astropy import constants as const
from astropy import units as u

from src.bag_state import (
    EXPANSION_TIME,
    LATENT_HEAT,
    RADIATED_FRACTION,
    SKY_FRACTION,
    SLUG_MASS,
    WARMING_TO_LIQUID,
    BagState,
    stored_field_energy,
)
from src.plume_state import FLOWN_BAG_DENSITY, plume_state
from src.plume_thermal import plume_ignition_energy

#: Bag radius the paper flies, and the radiated share it was sized at.
#:
#: **Left at 5.4 m by ADR 0029, deliberately.**  This module works the optical
#: depth as ``kappa rho r`` against :data:`~src.plume_state.FLOWN_BAG_DENSITY`,
#: which is the vendored solve's 0.323 kg/m^3, and 5.4 m is the radius that
#: holds 213 kg at that density.  The two are one pair from
#: ``puffsat_impact_simulation``, so moving one without the other would size the
#: bag at 217 kg.  The bag ``bag_state`` now adopts is the flown column's
#: 672.9 m^3, whose equal-volume sphere is 5.44 m at 0.3165 kg/m^3; ``tau``
#: differs by 0.7% between the two pairs, which does not reach any conclusion
#: here.
FLOWN_RADIUS = 5.4 * u.m
FLOWN_RADIATED_SHARE = 0.0117

#: Pessimistic absorption coefficient.  The plume stops being optically thick
#: once ``kappa rho r`` falls below one, which is the floor under the bag.
ABSORPTION_COEFFICIENT = 0.5 * u.m**2 / u.kg


def radiated_share(
    radius: u.Quantity, temperature: u.Quantity, ionised: float
) -> float:
    """Share of the pulse energy radiated, priced at the *solved* plume state.

    Differs from ``bag_state.radiated_fraction`` in both terms: the flux is
    taken at the solved temperature rather than 15 000 K, and the budget it is
    measured against includes the water's own ionisation, which at a hot pulse
    more than doubles it.  Both corrections matter and they partly cancel.

    Args:
        radius: Bag radius.
        temperature: Solved plume temperature.
        ionised: Solved ionisation fraction.

    Returns:
        Radiated share of the pulse energy.
    """
    area = 4.0 * np.pi * radius**2
    radiated = const.sigma_sb * temperature**4 * area * EXPANSION_TIME
    budget = plume_ignition_energy(temperature, ionised) * SLUG_MASS
    return float(radiated / budget)


def optical_depth(radius: u.Quantity) -> float:
    """``kappa rho D`` across the bag's full chord; below 1 it radiates from bulk.

    **Across the diameter, not the radius**, which is what reproduces the
    paper's stated limit: it says the plume "stops being optically thick past
    about 7 m", and ``2 kappa m / ((4/3) pi r^2) = 1`` gives exactly 7.13 m.
    Using the radius would put the limit at 5.04 m and make the flown bag
    already thin.

    Args:
        radius: Bag radius.

    Returns:
        Optical depth across the diameter.
    """
    density = SLUG_MASS / ((4.0 / 3.0) * np.pi * radius**3)
    return float(ABSORPTION_COEFFICIENT * density * 2.0 * radius)


def optically_thick_limit() -> u.Quantity:
    """Largest bag whose plume is still optically thick, ``tau = 1``.

    Returns:
        Bag radius (astropy Quantity, m).
    """
    coefficient = ABSORPTION_COEFFICIENT.to_value(u.m**2 / u.kg)
    mass = SLUG_MASS.to_value(u.kg)
    return float(np.sqrt(2.0 * coefficient * mass / ((4.0 / 3.0) * np.pi))) * u.m


def _radius_holding_radiated_share(
    temperature: u.Quantity, ionised: float, share: float
) -> u.Quantity:
    """Radius at which the solved plume radiates exactly ``share`` of the pulse.

    Closed form, since the share is linear in ``r^2`` at fixed plume state.

    Args:
        temperature: Solved plume temperature.
        ionised: Solved ionisation fraction.
        share: Radiated share to hold.

    Returns:
        Bag radius.
    """
    unit = radiated_share(1.0 * u.m, temperature, ionised)
    return float(np.sqrt(share / unit)) * u.m


def converge(
    closing_speed: float,
    leak_fraction: float,
    share: float = FLOWN_RADIATED_SHARE,
    iterations: int = 60,
) -> Tuple[u.Quantity, u.Quantity, float, BagState, List[float]]:
    """Iterate radius, plume state and bag state to a fixed point.

    Args:
        closing_speed: Impactor speed relative to the vehicle (km/s).
        leak_fraction: Residence-weighted field leak on this leg.
        share: Radiated share the bag is sized to hold.
        iterations: Fixed-point iterations; the map is a contraction and
            converges in well under ten.

    Returns:
        ``(radius, temperature, ionised, bag_state, radius_history)``.
    """
    radius = FLOWN_RADIUS
    history: List[float] = [radius.to_value(u.m)]
    temperature, ionised = plume_state(closing_speed, FLOWN_BAG_DENSITY)
    for _ in range(iterations):
        density = float(
            (SLUG_MASS / ((4.0 / 3.0) * np.pi * radius**3)).to_value(u.kg / u.m**3)
        )
        temperature, ionised = plume_state(closing_speed, min(max(density, 0.05), 2.0))
        radius = _radius_holding_radiated_share(temperature, ionised, share)
        history.append(radius.to_value(u.m))
    volume = (4.0 / 3.0) * np.pi * radius**3
    state = BagState(
        leak_fraction,
        stored_field_energy(temperature, ionised),
        "jupiter",
        bag_volume=volume,
    )
    return radius, temperature, ionised, state, history


def main() -> None:
    """Report the fixed point and the gap against the published tables."""
    from src.bag_state import (
        SOLVED_LEAK_FRACTIONS,
        melting_fixed_point,
        paper_bag_state,
        radiated_fraction,
    )
    from src.plume_state import QUOTED_SPEEDS

    print("=== The loop, closed ===")
    print(
        f"optically thick out to {optically_thick_limit().to_value(u.m):.2f} m "
        f"(the paper says about 7 m); the flown 5.4 m bag sits at "
        f"tau = {optical_depth(FLOWN_RADIUS):.2f}"
    )

    print(
        "\n--- Cut 1: the bag was sized at 15 000 K. What does it really radiate? ---"
    )
    print(
        f"{'w [km/s]':>9}{'T solved':>11}{'tabulated':>12}{'at solved T':>14}{'ratio':>8}"
    )
    for speed in QUOTED_SPEEDS:
        temperature, ionised = plume_state(speed, FLOWN_BAG_DENSITY)
        tabulated = radiated_fraction(FLOWN_RADIUS)
        actual = radiated_share(FLOWN_RADIUS, temperature, ionised)
        print(
            f"{speed:>9.2f}{temperature.to_value(u.K):>10.0f}K"
            f"{100 * tabulated:>11.2f}%{100 * actual:>13.2f}%{actual / tabulated:>7.1f}x"
        )

    print("\n--- The fixed point: what bag would each leg choose for itself? ---")
    print(
        f"{'w [km/s]':>9}{'radius':>10}{'vs flown':>10}{'T [K]':>10}"
        f"{'tau':>8}{'film':>10}"
    )
    radii = []
    for speed in QUOTED_SPEEDS:
        radius, temperature, _ionised, state, _history = converge(
            speed, SOLVED_LEAK_FRACTIONS["equilibrium"][speed]
        )
        radii.append(radius.to_value(u.m))
        print(
            f"{speed:>9.2f}{radius.to_value(u.m):>8.2f} m"
            f"{radius / FLOWN_RADIUS:>9.2f}x{temperature.to_value(u.K):>10.0f}"
            f"{optical_depth(radius):>8.2f}{state.film_mass.to_value(u.kg):>9.2f}kg"
        )
    print(
        f"\nHolding the radiated share fixed, the legs disagree by "
        f"{max(radii) / min(radii):.1f}x in radius. **That objective is the wrong\n"
        "one**, and the spread is an artefact of it: it drives the hot leg to a 2 m\n"
        "bag needing 32 T, well past the ~20 T commercial working point. Held at\n"
        "constant *field* instead the spread is 1.3x and runs the other way -- a\n"
        "hotter pulse wants a BIGGER bag, because thinning the slug drops pressure\n"
        "faster than the temperature rises. See bag_for_fixed_field."
    )
    print(
        "\nSize of the gap, stated fairly: cut 1 costs a factor of 3 in a term that\n"
        "is small either way -- 3.6% of the pulse radiated rather than 1.2%. It\n"
        "does not move the bag out of the optically thick band, and it does not\n"
        "reach the film, which cold storage holds under the handling floor\n"
        "at two thirds of a kilogram (item 10, ADR 0027)."
    )

    print("\n--- Cut 4: the warming row is an output of the mist row ---")
    print(f"{'case':>24}{'warming':>10}{'x':>9}{'mist':>9}{'film':>9}")
    for storage in ("jupiter", "earth"):
        published = paper_bag_state(storage)
        closed = melting_fixed_point(published.waste_heat, storage)
        for label, warming, fraction, temperature, film in (
            (
                "printed",
                published.warming,
                published.vapour_fraction,
                published.mist_temperature,
                published.film_mass,
            ),
            (
                "closed",
                closed.warming,
                closed.vapour_fraction,
                closed.mist_temperature,
                closed.film_mass,
            ),
        ):
            print(
                f"{storage + ', ' + label:>24}"
                f"{warming.to_value(u.MJ / u.kg):>10.3f}{fraction:>9.4f}"
                f"{temperature.to_value(u.K):>8.1f}K{film.to_value(u.kg):>8.2f}kg"
            )
    print(
        "Size of this gap: half a kilogram of film on the Earth leg, and a\n"
        "Jupiter column that ends as slush at the freezing point rather than\n"
        "dry. Neither overturns 'cold storage removes the pressure vessel',\n"
        "which is the conclusion the section rests on, so this is reported\n"
        "rather than applied -- the paper prints the warming row as an input."
    )


if __name__ == "__main__":
    main()


def _plume_field(closing_speed: float, radius: u.Quantity) -> u.Quantity:
    """Confinement field a bag of this radius needs at this closing speed.

    Args:
        closing_speed: Impactor speed relative to the vehicle (km/s).
        radius: Bag radius.

    Returns:
        Field strength (astropy Quantity, T).
    """
    from src.bag_state import DISSOCIATED_MOLAR_MASS

    density = SLUG_MASS / ((4.0 / 3.0) * np.pi * radius**3)
    temperature, ionised = plume_state(
        closing_speed,
        min(max(float(density.to_value(u.kg / u.m**3)), 0.05), 2.0),
    )
    molar = DISSOCIATED_MOLAR_MASS / (1.0 + ionised)
    return np.sqrt(2.0 * const.mu0 * density * const.R * temperature / molar).to(u.T)


def bag_for_fixed_field(
    closing_speed: float, field: u.Quantity = 4.11 * u.T
) -> u.Quantity:
    """Bag radius that holds the confinement field at a chosen value.

    **The counterintuitive direction, and it is worth stating.**  A *hotter*
    pulse needs a *bigger* bag to hold the same field, because thinning the slug
    drops the pressure faster than the temperature rises.  So per-speed bag
    sizing grows the bag on the hot legs, not shrinks it.

    Args:
        closing_speed: Impactor speed relative to the vehicle (km/s).
        field: Field to hold.

    Returns:
        Bag radius (astropy Quantity, m).
    """
    from scipy.optimize import brentq

    target = field.to_value(u.T)
    return (
        float(
            brentq(
                lambda r: _plume_field(closing_speed, r * u.m).to_value(u.T) - target,
                2.0,
                12.0,
            )
        )
        * u.m
    )
