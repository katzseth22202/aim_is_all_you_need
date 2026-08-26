"""What the slug bag weighs, and the field leak that decides it.

The bag holds 213 kg of slug spread over 659.6 m^3 so the plume has something
to be a plume *in*.  It is not free: waste heat boils part of the slug, the
vapour has a saturation pressure, and the film has to hold that pressure.  The
whole chain is six dependent steps and the paper prints it as
``tab:bag_state``:

    blackbody intercept + field leak  ->  waste heat
      - warming and melting the slug  ->  what is left to boil
      / latent heat                   ->  vapour fraction x
      -> saturation curve             ->  mist temperature and pressure
      -> eq:bag_film_mass             ->  film mass

**The field leak is the whole story and it is the one term nobody had
computed.**  The paper holds it at 4.4% of the 20.8 MJ/kg of stored field
energy, which was fine while ``E_B`` was 4.43 GJ.  Item 1 moved ``E_B`` to
12.2 GJ at the hottest pulse without moving the leak fraction, and at 4.4% of
*that* the film goes from 2.8 kg to 31 kg -- from a rounding error to 15% of
the slug, every pulse.  That is the bracket this module exists to close.

**It is closed, and in the design's favour.**  The leak fraction is
``~1/Rm`` -- the field diffuses out in ``mu0 sigma L^2`` against an expansion
clock of ``L/v``, and the ratio *is* the magnetic Reynolds number, which
``tab:seed_window`` has been tabulating two pages earlier.  So the calculation
is a quadrature, not a simulation: weight ``1/Rm(T)`` by how long the plume
spends at each ``T``.  ``puffsat_impact_simulation`` ran it on its solved
cooling history (Q-L, ``make analysis-expansion``) and gets **0.11% to 2.54%**
on the equilibrium branch -- and Q-M then established that equilibrium is the
branch that holds, by two to five decades.  See :data:`SOLVED_LEAK_FRACTIONS`.

See CONTEXT.md and the ledger's items 5-10.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from astropy import constants as const
from astropy import units as u

from src.plume_thermal import (
    NOZZLE_FLOOR_TEMPERATURE,
    WATER_MOLAR_MASS,
    plume_ignition_energy,
)

#: Slug mass the bag holds.
SLUG_MASS = 213.0 * u.kg
#: Enclosed bag volume, the standoff volume of ``tab:bag_sizing``.
BAG_VOLUME = 659.6 * u.m**3
#: Free particles per water molecule once dissociated (2 H + O).
ATOMS_PER_MOLECULE = 3

#: Share of the plume's energy that escapes as light, and the share of the sky
#: the bag subtends to it.  Their product times the ignition bill is the
#: blackbody row of ``tab:bag_state``.
RADIATED_FRACTION = 0.012
SKY_FRACTION = 0.1

#: Sensible-plus-latent heat to bring the slug from storage up to liquid.
#: Jupiter parks it at 122 K (0.26 warming ice + 0.33 melting + 0.14 warming
#: liquid); Earth at 278 K needs only the last part of that.
WARMING_TO_LIQUID: Dict[str, u.Quantity] = {
    "jupiter": 0.73 * u.MJ / u.kg,
    "earth": 0.21 * u.MJ / u.kg,
}
#: Temperature the slug is parked at before the burn, per location.
STORAGE_TEMPERATURE: Dict[str, u.Quantity] = {
    "jupiter": 122.0 * u.K,
    "earth": 278.0 * u.K,
}

#: Latent heat of vaporisation used by the paper's own table.  **Back-solved
#: from it, not quoted by it**: both columns give ``left-over / x`` = 2.36
#: (0.26/0.11 and 0.78/0.33), which is the value near 300 K rather than the
#: 2.26 at the boiling point.  Named here because it is the one input to the
#: cascade that has to be inferred.
LATENT_HEAT = 2.36 * u.MJ / u.kg

#: Film mass coefficient of ``eq:bag_film_mass``, ``rho_f R_g / (M sigma)``,
#: for polyethylene fibre at half of Toyobo's quoted 3.5 GPa to allow for weave
#: and seams.  Every factor of bag radius cancels out of the derivation, which
#: is why a bigger bag is not a heavier one.
FILM_COEFFICIENT = 2.6e-4 / u.K
#: Shape factor ``F = (2L + 2r)/(L + 4r/3)``: 1.5 for a sphere, 2.0 for a long
#: tube.
SPHERE_SHAPE_FACTOR = 1.5

#: Residence-weighted leak fractions from ``puffsat_impact_simulation``'s
#: solved cooling history, keyed by closing speed in km/s (Q-L, 2026-08-22,
#: ``make analysis-expansion``).  The equilibrium branch is the one that holds
#: (Q-M, by two to five decades); the frozen branch is carried because it is
#: what the paper's own 4.4% is closest to.
SOLVED_LEAK_FRACTIONS: Dict[str, Dict[float, float]] = {
    "equilibrium": {75.0: 0.0011, 65.0: 0.0017, 56.53: 0.0029, 45.58: 0.0254},
    "frozen": {75.0: 0.0121, 65.0: 0.0180, 56.53: 0.0258, 45.58: 0.0538},
}
#: The leak fraction the paper prints.  Its own table is internally
#: inconsistent by 3%: the row reads "4.4% of 20.8 MJ/kg" but prints 0.89
#: MJ/kg, and 0.89/20.8 is 4.28%.  Reproducing the printed table needs the
#: printed 0.89, so that is what :func:`paper_bag_state` uses.
PAPER_LEAK_FRACTION = 0.044


def stored_field_energy(
    temperature: u.Quantity = NOZZLE_FLOOR_TEMPERATURE,
    ionisation_fraction: float = 0.0,
    slug_mass: u.Quantity = SLUG_MASS,
) -> u.Quantity:
    """Magnetic energy the bag confines, ``E_B = B^2 V / 2 mu0 = P V = n R_g T``.

    **The bag radius does not appear**, which is the point: ``E_B`` is invariant
    across ``tab:bag_sizing``'s whole 15x range in radius, because a bigger bag
    trades field strength against volume exactly. So one thing a bigger bag does
    not buy is a lighter nozzle -- the nozzle is sized by the energy it must
    contain, and that is fixed by the slug and the temperature alone.

    Args:
        temperature: Plume temperature.
        ionisation_fraction: ``f``; every atom that loses an electron becomes
            two particles, and pressure counts particles rather than mass, so
            the count is ``3(1+f)`` per molecule.
        slug_mass: Mass of slug in the bag.

    Returns:
        Stored field energy (astropy Quantity, J).
    """
    moles = (slug_mass / WATER_MOLAR_MASS) * ATOMS_PER_MOLECULE
    particles = moles * (1.0 + ionisation_fraction)
    return (particles * const.R * temperature).to(u.J)


def saturation_pressure(temperature: u.Quantity) -> u.Quantity:
    """Saturation vapour pressure of water, by the Magnus-Tetens correlation.

    ``P = 0.61094 exp(17.625 t / (t + 243.04))`` kPa with ``t`` in Celsius.
    Chosen over a steam-table interpolation because it reproduces both of the
    paper's printed points (4.98 kPa against 4.9 at 306 K, 15.7 against 16 at
    328 K) and is valid across the whole 270-340 K range the mist occupies.

    Args:
        temperature: Vapour temperature.

    Returns:
        Saturation pressure (astropy Quantity, Pa).
    """
    celsius = temperature.to_value(u.K) - 273.15
    kilopascals = 0.61094 * np.exp(17.625 * celsius / (celsius + 243.04))
    return (kilopascals * u.kPa).to(u.Pa)


def saturation_density(temperature: u.Quantity) -> u.Quantity:
    """Density of saturated water vapour at this temperature.

    Args:
        temperature: Vapour temperature.

    Returns:
        Vapour density (astropy Quantity, kg/m^3).
    """
    pressure = saturation_pressure(temperature)
    return (pressure * WATER_MOLAR_MASS / (const.R * temperature)).to(u.kg / u.m**3)


def saturation_temperature(
    vapour_density: u.Quantity,
    bounds: Tuple[float, float] = (200.0, 600.0),
) -> u.Quantity:
    """Invert the saturation curve: the ``T`` at which vapour is this dense.

    Bisection rather than a closed form, because Magnus-Tetens is not
    analytically invertible once the ideal-gas density conversion is folded in.
    Saturation density is strictly increasing in temperature, which is all
    bisection needs.

    Args:
        vapour_density: Mass of vapour per unit bag volume.
        bounds: Temperature bracket to search, in kelvin.  The ceiling is
            below water's 647 K critical point, past which saturation is not a
            meaningful concept and a caller has left this model's domain.

    Returns:
        Mist temperature (astropy Quantity, K).

    Raises:
        ValueError: If the density lies outside the bracket.
    """
    low, high = bounds
    target = vapour_density.to_value(u.kg / u.m**3)
    if not (
        saturation_density(low * u.K).to_value(u.kg / u.m**3)
        <= target
        <= saturation_density(high * u.K).to_value(u.kg / u.m**3)
    ):
        raise ValueError(f"vapour density {vapour_density} is outside {bounds} K")
    for _ in range(200):
        mid = 0.5 * (low + high)
        if saturation_density(mid * u.K).to_value(u.kg / u.m**3) < target:
            low = mid
        else:
            high = mid
    return (0.5 * (low + high)) * u.K


def film_mass_fraction(
    vapour_fraction: float,
    temperature: u.Quantity,
    shape_factor: float = SPHERE_SHAPE_FACTOR,
) -> float:
    """Bag film mass as a fraction of the slug, ``eq:bag_film_mass``.

    ``m_bag/m_slug = F x rho_f R_g T / (M sigma)``.  Linear in both the vapour
    fraction and its temperature, and independent of bag radius.

    Args:
        vapour_fraction: ``x``, the share of the slug carried as vapour.
        temperature: Vapour temperature.
        shape_factor: ``F``; 1.5 sphere, up to 2.0 for a long tube.

    Returns:
        Film mass per unit slug mass.
    """
    return float(
        FILM_COEFFICIENT * shape_factor * vapour_fraction * temperature.to(u.K)
    )


class BagState:
    """One column of ``tab:bag_state``: the waste-heat cascade and its bag.

    Attributes:
        leak_fraction: Share of the stored field energy that diffuses out.
        intercept: Blackbody light absorbed, per kg of slug.
        leak: Field energy leaked, per kg of slug.
        waste_heat: Their sum, which the slug must absorb.
        warming: Heat spent bringing the slug up to liquid, per kg.
        boiling: What is left to boil, per kg.
        vapour_fraction: ``x``, the share of the slug carried as vapour.
        mist_temperature: Saturation temperature at that vapour density.
        mist_pressure: Saturation pressure there.
        film_fraction: Bag film mass per unit slug mass.
        film_mass: Bag film mass.
    """

    def __init__(
        self,
        leak_fraction: float,
        stored_energy: u.Quantity,
        storage: str = "jupiter",
        shape_factor: float = SPHERE_SHAPE_FACTOR,
        slug_mass: u.Quantity = SLUG_MASS,
        bag_volume: u.Quantity = BAG_VOLUME,
        leak_energy: u.Quantity | None = None,
    ) -> None:
        """Run the cascade.

        Args:
            leak_fraction: Share of ``stored_energy`` that leaks.
            stored_energy: ``E_B`` the bag confines.
            storage: ``"jupiter"`` (122 K) or ``"earth"`` (278 K).
            shape_factor: ``F`` of ``eq:bag_film_mass``.
            slug_mass: Slug in the bag.
            bag_volume: Enclosed volume, which sets the vapour density.
            leak_energy: Override the leak term with an explicit energy per kg,
                used only to reproduce the paper's printed 0.89 MJ/kg.

        Raises:
            KeyError: If ``storage`` is not a known location.
        """
        self.leak_fraction = leak_fraction
        self.intercept = (
            RADIATED_FRACTION
            * SKY_FRACTION
            * plume_ignition_energy(NOZZLE_FLOOR_TEMPERATURE)
        ).to(u.MJ / u.kg)
        self.leak = (
            (leak_fraction * stored_energy / slug_mass).to(u.MJ / u.kg)
            if leak_energy is None
            else leak_energy.to(u.MJ / u.kg)
        )
        self.waste_heat = self.intercept + self.leak
        self.warming = WARMING_TO_LIQUID[storage]
        storage_temperature = STORAGE_TEMPERATURE[storage]
        self.boiling = self.waste_heat - self.warming
        self.vapour_fraction = max(0.0, float(self.boiling / LATENT_HEAT))
        if self.vapour_fraction <= 0.0:
            # The waste heat does not even finish melting the slug, so there is
            # no vapour, no saturation pressure, and nothing for the film to
            # hold.  This is a real operating point rather than a degenerate
            # one -- it is where every leg lands at the solved leak from cold
            # storage -- and the bag stops being a pressure vessel and becomes
            # a containment membrane, sized by handling rather than by stress.
            self.mist_temperature = storage_temperature
            self.mist_pressure = 0.0 * u.Pa
            self.film_fraction = 0.0
            self.film_mass = 0.0 * u.kg
            return
        vapour_density = self.vapour_fraction * slug_mass / bag_volume
        self.mist_temperature = saturation_temperature(vapour_density)
        self.mist_pressure = saturation_pressure(self.mist_temperature)
        self.film_fraction = film_mass_fraction(
            self.vapour_fraction, self.mist_temperature, shape_factor
        )
        self.film_mass = (self.film_fraction * slug_mass).to(u.kg)


def paper_bag_state(storage: str = "jupiter") -> BagState:
    """``tab:bag_state`` exactly as printed, for regression.

    Uses the table's printed 0.89 MJ/kg leak rather than ``0.044 * 20.8``,
    which is 0.915: the paper's own row is internally inconsistent by 3% and
    the printed digits follow the smaller number.

    Args:
        storage: ``"jupiter"`` or ``"earth"``.

    Returns:
        The cascade as the paper prints it.
    """
    return BagState(
        leak_fraction=PAPER_LEAK_FRACTION,
        stored_energy=stored_field_energy(),
        storage=storage,
        leak_energy=0.89 * u.MJ / u.kg,
    )


#: Plume state per closing speed, from item 1: ``(T, ionisation fraction)``.
#: Each leg confines its own ``E_B`` and pays its own leak, and the two run in
#: opposite directions with closing speed -- which is why the bracket that held
#: one fixed while moving the other was never a real operating point.
LEG_PLUME_STATES: Dict[float, Tuple[float, float]] = {
    75.0: (26200.0, 0.573),
    65.0: (22400.0, 0.371),
    56.53: (19400.0, 0.217),
    45.58: (14700.0, 0.053),
}


def boil_onset_leak(stored_energy: u.Quantity, storage: str = "jupiter") -> float:
    """Leak fraction at which the bag first has to hold pressure.

    Closed form: the slug must absorb ``warming`` before any of it boils, and
    the blackbody intercept contributes to that regardless, so boiling starts
    at ``(warming - intercept) m / E_B``.

    Args:
        stored_energy: ``E_B`` the bag confines.
        storage: ``"jupiter"`` or ``"earth"``.

    Returns:
        The leak fraction at onset.
    """
    intercept = (
        RADIATED_FRACTION
        * SKY_FRACTION
        * plume_ignition_energy(NOZZLE_FLOOR_TEMPERATURE)
    )
    deficit = WARMING_TO_LIQUID[storage] - intercept
    return float(deficit * SLUG_MASS / stored_energy)


def main() -> None:
    """Print the bag-state reproduction and the closed leak bracket."""
    print("=== tab:bag_state, as printed ===")
    print(f"{'row':<32}{'Jupiter 122 K':>15}{'Earth 278 K':>15}")
    jupiter, earth = paper_bag_state("jupiter"), paper_bag_state("earth")
    for label, getter in (
        ("waste heat [MJ/kg]", lambda s: s.waste_heat.to_value(u.MJ / u.kg)),
        ("left to boil [MJ/kg]", lambda s: s.boiling.to_value(u.MJ / u.kg)),
        ("vapour fraction x", lambda s: s.vapour_fraction),
        ("mist temperature [K]", lambda s: s.mist_temperature.to_value(u.K)),
        ("mist pressure [kPa]", lambda s: s.mist_pressure.to_value(u.kPa)),
        ("bag film [kg]", lambda s: s.film_mass.to_value(u.kg)),
    ):
        print(f"{label:<32}{getter(jupiter):>15.4g}{getter(earth):>15.4g}")

    print("\n=== Item 10: the leak bracket, closed ===")
    print(
        f"{'w [km/s]':>9}{'E_B [GJ]':>10}{'leak':>9}{'onset':>9}{'margin':>9}"
        f"{'film Jup':>10}{'film Earth':>12}{'film at 4.4%':>14}"
    )
    for speed, (temp, ionised) in LEG_PLUME_STATES.items():
        energy = stored_field_energy(temp * u.K, ionised)
        leak = SOLVED_LEAK_FRACTIONS["equilibrium"][speed]
        onset = boil_onset_leak(energy)
        print(
            f"{speed:>9.2f}{energy.to_value(u.GJ):>10.3g}{100 * leak:>8.2f}%"
            f"{100 * onset:>8.2f}%{onset / leak:>8.1f}x"
            f"{BagState(leak, energy, 'jupiter').film_mass.to_value(u.kg):>10.2f}"
            f"{BagState(leak, energy, 'earth').film_mass.to_value(u.kg):>12.2f}"
            f"{BagState(PAPER_LEAK_FRACTION, energy).film_mass.to_value(u.kg):>14.2f}"
        )
    print(
        "\nLeak fractions are residence-weighted 1/Rm on the equilibrium branch,\n"
        "from puffsat_impact_simulation's solved cooling history (Q-L, Q-M)."
    )

    print("\n=== tab:bag_sizing (item 4) ===")
    print(
        f"{'radius':>9}{'density':>12}{'B cold':>9}{'B hot':>9}"
        f"{'radiated':>11}{'mist':>9}"
    )
    for radius in BAG_SIZING_RADII:
        metres = radius * u.m
        print(
            f"{radius:>8.1f}m{bag_density_at(metres).to_value(u.kg / u.m**3):>12.4g}"
            f"{confinement_field_at(metres).to_value(u.T):>8.2f}T"
            f"{confinement_field_at(metres, *HOT_PULSE_STATE).to_value(u.T):>8.2f}T"
            f"{100 * radiated_fraction(metres):>10.2f}%"
            f"{coldest_mist_temperature(metres).to_value(u.K):>8.0f}K"
        )
    print("The usable band is 3.5-7 m: radiative loss binds above, opacity below.")

    print("\n=== tab:axial_bag (item 9) ===")
    print(f"{'length':>9}{'bore':>9}{'conductor':>12}{'F':>7}{'film':>9}")
    reference = SPHERE_BORE_RADIUS.to_value(u.m) * 10.8
    for length in AXIAL_BAG_LENGTHS:
        metres = length * u.m
        radius = (
            SPHERE_BORE_RADIUS if length == 10.8 else bore_radius(metres)
        ).to_value(u.m)
        factor = shape_factor(metres)
        print(
            f"{length:>8.1f}m{radius:>8.2f}m{radius * length / reference:>12.2f}"
            f"{factor:>7.2f}{2.8 * factor / 1.5:>8.1f}kg"
        )
    print("Bore and conductor trade one for one; the launch envelope picks 23 m.")

    print("\n=== D4: the handling floor under the film ===")
    thin, thick = HANDLING_GAUGE_BAND
    coldest = 45.58
    temperature, ionised = LEG_PLUME_STATES[coldest]
    earth_film = BagState(
        SOLVED_LEAK_FRACTIONS["equilibrium"][coldest],
        stored_field_energy(temperature * u.K, ionised),
        "earth",
    ).film_mass
    print(
        f"{'length':>9}{'area':>10}{'6 um':>9}{'12.7 um':>10}{'25 um':>9}"
        f"{'pressure':>11}{'governs':>10}"
    )
    for length in AXIAL_BAG_LENGTHS:
        metres = length * u.m
        factor = shape_factor(metres)
        pressure = earth_film * factor / SPHERE_SHAPE_FACTOR
        governing = governing_film_mass(pressure, metres)
        governs = "handling" if governing > pressure else "pressure"
        print(
            f"{length:>8.1f}m"
            f"{bag_surface_area(metres).to_value(u.m**2):>9.0f}m2"
            f"{handling_film_mass(metres, thin).to_value(u.kg):>8.1f}kg"
            f"{handling_film_mass(metres).to_value(u.kg):>9.1f}kg"
            f"{handling_film_mass(metres, thick).to_value(u.kg):>8.1f}kg"
            f"{pressure.to_value(u.kg):>10.1f}kg{governs:>10}"
        )
    print(
        "The pressure column is the coldest leg from Earth storage, the only\n"
        "case left that still boils; from cold storage it is 0 kg everywhere.\n"
        "The handling floor is not, so the bag never weighs nothing -- and a\n"
        "capsule's area is 2 pi r L, so a longer column costs film as well as\n"
        "conductor. Seams, ripstop, metallisation and inflation hardware are\n"
        "all excluded, so this is a floor rather than a design."
    )


# --- Item 4: tab:bag_sizing ------------------------------------------------

#: Plume temperature the cold-pulse field column is worked at.
COLD_PULSE_TEMPERATURE = 15000.0 * u.K
#: Plume temperature and ionisation fraction of the hottest pulse, from item 1.
HOT_PULSE_STATE = (26200.0 * u.K, 0.573)
#: Effective molar mass of dissociated neutral water: three particles per
#: 18.015 g, so 6.005 g/mol.
DISSOCIATED_MOLAR_MASS = WATER_MOLAR_MASS / ATOMS_PER_MOLECULE
#: Expansion time the radiative-loss column integrates the surface flux over.
#: **Calibrated against the published table rather than derived** -- it
#: reproduces all six rows to the digits printed, and it matches the
#: "hundred-microsecond expansion" the paper quotes elsewhere.
EXPANSION_TIME = 200.0 * u.us
#: Bag radii ``tab:bag_sizing`` tabulates.
BAG_SIZING_RADII = (1.8, 3.5, 5.4, 8.7, 17.5, 28.0)


def bag_density_at(radius: u.Quantity, slug_mass: u.Quantity = SLUG_MASS) -> u.Quantity:
    """Slug density of a spherical bag of this radius.

    Args:
        radius: Bag radius.
        slug_mass: Slug the bag holds.

    Returns:
        Density (astropy Quantity, kg/m^3).
    """
    return (slug_mass / ((4.0 / 3.0) * np.pi * radius**3)).to(u.kg / u.m**3)


def confinement_field_at(
    radius: u.Quantity,
    temperature: u.Quantity = COLD_PULSE_TEMPERATURE,
    ionisation_fraction: float = 0.0,
    slug_mass: u.Quantity = SLUG_MASS,
) -> u.Quantity:
    """Field the bag must hold at this radius, ``B = sqrt(2 mu0 P)``.

    The pressure is ideal-gas, ``P = rho R_g T / M_eff``, **not** ``(gamma-1) e``:
    reproducing the published table to three figures is what settles which of
    those the table was built from.  Ionisation enters only through the particle
    count, since pressure counts particles rather than mass -- water goes from
    three particles per molecule to ``3(1+f)``.

    Args:
        radius: Bag radius.
        temperature: Plume temperature.
        ionisation_fraction: ``f``.
        slug_mass: Slug the bag holds.

    Returns:
        Field strength (astropy Quantity, T).
    """
    molar = DISSOCIATED_MOLAR_MASS / (1.0 + ionisation_fraction)
    pressure = bag_density_at(radius, slug_mass) * const.R * temperature / molar
    return np.sqrt(2.0 * const.mu0 * pressure).to(u.T)


def radiated_fraction(
    radius: u.Quantity,
    temperature: u.Quantity = COLD_PULSE_TEMPERATURE,
    slug_mass: u.Quantity = SLUG_MASS,
) -> float:
    """Share of the pulse's energy the plume radiates away from its surface.

    Blackbody flux over the bag's surface, integrated for one expansion time,
    against the ignition budget the pulse put in.  **It grows as radius squared
    while the energy inside is fixed**, which is the trade that bounds the bag
    from above: 0.1% at 1.8 m against 31% at 28 m.

    Args:
        radius: Bag radius.
        temperature: Plume temperature.
        slug_mass: Slug the bag holds.

    Returns:
        Radiated fraction of the pulse energy.
    """
    area = 4.0 * np.pi * radius**2
    radiated = const.sigma_sb * temperature**4 * area * EXPANSION_TIME
    return float(radiated / (plume_ignition_energy(temperature) * slug_mass))


# --- Item 9: tab:axial_bag -------------------------------------------------

#: Column lengths ``tab:axial_bag`` tabulates; the first is the sphere.
AXIAL_BAG_LENGTHS = (10.8, 16.0, 23.0, 32.0, 50.0)
#: Bore radius of the spherical row, which is the sphere's own radius rather
#: than ``eq:bore_from_length``'s.
SPHERE_BORE_RADIUS = 5.40 * u.m


def bore_radius(
    column_length: u.Quantity, volume: u.Quantity = BAG_VOLUME
) -> u.Quantity:
    """Bore of a column holding a fixed volume, ``eq:bore_from_length``.

    ``r = sqrt(V / pi l)``.  Bore falls as the inverse square root of length,
    and the conductor rises as its square root, so the two trade one for one.

    Args:
        column_length: Length of the column.
        volume: Enclosed volume, fixed by ``PV = n R_g T``.

    Returns:
        Bore radius (astropy Quantity, m).
    """
    return np.sqrt(volume / (np.pi * column_length)).to(u.m)


# --- D4: the handling floor under the film ---------------------------------
#
# ``eq:bag_film_mass`` sizes a *pressure vessel*: it is ``F x rho_f R_g T /
# (M sigma)``, linear in the vapour fraction ``x``.  With the solved leak the
# slug boils nothing from cold storage, so ``x`` is zero and the equation
# returns zero film *wherever it is evaluated* -- not only in the one table
# that reported it.  Zero is the right answer to the question the equation
# asks and the wrong answer to "what does the bag weigh": a bag still has to
# be manufactured, folded, packed, deployed and inflated, and that sets a
# gauge floor no pressure argument can go below.
#
# What binds is therefore ``max(pressure film, handling film)``, and at the
# gauges below the handling term is the larger of the two for every row of
# ``tab:axial_bag`` -- which is why the film column does not vanish when the
# pressure does.

#: Film gauge the handling floor is quoted at.  Echo 1 flew a 30 m sphere of
#: half-mil (12.7 um) metallised PET in 1960, packed into a canister, deployed
#: and inflated on orbit, which is the closest flown precedent for a membrane
#: this thin at this scale.  **The paper needs this claim cited before it
#: prints it**; the arithmetic here does not depend on the citation, only the
#: choice of gauge does.
HANDLING_FILM_GAUGE = 12.7e-6 * u.m
#: Bracket the gauge is quoted over: 6 um is about as thin as a metallised
#: film is handled in bulk, 25 um (1 mil) is a conservative deployable.
HANDLING_GAUGE_BAND = (6.0e-6 * u.m, 25.0e-6 * u.m)
#: Density of the polyethylene film of ``eq:bag_film_mass``.  The pressure
#: model carries it inside :data:`FILM_COEFFICIENT`; the handling model needs
#: it on its own, so it is named here rather than back-solved twice.
FILM_DENSITY = 920.0 * u.kg / u.m**3


def bag_surface_area(
    column_length: u.Quantity, volume: u.Quantity = BAG_VOLUME
) -> u.Quantity:
    """Membrane area of a capsule bag of this total length.

    A capsule is a cylinder of length ``L - 2r`` capped by two hemispheres, and
    its area collapses to ``2 pi r L`` exactly: the caps put back precisely what
    shortening the cylinder took away.  Stretching a fixed volume into a longer
    column therefore *costs* area, since ``r`` falls only as ``1/sqrt(L)``.

    Args:
        column_length: Total bag length, tip to tip.
        volume: Bag volume, which fixes the bore.

    Returns:
        Membrane area (astropy Quantity, m^2).
    """
    radius = (
        SPHERE_BORE_RADIUS
        if np.isclose(
            column_length.to_value(u.m), 2.0 * SPHERE_BORE_RADIUS.to_value(u.m)
        )
        else bore_radius(column_length, volume)
    )
    return (2.0 * np.pi * radius * column_length).to(u.m**2)


def handling_film_mass(
    column_length: u.Quantity,
    gauge: u.Quantity = HANDLING_FILM_GAUGE,
    volume: u.Quantity = BAG_VOLUME,
) -> u.Quantity:
    """Bag mass at a manufacturable gauge, holding no pressure at all.

    This is a floor, not a design: it charges area times gauge times density
    and nothing else -- no seams, no ripstop, no inflation hardware, no
    metallisation.  A real bag is heavier.  The point of the number is that it
    is not zero, and that it does not depend on whether the slug boils.

    Args:
        column_length: Total bag length, tip to tip.
        gauge: Film thickness.
        volume: Bag volume, which fixes the bore.

    Returns:
        Film mass (astropy Quantity, kg).
    """
    return (bag_surface_area(column_length, volume) * gauge * FILM_DENSITY).to(u.kg)


def governing_film_mass(
    pressure_film_mass: u.Quantity,
    column_length: u.Quantity,
    gauge: u.Quantity = HANDLING_FILM_GAUGE,
    volume: u.Quantity = BAG_VOLUME,
) -> u.Quantity:
    """The larger of the pressure vessel and the handling floor.

    Args:
        pressure_film_mass: What ``eq:bag_film_mass`` returns for this state.
        column_length: Total bag length, tip to tip.
        gauge: Film thickness for the handling floor.
        volume: Bag volume, which fixes the bore.

    Returns:
        Film mass that actually flies (astropy Quantity, kg).
    """
    floor = handling_film_mass(column_length, gauge, volume)
    return max(pressure_film_mass.to(u.kg), floor, key=lambda m: m.to_value(u.kg))


def shape_factor(column_length: u.Quantity, volume: u.Quantity = BAG_VOLUME) -> float:
    """``F = (2L + 2r)/(L + 4r/3)`` for a capsule of this total length.

    ``L`` is the *cylindrical* part, so the sphere (``L = 0``) gives exactly
    1.5 and a long tube tends to 2.0.  The film mass is linear in ``F``, so the
    whole cost of stretching the bag into a column is this factor.

    Args:
        column_length: Total length, caps included.
        volume: Enclosed volume.

    Returns:
        The shape factor.
    """
    radius = (
        SPHERE_BORE_RADIUS
        if np.isclose(
            column_length.to_value(u.m), 2.0 * SPHERE_BORE_RADIUS.to_value(u.m)
        )
        else bore_radius(column_length, volume)
    )
    cylinder = column_length - 2.0 * radius
    return float((2.0 * cylinder + 2.0 * radius) / (cylinder + 4.0 * radius / 3.0))


def coldest_mist_temperature(
    radius: u.Quantity,
    vapour_fraction: float = 0.1107,
    slug_mass: u.Quantity = SLUG_MASS,
) -> u.Quantity:
    """Mist temperature in a spherical bag of this radius.

    The vapour fraction is held at ``tab:bag_state``'s value and only the volume
    changes, so a bigger bag holds the same vapour more thinly and condenses
    colder.

    **This reproduces the published column to ~1 K across the usable band and
    overshoots outside it** -- +16 K at 1.8 m and +9 K at 28 m.  The deviation
    runs the way an optical-depth correction would: the paper notes the plume
    "stops being optically thick past about 7 m", and a bag that has gone thin
    reabsorbs less of its own radiation, so its vapour fraction should fall
    below the held 0.11 rather than stay at it.  That correction is not modelled
    here because the paper does not state one, and the usable band the paper
    itself settles on (3.5-7 m) is inside the region that reproduces.

    Args:
        radius: Bag radius.
        vapour_fraction: ``x`` carried as vapour.
        slug_mass: Slug the bag holds.

    Returns:
        Mist temperature (astropy Quantity, K).
    """
    volume = (4.0 / 3.0) * np.pi * radius**3
    return saturation_temperature(vapour_fraction * slug_mass / volume)


if __name__ == "__main__":
    main()
