"""What the slug bag weighs, and the field leak that decides it.

The bag holds 213 kg of slug spread over 672.9 m^3 so the plume has something
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
computed.**  The paper *used to* hold it at 4.4% of the 20.8 MJ/kg of stored
field energy, which was fine while ``E_B`` was 4.43 GJ.  Item 1 moved ``E_B``
to 12.2 GJ at the hottest pulse without moving the leak fraction, and at 4.4%
of *that* the film goes from 2.8 kg to 31 kg -- from a rounding error to 15%
of the slug, every pulse.  That is the bracket this module exists to close.

**It is closed, and in the design's favour.**  The leak fraction is
``~1/Rm`` -- the field diffuses out in ``mu0 sigma L^2`` against an expansion
clock of ``L/v``, and the ratio *is* the magnetic Reynolds number, which
``tab:seed_window`` has been tabulating two pages earlier.  So the calculation
is a quadrature, not a simulation: weight ``1/Rm(T)`` by how long the plume
spends at each ``T``.  ``puffsat_impact_simulation`` ran it on its solved
cooling history (Q-L, ``make analysis-expansion``) and gets **0.11% to 2.54%**
on the equilibrium branch -- and Q-M then established that equilibrium is the
branch that holds, by two to five decades.  See :data:`SOLVED_LEAK_FRACTIONS`.

**That bracket is now closed in print**, and :func:`paper_bag_state` follows
the table the paper prints today: the solved 2.54% at the cold end of the
growth push, which leaves ``x = 0.18`` from Earth storage and ``x = 0.022``
from Jupiter's.

**The Jupiter column used to read zero, and that was an accounting error**
(ADR 0027).  Melting is gated by warming the ice and fusing it -- 0.570 MJ/kg
-- and nothing else; the ladder's third term warms *liquid* water and was
being charged against a threshold it does not belong to.  Against the real
gate the cold leg's 0.646 MJ/kg of waste heat melts through with 0.076 to
spare, that surplus splits between warming the liquid and boiling it on the
saturation curve, and the bag holds 0.9 kPa and half a kilogram of film
instead of nothing.  The conclusion survives -- half a kilogram is far under
the handling floor, so cold storage still removes the *pressure vessel* -- but
the margin does not: melting completes at a 2.19% leak and the cold leg runs
at 2.54%, so it is past that line rather than short of it.  The superseded 4.4% column is kept as
:func:`superseded_bag_state`, because numbers computed downstream of it are
still in circulation and the comparison is what identifies them.

The bag also carries the ice **plug** of ADR 0018, and the plug is a heat sink
before it is a target: see :data:`PLUG_MASS` and ``BagState(plug_mass=...)``.

See CONTEXT.md and the ledger's items 5-10.
"""

from __future__ import annotations

from dataclasses import dataclass
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
#: The column the launch envelope picks: 23.8 m at a 3.00 m bore, which is
#: ``puffsat_impact_simulation``'s and which the paper adopted in reply R14.
#: Every film mass ``sec:axial_bag`` quotes is worked at this length.
FLOWN_COLUMN_LENGTH = 23.8 * u.m
#: Bore of that column.  ``eq:bore_from_length`` is ``r = sqrt(V / pi l)``, so
#: this and the length above are what fix :data:`BAG_VOLUME`.
FLOWN_BORE_RADIUS = 3.00 * u.m
#: Enclosed bag volume: the standoff volume ``tab:axial_bag`` pours into every
#: shape, and the volume the mist of ``tab:bag_state`` fills.  **Derived from
#: the column above rather than written down**, which is the whole of ADR 0029.
#:
#: **672.9 m^3 is the flown column's own volume, adopted 2026-09-05 in place of
#: the 659.6 that used to stand here** (paper reply R14, ADR 0029).  The two
#: numbers are two answers to the same question and the paper printed both.
#: 659.6 is the 5.4 m sphere of ``tab:bag_sizing``; 672.9 is
#: ``pi r^2 l`` for the 23.8 m column at a 3.00 m bore that
#: ``puffsat_impact_simulation`` flies and that the paper's *prose* has always
#: quoted -- "3.0 m", "28 m^2", "aspect 4" are roundings of the sim's pair and
#: not of the stated one.  The sim's pair is self-consistent, so it is the one
#: adopted, and the equal-volume sphere moves with it to
#: :data:`SPHERE_BORE_RADIUS` = 5.44 m rather than 5.40.
#:
#: **What it does and does not move.**  ``eq:bag_film_mass`` is
#: ``F x rho_f R_g T / (M sigma)`` -- volume does not appear -- so the film
#: column of ``tab:axial_bag`` inherits the change only through the mist
#: temperature, which falls 0.4 K because the same vapour is spread more
#: thinly.  That is a 0.1% effect on the film and a 2% one on the pressure.
BAG_VOLUME = (np.pi * FLOWN_BORE_RADIUS**2 * FLOWN_COLUMN_LENGTH).to(u.m**3)
#: Free particles per water molecule once dissociated (2 H + O).
ATOMS_PER_MOLECULE = 3

#: Share of the plume's energy that escapes as light, and the share of the sky
#: the bag subtends to it.  Their product times the ignition bill is the
#: blackbody row of ``tab:bag_state``.
RADIATED_FRACTION = 0.012
SKY_FRACTION = 0.1

#: Temperature the slug is parked at before the burn, per location.
STORAGE_TEMPERATURE: Dict[str, u.Quantity] = {
    "jupiter": 122.0 * u.K,
    "earth": 278.0 * u.K,
}
#: Melting point, and the "room temperature" the published ladder warms to.
FREEZING_POINT = 273.15 * u.K
ROOM_TEMPERATURE = 293.15 * u.K

#: Warming 122 K ice up to the melting point, **integrating the measured heat
#: capacity of ice Ih rather than holding it constant**.  Consumed here as a
#: published input: the integration itself is the paper's, in
#: ``tab:payload_sink``, whose "warming and melting from 122 K" row is
#: 570 kJ/kg and whose latent heat is 334, leaving 236 for this term.  Ice's
#: capacity falls by a factor of twelve between melting and 30 K, which is why
#: a constant is wrong and why this is 0.236 rather than the 0.26 a textbook
#: c_p near freezing gives.
WARMING_ICE_TO_MELT = 0.236 * u.MJ / u.kg
HEAT_OF_FUSION = 0.334 * u.MJ / u.kg
#: Specific heat of liquid water.  The paper's own Earth warming row implies
#: 4.14 kJ/kg/K when read back against the mist temperature it was closed at,
#: so the textbook value is what it was built from.
LIQUID_HEAT_CAPACITY = 4.18e-3 * u.MJ / (u.kg * u.K)

#: **Heat that must go in before any of the slug can boil.**  From ice that is
#: warming to the melting point plus the heat of fusion, and *only* those two:
#: 0.570 MJ/kg.  From Earth's 278 K liquid there is no gate at all -- water in
#: vacuum evaporates toward saturation at whatever temperature it is at.
#:
#: **This is the correction of ADR 0027.**  The gate used to be the whole
#: ``WARMING_TO_LIQUID`` ladder, 0.73 MJ/kg, which charged the *liquid*
#: warming against a melting threshold it does not belong to and made the
#: Jupiter column read "never finishes melting".  It does finish: 0.646 MJ/kg
#: of waste heat against a 0.570 gate clears it with 0.076 to spare.
MELTING_GATE: Dict[str, u.Quantity] = {
    "jupiter": WARMING_ICE_TO_MELT + HEAT_OF_FUSION,
    "earth": 0.0 * u.MJ / u.kg,
}

#: Liquid warming on top of the gate, as the paper publishes it.  Jupiter's is
#: the ladder's third term, 273 K to room temperature; Earth's is the whole
#: row, and 0.21 MJ/kg warms liquid water 50 K, to 328 K -- the mist of the
#: *superseded* 4.4% table.  The paper still prints Earth's as an input, so it
#: stays one here and :func:`melting_fixed_point` measures the gap.
LIQUID_WARMING: Dict[str, u.Quantity] = {
    "jupiter": LIQUID_HEAT_CAPACITY * (ROOM_TEMPERATURE - FREEZING_POINT),
    "earth": 0.21 * u.MJ / u.kg,
}

#: Sensible-plus-latent heat to bring the slug from storage all the way up to
#: liquid at room temperature: 0.236 + 0.334 + 0.084 = 0.653 MJ/kg from
#: Jupiter's 122 K ice, 0.21 from Earth's 278 K water.  **This is the ladder,
#: not the gate** -- see :data:`MELTING_GATE`.  It is still what the plug is
#: charged and what a shading window has to reject.
WARMING_TO_LIQUID: Dict[str, u.Quantity] = {
    key: MELTING_GATE[key] + LIQUID_WARMING[key] for key in MELTING_GATE
}

#: Columns whose liquid-warming term is an *output* of the mist row rather
#: than a published input.  Only the ice column, and only because the
#: correction above made it one: the surplus over the gate is 0.076 MJ/kg
#: against a 0.653 ladder, so *where* the liquid warming is charged decides
#: the entire column, and it has to be solved on the saturation curve instead
#: of assumed at room temperature.  Earth's row is still printed as an input.
SOLVED_WARMING = frozenset({"jupiter"})

#: The warming ladder the *superseded* 4.4% table was printed with, and the
#: model it was printed under: the whole ladder charged as the melting gate.
#: Kept only so :func:`superseded_bag_state` still reproduces that table cell
#: for cell -- 306 K and 328 K are the fingerprint downstream numbers are
#: grepped for, and a reproduction that drifted would stop finding them.  The
#: Jupiter entry carries both errors ADR 0027 corrects: a constant ice heat
#: capacity (0.26 rather than 0.236) and liquid warming inside the gate.
SUPERSEDED_WARMING_TO_LIQUID: Dict[str, u.Quantity] = {
    "jupiter": 0.73 * u.MJ / u.kg,
    "earth": 0.21 * u.MJ / u.kg,
}

#: Ice plug at the end the projectile enters, sized at half again the 25 kg
#: projectile's mass so the crater runs wider than the rod (ADR 0018).  It is
#: charged the full 122 K ice warming: the plug has to end up *condensed*, not
#: frozen, so melting it is free and boiling it is not.
PLUG_MASS = 37.5 * u.kg

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
#: The leak fraction the paper used to print, **superseded** by the solved
#: fractions above and kept only for comparison.  Its table was internally
#: inconsistent by 3%: the row read "4.4% of 20.8 MJ/kg" but printed 0.89
#: MJ/kg, and 0.89/20.8 is 4.28%.  Reproducing that table needs the printed
#: 0.89, so that is what :func:`superseded_bag_state` uses.
SUPERSEDED_LEAK_FRACTION = 0.044
#: The leak energy per kilogram of slug the superseded table printed.
SUPERSEDED_LEAK_ENERGY = 0.89 * u.MJ / u.kg
#: Closing speed ``tab:bag_state`` is quoted at: the cold end of the growth
#: push, which is the leg that leaks most and therefore the leg that sizes the
#: bag.  The whole table is one column of :data:`LEG_PLUME_STATES`.
TABLE_CLOSING_SPEED = 45.58


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
        warming: Heat spent bringing the slug up to liquid, per kg.  A
            published input on the Earth column; an output of the saturation
            solve on the ice column (see :data:`SOLVED_WARMING`).
        melting_gate: Heat that must go in before any of it boils -- warming
            the ice and fusing it, and nothing else.
        melts: Whether the waste heat clears that gate.  False only with the
            plug aboard on the ice column, which is what makes cold storage
            plus a plug the one dry case left.
        plug_sink: Heat the ice plug absorbs before boiling starts, charged
            per kg of *slug* so it subtracts from the same cascade.
        boiling: What is left to boil, per kg.
        vapour_fraction: ``x``, the share of the slug carried as vapour.
        vapour_mass: ``x`` times the slug, which is the mass the film holds
            back.  With a plug it is still ``x`` times the *slug*: the boiled
            water is drawn from slug and plug together, but the energy that
            boils it is booked per kg of slug, so the product is right.
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
        plug_mass: u.Quantity = 0.0 * u.kg,
        published_warming: u.Quantity | None = None,
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
                used only to reproduce the superseded table's printed 0.89
                MJ/kg.
            published_warming: Override the warming row, which then serves as
                the melting gate as well.  **This is the superseded model, not
                an option** -- charging the liquid-warming term against the
                melting threshold is exactly the error of ADR 0027, and the
                only caller is :func:`superseded_bag_state`.
            plug_mass: Ice plug at the bag's entrance.  **Pure added thermal
                mass at a fixed heat input**: it absorbs 122 K ice warming and
                melting without boiling, so it can only ever *lower* the vapour
                mass.  ``tab:bag_state`` itself is the no-plug case; the plug
                is priced separately in ``sec:needle_through_fog``.

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
        self.warming = (
            WARMING_TO_LIQUID[storage]
            if published_warming is None
            else published_warming
        )
        self.melting_gate = (
            MELTING_GATE[storage] if published_warming is None else published_warming
        )
        storage_temperature = STORAGE_TEMPERATURE[storage]
        # The plug is frozen at 122 K wherever the slug was stored, because a
        # plug that is not solid is not a target, so it is charged the Jupiter
        # warming on both columns.
        self.plug_sink = (plug_mass * WARMING_TO_LIQUID["jupiter"] / slug_mass).to(
            u.MJ / u.kg
        )
        # **The gate, not the ladder, decides whether anything boils** (ADR
        # 0027).  Only warming the ice and melting it stand between the waste
        # heat and the first gram of vapour; warming the liquid afterwards is
        # a real cost but it happens alongside boiling rather than before it.
        self.melts = self.waste_heat - self.plug_sink > self.melting_gate
        if self.melts and storage in SOLVED_WARMING and published_warming is None:
            # The melt completes from ice, and the surplus is small enough
            # (0.076 MJ/kg) that charging the liquid warming at an assumed
            # room temperature would swamp it.  So the residue's split between
            # warming the liquid and boiling it is solved on the saturation
            # curve, which makes this column's warming row an output.
            self._adopt(
                melting_fixed_point(
                    self.waste_heat,
                    storage,
                    shape_factor=shape_factor,
                    plug_mass=plug_mass,
                    slug_mass=slug_mass,
                    bag_volume=bag_volume,
                ),
                plug_mass=plug_mass,
                slug_mass=slug_mass,
            )
            return
        self.boiling = self.waste_heat - self.warming - self.plug_sink
        self.vapour_fraction = max(0.0, float(self.boiling / LATENT_HEAT))
        if self.vapour_fraction <= 0.0:
            # The waste heat does not even finish melting the slug, so there is
            # no vapour, no saturation pressure, and nothing for the film to
            # hold.  This is a real operating point rather than a degenerate
            # one -- from cold storage it is where the plugged column lands --
            # and the bag stops being a pressure vessel and becomes a
            # containment membrane, sized by handling rather than by stress.
            self.mist_temperature = storage_temperature
            self.mist_pressure = 0.0 * u.Pa
            self.vapour_mass = 0.0 * u.kg
            self.film_fraction = 0.0
            self.film_mass = 0.0 * u.kg
            return
        self.vapour_mass = (self.vapour_fraction * slug_mass).to(u.kg)
        vapour_density = self.vapour_fraction * slug_mass / bag_volume
        self.mist_temperature = saturation_temperature(vapour_density)
        self.mist_pressure = saturation_pressure(self.mist_temperature)
        self.film_fraction = film_mass_fraction(
            self.vapour_fraction, self.mist_temperature, shape_factor
        )
        self.film_mass = (self.film_fraction * slug_mass).to(u.kg)

    def _adopt(
        self,
        solved: "MeltingFixedPoint",
        plug_mass: u.Quantity,
        slug_mass: u.Quantity,
    ) -> None:
        """Take the cascade's tail from a solved fixed point.

        Used by the ice column, whose warming row is an output.  Every term is
        recharged at the solved mist temperature, including the plug, so the
        column's own book balances exactly: waste heat = warming + plug + boil.

        Args:
            solved: The state where the energy balance and the saturation
                curve agree.
            plug_mass: Ice plug at the bag's entrance.
            slug_mass: Slug in the bag.
        """
        self.warming = solved.warming
        self.plug_sink = (
            (plug_mass / slug_mass)
            * (
                MELTING_GATE["jupiter"]
                + LIQUID_HEAT_CAPACITY * (solved.mist_temperature - FREEZING_POINT)
            )
        ).to(u.MJ / u.kg)
        self.boiling = self.waste_heat - self.warming - self.plug_sink
        self.vapour_fraction = solved.vapour_fraction
        self.vapour_mass = solved.vapour_mass
        self.mist_temperature = solved.mist_temperature
        self.mist_pressure = solved.mist_pressure
        self.film_mass = solved.film_mass
        self.film_fraction = float(self.film_mass / slug_mass)


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


def table_stored_energy(closing_speed: float = TABLE_CLOSING_SPEED) -> u.Quantity:
    """``E_B`` on the leg ``tab:bag_state`` is quoted at.

    Args:
        closing_speed: Key into :data:`LEG_PLUME_STATES`.

    Returns:
        Stored field energy (astropy Quantity, J).

    Raises:
        KeyError: If the closing speed is not one of the solved legs.
    """
    temperature, ionised = LEG_PLUME_STATES[closing_speed]
    return stored_field_energy(temperature * u.K, ionised)


def paper_bag_state(
    storage: str = "jupiter",
    shape_factor: float = SPHERE_SHAPE_FACTOR,
    plug_mass: u.Quantity = 0.0 * u.kg,
) -> BagState:
    """``tab:bag_state`` exactly as printed, for regression.

    The table is one leg -- the 45.58 km/s cold end of the growth push, which
    leaks most -- at the solved 2.54% rather than at any assumed fraction.
    Both of its columns therefore come out of :data:`SOLVED_LEAK_FRACTIONS`
    and nothing here is hand-entered.

    Args:
        storage: ``"jupiter"`` or ``"earth"``.
        shape_factor: ``F`` of ``eq:bag_film_mass``; the table is the sphere.
        plug_mass: Ice plug, if the plug is being priced.  The table itself is
            the no-plug case.

    Returns:
        The cascade as the paper prints it.
    """
    return BagState(
        leak_fraction=SOLVED_LEAK_FRACTIONS["equilibrium"][TABLE_CLOSING_SPEED],
        stored_energy=table_stored_energy(),
        storage=storage,
        shape_factor=shape_factor,
        plug_mass=plug_mass,
    )


def superseded_bag_state(storage: str = "jupiter") -> BagState:
    """``tab:bag_state`` as it read before the leak was solved, for comparison.

    **Not what the paper prints, and kept only because things computed from it
    are still in circulation.**  The old table assumed a 4.4% leak against the
    4.43 GJ neutral-plume store, which put the mist at 306 K from Jupiter
    storage and 328 K from Earth storage.  Those two temperatures are the
    signature to grep for: anything still quoting them was computed before the
    leak was solved.

    Uses the table's printed 0.89 MJ/kg leak rather than ``0.044 * 20.8``,
    which is 0.915: that row was internally inconsistent by 3% and the printed
    digits followed the smaller number.

    Args:
        storage: ``"jupiter"`` or ``"earth"``.

    Returns:
        The cascade as the paper used to print it.
    """
    return BagState(
        leak_fraction=SUPERSEDED_LEAK_FRACTION,
        stored_energy=stored_field_energy(),
        storage=storage,
        leak_energy=SUPERSEDED_LEAK_ENERGY,
        published_warming=SUPERSEDED_WARMING_TO_LIQUID[storage],
    )


def boil_onset_leak(stored_energy: u.Quantity, storage: str = "jupiter") -> float:
    """Leak fraction at which the bag first has to hold pressure.

    Closed form: the slug must clear :data:`MELTING_GATE` before any of it
    boils, and the blackbody intercept contributes to that regardless, so
    boiling starts at ``(gate - intercept) m / E_B``.

    **The gate, not the ladder** (ADR 0027).  Charging the whole 0.653 MJ/kg
    ladder here put onset at 2.93% and left the cold leg's solved 2.54% a
    comfortable 1.2x short of it.  Against the real 0.570 gate onset is 2.19%
    and the cold leg is *past* it: the leg melts through and boils.

    Args:
        stored_energy: ``E_B`` the bag confines.
        storage: ``"jupiter"`` or ``"earth"``.

    Returns:
        The leak fraction at onset.  Negative from Earth storage, where there
        is no gate at all -- liquid water in vacuum needs no threshold to
        start evaporating, so the blackbody intercept alone is already past
        onset.
    """
    intercept = (
        RADIATED_FRACTION
        * SKY_FRACTION
        * plume_ignition_energy(NOZZLE_FLOOR_TEMPERATURE)
    )
    deficit = MELTING_GATE[storage] - intercept
    return float(deficit * SLUG_MASS / stored_energy)


@dataclass(frozen=True)
class MeltingFixedPoint:
    """The cascade with its one remaining loop closed, for gap reporting.

    ``tab:bag_state``'s warming row is not independent of its mist row: the
    last part of "warming and melting the slug up to liquid" warms liquid
    water to whatever temperature the mist ends up at, and the published 0.21
    MJ/kg was closed against 328 K -- the mist of the *superseded* 4.4% table.
    Solving the two together instead is a one-line root find.

    **The ice column now runs on this rather than reporting against it**
    (ADR 0027): once the melting gate is charged correctly the Jupiter surplus
    is 0.076 MJ/kg, small enough that assuming a room-temperature liquid
    warming decides the whole column, so :class:`BagState` solves that column
    here.  Earth's row is still a published input and this still measures what
    that costs.

    Attributes:
        warming: Heat to bring the slug up to liquid at the solved mist
            temperature, per kg of slug.
        vapour_fraction: ``x`` at the fixed point.
        vapour_mass: ``x`` times the slug.
        mist_temperature: Where the saturation curve and the energy balance
            agree.
        mist_pressure: Saturation pressure there.
        film_mass: ``eq:bag_film_mass`` at that state.
        below_freezing: True if the two curves only meet below 273 K, where
            the condensate is ice rather than water and this model has left
            its domain.  The vapour is then a sublimation equilibrium and the
            reported state is the freezing point, not the solution.
    """

    warming: u.Quantity
    vapour_fraction: float
    vapour_mass: u.Quantity
    mist_temperature: u.Quantity
    mist_pressure: u.Quantity
    film_mass: u.Quantity
    below_freezing: bool


def melting_fixed_point(
    waste_heat: u.Quantity,
    storage: str = "earth",
    shape_factor: float = SPHERE_SHAPE_FACTOR,
    plug_mass: u.Quantity = 0.0 * u.kg,
    slug_mass: u.Quantity = SLUG_MASS,
    bag_volume: u.Quantity = BAG_VOLUME,
) -> MeltingFixedPoint:
    """Solve the warming row and the mist row together instead of in sequence.

    Two curves in the mist temperature: the energy balance says how much
    vapour is left once the slug has been warmed *to that temperature*, and
    the saturation curve says how much vapour that temperature can hold.  The
    first falls and the second rises, so they cross exactly once and bisection
    is enough.

    Args:
        waste_heat: Waste heat the slug must absorb, per kg of slug.
        storage: ``"jupiter"`` (122 K ice, which must melt first) or
            ``"earth"`` (278 K liquid, which need only warm).
        shape_factor: ``F`` of ``eq:bag_film_mass``.
        plug_mass: Ice plug, charged the same melting ladder from 122 K.
        slug_mass: Slug in the bag.
        bag_volume: Enclosed volume, which sets the vapour density.

    Returns:
        The state where both rows hold at once.

    Raises:
        KeyError: If ``storage`` is not a known location.
    """
    freezing = FREEZING_POINT.to_value(u.K)
    to_liquid = MELTING_GATE[storage]
    from_ice = to_liquid > 0.0 * u.MJ / u.kg
    liquid_from = freezing if from_ice else STORAGE_TEMPERATURE[storage].to_value(u.K)
    capacity = LIQUID_HEAT_CAPACITY * u.K  # per kelvin of warming
    plug_share = float(plug_mass / slug_mass)

    def warming_at(kelvin: float) -> u.Quantity:
        return to_liquid + capacity * max(0.0, kelvin - liquid_from)

    def surplus(kelvin: float) -> float:
        """Vapour the energy balance leaves, less what saturation can hold."""
        plug = plug_share * (
            MELTING_GATE["jupiter"] + capacity * max(0.0, kelvin - freezing)
        )
        left = waste_heat - warming_at(kelvin) - plug
        by_energy = max(0.0, float(left / LATENT_HEAT))
        by_saturation = float(saturation_density(kelvin * u.K) * bag_volume / slug_mass)
        return by_energy - by_saturation

    low, high = freezing, 450.0
    below_freezing = surplus(low) <= 0.0
    if not below_freezing:
        for _ in range(200):
            mid = 0.5 * (low + high)
            if surplus(mid) > 0.0:
                low = mid
            else:
                high = mid
    kelvin = 0.5 * (low + high) if not below_freezing else freezing
    temperature = kelvin * u.K
    fraction = float(saturation_density(temperature) * bag_volume / slug_mass)
    mass = (fraction * slug_mass).to(u.kg)
    return MeltingFixedPoint(
        warming=warming_at(kelvin).to(u.MJ / u.kg),
        vapour_fraction=fraction,
        vapour_mass=mass,
        mist_temperature=temperature,
        mist_pressure=saturation_pressure(temperature),
        film_mass=(
            film_mass_fraction(fraction, temperature, shape_factor) * slug_mass
        ).to(u.kg),
        below_freezing=below_freezing,
    )


def main() -> None:
    """Print the bag-state reproduction and the closed leak bracket."""
    rows = (
        ("blackbody intercept [MJ/kg]", lambda s: s.intercept.to_value(u.MJ / u.kg)),
        ("field leak [MJ/kg]", lambda s: s.leak.to_value(u.MJ / u.kg)),
        ("waste heat [MJ/kg]", lambda s: s.waste_heat.to_value(u.MJ / u.kg)),
        ("melting gate [MJ/kg]", lambda s: s.melting_gate.to_value(u.MJ / u.kg)),
        ("warming charged [MJ/kg]", lambda s: s.warming.to_value(u.MJ / u.kg)),
        ("left to boil [MJ/kg]", lambda s: s.boiling.to_value(u.MJ / u.kg)),
        ("vapour fraction x", lambda s: s.vapour_fraction),
        ("vapour mass [kg]", lambda s: s.vapour_mass.to_value(u.kg)),
        ("mist temperature [K]", lambda s: s.mist_temperature.to_value(u.K)),
        ("mist pressure [kPa]", lambda s: s.mist_pressure.to_value(u.kPa)),
        ("bag film [kg]", lambda s: s.film_mass.to_value(u.kg)),
        ("bag film [% of slug]", lambda s: 100.0 * s.film_fraction),
    )

    print("=== tab:bag_state, as printed ===")
    print(f"{'row':<32}{'Jupiter 122 K':>15}{'Earth 278 K':>15}")
    jupiter, earth = paper_bag_state("jupiter"), paper_bag_state("earth")
    for label, getter in rows:
        print(f"{label:<32}{getter(jupiter):>15.4g}{getter(earth):>15.4g}")
    print(
        f"The leak is the solved {100 * SOLVED_LEAK_FRACTIONS['equilibrium'][TABLE_CLOSING_SPEED]:.2f}%"
        f" at the {TABLE_CLOSING_SPEED} km/s cold end of the growth push.\n"
        "Jupiter storage melts through with 0.076 MJ/kg to spare and boils\n"
        "2.2% of the flow, so that column holds ~0.9 kPa and half a kilogram\n"
        "of film -- far under the handling floor, but not the zero it read\n"
        "before the melting gate was separated from the warming ladder."
    )

    print("\n=== The same table before the leak was solved, for comparison ===")
    print(f"{'row':<32}{'Jupiter 122 K':>15}{'Earth 278 K':>15}")
    was_jupiter, was_earth = superseded_bag_state("jupiter"), superseded_bag_state(
        "earth"
    )
    for label, getter in rows:
        print(f"{label:<32}{getter(was_jupiter):>15.4g}{getter(was_earth):>15.4g}")
    print(
        "306 K and 328 K are the signature of the superseded 4.4% leak. Any\n"
        "number still quoting them was computed before the leak was solved."
    )

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
            f"{BagState(SUPERSEDED_LEAK_FRACTION, energy).film_mass.to_value(u.kg):>14.2f}"
        )
    print(
        "\nLeak fractions are residence-weighted 1/Rm on the equilibrium branch,\n"
        "from puffsat_impact_simulation's solved cooling history (Q-L, Q-M).\n"
        "Onset is the 0.570 MJ/kg melting gate, not the 0.653 ladder (ADR\n"
        "0027), so a margin under 1.0x means the leg melts through and boils.\n"
        "The cold leg does, at 0.9x; the three hot legs clear it 4.9-7.5x."
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
    sphere_length = SPHERE_BAG_LENGTH.to_value(u.m)
    reference = SPHERE_BORE_RADIUS.to_value(u.m) * sphere_length
    sphere_film = paper_bag_state("earth").film_mass
    for length in AXIAL_BAG_LENGTHS:
        metres = length * u.m
        radius = (
            SPHERE_BORE_RADIUS
            if np.isclose(length, sphere_length)
            else bore_radius(metres)
        ).to_value(u.m)
        factor = shape_factor(metres)
        film = sphere_film * factor / SPHERE_SHAPE_FACTOR
        print(
            f"{length:>8.1f}m{radius:>8.2f}m{radius * length / reference:>12.2f}"
            f"{factor:>7.2f}{film.to_value(u.kg):>8.1f}kg"
        )
    print(
        "Bore and conductor trade one for one; the launch envelope picks\n"
        "23.8 m -- the sim's column, adopted with its 672.9 m3 (ADR 0029).\n"
        "The film column is the Earth-storage pressure vessel, which is what\n"
        "the paper quotes; cold storage boils eight times less into a colder\n"
        "mist and so pays about a tenth of it -- 0.51 kg at the sphere to\n"
        "0.68 at 50 m, under the handling floor at every length (ADR 0027)."
    )

    print("\n=== sec:needle_through_fog: the plug as a heat sink ===")
    for storage in ("jupiter", "earth"):
        bare = paper_bag_state(storage)
        plugged = paper_bag_state(storage, plug_mass=PLUG_MASS)
        bill = (bare.waste_heat * SLUG_MASS).to(u.MJ)
        sink = (plugged.plug_sink * SLUG_MASS).to(u.MJ)
        print(
            f"{storage:>8} storage: bill {bill.to_value(u.MJ):5.1f} MJ, the plug "
            f"removes {sink.to_value(u.MJ):4.1f} MJ "
            f"({100 * float(sink / bill):.0f}% of it)"
        )
        for label, state in (("without plug", bare), ("with plug", plugged)):
            if state.vapour_fraction <= 0.0:
                print(f"{label:>21}: nothing boils")
                continue
            flown = FLOWN_COLUMN_LENGTH
            column = state.film_mass * shape_factor(flown) / SPHERE_SHAPE_FACTOR
            governing = governing_film_mass(column, flown)
            print(
                f"{label:>21}: vapour {state.vapour_mass.to_value(u.kg):5.2f} kg"
                f"  x {state.vapour_fraction:.4f}"
                f"  mist {state.mist_temperature.to_value(u.K):6.2f} K"
                f"  film(F=1.5) {state.film_mass.to_value(u.kg):5.2f} kg"
                f"  film(23.8 m) {column.to_value(u.kg):5.2f} kg"
                f"  flies {governing.to_value(u.kg):5.2f} kg"
                f" ({'handling' if governing > column else 'pressure'})"
            )
    print(
        "The plug is pure added thermal mass at a fixed heat input, so it can\n"
        "only lower the vapour mass and cool the mist. From cold storage that\n"
        "is now enough to matter: the bare column melts through by 0.076\n"
        "MJ/kg, and the plug's 24.5 MJ takes it back under the gate, so cold\n"
        "storage plus a plug is the one dry case left (ADR 0027). From Earth\n"
        "storage it takes the flown 23.8 m column below the 12.7 um handling\n"
        "floor, so the plug is what stops the bag being a pressure vessel at\n"
        "all on the leg that boils hardest."
    )

    print("\n=== The one loop the table still cuts: warming vs mist temperature ===")
    print(
        f"{'case':>26}{'warming':>10}{'x':>9}{'vapour':>10}{'mist':>9}"
        f"{'press':>9}{'film':>8}"
    )
    for storage in ("jupiter", "earth"):
        for label, plug in (("no plug", 0.0 * u.kg), ("with plug", PLUG_MASS)):
            published = paper_bag_state(storage, plug_mass=plug)
            closed = melting_fixed_point(published.waste_heat, storage, plug_mass=plug)
            for name, warming, x, mass, temp, press, film in (
                (
                    f"{storage} {label}, printed",
                    published.warming,
                    published.vapour_fraction,
                    published.vapour_mass,
                    published.mist_temperature,
                    published.mist_pressure,
                    published.film_mass,
                ),
                (
                    f"{storage} {label}, closed",
                    closed.warming,
                    closed.vapour_fraction,
                    closed.vapour_mass,
                    closed.mist_temperature,
                    closed.mist_pressure,
                    closed.film_mass,
                ),
            ):
                edge = (
                    "  (ice, not water: the two curves only meet below 273 K)"
                    if name.endswith("closed") and closed.below_freezing
                    else ""
                )
                print(
                    f"{name:>26}{warming.to_value(u.MJ / u.kg):>10.3f}{x:>9.4f}"
                    f"{mass.to_value(u.kg):>9.2f}kg{temp.to_value(u.K):>8.1f}K"
                    f"{press.to_value(u.kPa):>8.2f}k{film.to_value(u.kg):>7.2f}kg{edge}"
                )
    print(
        "Earth's printed warming row was closed against the superseded\n"
        "table's 328 K, so it overcharges the slug now: closing the loop moves\n"
        "that column up about 4 kg of vapour and 2 K. Reported, not applied --\n"
        "the paper prints that row as an input. The Jupiter column IS this\n"
        "solve now (ADR 0027), so its two lines agree by construction."
    )

    print("\n=== D4: the handling floor under the film ===")
    thin, thick = HANDLING_GAUGE_BAND
    earth_film = sphere_film
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
        "The pressure column is the coldest leg from Earth storage, which is\n"
        "the leg that boils hardest; from cold storage it is a tenth of this,\n"
        "0.51-0.68 kg, and under the floor at every length (ADR 0027).\n"
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

#: Bore radius of the spherical row, which is the sphere's own radius rather
#: than ``eq:bore_from_length``'s.  **Derived from :data:`BAG_VOLUME` rather
#: than written down**, because the two used to be able to drift: the table
#: normalises its conductor column to "the sphere of the same volume", and a
#: hardcoded 5.40 m stopped being that sphere the moment the standoff volume
#: moved to the flown column's 672.9 m^3.  It is now 5.4361 m, so the sphere
#: row is 10.9 m across rather than 10.8.
SPHERE_BORE_RADIUS = ((3.0 * BAG_VOLUME / (4.0 * np.pi)) ** (1.0 / 3.0)).to(u.m)
#: Length of the spherical row: its own diameter.
SPHERE_BAG_LENGTH = 2.0 * SPHERE_BORE_RADIUS
#: Column lengths ``tab:axial_bag`` tabulates; the first is the sphere and the
#: third is :data:`FLOWN_COLUMN_LENGTH`, which sits with :data:`BAG_VOLUME`
#: because it is what defines it.
AXIAL_BAG_LENGTHS = (
    SPHERE_BAG_LENGTH.to_value(u.m),
    16.0,
    FLOWN_COLUMN_LENGTH.to_value(u.m),
    32.0,
    50.0,
)


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
    vapour_fraction: float | None = None,
    slug_mass: u.Quantity = SLUG_MASS,
) -> u.Quantity:
    """Mist temperature in a spherical bag of this radius.

    The vapour fraction is held at ``tab:bag_state``'s Earth-storage value --
    the column the paper's caption names, and the one that boils eight times
    harder -- and only the volume changes, so a bigger bag holds the same
    vapour more thinly and condenses colder.

    **The default is read off the table rather than pinned**, because this
    column was left behind once already: it was written against the superseded
    4.4% leak's ``x`` = 0.11 and printed a 306 K flown row, which then survived
    into three sentences of prose after the leak was solved.  Taking the
    fraction from :func:`paper_bag_state` means the column cannot go stale
    without the table going stale with it.

    **This reproduces the published column to ~1 K across the usable band and
    overshoots outside it** -- +16 K at 1.8 m and +9 K at 28 m.  The deviation
    runs the way an optical-depth correction would: the paper notes the plume
    "stops being optically thick past about 7 m", and a bag that has gone thin
    reabsorbs less of its own radiation, so its vapour fraction should fall
    below the held value rather than stay at it.  That correction is not
    modelled here because the paper does not state one, and the usable band the
    paper itself settles on (3.5-7 m) is inside the region that reproduces.

    Args:
        radius: Bag radius.
        vapour_fraction: ``x`` carried as vapour.  Defaults to
            ``tab:bag_state``'s Earth-storage fraction.
        slug_mass: Slug the bag holds.

    Returns:
        Mist temperature (astropy Quantity, K).
    """
    if vapour_fraction is None:
        vapour_fraction = paper_bag_state("earth").vapour_fraction
    volume = (4.0 / 3.0) * np.pi * radius**3
    return saturation_temperature(vapour_fraction * slug_mass / volume)


if __name__ == "__main__":
    main()
