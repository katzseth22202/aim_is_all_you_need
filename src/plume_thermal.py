"""Whether the collision plume is hot enough for a magnetic nozzle to grip it.

ADR ``0013``'s impulse law is a momentum-out-of-energy ideal: it asks only how
much energy the blob carries, never what state the matter is in.  A magnetic
nozzle, unlike a pusher plate, can only steer a *conductor*, so the same
collision has a second admissibility condition the impulse law cannot see --
the plume must ionise.

The blob is impactor plus vaporised slug.  Only the *dissipated* part of the
collision heats it: the merge is inelastic, so of the impactor's vehicle-frame
kinetic energy ``w^2/2`` per impactor kilogram, the fraction ``k/(1+k)`` is
thermalised and the rest stays as bulk drift.  Spread over the ``1+k``
kilograms of blob that gives

    eps_th(w, k) = w^2 k / (2 (1+k)^2)

which peaks at ``k = 1`` and falls away on *both* sides.  That two-sidedness is
the module's main result: piling on slug spreads a fixed energy too thin, but
so does carrying almost none, because a vanishing slug dissipates nothing in
the merge.  The admissible slug ratios are therefore a closed **window**, not a
ceiling, and ``k -> 0`` is outside it.  A pusher plate has no such floor -- it
does not need a plasma -- so the plate is *not* recoverable as the small-``k``
limit of a nozzle, however cleanly the two impulse laws converge there.

The slug is water carrying a one-percent alkali seed.  The seed supplies the
free electrons, so the water itself is never charged for ionisation; what the
budget is actually dominated by is pulling H2O apart, 60% of the bill.  Both
impactor and slug are given water's caloric properties, which is exact nowhere
and close everywhere the answer matters (the blob is over 75% slug by mass for
``k >= 3``); below ``k ~ 1`` the blob is mostly impactor and the window's lower
root should be read as indicative.

See ``docs/adr/0014-magnetic-nozzle-on-both-legs.md`` and CONTEXT.md,
"Plume ignition window".
"""

from typing import Optional, Tuple

import numpy as np
from astropy import constants as const
from astropy import units as u

#: Molar mass of water.
WATER_MOLAR_MASS = 18.015 * u.g / u.mol
#: Total atomisation enthalpy, H2O -> 2H + O.  The dominant term in the whole
#: budget: 50.9 MJ/kg, against 31 MJ/kg of thermal energy at the floor
#: temperature and 0.16 MJ/kg for the seed.
WATER_ATOMISATION_ENTHALPY = 917 * u.kJ / u.mol
#: Sensible heat from ~300 K liquid plus latent heat of vaporisation.
WATER_VAPORISATION_ENERGY = 3.0 * u.MJ / u.kg
#: Free particles each water molecule contributes once dissociated (2 H + O).
WATER_ATOMS_PER_MOLECULE = 3

#: Alkali seed mass fraction of the slug.  Its job is conductivity, not thrust.
SEED_MASS_FRACTION = 0.01
#: Potassium.
SEED_MOLAR_MASS = 39.0983 * u.g / u.mol
#: First ionisation potential of potassium.
SEED_IONISATION_ENERGY = 4.34 * u.eV

#: Plume temperature the nozzle is designed to start from.  The nozzle converts
#: thermal energy into directed energy, so the plume only cools from here; this
#: is the post-merge peak, and holding it at the *coldest* instant of a burn is
#: what the window below enforces.
NOZZLE_FLOOR_TEMPERATURE = 15000 * u.K
#: Stagnation temperature below which the plume is not realistically hot, set
#: by Seth 2026-08-25 and used to bound the slug ratio.  It is *looser* than
#: :data:`NOZZLE_FLOOR_TEMPERATURE`, not tighter: 15 000 K is the design point
#: the nozzle is built around, 10 000 K is the gate below which
#: ``puffsat_impact_simulation``'s solved surface stops dissociating the water
#: at all (its seven excluded nodes sit at bond fractions 0.38-0.75).  The
#: seed is what makes the looser gate safe -- potassium keeps supplying
#: electrons long after the water stops, so conductivity is 1480 S/m here and
#: still 403 S/m at 4500 K, and it is dissociation rather than conduction that
#: fails first.
NOZZLE_GATE_TEMPERATURE = 10000 * u.K


def _particle_counts() -> Tuple[float, float]:
    """Free particles per kilogram of seeded, fully dissociated blob.

    Returns:
        ``(water_atoms, seed_pairs)`` per kg, where each seed pair is one ion
        plus one electron and therefore contributes two translational degrees
        of freedom's worth of particles.
    """
    avogadro = float(const.N_A.to_value(1 / u.mol))
    water_kg = 1.0 - SEED_MASS_FRACTION
    atoms = (
        water_kg
        / float(WATER_MOLAR_MASS.to_value(u.kg / u.mol))
        * avogadro
        * WATER_ATOMS_PER_MOLECULE
    )
    seed = SEED_MASS_FRACTION / float(SEED_MOLAR_MASS.to_value(u.kg / u.mol)) * avogadro
    return atoms, seed


def plume_ignition_energy(
    temperature: u.Quantity = NOZZLE_FLOOR_TEMPERATURE,
) -> u.Quantity:
    """Specific energy to take cold seeded water to a conducting plume.

    Charges vaporisation, full atomisation of the water, ionisation of the
    alkali seed only, and the translational energy of everything that is then
    free to move (dissociated atoms, seed ions and their electrons).  The water
    is never charged for ionisation because the seed is what carries the
    current.

    Args:
        temperature: Plume temperature to reach.

    Returns:
        Energy per kilogram of blob (astropy Quantity, J/kg).
    """
    atoms, seed = _particle_counts()
    boltzmann = float(const.k_B.to_value(u.J / u.K))
    water_kg = 1.0 - SEED_MASS_FRACTION
    chemical = water_kg * (
        float(WATER_VAPORISATION_ENERGY.to_value(u.J / u.kg))
        + float(WATER_ATOMISATION_ENTHALPY.to_value(u.J / u.mol))
        / float(WATER_MOLAR_MASS.to_value(u.kg / u.mol))
    )
    ionisation = seed * float(SEED_IONISATION_ENERGY.to_value(u.J))
    thermal = 1.5 * (atoms + 2.0 * seed) * boltzmann * float(temperature.to_value(u.K))
    return (chemical + ionisation + thermal) * u.J / u.kg


def specific_thermal_energy(closing_speed: u.Quantity, slug_ratio: float) -> u.Quantity:
    """Energy the inelastic merge actually thermalises, per kilogram of blob.

    Args:
        closing_speed: Impactor speed relative to the vehicle, ``w``.
        slug_ratio: Kilograms of slug per kilogram of impactor, ``k``.

    Returns:
        Specific internal energy of the blob (astropy Quantity, J/kg).

    Raises:
        ValueError: If the slug ratio is negative.
    """
    if slug_ratio < 0.0:
        raise ValueError("slug_ratio must be non-negative")
    speed = float(closing_speed.to_value(u.m / u.s))
    return (0.5 * speed * speed * slug_ratio / (1.0 + slug_ratio) ** 2) * u.J / u.kg


def plume_temperature(closing_speed: u.Quantity, slug_ratio: float) -> u.Quantity:
    """Post-merge temperature of the blob at a given closing speed and slug ratio.

    The inverse of :func:`plume_ignition_energy` at fixed composition.  Only
    meaningful once the chemical bill is paid; a blob with less energy than
    that is not a dissociated plume at all, and the function reports 0 K rather
    than extrapolating a model that no longer applies.

    Args:
        closing_speed: Impactor speed relative to the vehicle, ``w``.
        slug_ratio: Kilograms of slug per kilogram of impactor, ``k``.

    Returns:
        Plume temperature (astropy Quantity, K), or 0 K if the collision cannot
        even dissociate and seed-ionise the blob.
    """
    available = float(
        specific_thermal_energy(closing_speed, slug_ratio).to_value(u.J / u.kg)
    )
    chemical = float(plume_ignition_energy(0.0 * u.K).to_value(u.J / u.kg))
    atoms, seed = _particle_counts()
    per_kelvin = 1.5 * (atoms + 2.0 * seed) * float(const.k_B.to_value(u.J / u.K))
    return max(0.0, (available - chemical) / per_kelvin) * u.K


def slug_ratio_window(
    closing_speed: u.Quantity,
    temperature: u.Quantity = NOZZLE_FLOOR_TEMPERATURE,
) -> Optional[Tuple[float, float]]:
    """Slug ratios whose plume reaches ``temperature`` at this closing speed.

    Solving ``w^2 k / (2 (1+k)^2) = eps`` is a quadratic in ``k`` with two
    positive roots, so the admissible set is a closed interval.  The upper root
    is the familiar ceiling -- too much slug spreads the energy thin.  The lower
    root is the part that is easy to miss: too *little* slug means the merge
    dissipates almost nothing, and a magnetic nozzle has no plasma to hold.

    Args:
        closing_speed: Impactor speed relative to the vehicle, ``w``.  On a
            burn this must be the value at the coldest instant -- the end of an
            overtaking push, the start of a head-on burn.
        temperature: Plume temperature the nozzle requires.

    Returns:
        ``(k_min, k_max)``, or None if no slug ratio reaches the temperature at
        this closing speed (the peak of the curve, at ``k = 1``, is
        ``w^2 / 8``, so this happens when ``w^2 / 8 < eps``).
    """
    peak_scale = 0.5 * float(closing_speed.to_value(u.m / u.s)) ** 2
    required = float(plume_ignition_energy(temperature).to_value(u.J / u.kg))
    discriminant = peak_scale * (peak_scale - 4.0 * required)
    if discriminant < 0.0:
        return None
    root = float(np.sqrt(discriminant))
    lower = (peak_scale - 2.0 * required - root) / (2.0 * required)
    upper = (peak_scale - 2.0 * required + root) / (2.0 * required)
    return max(lower, 0.0), upper


def chemistry_efficiency(
    closing_speed: u.Quantity,
    slug_ratio: float,
    bond_fraction: float = 1.0,
) -> float:
    """Ceiling ``eta_chem`` that frozen dissociation puts on the jet efficiency.

    The plume leaves the nozzle chemically frozen: ``puffsat_impact_simulation``
    (Q-P, ``make analysis-fireball``) finds the three-body recombination rate
    crosses ``Da = 1`` at the *first station past the lip*, with 90-100% of the
    atomisation store still held.  The paper's own rate check is not wrong, it
    is evaluated at the wrong station -- at 1 kg/m^3 recombination really is
    ~0.01 us, but there the plume is fully atomised and has nothing to give
    back yet.  By the time it does, it sits at ~0.02 kg/m^3 and past the lip.

    So the loan does not come back, and the energy that pays it is unavailable
    to the jet.  The ideal gross jet places the whole collision energy
    ``w^2/2`` on one axis of the ``1+k`` kilograms of blob, giving
    ``w/sqrt(1+k)``; paying the toll first leaves
    ``sqrt(w^2/(1+k) - 2 E_a phi)``, and the ratio is what this returns.

    **This is a component of the paper's own ``eta_jet``, not a new term.**
    ``sec:jet_efficiency`` already defines ``eta_jet^2`` to include "frozen
    ionization or dissociation energy", so charging this separately as an
    energy debit would double-count.  Multiply it into the jet efficiency and
    sweep the remainder (divergence, exhaust-speed spread, radiative escape,
    and mass the field fails to grip) as the geometric factor.

    Args:
        closing_speed: Impactor speed relative to the vehicle, ``w``.
        slug_ratio: Kilograms of slug per kilogram of impactor, ``k``.
        bond_fraction: Share of the atomisation store still held at the
            freeze, ``phi``.  The default 1.0 is a *floor* on the true
            efficiency and can never overstate it, because ``phi <= 1`` and
            ``eta_chem`` falls with ``phi``; across the 74 conducting nodes of
            the solved surface it understates by at most 0.057.  Pass the
            solved ``bond_fraction`` from ``data/results/eta_chem.csv`` when
            that slack matters.

    Returns:
        ``eta_chem`` in [0, 1], or 0.0 where the toll exceeds the whole
        one-axis budget and no jet survives it.
    """
    w = float(closing_speed.to_value(u.m / u.s))
    toll = float((WATER_ATOMISATION_ENTHALPY / WATER_MOLAR_MASS).to_value(u.J / u.kg))
    remaining = 1.0 - 2.0 * toll * bond_fraction * (1.0 + slug_ratio) / (w * w)
    return float(np.sqrt(remaining)) if remaining > 0.0 else 0.0
