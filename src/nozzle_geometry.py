"""What slug ratio the projectile's arrival radius actually delivers.

``k`` is treated everywhere else in this repository as a design variable the
chain optimiser picks.  It is not: it is an *output* of the snowplow geometry.
The paper's model is ``m(x) = m_0 + rho A x`` with ``A`` the full bore area,
which is not a derivation of ``k = 8.5`` but the assumption that the sweeping
front spans the whole bore from the moment the projectile enters.  A compact
25 kg ice sphere is 0.187 m against a 3 m bore -- 16x in radius, 260x in area.

The relation is exact and quadratic when the front does not spread:

    k_eff / k_full = (r_arrival / R_bore)^2

so ``k = 8.5`` is really the statement *the front sweeps the whole bag*: the
volume it needs is 658.1 m^3 against the bag's 659.6.  Half the bore radius
delivers a quarter of ``k``, and there is no shallow region in between.

**The parameter only bites if the front does not spread.**  Letting it grow as
``dr/dx = c_exp/v`` with momentum shared inelastically, a front arriving at even
0.6 of the bore closes to the wall and recovers ``k`` = 7.3-8.0 at
``c_exp`` = 3-8 km/s.  It has 23.8 m of column to close 1.2 m of gap.  So the
honest statement is not "the arrival radius must be large" but "the arrival
radius matters exactly insofar as the front is rigid", and ``c_exp`` is a 2D
hydro question neither repository solves (``puffsat_impact_simulation``, Q-Q).

**The design arrival is compact, and the chain arithmetic is why.**  Because the
front widens itself, a 0.15 m arrival still sweeps most of the bag -- ``k`` =
7.21, which is *inside* ADR 0016's tolled optimum band, where the 8.60 a wide
arrival delivers overshoots it.  Compact therefore wins on delivered growth
before any aperture argument is made.  See :data:`DESIGN_ARRIVAL_FRACTION`.

**And then the rigid front turned out to be unphysical, 2026-08-26.** ``c_exp``
was being swept at 3-8 km/s because nobody had computed it. It is computable
without a 2D solve: the freshly shocked material at the front takes ``v^2/2`` of
specific energy, and ``eos_water`` returns its sound speed. At entry that is
**94 630 K and 21.1 km/s**, which is *half the closing speed* -- a 24.9 degree
half-angle. The front is strongly self-widening, it reaches the bore wall in the
first few metres from any arrival radius, and **the arrival radius is therefore
very nearly irrelevant**: even a compact 25 kg ice rod recovers ``k`` = 7.24
against the rigid model's 0.034.

Two checks that this is not an artefact. Balancing the shocked pressure against
the cold cloud's ram pressure, ``sqrt(P/rho_ambient)``, gives **1.6-1.9x faster**
lateral speed than ``c_s`` at every station, so ``c_s`` is the conservative
choice. And the shock-compression ratio, which is the one weakly-known input,
moves ``c_s`` by only 9% across 2x-16x.

See :func:`self_consistent_slug_ratio`.

See CONTEXT.md, "Plume ignition window", and
``docs/adr/0016-frozen-dissociation-is-charged-inside-the-momentum-debit.md``.

This module covers the ledger's items 11, 12 and 13.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

#: Slug mass the flown bag holds (kg).
BAG_SLUG_MASS = 213.0
#: Enclosed bag volume (m^3), the standoff volume of ``tab:bag_sizing``.
BAG_VOLUME = 659.6
#: Bore radius the snowplow sweeps through (m).
BORE_RADIUS = 3.0
#: Column length that makes the full-bore sweep reproduce the paper's k (m).
COLUMN_LENGTH = 23.8
#: Projectile mass (kg).
IMPACTOR_MASS = 25.0
#: Aperture the compact projectile enters through (m).  ``sec:needle_through_fog``
#: sizes it at 0.15 m into a 28 m^2 bore, which is **0.25% open**.
ENTRY_APERTURE_RADIUS = 0.15
#: Arrival radius as a fraction of the bore.
#:
#: **Corrected 2026-08-26 from the 0.8 set on 2026-08-25, and the calculation is
#: the reason, not the paper.**  Self-widening
#: (:func:`self_consistent_slug_ratio`) delivers ``k`` = 7.21 from a 0.15 m
#: arrival against 8.60 from 0.8 of the bore.  **7.21 sits inside ADR 0016's
#: tolled optimum band of 6.75-7.77; 8.60 overshoots it.**  So the compact
#: arrival delivers 99.4% of the achievable chain growth at ``eta_geom`` = 1
#: against the wide front's 91.7%.  On the chain arithmetic alone, compact wins.
#:
#: That it also matches ``sec:needle_through_fog`` is corroboration, not the
#: argument.  The paper's own reason is an aperture one -- a head-on nozzle is a
#: mirror with a hole where the projectile came in, 0.15 m is 0.25% of the bore
#: against 0.8's 64% -- and **that reason is weaker than the paper states**: a
#: magnetic mirror's leak is set by its mirror ratio's loss cone, not by the
#: physical open area, so "anything left open leaks it" is overstated for a
#: field. What does leave ballistically through a physical hole is the
#: **un-ionised** fraction, and at the cold leg that is most of the plume
#: (``f`` = 0.06 at 15 170 K).  So the aperture argument survives, but through
#: the neutrals rather than through the geometry the paper appeals to.  Recorded
#: as a paper edit rather than resolved here.
#:
#: The 0.8 case stays reachable as a parameter; nothing about it is unsafe, it
#: is simply dominated.
DESIGN_ARRIVAL_FRACTION = ENTRY_APERTURE_RADIUS / BORE_RADIUS
#: Integration steps down the column.  1e5 converges the sweep to <1e-3 in k.
_STEPS = 100_000


def bag_density() -> float:
    """Slug density of the flown bag (kg/m^3).

    Returns:
        ``BAG_SLUG_MASS / BAG_VOLUME``, the 0.323 kg/m^3 the plume state is
        solved at.
    """
    return BAG_SLUG_MASS / BAG_VOLUME


def full_bore_slug_ratio() -> float:
    """Slug ratio a front spanning the whole bore from ``x = 0`` delivers.

    Returns:
        ``rho A L / m``, which is 8.69 and is what the paper's 8.5 comes from.
    """
    return bag_density() * np.pi * BORE_RADIUS**2 * COLUMN_LENGTH / IMPACTOR_MASS


def swept_slug_ratio(
    arrival_fraction: float = DESIGN_ARRIVAL_FRACTION,
    expansion_speed: float = 0.0,
    closing_speed: float = 45.58,
) -> float:
    """Slug ratio a front arriving at ``arrival_fraction`` of the bore delivers.

    With ``expansion_speed = 0`` this is exactly
    ``full_bore_slug_ratio() * arrival_fraction**2``; the integration exists for
    the spreading case, where the front grows as ``dr/dx = c_exp/v`` while the
    projectile slows as it takes on swept mass.

    Args:
        arrival_fraction: Front radius at entry, as a fraction of the bore.
        expansion_speed: Radial spreading speed of the front (km/s).  Zero is
            the rigid-front bound and the conservative one.
        closing_speed: Impactor speed relative to the vehicle (km/s).  A faster
            projectile has less time to spread, so this only matters when
            ``expansion_speed`` is non-zero.

    Returns:
        The delivered slug ratio ``k_eff``.

    Raises:
        ValueError: If ``arrival_fraction`` is not in (0, 1].
    """
    if not 0.0 < arrival_fraction <= 1.0:
        raise ValueError("arrival_fraction must be in (0, 1]")
    rho, radius, mass = bag_density(), arrival_fraction * BORE_RADIUS, IMPACTOR_MASS
    if expansion_speed <= 0.0:
        return full_bore_slug_ratio() * arrival_fraction**2
    step = COLUMN_LENGTH / _STEPS
    rate = (expansion_speed / closing_speed) / IMPACTOR_MASS
    for _ in range(_STEPS):
        mass += rho * np.pi * radius * radius * step
        if radius < BORE_RADIUS:
            radius = min(BORE_RADIUS, radius + rate * mass * step)
    return (mass - IMPACTOR_MASS) / IMPACTOR_MASS


def arrival_fraction_for(slug_ratio: float) -> float:
    """Arrival radius a rigid front needs to deliver ``slug_ratio``.

    The inverse of :func:`swept_slug_ratio` at ``expansion_speed = 0``, which is
    the question the chain optimiser actually poses: it picks a ``k``, and this
    says whether the geometry can supply it.

    Args:
        slug_ratio: The delivered ``k`` wanted.

    Returns:
        Required front radius as a fraction of the bore; above 1.0 the rigid
        front cannot supply it at all and the bag must be resized or the front
        must spread.
    """
    return float(np.sqrt(slug_ratio / full_bore_slug_ratio()))


#: Post-shock sound speed of the plume [km/s] against the front's own speed
#: [km/s], which is what sets how fast the front widens.
#:
#: **Provenance, recorded because this repository cannot regenerate it.**
#: Computed from ``puffsat_impact_simulation`` at commit ``0216a09`` with
#: ``PYTHONPATH=python``::
#:
#:     from puffsat import eos_water
#:     from puffsat.expansion import temperature_at
#:     rho = 4.0 * 213.0 / 659.6                      # 4x shock compression
#:     T = temperature_at(rho, 0.5 * (v * 1000) ** 2, eos_water.pressure_energy)
#:     c_s = eos_water.sound_speed(rho, T) / 1000.0
#:
#: The freshly shocked layer takes ``v^2/2`` of specific energy (the snowplow's
#: own inelastic-accretion assumption); ``eos_water`` inverts that to a
#: temperature through the full dissociation and ``O+ .. O8+`` ionisation
#: ladder, which is what keeps the temperature finite.  The 4x compression is
#: the weakest input and barely matters: 2x-16x moves ``c_s`` by 9%.
_SOUND_SPEED_TABLE = (
    (2.0, 0.8432),  # T =    1249 K
    (3.0, 1.1151),  # T =    2297 K
    (4.0, 1.3038),  # T =    3071 K
    (5.0, 1.4856),  # T =    3586 K
    (7.5, 2.0433),  # T =    4597 K
    (10.0, 2.8878),  # T =    5886 K
    (12.5, 5.0178),  # T =   12288 K
    (15.0, 6.0632),  # T =   18514 K
    (20.0, 8.0531),  # T =   26206 K
    (25.0, 10.4874),  # T =   34620 K
    (30.0, 13.0844),  # T =   46734 K
    (35.0, 15.3991),  # T =   58880 K
    (40.0, 18.3456),  # T =   74783 K
    (45.0, 20.8207),  # T =   92623 K
    (50.0, 23.6094),  # T =  110970 K
    (60.0, 27.7779),  # T =  148361 K
    (80.0, 35.8412),  # T =  204860 K
)


def shocked_sound_speed(front_speed: float) -> float:
    """Sound speed of the freshly shocked plume at a front moving at this speed.

    This is the physical estimate of ``c_exp``: a hot, high-pressure layer vents
    sideways at roughly its own sound speed.  Log-log interpolated on
    :data:`_SOUND_SPEED_TABLE`, which is smooth apart from the dissociation knee
    between 10 and 12.5 km/s -- and that knee is real, not noise.  It is where
    the shocked layer stops being warm steam and starts being a dissociated
    plasma, which roughly doubles ``c_s``.

    Args:
        front_speed: Speed of the front relative to the unswept cloud (km/s).

    Returns:
        Sound speed of the shocked material (km/s), clamped to the tabulated
        range at both ends.
    """
    speeds = np.array([row[0] for row in _SOUND_SPEED_TABLE])
    values = np.array([row[1] for row in _SOUND_SPEED_TABLE])
    clamped = float(np.clip(front_speed, speeds[0], speeds[-1]))
    return float(np.exp(np.interp(np.log(clamped), np.log(speeds), np.log(values))))


def self_consistent_slug_ratio(
    arrival_fraction: float = DESIGN_ARRIVAL_FRACTION,
    closing_speed: float = 45.58,
) -> float:
    """Slug ratio when the front widens at its own computed sound speed.

    The honest replacement for both bounds: :func:`swept_slug_ratio` at
    ``expansion_speed = 0`` is a rigid front, which the shocked temperature says
    is unphysical, and at a fixed ``c_exp`` it is a guess.  Here the widening
    rate ``dr/dx = c_exp/v`` is evaluated from the local front speed at every
    step, with both the speed and the sound speed falling as the front loads up.

    **The result is that the arrival radius almost does not matter.** The front
    reaches the bore wall within the first few metres of a 23.8 m column from
    any plausible arrival radius, and ``k`` lands at 7.2-8.7 rather than the
    rigid model's 0.03-8.7.

    Args:
        arrival_fraction: Front radius at entry, as a fraction of the bore.
        closing_speed: Impactor speed relative to the vehicle (km/s).

    Returns:
        The delivered slug ratio.

    Raises:
        ValueError: If ``arrival_fraction`` is not in (0, 1].
    """
    if not 0.0 < arrival_fraction <= 1.0:
        raise ValueError("arrival_fraction must be in (0, 1]")
    rho, mass = bag_density(), IMPACTOR_MASS
    radius = arrival_fraction * BORE_RADIUS
    step = COLUMN_LENGTH / _STEPS
    for _ in range(_STEPS):
        speed = closing_speed * IMPACTOR_MASS / mass
        mass += rho * np.pi * radius * radius * step
        if radius < BORE_RADIUS:
            radius = min(
                BORE_RADIUS,
                radius + (shocked_sound_speed(speed) / speed) * step,
            )
    return (mass - IMPACTOR_MASS) / IMPACTOR_MASS


# --- Item 13: the two-term nozzle mass -------------------------------------
#
# ``sec:space_mortgages`` names this as the thing that has to settle the
# nozzle's mass and says it "has not been run at this pulse".  Two terms:
#
#   structure   the virial floor, ``M >= E_B / sigma_eff``.  A magnet has to
#               hold its own field pressure, and the mass needed to do that is
#               set by the structure's specific strength alone.
#   conductor   REBCO tape.  A solenoid of length ``l`` making field ``B`` needs
#               ``NI = B l / mu0`` ampere-turns and each turn is ``2 pi r`` of
#               tape, so the mass runs as ``B r l``.
#
# The paper quotes only the structure term (3.7-11 t at 4.43 GJ, 10-30 t at
# 12.2 GJ) and calls it "tens of tonnes".  The conductor is what it left out.

#: Specific strength of the coil structure [J/kg].  The low end is a plain
#: build; the high end is bought by winding under pre-compression, so the
#: assembly starts squeezed and has more room to stretch before the tape
#: reaches its ~0.4% strain limit.  A magnet that comes apart between legs
#: cannot hold pre-stress and returns to the low end.
VIRIAL_STRENGTH_RANGE = (0.4e6, 1.2e6)

#: Mass per unit length of thin-substrate REBCO tape [kg/m].  4 mm wide and
#: ~60 um overall at ~8000 kg/m^3 gives ~1.9 g/m; 1.5 g/m is the thin-substrate
#: figure the ledger's item 13 quotes.
TAPE_LINEAR_DENSITY = 1.5e-3

#: Operating current per tape [A].  **Not stated by the paper**, so it is a
#: parameter here rather than a constant, and the band is what commercial REBCO
#: delivers at magnet operating conditions rather than at 77 K self-field:
#: roughly 300 A pessimistic, 500 A central, 1000 A for a well-cooled
#: high-field winding.  The conductor term is exactly inversely proportional to
#: it, so this is the dominant uncertainty in that term and it is reported as a
#: band for that reason.
TAPE_CURRENT_RANGE = (300.0, 500.0, 1000.0)

_MU_0 = 4.0e-7 * np.pi


def confinement_field(stored_energy: float, volume: float = BAG_VOLUME) -> float:
    """Field strength that stores ``stored_energy`` in ``volume``.

    Inverts ``E_B = B^2 V / (2 mu0)``.  At the standoff volume and the
    dissociated-neutral 4.43 GJ this returns 4.11 T, which is the row of
    ``tab:bag_sizing`` at that radius.

    Args:
        stored_energy: ``E_B`` the field must contain (J).
        volume: Enclosed volume (m^3).

    Returns:
        Field strength (T).
    """
    return float(np.sqrt(2.0 * _MU_0 * stored_energy / volume))


def virial_structure_mass(stored_energy: float, specific_strength: float) -> float:
    """Structural mass needed to contain a field, ``M = E_B / sigma_eff``.

    The virial theorem's floor: a magnet must react its own field pressure, and
    no geometry escapes it.  **It tracks contained energy only**, which is why
    reshaping the bag does not change it -- contained energy is the ``n R_g T``
    that shape leaves alone.

    Args:
        stored_energy: ``E_B`` (J).
        specific_strength: Structure's strength-to-density ratio (J/kg).

    Returns:
        Structural mass (kg).
    """
    return stored_energy / specific_strength


def conductor_mass(
    field: float,
    column_length: float,
    tape_current: float = 500.0,
    volume: float = BAG_VOLUME,
) -> float:
    """REBCO tape mass for a solenoid of this field over this column.

    ``NI = B l / mu0`` ampere-turns, each turn ``2 pi r`` of tape, with the bore
    from ``eq:bore_from_length``.  Substituting makes the mass run as
    ``B sqrt(V l / pi)``, so it grows as the **square root** of column length
    while the bore falls as its inverse square root: bore and conductor trade
    inversely, one for one, and halving the bore doubles the tape.

    Args:
        field: Confinement field (T).
        column_length: Length of the solenoid (m).
        tape_current: Operating current per tape (A).
        volume: Enclosed volume, which sets the bore (m^3).

    Returns:
        Tape mass (kg).
    """
    bore = np.sqrt(volume / (np.pi * column_length))
    turns = field * column_length / (_MU_0 * tape_current)
    return float(turns * 2.0 * np.pi * bore * TAPE_LINEAR_DENSITY)


def two_term_nozzle_mass(
    stored_energy: float,
    column_length: float = COLUMN_LENGTH,
    tape_current: float = 500.0,
    volume: float = BAG_VOLUME,
) -> Tuple[float, float, float]:
    """Structure plus conductor, the model ``sec:space_mortgages`` asks for.

    Args:
        stored_energy: ``E_B`` (J).
        column_length: Solenoid length (m).
        tape_current: Operating current per tape (A).
        volume: Enclosed volume (m^3).

    Returns:
        ``(structure_low, structure_high, conductor)`` in kg, with the
        structure band spanning :data:`VIRIAL_STRENGTH_RANGE`.
    """
    field = confinement_field(stored_energy, volume)
    return (
        virial_structure_mass(stored_energy, VIRIAL_STRENGTH_RANGE[1]),
        virial_structure_mass(stored_energy, VIRIAL_STRENGTH_RANGE[0]),
        conductor_mass(field, column_length, tape_current, volume),
    )


# --- Item 12: mirror stagnation pressure versus plug position --------------
#
# The field is a wall and the condition is pressure: it holds if ``B^2/2mu0``
# exceeds the plume's static pressure plus its ram pressure.  Where the plug
# sits decides which of those the wall sees, and the two answers differ by 7x
# in field -- 56 T against 7.6 T -- which is the difference between impossible
# and already-built.

#: Adiabatic index of the dissipated plume, back-solved from the paper's own
#: ratio of 6.7 at the ship-end plug (1.2, not the monatomic 5/3, because
#: dissociation and ionisation are soaking energy that a monatomic gas would
#: put into translation).
PLUME_GAMMA = 1.2
#: Ice plug at the projectile's entrance, half again the projectile's mass so
#: the residual jet does not punch through into the mist.
PLUG_MASS = 37.5


def mirror_stagnation(
    added_mass: float,
    dissipated_volume: float,
    closing_speed: float = 56.0,
    impactor_mass: float = IMPACTOR_MASS,
    gamma: float = PLUME_GAMMA,
) -> Tuple[float, float, float]:
    """Ram-to-static ratio, wall pressure and field for one plug position.

    Momentum is conserved through the merge, so ``M v = m_p w`` sets the
    post-merge speed and the dissipated energy is whatever kinetic energy that
    leaves behind.  **Sweeping more mass dissipates more energy but leaves it
    moving slower**, and the ram term falls faster than the static term rises,
    which is the whole result: a plug at the throat end faces a seventh of the
    field a plug at the ship end does.

    Args:
        added_mass: Mass swept before the plume meets the wall (kg).  The plug
            alone for a ship-end plug; the plug plus the whole slug column for
            a throat-end one.
        dissipated_volume: Volume the dissipated energy occupies at the wall
            (m^3) -- one bore-length for a ship-end plug, the whole bag for a
            throat-end one.
        closing_speed: Impactor speed relative to the vehicle (km/s).
        impactor_mass: Projectile mass (kg).
        gamma: Adiabatic index of the plume.

    Returns:
        ``(ram_to_static_ratio, wall_pressure_Pa, field_T)``.
    """
    speed = closing_speed * 1000.0
    merged = impactor_mass + added_mass
    drift = impactor_mass * speed / merged
    dissipated = 0.5 * impactor_mass * speed**2 - 0.5 * merged * drift**2
    static = (gamma - 1.0) * dissipated / dissipated_volume
    ram = merged * drift**2 / dissipated_volume
    pressure = static + ram
    return ram / static, pressure, float(np.sqrt(2.0 * _MU_0 * pressure))


def main() -> None:
    """Print the snowplow geometry, the mirror trade and the nozzle mass."""
    print("=== Item 11: what slug ratio the arrival radius delivers ===")
    print(f"full-bore sweep k_full = {full_bore_slug_ratio():.3f}")
    print(f"{'r/R':>8}{'r [m]':>9}{'% open':>9}{'k rigid':>10}{'k self-widening':>18}")
    for fraction in (DESIGN_ARRIVAL_FRACTION, 0.3, 0.5, 0.8, 0.9, 1.0):
        print(
            f"{fraction:>8.3f}{fraction * BORE_RADIUS:>9.2f}"
            f"{100 * fraction**2:>8.2f}%{swept_slug_ratio(fraction):>10.3f}"
            f"{self_consistent_slug_ratio(fraction):>18.3f}"
        )
    print(
        f"\nc_s of the shocked plume at 45.58 km/s is "
        f"{shocked_sound_speed(45.58):.1f} km/s, a "
        f"{np.degrees(np.arctan(shocked_sound_speed(45.58) / 45.58)):.1f} deg half-angle."
    )

    print("\n=== Item 12: mirror stagnation pressure versus plug position ===")
    print(f"{'plug position':<28}{'ram/static':>12}{'wall':>14}{'field':>10}")
    for label, added, volume in (
        ("ship end (1 m of bore)", PLUG_MASS, np.pi * BORE_RADIUS**2),
        ("throat end (whole bag)", 213.0, BAG_VOLUME),
    ):
        ratio, pressure, field = mirror_stagnation(added, volume)
        print(f"{label:<28}{ratio:>12.2f}{pressure / 1e6:>11.1f} MPa{field:>8.1f} T")
    print("The plug therefore sits at the projectile's entrance on both legs.")

    print("\n=== Item 13: two-term nozzle mass ===")
    print(f"{'E_B [GJ]':>9}{'B [T]':>8}{'structure':>20}{'conductor':>12}{'total':>18}")
    for energy in (4.427e9, 12.15e9):
        low, high, tape = two_term_nozzle_mass(energy)
        print(
            f"{energy / 1e9:>9.2f}{confinement_field(energy):>8.2f}"
            f"{low / 1000:>12.1f} -{high / 1000:>6.1f} t{tape / 1000:>10.1f} t"
            f"{(low + tape) / 1000:>12.1f} -{(high + tape) / 1000:>5.1f} t"
        )
    print(
        "\nThe paper quotes the structure term only. The conductor exceeds its\n"
        "optimistic end, so the floor is ~8 t rather than 3.7 t. Tape operating\n"
        "current is the dominant uncertainty and the paper does not state one."
    )


if __name__ == "__main__":
    main()
