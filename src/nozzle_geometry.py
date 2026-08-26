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

Seth, 2026-08-26: **the design assumption is a front spanning 0.8 of the bore.**

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

This module covers the ledger's item 11 only.  Items 12 (mirror stagnation
pressure) and 13 (two-term nozzle mass) are still owed and are not here.
"""

from __future__ import annotations

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
#: Arrival radius as a fraction of the bore, decided by Seth 2026-08-26.
DESIGN_ARRIVAL_FRACTION = 0.8
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
