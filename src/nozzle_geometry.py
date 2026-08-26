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
