"""What a shallow dive costs the *direct* architecture once its node is charged.

ADR 0023 read the **depth conduction crossing** as the split dive being what
makes a shallow dive flyable at all: the direct single-impulse departure aims
~31 degrees off the arriving stream, so it cools itself through the burn and
drops below ADR 0022's 79.56 km/s floor at 19.80 solar radii, short of the 23
that ADR 0022 recommends starting at.  The companion paper held that claim,
because the crossing moves ~1.9 solar radii per km/s of **perihelion burn** and
nobody had priced a larger one: 38.15 km/s reaches 23 solar radii for 6 percent
more burn, which would make the crossing a statement about one *tuning* rather
than about the architecture.

This module settles it (ADR
``0025-the-shallow-dive-is-affordable-and-the-stated-survival-is-not.md``).
Two results, and the second is the larger:

* **The direct route can fly shallow.**  38.10 km/s holds it conducting at ADR
  0022's 22.93 solar-radii pad floor, and the cycle still grows -- 2.505 per
  pass, doubling in 0.657 yr.  The extra 6 percent of burn costs 7.4 percent of
  node survival and about 1 percent of cycle time (a faster climb-out shortens
  the loop slightly, which partly offsets it).  So the crossing shows the direct
  route cannot fly shallow *at the paper's tuning*, **not** that the split is
  required.  The weaker claim the paper already holds is the true one
  (:func:`conducting_burn`, :func:`shallow_dive_row`).
* **The stated node survival is what actually flatters the shallow rows.**
  ``paper_resonant_dive_ledger()`` holds ``periapsis_survival`` at 0.60 across
  the whole depth dial.  Derived from the boost the way ``dive_node()`` does it,
  survival is 0.5895 at 4 solar radii but **0.2589 at 22.93 and 0.1627 at 32**,
  because the node's exhaust speed collapses with the arrival speed (68.1 to
  23.8 km/s).  Holding it at 0.60 therefore reports the shallow dive as
  **1.92x** better than it is at 22.93 solar radii and **3.07x** at 32 -- and,
  worse, reports doubling as nearly flat across the dial (0.305 to 0.309 yr),
  which is the illusion that makes backing the dive out look free.

The trade is one-sided in the burn, so the *minimum* conducting burn is always
the right one: a larger burn only buys conduction depth already achieved while
costing survival exponentially.  Every row here is therefore the direct
architecture's best case at that depth.

The model is circular and coplanar and all algebra is float (km, s, km/s, rad).
The dive geometry is :func:`bielliptic_dive_split.resonant_dive_at_depth`, which
now passes the perihelion burn through to the closure rather than dropping it;
the node is :func:`solar_dive_depth_trade.dive_node` unchanged, and the cycle is
:func:`jovian_solar_dive_cycle.cycle_growth_ledger` unchanged, so these rows sit
on the same device as ADR 0019-0022.

Run with ``make shallow-dive``.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from .bielliptic_dive_split import (
    _MU_SUN,
    _PERIAPSIS_BURN,
    _SOLAR_RADIUS,
    DEFAULT_EXPANSION_MARGIN,
    PAD_ADMISSIBLE_DEPTH,
    direct_departure_conduction_depth,
    resonant_dive_at_depth,
    returning_beam,
    speed_components_at_radius,
    two_node_closure,
)
from .jovian_solar_dive_cycle import (
    DEFAULT_PERIAPSIS_SURVIVAL,
    DEFAULT_SLUG_RATIO,
    CycleGrowthLedger,
    _FlybyParams,
    _powered_flyby_params,
    cycle_growth_ledger,
    launch_ledger_verdict,
)
from .solar_dive_depth_trade import (
    DEFAULT_JET_ENERGY_EFFICIENCY,
    DEFAULT_PERIAPSIS_SLUG_RATIO,
    DiveNode,
    dive_node,
)
from .two_leg_nozzle_sweep import PAYLOAD_FRACTION_AT_INTERCEPT, RETURN_FLOOR

#: Depths the command-line tables sweep (solar radii).  Starts at the crossing
#: the paper's own tuning reaches and runs out to where the burn search fails.
DEPTH_GRID: Sequence[float] = (19.8, PAD_ADMISSIBLE_DEPTH, 26.0, 29.0, 32.0, 36.0)

#: Beam-reuse fraction of the phased split closure this scores -- ADR 0023's
#: headline pattern, 8 interleaved chains with a beam at Earth every 0.2849 yr.
#: The *phased* family is the one to score on the pad, because its far node eats
#: the beam's leftovers and so costs no second launch; the partial split's far
#: node needs a dedicated delivery that nothing has priced.
SPLIT_REUSE: Tuple[int, int] = (5, 8)

#: Bracket for the split's pad crossing (solar radii).  Recorded per the ADR
#: 0007 lesson.  The margin falls monotonically with depth and the root sits
#: near 5.6, so this straddles it with room; the upper edge is well inside where
#: the three-condition closure still solves.
SPLIT_PAD_BRACKET: Tuple[float, float] = (4.0, 8.0)

#: Bracket for the conducting-burn search (km/s).  Recorded per the ADR 0007
#: lesson.  The lower edge is well below the paper's 35.98 tuning; the upper is
#: where the crossing leaves ``bielliptic_dive_split``'s 4-48 solar-radii
#: conduction bracket, so no root exists above it and widening buys nothing.
BURN_BRACKET: Tuple[float, float] = (18.0, 46.0)


@dataclass(frozen=True)
class ShallowDiveRow:
    """The direct architecture at one depth, flown at its cheapest conducting burn.

    Attributes:
        dive_solar_radii: Perihelion distance in solar radii.
        periapsis_burn: Boost taken at the dive perihelion (km/s).
        burn_ratio: That over the paper's 35.98 km/s tuning.
        conducts: Whether the Earth departure stays above the conduction floor.
        earth_boost: The single-impulse Earth boost (km/s).
        stream_excess: Earth-relative excess of the returning beam (km/s).
        cycle_years: Departure-to-re-intercept time (yr).
        node: The dive node, survival derived from the boost.
        stated_survival: The fixed survival the paper's ledger uses instead.
        survival_ratio: ``stated_survival`` over the derived one -- how much the
            fixed value flatters this depth.
        growth: The cycle scored on the derived survival.
        doubling_stated: What the same cycle reports on the fixed 0.60.
        flattery: ``growth.doubling_years`` over ``doubling_stated``.
        pad_margin: Returned mass per pad kilogram over ADR 0021's 1/15 floor,
            on the derived survival.
    """

    dive_solar_radii: float
    periapsis_burn: float
    burn_ratio: float
    conducts: bool
    earth_boost: float
    stream_excess: float
    cycle_years: float
    node: DiveNode
    stated_survival: float
    survival_ratio: float
    growth: CycleGrowthLedger
    doubling_stated: float
    flattery: float
    pad_margin: float


def conducting_burn(
    dive_solar_radii: float,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    bracket: Tuple[float, float] = BURN_BRACKET,
    params: Optional[_FlybyParams] = None,
) -> Optional[float]:
    """Cheapest perihelion burn that keeps the direct departure conducting here.

    The **depth conduction crossing** rises monotonically with the burn (~1.9
    solar radii per km/s), so this inverts it.  The minimum is what matters: a
    larger burn buys conduction depth already in hand while costing node
    survival exponentially, so the cheapest conducting burn is the direct
    architecture's best play at every depth.

    Args:
        dive_solar_radii: Perihelion the departure must still conduct at.
        expansion_margin: Headroom demanded above the **expansion floor**.
        bracket: Burns to bisect between (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The burn in km/s, or None when no burn in ``bracket`` reaches this depth.
    """
    p = params if params is not None else _powered_flyby_params()

    def residual(burn: float) -> float:
        crossing = direct_departure_conduction_depth(
            expansion_margin=expansion_margin,
            periapsis_burn=float(burn),
            params=p,
        )
        return (crossing if crossing is not None else -1.0e3) - dive_solar_radii

    low, high = bracket
    if residual(low) * residual(high) > 0.0:
        return None
    return float(brentq(residual, low, high, xtol=1e-8))


def shallow_dive_row(
    dive_solar_radii: float,
    periapsis_burn: Optional[float] = None,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    periapsis_slug_ratio: float = DEFAULT_PERIAPSIS_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    stated_survival: float = DEFAULT_PERIAPSIS_SURVIVAL,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    params: Optional[_FlybyParams] = None,
) -> Optional[ShallowDiveRow]:
    """Score the direct single-impulse dive end to end, node charged for its boost.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        periapsis_burn: Boost at the dive perihelion (km/s).  ``None`` picks the
            cheapest burn that still conducts at this depth.
        slug_ratio: Departure slug ratio.
        periapsis_slug_ratio: Slug ratio at the dive node.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        stated_survival: The fixed node survival to compare against.
        expansion_margin: Headroom demanded above the **expansion floor**.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`ShallowDiveRow`, or None when no conducting burn exists or
        the departure burn delivers nothing.
    """
    p = params if params is not None else _powered_flyby_params()
    burn = (
        periapsis_burn
        if periapsis_burn is not None
        else conducting_burn(dive_solar_radii, expansion_margin, params=p)
    )
    if burn is None:
        return None
    perihelion = dive_solar_radii * _SOLAR_RADIUS
    aphelion, boost, aim, years = resonant_dive_at_depth(dive_solar_radii, burn)
    beam = returning_beam(0.5 * (aphelion + perihelion), perihelion, burn)
    tangential, radial = speed_components_at_radius(beam, _MU_SUN, p.r_earth_orbit)
    excess = float(np.hypot(radial, tangential - p.v_earth_orbit))
    axis = float(np.degrees(np.arctan2(radial, tangential - p.v_earth_orbit)))
    node = dive_node(
        dive_solar_radii, burn, periapsis_slug_ratio, jet_energy_efficiency, p
    )
    try:
        derived = cycle_growth_ledger(
            f"direct dive / {dive_solar_radii:g} R_sun / burn {burn:.2f}",
            boost,
            aim,
            years,
            excess,
            axis,
            slug_ratio,
            jet_energy_efficiency,
            node.survival,
            p,
        )
        stated = cycle_growth_ledger(
            "stated",
            boost,
            aim,
            years,
            excess,
            axis,
            slug_ratio,
            jet_energy_efficiency,
            stated_survival,
            p,
        )
    except ValueError:
        return None
    crossing = direct_departure_conduction_depth(
        expansion_margin=expansion_margin, periapsis_burn=burn, params=p
    )
    return ShallowDiveRow(
        dive_solar_radii=dive_solar_radii,
        periapsis_burn=burn,
        burn_ratio=burn / _PERIAPSIS_BURN,
        conducts=crossing is not None and crossing >= dive_solar_radii - 1e-6,
        earth_boost=boost,
        stream_excess=excess,
        cycle_years=years,
        node=node,
        stated_survival=stated_survival,
        survival_ratio=stated_survival / node.survival,
        growth=derived,
        doubling_stated=stated.doubling_years,
        flattery=derived.doubling_years / stated.doubling_years,
        pad_margin=launch_ledger_verdict(derived, p).stated_margin,
    )


# --------------------------------------------------------------------------
# The split dive on the pad, which is the only scoreboard it ever won
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPadRow:
    """The phased split dive charged in ADR 0021's pad currency, at one depth.

    ADR 0023's headline is that the split "buys the pad, not the clock" -- it is
    slower but launches far less mass.  That claim was only ever made at 4 solar
    radii and only in launched-slug-per-delivered-kilogram, never in ADR 0021's
    committed form: returned mass per kilogram off the pad, against a 1/15 floor.
    This scores it there, across the depth dial, with the node's survival derived
    rather than stated.

    Attributes:
        dive_solar_radii: Perihelion distance in solar radii.
        closes: Whether the three-condition phased closure solves at this depth.
        delivered_fraction: Mass surviving both of the split's burns.
        node_survival: Derived survival at the dive node.
        returned_per_pad_kg: ``0.25 * delivered * survival``.
        pad_margin: That over ADR 0021's 1/15 floor.
        clears: Whether the split earns its launch here.
        doubling_years: The split's doubling time at this depth.
    """

    dive_solar_radii: float
    closes: bool
    delivered_fraction: float
    node_survival: float
    returned_per_pad_kg: float
    pad_margin: float
    clears: bool
    doubling_years: float


def split_pad_row(
    dive_solar_radii: float,
    periapsis_burn: float = _PERIAPSIS_BURN,
    reuse: Tuple[int, int] = SPLIT_REUSE,
    periapsis_slug_ratio: float = DEFAULT_PERIAPSIS_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> Optional[SplitPadRow]:
    """Charge the phased split dive for its launched mass at one depth.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        periapsis_burn: Boost taken at the dive perihelion (km/s).
        reuse: Beam-reuse fraction of the closure to solve.
        periapsis_slug_ratio: Slug ratio at the dive node.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitPadRow`, or None when the phased closure finds no
        admissible root at this depth.
    """
    p = params if params is not None else _powered_flyby_params()
    node = dive_node(
        dive_solar_radii, periapsis_burn, periapsis_slug_ratio, jet_energy_efficiency, p
    )
    closure = two_node_closure(
        reuse[0],
        reuse[1],
        revolutions=1,
        dive_perihelion=dive_solar_radii * _SOLAR_RADIUS,
        periapsis_burn=periapsis_burn,
        periapsis_survival=node.survival,
        params=p,
    )
    if closure is None:
        return None
    delivered = closure.ledger.delivered_fraction
    returned = PAYLOAD_FRACTION_AT_INTERCEPT * delivered * node.survival
    return SplitPadRow(
        dive_solar_radii=dive_solar_radii,
        closes=True,
        delivered_fraction=delivered,
        node_survival=node.survival,
        returned_per_pad_kg=returned,
        pad_margin=returned / RETURN_FLOOR,
        clears=returned >= RETURN_FLOOR,
        doubling_years=closure.ledger.doubling_years,
    )


def split_pad_crossing(
    bracket: Tuple[float, float] = SPLIT_PAD_BRACKET,
    params: Optional[_FlybyParams] = None,
) -> Optional[float]:
    """Depth at which the split stops paying for its own launch.

    Bisected on the constraint rather than sampled near it, per ADR 0022's
    lesson.

    Args:
        bracket: Depths to bisect between, in solar radii.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The crossing depth in solar radii, or None when ``bracket`` does not
        straddle one.
    """
    p = params if params is not None else _powered_flyby_params()

    def residual(depth: float) -> float:
        row = split_pad_row(float(depth), params=p)
        return float("nan") if row is None else row.pad_margin - 1.0

    low, high = bracket
    if not np.isfinite(residual(low)) or not np.isfinite(residual(high)):
        return None
    if residual(low) * residual(high) > 0.0:
        return None
    return float(brentq(residual, low, high, xtol=1e-4))


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _burn_table(depths: Sequence[float] = DEPTH_GRID) -> str:
    """The burn each depth needs, and what the node pays for it.

    Args:
        depths: Dive depths to sweep (solar radii).

    Returns:
        A fixed-width table.
    """
    lines = [
        f"{'depth':>7} {'burn':>8} {'x paper':>8} {'cycle yr':>9} {'exhaust':>8} "
        f"{'survival':>9} {'growth':>8} {'doubling':>9} {'pad':>7}",
        "-" * 82,
    ]
    for depth in depths:
        row = shallow_dive_row(depth)
        if row is None:
            lines.append(f"{depth:7.2f}   no conducting burn in the bracket")
            continue
        lines.append(
            f"{depth:7.2f} {row.periapsis_burn:8.3f} {row.burn_ratio:7.3f}x "
            f"{row.cycle_years:9.4f} {row.node.exhaust_speed:8.2f} "
            f"{row.node.survival:9.4f} {row.growth.return_per_impactor_kg:8.3f} "
            f"{row.growth.doubling_years:9.4f} {row.pad_margin:7.3f}"
        )
    return "\n".join(lines)


def _flattery_table(depths: Sequence[float] = DEPTH_GRID) -> str:
    """What holding node survival at 0.60 hides, depth by depth.

    Args:
        depths: Dive depths to sweep (solar radii).

    Returns:
        A fixed-width table.
    """
    lines = [
        f"{'depth':>7} {'derived':>9} {'stated':>7} {'ratio':>7} | "
        f"{'doubling derived':>17} {'stated':>9} {'flattered by':>13}",
        "-" * 80,
    ]
    for depth in (4.0, *depths):
        row = shallow_dive_row(
            depth, periapsis_burn=_PERIAPSIS_BURN if depth <= 19.81 else None
        )
        if row is None:
            continue
        lines.append(
            f"{depth:7.2f} {row.node.survival:9.4f} {row.stated_survival:7.2f} "
            f"{row.survival_ratio:6.2f}x | {row.growth.doubling_years:17.4f} "
            f"{row.doubling_stated:9.4f} {row.flattery:12.3f}x"
        )
    return "\n".join(lines)


def _pad_table(depths: Sequence[float] = DEPTH_GRID) -> str:
    """Both architectures on ADR 0021's committed pad currency.

    Args:
        depths: Dive depths to sweep (solar radii).

    Returns:
        A fixed-width table.
    """
    lines = [
        f"{'depth':>7} {'split margin':>13} {'direct margin':>14} {'split edge':>11} "
        f"{'split clears':>13} {'split doubling':>15}",
        "-" * 78,
    ]
    for depth in (4.0, 8.0, 16.0, *depths):
        split = split_pad_row(depth)
        direct = shallow_dive_row(
            depth, periapsis_burn=_PERIAPSIS_BURN if depth <= 19.81 else None
        )
        if split is None or direct is None:
            continue
        lines.append(
            f"{depth:7.2f} {split.pad_margin:13.3f} {direct.pad_margin:14.3f} "
            f"{split.pad_margin / direct.pad_margin:10.2f}x "
            f"{('YES' if split.clears else 'no'):>13} "
            f"{split.doubling_years:15.4f}"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Price a shallow dive for the direct architecture, with the node "
            "charged for its own perihelion burn."
        )
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=list(DEPTH_GRID),
        help="dive depths to sweep, in solar radii",
    )
    return parser


def main() -> None:
    """Print the conducting-burn trade and the stated-survival comparison."""
    args = _parser().parse_args()
    depths: List[float] = list(args.depths)

    print(
        "Does the direct architecture need the split to fly shallow?\n\n"
        "1. The cheapest burn that keeps its departure conducting, with the\n"
        "   node charged for taking it.\n"
    )
    print(_burn_table(depths))
    crossing = direct_departure_conduction_depth()
    row = shallow_dive_row(PAD_ADMISSIBLE_DEPTH)
    if crossing is not None and row is not None:
        print(
            f"\n   At the paper's {_PERIAPSIS_BURN:.2f} km/s the crossing is "
            f"{crossing:.2f} R_sun. At {row.periapsis_burn:.2f} "
            f"({row.burn_ratio:.3f}x)\n"
            f"   it reaches ADR 0022's {PAD_ADMISSIBLE_DEPTH:.2f} pad floor, and "
            f"the cycle still grows\n"
            f"   ({row.growth.return_per_impactor_kg:.3f} per pass, doubling "
            f"{row.growth.doubling_years:.4f} yr). So the crossing is a statement\n"
            "   about the TUNING, not about the architecture: the split is not\n"
            "   required for a shallow dive to exist."
        )

    print(
        "\n\n2. But holding node survival at 0.60 across the dial is what really\n"
        "   flatters these rows.\n"
    )
    print(_flattery_table(depths))
    print(
        "\n   The node's exhaust speed collapses with the arrival speed, so a\n"
        "   gentler node is a less efficient one (ADR 0020's derived survival).\n"
        "   On the fixed 0.60 the doubling time is nearly flat across the whole\n"
        "   depth dial, which is the illusion that makes backing the dive out\n"
        "   look free. Derived, it is 2-3x worse."
    )

    print("\n\n3. And the split dive, on the only scoreboard it ever won: the pad.\n")
    print(_pad_table(depths))
    crossing = split_pad_crossing()
    if crossing is not None:
        print(
            f"\n   The split stops paying for its own launch at "
            f"{crossing:.2f} R_sun. Its pad\n"
            "   advantage is real but concentrated on deep dives: 1.80x the\n"
            "   direct route's margin at 4 R_sun, falling to 1.09x at 32. So the\n"
            "   architecture that 'buys the pad' buys it only where the thermal\n"
            "   case is worst, and neither route earns its launch at the shallow\n"
            "   end ADR 0022 recommends."
        )


if __name__ == "__main__":
    main()
