"""Magnetic nozzle on both legs, against the paper's pusher plate on the first.

ADR ``0013`` prices the growth push with a pusher plate (elasticity ``f``) and
only the head-on departure burn with a magnetic nozzle (recovery ``e``).  This
module asks the question that grid never posed: what if the *overtaking* leg is
a nozzle too, carrying its own water slug?

The change to the ledger is one substitution.  ADR 0013's

    g [ (1+sigma)/(r M d1) + sigma/k ] = 1

keeps its shape; only ``M``, the parked mass per arriving impactor kilogram,
moves from the plate's ``2 f / Lambda`` to the overtaking nozzle's
``k1 / sigma1``.  ``same_cycle_nozzle()`` takes both forms, so nothing is
re-derived here (see ``src/nozzle_analysis.py``).

Two admissibility conditions come with the second nozzle, and they bind in
complementary halves of the grid:

* **Plume ignition.**  A magnetic nozzle steers only a conductor, so the blob
  must reach 15,000 K at the *coldest* instant of each burn -- the end of the
  overtaking push, the start of the head-on burn.  That is a closed window in
  ``k``, not a ceiling (``src/plume_thermal.py``), and it governs the high-
  recovery corner.
* **Ground-launch economy.**  The slug for both legs is lofted from Earth, so
  the two nozzles compete for the same launched kilograms.  The floor below
  fixes how much of a liftoff must survive to return, and it governs the
  low-recovery half.

Run it with ``make two-leg``; compute-intensive, not part of ``make all``.

See ``docs/adr/0014-magnetic-nozzle-on-both-legs.md``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.astro_constants import STD_FUDGE_FACTOR
from src.nozzle_analysis import NozzlePricing, apoapsis_reversal_dv, same_cycle_nozzle
from src.plume_thermal import (
    NOZZLE_FLOOR_TEMPERATURE,
    chemistry_efficiency,
    slug_ratio_window,
)
from src.two_wave_growth import (
    DEFAULT_SPLIT_DAYS,
    VE_METHALOX,
    TwoWaveCycle,
    _cycle_periapsis_speed,
    adaptive_two_wave_cycles,
    coldest_closing_speeds,
    fleet_ignition_windows,
)

# --- Ground-launch ledger -------------------------------------------------
#
# The loop consumes freshly launched mass every cycle: the arriving wave pushes
# a payload that was lofted from Earth, and what survives both legs is what
# returns.  So "kilograms returned per kilogram off the pad" is a second
# currency alongside the growth rate, and it is the one that has to beat
# reusable rockets.
#
#: Propellant fraction of the ground rocket that lobs the payload to the
#: intercept altitude.  Exactly a 4.09 km/s lob at the 380 s methalox vacuum
#: Isp, which covers a 200 km zero-velocity toss with gravity and drag losses.
LAUNCH_PROPELLANT_FRACTION = 2.0 / 3.0
#: Launcher dry mass, as a fraction of what is left after propellant.  Puts the
#: stage's structural coefficient at ``dry / (dry + propellant)`` = 11%.
LAUNCHER_DRY_FRACTION = 0.25
#: What therefore reaches the intercept point, per kilogram off the pad.
PAYLOAD_FRACTION_AT_INTERCEPT = (1.0 - LAUNCH_PROPELLANT_FRACTION) * (
    1.0 - LAUNCHER_DRY_FRACTION
)
#: Floor on returned mass per kilogram launched from the ground.  Set against
#: reusable rockets: roughly 150 t to LEO from a ~5000 t liftoff is ~1/33, and
#: the returned mass here is at ~60 km/s Earth-relative rather than sitting in
#: LEO, so matching that ratio is already a win.
RETURN_FLOOR = 1.0 / 15.0

# --- Search boxes (recorded per the ADR 0007 lesson) ----------------------
#
# Both axes are searched on a log-spaced grid clipped to the fleet-wide plume
# window; each round re-spans both grids between the neighbours of the current
# best point, so the box shrinks by roughly the grid size per round.
_K_GRID_POINTS = 32
_REFINE_ROUNDS = 4
#: Recovery grids swept on each leg.
DEFAULT_RECOVERIES: Tuple[float, ...] = (0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
#: Plate elasticities the crossover search may consider.  Above 1.0 is
#: unphysical and is reported as such rather than clipped.
_PLATE_BRACKET = (0.02, 4.0)
#: Split gap the comparison is flown at, matching ADR 0013's headline table.
#: Re-exported from ``two_wave_growth`` rather than restated, so the sweep and a
#: direct ``adaptive_two_wave_cycles()`` call can never fly different chains.


def returned_launch_fraction(cycle: TwoWaveCycle, pricing: NozzlePricing) -> float:
    """Mass returning from Jupiter per kilogram lifted off the ground.

    Charges, in order: the ground rocket's propellant and dry mass, the slug
    consumed on each leg, the apoapsis reversal, and the growth wave's
    correction burns.  Everything the cycle actually throws away.

    Args:
        cycle: The flown cycle, which carries the correction burns.
        pricing: That cycle's pricing, which carries both legs' slug spend.

    Returns:
        Returned mass per kilogram of liftoff mass.
    """
    reversal = float(
        np.exp(
            -apoapsis_reversal_dv(cycle.split_days * u.day).to_value(u.km / u.s)
            / VE_METHALOX
        )
    )
    correction = float(
        np.exp(-(cycle.growth_wave_burn + cycle.nozzle_wave_dsm) / VE_METHALOX)
    )
    return (
        PAYLOAD_FRACTION_AT_INTERCEPT * reversal * correction / pricing.mass_multiplier
    )


@dataclass(frozen=True)
class TwoLegChain:
    """A flown chain priced with a nozzle (or a plate) on the growth push.

    Attributes:
        growth_recovery: Overtaking-leg recovery ``e1``; None for a plate.
        fudge: Plate elasticity, used only when ``growth_recovery`` is None.
        recovery: Head-on leg recovery ``e2``.
        growth_slug_ratio: Fleet-wide ``k1``; zero for a plate.
        slug_ratio: Fleet-wide ``k2``.
        rate: Chain e-foldings per year.
        doubling: Years to double at that rate.
        total_growth: Product of the per-cycle growth factors.
        horizon_years: Departure of the first cycle to return of the last.
        worst_return_fraction: Tightest per-cycle returned-mass fraction.
        ignition_limited: Whether ``k1`` sits on its plume-window ceiling.
        launch_limited: Whether the returned-mass floor is what stopped ``k1``.
    """

    growth_recovery: Optional[float]
    fudge: float
    recovery: float
    growth_slug_ratio: float
    slug_ratio: float
    rate: float
    doubling: float
    total_growth: float
    horizon_years: float
    worst_return_fraction: float
    ignition_limited: bool
    launch_limited: bool


def _score(
    cycles: Sequence[TwoWaveCycle],
    growth_recovery: Optional[float],
    fudge: float,
    recovery: float,
    k1: float,
    k2: float,
    geometric_efficiency: Optional[float] = None,
) -> Tuple[float, float]:
    """Chain log-growth and worst-cycle return fraction at one ``(k1, k2)``.

    Args:
        cycles: The flown cycles.
        growth_recovery: Overtaking-leg recovery, or None for a pusher plate.
        fudge: Plate elasticity, used only for the plate.
        recovery: Head-on leg recovery.
        k1: Growth-push slug ratio (ignored for a plate).
        k2: Departure-burn slug ratio.
        geometric_efficiency: Charge the frozen-dissociation toll, sweeping
            this as the remaining jet efficiency.  None reproduces the
            published arithmetic.  **A plate owes no chemistry**, so leg 1 is
            tolled only when it carries a nozzle -- which is exactly why the
            toll penalises the two-leg option and not the plate one.

    Returns:
        ``(sum of log growth, worst per-cycle returned fraction)``.
    """
    total, worst = 0.0, float("inf")
    for cycle in cycles:
        jet1 = jet2 = 1.0
        if geometric_efficiency is not None:
            push_end, burn_start = coldest_closing_speeds(cycle)
            jet2 = geometric_efficiency * chemistry_efficiency(
                burn_start * u.km / u.s, k2
            )
            if growth_recovery is not None:
                jet1 = geometric_efficiency * chemistry_efficiency(
                    push_end * u.km / u.s, k1
                )
        priced = same_cycle_nozzle(
            growth_collision_speed=cycle.growth_wave_v_b,
            growth_wave_burn=cycle.growth_wave_burn + cycle.nozzle_wave_dsm,
            nozzle_collision_speed=cycle.nozzle_wave_v_b,
            departure_dv=cycle.departure_burn,
            cycle=cycle.period_years,
            exhaust_speed=VE_METHALOX,
            recovery=recovery,
            slug_ratio=k2,
            fudge=fudge,
            reversal_period=cycle.split_days * u.day,
            growth_slug_ratio=None if growth_recovery is None else k1,
            growth_recovery=growth_recovery,
            jet_efficiency=jet2,
            growth_jet_efficiency=jet1,
        )
        if priced.growth <= 0.0:
            # Below the head-on leg's forward-thrust floor the burn produces
            # net *backward* impulse, so no slug ratio buys the delta-v and the
            # cycle delivers nothing.  Report it as such rather than letting
            # log(0) raise a warning on every grid point of a dead corner.
            return float("-inf"), 0.0
        total += float(np.log(priced.growth))
        worst = min(worst, returned_launch_fraction(cycle, priced))
    return total, worst


def price_chain_two_leg(
    cycles: Sequence[TwoWaveCycle],
    recovery: float,
    growth_recovery: Optional[float] = None,
    fudge: float = STD_FUDGE_FACTOR,
    temperature: u.Quantity = NOZZLE_FLOOR_TEMPERATURE,
    return_floor: float = RETURN_FLOOR,
    geometric_efficiency: Optional[float] = None,
) -> Optional[TwoLegChain]:
    """Search the fleet-wide slug ratios that maximise the chain's growth rate.

    One ``(k1, k2)`` flies every cycle, so the search is over the intersection
    of the per-cycle ignition windows, subject to every cycle clearing the
    returned-mass floor.  A losing chain is *not* filtered out -- per CONTEXT.md
    a negative rate is the gradient, not an infeasible point -- but a chain that
    cannot ignite or cannot pay for its own launch is genuinely inadmissible and
    returns None.

    Args:
        cycles: The flown cycles, in order.
        recovery: Head-on leg recovery ``e2``.
        growth_recovery: Overtaking leg recovery ``e1``; None prices the
            paper's pusher plate on that leg instead.
        fudge: Plate elasticity, used only when ``growth_recovery`` is None.
        temperature: Plume temperature both nozzles require.
        return_floor: Minimum returned mass per kilogram of liftoff mass.
        geometric_efficiency: Charge the frozen-dissociation toll on whichever
            legs carry a nozzle, sweeping this as the remaining jet efficiency;
            see :func:`_score`.  **Pass ``recovery = 1.0`` (and
            ``growth_recovery = 1.0``) when sweeping it**, since those derate
            from outside the momentum debit and this derates from inside:
            charging both applies the toll twice.

    Returns:
        The best admissible chain, or None if no slug ratios are admissible.

    Raises:
        ValueError: If no cycles are supplied.
    """
    if not cycles:
        raise ValueError("cycles must not be empty")
    window1, window2 = fleet_ignition_windows(cycles, temperature)
    if window2 is None or (growth_recovery is not None and window1 is None):
        return None
    horizon = sum(cycle.period_years for cycle in cycles)

    def axis(window: Tuple[float, float], lo: float, hi: float) -> np.ndarray:
        low = max(window[0], lo)
        high = min(window[1], hi)
        if high <= low:
            return np.array([np.clip(window[0], lo, hi)])
        return np.logspace(np.log10(low), np.log10(high), _K_GRID_POINTS)

    grid1 = (
        np.array([0.0])
        if growth_recovery is None
        else axis(window1 if window1 else (0.0, 0.0), 0.0, np.inf)
    )
    grid2 = axis(window2, 0.0, np.inf)
    best: Optional[Tuple[float, float, float, float]] = None
    for _ in range(_REFINE_ROUNDS):
        for k1 in grid1:
            for k2 in grid2:
                total, worst = _score(
                    cycles,
                    growth_recovery,
                    fudge,
                    recovery,
                    float(k1),
                    float(k2),
                    geometric_efficiency,
                )
                if worst < return_floor:
                    continue
                if best is None or total > best[0]:
                    best = (total, float(k1), float(k2), worst)
        if best is None:
            return None
        grid1 = _tighten(grid1, best[1])
        grid2 = _tighten(grid2, best[2])
    assert best is not None
    total, k1, k2, worst = best
    rate = total / horizon
    ceiling1 = window1[1] if window1 else 0.0
    return TwoLegChain(
        growth_recovery=growth_recovery,
        fudge=fudge,
        recovery=recovery,
        growth_slug_ratio=k1,
        slug_ratio=k2,
        rate=rate,
        doubling=float("inf") if rate <= 0.0 else float(np.log(2.0) / rate),
        total_growth=float(np.exp(total)),
        horizon_years=horizon,
        worst_return_fraction=worst,
        ignition_limited=growth_recovery is not None and k1 > 0.995 * ceiling1,
        launch_limited=worst < 1.02 * return_floor,
    )


def _tighten(grid: np.ndarray, centre: float) -> np.ndarray:
    """Re-span a log grid around its current best point.

    Args:
        grid: The grid just searched.
        centre: Its maximising value.

    Returns:
        A grid of the same size spanning the neighbours of ``centre``.
    """
    if grid.size <= 1:
        return grid
    index = int(np.argmin(np.abs(grid - centre)))
    low = float(grid[max(0, index - 1)])
    high = float(grid[min(grid.size - 1, index + 1)])
    if high <= low:
        return grid
    return np.logspace(np.log10(low), np.log10(high), grid.size)


def equivalent_plate_elasticity(
    cycles: Sequence[TwoWaveCycle], chain: TwoLegChain
) -> Optional[float]:
    """The pusher plate that would match this chain, at the same ``e2``.

    Converts a two-nozzle cell onto the incumbent's own axis, which is what
    makes the two architectures directly comparable: a value above 1.0 means no
    physically possible plate matches the nozzle, since ``f`` is a restitution
    coefficient.

    Args:
        cycles: The flown cycles.
        chain: A two-leg chain to convert.

    Returns:
        The matching elasticity, or None if the bracket does not contain it.
    """

    def gap(f: float) -> float:
        plate = price_chain_two_leg(cycles, chain.recovery, None, f)
        return -float("inf") if plate is None else plate.rate - chain.rate

    low, high = _PLATE_BRACKET
    if gap(low) > 0.0 or gap(high) < 0.0:
        return None
    return float(brentq(gap, low, high, xtol=1e-3))


@dataclass(frozen=True)
class TwoLegSweep:
    """The full comparison between the two architectures.

    Attributes:
        grid: One row per ``(e1, e2)`` pair, with both slug ratios, the growth
            it delivers, which limit bound it, and the plate it is worth.
        plate: One row per ``(f, e2)`` pair for the incumbent, same cycles.
        tolled: The matched diagonal ``e1 = e2 = eta_geom`` re-priced with the
            ADR 0016 dissociation toll, against a plate chain on the same
            ``eta_geom``.  This is what ``tab:two_leg_growth`` regenerates to,
            and the column that matters is ``nozzle_over_plate``: the toll
            reaches the two-leg option through *both* legs and the plate option
            through only one, so it is not a wash between architectures.
        windows: Fleet-wide ignition windows on each leg.
        split_days: Split gap the chain was flown at.
        horizon_years: Length of the flown chain.
        cycles: The flown cycles, for reporting the per-cycle cold ends.
    """

    grid: pd.DataFrame
    plate: pd.DataFrame
    tolled: pd.DataFrame
    windows: Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]
    split_days: float
    horizon_years: float
    cycles: List[TwoWaveCycle]


def analyze_two_leg_nozzle(
    recoveries: Sequence[float] = DEFAULT_RECOVERIES,
    split_days: float = DEFAULT_SPLIT_DAYS,
    fudges: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
) -> TwoLegSweep:
    """Fly the adaptive cadence and sweep both legs' recoveries against it.

    Args:
        recoveries: Recovery fractions swept on each leg independently.
        split_days: Split gap, which is also the parking-orbit period.
        fudges: Plate elasticities for the incumbent comparison table.

    Returns:
        The comparison tables.
    """
    cycles = adaptive_two_wave_cycles(split_days=split_days)
    windows = fleet_ignition_windows(cycles)
    rows = []
    for e1 in recoveries:
        for e2 in recoveries:
            chain = price_chain_two_leg(cycles, e2, e1)
            if chain is None:
                rows.append(
                    {
                        "growth_recovery": e1,
                        "recovery": e2,
                        "k1": float("nan"),
                        "k2": float("nan"),
                        "e_foldings_per_year": float("nan"),
                        "doubling_years": float("nan"),
                        "returned_per_liftoff": float("nan"),
                        "limit": "no ignition",
                        "equivalent_plate_f": float("nan"),
                    }
                )
                continue
            limits = []
            if chain.ignition_limited:
                limits.append("plume")
            if chain.launch_limited:
                limits.append("launch")
            equivalent = equivalent_plate_elasticity(cycles, chain)
            rows.append(
                {
                    "growth_recovery": e1,
                    "recovery": e2,
                    "k1": chain.growth_slug_ratio,
                    "k2": chain.slug_ratio,
                    "e_foldings_per_year": chain.rate,
                    "doubling_years": chain.doubling,
                    "returned_per_liftoff": chain.worst_return_fraction,
                    "limit": "+".join(limits) if limits else "interior",
                    "equivalent_plate_f": (
                        float("nan") if equivalent is None else equivalent
                    ),
                }
            )
    plate_rows = []
    for f in fudges:
        for e2 in recoveries:
            chain = price_chain_two_leg(cycles, e2, None, f)
            plate_rows.append(
                {
                    "fudge": f,
                    "recovery": e2,
                    "k2": float("nan") if chain is None else chain.slug_ratio,
                    "e_foldings_per_year": (
                        float("nan") if chain is None else chain.rate
                    ),
                    "doubling_years": float("nan") if chain is None else chain.doubling,
                    "returned_per_liftoff": (
                        float("nan") if chain is None else chain.worst_return_fraction
                    ),
                }
            )
    tolled_rows = []
    for geometric in recoveries:
        nozzle = price_chain_two_leg(cycles, 1.0, 1.0, geometric_efficiency=geometric)
        plate = price_chain_two_leg(
            cycles, 1.0, None, STD_FUDGE_FACTOR, geometric_efficiency=geometric
        )
        published = price_chain_two_leg(cycles, geometric, geometric)
        tolled_rows.append(
            {
                "eta_geom": geometric,
                "k1": float("nan") if nozzle is None else nozzle.growth_slug_ratio,
                "k2": float("nan") if nozzle is None else nozzle.slug_ratio,
                "nozzle_growth": (
                    float("nan") if nozzle is None else nozzle.total_growth
                ),
                "plate_growth": float("nan") if plate is None else plate.total_growth,
                "nozzle_over_plate": (
                    float("nan")
                    if nozzle is None or plate is None or plate.total_growth <= 0.0
                    else nozzle.total_growth / plate.total_growth
                ),
                "published_nozzle_growth": (
                    float("nan") if published is None else published.total_growth
                ),
            }
        )

    return TwoLegSweep(
        grid=pd.DataFrame(rows),
        plate=pd.DataFrame(plate_rows),
        tolled=pd.DataFrame(tolled_rows),
        windows=windows,
        split_days=split_days,
        horizon_years=sum(c.period_years for c in cycles),
        cycles=list(cycles),
    )


def _format_plain(frame: pd.DataFrame, formats: Dict[str, str]) -> str:
    """Render a flat table with per-column format specs.

    Args:
        frame: The table to render.
        formats: Column name to format spec.

    Returns:
        The rendered table.
    """
    shown = frame.copy()
    for column, spec in formats.items():
        shown[column] = shown[column].map(
            lambda v, s=spec: "--" if pd.isna(v) else format(v, s)
        )
    return str(shown.to_string(index=False))


def _pivot(frame: pd.DataFrame, value: str, spec: str) -> str:
    """Render one column of the sweep as an e1-by-e2 matrix.

    Args:
        frame: The sweep grid.
        value: Column to lay out.
        spec: Format spec for each cell.

    Returns:
        The rendered matrix.
    """
    table = frame.pivot(index="growth_recovery", columns="recovery", values=value)
    return str(table.map(lambda v: "--" if pd.isna(v) else format(v, spec)).to_string())


def main() -> None:
    """Print the two-leg nozzle comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-days", type=float, default=DEFAULT_SPLIT_DAYS, help="split gap"
    )
    parser.add_argument("--csv", default=None, help="write the sweep grid here")
    args = parser.parse_args()

    sweep = analyze_two_leg_nozzle(split_days=args.split_days)
    window1, window2 = sweep.windows
    print("=== ADR 0016: the matched diagonal with the dissociation toll charged ===")
    print(
        "    eta_geom is swept; eta_jet = eta_chem * eta_geom on each nozzle leg.\n"
        "    The plate column pays the toll on its head-on leg only, because a\n"
        "    plate owes no chemistry -- which is why the ratio column falls."
    )
    print(
        _format_plain(
            sweep.tolled,
            {
                "eta_geom": ".2f",
                "k1": ".2f",
                "k2": ".2f",
                "nozzle_growth": ".4g",
                "plate_growth": ".4g",
                "nozzle_over_plate": ".2f",
                "published_nozzle_growth": ".4g",
            },
        )
    )
    print()
    print("Magnetic nozzle on both legs, against the pusher plate on the first")
    print(
        f"chain: {len(sweep.cycles)} cycles / {sweep.horizon_years:.4f} yr / "
        f"{sweep.split_days:.0f} d split; methalox v_e {VE_METHALOX:.4f} km/s"
    )
    print(
        f"plume floor {NOZZLE_FLOOR_TEMPERATURE.to_value(u.K):.0f} K -> fleet-wide "
        f"slug windows: leg 1 {window1}, leg 2 {window2}"
    )
    print(
        f"ground ledger: {LAUNCH_PROPELLANT_FRACTION:.4f} propellant, "
        f"{LAUNCHER_DRY_FRACTION:.2f} of the remainder dry -> "
        f"{PAYLOAD_FRACTION_AT_INTERCEPT:.4f} reaches intercept; "
        f"floor 1/{1.0 / RETURN_FLOOR:.0f} returned per liftoff"
    )
    print(
        f"search box: {_K_GRID_POINTS} log-spaced points per axis inside the "
        f"fleet window, {_REFINE_ROUNDS} refinement rounds\n"
    )

    for title, column, spec in (
        ("e-foldings per year", "e_foldings_per_year", ".4f"),
        ("doubling time (yr)", "doubling_years", ".2f"),
        ("growth-push slug ratio k1", "k1", ".2f"),
        ("departure-burn slug ratio k2", "k2", ".2f"),
        ("returned mass per kg of liftoff", "returned_per_liftoff", ".4f"),
        ("binding limit", "limit", "s"),
        ("equivalent plate elasticity f", "equivalent_plate_f", ".3f"),
    ):
        print(f"=== {title} === rows e1 (growth push), columns e2 (departure burn)")
        print(_pivot(sweep.grid, column, spec))
        print()

    print("=== Incumbent: pusher plate on the growth push, same cycles ===")
    print(
        str(
            sweep.plate.pivot(
                index="fudge", columns="recovery", values="e_foldings_per_year"
            ).map(lambda v: "--" if pd.isna(v) else format(v, ".4f"))
        )
    )
    if args.csv:
        sweep.grid.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
