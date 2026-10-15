"""How the two-wave split is flown, priced through the ledger (ADRs 0068, 0069).

Three ways to get the growth wave home one parking orbit ahead of the nozzle
wave:

* **Batch, independent** (ADR 0013, the ledger's default): one departing batch,
  the growth wave re-solved with its own Earth departure, which no burn pays
  for.
* **Batch, divided at Jupiter** (ADR 0068): one outbound leg, each wave on its
  cheapest flyby (:func:`src.two_wave_growth.shared_split`).
* **By role** (ADR 0069): launch units dedicated as growth carriers or rod
  carriers, each departing on its own trajectory and paying its own departure
  burn through the chamber.  Nothing divides at Jupiter.

The role split's growth solves two supply balances (:func:`role_growth`).  On
three-synodic windows the growth carriers may also take an earlier, faster
return (the author's 2.8-synodic, ~68 km/s growth wave), held for more laps of
the 20-day orbit with hydrogen's cryostats scaled to the hold.

Every row flies the burn centred on periapsis, so the rows compare like for
like; the visibility tables cover only the independent chain's departures.
Run ``make split-by-role`` (about 10 min on four workers).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from tabulate import tabulate

from src.chamber_departure import LossModel, role_growth
from src.finite_burn_loss import fixed_direction_loss
from src.growth_ledger import (
    _orbit_loss,
    best_cycle_growth,
    chain_growth,
    hydrogen_boil_off,
    parking_period,
    price_cycle_growth,
    summarize_chain,
)
from src.nozzle_analysis import apoapsis_reversal_dv
from src.real_orbit_resonance import _ManeuverSolution
from src.seed_cost import Design
from src.survivable_chamber import HYDROGEN_BULGED_DESIGN, PLATE, survivable_design
from src.two_wave_growth import (
    VE_METHALOX,
    TwoWaveCycle,
    _cheapest_return,
    _cycle_periapsis_speed,
    adaptive_two_wave_cycles,
    window_waves,
)

#: The parking orbit and split gap ADRs 0068-0069 scouted on (days).  The
#: ledger's default moved to 10 days after (ADR 0071); :func:`split_gap_row`
#: sweeps the gap.
SCOUT_PARKING_DAYS = 20.0
#: Leads the growth carriers try on three-synodic windows (days); two-synodic
#: windows keep the 20-day parking orbit's lead.
EARLY_LEADS = (20.0, 40.0, 60.0, 80.0, 120.0)
#: The hold slush rides out on the 3% cryostat at 0.3%/day liquid insulation
#: (ADR 0051); a longer hold scales the cryostat with it (ADR 0069).
SLUSH_HOLD_DAYS = 60.0
DESIGNS: Dict[str, Design] = {
    "methane": survivable_design(100),
    "hydrogen": HYDROGEN_BULGED_DESIGN,
}

_Window = Tuple[int, _ManeuverSolution, _ManeuverSolution]


def growth_windows(
    cycles: Sequence[TwoWaveCycle], three_synodic_lead: float = SCOUT_PARKING_DAYS
) -> List[_Window]:
    """Each window's (synodic multiple, nozzle wave, growth wave), and the one after.

    Args:
        cycles: The flown chain, independent split.
        three_synodic_lead: Days the growth wave leads on three-synodic windows.

    Returns:
        One window per cycle departure, then the window after the chain.
    """
    out = []
    departures = [c.departure_jd for c in cycles] + [cycles[-1].return_jd]
    for departure in departures:
        multiple, return_jd, nozzle, growth = window_waves(
            departure, split_days=SCOUT_PARKING_DAYS, split="independent"
        )
        if multiple == 3 and three_synodic_lead != SCOUT_PARKING_DAYS:
            growth = _cheapest_return(departure, return_jd - three_synodic_lead)
        out.append((multiple, nozzle, growth))
    return out


def _departure_burn(solution: _ManeuverSolution) -> float:
    return float(solution.earth_departure_periapsis_speed - _cycle_periapsis_speed())


def chain_by_role(
    name: str, three_synodic_lead: float = SCOUT_PARKING_DAYS
) -> List[float]:
    """Each flown cycle's growth with launch units split by role.

    Args:
        name: A key of :data:`DESIGNS`.
        three_synodic_lead: As in :func:`growth_windows`.

    Returns:
        The cycles' growth factors.
    """
    design = DESIGNS[name]
    pairing = design.pairing
    assert pairing is not None and design.efficiency is not None
    cycles = adaptive_two_wave_cycles(
        split_days=SCOUT_PARKING_DAYS, split="independent"
    )
    windows = growth_windows(cycles, three_synodic_lead)
    loss = _orbit_loss(fixed_direction_loss, parking_period(cycles[0]))
    growths = []
    for i, cycle in enumerate(cycles):
        multiple, _, growth_wave = windows[i]
        _, next_nozzle, next_growth = windows[i + 1]
        # Each carrier is priced as a batch of one kind; the balance below
        # does the splitting.
        flown = replace(
            cycle,
            growth_wave_v_b=growth_wave.earth_return_collision_speed,
            growth_wave_burn=growth_wave.total_dv,
            onward_growth_burn=float("nan"),
        )
        hold = three_synodic_lead if multiple == 3 else SCOUT_PARKING_DAYS
        storage = pairing.storage
        cryostat = (
            0.0
            if storage is None
            else storage.cryostat_fraction * max(1.0, hold / SLUSH_HOLD_DAYS)
        )
        price = best_cycle_growth(
            flown,
            PLATE.efficiency,
            pairing,
            design.efficiency,
            pitch_ratio=design.pitch_ratio,
            loss_model=loss,
            max_slug_ratio=PLATE.max_slug_ratio,
            impactor_bond_energy=PLATE.impactor_bond_energy,
            film_per_impulse=PLATE.film_per_impulse,
            cryostat_fraction=cryostat,
        ).initial_water_price

        def unit(burn: float) -> Tuple[float, float, float]:
            assert pairing is not None and design.efficiency is not None
            ledger = price_cycle_growth(
                replace(flown, onward_burn=burn),
                PLATE.efficiency,
                pairing,
                design.efficiency,
                price,
                pitch_ratio=design.pitch_ratio,
                loss_model=loss,
                max_slug_ratio=PLATE.max_slug_ratio,
                impactor_bond_energy=PLATE.impactor_bond_energy,
                film_per_impulse=PLATE.film_per_impulse,
                cryostat_fraction=cryostat,
            )
            stack = float(ledger.departing_stack.to_value(u.t))
            return (
                float(ledger.puffsats.to_value(u.t)),
                ledger.departure.delivered_net * stack,
                ledger.departure.rod_mass_fraction * stack,
            )

        puffsats, onward_growth, rods_growth = unit(_departure_burn(next_growth))
        _, onward_rods, rods_rods = unit(_departure_burn(next_nozzle))
        growths.append(
            role_growth(
                puffsats,
                onward_growth,
                onward_rods,
                rods_growth,
                rods_rods,
                float(np.exp(-flown.growth_wave_burn / VE_METHALOX)),
                float(np.exp(-flown.nozzle_wave_dsm / VE_METHALOX)),
            )
        )
    return growths


def chain_by_batch(name: str, split: str) -> List[float]:
    """Each flown cycle's growth with one batch supplying both waves.

    Args:
        name: A key of :data:`DESIGNS`.
        split: ``"independent"`` or ``"shared"`` (divided at Jupiter).

    Returns:
        The cycles' growth factors.
    """
    design = DESIGNS[name]
    pairing = design.pairing
    assert pairing is not None and design.efficiency is not None
    cycles = [
        replace(c, onward_growth_burn=float("nan"))
        for c in adaptive_two_wave_cycles(split_days=SCOUT_PARKING_DAYS, split=split)
    ]
    storage = pairing.storage
    loss: LossModel = _orbit_loss(fixed_direction_loss, parking_period(cycles[0]))
    grown = chain_growth(
        cycles,
        PLATE.efficiency,
        pairing,
        design.efficiency,
        design.pitch_ratio,
        loss_model=loss,
        max_slug_ratio=PLATE.max_slug_ratio,
        impactor_bond_energy=PLATE.impactor_bond_energy,
        film_per_impulse=PLATE.film_per_impulse,
        cryostat_fraction=0.0 if storage is None else storage.cryostat_fraction,
        boil_off=(
            0.0
            if storage is None
            else hydrogen_boil_off(cycles, storage.boil_off_per_day)
        ),
    )
    return [g.growth for g in grown]


def _row(case: Tuple[str, str, float]) -> Dict[str, object]:
    name, how, lead = case
    if how == "role":
        growths = chain_by_role(name, lead)
        label = f"by role, 3S lead {lead:.0f} d"
    else:
        growths = chain_by_batch(name, how)
        label = {
            "independent": "batch, uncharged (ADR 0013)",
            "shared": "batch, divided at Jupiter",
        }[how]
    cycles = adaptive_two_wave_cycles(
        split_days=SCOUT_PARKING_DAYS, split="independent"
    )
    summary = summarize_chain([c.period_years for c in cycles], growths)
    return {
        "chamber": name,
        "split": label,
        "doubling yr": summary.doubling_years,
        "G": f"{min(growths):.2f}-{max(growths):.2f}",
    }


#: Split gaps the sweep tries (days); each is also the parking orbit.
SPLIT_GAPS = (4.0, 5.0, 7.0, 10.0, 15.0, 20.0)


def split_gap_row(case: Tuple[str, float]) -> Dict[str, object]:
    """The ledger's role split at one split gap, the stream kept in view.

    The gap is the parking orbit, so a shorter one buys a cheaper split (the
    growth carriers' departure stays closer to the rod carriers') at the price
    of a dearer apoapsis reversal.

    Args:
        case: (a key of :data:`DESIGNS`, split gap in days).

    Returns:
        The row.
    """
    name, gap = case
    design = DESIGNS[name]
    pairing = design.pairing
    assert pairing is not None and design.efficiency is not None
    cycles = adaptive_two_wave_cycles(split_days=gap)
    storage = pairing.storage
    grown = chain_growth(
        cycles,
        PLATE.efficiency,
        pairing,
        design.efficiency,
        design.pitch_ratio,
        max_slug_ratio=PLATE.max_slug_ratio,
        impactor_bond_energy=PLATE.impactor_bond_energy,
        film_per_impulse=PLATE.film_per_impulse,
        cryostat_fraction=0.0 if storage is None else storage.cryostat_fraction,
        boil_off=(
            0.0
            if storage is None
            else hydrogen_boil_off(cycles, storage.boil_off_per_day)
        ),
    )
    summary = summarize_chain(
        [c.period_years for c in cycles], [g.growth for g in grown]
    )
    return {
        "chamber": name,
        "split d": gap,
        "reversal m/s": float(apoapsis_reversal_dv(gap * u.day).to_value(u.m / u.s)),
        "growth carrier extra burn km/s": float(
            np.mean([c.onward_growth_burn - c.onward_burn for c in cycles])
        ),
        "growth wave correction km/s": float(
            np.mean([c.growth_wave_burn for c in cycles])
        ),
        "growth share": float(np.mean([g.growth_share for g in grown])),
        "doubling yr": summary.doubling_years,
    }


def cases() -> List[Tuple[str, str, float]]:
    """Every (design, split, three-synodic lead) the report flies."""
    rows: List[Tuple[str, str, float]] = []
    for name in DESIGNS:
        rows += [
            (name, "independent", SCOUT_PARKING_DAYS),
            (name, "shared", SCOUT_PARKING_DAYS),
        ]
        rows += [(name, "role", lead) for lead in EARLY_LEADS]
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the split comparison.

    Args:
        argv: ``--workers``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--split-sweep",
        action="store_true",
        help="the ledger's role split across split gaps, stream in view (ADR 0071)",
    )
    args = parser.parse_args(argv)
    if args.split_sweep:
        gaps = [(name, gap) for name in DESIGNS for gap in SPLIT_GAPS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            sweep = list(pool.map(split_gap_row, gaps))
        print(tabulate(sweep, headers="keys", floatfmt=".3f"))
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_row, cases()))
    print(
        "Flown chain, 20-day parking orbit, spray cup, film carried, burn centred "
        "on periapsis (ADRs 0068, 0069).\n"
    )
    print(tabulate(rows, headers="keys", floatfmt=".2f"))


if __name__ == "__main__":
    main()
