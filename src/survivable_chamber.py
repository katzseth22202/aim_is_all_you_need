"""The survivable methane chamber through the growth ledger and cost book (ADR 0044).

The parent's S17 (``docs/survivable_chamber_asks_for_aim_repo.md`` there,
applying the impact sim @ ``69d1f40``): the 20 m^3 sphere's bonded overwrap
delaminates under the stopped rod's line blast, so the departure chamber is
bulged to 212 m^3 and wrapped dry.  It fires a 5 kg rod at 2 Hz behind a 20 kg
frozen-methane plug.  ADR 0055 (impact sim @ 7afdba3) doubles its throat to
2.66 m^2: a 46.2 t wall (68.8 t maraging fallback), a 9.8 t (A/A* = 100) or
29.4 t (A/A* = 300) extension, and a 2.6 kg aramid throat film every shot,
charged with the 3.5-24 kg of pitch as carried mass that leaves in the exhaust.
:func:`src.chamber_isp.survivable_methane` builds the pairing, and its ``eta``
is the one that reproduces the impact sim's 885 / 941 kN s per pulse at 75 km/s
with the gate and finite-rate nozzle recombination charged
(:func:`src.chamber_isp.efficiency_for_impulse`): 0.472 at A/A* = 100 and 0.515
at 300 (ADR 0056).  ADR 0055's 868 / 899 (0.459 / 0.483) froze hydrogen
recombination suddenly; ADR 0044's 906 / 971 kN s (0.488 / 0.539) charged
neither the freeze nor the gate.

The parent's S24 (``docs/throat_and_turn_asks_for_aim_repo.md`` there) prices
the doubled throat as two options: (a) A/A* = 150 behind the old 22.5 m exit,
(b) A/A* = 300 behind a 31.9 m one; A/A* = 100 is ADR 0055's lead.  Each flies at
both carbon limits and at every rate :func:`src.chamber_isp.survivable_cycle`
fits: 2 Hz, and 3 Hz with the fill doubled to 2 t/s (:func:`cycle_report`).

The impact sim sizes the wall for one departure of ~4,200 pulses
(:data:`WALL_LIFE_PULSES`); the ledger reports the most pulses any one chamber
fires so the longest cycles can be checked against it.

Everything flies behind the vapor-shielded spray cup (ADR 0041/0042), its film
carried as launched mass as the parent's tables carry it (ADR 0043), on the
20-day orbit, with the integrated fixed-direction finite-burn loss charged and
solved with the burn, as every chamber row since ADR 0032.
The impact sim's +3.1-4.8% assumed thrust held along the velocity; that is the
``steered`` sensitivity here, which a head-on chamber cannot fly.

Each chain takes about four minutes; the cases run in parallel worker
processes.  ``--ledger`` is the growth ledger (~20 min on four workers),
``--cost`` the cost book and seed (~25 min).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from astropy import units as u
from tabulate import tabulate

from src.chamber_departure import chamber_unit_mass
from src.chamber_isp import (
    HYDROGEN_BULGED,
    HYDROGEN_BULGED_IMPULSE,
    SEAL_TIME_S17,
    SURVIVABLE_FALLBACK_WALL_MASS,
    SURVIVABLE_IMPULSE,
    SURVIVABLE_IMPULSE_CARBON_FROZEN,
    SURVIVABLE_PULSE_RATE,
    SURVIVABLE_THROAT_AREA,
    SURVIVABLE_THROAT_AREA_S17,
    SURVIVABLE_THROAT_FILM,
    SURVIVABLE_WALL_MASS,
    ChamberPairing,
    absolute_efficiency,
    efficiency_for_impulse,
    survivable_cycle,
    survivable_methane,
)
from src.finite_burn_loss import steered_loss
from src.growth_cost import ESTIMATE, PESSIMISTIC, SCENARIOS, DesignInputs, PriceBook
from src.growth_cost_inputs import design_inputs, seed_prices
from src.growth_cost_report import (
    climbing_lob,
    hardware,
    headline,
    rates,
    seed_routes,
)
from src.growth_ledger import (
    ADR_0033_PLATE,
    DEFAULT_PARKING_DAYS,
    DEPARTURE_ALTITUDE,
    LAUNCH_UNIT,
    PLATE_MAX_SLUG_RATIO,
    PLUG,
    SOLVED_EFFICIENCY,
    SPRAY_CUP_SHIELDED,
    SPRAY_CUP_UNSHIELDED,
    CycleGrowth,
    PlateDesign,
    _orbit_loss,
    chain_growth,
    departure_at_altitude,
    hydrogen_boil_off,
    parking_period,
    summarize_chain,
)
from src.lob_rise import OPERATING_RISE_SPEED, booster_growth
from src.seed_cost import (
    DESIGNS,
    STRIPPED_SHIPS,
    Design,
    DesignChain,
    _amortization,
    chains,
    design_chain,
    seed_excess_speed,
    seed_price_range,
)
from src.two_wave_growth import TwoWaveCycle

#: Pitch a pulse spends, between the wall-heat edges (parent S17).
SURVIVABLE_PITCH_RANGE = (3.5 * u.kg, 24.0 * u.kg)
#: The heavy end, which the matrix carries as it did 5.6 kg for the sphere.
SURVIVABLE_PITCH = SURVIVABLE_PITCH_RANGE[1]
AREA_RATIOS = (100, 300)
#: S24's three nozzles on the doubled throat: 18.4, 22.5 (option a) and 31.9 m
#: (option b) exits.
S24_AREA_RATIOS = (100, 150, 300)
#: The impulse limits each S24 row flies (ADR 0056).
IMPULSE_LIMITS = {
    "finite rate": SURVIVABLE_IMPULSE,
    "carbon frozen": SURVIVABLE_IMPULSE_CARBON_FROZEN,
}
#: The rates S24 asks for, and the faster fill 3 Hz needs (:func:`cycle_report`).
S24_RATES = (2.0 / u.s, 3.0 / u.s, 4.0 / u.s)
FAST_FILL_RATE = 2.0 * u.t / u.s
#: The blowdown clock's own uncertainty on the 1.33 m^2 throat (wall heat, a
#: discharge coefficient below one), halved with the doubled throat's ``V / A*``.
CLOCK_UNCERTAINTY_S17 = 45.0 * u.ms
#: Shares of the A/A* = 300 chemistry ceiling for the matrix's other rows.
#: A/A* = 100 returns fewer bonds, and its ceiling is not sized.
CEILING_SHARES = (0.50, 0.70, 0.90, 1.00)
#: Worker processes; each holds a few hundred MB.
WORKERS = 4
#: The plate every row flies behind: the parent's tables carry its film (S21).
PLATE = SPRAY_CUP_SHIELDED
#: Pulses the impact sim sizes one chamber's wall for: one departure of a
#: 500-600 t stack (its ``walled_nozzle_chamber_fatigue.md``).
WALL_LIFE_PULSES = 4200
#: Two 2.5 kg chambers' walls together: ADR 0044's 49.8 t against 48.2 t for
#: one 5 kg chamber, scaled to the doubled throat's 46.2 t.
SPLIT_WALL_MASS = 46.2 * 49.8 / 48.2 * u.t
#: The bulged hydrogen chamber at 2 Hz, at the impact sim's A/A* = 300 impulse
#: (ADR 0055).  The sphere's solved row stays beside it as the record.
HYDROGEN_BULGED_DESIGN = Design(
    "Hydrogen, bulged, 2 Hz",
    HYDROGEN_BULGED,
    efficiency_for_impulse(HYDROGEN_BULGED_IMPULSE, HYDROGEN_BULGED),
)


def survivable_efficiency(
    area_ratio: int, impulses: Dict[int, u.Quantity] = SURVIVABLE_IMPULSE
) -> float:
    """The ``eta`` that gives the impact sim's net impulse per pulse.

    Args:
        area_ratio: 100, 150 or 300.
        impulses: Net impulse per pulse by area ratio; finite rate by default.

    Returns:
        ``eta`` at 75 km/s, gate charged.
    """
    return efficiency_for_impulse(impulses[area_ratio], survivable_methane(area_ratio))


def survivable_design(
    area_ratio: int,
    pitch: u.Quantity = SURVIVABLE_PITCH,
    wall_mass: u.Quantity = SURVIVABLE_WALL_MASS,
    rods_split: int = 1,
    chambers: Optional[int] = None,
    efficiency: Optional[float] = None,
    pulse_rate: u.Quantity = SURVIVABLE_PULSE_RATE,
) -> Design:
    """The survivable chamber as a design the ledger and cost book fly.

    Args:
        area_ratio: 100 or 300.
        pitch: Pitch each 5 kg pulse spends; split chambers spend their share.
            The throat film is added to it: both leave in the exhaust.
        wall_mass: Wall of one 5 kg chamber, or of all the split ones.
        rods_split: As in :func:`src.chamber_isp.survivable_methane`.
        chambers: Fly exactly this many chambers; None picks the best count.
        efficiency: ``eta``; the impact sim's finite-rate one by default.
        pulse_rate: Pulses per second each chamber fires.

    Returns:
        The design.
    """
    pairing = survivable_methane(area_ratio, wall_mass, rods_split, pulse_rate)
    eta = survivable_efficiency(area_ratio) if efficiency is None else efficiency
    label = (
        f"CH4 AR{area_ratio}, {pitch.to_value(u.kg):g} kg"
        + (
            f", wall {wall_mass.to_value(u.t):g} t"
            if wall_mass != SURVIVABLE_WALL_MASS
            else ""
        )
        + (f", {rods_split}x{5 / rods_split:g} kg" if rods_split > 1 else "")
        + (f", n={chambers}" if chambers is not None else "")
        + (
            f", {pulse_rate.to_value(1 / u.s):g} Hz"
            if pulse_rate != SURVIVABLE_PULSE_RATE
            else ""
        )
    )
    carried = (pitch + SURVIVABLE_THROAT_FILM) / rods_split
    return Design(label, pairing, eta, carried, chambers)


@dataclass(frozen=True)
class Case:
    """One ledger run.

    Attributes:
        group: The table it prints in.
        label: Row label.
        design: What departs.
        steered: Charge the steered loss instead of the fixed-direction one.
    """

    group: str
    label: str
    design: Design
    steered: bool = False


def _fly(case: Case, plate: PlateDesign = PLATE) -> List[CycleGrowth]:
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    pairing = case.design.pairing
    assert pairing is not None and case.design.efficiency is not None
    # The steered loss is a sensitivity on the centred burn; everything else
    # keeps its stream in view (ADR 0066).
    steered = _orbit_loss(steered_loss, parking_period(flown[0]))
    storage = pairing.storage
    return chain_growth(
        flown,
        plate.efficiency,
        pairing,
        case.design.efficiency,
        case.design.pitch_ratio,
        max_slug_ratio=plate.max_slug_ratio,
        impactor_bond_energy=plate.impactor_bond_energy,
        film_per_impulse=plate.film_per_impulse,
        cryostat_fraction=0.0 if storage is None else storage.cryostat_fraction,
        boil_off=(
            0.0
            if storage is None
            else hydrogen_boil_off(flown, storage.boil_off_per_day)
        ),
        chambers=case.design.chambers,
        loss_model=steered if case.steered else None,
    )


def _span(values: Sequence[float], fmt: str = ".0f") -> str:
    low, high = min(values), max(values)
    return (
        f"{low:{fmt}}"
        if f"{low:{fmt}}" == f"{high:{fmt}}"
        else f"{low:{fmt}}-{high:{fmt}}"
    )


def ledger_row(case: Case) -> Dict[str, object]:
    """Fly one case over the chain and summarise it (runs in a worker).

    Args:
        case: The case.

    Returns:
        The row.
    """
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    try:
        grown = _fly(case)
    except (RuntimeError, ValueError) as error:
        # A forced chamber count can burn past the loss table's hour.
        return {"group": case.group, "departure": case.label, "doubling yr": str(error)}
    pairing = case.design.pairing
    assert pairing is not None
    summary = summarize_chain(
        [c.period_years for c in flown], [g.growth for g in grown]
    )
    stacks = [g.departing_stack.to_value(u.t) for g in grown]
    deps = [g.departure for g in grown]
    shares = [
        (d.finite_burn_loss / _burn(c)).to_value(u.one) for c, d in zip(flown, deps)
    ]
    return {
        "group": case.group,
        "departure": case.label,
        "eta": case.design.efficiency,
        "doubling yr": summary.doubling_years,
        "annual": summary.annual_growth,
        "10yr step": summary.ten_year_stepwise,
        "G_0": grown[0].growth,
        "stack t": _span(stacks),
        "n": _span([d.chambers for d in deps]),
        "hw t": _span(
            [d.chambers * chamber_unit_mass(pairing).to_value(u.t) for d in deps], ".1f"
        ),
        "pulses": _span([d.pulses for d in deps]),
        "max/chamber": f"{max(d.pulses / d.chambers for d in deps):.0f}",
        "burn min": _span([d.burn_time.to_value(u.min) for d in deps]),
        "loss m/s": _span([d.finite_burn_loss.to_value(u.m / u.s) for d in deps]),
        "loss %": _span([100.0 * b for b in shares], ".1f"),
        "payload t": float(
            np.mean([d.delivered_net * s for d, s in zip(deps, stacks)])
        ),
    }


def _burn(cycle: TwoWaveCycle) -> u.Quantity:
    """The impulsive departure burn a cycle's unit flies from 600 km."""
    return departure_at_altitude(
        cycle.onward_burn * u.km / u.s, DEPARTURE_ALTITUDE, parking_period(cycle)
    ).burn


def _ledger_row_at(index: int) -> Dict[str, object]:
    """:func:`ledger_row` of ``ledger_cases()[index]`` (in a worker)."""
    return ledger_row(ledger_cases()[index])


def s24_rates() -> List[u.Quantity]:
    """The S24 rates the doubled throat's cycle fits, with the fill at 2 t/s.

    Returns:
        Each rate whose margin clears the blowdown clock's uncertainty.
    """
    period = survivable_cycle(SURVIVABLE_THROAT_AREA, 0.3, FAST_FILL_RATE)
    slack = CLOCK_UNCERTAINTY_S17 * float(
        (SURVIVABLE_THROAT_AREA_S17 / SURVIVABLE_THROAT_AREA).to_value(u.one)
    )
    return [r for r in S24_RATES if (1.0 / r - period - slack) > 0.0 * u.ms]


def ledger_cases() -> List[Case]:
    """Every S17 ledger run: items 1-3, the ceiling shares, and S24's grid.

    Returns:
        The cases, grouped by table.
    """
    cases = [
        Case(
            "1. Survivable methane, solved",
            survivable_design(ar, p).label,
            survivable_design(ar, p),
        )
        for ar in AREA_RATIOS
        for p in SURVIVABLE_PITCH_RANGE
    ]
    pairing = survivable_methane(300)
    for share in CEILING_SHARES:
        eta = absolute_efficiency(pairing, share)
        cases.append(
            Case(
                "1b. A/A* = 300 at shares of the chemistry ceiling, 24 kg pitch",
                f"CH4 AR300 {share:.0%}",
                survivable_design(300, efficiency=eta),
            )
        )
    for ar in AREA_RATIOS:
        cases += [
            Case(
                "3. Redundancy, 24 kg pitch (n = count flown)",
                f"AR{ar}: 5 kg chambers, best count",
                survivable_design(ar),
            ),
            Case(
                "3. Redundancy, 24 kg pitch (n = count flown)",
                f"AR{ar}: 2.5 kg chambers "
                f"({SPLIT_WALL_MASS.to_value(u.t) / 2:.1f} t each), best count",
                survivable_design(ar, wall_mass=SPLIT_WALL_MASS, rods_split=2),
            ),
            Case(
                "4. Sensitivities, 24 kg pitch",
                f"AR{ar}: two 5 kg chambers (wall life)",
                survivable_design(ar, chambers=2),
            ),
            Case(
                "4. Sensitivities, 24 kg pitch",
                f"AR{ar}: maraging fallback wall",
                survivable_design(ar, wall_mass=SURVIVABLE_FALLBACK_WALL_MASS),
            ),
            Case(
                "4. Sensitivities, 24 kg pitch",
                f"AR{ar}: steered loss (cannot fly)",
                survivable_design(ar),
                steered=True,
            ),
        ]
    for limit, impulses in IMPULSE_LIMITS.items():
        for rate in s24_rates():
            for ar in S24_AREA_RATIOS:
                cases.append(
                    Case(
                        "S24. The doubled throat, 24 kg pitch, by impulse limit and rate",
                        f"AR{ar} {limit}, {rate.to_value(1 / u.s):g} Hz",
                        survivable_design(
                            ar,
                            efficiency=survivable_efficiency(ar, impulses),
                            pulse_rate=rate,
                        ),
                    )
                )
    return cases


def cycle_report() -> str:
    """S24 item 1: the 212 m^3 methane chamber's cycle at 2, 3 and 4 Hz.

    The same chamber and throat serve options (a) and (b): the exit does not
    touch the blowdown.

    Returns:
        The table.
    """
    rows = []
    for throat in (SURVIVABLE_THROAT_AREA_S17, SURVIVABLE_THROAT_AREA):
        slack = CLOCK_UNCERTAINTY_S17 * float(
            (SURVIVABLE_THROAT_AREA_S17 / throat).to_value(u.one)
        )
        for seal in SEAL_TIME_S17:
            for fill in (1.0 * u.t / u.s, FAST_FILL_RATE):
                period = survivable_cycle(throat, seal, fill)
                row: Dict[str, object] = {
                    "throat m^2": throat.to_value(u.m**2),
                    "film sealed at": f"{seal:g} bar"
                    + (" (polymer)" if seal < 1.0 else " (fill p, ~1,100 K)"),
                    "fill t/s": fill.to_value(u.t / u.s),
                    "seal ms": (
                        SEAL_TIME_S17[seal]
                        * float((SURVIVABLE_THROAT_AREA_S17 / throat).to_value(u.one))
                    ).to_value(u.ms),
                    "period ms": period.to_value(u.ms),
                    "fastest Hz": (1.0 / period).to_value(1 / u.s),
                }
                for rate in S24_RATES:
                    margin = (1.0 / rate - period).to_value(u.ms)
                    row[f"{rate.to_value(1 / u.s):g} Hz margin ms"] = margin
                row["clock +/- ms"] = slack.to_value(u.ms)
                rows.append(row)
    return (
        "S24 item 1: the cycle (seal, films 30 ms, fill 108 kg, lock 20 ms; seal "
        "times are the impact sim's real-gas solve scaled as V/A*).  A rate fits "
        "when its margin clears the blowdown clock's uncertainty.\n"
        + tabulate(rows, headers="keys", floatfmt=".4g")
    )


_In = TypeVar("_In")
_Out = TypeVar("_Out")


def _in_workers(function: Callable[[_In], _Out], items: Sequence[_In]) -> List[_Out]:
    """``map`` over worker processes, in order.

    Pass indices, not designs: a pickled pairing is a copy, and the ledger
    tells hydrogen from methane by identity (``pairing is HYDROGEN_5500K``).
    """
    workers = min(WORKERS, os.cpu_count() or 1, len(items))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


#: The paper's ``tab:growth_ledger_doubling`` columns: the spray cup with its
#: film carried and the plug at k <= 8.52 (k <= 10 in parentheses), and ADR
#: 0033's earlier plate at k <= 10; and the unshielded film of
#: ``tab:plate_designs_cost`` (ADR 0056).
PAPER_GRID_PLATES = (
    ("spray cup", PLATE, True),
    ("spray cup, unshielded film", SPRAY_CUP_UNSHIELDED, False),
    ("plug", PLUG, True),
    ("earlier 0.84", ADR_0033_PLATE, False),
)


def paper_grid_designs() -> List[Tuple[str, Design]]:
    """The methane rows of ``tab:growth_ledger_doubling``: A/A* = 300 at
    shares of the 0.669 ceiling and as solved, finite rate, 24 kg pitch, 2 Hz.

    Returns:
        (row label, design) pairs.
    """
    rows = [
        (
            f"Methane, {share:.0%}",
            survivable_design(
                300, efficiency=absolute_efficiency(survivable_methane(300), share)
            ),
        )
        for share in CEILING_SHARES
    ]
    solved = survivable_design(300)
    share = survivable_efficiency(300) / survivable_methane(300).chemistry_ceiling
    rows.insert(2, (f"Methane, solved ({share:.0%})", solved))
    return rows


def _paper_grid_cell(index: int) -> Dict[str, object]:
    """One (design, plate, cap) cell of :func:`paper_grid_designs` (in a worker)."""
    n_caps = 2
    per_design = len(PAPER_GRID_PLATES) * n_caps
    label, design = paper_grid_designs()[index // per_design]
    name, plate, both_caps = PAPER_GRID_PLATES[(index % per_design) // n_caps]
    loose = index % n_caps == 1
    if loose and not both_caps:
        return {}
    if loose:
        plate = replace(plate, max_slug_ratio=PLATE_MAX_SLUG_RATIO)
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    grown = _fly(Case("", label, design), plate)
    summary = summarize_chain(
        [c.period_years for c in flown], [g.growth for g in grown]
    )
    stacks = [g.departing_stack.to_value(u.t) for g in grown]
    payload = float(
        np.mean([g.departure.delivered_net * s for g, s in zip(grown, stacks)])
    )
    return {
        "row": label,
        "plate": name + (" k<=10" if loose else ""),
        "doubling yr": summary.doubling_years,
        "annual": summary.annual_growth,
        "10yr step": summary.ten_year_stepwise,
        "payload t": payload,
    }


def paper_grid_report() -> str:
    """The methane cells of ``tab:growth_ledger_doubling`` and
    ``tab:growth_ledger_ten_year`` on the doubled throat (ADR 0056).

    Returns:
        The table.
    """
    count = len(paper_grid_designs()) * len(PAPER_GRID_PLATES) * 2
    rows = [r for r in _in_workers(_paper_grid_cell, range(count)) if r]
    return (
        "Paper grid: survivable methane A/A* = 300, 2.66 m^2, finite-rate impulse, "
        "24 kg pitch + 2.6 kg film, 2 Hz, by plate (k <= 8.52 unless marked)\n"
        + tabulate(rows, headers="keys", floatfmt=".4g")
    )


#: The plates of the paper's ``tab:plate_designs_cost`` (ADR 0056).
PAPER_COST_PLATES = (PLATE, SPRAY_CUP_UNSHIELDED, PLUG, ADR_0033_PLATE)


def paper_cost_designs() -> List[Tuple[Design, PlateDesign]]:
    """The paper's priced methane rows on the doubled throat: A/A* = 300 as
    solved behind every plate, option (a) beside it, and 50% of the ceiling
    behind the spray cup (``tab:seed_return``).

    Returns:
        (design, plate) pairs.
    """
    half = survivable_design(
        300, efficiency=absolute_efficiency(survivable_methane(300), 0.5)
    )
    half = replace(half, label=f"{half.label}, 50%")
    return [(survivable_design(300), plate) for plate in PAPER_COST_PLATES] + [
        (survivable_design(150), PLATE),
        (half, PLATE),
    ]


def _paper_priced(index: int) -> DesignInputs:
    """The cost inputs of ``paper_cost_designs()[index]`` (in a worker)."""
    design, plate = paper_cost_designs()[index]
    return design_inputs(design, plate)


def paper_cost_report() -> str:
    """The paper's methane cost cells on the doubled throat, the chamber priced
    by its mass as the paper prices it (``tab:delivery_ledger``,
    ``tab:l1_comparison``, ``tab:seed_return``, ``tab:plate_designs_cost``).

    Returns:
        The report.
    """
    pairs = paper_cost_designs()
    inputs = _in_workers(_paper_priced, range(len(pairs)))
    labelled = [
        replace(i, label=f"{d.label} / {plate.label}")
        for (d, plate), i in zip(pairs, inputs)
    ]
    estimate, pessimistic = (
        climbing_lob(book, OPERATING_RISE_SPEED) for book in (ESTIMATE, PESSIMISTIC)
    )
    out = [
        "Paper cost cells: methane chamber priced by its mass against the sphere's "
        f"{_SPHERE_UNIT:.1f} t; lob x{booster_growth(OPERATING_RISE_SPEED):.4f}; "
        "10% a year, 50% odds unless stated"
    ]
    for book in (estimate, pessimistic):
        out.append(
            f"{book.label}, by line\n"
            + "\n".join(
                headline([i], _by_mass(book, d)).splitlines()[-1]
                for (d, _), i in zip(pairs, labelled)
            )
        )
    out.append(
        "Estimate, steady break-even by odds and schedule\n"
        + "\n".join(
            rates([i], _by_mass(estimate, d)).splitlines()[-1]
            for (d, _), i in zip(pairs, labelled)
        )
    )
    out.append(
        "Estimate, steady break-even with the seed flown on a route (ADR 0039)\n"
        + "\n".join(
            seed_routes([i], estimate=_by_mass(estimate, d)).splitlines()[-1]
            for (d, _), i in zip(pairs, labelled)
        )
    )
    return "\n\n".join(out)


def ledger_report() -> str:
    """The S17 growth-ledger tables.

    Returns:
        The tables.
    """
    rows = _in_workers(_ledger_row_at, range(len(ledger_cases())))
    out = []
    for group in dict.fromkeys(r["group"] for r in rows):
        picked = [
            {k: v for k, v in r.items() if k != "group"}
            for r in rows
            if r["group"] == group
        ]
        out.append(f"{group}\n" + tabulate(picked, headers="keys", floatfmt=".4g"))
    return "\n\n".join(out)


#: The sphere's chamber, wall plus extension, which the methane price is for.
_SPHERE_UNIT = 19.0 + 2.19


def cost_designs() -> List[Design]:
    """The designs the cost book prices: methalox, both survivable area
    ratios at both pitch edges, S24's option (a) and every S24 nozzle at the
    faster rates that fit (finite-rate impulse, 24 kg pitch), the sphere's
    solved hydrogen (wall unverified) and the bulged hydrogen chamber.

    Returns:
        The designs.
    """
    return (
        [DESIGNS[0]]
        + [
            survivable_design(ar, p)
            for ar in AREA_RATIOS
            for p in SURVIVABLE_PITCH_RANGE
        ]
        + [survivable_design(150)]
        + [
            survivable_design(ar, pulse_rate=rate)
            for rate in s24_rates()
            if rate != SURVIVABLE_PULSE_RATE
            for ar in S24_AREA_RATIOS
        ]
        + [DESIGNS[4], HYDROGEN_BULGED_DESIGN]
    )


def _priced(index: int) -> Tuple[DesignChain, DesignInputs]:
    """The chain and cost inputs of ``cost_designs()[index]`` (in a worker)."""
    design = cost_designs()[index]
    return design_chain(design, PLATE), design_inputs(design, PLATE)


def _by_mass(prices: PriceBook, design: Design) -> PriceBook:
    """The book with the methane chamber priced in proportion to its mass."""
    assert design.pairing is not None
    scale = chamber_unit_mass(design.pairing).to_value(u.t) / _SPHERE_UNIT
    curve = prices.methane_chamber
    return replace(
        prices,
        label=f"{prices.label}, chamber x{scale:.2f}",
        methane_chamber=replace(
            curve, first=curve.first * scale, floor=curve.floor * scale
        ),
    )


def cost_report() -> str:
    """The cost book and seed tables behind the spray cup, survivable methane.

    Returns:
        The report.
    """
    designs = cost_designs()
    priced = _in_workers(_priced, range(len(designs)))
    inputs = [i for _, i in priced]
    books = [climbing_lob(b, OPERATING_RISE_SPEED) for b in SCENARIOS]
    low, high = seed_prices()
    out = [
        f"Spray cup, film carried as launched mass; lob x{booster_growth(OPERATING_RISE_SPEED):.4f} "
        f"(1.1 km/s climb, brake held back).  Seed ${low:.0f} / ${high:.0f} per kg; "
        "'a / b' is cheap / dear seed.  The hydrogen sphere's wall is unverified "
        "(S18); the bulged one is solved (ADR 0055).",
        "Hardware per launch unit, cycle 0\n" + hardware(inputs),
    ]
    for prices in books:
        out.append(
            f"{prices.label}: 10% a year, 50% odds the cycle works (ADR 0040)\n"
            + headline(inputs, prices)
        )
    estimate = next(b for b in books if b.label == ESTIMATE.label)
    pessimistic = next(b for b in books if b.label == PESSIMISTIC.label)
    for book in (estimate, pessimistic):
        scaled = [
            (d, i)
            for d, i in zip(designs, inputs)
            if d.pairing is not None and not d.pairing.hydrogen
        ]
        out.append(
            f"{book.label} with each methane chamber priced by its mass against the "
            f"sphere's {_SPHERE_UNIT:.1f} t\n"
            + "\n".join(
                headline([i], _by_mass(book, d)).splitlines()[-1] for d, i in scaled
            )
        )
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    out.append(
        "tab:seed_amortization behind the spray cup, stripped baseline\n"
        + _amortization(
            [c for c, _ in priced],
            STRIPPED_SHIPS[0],
            seed_price_range(seed_excess_speed(flown[0])),
        )
    )
    return "\n\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the S17 recompute.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ledger", action="store_true", help="the growth ledger")
    parser.add_argument("--cost", action="store_true", help="the cost book and seed")
    parser.add_argument("--cycle", action="store_true", help="S24's cycle timing")
    parser.add_argument(
        "--paper-grid", action="store_true", help="the paper's methane ledger cells"
    )
    parser.add_argument(
        "--paper-cost", action="store_true", help="the paper's methane cost cells"
    )
    args = parser.parse_args(argv)
    if args.paper_grid or args.paper_cost:
        if args.paper_grid:
            print(paper_grid_report())
        if args.paper_cost:
            print(paper_cost_report())
        return
    both = not (args.ledger or args.cost or args.cycle)
    print(
        f"Survivable methane chamber (ADR 0044, parent S17) behind {PLATE.label}, "
        f"{LAUNCH_UNIT:g} unit, {DEFAULT_PARKING_DAYS:g}-day orbit, gate charged, "
        "fixed-direction loss integrated and solved with the burn.  "
        + ", ".join(
            f"AR{ar}: eta {survivable_efficiency(ar):.3f}" for ar in AREA_RATIOS
        )
        + f" (sphere's solved {SOLVED_EFFICIENCY['CH4 7000 K']:g})."
    )
    if args.cycle or args.ledger or both:
        print()
        print(cycle_report())
    if args.ledger or both:
        print()
        print(ledger_report())
    if args.cost or both:
        print()
        print(cost_report())


if __name__ == "__main__":
    main()
