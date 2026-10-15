"""Which departure lever moves growth most? A ranking on the flown chain (ADR 0063).

The departure burn is where growth goes.  On cycle 0 the methane chamber turns
an 805 t stack into 178 t sent onward: a 7.30 km/s burn plus 0.84 km/s of
fixed-direction loss at about 7 km/s of effective exhaust, then 56 t of chamber
and 12 t of tanks expended.  The author asked (2026-10-14) what to fix first.
Each row here is one lever, flown through :func:`src.growth_ledger.chain_growth`
behind today's push (the spray cup, film carried), on the flown chain:

* **Methane chamber** (AR100, 5 kg rod, 2 Hz, 24 kg pitch): at 3 Hz (the 2 t/s
  fill, ADR 0056), with two chambers, with the light 3.5 kg pitch, and 3 Hz with
  light pitch together.
* **Bulged hydrogen chamber** (4884 K, 2 Hz): at 2.4 Hz (the top of its
  2.2-2.4 Hz cycle, ADR 0055) and with two chambers.
* **Hotter hydrogen, solved** (:func:`solved_hydrogen`): the bulged chamber
  at 4884 (today), 5500, 6000 and 7000 K, each with the impact sim's charge,
  impulse, pulse rate and wrap (ADR 0064).  ADR 0063 first flew these as
  estimates scaled by a chemistry ceiling; the sim found the bonds come back at
  60-71%, not 94%, and the hot charge stops shrinking because the rod and plug
  store little energy.  10 000 K fails the wrap and is not flown.

Every hydrogen row is flown as 50% slush, as every hydrogen pairing now is
(ADR 0065), with its cryostat and hold boil-off charged as :mod:`src.seed_cost`
charges them; one liquid row is kept for comparison.  The rows used to charge
no storage at all, which flattered hydrogen against methane.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from tabulate import tabulate

from src.chamber_isp import (
    HYDROGEN_BULGED,
    LIQUID_HYDROGEN,
    ChamberPairing,
    efficiency_for_impulse,
    stored_as,
)
from src.departure_visibility import stored_tables
from src.finite_burn_loss import fixed_direction_loss
from src.growth_ledger import (
    DEFAULT_PARKING_DAYS,
    _orbit_loss,
    chain_growth,
    hydrogen_boil_off,
    parking_period,
    summarize_chain,
)
from src.seed_cost import Design, chains
from src.survivable_chamber import (
    HYDROGEN_BULGED_DESIGN,
    PLATE,
    SURVIVABLE_PITCH_RANGE,
    survivable_design,
)
from src.two_wave_growth import TwoWaveCycle

#: The bulged chamber at each temperature, solved by the impact sim (@ ca07938,
#: its ADR-0058 and ``walled_nozzle_hot_hydrogen.md``): the charge per rod at
#: 75 km/s (``solve_charge``, rod and plug included), net impulse per pulse at
#: A/A* = 300 with finite-rate recombination and the wall heat debited, and the
#: pulse rate sealed at 0.3 bar on the 0.5 t/s fill (today's row reproduces
#: the 2.2-2.4 Hz the bulge is flown at).  The dry wrap is solved in 2-D only
#: at 4884 K (37.0 t) and 7000 K (29.8 t); 5500 K and 6000 K fills lie between
#: two that pass, and are held at 37.0 t.
#: ``temperature: (k, impulse kN s, rate Hz, wall t)``.
SOLVED_HYDROGEN: Dict[float, Tuple[float, float, float, float]] = {
    4884.0: (23.97, 661.0, 2.27, 37.0),
    5500.0: (16.13, 485.0, 2.66, 37.0),
    6000.0: (12.65, 402.0, 2.94, 37.0),
    7000.0: (9.65, 327.0, 3.28, 29.8),
}
#: The temperature the bulged chamber is flown at (ADR 0065, reversing ADR
#: 0064's 5500 K): its fixed 60 kg charge.  A hotter charge's exhaust gains
#: only 6% per kilogram carried, because the bonds come back at 60-71%; each
#: rod then buys 27% less impulse, which nearly cancels it, and the thrust lost
#: costs a second chamber.
CHOSEN_TEMPERATURE = 4884.0


def solved_hydrogen(temperature: float) -> Tuple[ChamberPairing, float]:
    """The bulged hydrogen chamber at one of the impact sim's solved temperatures.

    Args:
        temperature: A key of :data:`SOLVED_HYDROGEN`.

    Returns:
        (pairing, efficiency), the efficiency inverted from the solved impulse.
    """
    slug, impulse, rate, wall = SOLVED_HYDROGEN[temperature]
    pairing = replace(
        HYDROGEN_BULGED,
        name=f"H2 bulged {temperature:.0f} K, {rate:g} Hz",
        temperature=temperature * u.K,
        reference_slug_ratio=slug,
        wall_mass=wall * u.t,
        pulse_rate=rate / u.s,
    )
    return pairing, efficiency_for_impulse(impulse * u.kN * u.s, pairing)


def thrust_matched(temperature: float, reference: float = 4884.0) -> float:
    """The pulse rate at which a solved row matches another's thrust (Hz).

    Thrust is impulse per pulse times pulse rate.

    Args:
        temperature: The row to speed up.
        reference: The row whose thrust it matches.

    Returns:
        The rate.
    """
    _, impulse, _, _ = SOLVED_HYDROGEN[temperature]
    _, ref_impulse, ref_rate, _ = SOLVED_HYDROGEN[reference]
    return ref_impulse * ref_rate / impulse


#: The parking orbit the rankings before ADR 0071 flew (days).
EARLIER_PARKING_DAYS = 20.0


def _flown(earlier: bool) -> Tuple[TwoWaveCycle, ...]:
    """The flown chain, or the one the rankings before ADRs 0066 and 0071 flew.

    Args:
        earlier: The 20-day orbit with one batch per window (no role split);
            the caller also centres the burn.

    Returns:
        The chain.
    """
    if not earlier:
        return chains(DEFAULT_PARKING_DAYS)[0]
    return tuple(
        replace(c, onward_growth_burn=float("nan"))
        for c in chains(EARLIER_PARKING_DAYS)[0]
    )


def temperature_row(case: Tuple[float, Optional[float], bool]) -> Dict[str, object]:
    """Cycle 0 of one solved temperature, the comparison behind ADR 0065.

    Args:
        case: (temperature, pulse rate in Hz or None for the solved one,
            centred), where ``centred`` reproduces ADR 0065's ledger: the burn
            centred on periapsis with the stream ignored, one batch rather
            than launch units split by role, on the 20-day orbit (ADR 0071).

    Returns:
        The row.
    """
    temperature, rate, centred = case
    pairing, eta = solved_hydrogen(temperature)
    if rate is not None:
        pairing = replace(pairing, pulse_rate=rate / u.s)
    flown = _flown(centred)
    first = flown[:1]
    storage = pairing.storage
    assert storage is not None
    grown = chain_growth(
        first,
        PLATE.efficiency,
        pairing,
        eta,
        max_slug_ratio=PLATE.max_slug_ratio,
        impactor_bond_energy=PLATE.impactor_bond_energy,
        film_per_impulse=PLATE.film_per_impulse,
        cryostat_fraction=storage.cryostat_fraction,
        boil_off=hydrogen_boil_off(flown, storage.boil_off_per_day),
        loss_model=(
            _orbit_loss(fixed_direction_loss, parking_period(first[0]))
            if centred
            else None
        ),
    )[0]
    departure = grown.departure
    stack = float(grown.departing_stack.to_value(u.t))
    push = stack / float(grown.puffsats.to_value(u.t))
    _, impulse, _, _ = SOLVED_HYDROGEN[temperature]
    hz = float(pairing.pulse_rate.to_value(1 / u.s))
    return {
        "temperature K": temperature,
        "rate Hz": hz,
        "thrust MN": impulse * hz / 1000.0,
        "chambers": departure.chambers,
        "delivered net": departure.delivered_net,
        "rods kg/t stack": 1000.0 * departure.rod_mass_fraction,
        "share pushing": 1.0 / (1.0 + departure.rod_mass_fraction * push),
        "growth": grown.growth,
    }


def temperature_cases(centred: bool) -> List[Tuple[float, Optional[float], bool]]:
    """ADR 0065's rows: 4884 K and 5500 K solved, and 5500 K at 4884 K's thrust."""
    return [
        (4884.0, None, centred),
        (5500.0, None, centred),
        (5500.0, thrust_matched(5500.0), centred),
    ]


def cases() -> List[Tuple[str, Design]]:
    """Every lever, labelled.  Hydrogen flies as slush but for one liquid row."""
    light = SURVIVABLE_PITCH_RANGE[0]
    rows: List[Tuple[str, Design]] = [
        ("methane, baseline", survivable_design(100)),
        ("methane, 3 Hz", survivable_design(100, pulse_rate=3.0 / u.s)),
        ("methane, two chambers", survivable_design(100, chambers=2)),
        ("methane, 3.5 kg pitch", survivable_design(100, pitch=light)),
        (
            "methane, 3 Hz + 3.5 kg pitch",
            survivable_design(100, pitch=light, pulse_rate=3.0 / u.s),
        ),
        (
            "hydrogen, baseline, liquid",
            replace(
                HYDROGEN_BULGED_DESIGN,
                pairing=stored_as(HYDROGEN_BULGED, LIQUID_HYDROGEN),
            ),
        ),
        ("hydrogen, baseline", HYDROGEN_BULGED_DESIGN),
        (
            "hydrogen, 2.4 Hz",
            replace(
                HYDROGEN_BULGED_DESIGN,
                label="Hydrogen, bulged, 2.4 Hz",
                pairing=replace(HYDROGEN_BULGED, pulse_rate=2.4 / u.s),
            ),
        ),
        (
            "hydrogen, two chambers",
            replace(HYDROGEN_BULGED_DESIGN, label="Hydrogen, bulged, n=2", chambers=2),
        ),
        (
            "hydrogen, two chambers, 2.4 Hz",
            replace(
                HYDROGEN_BULGED_DESIGN,
                label="Hydrogen, bulged, n=2, 2.4 Hz",
                pairing=replace(HYDROGEN_BULGED, pulse_rate=2.4 / u.s),
                chambers=2,
            ),
        ),
    ]
    for temperature in SOLVED_HYDROGEN:
        pairing, eta = solved_hydrogen(temperature)
        rows.append(
            (
                f"hydrogen {temperature:.0f} K, solved",
                Design(pairing.name, pairing, eta),
            )
        )
    return rows


def lever_row(index: int, centred: bool = False) -> Dict[str, object]:
    """Fly one case over the flown chain (runs in a worker, passed by index).

    Args:
        index: The case's index in :func:`cases`.
        centred: Fly the ranking before ADR 0066: the burn centred on
            periapsis with the stream ignored, one batch per window on the
            20-day orbit (ADR 0071).

    Returns:
        The row.
    """
    label, design = cases()[index]
    pairing = design.pairing
    assert pairing is not None and design.efficiency is not None
    flown = _flown(centred)
    storage = pairing.storage
    try:
        grown = chain_growth(
            flown,
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
                else hydrogen_boil_off(flown, storage.boil_off_per_day)
            ),
            chambers=design.chambers,
            loss_model=(
                _orbit_loss(fixed_direction_loss, parking_period(flown[0]))
                if centred
                else None
            ),
        )
    except (RuntimeError, ValueError) as error:
        return {"lever": label, "doubling yr": str(error)}
    summary = summarize_chain(
        [c.period_years for c in flown], [g.growth for g in grown]
    )
    deps = [g.departure for g in grown]
    return {
        "lever": label,
        "eta": design.efficiency,
        "k_75": pairing.reference_slug_ratio,
        "doubling yr": summary.doubling_years,
        "annual %": 100.0 * summary.annual_growth,
        "10-yr x": summary.ten_year_stepwise,
        "G": f"{min(g.growth for g in grown):.2f}-{max(g.growth for g in grown):.2f}",
        "onward share": f"{min(d.delivered_net for d in deps):.3f}-"
        f"{max(d.delivered_net for d in deps):.3f}",
        "tanks": float(np.mean([d.tank_share for d in deps])),
        "loss km/s": float(
            np.mean([d.finite_burn_loss.to_value(u.km / u.s) for d in deps])
        ),
        "burn min": float(np.mean([d.burn_time.to_value(u.min) for d in deps])),
        "n": f"{min(d.chambers for d in deps)}-{max(d.chambers for d in deps)}",
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print each cycle's stream and every lever's chain growth.

    Args:
        argv: ``--workers``; ``--centred`` flies the burn centred on periapsis
            with the stream ignored, the ranking before ADR 0066;
            ``--temperatures`` prints ADR 0065's cycle-0 comparison instead.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--centred", action="store_true")
    parser.add_argument(
        "--temperatures",
        action="store_true",
        help="cycle 0 at 4884 K and 5500 K, and 5500 K at 4884 K's thrust (ADR 0065)",
    )
    args = parser.parse_args(argv)
    if args.temperatures:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(temperature_row, temperature_cases(args.centred)))
        print(tabulate(rows, headers="keys", floatfmt=".4g"))
        return
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    streams = [
        {
            "cycle": stream.index,
            "S": stream.synodic_multiple,
            "aim sep deg": stream.aim_separation,
            "v_inf out": stream.departure_vinf,
            "v_inf in": stream.return_vinf,
        }
        for stream, _ in stored_tables(flown, DEFAULT_PARKING_DAYS)
    ]
    print(tabulate(streams, headers="keys", floatfmt=".4g") + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(
            pool.map(partial(lever_row, centred=args.centred), range(len(cases())))
        )
    print(tabulate(rows, headers="keys", floatfmt=".3g"))


if __name__ == "__main__":
    main()
