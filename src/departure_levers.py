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
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from tabulate import tabulate

from src.chamber_isp import HYDROGEN_BULGED, ChamberPairing, efficiency_for_impulse
from src.growth_ledger import DEFAULT_PARKING_DAYS, chain_growth, summarize_chain
from src.seed_cost import Design, chains
from src.survivable_chamber import (
    HYDROGEN_BULGED_DESIGN,
    PLATE,
    SURVIVABLE_PITCH_RANGE,
    survivable_design,
)

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
#: The temperature the bulged chamber is flown at (ADR 0064): the most impulse
#: per kilogram carried that its own charge can still cool by regeneration.
CHOSEN_TEMPERATURE = 5500.0


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


def cases() -> List[Tuple[str, Design]]:
    """Every lever, labelled."""
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


def lever_row(index: int) -> Dict[str, object]:
    """Fly one case over the flown chain (runs in a worker, passed by index)."""
    label, design = cases()[index]
    pairing = design.pairing
    assert pairing is not None and design.efficiency is not None
    flown, _ = chains(DEFAULT_PARKING_DAYS)
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
            chambers=design.chambers,
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
    """Print every lever's chain growth.

    Args:
        argv: ``--workers``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lever_row, range(len(cases()))))
    print(tabulate(rows, headers="keys", floatfmt=".3g"))


if __name__ == "__main__":
    main()
