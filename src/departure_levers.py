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
* **Hot hydrogen, unsolved** (:func:`hot_hydrogen`): the bulged chamber held at
  7000 K or 10000 K.  Each kilogram carries more energy (Cantera ``gri30``
  equilibrium at the chamber's density, extrapolated past its fits: 111 MJ/kg
  at 4884 K, 242 at 7000 K, 340-390 at 10000 K, so 360 is used), so the charge
  per rod that holds the temperature falls.  About 60% of that energy sits in
  broken bonds, and how much the nozzle gets back is unknown: rows at 94% (the
  5500 K sphere's AR300 figure), 70% and 50%.  The chamber's efficiency is
  scaled by the chemistry ceiling.  The wall, its mass, the pulse rate and the
  tanks per kilogram of hydrogen are held at the bulged chamber's, and it is
  flown at 2, 3 and 4 Hz: **none of it is solved**, which is what the
  hot-hydrogen ask (2026-10-14) requests.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from tabulate import tabulate

from src.chamber_isp import HYDROGEN_BULGED, ChamberPairing
from src.growth_ledger import DEFAULT_PARKING_DAYS, chain_growth, summarize_chain
from src.seed_cost import Design, chains
from src.survivable_chamber import (
    HYDROGEN_BULGED_DESIGN,
    PLATE,
    SURVIVABLE_PITCH_RANGE,
    survivable_design,
)

#: Hydrogen's equilibrium internal energy per kilogram at the bulged chamber's
#: charge density (0.57 kg/m^3), above H2 at 300 K (Cantera ``gri30``; past
#: ~6000 K its fits are extrapolated, and 10000 K takes the middle of 340-390).
HYDROGEN_ENERGY = {4884.0: 111.0e6, 7000.0: 242.0e6, 10000.0: 360.0e6}  # J/kg
#: Share of that energy in broken bonds when the charge is hot enough to be
#: mostly atomic.
HOT_BOND_SHARE = 0.60
#: Bonds the nozzle gives back: the 5500 K sphere's AR300 figure, then two
#: pessimistic ends.
RECOVERED = (0.94, 0.70, 0.50)
HOT_TEMPERATURES = (7000.0, 10000.0)
#: A smaller charge per rod gives less impulse per pulse, so at 2 Hz the hot
#: chamber burns longer or flies more chambers.  A smaller charge also fills
#: faster, so the faster rates are what the hot chamber would have to reach.
HOT_PULSE_RATES = (2.0, 3.0, 4.0)


def hot_hydrogen(temperature: float, recovered: float) -> Tuple[ChamberPairing, float]:
    """The bulged hydrogen chamber held at a hotter temperature, unsolved.

    The chamber's energy per kilogram ``u(T)`` holds ``(k + 1 + P) u = w^2 / 2``
    at the reference closing speed, so the reference charge scales as
    ``(k_ref + 1 + P) u_ref / u(T) - 1 - P``.  Efficiency is scaled by the new
    chemistry ceiling, ``1 - 0.60 (1 - recovered)``.

    Args:
        temperature: 7000 or 10000 K.
        recovered: Share of the bond energy the nozzle returns.

    Returns:
        (pairing, efficiency).
    """
    base = HYDROGEN_BULGED
    plug = base.plug_ratio
    ratio = HYDROGEN_ENERGY[4884.0] / HYDROGEN_ENERGY[temperature]
    slug = (base.reference_slug_ratio + 1.0 + plug) * ratio - 1.0 - plug
    ceiling = 1.0 - HOT_BOND_SHARE * (1.0 - recovered)
    eta0 = HYDROGEN_BULGED_DESIGN.efficiency
    assert eta0 is not None
    pairing = replace(
        base,
        name=f"H2 bulged {temperature:.0f} K, {recovered:.0%} back (unsolved)",
        temperature=temperature * u.K,
        reference_slug_ratio=float(slug),
        chemistry_ceiling=ceiling,
    )
    return pairing, eta0 * ceiling / base.chemistry_ceiling


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
    for temperature in HOT_TEMPERATURES:
        for back in RECOVERED:
            for rate in HOT_PULSE_RATES:
                pairing, eta = hot_hydrogen(temperature, back)
                pairing = replace(
                    pairing,
                    name=f"{pairing.name}, {rate:g} Hz",
                    pulse_rate=rate / u.s,
                )
                rows.append(
                    (
                        f"hydrogen {temperature:.0f} K, {back:.0%} back, {rate:g} Hz "
                        "(unsolved)",
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
