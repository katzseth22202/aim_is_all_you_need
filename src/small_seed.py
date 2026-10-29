"""The solar dive's seed: its tonnage at the headline, and a smaller one (S29, S63).

The parent's S63 item 4 asks which seed sits under the adopted headline (the
1 kg disk on a charge 1.2 times its radius, flown at 1.36 cm with the ship
holding its track; :mod:`src.disk_train_aim`); item 3 re-raises S29: is
anything indivisible below one 1500 t launch unit, and what do both
headlines give on a seed of one or two stripped Starships?

A smaller seed is one smaller unit (:func:`src.solar_dive_growth.sub_unit`),
lofted at a fraction of the 1500 t, and every generation stays one smaller
unit until its sheets fill a whole one.  Its push, plates and disks are sized
from it, but the Sun pulse is held at the 1 kg disk floor, so a smaller ship
flies fewer, heavier pulses on a shorter arc and keeps the full-size Sun
plate.  That plate is what is indivisible.

Run with ``make small-seed``.
"""

import argparse
import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from astropy import units as u
from scipy.optimize import brentq
from tabulate import tabulate

from src import disk_train_aim as dta
from src import solar_dive_growth as sdg
from src.seed_cost import STRIPPED_HIGH, STRIPPED_LOW, seed_payload

#: The closures both papers print (9.86 R_sun X 60, 4 R_sun X 90).
CLOSURES = (dta.CLOSURE, dta.DEEP_CLOSURE)
#: Seeds of one and two stripped Starships (S29 item 2).
SHIP_COUNTS = (1, 2)
#: Unit sizes the scale audit flies.
AUDIT_SIZES = (0.5, 0.25)
#: Lines whose share of the unit moves more than this are reported.
AUDIT_TOLERANCE = 0.02


@dataclass(frozen=True)
class Headline:
    """One closure's routes and its adopted design.

    Attributes:
        closure: (perihelion R_sun, return excess km/s).
        departure: The departure-wave route.
        push: The push-wave route.
        design: The dear-seed optimum.
        label: What design this is.
    """

    closure: Tuple[float, float]
    departure: sdg.DiveRoute
    push: sdg.DiveRoute
    design: sdg.Design
    label: str


def headline(closure: Tuple[float, float], adopted: bool = True) -> Headline:
    """The adopted headline (x1.2 charge, held track) or ADR 0085's, at a closure."""
    disk = dta.HEADLINE_X12 if adopted else dta.HEADLINE
    hold = dta.headline_hold(disk) if adopted else 0.0
    priced = dta.price(disk, hold, offset_loss=adopted, closure=closure)
    departure = sdg.departure_route(*closure)
    push = sdg.push_route(*closure, sdg.PARKING_DAYS)
    assert departure is not None and push is not None
    label = (
        "x1.2, 1.36 cm, held (adopted)" if adopted else "matched, centred (ADR 0085)"
    )
    return Headline(closure, departure, push, priced.optimum.design, label)


def seed_mass(h: Headline, units: float = 1.0) -> float:
    """The seed's departing ships (kg), one unit ``units`` the size."""
    pair = (
        sdg.sub_unit(h.departure, h.push, h.design, units)
        if units < 1.0
        else (
            sdg.fly_unit(h.push, h.push, h.departure, h.design),
            sdg.fly_unit(h.departure, h.push, h.departure, h.design),
        )
    )
    share = sdg.balance(*pair, 1.0 - sdg.MISS_FLOOR).push_share
    return share * pair[0].lines["departing"] + (1 - share) * pair[1].lines["departing"]


def ship_payload(h: Headline, dear: bool) -> float:
    """What one stripped Starship sends at the seed's excess (kg), cheap or dear ship."""
    ship = (STRIPPED_HIGH if dear else STRIPPED_LOW)[0]
    excess = h.departure.departure_excess * u.km / u.s
    return float(seed_payload(ship, excess).payload.to_value(u.kg))


# --------------------------------------------------------------------------
# Item 4: the headline seed's tonnage
# --------------------------------------------------------------------------


def tonnage_rows() -> List[List[str]]:
    """``tab:seed_amortization``'s first columns, both headlines, both closures."""
    rows = []
    for closure in CLOSURES:
        for adopted in (False, True):
            h = headline(closure, adopted)
            mass = seed_mass(h)
            ships = [mass / ship_payload(h, dear) for dear in (False, True)]
            prices = [sdg.seed_price(h.departure, dear) for dear in (False, True)]
            rows.append(
                [
                    dta._closure_label(closure),
                    h.label,
                    f"{h.departure.departure_excess:.2f}",
                    f"{mass / 1e3:.1f}",
                    f"{ships[0]:.2f} / {ships[1]:.2f}",
                    f"{prices[0] * mass / 1e9:.2f} / {prices[1] * mass / 1e9:.2f}",
                    f"{prices[0]:.0f} / {prices[1]:.0f}",
                ]
            )
    return rows


# --------------------------------------------------------------------------
# Item 3 (S29): what does not scale, and the smaller seed
# --------------------------------------------------------------------------


def scale_audit(h: Headline, units: float) -> List[Tuple[str, float, float]]:
    """Mass lines of the departure-wave ship whose share moves when it shrinks.

    Returns:
        ``(line, full-size share of departing, small share)`` where the share
        moved by more than :data:`AUDIT_TOLERANCE` of itself.
    """
    full = sdg.fly_unit(h.departure, h.push, h.departure, h.design).lines
    small = sdg.sub_unit(h.departure, h.push, h.design, units)[1].lines
    out = []
    for key in ("sun_plate", "bore_collar", "push_bag", "bowl", "disks", "sun_hold"):
        a, b = full[key] / full["departing"], small[key] / small["departing"]
        if a > 0 and abs(b / a - 1.0) > AUDIT_TOLERANCE:
            out.append((key, a, b))
    return out


@dataclass(frozen=True)
class SmallSeed:
    """A seed of whole stripped Starships at one end.

    Attributes:
        ships: Starships.
        dear: The dear end (60 t ship, $20M hull, Morgan Stanley) or the cheap.
        units: Its size in launch units.
        valuation: Valued with 15 synodic periods of growth.
        lag_years: Years more growth its fleet needs to match a one-unit
            seed's harvest, at the steady growth.
        lag_cycles: The same in whole three-synodic cycles, as the
            valuation's growth phase can be lengthened.
    """

    ships: int
    dear: bool
    units: float
    valuation: sdg.Valuation
    lag_years: float
    lag_cycles: int


def small_seed(h: Headline, ships: int, dear: bool) -> SmallSeed:
    """Value a seed of ``ships`` stripped Starships on one end's ship and price."""
    target = ships * ship_payload(h, dear)
    units = brentq(lambda f: seed_mass(h, f) - target, 0.05, 1.0, xtol=1e-6)
    price = sdg.seed_price(h.departure, dear)

    def value(synodic: int) -> sdg.Valuation:
        return sdg.value_design(
            h.departure,
            h.push,
            h.design,
            seed_price=price,
            growth_synodic=synodic,
            seed_units=units,
        )

    full = sdg.value_design(h.departure, h.push, h.design, seed_price=price)
    first = value(sdg.GROWTH_SYNODIC)
    lag = (
        math.log(full.fleet_multiple / first.fleet_multiple)
        / math.log(full.growth)
        * h.departure.cycle_years
    )
    cycles, matched = 0, first
    while matched.fleet_multiple < full.fleet_multiple:
        cycles += 1
        matched = value(sdg.GROWTH_SYNODIC + 3 * cycles)
    return SmallSeed(ships, dear, units, first, lag, cycles)


def small_rows() -> List[List[str]]:
    """S29 item 2 at both closures: one unit, one ship and two ships."""
    rows = []
    for closure in CLOSURES:
        h = headline(closure)
        for dear in (False, True):
            price = sdg.seed_price(h.departure, dear)
            full = sdg.value_design(h.departure, h.push, h.design, seed_price=price)
            cases = [("one unit", 1.0, full, 0.0, 0)] + [
                (
                    f"{s.ships} ship{'s' if s.ships > 1 else ''}",
                    s.units,
                    s.valuation,
                    s.lag_years,
                    s.lag_cycles,
                )
                for s in (small_seed(h, n, dear) for n in SHIP_COUNTS)
            ]
            for label, units, v, years, cycles in cases:
                rows.append(
                    [
                        dta._closure_label(closure),
                        "dear" if dear else "cheap",
                        label,
                        f"{units:.3f}",
                        f"{v.seed_mass / 1e3:.0f}",
                        f"{v.seed_outlay / 1e9:.2f}",
                        f"{sdg._money(v.break_even)}",
                        f"{v.fleet_multiple:.1f}",
                        f"{v.seed_per_fleet_kg:.0f}",
                        f"+{years:.1f} ({cycles})",
                    ]
                )
    return rows


def generation_rows(h: Headline, units: float) -> List[List[str]]:
    """A sub-unit ship beside the full one: arc, pulses, plate and growth."""
    rows = []
    for size in (1.0, units):
        pair = (
            sdg.sub_unit(h.departure, h.push, h.design, size)
            if size < 1.0
            else (
                sdg.fly_unit(h.push, h.push, h.departure, h.design),
                sdg.fly_unit(h.departure, h.push, h.departure, h.design),
            )
        )
        g = sdg.balance(*pair, 1.0 - sdg.MISS_FLOOR)
        d = pair[1].lines
        arc = (
            sdg.floor_arc(
                h.departure, h.push, replace(h.design, unit_mass=size * sdg.LAUNCH_UNIT)
            )
            if size < 1.0
            else h.design.arc_hours
        )
        rows.append(
            [
                f"{size:.3f}",
                f"{min(arc, h.design.arc_hours):.3f}",
                f"{d['pulse_disk']:.3f}",
                f"{(pair[0].lines['sun_plate'] + d['sun_plate']) / 2e3:.1f}",
                f"{d['departing'] / 1e3:.0f}",
                f"{g.growth:.3f}",
            ]
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the seed's tonnage (S63 item 4) and the smaller seed (S29).

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    print(
        "The one-unit seed (S63 item 4): one launch unit's departing ships in the"
        " steady mix of roles\n"
        + tabulate(
            tonnage_rows(),
            headers=[
                "closure",
                "design",
                "seed excess",
                "seed (t)",
                "ships 40 / 60 t",
                "launch $B cheap / dear",
                "$/kg sent",
            ],
        )
    )
    for closure in CLOSURES:
        h = headline(closure)
        print(f"\nWhat does not scale at {dta._closure_label(closure)} (S29 item 1)")
        for size in AUDIT_SIZES:
            moved = scale_audit(h, size)
            print(
                f"  at {size:g} units: "
                + ", ".join(f"{k} {a:.3f} -> {b:.3f} of departing" for k, a, b in moved)
            )
        one_ship = small_seed(h, 1, False).units
        print(
            tabulate(
                generation_rows(h, one_ship),
                headers=[
                    "units",
                    "arc (h)",
                    "disk a pulse (kg)",
                    "Sun plate, mean (t)",
                    "departure ship (t)",
                    "growth",
                ],
            )
        )
    print(
        "\nThe smaller seed (S29 item 2): 15 synodic periods of growth; the years"
        " more its fleet\nneeds to match the one-unit seed's harvest at the steady"
        " growth (whole 3S cycles)\n"
        + tabulate(
            small_rows(),
            headers=[
                "closure",
                "end",
                "seed",
                "units",
                "seed (t)",
                "up-front $B",
                "break-even",
                "fleet x",
                "seed $/fleet kg",
                "years to match (cycles)",
            ],
        )
    )


if __name__ == "__main__":
    main()
