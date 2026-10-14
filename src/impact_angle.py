"""What Earth's bending of the impact angle is worth on the flown departures (ADR 0066).

The ledger charges every pulse as exactly head-on: the rod arrives at the
vehicle's speed plus its own and its whole momentum is a debit
(:func:`src.chamber_isp.chamber_departure_burn`).  During a real burn Earth
turns the vehicle, so the impacts come in some degrees off head-on, the closing
speed falls, and part of the rod's momentum goes sideways.

For each flown cycle this flies the chamber's own ledger burn (its length and
mean exhaust speed) at the placement that keeps the stream in view
(:func:`src.departure_visibility.visible_burn`), and recomputes every pulse at
the actual relative velocity: the charge re-sized to the closing speed, the
axial debit ``w cos(theta)``, and the uncancelled sideways momentum.  It
reports the chamber's Isp three ways and the delivered-mass gain over the
ledger's head-on charge.

Run ``make impact-angle``.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.constants import g0
from tabulate import tabulate

from src.chamber_isp import GATE_THRUST_COST, ChamberPairing, slug_ratio_at
from src.departure_visibility import (
    OnwardStream,
    _in_view,
    burn_trajectory,
    stored_tables,
    visible_burn,
)
from src.growth_ledger import (
    DEFAULT_PARKING_DAYS,
    DEPARTURE_ALTITUDE,
    chain_growth,
    departure_at_altitude,
    hydrogen_boil_off,
    parking_period,
)
from src.orbit_turn_analysis import _incoming_routes
from src.seed_cost import Design, chains
from src.survivable_chamber import HYDROGEN_BULGED_DESIGN, PLATE, survivable_design

_G0 = float(g0.to_value(u.km / u.s**2))


@dataclass(frozen=True)
class AngleLedger:
    """One departure burn's pulses at their actual impact angle.

    Attributes:
        off_head_on_start: Impact angle off head-on at ignition (degrees).
        off_head_on_end: The same at cutoff.
        gross_isp: Exhaust momentum per kilogram carried, debit excluded (s).
        head_on_isp: Net of the head-on debit the ledger charges (s).
        bent_isp: Net of the axial debit at the actual angle (s).
        sideways: Mean uncancelled sideways momentum per kilogram carried (km/s).
        mass_gain: Delivered mass at the actual angle over the head-on charge.
    """

    off_head_on_start: float
    off_head_on_end: float
    gross_isp: float
    head_on_isp: float
    bent_isp: float
    sideways: float
    mass_gain: float


def _per_carried(
    closing: float, pairing: ChamberPairing, efficiency: float, pitch_ratio: float
) -> Tuple[float, float]:
    """Gross exhaust momentum and the carried mass, per kilogram of rod."""
    slug = slug_ratio_at(closing * u.km / u.s, pairing)
    carried = slug + pairing.plug_ratio + pitch_ratio
    exhausted = carried + 1.0
    exhaust = closing * np.sqrt(efficiency / exhausted)
    return (1.0 - GATE_THRUST_COST) * exhausted * exhaust, carried


def angle_ledger(
    stream: OnwardStream,
    burn_time: float,
    exhaust_speed: float,
    design: Design,
    period_days: float = DEFAULT_PARKING_DAYS,
) -> AngleLedger:
    """Recompute one departure's pulses at their actual impact angle.

    Args:
        stream: The stream the departure meets.
        burn_time: The ledger's burn length (s).
        exhaust_speed: The ledger burn's mean effective exhaust (km/s), which
            sets the mass-flow profile.
        design: The chamber.
        period_days: Parking-orbit period.

    Returns:
        The angle ledger.

    Raises:
        RuntimeError: If no placement keeps the stream in view.
    """
    pairing, efficiency = design.pairing, design.efficiency
    assert pairing is not None and efficiency is not None
    placement = visible_burn(stream, burn_time, period_days)
    if placement is None:
        raise RuntimeError("no placement keeps the stream in view")
    flown = burn_trajectory(
        stream, burn_time, placement.share_before, period_days, exhaust_speed
    )
    axis = np.array([0.0, 1.0])
    gross, head_on, bent, sideways, off = [], [], [], [], []
    for position, velocity in zip(flown.states[:2].T, flown.states[2:].T):
        if not _in_view(
            flown.mu, flown.earth_radius, position, stream.return_vinf, flown.incoming
        ):
            raise RuntimeError("the visible placement lost the stream")
        routes = [
            r
            for r in _incoming_routes(
                flown.mu,
                flown.earth_radius,
                position,
                stream.return_vinf,
                flown.incoming,
            )
            if r.visible
        ]
        route = min(routes, key=lambda r: abs(r.impact_parameter - flown.anchor))
        relative = route.velocity - velocity
        closing = float(np.linalg.norm(relative))
        cosine = float(relative @ axis / closing)
        momentum, carried = _per_carried(
            closing, pairing, efficiency, design.pitch_ratio
        )
        gross.append(momentum / carried)
        bent.append((momentum + closing * cosine) / carried)
        sideways.append(closing * np.sqrt(max(0.0, 1.0 - cosine * cosine)) / carried)
        off.append(np.degrees(np.pi - np.arccos(np.clip(cosine, -1.0, 1.0))))
        ledger_closing = float(np.linalg.norm(route.velocity)) + float(velocity @ axis)
        ledger_momentum, ledger_carried = _per_carried(
            ledger_closing, pairing, efficiency, design.pitch_ratio
        )
        head_on.append((ledger_momentum - ledger_closing) / ledger_carried)

    def log_ratio(exhaust: Sequence[float]) -> float:
        return float(
            np.trapezoid(flown.acceleration / np.asarray(exhaust), flown.times)
        )

    ideal = flown.ideal_burn
    return AngleLedger(
        off_head_on_start=off[0],
        off_head_on_end=off[-1],
        gross_isp=ideal / log_ratio(gross) / _G0,
        head_on_isp=ideal / log_ratio(head_on) / _G0,
        bent_isp=ideal / log_ratio(bent) / _G0,
        sideways=float(np.mean(sideways)),
        mass_gain=float(np.exp(log_ratio(head_on) - log_ratio(bent))),
    )


def flown_angles(design: Design) -> List[Dict[str, object]]:
    """Every flown cycle's departure for one chamber.

    Args:
        design: The chamber.

    Returns:
        One row per cycle.
    """
    pairing, efficiency = design.pairing, design.efficiency
    assert pairing is not None and efficiency is not None
    flown, _ = chains(DEFAULT_PARKING_DAYS)
    storage = pairing.storage
    grown = chain_growth(
        flown,
        PLATE.efficiency,
        pairing,
        efficiency,
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
    )
    rows = []
    for cycle, ledger, (stream, _) in zip(
        flown, grown, stored_tables(flown, DEFAULT_PARKING_DAYS)
    ):
        departure = ledger.departure
        burn = float(departure.burn_time.to_value(u.s))
        total = float(departure.finite_burn_loss.to_value(u.km / u.s))
        impulsive = departure_at_altitude(
            cycle.onward_burn * u.km / u.s, DEPARTURE_ALTITUDE, parking_period(cycle)
        ).burn.to_value(u.km / u.s)
        exhaust = (impulsive + total) / np.log(1.0 / departure.delivered_fraction)
        angles = angle_ledger(stream, burn, exhaust, design)
        rows.append(
            {
                "cycle": cycle.index,
                "aim sep deg": stream.aim_separation,
                "burn min": burn / 60.0,
                "off head-on deg": f"{angles.off_head_on_start:.1f} -> {angles.off_head_on_end:.1f}",
                "gross Isp s": angles.gross_isp,
                "head-on Isp s": angles.head_on_isp,
                "bent Isp s": angles.bent_isp,
                "sideways km/s": angles.sideways,
                "mass gain": angles.mass_gain,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print each flown departure's impact-angle ledger for both chambers."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    for label, design in (
        ("Bulged hydrogen, 2 Hz", HYDROGEN_BULGED_DESIGN),
        ("Methane AR100, 24 kg", survivable_design(100)),
    ):
        print(
            f"{label}: each burn at its ledger length, placed to keep the stream in view"
        )
        print(tabulate(flown_angles(design), headers="keys", floatfmt=".4g") + "\n")


if __name__ == "__main__":
    main()
