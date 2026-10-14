"""A slower outbound leg and a powered Jupiter flyby on the circular 3S cycle (ADR 0070).

The question (author, 2026-10-14): fly out to Jupiter slower, on a cheaper Earth
departure, pass deeper and burn at perijove to make up the time, and still
intercept Earth after exactly three synodic periods, perhaps faster.  The
Earth departure is a chamber burn and the Jupiter burn is methalox paid by the
whole batch.

Ideal circular, coplanar orbits and impulsive burns, as in
:mod:`src.circular_resonance_impulse`, whose Lambert pair this reuses.  That
module keeps only unpowered flybys, where the Jupiter-relative excess speed in
equals the speed out.  Here the mismatch is paid by a tangential perijove
burn, and the perijove is the radius at which the two hyperbolae, in and out,
bend through the turn the two legs need (:func:`perijove_for_turn`), no lower
than the 4,000 km floor.

Run ``make powered-resonance`` (about 30 s).
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from tabulate import tabulate

from src.circular_resonance_impulse import (
    _DEFAULT_ENCOUNTER_SAMPLES,
    _DEFAULT_PHASE_SAMPLES,
    _ENCOUNTER_MIN_YEARS,
    _RETURN_MIN_YEARS,
    _SECONDS_PER_YEAR,
    _evaluate_geometry,
    _synodic_period,
)
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.retrograde_return_legs import _FlybyParams, _powered_flyby_params
from src.two_wave_growth import VE_METHALOX

#: Jupiter's equatorial radius (km), for reporting perijoves.
JUPITER_RADIUS_KM = 71492.0
#: Chamber exhaust speeds the Earth departure is valued at (km/s): methane,
#: bulged hydrogen, and a generous hypothetical.
CHAMBER_EXHAUSTS = {"methane": 7.0, "hydrogen": 10.6, "hypothetical": 20.0}
#: Departure excess-speed bins the frontier is reported in (km/s).
FRONTIER_BINS = np.arange(8.5, 12.01, 0.5)


@dataclass(frozen=True)
class PoweredClosure:
    """One circular 3S (or other) closure with its Jupiter burn.

    Attributes:
        phase: Earth-minus-Jupiter longitude at departure (deg).
        outbound_years: Earth-to-Jupiter time of flight.
        departure_vinf: Earth departure excess speed (km/s).
        departure_burn: Burn from the cycle orbit's 200 km periapsis (km/s).
        jupiter_vinf_in: Excess speed arriving at Jupiter (km/s).
        jupiter_vinf_out: Excess speed the return leg needs (km/s).
        jupiter_turn: Turn the two legs need at Jupiter (deg).
        perijove: Perijove radius that gives that turn (km).
        jupiter_burn: Tangential perijove burn, signed (km/s).
        return_vinf: Earth return excess speed (km/s).
    """

    phase: float
    outbound_years: float
    departure_vinf: float
    departure_burn: float
    jupiter_vinf_in: float
    jupiter_vinf_out: float
    jupiter_turn: float
    perijove: float
    jupiter_burn: float
    return_vinf: float


def split_bend(perijove: float, vinf_in: float, vinf_out: float, mu: float) -> float:
    """The turn of a flyby whose two hyperbolae share one perijove (rad).

    Args:
        perijove: Perijove radius (km).
        vinf_in: Incoming excess speed (km/s).
        vinf_out: Outgoing excess speed (km/s).
        mu: Jupiter's gravitational parameter (km^3/s^2).

    Returns:
        The bend.
    """
    return float(
        np.arcsin(1.0 / (1.0 + perijove * vinf_in**2 / mu))
        + np.arcsin(1.0 / (1.0 + perijove * vinf_out**2 / mu))
    )


def perijove_for_turn(
    vinf_in: float, vinf_out: float, turn: float, mu: float, floor: float
) -> Optional[float]:
    """The perijove at which a flyby bends through ``turn``.

    Bend falls as the perijove rises, so the root is unique.

    Args:
        vinf_in: Incoming excess speed (km/s).
        vinf_out: Outgoing excess speed (km/s).
        turn: Turn the legs need (rad).
        mu: Jupiter's gravitational parameter (km^3/s^2).
        floor: Lowest admissible perijove radius (km).

    Returns:
        The radius, ``inf`` for no turn, or None if even the floor cannot bend
        that far.
    """
    if turn <= 1.0e-9:
        return float("inf")
    if split_bend(floor, vinf_in, vinf_out, mu) < turn:
        return None
    return float(
        brentq(lambda r: split_bend(r, vinf_in, vinf_out, mu) - turn, floor, 1.0e12)
    )


def perijove_burn(vinf_in: float, vinf_out: float, perijove: float, mu: float) -> float:
    """The tangential perijove burn that changes the excess speed (km/s).

    Args:
        vinf_in: Incoming excess speed (km/s).
        vinf_out: Outgoing excess speed (km/s).
        perijove: Perijove radius (km).
        mu: Jupiter's gravitational parameter (km^3/s^2).

    Returns:
        The burn, positive to speed up.
    """
    if not np.isfinite(perijove):
        return vinf_out - vinf_in
    escape_sq = 2.0 * mu / perijove
    return float(np.sqrt(vinf_out**2 + escape_sq) - np.sqrt(vinf_in**2 + escape_sq))


def powered_closures(
    synodic_multiple: int = 3,
    phase_samples: int = _DEFAULT_PHASE_SAMPLES,
    encounter_samples: int = _DEFAULT_ENCOUNTER_SAMPLES,
    params: Optional[_FlybyParams] = None,
) -> List[PoweredClosure]:
    """Every closure on the phase x encounter grid that a powered flyby can fly.

    Args:
        synodic_multiple: The cycle's length in synodic periods.
        phase_samples: Departure phases, over a full turn.
        encounter_samples: Jupiter encounter times.
        params: Physical inputs; the closed growth cycle's by default.

    Returns:
        The closures.
    """
    flyby = params or _powered_flyby_params(
        cycle_periapsis_speed=puffsat_cycle_periapsis_speed()
    )
    period = synodic_multiple * _synodic_period(flyby)
    out = []
    for phase in np.linspace(-np.pi, np.pi, phase_samples, endpoint=False):
        for encounter in np.linspace(
            _ENCOUNTER_MIN_YEARS * _SECONDS_PER_YEAR,
            period - _RETURN_MIN_YEARS * _SECONDS_PER_YEAR,
            encounter_samples,
        ):
            geometry = _evaluate_geometry(synodic_multiple, phase, encounter, flyby)
            if geometry is None:
                continue
            perijove = perijove_for_turn(
                geometry.jupiter_vinf_in,
                geometry.jupiter_vinf_out,
                geometry.jupiter_turn,
                flyby.mu_jupiter,
                flyby.periapsis_floor,
            )
            if perijove is None:
                continue
            out.append(
                PoweredClosure(
                    phase=float(np.degrees(phase)),
                    outbound_years=float(encounter / _SECONDS_PER_YEAR),
                    departure_vinf=geometry.departure_vinf,
                    departure_burn=geometry.departure_burn,
                    jupiter_vinf_in=geometry.jupiter_vinf_in,
                    jupiter_vinf_out=geometry.jupiter_vinf_out,
                    jupiter_turn=float(np.degrees(geometry.jupiter_turn)),
                    perijove=perijove,
                    jupiter_burn=perijove_burn(
                        geometry.jupiter_vinf_in,
                        geometry.jupiter_vinf_out,
                        perijove,
                        flyby.mu_jupiter,
                    ),
                    return_vinf=geometry.return_vinf,
                )
            )
    return out


def frontier(
    closures: Sequence[PoweredClosure], bins: Sequence[float] = tuple(FRONTIER_BINS)
) -> List[PoweredClosure]:
    """The least Jupiter burn in each departure excess-speed bin.

    Args:
        closures: Candidate closures.
        bins: Bin lower edges (km/s), each 0.5 km/s wide.

    Returns:
        One closure per bin that has any.
    """
    out = []
    for low in bins:
        within = [c for c in closures if low <= c.departure_vinf < low + 0.5]
        if within:
            out.append(min(within, key=lambda c: abs(c.jupiter_burn)))
    return out


def best_for(
    closures: Sequence[PoweredClosure], chamber_exhaust: float
) -> PoweredClosure:
    """The closure that delivers most, the departure at a chamber's exhaust and
    the Jupiter burn in methalox on the whole batch.

    Args:
        closures: Candidate closures.
        chamber_exhaust: The departure chamber's effective exhaust (km/s).

    Returns:
        The best closure.
    """
    return max(
        closures,
        key=lambda c: -c.departure_burn / chamber_exhaust
        - abs(c.jupiter_burn) / VE_METHALOX,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the powered 3S frontier and the best closure per chamber."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--synodic-multiple", type=int, default=3)
    args = parser.parse_args(argv)
    closures = powered_closures(args.synodic_multiple)
    print(
        f"{len(closures)} closures a powered flyby can fly, {args.synodic_multiple}S, "
        "circular and coplanar, impulsive burns.\n"
    )

    def row(c: PoweredClosure) -> Dict[str, object]:
        return {
            "dep v_inf": c.departure_vinf,
            "dep burn (200 km)": c.departure_burn,
            "Jupiter burn m/s": 1000.0 * c.jupiter_burn,
            "v_inf at Jupiter in/out": f"{c.jupiter_vinf_in:.2f} / {c.jupiter_vinf_out:.2f}",
            "turn deg": c.jupiter_turn,
            "perijove R_J": c.perijove / JUPITER_RADIUS_KM,
            "outbound yr": c.outbound_years,
            "return v_inf": c.return_vinf,
        }

    print("Least Jupiter burn per departure bin:")
    print(
        tabulate([row(c) for c in frontier(closures)], headers="keys", floatfmt=".2f")
    )
    print(
        "\nBest closure, departure at the chamber's exhaust, Jupiter burn in methalox:"
    )
    for name, exhaust in CHAMBER_EXHAUSTS.items():
        best = best_for(closures, exhaust)
        print(
            f"  {name} ({exhaust:g} km/s): dep v_inf {best.departure_vinf:.2f}, "
            f"Jupiter burn {1000 * best.jupiter_burn:+.0f} m/s, return {best.return_vinf:.2f}"
        )


if __name__ == "__main__":
    main()
