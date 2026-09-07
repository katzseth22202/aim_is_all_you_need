"""Fly a shorter, hotter Jupiter cycle and park the remainder to hold the clock.

The phased growth chain (:mod:`src.jovian_cycle_phasing`, ADR 0010 as corrected
by ADR 0030) settles onto cycles of exactly three Earth-Jupiter synodic periods
without being told to. That is not a preference for slowness: it is forced by the
**sweet phase**.

Vocabulary, defined once here because the rest of the module leans on it:

* **Departure phase** -- where Earth and Jupiter stand relative to one another
  when the payload leaves Earth, expressed as a fraction of one Earth-Jupiter
  synodic period (1.0923 yr). Phase 0.00 and phase 1.00 are the same geometry.
* **Sweet phase** -- the departure phase at which *reaching Jupiter is cheapest*.
  The cost of getting to Jupiter swings by roughly nine-fold around the circle
  (about 4.4 km/s of departure burn at the sweet phase against 38.5 km/s at the
  worst), because the transfer has to arrive where Jupiter actually is.
  :func:`sweet_phase` computes it rather than asserting it.
* **Usable phase** -- a departure phase from which at least one closing cycle
  *grows* the payload (``net_growth > 1``) at a given departure exhaust speed. A
  phase can be perfectly reachable and still be unusable, if every trajectory
  leaving it costs more mass than the arriving impactor mints.
* **Fly-and-park** -- flying a trajectory *shorter* than the 3S window and
  parking in the existing bound near-escape orbit for the remainder, so that
  flight + park is exactly 3.00 synodic periods and the next departure lands back
  on the same phase. Nothing drifts; the clock is preserved exactly.

The point of padding every candidate to a common 3.00 S is that **cycle time
cancels from the comparison**. What is left is a single exchange rate: how much
extra departure burn a hotter arrival speed is worth before the propellant it
costs eats the gain (:func:`exchange_rate`). That rate is set by the departure
stage's exhaust speed, and it decides the architecture:

* At **methalox** (Isp 380 s, ``v_e`` = 3.727 km/s) fly-and-park **loses**. It
  misses by about 5% -- the manoeuvre costs ~1.04 km/s of extra burn against a
  budget of 0.99.
* At any **impactor-driven** departure (ADR 0009/0012's head-on nozzle, Isp
  1200-2214 s) it **wins**, with three to five times the headroom it needs, and
  the arrival lands at 67-69 km/s -- the perfect-retrograde boundary the paper's
  catalog rows already assume -- with a zero perijove burn.

The same exhaust speed decides the launch cadence: the fraction of departure
phases that are **usable** runs 18% at Isp 380 and reaches 100% at Isp 1900, so
one narrow window per 1.09 yr becomes a continuous one.

Search box, recorded per CLAUDE.md's rule after ADR 0007: the phase grid is
``PHASE_SAMPLES`` departures spread over one synodic period; each phase is
enumerated by :func:`src.jovian_cycle_phasing._cycle_branches` at that module's
committed settings, with ``_OUTBOUND_TOF_MIN`` temporarily freed to
``OUTBOUND_FLOOR_FOR_SWEEP`` so short, hot departures are on offer. Growth is
re-scored here rather than taken from the branch, because ``_net_growth`` charges
methalox and this module's whole subject is what happens when it does not.

Run with ``make fly-park``. Not imported by :mod:`src.main`; not part of
``make all``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from tabulate import tabulate

from src import jovian_cycle_phasing as chain
from src.astro_constants import (
    JUPITER_FLYBY_MAX_TOF,
    METHALOX_VACUUM_ISP,
    PUFFSAT_CYCLE_ORBIT_PERIOD,
    STD_FUDGE_FACTOR,
)
from src.jovian_cycle_phasing import _EARTH_JUPITER_SYNODIC_YEARS
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.retrograde_return_legs import _assist_chain_params, _AssistChainParams

#: Departures sampled across one Earth-Jupiter synodic period.
PHASE_SAMPLES = 73
#: Outbound time-of-flight floor used while sweeping, in years. The production
#: chain floor is 1.1; freeing it to 0.70 puts short, hot departures on offer so
#: the sweep can find them rather than assuming they do not exist.
OUTBOUND_FLOOR_FOR_SWEEP = 0.70
#: Standard gravity in km/s^2, for Isp -> exhaust speed.
G0_KM_S2 = 9.80665e-3
#: Exhaust speeds the tables are reported at, as specific impulses in seconds.
#: 380 is methalox; 2214 is ADR 0019's departure-nozzle ledger at eta_jet^2 =
#: 0.6; 1200 is a deliberately conservative impactor-driven figure.
REPORTED_ISP: Tuple[int, ...] = (380, 700, 1000, 1200, 1500, 1800, 1900, 2214)
#: Arrival speeds the exchange rate is tabulated at (km/s).
REPORTED_COLLISION_SPEEDS: Tuple[float, ...] = (56.0, 58.0, 60.0, 62.0, 64.0, 68.0)
#: Methalox specific impulse, the departure stage fly-and-park *fails* on.
METHALOX_ISP_SECONDS = float(METHALOX_VACUUM_ISP.to_value(u.s))
#: Baseline arrival speed the exchange rate is measured against (km/s).
EXCHANGE_BASELINE_SPEED = 53.5
#: Half-width of the band counted as "exactly 3S", in synodic periods.
THREE_SYNODIC_TOLERANCE = 0.03
#: Minimum park, in synodic periods, for a cycle to count as fly-and-park.
MINIMUM_PARK = 0.02


def exhaust_speed_from_isp(specific_impulse: float) -> float:
    """Effective exhaust speed for a specific impulse.

    Args:
        specific_impulse: Specific impulse in seconds.

    Returns:
        Effective exhaust speed in km/s.
    """
    return specific_impulse * G0_KM_S2


def _chain_params() -> _AssistChainParams:
    """Build the chain parameter block this module scores against.

    Returns:
        The :class:`_AssistChainParams` the phased chain uses, carrying the
        20-day cycle orbit's periapsis speed as the push target.
    """
    cycle_speed = puffsat_cycle_periapsis_speed()
    return _assist_chain_params(
        target_collision_speed=float(cycle_speed.to_value(u.km / u.s)),
        cycle_periapsis_speed=cycle_speed,
    )


def payload_mass_ratio_float(collision_speed: float, push_target: float) -> float:
    """Payload minted per unit arriving impactor mass, in floats.

    ``eq:PuffSat_ratio``: ``2 f / ln(v_b / (v_b - v_rf))``. Inlined in floats
    because this module re-scores thousands of enumerated cycles.

    Args:
        collision_speed: Arrival speed ``v_b`` of the returning PuffSat (km/s).
        push_target: Cycle-orbit periapsis speed ``v_rf`` (km/s).

    Returns:
        The payload mass ratio, or 0.0 if the collision cannot reach the push
        target (no payload is minted).
    """
    if collision_speed <= push_target:
        return 0.0
    return (
        2.0
        * STD_FUDGE_FACTOR
        / float(np.log(collision_speed / (collision_speed - push_target)))
    )


@dataclass(frozen=True)
class Cycle:
    """One enumerated closing Earth -> Jupiter -> Earth cycle.

    Attributes:
        departure_phase: Departure phase as a fraction of one synodic period.
        outbound_years: Earth-to-Jupiter time of flight (yr).
        return_years: Jupiter-to-1 AU return time of flight (yr).
        departure_burn: Oberth departure burn above the cycle-orbit periapsis
            speed (km/s).
        flyby_burn: Perijove burn (km/s); zero on the unpowered enumeration.
        collision_speed: Achieved arrival speed ``v_b`` (km/s).
    """

    departure_phase: float
    outbound_years: float
    return_years: float
    departure_burn: float
    flyby_burn: float
    collision_speed: float

    @property
    def flight_years(self) -> float:
        """Flight time, outbound plus return, excluding any coast or park."""
        return self.outbound_years + self.return_years

    @property
    def flight_synodics(self) -> float:
        """Flight time in Earth-Jupiter synodic periods."""
        return self.flight_years / _EARTH_JUPITER_SYNODIC_YEARS

    def growth(self, exhaust: float, push_target: float) -> float:
        """Payload multiplier for this cycle at a given departure exhaust speed.

        Args:
            exhaust: Departure-stage effective exhaust speed (km/s).
            push_target: Cycle-orbit periapsis speed ``v_rf`` (km/s).

        Returns:
            ``mass_ratio x delivered_fraction``. Above 1 the cycle grows.
        """
        ratio = payload_mass_ratio_float(self.collision_speed, push_target)
        if ratio <= 0.0:
            return 0.0
        return ratio * float(np.exp(-(self.departure_burn + self.flyby_burn) / exhaust))


def enumerate_phase_grid(
    phase_samples: int = PHASE_SAMPLES,
    outbound_floor: float = OUTBOUND_FLOOR_FOR_SWEEP,
) -> Dict[float, List[Cycle]]:
    """Enumerate every closing cycle from every sampled departure phase.

    Temporarily lowers :mod:`src.jovian_cycle_phasing`'s outbound time-of-flight
    floor so short, hot departures are on offer, and restores it afterwards.

    Args:
        phase_samples: Departures sampled across one synodic period.
        outbound_floor: Outbound time-of-flight floor during the sweep (yr).

    Returns:
        Mapping from departure phase (fraction of a synodic period) to the
        closing cycles available from it.
    """
    params = _chain_params()
    year_s = float((1.0 * u.year).to_value(u.s))
    coast_s = float(PUFFSAT_CYCLE_ORBIT_PERIOD.to_value(u.s))
    max_tof = float(JUPITER_FLYBY_MAX_TOF.to_value(u.s))
    synodic_s = _EARTH_JUPITER_SYNODIC_YEARS * year_s

    saved_floor = chain._OUTBOUND_TOF_MIN
    chain._OUTBOUND_TOF_MIN = outbound_floor
    try:
        grid: Dict[float, List[Cycle]] = {}
        for index in range(phase_samples):
            phase = index / phase_samples
            branches = chain._cycle_branches(
                phase * synodic_s, False, 0.0, 0.0, max_tof, coast_s, params
            )
            grid[phase] = [
                Cycle(
                    departure_phase=phase,
                    outbound_years=branch.outbound_tof / year_s,
                    return_years=branch.return_tof / year_s,
                    departure_burn=branch.departure_burn,
                    flyby_burn=branch.flyby_burn,
                    collision_speed=branch.collision_speed,
                )
                for branch in branches
            ]
    finally:
        chain._OUTBOUND_TOF_MIN = saved_floor
    return grid


def sweet_phase(grid: Dict[float, List[Cycle]]) -> Tuple[float, float]:
    """The departure phase at which reaching Jupiter is cheapest.

    The **sweet phase**, defined operationally: the sampled departure phase whose
    cheapest available outbound burn is smallest. This is the phase the growth
    chain is pinned to when the departure stage cannot afford anything else, and
    it is why the loop ends up on a three-synodic clock nobody imposed.

    Args:
        grid: Output of :func:`enumerate_phase_grid`.

    Returns:
        ``(phase, cheapest departure burn in km/s)``.
    """
    best_phase, best_burn = 0.0, float("inf")
    for phase, cycles in grid.items():
        if not cycles:
            continue
        burn = min(c.departure_burn for c in cycles)
        if burn < best_burn:
            best_phase, best_burn = phase, burn
    return best_phase, best_burn


def departure_burn_span(grid: Dict[float, List[Cycle]]) -> Tuple[float, float]:
    """Cheapest and dearest "cheapest available" outbound burn across phases.

    The swing this reports is the whole mechanism behind the **sweet phase**: it
    is how much the price of reaching Jupiter depends on where Jupiter is.

    Args:
        grid: Output of :func:`enumerate_phase_grid`.

    Returns:
        ``(min, max)`` of the per-phase minimum departure burn, in km/s.
    """
    mins = [min(c.departure_burn for c in cs) for cs in grid.values() if cs]
    return min(mins), max(mins)


def exchange_rate(
    collision_speed: float,
    exhaust: float,
    push_target: float,
    baseline_speed: float = EXCHANGE_BASELINE_SPEED,
) -> float:
    """Extra departure burn a hotter arrival is worth before it stops paying.

    Because :func:`fly_and_park_comparison` pads every candidate to the same
    3.00 synodic periods, cycle time cancels and the entire trade is this one
    number: the burn at which the mass-ratio gain from a hotter ``v_b`` is
    exactly cancelled by the propellant it cost to buy it.

    Args:
        collision_speed: The hotter arrival speed ``v_b`` (km/s).
        exhaust: Departure-stage effective exhaust speed (km/s).
        push_target: Cycle-orbit periapsis speed ``v_rf`` (km/s).
        baseline_speed: Arrival speed to measure the gain against (km/s).

    Returns:
        Extra departure burn in km/s. Zero when the hotter speed is no better.
    """
    gain = payload_mass_ratio_float(
        collision_speed, push_target
    ) / payload_mass_ratio_float(baseline_speed, push_target)
    if gain <= 1.0:
        return 0.0
    return exhaust * float(np.log(gain))


def usable_phase_fraction(
    grid: Dict[float, List[Cycle]], exhaust: float, push_target: float
) -> float:
    """Fraction of departure phases offering at least one growing cycle.

    This is the launch-cadence quantity. A phase is **usable** when some closing
    trajectory leaving it grows the payload; a phase can be perfectly reachable
    and still be unusable.

    Args:
        grid: Output of :func:`enumerate_phase_grid`.
        exhaust: Departure-stage effective exhaust speed (km/s).
        push_target: Cycle-orbit periapsis speed ``v_rf`` (km/s).

    Returns:
        Fraction in ``[0, 1]``.
    """
    if not grid:
        return 0.0
    live = sum(
        1
        for cycles in grid.values()
        if any(c.growth(exhaust, push_target) > 1.0 for c in cycles)
    )
    return live / len(grid)


@dataclass(frozen=True)
class ParkComparison:
    """One departure phase's pure-3S cycle against its fly-and-park alternative.

    Attributes:
        phase: Departure phase as a fraction of one synodic period.
        pure: The best cycle whose flight time is already ~3.00 synodic periods.
        parked: The best cycle short enough to leave a park, or None.
        pure_growth: Payload multiplier of ``pure``.
        parked_growth: Payload multiplier of ``parked``, or 0.0.
    """

    phase: float
    pure: Cycle
    parked: Optional[Cycle]
    pure_growth: float
    parked_growth: float

    @property
    def park_synodics(self) -> float:
        """Park needed to pad the fly-and-park cycle to exactly 3.00 S."""
        if self.parked is None:
            return 0.0
        return 3.0 - self.parked.flight_synodics

    @property
    def gain(self) -> float:
        """Fly-and-park growth relative to the pure-3S cycle."""
        if self.parked is None or self.pure_growth <= 0.0:
            return 1.0
        return self.parked_growth / self.pure_growth

    @property
    def extra_departure_burn(self) -> float:
        """Extra departure burn the fly-and-park cycle costs (km/s)."""
        if self.parked is None:
            return 0.0
        return self.parked.departure_burn - self.pure.departure_burn


def fly_and_park_comparison(
    grid: Dict[float, List[Cycle]], exhaust: float, push_target: float
) -> List[ParkComparison]:
    """Compare, per phase, flying the full 3S window against flying short.

    Both alternatives occupy exactly 3.00 synodic periods -- one by flying it,
    the other by flying less and parking the difference -- so cycle time cancels
    and the comparison is purely per-cycle growth.

    Args:
        grid: Output of :func:`enumerate_phase_grid`.
        exhaust: Departure-stage effective exhaust speed (km/s).
        push_target: Cycle-orbit periapsis speed ``v_rf`` (km/s).

    Returns:
        One :class:`ParkComparison` per phase that has a growing pure-3S cycle,
        in phase order.
    """
    coast = (
        float((PUFFSAT_CYCLE_ORBIT_PERIOD).to_value(u.year))
        / _EARTH_JUPITER_SYNODIC_YEARS
    )
    out: List[ParkComparison] = []
    for phase in sorted(grid):
        cycles = grid[phase]
        if not cycles:
            continue
        three = [
            c
            for c in cycles
            if abs(c.flight_synodics + coast - 3.0) <= THREE_SYNODIC_TOLERANCE
        ]
        if not three:
            continue
        best_pure = max(three, key=lambda c: c.growth(exhaust, push_target))
        pure_growth = best_pure.growth(exhaust, push_target)
        if pure_growth <= 1.0:
            continue
        short = [c for c in cycles if c.flight_synodics <= 3.0 - MINIMUM_PARK]
        best_parked = (
            max(short, key=lambda c: c.growth(exhaust, push_target)) if short else None
        )
        out.append(
            ParkComparison(
                phase=phase,
                pure=best_pure,
                parked=best_parked,
                pure_growth=pure_growth,
                parked_growth=(
                    best_parked.growth(exhaust, push_target) if best_parked else 0.0
                ),
            )
        )
    return out


def phase_reaches(
    grid: Dict[float, List[Cycle]],
    phase: float,
    target_speed: float,
    minimum_park: float = MINIMUM_PARK,
) -> bool:
    """Does this phase offer *any* parkable cycle at or above a target speed?

    Availability, as distinct from profitability: a phase can offer a 68 km/s
    arrival that is nonetheless not worth flying. The paper document quotes this
    statistic, so it is computed here rather than inferred from the winner.

    Args:
        grid: Output of :func:`enumerate_phase_grid`.
        phase: The departure phase to test.
        target_speed: Arrival speed to reach or exceed (km/s).
        minimum_park: Park required, in synodic periods, to qualify.

    Returns:
        True if some cycle from that phase is short enough to park and arrives
        at or above ``target_speed``.
    """
    return any(
        c.collision_speed >= target_speed and c.flight_synodics <= 3.0 - minimum_park
        for c in grid.get(phase, [])
    )


def hottest_reachable(
    grid: Dict[float, List[Cycle]], minimum_park: float = MINIMUM_PARK
) -> float:
    """Highest arrival speed available on any cycle short enough to park.

    Args:
        grid: Output of :func:`enumerate_phase_grid`.
        minimum_park: Park required, in synodic periods, to qualify.

    Returns:
        The maximum ``v_b`` in km/s, or 0.0 if none qualifies.
    """
    speeds = [
        c.collision_speed
        for cycles in grid.values()
        for c in cycles
        if c.flight_synodics <= 3.0 - minimum_park
    ]
    return max(speeds) if speeds else 0.0


def _report(grid: Dict[float, List[Cycle]], params: _AssistChainParams) -> None:
    """Print every table this module exists to produce.

    Args:
        grid: Output of :func:`enumerate_phase_grid`.
        params: The chain parameter block, for the push target.
    """
    push = params.flyby.v_rf
    total = sum(len(c) for c in grid.values())
    phase, burn = sweet_phase(grid)
    low, high = departure_burn_span(grid)

    print(f"\n{total} closing cycles enumerated across {len(grid)} departure phases")
    print(f"push target v_rf = {push:.4f} km/s (20-day cycle orbit at 200 km)\n")
    print(
        f"SWEET PHASE: {phase:.3f} of a synodic period, cheapest outbound burn "
        f"{burn:.3f} km/s"
    )
    print(
        f"  the same 'cheapest available' burn runs {low:.3f} to {high:.3f} km/s "
        f"around the circle -- a {high / low:.1f}x swing, purely from where "
        f"Jupiter is\n"
    )

    print("EXCHANGE RATE -- extra departure burn a hotter arrival is worth")
    print(f"(km/s, against a {EXCHANGE_BASELINE_SPEED:.1f} km/s baseline)")
    rows = [
        [f"{speed:.0f}"]
        + [
            f"{exchange_rate(speed, exhaust_speed_from_isp(isp), push):.2f}"
            for isp in (380, 1200, 2214)
        ]
        for speed in REPORTED_COLLISION_SPEEDS
    ]
    print(
        tabulate(
            rows, headers=["v_b", "Isp 380", "Isp 1200", "Isp 2214"], tablefmt="grid"
        )
    )

    print("\nUSABLE PHASE FRACTION -- the launch-cadence result")
    rows = []
    for isp in REPORTED_ISP:
        exhaust = exhaust_speed_from_isp(isp)
        rows.append(
            [
                f"{isp}",
                f"{exhaust:.2f}",
                f"{usable_phase_fraction(grid, exhaust, push):.0%}",
            ]
        )
    print(
        tabulate(
            rows, headers=["Isp (s)", "v_e (km/s)", "usable phases"], tablefmt="grid"
        )
    )

    print("\nFLY-AND-PARK vs PURE 3S")
    rows = []
    for isp in (380, 1200, 2214):
        exhaust = exhaust_speed_from_isp(isp)
        comps = fly_and_park_comparison(grid, exhaust, push)
        wins = [c for c in comps if c.gain > 1.0]
        hot = [c for c in comps if phase_reaches(grid, c.phase, 60.0)]
        extra = [c.extra_departure_burn for c in comps if c.parked]
        rows.append(
            [
                f"{isp}",
                f"{len(comps)}",
                f"{len(hot)}",
                f"{len(wins)}",
                f"{max((c.gain for c in comps), default=1.0):.3f}",
                f"{float(np.median(extra)) if extra else 0.0:+.3f}",
            ]
        )
    print(
        tabulate(
            rows,
            headers=[
                "Isp (s)",
                "phases with 3S",
                "...reaching v_b>=60",
                "...park wins",
                "best gain",
                "median extra dv",
            ],
            tablefmt="grid",
        )
    )
    print(
        f"\nhottest arrival on any parkable cycle: {hottest_reachable(grid):.2f} km/s"
    )
    print(
        "  (the perfect-retrograde boundary; CONTEXT.md, 'Perfect-retrograde boundary')"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the fly-and-park analysis and print its tables.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--phase-samples",
        type=int,
        default=PHASE_SAMPLES,
        help=f"departures sampled across one synodic period (default {PHASE_SAMPLES})",
    )
    args = parser.parse_args(argv)
    params = _chain_params()
    grid = enumerate_phase_grid(phase_samples=args.phase_samples)
    _report(grid, params)


if __name__ == "__main__":
    main()
