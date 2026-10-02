"""Cheaper seeds on gravity assists and solar-electric propulsion (ADR 0039).

The seed (ADR 0035/0036) is the first batch of PuffSats, flown once to Jupiter
on an expended, refuelled Starship. Under the bank flight prices it costs
$9293 per kilogram sent, and the tankers are most of that. A route that lowers
the burn from low orbit sends more PuffSats per ship, so the seed costs less
per kilogram. But it arrives later, and the whole program slides with it.

**The test.** A route that makes the seed ``k`` times cheaper and delays its
return by ``dt`` years repays itself when ``k (1 + r)^-dt > 1``, with ``r`` the
rate charged before the cycle is proven (30%). Every later cost and revenue
slides by the same ``dt``, so the test does not depend on what the fleet is
worth (:func:`seed_route_worth`).

**Solar-electric propulsion is charged twice** (author, 2026-10-01). Its array,
power processing, thrusters and argon tankage ride with the seed and displace
PuffSats, and the hardware is bought (:class:`SepStage`). The mass figures are
ADR 0026's; the dollar price per watt is an unsourced hypothesis.
"""

import enum
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from scipy.optimize import brentq

from src import conic_kernel
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.retrograde_return_legs import (
    _assist_chain_params,
    _AssistBody,
    _AssistChainParams,
    _earth_phase_mismatch,
    _jupiter_assist_body,
    _LadderPricing,
    _phased_jovian_flyby,
    _phased_ladder_burn,
    _ReturnLeg,
)
from src.seed_cost import (
    SEED_LAUNCHES,
    SEED_PROPELLANT,
    STRIPPED_SHIPS,
    SeedShip,
    burn_from_circular,
    seed_payload,
)
from src.sep_split_correction import (
    ARGON_TANK_FRACTION,
    SEP_THRUSTER_EFFICIENCY,
    VE_ARGON,
)
from src.two_wave_growth import VE_METHALOX

#: ADR 0026's middle specific mass: array, power processing, thrusters and
#: gimbals per kW at 1 AU (swept 10/15/20).
DEFAULT_SPECIFIC_MASS = 15.0
#: Dollars per watt at 1 AU, an unsourced hypothesis (author, 2026-10-01): a
#: mass-produced Starlink-class array to a science-mission system.
PRICES_PER_WATT = (50.0, 200.0, 1000.0)


def seed_route_worth(cheaper: float, delay_years: float, rate: float) -> float:
    """``k (1 + r)^-dt``: above one, the slower, cheaper seed repays itself.

    Args:
        cheaper: How many times cheaper per kilogram the route makes the seed.
        delay_years: How much later its first return comes.
        rate: Annual cost of capital before the cycle is proven.

    Returns:
        The route's worth relative to the direct seed.
    """
    return float(cheaper * (1.0 + rate) ** (-delay_years))


def discounted_seed_per_dollar(
    seed_mass: float, return_years: float, dollars: float, rate: float
) -> float:
    """Seed mass per dollar, discounted from its return to the seed's purchase.

    The route search maximises this: it is the ``k (1 + r)^-dt`` test with the
    dollars, SEP hardware included, inside ``k``.

    Args:
        seed_mass: Seed sent per ship (kg).
        return_years: From purchase to the seed's return (yr).
        dollars: What the ship and its stage cost.
        rate: Annual cost of capital before the cycle is proven.

    Returns:
        Kilograms per dollar, discounted.
    """
    return float(seed_mass * (1.0 + rate) ** (-return_years) / dollars)


def ship_cost(flight_price: float, hull: float) -> float:
    """One seed ship: its twelve tankers, its own launch, and its hull ($)."""
    return float(SEED_LAUNCHES * flight_price + hull)


@dataclass(frozen=True)
class SepSplit:
    """How a departing stack divides once its SEP stage is charged (kg).

    Attributes:
        hardware: Array, power processing, thrusters and gimbals.
        argon: Argon the burns spend.
        tanks: Argon tankage.
        puffsats: What is left to fly as seed.
    """

    hardware: float
    argon: float
    tanks: float
    puffsats: float


@dataclass(frozen=True)
class SepStage:
    """An argon solar-electric stage riding with the seed (ADR 0026's figures).

    The array is sized per kilogram of departing stack, so a bigger seed gets a
    proportionally bigger array: ADR 0026 found the array scale-invariant.
    Mass falling as argon is spent is ignored, which understates the
    acceleration slightly, the conservative direction.

    Attributes:
        specific_power: Array power at 1 AU per kilogram of departing stack (W/kg).
        specific_mass: Stage hardware per kW at 1 AU (kg/kW).
        price_per_watt: Hardware dollars per watt at 1 AU.
    """

    specific_power: float
    specific_mass: float = DEFAULT_SPECIFIC_MASS
    price_per_watt: float = PRICES_PER_WATT[1]

    def acceleration_1au(self) -> float:
        """Thrust over mass at 1 AU, ``2 eta P / (m v_e)`` (m/s^2)."""
        return 2.0 * SEP_THRUSTER_EFFICIENCY * self.specific_power / (VE_ARGON * 1.0e3)

    def capacity(self, distances_au: Sequence[float], duration: float) -> float:
        """Delta-v the stage can deliver thrusting through an arc (km/s).

        Args:
            distances_au: Sun distances sampled evenly in time along the arc (AU).
            duration: The arc's duration (s).

        Returns:
            ``a_1AU x duration x mean(1 / r^2)``.
        """
        inverse_square = float(np.mean(np.asarray(distances_au, dtype=float) ** -2))
        return self.acceleration_1au() * duration * inverse_square / 1.0e3

    def split(self, stack: float, burn: float) -> SepSplit:
        """Charge the stage and its argon against a departing stack.

        Args:
            stack: Mass leaving Earth (kg).
            burn: Delta-v the stage delivers in total (km/s).

        Returns:
            The split.
        """
        hardware = self.specific_power * stack * self.specific_mass / 1.0e3
        argon = stack * (1.0 - float(np.exp(-burn / VE_ARGON)))
        tanks = ARGON_TANK_FRACTION * argon
        return SepSplit(hardware, argon, tanks, stack - hardware - argon - tanks)

    def price(self, stack: float) -> float:
        """Hardware dollars for the array sized to ``stack`` (kg)."""
        return self.specific_power * stack * self.price_per_watt


def shortfall(node_burns: Sequence[float], capacities: Sequence[float]) -> float:
    """How far a route's node burns outrun the thrusting that must precede them.

    The stage can thrust on any leg before a node, but not after it, so the
    test is cumulative: for every node, the burns up to and including it must
    fit within the capacity of the legs up to and including the one that
    arrives there.

    Args:
        node_burns: Delta-v needed at each node, in order (km/s).
        capacities: Delta-v the stage can deliver on the leg arriving at each
            node (km/s), parallel to ``node_burns``.

    Returns:
        The largest cumulative excess, zero when the route is deliverable.
    """
    needed = np.cumsum(np.asarray(node_burns, dtype=float))
    available = np.cumsum(np.asarray(capacities, dtype=float))
    return float(max(0.0, float(np.max(needed - available, initial=0.0))))


_AU_KM = 1.495978707e8
_YEAR_S = 365.25 * 86400.0
#: Samples per leg, evenly in time, for the sunlight average.
_ARC_SAMPLES = 16


@dataclass(frozen=True)
class RouteFlight:
    """One phased seed route, flown from Earth to Jupiter and back to Earth.

    Attributes:
        sequence: Bodies in order, e.g. ``"EVEJ"``.
        departure_burn: Burn at 200 km above the parameter block's start speed
            (km/s).
        v_infinity: Departure excess speed (km/s), what the seed ship must reach.
        node_burns: Burn at each intermediate flyby, in order (km/s).
        leg_years: Each leg's duration (yr).
        leg_distances_au: Sun distances sampled along each leg (AU).
        collision_speed: ``v_b`` of the returning wave (km/s).
        mismatch: Signed Earth-phase mismatch at the 1 AU crossing (rad).
        trip_years: Departure to the return's 1 AU crossing (yr).
    """

    sequence: str
    departure_burn: float
    v_infinity: float
    node_burns: Tuple[float, ...]
    leg_years: Tuple[float, ...]
    leg_distances_au: Tuple[Tuple[float, ...], ...]
    collision_speed: float
    mismatch: float
    trip_years: float


def route_params() -> _AssistChainParams:
    """The phased model's parameter block, as ADR 0008's optimum was flown."""
    v_rf = puffsat_cycle_periapsis_speed()
    return _assist_chain_params(
        target_collision_speed=float(v_rf.to_value("km/s")) + 0.5,
        cycle_periapsis_speed=v_rf,
    )


_BODY_NAMES = {"V": "venus", "E": "earth", "M": "mars", "J": "jupiter"}
#: Mean obliquity of the ecliptic at J2000 (deg).
_OBLIQUITY_DEG = 23.4392911


def planet_longitudes(sequence: str, jd: float) -> Tuple[float, ...]:
    """Each body's heliocentric ecliptic longitude on a date (rad).

    Reads astropy's built-in ephemeris and rotates equatorial into ecliptic
    coordinates, so the circular model's phases are the real ones.

    Args:
        sequence: Body symbols, e.g. ``"EVEJ"``.
        jd: TDB Julian date.

    Returns:
        One longitude per symbol, in order.
    """
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    epoch = Time(jd, format="jd", scale="tdb")
    sun = get_body_barycentric("sun", epoch).xyz.to_value("km")
    tilt = np.radians(_OBLIQUITY_DEG)
    out = []
    for symbol in sequence:
        x, y, z = (
            get_body_barycentric(_BODY_NAMES[symbol], epoch).xyz.to_value("km") - sun
        )
        ecliptic_y = float(np.cos(tilt) * y + np.sin(tilt) * z)
        out.append(float(np.arctan2(ecliptic_y, x)))
    return tuple(out)


def _bodies(sequence: str, params: _AssistChainParams) -> Tuple[_AssistBody, ...]:
    by_symbol: Dict[str, _AssistBody] = {b.symbol: b for b in params.bodies}
    by_symbol["J"] = _jupiter_assist_body(params)
    return tuple(by_symbol[symbol] for symbol in sequence)


def _arc_distances(
    position: np.ndarray, velocity: np.ndarray, tof: float, mu: float
) -> Tuple[float, ...]:
    """Sun distances (AU) at ``_ARC_SAMPLES`` instants evenly spaced in time."""
    times = np.linspace(0.0, tof, _ARC_SAMPLES)
    return tuple(
        float(
            np.linalg.norm(conic_kernel.kepler_propagate(position, velocity, t, mu)[0])
        )
        / _AU_KM
        for t in times
    )


def _fly_ladder(
    sequence: str,
    longitudes: Sequence[float],
    leg_years: Sequence[float],
    params: _AssistChainParams,
    powered_nodes: bool,
) -> Optional[_LadderPricing]:
    """The phased ladder from Earth to Jupiter, its nodes priced either way."""
    priced = _phased_ladder_burn(
        0.0,
        [years * _YEAR_S for years in leg_years],
        _bodies(sequence, params),
        list(longitudes),
        params,
        powered_nodes=powered_nodes,
    )
    if priced is None or priced.arrival_excess < 1e-6:
        return None
    return priced


def _bend(
    priced: _LadderPricing,
    log_perijove: float,
    bend_sign: float,
    earth_longitude: float,
    params: _AssistChainParams,
) -> Optional[Tuple[_ReturnLeg, float]]:
    """Jupiter's unpowered bend and the return: (leg, Earth-phase mismatch)."""
    perijove = params.flyby.periapsis_floor * float(10.0**log_perijove)
    leg = _phased_jovian_flyby(
        priced.arrival_excess_vector,
        priced.arrival_longitude,
        perijove,
        0.0,
        bend_sign,
        params,
    )
    if leg is None:
        return None
    mismatch = _earth_phase_mismatch(
        leg,
        priced.arrival_longitude,
        priced.arrival_time,
        earth_longitude,
        params.flyby,
    )
    return leg, mismatch


def _flight(
    sequence: str,
    priced: _LadderPricing,
    leg: _ReturnLeg,
    mismatch: float,
    leg_years: Sequence[float],
    params: _AssistChainParams,
) -> RouteFlight:
    fl = params.flyby
    start = fl.v_depart_from + priced.departure_burn
    return RouteFlight(
        sequence=sequence,
        departure_burn=priced.departure_burn,
        v_infinity=float(np.sqrt(max(0.0, start * start - fl.v_esc_leo**2))),
        node_burns=tuple(priced.node_burns),
        leg_years=tuple(float(y) for y in leg_years),
        leg_distances_au=tuple(
            _arc_distances(r, v, tof, fl.mu_sun) for r, v, tof in priced.legs
        ),
        collision_speed=leg.collision_speed,
        mismatch=mismatch,
        trip_years=(priced.arrival_time + leg.tof) / _YEAR_S,
    )


def fly_route(
    sequence: str,
    longitudes: Sequence[float],
    leg_years: Sequence[float],
    log_perijove: float,
    bend_sign: float,
    params: Optional[_AssistChainParams] = None,
    powered_nodes: bool = True,
) -> Optional[RouteFlight]:
    """Fly a phased route: the ladder to Jupiter, its bend, and the return.

    Every body sits at its given heliocentric longitude at departure and moves
    on its circular orbit; each leg is a Lambert arc between true positions.
    An intermediate flyby is charged as a methalox burn at periapsis, with
    Oberth leverage (``powered_nodes``), or as an unpowered flyby plus the
    excess-velocity change bought in deep space, which is how a thruster pays
    it. Jupiter's bend is unpowered (ADR 0006). Earth's longitude comes first.

    Args:
        sequence: Bodies in order, starting with ``"E"`` and ending with ``"J"``.
        longitudes: Each body's heliocentric longitude at departure (rad).
        leg_years: Each leg's duration (yr), one fewer than the bodies.
        log_perijove: Perijove radius as log10 multiples of the floor.
        bend_sign: Which side Jupiter is passed on (+1 or -1).
        params: The parameter block; defaults to :func:`route_params`.
        powered_nodes: Charge nodes at periapsis (methalox) rather than in
            deep space (a thruster).

    Returns:
        The flight, or None if a Lambert arc or the bend fails.
    """
    params = params or route_params()
    priced = _fly_ladder(sequence, longitudes, leg_years, params, powered_nodes)
    if priced is None:
        return None
    bent = _bend(priced, log_perijove, bend_sign, longitudes[0], params)
    if bent is None:
        return None
    return _flight(sequence, priced, bent[0], bent[1], leg_years, params)


#: The returning wave must be at least as fast as ADR 0007's target, the
#: powered-flyby optimum's, so every route delivers an equally useful seed.
RETURN_FLOOR = 51.134
#: The flown chain's first departure, 2026-11-09 TDB, opens the seed's window.
SEED_WINDOW_OPENS = 2461353.5
#: How long after it a seed may launch (yr).
SEED_WINDOW_YEARS = 6.0
#: Methalox tankage and engines for node burns flown chemically (ADR 0026).
METHALOX_STAGE_FRACTION = 0.08
#: The stripped seed ship's steered loss (ADR 0036), the search's proxy.
_PROXY_LOSS = 0.117
#: Perijove search interval, log10 multiples of the floor.
_LOG_PERIJOVE = (0.0, 2.5)
_PERIJOVE_SCAN = 24
#: Leg-time bounds (yr) by hop: inner-planet hops, Earth-Earth resonant loops,
#: and the leg to Jupiter.
_LEG_BOUNDS = {"inner": (0.08, 1.6), "EE": (0.9, 3.2), "J": (0.8, 5.0)}
#: Returned when a candidate cannot close at all.
_INFEASIBLE = 1.0e6


class Propulsion(enum.Enum):
    """What flies the node burns after the seed ship's departure burn."""

    METHALOX = "methalox"
    SEP = "sep"


def _hop(here: str, there: str) -> str:
    if there == "J":
        return "J"
    return "EE" if here == there == "E" else "inner"


def leg_bounds(sequence: str) -> List[Tuple[float, float]]:
    """Duration bounds for each leg of ``sequence`` (yr)."""
    return [_LEG_BOUNDS[_hop(a, b)] for a, b in zip(sequence, sequence[1:])]


def stack_per_ship(v_infinity: float, ship: SeedShip) -> float:
    """Mass one ship sends to ``v_infinity`` (kg), the search's quick proxy.

    The rocket equation from the 200 km parking orbit with the stripped ship's
    117 m/s steering loss. :func:`seed_payload` integrates the real loss; the
    search re-prices its winners with it.
    """
    burn = float(burn_from_circular(v_infinity * u.km / u.s).to_value(u.km / u.s))
    ratio = float(np.exp((burn + _PROXY_LOSS) / VE_METHALOX))
    propellant = float(SEED_PROPELLANT.to_value(u.kg))
    return propellant / (ratio - 1.0) - float(ship.dry_mass.to_value(u.kg))


def puffsats_after_nodes(
    stack: float,
    flight: RouteFlight,
    propulsion: Propulsion,
    stage: Optional[SepStage],
) -> Tuple[float, float]:
    """(PuffSats left, SEP shortfall km/s) once the node burns are paid for.

    Args:
        stack: Mass leaving Earth (kg).
        flight: The route.
        propulsion: What flies the nodes.
        stage: The SEP stage, for ``Propulsion.SEP``.

    Returns:
        The seed mass and how far the nodes outrun the stage (zero if
        deliverable, and always zero for methalox).
    """
    burn = float(sum(flight.node_burns))
    if propulsion is Propulsion.METHALOX:
        spent = 1.0 - float(np.exp(-burn / VE_METHALOX))
        return stack * (1.0 - (1.0 + METHALOX_STAGE_FRACTION) * spent), 0.0
    if stage is None:
        raise ValueError("SEP propulsion needs a stage")
    capacities = [
        stage.capacity(distances, years * _YEAR_S)
        for distances, years in zip(flight.leg_distances_au, flight.leg_years)
    ]
    # The stage pays node i on the legs up to the one arriving there.
    gap = shortfall(flight.node_burns, capacities[: len(flight.node_burns)])
    return stage.split(stack, burn).puffsats, gap


@dataclass(frozen=True)
class SeedRoute:
    """The best seed found on one route.

    Attributes:
        flight: Its trajectory.
        launch_jd: TDB Julian date it departs.
        return_jd: TDB Julian date it crosses 1 AU on return.
        propulsion: What flies the nodes.
        stage: The SEP stage, or None.
        ship: The seed ship.
        stack: Mass leaving Earth per ship, real finite-burn loss (kg).
        puffsats_per_ship: Seed mass per ship once the nodes are paid (kg).
        dollars: The ship and its stage ($).
        value: :func:`discounted_seed_per_dollar` at the search's rate.
    """

    flight: RouteFlight
    launch_jd: float
    return_jd: float
    propulsion: Propulsion
    stage: Optional[SepStage]
    ship: SeedShip
    stack: float
    puffsats_per_ship: float
    dollars: float
    value: float


class _RouteProblem:
    """pygmo problem: launch date and leg times; perijove is solved inside."""

    def __init__(
        self,
        sequence: str,
        bend_sign: float,
        propulsion: Propulsion,
        stage: Optional[SepStage],
        ship: SeedShip,
        ship_dollars: float,
        rate: float,
    ) -> None:
        self.sequence = sequence
        self.ship_dollars = ship_dollars
        self.rate = rate
        self.bend_sign = bend_sign
        self.propulsion = propulsion
        self.stage = stage
        self.ship = ship
        self.params = route_params()
        self.longitudes0 = planet_longitudes(sequence, SEED_WINDOW_OPENS)
        self.rates = [
            body.v_circ / body.orbit_radius for body in _bodies(sequence, self.params)
        ]

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        legs = leg_bounds(self.sequence)
        return (
            [0.0] + [lo for lo, _ in legs],
            [SEED_WINDOW_YEARS] + [hi for _, hi in legs],
        )

    def longitudes(self, launch_years: float) -> List[float]:
        seconds = launch_years * _YEAR_S
        return [lon + n * seconds for lon, n in zip(self.longitudes0, self.rates)]

    def solve(self, x: Sequence[float]) -> Optional[Tuple[RouteFlight, float]]:
        """The flight closing on Earth for this launch and these legs, if any."""
        launch, legs = float(x[0]), [float(v) for v in x[1:]]
        longitudes = self.longitudes(launch)
        priced = _fly_ladder(
            self.sequence,
            longitudes,
            legs,
            self.params,
            powered_nodes=self.propulsion is Propulsion.METHALOX,
        )
        if priced is None:
            return None

        def mismatch(log_perijove: float) -> float:
            bent = _bend(
                priced, log_perijove, self.bend_sign, longitudes[0], self.params
            )
            return float("nan") if bent is None else bent[1]

        grid = np.linspace(*_LOG_PERIJOVE, _PERIJOVE_SCAN)
        values = [mismatch(g) for g in grid]
        best: Optional[RouteFlight] = None
        for lo, hi, f_lo, f_hi in zip(grid, grid[1:], values, values[1:]):
            if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0.0:
                continue
            if abs(f_hi - f_lo) > np.pi:  # the wrap at +-pi, not a crossing
                continue
            try:
                root = float(brentq(mismatch, lo, hi, xtol=1e-12))
            except ValueError:  # the bend fails somewhere inside this bracket
                continue
            bent = _bend(priced, root, self.bend_sign, longitudes[0], self.params)
            if bent is None:
                continue
            flight = _flight(self.sequence, priced, bent[0], bent[1], legs, self.params)
            if best is None or flight.collision_speed > best.collision_speed:
                best = flight
        return None if best is None else (best, launch)

    def fitness(self, x: Sequence[float]) -> List[float]:
        solved = self.solve(x)
        if solved is None:
            return [_INFEASIBLE]
        flight, launch = solved
        stack = stack_per_ship(flight.v_infinity, self.ship)
        if stack <= 0.0:
            return [_INFEASIBLE / 2.0]
        seed, gap = puffsats_after_nodes(stack, flight, self.propulsion, self.stage)
        if seed <= 0.0:
            return [_INFEASIBLE / 4.0]
        dollars = self.ship_dollars + (self.stage.price(stack) if self.stage else 0.0)
        value = discounted_seed_per_dollar(
            seed, launch + flight.trip_years, dollars, self.rate
        )
        slow = max(0.0, RETURN_FLOOR - flight.collision_speed)
        # Log value, so a km/s of shortfall weighs like a factor of e^10.
        return [-float(np.log(value)) + 10.0 * gap + 10.0 * slow]


def _evolve_island(
    task: Tuple[
        str,
        float,
        Propulsion,
        Optional[SepStage],
        SeedShip,
        float,
        float,
        int,
        int,
        int,
        int,
    ],
) -> Tuple[List[float], float]:
    """One independent island: build the problem here, evolve, return its champion."""
    import pygmo as pg

    (sequence, sign, propulsion, stage, ship, dollars, rate, population, generations,
     evolutions, seed) = task  # fmt: skip
    udp = _RouteProblem(sequence, sign, propulsion, stage, ship, dollars, rate)
    pop = pg.population(pg.problem(udp), population, seed=seed)
    algo = pg.algorithm(pg.sade(gen=generations, seed=seed))
    for _ in range(evolutions):
        pop = algo.evolve(pop)
    return [float(v) for v in pop.champion_x], float(pop.champion_f[0])


# Peak resident memory of one island: numpy, astropy, scipy and pygmo loaded,
# measured at 341 MB on CPython 3.13 / aarch64, rounded up for headroom.
_ISLAND_BYTES = 450 * 2**20
# Left free for whatever else the machine runs while the islands evolve.
_RESERVE_BYTES = 2**30


def _available_bytes() -> Optional[int]:
    """MemAvailable from /proc/meminfo, or None where there is none (macOS)."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _island_workers(islands: int) -> int:
    """Processes to run islands in: no more than cores or free memory allow.

    Each island is a fresh spawned interpreter of :data:`_ISLAND_BYTES`; eight
    of them on a 3 GB machine with no swap ran it out of memory and the kernel
    killed the session. Islands are seeded individually, so the worker count
    changes only the wall time, never the champions.
    """
    workers = min(islands, os.cpu_count() or 1)
    available = _available_bytes()
    if available is not None:
        workers = min(workers, (available - _RESERVE_BYTES) // _ISLAND_BYTES)
    return max(1, workers)


def best_route(
    sequence: str,
    propulsion: Propulsion,
    stage: Optional[SepStage] = None,
    ship: SeedShip = STRIPPED_SHIPS[1],
    ship_dollars: float = 670.0e6,
    rate: float = 0.30,
    seed: int = 0,
    islands: int = 8,
    population: int = 40,
    generations: int = 150,
    evolutions: int = 3,
) -> Optional[SeedRoute]:
    """Most discounted seed per dollar on ``sequence``, launched in the window.

    Independent islands of pygmo's self-adaptive differential evolution, one
    per process and per bend side, search the launch date and leg times; the
    perijove is solved inside each evaluation so the return lands on Earth
    exactly. Islands do not migrate, as in pygmo's default archipelago; they
    run in a plain process pool because pygmo's multiprocessing island hung for
    minutes on shutdown here. The pool is capped by free memory
    (:func:`_island_workers`). Each champion is re-priced with the real
    finite-burn loss.

    Args:
        sequence: Bodies, e.g. ``"EVEJ"``.
        propulsion: What flies the node burns.
        stage: The SEP stage, for ``Propulsion.SEP``.
        ship: The seed ship.
        ship_dollars: What the ship costs (:func:`ship_cost`); the default
            is the dear end, Morgan Stanley's flights and a $20M hull.
        rate: Cost of capital until the seed returns.
        seed: Random seed; island ``i`` on side ``s`` uses ``seed + 2 i + s``.
        islands: Islands per bend side.
        population: Individuals per island.
        generations: Generations per evolution.
        evolutions: Evolutions per island.

    Returns:
        The best feasible route, or None if no candidate closes at the floor.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    tasks = [
        (sequence, sign, propulsion, stage, ship, ship_dollars, rate, population,
         generations, evolutions, seed + 2 * island + side)
        for side, sign in enumerate((1.0, -1.0))
        for island in range(islands)
    ]  # fmt: skip
    context = multiprocessing.get_context("spawn")
    workers = _island_workers(islands)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        champions = list(pool.map(_evolve_island, tasks))
    best: Optional[SeedRoute] = None
    for task, (x, _) in zip(tasks, champions):
        udp = _RouteProblem(
            sequence, task[1], propulsion, stage, ship, ship_dollars, rate
        )
        route = _price(udp, x)
        if route is not None and (best is None or route.value > best.value):
            best = route
    return best


def _price(udp: _RouteProblem, x: Sequence[float]) -> Optional[SeedRoute]:
    """Re-price a champion with the real finite-burn loss; None if infeasible."""
    solved = udp.solve(x)
    if solved is None:
        return None
    flight, launch = solved
    if flight.collision_speed < RETURN_FLOOR - 1e-6:
        return None
    stack = float(
        seed_payload(udp.ship, flight.v_infinity * u.km / u.s).payload.to_value(u.kg)
    )
    seed_mass, gap = puffsats_after_nodes(stack, flight, udp.propulsion, udp.stage)
    if gap > 1e-6 or seed_mass <= 0.0:
        return None
    launch_jd = SEED_WINDOW_OPENS + launch * 365.25
    dollars = udp.ship_dollars + (udp.stage.price(stack) if udp.stage else 0.0)
    return SeedRoute(
        flight=flight,
        launch_jd=launch_jd,
        return_jd=launch_jd + flight.trip_years * 365.25,
        propulsion=udp.propulsion,
        stage=udp.stage,
        ship=udp.ship,
        stack=stack,
        puffsats_per_ship=seed_mass,
        dollars=dollars,
        value=discounted_seed_per_dollar(
            seed_mass, launch + flight.trip_years, dollars, udp.rate
        ),
    )
