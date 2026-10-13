"""A Lucy-style Earth loop for the seed: leave slowly, come back fast (ADR 0059).

ADR 0039 flew the seed on one Earth loop (EEJ): a Lambert arc from Earth back
to Earth, then a powered Earth flyby onto the arc to Jupiter.  This module adds
the piece that makes NASA's Lucy and Juno trajectories work, a **deep-space
maneuver** in the loop (the delta-v Earth gravity assist, "ΔV-EGA"):

1. The refuelled seed ship burns from the 200 km parking orbit to a modest
   excess speed ``v0`` in a free direction ``alpha`` in the ecliptic.
2. It coasts a fraction ``eta`` of the loop and burns once in deep space,
   usually retrograde near aphelion.  That pumps the eccentricity, so it
   meets Earth again at a much higher excess speed than it left with.
3. At Earth a **powered flyby** (:func:`_powered_node_burn`, split hyperbolae,
   periapsis no lower than 300 km) bends and boosts that excess onto the
   Lambert arc to where Jupiter really is.  The node is solved, not assumed:
   it must leave the ship pointed on the Jupiter arc.
4. Jupiter's unpowered bend is solved so that the return lands on Earth at
   no less than the chain's first growth wave, 59.29 km/s
   (:data:`CHAIN_FLOOR`, :func:`_close_on_earth`).  ADR 0039's 51.134 km/s
   was slower than the cycle the cost model credits the seed with.

The loop is searched in three bands, about one, two and three years (Lucy
flew one- and two-year loops).  The deep-space and flyby burns are methalox at
**370 s**, held cold under blankets and cryocoolers (author, 2026-10-13), and
the Starship flies them itself, its dry mass carried (:func:`seed_carried`).
ADR 0039 instead dropped the ship after departure and flew later burns on a
light stage (8% tankage); that is staging, which the direct route could use
too, and is kept only as ``--staged``.  Everything is coplanar and circular, as in ADR
0039, and the loop's longest coast is under 3.6 years.

**The test** is ADR 0039's ``k x 1.1^-dt``: seed per ship over the direct
route's, discounted by the later return.  The ship's price cancels, so only
its dry mass matters.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.core.iod import izzo
from tabulate import tabulate

from src import conic_kernel
from src.astro_constants import LOW_ASSIST_FLYBY_ALTITUDE
from src.retrograde_return_legs import (
    _AssistChainParams,
    _body_state,
    _jupiter_assist_body,
    _LadderPricing,
    _mean_motion,
    _powered_node_burn,
)
from src.seed_cost import (
    SEED_PROPELLANT,
    STRIPPED_SHIPS,
    SeedShip,
    burn_from_circular,
    seed_payload,
)
from src.seed_route import (
    _PROXY_LOSS,
    METHALOX_STAGE_FRACTION,
    SEED_WINDOW_OPENS,
    SEED_WINDOW_YEARS,
    RouteFlight,
    _bodies,
    _close_on_earth,
    _island_workers,
    planet_longitudes,
    route_params,
    stack_per_ship,
)
from src.two_wave_growth import VE_METHALOX

#: The deep-space and flyby burns' specific impulse (s): methalox kept cold on
#: a multi-year coast (author, 2026-10-13).
STORED_ISP = 370.0
VE_STORED = STORED_ISP * 9.80665e-3
#: Loop-length bands (yr), one per resonance.
LOOP_BANDS: Dict[str, Tuple[float, float]] = {
    "1 yr": (0.9, 1.6),
    "2 yr": (1.6, 2.6),
    "3 yr": (2.6, 3.6),
}
#: Search bounds: launch excess speed (km/s), coast fraction, Earth-Jupiter
#: leg (yr).
V0_BOUNDS = (1.0, 12.0)
ETA_BOUNDS = (0.05, 0.95)
CRUISE_BOUNDS = (0.8, 5.0)
#: Time value of money (ADR 0040).
RATE = 0.10
#: The slowest return the seed may make: the chain's first (three-synodic)
#: cycle's growth wave, ``chains()[0][0].growth_wave_v_b`` (km/s).  ADR 0039's
#: 51.134 (``seed_route.RETURN_FLOOR``) is slower than the cycle the cost
#: model credits the seed with (ADR 0059).
CHAIN_FLOOR = 59.29
_YEAR_S = 365.25 * 86400.0
_INFEASIBLE = 1.0e6


def _unit(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return vector / float(np.linalg.norm(vector))


def _lambert(
    mu: float,
    start: npt.NDArray[np.float64],
    end: npt.NDArray[np.float64],
    tof: float,
    revolutions: int,
    low_path: bool,
) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    try:
        v_start, v_end = izzo(
            mu, start, end, tof, revolutions, True, low_path, 35, 1e-8
        )
    except (ValueError, RuntimeError):
        return None
    if not (np.all(np.isfinite(v_start)) and np.all(np.isfinite(v_end))):
        return None
    return v_start, v_end


@dataclass(frozen=True)
class EarthFlyby:
    """The powered Earth flyby at the end of the loop.

    Attributes:
        excess_in: Excess speed the loop brings back (km/s).
        excess_out: Excess speed the arc to Jupiter needs (km/s).
        turn: Angle between them (deg).
        periapsis_altitude: The solved pass (km above the surface).
    """

    excess_in: float
    excess_out: float
    turn: float
    periapsis_altitude: float


def loop_ladder(
    longitudes: Sequence[float],
    v0: float,
    alpha: float,
    eta: float,
    loop_years: float,
    cruise_years: float,
    params: _AssistChainParams,
) -> Optional[_LadderPricing]:
    """:func:`loop_flyby`'s ladder alone."""
    flown = loop_flyby(longitudes, v0, alpha, eta, loop_years, cruise_years, params)
    return None if flown is None else flown[0]


def loop_flyby(
    longitudes: Sequence[float],
    v0: float,
    alpha: float,
    eta: float,
    loop_years: float,
    cruise_years: float,
    params: _AssistChainParams,
) -> Optional[Tuple[_LadderPricing, EarthFlyby]]:
    """Earth, a deep-space burn, Earth again, then Jupiter, against real phases.

    The return to Earth after the burn is a prograde Lambert arc of up to one
    full revolution (both paths), and the cheapest burn is kept.

    Args:
        longitudes: Earth's and Jupiter's longitudes at launch (rad).
        v0: Launch excess speed (km/s).
        alpha: Its direction in the ecliptic, from Earth's velocity toward the
            Sun-Earth line outward (rad).
        eta: Share of the loop coasted before the burn.
        loop_years: Launch to the Earth flyby (yr).
        cruise_years: Earth flyby to Jupiter (yr).
        params: The parameter block.

    Returns:
        The ladder, with the deep-space burn and the flyby burn as its two
        nodes, and the flyby; or None if an arc fails.
    """
    mu = params.flyby.mu_sun
    (earth,) = _bodies("E", params)
    jupiter = _jupiter_assist_body(params)
    loop, cruise = loop_years * _YEAR_S, cruise_years * _YEAR_S
    r0, v_earth0 = _body_state(earth.orbit_radius, longitudes[0], earth.v_circ)
    v_launch = v_earth0 + v0 * (
        np.cos(alpha) * _unit(v_earth0) + np.sin(alpha) * _unit(r0)
    )
    coast = eta * loop
    r_burn, v_before = conic_kernel.kepler_propagate(r0, v_launch, coast, mu)
    r1, v_earth1 = _body_state(
        earth.orbit_radius,
        longitudes[0] + _mean_motion(earth) * loop,
        earth.v_circ,
    )
    best: Optional[Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    best = None
    for revolutions, low_path in ((0, True), (1, True), (1, False)):
        arc = _lambert(mu, r_burn, r1, loop - coast, revolutions, low_path)
        if arc is None:
            continue
        burn = float(np.linalg.norm(arc[0] - v_before))
        if best is None or burn < best[0]:
            best = (burn, arc[0], arc[1])
    if best is None:
        return None
    burn, v_after, v_arrive = best
    lon_jupiter = longitudes[1] + _mean_motion(jupiter) * (loop + cruise)
    r2, v_jupiter = _body_state(jupiter.orbit_radius, lon_jupiter, jupiter.v_circ)
    leg = _lambert(mu, r1, r2, cruise, 0, True)
    if leg is None:
        return None
    v_depart, v_reach = leg
    excess_in, excess_out = v_arrive - v_earth1, v_depart - v_earth1
    node, periapsis = _powered_node_burn(excess_in, excess_out, earth)
    arrival = v_reach - v_jupiter
    fl = params.flyby
    speed_in = float(np.linalg.norm(excess_in))
    speed_out = float(np.linalg.norm(excess_out))
    flyby = EarthFlyby(
        excess_in=speed_in,
        excess_out=speed_out,
        turn=float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(excess_in, excess_out) / (speed_in * speed_out), -1, 1
                    )
                )
            )
        ),
        periapsis_altitude=periapsis
        - earth.min_periapsis
        + float(LOW_ASSIST_FLYBY_ALTITUDE.to_value(u.km)),
    )
    pricing = _LadderPricing(
        departure_burn=float(np.sqrt(fl.v_esc_leo**2 + v0 * v0)) - fl.v_depart_from,
        node_total=burn + node,
        node_burns=[burn, node],
        arrival_excess=float(np.linalg.norm(arrival)),
        arrival_excess_vector=arrival,
        arrival_time=loop + cruise,
        arrival_longitude=lon_jupiter,
        legs=(
            (r0, v_launch, coast),
            (r_burn, v_after, loop - coast),
            (r1, v_depart, cruise),
        ),
    )
    return pricing, flyby


def seed_carried(
    ship: SeedShip, departure_ratio: float, burns: Sequence[float]
) -> float:
    """Seed when the ship itself flies every burn, its dry mass carried (kg).

    The ship keeps its methalox cold and fires its own engines in deep space
    and at the flyby, so nothing is staged:
    ``seed = P / (R_departure R_burns - 1) - dry``.

    Args:
        ship: The seed ship.
        departure_ratio: Mass ratio of the departure burn, loss included.
        burns: The deep-space and flyby burns (km/s), at 370 s.

    Returns:
        Seed mass (kg).
    """
    ratio = departure_ratio * float(np.exp(float(sum(burns)) / VE_STORED))
    propellant = float(SEED_PROPELLANT.to_value(u.kg))
    return propellant / (ratio - 1.0) - float(ship.dry_mass.to_value(u.kg))


def seed_after_burns(
    stack: float,
    burns: Sequence[float],
    stage_fraction: float = METHALOX_STAGE_FRACTION,
) -> float:
    """PuffSats left once the stored-methalox burns and their stage are paid.

    Args:
        stack: Mass leaving Earth (kg).
        burns: The deep-space and flyby burns (km/s).
        stage_fraction: Tankage and engines per kilogram of propellant.

    Returns:
        Seed mass (kg).
    """
    spent = 1.0 - float(np.exp(-float(sum(burns)) / VE_STORED))
    return stack * (1.0 - (1.0 + stage_fraction) * spent)


@dataclass(frozen=True)
class LoopSeed:
    """The best seed found in one loop band.

    Attributes:
        band: The loop band.
        flight: Its trajectory; ``node_burns`` are (deep-space, Earth flyby).
        launch_years: Launch after the window opens (yr).
        v0: Launch excess speed (km/s).
        alpha: Its direction (rad).
        eta: Coast share before the deep-space burn.
        ship: The seed ship.
        stack: Mass leaving Earth, real finite-burn loss (kg).
        seed: Seed per ship once the burns are paid (kg).
        flyby: The powered Earth flyby.
    """

    band: str
    flight: RouteFlight
    launch_years: float
    v0: float
    alpha: float
    eta: float
    ship: SeedShip
    stack: float
    seed: float
    flyby: EarthFlyby

    @property
    def x(self) -> Tuple[float, ...]:
        """The search vector: launch, v0, alpha, eta, loop and cruise years."""
        loop, cruise = self.flight.leg_years
        return (self.launch_years, self.v0, self.alpha, self.eta, loop, cruise)

    @property
    def return_years(self) -> float:
        """Window opening to the return's 1 AU crossing (yr)."""
        return self.launch_years + self.flight.trip_years

    def worth(self, direct_seed: float, direct_return: float) -> float:
        """``k x 1.1^-dt`` against the direct route on the same ship."""
        dt = self.return_years - direct_return
        return float(self.seed / direct_seed * (1.0 + RATE) ** (-dt))


class _LoopProblem:
    """pygmo problem: launch, v0, alpha, eta, loop and cruise; perijove inside."""

    def __init__(
        self,
        band: str,
        bend_sign: float,
        ship: SeedShip,
        floor: float = CHAIN_FLOOR,
        staged: bool = False,
    ) -> None:
        self.floor = floor
        self.staged = staged
        self.band = band
        self.bend_sign = bend_sign
        self.ship = ship
        self.params = route_params()
        self.longitudes0 = planet_longitudes("EJ", SEED_WINDOW_OPENS)
        self.rates = [_mean_motion(b) for b in (*_bodies("E", self.params),)] + [
            _mean_motion(_jupiter_assist_body(self.params))
        ]

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        lo, hi = LOOP_BANDS[self.band]
        return (
            [0.0, V0_BOUNDS[0], -np.pi, ETA_BOUNDS[0], lo, CRUISE_BOUNDS[0]],
            [
                SEED_WINDOW_YEARS,
                V0_BOUNDS[1],
                np.pi,
                ETA_BOUNDS[1],
                hi,
                CRUISE_BOUNDS[1],
            ],
        )

    def longitudes(self, launch_years: float) -> List[float]:
        """Earth's and Jupiter's longitudes at launch (rad)."""
        seconds = launch_years * _YEAR_S
        return [lon + n * seconds for lon, n in zip(self.longitudes0, self.rates)]

    def solve(self, x: Sequence[float]) -> Optional[RouteFlight]:
        """The flight closing on Earth for these parameters, if any."""
        launch, v0, alpha, eta, loop, cruise = (float(v) for v in x)
        longitudes = self.longitudes(launch)
        priced = loop_ladder(longitudes, v0, alpha, eta, loop, cruise, self.params)
        if priced is None:
            return None
        return _close_on_earth(
            "EEJ", priced, longitudes[0], (loop, cruise), self.bend_sign, self.params
        )

    def fitness(self, x: Sequence[float]) -> List[float]:
        flight = self.solve(x)
        if flight is None:
            return [_INFEASIBLE]
        if self.staged:
            stack = stack_per_ship(flight.v_infinity, self.ship)
            if stack <= 0.0:
                return [_INFEASIBLE / 2.0]
            seed = seed_after_burns(stack, flight.node_burns)
        else:
            burn = float(
                burn_from_circular(flight.v_infinity * u.km / u.s).to_value(u.km / u.s)
            )
            departure = float(np.exp((burn + _PROXY_LOSS) / VE_METHALOX))
            seed = seed_carried(self.ship, departure, flight.node_burns)
        if seed <= 0.0:
            # Closed but the burns eat the stack: slope toward smaller burns,
            # so the search is not left on a flat plateau.
            return [100.0 + float(sum(flight.node_burns))]
        years = float(x[0]) + flight.trip_years
        slow = max(0.0, self.floor - flight.collision_speed)
        return [-float(np.log(seed)) + years * float(np.log(1.0 + RATE)) + 10.0 * slow]


def _evolve(
    task: Tuple[str, float, SeedShip, int, int, int, int, float, bool],
) -> Tuple[List[float], float]:
    """One island: build the problem here, evolve, return its champion."""
    import pygmo as pg

    band, sign, ship, population, generations, evolutions, seed, floor, staged = task
    udp = _LoopProblem(band, sign, ship, floor, staged)
    pop = pg.population(pg.problem(udp), population, seed=seed)
    algo = pg.algorithm(pg.sade(gen=generations, seed=seed))
    for _ in range(evolutions):
        pop = algo.evolve(pop)
    return [float(v) for v in pop.champion_x], float(pop.champion_f[0])


def price_loop(
    band: str,
    sign: float,
    ship: SeedShip,
    x: Sequence[float],
    floor: float = CHAIN_FLOOR,
    staged: bool = False,
) -> Optional[LoopSeed]:
    """Re-price a champion with the real finite-burn loss; None if infeasible.

    The departure's loss is :func:`seed_payload`'s for a full 1200 t burn,
    which overstates it slightly when the ship keeps propellant back.
    """
    udp = _LoopProblem(band, sign, ship, floor, staged)
    flight = udp.solve(x)
    if flight is None or flight.collision_speed < floor - 1e-6:
        return None
    launch, v0, alpha, eta, loop, cruise = (float(v) for v in x)
    flown = loop_flyby(udp.longitudes(launch), v0, alpha, eta, loop, cruise, udp.params)
    assert flown is not None
    departure = seed_payload(ship, flight.v_infinity * u.km / u.s)
    if staged:
        stack = float(departure.payload.to_value(u.kg))
        seed = seed_after_burns(stack, flight.node_burns)
    else:
        seed = seed_carried(ship, departure.mass_ratio, flight.node_burns)
        stack = seed + float(ship.dry_mass.to_value(u.kg))
    if seed <= 0.0:
        return None
    return LoopSeed(band, flight, launch, v0, alpha, eta, ship, stack, seed, flown[1])


def best_loop(
    band: str,
    ship: SeedShip = STRIPPED_SHIPS[0],
    seed: int = 0,
    islands: int = 4,
    population: int = 40,
    generations: int = 150,
    evolutions: int = 3,
    floor: float = CHAIN_FLOOR,
    staged: bool = False,
) -> Optional[LoopSeed]:
    """Most discounted seed per ship in one loop band (pygmo SADE islands).

    Islands are independent, seeded ``seed + 2 i + s`` on bend side ``s``, and
    run in a process pool capped by free memory, as in ADR 0039.

    Args:
        band: A key of :data:`LOOP_BANDS`.
        ship: The seed ship.
        seed: Random seed.
        islands: Islands per bend side.
        population: Individuals per island.
        generations: Generations per evolution.
        evolutions: Evolutions per island.
        floor: Slowest acceptable return (km/s).
        staged: Fly the later burns on ADR 0039's light stage and drop the
            ship after departure, instead of carrying it (the default).

    Returns:
        The best feasible loop, or None.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    tasks = [
        (
            band,
            sign,
            ship,
            population,
            generations,
            evolutions,
            seed + 2 * i + side,
            floor,
            staged,
        )
        for side, sign in enumerate((1.0, -1.0))
        for i in range(islands)
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=_island_workers(len(tasks)), mp_context=context
    ) as pool:
        champions = list(pool.map(_evolve, tasks))
    best: Optional[LoopSeed] = None
    for task, (x, _) in zip(tasks, champions):
        found = price_loop(band, task[1], ship, x, floor, staged)
        if found is not None and (
            best is None
            or found.seed * (1.0 + RATE) ** (-found.return_years)
            > best.seed * (1.0 + RATE) ** (-best.return_years)
        ):
            best = found
    return best


def loop_row(
    found: LoopSeed, direct_seed: float, direct_return: float
) -> Dict[str, object]:
    """One table row for a loop against the direct route on its ship."""
    deep, flyby = found.flight.node_burns
    return {
        "ship": found.ship.name,
        "loop": found.band,
        "launch yr": found.launch_years,
        "cruise yr": found.flight.leg_years[1],
        "loop yr": found.flight.leg_years[0],
        "return yr": found.return_years,
        "v0 km/s": found.v0,
        "alpha deg": float(np.degrees(found.alpha)),
        "DSM at": found.eta,
        "DSM km/s": deep,
        "flyby km/s": flyby,
        "Earth v_inf in": found.flyby.excess_in,
        "out": found.flyby.excess_out,
        "turn deg": found.flyby.turn,
        "pass km": found.flyby.periapsis_altitude,
        "v_b km/s": found.flight.collision_speed,
        "seed t": found.seed / 1e3,
        "k": found.seed / direct_seed,
        "worth": found.worth(direct_seed, direct_return),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Search every band on both ships and print them against direct.

    Args:
        argv: ``--seed``, ``--islands`` and ``--generations`` override the
            search; ``--direct-seed`` and ``--direct-return`` (per ship, t and
            yr) the direct route it is scored against.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--bands", nargs="*", default=list(LOOP_BANDS))
    parser.add_argument("--floor", type=float, default=CHAIN_FLOOR)
    parser.add_argument(
        "--staged", action="store_true", help="drop the ship after departure"
    )
    args = parser.parse_args(argv)
    from src.seed_route import Propulsion, best_route

    rows = []
    for ship in STRIPPED_SHIPS:
        direct = best_route(
            "EJ", Propulsion.METHALOX, ship=ship, rate=RATE, seed=args.seed,
            islands=args.islands, generations=args.generations, floor=args.floor,
        )  # fmt: skip
        assert direct is not None
        direct_return = (direct.return_jd - SEED_WINDOW_OPENS) / 365.25
        print(
            f"{ship.name}: direct EJ launches {(direct.launch_jd - SEED_WINDOW_OPENS) / 365.25:.2f} yr, "
            f"v_inf {direct.flight.v_infinity:.2f} km/s, seed {direct.puffsats_per_ship / 1e3:.1f} t, "
            f"returns {direct_return:.2f} yr",
            flush=True,
        )
        for band in args.bands:
            found = best_loop(
                band, ship, seed=args.seed, islands=args.islands,
                generations=args.generations, floor=args.floor, staged=args.staged,
            )  # fmt: skip
            if found is None:
                print(f"  {band}: no loop closes", flush=True)
                continue
            rows.append(loop_row(found, direct.puffsats_per_ship, direct_return))
            print(
                f"  {band} champion x = {found.x!r}, seed {found.seed!r} kg, "
                f"direct {direct.puffsats_per_ship!r} kg, return "
                f"{found.return_years!r} vs {direct_return!r} yr",
                flush=True,
            )
            print(tabulate([rows[-1]], headers="keys", floatfmt=".3g"), flush=True)
    print(tabulate(rows, headers="keys", floatfmt=".3g"))


if __name__ == "__main__":
    main()
