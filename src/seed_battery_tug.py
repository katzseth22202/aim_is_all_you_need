"""An argon tug that stores a whole orbit's sunlight and spends it at perigee (ADR 0061).

ADR 0058's perigee pump failed on its duty cycle: thrust only near perigee,
and perigee comes round once an orbit, so the array idled most of the time and
had to be sized for the arcs.  The author's fix (2026-10-14): the seed's
payload is water, so **electrolyse it all round the orbit and recombine the
hydrogen and oxygen in a fuel cell during the perigee arc**.  The water is a
battery whose storage medium is already paid for, since it comes back as
water.  The array is then sized for the orbit's average power, and the
thrusters, fuel cells and power bus for the arc's peak, tens of MW to a
gigawatt.

**The thruster.**  A quasi-steady magnetoplasmadynamic (MPD) thruster on
argon: an arc whose own current and field push the plasma out, the device
built for MW-GW pulses.  An argon arcjet (heat alone) gives 3-5 km/s, barely
above methalox, and VASIMR loses most of its power ionising argon below about
5000 s, so neither is flown.  Exhaust speed is free in :data:`VE_BOUNDS`, at a
fixed :data:`MPD_EFFICIENCY`.  Fuel cells deliver low-voltage, high-current DC,
which is what an MPD arc wants, so the bus is light.

**The heat.**  The fuel cell sheds about as much heat as it delivers, and the
thruster keeps part of its loss in its electrodes.  That heat goes into the
payload water during the arc, as warmed liquid or melted ice
(:data:`WATER_SINKS`, whichever is better), and is radiated away over the
rest of the orbit, so the radiators are sized for the average.  The heat a
pass dumps must fit in the seed's water.

**The ship.**  The tug hands the fuelled ship over on an ellipse of
semi-major axis ``a_f``.  The ship burns at perigee onto its route
(:data:`ROUTES`): direct to 12.20 km/s (ADR 0060), or ADR 0059's two-year loop
(its champions' launch excess and later burns held fixed, dry mass carried).  Methalox does not boil
off: blankets, plus cryocoolers drawing a few kW off the tug's array
(author, 2026-10-14).

**The test.**  The seed's departure date does not move.  The launch date is
free (ADR 0059), so the climb starts early and is charged only as capital:
everything is bought ``climb`` years sooner, ``x 1.1^climb``.  The array is
sold at L1 (:func:`array_credit`).  ``worth`` is seed per net dollar over the
same route flown without a tug; above 1 the tug pays.

**The answer.**  The battery works: low hand-overs and wide arcs, peaks of
4-25 MW, two to three times the seed.  But the tug costs as much again as the
ship, so it reaches 1.31x direct only with a $10/W array and burst hardware
at $0.25/W, and loses at $3/W of burst hardware at any array price.  Every
best case climbs the full 2 years allowed at the 12 km/s exhaust floor.

Every hardware figure here is an assumption without a source; they are
gathered at the top so the ADR can list them.  Orbit-averaged, coplanar, no
drag or third bodies, and the arcs and eclipse are not phased against each
other.
"""

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from scipy.optimize import differential_evolution
from tabulate import tabulate

from src.finite_burn_loss import finite_burn_loss
from src.growth_ledger import RAPTOR3_THRUST
from src.seed_cost import (
    BASELINE_FLIGHT,
    FLIGHT_PRICES,
    SEED_PROPELLANT,
    SHIP_PRICES,
    STRIPPED_SHIPS,
    SeedShip,
)
from src.seed_loop import seed_carried
from src.seed_sep_tug import (
    _MU,
    _R_EARTH,
    _YEAR_S,
    ARRAY_SPECIFIC_MASS,
    BELT_DERATING,
    END_OF_LIFE,
    L1_SHIPPING,
    RATE,
    SALE_LAG,
    START_RADIUS,
    SUNLIT,
)
from src.sep_split_correction import ARGON_TANK_FRACTION
from src.two_wave_growth import VE_METHALOX

#: ADR 0059's two-year champions, per ship (index into ``STRIPPED_SHIPS``):
#: launch excess speed (km/s) and the (deep-space, Earth flyby) burns at 370 s
#: (km/s), from ``seed_loop.price_loop("2 yr", -1.0, ship, x)``.
LOOP_FLIGHTS: Dict[int, Tuple[float, Tuple[float, ...]]] = {
    0: (5.21552, (0.560729080578008, 2.708449472563413)),
    1: (5.21308, (0.5080241562264862, 2.7267225834495417)),
}
#: The direct route to the chain's floor (ADR 0059's comparator, ADR 0060's
#: headline): one departure to 12.20 km/s, no later burns.
DIRECT_FLIGHTS: Dict[int, Tuple[float, Tuple[float, ...]]] = {
    0: (12.20, ()),
    1: (12.20, ()),
}
#: The seed routes a tug can feed.
ROUTES: Dict[str, Dict[int, Tuple[float, Tuple[float, ...]]]] = {
    "direct": DIRECT_FLIGHTS,
    "2-yr loop": LOOP_FLIGHTS,
}

# --- The thruster (assumptions) ---------------------------------------------
#: Jet power over electrical input, argon quasi-steady MPD at MW class.
MPD_EFFICIENCY = 0.3
#: Share of the thruster's loss that leaves in the plume (ionisation, frozen
#: flow) rather than heating the electrodes.
MPD_PLUME_LOSS_SHARE = 0.5
#: Thrusters and bus, kg per kW of peak electrical input.
MPD_SPECIFIC_MASS = 1.0
#: Exhaust-speed bounds (km/s), about 1200-3000 s.
VE_BOUNDS = (12.0, 30.0)

# --- The water battery (assumptions) ----------------------------------------
#: Hydrogen's higher heating value (J/kg), what electrolysis stores.
HHV_HYDROGEN = 141.9e6
#: Electrolyser: chemical energy stored over electrical input; kg per kW.
ELECTROLYSER_EFFICIENCY = 0.8
ELECTROLYSER_SPECIFIC_MASS = 2.0
#: Fuel cell: electrical output over chemical input; kg per kW of output.
FUEL_CELL_EFFICIENCY = 0.6
FUEL_CELL_SPECIFIC_MASS = 1.0
#: Gas tanks per kilogram of hydrogen held: hydrogen at about 5.5% by weight,
#: plus its oxygen, half as many moles at the same pressure.
GAS_TANK_PER_HYDROGEN = 27.0

# --- The heat (assumptions) -------------------------------------------------
#: Payload water as the arc's heat sink: (J/kg it absorbs, the temperature the
#: radiator rejects it at, K).  Liquid warmed 290 -> 340 K, rejected at its cold
#: end; ice melted and refrozen at 273 K.
WATER_SINKS: Dict[str, Tuple[float, float]] = {
    "liquid": (4186.0 * 50.0, 290.0),
    "ice": (334.0e3, 273.0),
}
#: Radiator: two-sided, emissivity 0.85, 5 kg per m^2.
RADIATOR_EMISSIVITY = 0.85
RADIATOR_AREAL_MASS = 5.0
_SIGMA = 5.670374419e-8

# --- Prices (assumptions) ---------------------------------------------------
#: Array, gross $/W of rating; sold at L1 after the climb.
ARRAY_PRICES = (10.0, 25.0, 50.0)
#: Thrusters, bus, fuel cells and electrolysers, $/W of their rating; not sold.
BURST_PRICES = (0.25, 1.0, 3.0)
#: Tanks and radiators, $/kg.
PASSIVE_PRICE = 1000.0
#: The longest climb searched (yr).
MAX_CLIMB = 2.0

#: Search bounds: arc half-width (deg), hand-over semi-major axis (km), array
#: rating (MW), peak thruster input over array rating.
THETA_BOUNDS = (15.0, 150.0)
AXIS_BOUNDS = (9.0e3, 90.0e3)
ARRAY_MW_BOUNDS = (0.3, 40.0)
PEAK_RATIO_BOUNDS = (1.0, 300.0)

_ARC_SAMPLES = 181
_AXIS_STEP = 0.01


@dataclass(frozen=True)
class Climb:
    """One battery tug's climb (SI units unless named).

    Attributes:
        years: Climb time.
        argon: Argon spent (kg).
        delta_v: Tug delta-v (km/s).
        perigee: Perigee radius at hand-over (km).
        perigee_speed: Speed there (km/s).
        axis: Semi-major axis at hand-over (km).
        peak: Largest electrical input to the thrusters (W).
        fuel_cell: Largest fuel-cell output (W).
        hydrogen: Most hydrogen held at once (kg).
        heat_per_pass: Most heat the water must buffer in one arc, beyond what
            the radiator rejects during it (J).
        mean_heat: Most heat per second averaged over an orbit (W).
    """

    years: float
    argon: float
    delta_v: float
    perigee: float
    perigee_speed: float
    axis: float
    peak: float
    fuel_cell: float
    hydrogen: float
    heat_per_pass: float
    mean_heat: float


def climb(
    theta: float,
    handover_axis: float,
    exhaust: float,
    array: float,
    peak: float,
    start_mass: float,
) -> Climb:
    """Pump apogee on stored sunlight: Gauss's equations, orbit by orbit.

    Each orbit the array charges the battery except during the arc, when it
    feeds the thrusters directly; the arc spends what the battery returns,
    up to ``peak`` for the arc's length.  Thrust is tangential and constant
    across the arc, and the orbit is stepped several orbits at a time (1% of
    ``a`` per step), argon leaving as it is spent.

    Args:
        theta: Arc half-width about perigee (deg).
        handover_axis: Stop at this semi-major axis (km).
        exhaust: Exhaust speed (km/s).
        array: Array rating (W).
        peak: Thruster input rating (W).
        start_mass: Stack at the start, argon included (kg).

    Returns:
        The climb.
    """
    mu = _MU * 1.0e9
    nu = np.linspace(-np.radians(theta), np.radians(theta), _ARC_SAMPLES)
    ve = exhaust * 1.0e3
    supply = array * SUNLIT * BELT_DERATING
    round_trip = ELECTROLYSER_EFFICIENCY * FUEL_CELL_EFFICIENCY
    thruster_heat = (1.0 - MPD_EFFICIENCY) * (1.0 - MPD_PLUME_LOSS_SHARE)
    axis, ecc = START_RADIUS * 1.0e3 * 1.0001, 1.0e-4
    mass, elapsed, delta_v = start_mass, 0.0, 0.0
    used_peak = fuel_cell = hydrogen = pass_heat = mean_heat = 0.0
    while axis < handover_axis * 1.0e3:
        semi_latus = axis * (1.0 - ecc * ecc)
        momentum = np.sqrt(mu * semi_latus)
        radius = semi_latus / (1.0 + ecc * np.cos(nu))
        speed = np.sqrt(mu * (2.0 / radius - 1.0 / axis))
        dt_dnu = radius * radius / momentum
        arc = float(np.trapezoid(dt_dnu, nu))
        period = 2.0 * np.pi * np.sqrt(axis**3 / mu)
        direct = supply * arc
        stored = supply * (period - arc)
        energy = min(direct + round_trip * stored, peak * arc)
        drawn = max(0.0, energy - direct) / round_trip
        acceleration = 2.0 * MPD_EFFICIENCY * energy / arc / ve / mass
        d_axis = float(
            np.trapezoid(2.0 * axis * axis * speed / mu * acceleration * dt_dnu, nu)
        )
        d_ecc = float(
            np.trapezoid(2.0 * (ecc + np.cos(nu)) / speed * acceleration * dt_dnu, nu)
        )
        orbits = max(1, int(_AXIS_STEP * axis / abs(d_axis)))
        argon = 2.0 * MPD_EFFICIENCY * energy / (ve * ve)
        burst_heat = (
            ELECTROLYSER_EFFICIENCY * (1.0 - FUEL_CELL_EFFICIENCY) * drawn
            + thruster_heat * energy
        )
        orbit_heat = burst_heat + (1.0 - ELECTROLYSER_EFFICIENCY) * drawn
        used_peak = max(used_peak, energy / arc)
        fuel_cell = max(fuel_cell, (energy - min(direct, energy)) / arc)
        hydrogen = max(hydrogen, ELECTROLYSER_EFFICIENCY * drawn / HHV_HYDROGEN)
        pass_heat = max(pass_heat, burst_heat - orbit_heat / period * arc)
        mean_heat = max(mean_heat, orbit_heat / period)
        axis += orbits * d_axis
        ecc = min(ecc + orbits * d_ecc, 0.9999)
        mass -= orbits * argon
        elapsed += orbits * period
        delta_v += orbits * acceleration * arc
        if elapsed > 2.0 * MAX_CLIMB * _YEAR_S or mass <= 0.0:
            break
    perigee = axis * (1.0 - ecc)
    return Climb(
        years=elapsed / _YEAR_S,
        argon=start_mass - mass,
        delta_v=delta_v / 1.0e3,
        perigee=perigee / 1.0e3,
        perigee_speed=float(np.sqrt(mu * (2.0 / perigee - 1.0 / axis))) / 1.0e3,
        axis=axis / 1.0e3,
        peak=used_peak,
        fuel_cell=fuel_cell,
        hydrogen=hydrogen,
        heat_per_pass=pass_heat,
        mean_heat=mean_heat,
    )


def ship_seed(
    perigee: float,
    perigee_speed: float,
    axis: float,
    ship_index: int,
    route: str = "direct",
) -> float:
    """Seed the ship sends from a perigee onto a route of :data:`ROUTES` (kg).

    The departure is steered, its finite-burn loss charged for the propellant
    it actually burns (one refinement pass); the loop's later burns are fixed.

    Args:
        perigee: Perigee radius (km).
        perigee_speed: Speed there (km/s).
        axis: The orbit's semi-major axis (km).
        ship_index: Into ``STRIPPED_SHIPS``.
        route: A key of :data:`ROUTES`.

    Returns:
        Seed mass (kg); negative if the ship cannot fly it.
    """
    ship = STRIPPED_SHIPS[ship_index]
    v0, burns = ROUTES[route][ship_index]
    burn = float(np.sqrt(v0 * v0 + 2.0 * _MU / perigee)) - perigee_speed
    propellant = float(SEED_PROPELLANT.to_value(u.kg))
    dry = float(ship.dry_mass.to_value(u.kg))
    flow = float(
        (ship.engines * RAPTOR3_THRUST / (VE_METHALOX * u.km / u.s)).to_value(
            u.kg / u.s
        )
    )
    ratio = float(np.exp(burn / VE_METHALOX))
    for _ in range(2):
        seed = seed_carried(ship, ratio, burns)
        spent = (propellant + dry + seed) * (1.0 - 1.0 / ratio)
        loss = finite_burn_loss(
            burn * u.km / u.s,
            VE_METHALOX * u.km / u.s,
            spent / flow * u.s,
            steered=True,
            altitude=(perigee - _R_EARTH) * u.km,
            period=2.0 * np.pi * np.sqrt(axis**3 / _MU) * u.s,
        ).to_value(u.km / u.s)
        ratio = float(np.exp((burn + loss) / VE_METHALOX))
    return seed_carried(ship, ratio, burns)


def loop_seed(ship_index: int, route: str = "direct") -> float:
    """The route's seed flown from the 200 km circle, same accounting (kg)."""
    return ship_seed(
        START_RADIUS,
        float(np.sqrt(_MU / START_RADIUS)),
        START_RADIUS,
        ship_index,
        route,
    )


def array_credit(array: float, price: float) -> float:
    """The array sold at L1, valued at the ship's departure ($).

    What a buyer would otherwise pay, its gross price at end-of-life output
    plus shipping its mass there, received :data:`SALE_LAG` after hand-over.

    Args:
        array: Rating (W).
        price: Gross $/W.

    Returns:
        The credit.
    """
    value = (price * END_OF_LIFE + L1_SHIPPING * ARRAY_SPECIFIC_MASS / 1.0e3) * array
    return float(value / (1.0 + RATE) ** SALE_LAG)


def radiator_mass(heat: float, temperature: float) -> float:
    """Two-sided radiator rejecting ``heat`` (W) at ``temperature`` (K), kg."""
    flux = 2.0 * RADIATOR_EMISSIVITY * _SIGMA * temperature**4
    return heat / flux * RADIATOR_AREAL_MASS


@dataclass(frozen=True)
class BatteryTug:
    """One priced battery tug and the seed it lifts.

    Attributes:
        theta: Arc half-width (deg).
        exhaust: Exhaust speed (km/s).
        array: Array rating (W).
        climb: The climb.
        sink: Which :data:`WATER_SINKS` entry holds the heat.
        seed: Seed the ship sends (kg).
        water_needed: Water one pass's heat needs (kg).
        hardware: Tug hardware mass, argon tanks included (kg).
        dollars: Everything, valued at the ship's departure ($).
        credit: The array's sale ($).
        worth: Seed per net dollar over the route without a tug.
        x: The design vector :func:`price_tug` was given.
    """

    theta: float
    exhaust: float
    array: float
    climb: Climb
    sink: str
    seed: float
    water_needed: float
    hardware: float
    dollars: float
    credit: float
    worth: float
    x: Tuple[float, ...]


def plain_loop_cost(
    ship_index: int, flight_price: float, route: str = "direct"
) -> Tuple[float, float]:
    """(seed kg, dollars) without a tug: propellant and seed lofted, hull."""
    seed = loop_seed(ship_index, route)
    lofted = float(SEED_PROPELLANT.to_value(u.kg)) + seed
    return seed, flight_price / 1.0e5 * lofted + SHIP_PRICES[ship_index]


def price_tug(
    x: Sequence[float],
    array_price: float,
    burst_price: float,
    ship_index: int = 0,
    flight_price: float = FLIGHT_PRICES[BASELINE_FLIGHT],
    route: str = "direct",
) -> Optional[BatteryTug]:
    """Fly and price one tug; None if it cannot finish or its heat will not fit.

    ``x`` is (arc half-width deg, hand-over axis km, exhaust km/s, array MW,
    peak over array).  The tug's own mass rides its climb, so the climb is
    solved by fixed-point iteration on the argon and hardware.

    Args:
        x: The design.
        array_price: Gross $/W.
        burst_price: Thrusters, bus, fuel cells, electrolysers, $/W.
        ship_index: Into ``STRIPPED_SHIPS``.
        flight_price: Dollars per 100 t flight.
        route: A key of :data:`ROUTES`; the tug is scored against it flown
            without a tug.

    Returns:
        The tug, or None.
    """
    theta, axis, exhaust, array_mw, ratio = (float(v) for v in x)
    array, peak = array_mw * 1.0e6, array_mw * 1.0e6 * ratio
    ship: SeedShip = STRIPPED_SHIPS[ship_index]
    propellant = float(SEED_PROPELLANT.to_value(u.kg))
    dry = float(ship.dry_mass.to_value(u.kg))
    hardware, argon = 0.0, 0.0
    flown: Optional[Climb] = None
    seed = 0.0
    sink, water_needed = "", 0.0
    for _ in range(4):
        guess = flown
        seed = (
            ship_seed(guess.perigee, guess.perigee_speed, guess.axis, ship_index, route)
            if guess is not None
            else loop_seed(ship_index, route)
        )
        if seed <= 0.0:
            return None
        start = propellant + dry + seed + hardware + argon
        flown = climb(theta, axis, exhaust, array, peak, start)
        if flown.axis < 0.999 * axis or flown.years > MAX_CLIMB:
            return None
        argon = flown.argon
        options = []
        for name, (capacity, temperature) in WATER_SINKS.items():
            needed = flown.heat_per_pass / capacity
            mass = radiator_mass(flown.mean_heat, temperature)
            options.append((mass, needed, name))
        radiator, water_needed, sink = min(
            (o for o in options if o[1] <= seed), default=min(options)
        )
        hardware = (
            ARRAY_SPECIFIC_MASS * array / 1.0e3
            + MPD_SPECIFIC_MASS * flown.peak / 1.0e3
            + FUEL_CELL_SPECIFIC_MASS * flown.fuel_cell / 1.0e3
            + ELECTROLYSER_SPECIFIC_MASS * array / 1.0e3
            + GAS_TANK_PER_HYDROGEN * flown.hydrogen
            + radiator
            + ARGON_TANK_FRACTION * argon
        )
    assert flown is not None
    seed = ship_seed(flown.perigee, flown.perigee_speed, flown.axis, ship_index, route)
    if seed <= 0.0 or water_needed > seed:
        return None
    lofted = propellant + seed + hardware + argon
    passive = GAS_TANK_PER_HYDROGEN * flown.hydrogen + radiator
    spend = (
        flight_price / 1.0e5 * lofted
        + SHIP_PRICES[ship_index]
        + array_price * array
        + burst_price * (flown.peak + flown.fuel_cell + array)
        + PASSIVE_PRICE * passive
    )
    dollars = spend * (1.0 + RATE) ** flown.years
    credit = array_credit(array, array_price)
    loop, loop_dollars = plain_loop_cost(ship_index, flight_price, route)
    worth = (seed / (dollars - credit)) / (loop / loop_dollars)
    return BatteryTug(
        theta=theta,
        exhaust=exhaust,
        array=array,
        climb=flown,
        sink=sink,
        seed=seed,
        water_needed=water_needed,
        hardware=hardware,
        dollars=dollars,
        credit=credit,
        worth=float(worth),
        x=tuple(float(v) for v in x),
    )


def _objective(
    x: Sequence[float],
    array_price: float,
    burst_price: float,
    ship_index: int,
    route: str,
) -> float:
    tug = price_tug(x, array_price, burst_price, ship_index, route=route)
    return 10.0 if tug is None else -tug.worth


def best_tug(
    array_price: float,
    burst_price: float,
    ship_index: int = 0,
    seed: int = 7,
    route: str = "direct",
    maxiter: int = 60,
    popsize: int = 12,
) -> Optional[BatteryTug]:
    """The most valuable tug at one pair of prices (scipy differential evolution).

    Args:
        array_price: Gross $/W.
        burst_price: Burst hardware $/W.
        ship_index: Into ``STRIPPED_SHIPS``.
        seed: Random seed.
        maxiter: Generations.
        popsize: Population per parameter.
        route: A key of :data:`ROUTES`.

    Returns:
        The best tug found, or None.
    """
    found = differential_evolution(
        _objective,
        bounds=[
            THETA_BOUNDS,
            AXIS_BOUNDS,
            VE_BOUNDS,
            ARRAY_MW_BOUNDS,
            PEAK_RATIO_BOUNDS,
        ],
        args=(array_price, burst_price, ship_index, route),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        polish=False,
    )
    return price_tug(found.x, array_price, burst_price, ship_index, route=route)


def tug_row(
    tug: BatteryTug, array_price: float, burst_price: float
) -> Dict[str, object]:
    """One table row."""
    c = tug.climb
    return {
        "array $/W": array_price,
        "burst $/W": burst_price,
        "worth": tug.worth,
        "seed t": tug.seed / 1e3,
        "climb yr": c.years,
        "a_f km": c.axis,
        "theta": tug.theta,
        "ve km/s": tug.exhaust,
        "array MW": tug.array / 1e6,
        "peak MW": c.peak / 1e6,
        "tug km/s": c.delta_v,
        "argon t": c.argon / 1e3,
        "hardware t": tug.hardware / 1e3,
        "H2 t": c.hydrogen / 1e3,
        "sink": tug.sink,
        "water/pass t": tug.water_needed / 1e3,
        "net $M": (tug.dollars - tug.credit) / 1e6,
    }


def _search(task: Tuple[float, float, int, int, str]) -> Optional[Dict[str, object]]:
    array_price, burst_price, ship_index, seed, route = task
    tug = best_tug(array_price, burst_price, ship_index, seed, route=route)
    if tug is None:
        return None
    row = tug_row(tug, array_price, burst_price)
    row["x"] = tug.x
    return row


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Search every price pair on each ship and route and print the best tugs.

    Args:
        argv: ``--ships`` (indices), ``--routes``, ``--workers`` and ``--seed``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ships", nargs="*", type=int, default=[0])
    parser.add_argument("--routes", nargs="*", default=list(ROUTES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    from concurrent.futures import ProcessPoolExecutor

    for route, index in itertools.product(args.routes, args.ships):
        plain, plain_dollars = plain_loop_cost(
            index, FLIGHT_PRICES[BASELINE_FLIGHT], route
        )
        print(
            f"{STRIPPED_SHIPS[index].name}, {route}: without a tug sends "
            f"{plain / 1e3:.1f} t for ${plain_dollars / 1e6:.0f}M",
            flush=True,
        )
        tasks = [
            (a, b, index, args.seed, route)
            for a, b in itertools.product(ARRAY_PRICES, BURST_PRICES)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows: List[Dict[str, object]] = [
                r for r in pool.map(_search, tasks) if r is not None
            ]
        print(tabulate(rows, headers="keys", floatfmt=".3g"), flush=True)


if __name__ == "__main__":
    main()
