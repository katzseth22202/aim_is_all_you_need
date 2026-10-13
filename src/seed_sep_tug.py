"""Can a solar-electric tug lift the fuelled seed ship out of low orbit? (ADR 0058)

The seed ship (ADR 0035/0036) burns 8.50 km/s from a 200 km circular orbit.
From the periapsis of a high ellipse the same departure is 5.3-6.6 km/s, so a
ship that starts there sends three to four times the seed.  The question is
what it costs to put the **fuelled** ship there: its 1200 t of methalox must
be lifted too, or the chemical climb spends what the Oberth burn saves.  This
module charges an argon tug for that lift and asks whether the seed gets
cheaper per discounted dollar.

**Two ways to climb.**

* :func:`circular_spiral` -- the author's first proposal: spiral out to the
  20-day orbit's apogee on near-circular orbits (Edelbaum's
  ``v_LEO - v_circ``), then lower perigee to 200 km at apogee.  7.67 km/s,
  because a slow spiral spends its thrust where the speed is low and gives
  up the Oberth effect.
* :func:`perigee_pump` -- thrust only on an arc of half-width ``theta`` about
  perigee, which raises apogee and keeps perigee low.  Orbit-averaged Gauss
  equations for tangential thrust, stepped orbit by orbit.  It needs about
  3 km/s, but the duty cycle collapses as the orbit grows, so the tug hands
  the ship over on an ellipse of semi-major axis ``a_f`` well short of the
  20-day orbit and the ship's Oberth burn buys the rest.

**The tug's bill** (:func:`tug_seed`).  Array, power processing and thrusters
at ADR 0026's 15 kg/kW, bought by the watt; argon at 2000 s and efficiency 0.5,
with 15% tankage; everything launched at the flight price per kilogram.  Power
is sized so the climb takes the stated time with thrust only in sunlight
(:data:`SUNLIT`) and the array derated for the Van Allen belts
(:data:`BELT_DERATING`).  Methalox boils off while it waits
(``boil_off_per_day``).  Afterwards the tug flies on to the Sun-Earth L1 and
its array is sold there (:func:`l1_credit`).

**The test** is ADR 0039's: ``k x 1.1^-dt > 1``, with ``k`` the seed per
dollar over the direct route's and ``dt`` the wait for the next Jupiter window
the climb forces (:func:`window_delay`).  The odds the cycle works cancel.

**The answer.**  Only with hardware at $10/W, no boil-off and the array sold
at L1 does the tug reach 1.15x, about what ADR 0039's free Earth loop already
gives (1.16x).  At $25/W, or with 0.05% a day of boil-off, it loses.

Orbit-averaged, coplanar, no drag, no lunar or solar perturbation, no plane
change: enough to call the trade, not to fly it.
"""

import argparse
import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from boinor.bodies import Earth
from tabulate import tabulate

from src.astro_constants import LEO_ALTITUDE, PUFFSAT_CYCLE_ORBIT_PERIOD
from src.finite_burn_loss import finite_burn_loss
from src.growth_ledger import RAPTOR3_THRUST
from src.real_orbit_resonance import _FIXED_TWO_SYNODIC_DAYS
from src.seed_cost import (
    BASELINE_FLIGHT,
    FLIGHT_PRICES,
    SEED_PROPELLANT,
    SHIP_PRICES,
    STRIPPED_SHIPS,
    SeedShip,
    seed_payload,
)
from src.seed_route import DEFAULT_SPECIFIC_MASS
from src.sep_split_correction import (
    ARGON_TANK_FRACTION,
    SEP_THRUSTER_EFFICIENCY,
    VE_ARGON,
)
from src.two_wave_growth import VE_METHALOX

_MU = float(Earth.k.to_value(u.km**3 / u.s**2))
_R_EARTH = float(Earth.R.to_value(u.km))
_YEAR_S = 365.25 * 86400.0
#: The seed's parking orbit, where every climb starts (km from the centre).
START_RADIUS = _R_EARTH + float(LEO_ALTITUDE.to_value(u.km))
#: The chain's first excess speed (ADR 0035), what the seed must leave with.
SEED_EXCESS = 11.99276933457366
#: One Earth-Jupiter synodic period (yr): a climb that misses the window
#: waits for the next one.
SYNODIC_YEARS = _FIXED_TWO_SYNODIC_DAYS / 2.0 / 365.25
#: Share of a climb's time the array is in sunlight: the noon-midnight
#: (worst-case) eclipse fraction averaged over a circular spiral from 200 km,
#: uniformly in speed (:func:`mean_sunlit`).  Applied to the pump too.
SUNLIT = 0.87
#: Mean array output over the climb against its rating, for the radiation
#: belts.  An assumption (author, 2026-10-13: the belts are survivable over
#: months); all-electric GEO satellites cross them with modest loss.
BELT_DERATING = 0.9
#: What the array still delivers when it is sold at L1, against its rating.
END_OF_LIFE = 0.8
#: The array's own share of the 15 kg/kW (about 150 W/kg, ADR 0026), the part
#: that is sold.  Power processing and thrusters are not.
ARRAY_SPECIFIC_MASS = 7.0
#: The array's share of the hardware price.  An unsourced assumption.
ARRAY_PRICE_SHARE = 0.5
#: What a buyer at L1 would otherwise pay to ship a kilogram there ($/kg):
#: SpaceX to high orbit (author, 2026-10-13).
L1_SHIPPING = 500.0
#: From hand-over to the array's sale at L1 (yr), an assumption.
SALE_LAG = 0.25
#: Time value of money (ADR 0040).
RATE = 0.10
#: Hardware dollars per watt.  $50 and $200 are ADR 0039's lower tiers; $10
#: and $25 are added to find where the tug starts to pay (no source).
PRICES_PER_WATT = (10.0, 25.0, 50.0, 200.0)
#: Methalox lost per day while the tug climbs: none (an ideal store) and a
#: well-insulated tank (an assumption).
BOIL_OFF_PER_DAY = (0.0, 0.0005)
#: The grid the pump is searched on.
ARC_HALF_WIDTHS = (30.0, 45.0, 60.0, 90.0, 120.0)
HANDOVER_AXES = (15.0e3, 25.0e3, 40.0e3, 80.0e3, 160.0e3)
CLIMB_YEARS = (0.5, 1.0, 1.5, 2.0, 3.0)
#: Reference acceleration the pump's geometry is computed at (m/s^2).  The
#: delta-v and duty cycle barely depend on it while it is small; only the
#: time scales, as 1 / acceleration.
_REFERENCE_ACCELERATION = 2.0e-4
#: Angular samples per orbit for the orbit average.
_SAMPLES = 1441


def mean_sunlit(r_from: float = START_RADIUS, r_to: float = 6.0e5) -> float:
    """Sunlit share of a circular spiral's time, Sun in the orbit plane.

    Thrust is constant, so time runs uniformly in circular speed.  The shadow
    of a sphere covers ``asin(R / r) / pi`` of a circular orbit.

    Args:
        r_from: Starting radius (km).
        r_to: Final radius (km).

    Returns:
        The sunlit share.
    """
    speeds = np.linspace(np.sqrt(_MU / r_to), np.sqrt(_MU / r_from), 20001)
    radii = _MU / speeds**2
    shadow = np.arcsin(np.minimum(1.0, _R_EARTH / radii)) / np.pi
    return float(np.mean(1.0 - shadow))


def twenty_day_orbit() -> Tuple[float, float]:
    """(semi-major axis, apogee radius) of the 20-day orbit at 200 km (km)."""
    period = float(PUFFSAT_CYCLE_ORBIT_PERIOD.to_value(u.s))
    axis = float(np.cbrt(_MU * period**2 / (4.0 * np.pi**2)))
    return axis, 2.0 * axis - START_RADIUS


def circular_spiral() -> Tuple[float, float]:
    """The author's first climb: (spiral, perigee drop) delta-v (km/s).

    Edelbaum's coplanar circle-to-circle ``v_1 - v_2`` out to the 20-day
    orbit's apogee, then a tangential burn there to bring perigee to 200 km.
    """
    axis, apogee = twenty_day_orbit()
    v_circular = np.sqrt(_MU / apogee)
    v_apogee = np.sqrt(_MU * (2.0 / apogee - 1.0 / axis))
    return (
        float(np.sqrt(_MU / START_RADIUS) - v_circular),
        float(v_circular - v_apogee),
    )


@dataclass(frozen=True)
class Pump:
    """A perigee pump's geometry at the reference acceleration.

    Attributes:
        theta: Thrust-arc half-width about perigee (deg).
        handover_axis: Semi-major axis at hand-over (km).
        delta_v: Tug delta-v (km/s).
        duty: Thrusting time over elapsed time.
        perigee: Perigee radius at hand-over (km); the finite arcs raise it.
        perigee_speed: Speed there (km/s).
    """

    theta: float
    handover_axis: float
    delta_v: float
    duty: float
    perigee: float
    perigee_speed: float

    @property
    def departure_burn(self) -> float:
        """The ship's impulsive Oberth burn from perigee to the seed (km/s)."""
        final = np.sqrt(SEED_EXCESS**2 + 2.0 * _MU / self.perigee)
        return float(final - self.perigee_speed)


@lru_cache(maxsize=64)
def perigee_pump(theta: float, handover_axis: float) -> Pump:
    """Raise apogee by thrusting tangentially within ``theta`` of perigee.

    Gauss's equations for tangential thrust, ``da/dt = 2 a^2 v a_t / mu`` and
    ``de/dt = 2 (e + cos nu) a_t / v``, averaged over each orbit's arc and
    stepped several orbits at a time (each step moves ``a`` by 0.2%).  The
    arc is symmetric about perigee, so the line of apsides holds still.

    Args:
        theta: Arc half-width (deg).  180 thrusts all the way round.
        handover_axis: Stop when the semi-major axis reaches this (km).

    Returns:
        The pump.
    """
    mu = _MU * 1.0e9
    acceleration = _REFERENCE_ACCELERATION
    nu = np.linspace(-np.pi, np.pi, _SAMPLES)
    on = np.abs(nu) <= np.radians(theta)
    axis, ecc = START_RADIUS * 1.0e3 * 1.0001, 1.0e-4
    elapsed = delta_v = 0.0
    while axis < handover_axis * 1.0e3:
        semi_latus = axis * (1.0 - ecc * ecc)
        momentum = np.sqrt(mu * semi_latus)
        radius = semi_latus / (1.0 + ecc * np.cos(nu))
        speed = np.sqrt(mu * (2.0 / radius - 1.0 / axis))
        dt_dnu = radius * radius / momentum * on
        arc = float(np.trapezoid(dt_dnu, nu))
        d_axis = float(
            np.trapezoid(2.0 * axis * axis * speed / mu * acceleration * dt_dnu, nu)
        )
        d_ecc = float(
            np.trapezoid(2.0 * (ecc + np.cos(nu)) / speed * acceleration * dt_dnu, nu)
        )
        orbits = max(1, int(0.002 * axis / abs(d_axis)))
        axis += orbits * d_axis
        ecc = min(ecc + orbits * d_ecc, 0.9999)
        elapsed += orbits * 2.0 * np.pi * np.sqrt(axis**3 / mu)
        delta_v += orbits * acceleration * arc
    perigee = axis * (1.0 - ecc)
    return Pump(
        theta=theta,
        handover_axis=handover_axis,
        delta_v=delta_v / 1.0e3,
        duty=float(delta_v / (acceleration * elapsed)),
        perigee=perigee / 1.0e3,
        perigee_speed=float(np.sqrt(mu * (2.0 / perigee - 1.0 / axis))) / 1.0e3,
    )


def window_delay(climb_years: float) -> float:
    """Wait the climb forces: whole synodic periods until the window returns."""
    return float(np.ceil(climb_years / SYNODIC_YEARS - 1e-9) * SYNODIC_YEARS)


@dataclass(frozen=True)
class TugSeed:
    """One tug-lifted seed ship (kg, W, $).

    Attributes:
        pump: The climb.
        climb_years: How long it takes.
        seed: Seed the ship sends.
        argon: Argon the tug spends.
        power: Array rating.
        hardware: Tug hardware mass.
        dollars: Launches, hull and tug hardware.
        credit: The array's sale at L1, discounted to purchase.
        delay: Wait for the window (yr).
    """

    pump: Pump
    climb_years: float
    seed: float
    argon: float
    power: float
    hardware: float
    dollars: float
    credit: float
    delay: float


def _seed_from_perigee(pump: Pump, propellant: float, ship: SeedShip) -> float:
    """Seed the ship sends from the pump's perigee, steered loss charged (kg)."""
    exhaust = VE_METHALOX * u.km / u.s
    flow = float((ship.engines * RAPTOR3_THRUST / exhaust).to_value(u.kg / u.s))
    period = 2.0 * np.pi * np.sqrt(pump.handover_axis**3 / _MU)
    loss = finite_burn_loss(
        pump.departure_burn * u.km / u.s,
        exhaust,
        propellant / flow * u.s,
        steered=True,
        altitude=(pump.perigee - _R_EARTH) * u.km,
        period=period * u.s,
    ).to_value(u.km / u.s)
    ratio = float(np.exp((pump.departure_burn + loss) / VE_METHALOX))
    return propellant / (ratio - 1.0) - float(ship.dry_mass.to_value(u.kg))


def tug_seed(
    pump: Pump,
    climb_years: float,
    boil_off_per_day: float,
    price_per_watt: float,
    ship: SeedShip = STRIPPED_SHIPS[0],
    flight_price: float = FLIGHT_PRICES[BASELINE_FLIGHT],
    hull: float = SHIP_PRICES[0],
) -> TugSeed:
    """Price one ship lifted by the tug.

    The tug lifts the ship with its whole load; the methalox boils off on the
    way, and the ship burns what is left.  Power is sized from the climb's
    total impulse at the log-mean mass, thrusting ``duty x SUNLIT`` of the
    time at ``BELT_DERATING`` of its rating, and solved with the argon by
    fixed-point iteration (the hardware rides too).

    Args:
        pump: The climb.
        climb_years: How long it may take.
        boil_off_per_day: Share of the methalox lost per day.
        price_per_watt: Tug hardware, $/W of rating.
        ship: The seed ship.
        flight_price: Dollars per 100 t flight.
        hull: The expended hull.

    Returns:
        The tug-lifted seed.
    """
    loaded = float(SEED_PROPELLANT.to_value(u.kg))
    burned = loaded * (1.0 - boil_off_per_day) ** (climb_years * 365.25)
    seed = _seed_from_perigee(pump, burned, ship)
    dry = float(ship.dry_mass.to_value(u.kg))
    stack = loaded + dry + max(seed, 0.0)
    spent = 1.0 - float(np.exp(-pump.delta_v / VE_ARGON))
    hardware = argon = power = 0.0
    for _ in range(60):
        argon = spent * (stack + hardware) / (1.0 - spent * (1.0 + ARGON_TANK_FRACTION))
        start = stack + hardware + argon * (1.0 + ARGON_TANK_FRACTION)
        end = start - argon
        mean_mass = (start - end) / np.log(start / end)
        thrust = (
            mean_mass
            * pump.delta_v
            * 1.0e3
            / (pump.duty * SUNLIT * BELT_DERATING * climb_years * _YEAR_S)
        )
        power = thrust * VE_ARGON * 1.0e3 / (2.0 * SEP_THRUSTER_EFFICIENCY)
        hardware = DEFAULT_SPECIFIC_MASS * power / 1.0e3
    lofted = stack + argon * (1.0 + ARGON_TANK_FRACTION) + hardware
    dollars = flight_price / 1.0e5 * lofted + hull + price_per_watt * power
    return TugSeed(
        pump=pump,
        climb_years=climb_years,
        seed=seed,
        argon=argon,
        power=power,
        hardware=hardware,
        dollars=dollars,
        credit=l1_credit(power, price_per_watt, climb_years),
        delay=window_delay(climb_years),
    )


def l1_credit(power: float, price_per_watt: float, climb_years: float) -> float:
    """The array sold at L1 at what a buyer would otherwise pay ($, at purchase).

    Its build price (its share of the hardware, derated to end of life) plus
    shipping its mass there, received :data:`SALE_LAG` after the climb and
    discounted at :data:`RATE`.  Not risked: an array at L1 does not depend
    on the cycle working.

    Args:
        power: Array rating (W).
        price_per_watt: Tug hardware, $/W.
        climb_years: The climb (yr).

    Returns:
        The credit.
    """
    value = (
        ARRAY_PRICE_SHARE * price_per_watt * END_OF_LIFE * power
        + L1_SHIPPING * ARRAY_SPECIFIC_MASS * power / 1.0e3
    )
    return float(value / (1.0 + RATE) ** (climb_years + SALE_LAG))


def worth(tug: TugSeed, credited: bool) -> Tuple[float, float]:
    """(``k``, ``k x 1.1^-dt``): seed per dollar over direct's, then discounted.

    Args:
        tug: The tug-lifted seed.
        credited: Net the array's sale at L1.

    Returns:
        The ratio and its discounted value.
    """
    ship = STRIPPED_SHIPS[0]
    direct = float(seed_payload(ship, SEED_EXCESS * u.km / u.s).payload.to_value(u.kg))
    flights = 13.0 * FLIGHT_PRICES[BASELINE_FLIGHT] + SHIP_PRICES[0]
    net = tug.dollars - (tug.credit if credited else 0.0)
    k = (tug.seed / net) / (direct / flights)
    return float(k), float(k * (1.0 + RATE) ** (-tug.delay))


def best_tugs() -> List[Dict[str, object]]:
    """The best pump on the grid at each price, boil-off and credit."""
    pumps = [
        perigee_pump(t, a) for t, a in itertools.product(ARC_HALF_WIDTHS, HANDOVER_AXES)
    ]
    rows: List[Dict[str, object]] = []
    for boil, price, credited in itertools.product(
        BOIL_OFF_PER_DAY, PRICES_PER_WATT, (False, True)
    ):
        best: Optional[Tuple[float, float, TugSeed]] = None
        for pump, years in itertools.product(pumps, CLIMB_YEARS):
            tug = tug_seed(pump, years, boil, price)
            if tug.seed <= 0.0:
                continue
            k, value = worth(tug, credited)
            if best is None or value > best[1]:
                best = (k, value, tug)
        assert best is not None
        k, value, tug = best
        rows.append(
            {
                "boil %/d": 100.0 * boil,
                "$/W": price,
                "L1 sale": credited,
                "worth": value,
                "k": k,
                "delay yr": tug.delay,
                "theta": tug.pump.theta,
                "a_f km": tug.pump.handover_axis,
                "climb yr": tug.climb_years,
                "seed t": tug.seed / 1e3,
                "MW": tug.power / 1e6,
                "argon t": tug.argon / 1e3,
                "tug SEP km/s": tug.pump.delta_v,
                "ship km/s": tug.pump.departure_burn,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the climbs and the tug's best case at each price.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    spiral, drop = circular_spiral()
    print(
        f"Circular spiral to the 20-day apogee: {spiral:.2f} km/s, perigee drop "
        f"{drop:.2f} km/s (sunlit share {mean_sunlit():.3f})."
    )
    pumps = [
        perigee_pump(t, a) for t, a in itertools.product(ARC_HALF_WIDTHS, HANDOVER_AXES)
    ]
    print("\nPerigee pumps, at 2e-4 m/s^2 (time scales as 1 / acceleration):")
    print(
        tabulate(
            [
                {
                    "theta": p.theta,
                    "a_f km": p.handover_axis,
                    "tug km/s": p.delta_v,
                    "duty": p.duty,
                    "perigee alt km": p.perigee - _R_EARTH,
                    "ship km/s": p.departure_burn,
                }
                for p in pumps
            ],
            headers="keys",
            floatfmt=".3g",
        )
    )
    print(
        "\nBest tug at each price (40 t ship, $5M hull, $150/kg flights), against "
        "direct; worth = k x 1.1^-delay, above 1 the tug pays (ADR 0039's test):"
    )
    print(tabulate(best_tugs(), headers="keys", floatfmt=".3g"))


if __name__ == "__main__":
    main()
