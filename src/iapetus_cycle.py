"""The Iapetus water cycle: a single-use water NTR, its ledger and true cost (S51).

The parent's ask S51 (``docs/iapetus_cycle_ledger_asks_for_aim_repo.md``, carried
from ``Balloon-Pulse-Propulsion`` @ ``ec5995d``), answered in ADR 0078.

One round, 9.5 years:

1. **Out.** Earth hardware is pushed to the eccentric orbit (10.916 km/s at
   200 km, the 10-day orbit) by returning bags; a hydrogen NTP stage burns at
   perigee for Saturn on a 3.0-year transfer.  The seed round starts from LEO.
2. **Saturn.** A methalox stage captures at 150 000 km onto a 20 million km
   apoapsis, raises periapsis there, matches Iapetus and lands the kit
   (the dog-leg, 1.17 years each way).
3. **Load and return.** The water nuclear thermal stage (1300 K, 200 s, one
   560 MW engine per unit) loads ice in low orbit, spirals out of Iapetus,
   drops periapsis at apoapsis, and burns at 150 000 km for Earth: 10.4 km/s
   at Saturn, 51.1 km/s at 200 km after 3.7 years.  The PuffSats (25 kg water
   bags) separate at cutoff; the empty stage brakes and falls into Saturn.
4. **The Earth push.** The bags overtake the next round's ship and strike a
   ship-carried ice plug of ``k`` kg per kg of bag on the Earth bag plate
   (the impact simulation's Result 20; its S52 shaped plug is still unrun).

Everything scales with thrust, so the ledger is kept per 560 MW engine.  The
finite burns are integrated on point masses in 2-D, thrust along velocity:
they reproduce the parent's 1.3/2.8/8.5/20.3/27.2% periapsis losses and its
214/314/554/826/919 t of PuffSat per engine to the tonne (at its old 1%
tankage).

Floats throughout (km, s, kg, km/s, dollars), as :mod:`src.solar_dive_growth`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
from tabulate import tabulate

from src.conic_kernel import elliptic_time_from_periapsis, hyperbolic_tof_seconds
from src.learning_curve import LearningCurve
from src.solar_dive_growth import (
    MISS_FLOOR,
    Gate,
    GatedSchedule,
    miss_fraction,
)

# --------------------------------------------------------------------------
# Bodies and the route
# --------------------------------------------------------------------------

G0 = 9.80665e-3  # km/s^2
_YEAR = 365.25 * 86400.0
_HOUR = 3600.0

MU_SUN = 1.32712440018e11
AU = 1.495978707e8
SATURN_A = 9.5388 * AU
SATURN_YEAR = 29.457  # years

MU_SATURN = 37_931_187.0
#: The capture and departure periapsis: outside the main rings (parent).
SATURN_PERIAPSIS = 150_000.0
#: The dog-leg ellipse's apoapsis (parent).
DOGLEG_APOAPSIS = 20.0e6

MU_IAPETUS = 120.5
IAPETUS_RADIUS = 734.5
IAPETUS_A = 3_560_820.0
#: The loading orbit's *radius* (km).  The parent's 0.74 km/s impulsive escape
#: and its 1.37 km/s very-low-thrust limit both come from an 800 km radius
#: (65 km up), not 800 km altitude, and so do its spiral costs.
LOADING_RADIUS = 800.0

MU_EARTH = 398_600.4418
R_EARTH = 6378.137
PUSH_ALTITUDE = 200.0
#: The eccentric orbit's perigee speed (parent): a 10.0-day orbit.
ECCENTRIC_SPEED = 10.916
LEO_ALTITUDE = 200.0

#: Saturn departure excess and the Earth closing speed it buys (parent;
#: :func:`return_excess_for` reproduces 51.0 km/s at 10.4 in 3.70 years).
RETURN_EXCESS = 10.4
CLOSING_SPEED = 51.1
RETURN_YEARS = 3.7
#: The empty stage's brake to bound plus 0.2 km/s at apoapsis into Saturn.
DISPOSAL_DV = 2.89

#: Earth departure from the eccentric orbit's perigee for the 3.0-year
#: transfer (parent); from LEO the same excess costs 8.03 km/s.
EARTH_DEPARTURE_DV = 4.90
#: Per kg of an aerocapture aeroshell (assumption: the mining kit's $1000/kg).
AEROSHELL_PRICE = 1000.0
#: Arrival excess at Saturn: the parent's ~2.5 km/s capture implies 10.7.
SATURN_ARRIVAL_EXCESS = 10.7
CAPTURE_MARGIN = 0.02
#: The landing from a matched orbit (parent; 0.57 is the bare escape speed).
IAPETUS_LANDING = 0.65

OUT_YEARS = 3.0
DOGLEG_YEARS = 1.17
STAY_YEARS = 0.46
ROUND_YEARS = OUT_YEARS + 2.0 * DOGLEG_YEARS + STAY_YEARS + RETURN_YEARS
#: Earth-Saturn synodic period: a departure window.
SYNODIC_YEARS = 1.0 / (1.0 - 1.0 / SATURN_YEAR)
#: The horizon the whole-round count fits rounds into (parent: 20 years).
ROUND_HORIZON = 20.0


def _vis_viva(mu: float, r: float, a: float) -> float:
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def dogleg_dvs() -> Dict[str, float]:
    """The Saturn-side impulses of the dog-leg (km/s), from the conics.

    Returns:
        ``capture`` (from :data:`SATURN_ARRIVAL_EXCESS`, with its margin),
        ``raise`` (periapsis to Iapetus at apoapsis), ``match`` (onto
        Iapetus's circular speed), ``land``, and on the way home ``escape``
        (Iapetus's excess needed, as a speed), ``drop`` (periapsis back to
        150 000 km) and ``departure`` (the impulsive periapsis burn).
    """
    rp, ra = SATURN_PERIAPSIS, DOGLEG_APOAPSIS
    a_low = 0.5 * (rp + ra)
    a_high = 0.5 * (IAPETUS_A + ra)
    vp_low = _vis_viva(MU_SATURN, rp, a_low)
    capture = math.sqrt(SATURN_ARRIVAL_EXCESS**2 + 2.0 * MU_SATURN / rp) - vp_low
    shift = _vis_viva(MU_SATURN, ra, a_high) - _vis_viva(MU_SATURN, ra, a_low)
    vp_high = _vis_viva(MU_SATURN, IAPETUS_A, a_high)
    match = vp_high - math.sqrt(MU_SATURN / IAPETUS_A)
    departure = math.sqrt(RETURN_EXCESS**2 + 2.0 * MU_SATURN / rp) - vp_low
    return {
        "capture": capture * (1.0 + CAPTURE_MARGIN),
        "raise": shift,
        "match": match,
        "land": IAPETUS_LANDING,
        "escape": match,
        "drop": shift,
        "departure": departure,
    }


# --------------------------------------------------------------------------
# The heliocentric return: Saturn excess against Earth closing speed
# --------------------------------------------------------------------------


def earth_arrival(saturn_excess: float, angle: float) -> Optional[Tuple[float, float]]:
    """Speed at 200 km and years of flight, Saturn to 1 AU, coplanar circles.

    Args:
        saturn_excess: Departure excess at Saturn (km/s).
        angle: Its direction from Saturn's velocity, prograde (rad).

    Returns:
        ``(speed at 200 km km/s, years)``, or None if 1 AU is not reached
        inbound.  Earth's phase is taken as free (a departure every window).
    """
    v_s = math.sqrt(MU_SUN / SATURN_A)
    v_e = math.sqrt(MU_SUN / AU)
    vt = v_s + saturn_excess * math.cos(angle)
    vr = saturn_excess * math.sin(angle)
    h = SATURN_A * vt
    energy = 0.5 * (vt * vt + vr * vr) - MU_SUN / SATURN_A
    v1_sq = 2.0 * (energy + MU_SUN / AU)
    vt1 = h / AU
    if v1_sq < vt1 * vt1:
        return None
    vr1 = -math.sqrt(v1_sq - vt1 * vt1)
    excess = math.hypot(vt1 - v_e, vr1)
    speed = math.sqrt(excess**2 + 2.0 * MU_EARTH / (R_EARTH + PUSH_ALTITUDE))
    p = h * h / MU_SUN
    ecc = math.sqrt(max(1.0 + 2.0 * energy * h * h / MU_SUN**2, 0.0))
    if ecc < 1e-9:
        return None

    def nu(r: float, outbound: bool) -> float:
        c = max(-1.0, min(1.0, (p / r - 1.0) / ecc))
        n = math.acos(c)
        return n if outbound else 2.0 * math.pi - n

    if energy < 0.0:
        a = -MU_SUN / (2.0 * energy)
        period = 2.0 * math.pi * math.sqrt(a**3 / MU_SUN)
        t0 = elliptic_time_from_periapsis(MU_SUN, a, ecc, nu(SATURN_A, vr > 0.0))
        t1 = elliptic_time_from_periapsis(MU_SUN, a, ecc, nu(AU, False))
        tof = (t1 - t0) % period
    else:
        if vr > 0.0:
            return None
        a_abs = MU_SUN / (2.0 * energy)
        tof = hyperbolic_tof_seconds(
            MU_SUN,
            a_abs,
            ecc,
            math.acos(max(-1.0, min(1.0, (p / SATURN_A - 1.0) / ecc))),
        ) - hyperbolic_tof_seconds(
            MU_SUN, a_abs, ecc, math.acos(max(-1.0, min(1.0, (p / AU - 1.0) / ecc)))
        )
    return speed, tof / _YEAR


def fastest_arrival(saturn_excess: float, max_years: float = RETURN_YEARS) -> float:
    """The highest speed at 200 km a Saturn excess buys within ``max_years``."""
    best = 0.0
    for deg in np.arange(90.0, 270.0, 0.05):
        hit = earth_arrival(saturn_excess, math.radians(float(deg)))
        if hit is not None and hit[1] <= max_years and hit[0] > best:
            best = hit[0]
    return best


def return_excess_for(closing: float, max_years: float = RETURN_YEARS) -> float:
    """The Saturn excess that reaches ``closing`` km/s at 200 km in time (km/s)."""
    return float(
        brentq(lambda v: fastest_arrival(v, max_years) - closing, 8.0, 16.0, xtol=1e-3)
    )


# --------------------------------------------------------------------------
# The water engine and its barrier
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterEngine:
    """A water NTR at one exhaust temperature, at its reactor power.

    Attributes:
        label: Row name.
        exhaust: Exhaust temperature (K).
        isp: Specific impulse on water (s).
        heat: Heat per kilogram from 100 K ice to the exhaust (J/kg).
        power: Reactor power (W).
    """

    label: str
    exhaust: float
    isp: float
    heat: float
    power: float = 560.0e6

    @property
    def ve(self) -> float:
        """Exhaust speed (km/s)."""
        return self.isp * G0

    @property
    def mass_flow(self) -> float:
        """Water through the core (kg/s)."""
        return self.power / self.heat

    @property
    def thrust(self) -> float:
        """Thrust (kN): power over heat per kg times exhaust speed."""
        return self.mass_flow * self.ve

    def at_power(self, power: float) -> "WaterEngine":
        """The same engine at another reactor power."""
        return replace(self, power=power)


#: The base: the barrier holds at a ~1400 K wall at ~1 MPa (parent).
WATER_1300K = WaterEngine("1300 K", 1300.0, 200.0, 5.35e6)
#: Needs a barrier not yet demonstrated in pure high-pressure steam.
WATER_1800K = WaterEngine(
    "1800 K (needs undemonstrated barrier)", 1800.0, 235.0, 6.55e6
)
#: NASA's Mars engine as the parent prices it: 111 kN on hydrogen.
NERVA_CLASS_THRUST = 111.0


def sic_recession(
    wall: float, pressure_atm: float, speed: float, steam_fraction: float = 1.0
) -> float:
    """SiC recession in steam (micrometres an hour), the parent's extrapolation.

    Smialek, Robinson, Opila, Fox and Jacobson (NASA/TP-1999-208696), fitted
    at 1473-1723 K, 4-15 atm, 18-25 m/s in ~10% steam:
    ``k = 2.04 exp(-108 kJ/mol / RT) P^1.5 v^0.5`` mg/cm^2/h of SiC.  Its
    boundary-layer basis scales as ``P_H2O^2 P^-0.5``, so the rig's 10%
    steam is carried to ``steam_fraction`` as its square; SiC at 3.21 g/cm^3.

    Args:
        wall: Hot-wall temperature (K).
        pressure_atm: Total pressure (atm).
        speed: Gas speed past the wall (m/s).
        steam_fraction: Steam's share of the gas (1 for the water NTR).

    Returns:
        Recession rate (um/h).
    """
    rig = 2.04 * math.exp(-108.0e3 / (8.314 * wall)) * pressure_atm**1.5 * speed**0.5
    mg_cm2_h = rig * (steam_fraction / 0.10) ** 2
    return float(mg_cm2_h / 3.21 * 10.0)


# --------------------------------------------------------------------------
# Finite burns: one integrator for every central body
# --------------------------------------------------------------------------


def _ellipse_state(mu: float, rp: float, ra: float, before: float) -> List[float]:
    """Position and velocity ``before`` seconds ahead of periapsis."""
    a = 0.5 * (rp + ra)
    ecc = (ra - rp) / (ra + rp)
    mean = -math.sqrt(mu / a**3) * before
    ecc_anom = mean
    for _ in range(60):
        ecc_anom -= (ecc_anom - ecc * math.sin(ecc_anom) - mean) / (
            1.0 - ecc * math.cos(ecc_anom)
        )
    r = a * (1.0 - ecc * math.cos(ecc_anom))
    f = math.sqrt(mu * a) / r
    return [
        a * (math.cos(ecc_anom) - ecc),
        a * math.sqrt(1.0 - ecc * ecc) * math.sin(ecc_anom),
        -f * math.sin(ecc_anom),
        f * math.sqrt(1.0 - ecc * ecc) * math.cos(ecc_anom),
    ]


def _tangential_burn(
    mu: float, state: List[float], a0: float, ve: float, seconds: float
) -> float:
    """Specific energy after a constant-thrust burn along velocity (km^2/s^2)."""

    def rhs(t: float, s: np.ndarray) -> List[float]:
        x, y, vx, vy = s
        r = math.hypot(x, y)
        v = math.hypot(vx, vy)
        acc = a0 / (1.0 - a0 / ve * t)
        return [vx, vy, -mu * x / r**3 + acc * vx / v, -mu * y / r**3 + acc * vy / v]

    sol = solve_ivp(rhs, (0.0, seconds), state, rtol=1e-10, atol=1e-9)
    x, y, vx, vy = sol.y[:, -1]
    return float(0.5 * (vx * vx + vy * vy) - mu / math.hypot(x, y))


def _best_energy(
    mu: float, rp: float, ra: float, a0: float, ve: float, seconds: float
) -> float:
    """The burn's final energy with its start before periapsis optimised."""
    if ra <= rp:  # a circle: every start is the same
        return _tangential_burn(mu, _ellipse_state(mu, rp, rp, 0.0), a0, ve, seconds)
    found = minimize_scalar(
        lambda before: -_tangential_burn(
            mu, _ellipse_state(mu, rp, ra, before), a0, ve, seconds
        ),
        bounds=(0.0, seconds),
        method="bounded",
        options={"xatol": 1.0},
    )
    return float(-found.fun)


@lru_cache(maxsize=None)
def periapsis_acceleration(hours: float, ve: float, excess: float) -> float:
    """Start acceleration (km/s^2) at which an ``hours`` burn at 150 000 km
    reaches ``excess``, the start optimised (2-D, Saturn point mass)."""
    seconds = hours * _HOUR
    target = 0.5 * excess**2
    return float(
        brentq(
            lambda a0: _best_energy(
                MU_SATURN, SATURN_PERIAPSIS, DOGLEG_APOAPSIS, a0, ve, seconds
            )
            - target,
            1e-7,
            min(5e-3, 0.999 * ve / seconds),
            xtol=1e-11,
        )
    )


def periapsis_burn(hours: float, ve: float, excess: float) -> Tuple[float, float]:
    """The finite periapsis burn's mass ratio and its loss over the impulse.

    Returns:
        ``(end mass / start mass, fractional loss)``.
    """
    a0 = periapsis_acceleration(hours, ve, excess)
    kept = 1.0 - a0 / ve * hours * _HOUR
    impulsive = math.sqrt(excess**2 + 2.0 * MU_SATURN / SATURN_PERIAPSIS) - _vis_viva(
        MU_SATURN, SATURN_PERIAPSIS, 0.5 * (SATURN_PERIAPSIS + DOGLEG_APOAPSIS)
    )
    return kept, -ve * math.log(kept) / impulsive - 1.0


@lru_cache(maxsize=None)
def spiral_escape(a0: float, ve: float, excess: float) -> Tuple[float, float]:
    """Iapetus escape from :data:`LOADING_RADIUS`, thrust along velocity.

    Args:
        a0: Start acceleration (km/s^2).
        ve: Exhaust speed (km/s).
        excess: Iapetus excess wanted (km/s).

    Returns:
        ``(delta-v km/s, burn seconds)``.
    """
    r0 = LOADING_RADIUS
    target = 0.5 * excess**2

    def rhs(t: float, s: np.ndarray) -> List[float]:
        x, y, vx, vy = s
        r = math.hypot(x, y)
        v = math.hypot(vx, vy)
        acc = a0 / (1.0 - a0 / ve * t)
        return [
            vx,
            vy,
            -MU_IAPETUS * x / r**3 + acc * vx / v,
            -MU_IAPETUS * y / r**3 + acc * vy / v,
        ]

    def escaped(t: float, s: np.ndarray) -> float:
        x, y, vx, vy = s
        return float(0.5 * (vx * vx + vy * vy) - MU_IAPETUS / math.hypot(x, y) - target)

    escaped.terminal = True  # type: ignore[attr-defined]
    sol = solve_ivp(
        rhs,
        (0.0, 0.99 * ve / a0),
        [r0, 0.0, 0.0, math.sqrt(MU_IAPETUS / r0)],
        events=escaped,
        rtol=1e-10,
        atol=1e-10,
        max_step=600.0,
    )
    seconds = float(sol.t_events[0][0])
    return -ve * math.log(1.0 - a0 / ve * seconds), seconds


def impulsive_escape(excess: float, radius: float = LOADING_RADIUS) -> float:
    """Iapetus escape to ``excess`` as one impulse from a circular orbit (km/s)."""
    return math.sqrt(excess**2 + 2.0 * MU_IAPETUS / radius) - math.sqrt(
        MU_IAPETUS / radius
    )


@lru_cache(maxsize=None)
def earth_departure_acceleration(
    from_leo: bool, isp: float, loss: float, earth_dv: float = EARTH_DEPARTURE_DV
) -> float:
    """Start acceleration (km/s^2) at which the Earth departure loses ``loss``.

    From the eccentric orbit's 200 km perigee, or from a 200 km circular LEO,
    to the excess the 3.0-year transfer needs, thrust along velocity, its
    start optimised (2-D, Earth point mass).  The parent sized its outbound
    stage for at most 5%; the engines are dear, so the ledger picks the loss
    (:func:`outbound`).  ``earth_dv`` is the impulsive burn from the
    eccentric orbit (4.90 coplanar; the real arcs' median is larger).
    """
    rp = R_EARTH + PUSH_ALTITUDE
    vp = ECCENTRIC_SPEED
    excess_sq = (vp + earth_dv) ** 2 - 2.0 * MU_EARTH / rp
    if from_leo:
        ra = rp
        dv = math.sqrt(excess_sq + 2.0 * MU_EARTH / rp) - math.sqrt(MU_EARTH / rp)
    else:
        a = 1.0 / (2.0 / rp - vp * vp / MU_EARTH)
        ra = 2.0 * a - rp
        dv = earth_dv
    ve = isp * G0
    finite = dv * (1.0 + loss)

    def short(a0: float) -> float:
        seconds = ve / a0 * (1.0 - math.exp(-finite / ve))
        return _best_energy(MU_EARTH, rp, ra, a0, ve, seconds) - 0.5 * excess_sq

    return float(brentq(short, 1e-6, 0.1, xtol=1e-10))


def earth_departure_dv(from_leo: bool, earth_dv: float = EARTH_DEPARTURE_DV) -> float:
    """The impulsive Earth departure (km/s): ``earth_dv`` from the eccentric
    orbit, or the same excess from LEO."""
    if not from_leo:
        return earth_dv
    rp = R_EARTH + LEO_ALTITUDE
    excess_sq = (ECCENTRIC_SPEED + earth_dv) ** 2 - 2.0 * MU_EARTH / (
        R_EARTH + PUSH_ALTITUDE
    )
    return math.sqrt(excess_sq + 2.0 * MU_EARTH / rp) - math.sqrt(MU_EARTH / rp)


@lru_cache(maxsize=None)
def seed_ship_price(dear: bool = False, earth_dv: float = EARTH_DEPARTURE_DV) -> float:
    """Dollars per kg a refuelled, expended Starship sends to Saturn's excess.

    The repository's seed ship (:func:`src.seed_cost.seed_price_range`): one
    burn from LEO, steered finite loss charged, at the excess the 3.0-year
    transfer needs (11.36 km/s); cheap end ($5M hull, $150/kg flights) or
    dear.
    """
    from astropy import units as u

    from src.seed_cost import seed_price_range

    rp = R_EARTH + PUSH_ALTITUDE
    excess = math.sqrt((ECCENTRIC_SPEED + earth_dv) ** 2 - 2.0 * MU_EARTH / rp)
    prices = seed_price_range(excess * u.km / u.s)
    return float(prices.high if dear else prices.low)


# --------------------------------------------------------------------------
# The return stage, per engine
# --------------------------------------------------------------------------

#: The GE-711 cermet pair (NASA GRC, NTRS 20150007681): 266 MW at 1.5x less
#: mass than 564 MW's 2675 kg.  A fixed part and a per-MW part, scaled to
#: each row's mass at 560 MW.
_GE711_FIXED = 2675.0 / 1.5 - 266.0 * (2675.0 - 2675.0 / 1.5) / (564.0 - 266.0)
_GE711_PER_MW = (2675.0 - 2675.0 / 1.5) / (564.0 - 266.0)


def engine_mass(at_560: float, power: float) -> float:
    """An engine's mass at ``power`` W, on the GE-711 pair's shape (kg)."""
    mw = power / 1e6
    return (
        at_560
        * (_GE711_FIXED + _GE711_PER_MW * mw)
        / (_GE711_FIXED + _GE711_PER_MW * 560.0)
    )


@dataclass(frozen=True)
class StageInputs:
    """The per-line mass rows (parent's base / optimistic / pessimistic).

    Attributes:
        label: Row name.
        engine: Engine mass at 560 MW (kg): 3.23 t base (NASA's Mars engine
            as the parent prices it), 2.675 t optimistic (GE-711 cermet, 93%
            enriched), 3.544 t pessimistic (a 19.75% LEU cermet at 111 kN,
            "Design comparison of nuclear thermal rocket concepts", replacing
            the parent's unsourced 4.5 t).
        tankage: Walls holding the water, per kg of propellant.
        skins: PuffSat skins and electronics, per kg of PuffSat (Earth-sourced).
        kit: Mining and launch kit, kg per kg of ice loaded.
        redundancy: ``none``, ``second set`` (a second full engine, carried
            dead) or ``brake reactor`` (a brake-only reactor at
            ``brake_share`` of the main power, carried dead).
        brake_share: The brake-only reactor's power, of the main engine's
            (assumption: the empty stage is ~2% of the departing stack).
    """

    label: str = "base"
    engine: float = 3230.0
    tankage: float = 0.002
    skins: float = 0.05
    kit: float = 0.3e-3
    redundancy: str = "brake reactor"
    brake_share: float = 0.10


BASE_INPUTS = StageInputs()
OPTIMISTIC_INPUTS = StageInputs("optimistic", 2675.0, 0.001, 0.03, 0.1e-3)
PESSIMISTIC_INPUTS = StageInputs("pessimistic", 3544.0, 0.01, 0.10, 1.0e-3)
#: The parent's sizing behind its 826 t: the old 1% tankage, no redundancy.
PARENT_TABLE_INPUTS = StageInputs(
    "parent table (1% tank)", tankage=0.01, redundancy="none"
)


@dataclass(frozen=True)
class ReturnStage:
    """One engine's return stage, by mass (kg) and time.

    Attributes:
        hours: Periapsis burn length.
        engine: The water engine.
        inputs: The mass rows.
        start: Loaded stack in low Iapetus orbit.
        escape_dv: The finite Iapetus escape (km/s).
        escape_hours: Its burn.
        start_acceleration: At the escape's start (m/s^2).
        periapsis_loss: The periapsis burn's finite loss over the impulse.
        propellant: Water burned, disposal included.
        disposal: Of which the disposal brake.
        tank: Tank walls.
        dead: Redundant hardware carried.
        puffsat: PuffSats released at cutoff.
        ice: Ice loaded (propellant and the PuffSats' water).
        kit: Mining and launch kit.
        skins: PuffSat skins and electronics.
        run_hours: Engine run, disposal included.
    """

    hours: float
    engine: WaterEngine
    inputs: StageInputs
    start: float
    escape_dv: float
    escape_hours: float
    start_acceleration: float
    periapsis_loss: float
    propellant: float
    disposal: float
    tank: float
    dead: float
    puffsat: float
    ice: float
    kit: float
    skins: float
    run_hours: float

    @property
    def engine_mass(self) -> float:
        """The engine (kg)."""
        return engine_mass(self.inputs.engine, self.engine.power)

    @property
    def earth_kg(self) -> float:
        """Earth-sourced mass landed at Iapetus for this stage (kg)."""
        return self.engine_mass + self.dead + self.tank + self.kit + self.skins

    @property
    def puffsat_per_earth_kg(self) -> float:
        """PuffSat per Earth-sourced kilogram at Iapetus."""
        return self.puffsat / self.earth_kg


def redundant_mass(inputs: StageInputs, engine: WaterEngine) -> float:
    """The redundant hardware carried dead (kg)."""
    if inputs.redundancy == "none":
        return 0.0
    if inputs.redundancy == "second set":
        return engine_mass(inputs.engine, engine.power)
    if inputs.redundancy == "brake reactor":
        return engine_mass(inputs.engine, engine.power * inputs.brake_share)
    raise ValueError(f"unknown redundancy {inputs.redundancy!r}")


def return_stage(
    hours: float = 12.0,
    engine: WaterEngine = WATER_1300K,
    inputs: StageInputs = BASE_INPUTS,
    excess: float = RETURN_EXCESS,
) -> ReturnStage:
    """Close one engine's stage backwards from the periapsis burn.

    The thrust and the periapsis burn's length set the stack there; the
    finite Iapetus spiral (its start acceleration found with the stack) and
    the impulsive apoapsis drop set the loaded stack; the empty stage's
    disposal brake, the tank walls (a share of every kilogram burned) and the
    dead redundant hardware come out of what remains, and the rest is PuffSat.

    Args:
        hours: Periapsis burn length (the burn policy).
        engine: The water engine.
        inputs: The mass rows.
        excess: Saturn departure excess (km/s).

    Returns:
        The stage.
    """
    ve = engine.ve
    thrust = engine.thrust  # kN is kg km/s^2
    dvs = dogleg_dvs()
    a_peri = periapsis_acceleration(hours, ve, excess)
    m_peri = thrust / a_peri
    kept, loss = periapsis_burn(hours, ve, excess)
    m_cut = m_peri * kept
    drop = math.exp(dvs["drop"] / ve)

    def loaded(m_start: float) -> float:
        dv, _ = spiral_escape(thrust / m_start, ve, dvs["escape"])
        return m_start * math.exp(-dv / ve) / drop - m_peri

    m_start = float(brentq(loaded, m_peri, 20.0 * m_peri, xtol=1e-3))
    escape_dv, escape_s = spiral_escape(thrust / m_start, ve, dvs["escape"])
    dead = redundant_mass(inputs, engine)
    eng = engine_mass(inputs.engine, engine.power)
    rd = math.exp(DISPOSAL_DV / ve) - 1.0
    tau = inputs.tankage
    disposal = (eng + dead + tau * (m_start - m_cut)) * rd / (1.0 - tau * rd)
    propellant = m_start - m_cut + disposal
    tank = tau * propellant
    puffsat = m_cut - eng - dead - tank - disposal
    if puffsat <= 0.0:
        raise ValueError("the stage carries no PuffSat")
    ice = propellant + puffsat
    return ReturnStage(
        hours=hours,
        engine=engine,
        inputs=inputs,
        start=m_start,
        escape_dv=escape_dv,
        escape_hours=escape_s / _HOUR,
        start_acceleration=thrust / m_start * 1e3,  # m/s^2
        periapsis_loss=loss,
        propellant=propellant,
        disposal=disposal,
        tank=tank,
        dead=dead,
        puffsat=puffsat,
        ice=ice,
        kit=inputs.kit * ice,
        skins=inputs.skins * puffsat,
        run_hours=propellant / engine.mass_flow / _HOUR,
    )


# --------------------------------------------------------------------------
# The outbound stack
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OutboundBook:
    """The outbound stages (assumptions marked).

    Attributes:
        earth_stage: ``hydrogen NTP`` (the parent's) or ``methalox`` for the
            Earth departure.
        methalox_isp: Methalox (s): the ask's 370 s base.
        methalox_dry: Methalox stage dry mass, engines included, per kg of
            propellant (the ask's 8-12%, 10% base).
        methalox_loss: The methalox Earth departure's finite loss (high
            thrust: assumption).
        ntp_isp: Hydrogen NTP (s): NASA's ~900 s class.
        lh2_tank: LH2 tank and insulation per kg of hydrogen (assumption).
        ntp_engine: Hydrogen engine mass (kg): the same NASA Mars engine.
        ntp_thrust: Its thrust (kN).
        losses: The hydrogen departure's finite losses the ledger chooses
            among (the parent's 5% first).
    """

    earth_stage: str = "hydrogen NTP"
    methalox_isp: float = 370.0
    methalox_dry: float = 0.10
    methalox_loss: float = 0.02
    ntp_isp: float = 900.0
    lh2_tank: float = 0.15
    ntp_engine: float = 3230.0
    ntp_thrust: float = NERVA_CLASS_THRUST
    losses: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)


OUTBOUND_BOOK = OutboundBook()
METHALOX_DEPARTURE = replace(OUTBOUND_BOOK, earth_stage="methalox")


@dataclass(frozen=True)
class Outbound:
    """Per kilogram landed at Iapetus, what leaves the departure orbit (kg).

    Attributes:
        from_leo: Departing from LEO (the seed) instead of the eccentric orbit.
        earth_stage: ``hydrogen NTP`` or ``methalox``.
        loss: The Earth departure's finite loss.
        saturn_dv: The Saturn-side methalox stage's delta-v (km/s).
        methalox: Saturn-side methalox.
        methalox_dry: Its stage.
        earth_propellant: The Earth departure's propellant.
        earth_dry: Its tanks (LH2) or stage (methalox).
        ntp_engines: Hydrogen engines (count, per kg landed).
        ntp_mass: Their mass.
        earth_dv: The impulsive burn from the eccentric orbit it was sized for.
        aeroshell: Aeroshell shed after an aerocapture at Saturn
            (:mod:`src.saturn_aerocapture`), zero for a capture burn.
    """

    from_leo: bool
    earth_stage: str
    loss: float
    saturn_dv: float
    methalox: float
    methalox_dry: float
    earth_propellant: float
    earth_dry: float
    ntp_engines: float
    ntp_mass: float
    earth_dv: float = EARTH_DEPARTURE_DV
    aeroshell: float = 0.0

    @property
    def ratio(self) -> float:
        """Departing mass per kg landed."""
        return (
            1.0
            + self.methalox
            + self.methalox_dry
            + self.aeroshell
            + self.earth_propellant
            + self.earth_dry
            + self.ntp_mass
        )


def _stage_on(payload: float, dv: float, isp: float, dry: float) -> Tuple[float, float]:
    """Propellant and dry mass of a stage pushing ``payload`` through ``dv``."""
    r = math.exp(dv / (isp * G0))
    start = payload / (1.0 - (1.0 + dry) * (1.0 - 1.0 / r))
    propellant = (start - payload) / (1.0 + dry)
    return propellant, dry * propellant


def outbound_at(
    from_leo: bool,
    book: OutboundBook,
    loss: Optional[float] = None,
    saturn_dv: Optional[float] = None,
    earth_dv: float = EARTH_DEPARTURE_DV,
    aeroshell: float = 0.0,
) -> Outbound:
    """The outbound stack per kilogram landed, at one Earth departure loss.

    Saturn side: one methalox stage flies the capture (with its 2%), the
    periapsis raise, the match and the landing, and lands with the kit.
    With ``aeroshell`` (a share of entry mass) the capture is flown through
    Saturn's atmosphere instead: ``saturn_dv`` is then only the burns after
    the pass, and the aeroshell rides out on top of the stage.
    Earth side: a hydrogen NTP stage with engines enough to hold ``loss``
    (:func:`earth_departure_acceleration`), or a methalox stage.
    """
    if saturn_dv is None:
        d = dogleg_dvs()
        saturn_dv = d["capture"] + d["raise"] + d["match"] + d["land"]
    methalox, dry = _stage_on(1.0, saturn_dv, book.methalox_isp, book.methalox_dry)
    stack = (1.0 + methalox + dry) / (1.0 - aeroshell)
    shell = stack - (1.0 + methalox + dry)
    dv = earth_departure_dv(from_leo, earth_dv)
    if book.earth_stage == "methalox":
        loss = book.methalox_loss if loss is None else loss
        if from_leo:
            # The seed: one burn from LEO on a refuelled, expended Starship
            # (src.seed_cost), priced per kg it sends (:func:`seed_ship_price`).
            return Outbound(
                from_leo,
                "seed ship",
                0.0,
                saturn_dv,
                methalox,
                dry,
                0.0,
                0.0,
                0.0,
                0.0,
                earth_dv,
                shell,
            )
        # Recurring: the bags push the ship to the eccentric orbit, then one
        # methalox burn at perigee.
        prop, edry = _stage_on(
            stack, dv * (1.0 + loss), book.methalox_isp, book.methalox_dry
        )
        return Outbound(
            from_leo,
            "methalox",
            loss,
            saturn_dv,
            methalox,
            dry,
            prop,
            edry,
            0.0,
            0.0,
            earth_dv,
            shell,
        )
    if book.earth_stage != "hydrogen NTP":
        raise ValueError(f"unknown Earth stage {book.earth_stage!r}")
    loss = book.losses[0] if loss is None else loss
    a0 = earth_departure_acceleration(from_leo, book.ntp_isp, loss, earth_dv)
    burned = 1.0 - math.exp(-dv * (1.0 + loss) / (book.ntp_isp * G0))
    engines_per_kg = a0 / book.ntp_thrust  # thrust kN is kg km/s^2
    m0 = stack / (
        1.0 - burned * (1.0 + book.lh2_tank) - engines_per_kg * book.ntp_engine
    )
    if m0 <= 0.0:
        raise ValueError("the hydrogen stage does not close")
    hydrogen = burned * m0
    return Outbound(
        from_leo=from_leo,
        earth_stage="hydrogen NTP",
        loss=loss,
        saturn_dv=saturn_dv,
        methalox=methalox,
        methalox_dry=dry,
        earth_propellant=hydrogen,
        earth_dry=book.lh2_tank * hydrogen,
        ntp_engines=engines_per_kg * m0,
        ntp_mass=engines_per_kg * m0 * book.ntp_engine,
        earth_dv=earth_dv,
        aeroshell=shell,
    )


def outbound(
    from_leo: bool = False,
    book: OutboundBook = OUTBOUND_BOOK,
    per_kg: Optional[float] = None,
    engine: float = 60.0e6,
    hydrogen: float = 6.0,
    hardware: float = 500.0,
    saturn_dv: Optional[float] = None,
    earth_dv: float = EARTH_DEPARTURE_DV,
    aeroshell: float = 0.0,
) -> Outbound:
    """The outbound stack, its Earth departure loss chosen on price.

    For the hydrogen stage the loss is picked from ``book.losses`` to
    minimise, per kg landed, the engines, hydrogen and tanks plus ``per_kg``
    dollars on every kilogram it adds at the departure orbit (the push's
    cost there); with ``per_kg`` None the parent's 5% is kept.
    """
    if book.earth_stage != "hydrogen NTP" or per_kg is None:
        return outbound_at(
            from_leo, book, saturn_dv=saturn_dv, earth_dv=earth_dv, aeroshell=aeroshell
        )
    best: Optional[Tuple[float, Outbound]] = None
    for loss in book.losses:
        try:
            o = outbound_at(from_leo, book, loss, saturn_dv, earth_dv, aeroshell)
        except ValueError:
            continue
        dollars = (
            engine * o.ntp_engines
            + hydrogen * o.earth_propellant
            + hardware * o.earth_dry
            + AEROSHELL_PRICE * o.aeroshell
            + per_kg * o.ratio
        )
        if best is None or dollars < best[0]:
            best = (dollars, o)
    if best is None:
        raise ValueError("no hydrogen stage closes")
    return best[1]


# --------------------------------------------------------------------------
# The Earth push (sim Result 20; S52 pending)
# --------------------------------------------------------------------------

#: Leverage, kg pushed to 10.916 km/s per kg of bag at 51.1 km/s, ice plug,
#: 2 F paraboloid: (lossless rebound, frozen-rebound edge).  Sim Result 20
#: (puffsat_impact_simulation @ 1d7cedf), as S51 carries it.
LEVERAGE: Dict[str, Dict[float, Tuple[float, float]]] = {
    "compact": {
        4.0: (8.12, 7.26),
        8.5: (7.20, 6.21),
        15.0: (4.52, 3.87),
        20.0: (3.28, 2.72),
    },
    "shaped": {
        2.0: (7.89, 7.11),
        4.0: (8.42, 7.53),
        8.5: (8.83, 7.67),
        15.0: (8.37, 7.35),
        20.0: (8.06, 7.00),
    },
}
#: The column labels: the compact plug is simulated; the shaped plug's row is
#: the point-fireball ceiling, unrun until the sim's S52.
COLUMN_LABELS = {
    "compact": "compact plug (simulated)",
    "shaped": "shaped plug, point-fireball ceiling (unrun, S52)",
}
#: A backup pulse: the gun-spread bag misses the plug, J/mv 1.74 (Result 20).
BACKUP_B = 1.74
#: The liner's eroded graphite carried as charge: 7.20 -> 7.06x at k = 8.5.
LINER_LEVERAGE = 7.06 / 7.20
#: The parent's bare plate (eq:PuffSat_ratio, f = 0.8): reference only.
BARE_PLATE = 6.7
BAG_MASS = 25.0  # kg
#: Ship and plate at 1 g and 2 Hz on the 8 m spring (Result 20 item 4), t.
PLATE_TABLE = {
    "compact": {4.0: (625.0, 26.4), 8.5: (683.0, 28.8)},
    "shaped": {8.5: (758.0, 32.0)},
}
#: The springs' nitrogen bags and bladder, per kg of plate (ADR 0077: 5.4 t on 63 t).
SPRING_PER_PLATE = 5.4 / 63.0
#: Leverage at faster arrivals, over 51.1 km/s, measured at k = 8.5 only
#: (compact 9.34x / 11.26x, 2.5 m disk 10.74x / 12.88x at 60 / 68 km/s).
SPEED_GAIN = {
    "compact": {51.1: 1.0, 60.0: 9.34 / 7.20, 68.0: 11.26 / 7.20},
    "shaped": {51.1: 1.0, 60.0: 10.74 / 8.35, 68.0: 12.88 / 8.35},
}


def overtake_leverage(
    k: float, b: float, closing: float = CLOSING_SPEED, target: float = ECCENTRIC_SPEED
) -> float:
    """The parent's overtake law: kg pushed per kg of bag at plug ratio ``k``.

    ``M_f / m = k / ((w / (w - v_f))^(k/B) - 1)``, ``B = J / (m w)``.
    """
    x = closing / (closing - target)
    return float(k / (x ** (k / b) - 1.0))


def fitted_b(k: float, leverage: float) -> float:
    """The ``B`` that the law needs to give ``leverage`` at ``k``."""
    return float(brentq(lambda b: overtake_leverage(k, b) - leverage, 0.3, 50.0))


@dataclass(frozen=True)
class PushLaw:
    """The Earth push at one plug column and miss rate.

    Attributes:
        column: ``compact`` (base) or ``shaped`` (unrun ceiling).
        frozen: Use the frozen-rebound edge.
        miss: Miss rate ``p``: pulses on which the gun fires (placeholder 5%).
        closing: Closing speed at 200 km (km/s).
        liner_carried: Charge the liner's eroded graphite.
    """

    column: str = "compact"
    frozen: bool = False
    miss: float = 0.05
    closing: float = CLOSING_SPEED
    liner_carried: bool = True

    @property
    def k_range(self) -> Tuple[float, float]:
        """The tabled range of ``k``."""
        ks = sorted(LEVERAGE[self.column])
        return ks[0], ks[-1]

    def b(self, k: float) -> float:
        """``B`` interpolated in ``ln k`` between the tabled rows."""
        ks = sorted(LEVERAGE[self.column])
        bs = [fitted_b(x, LEVERAGE[self.column][x][int(self.frozen)]) for x in ks]
        lo, hi = ks[0], ks[-1]
        if not lo - 1e-9 <= k <= hi + 1e-9:
            raise ValueError(f"k = {k} outside the tabled {lo}-{hi}")
        return float(np.interp(math.log(k), np.log(ks), bs))

    def leverage(self, k: float) -> float:
        """Kg pushed per kg of bag, the miss and liner charged."""
        b = (1.0 - self.miss) * self.b(k) + self.miss * BACKUP_B
        base = overtake_leverage(k, b)
        gain = float(
            np.interp(
                self.closing,
                sorted(SPEED_GAIN[self.column]),
                [SPEED_GAIN[self.column][s] for s in sorted(SPEED_GAIN[self.column])],
            )
        )
        return base * gain * (LINER_LEVERAGE if self.liner_carried else 1.0)

    def plate_share(self, k: float) -> float:
        """Plate and spring per kg pushed."""
        rows = PLATE_TABLE[self.column]
        ks = sorted(rows)
        ship = float(np.interp(k, ks, [rows[x][0] for x in ks]))
        plate = float(np.interp(k, ks, [rows[x][1] for x in ks]))
        hardware = plate * (1.0 + SPRING_PER_PLATE)
        return hardware / (ship + hardware)

    def pushed_per_push(self, k: float) -> float:
        """Kg pushed per push (ship, plate and spring)."""
        rows = PLATE_TABLE[self.column]
        ks = sorted(rows)
        ship = float(np.interp(k, ks, [rows[x][0] for x in ks]))
        return ship * 1e3 / (1.0 - self.plate_share(k))

    def plate_mass(self, k: float) -> float:
        """The plate and its spring (kg)."""
        return self.pushed_per_push(k) * self.plate_share(k)


# --------------------------------------------------------------------------
# Prices and the two ledgers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CyclePrices:
    """Every price the cycle is charged, dollars (assumptions marked).

    Attributes:
        lob: Per kilogram lofted (the ledger's $25).
        lob_climb: The 1.1 km/s climbing lob (lob_rise, x1.0648): $26.6.
        leo: Per kilogram to LEO, the seed (SpaceX $150/kg, seed_cost).
        engine: Per 560 MW engine at scale ($40M / $60M / $100M rows, the
            parent's working figures, not cited).
        u235: Per kilogram of U-235 (HALEU at $20-25k/kgU, parent).
        u235_inventory: A 19.75% LEU cermet core's U-235 at 111 kN (kg;
            "Design comparison of nuclear thermal rocket concepts", read from
            a garbled table: verify).  The only fixed part of the engine's
            price; the rest scales with power.
        hydrogen: Per kg of LH2 (assumption).
        stage_hardware: Per kg of chemical stage dry mass and LH2 tanks
            (assumption: a stripped hull at $20M for 40 t, seed_cost).
        water_tank: Per kg of the water stage's net or film walls (assumption).
        kit: Per kg of mining and launch kit (assumption, as avionics).
        skin: Per kg of PuffSat skin (the solar dive's sheet body).
        package: The steering package curve, one per ``bags_per_package``.
        bags_per_package: Bags per package (the child's two per 60 kg).
        methalox: Per kg (growth_cost).
        graphite: Per kg of bulk isostatic graphite (Indian B2B listings
            ~INR 500-3000/kg, ~$6-36; assumption mid).
        liner: Graphite consumed per push (kg; Result 20's 2-18 t).
        gun: Per gun, one a plate, flown as the plate is (assumption).
        bullet: Ice bullet (kg).
        dear_seed: Price a chemical seed at the seed ship's dear end.
        aeroshell: Per kg of aeroshell shed after an aerocapture (assumption:
            thermal protection, structure and backshell at the kit's price).
    """

    lob: float = 25.0
    lob_climb: float = 1.0648
    leo: float = 150.0
    engine: float = 60.0e6
    u235: float = 112.5e3
    u235_inventory: float = 71.3
    hydrogen: float = 6.0
    stage_hardware: float = 500.0
    water_tank: float = 100.0
    kit: float = 1000.0
    skin: float = 3.0
    package: LearningCurve = LearningCurve(100.0, 10.0, 0.8, anchor=1.0e5)
    bags_per_package: float = 60.0 / 2.0 / BAG_MASS
    methalox: float = 0.2
    graphite: float = 20.0
    liner: float = 10.0e3
    dear_seed: bool = False
    gun: float = 5.0e6
    bullet: float = 0.1
    aeroshell: float = AEROSHELL_PRICE

    @property
    def lofted(self) -> float:
        """Per kilogram lofted on the climbing lob."""
        return self.lob * self.lob_climb

    def engine_price(self, power: float, base: Optional[float] = None) -> float:
        """An engine's price at ``power`` W: U-235 fixed, the rest per MW."""
        fixed = self.u235 * self.u235_inventory
        at_560 = self.engine if base is None else base
        return fixed + (at_560 - fixed) * power / 560.0e6


ESTIMATE = CyclePrices()


@dataclass(frozen=True)
class Ledger:
    """One of Seth's two whole ledgers (2026-10-20).

    Attributes:
        label: Name.
        plate_flight: The Earth bag plate per flight.
        plate_flights: Flights a plate flies in its life (A: $200K on a $5M
            plate is 25; B: expended).
        electronics: PuffSat electronics per kg pushed, or None for the
            ledger's existing prices (skin and package curve).
        hours: Burn policy off Iapetus.
    """

    label: str
    plate_flight: float
    plate_flights: float
    electronics: Optional[float]
    hours: float


LEDGER_A = Ledger("A: reuse at scale", 200.0e3, 25.0, 1.0, 12.0)
LEDGER_B = Ledger("B: no reuse", 5.0e6, 1.0, None, 6.0)
LEDGERS = (LEDGER_A, LEDGER_B)
#: The returns arrive over about half a year (departures spread across a
#: broad Saturn window), and a plate is recovered and reflown (assumption).
STREAM_YEARS = 0.5
PLATE_TURNAROUND_DAYS = 30.0


# --------------------------------------------------------------------------
# The cycle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cycle:
    """The cycle's per-engine unit and its Earth push, ready to value.

    Attributes:
        stage: The return stage.
        out: Outbound from the eccentric orbit.
        out_leo: Outbound from LEO (the seed).
        law: The Earth push.
        k: Plug per kg of bag.
        ledger: Ledger A or B.
        prices: The book.
    """

    stage: ReturnStage
    out: Outbound
    out_leo: Outbound
    law: PushLaw
    k: float
    ledger: Ledger
    prices: CyclePrices

    @property
    def leverage(self) -> float:
        """Kg pushed per kg of bag arriving."""
        return self.law.leverage(self.k)

    @property
    def useful(self) -> float:
        """Kg pushed per kg of bag, less the plate and its spring."""
        return self.leverage * (1.0 - self.law.plate_share(self.k))

    @property
    def earth_propellant_price(self) -> float:
        """Per kg of the Earth departure's propellant."""
        return (
            self.prices.hydrogen
            if self.out.earth_stage == "hydrogen NTP"
            else self.prices.methalox
        )

    @property
    def outbound_mass(self) -> float:
        """Eccentric-orbit kg per engine sent."""
        return self.stage.earth_kg * self.out.ratio

    def growth(self, miss: float = MISS_FLOOR) -> float:
        """Engines sent next round per engine this round."""
        return self.stage.puffsat * (1.0 - miss) * self.useful / self.outbound_mass

    def pulses_per_push(self) -> float:
        """Bags a push spends."""
        return self.law.pushed_per_push(self.k) / (self.leverage * BAG_MASS)

    def hardware(self, packages_built: List[float]) -> Dict[str, float]:
        """Dollars of Earth hardware and propellant per engine sent, by line."""
        p, s, o = self.prices, self.stage, self.out
        power = s.engine.power
        lines = {
            "water engine": p.engine_price(power),
            "redundancy": (
                0.0
                if s.inputs.redundancy == "none"
                else p.engine_price(
                    power
                    * (
                        1.0
                        if s.inputs.redundancy == "second set"
                        else s.inputs.brake_share
                    )
                )
            ),
            "tankage": p.water_tank * s.tank,
            "mining kit": p.kit * s.kit,
            "skins": p.skin * s.skins,
            "outbound hydrogen NTP engines": p.engine * o.ntp_engines * s.earth_kg,
            "Earth departure propellant": self.earth_propellant_price
            * o.earth_propellant
            * s.earth_kg,
            "stage hardware": p.stage_hardware
            * (o.earth_dry + o.methalox_dry)
            * s.earth_kg,
            "methalox": p.methalox * o.methalox * s.earth_kg,
            "aeroshell": p.aeroshell * o.aeroshell * s.earth_kg,
        }
        if self.ledger.electronics is None:
            count = s.puffsat / BAG_MASS / p.bags_per_package
            lines["electronics"] = p.package.batch_cost(packages_built[0], count)
            packages_built[0] += count
        return lines

    def push_lines(self, bags: float) -> Dict[str, float]:
        """Dollars to push ``bags`` kg of arriving bags, by line."""
        return push_dollars(self.law, self.k, self.ledger, self.prices, bags)


def push_dollars(
    law: PushLaw, k: float, ledger: Ledger, prices: CyclePrices, bags: float
) -> Dict[str, float]:
    """Dollars to push ``bags`` kg of arriving bags on the Earth bag plate, by line.

    Every kilogram pushed rides the lob (cargo, ship, plate and spring), and
    so do the plug, the liner and the bullets.  Ledger A's electronics are
    $1 per useful kilogram pushed.
    """
    pushed = bags * law.leverage(k)
    pushes = pushed / law.pushed_per_push(k)
    pulses = bags / BAG_MASS
    lines = {
        "lob": prices.lofted * (pushed + prices.liner * pushes),
        "plug": prices.lofted * k * bags,
        "plate": ledger.plate_flight * pushes,
        "liner": prices.graphite * prices.liner * pushes,
        "gun": prices.gun / ledger.plate_flights * pushes,
        "bullets": prices.lofted * prices.bullet * law.miss * pulses,
    }
    if ledger.electronics is not None:
        lines["electronics"] = ledger.electronics * pushed * (1.0 - law.plate_share(k))
    return lines


def build_cycle(
    ledger: Ledger = LEDGER_A,
    k: Optional[float] = None,
    law: PushLaw = PushLaw(),
    inputs: StageInputs = BASE_INPUTS,
    engine: WaterEngine = WATER_1300K,
    prices: CyclePrices = ESTIMATE,
    hours: Optional[float] = None,
    book: OutboundBook = OUTBOUND_BOOK,
    loss: Optional[float] = None,
    real_planes: bool = False,
    aero: Optional[Tuple[float, float]] = None,
) -> Cycle:
    """Assemble a cycle; ``k`` defaults to the column's lowest tabled ratio.

    The hydrogen Earth departure's finite loss (its engine count) is chosen
    from ``book.losses`` for the lowest gated break-even (10%, whole rounds)
    unless ``loss`` is given: the engines are dear, but every kilogram they
    save in hydrogen is a kilogram less to push, so neither steady dollars
    nor the parent's 5% rule is the right criterion.  The seed leaves LEO
    on the book's Earth stage: hydrogen NTP (its loss chosen on the launch
    price) or, chemical only, the expended Starship in one burn.

    ``real_planes`` takes the Saturn-side outbound and the return's Saturn
    excess from :mod:`src.saturn_moon_legs`' medians over 2035-2048 (the
    coplanar 10.4 km/s shifted by the same amount at other closing speeds).

    ``aero`` is ``(Saturn-side delta-v after the pass, aeroshell share)``:
    every recurring ship aerocaptures at Saturn (:mod:`src.saturn_aerocapture`);
    the seed keeps the capture burn.
    """
    excess = (
        RETURN_EXCESS
        if law.closing == CLOSING_SPEED
        else return_excess_for(law.closing)
    )
    saturn_dv: Optional[float] = None
    earth_dv = EARTH_DEPARTURE_DV
    if real_planes:
        from src.saturn_moon_legs import site_summary

        legs = site_summary("Iapetus")
        d = dogleg_dvs()
        excess += legs["saturn_excess"] - RETURN_EXCESS
        saturn_dv = (
            legs["capture"] * (1.0 + CAPTURE_MARGIN)
            + legs["apse"]
            + d["match"]
            + d["land"]
        )
        earth_dv = legs["earth_dv"]
    stage = return_stage(
        ledger.hours if hours is None else hours, engine, inputs, excess
    )
    out_leo = outbound(
        True,
        book,
        prices.leo,
        prices.engine,
        prices.hydrogen,
        prices.stage_hardware,
        saturn_dv,
        earth_dv,
    )
    shell = 0.0
    if aero is not None:
        saturn_dv, shell = aero
    draft = Cycle(
        stage=stage,
        out=outbound_at(False, book, loss, saturn_dv, earth_dv, shell),
        out_leo=out_leo,
        law=law,
        k=law.k_range[0] if k is None else k,
        ledger=ledger,
        prices=prices,
    )
    if loss is not None or book.earth_stage != "hydrogen NTP":
        return draft
    best: Optional[Tuple[float, Cycle]] = None
    for trial in book.losses:
        try:
            c = replace(
                draft, out=outbound_at(False, book, trial, saturn_dv, earth_dv, shell)
            )
        except ValueError:
            continue
        if c.growth() <= 1.0:
            continue
        v = value_cycle(c)
        score = v.break_even if v.break_even is not None else math.inf
        if best is None or score < best[0]:
            best = (score, c)
    return draft if best is None else best[1]


def iapetus_gates(
    odds: Tuple[float, float, float, float] = (0.95, 0.90, 0.85, 0.90)
) -> Tuple[Gate, ...]:
    """S51's gates (Seth, 2026-10-20), at the first round's times."""
    capture = OUT_YEARS + DOGLEG_YEARS
    departure = capture + STAY_YEARS + DOGLEG_YEARS
    return (
        Gate("captures at Saturn, lands the kit on Iapetus", capture, odds[0]),
        Gate(
            "water stage qualified; first departure, ice mined and loaded",
            departure,
            odds[1],
        ),
        Gate(
            "PuffSats coast and home onto the Earth bag plate",
            ROUND_YEARS - 1e-6,
            odds[2],
        ),
        Gate("first Earth push with the ship-carried plug", ROUND_YEARS, odds[3]),
    )


#: Two rounds reinvested (19 years, the paper's drug-timeline analogue), then
#: held level and the surplus sold (S51 item 3).
REINVESTMENTS = 2
HORIZON_YEARS = 300.0
SEED_ENGINES = 10.0


@dataclass(frozen=True)
class CycleValuation:
    """A cycle valued on the gated schedule.

    Attributes:
        growth: Engines per engine per round, miss at its floor.
        round_years: The round.
        break_even: PV of dollars over PV of kilograms at the eccentric orbit.
        steady_cost: Undiscounted $/kg in a level round.
        steady_lines: The same by line.
        seed: Seed dollars.
        cargo_share: Cargo per kg pushed in a level round.
    """

    growth: float
    round_years: float
    break_even: Optional[float]
    steady_cost: float
    steady_lines: Dict[str, float]
    seed: float
    cargo_share: float


def _seed_lines(
    cycle: Cycle, engines: float, packages: List[float]
) -> Dict[str, float]:
    lines = {key: value * engines for key, value in cycle.hardware(packages).items()}
    o, s = cycle.out_leo, cycle.stage
    if o.earth_stage == "seed ship":
        # One burn from LEO on an expended Starship: its price covers the
        # launches, the hull and the propellant.
        for key in ("outbound hydrogen NTP engines", "Earth departure propellant"):
            lines.pop(key, None)
        lines["stage hardware"] = (
            cycle.prices.stage_hardware * o.methalox_dry * s.earth_kg * engines
        )
        lines["aeroshell"] = cycle.prices.aeroshell * o.aeroshell * s.earth_kg * engines
        lines["seed ship"] = (
            seed_ship_price(cycle.prices.dear_seed, o.earth_dv)
            * o.ratio
            * s.earth_kg
            * engines
        )
        return lines
    lines["launch to LEO"] = cycle.prices.leo * s.earth_kg * o.ratio * engines
    # The seed flies from LEO: its Earth departure is the LEO stack's.
    lines["outbound hydrogen NTP engines"] = (
        cycle.prices.engine * o.ntp_engines * s.earth_kg * engines
    )
    lines["Earth departure propellant"] = (
        cycle.earth_propellant_price * o.earth_propellant * s.earth_kg * engines
    )
    lines["stage hardware"] = (
        cycle.prices.stage_hardware
        * (o.earth_dry + o.methalox_dry)
        * s.earth_kg
        * engines
    )
    lines["aeroshell"] = cycle.prices.aeroshell * o.aeroshell * s.earth_kg * engines
    return lines


def value_cycle(
    cycle: Cycle,
    rate: float = 0.10,
    gates: Optional[Tuple[Gate, ...]] = None,
    pipelined: bool = False,
    reinvestments: int = REINVESTMENTS,
) -> CycleValuation:
    """Grow from a seed, then hold level; the break-even $/kg at the eccentric orbit.

    Whole rounds (headline): one seed of :data:`SEED_ENGINES` from LEO at
    year 0; its bags push the next round at 9.5 and 19 years, all of it
    reinvested; from the third return on the fleet is held level and the
    surplus is sold.  Pipelined: the same seed every Earth-Saturn window until
    the first return (:data:`SYNODIC_YEARS`), each cohort flown the same way.
    Every flow is weighted by the program's gates on the calendar, and the
    bags' homing losses fall on the solar dive's attrition curve (20% to 3%)
    in cumulative bags.  Costs and cargo of a push are booked at the push;
    a round's hardware is bought then too.

    Args:
        cycle: The cycle.
        rate: Annual time value of money (10% gated rNPV; 4% the loan).
        gates: Milestones; S51's by default.
        pipelined: Seed every window until the first return.
        reinvestments: Returns reinvested in full before holding level.

    Returns:
        The valuation.
    """
    schedule = GatedSchedule(rate, iapetus_gates() if gates is None else gates)
    packages = [0.0]
    cohorts = int(math.ceil(ROUND_YEARS / SYNODIC_YEARS - 1e-9)) if pipelined else 1
    pv_cost = pv_cargo = seed_total = 0.0
    events: List[Tuple[float, int, int]] = []
    engines = {c: SEED_ENGINES for c in range(cohorts)}
    level: Dict[int, float] = {}
    for c in range(cohorts):
        start = c * SYNODIC_YEARS
        lines = _seed_lines(cycle, SEED_ENGINES, packages)
        seed = sum(lines.values())
        seed_total += seed
        pv_cost += schedule.factor(start) * seed
        j = 1
        while start + j * ROUND_YEARS <= HORIZON_YEARS:
            events.append((start + j * ROUND_YEARS, c, j))
            j += 1
    events.sort()
    first = SEED_ENGINES * cycle.stage.puffsat
    cumulative = 0.0
    for years, c, j in events:
        returned = engines[c] * cycle.stage.puffsat
        cumulative += returned
        bags = returned * (1.0 - miss_fraction(cumulative, first))
        useful = bags * cycle.useful
        possible = useful / cycle.outbound_mass
        if j <= reinvestments:
            sent = possible
        else:
            level.setdefault(c, engines[c])
            sent = min(possible, level[c])
        cargo = max(useful - sent * cycle.outbound_mass, 0.0)
        cost = sum(cycle.push_lines(bags).values()) + sent * sum(
            cycle.hardware(packages).values()
        )
        factor = schedule.factor(years)
        pv_cost += factor * cost
        pv_cargo += factor * cargo
        engines[c] = sent
    steady = steady_state(cycle)
    return CycleValuation(
        growth=cycle.growth(),
        round_years=ROUND_YEARS,
        break_even=pv_cost / pv_cargo if pv_cargo > 0 else None,
        steady_cost=steady[0],
        steady_lines=steady[1],
        seed=seed_total,
        cargo_share=steady[2],
    )


def steady_state(cycle: Cycle) -> Tuple[float, Dict[str, float], float]:
    """A level round at the miss floor and the package floor: $/kg by line.

    Returns:
        ``($/kg, lines $/kg, cargo per kg pushed)``.
    """
    bags = cycle.stage.puffsat * (1.0 - MISS_FLOOR)
    useful = bags * cycle.useful
    cargo = useful - cycle.outbound_mass
    if cargo <= 0.0:
        return math.inf, {}, 0.0
    lines = dict(cycle.push_lines(bags))
    for key, value in cycle.hardware([1e12]).items():
        lines[key] = lines.get(key, 0.0) + value
    per_kg = {key: value / cargo for key, value in lines.items()}
    return sum(per_kg.values()), per_kg, cargo / (bags * cycle.leverage)


def best_k(
    ledger: Ledger,
    law: PushLaw = PushLaw(),
    rate: float = 0.10,
    **kwargs: object,
) -> Tuple[float, CycleValuation]:
    """The plug ratio with the lowest gated break-even, inside the tabled k."""
    lo, hi = law.k_range
    base = build_cycle(ledger, lo, law, **kwargs)  # type: ignore[arg-type]

    def score(log_k: float) -> float:
        v = value_cycle(replace(base, k=math.exp(log_k)), rate)
        return v.break_even if v.break_even is not None else 1e9

    found = minimize_scalar(
        score,
        bounds=(math.log(lo), math.log(hi)),
        method="bounded",
        options={"xatol": 1e-3},
    )
    k = float(math.exp(found.x))
    for edge in (lo, hi):
        if score(math.log(edge)) <= found.fun:
            k = edge
    return k, value_cycle(replace(base, k=k), rate)


# --------------------------------------------------------------------------
# Growth rate (item 6) and the comparison (item 4)
# --------------------------------------------------------------------------


def rate_rows(growth: float, round_years: float) -> Dict[str, float]:
    """``ln(G)/T``, doubling, and the 19-year multiple, pipelined and whole."""
    r = math.log(growth) / round_years
    return {
        "rate": r,
        "doubling": math.log(2.0) / r if r > 0.0 else math.inf,
        "pipelined_19": math.exp(19.0 * r),
        "whole_rounds": float(math.floor(ROUND_HORIZON / round_years + 1e-9)),
        "whole_19": growth ** math.floor(ROUND_HORIZON / round_years + 1e-9),
    }


#: The solar dive's dear-seed break-even ($/kg) at its two headline closures,
#: delivering to the eccentric orbit (Design.delivery_target = "eccentric",
#: re-optimised from ADR 0077's starts) and to L1 (ADR 0077), on this
#: commit; ``make solar-dive ARGS=--eccentric`` reproduces them.
SOLAR_DIVE_HEADLINE: Dict[str, Dict[str, float]] = {
    "9.86 R_sun, X 60": {"eccentric": 143.4, "l1": 145.7, "steady eccentric": 92.7},
    "4 R_sun, X 90": {"eccentric": 85.9, "l1": 87.2, "steady eccentric": 72.4},
}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _m(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "none"
    return f"{value:,.0f}" if value >= 100 else f"{value:.2f}"


def stage_report() -> str:
    """Item 1: the return stage across burn policies, both tank rows."""
    rows = []
    for inputs in (PARENT_TABLE_INPUTS, BASE_INPUTS):
        for hours in (2.0, 3.0, 6.0, 12.0, 16.0):
            s = return_stage(hours, WATER_1300K, inputs)
            rows.append(
                [
                    inputs.label,
                    f"{hours:g} h",
                    f"{100 * s.periapsis_loss:.1f}%",
                    f"{s.escape_dv:.2f}",
                    f"{s.start_acceleration:.4f}",
                    f"{s.ice / 1e3:,.0f}",
                    f"{s.puffsat / 1e3:,.0f}",
                    f"{s.tank / 1e3:,.1f}",
                    f"{s.disposal / 1e3:,.0f}",
                    f"{s.run_hours:.1f}",
                    f"{s.puffsat_per_earth_kg:.1f}",
                ]
            )
    d = dogleg_dvs()
    head = (
        "Item 1. The water stage per 560 MW engine at 1300 K "
        f"(thrust {WATER_1300K.thrust:.0f} kN, {WATER_1300K.mass_flow:.1f} kg/s).\n"
        f"Saturn side (conics): capture {d['capture']:.2f} (incl. 2%), raise {d['raise']:.3f}, "
        f"match {d['match']:.3f}, land {d['land']:.2f}; home: Iapetus excess "
        f"{d['escape']:.3f} (impulsive escape from 800 km radius "
        f"{impulsive_escape(d['escape']):.2f}), drop {d['drop']:.3f}, periapsis departure "
        f"{d['departure']:.3f} km/s.\n"
        "The parent table row (1% tank, no redundancy) reproduces S51's 214/314/554/826/919 t;\n"
        "the base row is S51's 0.2% tank and a brake-only reactor at 10% power.\n"
    )
    return head + tabulate(
        rows,
        headers=[
            "rows",
            "burn",
            "Oberth loss",
            "escape km/s",
            "a0 m/s^2",
            "ice t",
            "PuffSat t",
            "tank t",
            "disposal t",
            "run h",
            "PuffSat/Earth kg",
        ],
    )


def recession_report() -> str:
    """The barrier: the parent's coat table, recomputed."""
    rows = []
    for wall, atm, run in (
        (1950.0, 50.0, 32.5),
        (1950.0, 10.0, 32.5),
        (1700.0, 20.0, 32.5),
        (1400.0, 10.0, 32.5),
    ):
        lo, hi = sic_recession(wall, atm, 30.0), sic_recession(wall, atm, 100.0)
        rows.append(
            [
                f"{wall:.0f} K",
                f"{atm:g} atm",
                f"{lo:,.1f}-{hi:,.1f} um/h",
                f"{lo * run / 1e3:.2f}-{hi * run / 1e3:.2f} mm",
            ]
        )
    return (
        "SiC recession (parent's extrapolation of NASA/TP-1999-208696):\n"
        + tabulate(
            rows,
            headers=["hot wall", "steam", "recession 30-100 m/s", "coat for 32.5 h"],
        )
    )


def outbound_report() -> str:
    """The outbound stack against the parent's 8.2 (13.9 from LEO)."""
    rows = []
    for book in (OUTBOUND_BOOK, METHALOX_DEPARTURE):
        losses = (
            book.losses if book.earth_stage == "hydrogen NTP" else (book.methalox_loss,)
        )
        for leo in (False, True):
            for loss in losses:
                try:
                    o = outbound_at(leo, book, loss)
                except ValueError:
                    continue
                engines = ESTIMATE.engine * o.ntp_engines
                if o.earth_stage == "seed ship":
                    rows.append(
                        [
                            "seed ship (one burn, expended Starship)",
                            "LEO",
                            "steered",
                            f"{o.ratio:.2f} sent",
                            f"${seed_ship_price():,.0f}-{seed_ship_price(True):,.0f} per kg sent",
                            "0",
                            "0",
                        ]
                    )
                    continue
                rows.append(
                    [
                        book.earth_stage,
                        "LEO" if leo else "eccentric",
                        f"{loss:.0%}",
                        f"{o.ratio:.2f}",
                        f"{o.earth_propellant:.2f}",
                        f"{o.ntp_engines * 1e3:.3f}",
                        f"{engines:,.0f}",
                    ]
                )
    return (
        f"Outbound per kg landed (Saturn side methalox {outbound_at(False, OUTBOUND_BOOK).saturn_dv:.2f} km/s at "
        f"{OUTBOUND_BOOK.methalox_isp:.0f} s, {OUTBOUND_BOOK.methalox_dry:.0%} dry; hydrogen NTP "
        f"{OUTBOUND_BOOK.ntp_isp:.0f} s, LH2 tanks {OUTBOUND_BOOK.lh2_tank:.0%}, "
        f"{OUTBOUND_BOOK.ntp_engine / 1e3:.2f} t a {OUTBOUND_BOOK.ntp_thrust:.0f} kN engine). "
        "Parent: 8.2 / 13.9 at its 5%.\n"
        + tabulate(
            rows,
            headers=[
                "Earth stage",
                "from",
                "finite loss",
                "kg per kg landed",
                "Earth propellant",
                "NTP engines per t landed",
                "engine $ per kg landed at $60M",
            ],
        )
    )


def _cells(ledger: Ledger, law: PushLaw, k: float, rates: Sequence[float]) -> List[str]:
    c = build_cycle(ledger, k, law)
    out = []
    for rate in rates:
        v = value_cycle(c, rate)
        out.append(_m(v.break_even))
    return out


def ledger_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Items 2-3: the two ledgers, k solved, four cells, both counts."""
    out = []
    rows = []
    for ledger in LEDGERS:
        for column in ("compact", "shaped"):
            law = PushLaw(column)
            k, v = best_k(ledger, law, book=book)
            c = build_cycle(ledger, k, law, book=book)
            loan = value_cycle(c, 0.04)
            pipe = value_cycle(c, 0.10, pipelined=True)
            pipe_loan = value_cycle(c, 0.04, pipelined=True)
            rows.append(
                [
                    ledger.label,
                    COLUMN_LABELS[column],
                    f"{k:.2f}",
                    f"{c.leverage:.2f}",
                    f"{c.stage.puffsat / 1e3:,.0f}",
                    f"{v.growth:.2f}",
                    _m(v.steady_cost),
                    _m(v.break_even),
                    _m(loan.break_even),
                    _m(pipe.break_even),
                    _m(pipe_loan.break_even),
                ]
            )
    out.append(
        "Items 2-3. $/kg at the eccentric orbit. k solved on the gated rNPV (10%, whole rounds);\n"
        "miss p = 5% (placeholder), liner carried, frozen-rebound edge in the sensitivities.\n"
        + tabulate(
            rows,
            headers=[
                "ledger",
                "plug column",
                "k",
                "leverage",
                "PuffSat t/engine",
                "G/round",
                "steady",
                "rNPV 10%",
                "loan 4%",
                "rNPV pipelined",
                "loan pipelined",
            ],
        )
    )
    for ledger in LEDGERS:
        k, v = best_k(ledger, book=book)
        c = build_cycle(ledger, k, book=book)
        lines = ", ".join(
            f"{key} {value:.2f}"
            for key, value in sorted(v.steady_lines.items(), key=lambda kv: -kv[1])
        )
        pushes = c.stage.puffsat / BAG_MASS / c.pulses_per_push()
        per_round = min(
            STREAM_YEARS * 365.25 / PLATE_TURNAROUND_DAYS, ledger.plate_flights
        )
        out.append(
            f"{ledger.label} (compact, k {k:.2f}): steady ${v.steady_cost:.2f}/kg by line: {lines}\n"
            f"  per engine: {c.stage.puffsat / 1e3:,.0f} t of bags, {pushes:.1f} pushes of "
            f"{c.pulses_per_push():,.0f} pulses ({c.law.pushed_per_push(k) / 1e3:,.0f} t, plate + spring "
            f"{c.law.plate_mass(k) / 1e3:.1f} t); a plate flies {per_round:.1f} pushes a round "
            f"({STREAM_YEARS:g}-yr stream, {PLATE_TURNAROUND_DAYS:g}-day turnaround), so "
            f"{pushes / per_round:.2f} plates per engine per round; "
            f"seed ${v.seed / 1e9:.2f}B for {SEED_ENGINES:g} engines; cargo {100 * v.cargo_share:.0f}% of pushed"
        )
    return "\n\n".join(out)


def k_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """k on the whole ledger, and the miss-rate rows for k <= 4."""
    rows = []
    for column in ("compact", "shaped"):
        lo, hi = PushLaw(column).k_range
        for k in sorted(LEVERAGE[column]):
            for miss in (0.01, 0.05, 0.20):
                if miss != 0.05 and k > 4.0:
                    continue
                law = PushLaw(column, miss=miss)
                c = build_cycle(LEDGER_A, k, law, book=book)
                v = value_cycle(c)
                rows.append(
                    [
                        COLUMN_LABELS[column],
                        f"{k:g}",
                        f"{miss:.0%}",
                        f"{c.leverage:.2f}",
                        _m(v.steady_cost),
                        _m(v.break_even),
                    ]
                )
    return "k on Ledger A's whole ledger (gated rNPV 10%):\n" + tabulate(
        rows, headers=["column", "k", "miss p", "leverage", "steady $/kg", "rNPV $/kg"]
    )


def rate_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Item 6 for the water stage at Iapetus: growth rate, both counts."""
    rows = []
    for ledger in LEDGERS:
        for column in ("compact", "shaped"):
            k, _ = best_k(ledger, PushLaw(column), book=book)
            c = build_cycle(ledger, k, PushLaw(column), book=book)
            for frozen in (False, True):
                cf = replace(c, law=replace(c.law, frozen=frozen))
                g = cf.growth()
                r = rate_rows(g, ROUND_YEARS)
                rows.append(
                    [
                        ledger.label,
                        COLUMN_LABELS[column] + (" frozen edge" if frozen else ""),
                        f"{g:.2f}",
                        f"{ROUND_YEARS:.2f}",
                        f"{r['rate']:.3f}",
                        f"{r['doubling']:.2f}",
                        f"x{r['pipelined_19']:,.0f}",
                        f"x{r['whole_19']:,.0f} ({r['whole_rounds']:.0f} rounds)",
                    ]
                )
    return (
        "Item 6 (water NTR at Iapetus). Growth per round G with the miss at its floor;\n"
        "the gated rNPV uses whole rounds (headline) and prints the pipeline beside it.\n"
        + tabulate(
            rows,
            headers=[
                "ledger",
                "push",
                "G",
                "T yr",
                "ln(G)/T",
                "doubling yr",
                "19 yr pipelined",
                "whole rounds",
            ],
        )
    )


def parent_check_report() -> str:
    """The parent's stale growth rows, by its own inputs, as a method check.

    ``G = leverage x PuffSat per Earth kg / outbound``, round 1 from LEO.  The
    parent's 5.4 (5.0 with a second engine set) came from an earlier 3-4 h
    life sizing this module does not rebuild; its 8.2 / 13.9 outbound is
    lighter than this module's (:func:`outbound_report`).
    """
    rows = []
    for label, lev, ratio in (
        ("bare plate 6.7x", BARE_PLATE, 5.4),
        (
            "10.3x (k 8.5, eta_jet 0.775), second engine set",
            overtake_leverage(8.5, 1.0 + 0.775 * math.sqrt(9.5)),
            5.0,
        ),
    ):
        g2 = lev * ratio / 8.2
        g1 = g2 * 8.2 / 13.9
        rows.append(
            [label, f"{ratio:.1f}", f"{g1:.2f}", f"{g2:.2f}", f"x{g1 * g2:.0f}"]
        )
    return (
        "Parent check (its stale inputs): G = leverage x PuffSat per Earth kg / outbound (8.2; 13.9 from LEO).\n"
        "Parent: 2.6 / 4.4 bare, 3.7 / 6.3 at 10.3x, x10 / x23 over 19 years.\n"
        + tabulate(
            rows,
            headers=["push", "PuffSat/Earth kg", "G round 1", "G round 2", "19 yr"],
        )
    )


def comparison_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Item 4: this cycle against the solar dive, at the eccentric orbit and L1."""
    rows = []
    for ledger in LEDGERS:
        k, v = best_k(ledger, book=book)
        rows.append(
            [
                f"Iapetus water NTR, ledger {ledger.label[0]}, compact plug",
                _m(v.steady_cost),
                _m(v.break_even),
                "-",
            ]
        )
    for key, value in SOLAR_DIVE_HEADLINE.items():
        rows.append(
            [
                f"solar dive {key}",
                _m(value["steady eccentric"]),
                _m(value["eccentric"]),
                _m(value["l1"]),
            ]
        )
    return (
        "Item 4. Against the paper's current route (the solar-dive charge pusher, ADR 0077 headlines,\n"
        "dear seed, re-optimised to deliver to the 10-day eccentric orbit; L1 beside it). $/kg:\n"
        + tabulate(
            rows,
            headers=["route", "steady", "break-even (eccentric)", "break-even (L1)"],
        )
    )


def target_report(target: float = 2.0, book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Item 5's target: what must fall to reach ~$1-2/kg (Ledger A, the loan).

    The pushed mass itself rides the lob, so the lob sets a floor under any
    Saturn-side saving: ``lob x (pushed + plug) / cargo``.
    """
    k, v = best_k(LEDGER_A, rate=0.04, book=book)
    c = build_cycle(LEDGER_A, k, book=book)
    lines = v.steady_lines
    lofted = ("lob", "plug", "bullets")
    lob_lines = sum(lines.get(key, 0.0) for key in lofted)
    rest = v.steady_cost - lob_lines
    pushed = 1.0 / v.cargo_share
    plug = k / c.leverage * pushed
    alone = ESTIMATE.lofted * target / lob_lines
    biggest = ", ".join(
        f"{key} ${value:.2f}"
        for key, value in sorted(lines.items(), key=lambda kv: -kv[1])
        if key not in lofted and value >= 0.5
    )
    return (
        f"Target ${target:g}/kg (Ledger A, k {k:.2f}, compact): steady ${v.steady_cost:.2f}, loan break-even ${_m(v.break_even)}.\n"
        f"  Lofted mass: {pushed:.2f} kg pushed and {plug:.2f} kg of plug per kg of cargo at ${ESTIMATE.lofted:.1f}/kg = ${lob_lines:.2f}/kg.\n"
        f"  Everything else: ${rest:.2f}/kg ({biggest}).\n"
        f"  With everything else free, the lofted kilogram must cost ${alone:.2f} (x{ESTIMATE.lofted / alone:.0f} cheaper), "
        f"plug and cargo alike; the rest must fall x{rest / target:.0f} to fit inside ${target:g} on its own.\n"
        "  So the target needs the ship and its plug lofted for about a dollar a kilogram, not only the plug."
    )


def residual_loss_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """The residual plate-loss row: per pulse p x q, a plate and its ship lost."""
    rows = []
    for ledger in LEDGERS:
        k, v = best_k(ledger, book=book)
        c = build_cycle(ledger, k, book=book)
        cargo_per_push = c.law.pushed_per_push(k) * v.cargo_share
        ship = (
            c.law.pushed_per_push(k)
            * (1.0 - c.law.plate_share(k))
            * (v.break_even or v.steady_cost)
        )
        value = 5.0e6 + ship
        per_q = c.pulses_per_push() * c.law.miss * value / cargo_per_push
        q1 = 0.01 * v.steady_cost / per_q
        rows.append(
            [
                ledger.label,
                f"{c.pulses_per_push():,.0f}",
                f"${value / 1e6:,.0f}M",
                f"${per_q * 1e-3:.3f}",
                f"{q1:.1e}",
            ]
        )
    return (
        "Residual loss (labelled, not stacked): a push loses plate and ship with odds p x q a pulse;\n"
        "the ship valued at the break-even price. At q = 1e-3 the row adds the $/kg shown.\n"
        + tabulate(
            rows,
            headers=[
                "ledger",
                "pulses/push",
                "plate + ship",
                "$/kg at q = 1e-3",
                "q for 1% of $/kg",
            ],
        )
    )


def plate_check_report() -> str:
    """Seth's per-flight plate prices against the ledger's curve at 25-32 t."""
    from src.solar_dive_growth import ESTIMATE as SD

    curve = SD.plate_curve(SD.earth_plate_share)
    rows = []
    for column, k in (("compact", 4.0), ("compact", 8.5), ("shaped", 8.5)):
        plate = PushLaw(column).plate_mass(k) / (1.0 + SPRING_PER_PLATE)
        rows.append(
            [
                COLUMN_LABELS[column],
                f"{k:g}",
                f"{plate / 1e3:.1f}",
                f"${curve.batch_cost(0.0, 1.0) * plate / 1e6:.2f}M",
                f"${curve.floor * plate / 1e6:.2f}M",
            ]
        )
    return (
        "Plate check: the ledger's spray-cup curve at half price (the solar dive's Earth plates), expended,\n"
        "against Seth's $5M (B) and $200K a flight (A):\n"
        + tabulate(rows, headers=["column", "k", "plate t", "first unit", "floor"])
    )


def saturn_tank_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Tank walls made at Saturn by a 40 kWe fission surface power unit."""
    stage = return_stage(12.0)
    tonnes_a_year = 40.0e3 * _YEAR / 75.0e6 / 1e3
    per_round = tonnes_a_year * ROUND_YEARS
    stages = per_round * 1e3 / stage.tank
    c = build_cycle(LEDGER_A, 4.0, book=book)
    landed = (sum(c.push_lines(1.0).values()) / c.useful) * c.out.ratio
    shipped = (ESTIMATE.water_tank + landed) * stage.tank
    unit_mass = 6.0e3
    return (
        f"Saturn-made tank walls: a 40 kWe unit at 75 MJ/kg makes {tonnes_a_year:.1f} t a year, "
        f"{per_round:.0f} t a round, the walls of {stages:.1f} 12-hour stages ({stage.tank / 1e3:.1f} t each at 0.2%).\n"
        f"  Shipping one stage's walls costs ${shipped / 1e6:.2f}M (${landed:,.0f} per kg landed, Ledger A).\n"
        f"  The unit pays if it lands for less than ${stages * shipped / 1e6:.0f}M a round it serves, its "
        f"{unit_mass / 1e3:.0f} t (assumption) costing ${unit_mass * landed / 1e6:.1f}M to land."
    )


def power_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Reactor power per engine: U-235 fixed, everything else per MW.

    The most power the core reaches at the 1400 K wall is not derived here;
    the sweep is labelled.
    """
    rows = []
    for mw in (280.0, 560.0, 840.0, 1120.0):
        engine = WATER_1300K.at_power(mw * 1e6)
        c = build_cycle(LEDGER_A, 4.0, engine=engine, book=book)
        v = value_cycle(c)
        rows.append(
            [
                f"{mw:.0f}",
                f"{engine.thrust:.0f}",
                f"{engine_mass(BASE_INPUTS.engine, engine.power) / 1e3:.2f}",
                f"${ESTIMATE.engine_price(engine.power) / 1e6:.0f}M",
                f"{c.stage.puffsat / 1e3:,.0f}",
                f"{c.growth():.2f}",
                _m(v.steady_cost),
                _m(v.break_even),
            ]
        )
    return (
        "Power per engine (Ledger A, k 4, 12 h): price = U-235 (71.3 kg at $112.5k) + the rest per MW;\n"
        "mass on the GE-711 pair's shape. The ceiling at the 1400 K wall is unsolved (labelled).\n"
        + tabulate(
            rows,
            headers=[
                "MW",
                "kN",
                "engine t",
                "engine $",
                "PuffSat t",
                "G",
                "steady $/kg",
                "rNPV $/kg",
            ],
        )
    )


@dataclass(frozen=True)
class Sensitivity:
    """One row: a label and the cycle's arguments.

    Attributes:
        label: What moves.
        kwargs: Arguments to :func:`build_cycle`.
    """

    label: str
    kwargs: Dict[str, object] = field(default_factory=dict)


SENSITIVITIES = (
    Sensitivity("base (Ledger A, compact, 12 h, 1300 K)"),
    Sensitivity("arrival 60 km/s", {"law": PushLaw(closing=60.0)}),
    Sensitivity("arrival 68 km/s", {"law": PushLaw(closing=68.0)}),
    Sensitivity("engine $40M", {"prices": replace(ESTIMATE, engine=40.0e6)}),
    Sensitivity(
        "engine $100M (pessimistic)", {"prices": replace(ESTIMATE, engine=100.0e6)}
    ),
    Sensitivity("1800 K (undemonstrated barrier)", {"engine": WATER_1800K}),
    Sensitivity("3 h burn (growth policy)", {"hours": 3.0}),
    Sensitivity("6 h burn", {"hours": 6.0}),
    Sensitivity("16 h burn", {"hours": 16.0}),
    Sensitivity("frozen-rebound edge", {"law": PushLaw(frozen=True)}),
    Sensitivity("shaped-plug ceiling (unrun)", {"law": PushLaw("shaped")}),
    Sensitivity("miss p 1%", {"law": PushLaw(miss=0.01)}),
    Sensitivity("miss p 20%", {"law": PushLaw(miss=0.20)}),
    Sensitivity(
        "tankage 1% (old base)", {"inputs": replace(BASE_INPUTS, tankage=0.01)}
    ),
    Sensitivity("optimistic masses", {"inputs": OPTIMISTIC_INPUTS}),
    Sensitivity("pessimistic masses", {"inputs": PESSIMISTIC_INPUTS}),
    Sensitivity(
        "redundancy: second full engine set",
        {"inputs": replace(BASE_INPUTS, redundancy="second set")},
    ),
    Sensitivity(
        "redundancy: none", {"inputs": replace(BASE_INPUTS, redundancy="none")}
    ),
    Sensitivity(
        "graphite $1000/kg (paper's disk price)",
        {"prices": replace(ESTIMATE, graphite=1000.0)},
    ),
    Sensitivity("liner 18 t a push", {"prices": replace(ESTIMATE, liner=18.0e3)}),
)


def sensitivity_report(
    rows_in: Sequence[Sensitivity] = SENSITIVITIES, book: OutboundBook = OUTBOUND_BOOK
) -> str:
    """Item 7: one line moved at a time, k re-solved on each."""
    rows = []
    for sens in rows_in:
        law = sens.kwargs.get("law", PushLaw())
        kwargs = {key: value for key, value in sens.kwargs.items() if key != "law"}
        k, v = best_k(LEDGER_A, law, **kwargs, book=book)  # type: ignore[arg-type]
        c = build_cycle(LEDGER_A, k, law, **kwargs, book=book)  # type: ignore[arg-type]
        loan = value_cycle(c, 0.04)
        rows.append(
            [
                sens.label,
                f"{k:.2f}",
                f"{c.leverage:.2f}",
                f"{c.out.ratio:.1f}",
                f"{c.growth():.2f}",
                _m(v.steady_cost),
                _m(v.break_even),
                _m(loan.break_even),
            ]
        )
    return "Item 7. Sensitivities, Ledger A, k re-solved on each:\n" + tabulate(
        rows,
        headers=[
            "row",
            "k",
            "leverage",
            "outbound",
            "G",
            "steady",
            "rNPV 10%",
            "loan 4%",
        ],
    )


#: The two Earth-departure scenarios every table is printed under: the
#: parent's hydrogen NTP, and chemical only near Earth (no reactor run inside
#: the magnetosphere; Seth, 2026-10-21: an accident after startup scatters
#: fission products, a cold core does not).
SCENARIOS: Tuple[Tuple[str, OutboundBook], ...] = (
    ("hydrogen NTP Earth departure (the parent's)", OUTBOUND_BOOK),
    (
        "no nuclear near Earth: methalox Earth departure, seed on the expended Starship",
        METHALOX_DEPARTURE,
    ),
)


def scenario_summary() -> str:
    """The headline cells under both Earth-departure scenarios."""
    rows = []
    for name, book in SCENARIOS:
        for ledger in LEDGERS:
            k, v = best_k(ledger, book=book)
            c = build_cycle(ledger, k, book=book)
            loan = value_cycle(c, 0.04)
            pipe = value_cycle(c, pipelined=True)
            rows.append(
                [
                    book.earth_stage,
                    ledger.label[0],
                    f"{c.out.ratio:.1f}",
                    f"{c.growth():.2f}",
                    _m(v.steady_cost),
                    _m(v.break_even),
                    _m(loan.break_even),
                    _m(pipe.break_even),
                    f"${v.seed / 1e9:.1f}B",
                ]
            )
    return (
        "Both scenarios, water NTR at Iapetus, compact k 4, $/kg at the eccentric orbit:\n"
        + tabulate(
            rows,
            headers=[
                "Earth departure",
                "ledger",
                "outbound",
                "G",
                "steady",
                "rNPV 10%",
                "loan 4%",
                "rNPV pipelined",
                "seed (10 engines)",
            ],
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the S51 report under both Earth-departure scenarios."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        choices=("both", "hydrogen", "chemical"),
        default="both",
        help="which Earth-departure scenario's full tables to print",
    )
    args = parser.parse_args(argv)
    for section in (
        stage_report(),
        recession_report(),
        outbound_report(),
        parent_check_report(),
        plate_check_report(),
        scenario_summary(),
    ):
        print(section)
        print()
    for (name, book), key in zip(SCENARIOS, ("hydrogen", "chemical")):
        if args.scenario not in ("both", key):
            continue
        print("=" * 100 + f"\nScenario: {name}\n" + "=" * 100)
        for section in (
            ledger_report(book),
            k_report(book),
            rate_report(book),
            comparison_report(book),
            target_report(book=book),
            residual_loss_report(book),
            saturn_tank_report(book),
            power_report(book),
            sensitivity_report(book=book),
        ):
            print(section)
            print()


if __name__ == "__main__":
    main()
