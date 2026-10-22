"""The solar dive grown at Ceres: the parent's S56.

The author's ISRU variant of the solar-dive charge pusher
(:mod:`src.solar_dive_growth`) removes the growth lob.  Ceres supplies the
bulk: polar water at $0.02/kg, and graphite, tank walls and carbon foam made
there on nuclear power.  Only the plates and the electronics come from Earth.

1. **Ceres push.** A returning **Ceres stream** of compact ice pellets meets
   the ship at Ceres' orbit, about 80 degrees off its thrust axis.  Each
   pellet threads the hole in a departure bowl and stops in a water charge
   at its focus.  The ship goes onto a prograde dive, and its graphite disks
   onto the mirror-image retrograde dive, so that the two meet head-on at
   perihelion.  Ceres' escape speed is 0.51 km/s, so the push starts from
   rest in Ceres' orbit, and a burn of hours is impulsive on the heliocentric
   orbit.
2. **Sun.** The disks strike polyethylene charges in the Sun bowl over an arc
   around perihelion, as in the solar dive, and send the ship out on a fast
   hyperbola.
3. **Return.** The ship's remainder is released as the next stream: ice
   pellets back to Ceres' orbit, or Kapton-grade sheets (Ceres film) to
   Earth.  At Earth the sheets push lofted payload to L1 on reused plates,
   the solar dive's delivery leg (:func:`src.solar_dive_growth.delivery_unit`).

The loop has no growth lob.  The lofted mass is the payload, the Earth
charges and the Earth plates; the Earth hardware for Ceres (plates, bowls,
avionics) flies there on the paper's departure.

Everything is float arithmetic (km, km/s, kg, years, dollars), like the solar
dive.  The heliocentric legs are two-body conics.  Ceres is coplanar and
circular unless stated (:func:`real_orbit_rows` checks what that hides).

Run with ``make ceres-dive``.
"""

import argparse
import functools
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from tabulate import tabulate

from src.conic_kernel import wrap_pi
from src.solar_dive_growth import (
    AVIONICS_FRACTION,
    BORE_COLLAR,
    CARRIER_DRY_FRACTION,
    CHARGE_MOUNT_FRACTION,
    FILM_PER_IMPULSE,
    LAUNCH_UNIT,
    MATURE,
    MISS_FLOOR,
    PUSH_ALTITUDE,
    SHIELD_FRACTION,
    SNOW_FOAM,
    STEERING_PACKAGE_BEST,
    SUN_PLATE_AREA,
    SUN_PLATE_PULSE,
    SUN_PLATE_SPEED,
    SUN_PULSE_RATE,
    DepartureGeometry,
    Design,
    EarthLeg,
    StageLaws,
    _plate_for,
    delivery_unit,
    fly_earth,
    gas_spring,
    lob_floor,
    sheet_material,
    wave_speed,
)
from src.two_wave_growth import VE_METHALOX

MU_SUN = 1.32712440018e11  # km^3/s^2
AU = 1.495978707e8  # km
R_SUN = 695700.0  # km
_MU_EARTH = 398600.4418
_R_EARTH = 6378.137
_G0 = 9.80665e-3  # km/s^2
_YEAR = 365.25 * 86400.0
_HOUR = 3600.0

# --- Ceres (JPL small-body database, epoch 2026).  The ledger flies the
# coplanar circular orbit at the semi-major axis, as the parent did.
CERES_A = 2.767 * AU
CERES_E = 0.0785
CERES_I = 10.59  # deg to the ecliptic
EARTH_A = 1.0 * AU
#: Mars, for the flyby check (item 3d).
MARS_A = 1.5237 * AU
MU_MARS = 42828.37
R_MARS = 3389.5


def circular_speed(r: float) -> float:
    """Circular speed about the Sun at ``r`` km (km/s)."""
    return math.sqrt(MU_SUN / r)


def mean_motion(r: float) -> float:
    """Circular mean motion about the Sun at ``r`` km (rad/s)."""
    return math.sqrt(MU_SUN / r**3)


def synodic_years(r1: float, r2: float) -> float:
    """Synodic period of two circular orbits (years)."""
    return 2.0 * math.pi / abs(mean_motion(r1) - mean_motion(r2)) / _YEAR


# --------------------------------------------------------------------------
# 1. The Ceres push law
# --------------------------------------------------------------------------

#: The parent's law 1: the bowl's axial turn, mid of the departure-hole runs'
#: 60-73%.
ETA_GEOM = 0.65
#: Atomization tolls, MJ/kg: H2O -> 2H + O at 9.5 eV a molecule; and the
#: parent's polyethylene figure.
WATER_BOND = 51.0
POLYETHYLENE_BOND = 84.0
#: Share of water's toll the expansion returns before the gas leaves the
#: bowl: the companion's 23-31% for water on a plate (ADR 0033 item 5).
#: Water's bonds re-form only near 3000-4000 K, late in the expansion, so the
#: headline keeps the frozen toll (0).
WATER_RECOVERY = (0.23, 0.31)
#: The parent's departure check: polyethylene at 84 km/s, 130 deg, k = 15,
#: against the paper's 1015 t -> 566 t over 4.67 km/s on 29.5 t of impactor.
PAPER_DEPARTURE = (84.0, 130.0, 15.0)
PAPER_DEPARTURE_VE = 4.67 / math.log(1015.0 / 566.0)
#: The Ceres push's slug-ratio cap, the parent's (the hole runs reach 60).
K_CAP = 30.0


def law_one_ve(
    k: float,
    w: float,
    theta_deg: float,
    bond: float,
    impactor_bond: float = 0.0,
    eta_geom: float = ETA_GEOM,
) -> float:
    """The parent's law 1, per kilogram of charge (km/s).

    ``I / (m w) = eta_geom eta_chem sqrt(1 + k) + cos(theta)`` per kilogram of
    impactor, ``eta_chem = sqrt(1 - 2 (k E_b + E_imp) / w^2)``, and ``v_e =
    I / (k m)``.  The impactor's own toll (``impactor_bond``) is ADR 0033's
    two-material form; the parent's check leaves it out.

    Args:
        k: Charge per kilogram of impactor.
        w: Closing speed (km/s).
        theta_deg: Angle between the thrust and the stream's travel (0 is an
            overtake).
        bond: The charge's atomization (MJ/kg).
        impactor_bond: The impactor's (MJ/kg).
        eta_geom: The bowl's axial turn.

    Returns:
        The exhaust speed; 0 if the toll eats the whole impact.
    """
    toll = 2.0 * (k * bond + impactor_bond) / (w * w)
    if toll >= 1.0:
        return 0.0
    beta = eta_geom * math.sqrt(1.0 - toll) * math.sqrt(1.0 + k) + math.cos(
        math.radians(theta_deg)
    )
    return w * beta / k


def water_chemistry(k: float, w: float, recovered: float = 0.0) -> float:
    """Law 1's ``eta_chem`` for a water charge on an ice pellet, frozen toll."""
    toll = 2.0 * (k + 1.0) * WATER_BOND * (1.0 - recovered) / (w * w)
    return math.sqrt(1.0 - toll) if toll < 1.0 else 0.0


def _hole_laws() -> StageLaws:
    """The departure-hole surface with every bond returned (its ``condensed`` rows)."""
    return StageLaws.load(bonds="bonds returned")


def _hole_cover(laws: StageLaws) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ks = sorted({row[1] for row in laws.capture})
    speeds = sorted({row[1] for row in laws.hole})
    return (ks[0], ks[-1]), (speeds[0], speeds[-1])


def ceres_ve(
    k: float,
    w: float,
    theta_deg: float,
    recovered: float = 0.0,
    law: str = "hole",
    clamp_angle: bool = False,
) -> float:
    """Exhaust speed of a water charge in the departure bowl (km/s per kg charge).

    ``hole`` (headline): the departure-hole simulation's surface with every
    bond returned (:meth:`StageLaws.hole_ve` on its ``condensed`` rows, the
    charge at 2.5 F below the rim line), which carries the bowl's capture,
    rebound and mixing, times law 1's frozen toll for water on an ice pellet
    (:func:`water_chemistry`).  The toll scales the whole impulse, the
    pellet's own momentum included, so it is slightly pessimistic below 90
    deg.  Outside the surface's ``k`` and ``w`` it is law 1 calibrated at the
    surface's nearest edge.  Below the shallowest run angle (89.2 deg) the
    surface is extrapolated linearly in ``cos theta``, as the simulation
    does; ``clamp_angle`` holds it at 89.2.  ``law1``: the parent's law 1
    alone, the pellet's toll included.

    The parent's ratio transplant (the polyethylene surface times law 1's
    water over polyethylene) is not used: law 1's polyethylene toll reaches
    zero at ``k = w^2 / 168``, 26 at 67 km/s, where the ratio diverges.

    Args:
        k: Charge per kilogram of impactor.
        w: Closing speed (km/s).
        theta_deg: Stream angle to the thrust axis.
        recovered: Share of water's toll the expansion returns.
        law: ``hole`` or ``law1``.
        clamp_angle: Hold the angle at the shallowest run.

    Returns:
        The exhaust speed.
    """
    bond = WATER_BOND * (1.0 - recovered)
    if law == "law1":
        return law_one_ve(k, w, theta_deg, bond, bond)
    laws = _hole_laws()
    (k_lo, k_hi), (w_lo, w_hi) = _hole_cover(laws)
    kk = min(max(k, k_lo), k_hi)
    ww = min(max(w, w_lo), w_hi)
    angle = max(theta_deg, 89.2) if clamp_angle else theta_deg
    surface = laws.hole_ve(kk, ww, angle) * water_chemistry(kk, ww, recovered)
    if kk == k and ww == w:
        return surface
    edge = law_one_ve(kk, ww, angle, bond, bond)
    return surface * law_one_ve(k, w, angle, bond, bond) / edge if edge > 0 else 0.0


@dataclass(frozen=True)
class CeresPush:
    """One push from Ceres' orbit at its best slug ratio, per kilogram pushed.

    Attributes:
        k: Water per kilogram of pellet.
        ve: Exhaust speed (km/s).
        water: Water charge per kilogram after the push (``R - 1``).
        stream: Pellets per kilogram after the push (``(R - 1) / k``).
        closing: Closing speed (km/s).
        theta: Stream angle to the thrust axis (deg).
        dv: The push (km/s).
    """

    k: float
    ve: float
    water: float
    stream: float
    closing: float
    theta: float
    dv: float


def best_push(
    dv: float,
    w: float,
    theta_deg: float,
    cap: float = K_CAP,
    recovered: float = 0.0,
    law: str = "hole",
    clamp_angle: bool = False,
    k: Optional[float] = None,
) -> CeresPush:
    """The push at the slug ratio that spends the fewest pellets.

    Water is nearly free and the pellets are not, so ``k`` minimizes ``(R -
    1) / k``.  The charges ride on the ship, so past some ``k`` the stream
    needed rises again.

    Args:
        dv: The push (km/s).
        w: Closing speed (km/s).
        theta_deg: Stream angle to the thrust axis.
        cap: The highest ``k`` allowed (the floor is the surface's 5).
        recovered: Share of water's toll returned.
        law: :func:`ceres_ve`'s law.
        clamp_angle: :func:`ceres_ve`'s.
        k: Fly this ``k`` instead of the best.

    Returns:
        The push.
    """
    (k_lo, _), _ = _hole_cover(_hole_laws())

    def stream(log_k: float) -> float:
        kk = math.exp(log_k)
        ve = ceres_ve(kk, w, theta_deg, recovered, law, clamp_angle)
        if ve <= 0.0 or dv / ve > 50.0:
            return 1e9
        return (math.exp(dv / ve) - 1.0) / kk

    if k is None:
        # The surface is interpolated piecewise in log k, so the stream is not
        # unimodal: grid first, then refine within the best cell.
        grid = np.linspace(math.log(k_lo), math.log(cap), 41)
        best = int(np.argmin([stream(x) for x in grid]))
        low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        found = minimize_scalar(
            stream, bounds=(low, high), method="bounded", options={"xatol": 1e-5}
        )
        k = float(math.exp(found.x))
    ve = ceres_ve(k, w, theta_deg, recovered, law, clamp_angle)
    water = math.exp(dv / ve) - 1.0
    return CeresPush(k, ve, water, water / k, w, theta_deg, dv)


# --------------------------------------------------------------------------
# 2. Heliocentric conics (2D)
# --------------------------------------------------------------------------


def _elements(
    r: np.ndarray, v: np.ndarray
) -> Tuple[float, float, np.ndarray, float, float]:
    """``(h, e, e-vector, p, a)`` of a planar heliocentric state."""
    rn = float(np.linalg.norm(r))
    h = float(r[0] * v[1] - r[1] * v[0])
    v2 = float(v @ v)
    evec = ((v2 - MU_SUN / rn) * r - float(r @ v) * v) / MU_SUN
    e = float(np.linalg.norm(evec))
    p = h * h / MU_SUN
    a = -MU_SUN / (2.0 * (v2 / 2.0 - MU_SUN / rn))
    return h, e, evec, p, a


def _anomaly(r: np.ndarray, h: float, evec: np.ndarray) -> float:
    """True anomaly along the motion, in (-pi, pi]."""
    omega = math.atan2(evec[1], evec[0])
    phi = math.atan2(r[1], r[0])
    return wrap_pi(phi - omega if h > 0 else omega - phi)


def _time_from_periapsis(nu: float, e: float, a: float) -> float:
    """Signed time from periapsis to true anomaly ``nu`` (s)."""
    if e < 1.0:
        big_e = 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(nu / 2.0),
            math.sqrt(1.0 + e) * math.cos(nu / 2.0),
        )
        return (big_e - e * math.sin(big_e)) / math.sqrt(MU_SUN / a**3)
    f = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(nu / 2.0))
    return (e * math.sinh(f) - f) / math.sqrt(MU_SUN / abs(a) ** 3)


def _state_at(
    nu: float, h: float, e: float, evec: np.ndarray, p: float
) -> Tuple[np.ndarray, np.ndarray]:
    omega = math.atan2(evec[1], evec[0])
    sign = 1.0 if h > 0 else -1.0
    phi = omega + sign * nu
    r = p / (1.0 + e * math.cos(nu))
    radial = np.array([math.cos(phi), math.sin(phi)])
    along = sign * np.array([-math.sin(phi), math.cos(phi)])
    v_r = MU_SUN / abs(h) * e * math.sin(nu)
    v_t = abs(h) / r
    return r * radial, v_r * radial + v_t * along


def coast(
    r: np.ndarray, v: np.ndarray, nu_target: float, revs: int = 0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Coast forward to true anomaly ``nu_target``.

    Args:
        r: Position (km).
        v: Velocity (km/s).
        nu_target: The anomaly to stop at (rad, along the motion).
        revs: Whole extra revolutions first (ellipses only).

    Returns:
        ``(position, velocity, seconds)``.

    Raises:
        ValueError: If a hyperbola would have to run backward.
    """
    h, e, evec, p, a = _elements(r, v)
    dt = _time_from_periapsis(nu_target, e, a) - _time_from_periapsis(
        _anomaly(r, h, evec), e, a
    )
    if e < 1.0:
        period = 2.0 * math.pi * math.sqrt(a**3 / MU_SUN)
        dt = dt % period + revs * period
    elif dt < -1.0:
        raise ValueError("the hyperbola has already passed that anomaly")
    r1, v1 = _state_at(nu_target, h, e, evec, p)
    return r1, v1, max(dt, 0.0)


def coast_to_radius(
    r: np.ndarray, v: np.ndarray, radius: float, outbound: bool = True, revs: int = 0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Coast to the next crossing of ``radius`` on the given branch."""
    h, e, evec, p, a = _elements(r, v)
    cos_nu = (p / radius - 1.0) / e
    if abs(cos_nu) > 1.0:
        raise ValueError(f"radius {radius / AU:.3f} AU is not reached")
    nu = math.acos(cos_nu)
    return coast(r, v, nu if outbound else -nu, revs)


def dive_velocity(
    r_dep: float, r_p: float, r_a: float, retrograde: bool, inbound: bool
) -> np.ndarray:
    """Velocity at ``(r_dep, 0)`` on the ellipse ``r_p x r_a`` (km/s)."""
    a = (r_p + r_a) / 2.0
    p = 2.0 * r_p * r_a / (r_p + r_a)
    v_t = math.sqrt(MU_SUN * p) / r_dep
    v2 = MU_SUN * (2.0 / r_dep - 1.0 / a)
    v_r = math.sqrt(max(v2 - v_t * v_t, 0.0))
    return np.array([-v_r if inbound else v_r, -v_t if retrograde else v_t])


@dataclass(frozen=True)
class SunRoute:
    """A dive from Ceres' orbit through the Sun boost to a crossing radius.

    Attributes:
        depth: Perihelion (solar radii).
        excess: The climb-out's solar excess (km/s).
        retrograde: Dives against Ceres' motion.
        aphelion: The dive ellipse's aphelion (km); Ceres' radius is the
            direct dive.
        inbound: Departs on the dive ellipse's inbound branch (False: climbs
            to aphelion first).
        revs: Whole revolutions of the dive ellipse before perihelion.
        departure_dv: The Ceres push (km/s).
        thrust: Its unit axis in the departure point's (radial, along-track)
            frame.
        periapsis_speed: Arriving at perihelion (km/s).
        boost: The Sun boost (km/s).
        boost_offset_deg: True anomaly of the boost from perihelion.
        boost_turn_deg: The boosted velocity's turn off the arriving one.
        sun_years: Departure to perihelion.
        years: Departure to the crossing.
        sweep_deg: The crossing's polar angle from the departure point.
        crossing: The crossing radius (km).
        position: Crossing position (km).
        velocity: Crossing velocity (km/s).
    """

    depth: float
    excess: float
    retrograde: bool
    aphelion: float
    inbound: bool
    revs: int
    departure_dv: float
    thrust: Tuple[float, float]
    periapsis_speed: float
    boost: float
    boost_offset_deg: float
    boost_turn_deg: float
    sun_years: float
    years: float
    sweep_deg: float
    crossing: float
    position: Tuple[float, float]
    velocity: Tuple[float, float]

    @property
    def miss_deg(self) -> float:
        """Stream crossing angle less a co-orbiting departure point's, at Ceres."""
        phase = math.degrees(mean_motion(CERES_A) * self.years * _YEAR)
        return math.degrees(wrap_pi(math.radians(self.sweep_deg - phase)))

    def earth_phase_deg(self) -> float:
        """Earth's lead on the departure point at departure for the stream to meet it."""
        travel = math.degrees(mean_motion(EARTH_A) * self.years * _YEAR)
        return math.degrees(wrap_pi(math.radians(self.sweep_deg - travel)))


def sun_route(
    depth: float,
    excess: float,
    retrograde: bool = False,
    aphelion: Optional[float] = None,
    inbound: bool = True,
    revs: int = 0,
    crossing: float = CERES_A,
    r_dep: float = CERES_A,
    boost_offset_deg: float = 0.0,
    boost_turn_deg: float = 0.0,
) -> SunRoute:
    """Fly a dive from a circular departure point to a crossing radius.

    Args:
        depth: Perihelion (solar radii).
        excess: Solar excess after the boost (km/s).
        retrograde: Dive against the departure point's motion.
        aphelion: The dive ellipse's aphelion (km); ``r_dep`` by default.
        inbound: Depart on the inbound branch.
        revs: Whole dive-ellipse revolutions before the boost.
        crossing: Where the stream is caught (km).
        r_dep: The departure orbit's radius (km).
        boost_offset_deg: Boost this far (true anomaly) from perihelion,
            along the velocity; negative is before.
        boost_turn_deg: Turn the boosted velocity this far off the arriving
            one, toward the outward radial.

    Returns:
        The route.
    """
    r_p = depth * R_SUN
    r_a = r_dep if aphelion is None else aphelion
    if r_a < r_dep * (1.0 - 1e-12):
        raise ValueError("the dive ellipse must reach the departure radius")
    r0 = np.array([r_dep, 0.0])
    v0 = dive_velocity(r_dep, r_p, r_a, retrograde, inbound and r_a > r_dep)
    depot = np.array([0.0, circular_speed(r_dep)])
    dv_vec = v0 - depot
    dv = float(np.linalg.norm(dv_vec))
    # In the departure frame radial is +x and along-track is +y.
    thrust = (float(dv_vec[0] / dv), float(dv_vec[1] / dv))
    rp_vec, vp_vec, t_in = coast(r0, v0, 0.0, revs)
    v_p = float(np.linalg.norm(vp_vec))
    r_b, v_b, t_b = rp_vec, vp_vec, 0.0
    if boost_offset_deg:
        r_b, v_b, t_b = coast(r0, v0, math.radians(boost_offset_deg), revs)
        t_b -= t_in
    rb = float(np.linalg.norm(r_b))
    speed = math.sqrt(excess**2 + 2.0 * MU_SUN / rb)
    direction = v_b / np.linalg.norm(v_b)
    if boost_turn_deg:
        out = r_b / rb
        side = 1.0 if (r_b[0] * v_b[1] - r_b[1] * v_b[0]) > 0 else -1.0
        # Rotate toward the outward radial, keeping the sense of motion.
        c, s = math.cos(math.radians(boost_turn_deg)), math.sin(
            math.radians(boost_turn_deg)
        )
        perp = np.array([-direction[1], direction[0]]) * side
        if float(perp @ out) < 0:
            perp = -perp
        direction = c * direction + s * perp
    v_new = speed * direction
    boost = float(np.linalg.norm(v_new - v_b))
    r1, v1, t_out = coast_to_radius(r_b, v_new, crossing, outbound=True)
    total = t_in + t_b + t_out
    return SunRoute(
        depth=depth,
        excess=excess,
        retrograde=retrograde,
        aphelion=r_a,
        inbound=inbound,
        revs=revs,
        departure_dv=dv,
        thrust=thrust,
        periapsis_speed=v_p,
        boost=boost,
        boost_offset_deg=boost_offset_deg,
        boost_turn_deg=boost_turn_deg,
        sun_years=(t_in + t_b) / _YEAR,
        years=total / _YEAR,
        sweep_deg=math.degrees(math.atan2(r1[1], r1[0])),
        crossing=crossing,
        position=(float(r1[0]), float(r1[1])),
        velocity=(float(v1[0]), float(v1[1])),
    )


def stream_at(stream: SunRoute, pushed: SunRoute) -> Tuple[float, float]:
    """Closing speed and angle of ``stream`` on a ship departing on ``pushed``.

    Both are taken in the crossing point's local (radial, along-track)
    frame: the next ship departs from where the stream crosses, on the same
    footing as the first.

    Returns:
        ``(w km/s, theta deg)``.
    """
    r = np.array(stream.position)
    v = np.array(stream.velocity)
    rn = float(np.linalg.norm(r))
    radial = r / rn
    along = np.array([-radial[1], radial[0]])
    rel = v - circular_speed(rn) * along
    w = float(np.linalg.norm(rel))
    local = np.array([float(rel @ radial), float(rel @ along)]) / w
    cos_theta = float(local @ np.array(pushed.thrust))
    return w, math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def earth_stream(route: SunRoute) -> Tuple[float, float]:
    """An Earth-bound stream's Earth excess and its speed at 400 km (km/s)."""
    r = np.array(route.position)
    v = np.array(route.velocity)
    rn = float(np.linalg.norm(r))
    radial = r / rn
    along = np.array([-radial[1], radial[0]])
    v_inf = float(np.linalg.norm(v - circular_speed(rn) * along))
    return v_inf, wave_speed(v_inf, PUSH_ALTITUDE)


def earth_excess_for(depth: float, w_400: float, **route: float) -> float:
    """The solar excess whose Earth-bound stream arrives at ``w_400`` km/s."""

    def residual(x: float) -> float:
        flown = sun_route(depth, x, crossing=EARTH_A, **route)  # type: ignore[arg-type]
        return earth_stream(flown)[1] - w_400

    return float(brentq(residual, 20.0, 400.0, xtol=1e-6))


def sun_closing(ship: SunRoute, disks: SunRoute) -> float:
    """Disk closing speed at perihelion, head-on (km/s)."""
    return ship.periapsis_speed + disks.periapsis_speed


# --------------------------------------------------------------------------
# 3. Phasing: where the stream lands
# --------------------------------------------------------------------------

#: The dive ellipse's aphelia scanned for closures (AU), and the revolutions.
APHELION_SCAN = (CERES_A, 6.0 * AU)
_SCAN_POINTS = 240
REVS = (0, 1)


def _family(
    depth: float, excess: float, retrograde: bool, inbound: bool, revs: int
) -> List[Tuple[float, SunRoute]]:
    out = []
    for r_a in np.linspace(*APHELION_SCAN, _SCAN_POINTS):
        try:
            out.append(
                (float(r_a), sun_route(depth, excess, retrograde, r_a, inbound, revs))
            )
        except ValueError:
            continue
    return out


def closures(
    depth: float,
    excess: float,
    retrograde: bool,
    target_deg: float = 0.0,
    measure: str = "ceres",
    revolutions: Sequence[int] = REVS,
) -> List[SunRoute]:
    """Every dive whose stream lands ``target_deg`` from its departure point.

    The family is the dive ellipse's aphelion (Ceres' radius to 6 AU), its
    branch at departure and whole revolutions before perihelion.

    Args:
        depth: Perihelion (solar radii).
        excess: Solar excess (km/s).
        retrograde: The dive's sense.
        target_deg: The landing phase wanted (``miss_deg``), or for ``earth``
            the Earth phase at departure (``earth_phase_deg``).
        measure: ``ceres`` (the stream returns to Ceres' orbit) or ``earth``.
        revolutions: Dive-ellipse revolutions allowed before perihelion (each
            one is an unboosted perihelion pass).

    Returns:
        The closures, cheapest departure first.
    """
    found = []
    crossing = CERES_A if measure == "ceres" else EARTH_A

    def value(route: SunRoute) -> float:
        raw = route.miss_deg if measure == "ceres" else route.earth_phase_deg()
        return math.degrees(wrap_pi(math.radians(raw - target_deg)))

    for inbound in (True, False):
        for revs in revolutions:
            family = _family(depth, excess, retrograde, inbound, revs)
            for (r1, a), (r2, b) in zip(family, family[1:]):
                va, vb = value(a), value(b)
                if va * vb > 0 or abs(va - vb) > 90.0:
                    continue

                def residual(r_a: float) -> float:
                    return value(
                        sun_route(
                            depth,
                            excess,
                            retrograde,
                            r_a,
                            inbound,
                            revs,
                            crossing=crossing,
                        )
                    )

                if measure == "earth":
                    va, vb = residual(r1), residual(r2)
                    if va * vb > 0 or abs(va - vb) > 90.0:
                        continue
                r_a = float(brentq(residual, r1, r2, xtol=1.0))
                found.append(
                    sun_route(
                        depth, excess, retrograde, r_a, inbound, revs, crossing=crossing
                    )
                )
    return sorted(found, key=lambda route: route.departure_dv)


def tug_dv(phase_deg: float, years: float) -> float:
    """Two-burn drift of ``phase_deg`` along Ceres' orbit in ``years`` (km/s).

    ``v (2/3) dtheta / (n t)``: the phasing orbit's ``da/a`` and back.
    """
    return (
        circular_speed(CERES_A)
        * (2.0 / 3.0)
        * math.radians(abs(phase_deg))
        / (mean_motion(CERES_A) * years * _YEAR)
    )


@dataclass(frozen=True)
class Ring:
    """Co-orbital depots the Ceres-bound stream steps round.

    A depot holds a fixed phase on Ceres' orbit, but the stream lands
    ``miss`` beyond the point its ships left from, so a single depot does not
    close: ships from the depot send their stream on again.  ``N`` depots
    spaced ``360 / N`` close when each one's ships land on the next but
    ``j``: ``miss = -360 j / N``.  Ceres is depot 0, and the others are fed
    from it by a steam tug.

    Attributes:
        depots: ``N``.
        step: ``j``.
        route: The ship's closing dive.
        mean_offset_deg: Mean phase of the depots from Ceres.
    """

    depots: int
    step: int
    route: SunRoute

    @property
    def mean_offset_deg(self) -> float:
        """Mean phase a depot sits from Ceres (deg)."""
        return float(
            np.mean(
                [
                    abs(math.degrees(wrap_pi(2.0 * math.pi * i / self.depots)))
                    for i in range(self.depots)
                ]
            )
        )


def rings(
    depth: float,
    excess: float,
    retrograde: bool = False,
    most: int = 6,
    revolutions: Sequence[int] = REVS,
) -> List[Ring]:
    """The cheapest closing dive for each ring of up to ``most`` depots."""
    out = []
    for n in range(1, most + 1):
        for j in range(n):
            if j and math.gcd(n, j) != 1:
                continue
            if n > 1 and j == 0:
                continue
            target = math.degrees(wrap_pi(-2.0 * math.pi * j / n))
            found = closures(depth, excess, retrograde, target, "ceres", revolutions)
            if found:
                out.append(Ring(n, j, found[0]))
    return sorted(out, key=lambda ring: ring.route.departure_dv)


def disk_route(
    ship: SunRoute, depot_deg: float, retrograde: Optional[bool] = None
) -> Optional[SunRoute]:
    """The disks' cheapest mirror-sense dive from a depot onto the ship, head-on.

    The disks leave a depot ``depot_deg`` ahead of the ship's (any time: the
    flow is steady) and reach perihelion where the ship does, against its
    motion.  With the depots co-orbiting, that needs ``depot +
    sweep_disk - n (T_ship - T_disk) = sweep_ship`` in the ship's sense,
    where the sweeps run to perihelion.  ``retrograde`` is the disks' sense,
    against the ship's by default.
    """
    sense = (not ship.retrograde) if retrograde is None else retrograde
    n = mean_motion(CERES_A) * _YEAR
    ship_sweep = math.degrees(
        math.atan2(*_perihelion_direction(ship)[::-1])
    )  # inertial
    best: Optional[SunRoute] = None

    def residual(route: SunRoute) -> float:
        sweep = math.degrees(math.atan2(*_perihelion_direction(route)[::-1]))
        landed = (
            depot_deg + sweep + math.degrees(n * (ship.sun_years - route.sun_years))
        )
        return math.degrees(wrap_pi(math.radians(landed - ship_sweep)))

    for inbound in (True, False):
        for revs in REVS:
            routes = []
            for aphelion in np.linspace(*APHELION_SCAN, _SCAN_POINTS):
                try:
                    routes.append(
                        (
                            float(aphelion),
                            sun_route(
                                ship.depth,
                                ship.excess,
                                sense,
                                float(aphelion),
                                inbound,
                                revs,
                            ),
                        )
                    )
                except ValueError:
                    continue
            for (r1, a), (r2, b) in zip(routes, routes[1:]):
                va, vb = residual(a), residual(b)
                if va * vb > 0 or abs(va - vb) > 90.0:
                    continue
                r_a = float(
                    brentq(
                        lambda x: residual(
                            sun_route(ship.depth, ship.excess, sense, x, inbound, revs)
                        ),
                        r1,
                        r2,
                        xtol=1.0,
                    )
                )
                route = sun_route(ship.depth, ship.excess, sense, r_a, inbound, revs)
                if best is None or route.departure_dv < best.departure_dv:
                    best = route
    return best


def _perihelion_direction(route: SunRoute) -> Tuple[float, float]:
    """Unit vector to the route's perihelion, in the departure frame."""
    r_p = route.depth * R_SUN
    r0 = np.array([CERES_A, 0.0])
    v0 = dive_velocity(
        CERES_A,
        r_p,
        route.aphelion,
        route.retrograde,
        route.inbound and route.aphelion > CERES_A,
    )
    rp, _, _ = coast(r0, v0, 0.0, route.revs)
    return float(rp[0] / np.linalg.norm(rp)), float(rp[1] / np.linalg.norm(rp))


# --------------------------------------------------------------------------
# 4-5. The steady ledger
# --------------------------------------------------------------------------

#: The departure the Earth hardware flies to Ceres: a Hohmann transfer.
HOHMANN_DEPARTURE = math.sqrt(MU_SUN / EARTH_A) * (
    math.sqrt(2.0 * CERES_A / (EARTH_A + CERES_A)) - 1.0
)
HOHMANN_ARRIVAL = math.sqrt(MU_SUN / CERES_A) * (
    1.0 - math.sqrt(2.0 * EARTH_A / (EARTH_A + CERES_A))
)
#: The freight's departure bowl sees its stream at the visible sign's
#: middle (the solar dive flies 117-140 deg); its push and departure knobs
#: are the solar dive's 4 R_sun, X 90 headline optimum (ADR 0077).
FREIGHT_ANGLE = 130.0
FREIGHT_DESIGN = Design(k_push=7.55, k_earth=12.25)
#: Water NTR on the depot tug, the Iapetus cycle's (ADR 0078).
TUG_ISP = 200.0


@dataclass(frozen=True)
class CeresPrices:
    """Every price the Ceres loop is charged, dollars per kilogram unless noted.

    The shared lines are S53's best case (ADR 0082): the lob at its
    propellant floor, snow-foam Earth charges, Earth plates flown 100 times
    and steering packages at $1.  The Ceres lines are the author's
    (2026-10-22).

    Attributes:
        water: Ceres polar water.
        disk: Ceres-made graphite Sun disks.
        structure: Ceres-made structure: shade, carrier, mounts, carbon foam.
        polyethylene: Ceres-made Sun charges.
        film: Ceres film: the Earth sheets (Kapton-grade, a hypothesis) and
            the Ceres plates' ablative film.
        earth_hardware: Earth-made plates, bowls, bags and avionics, before
            freight.
        freight: Earth to a Ceres depot, per kilogram; None prices the leg
            (:func:`freight_leg`).
        package: Per steering package.
        packages_per_kg: Packages per kilogram of stream (the paper's two per
            60 kg PuffSat).
        lob: Per kilogram lofted on the book's lob axis; None is the
            propellant floor (:func:`src.solar_dive_growth.lob_floor`).
        lob_climb: The child's lob climb factor (MATURE's).
        earth_charge: Earth-side charges: snow foam.
        earth_film: The Earth plates' film (MATURE's).
        methalox: MATURE's.
        earth_mounts: MATURE's.
        earth_plate_uses: Pushes per Earth plate (delivery and freight).
        ceres_plate_uses: Pushes per recovered Ceres plate.
        misses: Stream that misses its bowl (the solar dive's floor).
        tug_years: The depot tug's lead time.
    """

    water: float = 0.02
    disk: float = 30.0
    structure: float = 10.0
    polyethylene: float = 1.0
    film: float = 3.0
    earth_hardware: float = 30.0
    freight: Optional[float] = None
    package: float = STEERING_PACKAGE_BEST
    packages_per_kg: float = MATURE.packages_per_kg
    lob: Optional[float] = None
    lob_climb: float = MATURE.lob_climb
    earth_charge: float = SNOW_FOAM
    earth_film: float = MATURE.film
    methalox: float = MATURE.methalox
    earth_mounts: float = MATURE.mounts
    earth_plate_uses: float = 100.0
    ceres_plate_uses: float = 100.0
    misses: float = MISS_FLOOR
    tug_years: float = 20.0

    @property
    def lob_price(self) -> float:
        """Per kilogram lofted on the book's lob axis."""
        return lob_floor() if self.lob is None else self.lob


CERES_BOOK = CeresPrices()


@dataclass(frozen=True)
class CeresDesign:
    """The knobs and the stated assumptions of one Ceres loop.

    Attributes:
        depth: Perihelion (solar radii).
        ceres_excess: The Ceres-bound stream's solar excess, ``X_C`` (km/s).
        earth_stream: The Earth-bound stream's speed at 400 km (km/s); its
            excess ``X_E`` is solved for it.
        k_ship: Water per pellet on the ship's Ceres push; None flies the
            fewest pellets (:func:`best_push`).
        k_disk: The same for the disks' push.
        k_sun: Polyethylene per kg of disk at the Sun.
        arc_hours: The Sun boost's arc.
        k_delivery: The L1 delivery's slug ratio.
        push_hours: The Ceres push's length.
        push_rate: Ceres pellets a second.
        plate: The Ceres pusher plate: ``recover`` (pushed back by the
            stream and reused), ``expend`` (dives with the ship) or
            ``sun_bowl`` (rides on and is the Sun bowl).
        ship_mass: The ship after its Ceres push (kg), plate excluded.
        closure: ``ring`` (co-orbital depots) or ``single`` (one depot, the
            ship's cheapest closing dive in ``retrograde`` sense).
        retrograde: The ship's dive sense (``single`` only).
        reserve: Dead mass at the Sun not otherwise charged, per kg of ship.
        recovered: Share of water's toll the expansion returns.
        law: :func:`ceres_ve`'s law.
        clamp_angle: :func:`ceres_ve`'s.
        k_cap: The Ceres pushes' slug-ratio cap.
        stream_guidance: Charge steering packages on the Ceres stream.
        child_delivery: Fly the L1 delivery at the child's 4 Hz and 3.26 g
            from its 1.1 km/s lob (the solar dive's ``slow_delivery`` False;
            the sheets' survival there is the simulation's open S37).
        spring: Every plate's gas-spring knobs (the solar dive's).
    """

    depth: float = 4.0
    ceres_excess: float = 150.0
    earth_stream: float = 137.0
    k_ship: Optional[float] = None
    k_disk: Optional[float] = None
    k_sun: float = 20.0
    arc_hours: float = 2.2
    k_delivery: float = 5.0
    push_hours: float = 8.0
    push_rate: float = 0.5
    plate: str = "recover"
    ship_mass: float = 1.0e6
    closure: str = "ring"
    retrograde: bool = False
    reserve: float = 0.0
    recovered: float = 0.0
    law: str = "hole"
    clamp_angle: bool = False
    k_cap: float = K_CAP
    stream_guidance: bool = True
    child_delivery: bool = False
    spring: Design = field(default_factory=Design)


@dataclass(frozen=True)
class Routes:
    """The dives one design flies.

    Attributes:
        ship: The Ceres-bound ship's closing dive (its stream).
        disks: Its disks' head-on dive from the same depot.
        earth: The Earth-bound ship's dive (direct, to 1 AU at ``X_E``).
        earth_disks: Its disks'.
        ring: The depots (None for a single depot).
        earth_excess: ``X_E`` (km/s).
    """

    ship: SunRoute
    disks: SunRoute
    earth: SunRoute
    earth_disks: SunRoute
    ring: Optional[Ring]
    earth_excess: float


_ROUTES: Dict[Tuple[float, float, float, str, bool], Routes] = {}


def routes_for(design: CeresDesign) -> Routes:
    """The design's dives (cached by depth, excesses and closure)."""
    key = (
        design.depth,
        design.ceres_excess,
        design.earth_stream,
        design.closure,
        design.retrograde,
    )
    if key in _ROUTES:
        return _ROUTES[key]
    ring: Optional[Ring] = None
    if design.closure == "ring":
        ring = rings(design.depth, design.ceres_excess, revolutions=(0,))[0]
        ship = ring.route
    elif design.closure == "single":
        found = closures(
            design.depth, design.ceres_excess, design.retrograde, revolutions=(0,)
        )
        if not found:
            raise ValueError("no single-depot closure at this depth and excess")
        ship = found[0]
    else:
        raise ValueError(f"unknown closure {design.closure!r}")
    disks = disk_route(ship, 0.0)
    if disks is None:
        raise ValueError("the disks cannot meet the ship from its depot")
    x_e = earth_excess_for(design.depth, design.earth_stream)
    earth = sun_route(design.depth, x_e, crossing=EARTH_A)
    earth_disks = sun_route(design.depth, x_e, retrograde=True, crossing=EARTH_A)
    found_routes = Routes(ship, disks, earth, earth_disks, ring, x_e)
    _ROUTES[key] = found_routes
    return found_routes


@dataclass(frozen=True)
class ShipLedger:
    """One ship from its depot to its released stream, per ship.

    Attributes:
        mass: Every mass line (kg) and the leg's speeds.
        dollars: Every dollar line except the Ceres stream it spends.
        stream: Ceres stream it spends (kg): its push, its disks', the plate's
            return.
        cargo: Stream it releases (kg).
        earth_hardware: Earth-made kilograms it spends (freight-bearing).
    """

    mass: Dict[str, float]
    dollars: Dict[str, float]
    stream: float
    cargo: float
    earth_hardware: float


def _ceres_plate(
    pushed: float, push: CeresPush, design: CeresDesign
) -> Tuple[float, float, float]:
    """The Ceres plate's mass, its bags and the bore collar (kg) for ``pushed`` kg.

    The spring law at the push's mean force, held at the paraboloid's
    point-blast floor for the pellet's energy (the solar dive's bowl).
    """
    force = push.water * pushed * push.ve * 1e3 / (design.push_hours * _HOUR)
    pellet = force / (design.push_rate * push.k * push.ve * 1e3)
    floor = _plate_for(
        pellet, SUN_PLATE_PULSE, SUN_PLATE_AREA, push.closing / SUN_PLATE_SPEED
    )
    spring = gas_spring(force, design.push_rate, pushed, design.spring, floor)
    return spring.plate, spring.bag, BORE_COLLAR


def _push(
    route: SunRoute, stream: SunRoute, design: CeresDesign, k: Optional[float]
) -> CeresPush:
    w, theta = stream_at(stream, route)
    return best_push(
        route.departure_dv,
        w,
        theta,
        design.k_cap,
        design.recovered,
        design.law,
        design.clamp_angle,
        k,
    )


def _return_push(route: SunRoute, stream: SunRoute, design: CeresDesign) -> CeresPush:
    """The plate's push back to its depot after it has pushed the ship.

    It flies the ship's departure velocity and must lose it; the stream comes
    in relative to that velocity.
    """
    r = np.array(stream.position)
    rn = float(np.linalg.norm(r))
    radial = r / rn
    along = np.array([-radial[1], radial[0]])
    v_local = np.array(
        [
            float(np.array(stream.velocity) @ radial),
            float(np.array(stream.velocity) @ along),
        ]
    )
    depot = np.array([0.0, circular_speed(rn)])
    plate = depot + route.departure_dv * np.array(route.thrust)
    rel = v_local - plate
    w = float(np.linalg.norm(rel))
    back = -np.array(route.thrust)
    theta = math.degrees(math.acos(max(-1.0, min(1.0, float(rel @ back) / w))))
    return best_push(
        route.departure_dv,
        w,
        theta,
        design.k_cap,
        design.recovered,
        design.law,
        design.clamp_angle,
    )


def tug_per_kg(ring: Optional[Ring], prices: CeresPrices) -> float:
    """Tug water per kilogram the ring's depots are fed from Ceres (kg)."""
    if ring is None or ring.depots == 1:
        return 0.0
    dv = tug_dv(ring.mean_offset_deg, prices.tug_years)
    return math.exp(dv / (TUG_ISP * _G0)) - 1.0


def fly_ship(
    design: CeresDesign,
    prices: CeresPrices,
    bound: str,
    freight: float,
    reference_ve: float,
) -> ShipLedger:
    """One ship of ``design.ship_mass`` from its depot to its stream.

    Args:
        design: The knobs.
        prices: The book.
        bound: ``ceres`` (ice pellets back to the ring) or ``earth`` (Ceres
            film sheets to Earth).
        freight: Earth to Ceres, per kilogram of Earth hardware.
        reference_ve: The Earth push's exhaust speed, against which the Sun
            plate's film is scaled (the solar dive's convention).

    Returns:
        The ledger.
    """
    flown = routes_for(design)
    route, disk_route_ = (
        (flown.ship, flown.disks)
        if bound == "ceres"
        else (flown.earth, flown.earth_disks)
    )
    laws = StageLaws.load()
    m: Dict[str, float] = {}
    ship_mass = design.ship_mass
    # --- The Sun: the solar dive's boost on the arc.
    w_sun = sun_closing(route, disk_route_)
    ve_sun = laws.sun_ve(design.k_sun, w_sun)
    tau = route.depth * R_SUN / route.periapsis_speed / _HOUR
    loss = (design.arc_hours / tau) ** 2 * (1.0 / 96.0 + 1.0 / 24.0)
    if loss >= 0.5:
        raise ValueError("the Sun arc is too long for its quadratic loss")
    boost = route.boost / (1.0 - loss)
    mr_sun = math.exp(boost / ve_sun)
    film_rate = FILM_PER_IMPULSE * ve_sun / reference_ve
    film_per = film_rate * (1.0 - 1.0 / mr_sun) * ve_sun * 1e3
    ship = ship_mass / (1.0 + film_per)
    sun_charge = ship * (1.0 - 1.0 / mr_sun)
    disks = sun_charge / design.k_sun
    final = ship / mr_sun
    pulses = SUN_PULSE_RATE * design.arc_hours * _HOUR
    pulse_disk = disks / pulses
    sun_spring = gas_spring(
        pulse_disk * design.k_sun * ve_sun * 1e3 * SUN_PULSE_RATE,
        SUN_PULSE_RATE,
        final,
        design.spring,
        _plate_for(
            pulse_disk, SUN_PLATE_PULSE, SUN_PLATE_AREA, w_sun / SUN_PLATE_SPEED
        ),
    )
    if sun_spring.plate >= final or sun_spring.bag < 0.0:
        raise ValueError("the Sun bowl's survival floor outweighs the ship it pushes")
    # --- The Ceres push.
    push = _push(route, flown.ship, design, design.k_ship)
    plate, bags, collar = 0.0, 0.0, 0.0
    for _ in range(6):
        pushed = ship_mass + (
            0.0 if design.plate == "sun_bowl" else plate + bags + collar
        )
        plate, bags, collar = _ceres_plate(pushed, push, design)
    sun_plate, sun_bags = sun_spring.plate, sun_spring.bag
    if design.plate == "sun_bowl":
        sun_plate = max(sun_plate, plate)
        sun_bags = max(sun_bags, bags)
        carried = 0.0
    else:
        carried = plate + bags + collar
    pushed = ship_mass + carried
    dead = {
        "sun_plate": sun_plate + sun_bags,
        "bore_collar": BORE_COLLAR,
        "shield": SHIELD_FRACTION * ship,
        "avionics": AVIONICS_FRACTION * ship,
        "sun_mounts": CHARGE_MOUNT_FRACTION * sun_charge,
        "reserve": design.reserve * ship_mass,
    }
    cargo = final - sum(dead.values())
    if cargo <= 0.0:
        raise ValueError("the Sun leg returns nothing")
    water = push.water * pushed
    stream = push.stream * pushed
    film_ceres = FILM_PER_IMPULSE * water * push.ve * 1e3
    # --- The disks: their own stack and plate, onto the mirror dive.
    disk_push = _push(disk_route_, flown.ship, design, design.k_disk)
    carrier = CARRIER_DRY_FRACTION * disks
    d_plate, d_bags, d_collar = 0.0, 0.0, 0.0
    for _ in range(6):
        d_pushed = disks + carrier + d_plate + d_bags + d_collar
        d_plate, d_bags, d_collar = _ceres_plate(d_pushed, disk_push, design)
    d_carried = d_plate + d_bags + d_collar
    d_pushed = disks + carrier + d_carried
    d_water = disk_push.water * d_pushed
    d_stream = disk_push.stream * d_pushed
    film_ceres += FILM_PER_IMPULSE * d_water * disk_push.ve * 1e3
    # --- The plates after the push: recovered, expended, or the Sun bowl.
    hardware_price = prices.earth_hardware + freight
    plate_dollars = 0.0
    back_water = back_stream = 0.0
    earth_kg = dead["sun_plate"] + dead["bore_collar"] + dead["avionics"]
    if design.plate == "recover":
        back = _return_push(route, flown.ship, design)
        d_back = _return_push(disk_route_, flown.ship, design)
        back_water = back.water * carried + d_back.water * d_carried
        back_stream = back.stream * carried + d_back.stream * d_carried
        spent = (carried + d_carried) / prices.ceres_plate_uses
        film_ceres += (
            FILM_PER_IMPULSE
            * (back.water * carried * back.ve + d_back.water * d_carried * d_back.ve)
            * 1e3
        )
    elif design.plate in ("expend", "sun_bowl"):
        spent = carried + d_carried
    else:
        raise ValueError(f"unknown plate policy {design.plate!r}")
    plate_dollars = spent * hardware_price
    earth_kg += spent
    charges = water + d_water + back_water
    mounts = CHARGE_MOUNT_FRACTION * charges
    ceres_made = (
        charges
        + mounts
        + film_ceres
        + cargo
        + sun_charge
        + disks
        + carrier
        + dead["shield"]
        + dead["sun_mounts"]
        + dead["reserve"]
    )
    tug = tug_per_kg(flown.ring, prices)
    packages = (
        cargo * prices.packages_per_kg * prices.package
        if (bound == "earth" or design.stream_guidance)
        else 0.0
    )
    dollars = {
        "water_charges": prices.water * charges,
        "charge_mounts": prices.structure * mounts,
        "ceres_film": prices.film * film_ceres,
        "ceres_plates": plate_dollars,
        "disks": prices.disk * disks,
        "carrier": prices.structure * carrier,
        "sun_charges": prices.polyethylene * sun_charge,
        "sun_bowl": hardware_price * (dead["sun_plate"] + dead["bore_collar"]),
        "shield": prices.structure * dead["shield"],
        "avionics": hardware_price * dead["avionics"],
        "sun_mounts": prices.structure * dead["sun_mounts"],
        "sun_film": prices.film * film_per * ship,
        "reserve": prices.structure * dead["reserve"],
        "cargo": (prices.water if bound == "ceres" else prices.film) * cargo,
        "packages": packages,
        "depot_tug": prices.water * tug * ceres_made,
    }
    m.update(dead)
    m.update(
        ship=ship,
        pushed=pushed,
        ceres_plate=plate,
        ceres_bags=bags,
        disk_plate=d_plate + d_bags,
        water=water,
        disk_water=d_water,
        back_water=back_water,
        stream=stream,
        disk_stream=d_stream,
        back_stream=back_stream,
        disks=disks,
        carrier=carrier,
        sun_charge=sun_charge,
        film_ceres=film_ceres,
        film_sun=film_per * ship,
        cargo=cargo,
        k=push.k,
        ve=push.ve,
        closing=push.closing,
        theta=push.theta,
        departure_dv=route.departure_dv,
        k_disk=disk_push.k,
        disk_dv=disk_route_.departure_dv,
        boost=boost,
        mr_sun=mr_sun,
        ve_sun=ve_sun,
        w_sun=w_sun,
        tug=tug,
        pulse_pellet=push.water
        * pushed
        * push.ve
        * 1e3
        / (design.push_hours * _HOUR * design.push_rate * push.k * push.ve * 1e3),
    )
    return ShipLedger(m, dollars, stream + d_stream + back_stream, cargo, earth_kg)


@dataclass(frozen=True)
class Freight:
    """Earth hardware's trip to a Ceres depot, per kilogram arriving.

    Attributes:
        earth_dollars: The lob, charges and plates per kg departing.
        earth_sheets: Earth stream per kg departing (kg).
        capture_dollars: The water and plate at the Ceres capture.
        capture_stream: Ceres stream per kg captured (kg).
        lofted: Kilograms lofted per kilogram arriving.
    """

    earth_dollars: float
    earth_sheets: float
    capture_dollars: float
    capture_stream: float
    lofted: float

    def price(self, c_c: float, c_e: float) -> float:
        """Dollars per kilogram arriving at the depot."""
        return (
            self.earth_dollars
            + self.earth_sheets * c_e
            + self.capture_dollars
            + self.capture_stream * c_c
        )


@functools.lru_cache(maxsize=None)
def _freight_departure(v_inf: float, w: float) -> EarthLeg:
    """The solar dive's Earth leg to the Hohmann excess on an Earth stream.

    The burn (1.8 km/s from the 10-day orbit) is below the solar dive's loss
    table, so its fixed-direction loss is integrated directly.
    """
    days = 10.0

    def loss(burn: float, seconds: float) -> float:
        from astropy import units as u

        from src.finite_burn_loss import TABLE_EXHAUST, finite_burn_loss

        return float(
            finite_burn_loss(
                burn * u.km / u.s,
                TABLE_EXHAUST,
                seconds * u.s,
                steered=False,
                period=days * u.day,
            ).to_value(u.km / u.s)
        )

    geometry = DepartureGeometry(0.0, FREIGHT_ANGLE, wave_speed(v_inf, 600.0), 90.0)
    return fly_earth(
        w,
        0.0,
        geometry,
        HOHMANN_DEPARTURE,
        days,
        FREIGHT_DESIGN,
        StageLaws.load(),
        loss,
    )


def freight_leg(design: CeresDesign, prices: CeresPrices) -> Freight:
    """The paper's departure to the Hohmann excess, then a Ceres-stream capture.

    The Earth half is the solar dive's own leg (:func:`fly_earth`): a 1500 t
    unit pushed at 400 km by the Earth stream, reversed at apoapsis, and
    departed from the 10-day orbit's 600 km periapsis on the bowl.  Its
    stream is the Earth-bound one; the bowl sees it at
    :data:`FREIGHT_ANGLE`.  At Ceres the freight is slower than the depot by
    the Hohmann arrival excess, and the Ceres stream pushes it up to the
    depot on a recovered plate.
    """
    flown = routes_for(design)
    leg = _freight_departure(*earth_stream(flown.earth))
    lines = leg.lines
    departing = leg.departing
    lob = prices.lob_price * prices.lob_climb * lines["lob_climb"] * LAUNCH_UNIT
    earth = (
        lob
        + prices.earth_charge * (lines["push_charge"] + lines["departure_charge"])
        + prices.earth_hardware * lines["bowl"]
        + prices.earth_film * lines["film_earth"]
        + prices.methalox * lines["methalox"]
        + prices.earth_mounts * lines["earth_mounts"]
    ) / departing
    sheets = (leg.push_sheets + leg.departure_sheets) / departing
    # The capture: the stream overtakes the slower freight from behind and
    # pushes it forward along the track.
    r = np.array(flown.ship.position)
    rn = float(np.linalg.norm(r))
    radial = r / rn
    along = np.array([-radial[1], radial[0]])
    v = np.array(flown.ship.velocity)
    local = np.array([float(v @ radial), float(v @ along)])
    rel = local - np.array([0.0, circular_speed(rn) - HOHMANN_ARRIVAL])
    w_c = float(np.linalg.norm(rel))
    theta = math.degrees(math.acos(float(rel[1]) / w_c))
    capture = best_push(
        HOHMANN_ARRIVAL, w_c, theta, design.k_cap, design.recovered, design.law
    )
    plate, bags, collar = _ceres_plate(1.0e6, capture, design)
    capture_dollars = (
        prices.water + CHARGE_MOUNT_FRACTION * prices.structure
    ) * capture.water + prices.earth_hardware * (
        plate + bags + collar
    ) / 1.0e6 / prices.ceres_plate_uses
    return Freight(
        earth_dollars=earth,
        earth_sheets=sheets,
        capture_dollars=capture_dollars,
        capture_stream=capture.stream,
        lofted=LAUNCH_UNIT / departing,
    )


@dataclass(frozen=True)
class SteadyLoop:
    """The Ceres loop in the steady state, and the price at L1.

    Attributes:
        c_ceres: Dollars per kilogram of Ceres stream caught.
        c_earth: Dollars per kilogram of Earth stream caught.
        freight: Earth to a depot, per kilogram.
        lines: $/kg at L1 by line; the Ceres-side lines are carried into the
            Earth stream's price and reported apart (``ceres_lines``).
        ceres_lines: The Earth stream's $/kg at L1, split by the Earth-bound
            ship's lines.
        growth: Ceres stream out over Ceres stream in, per loop.
        loop_years: Departure to the stream's return.
        doubling_years: The Ceres loop's doubling time.
        ceres_ship: The Ceres-bound ship's ledger.
        earth_ship: The Earth-bound ship's ledger.
        delivery: The L1 delivery's masses.
        lofted: Kilograms lofted per kilogram at L1.
    """

    c_ceres: float
    c_earth: float
    freight: float
    lines: Dict[str, float]
    ceres_lines: Dict[str, float]
    growth: float
    loop_years: float
    doubling_years: float
    ceres_ship: ShipLedger
    earth_ship: ShipLedger
    delivery: Dict[str, float]
    lofted: float

    @property
    def price(self) -> float:
        """Steady $/kg at L1."""
        return float(sum(self.lines.values()))


def steady_loop(design: CeresDesign, prices: CeresPrices = CERES_BOOK) -> SteadyLoop:
    """Solve both stream prices as fixed points and price a kilogram at L1.

    The Ceres stream's price is what a Ceres-bound ship costs per kilogram
    of stream it returns, its own pellets included: ``c_C (Y - S) = O``.  The
    Earth stream's is the Earth-bound ship's cost per kilogram it returns.
    Freight depends on both, so the three are iterated.

    Args:
        design: The knobs.
        prices: The book.

    Returns:
        The steady loop.

    Raises:
        ValueError: If the Ceres loop does not grow.
    """
    flown = routes_for(design)
    laws = StageLaws.load()
    v_inf, w_e = earth_stream(flown.earth)
    earth_design = replace(
        design.spring,
        k_delivery=design.k_delivery,
        slow_delivery=not design.child_delivery,
    )
    delivery = delivery_unit(v_inf, earth_design, laws)
    reference_ve = laws.push_ve(
        design.k_delivery, w_e, sheet_material(w_e), earth_design.push_law
    )
    leg = freight_leg(design, prices) if prices.freight is None else None
    caught = 1.0 - prices.misses
    # The masses do not depend on the prices, and each ship's dollars are
    # linear in the freight on its Earth hardware: fly once, iterate scalars.
    c_free = fly_ship(design, prices, "ceres", 0.0, reference_ve)
    e_free = fly_ship(design, prices, "earth", 0.0, reference_ve)
    net = c_free.cargo * caught - c_free.stream
    if net <= 0.0:
        raise ValueError("the Ceres loop does not grow")
    # Three linear conditions on c_C, c_E and the freight F:
    #   c_C net = O_C + H_C F;  c_E Y_E = O_E + H_E F + S_E c_C;
    #   F = F_0 + F_E c_E + F_C c_C (or F fixed by the book).
    y_e = e_free.cargo * caught
    if leg is not None:
        f_0 = leg.earth_dollars + leg.capture_dollars
        f_e, f_c = leg.earth_sheets, leg.capture_stream
    else:
        f_0, f_e, f_c = float(prices.freight or 0.0), 0.0, 0.0
    matrix = np.array(
        [
            [net, 0.0, -c_free.earth_hardware],
            [-e_free.stream, y_e, -e_free.earth_hardware],
            [-f_c, -f_e, 1.0],
        ]
    )
    rhs = np.array([sum(c_free.dollars.values()), sum(e_free.dollars.values()), f_0])
    c_c, c_e, freight = (float(x) for x in np.linalg.solve(matrix, rhs))
    if min(c_c, c_e, freight) <= 0.0:
        raise ValueError(
            "the stream prices do not close: the loop feeds itself too little"
        )
    c_ship = fly_ship(design, prices, "ceres", freight, reference_ve)
    e_ship = fly_ship(design, prices, "earth", freight, reference_ve)
    cargo = delivery.cargo
    lob = prices.lob_price * prices.lob_climb
    sheets_per_kg = delivery.sheets / cargo
    lines = {
        "delivery_lob": lob * delivery.climb * LAUNCH_UNIT / cargo,
        "delivery_charge": prices.earth_charge * delivery.charge / cargo,
        "delivery_plate": prices.earth_hardware
        * (delivery.plate + delivery.bag)
        / prices.earth_plate_uses
        / cargo,
        "earth_stream": sheets_per_kg * c_e,
    }
    scale = sheets_per_kg / (e_ship.cargo * caught)
    ceres_lines = {key: value * scale for key, value in e_ship.dollars.items()}
    ceres_lines["ceres_stream"] = e_ship.stream * c_c * scale
    growth = c_ship.cargo * caught / c_ship.stream
    years = flown.ship.years
    return SteadyLoop(
        c_ceres=c_c,
        c_earth=c_e,
        freight=freight,
        lines=lines,
        ceres_lines=ceres_lines,
        growth=growth,
        loop_years=years,
        doubling_years=years * math.log(2.0) / math.log(growth),
        ceres_ship=c_ship,
        earth_ship=e_ship,
        delivery={
            "cargo": cargo,
            "charge": delivery.charge,
            "sheets": delivery.sheets,
            "plate": delivery.plate + delivery.bag,
            "climb": delivery.climb,
            "stream_400": w_e,
        },
        lofted=LAUNCH_UNIT / cargo,
    )


# --------------------------------------------------------------------------
# The optimizer
# --------------------------------------------------------------------------

#: Search bounds (recorded so the optimum reproduces; CLAUDE.md): the solar
#: dive's own (K_SUN, ARC_HOURS, the push law's k floor of 5).
CERES_BOUNDS = {
    "k_sun": (20.0, 100.0),
    "arc_hours": (0.5, 12.0),
    "k_delivery": (5.0, 40.0),
}
SUN_GRID = (20.0, 25.0, 29.907, 36.57, 44.72, 60.0)
ARC_GRID = (1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0)
CERES_EXCESSES = (60.0, 75.0, 90.0, 110.0, 130.0, 150.0, 175.0, 200.0, 250.0, 300.0)
DEPTHS = (4.0, 9.86)
#: The Earth stream each depth holds: the paper's speeds at 400 km.
EARTH_STREAMS = {4.0: 137.0, 9.86: 99.0}


def optimize_loop(
    design: CeresDesign, prices: CeresPrices = CERES_BOOK
) -> Tuple[CeresDesign, SteadyLoop]:
    """The cheapest steady $/kg at L1 over the Sun's ``k`` and arc and the delivery's ``k``.

    Powell's method on the box :data:`CERES_BOUNDS`, from the best point of
    a coarse grid (:data:`SUN_GRID` x :data:`ARC_GRID`).  The Ceres pushes fly
    their fewest-pellet ``k``.

    Args:
        design: The knobs to start from and every stated assumption.
        prices: The book.

    Returns:
        The best design and its loop.
    """
    names = tuple(CERES_BOUNDS)

    def build(x: Sequence[float]) -> CeresDesign:
        return replace(design, **dict(zip(names, (float(v) for v in x))))  # type: ignore[arg-type]

    best: List[Tuple[float, CeresDesign]] = []

    def price(x: Sequence[float]) -> float:
        trial = build(x)
        try:
            value = steady_loop(trial, prices).price
        except (ValueError, ZeroDivisionError):
            return 1e6
        if not best or value < best[0][0]:
            best[:] = [(value, trial)]
        return value

    # A coarse grid first: the Sun law is interpolated piecewise, so the
    # surface has kinks at its nodes.
    grid = [
        (price([k, arc, design.k_delivery]), [k, arc, design.k_delivery])
        for k in SUN_GRID
        for arc in ARC_GRID
    ]
    start = min(grid, key=lambda row: row[0])[1]
    minimize(
        price,
        start,
        method="Powell",
        bounds=[CERES_BOUNDS[name] for name in names],
        options={"xtol": 1e-3, "ftol": 1e-6},
    )
    if not best:
        raise ValueError("no flyable design near the start")
    return best[0][1], steady_loop(best[0][1], prices)


def excess_sweep(
    depth: float,
    prices: CeresPrices = CERES_BOOK,
    excesses: Sequence[float] = CERES_EXCESSES,
    base: Optional[CeresDesign] = None,
) -> List[Tuple[CeresDesign, SteadyLoop]]:
    """The optimum at each Ceres-stream excess, the Earth stream held."""
    out = []
    for x_c in excesses:
        start = replace(
            base or CeresDesign(),
            depth=depth,
            ceres_excess=x_c,
            earth_stream=EARTH_STREAMS[depth],
        )
        try:
            out.append(optimize_loop(start, prices))
        except (ValueError, ZeroDivisionError):
            continue
    return out


# --------------------------------------------------------------------------
# What the coplanar circular orbit hides (item 2) and the other closures (3)
# --------------------------------------------------------------------------


def eccentric_dive(
    depth: float, excess: float, mean_anomaly_deg: float
) -> Tuple[float, float]:
    """A direct prograde dive from Ceres' real (eccentric) orbit.

    Ceres is placed at ``mean_anomaly_deg``; the ship's dive ellipse has its
    aphelion there, tangential.  The stream's landing is measured against
    Ceres' own position when the stream reaches its radius (three passes).

    Returns:
        ``(departure dv km/s, miss deg)``.
    """
    from src.conic_kernel import kepler_propagate

    a, e = CERES_A, CERES_E
    n = mean_motion(a)
    m0 = math.radians(mean_anomaly_deg)
    big_e = float(brentq(lambda x: x - e * math.sin(x) - m0, -10.0, 10.0))
    nu = 2.0 * math.atan2(
        math.sqrt(1 + e) * math.sin(big_e / 2), math.sqrt(1 - e) * math.cos(big_e / 2)
    )
    p = a * (1 - e * e)
    r = p / (1 + e * math.cos(nu))
    pos = r * np.array([math.cos(nu), math.sin(nu)])
    h = math.sqrt(MU_SUN * p)
    vel = np.array([-MU_SUN / h * math.sin(nu), MU_SUN / h * (e + math.cos(nu))])
    radial = pos / r
    along = np.array([-radial[1], radial[0]])
    v_a = float(dive_velocity(r, depth * R_SUN, r, False, False)[1])
    ship_v = v_a * along
    dv = float(np.linalg.norm(ship_v - vel))
    rp, vp, t_in = coast(pos, ship_v, 0.0)
    speed = math.sqrt(excess**2 + 2.0 * MU_SUN / float(np.linalg.norm(rp)))
    v_new = speed * vp / np.linalg.norm(vp)
    target = r
    for _ in range(4):
        r1, _, t_out = coast_to_radius(rp, v_new, target)
        c_pos, _ = kepler_propagate(pos, vel, t_in + t_out, MU_SUN)
        target = float(np.linalg.norm(c_pos))
    miss = math.atan2(r1[1], r1[0]) - math.atan2(c_pos[1], c_pos[0])
    return dv, math.degrees(wrap_pi(miss))


def tilted_dive_dv(depth: float, tilt_deg: float) -> float:
    """A direct dive whose plane is tilted ``tilt_deg`` from Ceres' (km/s).

    The dive ellipse's aphelion speed is turned out of Ceres' plane about
    the radius: ``dv^2 = v_C^2 + v_a^2 - 2 v_C v_a cos(tilt)``.
    """
    v_c = circular_speed(CERES_A)
    v_a = float(dive_velocity(CERES_A, depth * R_SUN, CERES_A, False, False)[1])
    tilt = math.radians(tilt_deg)
    return math.sqrt(v_c * v_c + v_a * v_a - 2.0 * v_c * v_a * math.cos(tilt))


def earth_plane_tilt(route: SunRoute) -> float:
    """The worst tilt an Earth-bound dive needs from Ceres' plane (deg).

    The dive's plane holds the Sun, the departure point and Earth at the
    stream's arrival.  Earth lies in the ecliptic, at most Ceres'
    inclination out of Ceres' plane, and ``psi`` (the route's sweep) round
    from the departure point: ``tan(tilt) = tan(i) / |sin(psi)|``.
    """
    psi = math.radians(route.sweep_deg)
    return math.degrees(
        math.atan(math.tan(math.radians(CERES_I)) / max(abs(math.sin(psi)), 1e-9))
    )


def mars_flyby_turn(depth: float, altitude: float = 300.0) -> Tuple[float, float]:
    """A dive ellipse's excess at Mars' orbit and the turn a Mars flyby gives.

    Returns:
        ``(v_inf km/s, turn deg)`` at ``altitude`` km.
    """
    r_p = depth * R_SUN
    a = (r_p + CERES_A) / 2.0
    p = 2.0 * r_p * CERES_A / (r_p + CERES_A)
    v_t = math.sqrt(MU_SUN * p) / MARS_A
    v_r = math.sqrt(MU_SUN * (2.0 / MARS_A - 1.0 / a) - v_t * v_t)
    v_inf = math.hypot(v_r, v_t - circular_speed(MARS_A))
    e = 1.0 + (R_MARS + altitude) * v_inf**2 / MU_MARS
    return v_inf, math.degrees(2.0 * math.asin(1.0 / e))


def boost_rotation(
    depth: float, excess: float, offsets: Sequence[float]
) -> List[Tuple[float, float, float]]:
    """What an off-perihelion or turned boost does to the landing.

    Returns:
        ``(offset or turn deg, landing shift deg, extra boost km/s)`` rows:
        positive values are boost offsets from perihelion along the arc;
        the turn rows are tagged by a negative zero-offset.
    """
    base = sun_route(depth, excess)
    rows = []
    for offset in offsets:
        flown = sun_route(depth, excess, boost_offset_deg=offset)
        rows.append(
            (
                offset,
                math.degrees(wrap_pi(math.radians(flown.miss_deg - base.miss_deg))),
                flown.boost - base.boost,
            )
        )
    return rows


def boost_turn(
    depth: float, excess: float, turns: Sequence[float]
) -> List[Tuple[float, float, float]]:
    """The landing shift and extra boost of a perihelion boost turned outward."""
    base = sun_route(depth, excess)
    rows = []
    for turn in turns:
        flown = sun_route(depth, excess, boost_turn_deg=turn)
        rows.append(
            (
                turn,
                math.degrees(wrap_pi(math.radians(flown.miss_deg - base.miss_deg))),
                flown.boost - base.boost,
            )
        )
    return rows


# --------------------------------------------------------------------------
# Two clocks (item 4)
# --------------------------------------------------------------------------


def earth_window_days(depth: float, excess: float, budget: float) -> float:
    """Days a departure point's Earth window stays open for ``budget`` km/s more.

    The Earth-bound dive's aphelion is raised (inbound or outbound branch)
    until its push costs ``budget`` more than the direct dive; the Earth
    phase it needs moves with it.  The window is that phase range over the
    Earth-Ceres synodic drift.
    """
    direct = sun_route(depth, excess, crossing=EARTH_A)
    phases = [direct.earth_phase_deg()]
    for inbound in (True, False):
        for r_a in np.linspace(CERES_A, 4.0 * AU, 400):
            flown = sun_route(
                depth, excess, aphelion=float(r_a), inbound=inbound, crossing=EARTH_A
            )
            if flown.departure_dv > direct.departure_dv + budget:
                break
            phases.append(flown.earth_phase_deg())
    shifts = [math.degrees(wrap_pi(math.radians(p - phases[0]))) for p in phases]
    span = max(shifts) - min(shifts)
    drift = 360.0 / (synodic_years(EARTH_A, CERES_A) * 365.25)
    return span / drift


# --------------------------------------------------------------------------
# Reports (items 1-9)
# --------------------------------------------------------------------------

#: Where the reports start each depth's optimizer: the optima of the run
#: recorded in ADR 0083 (so the headline reproduces; bounds CERES_BOUNDS).
HEADLINE = {
    4.0: CeresDesign(
        depth=4.0,
        ceres_excess=110.0,
        earth_stream=137.0,
        k_sun=29.907,
        arc_hours=3.14,
        k_delivery=5.62,
    ),
    9.86: CeresDesign(
        depth=9.86,
        ceres_excess=90.0,
        earth_stream=99.0,
        k_sun=27.8,
        arc_hours=7.38,
        k_delivery=5.0,
    ),
}
PUSH_SPEEDS = (65.0, 95.0, 125.0, 150.0, 200.0, 250.0, 300.0)
PUSH_ANGLE = 80.0
SWEEP_EXCESSES = (60.0, 90.0, 120.0, 150.0, 200.0, 250.0, 300.0)


def _f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def push_law_report() -> str:
    """Item 1: the departure check, then the Ceres push across closing speeds."""
    w, theta, k = PAPER_DEPARTURE
    check = [
        [
            "the paper's departure (1015 t -> 566 t over 4.67 km/s)",
            _f(PAPER_DEPARTURE_VE),
        ],
        [
            "law 1, polyethylene, no impactor toll (the parent's)",
            _f(law_one_ve(k, w, theta, POLYETHYLENE_BOND)),
        ],
        [
            "the departure-hole surface, polyethylene, carbon atomic",
            _f(StageLaws.load().hole_ve(k, w, theta)),
        ],
        [
            "the same surface, every bond returned",
            _f(_hole_laws().hole_ve(k, w, theta)),
        ],
    ]
    out = [
        f"Departure check: polyethylene at {w:g} km/s, {theta:g} deg, k = {k:g} (v_e km/s)",
        tabulate(check, tablefmt="github"),
    ]
    dv = sun_route(4.0, 110.0).departure_dv
    rows = []
    for speed in PUSH_SPEEDS:
        row = [f"{speed:g}"]
        for law, recovered in (
            ("hole", 0.0),
            ("hole", WATER_RECOVERY[1]),
            ("law1", 0.0),
        ):
            p = best_push(dv, speed, PUSH_ANGLE, recovered=recovered, law=law)
            row += [_f(p.k, 1), _f(p.ve), _f(p.stream, 3), _f(p.water)]
            if law == "hole" and recovered == 0.0:
                capped = best_push(dv, speed, PUSH_ANGLE, cap=60.0)
                row.append(f"{capped.k:.0f} / {capped.stream:.3f}")
        rows.append(row)
    out.append(
        f"\nThe Ceres push at {PUSH_ANGLE:g} deg, {dv:.2f} km/s (4 R_sun prograde), water on an ice pellet."
        " Each block: best k (cap 30), v_e, pellets and water per kg after the push"
    )
    out.append(
        tabulate(
            rows,
            headers=[
                "w",
                "k",
                "v_e",
                "pellets",
                "water",
                "k / pellets (cap 60)",
                "k",
                "v_e",
                "pellets",
                "water",
                "k",
                "v_e",
                "pellets",
                "water",
            ],
            tablefmt="github",
        )
    )
    out.append(
        "Blocks: the hole surface with water's frozen toll (headline); the same with 31% of the"
        " toll returned (ADR 0033's companion figure); the parent's law 1."
    )
    return "\n".join(out)


def orbits_report() -> str:
    """Item 2: Ceres-side speeds, coplanar circular, then what that hides."""
    rows = []
    for depth in DEPTHS:
        pro, retro = sun_route(depth, 90.0), sun_route(depth, 90.0, True)
        for x in (60.0, 90.0, 120.0, 150.0, 200.0, 250.0, 300.0):
            flown = sun_route(depth, x)
            w, theta = stream_at(flown, flown)
            rows.append(
                [
                    f"{depth:g}",
                    f"{x:g}",
                    _f(pro.departure_dv),
                    _f(retro.departure_dv),
                    _f(flown.periapsis_speed, 1),
                    _f(flown.boost),
                    _f(w, 1),
                    _f(theta, 1),
                    _f(flown.years),
                ]
            )
    out = [
        tabulate(
            rows,
            headers=[
                "R_sun",
                "X",
                "prograde dv",
                "retro dv",
                "v_p",
                "boost",
                "w at Ceres",
                "theta",
                "loop yr",
            ],
            tablefmt="github",
        )
    ]
    real = []
    for depth in DEPTHS:
        cells = [f"{depth:g}"]
        for anomaly in (0.0, 90.0, 180.0, 270.0):
            dv, miss = eccentric_dive(depth, 110.0, anomaly)
            cells.append(f"{dv:.2f} / {miss:.0f}")
        cells.append(
            " / ".join(_f(tilted_dive_dv(depth, t)) for t in (0.0, CERES_I, 20.0, 30.0))
        )
        real.append(cells)
    out.append(
        "\nCeres' real orbit (e 0.0785): direct prograde dive dv / landing (deg) at X 110, by Ceres'"
        " mean anomaly at departure; and the dive dv tilted 0 / 10.6 / 20 / 30 deg out of Ceres' plane"
    )
    out.append(
        tabulate(
            real,
            headers=[
                "R_sun",
                "M 0 (perihelion)",
                "M 90",
                "M 180 (aphelion)",
                "M 270",
                "tilted",
            ],
            tablefmt="github",
        )
    )
    return "\n".join(out)


def phasing_report() -> str:
    """Item 3: the misses, the rings, the single-depot closures and the boost levers."""
    misses = []
    for depth in (9.86, 4.0):
        for x in (60.0, 90.0):
            for retro in (False, True):
                flown = sun_route(depth, x, retro)
                found = closures(depth, x, retro, revolutions=(0,))
                misses.append(
                    [
                        f"{depth:g}",
                        f"{x:g}",
                        "retro" if retro else "pro",
                        _f(flown.miss_deg, 0),
                        (
                            (
                                f"{found[0].departure_dv:.2f} km/s, aphelion {found[0].aphelion / AU:.2f} AU,"
                                f" {'in' if found[0].inbound else 'out'}bound, {found[0].years:.2f} yr"
                            )
                            if found
                            else "none without an extra perihelion pass"
                        ),
                    ]
                )
    out = [
        "Direct dives' landing from their departure point, and the cheapest single-depot closure",
        tabulate(
            misses,
            headers=["R_sun", "X", "sense", "miss deg", "closes at"],
            tablefmt="github",
        ),
    ]
    ring_rows = []
    for depth, x in ((4.0, 110.0), (4.0, 150.0), (9.86, 90.0)):
        for ring in rings(depth, x, revolutions=(0,))[:4]:
            ring_rows.append(
                [
                    f"{depth:g}",
                    f"{x:g}",
                    f"{ring.depots} / {ring.step}",
                    _f(ring.route.departure_dv),
                    _f(ring.route.aphelion / AU, 3),
                    "in" if ring.route.inbound else "out",
                    _f(ring.route.years),
                    _f(ring.mean_offset_deg, 0),
                    f"{tug_dv(ring.mean_offset_deg, 10.0):.2f} / {tug_dv(ring.mean_offset_deg, 20.0):.2f}",
                    _f(tug_per_kg(ring, CERES_BOOK), 3),
                ]
            )
    out.append(
        "\nRings of co-orbital depots (N / j: each depot's stream lands j depots on), cheapest four,"
        " no extra perihelion pass"
    )
    out.append(
        tabulate(
            ring_rows,
            headers=[
                "R_sun",
                "X_C",
                "N / j",
                "dv",
                "aphelion AU",
                "branch",
                "loop yr",
                "mean offset",
                "tug dv 10 / 20 yr",
                "tug water / kg",
            ],
            tablefmt="github",
        )
    )
    levers = []
    for depth in DEPTHS:
        for offset, shift, extra in boost_rotation(
            depth, 110.0, (-20.0, -10.0, 10.0, 20.0)
        ):
            levers.append(
                [
                    f"{depth:g}",
                    f"boost {offset:+g} deg from perihelion",
                    _f(shift, 1),
                    _f(extra),
                ]
            )
        for turn, shift, extra in boost_turn(depth, 110.0, (1.0, 2.0, 5.0)):
            levers.append(
                [
                    f"{depth:g}",
                    f"perihelion boost turned {turn:g} deg",
                    _f(shift, 1),
                    _f(extra),
                ]
            )
        v_inf, bend = mars_flyby_turn(depth)
        levers.append(
            [
                f"{depth:g}",
                f"Mars flyby at 300 km (v_inf {v_inf:.1f})",
                f"turn {bend:.1f}",
                "-",
            ]
        )
    out.append(
        "\nOther closures (item 3c-d) at X 110: landing shift (deg) and extra Sun boost (km/s)"
    )
    out.append(
        tabulate(
            levers,
            headers=["R_sun", "lever", "shift", "extra boost"],
            tablefmt="github",
        )
    )
    disks = []
    for depth in DEPTHS:
        headline = routes_for(HEADLINE[depth])
        direct = sun_route(depth, HEADLINE[depth].ceres_excess, True)
        disks.append(
            [
                f"{depth:g}",
                _f(headline.ship.departure_dv),
                _f(headline.disks.departure_dv),
                _f(direct.departure_dv),
                _f(headline.disks.aphelion / AU, 3),
                _f(abs(headline.ship.sun_years - headline.disks.sun_years) * 365.25, 1),
            ]
        )
    out.append(
        "\nThe disks' head-on dive from the ship's own depot (headline closures)"
    )
    out.append(
        tabulate(
            disks,
            headers=[
                "R_sun",
                "ship dv",
                "disk dv",
                "direct mirror dv",
                "disk aphelion AU",
                "launch gap days",
            ],
            tablefmt="github",
        )
    )
    return "\n".join(out)


def clocks_report() -> str:
    """Item 4: the Ceres loop's clock against the Earth windows."""
    rows = []
    synodic = synodic_years(EARTH_A, CERES_A) * 365.25
    for depth in DEPTHS:
        flown = routes_for(HEADLINE[depth])
        windows = [
            earth_window_days(depth, flown.earth_excess, b) for b in (0.25, 0.5, 1.0)
        ]
        depots = flown.ring.depots if flown.ring else 1
        rows.append(
            [
                f"{depth:g}",
                _f(flown.ship.years),
                _f(flown.earth.years),
                " / ".join(f"{d:.0f}" for d in windows),
                f"{min(1.0, depots * windows[1] / synodic):.0%}",
                _f(earth_plane_tilt(flown.earth), 1),
                _f(
                    tilted_dive_dv(depth, earth_plane_tilt(flown.earth))
                    - tilted_dive_dv(depth, 0.0)
                ),
            ]
        )
    return f"Earth-Ceres synodic period {synodic:.0f} days\n" + tabulate(
        rows,
        headers=[
            "R_sun",
            "Ceres loop yr",
            "Earth leg yr",
            "window days at +0.25 / 0.5 / 1 km/s",
            "ring coverage at +0.5",
            "Earth plane tilt",
            "its dv",
        ],
        tablefmt="github",
    )


def _loop_rows(loop: SteadyLoop) -> List[List[str]]:
    rows = [[key, _f(value, 3)] for key, value in loop.lines.items()]
    rows.append(["**total, $/kg at L1**", f"**{loop.price:.3f}**"])
    return rows


def ledger_report() -> Tuple[str, Dict[float, Tuple[CeresDesign, SteadyLoop]]]:
    """Item 5: the headline at both depths, then the Ceres-stream speed swept."""
    optima = {depth: optimize_loop(HEADLINE[depth]) for depth in DEPTHS}
    out = []
    for depth, (design, loop) in optima.items():
        flown = routes_for(design)
        ship = loop.earth_ship.mass
        out.append(
            f"\n{depth:g} R_sun: X_C {design.ceres_excess:g}, X_E {flown.earth_excess:.1f}"
            f" ({design.earth_stream:g} km/s at 400 km); k_sun {design.k_sun:.1f}, arc"
            f" {design.arc_hours:.2f} h, k_delivery {design.k_delivery:.2f};"
            f" ring of {flown.ring.depots if flown.ring else 1}\n"
            f"c_C ${loop.c_ceres:.2f}/kg, c_E ${loop.c_earth:.2f}/kg, freight ${loop.freight:.2f}/kg;"
            f" Ceres loop x{loop.growth:.2f} per {loop.loop_years:.2f} yr"
            f" (doubles in {loop.doubling_years * 12:.1f} months)\n"
            f"Earth-bound ship per 1000 t after its push: water {ship['water'] / 1e3:.0f} t,"
            f" Ceres plate {ship['ceres_plate'] / 1e3:.1f} t, Sun charges {ship['sun_charge'] / 1e3:.0f} t,"
            f" disks {ship['disks'] / 1e3:.1f} t, Sun bowl {ship['sun_plate'] / 1e3:.1f} t,"
            f" sheets {ship['cargo'] / 1e3:.0f} t"
        )
        out.append(
            tabulate(_loop_rows(loop), headers=["line", "$/kg"], tablefmt="github")
        )
        split = sorted(loop.ceres_lines.items(), key=lambda kv: -kv[1])
        out.append(
            "earth_stream by the Earth-bound ship's lines: "
            + ", ".join(f"{key} {value:.3f}" for key, value in split if value >= 0.001)
        )
    sweep = []
    for depth in DEPTHS:
        for design, loop in excess_sweep(
            depth, excesses=SWEEP_EXCESSES, base=optima[depth][0]
        ):
            push = loop.ceres_ship.mass
            sweep.append(
                [
                    f"{depth:g}",
                    f"{design.ceres_excess:g}",
                    _f(push["closing"], 0),
                    _f(push["k"], 1),
                    _f(push["stream"] / push["pushed"], 3),
                    _f(push["boost"], 1),
                    _f(loop.c_ceres),
                    _f(loop.c_earth),
                    _f(loop.growth),
                    f"{loop.price:.3f}",
                ]
            )
    out.append("\nThe Ceres stream's speed swept, the Earth stream held, re-optimized")
    out.append(
        tabulate(
            sweep,
            headers=[
                "R_sun",
                "X_C",
                "w",
                "k",
                "pellets/kg",
                "Sun boost",
                "c_C",
                "c_E",
                "growth",
                "$/kg L1",
            ],
            tablefmt="github",
        )
    )
    return "\n".join(out), optima


def earth_push_report(optima: Dict[float, Tuple[CeresDesign, SteadyLoop]]) -> str:
    """Item 6: the delivery push re-solved on the Earth stream's price."""
    rows = [["parent (S56 section 5)", "-", "318", "0.053", "1182", "-"]]
    for depth, (design, loop) in optima.items():
        for k in sorted({5.0, round(design.k_delivery, 2), 10.0}):
            trial = steady_loop(replace(design, k_delivery=k))
            d = trial.delivery
            rows.append(
                [
                    f"{depth:g} R_sun, k {k:.2f}",
                    f"{d['stream_400']:.0f}",
                    f"{d['charge'] / 1e3:.0f}",
                    f"{d['sheets'] / d['cargo']:.4f}",
                    f"{d['cargo'] / 1e3:.0f}",
                    f"{trial.price:.3f}",
                ]
            )
    return tabulate(
        rows,
        headers=[
            "delivery",
            "stream km/s",
            "charge t",
            "sheets / kg",
            "delivered t",
            "$/kg L1",
        ],
        tablefmt="github",
    )


PLATE_RATES = (0.5, 1.0, 2.0)
PLATE_HOURS = (8.0, 24.0, 72.0)


def plate_report(optima: Dict[float, Tuple[CeresDesign, SteadyLoop]]) -> str:
    """Item 7: the Ceres plate on the spring law, and what to do with it."""
    design, _ = optima[4.0]
    flown = routes_for(design)
    rows = []
    for rate in PLATE_RATES:
        for hours in PLATE_HOURS:
            trial = replace(design, push_rate=rate, push_hours=hours)
            push = _push(flown.earth, flown.ship, trial, None)
            plate, bags, collar = _ceres_plate(1.0e6, push, trial)
            # One 63 t plate (with its bags): the force it carries at 8 m.
            per = (plate + bags + collar) / 1.0e6
            cells = [f"{rate:g}", f"{hours:g}", _f((plate + bags + collar) / 1e3, 1)]
            cells.append(_f(63.0 / per, 0) if per > 0 else "-")
            for policy in ("recover", "expend", "sun_bowl"):
                cells.append(f"{steady_loop(replace(trial, plate=policy)).price:.3f}")
            rows.append(cells)
    return (
        "The Ceres push on the spring law (8 m stroke, 2 pi/3), 4 R_sun headline; plate per 1000 t"
        " pushed, the ship one 63 t-class plate pushes, and $/kg at L1 by what is done with it\n"
        + tabulate(
            rows,
            headers=[
                "Hz",
                "hours",
                "plate t / 1000 t",
                "t pushed by 63 t",
                "recover",
                "expend",
                "Sun bowl",
            ],
            tablefmt="github",
        )
    )


@dataclass(frozen=True)
class CeresLever:
    """One sensitivity: a design or book change from the headline."""

    name: str
    design: Dict[str, object] = field(default_factory=dict)
    prices: Dict[str, object] = field(default_factory=dict)


CERES_SENSITIVITIES = (
    CeresLever("Ceres film $1/kg", prices={"film": 1.0}),
    CeresLever("Ceres film $10/kg", prices={"film": 10.0}),
    CeresLever("Earth hardware $100/kg", prices={"earth_hardware": 100.0}),
    CeresLever("freight $3/kg (the parent's placeholder)", prices={"freight": 3.0}),
    CeresLever("freight $10/kg", prices={"freight": 10.0}),
    CeresLever("unlisted dead mass 5% of the ship", design={"reserve": 0.05}),
    CeresLever(
        "unlisted dead mass 15% (cargo fraction ~0.6)", design={"reserve": 0.15}
    ),
    CeresLever("misses 10%", prices={"misses": 0.10}),
    CeresLever("steering packages $15 (MATURE's curve)", prices={"package": 15.0}),
    CeresLever("no guidance on the Ceres stream", design={"stream_guidance": False}),
    CeresLever(
        "Earth plates flown 10 times",
        prices={"earth_plate_uses": 10.0, "ceres_plate_uses": 10.0},
    ),
    CeresLever("lob $0.30 (methalox $0.12)", prices={"lob": 0.30}),
    CeresLever("lob $0.66 (methalox $0.27)", prices={"lob": 0.66}),
    CeresLever("lob $1/kg lofted", prices={"lob": 1.0}),
    CeresLever("water's toll 31% returned", design={"recovered": WATER_RECOVERY[1]}),
    CeresLever("law 1 for the Ceres push", design={"law": "law1"}),
    CeresLever("hole surface held at 89.2 deg", design={"clamp_angle": True}),
    CeresLever("ship 3000 t", design={"ship_mass": 3.0e6}),
    CeresLever("ship 500 t", design={"ship_mass": 5.0e5}),
    CeresLever("Ceres push 2 Hz", design={"push_rate": 2.0}),
    CeresLever("child's 4 Hz delivery (S37)", design={"child_delivery": True}),
)

#: The cascade to $1 (item 8), cumulative, in the order each lever's size
#: at the headline ranks it.
CASCADE = (
    CeresLever(
        "child's 4 Hz, 3.26 g delivery from the 1.1 km/s lob (S37)",
        design={"child_delivery": True},
    ),
    CeresLever("Ceres film $1/kg", prices={"film": 1.0}),
    CeresLever("ship 3000 t", design={"ship_mass": 3.0e6}),
    CeresLever("Ceres push at 2 Hz", design={"push_rate": 2.0}),
    CeresLever("water's toll 31% returned", design={"recovered": WATER_RECOVERY[1]}),
    CeresLever("methalox $0.12/kg (lob $0.30)", prices={"lob": 0.30}),
)


def _apply(
    design: CeresDesign, prices: CeresPrices, lever: CeresLever
) -> Tuple[CeresDesign, CeresPrices]:
    return replace(design, **lever.design), replace(prices, **lever.prices)  # type: ignore[arg-type]


def sensitivity_report(optima: Dict[float, Tuple[CeresDesign, SteadyLoop]]) -> str:
    """Item 8: the levers one at a time, then the cascade to $1 and the floor."""
    rows = []
    for lever in CERES_SENSITIVITIES:
        cells = [lever.name]
        for depth in DEPTHS:
            design, loop = optima[depth]
            d, p = _apply(design, CERES_BOOK, lever)
            try:
                cells.append(f"{steady_loop(d, p).price:.3f}")
            except ValueError:
                cells.append("does not close")
        rows.append(cells)
    head = [["headline"] + [f"{optima[d][1].price:.3f}" for d in DEPTHS]]
    out = [
        "One at a time, design held ($/kg at L1)",
        tabulate(
            head + rows,
            headers=["lever"] + [f"{d:g} R_sun" for d in DEPTHS],
            tablefmt="github",
        ),
    ]
    cascade = [["headline"] + [f"{optima[d][1].price:.3f}" for d in DEPTHS]]
    states = {d: (optima[d][0], CERES_BOOK) for d in DEPTHS}
    for lever in CASCADE:
        cells = [lever.name]
        for depth in DEPTHS:
            d, p = _apply(*states[depth], lever)
            d, loop = optimize_loop(d, p)
            states[depth] = (d, p)
            cells.append(f"{loop.price:.3f} (lob {loop.lines['delivery_lob']:.3f})")
        cascade.append(cells)
    out.append("\nWhat must fall to reach $1: cumulative, each step re-optimized")
    out.append(
        tabulate(
            cascade,
            headers=["step"] + [f"{d:g} R_sun" for d in DEPTHS],
            tablefmt="github",
        )
    )
    return "\n".join(out)


def uncharged_report(optima: Dict[float, Tuple[CeresDesign, SteadyLoop]]) -> str:
    """Item 9: what reaches $0.02/kg at L1 and is not charged, sized where it can be."""
    _, loop = optima[4.0]
    delivery = loop.delivery
    # Booster operations at $1M a lob: one delivery lob per cargo.
    ops = 1.0e6 / delivery["cargo"]
    rows = [
        ["booster operations and hardware, per $1M a lob", f"{ops:.2f}"],
        [
            "the lob's climb above the child's 1.1 km/s (already charged)",
            "in delivery_lob",
        ],
        [
            "Ceres industry: reactors, mining, the plant (assumed, unpriced)",
            "-",
        ],
        [
            "the seed: the first Ceres stream (one loop's pellets per ship)",
            f"{loop.ceres_ship.stream / 1e3:.0f} t a 1000 t ship",
        ],
        ["the depot tug's reactor and tanks (water charged)", "-"],
        ["the Earth plates' recovery from the high orbit", "-"],
        [
            "guidance on 0.5 Hz ice pellets (charged at the sheets' packages)",
            f"{loop.ceres_ship.mass['pulse_pellet']:.1f} kg a pellet",
        ],
        ["Earth-bound ships per window (bursty Earth deliveries)", "idle Earth plates"],
    ]
    return tabulate(
        rows, headers=["not charged", "size ($/kg at L1)"], tablefmt="github"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the S56 answers.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    print("Item 1. The Ceres push law\n" + push_law_report())
    print("\nItem 2. Ceres-side orbits\n" + orbits_report())
    print("\nItem 3. Phasing\n" + phasing_report())
    print("\nItem 4. Two clocks\n" + clocks_report())
    ledger, optima = ledger_report()
    print("\nItem 5. The steady ledger" + ledger)
    print("\nItem 6. The Earth push re-solved on c_E\n" + earth_push_report(optima))
    print("\nItem 7. The Ceres plate\n" + plate_report(optima))
    print("\nItem 8. Sensitivities and the $1 line\n" + sensitivity_report(optima))
    print("\nItem 9. Not charged\n" + uncharged_report(optima))


if __name__ == "__main__":
    main()
