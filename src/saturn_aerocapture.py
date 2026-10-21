"""Aerocapture at Saturn for the outbound ship, rings included (author, 2026-10-21).

The Saturn-side outbound of ADR 0078/0079 is all methalox: a 2.55 km/s
capture burn at 150 000 km (2.81 on the real arcs), the dog-leg's apoapsis
raise, the match and the landing, 5.1 kg of stage per kg landed at Iapetus.
Here the capture is flown through Saturn's atmosphere instead, and the
methalox stage keeps only the burns after it.

**The pass.** A 3-DOF point mass in Saturn's inertial frame (ecliptic J2000,
:mod:`src.saturn_moon_legs`), point-mass gravity, an atmosphere co-rotating
with the planet (10.66 h), drag at ballistic coefficient ``beta`` and lift
at ``L/D`` steered in the vertical plane only (bank reversals taken as
perfect).  It starts on the arrival hyperbola at 1000 km above the 1-bar
ellipsoid and ends when it climbs back through it, or fails below 1 bar.

- **Corridor**: the vacuum-periapsis altitudes at which full lift up (steep
  limit) and full lift down (shallow limit) exit onto the target apoapsis;
  the nominal flies their midpoint with the bank solved to hit it.
- **Heating** at the stagnation point: AMAT's Saturn coefficients (Girija,
  ``AMAT`` 2.4.0 ``vehicle.py``): Sutton-Graves convective at
  ``k = 0.6356e-4`` and the ESTEC Jupiter-probe radiative fit AMAT uses for
  Saturn.  Peak rate and peak deceleration are taken over the three passes,
  and the heat load too.
- **Aeroshell**: thermal protection from the heat load (the HEEET fit of
  Girija et al. 2023, arXiv 2308.07005: ``5.91 log10(Q kJ/cm^2) + 5`` per
  cent; the legacy carbon-phenolic fit of Laub and Venkatapathy,
  ``0.091 Q[J/cm^2]^0.51575`` per cent, as the dear row), plus structure,
  backshell and mechanisms at 10% of entry mass (assumption).  It is shed
  after the pass.

**The atmosphere** is hydrostatic in H2/He at 2.2 g/mol over a temperature
profile shaped after the Voyager/Cassini picture (134 K at 1 bar, an 84 K
tropopause near 100 mbar, a ~145 K stratosphere, a ~400 K thermosphere above
a microbar): labelled assumptions, not a fitted model.  NASA's own Saturn
study (Putnam et al. 2023, NTRS 20230005179) bridges Yelle's upper and
Sayanagi's lower atmosphere; its near-equatorial example sheds ~4.5 km/s to an
apoapsis near Titan at ~2.5 g and ~1400 W/cm^2.

**The rings.** The pass's periapsis is a few hundred kilometres above the
cloud tops, so the arrival hyperbola and the exit ellipse cross Saturn's
equator, the ring plane, somewhere.  A crossing is *clear* inside 65 000 km
(below the D ring, where Cassini's Grand Finale flew) or beyond the G ring
(175 000 km), *dusty* in the F-G gap (140 220-166 000 km, where Cassini's
ring-grazing orbits crossed), and *blocked* anywhere else.  As in
:mod:`src.saturn_moon_legs`, the apse line lies on the asymptote's cone and
in the moon's plane, so each window has at most two geometries; the ring
check picks among them.

**Burns after the pass**, all methalox: the apoapsis burn (periapsis raise
and plane turn at Iapetus, the match at Phoebe) from the *exit* periapsis, the
Iapetus match, the landing, and a 50 m/s apoapsis-targeting correction
(assumption).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tabulate import tabulate

if TYPE_CHECKING:
    from src.iapetus_cycle import CyclePrices, OutboundBook
    from src.saturn_methalox import PlantCycle, PlantValuation, Site

from src.saturn_moon_legs import (
    DOGLEG_APOAPSIS,
    F_RING,
    G_RING,
    IAPETUS_A,
    IAPETUS_NORMAL,
    MU_SATURN,
    PHOEBE_A,
    PHOEBE_NORMAL,
    SATURN_POLE,
    SYNODIC_DAYS,
    WINDOWS,
    Vec,
    _unit,
    _vis_viva,
    apse_candidates,
    asymptote_angle,
    outbound_arc,
)

#: Saturn's 1-bar ellipsoid (km) and spin (rad/s, AMAT's 10.66 h).
R_EQ = 60_268.0
R_POL = 54_364.0
OMEGA = 1.6379e-4
#: Entry interface above the 1-bar ellipsoid (km, AMAT's ``h_thres``).
ENTRY_ALTITUDE = 1000.0
#: Molar mass of the H2/He gas (kg/mol; assumption, helium ~13% by mass).
MOLAR_MASS = 2.2e-3
#: Temperature against log10(pressure / bar) (assumption, see the docstring).
TEMPERATURE_NODES = (
    (0.0, 134.0),
    (-1.0, 84.0),
    (-2.5, 135.0),
    (-4.0, 145.0),
    (-6.0, 145.0),
    (-7.5, 400.0),
    (-11.0, 420.0),
)
#: AMAT's Saturn Sutton-Graves constant (SI: W/m^2 from kg/m^3, m, m/s).
SUTTON_GRAVES = 0.6356e-4
#: Structure, backshell and mechanisms per kg of entry mass (assumption).
STRUCTURE = 0.10
#: The apoapsis-targeting correction after the pass (km/s, assumption).
TARGETING = 0.050
#: Ring-plane zones (km).
D_RING_GAP = 65_000.0
_STEP = 0.5  # s, the pass integrator's step
_G0 = 9.80665


# --------------------------------------------------------------------------
# The atmosphere
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _density_table() -> Tuple[float, float, Tuple[float, ...]]:
    """ln(density) on a 1 km grid of altitude above 1 bar: (h0, step, values)."""
    lp = np.arange(0.0, -11.0 - 1e-9, -0.002)
    nodes = np.array(TEMPERATURE_NODES)
    temp = np.interp(lp, nodes[::-1, 0], nodes[::-1, 1])
    gas = 8.314462618 / MOLAR_MASS
    z = np.zeros_like(lp)
    for i in range(1, len(lp)):
        r = R_EQ + z[i - 1]
        g = MU_SATURN / r**2 * 1e3  # m/s^2
        dlnp = (lp[i - 1] - lp[i]) * math.log(10.0)
        z[i] = z[i - 1] + gas * 0.5 * (temp[i] + temp[i - 1]) / g * dlnp / 1e3
    rho = 10.0**lp * 1e5 / (gas * temp)
    grid = np.arange(0.0, float(z[-1]), 1.0)
    return 0.0, 1.0, tuple(np.interp(grid, z, np.log(rho)))


def density(h: float) -> float:
    """Gas density (kg/m^3) at ``h`` km above the 1-bar level."""
    h0, step, table = _density_table()
    x = (h - h0) / step
    if x >= len(table) - 1:
        return 0.0
    if x <= 0.0:
        return math.exp(table[0])
    i = int(x)
    f = x - i
    return math.exp(table[i] * (1.0 - f) + table[i + 1] * f)


def surface_radius(sin_lat: float) -> float:
    """The 1-bar ellipsoid's radius at geocentric latitude (km)."""
    c2 = 1.0 - sin_lat**2
    return R_EQ * R_POL / math.sqrt((R_POL**2) * c2 + (R_EQ**2) * sin_lat**2)


# --------------------------------------------------------------------------
# The vehicle and the pass
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Vehicle:
    """The entering ship (assumptions marked).

    Attributes:
        beta: Ballistic coefficient m / (C_D A) (kg/m^2): a rigid blunt
            shield on a dense stack; 150 / 500 / 1500 rows.
        lift_drag: L/D at trim (0.25 a blunt capsule, 0.4 an ellipsled).
        entry_mass: One ship (kg): sets the shield's size, so its nose radius.
        drag_coefficient: Hypersonic C_D of a 70-degree sphere-cone.
        nose_share: Nose radius per shield diameter.
    """

    beta: float = 500.0
    lift_drag: float = 0.25
    entry_mass: float = 500.0e3
    drag_coefficient: float = 1.6
    nose_share: float = 0.5

    @property
    def diameter(self) -> float:
        """Shield diameter (m)."""
        area = self.entry_mass / (self.drag_coefficient * self.beta)
        return math.sqrt(4.0 * area / math.pi)

    @property
    def nose_radius(self) -> float:
        """Stagnation-point radius (m)."""
        return self.nose_share * self.diameter


VEHICLE = Vehicle()


@dataclass(frozen=True)
class Pass:
    """One atmospheric pass.

    Attributes:
        exited: Climbed back through the entry interface.
        apoapsis: Exit orbit's apoapsis (km; inf if not captured).
        periapsis: Exit orbit's periapsis (km).
        drag_dv: Inertial speed lost between the interfaces (km/s).
        entry_speed: Atmosphere-relative speed at the interface (km/s).
        entry_angle: Flight-path angle at the interface (deg).
        peak_heat: Stagnation-point heat rate (W/cm^2, convective + radiative).
        heat_load: Its integral (J/cm^2).
        peak_g: Peak sensed deceleration (Earth g).
        lowest: Lowest altitude above 1 bar (km).
    """

    exited: bool
    apoapsis: float
    periapsis: float
    drag_dv: float
    entry_speed: float
    entry_angle: float
    peak_heat: float
    heat_load: float
    peak_g: float
    lowest: float


def _entry_state(
    periapsis_dir: Vec, normal: Vec, rp: float, excess: float
) -> Tuple[List[float], List[float]]:
    """The arrival hyperbola's state at the entry interface (km, km/s)."""
    q = np.cross(normal, periapsis_dir)
    ecc = 1.0 + rp * excess**2 / MU_SATURN
    p = rp * (1.0 + ecc)
    r_ei = surface_radius(float(periapsis_dir @ SATURN_POLE)) + ENTRY_ALTITUDE
    nu = -math.acos(max(min((p / r_ei - 1.0) / ecc, 1.0), -1.0))
    r = r_ei * (math.cos(nu) * periapsis_dir + math.sin(nu) * q)
    v = math.sqrt(MU_SATURN / p) * (
        -math.sin(nu) * periapsis_dir + (ecc + math.cos(nu)) * q
    )
    return [float(x) for x in r], [float(x) for x in v]


def fly(
    periapsis_dir: Vec,
    normal: Vec,
    vacuum_altitude: float,
    excess: float,
    lift_up: float,
    vehicle: Vehicle = VEHICLE,
) -> Pass:
    """Fly one pass at a constant vertical lift share ``lift_up`` (cos bank).

    Args:
        periapsis_dir: Unit vector to the vacuum periapsis.
        normal: The orbit's unit angular momentum.
        vacuum_altitude: Vacuum periapsis above the 1-bar ellipsoid (km).
        excess: Arrival v-infinity (km/s).
        lift_up: cos(bank): +1 full lift up, -1 full lift down.
        vehicle: The ship.
    """
    sin_lat = float(periapsis_dir @ SATURN_POLE)
    rp = surface_radius(sin_lat) + vacuum_altitude
    r, v = _entry_state(periapsis_dir, normal, rp, excess)
    px, py, pz = (float(x) for x in SATURN_POLE)
    wx, wy, wz = OMEGA * px, OMEGA * py, OMEGA * pz
    beta, ld = vehicle.beta, vehicle.lift_drag * lift_up
    ld_total = math.hypot(1.0, vehicle.lift_drag)
    rn = vehicle.nose_radius
    conv = SUTTON_GRAVES / math.sqrt(rn)
    rad = 9.7632379e-40 * (2.0 * rn) ** -0.17905 * 1e3  # W/m^2 coefficient
    mu = MU_SATURN
    peak_q = peak_g = heat = 0.0
    lowest = math.inf

    def deriv(s: List[float]) -> Tuple[List[float], float, float, float]:
        x, y, z, vx, vy, vz = s
        rr = math.sqrt(x * x + y * y + z * z)
        h = rr - surface_radius((x * px + y * py + z * pz) / rr)
        g = -mu / rr**3
        ax, ay, az = g * x, g * y, g * z
        rho = density(h)
        q = dec = 0.0
        if rho > 0.0:
            ux = vx - (wy * z - wz * y)
            uy = vy - (wz * x - wx * z)
            uz = vz - (wx * y - wy * x)
            speed = math.sqrt(ux * ux + uy * uy + uz * uz)
            drag = 0.5 * rho * speed * speed * 1e3 / beta  # km/s^2
            ex, ey, ez = ux / speed, uy / speed, uz / speed
            rx, ry, rz = x / rr, y / rr, z / rr
            dot = rx * ex + ry * ey + rz * ez
            nx, ny, nz = rx - dot * ex, ry - dot * ey, rz - dot * ez
            nn = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            lift = ld * drag / nn
            ax += -drag * ex + lift * nx
            ay += -drag * ey + lift * ny
            az += -drag * ez + lift * nz
            vm = speed * 1e3
            q = (
                conv * math.sqrt(rho) * vm**3 + rad * rho**1.763827469 * vm**10.993852
            ) / 1e4
            dec = drag * ld_total * 1e3 / _G0
        return [vx, vy, vz, ax, ay, az], q, dec, h

    s = r + v
    v_in = math.sqrt(sum(c * c for c in v))
    ux = v[0] - (wy * r[2] - wz * r[1])
    uy = v[1] - (wz * r[0] - wx * r[2])
    uz = v[2] - (wx * r[1] - wy * r[0])
    entry_speed = math.sqrt(ux * ux + uy * uy + uz * uz)
    rv = sum(a * b for a, b in zip(r, v))
    entry_angle = math.degrees(
        math.asin(rv / (math.sqrt(sum(c * c for c in r)) * v_in))
    )
    dt = _STEP
    exited = False
    for _ in range(40_000):
        k1, q1, g1, h1 = deriv(s)
        lowest = min(lowest, h1)
        if h1 < 0.0:
            break
        if h1 > ENTRY_ALTITUDE and sum(s[i] * s[i + 3] for i in range(3)) > 0.0:
            exited = True
            break
        peak_q, peak_g = max(peak_q, q1), max(peak_g, g1)
        heat += q1 * dt
        k2 = deriv([a + 0.5 * dt * b for a, b in zip(s, k1)])[0]
        k3 = deriv([a + 0.5 * dt * b for a, b in zip(s, k2)])[0]
        k4 = deriv([a + dt * b for a, b in zip(s, k3)])[0]
        s = [
            a + dt / 6.0 * (b + 2.0 * c + 2.0 * d + e)
            for a, b, c, d, e in zip(s, k1, k2, k3, k4)
        ]
    rr = math.sqrt(sum(c * c for c in s[:3]))
    vv = math.sqrt(sum(c * c for c in s[3:]))
    energy = 0.5 * vv * vv - mu / rr
    apo, peri = math.inf, math.nan
    if exited and energy < 0.0:
        a = -mu / (2.0 * energy)
        hv = np.cross(s[:3], s[3:])
        ecc = math.sqrt(max(1.0 + 2.0 * energy * float(hv @ hv) / mu**2, 0.0))
        apo, peri = a * (1.0 + ecc), a * (1.0 - ecc)
    return Pass(
        exited=exited,
        apoapsis=apo,
        periapsis=peri,
        drag_dv=v_in - vv,
        entry_speed=entry_speed,
        entry_angle=entry_angle,
        peak_heat=peak_q,
        heat_load=heat,
        peak_g=peak_g,
        lowest=lowest,
    )


def _miss(p: Pass, target: float) -> float:
    """Signed log miss of the target apoapsis: + too high, - too low or failed."""
    if not p.exited:
        return -50.0
    if not math.isfinite(p.apoapsis):
        return 50.0
    return math.log(p.apoapsis / target)


def _bisect(
    fn: Callable[[float], float], lo: float, hi: float, iterations: int = 32
) -> Optional[float]:
    """Root of an increasing ``fn`` on [lo, hi], or None if not bracketed."""
    flo, fhi = fn(lo), fn(hi)
    if flo > 0.0 or fhi < 0.0:
        return None
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if fn(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Corridor:
    """One geometry's corridor and its sizing passes.

    Attributes:
        steep: Vacuum periapsis altitude, full lift up (km).
        shallow: The same, full lift down (km).
        width_deg: The two entry flight-path angles apart (deg).
        nominal: The midpoint pass, bank solved.
        bank_up: Its cos(bank).
        peak_heat: Worst of the three passes (W/cm^2).
        heat_load: Worst of the three (J/cm^2).
        peak_g: Worst of the three (g).
    """

    steep: float
    shallow: float
    width_deg: float
    nominal: Pass
    bank_up: float
    peak_heat: float
    heat_load: float
    peak_g: float


def corridor(
    periapsis_dir: Vec,
    normal: Vec,
    excess: float,
    target: float,
    vehicle: Vehicle = VEHICLE,
) -> Optional[Corridor]:
    """The steep and shallow limits onto ``target`` apoapsis, and the nominal pass."""

    def at(alt: float, up: float) -> Pass:
        return fly(periapsis_dir, normal, alt, excess, up, vehicle)

    steep = _bisect(lambda a: _miss(at(a, 1.0), target), -300.0, 1000.0)
    shallow = _bisect(lambda a: _miss(at(a, -1.0), target), -300.0, 1000.0)
    if steep is None or shallow is None:
        return None
    mid = 0.5 * (steep + shallow)
    up = _bisect(lambda c: _miss(at(mid, c), target), -1.0, 1.0, 24)
    if up is None:
        return None
    passes = (at(steep, 1.0), at(shallow, -1.0), at(mid, up))
    return Corridor(
        steep=steep,
        shallow=shallow,
        width_deg=abs(passes[0].entry_angle - passes[1].entry_angle),
        nominal=passes[2],
        bank_up=up,
        peak_heat=max(p.peak_heat for p in passes),
        heat_load=max(p.heat_load for p in passes),
        peak_g=max(p.peak_g for p in passes),
    )


# --------------------------------------------------------------------------
# The aeroshell
# --------------------------------------------------------------------------


def tps_share(heat_load: float, fit: str = "HEEET") -> float:
    """Thermal protection per kg of entry mass from a heat load (J/cm^2)."""
    if fit == "HEEET":
        return max(5.91 * math.log10(heat_load / 1e3) + 5.0, 0.0) / 100.0
    if fit == "carbon phenolic":
        return float(0.091 * heat_load**0.51575 / 100.0)
    raise ValueError(fit)


def aeroshell_share(
    heat_load: float, fit: str = "HEEET", structure: float = STRUCTURE
) -> float:
    """Aeroshell (shed after the pass) per kg of entry mass."""
    return tps_share(heat_load, fit) + structure


# --------------------------------------------------------------------------
# The rings
# --------------------------------------------------------------------------


def ring_zone(radius: float) -> str:
    """``clear``, ``dusty`` or ``blocked`` for an equator crossing (km)."""
    if radius < D_RING_GAP or radius > G_RING[1]:
        return "clear"
    if F_RING < radius < G_RING[0]:
        return "dusty"
    return "blocked"


def ring_crossings(
    periapsis_dir: Vec, normal: Vec, rp: float, excess: float, apoapsis: float
) -> List[Tuple[str, float]]:
    """Equator crossings of the arrival hyperbola and the exit ellipse (km).

    The hyperbola is flown from infinity to periapsis, the exit ellipse from
    periapsis to the first apoapsis (where the apoapsis burn lifts the
    periapsis clear).  The atmospheric arc is taken on the conics.
    """
    line = np.cross(SATURN_POLE, normal)
    if np.linalg.norm(line) < 1e-9:
        return [("in", rp)]
    line = _unit(line)
    e_in = 1.0 + rp * excess**2 / MU_SATURN
    nu_inf = math.acos(-1.0 / e_in)
    e_out = (apoapsis - rp) / (apoapsis + rp)
    out = []
    for d in (line, -line):
        nu = math.atan2(
            float(np.cross(periapsis_dir, d) @ normal), float(periapsis_dir @ d)
        )
        if -nu_inf < nu <= 0.0:
            out.append(("in", rp * (1.0 + e_in) / (1.0 + e_in * math.cos(nu))))
        if 0.0 <= nu <= math.pi:
            out.append(("out", rp * (1.0 + e_out) / (1.0 + e_out * math.cos(nu))))
    return out


def ring_status(crossings: Sequence[Tuple[str, float]]) -> str:
    """The worst zone among the crossings."""
    zones = [ring_zone(r) for _, r in crossings]
    for z in ("blocked", "dusty"):
        if z in zones:
            return z
    return "clear"


# --------------------------------------------------------------------------
# The windows
# --------------------------------------------------------------------------

#: The vacuum periapsis the apse cone is drawn for (km above 1 bar).
NOMINAL_ALTITUDE = 400.0
#: Methalox exhaust speed (km/s), to weigh an aeroshell against a burn.
_VE_METHALOX = 370.0 * _G0 / 1e3
#: The first apoapsis the pass aims for (km): the parent's dog-leg at
#: Iapetus; above Phoebe's orbit, so the apoapsis burn can reach it anywhere.
CAPTURE_APOAPSES: Dict[str, Tuple[float, ...]] = {
    "Iapetus": (DOGLEG_APOAPSIS,),
    "Phoebe": (1.05 * PHOEBE_A, 1.2 * PHOEBE_A, 1.5 * PHOEBE_A, 2.0 * PHOEBE_A),
}
MOON_RADIUS = {"Iapetus": IAPETUS_A, "Phoebe": PHOEBE_A}
MOON_NORMAL = {"Iapetus": IAPETUS_NORMAL, "Phoebe": PHOEBE_NORMAL}
_CONE_STEPS = 180
_PLANE_STEPS = 360


@dataclass(frozen=True)
class MoonLeg:
    """From the first apoapsis onto the moon's orbit (methalox, impulsive).

    The burn at apoapsis puts the ship on an ellipse whose plane holds the
    apse line and a point ``d`` of the moon's orbit; it meets the moon there
    and matches its velocity.  ``d`` opposite the apse line is the parent's
    tangent dog-leg; elsewhere the match is not tangent and costs more.

    Attributes:
        burn: At the first apoapsis (km/s).
        match: At the moon (km/s).
        years: Periapsis of the pass to the moon.
        meet_angle: ``d`` from the tangent point (deg).
    """

    burn: float
    match: float
    years: float
    meet_angle: float

    @property
    def dv(self) -> float:
        """Both burns (km/s)."""
        return self.burn + self.match


def _kepler_time(ecc: float, nu: float, a: float) -> float:
    """Seconds from periapsis to true anomaly ``nu`` (0..pi) on an ellipse."""
    big_e = 2.0 * math.atan(math.sqrt((1.0 - ecc) / (1.0 + ecc)) * math.tan(0.5 * nu))
    return (big_e - ecc * math.sin(big_e)) * math.sqrt(a**3 / MU_SATURN)


def moon_leg(
    apse: Vec,
    normal: Vec,
    rp: float,
    ra: float,
    site: str,
    max_years: Optional[float] = None,
) -> Optional[MoonLeg]:
    """The cheapest leg from the exit ellipse's apoapsis onto the moon.

    ``d`` is swept round the moon's orbit (1 degree), in both senses of the
    second ellipse; legs longer than ``max_years`` (from the pass) are dropped.
    """
    n_moon, r_moon = MOON_NORMAL[site], MOON_RADIUS[site]
    v_low = _vis_viva(ra, 0.5 * (rp + ra))
    moving = _unit(np.cross(normal, apse))
    e1 = _unit(np.cross(n_moon, SATURN_POLE))
    e2 = np.cross(n_moon, e1)
    v_moon = math.sqrt(MU_SATURN / r_moon)
    t_exit = math.pi * math.sqrt((0.5 * (rp + ra)) ** 3 / MU_SATURN)
    psi = np.linspace(0.0, 2.0 * math.pi, _PLANE_STEPS, endpoint=False)
    d = np.outer(np.cos(psi), e1) + np.outer(np.sin(psi), e2)
    cos_a = np.clip(-(d @ apse), -1.0, 1.0)
    alpha = np.arccos(cos_a)
    keep = alpha < math.pi - 0.05
    d, cos_a, alpha = d[keep], cos_a[keep], alpha[keep]
    ecc = (ra - r_moon) / (ra + r_moon * cos_a)
    p = ra * (1.0 - ecc)
    a2 = p / (1.0 - ecc**2)
    v_apo = np.sqrt(MU_SATURN * (2.0 / ra - 1.0 / a2))
    perp = d - np.outer(d @ apse, apse)
    norm = np.linalg.norm(perp, axis=1)
    tangent = _unit(np.cross(n_moon, apse))
    base = np.where(
        (norm > 1e-6)[:, None], perp / np.maximum(norm, 1e-12)[:, None], tangent
    )
    scale = np.sqrt(MU_SATURN / p)
    big_e = 2.0 * np.arctan(np.sqrt((1.0 - ecc) / (1.0 + ecc)) * np.tan(0.5 * alpha))
    t_d = (big_e - ecc * np.sin(big_e)) * np.sqrt(a2**3 / MU_SATURN)
    half = math.pi * np.sqrt(a2**3 / MU_SATURN)
    best: Optional[MoonLeg] = None
    for sense in (1.0, -1.0):
        t_hat = sense * base
        h2 = np.cross(apse, t_hat)
        h2 /= np.linalg.norm(h2, axis=1)[:, None]
        inbound = np.einsum("ij,ij->i", t_hat, d) >= 0.0
        nu = np.where(inbound, -alpha, alpha)
        v_ship = (scale * ecc * np.sin(nu))[:, None] * d + (
            scale * (1.0 + ecc * np.cos(nu))
        )[:, None] * np.cross(h2, d)
        match = np.linalg.norm(v_ship - v_moon * np.cross(n_moon, d), axis=1)
        burn = np.linalg.norm(v_apo[:, None] * t_hat - v_low * moving, axis=1)
        years = (t_exit + np.where(inbound, half - t_d, half + t_d)) / (
            365.25 * 86400.0
        )
        total = burn + match
        if max_years is not None:
            total = np.where(years <= max_years, total, np.inf)
        i = int(np.argmin(total))
        if not math.isfinite(float(total[i])):
            continue
        leg = MoonLeg(
            float(burn[i]), float(match[i]), float(years[i]), math.degrees(alpha[i])
        )
        if best is None or leg.dv < best.dv:
            best = leg
    return best


@dataclass(frozen=True)
class Geometry:
    """One window's arrival at one moon, aerocaptured.

    Attributes:
        site: ``Iapetus`` or ``Phoebe``.
        window: Window index from 2035.
        excess: Arrival v-infinity (km/s).
        inclination: The arrival orbit on Saturn's equator (deg).
        periapsis_lat: Latitude of the vacuum periapsis (deg).
        rings: ``clear``, ``dusty`` or ``blocked``.
        crossings: The equator crossings (km).
        capture_apoapsis: The pass's target apoapsis (km).
        corridor: The flown corridor (None if blocked or not closing).
        leg: The methalox leg from the exit apoapsis to the moon.
        aeroshell: Shed per kg of entry mass (HEEET).
    """

    site: str
    window: int
    excess: float
    inclination: float
    periapsis_lat: float
    rings: str
    crossings: Tuple[float, ...]
    capture_apoapsis: float
    corridor: Optional[Corridor]
    leg: Optional[MoonLeg]
    aeroshell: float

    @property
    def equivalent_dv(self) -> float:
        """Leg plus the aeroshell as methalox delta-v (km/s), for ranking."""
        assert self.leg is not None
        return self.leg.dv - _VE_METHALOX * math.log(1.0 - self.aeroshell)


_RING_RANK = {"clear": 0, "dusty": 1, "blocked": 2}
#: The capture apoapsis each site's pass is flown for (heating barely
#: depends on it; the legs are then costed at every apoapsis).
BASE_APOAPSIS = {"Iapetus": DOGLEG_APOAPSIS, "Phoebe": 1.2 * PHOEBE_A}


def screen(
    window: int,
    site: str,
    ra: float,
    prograde: bool,
    rp: float,
    max_years: Optional[float] = None,
) -> Optional[Tuple[Vec, Vec, Geometry]]:
    """The best apse line of one sense on the cone, ring-checked, by its leg.

    Clear beats dusty; blocked lines are dropped.  The leg is costed from
    periapsis ``rp`` (the nominal pass's exit periapsis once flown).

    Returns:
        ``(apse direction, orbit normal, geometry without a corridor)``.
    """
    arc = outbound_arc(window)
    excess = float(np.linalg.norm(arc.saturn_excess))
    s_hat = _unit(arc.saturn_excess)
    angle = asymptote_angle(excess, R_EQ + NOMINAL_ALTITUDE)
    axis = -s_hat
    e1 = _unit(np.cross(axis, SATURN_POLE))
    e2 = np.cross(axis, e1)
    best: Optional[Tuple[Tuple[int, float], Vec, Vec, Geometry]] = None
    for k in range(_CONE_STEPS):
        phi = 2.0 * math.pi * k / _CONE_STEPS
        apse = _unit(
            math.cos(angle) * axis
            + math.sin(angle) * (math.cos(phi) * e1 + math.sin(phi) * e2)
        )
        h = _unit(np.cross(s_hat, apse))
        inc = math.degrees(math.acos(float(np.clip(h @ SATURN_POLE, -1.0, 1.0))))
        if (inc < 90.0) != prograde:
            continue
        crossings = ring_crossings(-apse, h, rp, excess, ra)
        rings = ring_status(crossings)
        if rings == "blocked":
            continue
        leg = moon_leg(apse, h, rp, ra, site, max_years)
        if leg is None:
            continue
        score = (_RING_RANK[rings], leg.dv)
        if best is None or score < best[0]:
            g = Geometry(
                site,
                window,
                excess,
                inc,
                math.degrees(math.asin(float(-apse @ SATURN_POLE))),
                rings,
                tuple(r for _, r in crossings),
                ra,
                None,
                leg,
                math.nan,
            )
            best = (score, apse, h, g)
    return None if best is None else (best[1], best[2], best[3])


@lru_cache(maxsize=None)
def flown_pass(
    window: int, site: str, prograde: bool, vehicle: Vehicle = VEHICLE
) -> Optional[Corridor]:
    """The corridor of the window's best geometry of one sense, at the base apoapsis."""
    ra = BASE_APOAPSIS[site]
    picked = screen(window, site, ra, prograde, R_EQ + NOMINAL_ALTITUDE)
    if picked is None:
        return None
    apse, h, g = picked
    return corridor(-apse, h, g.excess, ra, vehicle)


@lru_cache(maxsize=None)
def geometry(
    window: int,
    site: str,
    ra: Optional[float] = None,
    max_years: Optional[float] = None,
    vehicle: Vehicle = VEHICLE,
) -> Optional[Geometry]:
    """The window's best ring-passable aerocapture at ``site``.

    For each sense (prograde, retrograde) the pass is flown once
    (:func:`flown_pass`); the leg is re-screened at capture apoapsis ``ra``
    from that pass's exit periapsis, and the cheaper of the two senses (leg
    plus aeroshell as methalox delta-v; clear before dusty) is kept.
    """
    ra = BASE_APOAPSIS[site] if ra is None else ra
    best: Optional[Geometry] = None
    for prograde in (True, False):
        c = flown_pass(window, site, prograde, vehicle)
        if c is None:
            continue
        picked = screen(window, site, ra, prograde, c.nominal.periapsis, max_years)
        if picked is None:
            continue
        flown = replace(picked[2], corridor=c, aeroshell=aeroshell_share(c.heat_load))
        if best is None or (_RING_RANK[flown.rings], flown.equivalent_dv) < (
            _RING_RANK[best.rings],
            best.equivalent_dv,
        ):
            best = flown
    return best


def summary(
    site: str,
    ra: Optional[float] = None,
    max_years: Optional[float] = None,
    windows: Sequence[int] = WINDOWS,
    vehicle: Vehicle = VEHICLE,
) -> Dict[str, float]:
    """Medians over the windows that close, and how many are clear."""
    flown = [geometry(w, site, ra, max_years, vehicle) for w in windows]
    ok = [
        (g, g.corridor, g.leg)
        for g in flown
        if g is not None and g.corridor is not None and g.leg is not None
    ]
    out: Dict[str, float] = {
        "windows": float(len(windows)),
        "flyable": float(len(ok)),
        "clear": float(sum(g.rings == "clear" for g, _, _ in ok)),
        "prograde": float(sum(g.inclination < 90.0 for g, _, _ in ok)),
    }
    if not ok:
        return out
    getters: Dict[str, Callable[[Geometry, Corridor, MoonLeg], float]] = {
        "leg": lambda g, c, leg: leg.dv,
        "burn": lambda g, c, leg: leg.burn,
        "match": lambda g, c, leg: leg.match,
        "leg_years": lambda g, c, leg: leg.years,
        "aeroshell": lambda g, c, leg: g.aeroshell,
        "heat_load": lambda g, c, leg: c.heat_load,
        "peak_heat": lambda g, c, leg: c.peak_heat,
        "peak_g": lambda g, c, leg: c.peak_g,
        "exit_periapsis": lambda g, c, leg: c.nominal.periapsis,
        "entry_speed": lambda g, c, leg: c.nominal.entry_speed,
        "drag_dv": lambda g, c, leg: c.nominal.drag_dv,
        "width": lambda g, c, leg: c.width_deg,
        "width_km": lambda g, c, leg: c.shallow - c.steep,
    }
    for key, get in getters.items():
        out[key] = float(np.median([get(*x) for x in ok]))
    out["aeroshell_worst"] = max(g.aeroshell for g, _, _ in ok)
    out["leg_worst"] = max(leg.dv for _, _, leg in ok)
    return out


def report() -> str:
    """Per-window passes at both moons, and the medians."""
    rows = []
    for w in WINDOWS:
        for site in ("Iapetus", "Phoebe"):
            g = geometry(w, site)
            year = f"{2035 + w * SYNODIC_DAYS / 365.25:.1f}"
            if g is None or g.corridor is None or g.leg is None:
                rows.append([year, site, "no ring-passable geometry closes"])
                continue
            c, n, leg = g.corridor, g.corridor.nominal, g.leg
            rows.append(
                [
                    year,
                    site,
                    f"{g.excess:.2f}",
                    f"{g.inclination:.0f}",
                    f"{g.periapsis_lat:.0f}",
                    f"{g.rings} ({', '.join(f'{r / 1e3:,.0f}' for r in g.crossings)})",
                    f"{n.entry_speed:.1f}",
                    f"{c.width_deg:.2f}",
                    f"{n.drag_dv:.2f}",
                    f"{(n.periapsis - R_EQ):,.0f}",
                    f"{c.peak_heat:,.0f}",
                    f"{c.heat_load / 1e3:,.0f}",
                    f"{c.peak_g:.1f}",
                    f"{g.aeroshell:.1%}",
                    f"{g.capture_apoapsis / 1e6:.1f}",
                    f"{leg.burn:.2f} + {leg.match:.2f}",
                    f"{leg.meet_angle:.0f}",
                    f"{leg.years:.2f}",
                ]
            )
    v = VEHICLE
    head = (
        f"Aerocapture at Saturn on the real arcs (beta {v.beta:.0f} kg/m^2, L/D {v.lift_drag}, "
        f"{v.entry_mass / 1e3:.0f} t, shield {v.diameter:.0f} m, nose {v.nose_radius:.0f} m).\n"
        "Ring crossings in 1000 km; heat and g the worst of steep, shallow and nominal;\n"
        "aeroshell = HEEET fit on the heat load + 10% structure; leg = methalox burn at the\n"
        "first apoapsis + match at the moon (meet angle 0 = the tangent dog-leg).\n"
    )
    table = tabulate(
        rows,
        headers=[
            "window",
            "site",
            "v_inf",
            "incl",
            "peri lat",
            "rings",
            "V_rel EI",
            "corridor deg",
            "drag dv",
            "exit peri - R_eq km",
            "peak W/cm^2",
            "load kJ/cm^2",
            "peak g",
            "aeroshell",
            "apo 1e6 km",
            "burn + match",
            "meet deg",
            "leg yr",
        ],
    )
    lines = []
    for site in ("Iapetus", "Phoebe"):
        m = summary(site)
        if "leg" not in m:
            lines.append([site, f"0 of {m['windows']:.0f}"])
            continue
        lines.append(
            [
                site,
                f"{m['flyable']:.0f} of {m['windows']:.0f} ({m['clear']:.0f} clear, {m['prograde']:.0f} prograde)",
                f"{m['entry_speed']:.1f}",
                f"{m['peak_heat']:,.0f}",
                f"{m['heat_load'] / 1e3:,.0f}",
                f"{m['peak_g']:.1f}",
                f"{m['aeroshell']:.1%} (worst {m['aeroshell_worst']:.1%})",
                f"{m['leg']:.2f} (worst {m['leg_worst']:.2f})",
                f"{m['leg_years']:.2f}",
            ]
        )
    return (
        head
        + table
        + "\n\nMedians over the windows that close:\n"
        + tabulate(
            lines,
            headers=[
                "site",
                "windows",
                "V_rel EI",
                "peak W/cm^2",
                "load kJ/cm^2",
                "peak g",
                "aeroshell",
                "leg km/s",
                "leg yr",
            ],
        )
    )


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------

#: Phoebe's capture options the ledger chooses among on the rNPV: the first
#: apoapsis, and the longest leg to the moon allowed (years; None: no cap).
PHOEBE_OPTIONS: Tuple[Tuple[float, Optional[float]], ...] = tuple(
    (ra, cap) for ra in CAPTURE_APOAPSES["Phoebe"] for cap in (0.5, 0.75, 1.0, None)
)


def _coplanar_leg(site: str, rp: float) -> float:
    """The coplanar, tangent methalox leg after the pass (km/s)."""
    if site == "Iapetus":
        ra = DOGLEG_APOAPSIS
        raise_ = _vis_viva(ra, 0.5 * (IAPETUS_A + ra)) - _vis_viva(ra, 0.5 * (rp + ra))
        match = _vis_viva(IAPETUS_A, 0.5 * (IAPETUS_A + ra)) - math.sqrt(
            MU_SATURN / IAPETUS_A
        )
        return raise_ + match
    return math.sqrt(MU_SATURN / PHOEBE_A) - _vis_viva(PHOEBE_A, 0.5 * (rp + PHOEBE_A))


@dataclass(frozen=True)
class AeroCase:
    """The aerocapture assumptions one ledger row takes (assumptions marked).

    Attributes:
        label: Row name.
        vehicle: The entering ship.
        fit: The thermal-protection fit: ``HEEET`` or ``carbon phenolic``.
        structure: Structure, backshell and mechanisms per kg of entry mass.
        seed_aerocaptures: The seed flies the pass too (base: it keeps the
            capture burn, as the first of its kind).
        price: Aeroshell $/kg, or None for the ledger's
            :data:`~src.iapetus_cycle.AEROSHELL_PRICE`.
    """

    label: str
    vehicle: Vehicle = VEHICLE
    fit: str = "HEEET"
    structure: float = STRUCTURE
    seed_aerocaptures: bool = False
    price: Optional[float] = None

    def prices(self) -> "CyclePrices":
        """The ledger's prices with this row's aeroshell price."""
        from src.iapetus_cycle import ESTIMATE

        return (
            ESTIMATE if self.price is None else replace(ESTIMATE, aeroshell=self.price)
        )


BASE_CASE = AeroCase("base (HEEET, 10% structure, $1000/kg)")


def aero_site(
    site: str,
    real: bool,
    ra: Optional[float] = None,
    max_years: Optional[float] = None,
    case: AeroCase = BASE_CASE,
) -> Optional["Site"]:
    """A :class:`~src.saturn_methalox.Site` whose recurring ships aerocapture.

    The aeroshell is the windows' median, the burns after the pass are the
    methalox leg plus the landing and the targeting correction.  Coplanar
    rows take the tangent leg from the median exit periapsis and keep the
    old leg time; real-plane rows take the screened legs' medians and their
    time.  The seed keeps the capture burn unless ``case.seed_aerocaptures``.
    """
    from src.iapetus_cycle import dogleg_dvs
    from src.saturn_methalox import IAPETUS, PHOEBE, PHOEBE_LANDING, real_plane_sites

    m = summary(site, ra, max_years, vehicle=case.vehicle)
    if "leg" not in m:
        return None
    shell = aeroshell_share(m["heat_load"], case.fit, case.structure)
    land = dogleg_dvs()["land"] if site == "Iapetus" else PHOEBE_LANDING
    base = {
        (False, "Iapetus"): IAPETUS,
        (False, "Phoebe"): PHOEBE,
        (True, "Iapetus"): real_plane_sites()[0],
        (True, "Phoebe"): real_plane_sites()[1],
    }[(real, site)]
    if real:
        after = m["leg"] + land + TARGETING
        arrival: Optional[float] = m["leg_years"]
    else:
        after = _coplanar_leg(site, m["exit_periapsis"]) + land + TARGETING
        arrival = None
    label = "aerocapture" + ("" if ra is None else f", apo {ra / 1e6:.1f}e6 km")
    if max_years is not None:
        label += f", leg <= {max_years:g} yr"
    return replace(
        base,
        name=label,
        outbound_dv=after,
        aeroshell=shell,
        seed_outbound_dv=None if case.seed_aerocaptures else base.outbound_dv,
        arrival_years=arrival,
    )


Valued = Tuple["PlantCycle", "PlantValuation", "PlantValuation"]


def _plant_row(
    site: "Site", book: "OutboundBook", case: AeroCase = BASE_CASE
) -> Valued:
    """The electric-plant cycle at ``site``, valued at 10% and on the 4% loan."""
    from src.saturn_methalox import ELECTRIC, build_plant_cycle, value_plant_cycle

    c = build_plant_cycle(site=site, plant=ELECTRIC, book=book, prices=case.prices())
    return c, value_plant_cycle(c), value_plant_cycle(c, 0.04)


def _fmt(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value) or value > 1.0e6:
        return "does not grow"
    return f"{value:,.0f}" if value >= 100 else f"{value:.2f}"


def best_phoebe(
    real: bool, book: "OutboundBook", case: AeroCase = BASE_CASE
) -> Optional[Tuple["Site", Valued]]:
    """Phoebe's capture option with the lowest gated rNPV (methalox, electric).

    Coplanar rows have one option (the tangent capture onto Phoebe); on the
    real planes every :data:`PHOEBE_OPTIONS` pair is valued.
    """
    best: Optional[Tuple[float, "Site", Valued]] = None
    options = PHOEBE_OPTIONS if real else ((None, None),)
    for ra, cap in options:
        site = aero_site("Phoebe", real, ra, cap, case)
        if site is None:
            continue
        valued = _plant_row(site, book, case)
        v = valued[1]
        score = v.break_even if v.break_even is not None else math.inf
        if best is None or score < best[0]:
            best = (score, site, valued)
    return None if best is None else (best[1], best[2])


def _water_row(
    real: bool, book: "OutboundBook", case: Optional[AeroCase]
) -> Tuple[float, float, Optional[float], Optional[float], Optional[float]]:
    """The water NTR at Iapetus, capture burn (``case`` None) or aerocaptured.

    Returns:
        ``(outbound ratio, growth, steady, rNPV 10%, loan 4%)``.
    """
    from src.iapetus_cycle import (
        ESTIMATE,
        LEDGER_A,
        best_k,
        build_cycle,
        value_cycle,
    )

    k, _ = best_k(LEDGER_A, book=book, real_planes=real)
    aero: Optional[Tuple[float, float]] = None
    prices = ESTIMATE
    if case is not None:
        site = aero_site("Iapetus", real, case=case)
        assert site is not None
        aero = (site.outbound_dv, site.aeroshell)
        prices = case.prices()
    c = build_cycle(LEDGER_A, k, prices=prices, book=book, real_planes=real, aero=aero)
    v, loan = value_cycle(c), value_cycle(c, 0.04)
    return c.out.ratio, c.growth(), v.steady_cost, v.break_even, loan.break_even


def ledger_report() -> str:
    """Capture burn against aerocapture, every Saturn option, both scenarios."""
    from src.iapetus_cycle import SCENARIOS
    from src.saturn_methalox import IAPETUS, PHOEBE, real_plane_sites

    def plant_cells(valued: Valued) -> List[str]:
        c, v, loan = valued
        return [
            f"{c.out.ratio:.2f}",
            f"{c.growth():.2f}",
            f"{c.round_years:.2f}",
            _fmt(v.steady_cost),
            _fmt(v.break_even),
            _fmt(loan.break_even),
        ]

    out = []
    for name, book in SCENARIOS:
        rows = []
        for real in (False, True):
            planes = "real" if real else "coplanar"
            for label, case in (("capture burn", None), ("aerocapture", BASE_CASE)):
                ratio, growth, steady, rnpv, loan = _water_row(real, book, case)
                rows.append(
                    [
                        "water NTR, Iapetus",
                        planes,
                        label,
                        f"{ratio:.2f}",
                        f"{growth:.2f}",
                        "9.50",
                    ]
                    + [_fmt(steady), _fmt(rnpv), _fmt(loan)]
                )
            iapetus, phoebe = real_plane_sites() if real else (IAPETUS, PHOEBE)
            rows.append(
                ["methalox, Iapetus", planes, "capture burn"]
                + plant_cells(_plant_row(iapetus, book))
            )
            aero = aero_site("Iapetus", real)
            assert aero is not None
            rows.append(
                ["methalox, Iapetus", planes, aero.name]
                + plant_cells(_plant_row(aero, book))
            )
            rows.append(
                ["methalox, Phoebe", planes, "capture burn"]
                + plant_cells(_plant_row(phoebe, book))
            )
            found = best_phoebe(real, book)
            assert found is not None
            rows.append(
                ["methalox, Phoebe", planes, found[0].name] + plant_cells(found[1])
            )
        out.append(
            f"Scenario: {name}. Ledger A, k 4 compact, electric plant, $/kg at the eccentric orbit;\n"
            "the seed keeps the capture burn. Outbound = kg leaving the eccentric orbit per kg landed.\n"
            + tabulate(
                rows,
                headers=[
                    "Saturn side",
                    "planes",
                    "capture",
                    "outbound",
                    "G",
                    "T yr",
                    "steady",
                    "rNPV 10%",
                    "loan 4%",
                ],
            )
        )
    return "\n\n".join(out)


#: One assumption moved at a time.
SENSITIVITIES: Tuple[AeroCase, ...] = (
    BASE_CASE,
    AeroCase("aeroshell $5000/kg", price=5000.0),
    AeroCase("carbon-phenolic fit", fit="carbon phenolic"),
    AeroCase("structure 15%", structure=0.15),
    AeroCase("seed aerocaptures too", seed_aerocaptures=True),
    AeroCase("L/D 0.4", vehicle=replace(VEHICLE, lift_drag=0.4)),
    AeroCase("beta 150 kg/m^2", vehicle=replace(VEHICLE, beta=150.0)),
    AeroCase("beta 1500 kg/m^2", vehicle=replace(VEHICLE, beta=1500.0)),
)


def sensitivity_report() -> str:
    """The best Saturn cells on the real planes, one assumption moved at a time."""
    from src.iapetus_cycle import SCENARIOS

    rows = []
    for case in SENSITIVITIES:
        m = summary("Iapetus", vehicle=case.vehicle)
        site = aero_site("Iapetus", True, case=case)
        assert site is not None
        cells = [
            case.label,
            f"{m['width']:.2f} ({m['width_km']:.0f} km)",
            f"{m['peak_heat']:,.0f}",
            f"{m['peak_g']:.1f}",
            f"{site.aeroshell:.1%}",
        ]
        for _, book in SCENARIOS:
            cells.append(_fmt(_water_row(True, book, case)[3]))
            found = best_phoebe(True, book, case)
            cells.append(_fmt(found[1][1].break_even) if found else "-")
        rows.append(cells)
    return (
        "Sensitivities on the real planes (rNPV 10%, Ledger A). Corridor (entry angle, and vacuum\n"
        "periapsis in km), peak heat, peak g and aeroshell are Iapetus medians for the row's vehicle.\n"
        "The water NTR's seed always keeps its capture burn.\n"
        + tabulate(
            rows,
            headers=[
                "row",
                "corridor deg (km)",
                "peak W/cm^2",
                "peak g",
                "aeroshell",
                "H2 NTP: water NTR Iapetus",
                "methalox Phoebe",
                "chemical: water NTR Iapetus",
                "methalox Phoebe",
            ],
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the passes, the ledger and the sensitivities."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    print(report())
    print()
    print(ledger_report())
    print()
    print(sensitivity_report())


if __name__ == "__main__":
    main()
