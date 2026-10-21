"""Saturn-side legs on the real asymptotes and the moons' real planes (S51 item 6).

The parent's Saturn-side delta-v is coplanar: circular moons in one plane
with the arrival.  This module flies the Earth-Saturn and Saturn-Earth arcs
as Lambert solutions on astropy's built-in ephemeris (2035-2050 windows), and
then places each capture or departure orbit in 3-D:

- A hyperbola with excess ``v`` at periapsis ``rp`` has its apoapsis (after a
  capture burn, or before a departure burn) at a fixed angle from its
  asymptote, ``pi - nu_inf``, but anywhere on that cone about it.
- The dog-leg to Iapetus needs the apse line *in Iapetus's plane*, so the
  raised periapsis lands on its orbit.  Phoebe needs the apoapsis on
  Phoebe's orbit.  Either is a cone meeting a great circle: at most two
  apse directions.
- The burn at apoapsis then turns the plane onto the moon's, rotating about
  the apse line, and costs the vector difference of the two velocities.

The moons' planes are Horizons osculating elements (``sat441l``) on the
ecliptic of J2000 at 2042-07-01.  Over 2035-2050, Iapetus moves from 16.93 to
16.75 deg and node 138.27 to 137.83 deg, and Phoebe from 172.91 to 172.88 deg
and node 273.4 to 278.0 deg, so one epoch serves.  Phoebe's orbit is
eccentric (0.164): it is met at 10.9, 12.95 or 15.1 million km.

Phasing (the moon being where the apse line points when the ship gets there)
is not solved: it is a matter of when, within a window, the ship arrives.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from boinor.core.iod import izzo
from tabulate import tabulate

from src.real_orbit_resonance import _heliocentric_state

Vec = npt.NDArray[np.float64]

MU_SUN = 1.32712440018e11
MU_SATURN = 37_931_187.0
MU_EARTH = 398_600.4418
R_EARTH = 6378.137
DAY = 86400.0
YEAR = 365.25 * DAY
SYNODIC_DAYS = 378.09  # Earth-Saturn

CAPTURE_PERIAPSIS = 150_000.0
DOGLEG_APOAPSIS = 20.0e6
IAPETUS_A = 3_560_820.0
#: Phoebe (Horizons 2042-07-01): a, e.
PHOEBE_A = 12_947_780.0
PHOEBE_E = 0.1635
PHOEBE_RADII = (
    PHOEBE_A * (1.0 - PHOEBE_E),
    PHOEBE_A,
    PHOEBE_A * (1.0 + PHOEBE_E),
)
#: Earth arrival altitude the closing speed is quoted at.
ARRIVAL_ALTITUDE = 200.0
OUT_YEARS = 3.0
RETURN_YEARS = 3.7
CLOSING = 51.1
#: The eccentric orbit's perigee speed at 200 km (the 10-day orbit).
ECCENTRIC_SPEED = 10.916
#: Ring radii (km): the F ring's outer edge and the G ring.
F_RING = 140_220.0
G_RING = (166_000.0, 175_000.0)


def _unit(v: Vec) -> Vec:
    return np.asarray(v / np.linalg.norm(v), dtype=np.float64)


def plane_normal(inclination_deg: float, node_deg: float) -> Vec:
    """An orbit's angular-momentum direction from its inclination and node."""
    i, o = math.radians(inclination_deg), math.radians(node_deg)
    return np.array(
        [math.sin(i) * math.sin(o), -math.sin(i) * math.cos(o), math.cos(i)]
    )


#: Iapetus and Phoebe, ecliptic J2000 (Horizons, 2042-07-01).
IAPETUS_NORMAL = plane_normal(16.778, 138.167)
PHOEBE_NORMAL = plane_normal(172.773, 277.099)


def _saturn_pole() -> Vec:
    """Saturn's spin axis (IAU: RA 40.589, Dec 83.537 deg), on the ecliptic."""
    ra, dec = math.radians(40.589), math.radians(83.537)
    eq = np.array(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)]
    )
    eps = math.radians(23.4392911)
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(eps), math.sin(eps)],
            [0.0, -math.sin(eps), math.cos(eps)],
        ]
    )
    return _unit(rot @ eq)


SATURN_POLE = _saturn_pole()


def _vis_viva(r: float, a: float, mu: float = MU_SATURN) -> float:
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def asymptote_angle(excess: float, rp: float = CAPTURE_PERIAPSIS) -> float:
    """The apoapsis's angle from the asymptote, ``pi - nu_inf`` (rad)."""
    ecc = 1.0 + rp * excess**2 / MU_SATURN
    return math.pi - math.acos(-1.0 / ecc)


def apse_options(axis: Vec, angle: float, normal: Vec) -> List[Vec]:
    """Unit vectors in the plane of ``normal`` at ``angle`` from ``axis``."""
    e1 = _unit(
        np.cross(normal, np.array([0.0, 0.0, 1.0]))
        if abs(normal[2]) < 0.999
        else np.array([1.0, 0.0, 0.0])
    )
    e2 = _unit(np.cross(normal, e1))
    u1, u2 = float(axis @ e1), float(axis @ e2)
    rho = math.hypot(u1, u2)
    c = math.cos(angle)
    if rho < abs(c):
        return []
    base = math.atan2(u2, u1)
    spread = math.acos(c / rho)
    return [
        _unit(math.cos(base + s) * e1 + math.sin(base + s) * e2)
        for s in (spread, -spread)
    ]


@dataclass(frozen=True)
class ApseBurn:
    """The apoapsis burn that joins a Saturn hyperbola to a moon's plane.

    Attributes:
        dv: The burn (km/s): periapsis shift, plane turn and any match.
        coplanar: The same burn with no plane turn (km/s).
        turn_deg: The plane turn at apoapsis.
        node_radii: Where the 150 000 km orbit crosses Saturn's equator (km).
    """

    dv: float
    coplanar: float
    turn_deg: float
    node_radii: Tuple[float, float]


def _node_radii(h: Vec, periapsis: Vec, rp: float, ra: float) -> Tuple[float, float]:
    """Radii of the ellipse's two equator crossings (km)."""
    line = np.cross(SATURN_POLE, h)
    if np.linalg.norm(line) < 1e-9:
        return (math.inf, math.inf)
    line = _unit(line)
    ecc = (ra - rp) / (ra + rp)
    p = rp * (1.0 + ecc)
    out = []
    for d in (line, -line):
        nu = math.atan2(float(np.cross(periapsis, d) @ h), float(periapsis @ d))
        out.append(p / (1.0 + ecc * math.cos(nu)))
    return (min(out), max(out))


def apse_burn(
    asymptote: Vec,
    excess: float,
    normal: Vec,
    apoapsis: float,
    after: Tuple[float, float],
    inbound: bool,
) -> Optional[ApseBurn]:
    """The cheapest apoapsis burn joining a hyperbola to a moon's plane.

    Args:
        asymptote: Unit excess direction (arrival: the incoming velocity;
            departure: the outgoing velocity).
        excess: Its speed (km/s).
        normal: The moon's orbit normal.
        apoapsis: The 150 000 km ellipse's apoapsis (km).
        after: ``(semi-major axis, radius)`` of the orbit the burn joins, in the
            moon's plane and sense: the dog-leg's raised ellipse, or the moon.
        inbound: Arrival (True) or departure.

    Returns:
        The burn, or None if no apse line lies in the moon's plane.
    """
    angle = asymptote_angle(excess)
    axis = -asymptote if inbound else asymptote
    v_low = _vis_viva(apoapsis, 0.5 * (CAPTURE_PERIAPSIS + apoapsis))
    v_high = _vis_viva(after[1], after[0])
    best: Optional[ApseBurn] = None
    for apse in apse_options(axis, angle, normal):
        h = _unit(np.cross(asymptote, apse))
        moving = np.cross(h, apse)
        moon = np.cross(normal, apse)
        dv = float(np.linalg.norm(v_high * moon - v_low * moving))
        cos_turn = float(np.clip(moving @ moon, -1.0, 1.0))
        burn = ApseBurn(
            dv=dv,
            coplanar=abs(v_high - v_low),
            turn_deg=math.degrees(math.acos(cos_turn)),
            node_radii=_node_radii(h, -apse, CAPTURE_PERIAPSIS, apoapsis),
        )
        if best is None or burn.dv < best.dv:
            best = burn
    return best


# --------------------------------------------------------------------------
# The heliocentric arcs
# --------------------------------------------------------------------------

_J2035 = 2464328.5  # 2035-01-01 TDB


def _lambert(r0: Vec, r1: Vec, tof: float, prograde: bool) -> Optional[Tuple[Vec, Vec]]:
    try:
        v0, v1 = izzo(MU_SUN, r0, r1, tof, 0, prograde, True, 80, 1e-10)
    except (AssertionError, RuntimeError, ValueError):
        return None
    return np.asarray(v0), np.asarray(v1)


@dataclass(frozen=True)
class Arc:
    """One Earth-Saturn or Saturn-Earth arc.

    Attributes:
        depart_jd: Departure (TDB Julian date).
        years: Time of flight.
        earth_excess: Earth departure or arrival excess (km/s).
        saturn_excess: Saturn arrival or departure excess vector (km/s).
        closing: Speed at 200 km (arrivals at Earth only).
    """

    depart_jd: float
    years: float
    earth_excess: float
    saturn_excess: Vec
    closing: float


@lru_cache(maxsize=None)
def outbound_arc(window: int, years: float = OUT_YEARS) -> Arc:
    """The window's cheapest Earth departure onto a ``years`` arc to Saturn."""
    best: Optional[Arc] = None
    start = _J2035 + window * SYNODIC_DAYS
    for jd in np.arange(start, start + SYNODIC_DAYS, 3.0):
        re, ve = _heliocentric_state("earth", float(jd))
        rs, vs = _heliocentric_state("saturn", float(jd) + years * 365.25)
        sol = _lambert(re, rs, years * YEAR, True)
        if sol is None:
            continue
        excess = float(np.linalg.norm(sol[0] - ve))
        if best is None or excess < best.earth_excess:
            best = Arc(float(jd), years, excess, sol[1] - vs, 0.0)
    assert best is not None
    return best


@lru_cache(maxsize=None)
def return_arc(
    window: int, closing: float = CLOSING, max_years: float = RETURN_YEARS
) -> Optional[Arc]:
    """The window's smallest Saturn excess that meets Earth at ``closing``.

    Scans departure date (3-day steps across the window) and time of flight
    (3.0 years to ``max_years``), both senses of the heliocentric arc, and
    keeps the smallest Saturn excess whose speed at 200 km reaches
    ``closing``.
    """
    best: Optional[Arc] = None
    start = _J2035 + window * SYNODIC_DAYS
    for jd in np.arange(start, start + SYNODIC_DAYS, 3.0):
        rs, vs = _heliocentric_state("saturn", float(jd))
        for years in np.arange(3.0, max_years + 1e-9, 0.1):
            re, ve = _heliocentric_state("earth", float(jd) + years * 365.25)
            for prograde in (True, False):
                sol = _lambert(rs, re, years * YEAR, prograde)
                if sol is None:
                    continue
                arrive = float(np.linalg.norm(sol[1] - ve))
                speed = math.sqrt(
                    arrive**2 + 2.0 * MU_EARTH / (R_EARTH + ARRIVAL_ALTITUDE)
                )
                if speed < closing:
                    continue
                excess = sol[0] - vs
                if best is None or np.linalg.norm(excess) < np.linalg.norm(
                    best.saturn_excess
                ):
                    best = Arc(float(jd), float(years), arrive, excess, speed)
    return best


# --------------------------------------------------------------------------
# The legs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteLegs:
    """One window's Saturn-side delta-v at one moon (km/s).

    Attributes:
        site: ``Iapetus`` or ``Phoebe at <radius>``.
        window: Window index from 2035.
        capture: Capture at 150 000 km onto the ellipse.
        apse: The apoapsis burn on the way in (raise and turn, or match).
        apse_coplanar: The same with no turn.
        arrival_turn: Its plane turn (deg).
        home_apse: The apoapsis burn on the way out.
        home_coplanar: The same with no turn.
        departure: The periapsis burn for Earth (impulsive).
        saturn_excess: The return's Saturn excess.
        node_min: The lowest equator crossing of either 150 000 km ellipse.
    """

    site: str
    window: int
    capture: float
    apse: float
    apse_coplanar: float
    arrival_turn: float
    home_apse: float
    home_coplanar: float
    departure: float
    saturn_excess: float
    node_min: float


def site_legs(
    window: int, site: str = "Iapetus", phoebe_radius: float = PHOEBE_A
) -> Optional[SiteLegs]:
    """Both Saturn-side apse burns for a window, on the real planes."""
    out = outbound_arc(window)
    home = return_arc(window + (6 if site == "Iapetus" else 4))
    if home is None:
        return None
    v_in = float(np.linalg.norm(out.saturn_excess))
    v_out = float(np.linalg.norm(home.saturn_excess))
    if site == "Iapetus":
        normal, apo = IAPETUS_NORMAL, DOGLEG_APOAPSIS
        after = (0.5 * (IAPETUS_A + DOGLEG_APOAPSIS), DOGLEG_APOAPSIS)
    else:
        normal, apo = PHOEBE_NORMAL, phoebe_radius
        after = (PHOEBE_A, phoebe_radius)
    arrive = apse_burn(_unit(out.saturn_excess), v_in, normal, apo, after, True)
    leave = apse_burn(_unit(home.saturn_excess), v_out, normal, apo, after, False)
    if arrive is None or leave is None:
        return None
    vp = _vis_viva(CAPTURE_PERIAPSIS, 0.5 * (CAPTURE_PERIAPSIS + apo))
    capture = math.sqrt(v_in**2 + 2.0 * MU_SATURN / CAPTURE_PERIAPSIS) - vp
    departure = math.sqrt(v_out**2 + 2.0 * MU_SATURN / CAPTURE_PERIAPSIS) - vp
    return SiteLegs(
        site=site if site == "Iapetus" else f"Phoebe at {phoebe_radius / 1e6:.2f}e6 km",
        window=window,
        capture=capture,
        apse=arrive.dv,
        apse_coplanar=arrive.coplanar,
        arrival_turn=arrive.turn_deg,
        home_apse=leave.dv,
        home_coplanar=leave.coplanar,
        departure=departure,
        saturn_excess=v_out,
        node_min=min(arrive.node_radii[0], leave.node_radii[0]),
    )


WINDOWS = tuple(range(0, 13))


def site_summary(
    site: str, phoebe_radius: float = PHOEBE_A, windows: Sequence[int] = WINDOWS
) -> Dict[str, float]:
    """Medians over the windows of each leg (km/s), and the worst."""
    legs = [
        x for x in (site_legs(w, site, phoebe_radius) for w in windows) if x is not None
    ]
    if not legs:
        return {}
    keys = (
        "capture",
        "apse",
        "apse_coplanar",
        "home_apse",
        "home_coplanar",
        "departure",
        "saturn_excess",
        "node_min",
    )
    out = {k: float(np.median([getattr(x, k) for x in legs])) for k in keys}
    excess = float(
        np.median([np.linalg.norm(outbound_arc(w).earth_excess) for w in windows])
    )
    rp = R_EARTH + ARRIVAL_ALTITUDE
    out["earth_excess"] = excess
    out["earth_dv"] = math.sqrt(excess**2 + 2.0 * MU_EARTH / rp) - ECCENTRIC_SPEED
    out["apse_worst"] = max(x.apse for x in legs)
    out["home_worst"] = max(x.home_apse for x in legs)
    out["node_lowest"] = min(x.node_min for x in legs)
    out["windows"] = float(len(legs))
    return out


def report() -> str:
    """Per-window legs at both moons, and the medians."""
    rows = []
    for w in WINDOWS:
        out = outbound_arc(w)
        for site, radius in (("Iapetus", PHOEBE_A), ("Phoebe", PHOEBE_A)):
            x = site_legs(w, site, radius)
            if x is None:
                rows.append(
                    [
                        f"{2035 + w * SYNODIC_DAYS / 365.25:.1f}",
                        site,
                        "no apse in plane / no return",
                    ]
                )
                continue
            rows.append(
                [
                    f"{2035 + w * SYNODIC_DAYS / 365.25:.1f}",
                    x.site,
                    f"{out.earth_excess:.2f}",
                    f"{np.linalg.norm(out.saturn_excess):.2f}",
                    f"{x.capture:.2f}",
                    f"{x.apse:.3f} ({x.apse_coplanar:.3f})",
                    f"{x.arrival_turn:.1f}",
                    f"{x.home_apse:.3f} ({x.home_coplanar:.3f})",
                    f"{x.saturn_excess:.2f}",
                    f"{x.departure:.2f}",
                    f"{x.node_min / 1e3:,.0f}",
                ]
            )
    head = (
        "Saturn-side legs on the real arcs and planes (Lambert on astropy's built-in ephemeris;\n"
        "Horizons planes). Outbound 3.0 yr, the window's cheapest Earth excess; return the smallest\n"
        f"Saturn excess reaching {CLOSING} km/s at 200 km within {RETURN_YEARS} yr. Apse burns: with the turn (coplanar).\n"
    )
    table = tabulate(
        rows,
        headers=[
            "window",
            "site",
            "Earth v_inf",
            "Saturn v_inf in",
            "capture",
            "apse in",
            "turn deg",
            "apse home",
            "Saturn v_inf out",
            "departure",
            "lowest node, 1000 km",
        ],
    )
    lines = []
    for site, radius in [("Iapetus", PHOEBE_A)] + [("Phoebe", r) for r in PHOEBE_RADII]:
        s = site_summary(site, radius)
        lines.append(
            [
                site if site == "Iapetus" else f"Phoebe at {radius / 1e6:.2f}e6 km",
                f"{s['capture']:.2f}",
                f"{s['apse']:.3f} ({s['apse_coplanar']:.3f}); worst {s['apse_worst']:.3f}",
                f"{s['home_apse']:.3f} ({s['home_coplanar']:.3f}); worst {s['home_worst']:.3f}",
                f"{s['saturn_excess']:.2f}",
                f"{s['departure']:.2f}",
                f"{s['node_lowest'] / 1e3:,.0f}",
            ]
        )
    summary = tabulate(
        lines,
        headers=[
            "site",
            "capture",
            "apse in (coplanar)",
            "apse home (coplanar)",
            "Saturn v_inf out",
            "departure",
            "lowest node",
        ],
    )
    return head + table + "\n\nMedians over the windows:\n" + summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the legs."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    print(report())


if __name__ == "__main__":
    main()
