"""The solar-dive charge-pusher cycle through the growth ledger and the cost book.

The impact simulation (``puffsat_impact_simulation`` a6742a2,
``docs/solar_dive_cycle_ask_for_calcs.md``, asks D1-D7) proposes a different
growth loop from the child paper's chamber cycle:

1. **Push.** The returning **Earth wave** (Kapton sheets at ~100 km/s) overtakes
   a lofted launch unit at 400 km and pushes it onto the 10-day parking orbit,
   each sheet striking a polyethylene charge held off a flat plate.
2. **Departure.** Ten days later a second Earth wave meets the ship at the
   600 km periapsis at ~100 degrees to its thrust axis, sweeping past the same
   flat plate (the **canted plate**), and sends it onto the three-synodic
   Earth -> Jupiter -> Sun -> Earth closure of
   :mod:`src.jovian_solar_dive_cycle`.  The plate then diverts into Jupiter.
3. **Jupiter.** The ship rides an unpowered flyby into a prograde dive.  Its
   **Sun-wave** graphite disks ride a carrier bus to a lower perijove and burn
   onto the mirror-image retrograde dive, so they meet the ship head-on.
4. **Sun.** The disks strike polyethylene charges in a paraboloid over an arc
   of hours around periapsis, at 2 Hz.
5. **Return.** The ship's remainder is released as furled sheets: the next
   Earth wave.

Everything is float arithmetic (km, km/s, kg, years, dollars), like
:mod:`src.growth_cost`; the closures come from the Quantity-free conic code.
The stage laws (exhaust speed against slug ratio ``k``) are the impact
simulation's, vendored in ``data/solar_dive_stage_laws.csv``.

**Growth is counted in launch units.**  Both Earth waves are fleet mass that
flies the cycle and returns, like the child's growth PuffSats and rods.  The
Sun-wave disks are not: they are built on Earth every cycle and spent at the
Sun, so they cost launch mass (charged inside the launch unit) and dollars,
but never returned mass.  The simulation's "growth per total impactor" divides
by them too; it is reported as a comparison.

Run with ``make solar-dive``.
"""

import argparse
import csv
import functools
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, fsolve, minimize
from tabulate import tabulate

from src.chamber_departure import role_growth
from src.conic_kernel import (
    half_turn_angle,
    hyperbolic_eccentricity,
    speed_with_escape_energy,
    unpowered_bend_angle,
    wrap_pi,
)
from src.growth_cost import DesignInputs, PriceBook
from src.jovian_solar_dive_cycle import (
    SynodicCycleClosure,
    _dive_periapsis_radius,
    _evaluate,
    _intercept_miss,
    _solve_dive_state,
    departure_nozzle_ledger,
    solve_synodic_closure,
    synodic_period,
)
from src.learning_curve import LearningCurve
from src.retrograde_return_legs import _FlybyParams, _powered_flyby_params
from src.two_wave_growth import VE_METHALOX

STAGE_LAWS = (
    Path(__file__).resolve().parent.parent / "data" / "solar_dive_stage_laws.csv"
)

_MU_EARTH = 398600.4418
_R_EARTH = 6378.137
_G0 = 9.80665e-3  # km/s^2
_R_JUPITER = 71492.0
_SECONDS_PER_YEAR = 365.25 * 86400.0
_DAY = 86400.0

# --- The flown geometry (child's ledger, ADR 0071).
LAUNCH_UNIT = 1500.0e3  # kg lofted per unit (growth_ledger.LAUNCH_UNIT)
PARKING_DAYS = 10.0  # growth_ledger.DEFAULT_PARKING_DAYS: the split gap
PUSH_ALTITUDE = 400.0
DEPARTURE_ALTITUDE = 600.0
#: The child's pulse (growth_ledger.PULSE_IMPULSE).  Keeping it keeps the
#: child's push schedule, so its lob and gravity charges carry over.
PULSE_IMPULSE = 12.0e6  # N s
#: Sun pulse rate cap (author, 2026-10-16: the methane chamber's 2 Hz).  The
#: Earth departure fires at the same rate.
SUN_PULSE_RATE = 2.0  # Hz
EARTH_PULSE_RATE = 2.0  # Hz
_LOSS_ITERATIONS = 6

# --- Plates (impact simulation's charge ledger and overtake handoff).
SURVIVAL_AREAL = 30.0  # kg/m^2, the parent's "tens of kilograms per square metre"
#: The flat plate with a 4 m skirt (charge_ledger "flat + skirt 4 m"), and the
#: impactor per pulse it takes at 400 MPa (overtake handoff: ~11-14 kg).
FLAT_PLATE_AREA = 565.487  # m^2
FLAT_PLATE_PULSE = 12.5  # kg of ~100 km/s sheet per pulse
#: The delta/D 0.5 paraboloid at 1.5 F, rated for a 1 kg disk at 400 km/s
#: (the simulation's own dead-fraction convention, mission.plate_dead_fraction).
SUN_PLATE_AREA = 533.041  # m^2
SUN_PLATE_PULSE = 1.0  # kg at 400 km/s
SUN_PLATE_SPEED = 400.0  # km/s
BORE_COLLAR = 1.0e3  # kg, the simulation's ~0.5-1 t
CHARGE_MOUNT_FRACTION = 0.01  # mounts per kg of charge they hold
ABSORBER_STROKE = 2.0e-3  # km: a short-stroke absorber
G_CAP = 100.0  # peak g the ship's robust cargo takes (author, 2026-10-16)
#: Film per unit impulse: the spray cup's vapor-shielded 4-6 kg per 12 MN s
#: pulse (growth_ledger.SPRAY_CUP_FILM_PER_PULSE).  The Sun plate's is scaled
#: by its gas's energy per unit impulse (~v_e) over the Earth push's, to the
#: power ``film_exponent`` (1 headline, 0 the author's view).
FILM_PER_IMPULSE = 5.0 / 12.0e6  # kg per N s

# --- The Sun ship (assumptions; D3).
SHIELD_FRACTION = 0.005  # Parker-scaled shade, of the ship at the Sun
PARKER_SOLAR_RADII = 9.86  # Parker Solar Probe's perihelion: the shade's flight anchor
AVIONICS_FRACTION = 0.01  # avionics and attitude control, of the ship at the Sun
CARRIER_DRY_FRACTION = 0.10  # carrier bus and stage dry mass, per kg of disk
STORABLE_ISP = 330.0  # s, the carrier's stage (the simulation's)
CARRIER_PERIJOVE = 6.5  # R_J: the highest that still turns the disks (<= 6.7)
PLATE_DIVERT = 0.020  # km/s: the flat plate's nudge into Jupiter

# --- The L1 delivery (harvest.py): fragile cargo keeps the child's plate.
L1_DISTANCE = 1.5e6  # km
HALO_INSERTION = 0.030  # km/s
DELIVERY_PLATE = 156.0e3  # kg: 150 t plate and 6 t absorber

# --- Search bounds (recorded so the optimum is reproducible; CLAUDE.md).
K_PUSH = (5.0, 40.0)
K_EARTH = (5.0, 60.0)
K_SUN = (20.0, 100.0)
ARC_HOURS = (0.5, 12.0)
DEPARTURE_PULSE = (2.0e6, 60.0e6)  # N s
#: Return excesses each depth is closed at; the optimizer picks among them.
RETURN_EXCESSES = (60.0, 75.0, 90.0, 110.0, 130.0, 150.0, 175.0)


# --------------------------------------------------------------------------
# Stage laws
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageLaws:
    """The impact simulation's exhaust speeds, vendored.

    Attributes:
        earth: Canted departure: ``(theta_deg, w km/s, k, v_e km/s)`` rows.
        sun: Sun boost: ``(w km/s, k, v_e km/s)`` rows.
        push: Overtake push at 100 km/s: ``(k, v_e km/s)`` rows.
    """

    earth: Tuple[Tuple[float, float, float, float], ...]
    sun: Tuple[Tuple[float, float, float], ...]
    push: Tuple[Tuple[float, float], ...]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load(path: Path = STAGE_LAWS, bonds: str = "carbon atomic") -> "StageLaws":
        """Read the vendored table.

        Args:
            path: The CSV.
            bonds: ``carbon atomic`` (headline) or ``bonds returned``.

        Returns:
            The laws.
        """
        earth, sun, push = [], [], []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                k, w = float(row["k"]), float(row["w_m_s"]) / 1e3
                v = float(row["v_e_m_s"]) / 1e3
                if row["stage"] == "earth_departure" and row["bonds"] == bonds:
                    earth.append((float(row["theta_deg"]), w, k, v))
                elif row["stage"] == "sun_boost":
                    sun.append((w, k, v))
                elif row["stage"] == "overtake_push":
                    push.append((k, v))
        return StageLaws(tuple(earth), tuple(sun), tuple(push))

    def earth_ve(self, k: float, w: float, theta_deg: float) -> float:
        """Canted departure exhaust speed (km/s).

        The table has the two closures only (9.86 and 4 solar radii).  ``v_e
        / w`` is interpolated linearly in the impact angle between them (held
        flat outside) and scaled by the closing speed: the canted ceiling's
        impulse goes as ``m w``.
        """
        by_theta: Dict[float, List[Tuple[float, float, float]]] = {}
        for theta, speed, kk, v in self.earth:
            by_theta.setdefault(theta, []).append((speed, kk, v))
        thetas = sorted(by_theta)
        ratios = []
        for theta in thetas:
            rows = sorted(by_theta[theta], key=lambda r: r[1])
            ratios.append(
                _log_interp(k, [r[1] for r in rows], [r[2] / r[0] for r in rows])
            )
        return w * float(np.interp(theta_deg, thetas, ratios))

    def sun_ve(self, k: float, w: float) -> float:
        """Sun boost exhaust speed (km/s) at slug ratio ``k`` and closing ``w``."""
        speeds = sorted({row[0] for row in self.sun})
        values = []
        for speed in speeds:
            rows = sorted((r for r in self.sun if r[0] == speed), key=lambda r: r[1])
            values.append(_log_interp(k, [r[1] for r in rows], [r[2] for r in rows]))
        _check(w, speeds[0], speeds[-1], "Sun closing speed")
        return float(np.interp(w, speeds, values))

    def push_ve(self, k: float, w: float) -> float:
        """Overtake push exhaust speed (km/s), carried from 100 km/s by ``w / 100``."""
        rows = sorted(self.push)
        return _log_interp(k, [r[0] for r in rows], [r[1] for r in rows]) * w / 100.0


def _check(value: float, low: float, high: float, name: str) -> None:
    if not low * (1 - 1e-9) <= value <= high * (1 + 1e-9):
        raise ValueError(f"{name} {value:.4g} is outside the table's {low:g}-{high:g}")


def _log_interp(k: float, ks: Sequence[float], values: Sequence[float]) -> float:
    _check(k, ks[0], ks[-1], "slug ratio")
    return float(np.interp(math.log(k), np.log(ks), values))


# --------------------------------------------------------------------------
# Closures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiveRoute:
    """One ship type's three-synodic route, as the ledger needs it.

    Attributes:
        dive_solar_radii: Perihelion.
        return_excess: The climb-out's solar excess, the closure's input (km/s).
        departure_excess: Earth departure excess (km/s).
        departure_aim_deg: Its aim from Earth's prograde (deg).
        cycle_years: Departure to Earth return.
        jupiter_years: Departure to the Jovian flyby.
        sun_years: Departure to perihelion.
        earth_excess: The returning wave's Earth-relative excess (km/s).
        push_axis_deg: Direction the returning wave travels at Earth (deg).
        thrust_axis_deg: The departure burn's thrust axis (deg).
        impact_angle_deg: Mean stream angle to the thrust axis over the burn.
        departure_closing: Mean stream closing speed over the burn (km/s).
        periapsis_boost: Boost the Sun stage gives (km/s).
        periapsis_speed: The ship's speed at perihelion (km/s).
        periapsis_radius: Perihelion radius (km).
        jupiter_arrival_excess: Jupiter-relative arrival excess (km/s).
        mirror_excess: The disks' excess on the mirror-image retrograde dive.
        disk_bend_deg: Turn the disks' flyby must make.
        ship_bend_deg: Turn the ship's own unpowered flyby makes.
    """

    dive_solar_radii: float
    return_excess: float
    departure_excess: float
    departure_aim_deg: float
    cycle_years: float
    jupiter_years: float
    sun_years: float
    earth_excess: float
    push_axis_deg: float
    thrust_axis_deg: float
    impact_angle_deg: float
    departure_closing: float
    periapsis_boost: float
    periapsis_speed: float
    periapsis_radius: float
    jupiter_arrival_excess: float
    mirror_excess: float
    disk_bend_deg: float
    ship_bend_deg: float

    @property
    def sun_closing(self) -> float:
        """Disk closing speed at perihelion: the mirror orbit's equal and opposite speed."""
        return 2.0 * self.periapsis_speed

    @property
    def arc_time_scale(self) -> float:
        """``r_p / v_p``: the hours of arc per radian of true anomaly at perihelion."""
        return self.periapsis_radius / self.periapsis_speed / 3600.0


def _route(
    closure: SynodicCycleClosure,
    dive_solar_radii: float,
    return_excess: float,
    p: _FlybyParams,
) -> DiveRoute:
    dive_radius = _dive_periapsis_radius(dive_solar_radii)
    evaluation = _evaluate(
        closure.departure_excess,
        closure.departure_aim_deg,
        p,
        dive_radius,
        return_excess,
        1.0,
    )
    assert evaluation is not None
    return _route_from(
        evaluation,
        closure.departure_excess,
        closure.departure_aim_deg,
        dive_solar_radii,
        return_excess,
        p,
    )


def _route_from(
    evaluation: object,
    departure_excess: float,
    aim_deg: float,
    dive_solar_radii: float,
    return_excess: float,
    p: _FlybyParams,
) -> DiveRoute:
    ev = evaluation
    dive_radius = _dive_periapsis_radius(dive_solar_radii)
    outbound, dive, climb = ev.outbound, ev.dive, ev.climb  # type: ignore[attr-defined]
    state = _solve_dive_state(dive.excess_out, 1.0, dive_radius, p)
    assert state is not None
    v_t, v_r = state
    incoming = np.array([outbound.v_tangential - p.v_jupiter_orbit, outbound.v_radial])
    mirror = np.array([-v_t - p.v_jupiter_orbit, v_r])
    w_in, w_mirror = float(np.linalg.norm(incoming)), float(np.linalg.norm(mirror))
    bend = math.degrees(
        math.acos(float(np.clip(incoming @ mirror / (w_in * w_mirror), -1.0, 1.0)))
    )
    periapsis_speed = math.sqrt(2.0 * dive.energy + 2.0 * p.mu_sun / dive_radius)
    closure_total = ev.total_tof  # type: ignore[attr-defined]
    # The departure burn's geometry, from the same ledger the closure is
    # priced by (its mirror sign chosen for the better delivery).
    earth_excess = _earth_relative_return(ev, p)
    push_axis = _push_axis_deg(ev, p)
    nozzle = departure_nozzle_ledger(
        departure_excess, aim_deg, earth_excess, push_axis, 30.0, 0.6, p
    )
    return DiveRoute(
        dive_solar_radii=dive_solar_radii,
        return_excess=return_excess,
        departure_excess=departure_excess,
        departure_aim_deg=aim_deg,
        cycle_years=closure_total / _SECONDS_PER_YEAR,
        jupiter_years=outbound.tof / _SECONDS_PER_YEAR,
        sun_years=(outbound.tof + dive.tof) / _SECONDS_PER_YEAR,
        earth_excess=earth_excess,
        push_axis_deg=push_axis,
        thrust_axis_deg=nozzle.thrust_axis_deg,
        impact_angle_deg=0.5
        * (nozzle.impact_angle_start_deg + nozzle.impact_angle_end_deg),
        departure_closing=0.5 * (nozzle.closing_speed_start + nozzle.closing_speed_end),
        periapsis_boost=climb_boost(ev),
        periapsis_speed=periapsis_speed,
        periapsis_radius=dive_radius,
        jupiter_arrival_excess=w_in,
        mirror_excess=w_mirror,
        disk_bend_deg=bend,
        ship_bend_deg=math.degrees(dive.bend_required),
    )


def climb_boost(evaluation: object) -> float:
    """The perihelion boost the climb leg takes (km/s)."""
    return float(evaluation.climb.periapsis_boost)  # type: ignore[attr-defined]


def _earth_relative_return(evaluation: object, p: _FlybyParams) -> float:
    climb = evaluation.climb  # type: ignore[attr-defined]
    return float(np.hypot(climb.v_tangential - p.v_earth_orbit, climb.v_radial))


def _push_axis_deg(evaluation: object, p: _FlybyParams) -> float:
    climb = evaluation.climb  # type: ignore[attr-defined]
    return math.degrees(
        math.atan2(climb.v_radial, climb.v_tangential - p.v_earth_orbit)
    )


@functools.lru_cache(maxsize=None)
def departure_route(
    dive_solar_radii: float, return_excess: float
) -> Optional[DiveRoute]:
    """The departure-wave ship: the plain three-synodic closure.

    Args:
        dive_solar_radii: Perihelion in solar radii.
        return_excess: Solar excess of the climb-out (km/s).

    Returns:
        The route, or None if it does not close or its flyby cannot turn.
    """
    p = _powered_flyby_params()
    closure = solve_synodic_closure(3, p, dive_solar_radii, return_excess, 1.0)
    if closure is None or not closure.feasible:
        return None
    return _route(closure, dive_solar_radii, return_excess, p)


@functools.lru_cache(maxsize=None)
def push_route(
    dive_solar_radii: float, return_excess: float, early_days: float = PARKING_DAYS
) -> Optional[DiveRoute]:
    """The push-wave ship: departs with the others, returns ``early_days`` early.

    Its sheets push the next window's launch units into the parking orbit, so
    they must arrive one parking period before the departure wave.  The
    closure solver lets Jupiter's phase float, so the push-wave ship is solved
    on three knobs -- departure excess, aim and its own climb-out excess --
    against three conditions: the shortened clock, the Earth intercept, and
    reaching Jupiter where the departure-wave ship's flyby put it.

    Args:
        dive_solar_radii: Perihelion in solar radii.
        return_excess: The departure-wave ship's climb-out excess (km/s).
        early_days: How much earlier its wave must arrive.

    Returns:
        The route, or None if no solution is found.
    """
    base = departure_route(dive_solar_radii, return_excess)
    if base is None:
        return None
    p = _powered_flyby_params()
    dive_radius = _dive_periapsis_radius(dive_solar_radii)
    target = 3.0 * synodic_period(p) - early_days * _DAY
    base_ev = _evaluate(
        base.departure_excess,
        base.departure_aim_deg,
        p,
        dive_radius,
        return_excess,
        1.0,
    )
    assert base_ev is not None
    n_j = p.v_jupiter_orbit / p.r_jupiter_orbit
    jupiter_start = base_ev.outbound.swept - n_j * base_ev.outbound.tof

    def residual(x: np.ndarray) -> List[float]:
        ev = _evaluate(float(x[0]), float(x[1]), p, dive_radius, float(x[2]), 1.0)
        if ev is None:
            return [1e3, 1e3, 1e3]
        return [
            (ev.total_tof - target) / _DAY,
            math.degrees(_intercept_miss(ev, p)),
            math.degrees(
                wrap_pi(ev.outbound.swept - jupiter_start - n_j * ev.outbound.tof)
            ),
        ]

    guess = np.array([base.departure_excess, base.departure_aim_deg, return_excess])
    solution, _, ok, _ = fsolve(residual, guess, full_output=True, xtol=1e-12)
    if ok != 1 or max(abs(v) for v in residual(solution)) > 1e-6:
        return None
    ev = _evaluate(
        float(solution[0]), float(solution[1]), p, dive_radius, float(solution[2]), 1.0
    )
    if ev is None or ev.dive.bend_margin < 0.0:
        return None
    return _route_from(
        ev,
        float(solution[0]),
        float(solution[1]),
        dive_solar_radii,
        float(solution[2]),
        p,
    )


# --------------------------------------------------------------------------
# Orbit mechanics at Earth and Jupiter
# --------------------------------------------------------------------------


def _orbit_a(days: float) -> float:
    return float((_MU_EARTH * (days * _DAY / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0))


def periapsis_speed(days: float, altitude: float) -> float:
    """Periapsis speed of the ``days``-period orbit at ``altitude`` (km/s)."""
    r = _R_EARTH + altitude
    return math.sqrt(_MU_EARTH * (2.0 / r - 1.0 / _orbit_a(days)))


def departure_burn(excess: float, days: float = PARKING_DAYS) -> float:
    """The burn from the parking orbit's 600 km periapsis to ``excess`` (km/s)."""
    r = _R_EARTH + DEPARTURE_ALTITUDE
    return math.sqrt(excess**2 + 2.0 * _MU_EARTH / r) - periapsis_speed(
        days, DEPARTURE_ALTITUDE
    )


def apoapsis_methalox(rotation_deg: float, days: float = PARKING_DAYS) -> float:
    """Methalox at apoapsis: periapsis 400 -> 600 km, and the orbit turned (km/s).

    The push leaves the periapsis velocity along the push wave's direction;
    the departure needs it along the thrust axis.  With periapsis placed
    perpendicular to both, a rotation of the orbit about the apse line at
    apoapsis turns one into the other for ``2 v_a sin(angle/2)`` -- the
    child's reversal (ADR 0009) is the 180 degree case.
    """
    a = _orbit_a(days)
    low, high = _R_EARTH + PUSH_ALTITUDE, _R_EARTH + DEPARTURE_ALTITUDE
    apo = 2.0 * a - low
    before = math.sqrt(_MU_EARTH * (2.0 / apo - 1.0 / a))
    after = math.sqrt(_MU_EARTH * (2.0 / apo - 2.0 / (apo + high)))
    return (after - before) + 2.0 * after * math.sin(math.radians(rotation_deg) / 2.0)


def wave_speed(excess: float, altitude: float) -> float:
    """A returning wave's speed at ``altitude`` (km/s)."""
    return speed_with_escape_energy(
        excess, math.sqrt(2.0 * _MU_EARTH / (_R_EARTH + altitude))
    )


def l1_transfer_speed() -> float:
    """Periapsis speed at 400 km of a transfer with apoapsis at L1 (km/s)."""
    r_p = _R_EARTH + PUSH_ALTITUDE
    return math.sqrt(2.0 * _MU_EARTH * (1.0 / r_p - 1.0 / (r_p + L1_DISTANCE)))


@dataclass(frozen=True)
class CarrierBurn:
    """The disks' perijove burn onto the mirror-image retrograde dive.

    Attributes:
        perijove_rj: Perijove flown (Jupiter radii).
        burn: The burn (km/s).
        bend_available_deg: Turn the powered flyby can make there.
        bend_required_deg: Turn it must make.
    """

    perijove_rj: float
    burn: float
    bend_available_deg: float
    bend_required_deg: float


def carrier_burn(
    route: DiveRoute, perijove_rj: float = CARRIER_PERIJOVE
) -> CarrierBurn:
    """Price the disks' split at Jupiter (G10).

    The mirror image of the ship's post-flyby state reaches perihelion at the
    same time and meets it on the symmetry line, but it needs the ship's own
    radial speed reflected: ``|(-v_t - V_J, v_r)|``, not the retrograde floor.

    Args:
        route: The ship's route.
        perijove_rj: The carrier's perijove.

    Returns:
        The burn and the turn check.
    """
    p = _powered_flyby_params()
    r = perijove_rj * _R_JUPITER
    v_esc = math.sqrt(2.0 * p.mu_jupiter / r)
    w_in, w_out = route.jupiter_arrival_excess, route.mirror_excess
    available = math.degrees(
        half_turn_angle(hyperbolic_eccentricity(p.mu_jupiter, r, w_in))
        + half_turn_angle(hyperbolic_eccentricity(p.mu_jupiter, r, w_out))
    )
    return CarrierBurn(
        perijove_rj,
        speed_with_escape_energy(w_out, v_esc) - speed_with_escape_energy(w_in, v_esc),
        available,
        route.disk_bend_deg,
    )


def ship_perijove_rj(route: DiveRoute) -> float:
    """The perijove the ship's unpowered turn pins (Jupiter radii)."""
    p = _powered_flyby_params()

    def gap(rj: float) -> float:
        e = hyperbolic_eccentricity(
            p.mu_jupiter, rj * _R_JUPITER, route.jupiter_arrival_excess
        )
        return math.degrees(unpowered_bend_angle(e)) - route.ship_bend_deg

    return float(brentq(gap, 1.0, 200.0))


def carrier_perijove(route: DiveRoute, highest: float = CARRIER_PERIJOVE) -> float:
    """The highest perijove up to ``highest`` at which the disks' flyby turns."""
    p = _powered_flyby_params()

    def margin(rj: float) -> float:
        burn = carrier_burn(route, rj)
        return burn.bend_available_deg - burn.bend_required_deg

    if margin(highest) >= 0.0:
        return highest
    floor = p.periapsis_floor / _R_JUPITER
    if margin(floor) < 0.0:
        return floor
    return float(brentq(margin, floor, highest, xtol=1e-6))


def departure_loss(impulsive: float, burn_seconds: float) -> float:
    """Fixed-direction finite-burn loss from the 10-day orbit at 600 km (km/s)."""
    from astropy import units as u

    from src.finite_burn_loss import TABLE_TIMES, fixed_direction_loss

    seconds = min(burn_seconds, float(TABLE_TIMES[-1]))
    if burn_seconds > TABLE_TIMES[-1]:
        raise ValueError("departure burn longer than the loss table")
    loss = fixed_direction_loss(
        impulsive * u.km / u.s, seconds * u.s, PARKING_DAYS * u.day
    )
    return float(loss.to_value(u.km / u.s))


def arc_loss(arc_hours: float, route: DiveRoute) -> float:
    """Share of the Sun boost a finite arc centred on perihelion loses.

    Two terms, both in ``(T / tau)^2`` with ``tau = r_p / v_p``:

    * **Oberth:** the ship is slower off perihelion, ``v ~ v_p (1 - nu^2/8)``,
      so the energy each pulse adds falls: ``T^2 / (96 tau^2)``.
    * **Off head-on:** the mirror-image disks meet the ship head-on only on
      the symmetry line; elsewhere they come in about ``nu`` off its velocity,
      and the thrust's useful share is ``cos nu``: ``T^2 / (24 tau^2)``.
    """
    ratio = arc_hours / route.arc_time_scale
    return ratio * ratio * (1.0 / 96.0 + 1.0 / 24.0)


# --------------------------------------------------------------------------
# One launch unit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """The knobs the optimizer turns, and the stated assumptions.

    Attributes:
        k_push: Polyethylene per kg of sheet in the 400 km overtake push.
        k_earth: The same at the canted departure.
        k_sun: Polyethylene per kg of disk at the Sun.
        arc_hours: The Sun boost's arc, centred on perihelion.
        k_delivery: The L1 delivery push's slug ratio.
        departure_pulse: Impulse per departure pulse (N s): a larger pulse
            shortens the burn and its loss but needs a larger plate.
        film_exponent: The Sun film's scaling with energy per unit impulse.
        g_cap: Peak g the ship takes.
        carrier_perijove: The disks' highest perijove (Jupiter radii); lower
            if their turn needs it.
        bonds: ``carbon atomic`` (headline) or ``bonds returned``.
        shield_exponent: The shade's mass scales as sunlight at perihelion to
            this power, from Parker's 9.86 solar radii (0 headline: unsolved
            below Parker, the parent's E-items).
    """

    k_push: float = 10.0
    k_earth: float = 24.0
    k_sun: float = 30.0
    arc_hours: float = 4.0
    k_delivery: float = 10.0
    departure_pulse: float = PULSE_IMPULSE
    film_exponent: float = 1.0
    g_cap: float = G_CAP
    carrier_perijove: float = CARRIER_PERIJOVE
    bonds: str = "carbon atomic"
    shield_exponent: float = 0.0


@dataclass(frozen=True)
class UnitLedger:
    """One launch unit flown by one ship type, kilograms.

    Attributes:
        push_sheets: Push-wave sheets its 400 km push spends.
        departure_sheets: Departure-wave sheets its departure spends.
        returned: Sheets its Sun ship releases as the next wave.
        disks: Sun-wave disks it carries.
        lines: Every mass line, for the report and the cost book.
    """

    push_sheets: float
    departure_sheets: float
    returned: float
    disks: float
    lines: Dict[str, float]


def _plate_for(pulse: float, rated: float, area: float, speed_ratio: float) -> float:
    """Survival floor: peak pressure held as the pulse's energy scales the plate."""
    return float(
        SURVIVAL_AREAL * area * (pulse / rated * speed_ratio**2) ** (2.0 / 3.0)
    )


def _inertia_plate(pulse_impulse: float, ship: float, g_cap: float) -> float:
    """Plate mass that holds the ship under ``g_cap`` over the absorber's stroke."""
    stroke_m = ABSORBER_STROKE * 1e3
    return pulse_impulse**2 / (2.0 * stroke_m * ship * g_cap * 9.80665)


def fly_unit(
    route: DiveRoute,
    push_wave: DiveRoute,
    design: Design,
    laws: Optional[StageLaws] = None,
) -> UnitLedger:
    """Carry one launch unit from the lob to its returned sheets.

    Args:
        route: This ship's own route.
        push_wave: The push-wave route (sets the push's sheet speed).
        design: The knobs.
        laws: The stage laws; the vendored table by default.

    Returns:
        The ledger.
    """
    law = laws or StageLaws.load(bonds=design.bonds)
    lines: Dict[str, float] = {}
    # 1. The overtake push at 400 km, from rest to the parking orbit.
    w_push = wave_speed(push_wave.earth_excess, PUSH_ALTITUDE)
    ve_push = law.push_ve(design.k_push, w_push)
    push_dv = periapsis_speed(PARKING_DAYS, PUSH_ALTITUDE)
    mr_push = math.exp(push_dv / ve_push)
    pushed = LAUNCH_UNIT / mr_push
    push_charge = LAUNCH_UNIT - pushed
    push_sheets = push_charge / design.k_push
    push_impulse = push_charge * ve_push * 1e3
    # 3. Apoapsis: raise periapsis and turn the orbit (methalox).
    rotation = abs(
        math.degrees(
            wrap_pi(math.radians(push_wave.push_axis_deg - route.thrust_axis_deg))
        )
    )
    methalox_dv = apoapsis_methalox(rotation)
    stack = pushed * math.exp(-methalox_dv / VE_METHALOX)
    lines["methalox"] = pushed - stack
    # 4. The canted departure; the plate rides through it, then diverts.  The
    # burn lasts as long as its pulses take at the Earth rate, and that finite
    # burn, its thrust held in one direction (the stream's angle is the
    # plate's law), loses Oberth at the 11-minute periapsis.
    ve_earth = law.earth_ve(
        design.k_earth, route.departure_closing, route.impact_angle_deg
    )
    impulsive = departure_burn(route.departure_excess)
    dv = impulsive
    for _ in range(_LOSS_ITERATIONS):
        mr_earth = math.exp(dv / ve_earth)
        departure_charge = stack * (1.0 - 1.0 / mr_earth)
        departure_impulse = departure_charge * ve_earth * 1e3
        burn_seconds = departure_impulse / (design.departure_pulse * EARTH_PULSE_RATE)
        dv = impulsive + departure_loss(impulsive, burn_seconds)
    after = stack / mr_earth
    departure_sheets = departure_charge / design.k_earth
    lines["departure_loss"] = dv - impulsive
    lines["departure_minutes"] = burn_seconds / 60.0
    # 2. The flat plate, sized for the larger of its two pulses (the push's and
    # the departure's sheet per pulse, by energy) and checked for the g cap.
    push_pulse_sheet = PULSE_IMPULSE / (design.k_push * ve_push * 1e3)
    departure_pulse_sheet = design.departure_pulse / (design.k_earth * ve_earth * 1e3)
    flat = max(
        _plate_for(push_pulse_sheet, FLAT_PLATE_PULSE, FLAT_PLATE_AREA, w_push / 100.0),
        _plate_for(
            departure_pulse_sheet,
            FLAT_PLATE_PULSE,
            FLAT_PLATE_AREA,
            route.departure_closing / 100.0,
        ),
        _inertia_plate(PULSE_IMPULSE, pushed, design.g_cap),
        _inertia_plate(design.departure_pulse, after, design.g_cap),
    )
    lines["flat_plate"] = flat
    film_earth = FILM_PER_IMPULSE * (push_impulse + departure_impulse)
    divert = flat * (math.exp(PLATE_DIVERT / (STORABLE_ISP * _G0)) - 1.0)
    mounts_earth = CHARGE_MOUNT_FRACTION * (push_charge + departure_charge)
    lines.update(
        push_charge=push_charge,
        departure_charge=departure_charge,
        film_earth=film_earth,
        plate_divert=divert,
        earth_mounts=mounts_earth,
    )
    departing = after - flat - divert - film_earth - mounts_earth
    # 5. Jupiter: the carrier takes the disks onto the mirror dive.
    burn = carrier_burn(route, carrier_perijove(route, design.carrier_perijove))
    if burn.bend_available_deg < burn.bend_required_deg - 1e-9:
        raise ValueError("the disks' flyby cannot turn even at Jupiter's surface")
    carrier_per_disk = (1.0 + CARRIER_DRY_FRACTION) * math.exp(
        burn.burn / (STORABLE_ISP * _G0)
    )
    # 6. The Sun: boost on the arc, disks per kg of ship, paraboloid, film.
    w_sun = route.sun_closing
    ve_sun = law.sun_ve(design.k_sun, w_sun)
    boost = route.periapsis_boost / (1.0 - arc_loss(design.arc_hours, route))
    mr_sun = math.exp(boost / ve_sun)
    disks_per_ship = (1.0 - 1.0 / mr_sun) / design.k_sun
    film_rate = FILM_PER_IMPULSE * (ve_sun / ve_push) ** design.film_exponent
    film_per_ship = film_rate * (1.0 - 1.0 / mr_sun) * ve_sun * 1e3
    ship = departing / (1.0 + carrier_per_disk * disks_per_ship + film_per_ship)
    disks = disks_per_ship * ship
    sun_charge = ship * (1.0 - 1.0 / mr_sun)
    final = ship / mr_sun
    pulses = SUN_PULSE_RATE * design.arc_hours * 3600.0
    pulse_disk = disks / pulses
    sun_plate = max(
        _plate_for(
            pulse_disk, SUN_PLATE_PULSE, SUN_PLATE_AREA, w_sun / SUN_PLATE_SPEED
        ),
        _inertia_plate(pulse_disk * design.k_sun * ve_sun * 1e3, final, design.g_cap),
    )
    dead = {
        "sun_plate": sun_plate,
        "bore_collar": BORE_COLLAR,
        "shield": SHIELD_FRACTION
        * ship
        * (PARKER_SOLAR_RADII / route.dive_solar_radii)
        ** (2.0 * design.shield_exponent),
        "avionics": AVIONICS_FRACTION * ship,
        "sun_mounts": CHARGE_MOUNT_FRACTION * sun_charge,
    }
    returned = final - sum(dead.values())
    lines.update(dead)
    lines.update(
        disks=disks,
        carrier_dry=CARRIER_DRY_FRACTION * disks,
        carrier_propellant=disks * (carrier_per_disk - 1.0 - CARRIER_DRY_FRACTION),
        sun_charge=sun_charge,
        film_sun=film_per_ship * ship,
        sheets=max(returned, 0.0),
        departing=departing,
        ship_at_sun=ship,
        mr_push=mr_push,
        mr_earth=mr_earth,
        mr_sun=mr_sun,
        ve_push=ve_push,
        ve_earth=ve_earth,
        ve_sun=ve_sun,
        methalox_dv=methalox_dv,
        departure_dv=dv,
        carrier_burn=burn.burn,
    )
    return UnitLedger(push_sheets, departure_sheets, max(returned, 0.0), disks, lines)


@dataclass(frozen=True)
class CycleGrowth:
    """A cycle's growth, split by role.

    Attributes:
        growth: Launch units out over launch units in, per three-synodic cycle.
        push_share: Share of departing units flown as push-wave ships.
        push_unit: A push-wave ship's ledger.
        departure_unit: A departure-wave ship's ledger.
        per_total_impactor: The simulation's measure, disks in the divisor.
    """

    growth: float
    push_share: float
    push_unit: UnitLedger
    departure_unit: UnitLedger
    per_total_impactor: float

    @property
    def sheets_per_unit(self) -> float:
        """Sheets one mean unit spends (push and departure)."""
        s = self.push_share
        return self.push_unit.push_sheets + (
            s * self.push_unit.departure_sheets
            + (1.0 - s) * self.departure_unit.departure_sheets
        )


def cycle_growth(
    departure: DiveRoute, push: DiveRoute, design: Design, arrives: float = 1.0
) -> CycleGrowth:
    """Balance the two waves (ADR 0069's role split).

    Args:
        departure: The departure-wave route.
        push: The push-wave route.
        design: The knobs.
        arrives: Share of each wave's sheets that arrive and are caught.

    Returns:
        The growth.
    """
    return balance(
        fly_unit(push, push, design), fly_unit(departure, push, design), arrives
    )


def balance(p_unit: UnitLedger, d_unit: UnitLedger, arrives: float) -> CycleGrowth:
    """The role split for two flown unit types (see :func:`cycle_growth`)."""
    g = role_growth(
        p_unit.push_sheets,
        p_unit.returned,
        d_unit.returned,
        p_unit.departure_sheets,
        d_unit.departure_sheets,
        arrives,
        arrives,
    )
    share = g * p_unit.push_sheets / (p_unit.returned * arrives) if g > 0 else 0.0
    spent = (
        p_unit.push_sheets
        + share * p_unit.departure_sheets
        + (1 - share) * d_unit.departure_sheets
    )
    disks = share * p_unit.disks + (1 - share) * d_unit.disks
    per_total = g * spent / (spent + disks)
    return CycleGrowth(g, share, p_unit, d_unit, per_total)


def doubling_years(growth: float, cycle_years: float) -> float:
    """Doubling time of a cycle that multiplies by ``growth``."""
    return cycle_years * math.log(2.0) / math.log(growth) if growth > 1.0 else math.inf


# --------------------------------------------------------------------------
# The cost book
# --------------------------------------------------------------------------

#: The child's spray cup is $30M falling to $6M on an 80% curve per 156 t
#: plate and absorber (growth_cost.ESTIMATE); here it is per kilogram so a
#: scaled plate is priced by its mass.
_SPRAY_CUP_FIRST = 30.0e6 / 156.0e3
_SPRAY_CUP_FLOOR = 6.0e6 / 156.0e3


@dataclass(frozen=True)
class SolarDivePrices:
    """Every price the solar-dive cycle is charged, dollars (assumptions marked).

    Attributes:
        lob: Per kilogram lofted (growth_cost.ESTIMATE).
        flat_plate_share: The flat plate's price on the spray cup's per-kg
            curve (author, 2026-10-16: half).
        sun_plate_share: The paraboloid's (author: the full curve).
        film: Per kilogram of film (growth_cost.ESTIMATE).
        methalox: Per kilogram (growth_cost.ESTIMATE).
        polyethylene: Per kilogram of charge (assumption: bulk resin, below the
            child's $4/kg rod with tethers).
        disk: Per kilogram of Sun-wave disk (author: $1000/kg baseline).
        carrier: Per kilogram of carrier dry mass (assumption: as a disk).
        storable: Per kilogram of the carrier's propellant (assumption).
        shield: Per kilogram of shade (assumption).
        avionics: Per kilogram (assumption).
        sheet_body: Per kilogram of sheet body (the child's PuffSat body).
        mounts: Per kilogram of charge mount (assumption).
        lob_climb: The child's lob arriving at 400 km still climbing at 1.1
            km/s (lob_rise.booster_growth, ADR 0042-0043), for parity.
        package: The child's steering package curve, per package.
        packages_per_kg: Packages per kilogram of sheet: the child's two per
            60 kg PuffSat, read as a 5% guidance mass (scaled by the bracket).
    """

    lob: float = 25.0
    flat_plate_share: float = 0.5
    sun_plate_share: float = 1.0
    film: float = 10.0
    methalox: float = 0.2
    polyethylene: float = 2.0
    disk: float = 1000.0
    carrier: float = 1000.0
    storable: float = 10.0
    shield: float = 1000.0
    avionics: float = 1000.0
    sheet_body: float = 3.0
    mounts: float = 50.0
    lob_climb: float = 1.0648
    package: LearningCurve = LearningCurve(100.0, 10.0, 0.8, anchor=1.0e5)
    packages_per_kg: float = 2.0 / 60.0

    def plate_curve(self, share: float) -> LearningCurve:
        """The spray cup's per-kilogram curve at ``share`` of its price."""
        return LearningCurve(share * _SPRAY_CUP_FIRST, share * _SPRAY_CUP_FLOOR, 0.8)


ESTIMATE = SolarDivePrices()


@dataclass(frozen=True)
class Gate:
    """One milestone the program's odds step at.

    Attributes:
        label: What is proved.
        years: When, from the seed's departure.
        odds: Chance it works, given every earlier gate did.
    """

    label: str
    years: float
    odds: float


@dataclass(frozen=True)
class GatedSchedule:
    """Risk-adjusted NPV: time at ``rate``, each flow weighted by the gates before it.

    The drug-development convention (rNPV): technical risk is a probability
    that steps up at each milestone, not a higher discount rate.

    Attributes:
        rate: Annual time value of money.
        gates: The milestones.
    """

    rate: float
    gates: Tuple[Gate, ...]

    def factor(self, years: float) -> float:
        """Present value of a dollar paid at ``years``."""
        odds = 1.0
        for gate in self.gates:
            if years >= gate.years:
                odds *= gate.odds
        return float(odds * (1.0 + self.rate) ** (-years))

    @property
    def cumulative(self) -> float:
        """The program's overall odds."""
        return float(np.prod([g.odds for g in self.gates]))


def solar_dive_gates(route: DiveRoute, push: DiveRoute) -> Tuple[Gate, ...]:
    """The cycle's milestones, at the proposed odds (2026-10-16).

    The Sun boost carries most of the risk: it is the one gate with no flight
    analog.  Flows at the first push-wave arrival (the first growth lob) are
    weighted by the first three; everything after the first departure by all.
    """
    return (
        Gate(
            "seed departs, carrier splits the disks at Jupiter",
            route.jupiter_years,
            0.95,
        ),
        Gate("first Sun boost", push.sun_years, 0.75),
        # Settled as the first growth lob is bought, at the push wave's arrival.
        Gate("sheets unfurl and arrive face-on", push.cycle_years - 1e-6, 0.85),
        Gate("first push and canted departure", route.cycle_years, 0.90),
    )


#: Sheets missed or failed to unfurl: 20% on the first return, falling on an
#: 80% learning curve in cumulative sheets returned, to a 3% floor.
MISS_FIRST = 0.20
MISS_FLOOR = 0.03
MISS_LEARNING = 0.8


def miss_fraction(cumulative_returned: float, first_return: float) -> float:
    """Share of a return's sheets lost, by experience."""
    doublings = math.log2(max(cumulative_returned / first_return, 1.0))
    return float(max(MISS_FLOOR, MISS_FIRST * MISS_LEARNING**doublings))


#: Growth phase: 15 synodic periods (author, 2026-10-16), five 3S cycles,
#: before any mass is delivered.  Held level after, to the child's horizon.
GROWTH_SYNODIC = 15
HORIZON_YEARS = 300.0
GUIDANCE_FRACTIONS = (0.02, 0.05, 0.10)


@dataclass(frozen=True)
class Valuation:
    """A design valued on the gated schedule.

    Attributes:
        growth: The steady cycle's growth (miss at its floor).
        doubling: Its doubling time (years).
        steady_cost: Undiscounted $/kg at L1 in the steady state.
        steady_lines: The same by line.
        break_even: The L1 price that repays the seed and growth.
        seed_per_fleet_kg: Seed dollars per kilogram of sheets at the harvest.
        fleet_multiple: Sheets at the harvest per seed sheet.
    """

    growth: float
    doubling: float
    steady_cost: float
    steady_lines: Dict[str, float]
    break_even: Optional[float]
    seed_per_fleet_kg: float
    fleet_multiple: float


def _unit_dollars(
    unit: UnitLedger,
    prices: SolarDivePrices,
    plates: Dict[str, float],
    packages: List[float],
    guidance: float,
    lob: float,
) -> Dict[str, float]:
    """Dollars one launch unit of this type costs, by line."""
    m = unit.lines
    flat_curve = prices.plate_curve(prices.flat_plate_share)
    sun_curve = prices.plate_curve(prices.sun_plate_share)
    count = m["sheets"] * prices.packages_per_kg * guidance / 0.05
    package = prices.package.batch_cost(packages[0], count)
    packages[0] += count
    lines = {
        "lob": lob * LAUNCH_UNIT,
        "flat_plate": (
            flat_curve.batch_cost(plates["flat"], 1.0) * m["flat_plate"] if lob else 0.0
        ),
        "sun_plate": sun_curve.batch_cost(plates["sun"], 1.0)
        * (m["sun_plate"] + m["bore_collar"]),
        "film": prices.film * (m["film_earth"] + m["film_sun"]),
        "methalox": prices.methalox * m["methalox"],
        "polyethylene": prices.polyethylene
        * (
            m["sun_charge"] + (m["push_charge"] + m["departure_charge"] if lob else 0.0)
        ),
        "mounts": prices.mounts
        * (m["sun_mounts"] + (m["earth_mounts"] if lob else 0.0)),
        "disks": prices.disk * m["disks"],
        "carrier": prices.carrier * m["carrier_dry"]
        + prices.storable * m["carrier_propellant"],
        "shield": prices.shield * m["shield"],
        "avionics": prices.avionics * m["avionics"],
        "sheets": prices.sheet_body * m["sheets"] + package,
    }
    plates["flat"] += 1.0
    plates["sun"] += 1.0
    return lines


@dataclass(frozen=True)
class DeliveryUnit:
    """One 1500 t lob pushed to L1 by returning sheets.

    Attributes:
        sheets: Sheets the push spends.
        cargo: Mass left at L1.
        charge: Polyethylene burned.
    """

    sheets: float
    cargo: float
    charge: float


def delivery_unit(route: DiveRoute, k: float, laws: StageLaws) -> DeliveryUnit:
    """The harvest push: the child's plate and absorber for fragile cargo."""
    w = wave_speed(route.earth_excess, PUSH_ALTITUDE)
    ve = laws.push_ve(k, w)
    mr = math.exp(l1_transfer_speed() / ve)
    pushed = LAUNCH_UNIT / mr
    halo = math.exp(-HALO_INSERTION / VE_METHALOX)
    charge = LAUNCH_UNIT - pushed
    return DeliveryUnit(charge / k, (pushed - DELIVERY_PLATE) * halo, charge)


def value_design(
    departure: DiveRoute,
    push: DiveRoute,
    design: Design,
    prices: SolarDivePrices = ESTIMATE,
    seed_price: float = 1367.0,
    guidance: float = 0.05,
    rate: float = 0.10,
    gates: Optional[Tuple[Gate, ...]] = None,
    growth_synodic: int = GROWTH_SYNODIC,
) -> Valuation:
    """Grow from a seed for ``growth_synodic`` synodic periods, then hold level.

    The seed is one launch unit's worth of departing ships, both types in the
    steady split, flown from LEO on an expended, refuelled Starship at the
    route's departure excess (:func:`src.seed_cost.seed_price_range`); its
    contents are built new.  Each return's sheets push and depart new launch
    units; from the harvest on, the surplus pushes cargo to L1.

    The delivery schedule does not depend on the sale price, so every flow is
    ``price x cargo - cost`` and the break-even is the ratio of the two
    present values: no root-finding.

    Args:
        departure: The departure-wave route.
        push: The push-wave route.
        design: The knobs.
        prices: The book.
        seed_price: Dollars per seed kilogram sent toward Jupiter.
        guidance: Sheet guidance mass share (2/5/10% bracket).
        rate: Annual time value of money.
        gates: Milestones; the cycle's own by default.
        growth_synodic: Synodic periods of growth before the first delivery.

    Returns:
        The valuation.
    """
    laws = StageLaws.load(bonds=design.bonds)
    schedule = GatedSchedule(
        rate, solar_dive_gates(departure, push) if gates is None else gates
    )
    p_unit = fly_unit(push, push, design, laws)
    d_unit = fly_unit(departure, push, design, laws)
    steady = balance(p_unit, d_unit, 1.0 - MISS_FLOOR)
    cycle_years = departure.cycle_years
    harvest = int(round(growth_synodic / 3))
    delivery = delivery_unit(departure, design.k_delivery, laws)
    lob = prices.lob * prices.lob_climb

    plates = {"flat": 0.0, "sun": 0.0}
    packages = [0.0]
    s = steady.push_share
    seed_mass = s * p_unit.lines["departing"] + (1 - s) * d_unit.lines["departing"]
    seed_contents = 0.0
    for unit, share in ((p_unit, s), (d_unit, 1 - s)):
        lines = _unit_dollars(unit, prices, plates, packages, guidance, lob=0.0)
        seed_contents += share * sum(lines.values())
    seed_outlay = seed_price * seed_mass + seed_contents
    pv_cost = seed_outlay
    pv_cargo = 0.0
    per_unit = s * p_unit.returned + (1 - s) * d_unit.returned
    if per_unit <= 0.0:
        raise ValueError("the launch unit returns no sheets")
    units, cumulative, fleet = 1.0, 0.0, 0.0
    tally: Dict[str, float] = {}
    tally_cargo = 0.0
    n = 1
    while True:
        years = n * cycle_years - PARKING_DAYS * _DAY / _SECONDS_PER_YEAR
        if years > HORIZON_YEARS:
            break
        returned = units * per_unit
        cumulative += returned
        arrives = 1.0 - miss_fraction(cumulative, per_unit)
        growth = balance(p_unit, d_unit, arrives)
        caught = returned * arrives
        harvesting = n >= harvest
        if n == harvest:
            fleet = caught
        new_units = units * (min(growth.growth, 1.0) if harvesting else growth.growth)
        surplus = (
            max(caught - new_units * growth.sheets_per_unit, 0.0) if harvesting else 0.0
        )
        cost: Dict[str, float] = {}
        for unit, share in (
            (p_unit, growth.push_share),
            (d_unit, 1 - growth.push_share),
        ):
            for key, value in _unit_dollars(
                unit, prices, plates, packages, guidance, lob
            ).items():
                cost[key] = cost.get(key, 0.0) + share * value * new_units
        deliveries = surplus / delivery.sheets
        cost["delivery_lob"] = lob * LAUNCH_UNIT * deliveries
        cost["delivery_plate"] = (
            prices.plate_curve(prices.flat_plate_share).batch_cost(
                plates["flat"], deliveries
            )
            * DELIVERY_PLATE
        )
        cost["delivery_charge"] = prices.polyethylene * delivery.charge * deliveries
        plates["flat"] += deliveries
        cargo = deliveries * delivery.cargo
        factor = schedule.factor(years)
        pv_cost += factor * sum(cost.values())
        pv_cargo += factor * cargo
        if n == harvest + 6:
            tally, tally_cargo = cost, cargo
        units = new_units
        n += 1
    lines = (
        {key: value / tally_cargo for key, value in tally.items()}
        if tally_cargo
        else {}
    )
    return Valuation(
        growth=steady.growth,
        doubling=doubling_years(steady.growth, cycle_years),
        steady_cost=float(sum(lines.values())),
        steady_lines=lines,
        break_even=pv_cost / pv_cargo if pv_cargo > 0 else None,
        seed_per_fleet_kg=seed_outlay / fleet if fleet > 0 else math.inf,
        fleet_multiple=fleet / per_unit if per_unit > 0 else 0.0,
    )


# --------------------------------------------------------------------------
# The optimizer and the depth sweep
# --------------------------------------------------------------------------


def seed_price(route: DiveRoute, dear: bool = False) -> float:
    """Seed dollars per kilogram at the route's departure excess (ADR 0057's ends)."""
    from astropy import units as u

    from src.seed_cost import seed_price_range

    prices = seed_price_range(route.departure_excess * u.km / u.s)
    return prices.high if dear else prices.low


_KNOBS = ("k_push", "k_earth", "k_sun", "arc_hours", "k_delivery", "departure_pulse")
_BOUNDS = {
    "k_push": K_PUSH,
    "k_earth": K_EARTH,
    "k_sun": K_SUN,
    "arc_hours": ARC_HOURS,
    "k_delivery": K_PUSH,
    "departure_pulse": DEPARTURE_PULSE,
}


@dataclass(frozen=True)
class Optimum:
    """The cheapest design at one closure.

    Attributes:
        departure: The departure-wave route.
        push: The push-wave route.
        design: The knobs at the optimum.
        valuation: Its valuation on the cheap seed.
        dear_break_even: Break-even on the dear seed (the objective by default).
    """

    departure: DiveRoute
    push: DiveRoute
    design: Design
    valuation: Valuation
    dear_break_even: Optional[float]


def optimize_design(
    departure: DiveRoute,
    push: DiveRoute,
    start: Design = Design(),
    objective: str = "break_even",
    prices: SolarDivePrices = ESTIMATE,
    guidance: float = 0.05,
    dear: bool = True,
    rate: float = 0.10,
    gates: Optional[Tuple[Gate, ...]] = None,
    growth_synodic: int = GROWTH_SYNODIC,
) -> Optimum:
    """Choose the slug ratios and the Sun arc for the lowest cost per kilogram.

    The objective is the break-even L1 price on the gated schedule, on the
    dear seed by default (author, 2026-10-16: the more realistic end)
    (``break_even``), or the steady $/kg (``steady``) or the doubling time
    (``doubling``) for comparison.  Nelder-Mead in log space inside the
    recorded bounds (``_BOUNDS``), which are the simulated ranges.
    """
    cheap = seed_price(departure, dear)
    lows = np.log([_BOUNDS[k][0] for k in _KNOBS])
    highs = np.log([_BOUNDS[k][1] for k in _KNOBS])

    def design_at(x: np.ndarray) -> Design:
        clipped = np.exp(np.clip(x, lows, highs))
        return replace(start, **dict(zip(_KNOBS, (float(v) for v in clipped))))  # type: ignore[arg-type]

    def score(x: np.ndarray) -> float:
        penalty = float(np.sum(np.maximum(x - highs, 0.0) + np.maximum(lows - x, 0.0)))
        try:
            d = design_at(x)
            if objective == "doubling":
                g = cycle_growth(departure, push, d, 1.0 - MISS_FLOOR)
                return doubling_years(g.growth, departure.cycle_years) + 100.0 * penalty
            v = value_design(
                departure, push, d, prices, cheap, guidance, rate, gates, growth_synodic
            )
        except ValueError:
            return 1e6
        value = v.steady_cost if objective == "steady" else (v.break_even or 1e5)
        return value * (1.0 + penalty)

    x0 = np.log([getattr(start, k) for k in _KNOBS])
    result = minimize(
        score,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 600},
    )
    best = design_at(result.x)
    schedule = (rate, gates, growth_synodic)
    valuation = value_design(
        departure, push, best, prices, seed_price(departure), guidance, *schedule
    )
    other = value_design(
        departure, push, best, prices, seed_price(departure, True), guidance, *schedule
    )
    return Optimum(departure, push, best, valuation, other.break_even)


def depth_optimum(
    dive_solar_radii: float, excesses: Sequence[float] = RETURN_EXCESSES
) -> Optional[Optimum]:
    """The cheapest closure at one perihelion, over the recorded return excesses."""
    best: Optional[Optimum] = None
    start = Design()
    for excess in excesses:
        departure = departure_route(dive_solar_radii, excess)
        push = push_route(dive_solar_radii, excess) if departure else None
        if departure is None or push is None:
            continue
        try:
            law = StageLaws.load()
            law.sun_ve(30.0, departure.sun_closing)
        except ValueError:
            continue
        found = optimize_design(departure, push, start)
        start = found.design
        be = found.valuation.break_even
        if be is not None and (
            best is None or be < (best.valuation.break_even or math.inf)
        ):
            best = found
    return best


# --------------------------------------------------------------------------
# The child's rows on the same footing (D7)
# --------------------------------------------------------------------------

#: The child's designs D7 compares against (survivable_chamber.cost_designs()).
CHILD_ROWS = (0, 2, 6, 10)


def child_gates(proof_years: float, seed_return_years: float) -> Tuple[Gate, ...]:
    """The child's milestones at the same odds as the solar dive's.

    Its novel physics (the chamber) is proved at the first growth departure,
    after the first growth lob is bought; the solar dive's (the Sun boost)
    before.  The seed's own return is guided as the Earth wave is.
    """
    del proof_years
    return (
        Gate("seed departs and flies by Jupiter", 1.3, 0.95),
        Gate("returning batch arrives guided", seed_return_years - 1e-6, 0.85),
        Gate("first plate push", seed_return_years, 0.90),
        Gate(
            "first chamber departure", seed_return_years + PARKING_DAYS / 365.25, 0.75
        ),
    )


def child_break_even(
    inputs: DesignInputs,
    prices: PriceBook,
    seed_price_per_kg: float,
    growth_synodic: int = GROWTH_SYNODIC,
    rate: float = 0.10,
    gated: bool = True,
) -> Optional[float]:
    """A child design's break-even, harvesting at ``growth_synodic`` (15 S).

    ``gated`` False drops the odds, for a flat venture rate that carries the
    risk itself (ADR 0037's 30%).
    """
    from src.growth_cost import run_program

    p = _powered_flyby_params()
    horizon = growth_synodic * synodic_period(p) / _SECONDS_PER_YEAR
    times = inputs.times
    harvest = max(i for i, t in enumerate(times) if t <= horizon + 1e-6)
    staged = replace(inputs, harvest=harvest)
    gates = child_gates(staged.proof_years, times[0]) if gated else ()
    schedule = GatedSchedule(rate, gates)

    def worth(price: float) -> float:
        program = run_program(staged, prices, seed_price_per_kg, price, True)
        return sum(a * schedule.factor(t) for t, a in program.flows)

    if worth(5000.0) < 0.0:
        return None
    return float(brentq(worth, 0.0, 5000.0, xtol=1e-3))


# --------------------------------------------------------------------------
# The report (make solar-dive)
# --------------------------------------------------------------------------

#: The two headline closures: Parker's perihelion (the shade has flown there)
#: at its cheapest return excess in the sweep, and the 4 R_sun upgrade, whose
#: shade is unsolved (the parent's E-items).
HEADLINES = ((9.86, 60.0), (4.0, 90.0))
#: Where the optimizer starts each headline: the depth sweep's cheap-seed
#: optima (recorded so the headline reproduces; the sweep itself is
#: ``--sweep``, its bounds ``_BOUNDS`` and ``RETURN_EXCESSES``).
HEADLINE_STARTS = {
    (9.86, 60.0): Design(
        k_push=6.74,
        k_earth=5.21,
        k_sun=20.0,
        arc_hours=4.86,
        k_delivery=5.10,
        departure_pulse=3.71e6,
    ),
    (4.0, 90.0): Design(
        k_push=6.83,
        k_earth=5.0,
        k_sun=29.9,
        arc_hours=2.37,
        k_delivery=5.0,
        departure_pulse=3.66e6,
    ),
}
SWEEP_RADII = (4.0, 6.0, 8.0, 9.86, 12.0, 15.0, 20.0)
_SWEEP_START = Design(
    k_push=8.0,
    k_earth=8.0,
    k_sun=25.0,
    arc_hours=3.0,
    k_delivery=6.0,
    departure_pulse=6.0e6,
)


def _money(value: Optional[float]) -> str:
    return ">5000" if value is None else f"{value:.0f}"


def corrections_report() -> str:
    """What the simulation copied from this repository, checked (G1, G4, G10)."""
    rows = []
    for radii, excess, sim_burn, sim_floor in (
        (9.86, 90.0, 4.15, 14.78),
        (4.0, 150.0, 4.23, 14.16),
    ):
        route = departure_route(radii, excess)
        assert route is not None
        burn = carrier_burn(route, carrier_perijove(route, 10.0))
        rows.append(
            [
                f"{radii:g} R_sun, X {excess:g}",
                f"{route.departure_excess:.3f}",
                f"{sim_burn:.2f} -> {departure_burn(route.departure_excess):.3f}",
                f"{route.earth_excess:.2f}",
                f"{route.periapsis_boost:.2f}",
                f"{sim_floor:.2f} -> {route.mirror_excess:.3f}",
                f"{ship_perijove_rj(route):.2f}",
                f"{burn.perijove_rj:.2f}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "closure",
            "departure excess",
            "burn: sim -> 10-d orbit at 600 km",
            "Earth wave excess",
            "Sun boost",
            "disks' excess: floor -> true mirror",
            "ship perijove (R_J)",
            "disks' highest turning perijove",
        ],
    )


def _ledger_rows(unit: UnitLedger) -> List[List[str]]:
    keys = (
        "flat_plate",
        "push_charge",
        "methalox",
        "departure_charge",
        "departure_loss",
        "departure_minutes",
        "departing",
        "disks",
        "carrier_propellant",
        "ship_at_sun",
        "sun_charge",
        "sun_plate",
        "shield",
        "avionics",
        "sheets",
    )
    rows = []
    for key in keys:
        value = unit.lines[key]
        unit_label = (
            "km/s"
            if key == "departure_loss"
            else "min" if key == "departure_minutes" else "t"
        )
        shown = value if unit_label != "t" else value / 1e3
        rows.append(
            [
                key,
                f"{shown:.3f}" if unit_label == "km/s" else f"{shown:.1f}",
                unit_label,
            ]
        )
    rows.append(
        [
            "push / departure sheets spent",
            f"{unit.push_sheets / 1e3:.1f} / {unit.departure_sheets / 1e3:.1f}",
            "t",
        ]
    )
    return rows


def headline_report() -> Tuple[str, Dict[Tuple[float, float], Optimum]]:
    """The two headline closures, optimized for the dear-seed break-even."""
    out, optima = [], {}
    for key in HEADLINES:
        departure = departure_route(*key)
        push = push_route(*key)
        assert departure is not None and push is not None
        found = optimize_design(departure, push, HEADLINE_STARTS[key])
        optima[key] = found
        v, d = found.valuation, found.design
        growth = cycle_growth(departure, push, d, 1.0 - MISS_FLOOR)
        out.append(
            f"{key[0]:g} R_sun, return excess {key[1]:g} km/s (push wave {push.return_excess:.1f}):"
            f" k_P {d.k_push:.2f}, k_E {d.k_earth:.2f}, k_S {d.k_sun:.2f}, arc {d.arc_hours:.2f} h,"
            f" departure pulse {d.departure_pulse / 1e6:.2f} MN s, k_L {d.k_delivery:.2f}\n"
            f"  growth {v.growth:.2f} per 3S ({growth.per_total_impactor:.2f} with the disks"
            f" counted as impactor), doubling {v.doubling:.2f} yr, push share {growth.push_share:.2f}\n"
            f"  steady ${v.steady_cost:.0f}/kg at L1; break-even ${_money(v.break_even)} /"
            f" ${_money(found.dear_break_even)} (cheap / dear seed at"
            f" ${seed_price(departure):.0f} / ${seed_price(departure, True):.0f} per kg);"
            f" seed ${v.seed_per_fleet_kg:.0f} per fleet kg at x{v.fleet_multiple:.0f}\n"
            + tabulate(
                _ledger_rows(growth.departure_unit),
                headers=["departure-wave unit", "", ""],
            )
            + "\n  $/kg at L1 by line: "
            + ", ".join(
                f"{k} {x:.1f}"
                for k, x in sorted(v.steady_lines.items(), key=lambda kv: -kv[1])
                if x >= 0.05
            )
        )
    return "\n\n".join(out), optima


@dataclass(frozen=True)
class Sensitivity:
    """One sensitivity row: what it changes from the headline.

    Attributes:
        name: The row's label.
        design: Design fields replaced.
        prices: Price-book fields replaced.
        odds: Gate odds in order; None keeps the headline's, () drops them.
        guidance: Sheet guidance mass share.
        rate: Annual time value of money.
        growth_synodic: Synodic periods of growth before delivery.
    """

    name: str
    design: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    odds: Optional[Tuple[float, ...]] = None
    guidance: float = 0.05
    rate: float = 0.10
    growth_synodic: int = GROWTH_SYNODIC
    bonds: Optional[str] = None


SENSITIVITIES = (
    Sensitivity("headline"),
    Sensitivity("guidance 2%", guidance=0.02),
    Sensitivity("guidance 10%", guidance=0.10),
    Sensitivity("Sun film flat in v_e (exponent 0)", design={"film_exponent": 0.0}),
    Sensitivity("g cap 20", design={"g_cap": 20.0}),
    Sensitivity("g cap 500", design={"g_cap": 500.0}),
    Sensitivity("bonds returned", bonds="bonds returned"),
    Sensitivity("shade scales with sunlight", design={"shield_exponent": 1.0}),
    Sensitivity("carrier perijove <= 2 R_J", design={"carrier_perijove": 2.0}),
    Sensitivity("flat plate 25% of spray cup", prices={"flat_plate_share": 0.25}),
    Sensitivity("flat plate 100%", prices={"flat_plate_share": 1.0}),
    Sensitivity("disks $500/kg", prices={"disk": 500.0}),
    Sensitivity("disks $2000/kg", prices={"disk": 2000.0}),
    Sensitivity("lob $50/kg", prices={"lob": 50.0}),
    Sensitivity("optimistic odds 98/90/90/95", odds=(0.98, 0.90, 0.90, 0.95)),
    Sensitivity("growth 10 synodic", growth_synodic=10),
    Sensitivity("growth 21 synodic", growth_synodic=21),
    Sensitivity("flat 30%, no odds, 9 synodic", odds=(), rate=0.30, growth_synodic=9),
)


def sensitivity_report(optimum: Optimum) -> str:
    """Each sensitivity at the headline design (not re-optimized)."""
    departure, push = optimum.departure, optimum.push
    rows = []
    for case in SENSITIVITIES:
        design = replace(optimum.design, **case.design)  # type: ignore[arg-type]
        if case.bonds is not None:
            design = replace(design, bonds=case.bonds)
        prices = replace(ESTIMATE, **case.prices)  # type: ignore[arg-type]
        gates: Optional[Tuple[Gate, ...]] = None
        if case.odds is not None:
            base = solar_dive_gates(departure, push)
            gates = tuple(replace(g, odds=o) for g, o in zip(base, case.odds))
        values = [
            value_design(
                departure,
                push,
                design,
                prices,
                seed_price(departure, dear),
                case.guidance,
                case.rate,
                gates,
                case.growth_synodic,
            )
            for dear in (False, True)
        ]
        cheap = values[0]
        rows.append(
            [
                case.name,
                f"{cheap.growth:.2f}",
                f"{cheap.doubling:.2f}",
                f"{cheap.steady_cost:.0f}",
                " / ".join(_money(v.break_even) for v in values),
                f"{cheap.seed_per_fleet_kg:.0f}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "case",
            "growth",
            "doubling (yr)",
            "steady $/kg",
            "break-even cheap / dear",
            "seed $/fleet kg",
        ],
    )


def _sweep_point(point: Tuple[float, float]) -> List[str]:
    radii, excess = point
    try:
        departure = departure_route(radii, excess)
        push = push_route(radii, excess) if departure else None
        if departure is None or push is None:
            return [f"{radii:g}", f"{excess:g}", "no closure"]
        found = optimize_design(departure, push, _SWEEP_START)
    except ValueError as error:
        return [f"{radii:g}", f"{excess:g}", str(error)]
    v = found.valuation
    if v.break_even is None or v.growth <= 1.0:
        return [f"{radii:g}", f"{excess:g}", f"does not grow ({v.growth:.2f})"]
    return [
        f"{radii:g}",
        f"{excess:g}",
        f"{push.return_excess:.1f}",
        f"{v.growth:.2f}",
        f"{v.doubling:.2f}",
        f"{v.steady_cost:.0f}",
        f"{_money(v.break_even)} / {_money(found.dear_break_even)}",
    ]


def sweep_report(workers: int = 4) -> str:
    """Every depth and return excess, each optimized for the dear seed."""
    from multiprocessing import Pool

    points = [(r, x) for r in SWEEP_RADII for x in RETURN_EXCESSES]
    with Pool(workers) as pool:
        rows = pool.map(_sweep_point, points)
    return tabulate(
        rows,
        headers=[
            "R_sun",
            "return excess",
            "push wave's",
            "growth",
            "doubling",
            "steady $/kg",
            "break-even cheap / dear",
        ],
    )


def child_report() -> str:
    """The child's chamber and methalox rows on the same footing, both schedules."""
    from src.growth_cost import ESTIMATE as CHILD_ESTIMATE
    from src.growth_cost import steady_state_cost
    from src.growth_cost_inputs import seed_prices
    from src.growth_cost_report import climbing_lob
    from src.lob_rise import OPERATING_RISE_SPEED
    from src.survivable_chamber import _in_workers, _priced, cost_designs

    designs = cost_designs()
    priced = _in_workers(_priced, list(CHILD_ROWS))
    book = climbing_lob(CHILD_ESTIMATE, OPERATING_RISE_SPEED)
    low, high = seed_prices()
    rows = []
    for index, (_, inputs) in zip(CHILD_ROWS, priced):
        steady, _ = steady_state_cost(inputs, book)
        gated = [child_break_even(inputs, book, s) for s in (low, high)]
        flat = [
            child_break_even(inputs, book, s, 9, 0.30, gated=False) for s in (low, high)
        ]
        rows.append(
            [
                designs[index].label,
                f"{steady:.0f}",
                " / ".join(_money(v) for v in gated),
                " / ".join(_money(v) for v in flat),
            ]
        )
    return tabulate(
        rows,
        headers=[
            "design",
            "steady $/kg",
            "gated 10%, 15 S: cheap / dear",
            "flat 30%, 9 S: cheap / dear",
        ],
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the solar-dive re-evaluation (D1-D7).

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sweep", action="store_true", help="the depth sweep (~25 min)"
    )
    parser.add_argument(
        "--child", action="store_true", help="the child's rows (~10 min)"
    )
    args = parser.parse_args(argv)
    print(
        "Inputs the simulation copied, checked (G1, G4, G10)\n" + corrections_report()
    )
    report, optima = headline_report()
    print(
        "\nHeadline: gated rNPV at 10%/yr, 15 synodic periods of growth, dear-seed optimum\n"
        + report
    )
    for key, optimum in optima.items():
        print(
            f"\nSensitivities at {key[0]:g} R_sun (headline design held)\n"
            + sensitivity_report(optimum)
        )
    if args.sweep:
        print("\nDepth sweep\n" + sweep_report())
    if args.child:
        print("\nThe child's rows on the same footing\n" + child_report())


if __name__ == "__main__":
    main()
