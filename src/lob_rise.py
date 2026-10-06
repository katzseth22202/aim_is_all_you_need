"""What a lob that is still climbing at the 400 km intercept costs (ADR 0041-0043).

The growth push lasts about 300 s.  Unsupported, the launch unit falls about
200 km over it while the stream, nearly straight at 60 km/s, rises about
185 km as the Earth curves away beneath it.  A plate cannot tilt its thrust to
hold altitude (ADR 0009), so the lob has to arrive at 400 km still climbing,
at ~1.0-1.2 km/s (:func:`push_track`; the impact-sim handoff's figure).  That
is lofted mass the booster no longer carries.

:func:`lofted_mass` integrates a vertical ascent with ADR 0037's scaling: a
5000 t liftoff, 250 t dry, 75 MN, scaled together, no drag.  At fixed
thrust-to-weight and propellant fraction the lofted mass scales with the
vehicle, so the climb is charged as the ratio :func:`booster_growth`,
independent of the booster's size.

The booster holds back its brake (ADR 0043).  Falling straight down from its
~430 km top it would cross 60 km at ~2.6 km/s, a ~21 g entry, so the parent's
``sec:vertical_lob`` brakes right after separation until it will cross 60 km
at 1.5 km/s (~9 g), then lands on a 0.3 km/s burn.  A vertical entry's peak
deceleration depends on its entry speed alone (Allen and Eggers), so the same
criterion holds at any climb.  A faster climb separates the booster higher and
faster, so its brake grows: 1.39 km/s at 0.75 km/s, 1.58 at 1.1 (380 s).  The
brake and landing come out of what the booster would have burned climbing
(:func:`braked_lob`), which takes the charge at 1.1 km/s from x1.044 to
x1.065.  This integration's 0.75 km/s brake (1.35-1.39 km/s, 143-155 t) sits
on the parent's 1.33-1.38 km/s and 114-185 t.

The ratio is taken against the lob ADR 0037 priced, not against an apex at
400 km (ADR 0042).  The parent's ``sec:vertical_lob`` lofts its 1250-1430 t to
a ~430 km top, so that lob already passes 400 km at ~0.75 km/s
(:data:`BASELINE_RISE_SPEED`), and ADR 0037's 1.1-1.2x booster was sized from
it.  Only the climb above that is new: x1.044 at the 1.1 km/s operating point
(:data:`OPERATING_RISE_SPEED`), against ADR 0041's x1.085 from an apex.  This integration
lofts 1575 t (380 s) where ADR 0037 quoted 1422 t for the same vehicle; only
ratios are taken from it.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from tabulate import tabulate

_MU = 3.986004418e14  # m^3/s^2
_R_EARTH = 6378.137e3  # m
_G0 = 9.80665
#: ADR 0037's booster at scale 1.
LIFTOFF_MASS = 5.0e6  # kg
DRY_MASS = 2.5e5  # kg
THRUST = 7.5e7  # N
#: The parent's vacuum methalox Isp, and ADR 0037's pessimistic average.
ISP = 380.0
ISP_PESSIMISTIC = 350.0
INTERCEPT_ALTITUDE = 400.0e3  # m
#: Climb rate at the intercept the push needs (ADR 0041).
RISE_SPEEDS = (1.0e3, 1.2e3)  # m/s
#: The climb that holds the unit within ~65 km of the stream (ADR 0041), the
#: one the parent prices.
OPERATING_RISE_SPEED = 1.1e3  # m/s
#: What ADR 0037's lob already does at 400 km: the parent's lob tops out near
#: 430 km (ADR 0042).
BASELINE_RISE_SPEED = 0.75e3  # m/s
#: The booster's brake: it crosses 60 km falling at 1.5 km/s, ~9 g straight
#: down, then lands on 0.3 km/s (parent ``sec:vertical_lob``, ADR 0043).
ENTRY_ALTITUDE = 60.0e3  # m
ENTRY_SPEED = 1.5e3  # m/s
LANDING_BURN = 0.3e3  # m/s
_RESERVE_TOLERANCE = 1.0  # kg
_RESERVE_ITERATIONS = 50


def _burnout(
    payload: float, scale: float, isp: float, reserve: float = 0.0
) -> Tuple[float, float]:
    """Altitude and speed at separation, holding ``reserve`` back (m, m/s)."""
    m0, dry, thrust = LIFTOFF_MASS * scale, DRY_MASS * scale, THRUST * scale
    propellant = m0 - dry - payload - reserve
    flow = thrust / (isp * _G0)

    def rhs(t: float, y: np.ndarray) -> list:
        return [y[1], thrust / (m0 - flow * t) - _MU / (_R_EARTH + y[0]) ** 2]

    h, v = solve_ivp(
        rhs, (0.0, propellant / flow), [0.0, 0.0], rtol=1e-10, atol=1e-6
    ).y[:, -1]
    return float(h), float(v)


def _coast_speed(altitude: float, speed: float, to_altitude: float) -> float:
    """Signed vertical speed at ``to_altitude`` (negative: never reaches it)."""
    v_sq = speed * speed + 2.0 * _MU * (
        1.0 / (_R_EARTH + to_altitude) - 1.0 / (_R_EARTH + altitude)
    )
    return float(np.sign(v_sq) * np.sqrt(abs(v_sq)))


def _speed_at_intercept(
    payload: float, scale: float, isp: float, reserve: float = 0.0
) -> float:
    """Signed vertical speed at 400 km (negative: the apex falls short)."""
    if LIFTOFF_MASS * scale - DRY_MASS * scale - payload - reserve <= 0.0:
        return -np.inf
    return _coast_speed(
        *_burnout(payload, scale, isp, reserve), to_altitude=INTERCEPT_ALTITUDE
    )


def brake_dv(altitude: float, speed: float) -> float:
    """The brake that makes a booster separating here cross 60 km at 1.5 km/s.

    Impulsive at separation.  A finite brake is helped by gravity, which slows
    the climb too, and hurt by spending its thrust at falling speed; under 4 g
    it lasts ~40 s, and the impulse is taken as the estimate.

    Args:
        altitude: Separation altitude, m.
        speed: Vertical speed at separation, m/s.

    Returns:
        The brake, m/s.
    """
    after_sq = ENTRY_SPEED**2 - 2.0 * _MU * (
        1.0 / (_R_EARTH + ENTRY_ALTITUDE) - 1.0 / (_R_EARTH + altitude)
    )
    return float(speed - np.sqrt(after_sq))


@dataclass(frozen=True)
class BrakedLob:
    """A vertical lob whose booster holds back its brake and landing burn.

    Attributes:
        lofted: Payload carried to 400 km, kg.
        reserve: Propellant held for the brake and landing, kg.
        brake: The brake, m/s.
        separation_altitude: Where the booster lets go, m.
        separation_speed: Its vertical speed there, m/s.
        unbraked_entry: The speed it would cross 60 km at without the brake, m/s.
    """

    lofted: float
    reserve: float
    brake: float
    separation_altitude: float
    separation_speed: float
    unbraked_entry: float


@lru_cache(maxsize=64)
def braked_lob(rise_speed: float, scale: float = 1.0, isp: float = ISP) -> BrakedLob:
    """A vertical lob to 400 km climbing at ``rise_speed``, the brake held back.

    The reserve brakes the empty booster and lands it, so it depends on where
    the booster separates, which depends on the reserve; solved as a fixed
    point (it converges in a few steps).

    Args:
        rise_speed: Vertical speed at 400 km, m/s.
        scale: Booster size relative to ADR 0037's 5000 t vehicle.
        isp: Average specific impulse, s.

    Returns:
        The lob.
    """
    dry = DRY_MASS * scale
    reserve = 0.0
    for _ in range(_RESERVE_ITERATIONS):
        held = reserve
        payload = float(
            brentq(
                lambda p: _speed_at_intercept(p, scale, isp, held) - rise_speed,
                1.0,
                LIFTOFF_MASS * scale - dry - held - 1.0,
                xtol=1.0,
            )
        )
        altitude, speed = _burnout(payload, scale, isp, held)
        brake = brake_dv(altitude, speed)
        reserve = dry * float(np.expm1((brake + LANDING_BURN) / (isp * _G0)))
        if abs(reserve - held) < _RESERVE_TOLERANCE * scale:
            break
    else:
        raise RuntimeError("the brake's reserve did not converge")
    return BrakedLob(
        lofted=payload,
        reserve=reserve,
        brake=brake,
        separation_altitude=altitude,
        separation_speed=speed,
        unbraked_entry=_coast_speed(altitude, speed, ENTRY_ALTITUDE),
    )


def lofted_mass(rise_speed: float, scale: float = 1.0, isp: float = ISP) -> float:
    """Payload (kg) a vertical lob carries to 400 km still climbing at ``rise_speed``.

    Burned to depletion, with no brake held back (ADR 0041-0042's measure; see
    :func:`braked_lob`).

    Args:
        rise_speed: Vertical speed at 400 km, m/s; zero is an apex there.
        scale: Booster size relative to ADR 0037's 5000 t vehicle.
        isp: Average specific impulse, s.

    Returns:
        The lofted mass, kg.
    """
    return float(
        brentq(
            lambda p: _speed_at_intercept(p, scale, isp) - rise_speed,
            1.0,
            LIFTOFF_MASS * scale - DRY_MASS * scale - 1.0,
            xtol=1.0,
        )
    )


def booster_growth(
    rise_speed: float,
    isp: float = ISP,
    baseline: float = BASELINE_RISE_SPEED,
    brake: bool = True,
) -> float:
    """How much bigger the booster must be to loft the same mass climbing.

    Also the rise in the lob's price per kilogram lofted, since ADR 0037 prices
    a flight in proportion to the booster's size.

    Args:
        rise_speed: Vertical speed at 400 km, m/s.
        isp: Average specific impulse, s.
        baseline: The climb rate at 400 km the existing lob price already
            pays for (ADR 0042); 0.0 measures from an apex there, as ADR 0041
            first did.
        brake: Hold back the booster's brake and landing (ADR 0043), as the
            parent's priced lob does; False is ADR 0041-0042's unbraked charge.

    Returns:
        Lofted mass at ``baseline`` over lofted mass at ``rise_speed``.
    """
    if brake:
        return (
            braked_lob(baseline, isp=isp).lofted
            / braked_lob(rise_speed, isp=isp).lofted
        )
    return lofted_mass(baseline, isp=isp) / lofted_mass(rise_speed, isp=isp)


@dataclass(frozen=True)
class PushTrack:
    """The launch unit's altitude against the stream's over one push.

    Attributes:
        duration: Push time, s.
        below: Deepest the craft sits below the stream's path, km (>= 0).
        above: Highest it sits above the path, km (>= 0).
        stream_rise: How far the stream's path climbs over the push, km.
        final_periapsis: Periapsis altitude of the orbit the push ends on, km.
    """

    duration: float
    below: float
    above: float
    stream_rise: float
    final_periapsis: float


def push_track(
    rise_speed: float,
    stream_speed: float,
    end_speed: float,
    delivered_fraction: float,
    thrust_to_mass: float = 48.0e6 / 1.5e6,
) -> PushTrack:
    """Fly the push in the orbit plane, thrust along the stream, from 400 km.

    The stream is one hyperbola with its periapsis at 400 km, where the craft
    starts at rest horizontally and climbing at ``rise_speed``.  The thrust is
    held constant (48 MN, 4 Hz of 12 MN s) and the craft's mass falls linearly
    with speed to ``delivered_fraction`` (an estimate of the push's schedule).

    Args:
        rise_speed: Vertical speed at the intercept, km/s.
        stream_speed: The stream's periapsis speed, km/s.
        end_speed: The push ends at this speed (the parking orbit's), km/s.
        delivered_fraction: Craft mass after the push over mass before it.
        thrust_to_mass: Thrust over the craft's starting mass, m/s^2.

    Returns:
        The track.
    """
    mu = _MU * 1e-9
    rp = (_R_EARTH + INTERCEPT_ALTITUDE) * 1e-3
    e = stream_speed**2 * rp / mu - 1.0
    p = rp * (1.0 + e)
    a0 = thrust_to_mass * 1e-3

    def stream_radius(theta: np.ndarray) -> np.ndarray:
        return p / (1.0 + e * np.cos(theta))

    def rhs(t: float, y: np.ndarray) -> list:
        r, theta, vr, vt = y
        speed = np.hypot(vr, vt)
        mass = 1.0 - (1.0 - delivered_fraction) * min(speed / end_speed, 1.0)
        sr, st = e * np.sin(theta), 1.0 + e * np.cos(theta)
        norm = np.hypot(sr, st)
        a = a0 / mass
        return [
            vr,
            vt / r,
            vt * vt / r - mu / r**2 + a * sr / norm,
            -vr * vt / r + a * st / norm,
        ]

    def done(t: float, y: np.ndarray) -> float:
        return float(np.hypot(y[2], y[3]) - end_speed)

    done.terminal = True  # type: ignore[attr-defined]
    flown = solve_ivp(
        rhs, (0.0, 3000.0), [rp, 0.0, rise_speed, 0.0], events=done,
        max_step=1.0, rtol=1e-9,
    )  # fmt: skip
    r, theta, vr, vt = flown.y
    gap = r - stream_radius(theta)
    h = r[-1] * vt[-1]
    a = -mu / (2.0 * (0.5 * (vr[-1] ** 2 + vt[-1] ** 2) - mu / r[-1]))
    ecc = np.sqrt(1.0 - h * h / (mu * a))
    return PushTrack(
        duration=float(flown.t[-1]),
        below=float(max(-gap.min(), 0.0)),
        above=float(max(gap.max(), 0.0)),
        stream_rise=float(stream_radius(theta[-1]) - rp),
        final_periapsis=float(a * (1.0 - ecc) - _R_EARTH * 1e-3),
    )


def main() -> None:
    """Print the braked lob across climb rates, and the charge it sets (ADR 0043)."""
    rows = []
    for isp in (ISP, ISP_PESSIMISTIC):
        for rise in (
            BASELINE_RISE_SPEED,
            RISE_SPEEDS[0],
            OPERATING_RISE_SPEED,
            RISE_SPEEDS[1],
        ):
            lob = braked_lob(rise, isp=isp)
            rows.append(
                {
                    "Isp s": isp,
                    "climb km/s": rise / 1e3,
                    "lofted t": lob.lofted / 1e3,
                    "separation km": lob.separation_altitude / 1e3,
                    "at km/s": lob.separation_speed / 1e3,
                    "unbraked entry km/s": lob.unbraked_entry / 1e3,
                    "brake km/s": lob.brake / 1e3,
                    "reserve t": lob.reserve / 1e3,
                    "charge": booster_growth(rise, isp),
                    "unbraked charge": booster_growth(rise, isp, brake=False),
                }
            )
    print(
        "Vertical lob to 400 km (ADR 0037's 5000 t / 250 t dry / 75 MN, no drag), "
        f"the booster braked after separation to cross {ENTRY_ALTITUDE / 1e3:g} km at "
        f"{ENTRY_SPEED / 1e3:g} km/s, landing on {LANDING_BURN / 1e3:g} km/s (ADR 0043).  "
        f"Charge = lofted at {BASELINE_RISE_SPEED / 1e3:g} km/s over lofted at the climb."
    )
    print(tabulate(rows, headers="keys", floatfmt=".4g"))


if __name__ == "__main__":
    main()
