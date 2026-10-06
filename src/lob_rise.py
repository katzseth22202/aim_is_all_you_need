"""What a lob that is still climbing at the 400 km intercept costs (ADR 0041, 0042).

The growth push lasts about 300 s.  Unsupported, the launch unit falls about
200 km over it while the stream, nearly straight at 60 km/s, rises about
185 km as the Earth curves away beneath it.  A plate cannot tilt its thrust to
hold altitude (ADR 0009), so the lob has to arrive at 400 km still climbing,
at ~1.0-1.2 km/s (:func:`push_track`; the impact-sim handoff's figure).  That
is lofted mass the booster no longer carries.

:func:`lofted_mass` integrates a vertical ascent with ADR 0037's scaling: a
5000 t liftoff, 250 t dry, 75 MN, scaled together, burned to depletion, no
drag and no braking reserve.  At fixed thrust-to-weight and propellant fraction
the lofted mass scales with the vehicle, so the climb is charged as the ratio
:func:`booster_growth`, independent of the booster's size.

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

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

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


def _speed_at_intercept(payload: float, scale: float, isp: float) -> float:
    """Signed vertical speed at 400 km (negative: the apex falls short)."""
    m0, dry, thrust = LIFTOFF_MASS * scale, DRY_MASS * scale, THRUST * scale
    propellant = m0 - dry - payload
    if propellant <= 0.0:
        return -np.inf
    flow = thrust / (isp * _G0)

    def rhs(t: float, y: np.ndarray) -> list:
        return [y[1], thrust / (m0 - flow * t) - _MU / (_R_EARTH + y[0]) ** 2]

    h, v = solve_ivp(
        rhs, (0.0, propellant / flow), [0.0, 0.0], rtol=1e-10, atol=1e-6
    ).y[:, -1]
    energy = 0.5 * v * v - _MU / (_R_EARTH + h)
    v_sq = 2.0 * (energy + _MU / (_R_EARTH + INTERCEPT_ALTITUDE))
    return float(np.sign(v_sq) * np.sqrt(abs(v_sq)))


def lofted_mass(rise_speed: float, scale: float = 1.0, isp: float = ISP) -> float:
    """Payload (kg) a vertical lob carries to 400 km still climbing at ``rise_speed``.

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
    rise_speed: float, isp: float = ISP, baseline: float = BASELINE_RISE_SPEED
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

    Returns:
        Lofted mass at ``baseline`` over lofted mass at ``rise_speed``.
    """
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
