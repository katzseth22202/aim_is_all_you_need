"""The speed a finite departure burn loses against an impulsive one.

A burn ``dv`` changes the specific orbital energy by about ``v dv``, so it is
worth most at periapsis, where ``v`` peaks.  A burn of finite length spends part
of itself away from periapsis, and a head-on chamber also cannot follow the
velocity as it turns: it must point along the arriving stream, one fixed
direction.  This module integrates that burn instead of scaling the parent's
single fixed-direction figure.

The model is planar two-body motion about Earth.  The burn is centred in time
on the periapsis of the 20-day cycle orbit at 600 km (``sec:jovian_meeting_altitudes``),
with constant mass flow, since the chamber fires a fixed pulse rate.  The loss
is the extra ideal burn, ``v_e ln(m0/m1)``, needed to reach the orbital energy
an impulsive burn at periapsis would, and so the same excess speed.

Pinned to the parent's ``sec:ntr_departure``: steered along the velocity, the
reference reactor's 5.39 km/s at 906 s loses 22, 226 and 486 m/s over 320,
1200 and 2100 s.  Held in one direction, a pulsed burn loses 27 m/s over 320 s
and 101 m/s over 640 s.
"""

from functools import lru_cache
from typing import Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.bodies import Earth
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq, minimize_scalar

from src.astro_constants import PUFFSAT_CYCLE_ORBIT_PERIOD

#: Periapsis altitude the departure burns at (``sec:jovian_meeting_altitudes``).
DEPARTURE_ALTITUDE = 600.0 * u.km
_RTOL = 1.0e-10
_ATOL = 1.0e-9


def _orbit(altitude: u.Quantity, period: u.Quantity) -> Tuple[float, float, float]:
    """Earth's ``mu`` (km^3/s^2), periapsis radius (km) and speed (km/s)."""
    mu = float(Earth.k.to_value(u.km**3 / u.s**2))
    r_p = float((Earth.R + altitude).to_value(u.km))
    seconds = float(period.to_value(u.s))
    a = (mu * (seconds / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    return mu, r_p, float(np.sqrt(mu * (2.0 / r_p - 1.0 / a)))


def _final_energy(
    mu: float,
    start: npt.NDArray[np.float64],
    ideal_dv: float,
    exhaust: float,
    seconds: float,
    direction: float,
    steered: bool,
) -> float:
    """Specific orbital energy after a constant-mass-flow burn from ``start``."""
    spent = 1.0 - np.exp(-ideal_dv / exhaust)
    heading = np.array([np.cos(direction), np.sin(direction)])

    def rhs(t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r = y[:2]
        v = y[2:]
        thrust = exhaust * spent / seconds / (1.0 - spent * t / seconds)
        axis = v / np.linalg.norm(v) if steered else heading
        gravity = -mu * r / np.linalg.norm(r) ** 3
        return np.concatenate((v, gravity + thrust * axis))

    end = solve_ivp(rhs, (0.0, seconds), start, rtol=_RTOL, atol=_ATOL).y[:, -1]
    return float(0.5 * end[2:] @ end[2:] - mu / np.linalg.norm(end[:2]))


def _coast_back(
    mu: float, r_p: float, v_p: float, seconds: float
) -> npt.NDArray[np.float64]:
    """State ``seconds`` before periapsis on the unpowered orbit.

    Periapsis sits on the +x axis with the velocity along +y.
    """

    def rhs(t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.concatenate((y[2:], -mu * y[:2] / np.linalg.norm(y[:2]) ** 3))

    periapsis = np.array([r_p, 0.0, 0.0, v_p])
    back = solve_ivp(rhs, (0.0, -seconds), periapsis, rtol=_RTOL, atol=_ATOL)
    return np.asarray(back.y[:, -1], dtype=np.float64)


def finite_burn_loss(
    impulsive_dv: u.Quantity,
    exhaust_speed: u.Quantity,
    burn_time: u.Quantity,
    steered: bool = False,
    altitude: u.Quantity = DEPARTURE_ALTITUDE,
    period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD,
) -> u.Quantity:
    """Extra ideal burn a finite burn needs to match an impulsive one's energy.

    Args:
        impulsive_dv: Burn an instantaneous impulse at periapsis would need.
        exhaust_speed: Effective exhaust speed, held over the burn.
        burn_time: Burn length, centred on periapsis.
        steered: Thrust along the velocity; otherwise in the one fixed
            direction that loses least, as a head-on chamber must.
        altitude: Periapsis altitude of the orbit the burn starts on.
        period: Period of that orbit.

    Returns:
        The loss (m/s).
    """
    mu, r_p, v_p = _orbit(altitude, period)
    dv = float(impulsive_dv.to_value(u.km / u.s))
    exhaust = float(exhaust_speed.to_value(u.km / u.s))
    seconds = float(burn_time.to_value(u.s))
    target = 0.5 * (v_p + dv) ** 2 - mu / r_p
    start = _coast_back(mu, r_p, v_p, 0.5 * seconds)

    def needed(direction: float) -> float:
        def gap(ideal: float) -> float:
            return (
                _final_energy(mu, start, ideal, exhaust, seconds, direction, steered)
                - target
            )

        return float(brentq(gap, dv, 2.0 * dv + 1.0, xtol=1.0e-9))

    along = 0.5 * np.pi
    if steered:
        ideal = needed(along)
    else:
        best = minimize_scalar(
            needed, bounds=(along - 0.5, along + 0.5), method="bounded"
        )
        ideal = float(best.fun)
    return ((ideal - dv) * u.km / u.s).to(u.m / u.s)


#: Table nodes for :func:`fixed_direction_loss`.  Burns span the flown chain's
#: 5.3-7.2 km/s with room for losses and the 600 km premium; times span one
#: chamber on a large stack.  The loss barely depends on the exhaust speed (under
#: 1% from 800 s to 1431 s), so the table is built at one representative value.
TABLE_BURNS = np.arange(4.0, 10.01, 1.0)
TABLE_TIMES = np.arange(0.0, 3600.01, 200.0)
TABLE_EXHAUST = 11.0 * u.km / u.s


@lru_cache(maxsize=1)
def _loss_table() -> RegularGridInterpolator:
    """Fixed-direction loss (m/s) over :data:`TABLE_BURNS` x :data:`TABLE_TIMES`."""
    values = np.zeros((TABLE_BURNS.size, TABLE_TIMES.size))
    for i, burn in enumerate(TABLE_BURNS):
        for j, seconds in enumerate(TABLE_TIMES[1:], start=1):
            values[i, j] = finite_burn_loss(
                burn * u.km / u.s, TABLE_EXHAUST, seconds * u.s
            ).to_value(u.m / u.s)
    return RegularGridInterpolator((TABLE_BURNS, TABLE_TIMES), values, method="cubic")


def fixed_direction_loss(impulsive_dv: u.Quantity, burn_time: u.Quantity) -> u.Quantity:
    """Fixed-direction loss read off a cached table of :func:`finite_burn_loss`.

    The table is built on first use, at 600 km on the 20-day orbit.

    Args:
        impulsive_dv: Burn an instantaneous impulse at periapsis would need.
        burn_time: Burn length, centred on periapsis.

    Returns:
        The loss (m/s).

    Raises:
        ValueError: If the burn or its length lies outside the table.
    """
    burn = float(impulsive_dv.to_value(u.km / u.s))
    seconds = float(burn_time.to_value(u.s))
    if not (TABLE_BURNS[0] <= burn <= TABLE_BURNS[-1]):
        raise ValueError(f"burn {impulsive_dv} is outside the loss table")
    if not (0.0 <= seconds <= TABLE_TIMES[-1]):
        raise ValueError(f"burn time {burn_time} is outside the loss table")
    return float(_loss_table()([[burn, seconds]])[0]) * u.m / u.s
