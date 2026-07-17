"""Float-valued two-body conic geometry: the substrate under the growth loop.

The Jovian flyby and assist-chain searches (`scenario.py`) and the nozzle
analysis (`nozzle_analysis.py`) all evaluate the same planar conic-section
geometry -- ellipses and hyperbolas, no perturbations -- many thousands of
times inside optimizer hot loops. This module is that shared geometry: plain
floats (km, s, km/s, rad) rather than `astropy.units.Quantity`, because a
Quantity's unit bookkeeping is too slow to pay per-evaluation inside a
`differential_evolution` objective or a beam search.

`orbit_utils.py`'s Quantity-based time-of-flight and true-anomaly functions
delegate to this module (see `elliptic_time_of_flight`,
`hyperbolic_time_of_flight`, `true_anomaly_at_radius`, `hyperbolic_eccentricity`
there) rather than duplicating the algebra, so this module sits *below*
`orbit_utils` in the dependency hierarchy:

    astro_constants -> conic_kernel -> orbit_utils -> propulsion -> scenario -> main

CONTEXT.md calls this module the "conic kernel".
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np


def hyperbolic_eccentricity(
    mu: float, periapsis_radius: float, v_infinity: float
) -> float:
    """Eccentricity of the hyperbola with a given periapsis and excess speed.

    ``e = 1 + r_p * v_infinity**2 / mu`` (specific energy ``v_infinity**2 / 2``
    gives ``a = -mu / v_infinity**2``, and ``r_p = a(1 - e)`` inverts to this).

    Args:
        mu: Gravitational parameter of the attractor (km^3/s^2).
        periapsis_radius: Periapsis distance from the attractor's center (km).
        v_infinity: Hyperbolic-excess speed far from the attractor (km/s).

    Returns:
        The eccentricity, strictly greater than 1 for v_infinity > 0.
    """
    return 1.0 + periapsis_radius * v_infinity * v_infinity / mu


def eccentricity_from_energy_and_h(mu: float, energy: float, h: float) -> float:
    """Eccentricity of a conic from its specific energy and angular momentum.

    ``e = sqrt(1 + 2*energy*h**2/mu**2)``. Works for either sign of ``energy``
    (ellipse or hyperbola); clamped at 0 so round-off on a near-circular or
    near-parabolic orbit cannot take the square root negative.

    Args:
        mu: Gravitational parameter of the attractor (km^3/s^2).
        energy: Specific orbital energy, ``v**2/2 - mu/r`` (km^2/s^2).
        h: Specific angular momentum, ``r * v_tangential`` (km^2/s).

    Returns:
        The (dimensionless) eccentricity, >= 0.
    """
    return float(np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / (mu * mu))))


def semimajor_axis_from_energy(mu: float, energy: float) -> float:
    """Semi-major axis from specific energy, ``a = -mu / (2*energy)``.

    Positive for an ellipse (energy < 0), negative for a hyperbola.

    Args:
        mu: Gravitational parameter of the attractor (km^3/s^2).
        energy: Specific orbital energy (km^2/s^2), nonzero.

    Returns:
        The semi-major axis (km).
    """
    return -mu / (2.0 * energy)


def periapsis_radius_of_conic(semi_latus_rectum: float, eccentricity: float) -> float:
    """Periapsis radius of a conic, ``p / (1 + e)``.

    Args:
        semi_latus_rectum: The conic's semi-latus rectum ``p`` (km).
        eccentricity: The (dimensionless) eccentricity, >= 0.

    Returns:
        The periapsis radius (km).
    """
    return semi_latus_rectum / (1.0 + eccentricity)


def half_turn_angle(eccentricity: float) -> float:
    """Single-asymptote turning half-angle of a hyperbola, ``asin(1/e)`` (rad).

    The angle between a hyperbola's incoming asymptote and its periapsis
    direction. An *unpowered* flyby bends the excess velocity by twice this
    (see :func:`unpowered_bend_angle`); a *powered* one splits into an inbound
    and outbound hyperbola of different eccentricities and bends by the sum of
    their two half-angles (see :func:`powered_bend_angle`) -- the unpowered
    formula does not apply across a burn (CONTEXT.md, "Powered Jovian flyby").

    Args:
        eccentricity: The hyperbola's eccentricity, > 1 (values within
            round-off of 1 from below are clamped to the asymptotic 90 deg).

    Returns:
        The half-angle (rad), in (0, pi/2].
    """
    return float(np.arcsin(min(1.0, 1.0 / eccentricity)))


def unpowered_bend_angle(eccentricity: float) -> float:
    """Total turn angle of an unpowered flyby, ``2 * asin(1/e)`` (rad).

    Args:
        eccentricity: The flyby hyperbola's eccentricity, > 1.

    Returns:
        The turn angle (rad), in (0, pi].
    """
    return 2.0 * half_turn_angle(eccentricity)


def powered_bend_angle(eccentricity_in: float, eccentricity_out: float) -> float:
    """Total turn angle of a powered (split-hyperbola) flyby (rad).

    The periapsis burn joins an inbound hyperbola of ``eccentricity_in`` to an
    outbound one of ``eccentricity_out`` sharing the same periapsis; the total
    bend is the sum of their half-angles, ``asin(1/e_in) + asin(1/e_out)``.

    Args:
        eccentricity_in: Eccentricity of the inbound hyperbola, > 1.
        eccentricity_out: Eccentricity of the outbound hyperbola, > 1.

    Returns:
        The turn angle (rad).
    """
    return half_turn_angle(eccentricity_in) + half_turn_angle(eccentricity_out)


@dataclass(frozen=True)
class ConicState:
    """The orbit-wide constants of a planar two-body conic.

    Attributes:
        energy: Specific orbital energy, ``v**2/2 - mu/r`` (km^2/s^2).
        h: Specific angular momentum, ``r * v_tangential`` (km^2/s). Positive
            for prograde motion.
        p: Semi-latus rectum, ``h**2 / mu`` (km).
        ecc: Eccentricity (dimensionless): < 1 ellipse, > 1 hyperbola.
    """

    energy: float
    h: float
    p: float
    ecc: float


def conic_state_at_radius(
    mu: float, r: float, v_tangential: float, v_radial: float
) -> ConicState:
    """The conic's orbit-wide constants from a state at radius ``r``.

    ``energy`` and ``h`` (and so ``p`` and ``ecc``) are conserved along the
    orbit, so the radius at which the state happens to be known does not
    appear in the result -- this is what lets a caller evaluate the returned
    :class:`ConicState` at any other radius via
    :func:`speed_components_at_radius`.

    Args:
        mu: Gravitational parameter of the attractor (km^3/s^2).
        r: Radius at which the state below is given (km).
        v_tangential: Tangential speed at ``r`` (km/s), positive prograde.
        v_radial: Radial-outward speed at ``r`` (km/s).

    Returns:
        The :class:`ConicState`.
    """
    energy = (v_tangential * v_tangential + v_radial * v_radial) / 2.0 - mu / r
    h = r * v_tangential
    p = h * h / mu
    ecc = eccentricity_from_energy_and_h(mu, energy, h)
    return ConicState(energy=energy, h=h, p=p, ecc=ecc)


def speed_components_at_radius(
    state: ConicState, mu: float, r: float
) -> Tuple[float, float]:
    """Tangential and (unsigned) radial speed at radius ``r`` on a conic.

    Vis-viva from the orbit's conserved energy and angular momentum:
    ``v_tangential = h / r``, ``v_radial = sqrt(2*(energy + mu/r) - v_tangential**2)``.
    The sign of ``v_radial`` (inbound vs outbound) is not determined by the
    conic alone; the caller supplies it from context.

    Args:
        state: The conic's orbit-wide constants.
        mu: Gravitational parameter of the attractor (km^3/s^2).
        r: Radius at which to evaluate the state (km).

    Returns:
        (v_tangential km/s, |v_radial| km/s).
    """
    v_tangential = state.h / r
    v_squared = 2.0 * (state.energy + mu / r)
    v_radial = float(np.sqrt(max(0.0, v_squared - v_tangential * v_tangential)))
    return v_tangential, v_radial


def elliptic_tof_seconds(mu: float, a: float, ecc: float, nu: float) -> float:
    """Time from periapsis to true anomaly ``nu`` (rad, [0, pi]) on an ellipse.

    Float twin of :func:`orbit_utils.elliptic_time_of_flight` -- this is the
    implementation that function's Quantity interface now delegates to.

    Args:
        mu: Gravitational parameter (km^3/s^2).
        a: Semi-major axis (km).
        ecc: Eccentricity, 0 <= ecc < 1.
        nu: True anomaly from periapsis (rad, in [0, pi]).

    Returns:
        The time of flight (s).
    """
    eccentric_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(nu / 2.0),
        np.sqrt(1.0 + ecc) * np.cos(nu / 2.0),
    )
    mean_anomaly = eccentric_anomaly - ecc * np.sin(eccentric_anomaly)
    return float(mean_anomaly / np.sqrt(mu / a**3))


def hyperbolic_tof_seconds(mu: float, a_abs: float, ecc: float, nu: float) -> float:
    """Time from periapsis to true anomaly ``nu`` (rad, [0, pi)) on a hyperbola.

    Float twin of :func:`orbit_utils.hyperbolic_time_of_flight` -- this is the
    implementation that function's Quantity interface now delegates to.

    Args:
        mu: Gravitational parameter (km^3/s^2).
        a_abs: Absolute semi-major axis |a| (km).
        ecc: Eccentricity, ecc > 1.
        nu: True anomaly from periapsis (rad), within the reachable branch.

    Returns:
        The time of flight (s).
    """
    cosh_f = (ecc + np.cos(nu)) / (1.0 + ecc * np.cos(nu))
    hyperbolic_anomaly = float(np.arccosh(max(cosh_f, 1.0)))
    mean_anomaly = ecc * np.sinh(hyperbolic_anomaly) - hyperbolic_anomaly
    return float(mean_anomaly / np.sqrt(mu / a_abs**3))


def elliptic_time_from_periapsis(mu: float, a: float, ecc: float, nu: float) -> float:
    """Time from periapsis to true anomaly ``nu`` in [0, 2*pi) on an ellipse.

    Extends :func:`elliptic_tof_seconds` past pi by period symmetry.

    Args:
        mu: Gravitational parameter (km^3/s^2).
        a: Semi-major axis (km).
        ecc: Eccentricity, 0 <= ecc < 1.
        nu: True anomaly from periapsis (rad, in [0, 2*pi)).

    Returns:
        The time since periapsis passage (s).
    """
    if nu > np.pi:
        period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
        return period - elliptic_tof_seconds(mu, a, ecc, 2.0 * np.pi - nu)
    return elliptic_tof_seconds(mu, a, ecc, nu)


def true_anomaly_at_radius_rad(p: float, ecc: float, r: float) -> Optional[float]:
    """Principal true anomaly (rad, [0, pi]) where a conic reaches radius ``r``.

    Float twin of :func:`orbit_utils.true_anomaly_at_radius` -- this is the
    implementation that function's Quantity interface now delegates to.

    Args:
        p: Semi-latus rectum (km).
        ecc: Eccentricity (> 0).
        r: Radius to reach (km).

    Returns:
        The true anomaly in [0, pi], or None if the radius is unreachable.
    """
    cos_nu = (p / r - 1.0) / ecc
    if abs(cos_nu) > 1.0 + 1e-9:
        return None
    return float(np.arccos(np.clip(cos_nu, -1.0, 1.0)))


class RadiusCrossing(NamedTuple):
    """One future crossing of a target radius from a heliocentric state.

    Attributes:
        tof: Time of flight from the starting state to this crossing (s).
        v_tangential: Tangential speed at the crossing (km/s).
        v_radial: Signed radial-outward speed at the crossing (km/s).
        outbound: True if the crossing is moving away from the attractor.
        swept: Heliocentric longitude swept to reach the crossing (rad,
            positive; the starting state's motion is always prograde by
            convention, see :func:`conic_radius_crossings`).
    """

    tof: float
    v_tangential: float
    v_radial: float
    outbound: bool
    swept: float


def conic_radius_crossings(
    mu: float,
    r0: float,
    v_t0: float,
    v_r0: float,
    r1: float,
    min_leg_time: float = 0.0,
) -> List[RadiusCrossing]:
    """Future crossings of radius ``r1`` from a heliocentric state at ``r0``.

    The state is (tangential, radial-outward) at radius ``r0`` with positive
    angular momentum. Within one revolution (ellipse) or on the remaining
    branch (hyperbola) the orbit crosses ``r1`` at most twice: once moving
    outward and once moving inward.

    Args:
        mu: Gravitational parameter of the attractor (km^3/s^2).
        r0: Current radius (km).
        v_t0: Tangential speed, > 0 for prograde (km/s).
        v_r0: Radial-outward speed (km/s).
        r1: Target radius (km).
        min_leg_time: Crossings reached in less than this many seconds are
            dropped (a caller's degenerate-leg filter; the kernel itself
            imposes no minimum). Defaults to no filtering.

    Returns:
        Up to two :class:`RadiusCrossing` records. The swept angle is what a
        *phased* chain needs: paired with the leg time it says where the craft
        arrives, not merely when, so an arrival can be matched against where
        the target body will actually be. Motion is prograde (h > 0), so it is
        positive.
    """
    state = conic_state_at_radius(mu, r0, v_t0, v_r0)
    h = state.h
    if h <= 0.0:
        return []
    ecc = state.ecc
    if ecc < 1e-9 or abs(ecc - 1.0) < 1e-9:
        return []
    nu0 = true_anomaly_at_radius_rad(state.p, ecc, r0)
    nu1 = true_anomaly_at_radius_rad(state.p, ecc, r1)
    if nu0 is None or nu1 is None:
        return []
    if v_r0 < 0.0:
        nu0 = -nu0
    v_t1, v_r1 = speed_components_at_radius(state, mu, r1)
    crossings: List[RadiusCrossing] = []
    if ecc < 1.0:
        a = state.p / (1.0 - ecc * ecc)
        period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
        t0 = elliptic_time_from_periapsis(mu, a, ecc, nu0 % (2.0 * np.pi))
        for nu_target, v_r_signed, outbound in (
            (nu1, v_r1, True),
            (2.0 * np.pi - nu1, -v_r1, False),
        ):
            dt = (elliptic_time_from_periapsis(mu, a, ecc, nu_target) - t0) % period
            if dt > min_leg_time:
                swept = (nu_target - nu0) % (2.0 * np.pi)
                crossings.append(RadiusCrossing(dt, v_t1, v_r_signed, outbound, swept))
    else:
        a_abs = state.p / (ecc * ecc - 1.0)
        t0 = float(np.sign(nu0)) * hyperbolic_tof_seconds(mu, a_abs, ecc, abs(nu0))
        for nu_target, v_r_signed, outbound in (
            (nu1, v_r1, True),
            (-nu1, -v_r1, False),
        ):
            dt = (
                float(np.sign(nu_target))
                * hyperbolic_tof_seconds(mu, a_abs, ecc, abs(nu_target))
                - t0
            )
            if dt > min_leg_time:
                # A hyperbola never wraps, so the sweep is the bare difference;
                # a positive dt guarantees nu_target is ahead of nu0.
                swept = nu_target - nu0
                crossings.append(RadiusCrossing(dt, v_t1, v_r_signed, outbound, swept))
    return crossings


def mean_motion(v_circ: float, orbit_radius: float) -> float:
    """Mean motion of a circular orbit (rad/s), ``v_circ / orbit_radius``.

    Args:
        v_circ: Circular speed (km/s).
        orbit_radius: Circular orbit radius (km).

    Returns:
        Angular rate, positive (prograde).
    """
    return v_circ / orbit_radius


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi).

    Args:
        angle: Angle in radians.

    Returns:
        The equivalent angle in [-pi, pi). Odd multiples of pi resolve to
        -pi (the branch cut), never +pi.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
