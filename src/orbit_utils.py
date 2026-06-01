"""Orbital mechanics utilities and calculations.

This module provides fundamental orbital mechanics functions for calculating
orbital parameters, velocities, and trajectories. It serves as the foundation
for orbital calculations used throughout the PuffSat propulsion system.

Key Functions:
    - body_speed: Calculate orbital speed at given altitude
    - escape_velocity: Calculate escape velocity from celestial bodies
    - orbit_from_rp_ra: Create orbits from periapsis/apoapsis radii
    - orbit_from_periapsis_speed_and_apoapsis_radius: Create orbits from a
      periapsis speed and apoapsis radius
    - periapsis_speed/apoapsis_speed: Extract speed at orbit extremes
    - speed_at_distance: Calculate speed at any point in orbit

Dependencies:
    - astro_constants: Physical constants and orbital parameters
    - boinor: Core orbital mechanics library
    - astropy: Units and astronomical calculations

This module is designed to be imported by propulsion.py and scenario.py
for higher-level orbital maneuver calculations.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from boinor.bodies import Body, Earth, Moon, Saturn, Sun
from boinor.maneuver import Maneuver
from boinor.twobody import Orbit


def body_speed(body: Body, altitude: u.Quantity) -> u.Quantity:
    """Compute the orbital speed at a given altitude above a body's surface.

    Args:
        body: A boinor Body instance
        altitude: Altitude above the body's surface (astropy Quantity)

    Returns:
        The orbital speed at the given altitude (astropy Quantity, m/s)
    """

    orbit: Orbit = Orbit.circular(body, altitude)
    _, velocity_vector = orbit.rv()
    speed = np.linalg.norm(velocity_vector.value) * velocity_vector.unit
    return speed.to(u.km / u.s)


def speed_around_attractor(a: u.Quantity, attractor: Body = Sun) -> u.Quantity:
    """Compute the orbital speed at a given altitude above an attractor's surface.

    Args:
        a: Altitude above the attractor's surface (astropy Quantity).
        attractor: The central body (boinor Body, default Sun).

    Returns:
        The orbital speed at the given altitude (astropy Quantity, km/s).
    """
    orbit: Orbit = Orbit.circular(attractor, a - attractor.R)
    _, velocity_vector = orbit.rv()
    speed = np.linalg.norm(velocity_vector.value) * velocity_vector.unit
    return speed.to(u.km / u.s)


def escape_velocity(body: Body, altitude: u.Quantity = 0 * u.km) -> u.Quantity:
    """Compute the escape velocity from a body's surface or at a given altitude.

    Args:
        body: A boinor Body instance.
        altitude: Altitude above the body's surface (astropy Quantity, default 0 km).

    Returns:
        The escape velocity at the given altitude (astropy Quantity, km/s).
    """
    # Distance from the center of the body
    r: u.Quantity = body.R + altitude
    v_esc: u.Quantity = np.sqrt(2 * body.k / r)
    return v_esc.to(u.km / u.s)


def speed_with_escape_energy(
    v_infinity: u.Quantity, body: Body, altitude: u.Quantity = 0 * u.km
) -> u.Quantity:
    """Local two-body speed v(r) = sqrt(v_infinity**2 + v_esc(r)**2).

    Vis-viva for a hyperbolic orbit: the speed at radius ``r`` (the body's
    surface plus ``altitude``) given the hyperbolic-excess speed ``v_infinity``
    far from the body. The escape energy at ``r`` and the excess kinetic energy
    add in quadrature.

    The caller is responsible for supplying a physically appropriate
    ``v_infinity``; it may itself be an approximation (e.g. a heliocentric or
    geocentric speed treated as body-relative excess).

    Args:
        v_infinity: Hyperbolic-excess speed far from the body (astropy Quantity).
        body: A boinor Body instance.
        altitude: Altitude above the body's surface (astropy Quantity, default 0 km).

    Returns:
        The local speed at the given altitude (astropy Quantity, km/s).
    """
    v_esc: u.Quantity = escape_velocity(body, altitude)
    return np.sqrt(v_infinity**2 + v_esc**2).to(u.km / u.s)


def get_period(body: Body, a: u.Quantity) -> u.Quantity:
    """Compute the orbital period for a given semi-major axis around a body.

    Args:
        body: A boinor Body instance.
        a: Semi-major axis (astropy Quantity).

    Returns:
        The orbital period (astropy Quantity, seconds).
    """
    T = (2 * np.pi / np.sqrt(body.k)) * (a**1.5)
    return T.to(u.second)


def get_semimajor_axis(body: Body, T: u.Quantity) -> u.Quantity:
    """Compute the semi-major axis for a given orbital period around a body.

    Args:
        body: A boinor Body instance.
        T: Orbital period (astropy Quantity).

    Returns:
        The semi-major axis (astropy Quantity, km).
    """
    a_cubed = (T**2 * body.k) / (4 * np.pi**2)
    a = a_cubed ** (1 / 3)
    return a.to(u.km)


def distance_to_center(altitude: u.Quantity, body: Body) -> u.Quantity:
    """Compute the distance from the center of a body given altitude and body radius.

    Args:
        altitude: Altitude above the body's surface (astropy Quantity).
        body: The celestial body (boinor Body).

    Returns:
        The distance from the center of the body (astropy Quantity, km).
    """
    return (body.R + altitude).to(u.km)


def orbit_from_rp_ra(
    apoapsis_radius: u.Quantity,
    periapsis_radius: u.Quantity,
    attractor_body: Body = Sun,
) -> Orbit:
    """
    Generates a boinor Orbit object aligned with the y-axis (periapsis on +y)
    and no z-component of motion (orbit in the XY-plane).

    Parameters
    ----------
       apoapsis_radius : astropy.units.Quantity
        The radius of the apoapsis (farthest point from the attractor).
        Must be an astropy Quantity with units of length (e.g., 10000 * u.km).
    periapsis_radius : astropy.units.Quantity
        The radius of the periapsis (closest point to the attractor).
        Must be an astropy Quantity with units of length (e.g., 6678 * u.km).
    attractor_body : boinor.bodies.Body
        The central celestial body (e.g., Earth, Sun, Mars).

    Returns
    -------
    boinor.twobody.Orbit
        The generated boinor Orbit object.

    Raises
    ------
    ValueError
        If periapsis_radius is greater than or equal to apoapsis_radius.
    """

    if periapsis_radius >= apoapsis_radius:
        raise ValueError(
            "Periapsis radius must be less than apoapsis radius for a valid elliptical orbit."
        )

    # Calculate semi-major axis (a)
    semimajor_axis: u.Quantity = (apoapsis_radius + periapsis_radius) / 2

    # Calculate eccentricity (ecc)
    eccentricity: float = (apoapsis_radius - periapsis_radius) / (
        apoapsis_radius + periapsis_radius
    )

    # Set classical orbital elements for desired alignment
    # Inclination: 0 degrees for orbit in the XY-plane (no Z component)
    inclination: u.Quantity = 0 * u.deg
    # Right Ascension of the Ascending Node (RAAN): 0 degrees
    raan: u.Quantity = 0 * u.deg
    # Argument of Periapsis: 90 degrees to align periapsis with the positive Y-axis
    argp: u.Quantity = 90 * u.deg
    # True Anomaly: 0 degrees to start at periapsis
    true_anomaly: u.Quantity = 0 * u.deg

    # Create the Orbit object
    orbit: Orbit = Orbit.from_classical(
        attractor_body,
        semimajor_axis,
        eccentricity,
        inclination,
        raan,
        argp,
        true_anomaly,
    )

    return orbit


def periapsis_speed(orbit: Orbit) -> u.Quantity:
    """Return the scalar speed at periapsis for a given boinor Orbit.

    Args:
        orbit: A boinor Orbit object.

    Returns:
        The scalar speed at periapsis (astropy Quantity, km/s).
    """
    # Create a new orbit at true anomaly = 0 deg (periapsis)
    orbit_at_periapsis = Orbit.from_classical(
        orbit.attractor,
        orbit.a,
        orbit.ecc,
        orbit.inc,
        orbit.raan,
        orbit.argp,
        0 * u.deg,
    )
    _, v_vec = orbit_at_periapsis.rv()
    return as_scalar(v_vec)


def apoapsis_speed(orbit: Orbit) -> u.Quantity:
    """Return the scalar speed at apoapsis for a given boinor Orbit.

    Args:
        orbit: A boinor Orbit object.

    Returns:
        The scalar speed at apoapsis (astropy Quantity, km/s).
    """
    # Create a new orbit at true anomaly = 180 deg (apoapsis)
    orbit_at_apoapsis = Orbit.from_classical(
        orbit.attractor,
        orbit.a,
        orbit.ecc,
        orbit.inc,
        orbit.raan,
        orbit.argp,
        180 * u.deg,
    )
    _, v_vec = orbit_at_apoapsis.rv()
    return as_scalar(v_vec)


def speed_at_distance(
    radius_periapsis: u.Quantity,
    periapsis_speed: u.Quantity,
    distance: u.Quantity,
    attractor_body: Body = Sun,
) -> u.Quantity:
    """
    Compute the scalar orbital speed at a given distance from the central body, given the periapsis radius and speed.

    Parameters
    ----------
    radius_periapsis : astropy.units.Quantity
        The radius at periapsis (with length units).
    periapsis_speed : astropy.units.Quantity
        The scalar speed at periapsis (with velocity units).
    distance : astropy.units.Quantity
        The distance from the center of the attractor at which to compute the speed (with length units).
    attractor_body : boinor.bodies.Body
        The central celestial body (e.g., Earth, Sun).

    Returns
    -------
    astropy.units.Quantity
        The scalar orbital speed at the given distance (with velocity units).

    Raises
    ------
    ValueError
        If the computed speed is not real (e.g., for unphysical parameters).
    """
    mu = attractor_body.k
    r_p = radius_periapsis
    v_p = periapsis_speed
    # Compute semi-major axis from vis-viva at periapsis
    # v_p^2 = mu * (2/r_p - 1/a)  => 1/a = 2/r_p - v_p^2/mu
    one_over_a = 2 / r_p - v_p**2 / mu

    # Handle parabolic orbit case (1/a ≈ 0)
    if np.isclose(one_over_a, 0, atol=1e-15):
        # For parabolic orbit: v = sqrt(2*mu/r)
        v2 = 2 * mu / distance
    else:
        a = 1 / one_over_a
        # Now compute speed at the given distance
        v2 = mu * (2 / distance - 1 / a)
    if v2 < 0 * v2.unit:
        raise ValueError("No real speed at this distance for the given orbit.")
    return np.sqrt(v2).to(u.km / u.s)


def as_scalar(vec: npt.ArrayLike) -> u.Quantity:
    """Return the norm of a vector as an astropy Quantity in km/s if possible, else as a float."""
    norm = np.linalg.norm(vec)
    # If input is an astropy Quantity, norm will be a Quantity; else, it's a float
    if isinstance(norm, u.Quantity):
        return norm.to(u.km / u.s)
    else:
        return norm * u.dimensionless_unscaled


def find_periapsis_radius_from_apoapsis_and_speed(
    apoapsis_radius: u.Quantity,
    periapsis_speed: u.Quantity,
    attractor_body: Body = Sun,
) -> u.Quantity:
    """
    Compute the periapsis radius of an orbit given the apoapsis radius, the scalar speed at periapsis, and the central attractor.

    The function solves the vis-viva equation for the periapsis radius, assuming an elliptical orbit
    aligned with the y-axis and no z-component (orbit in the XY-plane).

    Parameters
    ----------
    apoapsis_radius : astropy.units.Quantity
        The radius of the apoapsis (farthest point from the attractor), with length units.
    periapsis_speed : astropy.units.Quantity
        The scalar speed at periapsis, with velocity units.
    attractor_body : boinor.bodies.Body, optional
        The central celestial body (default: Sun).

    Returns
    -------
    astropy.units.Quantity
        The computed periapsis radius (with length units).

    Raises
    ------
    ValueError
        If the input parameters do not yield a real, positive periapsis radius.
    """
    # The quadratic equation is: A * rp^2 + B * rp + C = 0
    A = periapsis_speed**2
    B = periapsis_speed**2 * apoapsis_radius
    C = -2 * attractor_body.k * apoapsis_radius

    # Calculate the discriminant
    discriminant = B**2 - 4 * A * C

    # Ensure the discriminant is non-negative for a real solution
    if discriminant < 0:
        raise ValueError(
            "Invalid parameters: No real solution for periapsis radius. Check input values."
        )

    # With A = v_p**2 > 0, B = v_p**2 * r_a > 0 and C = -2*mu*r_a < 0, the
    # discriminant exceeds B**2, so the '+' root is strictly positive and the
    # '-' root is strictly negative. The physical periapsis radius is the '+'
    # root.
    periapsis_radius = ((-B + np.sqrt(discriminant)) / (2 * A)).to(u.km)
    if periapsis_radius <= 0 * u.km:
        raise ValueError("No positive solution for periapsis radius found.")
    return periapsis_radius


def orbit_from_periapsis_speed_and_apoapsis_radius(
    periapsis_speed: u.Quantity,
    apoapsis_radius: u.Quantity,
    attractor_body: Body = Sun,
) -> Orbit:
    """Generate a boinor Orbit from the periapsis speed and apoapsis radius.

    The orbit is aligned with the y-axis (periapsis on +y) and lies in the
    XY-plane, matching :func:`orbit_from_rp_ra`. The periapsis radius is solved
    from the periapsis speed and apoapsis radius via
    :func:`find_periapsis_radius_from_apoapsis_and_speed`, then the orbit is
    built with :func:`orbit_from_rp_ra` -- so the vis-viva quadratic lives in
    exactly one place.

    Args:
        periapsis_speed: The scalar speed at periapsis (astropy Quantity,
            velocity units).
        apoapsis_radius: The radius of the apoapsis (astropy Quantity, length
            units).
        attractor_body: The central celestial body (boinor Body, default Sun).

    Returns:
        The generated boinor Orbit object.

    Raises:
        ValueError: If no physically valid periapsis radius exists for the given
            parameters, or if it is not less than the apoapsis radius.
    """
    periapsis_radius = find_periapsis_radius_from_apoapsis_and_speed(
        apoapsis_radius=apoapsis_radius,
        periapsis_speed=periapsis_speed,
        attractor_body=attractor_body,
    )
    return orbit_from_rp_ra(
        apoapsis_radius=apoapsis_radius,
        periapsis_radius=periapsis_radius,
        attractor_body=attractor_body,
    )
