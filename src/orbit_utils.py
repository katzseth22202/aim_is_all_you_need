import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from poliastro.bodies import Body, Earth, Moon, Saturn, Sun
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit

from src.astro_constants import (
    EARTH_A,
    EFFECTIVE_DV_LUNAR,
    JUPITER_A,
    LEO_ALTITUDE,
    LOW_SATURN_ALTITUDE,
    MOON_A,
    PARKER_PERIAPSIS,
    PERIAPSIS_SOLAR_BURN,
    PERIAPSIS_SOLAR_V,
    PHOEBE_A,
    REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
    REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
    RETROGRADE_FRACTION,
    TARGET_LAUNCH_CAPACITY_MULTIPLE,
)


def body_speed(body: Body, altitude: u.Quantity) -> u.Quantity:
    """Compute the orbital speed at a given altitude above a body's surface.

    Args:
        body: A poliastro Body instance
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
        attractor: The central body (poliastro Body, default Sun).

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
        body: A poliastro Body instance.
        altitude: Altitude above the body's surface (astropy Quantity, default 0 km).

    Returns:
        The escape velocity at the given altitude (astropy Quantity, km/s).
    """
    # Distance from the center of the body
    r: u.Quantity = body.R + altitude
    v_esc: u.Quantity = np.sqrt(2 * body.k / r)
    return v_esc.to(u.km / u.s)


def get_period(body: Body, a: u.Quantity) -> u.Quantity:
    """Compute the orbital period for a given semi-major axis around a body.

    Args:
        body: A poliastro Body instance.
        a: Semi-major axis (astropy Quantity).

    Returns:
        The orbital period (astropy Quantity, seconds).
    """
    T = (2 * np.pi / np.sqrt(body.k)) * (a**1.5)
    return T.to(u.second)


def get_semimajor_axis(body: Body, T: u.Quantity) -> u.Quantity:
    """Compute the semi-major axis for a given orbital period around a body.

    Args:
        body: A poliastro Body instance.
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
        body: The celestial body (poliastro Body).

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
    Generates a poliastro Orbit object aligned with the y-axis (periapsis on +y)
    and no z-component of motion (orbit in the XY-plane).

    Parameters
    ----------
       apoapsis_radius : astropy.units.Quantity
        The radius of the apoapsis (farthest point from the attractor).
        Must be an astropy Quantity with units of length (e.g., 10000 * u.km).
    periapsis_radius : astropy.units.Quantity
        The radius of the periapsis (closest point to the attractor).
        Must be an astropy Quantity with units of length (e.g., 6678 * u.km).
    attractor_body : poliastro.bodies.Body
        The central celestial body (e.g., Earth, Sun, Mars).

    Returns
    -------
    poliastro.twobody.Orbit
        The generated poliastro Orbit object.

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


def periapsis_velocity(orbit: Orbit) -> u.Quantity:
    """Return the velocity vector at periapsis for a given poliastro Orbit.

    Args:
        orbit: A poliastro Orbit object.

    Returns:
        The velocity vector at periapsis (astropy Quantity, km/s).
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


def apoapsis_velocity(orbit: Orbit) -> u.Quantity:
    """Return the velocity vector at apoapsis for a given poliastro Orbit.

    Args:
        orbit: A poliastro Orbit object.

    Returns:
        The velocity vector at apoapsis (astropy Quantity, km/s).
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


def velocity_at_distance(
    radius_periapsis: u.Quantity,
    velocity_periapsis: u.Quantity,
    distance: u.Quantity,
    attractor_body: Body = Sun,
) -> u.Quantity:
    """
    Compute the scalar orbital velocity at a given distance from the central body, given the periapsis radius and velocity.

    Parameters
    ----------
    radius_periapsis : astropy.units.Quantity
        The radius at periapsis (with length units).
    velocity_periapsis : astropy.units.Quantity
        The scalar velocity at periapsis (with velocity units).
    distance : astropy.units.Quantity
        The distance from the center of the attractor at which to compute the velocity (with length units).
    attractor_body : poliastro.bodies.Body
        The central celestial body (e.g., Earth, Sun).

    Returns
    -------
    astropy.units.Quantity
        The scalar orbital velocity at the given distance (with velocity units).

    Raises
    ------
    ValueError
        If the computed velocity is not real (e.g., for unphysical parameters).
    """
    mu = attractor_body.k
    r_p = radius_periapsis
    v_p = velocity_periapsis
    # Compute semi-major axis from vis-viva at periapsis
    # v_p^2 = mu * (2/r_p - 1/a)  => 1/a = 2/r_p - v_p^2/mu
    one_over_a = 2 / r_p - v_p**2 / mu

    # Handle parabolic orbit case (1/a ≈ 0)
    if np.isclose(one_over_a, 0, atol=1e-15):
        # For parabolic orbit: v = sqrt(2*mu/r)
        v2 = 2 * mu / distance
    else:
        a = 1 / one_over_a
        # Now compute velocity at the given distance
        v2 = mu * (2 / distance - 1 / a)
    if v2 < 0 * v2.unit:
        raise ValueError("No real velocity at this distance for the given orbit.")
    return np.sqrt(v2).to(u.km / u.s)


def as_scalar(vec: npt.ArrayLike) -> u.Quantity:
    """Return the norm of a vector as an astropy Quantity in km/s if possible, else as a float."""
    norm = np.linalg.norm(vec)
    # If input is an astropy Quantity, norm will be a Quantity; else, it's a float
    if isinstance(norm, u.Quantity):
        return norm.to(u.km / u.s)
    else:
        return norm * u.dimensionless_unscaled


def find_periapsis_radius_from_apoapsis_and_velocity(
    apoapsis_radius: u.Quantity,
    periapsis_velocity: u.Quantity,
    attractor_body: Body = Sun,
) -> u.Quantity:
    """
    Compute the periapsis radius of an orbit given the apoapsis radius, the scalar velocity at periapsis, and the central attractor.

    The function solves the vis-viva equation for the periapsis radius, assuming an elliptical orbit
    aligned with the y-axis and no z-component (orbit in the XY-plane).

    Parameters
    ----------
    apoapsis_radius : astropy.units.Quantity
        The radius of the apoapsis (farthest point from the attractor), with length units.
    periapsis_velocity : astropy.units.Quantity
        The scalar velocity at periapsis, with velocity units.
    attractor_body : poliastro.bodies.Body, optional
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
    A = periapsis_velocity**2
    B = periapsis_velocity**2 * apoapsis_radius
    C = -2 * attractor_body.k * apoapsis_radius

    # Calculate the discriminant
    discriminant = B**2 - 4 * A * C

    # Ensure the discriminant is non-negative for a real solution
    if discriminant < 0:
        raise ValueError(
            "Invalid parameters: No real solution for periapsis radius. Check input values."
        )

    # Calculate the two possible solutions for periapsis radius
    rp1 = (-B + np.sqrt(discriminant)) / (2 * A)
    rp2 = (-B - np.sqrt(discriminant)) / (2 * A)
    rp1 = rp1.to(u.km)
    rp2 = rp2.to(u.km)

    # In orbital mechanics, radius must be positive.
    # We take the positive root. If both are positive, the problem context implies a physically
    # meaningful solution. In this case, the larger velocity at periapsis implies a smaller
    # periapsis radius, so we take the positive root.
    if rp1 > 0 and rp2 > 0:
        # Both roots are positive. Choose the one that makes physical sense.
        # Since Vp is given as the periapsis velocity, it implies rp < ra.
        # The equation derived earlier directly yields the correct physical radius.
        return rp1 if rp1 < apoapsis_radius else rp2
    elif rp1 > 0:
        return rp1
    elif rp2 > 0:
        return rp2
    else:
        raise ValueError("No positive solution for periapsis radius found.")


def get_orbital_velocity_at_radius(orbit: Orbit, radius: u.Quantity) -> u.Quantity:
    """
    Calculates the scalar orbital velocity at a given radial distance from the attractor body.

    Parameters
    ----------
    orbit : poliastro.twobody.orbit.Orbit
        The poliastro Orbit object, containing the orbital elements and attractor body.
    radius : astropy.units.Quantity
        The radial distance from the attractor body at which to calculate the velocity.
        Must be a scalar astropy Quantity with units of length.

    Returns
    -------
    astropy.units.Quantity
        The scalar orbital velocity at the given radius, with units of velocity.

    Raises
    ------
    ValueError
        If the provided radius is outside the valid range for the given orbit
        (e.g., negative for elliptical/parabolic/hyperbolic orbits, or
        greater than apoapsis for elliptical orbits if not handled correctly
        for specific velocity calculations).
    """
    if not isinstance(radius, u.Quantity) or not radius.unit.is_equivalent(u.km):
        raise TypeError("Radius must be an astropy Quantity with units of length.")
    if radius.size != 1:
        raise ValueError("Radius must be a scalar quantity.")

    # Gravitational parameter of the attractor body
    mu = orbit.attractor.k

    # Semimajor axis
    a = orbit.a

    # Specific energy (epsilon) for the orbit
    # epsilon = -mu / (2 * a)

    # Velocity formula for any conic section (vis-viva equation):
    # v = sqrt(mu * (2/r - 1/a))
    # For parabolic orbit, a approaches infinity, so 1/a approaches 0.
    # For hyperbolic orbit, a is negative, so 1/a is also negative.

    # Check for valid radius range depending on orbit type
    if orbit.ecc < 1:  # Elliptical orbit
        if radius < orbit.r_p or radius > orbit.r_a:
            # For an elliptical orbit, the radius must be between periapsis and apoapsis.
            # However, the Vis-Viva equation itself doesn't strictly break outside this range;
            # it would just give an imaginary velocity, indicating an invalid physical point.
            # We can allow the calculation as long as 2/r - 1/a > 0.
            pass  # The formula will handle it by returning a NaN if the sqrt is of a negative number
    elif orbit.ecc == 1:  # Parabolic orbit
        if radius < 0 * u.km:  # Radius must be non-negative
            raise ValueError("Radius cannot be negative for a parabolic orbit.")
        # For parabolic orbit, 1/a term becomes 0, Vis-Viva simplifies to sqrt(2*mu/r)
        # However, the general formula still works as a approaches infinity.
        pass
    else:  # Hyperbolic orbit (ecc > 1)
        if radius < 0 * u.km:  # Radius must be non-negative
            raise ValueError("Radius cannot be negative for a hyperbolic orbit.")
        # For hyperbolic orbits, 'a' is negative, but the Vis-Viva equation accounts for this.
        pass

    # Vis-viva equation
    # Make sure units are consistent before calculation
    velocity_squared = mu * (2 / radius - 1 / a)

    # Ensure the term inside the square root is non-negative
    if velocity_squared.value < 0:
        raise ValueError(
            f"The provided radius ({radius}) is not physically reachable for this orbit. "
            "Velocity would be imaginary. Ensure radius is within the orbit's bounds."
        )

    velocity = np.sqrt(velocity_squared)

    return velocity


def retrograde_orbit(orbit: Orbit) -> Orbit:
    """Return a new Orbit with the same shape as the input but with retrograde velocity.

    Args:
        orbit: A poliastro Orbit object.

    Returns:
        A new Orbit object with the same position but velocity reversed (retrograde).
    """
    r_vec, v_vec = orbit.rv()
    retrograde_v_vec = -v_vec
    return Orbit.from_vectors(
        orbit.attractor, r_vec, retrograde_v_vec, epoch=orbit.epoch
    )
