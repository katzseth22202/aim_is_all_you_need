import re
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from poliastro.bodies import Body, Earth, Jupiter, Sun
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit

from src.astro_constants import EARTH_A, JUPITER_A

STD_FUDGE_FACTOR: float = 0.8


def get_burn(impulse: Tuple[npt.ArrayLike, npt.ArrayLike]) -> u.Quantity:
    """Compute the magnitude of the second impulse vector in a maneuver.

    Args:
        impulse: Tuple containing two impulse vectors (numpy arrays).

    Returns:
        The magnitude of the second impulse (astropy Quantity).
    """
    return as_scalar(impulse[1])


def get_hohmann_burns(h: Maneuver) -> List[u.Quantity]:
    """Compute the burn magnitudes for a Hohmann transfer maneuver.

    Args:
        h: A Maneuver object representing a Hohmann transfer.

    Returns:
        A list containing the magnitudes of the two burns (astropy Quantities).
    """
    i_1, i_2 = h.impulses
    return [get_burn(i_1), get_burn(i_2)]


def hohmann_transfer(
    r_i: u.Quantity, r_f: u.Quantity, attractor: Body = Sun
) -> Maneuver:
    """Compute the Hohmann transfer maneuver between two circular orbits.

    Args:
        r_i: Initial orbit radius (astropy Quantity).
        r_f: Final orbit radius (astropy Quantity).
        attractor: The central body (poliastro Body, default Sun).

    Returns:
        A Maneuver object representing the Hohmann transfer.
    """
    initial_orbit: Orbit = Orbit.circular(attractor, r_i)
    return Maneuver.hohmann(initial_orbit, r_f)


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


def payload_mass_ratio(
    v_rf: u.Quantity,
    v_b: u.Quantity,
    v_ri: u.Quantity = 0 * u.km / u.s,
    fudge_factor: float = STD_FUDGE_FACTOR,
) -> float:
    """Compute the ratio of payload mass to balloon propulsion mass.
    Args:
        v_rf: Final velocity of the rocket (astropy Quantity, km/s).
        v_b: Velocity of the balloon  (astropy Quantity, km/s).
        v_ri: Initial velocity of the rocket (astropy Quantity, km/s, default 0 km/s).
        fudge_factor: Fudge factor for the mass calculation (default 0.8).

    Returns:
        The payload mass to balloon propulsion mass ratio.
    """

    denom: float = np.log((v_b - v_ri) / (v_b - v_rf)).item()
    return 2 * fudge_factor / denom


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


def retrograde_jovian_hohmann_transfer() -> u.Quantity:
    """Compute the reverse Hohmann transfer maneuver from Jupiter to Earth.

    Args:
        None

    Returns:
        the Earth crossing velocity in km/s
    """

    prograde_hohmann_speed = get_hohmann_burns(hohmann_transfer(JUPITER_A, EARTH_A))[1]
    earth_speed = speed_around_attractor(EARTH_A)
    # since we are retrograde, we add twice the Earth's speed
    retrograde_speed = prograde_hohmann_speed + 2 * earth_speed
    # add the poential energy of the Earth's orbit, which equals the kinetic energy of Earth's escape velocity
    earth_escape_velocity = escape_velocity(Earth)
    return np.sqrt(retrograde_speed**2 + earth_escape_velocity**2)


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


def distance_to_center(altitute: u.Quantity, body: Body) -> u.Quantity:
    """Compute the distance from the center of a body given altitude and body radius.

    Args:
        altitute: Altitude above the body's surface (astropy Quantity).
        body_radius: Radius of the body (astropy Quantity).

    Returns:
        The distance from the center of the body (astropy Quantity, km).
    """
    return (body.R + altitute).to(u.km)


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


def as_scalar(vec: npt.ArrayLike) -> u.Quantity:
    """Return the norm of a vector as an astropy Quantity in km/s if possible, else as a float."""
    norm = np.linalg.norm(vec)
    # If input is an astropy Quantity, norm will be a Quantity; else, it's a float
    if isinstance(norm, u.Quantity):
        return norm.to(u.km / u.s)
    else:
        return norm * u.dimensionless_unscaled


def orbit_from_radius_velocity(
    radius: u.Quantity,
    velocity: u.Quantity,
    attractor_body: Body = Sun,
) -> Orbit:
    """Generate a poliastro Orbit aligned with the y-axis, given a scalar radius and velocity.

    The position is set to (0, +radius, 0) and the velocity to (+velocity, 0, 0),
    so the orbit is in the XY-plane and periapsis is on the +y axis.

    Args:
        radius: Distance from the attractor (astropy Quantity, length units).
        velocity: Velocity magnitude at that position (astropy Quantity, speed units).
        attractor_body: The central body (poliastro Body, default Sun).

    Returns:
        A poliastro Orbit object with the specified state.
    """
    r_vec = np.array([0.0, radius.to_value(u.km), 0.0]) * u.km
    v_vec = np.array([velocity.to_value(u.km / u.s), 0.0, 0.0]) * (u.km / u.s)
    return Orbit.from_vectors(attractor_body, r_vec, v_vec)


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
    return Orbit.from_vectors(orbit.attractor, r_vec, retrograde_v_vec, epoch=orbit.epoch)
