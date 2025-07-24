import re
from typing import List, Tuple

import numpy as np
from astropy import units as u
from poliastro.bodies import Body, Earth, Jupiter, Sun
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit

from src.astro_constants import EARTH_A, JUPITER_A

STD_FUDGE_FACTOR: float = 0.8


def get_burn(impulse: Tuple[np.ndarray, np.ndarray]) -> u.Quantity:
    """Compute the magnitude of the second impulse vector in a maneuver.

    Args:
        impulse: Tuple containing two impulse vectors (numpy arrays).

    Returns:
        The magnitude of the second impulse (astropy Quantity).
    """
    return np.linalg.norm(impulse[1])


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
