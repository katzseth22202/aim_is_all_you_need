from typing import List, Tuple

import numpy as np
from astropy import units as u
from poliastro.bodies import Body, Sun
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit

STD_FUDGE_FACTOR: float = 0.8


def get_burn(impulse: Tuple[np.ndarray, np.ndarray]) -> u.Quantity:
    return np.linalg.norm(impulse[1])


def get_hohmann_burns(h: Maneuver) -> List[u.Quantity]:
    i_1, i_2 = h.impulses
    return [get_burn(i_1), get_burn(i_2)]


def hohmann_transfer(
    r_i: u.Quantity, r_f: u.Quantity, attractor: Body = Sun
) -> Maneuver:
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
    orbit: Orbit = Orbit.circular(attractor, a - attractor.R)
    _, velocity_vector = orbit.rv()
    speed = np.linalg.norm(velocity_vector.value) * velocity_vector.unit
    return speed.to(u.km / u.s)


def balloon_mass(
    v_rf: u.Quantity,
    v_b: u.Quantity,
    v_ri: u.Quantity = 0 * u.km / u.s,
    fudge_factor: float = STD_FUDGE_FACTOR,
) -> float:

    denom: float = np.log((v_b - v_ri) / (v_b - v_rf)).item()
    return 2 * fudge_factor / denom
