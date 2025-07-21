import numpy as np
from astropy import units as u
from poliastro.bodies import Body
from poliastro.twobody import Orbit

STD_FUDGE_FACTOR: float = 0.8


def body_speed(body: Body, altitude: u.Quantity) -> u.Quantity:
    """Compute the orbital speed at a given altitude above a body's surface.

    Args:
        body: A poliastro Body instance
        altitude: Altitude above the body's surface (astropy Quantity)

    Returns:
        The orbital speed at the given altitude (astropy Quantity, m/s)
    """
    orbit: Orbit = Orbit.circular(body, altitude)
    velocity_vector = orbit.rv()[1]
    speed = np.linalg.norm(velocity_vector.value) * velocity_vector.unit
    return speed.to(u.m / u.s)


def balloon_mass(
    v_rf: u.Quantity,
    v_b: u.Quantity,
    v_ri: u.Quantity = 0 * u.km / u.s,
    fudge_factor: float = STD_FUDGE_FACTOR,
) -> float:

    denom: float = np.log((v_b - v_ri) / (v_b - v_rf)).item()
    return 2 * fudge_factor / denom
