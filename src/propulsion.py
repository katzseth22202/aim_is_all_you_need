"""Propulsion-related functions and calculations.

This module provides rocket propulsion calculations and maneuver planning
for the PuffSat propulsion system. It builds on orbital mechanics from
orbit_utils.py to provide higher-level propulsion analysis.

Key Functions:
    - rocket_equation: Tsiolkovsky rocket equation for propellant mass
    - payload_mass_ratio: Calculate payload/PuffSat mass ratios
    - hohmann_transfer: Plan Hohmann transfer maneuvers
    - burn_for_v_infinity: Calculate burns for hyperbolic trajectories
    - retrograde_jovian_hohmann_transfer: Specialized Jupiter-Earth transfer

Dependencies:
    - orbit_utils: Orbital mechanics calculations
    - astro_constants: Physical constants and parameters

This module is designed to be imported by scenario.py for scenario analysis
and by main.py for high-level propulsion calculations.
"""

from typing import List, Tuple

import astropy.units as u
import numpy as np
import numpy.typing as npt
from poliastro.bodies import Body, Earth, Sun
from poliastro.maneuver import Maneuver
from poliastro.twobody import Orbit

from src.astro_constants import EARTH_A, JUPITER_A, LEO_ALTITUDE, STD_FUDGE_FACTOR
from src.orbit_utils import as_scalar, escape_velocity, speed_around_attractor


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


def payload_mass_ratio(
    v_rf: u.Quantity,
    v_b: u.Quantity,
    v_ri: u.Quantity = 0 * u.km / u.s,
    fudge_factor: float = STD_FUDGE_FACTOR,
) -> float:
    """Compute the ratio of payload mass to PuffSat propulsion mass.
    Args:
        v_rf: Final velocity of the rocket (astropy Quantity, km/s).
        v_b: Velocity of the PuffSat  (astropy Quantity, km/s).
        v_ri: Initial velocity of the rocket (astropy Quantity, km/s, default 0 km/s).
        fudge_factor: Fudge factor for the mass calculation (default 0.8).

    Returns:
        The payload mass to PuffSat propulsion mass ratio.
    """

    denom: float = np.log((v_b - v_ri) / (v_b - v_rf)).item()
    return 2 * fudge_factor / denom


def rocket_equation(delta_v: u.Quantity, exhaust_v: u.Quantity) -> u.Quantity:
    """
    Compute the fractional propellant mass required for a given delta-v using the Tsoilkovksy trocket equation.

    Parameters
    ----------
    delta_v : astropy.units.Quantity
        The total change in velocity required (delta-v), with velocity units.
    exhaust_v : astropy.units.Quantity
        The effective exhaust velocity of the rocket, with velocity units.

    Returns
    -------
    astropy.units.Quantity
        The fractional propellant mass required (dimensionless, as a Quantity).
    """
    return 1 - np.exp(-delta_v / exhaust_v)


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


def burn_for_v_infinity(
    v_infinity: u.Quantity,
    body: Body = Earth,
    altitude: u.Quantity = LEO_ALTITUDE,
    initial_velocity: u.Quantity = 0 * u.km / u.s,
) -> u.Quantity:
    """Calculate the burn required to achieve a specific v_infinity.

    This function computes the delta-v needed to achieve a desired v_infinity
    (hyperbolic excess velocity) when starting from a given altitude above a celestial body.
    The burn is applied at the specified altitude to achieve the target v_infinity.

    Args:
        v_infinity: The desired hyperbolic excess velocity (astropy Quantity).
        body: The celestial body to escape from (poliastro Body, default Earth).
        altitude: Altitude above the body's surface where the burn occurs (astropy Quantity, default LEO_ALTITUDE).
        initial_velocity: Initial velocity at the burn altitude, defaults to 0 km/s (astropy Quantity).

    Returns:
        The required burn (delta-v) to achieve the v_infinity (astropy Quantity).

    Raises:
        ValueError: If v_infinity is less than or equal to zero.
    """
    if v_infinity <= 0 * u.km / u.s:
        raise ValueError("v_infinity must be positive.")

    # Distance from the center of the body at burn altitude
    burn_radius: u.Quantity = body.R + altitude

    # Escape velocity at the burn altitude
    escape_velocity_at_altitude: u.Quantity = escape_velocity(body, altitude)

    # For a hyperbolic orbit, the total velocity at the burn point is:
    # v_total^2 = v_escape^2 + v_infinity^2
    # This is derived from the vis-viva equation for hyperbolic orbits

    total_velocity_squared: u.Quantity = escape_velocity_at_altitude**2 + v_infinity**2
    total_velocity: u.Quantity = np.sqrt(total_velocity_squared)

    # The burn required is the difference between total velocity and initial velocity
    required_burn: u.Quantity = total_velocity - initial_velocity

    return required_burn.to(u.km / u.s)


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
