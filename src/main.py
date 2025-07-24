"""Main entry point for the application."""

from typing import List

from astropy import units as u
from poliastro.bodies import Earth

from src.astro_constants import LEO_ALTITUDE, MOON_A
from src.compute_utils import (
    apoapsis_velocity,
    orbit_from_rp_ra,
    payload_mass_ratio,
    periapsis_velocity,
)


def greet(name: str) -> str:
    """Return a greeting message for the given name.

    Args:
        name: The name to greet

    Returns:
        A greeting message
    """
    return f"Hello, {name}! Welcome to AIM is all you need!"


def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers.

    Args:
        numbers: List of integers to sum

    Returns:
        The sum of all numbers in the list
    """
    return sum(numbers)


def main() -> None:
    print(payload_mass_ratio(v_rf=11 * u.km / u.s, v_b=69 * u.km / u.s))
    orb = orbit_from_rp_ra(
        apoapsis_radius=MOON_A,
        periapsis_radius=Earth.R + LEO_ALTITUDE,
        attractor_body=Earth,
    )
    print(periapsis_velocity(orb))
    print(apoapsis_velocity(orb))


if __name__ == "__main__":
    main()
