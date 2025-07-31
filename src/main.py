"""Main entry point for the application."""

from typing import List

import pandas as pd
from astropy import units as u
from poliastro.bodies import Earth
from tabulate import tabulate

from src.astro_constants import LEO_ALTITUDE, LUNAR_MONTH, MOON_A
from src.compute_utils import (
    BalloonScenario,
    earth_velocity_200km_periapsis,
    find_best_lunar_return,
    launch_capacity_time,
    orbit_from_rp_ra,
    solar_fusion_velocity,
)


def main() -> None:
    orb = orbit_from_rp_ra(
        apoapsis_radius=MOON_A,
        periapsis_radius=Earth.R + LEO_ALTITUDE,
        attractor_body=Earth,
    )
    print(f"best lunar return {find_best_lunar_return()}")

    # Get the scenarios DataFrame
    scenarios_df = BalloonScenario.paper_scenarios()

    # Display the table using tabulate with better formatting
    print("\nBalloon Propulsion Scenarios:")
    print("=" * 80)
    print(
        tabulate(
            scenarios_df,
            headers="keys",
            tablefmt="grid",
            showindex=False,
            maxcolwidths=[None, 15, 15, 15, 40],
        )
    )
    print("=" * 80)
    lunar_ratio = find_best_lunar_return().combined_mass_ratio
    print(
        f"lunarh launch cycle capacity time = {launch_capacity_time(lunar_ratio, LUNAR_MONTH)}"
    )
    print(
        f"launch cycle capacity of solar periapsis scenario = {launch_capacity_time(2, 1/2*u.year)}"
    )
    print(
        f"Velocity of rocket at Earth distance after periapsis at 200km/s = {earth_velocity_200km_periapsis()}"
    )
    print(
        f"Velocity of impact in reference frame of rocket for triggering nuclear impact fusion = {solar_fusion_velocity()} "
    )


if __name__ == "__main__":
    main()
