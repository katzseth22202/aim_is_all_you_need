"""Main entry point for the application."""

from typing import List

import pandas as pd
from astropy import units as u
from poliastro.bodies import Earth

from src.astro_constants import LEO_ALTITUDE, LUNAR_MONTH, MOON_A
from src.compute_utils import (
    BalloonScenario,
    find_best_lunar_return,
    launch_capacity_time,
    orbit_from_rp_ra,
    payload_mass_ratio,
)


def main() -> None:
    orb = orbit_from_rp_ra(
        apoapsis_radius=MOON_A,
        periapsis_radius=Earth.R + LEO_ALTITUDE,
        attractor_body=Earth,
    )
    print(f"best lunar return {find_best_lunar_return()}")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    ):
        print(BalloonScenario.paper_scenarios())
    lunar_ratio = find_best_lunar_return().combined_mass_ratio
    print(
        f"lunarh launch cycle capacity time = {launch_capacity_time(lunar_ratio, LUNAR_MONTH)}"
    )
    print(
        f"launch cycle capacity of solar periapsis scenario = {launch_capacity_time(2, 1/4*u.year)}"
    )


if __name__ == "__main__":
    main()
