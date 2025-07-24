"""Main entry point for the application."""

from typing import List

import pandas as pd
from astropy import units as u
from poliastro.bodies import Earth

from src.astro_constants import LEO_ALTITUDE, MOON_A
from src.compute_utils import (
    BalloonScenario,
    apoapsis_velocity,
    orbit_from_rp_ra,
    payload_mass_ratio,
    periapsis_velocity,
)


def main() -> None:
    print(payload_mass_ratio(v_rf=11 * u.km / u.s, v_b=69 * u.km / u.s))
    orb = orbit_from_rp_ra(
        apoapsis_radius=MOON_A,
        periapsis_radius=Earth.R + LEO_ALTITUDE,
        attractor_body=Earth,
    )
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000
    ):
        print(BalloonScenario.paper_scenarios())


if __name__ == "__main__":
    main()
