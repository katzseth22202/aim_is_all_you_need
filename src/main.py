"""Main entry point for the application."""

from typing import List

import pandas as pd
from astropy import units as u
from boinor.bodies import Earth
from tabulate import tabulate

from src.astro_constants import (
    CERES_A,
    EARTH_A,
    LEO_ALTITUDE,
    LUNAR_MONTH,
    MARS_A,
    MOON_A,
    REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
    REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
    SATURN_A,
    VENUS_A,
)
from src.orbit_utils import orbit_from_rp_ra
from src.scenario import (
    earth_reintercept_cycle_floor,
    earth_velocity_200km_periapsis,
    find_best_lunar_return,
    find_parker_orbit_period,
    launch_capacity_time,
    lunar_return_transfer_dv,
    millionfold_scaling_time,
    paper_scenarios,
    scenarios_to_dataframe,
    single_impulse_resonant_dive,
    solar_dive_reintercept_gap,
    solar_fusion_velocity,
    solar_impact_dv,
    suborbital_200km_propellant_fraction,
)


def main() -> None:
    orb = orbit_from_rp_ra(
        apoapsis_radius=MOON_A,
        periapsis_radius=Earth.R + LEO_ALTITUDE,
        attractor_body=Earth,
    )

    # Calculate best lunar return once and reuse the result
    best_lunar_return = find_best_lunar_return()
    print(f"best lunar return {best_lunar_return}")

    # Project the scenario catalog into a display DataFrame at the edge.
    scenarios_df = scenarios_to_dataframe(paper_scenarios())

    # Display the table using tabulate with better formatting
    print("\nPuffSat Propulsion Scenarios:")
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
    # The lunar-return optimum is not a single-collision scenario, so it is
    # presented on its own rather than as a row in the table above.
    print(
        f"lunar-return optimum: after LEO Earth burn = {best_lunar_return.burn}, "
        "the PuffSat comes towards the moon at optimal speed "
        f"(incoming v = {best_lunar_return.incoming_v}); "
        f"payload/PuffSat mass ratio = {best_lunar_return.combined_mass_ratio} "
        f"for required transfer dv = ({REQUIRED_DV_LUNAR_TRANSFER_PROGRADE}, "
        f"{REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE})"
    )
    lunar_mass_ratio = best_lunar_return.combined_mass_ratio
    print(
        f"lunar launch cycle capacity time = {launch_capacity_time(lunar_mass_ratio, LUNAR_MONTH)}"
    )
    # A boosted solar-dive return crosses 1 AU ~136 deg from Earth, so the growth
    # cycle is set by phasing the return to an Earth resonance (~0.82 yr floor),
    # not by the bare ~0.5 yr dive. See paper Appendix sec:earth_reintercept.
    print(
        "solar-dive re-intercept: unphased 1 AU miss = "
        f"{solar_dive_reintercept_gap():.0f}, cycle floor = "
        f"{earth_reintercept_cycle_floor():.3f}"
    )
    print(
        "Sorry I don't need ISRU launch capacity millionfold time (double each re-intercept cycle) = "
        f"{millionfold_scaling_time()}"
    )
    # The single-impulse resonant dive folds the phasing into one Earth boost: it
    # solves the outbound aphelion that closes the return geometry, at the cost of
    # a heavier boost. See paper Appendix sec:earth_reintercept.
    dive = single_impulse_resonant_dive()
    print(
        "single-impulse resonant dive: aphelion "
        f"{dive.closing_aphelion:.2f} closes the return in {dive.reintercept_time:.2f}, "
        f"Earth boost = {dive.earth_boost:.0f} "
        f"({dive.retrograde_component:.0f} retrograde + {dive.radial_component:.0f} radial)"
    )
    print(
        f"Velocity of rocket at Earth distance after periapsis at 200km/s = {earth_velocity_200km_periapsis()}"
    )
    print(
        f"Velocity of impact in reference frame of rocket for triggering nuclear impact fusion = {solar_fusion_velocity()} "
    )
    print(
        f"orbital period of transfer orbit to Parker periapsis = {find_parker_orbit_period()}"
    )
    print(
        f"solar-impact delta-v from Earth (cancel orbital velocity) = {solar_impact_dv(EARTH_A)}"
    )
    print(
        f"solar-impact delta-v from Saturn (cancel orbital velocity) = {solar_impact_dv(SATURN_A)}"
    )
    print(
        f"solar-impact delta-v from Ceres (cancel orbital velocity) = {solar_impact_dv(CERES_A)}"
    )
    print(
        "lunar-return delta-v beyond lunar escape to Venus transfer (200 km perigee) = "
        f"{lunar_return_transfer_dv(VENUS_A).to(u.m / u.s)}"
    )
    print(
        "lunar-return delta-v beyond lunar escape to Mars transfer (200 km perigee) = "
        f"{lunar_return_transfer_dv(MARS_A).to(u.m / u.s)}"
    )
    suborbital_frac = float(suborbital_200km_propellant_fraction())
    print(
        "suborbital 200 km propellant mass fraction (methalox Isp=310 s, dv=2.5 km/s) = "
        f"{suborbital_frac:.1%} ({'under' if suborbital_frac < 0.6 else 'NOT under'} 60%)"
    )


if __name__ == "__main__":
    main()
