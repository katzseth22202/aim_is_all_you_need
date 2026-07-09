"""Main entry point: prints the numbers behind the paper's claims.

Every block below names the paper section it supports -- by section title and
LaTeX ``\\label`` (e.g. ``sec:no_isru_rocket``) as used in the paper source at
https://github.com/katzseth22202/Balloon-Pulse-Propulsion -- and quotes the
claim from ``paper/Aim_Is_All_You_Need.pdf`` that the computed values back up.
"""

import pandas as pd
from astropy import units as u
from tabulate import tabulate

from src.astro_constants import (
    CERES_A,
    EARTH_A,
    LUNAR_MONTH,
    MARS_A,
    REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
    REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
    SATURN_A,
    VENUS_A,
)
from src.scenario import (
    earth_reintercept_cycle_floor,
    earth_reintercept_scenarios,
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


def print_paper_point(section: str, claim: str, *results: str) -> None:
    """Print computed results under the paper section and claim they support.

    Args:
        section: Paper section title plus its LaTeX label, e.g.
            ``The Need For Speed (sec:no_isru_rocket)``.
        claim: The paper's claim (abridged quote) that the results back up.
        results: One line per computed value supporting the claim.
    """
    print(f"\n[{section}]")
    print(f'  paper claim: "{claim}"')
    for result in results:
        print(f"  computed:    {result}")


def print_scenario_table(title: str, note: str, df: pd.DataFrame) -> None:
    """Print a scenario table with its paper anchor and a per-table note.

    Args:
        title: Table heading naming the paper table/section it reproduces.
        note: One-line explanation of how the rows map onto the paper.
        df: The scenario DataFrame to render.
    """
    print(f"\n{title}")
    print(f"  {note}")
    print("=" * 80)
    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="grid",
            showindex=False,
            maxcolwidths=[None, 15, 15, 15, 36, 28],
        )
    )
    print("=" * 80)


def main() -> None:
    print(
        "Each block cites the paper section (title + LaTeX label from the\n"
        "Balloon-Pulse-Propulsion source) whose claim the computed numbers support."
    )

    print_scenario_table(
        "PuffSat Propulsion Scenarios -- reproduces Table 'tab:mass_scenarios' in "
        "'Mass Fraction Of Rocket To PuffSat Mass' (sec:mass_fraction)",
        "Each row backs the matching row of the paper's mass-ratio table (f=0.8); "
        "paper_ref names the section that develops that mission.",
        scenarios_to_dataframe(paper_scenarios()),
    )

    # The Parker rows above are outbound injection only: their 1 AU crossing is
    # pinned at aphelion and misses Earth by ~125 deg. The phased Earth-return
    # scenario folds the return phasing into a heavier boost, so it is presented
    # separately.
    print_scenario_table(
        "Phased Earth-return scenarios -- the 'Phased single-impulse resonant "
        "dive' row of Table 'tab:mass_scenarios', derived in Appendix 'Earth "
        "Re-Intercept and the Phasing Loop' (sec:earth_reintercept)",
        "Backs the paper's 37.53 / 69.272 / 2.05 table row: folding return "
        "phasing into one Earth boost costs mass ratio vs. the ~3.83 Parker row.",
        scenarios_to_dataframe(earth_reintercept_scenarios()),
    )

    # The lunar-return optimum is not a single-collision scenario, so it is
    # presented on its own rather than as a row in the tables above.
    best_lunar_return = find_best_lunar_return()
    lunar_mass_ratio = best_lunar_return.combined_mass_ratio
    print_paper_point(
        "Lunar Rockets Without Lunar Rocket Fuel (sec:lunar_rockets_no_fuel)",
        "if the main rocket accelerates at its Earth periapsis with a burn of "
        "1.9 km/s it will reach the moon at 7.2 km/s ... each loop around the "
        "moon launches about 1.455 times the starting mass "
        "(also the '3 and 3.7 / 7.20 / 1.455' row of tab:mass_scenarios)",
        f"optimal LEO Earth burn = {best_lunar_return.burn}",
        f"incoming velocity at the moon = {best_lunar_return.incoming_v}",
        f"payload/PuffSat mass ratio = {lunar_mass_ratio} for required "
        f"transfer dv = ({REQUIRED_DV_LUNAR_TRANSFER_PROGRADE}, "
        f"{REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE})",
    )
    print_paper_point(
        "Lunar Rockets Without Lunar Rocket Fuel (sec:lunar_rockets_no_fuel)",
        "if we do this once per month, we have increased our initial launch "
        "mass capacity by a factor of 1 million in about 2.8 years",
        "millionfold lunar launch capacity time = "
        f"{launch_capacity_time(lunar_mass_ratio, LUNAR_MONTH)}",
    )

    print_paper_point(
        "The Need For Speed (sec:no_isru_rocket) + Appendix sec:earth_reintercept",
        "a boosted projectile ... crosses 1 AU only once, about 125 deg of "
        "heliocentric longitude from where Earth has moved to ... it sets the "
        "shortest re-intercepting cycle near 0.86 yr",
        f"unphased 1 AU miss = {solar_dive_reintercept_gap():.0f}",
        f"re-intercepting cycle floor = {earth_reintercept_cycle_floor():.3f}",
    )
    print_paper_point(
        "The Need For Speed (sec:no_isru_rocket) + Appendix sec:earth_reintercept",
        "doubling the payload each cycle then reaches a factor of a million in "
        "about 17 years",
        f"millionfold scaling time (double each re-intercept cycle) = "
        f"{millionfold_scaling_time()}",
    )

    dive = single_impulse_resonant_dive()
    print_paper_point(
        "Appendix: Earth Re-Intercept and the Phasing Loop (sec:earth_reintercept)",
        "aim it outbound onto an orbit with aphelion near 1.9 AU ... it "
        "re-crosses after about 0.89 yr with Earth waiting ... the boost grows "
        "to about 37.5 km/s: the same 24 km/s retrograde component as a direct "
        "dive plus a 29 km/s outbound radial component",
        f"closing aphelion = {dive.closing_aphelion:.2f}, "
        f"re-intercept time = {dive.reintercept_time:.2f}",
        f"Earth boost = {dive.earth_boost:.0f} "
        f"({dive.retrograde_component:.0f} retrograde + "
        f"{dive.radial_component:.0f} radial)",
    )
    print_paper_point(
        "Appendix: Earth Re-Intercept and the Phasing Loop (sec:earth_reintercept)",
        "a minimum-energy dive ... is half the period of the transfer ellipse "
        "-- the timescale behind the ~0.21 yr round trip",
        "full period of the Earth-to-Parker-periapsis transfer orbit = "
        f"{find_parker_orbit_period()}",
    )

    print_paper_point(
        "The Need For Speed (sec:no_isru_rocket)",
        "perhaps we accelerate from 200 km/s to 250 km/s with Earth crossing "
        "speeds around 150 km/s",
        "velocity at Earth distance after the periapsis burn (200 km/s "
        f"periapsis) = {earth_velocity_200km_periapsis()}",
    )
    print_paper_point(
        "Fusion Propulsion And Epstein Drives (sec:epstein_drives)",
        "at this close solar periapsis [2 solar radii], prograde and retrograde "
        "projectile collision velocities would exceed 869 km/s ... in the "
        "reference frame of a fusion pulsed propulsion chamber",
        "impact velocity in the rocket's reference frame at 2 solar radii = "
        f"{solar_fusion_velocity()}",
    )

    print_paper_point(
        "When We Get Greedy, We'll Go to Phoebe (sec:greedy_phoebe)",
        "transferring to a solar impact trajectory from Saturn requires only "
        "about 10 km/s, compared to roughly 30 km/s from Earth",
        f"solar-impact delta-v from Earth = {solar_impact_dv(EARTH_A)}",
        f"solar-impact delta-v from Saturn = {solar_impact_dv(SATURN_A)}",
    )
    print_paper_point(
        "A Ceres-ly Good Alternative (sec:ceresly_good)",
        "a solar impact trajectory from Ceres requires only about 18 km/s, "
        "considerably less than the roughly 30 km/s required from Earth",
        f"solar-impact delta-v from Ceres = {solar_impact_dv(CERES_A)}",
    )

    print_paper_point(
        "Inner Planets Combined With Lunar ISRU "
        "(subsubsection under sec:jupiter_only_growth)",
        "if its rockets fire at a 200 km perigee, only 300 m/s to 600 m/s "
        "delta-v beyond lunar escape is needed [to reach Venus or Mars]",
        "lunar-return delta-v beyond lunar escape to Venus transfer = "
        f"{lunar_return_transfer_dv(VENUS_A).to(u.m / u.s)}",
        "lunar-return delta-v beyond lunar escape to Mars transfer = "
        f"{lunar_return_transfer_dv(MARS_A).to(u.m / u.s)}",
    )

    suborbital_frac = float(suborbital_200km_propellant_fraction())
    print_paper_point(
        "Making Starship's Excess Capacity Useful To Satellite Customers "
        "(sec:starship_safelaunch)",
        "a suborbital rocket that merely reaches 200 km in altitude might have "
        "a propellant mass fraction under 60 percent",
        "suborbital 200 km propellant mass fraction (methalox Isp=310 s, "
        f"dv=2.5 km/s) = {suborbital_frac:.1%} "
        f"({'under' if suborbital_frac < 0.6 else 'NOT under'} 60%)",
    )


if __name__ == "__main__":
    main()
