"""Main entry point: prints the numbers behind the paper's claims.

Every block below names the paper section it supports -- by section title and
LaTeX ``\\label`` (e.g. ``sec:no_isru_rocket``) as used in the paper source at
https://github.com/katzseth22202/Balloon-Pulse-Propulsion -- and quotes the
paper claim (DOI 10.5281/zenodo.16741183) that the computed values back up.
"""

import pandas as pd
from astropy import units as u
from tabulate import tabulate

from src.astro_constants import (
    ASSIST_CHAIN_MAX_TRIP_TIME,
    CERES_A,
    EARTH_A,
    JUPITER_FLYBY_MAX_TOF,
    LUNAR_MONTH,
    MARS_A,
    REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
    REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
    SATURN_A,
    VENUS_A,
)
from src.propulsion import payload_mass_ratio
from src.scenario import (
    apoapsis_raise_economics,
    apoapsis_raise_finite_burn,
    apoapsis_raise_reintercept,
    assist_chain_return,
    earth_reintercept_cycle_floor,
    earth_reintercept_scenarios,
    earth_velocity_200km_periapsis,
    find_best_lunar_return,
    find_parker_orbit_period,
    jupiter_flyby_vb_trade_curve,
    launch_capacity_time,
    lunar_return_transfer_dv,
    millionfold_scaling_time,
    minimum_departure_burn_assist_chain,
    paper_scenarios,
    parker_injection_burns,
    powered_jovian_flyby_return,
    scenarios_to_dataframe,
    single_impulse_resonant_dive,
    solar_dive_reintercept_gap,
    solar_fusion_velocity,
    solar_impact_dv,
    suborbital_200km_propellant_fraction,
    venus_reach_departure_floor,
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
        "Phased Earth-return scenarios -- the two phased rows of Table "
        "'tab:mass_scenarios' that fold Earth-return phasing into the boost",
        "Row 1 backs the 37.53 / 69.272 / 2.05 resonant-dive row, derived in "
        "Appendix 'Earth Re-Intercept and the Phasing Loop' (sec:earth_reintercept) "
        "-- folding phasing into one Earth boost costs mass ratio vs. the ~3.83 "
        "Parker row. Row 2 is the apoapsis-raise re-intercept: v_rf=11.009 / "
        "v_b=24.059 / ratio 2.616, matching its own subsection 'Apoapsis-Raise "
        "Return With Onboard Burns' (sec:apoapsis_raise_loop), which "
        "tab:mass_scenarios now autorefs directly.",
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

    # Apoapsis-raise Earth re-intercept: the gentlest member of the re-intercept
    # family, now its own paper subsection rather than a standalone design doc.
    # The impulsive design point, its economics, and a finite-thrust SEP check
    # that confirms the impulsive burn -- the citable reproduction of the text.
    apoapsis_raise = apoapsis_raise_reintercept()
    print_paper_point(
        "Apoapsis-Raise Return With Onboard Burns (sec:apoapsis_raise_loop)",
        "a craft leaving Earth at escape speed spends 1.2 km/s of methalox at "
        "200 km to lift its aphelion to ~2.26 AU, where an argon solar-electric "
        "burn sheds 4 km/s of tangential speed ... the craft then falls back and "
        "meets Earth 1.69 yr after departure, closing at about 24 km/s, just over "
        "twice the local escape speed at the 200 km interception altitude",
        f"phasing-exact aphelion Q = {apoapsis_raise.aphelion:.3f} "
        f"(leg-1 a1 = {apoapsis_raise.leg1_semimajor_axis:.3f}); "
        f"phasing residual = {apoapsis_raise.phasing_residual.to(u.deg).value:.1e} "
        "deg (root-solved to ~0)",
        f"departure burn dv1 = {apoapsis_raise.departure_burn:.3f} (methalox), "
        f"apoapsis burn dv2 = {apoapsis_raise.sep_burn:.3f} (argon SEP)",
        f"closing speed at 200 km = {apoapsis_raise.closing_speed:.3f} vs 2x "
        f"escape target {apoapsis_raise.twice_escape_target:.3f} "
        f"(v_inf at SOI = {apoapsis_raise.v_infinity_arrival:.3f}); "
        f"transit = {apoapsis_raise.transit_time:.3f}",
        f"combined dry fraction reaching Earth = "
        f"{apoapsis_raise.combined_dry_fraction:.3f} "
        f"(methalox {apoapsis_raise.methalox_mass_fraction:.3f} x argon "
        f"{apoapsis_raise.sep_mass_fraction:.3f}); truncated perihelion "
        f"{apoapsis_raise.truncated_perihelion:.3f} (never reached)",
    )
    apoapsis_econ = apoapsis_raise_economics(apoapsis_raise)
    print_paper_point(
        "Apoapsis-Raise Return With Onboard Burns (sec:apoapsis_raise_loop)",
        "these burns total 5.2 km/s from two flown engine types and cost 41% of "
        "the departing mass; the collision then pushes 2.6 times the arriving "
        "mass to escape speed (tab:mass_scenarios), so the fleet multiplies by "
        "1.55 each cycle and reaches a millionfold in about 54 years, three "
        "times the 17 of the phased solar-dive loop (sec:earth_reintercept)",
        f"payload/PuffSat mass ratio m_r/m_p = "
        f"{apoapsis_econ.payload_puffsat_mass_ratio:.3f} (f=0.8, v_rf=Earth escape "
        "at 200 km, v_p=closing speed) -- matches tab:mass_scenarios' "
        "11.009 / 24.059 / 0 / 2.616 row",
        f"net growth per cycle = {apoapsis_econ.net_growth_per_cycle:.4f} "
        f"(+{100 * (apoapsis_econ.net_growth_per_cycle - 1):.1f}% per "
        f"{apoapsis_econ.cycle_time:.3f}; {apoapsis_econ.doublings_per_year:.3f} "
        "doublings/yr)",
        f"time to millionfold = {apoapsis_econ.time_to_millionfold:.1f} "
        f"({apoapsis_econ.cycles_to_millionfold:.1f} cycles)",
    )
    apoapsis_finite = apoapsis_raise_finite_burn(reintercept=apoapsis_raise)
    print_paper_point(
        "Apoapsis-Raise Return With Onboard Burns -- finite-thrust SEP check "
        "(sec:apoapsis_raise_loop)",
        "a finite-thrust check confirms that a 60-90 day burn centered on "
        "apoapsis reproduces the ideal impulse to under 1%",
        f"finite {apoapsis_finite.burn_duration:.0f} burn: closing speed = "
        f"{apoapsis_finite.closing_speed:.3f} vs impulsive "
        f"{apoapsis_raise.closing_speed:.3f} "
        f"({apoapsis_finite.closing_speed_error:.2%} difference)",
        f"finite transit = {apoapsis_finite.transit_time:.3f} vs impulsive "
        f"{apoapsis_raise.transit_time:.3f}; finite perihelion = "
        f"{apoapsis_finite.truncated_perihelion:.3f} vs impulsive "
        f"{apoapsis_raise.truncated_perihelion:.3f}; finite phasing residual = "
        f"{apoapsis_finite.phasing_residual:.2f}",
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

    # Powered Jovian flyby retrograde return: the leg the three Jovian-return
    # catalog rows assume but never derive. PLANNED paper subsection under
    # sec:jupiter_only_growth (ADR 0002-jupiter-flyby-objective); the catalog
    # rows above deliberately keep the paper's published retrograde-Hohmann
    # v_b ~ 69.27 km/s until the paper adopts these numbers.
    flyby = powered_jovian_flyby_return()
    parker_prograde_burn, parker_retrograde_burn = parker_injection_burns()
    print_paper_point(
        "Powered Jovian Flyby Retrograde Return -- PLANNED subsection under "
        "Jupiter-Only Exponential Launch Growth (sec:jupiter_only_growth); "
        "not yet in the published paper (ADR 0002)",
        "proposed claim: maximizing delivered mass x collision mass ratio -- "
        "not minimizing delta-v, which degenerates into a barely-retrograde "
        "plunge (ADR 0002) -- sets both burns of the leg that puts a PuffSat "
        "onto a retrograde Earth-crossing orbit",
        f"departure burn dv1 = {flyby.departure_burn:.3f} above escape at "
        f"200 km (v_inf = {flyby.v_infinity_earth:.3f}; free-aim angle = "
        f"{flyby.aim_angle:.2f} -- the optimizer picks a tangential departure)",
        f"flyby burn dv2 = {flyby.flyby_burn:.4f} at periapsis "
        f"{flyby.flyby_periapsis_radius:.0f} (altitude "
        f"{flyby.flyby_periapsis_altitude:.0f}): the unpowered bend suffices "
        f"at the optimum; turn = {flyby.turn_angle:.1f}, v_inf "
        f"{flyby.v_infinity_jupiter_in:.2f} -> "
        f"{flyby.v_infinity_jupiter_out:.2f}",
        f"achieved v_b = {flyby.collision_speed:.2f} vs the catalog rows' "
        "retrograde-Hohmann 69.27 km/s; truncated return perihelion = "
        f"{flyby.return_perihelion:.4f} (never reached -- intercepted at 1 AU)",
        f"delivered fraction {flyby.delivered_fraction:.3f} x mass ratio "
        f"{flyby.payload_puffsat_mass_ratio:.3f} (vs the "
        "sec:jupiter_only_growth push) = end-to-end "
        f"{flyby.end_to_end_mass_ratio:.3f}",
        f"time of flight = {flyby.outbound_time:.2f} out + "
        f"{flyby.return_time:.2f} back = {flyby.total_time:.2f} "
        f"(cap {JUPITER_FLYBY_MAX_TOF})",
        "Parker rows re-scored at the achieved v_b: prograde ratio = "
        f"{payload_mass_ratio(v_rf=parker_prograde_burn, v_b=flyby.collision_speed):.3f}, "
        "retrograde ratio = "
        f"{payload_mass_ratio(v_rf=parker_retrograde_burn, v_b=flyby.collision_speed):.3f}",
    )
    print(
        "\nPowered Jovian flyby -- min total burn vs target v_b (ADR 0002 "
        "trade curve; shows how flat the plateau around the optimum is)"
    )
    print(
        tabulate(
            [
                [
                    f"{point.target_collision_speed.to_value(u.km / u.s):.0f}",
                    "yes" if point.feasible else "no",
                    f"{point.total_burn.to_value(u.km / u.s):.2f}",
                    f"{point.departure_burn.to_value(u.km / u.s):.2f}",
                    f"{point.flyby_burn.to_value(u.km / u.s):.2f}",
                    f"{point.achieved_collision_speed.to_value(u.km / u.s):.2f}",
                    f"{point.end_to_end_mass_ratio:.3f}",
                    f"{point.total_time.to_value(u.year):.2f}",
                ]
                for point in jupiter_flyby_vb_trade_curve()
            ],
            headers=[
                "target v_b",
                "feasible",
                "total dv",
                "dv1 Earth",
                "dv2 Jupiter",
                "achieved v_b",
                "end-to-end",
                "TOF (yr)",
            ],
            tablefmt="grid",
        )
    )

    # Unpowered V/E/M assist chain: the same retrograde return with gravity
    # assists in place of the 4.45 km/s departure burn. Phasing-free model; a
    # fixed 300 m/s deep-space-maneuver budget is charged as spent methalox so
    # the mass numbers carry the cost of real planetary phasing.
    venus_floor = venus_reach_departure_floor()
    chain_scan = minimum_departure_burn_assist_chain(
        target_collision_speed=flyby.collision_speed
    )
    chain_minimum = chain_scan.minimum
    chain_fast = assist_chain_return(
        departure_burn=0.300 * u.km / u.s,
        target_collision_speed=flyby.collision_speed,
    )
    if chain_minimum is None or chain_fast is None:
        raise RuntimeError("assist-chain scan unexpectedly found no feasible chain")
    infeasible_probes = ", ".join(
        f"{burn.to_value(u.m / u.s):.0f}" for burn in chain_scan.infeasible_burns
    )
    print_paper_point(
        "Unpowered V/E/M Assist Chain to the Retrograde Return -- PLANNED "
        "companion to the powered Jovian flyby (sec:jupiter_only_growth); not "
        "yet in the published paper",
        "proposed claim: a Venus/Earth/Mars gravity-assist ladder replaces the "
        "4.45 km/s powered-flyby departure burn with ~0.3 km/s -- barely above "
        "the Venus-reach floor -- at the price of trip time and planetary "
        "phasing (charged here as a 300 m/s deep-space-maneuver budget)",
        f"Venus-reach floor = {venus_floor.to(u.m / u.s):.1f}: below it Venus is "
        "unreachable; exactly at it the arrival is tangent to Venus's orbit and "
        "Tisserand-locked (an unpowered flyby can rotate but never grow the "
        "excess speed, so a tangent arrival is a dead end)",
        f"beam-search scan at target v_b >= {flyby.collision_speed:.2f}: "
        f"infeasible at [{infeasible_probes}] m/s; minimum feasible burn = "
        f"{chain_minimum.departure_burn.to(u.m / u.s):.0f} "
        f"({chain_minimum.sequence}, total {chain_minimum.total_time:.2f}, "
        f"v_b {chain_minimum.collision_speed:.2f})",
        f"at 300 m/s the chain is fast: {chain_fast.sequence}, "
        f"{chain_fast.flyby_count} inner-planet flybys, total "
        f"{chain_fast.total_time:.2f} (chain {chain_fast.chain_time:.2f} + "
        f"return {chain_fast.return_time:.2f}; cap {ASSIST_CHAIN_MAX_TRIP_TIME}), "
        f"v_inf at Jupiter {chain_fast.v_infinity_jupiter:.2f}, Jovian bend "
        f"{chain_fast.jovian_bend_angle:.1f}, v_b {chain_fast.collision_speed:.2f}",
        f"mass accounting at 300 m/s: delivered fraction "
        f"{chain_fast.delivered_fraction:.3f} (burn + "
        f"{chain_fast.phasing_budget.to(u.m / u.s):.0f} phasing budget) x mass "
        f"ratio {chain_fast.payload_puffsat_mass_ratio:.3f} = end-to-end "
        f"{chain_fast.end_to_end_mass_ratio:.3f} vs the powered flyby's "
        f"{flyby.end_to_end_mass_ratio:.3f}",
    )
    print(
        "\nUnpowered assist chain at 300 m/s departure -- node-by-node "
        "(rotations are free unpowered flyby bends; v_inf is the Tisserand "
        "invariant each flyby preserves)"
    )
    print(
        tabulate(
            [
                [
                    step.body,
                    f"{step.rotation_angle.to_value(u.deg):+.1f}",
                    f"{step.v_infinity.to_value(u.km / u.s):.3f}",
                    step.target,
                    "outbound" if step.outbound_arrival else "inbound",
                    f"{step.leg_time.to_value(u.year):.2f}",
                    f"{step.elapsed.to_value(u.year):.2f}",
                ]
                for step in chain_fast.steps
            ],
            headers=[
                "at body",
                "rotate (deg)",
                "v_inf (km/s)",
                "leg to",
                "arrival",
                "leg (yr)",
                "elapsed (yr)",
            ],
            tablefmt="grid",
        )
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
