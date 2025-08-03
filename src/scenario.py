"""Balloon propulsion scenarios and data structures.

This module contains the BalloonScenario class and related functions for managing
balloon propulsion scenarios and their calculations.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from poliastro.bodies import Body, Earth, Moon, Saturn, Sun
from poliastro.twobody import Orbit

from src.astro_constants import (
    EARTH_A,
    EFFECTIVE_DV_LUNAR,
    JUPITER_A,
    LEO_ALTITUDE,
    LOW_SATURN_ALTITUDE,
    MOON_A,
    PARKER_PERIAPSIS,
    PERIAPSIS_SOLAR_BURN,
    PERIAPSIS_SOLAR_V,
    PHOEBE_A,
    REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
    REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
    RETROGRADE_FRACTION,
    TARGET_LAUNCH_CAPACITY_MULTIPLE,
)

# Import orbital mechanics functions from orbit_utils
from src.orbit_utils import (
    apoapsis_velocity,
    escape_velocity,
    orbit_from_rp_ra,
    periapsis_velocity,
    speed_around_attractor,
    velocity_at_distance,
)

# Import propulsion functions from propulsion
from src.propulsion import (
    burn_for_v_infinity,
    payload_mass_ratio,
    retrograde_jovian_hohmann_transfer,
    rocket_equation,
)


@dataclass(frozen=True)
class BalloonScenario:
    v_rf: u.Quantity
    v_b: u.Quantity
    desc: str
    v_ri: u.Quantity = 0

    @staticmethod
    def scenario_table() -> pd.DataFrame:
        """Create an empty DataFrame for storing balloon scenario results.

        Returns:
            A pandas DataFrame with columns for payload/balloon mass ratio, velocities, and description.
        """
        df = pd.DataFrame(
            columns=["payload_balloon_mass_ratio", "v_rf", "v_ri", "v_b", "desc"]
        )
        return df

    def append(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append this scenario's results to the provided DataFrame.

        Args:
            df: The DataFrame to append to.

        Returns:
            The DataFrame with this scenario's results added as a new row.
        """
        mass_ratio = payload_mass_ratio(v_rf=self.v_rf, v_b=self.v_b, v_ri=self.v_ri)
        df.loc[len(df)] = [mass_ratio, self.v_rf, self.v_ri, self.v_b, self.desc]
        return df

    @staticmethod
    def paper_scenarios() -> pd.DataFrame:
        low_earth_periapsis = Earth.R + LEO_ALTITUDE
        lunar_transfer_orbit = orbit_from_rp_ra(
            apoapsis_radius=MOON_A,
            periapsis_radius=low_earth_periapsis,
            attractor_body=Earth,
        )
        lunar_transfer_periapsis_velocity = periapsis_velocity(
            orbit=lunar_transfer_orbit
        )
        leo_speed = speed_around_attractor(a=low_earth_periapsis, attractor=Earth)
        desc = """Eccentric balloons with apogee at lunar distance pushes
rocket to minimal low Earth orbit"""
        scenario_table = BalloonScenario.scenario_table()
        BalloonScenario(
            v_rf=leo_speed, v_b=lunar_transfer_periapsis_velocity, desc=desc
        ).append(scenario_table)
        desc = """Decelerate intercity rocket for powered reentry with retrograde balloons in low orbit"""
        BalloonScenario(v_rf=0, v_b=-leo_speed, desc=desc, v_ri=leo_speed).append(
            scenario_table
        )
        desc = """Decelerate intercity rocket for powered reentry with retrograde balloons from lunar orbit"""
        BalloonScenario(
            v_rf=0, v_b=-lunar_transfer_periapsis_velocity, desc=desc, v_ri=leo_speed
        ).append(scenario_table)
        parker_orbit = orbit_from_rp_ra(
            apoapsis_radius=EARTH_A, periapsis_radius=PARKER_PERIAPSIS
        )
        parker_apoapsis_velocity = apoapsis_velocity(orbit=parker_orbit)
        earth_speed = speed_around_attractor(a=EARTH_A, attractor=Sun)
        # v_infinity for prograde transfer from Earth to Parker orbit apoapsis
        prograde_v_infinity_earth_to_parker = earth_speed - parker_apoapsis_velocity
        # calculate burn needed to achieve this v_infinity from Earth
        prograde_dv_parker_burn = burn_for_v_infinity(
            prograde_v_infinity_earth_to_parker
        )
        retrograde_jovian_speed = retrograde_jovian_hohmann_transfer()
        desc = """Balloons approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe"""
        BalloonScenario(
            v_rf=prograde_dv_parker_burn, v_b=retrograde_jovian_speed, desc=desc
        ).append(scenario_table)
        desc = """Balloons approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe but in a retrograde orbit around the Sun"""
        # v_infinity for retrograde transfer from Earth to Parker orbit apoapsis
        retrograde_v_infinity_earth_to_parker = earth_speed + parker_apoapsis_velocity
        # calculate burn needed to achieve this v_infinity from Earth
        retrograde_dv_parker_burn = burn_for_v_infinity(
            retrograde_v_infinity_earth_to_parker
        )
        BalloonScenario(
            v_rf=retrograde_dv_parker_burn, v_b=retrograde_jovian_speed, desc=desc
        ).append(scenario_table)
        desc = """Balloons approach Earth from Jupiter and push a rocket into an elliptical orbit"""
        BalloonScenario(
            v_rf=lunar_transfer_periapsis_velocity,
            v_b=retrograde_jovian_speed,
            desc=desc,
        ).append(scenario_table)
        desc = """Decelerate trans-lunar payloads to land on the moon"""
        lunar_esc = escape_velocity(body=Moon)
        BalloonScenario(
            v_rf=0 * u.km / u.s, v_b=-lunar_esc, desc=desc, v_ri=lunar_esc
        ).append(scenario_table)
        min_saturn_altitude = Saturn.R + LOW_SATURN_ALTITUDE
        min_saturn_speed = speed_around_attractor(
            a=min_saturn_altitude, attractor=Saturn
        )
        phoebe_low_orbit = orbit_from_rp_ra(
            apoapsis_radius=PHOEBE_A,
            periapsis_radius=min_saturn_altitude,
            attractor_body=Saturn,
        )
        phoebe_low_periapsis_velocity = periapsis_velocity(orbit=phoebe_low_orbit)
        desc = """Balloons approach Saturn from Phoebe and push a Helium-3 payload into a temporary very low orbit around Saturn"""
        BalloonScenario(
            v_rf=min_saturn_speed, v_b=phoebe_low_periapsis_velocity, desc=desc
        ).append(scenario_table)
        lunar_balloon_speed: BurnInfo = find_best_lunar_return()
        lunar_required_dv_prograde: u.Quantity = REQUIRED_DV_LUNAR_TRANSFER_PROGRADE
        desc = f"""After LEO Earth burn =  {lunar_balloon_speed.burn}, the balloon comes towards the moon at optimal speed."""
        scenario_table.loc[len(scenario_table)] = [
            lunar_balloon_speed.combined_mass_ratio,
            (
                REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
                REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
            ),
            0 * u.km / u.s,
            lunar_balloon_speed.incoming_v,
            desc,
        ]

        return scenario_table


def orbit_from_periapsis_speed_and_apoapsis_radius(
    periapsis_speed: u.Quantity,
    apoapsis_radius: u.Quantity,
    attractor_body: Body = Sun,
) -> Orbit:
    """
    Generate a poliastro Orbit aligned with the y-axis (periapsis on +y, no z-component),
    given the scalar speed at periapsis and the radius at apoapsis.

    Parameters
    ----------
    periapsis_speed : astropy.units.Quantity
        The scalar speed at periapsis (with velocity units).
    apoapsis_radius : astropy.units.Quantity
        The radius of the apoapsis (with length units).
    attractor_body : poliastro.bodies.Body, optional
        The central celestial body (default: Sun).

    Returns
    -------
    poliastro.twobody.Orbit
        The generated poliastro Orbit object.

    Raises
    ------
    ValueError
        If the computed periapsis radius is not physically valid (e.g., negative or zero).
    """
    # At periapsis, r = r_p, v = periapsis_speed
    # At apoapsis, r = r_a
    # Use vis-viva equation to solve for r_p:
    # v_p^2 = mu * (2/r_p - 1/a)
    # a = (r_p + r_a)/2
    mu = attractor_body.k
    r_a = apoapsis_radius
    v_p = periapsis_speed

    # Solve quadratic for r_p: v_p^2 = mu * (2/r_p - 2/(r_p + r_a))
    # Rearranged: v_p^2 = mu * (2(r_a) / (r_p * (r_p + r_a)))
    # Let x = r_p
    # v_p^2 * x^2 + v_p^2 * r_a * x - 2 * mu * r_a = 0
    # Quadratic: ax^2 + bx + c = 0
    a = v_p**2
    b = v_p**2 * r_a
    c = -2 * mu * r_a
    # Solve for x = r_p
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No real solution for periapsis radius with given parameters.")
    r_p = (-b + np.sqrt(discriminant)) / (2 * a)
    if r_p <= 0 * u.km:
        r_p = (-b - np.sqrt(discriminant)) / (2 * a)
    if r_p <= 0 * u.km:
        raise ValueError("Computed periapsis radius is not physically valid.")
    # Now use orbit_from_rp_ra to construct the orbit
    return orbit_from_rp_ra(
        apoapsis_radius=r_a, periapsis_radius=r_p, attractor_body=attractor_body
    )


@dataclass(frozen=True)
class BurnInfo:
    burn: u.Quantity
    combined_mass_ratio: u.Quantity
    incoming_v: u.Quantity


def launch_capacity_time(
    capacity_multiple_per_loop: float,
    one_loop_elapsed_time: u.Quantity,
    target_launch_capacity_multiple: float = TARGET_LAUNCH_CAPACITY_MULTIPLE,
) -> u.Quantity:
    time_elapsed: u.Quantity = (
        np.log(target_launch_capacity_multiple)
        / np.log(capacity_multiple_per_loop)
        * one_loop_elapsed_time
    )
    return time_elapsed.to(u.year)


def find_best_lunar_return(
    effective_exhaust_speed: u.Quantity = EFFECTIVE_DV_LUNAR,
    prograde_dv_required: u.Quantity = REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
    retrograde_dv_required: u.Quantity = REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
) -> BurnInfo:
    """Find the optimal burn for lunar return trajectory with maximum mass ratio.

    This function analyzes various burn magnitudes to find the optimal trajectory
    for returning from lunar orbit to Earth, maximizing the combined mass ratio
    while meeting the required delta-v constraints.

    Args:
        effective_exhaust_speed: The effective exhaust velocity for the rocket
            (astropy Quantity, default EFFECTIVE_DV_LUNAR).
        prograde_dv_required: The required delta-v for pushing moon's off balloon to a prograde lunar transfer orbit with Earth leo periapsis
        retrograde_dv_required: The required delta-v for pushing moon's off balloon to a retrograde lunar transfer orbit with Earth leo periapsis


    Returns:
        Optional[BurnInfo]: The optimal burn information containing the burn magnitude,
            combined mass ratio, and incoming velocity. Returns None if no valid
            solution is found.

    Note:
        The function evaluates burns from 0.01 to 6 km/s in 100 steps, calculating
        the trajectory from lunar orbit back to Earth's low orbit. It considers
        both the central body burn mass ratio and the target mass ratio to find
        the optimal combined solution.
    """
    candidate_burns: npt.NDArray[u.Quantity] = np.linspace(
        start=0.01 * u.km / u.s, stop=6 * u.km / u.s, num=100
    )
    periapsis_radius = Earth.R + LEO_ALTITUDE
    orbit: Orbit = orbit_from_rp_ra(
        apoapsis_radius=MOON_A, periapsis_radius=periapsis_radius, attractor_body=Earth
    )
    periapsis_v: u.Quantity = periapsis_velocity(orbit)
    max_burn: Optional[BurnInfo] = None
    for burn in candidate_burns:
        after_burn: u.Quantity = periapsis_v + burn
        incoming_v_before_moon_gravity = velocity_at_distance(
            radius_periapsis=periapsis_radius,
            velocity_periapsis=after_burn,
            distance=MOON_A,
            attractor_body=Earth,
        )
        incoming_v: u.Quantity = np.sqrt(
            incoming_v_before_moon_gravity**2 + escape_velocity(Moon) ** 2
        )
        if incoming_v <= retrograde_dv_required:
            continue
        prograde_lunar_payload_mass_ratio = payload_mass_ratio(
            v_rf=prograde_dv_required, v_b=incoming_v
        )
        if prograde_lunar_payload_mass_ratio < 1:
            continue
        retrograde_lunar_payload_mass_ratio = payload_mass_ratio(
            v_rf=retrograde_dv_required, v_b=incoming_v
        )
        leo_propulsion_mass_ratio = 1 - rocket_equation(
            delta_v=burn, exhaust_v=effective_exhaust_speed
        )
        prograde_combined_ratio = (
            prograde_lunar_payload_mass_ratio * leo_propulsion_mass_ratio
        )
        retrograde_combined_ratio = (
            retrograde_lunar_payload_mass_ratio * leo_propulsion_mass_ratio
        )
        prograde_fraction = 1 - RETROGRADE_FRACTION

        # Send retrograde_fraction of mass into retrograde transfer orbit and the rest into prograde orbit.
        # This achieves the effective velocity in the pulsed propulsion chamber when masses collide at maximum speed in LEO.
        combined_mass_ratio = (
            RETROGRADE_FRACTION * retrograde_combined_ratio
            + prograde_fraction * prograde_combined_ratio
        )
        burn_info = BurnInfo(
            burn=burn, combined_mass_ratio=combined_mass_ratio, incoming_v=incoming_v
        )
        if not max_burn:
            max_burn = burn_info
        else:
            if burn_info.combined_mass_ratio > max_burn.combined_mass_ratio:
                max_burn = burn_info
    if not max_burn:
        raise ValueError("No valid burn found that maximizes combined mass ratio")
    return max_burn


def earth_velocity_200km_periapsis(
    periapsis_v: u.Quantity = PERIAPSIS_SOLAR_V,
    periapsis_solar_burn: u.Quantity = PERIAPSIS_SOLAR_BURN,
) -> u.Quantity:
    """Calculate the velocity at Earth distance after a burn at periapsis.

    This function calculates the velocity of an object at Earth's orbital distance
    after it has accelerated at periapsis. The initial orbit has:
    - Periapsis velocity: 200 km/s (default PERIAPSIS_SOLAR_V)
    - Apoapsis: Earth's distance from the Sun (1 AU)
    - Burn at periapsis: additional velocity increase (default PERIAPSIS_SOLAR_BURN)

    Args:
        periapsis_v: Initial velocity at periapsis (astropy Quantity, default PERIAPSIS_SOLAR_V).
        periapsis_solar_burn: Additional velocity from burn at periapsis (astropy Quantity, default PERIAPSIS_SOLAR_BURN).

    Returns:
        The velocity at Earth distance after the burn (astropy Quantity, km/s).

    Note:
        Uses the vis-viva equation to calculate velocity at Earth distance based on
        the new periapsis velocity after the burn.
    """
    # Create the initial orbit with given periapsis velocity and Earth apoapsis
    initial_orbit: Orbit = orbit_from_periapsis_speed_and_apoapsis_radius(
        periapsis_speed=periapsis_v, apoapsis_radius=EARTH_A, attractor_body=Sun
    )

    # Get the periapsis radius from the orbit
    periapsis_radius: u.Quantity = initial_orbit.r_p

    # Calculate the new velocity at periapsis after the burn
    new_periapsis_velocity: u.Quantity = periapsis_v + periapsis_solar_burn

    # Calculate the velocity at Earth distance using the new periapsis velocity
    earth_velocity: u.Quantity = velocity_at_distance(
        radius_periapsis=periapsis_radius,
        velocity_periapsis=new_periapsis_velocity,
        distance=EARTH_A,
        attractor_body=Sun,
    )

    return earth_velocity


def solar_fusion_velocity() -> u.Quantity:
    """Calculate the impact velocity in reference frame of the prograde rocket at 2 solar radii

    This function creates an elliptical orbit around the Sun with:
    - Periapsis: 2 solar radii from the Sun's center
    - Apoapsis: Earth's orbital distance from the Sun (1 AU)

    Returns:
        The velocity at impact in reference frame of prograde rocket (astropy Quantity, km/s).

    Note:
        The factor of 2 accounts for the relative velocity between prograde and retrograde trajectories.
    """
    # Calculate periapsis radius: 2 solar radii
    periapsis_radius: u.Quantity = 2 * Sun.R

    # Use Earth's distance from the Sun as apoapsis
    apoapsis_radius: u.Quantity = EARTH_A

    # Create the orbit using existing function
    orbit: Orbit = orbit_from_rp_ra(
        apoapsis_radius=apoapsis_radius,
        periapsis_radius=periapsis_radius,
        attractor_body=Sun,
    )

    # The velocity of impact in reference frame of the rocket.
    # Factor of 2 accounts for relative velocity between prograde and retrograde trajectories.
    return 2 * periapsis_velocity(orbit)
