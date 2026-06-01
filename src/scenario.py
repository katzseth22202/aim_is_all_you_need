"""PuffSat propulsion scenarios and data structures.

This module provides scenario analysis and data structures for PuffSat propulsion
missions. It combines orbital mechanics and propulsion calculations to analyze
various mission scenarios and their feasibility.

Key Components:
    - PuffSatScenario: Data structure for propulsion scenarios
    - BurnInfo: Data structure for burn analysis results
    - paper_scenarios(): Generate predefined mission scenarios
    - find_best_lunar_return(): Optimize lunar return trajectories
    - launch_capacity_time(): Calculate mission timing requirements

Key Functions:
    - earth_velocity_200km_periapsis: Solar periapsis velocity calculations
    - solar_fusion_velocity: Nuclear fusion impact velocity analysis
    - orbit_from_periapsis_speed_and_apoapsis_radius: Orbit construction

Dependencies:
    - orbit_utils: Orbital mechanics calculations
    - propulsion: Propulsion analysis functions
    - astro_constants: Physical constants and mission parameters

This module serves as the highest-level analysis layer, combining all
lower-level calculations to provide comprehensive mission analysis.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from boinor.bodies import Body, Earth, Moon, Saturn, Sun
from boinor.twobody import Orbit

from src.astro_constants import (
    EARTH_A,
    EFFECTIVE_DV_LUNAR,
    JUPITER_A,
    LEO_ALTITUDE,
    LOW_SATURN_ALTITUDE,
    METHALOX_SEA_LEVEL_ISP,
    MOON_A,
    PARKER_PERIAPSIS,
    PERIAPSIS_SOLAR_BURN,
    PERIAPSIS_SOLAR_V,
    PHOEBE_A,
    REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
    REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
    RETROGRADE_FRACTION,
    SUBORBITAL_DV_TO_200KM,
    TARGET_LAUNCH_CAPACITY_MULTIPLE,
)

# Import orbital mechanics functions from orbit_utils
from src.orbit_utils import (
    apoapsis_velocity,
    escape_velocity,
    get_period,
    orbit_from_rp_ra,
    periapsis_velocity,
    speed_around_attractor,
    speed_with_escape_energy,
    velocity_at_distance,
)

# Import propulsion functions from propulsion
from src.propulsion import (
    burn_for_v_infinity,
    exhaust_velocity_from_isp,
    payload_mass_ratio,
    retrograde_jovian_hohmann_transfer,
    rocket_equation,
)


@dataclass(frozen=True)
class PuffSatScenario:
    """A single externally-pulsed (PuffSat) propulsion event, modelled as one
    elastic collision.

    A scenario is fully defined by three velocities; its payload-to-PuffSat mass
    ratio follows from them via :func:`payload_mass_ratio`. This is a single
    elastic-collision scenario only: blended optimization results such as the
    lunar-return optimum (see :func:`find_best_lunar_return`) are not scenarios.

    Attributes:
        v_rf: Final velocity of the payload after the collision (astropy Quantity).
        v_b: Collision velocity of the PuffSat (astropy Quantity).
        desc: Human-readable description of the scenario.
        v_ri: Initial velocity of the payload before the collision (astropy
            Quantity, default 0 km/s).
    """

    v_rf: u.Quantity
    v_b: u.Quantity
    desc: str
    v_ri: u.Quantity = 0 * u.km / u.s

    @property
    def mass_ratio(self) -> float:
        """Payload-to-PuffSat-propulsion mass ratio for this scenario.

        Returns:
            The mass ratio, computed from this scenario's collision, final, and
            initial velocities via :func:`payload_mass_ratio`.
        """
        return payload_mass_ratio(v_rf=self.v_rf, v_b=self.v_b, v_ri=self.v_ri)


# Columns of the scenario table -- the DataFrame projection of a scenario
# catalog. This is the single home for the table schema.
SCENARIO_COLUMNS = ["payload_puffsat_mass_ratio", "v_rf", "v_ri", "v_b", "desc"]


def paper_scenarios() -> List[PuffSatScenario]:
    """Build the catalog of PuffSat scenarios analyzed in the paper.

    The lunar-return optimum is intentionally excluded: it is not a single
    elastic-collision scenario (its mass ratio is a blended optimization result
    from :func:`find_best_lunar_return`), so it is presented separately rather
    than as a row here.

    Returns:
        The ordered list of :class:`PuffSatScenario` from the paper.
    """
    low_earth_periapsis = Earth.R + LEO_ALTITUDE
    lunar_transfer_orbit = orbit_from_rp_ra(
        apoapsis_radius=MOON_A,
        periapsis_radius=low_earth_periapsis,
        attractor_body=Earth,
    )
    lunar_transfer_periapsis_velocity = periapsis_velocity(orbit=lunar_transfer_orbit)
    leo_speed = speed_around_attractor(a=low_earth_periapsis, attractor=Earth)

    parker_orbit = orbit_from_rp_ra(
        apoapsis_radius=EARTH_A, periapsis_radius=PARKER_PERIAPSIS
    )
    parker_apoapsis_velocity = apoapsis_velocity(orbit=parker_orbit)
    earth_speed = speed_around_attractor(a=EARTH_A, attractor=Sun)
    # v_infinity for prograde transfer from Earth to Parker orbit apoapsis
    prograde_v_infinity_earth_to_parker = earth_speed - parker_apoapsis_velocity
    prograde_dv_parker_burn = burn_for_v_infinity(prograde_v_infinity_earth_to_parker)
    # v_infinity for retrograde transfer from Earth to Parker orbit apoapsis
    retrograde_v_infinity_earth_to_parker = earth_speed + parker_apoapsis_velocity
    retrograde_dv_parker_burn = burn_for_v_infinity(
        retrograde_v_infinity_earth_to_parker
    )
    retrograde_jovian_speed = retrograde_jovian_hohmann_transfer()

    lunar_esc = escape_velocity(body=Moon)

    min_saturn_altitude = Saturn.R + LOW_SATURN_ALTITUDE
    min_saturn_speed = speed_around_attractor(a=min_saturn_altitude, attractor=Saturn)
    phoebe_low_orbit = orbit_from_rp_ra(
        apoapsis_radius=PHOEBE_A,
        periapsis_radius=min_saturn_altitude,
        attractor_body=Saturn,
    )
    phoebe_low_periapsis_velocity = periapsis_velocity(orbit=phoebe_low_orbit)

    return [
        PuffSatScenario(
            v_rf=leo_speed,
            v_b=lunar_transfer_periapsis_velocity,
            desc="""Eccentric PuffSats with apogee at lunar distance push
rocket to minimal low Earth orbit""",
        ),
        PuffSatScenario(
            v_rf=0 * u.km / u.s,
            v_b=-leo_speed,
            v_ri=leo_speed,
            desc="""Decelerate intercity rocket for powered reentry with retrograde PuffSats in low orbit""",
        ),
        PuffSatScenario(
            v_rf=0 * u.km / u.s,
            v_b=-lunar_transfer_periapsis_velocity,
            v_ri=leo_speed,
            desc="""Decelerate intercity rocket for powered reentry with retrograde PuffSats from lunar orbit""",
        ),
        PuffSatScenario(
            v_rf=prograde_dv_parker_burn,
            v_b=retrograde_jovian_speed,
            desc="""PuffSats approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe""",
        ),
        PuffSatScenario(
            v_rf=retrograde_dv_parker_burn,
            v_b=retrograde_jovian_speed,
            desc="""PuffSats approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe but in a retrograde orbit around the Sun""",
        ),
        PuffSatScenario(
            v_rf=lunar_transfer_periapsis_velocity,
            v_b=retrograde_jovian_speed,
            desc="""PuffSats approach Earth from Jupiter and push a rocket into an elliptical orbit""",
        ),
        PuffSatScenario(
            v_rf=0 * u.km / u.s,
            v_b=-lunar_esc,
            v_ri=lunar_esc,
            desc="""Decelerate trans-lunar payloads to land on the moon""",
        ),
        PuffSatScenario(
            v_rf=min_saturn_speed,
            v_b=phoebe_low_periapsis_velocity,
            desc="""PuffSats approach Saturn from Phoebe and push a Helium-3 payload into a temporary very low orbit around Saturn""",
        ),
    ]


def scenarios_to_dataframe(scenarios: List[PuffSatScenario]) -> pd.DataFrame:
    """Project a scenario catalog into a display DataFrame.

    A pure, one-way projection: it reads each scenario's mass ratio and lays the
    fields out under :data:`SCENARIO_COLUMNS`. This is the only function in this
    path that touches pandas.

    Args:
        scenarios: The scenario catalog (e.g. from :func:`paper_scenarios`).

    Returns:
        A DataFrame with :data:`SCENARIO_COLUMNS`, one row per scenario.
    """
    df = pd.DataFrame(columns=SCENARIO_COLUMNS)
    for scenario in scenarios:
        df.loc[len(df)] = [
            scenario.mass_ratio,
            scenario.v_rf,
            scenario.v_ri,
            scenario.v_b,
            scenario.desc,
        ]
    return df


def orbit_from_periapsis_speed_and_apoapsis_radius(
    periapsis_speed: u.Quantity,
    apoapsis_radius: u.Quantity,
    attractor_body: Body = Sun,
) -> Orbit:
    """
    Generate a boinor Orbit aligned with the y-axis (periapsis on +y, no z-component),
    given the scalar speed at periapsis and the radius at apoapsis.

    Parameters
    ----------
    periapsis_speed : astropy.units.Quantity
        The scalar speed at periapsis (with velocity units).
    apoapsis_radius : astropy.units.Quantity
        The radius of the apoapsis (with length units).
    attractor_body : boinor.bodies.Body, optional
        The central celestial body (default: Sun).

    Returns
    -------
    boinor.twobody.Orbit
        The generated boinor Orbit object.

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
        prograde_dv_required: The required delta-v for pushing payloads off PuffSat to a prograde lunar transfer orbit with Earth leo periapsis
        retrograde_dv_required: The required delta-v for pushing payloads off PuffSat to a retrograde lunar transfer orbit with Earth leo periapsis


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
        incoming_v: u.Quantity = speed_with_escape_energy(
            incoming_v_before_moon_gravity, Moon
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


def solar_impact_dv(
    heliocentric_distance: u.Quantity, attractor: Body = Sun
) -> u.Quantity:
    """Compute the delta-v needed to drop a circular heliocentric orbit into the Sun.

    To fall into the Sun from a circular orbit, a spacecraft must cancel
    essentially all of its tangential (orbital) velocity; it then free-falls
    radially inward and impacts the Sun. The required delta-v therefore equals
    the circular orbital velocity at that heliocentric distance. This is the
    standard back-of-envelope "solar-impact delta-v is approximately the orbital
    velocity" estimate used in the paper (e.g. ~30 km/s from Earth, ~10 km/s
    from Saturn, ~18 km/s from Ceres).

    Args:
        heliocentric_distance: Distance from the attractor's center (astropy
            Quantity, length units), e.g. a body's orbital semi-major axis.
        attractor: The central body (boinor Body, default Sun).

    Returns:
        The solar-impact delta-v (astropy Quantity, km/s), equal to the circular
        orbital velocity at the given distance.
    """
    return speed_around_attractor(a=heliocentric_distance, attractor=attractor)


def hohmann_v_infinity(
    target_semimajor_axis: u.Quantity,
    departure_semimajor_axis: u.Quantity = EARTH_A,
) -> u.Quantity:
    """Heliocentric hyperbolic-excess velocity at departure for a Hohmann transfer.

    Computes the magnitude of the velocity difference between the departure
    planet's circular heliocentric velocity and the transfer ellipse's velocity at
    the departure point, for a Hohmann transfer between two circular, coplanar
    heliocentric orbits. For an outward transfer (target farther than departure)
    the departure planet sits at the transfer ellipse's perihelion; for an inward
    transfer it sits at the aphelion.

    Args:
        target_semimajor_axis: Heliocentric semi-major axis of the destination
            planet (astropy Quantity, length units).
        departure_semimajor_axis: Heliocentric semi-major axis of the departure
            planet (astropy Quantity, default EARTH_A).

    Returns:
        The hyperbolic-excess velocity v_infinity at departure (astropy Quantity,
        km/s).
    """
    v_departure = speed_around_attractor(a=departure_semimajor_axis, attractor=Sun)
    if target_semimajor_axis > departure_semimajor_axis:
        # Outward transfer: departure planet is at the transfer ellipse's perihelion.
        transfer = orbit_from_rp_ra(
            apoapsis_radius=target_semimajor_axis,
            periapsis_radius=departure_semimajor_axis,
            attractor_body=Sun,
        )
        v_transfer = periapsis_velocity(transfer)
    else:
        # Inward transfer: departure planet is at the transfer ellipse's aphelion.
        transfer = orbit_from_rp_ra(
            apoapsis_radius=departure_semimajor_axis,
            periapsis_radius=target_semimajor_axis,
            attractor_body=Sun,
        )
        v_transfer = apoapsis_velocity(transfer)
    return abs(v_transfer - v_departure).to(u.km / u.s)


def lunar_return_transfer_dv(
    target_semimajor_axis: u.Quantity,
    perigee_altitude: u.Quantity = LEO_ALTITUDE,
) -> u.Quantity:
    """Extra delta-v to redirect a lunar-return payload onto an interplanetary Hohmann transfer.

    A payload launched from the Moon toward Earth arrives at perigee already moving
    at the perigee speed of an Earth->Moon transfer ellipse (apogee at lunar
    distance, periapsis at ``perigee_altitude``) -- i.e. it is already at "lunar
    escape" energy with respect to the Earth-Moon system. Firing its rocket at that
    perigee, the *additional* delta-v needed to reach the hyperbolic-excess velocity
    of a heliocentric Hohmann transfer to a planet at ``target_semimajor_axis`` is

        dv = sqrt(v_esc**2 + v_inf**2) - v_perigee_lunar_return

    where ``v_esc`` is Earth's escape velocity at the perigee altitude and ``v_inf``
    is the heliocentric Hohmann excess velocity at Earth. This backs the paper's
    claim that only ~300-600 m/s beyond lunar escape reaches Venus or Mars.

    Args:
        target_semimajor_axis: Heliocentric semi-major axis of the destination
            planet (astropy Quantity, length units), e.g. VENUS_A or MARS_A.
        perigee_altitude: Altitude of the Earth perigee burn (astropy Quantity,
            default LEO_ALTITUDE = 200 km).

    Returns:
        The required delta-v beyond the lunar-return perigee speed (astropy
        Quantity, km/s).
    """
    perigee_radius = Earth.R + perigee_altitude
    # Speed a lunar-return payload already carries at perigee (apogee at the Moon).
    lunar_return_orbit = orbit_from_rp_ra(
        apoapsis_radius=MOON_A,
        periapsis_radius=perigee_radius,
        attractor_body=Earth,
    )
    v_perigee = periapsis_velocity(lunar_return_orbit)
    v_inf = hohmann_v_infinity(target_semimajor_axis)
    # The extra delta-v is exactly burn_for_v_infinity's job: reach the
    # hyperbolic-excess velocity v_inf from this perigee, given the speed the
    # lunar-return payload already carries there (v_perigee).
    return burn_for_v_infinity(
        v_inf, body=Earth, altitude=perigee_altitude, initial_velocity=v_perigee
    ).to(u.km / u.s)


def suborbital_200km_propellant_fraction(
    delta_v: u.Quantity = SUBORBITAL_DV_TO_200KM,
    specific_impulse: u.Quantity = METHALOX_SEA_LEVEL_ISP,
) -> u.Quantity:
    """Propellant mass fraction for a suborbital rocket that merely reaches 200 km.

    Backs the paper's Section 2.1 claim that a suborbital rocket which merely
    reaches ~200 km altitude -- and relies on PuffSat momentum pulses to reach
    orbit, rather than reaching orbit on its own -- can have a propellant mass
    fraction under 60 percent. Applies the Tsiolkovsky rocket equation
    (``rocket_equation``) to the ~2.5 km/s suborbital delta-v budget
    (``SUBORBITAL_DV_TO_200KM``) with a methalox sea-level exhaust velocity
    derived from ``METHALOX_SEA_LEVEL_ISP``.

    The defaults give ~0.56 (56 percent), comfortably under the paper's 60
    percent figure. Using the conservative sea-level Isp makes this an upper
    bound: a higher (vacuum) Isp would lower the fraction further.

    Args:
        delta_v: Suborbital delta-v budget to reach 200 km altitude (astropy
            Quantity, default SUBORBITAL_DV_TO_200KM = 2.5 km/s).
        specific_impulse: Engine specific impulse (astropy Quantity, default
            METHALOX_SEA_LEVEL_ISP = 310 s).

    Returns:
        The propellant mass fraction (dimensionless astropy Quantity).
    """
    exhaust_v = exhaust_velocity_from_isp(specific_impulse)
    return rocket_equation(delta_v=delta_v, exhaust_v=exhaust_v)


def find_parker_orbit_period() -> u.Quantity:
    """Calculate the orbital period of a transfer orbit between Earth and Parker Space Probe.

    This function calculates the orbital period of an elliptical transfer orbit around the Sun
    with periapsis near the Parker Space Probe's closest approach distance and apoapsis at
    Earth's orbital distance. This represents the time required for a spacecraft to complete
    one full orbit between these two points.

    The orbit is defined by:
    - Periapsis: Near Parker Space Probe's closest approach to the Sun (PARKER_PERIAPSIS)
    - Apoapsis: Earth's orbital distance from the Sun (EARTH_A)
    - Central body: Sun

    Returns:
        The orbital period of the transfer orbit (astropy Quantity, years).

    Note:
        Uses the standard orbital period formula T = 2π * sqrt(a³/μ) where 'a' is the
        semi-major axis and 'μ' is the Sun's gravitational parameter.
    """
    semimajor_axis: u.Quantity = (EARTH_A + PARKER_PERIAPSIS) / 2
    period: u.Quantity = get_period(Sun, semimajor_axis)
    return period.to(u.year)
