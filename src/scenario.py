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
from scipy.optimize import brentq

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
    SOLAR_DIVE_PERIAPSIS_BURN,
    SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
    SUBORBITAL_DV_TO_200KM,
    TARGET_LAUNCH_CAPACITY_MULTIPLE,
    TWO_IMPULSE_DIP_PERIAPSIS,
)

# Import orbital mechanics functions from orbit_utils
from src.orbit_utils import (
    apoapsis_speed,
    elliptic_time_of_flight,
    escape_velocity,
    get_period,
    hyperbolic_eccentricity,
    hyperbolic_time_of_flight,
    orbit_from_periapsis_speed_and_apoapsis_radius,
    orbit_from_rp_ra,
    periapsis_speed,
    speed_around_attractor,
    speed_at_distance,
    speed_with_escape_energy,
    true_anomaly_at_radius,
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
    lunar_transfer_periapsis_speed = periapsis_speed(orbit=lunar_transfer_orbit)
    leo_speed = speed_around_attractor(a=low_earth_periapsis, attractor=Earth)

    parker_orbit = orbit_from_rp_ra(
        apoapsis_radius=EARTH_A, periapsis_radius=PARKER_PERIAPSIS
    )
    parker_apoapsis_speed = apoapsis_speed(orbit=parker_orbit)
    earth_speed = speed_around_attractor(a=EARTH_A, attractor=Sun)
    # v_infinity for prograde transfer from Earth to Parker orbit apoapsis
    prograde_v_infinity_earth_to_parker = earth_speed - parker_apoapsis_speed
    prograde_dv_parker_burn = burn_for_v_infinity(prograde_v_infinity_earth_to_parker)
    # v_infinity for retrograde transfer from Earth to Parker orbit apoapsis
    retrograde_v_infinity_earth_to_parker = earth_speed + parker_apoapsis_speed
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
    phoebe_low_periapsis_speed = periapsis_speed(orbit=phoebe_low_orbit)

    return [
        PuffSatScenario(
            v_rf=leo_speed,
            v_b=lunar_transfer_periapsis_speed,
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
            v_b=-lunar_transfer_periapsis_speed,
            v_ri=leo_speed,
            desc="""Decelerate intercity rocket for powered reentry with retrograde PuffSats from lunar orbit""",
        ),
        PuffSatScenario(
            v_rf=prograde_dv_parker_burn,
            v_b=retrograde_jovian_speed,
            desc="""PuffSats approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe. Outbound injection only -- the phased Earth-return leg is scored separately (see earth_reintercept_scenarios)""",
        ),
        PuffSatScenario(
            v_rf=retrograde_dv_parker_burn,
            v_b=retrograde_jovian_speed,
            desc="""PuffSats approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe but in a retrograde orbit around the Sun. Outbound injection only; the retrograde orbit is for head-on collision energetics, not Earth-return phasing""",
        ),
        PuffSatScenario(
            v_rf=lunar_transfer_periapsis_speed,
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
            v_b=phoebe_low_periapsis_speed,
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


def earth_reintercept_scenarios() -> List[PuffSatScenario]:
    """Phased Earth-return dive scenarios (paper Appendix sec:earth_reintercept).

    The minimum-energy Parker-injection rows in :func:`paper_scenarios` pin the
    1 AU crossing at the transfer ellipse's aphelion, so the payload re-crosses
    1 AU ~125 deg from where Earth has moved to (:func:`solar_dive_reintercept_gap`)
    -- they are an *outbound injection* cost, not a returning orbit. This sibling
    catalog folds the Earth-return phasing into the boost instead: the
    single-impulse resonant dive (:func:`single_impulse_resonant_dive`) aims the
    payload outbound to a ~1.9 AU aphelion so its boosted solar-dive return
    re-intercepts Earth after ~0.89 yr.

    The phasing is not free: off the same ~69 km/s Jovian-return PuffSat, the
    boost grows from the prograde Parker row's ~23.7 km/s to ~37.5 km/s, which
    lowers the payload/PuffSat mass ratio from ~3.83 to ~2.05. This is the row
    that is actually "in the right phase for Earth return".

    Returns:
        The ordered list of phased Earth-return :class:`PuffSatScenario`.
    """
    retrograde_jovian_speed = retrograde_jovian_hohmann_transfer()
    resonant_dive = single_impulse_resonant_dive()
    return [
        PuffSatScenario(
            v_rf=resonant_dive.earth_boost,
            v_b=retrograde_jovian_speed,
            desc="""Phased single-impulse resonant dive: aim the payload outbound to a ~1.9 AU aphelion so its boosted solar-dive return re-intercepts Earth after ~0.89 yr. Folding the phasing into one Earth boost raises it to ~37.5 km/s, lowering the mass ratio from the prograde Parker injection's ~3.83""",
        ),
    ]


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
        BurnInfo: The optimal burn -- its burn magnitude, combined mass ratio,
            and incoming speed at the Moon.

    Raises:
        ValueError: If no candidate burn is feasible (none yields an incoming
            speed above retrograde_dv_required with a prograde mass ratio >= 1).

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
    periapsis_v: u.Quantity = periapsis_speed(orbit)
    feasible_burns: List[BurnInfo] = []
    for burn in candidate_burns:
        after_burn: u.Quantity = periapsis_v + burn
        incoming_v_before_moon_gravity = speed_at_distance(
            radius_periapsis=periapsis_radius,
            periapsis_speed=after_burn,
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
        feasible_burns.append(
            BurnInfo(
                burn=burn,
                combined_mass_ratio=combined_mass_ratio,
                incoming_v=incoming_v,
            )
        )
    if not feasible_burns:
        raise ValueError("No valid burn found that maximizes combined mass ratio")
    return max(feasible_burns, key=lambda info: info.combined_mass_ratio)


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
    new_periapsis_speed: u.Quantity = periapsis_v + periapsis_solar_burn

    # Calculate the velocity at Earth distance using the new periapsis velocity
    earth_velocity: u.Quantity = speed_at_distance(
        radius_periapsis=periapsis_radius,
        periapsis_speed=new_periapsis_speed,
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
    return 2 * periapsis_speed(orbit)


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
        v_transfer = periapsis_speed(transfer)
    else:
        # Inward transfer: departure planet is at the transfer ellipse's aphelion.
        transfer = orbit_from_rp_ra(
            apoapsis_radius=departure_semimajor_axis,
            periapsis_radius=target_semimajor_axis,
            attractor_body=Sun,
        )
        v_transfer = apoapsis_speed(transfer)
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
    v_perigee = periapsis_speed(lunar_return_orbit)
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


# ---------------------------------------------------------------------------
# "Sorry, I Don't Need ISRU" solar-dive Earth re-intercept
# (paper Appendix sec:earth_reintercept).
#
# A boosted solar-dive projectile leaves periapsis on an escaping hyperbola and
# crosses 1 AU only once, far from where Earth has moved to. Crossing 1 AU is not
# reaching Earth: the return must be phased to an Earth resonance, and that
# phasing -- not a 6-month dive -- sets the payload-doubling cycle. The functions
# below prove the appendix's cited figures from the repo's own primitives and
# supply the derived cycle floor used by the growth-rate estimate.
# ---------------------------------------------------------------------------

SOLAR_DIVE_PERIAPSIS = SOLAR_DIVE_PERIAPSIS_SOLAR_RADII * Sun.R


def solar_dive_periapsis_speed(
    periapsis_radius: u.Quantity = SOLAR_DIVE_PERIAPSIS,
    apoapsis_radius: u.Quantity = EARTH_A,
) -> u.Quantity:
    """Speed at periapsis of a minimum-energy dive from 1 AU to the solar periapsis.

    The projectile falls from an aphelion at ``apoapsis_radius`` (Earth's orbit)
    to ``periapsis_radius`` (4 solar radii by default). This backs the appendix's
    ~309 km/s figure -- the code gives ~306 km/s for the dive itself, which the
    paper rounds up to the ~309 km/s local escape speed at that radius (see
    :func:`boosted_solar_dive_v_infinity`).

    Args:
        periapsis_radius: Periapsis distance from the Sun's center (astropy
            Quantity, default 4 solar radii).
        apoapsis_radius: Aphelion distance, i.e. the launch orbit (astropy
            Quantity, default EARTH_A).

    Returns:
        The periapsis speed of the dive (astropy Quantity, km/s).
    """
    dive_orbit: Orbit = orbit_from_rp_ra(
        apoapsis_radius=apoapsis_radius,
        periapsis_radius=periapsis_radius,
        attractor_body=Sun,
    )
    return periapsis_speed(dive_orbit).to(u.km / u.s)


def boosted_solar_dive_v_infinity(
    periapsis_radius: u.Quantity = SOLAR_DIVE_PERIAPSIS,
    periapsis_burn: u.Quantity = SOLAR_DIVE_PERIAPSIS_BURN,
) -> u.Quantity:
    """Hyperbolic-excess speed left after a periapsis boost at the solar dive.

    A pulsed-propulsion boost of ``periapsis_burn`` (~34.5 km/s) at the 4
    solar-radii periapsis lifts the ~309 km/s local escape speed to ~343 km/s.
    The projectile then escapes with ``sqrt(v_boosted**2 - v_esc**2)`` to spare,
    the appendix's ~150 km/s -- matching the main text's ~150 km/s
    Earth-crossing scale for the no-ISRU cycle.

    Args:
        periapsis_radius: Periapsis distance from the Sun's center (astropy
            Quantity, default 4 solar radii).
        periapsis_burn: Speed increase from the PuffSat boost at periapsis
            (astropy Quantity, default SOLAR_DIVE_PERIAPSIS_BURN).

    Returns:
        The hyperbolic-excess (escape-to-spare) speed (astropy Quantity, km/s).
    """
    v_escape: u.Quantity = escape_velocity(Sun, altitude=periapsis_radius - Sun.R)
    v_boosted: u.Quantity = v_escape + periapsis_burn
    return np.sqrt(v_boosted**2 - v_escape**2).to(u.km / u.s)


def min_energy_solar_dive_time(
    periapsis_radius: u.Quantity = SOLAR_DIVE_PERIAPSIS,
    apoapsis_radius: u.Quantity = EARTH_A,
) -> u.Quantity:
    """Fall time from 1 AU to the solar periapsis on a minimum-energy dive.

    The dive is half of the transfer ellipse whose aphelion is ``apoapsis_radius``
    and periapsis is ``periapsis_radius`` (a ~0.509 AU semi-major axis for the
    default 4 solar-radii dive), so the fall takes half the orbital period -- the
    appendix's ~66 days.

    Args:
        periapsis_radius: Periapsis distance from the Sun's center (astropy
            Quantity, default 4 solar radii).
        apoapsis_radius: Aphelion distance, i.e. the launch orbit (astropy
            Quantity, default EARTH_A).

    Returns:
        The fall time from aphelion to periapsis (astropy Quantity, days).
    """
    dive_orbit: Orbit = orbit_from_rp_ra(
        apoapsis_radius=apoapsis_radius,
        periapsis_radius=periapsis_radius,
        attractor_body=Sun,
    )
    return (get_period(Sun, dive_orbit.a) / 2).to(u.day)


def solar_dive_whip_around_angle(
    periapsis_radius: u.Quantity = SOLAR_DIVE_PERIAPSIS,
    apoapsis_radius: u.Quantity = EARTH_A,
) -> u.Quantity:
    """Heliocentric longitude the boosted projectile sweeps from launch to 1 AU re-crossing.

    Falling from aphelion to periapsis always sweeps 180 deg of heliocentric
    longitude; the boosted climb-out is an escaping hyperbola that adds its true
    anomaly at the 1 AU re-crossing (~130 deg). The total whip-around is the
    appendix's ~310 deg, and it barely depends on periapsis depth.

    Args:
        periapsis_radius: Periapsis distance from the Sun's center (astropy
            Quantity, default 4 solar radii).
        apoapsis_radius: Re-crossing distance, i.e. Earth's orbit (astropy
            Quantity, default EARTH_A).

    Returns:
        The total whip-around angle (astropy Quantity, degrees).
    """
    v_infinity: u.Quantity = boosted_solar_dive_v_infinity(
        periapsis_radius=periapsis_radius
    )
    eccentricity: float = hyperbolic_eccentricity(periapsis_radius, v_infinity, Sun)
    climb_true_anomaly: u.Quantity = true_anomaly_at_radius(
        periapsis_radius, eccentricity, apoapsis_radius
    )
    return (180 * u.deg + climb_true_anomaly).to(u.deg)


def solar_dive_reintercept_gap(
    periapsis_radius: u.Quantity = SOLAR_DIVE_PERIAPSIS,
    apoapsis_radius: u.Quantity = EARTH_A,
) -> u.Quantity:
    """Angular gap between where the projectile re-crosses 1 AU and where Earth is.

    Over the ~0.21 yr round trip (fall + hyperbolic climb-out) Earth advances only
    ~76 deg, while the projectile whips ~310 deg around the Sun and re-crosses
    ~50 deg *behind* its launch longitude. The unphased miss is therefore ~125 deg
    -- set by the whip-around, not by Earth's drift.

    Args:
        periapsis_radius: Periapsis distance from the Sun's center (astropy
            Quantity, default 4 solar radii).
        apoapsis_radius: Re-crossing distance, i.e. Earth's orbit (astropy
            Quantity, default EARTH_A).

    Returns:
        The unphased heliocentric miss angle (astropy Quantity, degrees).
    """
    v_infinity: u.Quantity = boosted_solar_dive_v_infinity(
        periapsis_radius=periapsis_radius
    )
    eccentricity: float = hyperbolic_eccentricity(periapsis_radius, v_infinity, Sun)
    climb_true_anomaly: u.Quantity = true_anomaly_at_radius(
        periapsis_radius, eccentricity, apoapsis_radius
    )
    whip_around: u.Quantity = 180 * u.deg + climb_true_anomaly
    round_trip: u.Quantity = (
        min_energy_solar_dive_time(periapsis_radius, apoapsis_radius)
        + hyperbolic_time_of_flight(
            periapsis_radius, v_infinity, climb_true_anomaly, Sun
        )
    ).to(u.year)
    earth_advance: u.Quantity = (360 * u.deg) * (round_trip / (1.0 * u.year))
    # The re-crossing lands (whip_around - 360 deg) relative to launch (negative =
    # behind); the gap to Earth is that plus Earth's prograde advance.
    crossing_relative_to_launch: u.Quantity = whip_around - 360 * u.deg
    return (earth_advance - crossing_relative_to_launch).to(u.deg)


def periapsis_reaim_cost_per_degree(
    periapsis_radius: u.Quantity = SOLAR_DIVE_PERIAPSIS,
) -> u.Quantity:
    """Delta-v to turn the velocity vector by one degree at the solar periapsis.

    Turning the velocity by an angle ``theta`` costs ``2 * v_p * sin(theta/2)``.
    At the ~309 km/s periapsis (local escape) speed this is the appendix's
    ~5.4 km/s per degree -- prohibitive against a ~24 km/s dive boost, which is
    why the miss is fixed by phasing (timing), not by re-aiming at periapsis.

    Args:
        periapsis_radius: Periapsis distance from the Sun's center (astropy
            Quantity, default 4 solar radii).

    Returns:
        The re-aim cost for a one-degree turn (astropy Quantity, km/s per degree).
    """
    v_periapsis: u.Quantity = escape_velocity(Sun, altitude=periapsis_radius - Sun.R)
    return (2 * v_periapsis * np.sin(np.deg2rad(0.5))).to(u.km / u.s)


@dataclass(frozen=True)
class TwoImpulseLoop:
    """Result of the two-impulse phasing loop at 1 AU (all speeds tangential there).

    Attributes:
        earth_speed: Earth's circular heliocentric speed (astropy Quantity).
        dip_aphelion_speed: Speed at 1 AU after the first boost onto the shallow
            dip orbit (astropy Quantity).
        deep_dive_aphelion_speed: Speed at 1 AU after the second boost onto the
            deep dive (astropy Quantity).
        total_boost: Combined magnitude of the two colinear retrograde boosts,
            equal to a direct dive's boost (astropy Quantity).
        dip_return_time: Time for the dip orbit to return to 1 AU (astropy
            Quantity).
    """

    earth_speed: u.Quantity
    dip_aphelion_speed: u.Quantity
    deep_dive_aphelion_speed: u.Quantity
    total_boost: u.Quantity
    dip_return_time: u.Quantity


def two_impulse_phasing_loop(
    dip_periapsis: u.Quantity = TWO_IMPULSE_DIP_PERIAPSIS,
    deep_dive_periapsis: u.Quantity = SOLAR_DIVE_PERIAPSIS,
    apoapsis_radius: u.Quantity = EARTH_A,
) -> TwoImpulseLoop:
    """Prove the two-impulse phasing loop is free in total impulse.

    A first PuffSat boost at 1 AU drops the projectile to a shallow ``dip_periapsis``
    orbit; it returns to 1 AU tangentially after one dip period (~0.65 yr), where a
    second boost drops it into the deep dive. Both boosts are retrograde and
    colinear at 1 AU, so their magnitudes sum to a direct dive's single boost
    (~24 km/s) -- the delay costs no extra impulse, holding the doubling factor at
    two. The appendix's boost sequence is 29.78, 24.4, 5.7 km/s.

    Args:
        dip_periapsis: Periapsis of the shallow dip orbit (astropy Quantity,
            default TWO_IMPULSE_DIP_PERIAPSIS).
        deep_dive_periapsis: Periapsis of the deep dive (astropy Quantity,
            default 4 solar radii).
        apoapsis_radius: The shared aphelion / boost point (astropy Quantity,
            default EARTH_A).

    Returns:
        A :class:`TwoImpulseLoop` with the three 1 AU speeds, the total boost, and
        the dip return time.
    """
    earth_speed: u.Quantity = speed_around_attractor(a=apoapsis_radius, attractor=Sun)
    dip_orbit: Orbit = orbit_from_rp_ra(
        apoapsis_radius=apoapsis_radius,
        periapsis_radius=dip_periapsis,
        attractor_body=Sun,
    )
    dip_aphelion_speed: u.Quantity = apoapsis_speed(dip_orbit)
    deep_dive_orbit: Orbit = orbit_from_rp_ra(
        apoapsis_radius=apoapsis_radius,
        periapsis_radius=deep_dive_periapsis,
        attractor_body=Sun,
    )
    deep_dive_aphelion_speed: u.Quantity = apoapsis_speed(deep_dive_orbit)
    # Both boosts are retrograde and colinear at 1 AU, so the total is just the
    # single direct-dive boost from Earth's speed down to the deep-dive aphelion.
    total_boost: u.Quantity = (earth_speed - deep_dive_aphelion_speed).to(u.km / u.s)
    dip_return_time: u.Quantity = get_period(Sun, dip_orbit.a).to(u.year)
    return TwoImpulseLoop(
        earth_speed=earth_speed.to(u.km / u.s),
        dip_aphelion_speed=dip_aphelion_speed.to(u.km / u.s),
        deep_dive_aphelion_speed=deep_dive_aphelion_speed.to(u.km / u.s),
        total_boost=total_boost,
        dip_return_time=dip_return_time,
    )


@dataclass(frozen=True)
class SingleImpulseResonantDive:
    """Closure of the single-impulse resonant dive (paper Appendix sec:earth_reintercept).

    One PuffSat boost at Earth aims the projectile outbound onto an ellipse whose
    aphelion is tuned so that, after coasting out, falling back through 1 AU,
    diving to the solar periapsis, and climbing back out on the boosted hyperbola,
    it re-crosses 1 AU exactly where Earth has moved to. The aphelion is the free
    knob that closes the geometry; everything else follows from it.

    Attributes:
        closing_aphelion: The aphelion that makes the return re-intercept Earth
            (astropy Quantity, ~1.9 AU for the default dive).
        reintercept_time: Time from the Earth boost to the 1 AU re-crossing
            (astropy Quantity, ~0.89 yr).
        earth_boost: Magnitude of the single Earth boost (astropy Quantity,
            ~37.5 km/s), the vector sum of its retrograde and radial components.
        retrograde_component: Tangential (retrograde) part of the boost -- the
            same ~24 km/s a direct dive spends to drop to the solar periapsis
            (astropy Quantity).
        radial_component: Outbound radial part of the boost that buys the phasing
            coast (astropy Quantity, ~29 km/s).
        launch_true_anomaly: True anomaly of the 1 AU launch point on the closing
            ellipse, just short of aphelion (astropy Quantity, ~169 deg).
    """

    closing_aphelion: u.Quantity
    reintercept_time: u.Quantity
    earth_boost: u.Quantity
    retrograde_component: u.Quantity
    radial_component: u.Quantity
    launch_true_anomaly: u.Quantity


def single_impulse_resonant_dive(
    periapsis_radius: u.Quantity = SOLAR_DIVE_PERIAPSIS,
    launch_radius: u.Quantity = EARTH_A,
) -> SingleImpulseResonantDive:
    """Solve the aphelion that makes a single-impulse solar dive re-intercept Earth.

    The direct dive re-crosses 1 AU ~125 deg from Earth (:func:`solar_dive_reintercept_gap`).
    Folding the phasing into the single Earth boost aims the projectile *outbound*
    first, onto an ellipse with periapsis at the solar dive and a raised aphelion.
    The re-crossing longitude and its arrival time both grow with that aphelion, so
    exactly one aphelion makes Earth's advance equal the longitude the projectile
    sweeps -- the geometry closes. This roots that condition rather than hardcoding
    it, mirroring :func:`earth_reintercept_cycle_floor`, and reproduces the
    appendix's ~1.9 AU aphelion, ~0.89 yr re-cross, and ~37.5 km/s boost (a ~24 km/s
    retrograde component plus a ~29 km/s outbound radial one).

    The closure residual is monotonic in the aphelion over ``(launch_radius,
    4 * launch_radius)``, so the root found is the first (shortest) resonance.

    Args:
        periapsis_radius: Solar-dive periapsis (astropy Quantity, default 4 solar
            radii).
        launch_radius: The boost point / re-crossing distance, i.e. Earth's orbit
            (astropy Quantity, default EARTH_A).

    Returns:
        A :class:`SingleImpulseResonantDive` with the closing aphelion, the
        re-intercept time, and the boost with its retrograde/radial decomposition.
    """
    # The boosted climb-out is fixed by the periapsis and its boost; it does not
    # depend on the aphelion, so compute it once outside the root solve.
    v_infinity: u.Quantity = boosted_solar_dive_v_infinity(
        periapsis_radius=periapsis_radius
    )
    climb_eccentricity: float = hyperbolic_eccentricity(
        periapsis_radius, v_infinity, Sun
    )
    climb_true_anomaly: u.Quantity = true_anomaly_at_radius(
        periapsis_radius, climb_eccentricity, launch_radius
    )
    climb_time: u.Quantity = hyperbolic_time_of_flight(
        periapsis_radius, v_infinity, climb_true_anomaly, Sun
    )
    earth_speed: u.Quantity = speed_around_attractor(a=launch_radius, attractor=Sun)

    def closure_residual_deg(aphelion_au: float) -> float:
        """Heliocentric gap (deg) between Earth and the 1 AU re-crossing for a trial aphelion."""
        aphelion: u.Quantity = aphelion_au * u.AU
        eccentricity: float = float(
            ((aphelion - periapsis_radius) / (aphelion + periapsis_radius))
            .decompose()
            .value
        )
        semimajor_axis: u.Quantity = (aphelion + periapsis_radius) / 2
        launch_nu: u.Quantity = true_anomaly_at_radius(
            periapsis_radius, eccentricity, launch_radius
        )
        # Launched outbound at launch_nu, the projectile swings the long way through
        # aphelion to periapsis: the complement of the periapsis -> launch_nu time.
        ellipse_time: u.Quantity = get_period(
            Sun, semimajor_axis
        ) - elliptic_time_of_flight(periapsis_radius, eccentricity, launch_nu, Sun)
        total_time: u.Quantity = (ellipse_time + climb_time).to(u.year)
        swept: u.Quantity = (360 * u.deg - launch_nu) + climb_true_anomaly
        earth_longitude: u.Quantity = (360 * u.deg) * (total_time / (1.0 * u.year))
        return float((earth_longitude - swept).to(u.deg).value)

    launch_au: float = launch_radius.to(u.AU).value
    closing_aphelion: u.Quantity = (
        brentq(closure_residual_deg, launch_au * 1.0001, launch_au * 4.0) * u.AU
    )

    # Re-build the closing geometry once to report the timing and boost.
    eccentricity = float(
        ((closing_aphelion - periapsis_radius) / (closing_aphelion + periapsis_radius))
        .decompose()
        .value
    )
    semimajor_axis = (closing_aphelion + periapsis_radius) / 2
    launch_nu = true_anomaly_at_radius(periapsis_radius, eccentricity, launch_radius)
    ellipse_time = get_period(Sun, semimajor_axis) - elliptic_time_of_flight(
        periapsis_radius, eccentricity, launch_nu, Sun
    )
    reintercept_time: u.Quantity = (ellipse_time + climb_time).to(u.year)

    closing_orbit: Orbit = orbit_from_rp_ra(
        apoapsis_radius=closing_aphelion,
        periapsis_radius=periapsis_radius,
        attractor_body=Sun,
    )
    v_periapsis: u.Quantity = periapsis_speed(closing_orbit)
    # Angular momentum h = v_p * r_p is conserved; the tangential speed at launch is
    # h / r, and the radial speed closes the vis-viva speed by Pythagoras.
    tangential_speed: u.Quantity = (v_periapsis * periapsis_radius / launch_radius).to(
        u.km / u.s
    )
    launch_speed: u.Quantity = speed_at_distance(
        radius_periapsis=periapsis_radius,
        periapsis_speed=v_periapsis,
        distance=launch_radius,
        attractor_body=Sun,
    )
    radial_component: u.Quantity = np.sqrt(
        launch_speed**2 - tangential_speed**2
    ).to(u.km / u.s)
    retrograde_component: u.Quantity = (earth_speed - tangential_speed).to(u.km / u.s)
    earth_boost: u.Quantity = np.sqrt(
        retrograde_component**2 + radial_component**2
    ).to(u.km / u.s)

    return SingleImpulseResonantDive(
        closing_aphelion=closing_aphelion.to(u.AU),
        reintercept_time=reintercept_time,
        earth_boost=earth_boost,
        retrograde_component=retrograde_component,
        radial_component=radial_component,
        launch_true_anomaly=launch_nu.to(u.deg),
    )


def earth_reintercept_cycle_floor(
    periapsis_radius: u.Quantity = SOLAR_DIVE_PERIAPSIS,
    apoapsis_radius: u.Quantity = EARTH_A,
) -> u.Quantity:
    """Shortest solar-dive cycle that actually re-intercepts Earth.

    The projectile re-crosses 1 AU at a fixed longitude ``whip_around - 360 deg``
    behind its launch point. Earth first reaches that longitude after sweeping the
    complementary ``whip_around`` degrees prograde, i.e. a fraction
    ``whip_around / 360 deg`` of a year. With a ~310 deg whip-around this floor is
    the appendix's ~0.86 yr -- it supersedes the paper's earlier implied ~0.5 yr
    ("6 month") cycle and is the doubling interval for the growth estimate.

    Args:
        periapsis_radius: Periapsis distance from the Sun's center (astropy
            Quantity, default 4 solar radii).
        apoapsis_radius: Re-crossing distance, i.e. Earth's orbit (astropy
            Quantity, default EARTH_A).

    Returns:
        The re-intercepting cycle floor (astropy Quantity, years).
    """
    whip_around: u.Quantity = solar_dive_whip_around_angle(
        periapsis_radius, apoapsis_radius
    )
    return ((whip_around / (360 * u.deg)) * u.year).to(u.year)


def millionfold_scaling_time(
    doubling_factor: float = 2.0,
    target_multiple: float = TARGET_LAUNCH_CAPACITY_MULTIPLE,
    cycle_time: Optional[u.Quantity] = None,
) -> u.Quantity:
    """Time to scale launch capacity a millionfold at one doubling per re-intercept cycle.

    Applies :func:`launch_capacity_time` with the Earth re-intercept cycle floor
    (:func:`earth_reintercept_cycle_floor`, ~0.86 yr) as the doubling interval.
    At ``doubling_factor`` = 2 this is ~20 doublings over ~0.86 yr each, the
    appendix's ~17 years -- not the "under a decade" a ~0.5 yr cycle would imply.
    Only the two-impulse phasing loop holds the factor at two, so ~17 yr is itself
    a floor.

    Args:
        doubling_factor: Payload growth per cycle (default 2.0, the two-impulse
            loop).
        target_multiple: Target launch-capacity multiple (default
            TARGET_LAUNCH_CAPACITY_MULTIPLE = 1e6).
        cycle_time: Doubling interval (astropy Quantity); defaults to the derived
            Earth re-intercept cycle floor when None.

    Returns:
        The time to reach the target multiple (astropy Quantity, years).
    """
    if cycle_time is None:
        cycle_time = earth_reintercept_cycle_floor()
    return launch_capacity_time(
        capacity_multiple_per_loop=doubling_factor,
        one_loop_elapsed_time=cycle_time,
        target_launch_capacity_multiple=target_multiple,
    )
