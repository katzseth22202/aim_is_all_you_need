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
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from boinor.bodies import Body, Earth, Moon, Saturn, Sun
from boinor.twobody import Orbit
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src.astro_constants import (
    APOAPSIS_RAISE_APHELION_BRACKET,
    APOAPSIS_RAISE_SEP_BURN_DURATION,
    APOAPSIS_RAISE_SEP_DV,
    ARGON_SEP_ISP,
    EARTH_A,
    EFFECTIVE_DV_LUNAR,
    JUPITER_A,
    LEO_ALTITUDE,
    LOW_SATURN_ALTITUDE,
    METHALOX_SEA_LEVEL_ISP,
    METHALOX_VACUUM_ISP,
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
        paper_ref: LaTeX ``\\label`` of the paper section that develops this
            mission -- the same section each row of the paper's Table
            ``tab:mass_scenarios`` cites via ``\\autoref``. Empty for ad-hoc
            scenarios that are not paper rows.
    """

    v_rf: u.Quantity
    v_b: u.Quantity
    desc: str
    v_ri: u.Quantity = 0 * u.km / u.s
    paper_ref: str = ""

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
SCENARIO_COLUMNS = [
    "payload_puffsat_mass_ratio",
    "v_rf",
    "v_ri",
    "v_b",
    "desc",
    "paper_ref",
]


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
            paper_ref="sec:starship_safelaunch",
        ),
        PuffSatScenario(
            v_rf=0 * u.km / u.s,
            v_b=-leo_speed,
            v_ri=leo_speed,
            desc="""Decelerate intercity rocket for powered reentry with retrograde PuffSats in low orbit""",
            paper_ref="sec:200_mile_high",
        ),
        PuffSatScenario(
            v_rf=0 * u.km / u.s,
            v_b=-lunar_transfer_periapsis_speed,
            v_ri=leo_speed,
            desc="""Decelerate intercity rocket for powered reentry with retrograde PuffSats from lunar orbit""",
            paper_ref="sec:200_mile_high",
        ),
        PuffSatScenario(
            v_rf=prograde_dv_parker_burn,
            v_b=retrograde_jovian_speed,
            desc="""PuffSats approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe. Outbound injection only -- the phased Earth-return leg is scored separately (see earth_reintercept_scenarios)""",
            paper_ref="sec:jupiter_gravity_initial",
        ),
        PuffSatScenario(
            v_rf=retrograde_dv_parker_burn,
            v_b=retrograde_jovian_speed,
            desc="""PuffSats approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe but in a retrograde orbit around the Sun. Outbound injection only; the retrograde orbit is for head-on collision energetics, not Earth-return phasing""",
            paper_ref="sec:jupiter_gravity_initial",
        ),
        PuffSatScenario(
            v_rf=lunar_transfer_periapsis_speed,
            v_b=retrograde_jovian_speed,
            desc="""PuffSats approach Earth from Jupiter and push a rocket into an elliptical orbit""",
            paper_ref="sec:jupiter_only_growth",
        ),
        PuffSatScenario(
            v_rf=0 * u.km / u.s,
            v_b=-lunar_esc,
            v_ri=lunar_esc,
            desc="""Decelerate trans-lunar payloads to land on the moon""",
            paper_ref="sec:no_isru_rocket",
        ),
        PuffSatScenario(
            v_rf=min_saturn_speed,
            v_b=phoebe_low_periapsis_speed,
            desc="""PuffSats approach Saturn from Phoebe and push a Helium-3 payload into a temporary very low orbit around Saturn""",
            paper_ref="sec:mining_helium_3",
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
            scenario.paper_ref,
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

    The second row is the apoapsis-raise Earth re-intercept
    (:func:`apoapsis_raise_reintercept`), the lowest-closing-speed member of the
    family: a returning PuffSat pushes the next payload to Earth escape and it
    re-intercepts Earth at ~24 km/s after raising aphelion to ~2.26 AU -- no solar
    dive, no gravity assist, no off-Earth boost node. Its v_rf/v_b are the
    departure escape speed and the closing speed, not a Jovian-return boost, so it
    is scored on its own collision rather than off the ~69 km/s PuffSat.

    Returns:
        The ordered list of phased Earth-return :class:`PuffSatScenario`.
    """
    retrograde_jovian_speed = retrograde_jovian_hohmann_transfer()
    resonant_dive = single_impulse_resonant_dive()
    apoapsis_raise = apoapsis_raise_reintercept()
    return [
        PuffSatScenario(
            v_rf=resonant_dive.earth_boost,
            v_b=retrograde_jovian_speed,
            desc="""Phased single-impulse resonant dive: aim the payload outbound to a ~1.9 AU aphelion so its boosted solar-dive return re-intercepts Earth after ~0.89 yr. Folding the phasing into one Earth boost raises it to ~37.5 km/s, lowering the mass ratio from the prograde Parker injection's ~3.83""",
            paper_ref="sec:earth_reintercept",
        ),
        PuffSatScenario(
            v_rf=escape_velocity(Earth, LEO_ALTITUDE),
            v_b=apoapsis_raise.closing_speed,
            desc="""Apoapsis-raise Earth re-intercept: a methalox Oberth burn raises heliocentric aphelion to ~2.26 AU, one retrograde argon-SEP burn at apoapsis lowers perihelion, and the craft falls back to intercept Earth at ~24 km/s after ~1.69 yr, pushing the next payload to Earth escape. Lowest-closing-speed member of sec:earth_reintercept -- no solar dive, no gravity assist, no off-Earth boost node, only onboard propellant""",
            paper_ref="sec:earth_reintercept",
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


# ---------------------------------------------------------------------------
# Apoapsis-raise Earth re-intercept (candidate sec:earth_reintercept option;
# design doc apoapsis_raise_reintercept_design.md in the paper source repo).
#
# A projectile leaves Earth at a 200 km altitude on a C3=0 escape, an Oberth
# methalox burn (Leg 1) raises its heliocentric aphelion to Q with perihelion
# pinned at 1 AU, a retrograde argon-SEP burn at apoapsis (Leg 2) lowers the
# perihelion, and it falls back to intercept Earth at 1 AU on the inbound leg.
# The single free knob Q is root-solved for phasing-exact Earth re-intercept, the
# same closure method single_impulse_resonant_dive() uses. This is the lowest-
# closing-speed member of the sec:earth_reintercept family: ~24 km/s instead of
# ~150 km/s, needing no solar dive, no gravity assist, and no off-Earth boost
# node -- only onboard propellant. The functions below reproduce the design doc's
# locked design point (Sec. 4), economics (Sec. 5), and finite-thrust check
# (Sec. 3) from the repo's own primitives.
# ---------------------------------------------------------------------------


def _wrap_to_180_deg(angle: u.Quantity) -> u.Quantity:
    """Wrap an angle into the half-open interval ``(-180, 180]`` degrees.

    Args:
        angle: The angle to wrap (astropy Quantity, angle units).

    Returns:
        The equivalent angle in ``(-180, 180]`` degrees (astropy Quantity).
    """
    deg: float = angle.to(u.deg).value
    return (((deg + 180.0) % 360.0) - 180.0) * u.deg


@dataclass(frozen=True)
class _ApoapsisRaiseGeometry:
    """Single-pass geometry of one apoapsis-raise trajectory (internal).

    Built by :func:`_apoapsis_raise_geometry` for a trial aphelion and SEP burn.
    :func:`apoapsis_raise_reintercept` drives ``phasing_residual`` to zero over the
    aphelion, then reads the closing solution's timing, perihelion, and speeds off
    the returned instance.

    Attributes:
        leg1_semimajor_axis: Semi-major axis of the outbound injection ellipse
            (astropy Quantity, AU).
        departure_burn: Methalox Oberth burn at 200 km that injects Leg 1 (astropy
            Quantity, km/s).
        truncated_perihelion: Perihelion of the post-burn orbit; the craft is
            intercepted at 1 AU and never reaches it (astropy Quantity, AU).
        transit_time: Earth-to-Earth flight time (astropy Quantity, years).
        phasing_residual: Heliocentric gap between the craft's re-crossing and
            Earth, wrapped to ``(-180, 180]`` (astropy Quantity, degrees).
        v_infinity_arrival: Hyperbolic-excess speed at Earth SOI entry (astropy
            Quantity, km/s).
        closing_speed: Closest-approach speed at 200 km, the Oberth-boosted
            ``sqrt(v_inf**2 + v_esc200**2)`` (astropy Quantity, km/s).
    """

    leg1_semimajor_axis: u.Quantity
    departure_burn: u.Quantity
    truncated_perihelion: u.Quantity
    transit_time: u.Quantity
    phasing_residual: u.Quantity
    v_infinity_arrival: u.Quantity
    closing_speed: u.Quantity


def _apoapsis_raise_geometry(
    aphelion: u.Quantity,
    sep_dv: u.Quantity,
    launch_radius: u.Quantity = EARTH_A,
) -> _ApoapsisRaiseGeometry:
    """Compute the single-pass geometry for a trial aphelion and SEP burn.

    All burns are purely tangential and Earth's orbit is treated as circular at
    ``launch_radius``. Leg 1 injects tangentially at ``launch_radius`` (perihelion)
    onto an ellipse with aphelion ``aphelion``; Leg 2 removes ``sep_dv`` of
    tangential speed at apoapsis, dropping the perihelion; the craft then falls to
    the inbound ``launch_radius`` crossing, where the relative velocity to Earth
    sets the closing speed. See the design doc's Sec. 8 methodology.

    Args:
        aphelion: Trial heliocentric aphelion Q (astropy Quantity, length units).
        sep_dv: Retrograde SEP burn at apoapsis (astropy Quantity, velocity units).
        launch_radius: Earth's heliocentric distance / the boost and re-crossing
            radius (astropy Quantity, default EARTH_A).

    Returns:
        The :class:`_ApoapsisRaiseGeometry` for this trial point.
    """
    earth_speed: u.Quantity = speed_around_attractor(a=launch_radius, attractor=Sun)
    v_esc_200: u.Quantity = escape_velocity(Earth, LEO_ALTITUDE)

    # Leg 1: purely tangential Oberth injection at launch_radius (perihelion) onto
    # an ellipse with aphelion Q. The methalox burn at 200 km supplies the
    # heliocentric excess v_inf1 = v_perihelion_1 - v_Earth.
    leg1: Orbit = orbit_from_rp_ra(
        apoapsis_radius=aphelion, periapsis_radius=launch_radius
    )
    v_perihelion_1: u.Quantity = periapsis_speed(leg1)
    v_infinity_1: u.Quantity = v_perihelion_1 - earth_speed
    departure_burn: u.Quantity = burn_for_v_infinity(
        v_infinity_1, body=Earth, altitude=LEO_ALTITUDE, initial_velocity=v_esc_200
    )

    # Leg 2: retrograde SEP burn at apoapsis. It has no radial component, so Q stays
    # an apsis; slowing there drops the apoapsis speed to va2, making Q the new
    # aphelion and lowering the perihelion, which vis-viva at apoapsis fixes.
    v_apoapsis_1: u.Quantity = apoapsis_speed(leg1)
    v_apoapsis_2: u.Quantity = v_apoapsis_1 - sep_dv
    inv_semimajor_2: u.Quantity = 2.0 / aphelion - v_apoapsis_2**2 / Sun.k
    semimajor_2: u.Quantity = 1.0 / inv_semimajor_2
    perihelion_2: u.Quantity = (2.0 * semimajor_2 - aphelion).to(u.AU)

    leg2: Orbit = orbit_from_rp_ra(
        apoapsis_radius=aphelion, periapsis_radius=perihelion_2
    )
    ecc_2: float = float(leg2.ecc.value)
    v_perihelion_2: u.Quantity = periapsis_speed(leg2)

    # True anomaly of the inbound launch_radius crossing (perihelion_2 < 1 AU < Q,
    # so 1 AU is reached on the way back in).
    nu_crossing: u.Quantity = true_anomaly_at_radius(perihelion_2, ecc_2, launch_radius)

    # Transit = leg-1 half period (perihelion -> apoapsis) plus leg-2 apoapsis ->
    # inbound crossing, the latter being P2/2 - t(perihelion -> nu_crossing).
    leg1_half_period: u.Quantity = get_period(Sun, leg1.a) / 2.0
    leg2_half_period: u.Quantity = get_period(Sun, leg2.a) / 2.0
    time_periapsis_to_crossing: u.Quantity = elliptic_time_of_flight(
        perihelion_2, ecc_2, nu_crossing, Sun
    )
    transit_time: u.Quantity = (
        leg1_half_period + leg2_half_period - time_periapsis_to_crossing
    ).to(u.year)

    # The craft sweeps 180 deg on Leg 1 and (180 deg - nu_crossing) on Leg 2, for a
    # net (360 deg - nu_crossing). Phasing closes when that equals Earth's advance.
    swept: u.Quantity = 360.0 * u.deg - nu_crossing
    earth_advance: u.Quantity = 360.0 * u.deg * (transit_time / (1.0 * u.year))
    phasing_residual: u.Quantity = _wrap_to_180_deg(swept - earth_advance)

    # Arrival velocity at the inbound crossing: tangential v_t = h / r (angular
    # momentum conserved), radial closes the vis-viva speed by Pythagoras. Subtract
    # Earth's purely tangential circular speed to get the SOI-entry excess.
    v_arrival: u.Quantity = speed_at_distance(
        radius_periapsis=perihelion_2,
        periapsis_speed=v_perihelion_2,
        distance=launch_radius,
        attractor_body=Sun,
    )
    v_tangential: u.Quantity = (v_perihelion_2 * perihelion_2 / launch_radius).to(
        u.km / u.s
    )
    v_radial: u.Quantity = np.sqrt(v_arrival**2 - v_tangential**2).to(u.km / u.s)
    v_infinity_arrival: u.Quantity = np.sqrt(
        v_radial**2 + (v_tangential - earth_speed) ** 2
    ).to(u.km / u.s)
    closing_speed: u.Quantity = speed_with_escape_energy(
        v_infinity_arrival, Earth, LEO_ALTITUDE
    )

    return _ApoapsisRaiseGeometry(
        leg1_semimajor_axis=leg1.a.to(u.AU),
        departure_burn=departure_burn.to(u.km / u.s),
        truncated_perihelion=perihelion_2,
        transit_time=transit_time,
        phasing_residual=phasing_residual.to(u.deg),
        v_infinity_arrival=v_infinity_arrival,
        closing_speed=closing_speed,
    )


@dataclass(frozen=True)
class ApoapsisRaiseReintercept:
    """Closure of the impulsive apoapsis-raise Earth re-intercept (design doc Sec. 4).

    One methalox Oberth burn at Earth injects the craft onto an ellipse whose
    aphelion ``aphelion`` is root-solved so that, after a retrograde SEP burn at
    apoapsis and the fall back inward, it re-crosses 1 AU exactly where Earth has
    moved to. Everything else follows from that aphelion.

    Attributes:
        aphelion: The phasing-exact heliocentric aphelion Q (astropy Quantity, AU).
        leg1_semimajor_axis: Semi-major axis of the injection ellipse (astropy
            Quantity, AU).
        departure_burn: Methalox Oberth burn at 200 km, Delta-v1 (astropy Quantity,
            km/s).
        sep_burn: Retrograde argon-SEP burn at apoapsis, Delta-v2 (astropy
            Quantity, km/s).
        methalox_mass_fraction: Mass fraction retained after the methalox burn.
        sep_mass_fraction: Mass fraction retained after the SEP burn.
        combined_dry_fraction: Product of the two retained fractions -- the dry
            mass reaching Earth per unit departing mass.
        truncated_perihelion: Perihelion of the post-burn orbit; never reached, the
            craft is intercepted at 1 AU (astropy Quantity, AU).
        v_infinity_arrival: Hyperbolic-excess speed at Earth SOI entry (astropy
            Quantity, km/s).
        closing_speed: Closest-approach speed at 200 km, v_close200 (astropy
            Quantity, km/s).
        twice_escape_target: The 2x escape-velocity design bar (astropy Quantity,
            km/s).
        transit_time: Earth-to-Earth flight time (astropy Quantity, years).
        phasing_residual: Residual heliocentric miss at closure (astropy Quantity,
            degrees), driven to ~0 by the root solve.
    """

    aphelion: u.Quantity
    leg1_semimajor_axis: u.Quantity
    departure_burn: u.Quantity
    sep_burn: u.Quantity
    methalox_mass_fraction: float
    sep_mass_fraction: float
    combined_dry_fraction: float
    truncated_perihelion: u.Quantity
    v_infinity_arrival: u.Quantity
    closing_speed: u.Quantity
    twice_escape_target: u.Quantity
    transit_time: u.Quantity
    phasing_residual: u.Quantity


def apoapsis_raise_reintercept(
    sep_dv: u.Quantity = APOAPSIS_RAISE_SEP_DV,
    launch_radius: u.Quantity = EARTH_A,
    aphelion_bracket: Tuple[float, float] = APOAPSIS_RAISE_APHELION_BRACKET,
) -> ApoapsisRaiseReintercept:
    """Solve the aphelion that makes an apoapsis-raise dive re-intercept Earth.

    The phasing residual (:func:`_apoapsis_raise_geometry`) changes sign across the
    lone phasing-exact aphelion inside ``aphelion_bracket`` (~2.26 AU for the
    default 4 km/s SEP burn), so this roots that condition rather than hardcoding
    it, mirroring :func:`single_impulse_resonant_dive`. Raising ``sep_dv`` deepens
    the closing speed but barely moves the aphelion, because the phasing balance is
    dominated by the long Leg-1 arc that the apoapsis kick hardly perturbs. The
    default reproduces the design doc's locked point: Q ~ 2.26 AU, Delta-v1 ~ 1.20
    km/s, v_close200 ~ 24.06 km/s, transit ~ 1.69 yr.

    Args:
        sep_dv: Retrograde SEP burn at apoapsis (astropy Quantity, default
            APOAPSIS_RAISE_SEP_DV = 4 km/s).
        launch_radius: Earth's heliocentric distance / the boost and re-crossing
            radius (astropy Quantity, default EARTH_A).
        aphelion_bracket: ``(low, high)`` aphelion bracket in AU for the root solve
            (default APOAPSIS_RAISE_APHELION_BRACKET).

    Returns:
        The :class:`ApoapsisRaiseReintercept` closing solution.
    """

    def residual_deg(aphelion_au: float) -> float:
        """Phasing residual (deg) for a trial aphelion, for the root solve."""
        geometry = _apoapsis_raise_geometry(aphelion_au * u.AU, sep_dv, launch_radius)
        return float(geometry.phasing_residual.to(u.deg).value)

    aphelion: u.Quantity = (
        brentq(residual_deg, aphelion_bracket[0], aphelion_bracket[1]) * u.AU
    )
    geometry = _apoapsis_raise_geometry(aphelion, sep_dv, launch_radius)

    # Rocket-equation mass fractions retained (exp(-dv/v_exh) = 1 - propellant
    # fraction), one per propulsion type.
    methalox_exhaust: u.Quantity = exhaust_velocity_from_isp(METHALOX_VACUUM_ISP)
    argon_exhaust: u.Quantity = exhaust_velocity_from_isp(ARGON_SEP_ISP)
    methalox_mass_fraction: float = float(
        1.0 - rocket_equation(geometry.departure_burn, methalox_exhaust)
    )
    sep_mass_fraction: float = float(1.0 - rocket_equation(sep_dv, argon_exhaust))
    combined_dry_fraction: float = methalox_mass_fraction * sep_mass_fraction
    twice_escape_target: u.Quantity = 2.0 * escape_velocity(Earth, LEO_ALTITUDE)

    return ApoapsisRaiseReintercept(
        aphelion=aphelion.to(u.AU),
        leg1_semimajor_axis=geometry.leg1_semimajor_axis,
        departure_burn=geometry.departure_burn,
        sep_burn=sep_dv.to(u.km / u.s),
        methalox_mass_fraction=methalox_mass_fraction,
        sep_mass_fraction=sep_mass_fraction,
        combined_dry_fraction=combined_dry_fraction,
        truncated_perihelion=geometry.truncated_perihelion,
        v_infinity_arrival=geometry.v_infinity_arrival,
        closing_speed=geometry.closing_speed,
        twice_escape_target=twice_escape_target.to(u.km / u.s),
        transit_time=geometry.transit_time,
        phasing_residual=geometry.phasing_residual,
    )


@dataclass(frozen=True)
class ApoapsisRaiseEconomics:
    """Growth economics of the apoapsis-raise cycle (design doc Sec. 5).

    Uses ``eq:PuffSat_ratio`` (:func:`payload_mass_ratio`, f = 0.8) with v_rf = the
    200 km Earth escape speed (the departure condition for the next cycle) and v_p =
    the closing speed. The net growth per cycle is that ratio times the dry-mass
    fraction that survives both burns to reach Earth.

    Attributes:
        payload_puffsat_mass_ratio: New payload pushed to escape per unit returning
            PuffSat mass, m_r / m_p.
        net_growth_per_cycle: payload_puffsat_mass_ratio times the combined dry
            fraction -- the cascade's per-cycle multiplier.
        cycle_time: One Earth-to-Earth cycle (astropy Quantity, years).
        cycles_to_millionfold: Cycles to multiply launch capacity a millionfold.
        time_to_millionfold: cycles_to_millionfold times the cycle time (astropy
            Quantity, years).
        doublings_per_year: Payload doublings per year at this growth and cadence.
    """

    payload_puffsat_mass_ratio: float
    net_growth_per_cycle: float
    cycle_time: u.Quantity
    cycles_to_millionfold: float
    time_to_millionfold: u.Quantity
    doublings_per_year: float


def apoapsis_raise_economics(
    reintercept: Optional[ApoapsisRaiseReintercept] = None,
    target_multiple: float = TARGET_LAUNCH_CAPACITY_MULTIPLE,
) -> ApoapsisRaiseEconomics:
    """Compute the apoapsis-raise growth economics (design doc Sec. 5).

    Args:
        reintercept: The closing solution to score; solved with defaults when None.
        target_multiple: Launch-capacity growth target (default
            TARGET_LAUNCH_CAPACITY_MULTIPLE = 1e6).

    Returns:
        The :class:`ApoapsisRaiseEconomics` for the cycle. With the default design
        point this reproduces the doc's m_r/m_p ~ 2.62, net growth ~ 1.547x, and
        ~54 yr to a millionfold.
    """
    if reintercept is None:
        reintercept = apoapsis_raise_reintercept()
    # v_rf is the 200 km Earth escape speed (self-consistent with the departure
    # condition for the next cycle); the doc rounds this ~11.01 km/s to 11.0.
    v_rf: u.Quantity = escape_velocity(Earth, LEO_ALTITUDE)
    # payload_mass_ratio returns a dimensionless Quantity; float() it so the
    # economics fields (and any downstream pytest.approx) are plain floats.
    mass_ratio: float = float(
        payload_mass_ratio(v_rf=v_rf, v_b=reintercept.closing_speed)
    )
    net_growth: float = mass_ratio * reintercept.combined_dry_fraction
    cycle_time: u.Quantity = reintercept.transit_time
    cycles_to_millionfold: float = float(np.log(target_multiple) / np.log(net_growth))
    time_to_millionfold: u.Quantity = launch_capacity_time(
        capacity_multiple_per_loop=net_growth,
        one_loop_elapsed_time=cycle_time,
        target_launch_capacity_multiple=target_multiple,
    )
    doublings_per_year: float = float(np.log2(net_growth) / cycle_time.to(u.year).value)
    return ApoapsisRaiseEconomics(
        payload_puffsat_mass_ratio=mass_ratio,
        net_growth_per_cycle=net_growth,
        cycle_time=cycle_time.to(u.year),
        cycles_to_millionfold=cycles_to_millionfold,
        time_to_millionfold=time_to_millionfold,
        doublings_per_year=doublings_per_year,
    )


@dataclass(frozen=True)
class ApoapsisRaiseFiniteBurn:
    """Finite-thrust check of the apoapsis SEP burn (design doc Sec. 3).

    A 2D finite-thrust propagation (constant thrust held anti-velocity, mass loss
    by the rocket equation, the burn centered on apoapsis) of the same locked
    design point, compared to the instantaneous kick. A 60-90 day burn reproduces
    the impulsive closing speed to within ~1%.

    Attributes:
        burn_duration: The SEP burn duration modeled (astropy Quantity, days).
        closing_speed: Finite-burn closest-approach speed at 200 km (astropy
            Quantity, km/s).
        transit_time: Finite-burn Earth-to-Earth flight time (astropy Quantity,
            years).
        truncated_perihelion: Perihelion of the finite-burn post-burn orbit
            (astropy Quantity, AU).
        phasing_residual: Finite-burn heliocentric miss at the 1 AU crossing
            (astropy Quantity, degrees).
        closing_speed_error: Fractional difference of the finite-burn closing speed
            from the impulsive one (dimensionless).
    """

    burn_duration: u.Quantity
    closing_speed: u.Quantity
    transit_time: u.Quantity
    truncated_perihelion: u.Quantity
    phasing_residual: u.Quantity
    closing_speed_error: float


def apoapsis_raise_finite_burn(
    burn_duration: u.Quantity = APOAPSIS_RAISE_SEP_BURN_DURATION,
    sep_dv: u.Quantity = APOAPSIS_RAISE_SEP_DV,
    reintercept: Optional[ApoapsisRaiseReintercept] = None,
) -> ApoapsisRaiseFiniteBurn:
    """Propagate the apoapsis SEP burn as finite thrust and compare to the impulse.

    Integrates the heliocentric plane trajectory in three segments -- coast to just
    before apoapsis, a constant-thrust retrograde burn centered on apoapsis, then a
    coast to the inbound 1 AU crossing -- reusing the impulsive design point's
    aphelion. The closing speed, transit, truncated perihelion, and phasing are read
    off the arrival state and compared with the impulsive solution. Centering the
    burn on apoapsis is the one requirement for the impulsive approximation to hold
    (design doc Sec. 3).

    Args:
        burn_duration: SEP burn duration (astropy Quantity, default
            APOAPSIS_RAISE_SEP_BURN_DURATION = 90 days).
        sep_dv: Retrograde SEP burn magnitude (astropy Quantity, default
            APOAPSIS_RAISE_SEP_DV = 4 km/s).
        reintercept: The impulsive design point to check against; solved with the
            given ``sep_dv`` when None.

    Returns:
        The :class:`ApoapsisRaiseFiniteBurn` result and its fractional closing-speed
        error versus the impulsive kick.
    """
    if reintercept is None:
        reintercept = apoapsis_raise_reintercept(sep_dv=sep_dv)

    mu: float = Sun.k.to(u.km**3 / u.s**2).value
    au_km: float = (1.0 * u.AU).to(u.km).value
    year_s: float = (1.0 * u.year).to(u.s).value
    earth_speed: float = (
        speed_around_attractor(a=EARTH_A, attractor=Sun).to(u.km / u.s).value
    )
    v_esc_200: float = escape_velocity(Earth, LEO_ALTITUDE).to(u.km / u.s).value

    leg1: Orbit = orbit_from_rp_ra(
        apoapsis_radius=reintercept.aphelion, periapsis_radius=EARTH_A
    )
    v_perihelion_1: float = periapsis_speed(leg1).to(u.km / u.s).value
    leg1_semimajor_km: float = leg1.a.to(u.km).value
    leg1_period: float = 2.0 * np.pi * np.sqrt(leg1_semimajor_km**3 / mu)

    exhaust: float = exhaust_velocity_from_isp(ARGON_SEP_ISP).to(u.km / u.s).value
    burn_dv: float = sep_dv.to(u.km / u.s).value
    burn_seconds: float = burn_duration.to(u.s).value
    retained_fraction: float = float(np.exp(-burn_dv / exhaust))
    # Mass flow per unit initial mass (normalized m0 = 1); constant thrust.
    mass_flow: float = (1.0 - retained_fraction) / burn_seconds

    def gravity_rhs(t: float, state: npt.NDArray[np.float64]) -> List[float]:
        """Two-body gravitational RHS in the heliocentric plane (km, s)."""
        x, y, vx, vy, _mass = state
        r_cubed = (x * x + y * y) ** 1.5
        return [vx, vy, -mu * x / r_cubed, -mu * y / r_cubed, 0.0]

    def burn_rhs(t: float, state: npt.NDArray[np.float64]) -> List[float]:
        """Gravity plus constant anti-velocity thrust; mass shrinks at mass_flow."""
        x, y, vx, vy, mass = state
        r_cubed = (x * x + y * y) ** 1.5
        speed = np.hypot(vx, vy)
        thrust_acc = mass_flow * exhaust / mass
        return [
            vx,
            vy,
            -mu * x / r_cubed - thrust_acc * vx / speed,
            -mu * y / r_cubed - thrust_acc * vy / speed,
            -mass_flow,
        ]

    def crosses_launch_radius(t: float, state: npt.NDArray[np.float64]) -> float:
        """Zero when the craft is at 1 AU; fired inbound (direction < 0), terminal."""
        return float(np.hypot(state[0], state[1]) - au_km)

    setattr(crosses_launch_radius, "terminal", True)
    setattr(crosses_launch_radius, "direction", -1.0)

    # Depart at perihelion (angle 180 deg, on -x) moving prograde (counterclockwise),
    # so apoapsis falls at angle 0 after half a period. Center the burn on apoapsis.
    state0: npt.NDArray[np.float64] = np.array([-au_km, 0.0, 0.0, -v_perihelion_1, 1.0])
    coast_to_burn: float = leg1_period / 2.0 - burn_seconds / 2.0

    coast = solve_ivp(
        gravity_rhs,
        (0.0, coast_to_burn),
        state0,
        method="DOP853",
        rtol=1e-10,
        atol=1e-6,
    )
    burn = solve_ivp(
        burn_rhs,
        (0.0, burn_seconds),
        coast.y[:, -1],
        method="DOP853",
        rtol=1e-10,
        atol=1e-6,
    )
    fall = solve_ivp(
        gravity_rhs,
        (0.0, 4.0 * year_s),
        burn.y[:, -1],
        method="DOP853",
        rtol=1e-10,
        atol=1e-6,
        events=crosses_launch_radius,
    )

    arrival: npt.NDArray[np.float64] = fall.y_events[0][0]
    arrival_time: float = coast_to_burn + burn_seconds + float(fall.t_events[0][0])

    x, y, vx, vy, _mass = arrival
    radius = np.hypot(x, y)
    v_radial = (vx * x + vy * y) / radius
    v_tangential = (-vx * y + vy * x) / radius
    v_infinity = np.hypot(v_radial, v_tangential - earth_speed)
    closing_speed = float(np.hypot(v_infinity, v_esc_200))

    # Truncated perihelion from the arrival state vector (energy + angular momentum).
    speed_squared = vx * vx + vy * vy
    specific_energy = speed_squared / 2.0 - mu / radius
    semimajor = -mu / (2.0 * specific_energy)
    angular_momentum = x * vy - y * vx
    ecc = np.sqrt(1.0 + 2.0 * specific_energy * angular_momentum**2 / mu**2)
    perihelion_km = float(semimajor * (1.0 - ecc))

    swept = (np.degrees(np.arctan2(y, x)) - 180.0) % 360.0
    earth_advance = 360.0 * (arrival_time / year_s)
    residual = ((swept - earth_advance + 180.0) % 360.0) - 180.0

    impulsive_closing: float = reintercept.closing_speed.to(u.km / u.s).value
    closing_speed_error: float = float(
        abs(closing_speed - impulsive_closing) / impulsive_closing
    )

    return ApoapsisRaiseFiniteBurn(
        burn_duration=burn_duration.to(u.day),
        closing_speed=closing_speed * u.km / u.s,
        transit_time=(arrival_time * u.s).to(u.year),
        truncated_perihelion=(perihelion_km * u.km).to(u.AU),
        phasing_residual=float(residual) * u.deg,
        closing_speed_error=closing_speed_error,
    )
