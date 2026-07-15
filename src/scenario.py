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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from boinor.bodies import Body, Earth, Jupiter, Mars, Moon, Saturn, Sun, Venus
from boinor.core.iod import izzo
from boinor.twobody import Orbit
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, differential_evolution

from src.astro_constants import (
    APOAPSIS_RAISE_APHELION_BRACKET,
    APOAPSIS_RAISE_SEP_BURN_DURATION,
    APOAPSIS_RAISE_SEP_DV,
    ARGON_SEP_ISP,
    ASSIST_CHAIN_BURN_CANDIDATES,
    ASSIST_CHAIN_MAX_FLYBYS,
    ASSIST_CHAIN_MAX_TRIP_TIME,
    ASSIST_CHAIN_PHASING_BUDGET,
    EARTH_A,
    EFFECTIVE_DV_LUNAR,
    JUPITER_A,
    JUPITER_FLYBY_MAX_TOF,
    JUPITER_FLYBY_VB_TRADE_TARGETS,
    LEO_ALTITUDE,
    LOW_ASSIST_FLYBY_ALTITUDE,
    LOW_JUPITER_ALTITUDE,
    LOW_SATURN_ALTITUDE,
    MARS_A,
    METHALOX_SEA_LEVEL_ISP,
    METHALOX_VACUUM_ISP,
    MOON_A,
    PARKER_PERIAPSIS,
    PERIAPSIS_SOLAR_BURN,
    PERIAPSIS_SOLAR_V,
    PHOEBE_A,
    PUFFSAT_CYCLE_ORBIT_PERIOD,
    REQUIRED_DV_LUNAR_TRANSFER_PROGRADE,
    REQUIRED_DV_LUNAR_TRANSFER_RETROGRADE,
    RETROGRADE_FRACTION,
    SOLAR_DIVE_PERIAPSIS_BURN,
    SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
    STD_FUDGE_FACTOR,
    SUBORBITAL_DV_TO_200KM,
    TARGET_LAUNCH_CAPACITY_MULTIPLE,
    TWO_IMPULSE_DIP_PERIAPSIS,
    VENUS_A,
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


def lunar_transfer_periapsis_speed() -> u.Quantity:
    """Speed at the 200 km Earth periapsis of the LEO-to-Moon transfer ellipse.

    The `v_rf` of the ``sec:jupiter_only_growth`` catalog row and the push target
    the powered Jovian flyby's end-to-end mass ratio is scored against.

    Returns:
        The periapsis speed of the lunar transfer orbit (astropy Quantity, km/s).
    """
    return periapsis_speed(
        orbit=orbit_from_rp_ra(
            apoapsis_radius=MOON_A,
            periapsis_radius=Earth.R + LEO_ALTITUDE,
            attractor_body=Earth,
        )
    )


def parker_injection_burns() -> Tuple[u.Quantity, u.Quantity]:
    """Burns to inject a payload from Earth toward a Parker-class periapsis.

    The `v_rf` values of the two ``sec:jupiter_gravity_initial`` catalog rows: the
    delta-v at 200 km that sends the payload onto a prograde (respectively
    retrograde) heliocentric transfer whose aphelion is 1 AU and whose periapsis
    is the Parker Space Probe's.

    Returns:
        Tuple of (prograde burn, retrograde burn) (astropy Quantities, km/s).
    """
    parker_orbit = orbit_from_rp_ra(
        apoapsis_radius=EARTH_A, periapsis_radius=PARKER_PERIAPSIS
    )
    parker_apoapsis_speed = apoapsis_speed(orbit=parker_orbit)
    earth_speed = speed_around_attractor(a=EARTH_A, attractor=Sun)
    # v_infinity for prograde transfer from Earth to Parker orbit apoapsis
    prograde = burn_for_v_infinity(earth_speed - parker_apoapsis_speed)
    # v_infinity for retrograde transfer from Earth to Parker orbit apoapsis
    retrograde = burn_for_v_infinity(earth_speed + parker_apoapsis_speed)
    return prograde, retrograde


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
    lunar_periapsis_speed = lunar_transfer_periapsis_speed()
    leo_speed = speed_around_attractor(a=low_earth_periapsis, attractor=Earth)

    prograde_dv_parker_burn, retrograde_dv_parker_burn = parker_injection_burns()
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
            v_b=lunar_periapsis_speed,
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
            v_b=-lunar_periapsis_speed,
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
            v_rf=lunar_periapsis_speed,
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
    """Phased Earth-return dive scenarios (resonant dive: paper Appendix
    sec:earth_reintercept; apoapsis-raise: sec:apoapsis_raise_loop).

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
    is scored on its own collision rather than off the ~69 km/s PuffSat. It now has
    its own paper subsection, "Apoapsis-Raise Return With Onboard Burns"
    (sec:apoapsis_raise_loop), folded in under Jupiter-Only Exponential Launch
    Growth rather than the sec:earth_reintercept appendix that Row 1 cites.

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
            desc="""Apoapsis-raise Earth re-intercept: a methalox Oberth burn raises heliocentric aphelion to ~2.26 AU, one retrograde argon-SEP burn at apoapsis lowers perihelion, and the craft falls back to intercept Earth at ~24 km/s after ~1.69 yr, pushing the next payload to Earth escape. The gentlest member of the family -- no solar dive, no gravity assist, no off-Earth boost node, only onboard propellant""",
            paper_ref="sec:apoapsis_raise_loop",
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


# ---------------------------------------------------------------------------
# Powered Jovian flyby retrograde return (planned subsection under
# sec:jupiter_only_growth; ADR 0002-jupiter-flyby-objective; CONTEXT.md
# "Jupiter powered-flyby retrograde return").
#
# The leg the catalog's Jovian-return rows assume but never derive: an Oberth
# methalox burn at 200 km above Earth (from C3 = 0, PuffSat-provided), a coast
# to Jupiter, and a powered gravity assist -- a second impulsive methalox burn
# at Jovian periapsis -- that bends and pumps the trajectory into a retrograde
# heliocentric return crossing 1 AU. Planar patched conic; circular coplanar
# planet orbits; velocities in the local (tangential, radial-outward) basis.
#
# "Minimize total delta-v" is ill-posed here (the retrograde-plunge degeneracy,
# see the ADR), so the optimizer maximizes the end-to-end mass ratio instead:
# delivered mass fraction x payload mass ratio at the achieved collision speed.
# The internal helpers work in floats (km, s, km/s) for optimizer speed, like
# apoapsis_raise_finite_burn; Quantities appear only at the public boundary.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FlybyParams:
    """Float-valued physical inputs of the powered-flyby search (km, s, km/s).

    Attributes:
        mu_sun: Sun gravitational parameter (km^3/s^2).
        mu_jupiter: Jupiter gravitational parameter (km^3/s^2).
        r_earth_orbit: Earth heliocentric orbit radius, 1 AU (km).
        r_jupiter_orbit: Jupiter heliocentric orbit radius (km).
        v_earth_orbit: Earth circular heliocentric speed (km/s).
        v_jupiter_orbit: Jupiter circular heliocentric speed (km/s).
        v_esc_leo: Earth escape speed at the 200 km burn altitude (km/s). Sets
            the Oberth leverage: leaving with excess ``w`` needs periapsis speed
            ``hypot(w, v_esc_leo)``.
        v_depart_from: Speed the mass already carries at that periapsis when the
            departure burn lights, so the burn is
            ``hypot(w, v_esc_leo) - v_depart_from``. Equals ``v_esc_leo`` under
            the legacy C3 = 0 convention, or the cycle orbit's periapsis speed
            when the growth loop is closed (see
            :func:`puffsat_cycle_periapsis_speed`).
        v_esc_surface: Earth surface escape speed -- the collision-speed
            convention of retrograde_jovian_hohmann_transfer (km/s).
        periapsis_floor: Minimum Jovian flyby periapsis radius, center-based (km).
        exhaust_speed: Methalox vacuum effective exhaust speed (km/s).
        v_rf: Push target of the growth row the mass ratio is scored on (km/s).
        max_tof: Cap on outbound + return heliocentric time of flight (s).
    """

    mu_sun: float
    mu_jupiter: float
    r_earth_orbit: float
    r_jupiter_orbit: float
    v_earth_orbit: float
    v_jupiter_orbit: float
    v_esc_leo: float
    v_depart_from: float
    v_esc_surface: float
    periapsis_floor: float
    exhaust_speed: float
    v_rf: float
    max_tof: float


def puffsat_cycle_periapsis_speed(
    period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD,
    altitude: u.Quantity = LEO_ALTITUDE,
) -> u.Quantity:
    """Periapsis speed of the bound orbit the PuffSat collision leaves behind.

    The growth loop closes on itself here. A returning PuffSat pushes the mass to
    just under Earth escape, which puts it on a bound ellipse of the given period
    with periapsis at ``altitude``; the mass coasts to apoapsis, falls back, and
    the next cycle's Oberth burn fires at that periapsis. So this one speed is
    both the collision's push target (``v_rf`` of ``eq:PuffSat_ratio``) and the
    speed the next departure burn starts from -- they are the same state, and
    modelling them as one number is what makes the cycle self-consistent.

    Args:
        period: Orbital period of the post-collision ellipse (astropy Quantity).
        altitude: Periapsis altitude above Earth's surface (astropy Quantity).

    Returns:
        The periapsis speed (astropy Quantity, km/s). ~10.9503 km/s at the
        default 20 days, 58.3 m/s below the 11.0086 km/s escape speed there.
    """
    mu = Earth.k.to(u.km**3 / u.s**2)
    r_p = (Earth.R + altitude).to(u.km)
    semi_major_axis = np.cbrt(mu * period.to(u.s) ** 2 / (4.0 * np.pi**2))
    return np.sqrt(mu * (2.0 / r_p - 1.0 / semi_major_axis)).to(u.km / u.s)


@dataclass(frozen=True)
class CycleGrowth:
    """Growth economics of one closed Earth-to-Earth PuffSat cycle.

    Attributes:
        mass_ratio: New payload pushed to ``v_rf`` per unit returning PuffSat
            mass, ``eq:PuffSat_ratio`` at the achieved collision speed.
        delivered_fraction: Mass fraction surviving every burn of the chain.
        net_growth: mass_ratio x delivered_fraction, the per-cycle multiplier.
        cycle_time: Departure to the next departure (astropy Quantity, years),
            including the coast back to periapsis.
        doubling_time: Time to double the payload (astropy Quantity, years);
            infinite when the cycle does not grow.
    """

    mass_ratio: float
    delivered_fraction: float
    net_growth: float
    cycle_time: u.Quantity
    doubling_time: u.Quantity


def puffsat_cycle_growth(
    total_dv: u.Quantity,
    collision_speed: u.Quantity,
    trip_time: u.Quantity,
    cycle_period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD,
    specific_impulse: u.Quantity = METHALOX_VACUUM_ISP,
) -> CycleGrowth:
    """Score a chain on doubling time, the growth loop's real objective.

    Total delta-v is the wrong thing to minimize on its own: it buys cheapness
    with trip time, and the loop is paid in payload per *year*. A chain that
    halves its delta-v while doubling its cycle is a worse machine. This trades
    the two properly, via ``cycle x ln2 / ln(net growth)``.

    The cycle is closed: the collision pushes the mass to ``v_rf``, the periapsis
    speed of the ``cycle_period`` orbit, and the next departure burn starts from
    that same speed one coast later (see :func:`puffsat_cycle_periapsis_speed`).

    Args:
        total_dv: Summed delta-v of every burn in the chain (astropy Quantity).
        collision_speed: Achieved v_b of the returning PuffSat (astropy Quantity).
        trip_time: Departure to the 1 AU return crossing (astropy Quantity).
        cycle_period: Period of the post-collision orbit (astropy Quantity).
        specific_impulse: Isp charged to every burn (astropy Quantity).

    Returns:
        The :class:`CycleGrowth`. ``doubling_time`` is ``inf`` years when
        ``net_growth <= 1``, i.e. the cycle shrinks and never doubles.
    """
    v_rf = puffsat_cycle_periapsis_speed(period=cycle_period)
    if collision_speed <= v_rf:
        raise ValueError(
            f"collision speed {collision_speed} does not exceed the push target "
            f"{v_rf}; the PuffSat cannot drive the mass to the cycle orbit"
        )
    mass_ratio = float(payload_mass_ratio(v_rf=v_rf, v_b=collision_speed))
    exhaust = exhaust_velocity_from_isp(specific_impulse)
    delivered = float(np.exp(-(total_dv / exhaust).to_value(u.dimensionless_unscaled)))
    net_growth = mass_ratio * delivered
    # The mass coasts one full period from the collision at periapsis, out to
    # apoapsis and back, before the next burn can fire.
    cycle_time = (trip_time + cycle_period).to(u.year)
    if net_growth <= 1.0:
        doubling_time = np.inf * u.year
    else:
        doubling_time = cycle_time * float(np.log(2.0) / np.log(net_growth))
    return CycleGrowth(
        mass_ratio=mass_ratio,
        delivered_fraction=delivered,
        net_growth=net_growth,
        cycle_time=cycle_time,
        doubling_time=doubling_time,
    )


def _powered_flyby_params(
    max_total_tof: u.Quantity = JUPITER_FLYBY_MAX_TOF,
    periapsis_floor_altitude: u.Quantity = LOW_JUPITER_ALTITUDE,
    cycle_periapsis_speed: Optional[u.Quantity] = None,
) -> _FlybyParams:
    """Build the float parameter block for the powered-flyby search.

    Args:
        max_total_tof: Cap on outbound + return time of flight (astropy Quantity).
        periapsis_floor_altitude: Minimum flyby altitude above Jupiter's 1-bar
            level (astropy Quantity).
        cycle_periapsis_speed: Speed the departure burn starts from, and the
            collision's push target -- see :func:`puffsat_cycle_periapsis_speed`.
            When None the legacy convention is kept: depart from escape at 200 km
            (C3 = 0) and score the mass ratio against the lunar-transfer periapsis
            speed. Those are two different states, so the legacy pair is not a
            closed cycle; pass the cycle speed to close it.

    Returns:
        The :class:`_FlybyParams` with everything converted to km / s / km/s.
    """
    mu_sun = float(Sun.k.to_value(u.km**3 / u.s**2))
    r_earth_orbit = float(EARTH_A.to_value(u.km))
    r_jupiter_orbit = float(JUPITER_A.to_value(u.km))
    v_esc_leo = float(escape_velocity(Earth, LEO_ALTITUDE).to_value(u.km / u.s))
    if cycle_periapsis_speed is None:
        v_depart_from = v_esc_leo
        v_rf = float(lunar_transfer_periapsis_speed().to_value(u.km / u.s))
    else:
        v_depart_from = float(cycle_periapsis_speed.to_value(u.km / u.s))
        v_rf = v_depart_from
    return _FlybyParams(
        mu_sun=mu_sun,
        mu_jupiter=float(Jupiter.k.to_value(u.km**3 / u.s**2)),
        r_earth_orbit=r_earth_orbit,
        r_jupiter_orbit=r_jupiter_orbit,
        v_earth_orbit=float(np.sqrt(mu_sun / r_earth_orbit)),
        v_jupiter_orbit=float(np.sqrt(mu_sun / r_jupiter_orbit)),
        v_esc_leo=v_esc_leo,
        v_depart_from=v_depart_from,
        v_esc_surface=float(escape_velocity(Earth).to_value(u.km / u.s)),
        periapsis_floor=float((Jupiter.R + periapsis_floor_altitude).to_value(u.km)),
        exhaust_speed=float(
            exhaust_velocity_from_isp(METHALOX_VACUUM_ISP).to_value(u.km / u.s)
        ),
        v_rf=v_rf,
        max_tof=float(max_total_tof.to_value(u.s)),
    )


def _elliptic_tof_seconds(mu: float, a: float, ecc: float, nu: float) -> float:
    """Time from periapsis to true anomaly ``nu`` (rad, [0, pi]) on an ellipse.

    Float twin of :func:`orbit_utils.elliptic_time_of_flight` for the optimizer's
    hot loop.

    Args:
        mu: Gravitational parameter (km^3/s^2).
        a: Semi-major axis (km).
        ecc: Eccentricity, 0 <= ecc < 1.
        nu: True anomaly from periapsis (rad, in [0, pi]).

    Returns:
        The time of flight (s).
    """
    eccentric_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(nu / 2.0),
        np.sqrt(1.0 + ecc) * np.cos(nu / 2.0),
    )
    mean_anomaly = eccentric_anomaly - ecc * np.sin(eccentric_anomaly)
    return float(mean_anomaly / np.sqrt(mu / a**3))


def _hyperbolic_tof_seconds(mu: float, a_abs: float, ecc: float, nu: float) -> float:
    """Time from periapsis to true anomaly ``nu`` (rad, [0, pi)) on a hyperbola.

    Float twin of :func:`orbit_utils.hyperbolic_time_of_flight` for the
    optimizer's hot loop.

    Args:
        mu: Gravitational parameter (km^3/s^2).
        a_abs: Absolute semi-major axis |a| (km).
        ecc: Eccentricity, ecc > 1.
        nu: True anomaly from periapsis (rad), within the reachable branch.

    Returns:
        The time of flight (s).
    """
    cosh_f = (ecc + np.cos(nu)) / (1.0 + ecc * np.cos(nu))
    hyperbolic_anomaly = float(np.arccosh(max(cosh_f, 1.0)))
    mean_anomaly = ecc * np.sinh(hyperbolic_anomaly) - hyperbolic_anomaly
    return float(mean_anomaly / np.sqrt(mu / a_abs**3))


def _true_anomaly_at_radius_rad(p: float, ecc: float, r: float) -> Optional[float]:
    """Principal true anomaly (rad, [0, pi]) where a conic reaches radius ``r``.

    Args:
        p: Semi-latus rectum (km).
        ecc: Eccentricity (> 0).
        r: Radius to reach (km).

    Returns:
        The true anomaly in [0, pi], or None if the radius is unreachable.
    """
    cos_nu = (p / r - 1.0) / ecc
    if abs(cos_nu) > 1.0 + 1e-9:
        return None
    return float(np.arccos(np.clip(cos_nu, -1.0, 1.0)))


@dataclass(frozen=True)
class _ReturnLeg:
    """Float summary of the retrograde Jupiter-to-1 AU return leg.

    Attributes:
        perihelion: Perihelion radius of the return orbit (km).
        tof: Time of flight from the flyby to the first inbound 1 AU crossing (s).
        closing_speed: Earth-relative speed at the 1 AU crossing (km/s).
        collision_speed: Closing speed folded through Earth's gravity well to the
            surface-escape convention of retrograde_jovian_hohmann_transfer (km/s).
    """

    perihelion: float
    tof: float
    closing_speed: float
    collision_speed: float


def _flyby_return_leg(
    v_tangential: float, v_radial: float, params: _FlybyParams
) -> Optional[_ReturnLeg]:
    """Score the heliocentric return from Jupiter's orbit radius to 1 AU.

    Takes the post-flyby heliocentric velocity at ``r_jupiter_orbit`` in the
    (tangential, radial-outward) basis. The orbit must be retrograde
    (``v_tangential < 0``) and cross 1 AU; the leg ends at the first *inbound*
    1 AU crossing. Internally the orbit is mirrored to a prograde twin (radii
    and times are unchanged) so the conic formulas keep a positive angular
    momentum.

    Args:
        v_tangential: Post-flyby tangential heliocentric speed (km/s, negative
            for retrograde).
        v_radial: Post-flyby radial-outward heliocentric speed (km/s).
        params: The float parameter block.

    Returns:
        The :class:`_ReturnLeg`, or None if the state is not retrograde, never
        crosses 1 AU, or never comes back inward.
    """
    if v_tangential >= 0.0:
        return None
    mu = params.mu_sun
    r_j = params.r_jupiter_orbit
    r_e = params.r_earth_orbit
    # Mirrored prograde twin: same radii and times, positive angular momentum.
    v_t = -v_tangential
    v_r = v_radial
    energy = (v_t * v_t + v_r * v_r) / 2.0 - mu / r_j
    h = r_j * v_t
    p = h * h / mu
    ecc = float(np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / (mu * mu))))
    if abs(ecc - 1.0) < 1e-9 or ecc < 1e-9:
        return None
    perihelion = p / (1.0 + ecc)
    if perihelion > r_e:
        return None
    nu_jupiter = _true_anomaly_at_radius_rad(p, ecc, r_j)
    nu_earth = _true_anomaly_at_radius_rad(p, ecc, r_e)
    if nu_jupiter is None or nu_earth is None:
        return None
    if ecc > 1.0:
        # Hyperbolic return: only an already-inbound state ever re-crosses 1 AU.
        if v_r > 0.0:
            return None
        a_abs = p / (ecc * ecc - 1.0)
        tof = _hyperbolic_tof_seconds(mu, a_abs, ecc, nu_jupiter)
        tof -= _hyperbolic_tof_seconds(mu, a_abs, ecc, nu_earth)
    else:
        a = p / (1.0 - ecc * ecc)
        period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
        t_jupiter = _elliptic_tof_seconds(mu, a, ecc, nu_jupiter)
        t_earth = _elliptic_tof_seconds(mu, a, ecc, nu_earth)
        if v_r > 0.0:
            # Out to aphelion first, then back in through 1 AU.
            tof = (period - t_earth) - t_jupiter
        else:
            tof = t_jupiter - t_earth
    if tof <= 0.0:
        return None
    v1_sq = 2.0 * (energy + mu / r_e)
    v_t1 = h / r_e
    v_r1_sq = max(0.0, v1_sq - v_t1 * v_t1)
    # Un-mirrored the tangential speed is -v_t1, so the Earth-relative closing
    # speed is hypot(v_t1 + v_earth, v_r1).
    closing = float(np.hypot(v_t1 + params.v_earth_orbit, np.sqrt(v_r1_sq)))
    collision = float(np.hypot(closing, params.v_esc_surface))
    return _ReturnLeg(
        perihelion=perihelion, tof=tof, closing_speed=closing, collision_speed=collision
    )


@dataclass(frozen=True)
class _FlybyLeg:
    """Float summary of one full Earth-to-1-AU powered-flyby trajectory.

    Attributes:
        departure_burn: Oberth burn above escape at 200 km, delta-v1 (km/s).
        flyby_burn: Impulsive burn at Jovian periapsis, delta-v2 (km/s).
        v_infinity_earth: Hyperbolic excess speed leaving Earth (km/s).
        aim_angle: Angle of the departure excess velocity off Earth's orbital
            velocity (rad; positive tilts radially outward).
        periapsis_radius: Jovian flyby periapsis radius, center-based (km).
        v_infinity_in: Jupiter-relative excess speed before the burn (km/s).
        v_infinity_out: Jupiter-relative excess speed after the burn (km/s).
        turn_angle: Total split-hyperbola bend asin(1/e_in) + asin(1/e_out) (rad).
        return_leg: The scored retrograde return.
        outbound_tof: Earth-to-Jupiter heliocentric time of flight (s).
        delivered_fraction: Mass fraction surviving both burns.
        mass_ratio: Payload/PuffSat mass ratio at the achieved collision speed.
        end_to_end: delivered_fraction x mass_ratio, the objective.
    """

    departure_burn: float
    flyby_burn: float
    v_infinity_earth: float
    aim_angle: float
    periapsis_radius: float
    v_infinity_in: float
    v_infinity_out: float
    turn_angle: float
    return_leg: _ReturnLeg
    outbound_tof: float
    delivered_fraction: float
    mass_ratio: float
    end_to_end: float


def _powered_flyby_leg(
    v_infinity_earth: float,
    aim_angle: float,
    periapsis_radius: float,
    flyby_burn: float,
    descending_arrival: bool,
    bend_sign: float,
    params: _FlybyParams,
) -> Optional[_FlybyLeg]:
    """Evaluate one powered-flyby trajectory from its four knobs.

    Free-aim departure: the burn cost depends only on ``v_infinity_earth``;
    ``aim_angle`` orients the excess velocity for free. The flyby is a split
    hyperbola -- an impulsive tangential burn at periapsis joins an inbound and
    an outbound hyperbola of different eccentricities, so the total bend is
    ``asin(1/e_in) + asin(1/e_out)``, rotated to the side ``bend_sign`` picks.

    Args:
        v_infinity_earth: Hyperbolic excess speed leaving Earth (km/s).
        aim_angle: Angle of the excess velocity off Earth's orbital velocity
            (rad; positive tilts radially outward).
        periapsis_radius: Jovian flyby periapsis radius, center-based (km).
        flyby_burn: Impulsive burn at Jovian periapsis, >= 0 (km/s).
        descending_arrival: Arrive at Jupiter's orbit radius past aphelion
            (inward-moving) instead of on the outbound branch.
        bend_sign: +1 rotates the excess velocity from tangential toward
            radial-outward, -1 the other way (which side Jupiter is passed on).
        params: The float parameter block.

    Returns:
        The :class:`_FlybyLeg`, or None if infeasible (cannot reach Jupiter,
        non-retrograde return, no 1 AU crossing, or over the time-of-flight cap).
    """
    mu = params.mu_sun
    r_e = params.r_earth_orbit
    r_j = params.r_jupiter_orbit
    if v_infinity_earth <= 0.0 or flyby_burn < 0.0:
        return None
    if periapsis_radius < params.periapsis_floor * (1.0 - 1e-12):
        return None
    departure_burn = float(
        np.hypot(v_infinity_earth, params.v_esc_leo) - params.v_depart_from
    )

    # Outbound heliocentric transfer from the free-aimed departure state.
    v_t0 = params.v_earth_orbit + v_infinity_earth * float(np.cos(aim_angle))
    v_r0 = v_infinity_earth * float(np.sin(aim_angle))
    if v_t0 <= 0.0:
        return None
    energy = (v_t0 * v_t0 + v_r0 * v_r0) / 2.0 - mu / r_e
    h = r_e * v_t0
    p = h * h / mu
    ecc = float(np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / (mu * mu))))
    if abs(ecc - 1.0) < 1e-9 or ecc < 1e-9:
        return None
    if ecc < 1.0 and p / (1.0 - ecc) < r_j:
        return None
    if ecc > 1.0 and descending_arrival:
        return None
    nu_depart = _true_anomaly_at_radius_rad(p, ecc, r_e)
    nu_arrive = _true_anomaly_at_radius_rad(p, ecc, r_j)
    if nu_depart is None or nu_arrive is None:
        return None
    if ecc < 1.0:
        a = p / (1.0 - ecc * ecc)
        period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
        t_depart = _elliptic_tof_seconds(mu, a, ecc, nu_depart)
        t_arrive = _elliptic_tof_seconds(mu, a, ecc, nu_arrive)
        if descending_arrival:
            t_arrive = period - t_arrive
    else:
        a_abs = p / (ecc * ecc - 1.0)
        t_depart = _hyperbolic_tof_seconds(mu, a_abs, ecc, nu_depart)
        t_arrive = _hyperbolic_tof_seconds(mu, a_abs, ecc, nu_arrive)
    if v_r0 < 0.0:
        t_depart = -t_depart
    outbound_tof = t_arrive - t_depart
    if outbound_tof <= 0.0:
        return None

    # Jupiter-relative arrival state.
    v_arr_sq = 2.0 * (energy + mu / r_j)
    v_t_arr = h / r_j
    v_r_arr_sq = v_arr_sq - v_t_arr * v_t_arr
    if v_r_arr_sq < -1e-9:
        return None
    v_r_arr = float(np.sqrt(max(0.0, v_r_arr_sq)))
    if descending_arrival:
        v_r_arr = -v_r_arr
    rel_t = v_t_arr - params.v_jupiter_orbit
    rel_r = v_r_arr
    w_in = float(np.hypot(rel_t, rel_r))
    if w_in < 1e-6:
        return None

    # Split-hyperbola powered flyby at periapsis_radius.
    mu_j = params.mu_jupiter
    ecc_in = 1.0 + periapsis_radius * w_in * w_in / mu_j
    half_turn_in = float(np.arcsin(1.0 / ecc_in))
    v_peri_in = float(np.sqrt(w_in * w_in + 2.0 * mu_j / periapsis_radius))
    v_peri_out = v_peri_in + flyby_burn
    w_out_sq = v_peri_out * v_peri_out - 2.0 * mu_j / periapsis_radius
    if w_out_sq <= 0.0:
        return None
    w_out = float(np.sqrt(w_out_sq))
    ecc_out = 1.0 + periapsis_radius * w_out * w_out / mu_j
    half_turn_out = float(np.arcsin(1.0 / ecc_out))
    turn = half_turn_in + half_turn_out
    cos_turn = float(np.cos(bend_sign * turn))
    sin_turn = float(np.sin(bend_sign * turn))
    unit_t = rel_t / w_in
    unit_r = rel_r / w_in
    out_t = cos_turn * unit_t - sin_turn * unit_r
    out_r = sin_turn * unit_t + cos_turn * unit_r
    v_t_out = params.v_jupiter_orbit + w_out * out_t
    v_r_out = w_out * out_r

    return_leg = _flyby_return_leg(v_t_out, v_r_out, params)
    if return_leg is None:
        return None
    if outbound_tof + return_leg.tof > params.max_tof:
        return None
    if return_leg.collision_speed <= params.v_rf:
        return None

    delivered = float(np.exp(-(departure_burn + flyby_burn) / params.exhaust_speed))
    # payload_mass_ratio with v_ri = 0, inlined for the float hot loop.
    mass_ratio = float(
        2.0
        * STD_FUDGE_FACTOR
        / np.log(
            return_leg.collision_speed / (return_leg.collision_speed - params.v_rf)
        )
    )
    return _FlybyLeg(
        departure_burn=departure_burn,
        flyby_burn=flyby_burn,
        v_infinity_earth=v_infinity_earth,
        aim_angle=aim_angle,
        periapsis_radius=periapsis_radius,
        v_infinity_in=w_in,
        v_infinity_out=w_out,
        turn_angle=turn,
        return_leg=return_leg,
        outbound_tof=outbound_tof,
        delivered_fraction=delivered,
        mass_ratio=mass_ratio,
        end_to_end=delivered * mass_ratio,
    )


# Search-space bounds for the differential-evolution knobs:
# (v_infinity_earth km/s, aim_angle rad, log10(periapsis/floor), flyby_burn km/s).
# The Hohmann departure needs ~8.79 km/s of excess, so 8.5 undercuts the
# feasible edge; 25 km/s costs a ~16 km/s burn the objective would never pay.
_FLYBY_BOUNDS: List[Tuple[float, float]] = [
    (8.5, 25.0),
    (-np.pi, np.pi),
    (0.0, 2.5),
    (0.0, 12.0),
]
_FLYBY_INFEASIBLE_PENALTY = 1e3


def _flyby_from_vector(
    x: npt.NDArray[np.float64],
    descending_arrival: bool,
    bend_sign: float,
    params: _FlybyParams,
) -> Optional[_FlybyLeg]:
    """Evaluate a knob vector (see ``_FLYBY_BOUNDS``) into a flyby leg.

    Args:
        x: Knob vector (v_infinity_earth, aim_angle, log10(rp/floor), flyby_burn).
        descending_arrival: Arrival-branch discrete choice.
        bend_sign: Flyby bend-side discrete choice (+1 or -1).
        params: The float parameter block.

    Returns:
        The evaluated :class:`_FlybyLeg`, or None if infeasible.
    """
    periapsis_radius = params.periapsis_floor * float(10.0 ** x[2])
    return _powered_flyby_leg(
        v_infinity_earth=float(x[0]),
        aim_angle=float(x[1]),
        periapsis_radius=periapsis_radius,
        flyby_burn=float(x[3]),
        descending_arrival=descending_arrival,
        bend_sign=bend_sign,
        params=params,
    )


def _flyby_objective(
    x: npt.NDArray[np.float64],
    descending_arrival: bool,
    bend_sign: float,
    params: _FlybyParams,
) -> float:
    """Penalized objective for the end-to-end optimum: minimize -end_to_end."""
    leg = _flyby_from_vector(x, descending_arrival, bend_sign, params)
    if leg is None:
        return _FLYBY_INFEASIBLE_PENALTY
    return -leg.end_to_end


def _flyby_trade_objective(
    x: npt.NDArray[np.float64],
    descending_arrival: bool,
    bend_sign: float,
    params: _FlybyParams,
    target_collision_speed: float,
) -> float:
    """Penalized objective for the trade curve: minimize total burn at a v_b floor."""
    leg = _flyby_from_vector(x, descending_arrival, bend_sign, params)
    if leg is None:
        return _FLYBY_INFEASIBLE_PENALTY
    shortfall = target_collision_speed - leg.return_leg.collision_speed
    if shortfall > 0.0:
        # Keep the penalty smooth but strictly above any feasible burn total
        # (bounds cap the total near ~28 km/s).
        return 50.0 + shortfall
    return leg.departure_burn + leg.flyby_burn


def _optimize_flyby(
    objective_args: Tuple[object, ...],
    objective: object,
    params: _FlybyParams,
    seed: int,
    popsize: int,
    maxiter: int,
) -> Optional[Tuple[_FlybyLeg, bool, float]]:
    """Run differential evolution over the four discrete branch/side combos.

    Args:
        objective_args: Extra args appended after (descending, bend_sign, params).
        objective: The penalized objective callable.
        params: The float parameter block.
        seed: Random seed for deterministic results.
        popsize: Differential-evolution population multiplier.
        maxiter: Differential-evolution iteration cap.

    Returns:
        Tuple of (best leg, descending_arrival, bend_sign) by objective value,
        or None if every combo came back infeasible.
    """
    best: Optional[Tuple[_FlybyLeg, bool, float]] = None
    best_value = _FLYBY_INFEASIBLE_PENALTY
    for descending_arrival in (False, True):
        for bend_sign in (1.0, -1.0):
            result = differential_evolution(
                objective,
                bounds=_FLYBY_BOUNDS,
                args=(descending_arrival, bend_sign, params) + objective_args,
                seed=seed,
                popsize=popsize,
                maxiter=maxiter,
                tol=1e-10,
                polish=True,
            )
            if result.fun >= _FLYBY_INFEASIBLE_PENALTY:
                continue
            leg = _flyby_from_vector(result.x, descending_arrival, bend_sign, params)
            if leg is not None and result.fun < best_value:
                best = (leg, descending_arrival, bend_sign)
                best_value = float(result.fun)
    return best


@dataclass(frozen=True)
class PoweredJovianFlybyReturn:
    """Optimum of the powered Jovian flyby retrograde return (ADR 0002).

    The two burns of the leg that puts a PuffSat onto a retrograde
    Earth-crossing orbit, chosen to maximize the end-to-end mass ratio
    (delivered mass fraction x payload mass ratio at the achieved collision
    speed) under the seven-year time-of-flight cap. Not a
    :class:`PuffSatScenario`: like the lunar-return optimum, it blends a rocket
    burn cost with a collision mass ratio, so it is reported on its own.

    Attributes:
        v_infinity_earth: Hyperbolic excess speed leaving Earth (km/s).
        aim_angle: Free-aim angle of the excess velocity off Earth's orbital
            velocity (deg; positive tilts radially outward).
        departure_burn: Oberth burn above escape at 200 km, delta-v1 (km/s).
        flyby_periapsis_radius: Jovian flyby periapsis, center-based (km).
        flyby_periapsis_altitude: The same periapsis above the 1-bar level (km).
        flyby_burn: Impulsive periapsis burn at Jupiter, delta-v2 (km/s).
        v_infinity_jupiter_in: Jupiter-relative excess speed before the burn (km/s).
        v_infinity_jupiter_out: Jupiter-relative excess speed after the burn (km/s).
        turn_angle: Total split-hyperbola bend (deg).
        descending_arrival: Whether the optimum arrives at Jupiter's orbit radius
            past aphelion (inward-moving).
        bend_sign: Which side the flyby bends toward (+1 tangential-to-outward).
        return_perihelion: Perihelion of the retrograde return orbit (AU).
        closing_speed_1au: Earth-relative speed at the 1 AU crossing (km/s).
        collision_speed: The closing speed folded through Earth's well -- the
            achieved v_b, on the retrograde_jovian_hohmann_transfer convention
            (km/s).
        outbound_time: Earth-to-Jupiter time of flight (yr).
        return_time: Jupiter-to-1 AU time of flight (yr).
        total_time: Sum of the two legs (yr), <= the seven-year cap.
        delivered_fraction: Mass fraction surviving both methalox burns.
        payload_puffsat_mass_ratio: Mass ratio at the achieved collision speed
            against the sec:jupiter_only_growth push.
        end_to_end_mass_ratio: The objective, delivered x ratio.
    """

    v_infinity_earth: u.Quantity
    aim_angle: u.Quantity
    departure_burn: u.Quantity
    flyby_periapsis_radius: u.Quantity
    flyby_periapsis_altitude: u.Quantity
    flyby_burn: u.Quantity
    v_infinity_jupiter_in: u.Quantity
    v_infinity_jupiter_out: u.Quantity
    turn_angle: u.Quantity
    descending_arrival: bool
    bend_sign: float
    return_perihelion: u.Quantity
    closing_speed_1au: u.Quantity
    collision_speed: u.Quantity
    outbound_time: u.Quantity
    return_time: u.Quantity
    total_time: u.Quantity
    delivered_fraction: float
    payload_puffsat_mass_ratio: float
    end_to_end_mass_ratio: float


def _leg_to_result(
    leg: _FlybyLeg, descending_arrival: bool, bend_sign: float, params: _FlybyParams
) -> PoweredJovianFlybyReturn:
    """Wrap a float flyby leg into the public Quantity-valued result."""
    return PoweredJovianFlybyReturn(
        v_infinity_earth=leg.v_infinity_earth * u.km / u.s,
        aim_angle=(np.degrees(leg.aim_angle) * u.deg),
        departure_burn=leg.departure_burn * u.km / u.s,
        flyby_periapsis_radius=leg.periapsis_radius * u.km,
        flyby_periapsis_altitude=(leg.periapsis_radius * u.km - Jupiter.R.to(u.km)),
        flyby_burn=leg.flyby_burn * u.km / u.s,
        v_infinity_jupiter_in=leg.v_infinity_in * u.km / u.s,
        v_infinity_jupiter_out=leg.v_infinity_out * u.km / u.s,
        turn_angle=(np.degrees(leg.turn_angle) * u.deg),
        descending_arrival=descending_arrival,
        bend_sign=bend_sign,
        return_perihelion=(leg.return_leg.perihelion * u.km).to(u.AU),
        closing_speed_1au=leg.return_leg.closing_speed * u.km / u.s,
        collision_speed=leg.return_leg.collision_speed * u.km / u.s,
        outbound_time=(leg.outbound_tof * u.s).to(u.year),
        return_time=(leg.return_leg.tof * u.s).to(u.year),
        total_time=((leg.outbound_tof + leg.return_leg.tof) * u.s).to(u.year),
        delivered_fraction=leg.delivered_fraction,
        payload_puffsat_mass_ratio=leg.mass_ratio,
        end_to_end_mass_ratio=leg.end_to_end,
    )


def powered_jovian_flyby_return(
    max_total_tof: u.Quantity = JUPITER_FLYBY_MAX_TOF,
    periapsis_floor_altitude: u.Quantity = LOW_JUPITER_ALTITUDE,
    seed: int = 0,
) -> PoweredJovianFlybyReturn:
    """Optimize the powered Jovian flyby that returns a PuffSat retrograde.

    Searches the four continuous knobs (departure excess speed, free-aim angle,
    flyby periapsis radius, flyby burn) and the two discrete choices (arrival
    branch, bend side) for the trajectory maximizing the end-to-end mass ratio,
    subject to a retrograde 1 AU crossing and the seven-year cap. See ADR
    0002-jupiter-flyby-objective for why the objective is not minimum delta-v
    (the retrograde-plunge degeneracy) and CONTEXT.md for the vocabulary.

    The achieved collision speed is expected to land strictly between the
    barely-retrograde plunge (~50 km/s) and the retrograde Hohmann the catalog
    rows assume (~69.3 km/s); the catalog rows deliberately keep the paper's
    published value (see the ADR).

    Args:
        max_total_tof: Cap on outbound + return time of flight (astropy
            Quantity, default JUPITER_FLYBY_MAX_TOF = 7 yr).
        periapsis_floor_altitude: Minimum flyby altitude above Jupiter's 1-bar
            level (astropy Quantity, default LOW_JUPITER_ALTITUDE = 4000 km).
        seed: Random seed for the differential-evolution search (deterministic).

    Returns:
        The :class:`PoweredJovianFlybyReturn` optimum.

    Raises:
        ValueError: If no feasible trajectory exists under the constraints.
    """
    params = _powered_flyby_params(max_total_tof, periapsis_floor_altitude)
    best = _optimize_flyby(
        objective_args=(),
        objective=_flyby_objective,
        params=params,
        seed=seed,
        popsize=20,
        maxiter=300,
    )
    if best is None:
        raise ValueError(
            "No feasible powered Jovian flyby found under the given constraints."
        )
    leg, descending_arrival, bend_sign = best
    return _leg_to_result(leg, descending_arrival, bend_sign, params)


@dataclass(frozen=True)
class JupiterFlybyTradePoint:
    """One point of the delta-v versus collision-speed trade curve (ADR 0002).

    Attributes:
        target_collision_speed: The v_b floor this point was solved for (km/s).
        feasible: Whether any trajectory meets the floor under the constraints.
        departure_burn: Delta-v1 of the cheapest meeting trajectory (km/s).
        flyby_burn: Delta-v2 of the cheapest meeting trajectory (km/s).
        total_burn: departure_burn + flyby_burn (km/s).
        achieved_collision_speed: The v_b actually reached (km/s, >= target).
        end_to_end_mass_ratio: The end-to-end metric at this point (not what it
            optimizes -- it minimizes total burn).
        total_time: Outbound + return time of flight (yr).
    """

    target_collision_speed: u.Quantity
    feasible: bool
    departure_burn: u.Quantity
    flyby_burn: u.Quantity
    total_burn: u.Quantity
    achieved_collision_speed: u.Quantity
    end_to_end_mass_ratio: float
    total_time: u.Quantity


def jupiter_flyby_vb_trade_curve(
    targets: Tuple[float, ...] = JUPITER_FLYBY_VB_TRADE_TARGETS,
    max_total_tof: u.Quantity = JUPITER_FLYBY_MAX_TOF,
    periapsis_floor_altitude: u.Quantity = LOW_JUPITER_ALTITUDE,
    seed: int = 0,
) -> List[JupiterFlybyTradePoint]:
    """Sweep minimum total burn against a floor on the collision speed v_b.

    The defence of the point optimum: shows how flat the propellant landscape is
    between plunge-like returns and the retrograde Hohmann, pre-answering the
    sensitivity question when the paper adopts the numbers (ADR 0002).

    Args:
        targets: Collision-speed floors in km/s (default
            JUPITER_FLYBY_VB_TRADE_TARGETS).
        max_total_tof: Cap on outbound + return time of flight (astropy Quantity).
        periapsis_floor_altitude: Minimum flyby altitude above Jupiter's 1-bar
            level (astropy Quantity).
        seed: Random seed for the differential-evolution search (deterministic).

    Returns:
        One :class:`JupiterFlybyTradePoint` per target, in the given order;
        infeasible targets come back with ``feasible=False`` and NaN quantities.
    """
    params = _powered_flyby_params(max_total_tof, periapsis_floor_altitude)
    kms = u.km / u.s
    points: List[JupiterFlybyTradePoint] = []
    for target in targets:
        best = _optimize_flyby(
            objective_args=(float(target),),
            objective=_flyby_trade_objective,
            params=params,
            seed=seed,
            popsize=16,
            maxiter=200,
        )
        if best is None:
            points.append(
                JupiterFlybyTradePoint(
                    target_collision_speed=target * kms,
                    feasible=False,
                    departure_burn=np.nan * kms,
                    flyby_burn=np.nan * kms,
                    total_burn=np.nan * kms,
                    achieved_collision_speed=np.nan * kms,
                    end_to_end_mass_ratio=float("nan"),
                    total_time=np.nan * u.year,
                )
            )
            continue
        leg, _, _ = best
        points.append(
            JupiterFlybyTradePoint(
                target_collision_speed=target * kms,
                feasible=True,
                departure_burn=leg.departure_burn * kms,
                flyby_burn=leg.flyby_burn * kms,
                total_burn=(leg.departure_burn + leg.flyby_burn) * kms,
                achieved_collision_speed=leg.return_leg.collision_speed * kms,
                end_to_end_mass_ratio=leg.end_to_end,
                total_time=((leg.outbound_tof + leg.return_leg.tof) * u.s).to(u.year),
            )
        )
    return points


# ---------------------------------------------------------------------------
# Unpowered V/E/M assist chain to the same retrograde return (CONTEXT.md
# "Unpowered assist chain").
#
# Same phasing-free, coplanar-circular patched-conic model as the powered
# Jovian flyby above, but after the LEO departure burn no propellant is spent
# en route: every velocity change comes from unpowered gravity assists at
# Venus, Earth, and Mars, then one unpowered Jovian bend into the retrograde
# return scored by _flyby_return_leg. A fixed phasing budget
# (ASSIST_CHAIN_PHASING_BUDGET) is charged as spent methalox so the mass
# accounting carries the cost of making real ephemerides line up.
#
# The search is a deterministic beam search over flyby chains. States live at
# a body's orbit radius as planet-relative excess velocities; an unpowered
# flyby rotates the excess velocity within the bend limit its periapsis floor
# allows (it can never change the excess *speed* -- the Tisserand invariant),
# and legs between bodies are conic arcs between orbit radii. Pruning is
# time-bucketed: per (body, 0.2 yr time-of-flight bucket) the top states by
# excess speed are kept, which keeps slow-but-closable chains alive alongside
# fast high-energy ones (global top-by-speed pruning made feasibility
# non-monotone in the burn). The sample counts below were calibrated so the
# feasibility edge sits at ~0.29 km/s (a finer beam moves it to ~0.285 km/s,
# still above the ~0.2794 km/s analytic Venus-reach floor) at ~10 s per burn
# probe.
_ASSIST_DEPARTURE_AIM_SAMPLES = 181  # free-aim samples at Earth departure
_ASSIST_ROTATION_SAMPLES = 31  # unpowered-bend samples per inner-planet flyby
_ASSIST_JOVIAN_BEND_SAMPLES = 121  # bend samples at the Jovian terminal
_ASSIST_BEAM_PER_BUCKET = 80  # states kept per (body, time bucket)
_ASSIST_BEAM_CAP_PER_BODY = 1500  # states kept per body per search depth
_ASSIST_BUCKET_WIDTH_YEARS = 0.2  # width of the pruning time buckets
_ASSIST_DEDUP_SPEED_BIN = 0.05  # km/s; velocity grid for state deduplication
_ASSIST_DEDUP_TIME_BIN_YEARS = 0.05  # yr; time grid for state deduplication
_ASSIST_MIN_LEG_TIME = 86400.0  # s; ignore degenerate sub-day legs
_ASSIST_JOVIAN_LEG_RESERVE_YEARS = 0.3  # yr kept free for the Jovian leg
_SECONDS_PER_YEAR = float((1.0 * u.year).to_value(u.s))


@dataclass(frozen=True)
class _AssistBody:
    """One inner-planet flyby body of the assist chain (km, s, km/s floats).

    Attributes:
        symbol: One-letter tag used in sequence strings ("V", "E", "M").
        name: Full body name for the public result.
        orbit_radius: Heliocentric orbit radius (km).
        mu: Gravitational parameter (km^3/s^2).
        min_periapsis: Minimum flyby periapsis radius, center-based (km).
        v_circ: Circular heliocentric speed at orbit_radius (km/s).
    """

    symbol: str
    name: str
    orbit_radius: float
    mu: float
    min_periapsis: float
    v_circ: float


@dataclass(frozen=True)
class _AssistChainParams:
    """Inputs of the assist-chain beam search.

    Attributes:
        flyby: The powered-flyby parameter block, reused for the Sun/Jupiter
            constants, the Jovian periapsis floor, the exhaust speed, and the
            v_rf push target (its max_tof field is unused here).
        bodies: The flyby bodies, in (Venus, Earth, Mars) order.
        earth_index: Index of Earth in ``bodies`` (the departure body).
        target_collision_speed: Minimum acceptable v_b of the return (km/s).
        max_trip_time: Cap on departure-to-1 AU-crossing time (s).
        max_flybys: Cap on the number of inner-planet flybys.
    """

    flyby: _FlybyParams
    bodies: Tuple[_AssistBody, ...]
    earth_index: int
    target_collision_speed: float
    max_trip_time: float
    max_flybys: int


def _assist_chain_params(
    target_collision_speed: float,
    max_trip_time: u.Quantity = ASSIST_CHAIN_MAX_TRIP_TIME,
    max_flybys: int = ASSIST_CHAIN_MAX_FLYBYS,
    cycle_periapsis_speed: Optional[u.Quantity] = None,
) -> _AssistChainParams:
    """Build the float parameter block for the assist-chain search.

    Args:
        target_collision_speed: Minimum acceptable collision speed v_b (km/s).
        max_trip_time: Cap on total trip time (astropy Quantity).
        max_flybys: Cap on the number of inner-planet flybys.
        cycle_periapsis_speed: Speed the departure burn starts from; see
            :func:`_powered_flyby_params`. None keeps the legacy C3 = 0
            convention.

    Returns:
        The :class:`_AssistChainParams` with everything in km / s / km/s.
    """
    flyby = _powered_flyby_params(
        max_total_tof=max_trip_time, cycle_periapsis_speed=cycle_periapsis_speed
    )
    altitude = float(LOW_ASSIST_FLYBY_ALTITUDE.to_value(u.km))
    bodies: List[_AssistBody] = []
    for symbol, body, semi_major_axis in (
        ("V", Venus, VENUS_A),
        ("E", Earth, EARTH_A),
        ("M", Mars, MARS_A),
    ):
        orbit_radius = float(semi_major_axis.to_value(u.km))
        bodies.append(
            _AssistBody(
                symbol=symbol,
                name=str(body.name),
                orbit_radius=orbit_radius,
                mu=float(body.k.to_value(u.km**3 / u.s**2)),
                min_periapsis=float(body.R.to_value(u.km)) + altitude,
                v_circ=float(np.sqrt(flyby.mu_sun / orbit_radius)),
            )
        )
    return _AssistChainParams(
        flyby=flyby,
        bodies=tuple(bodies),
        earth_index=1,
        target_collision_speed=target_collision_speed,
        max_trip_time=float(max_trip_time.to_value(u.s)),
        max_flybys=max_flybys,
    )


def _elliptic_time_from_periapsis(mu: float, a: float, ecc: float, nu: float) -> float:
    """Time from periapsis to true anomaly ``nu`` in [0, 2*pi) on an ellipse.

    Extends :func:`_elliptic_tof_seconds` past pi by period symmetry.

    Args:
        mu: Gravitational parameter (km^3/s^2).
        a: Semi-major axis (km).
        ecc: Eccentricity, 0 <= ecc < 1.
        nu: True anomaly from periapsis (rad, in [0, 2*pi)).

    Returns:
        The time since periapsis passage (s).
    """
    if nu > np.pi:
        period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
        return period - _elliptic_tof_seconds(mu, a, ecc, 2.0 * np.pi - nu)
    return _elliptic_tof_seconds(mu, a, ecc, nu)


def _conic_radius_crossings(
    mu: float, r0: float, v_t0: float, v_r0: float, r1: float
) -> List[Tuple[float, float, float, bool, float]]:
    """Future crossings of radius ``r1`` from a heliocentric state at ``r0``.

    The state is (tangential, radial-outward) at radius ``r0`` with positive
    angular momentum. Within one revolution (ellipse) or on the remaining
    branch (hyperbola) the orbit crosses ``r1`` at most twice: once moving
    outward and once moving inward.

    Args:
        mu: Gravitational parameter of the Sun (km^3/s^2).
        r0: Current radius (km).
        v_t0: Tangential speed, > 0 for prograde (km/s).
        v_r0: Radial-outward speed (km/s).
        r1: Target radius (km).

    Returns:
        Up to two tuples (time of flight s, tangential speed at r1, signed
        radial speed at r1, outbound flag, swept heliocentric longitude rad),
        skipping degenerate sub-day legs. The swept angle is what a *phased*
        chain needs: paired with the leg time it says where the craft arrives,
        not merely when, so an arrival can be matched against where the target
        body will actually be. Motion is prograde (h > 0), so it is positive.
    """
    energy = (v_t0 * v_t0 + v_r0 * v_r0) / 2.0 - mu / r0
    h = r0 * v_t0
    if h <= 0.0:
        return []
    p = h * h / mu
    ecc = float(np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / (mu * mu))))
    if ecc < 1e-9 or abs(ecc - 1.0) < 1e-9:
        return []
    nu0 = _true_anomaly_at_radius_rad(p, ecc, r0)
    nu1 = _true_anomaly_at_radius_rad(p, ecc, r1)
    if nu0 is None or nu1 is None:
        return []
    if v_r0 < 0.0:
        nu0 = -nu0
    v1_sq = 2.0 * (energy + mu / r1)
    v_t1 = h / r1
    v_r1 = float(np.sqrt(max(0.0, v1_sq - v_t1 * v_t1)))
    crossings: List[Tuple[float, float, float, bool, float]] = []
    if ecc < 1.0:
        a = p / (1.0 - ecc * ecc)
        period = 2.0 * np.pi * float(np.sqrt(a**3 / mu))
        t0 = _elliptic_time_from_periapsis(mu, a, ecc, nu0 % (2.0 * np.pi))
        for nu_target, v_r_signed, outbound in (
            (nu1, v_r1, True),
            (2.0 * np.pi - nu1, -v_r1, False),
        ):
            dt = (_elliptic_time_from_periapsis(mu, a, ecc, nu_target) - t0) % period
            if dt > _ASSIST_MIN_LEG_TIME:
                swept = (nu_target - nu0) % (2.0 * np.pi)
                crossings.append((dt, v_t1, v_r_signed, outbound, swept))
    else:
        a_abs = p / (ecc * ecc - 1.0)
        t0 = float(np.sign(nu0)) * _hyperbolic_tof_seconds(mu, a_abs, ecc, abs(nu0))
        for nu_target, v_r_signed, outbound in (
            (nu1, v_r1, True),
            (-nu1, -v_r1, False),
        ):
            dt = (
                float(np.sign(nu_target))
                * _hyperbolic_tof_seconds(mu, a_abs, ecc, abs(nu_target))
                - t0
            )
            if dt > _ASSIST_MIN_LEG_TIME:
                # A hyperbola never wraps, so the sweep is the bare difference;
                # a positive dt guarantees nu_target is ahead of nu0.
                swept = nu_target - nu0
                crossings.append((dt, v_t1, v_r_signed, outbound, swept))
    return crossings


def _mean_motion(body: _AssistBody) -> float:
    """Mean motion of a chain body on its circular orbit (rad/s).

    Args:
        body: The chain body.

    Returns:
        Angular rate v_circ / r, positive (prograde).
    """
    return body.v_circ / body.orbit_radius


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        The equivalent angle in (-pi, pi].
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _phased_leg_rotations(
    body: _AssistBody,
    body_longitude: float,
    excess_t: float,
    excess_r: float,
    target: _AssistBody,
    target_longitude: float,
    outbound: bool,
    params: _AssistChainParams,
    samples: int = _ASSIST_ROTATION_SAMPLES,
    rotation_limit: Optional[float] = None,
) -> List[Tuple[float, float, float, float]]:
    """Flyby rotations at ``body`` that arrive where ``target`` actually is.

    This is what turns the phasing-free ladder into a phased one. The
    phasing-free search samples the rotation freely and lets the target planet
    be wherever the leg lands; here the rotation is *solved* so the leg lands on
    the planet. Rotating by ``theta`` fixes both the swept longitude and the
    flight time, so "arrive where the target will be" is one equation in one
    unknown -- rooted with brentq over the bend-limited rotation, the same
    scan-for-sign-change-then-root pattern as
    :func:`apoapsis_raise_reintercept`.

    Wrapped residuals jump by 2*pi where they cross the branch cut, which looks
    exactly like a sign change; brackets spanning more than pi of residual are
    rejected so those phantom roots are never rooted.

    Args:
        body: Body the flyby happens at.
        body_longitude: Its heliocentric longitude at the flyby (rad).
        excess_t: Tangential component of the body-relative excess (km/s).
        excess_r: Radial-outward component of the body-relative excess (km/s).
        target: Body the leg must arrive at.
        target_longitude: Target's heliocentric longitude at the *same* instant
            as ``body_longitude`` -- both are stated at the flyby, so the
            absolute epoch never enters and the caller keeps that bookkeeping.
        outbound: Which arrival branch to solve on.
        params: The assist-chain parameter block.
        samples: Rotations scanned for sign changes before rooting.
        rotation_limit: Override for the reachable rotation half-range (rad).
            Defaults to the body's unpowered bend limit. Pass ``pi`` for the
            *departure*, where there is no flyby and the excess may be aimed
            anywhere for free -- a bend limit there would be fictitious.

    Returns:
        One (rotation rad, leg time s, tangential speed at arrival km/s,
        signed radial speed at arrival km/s) per solved rotation, possibly
        empty. Several roots can coexist: the equation is transcendental.
    """
    w = float(np.hypot(excess_t, excess_r))
    if w < 1e-9:
        return []
    phi = float(np.arctan2(excess_r, excess_t))
    bend_limit = (
        2.0 * float(np.arcsin(1.0 / (1.0 + body.min_periapsis * w * w / body.mu)))
        if rotation_limit is None
        else rotation_limit
    )
    n_target = _mean_motion(target)

    def evaluate(theta: float) -> Optional[Tuple[float, float, float, float]]:
        angle = phi + theta
        v_t0 = body.v_circ + w * float(np.cos(angle))
        v_r0 = w * float(np.sin(angle))
        for leg_tof, v_t1, v_r1, out, swept in _conic_radius_crossings(
            params.flyby.mu_sun,
            body.orbit_radius,
            v_t0,
            v_r0,
            target.orbit_radius,
        ):
            if out != outbound:
                continue
            # Both longitudes are stated at `epoch`, so the target advances by
            # the leg time only -- advancing it from zero would double-count
            # the epoch and manufacture a spurious window cadence.
            arrival = body_longitude + swept
            where_target_is = target_longitude + n_target * leg_tof
            return (_wrap_pi(arrival - where_target_is), leg_tof, v_t1, v_r1)
        return None

    def residual(theta: float) -> float:
        found = evaluate(theta)
        return float("nan") if found is None else found[0]

    grid = np.linspace(-bend_limit, bend_limit, samples)
    values = [residual(float(t)) for t in grid]
    solved: List[Tuple[float, float, float, float]] = []
    for i in range(len(grid) - 1):
        lo, hi = values[i], values[i + 1]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if lo == 0.0:
            root = float(grid[i])
        elif lo * hi > 0.0:
            continue
        elif abs(lo - hi) > np.pi:
            # Branch-cut jump, not a crossing: the residual wrapped rather than
            # passed through zero.
            continue
        else:
            try:
                root = float(brentq(residual, grid[i], grid[i + 1], xtol=1e-12))
            except (ValueError, RuntimeError):
                continue
        found = evaluate(root)
        if found is None:
            continue
        solved.append((root, found[1], found[2], found[3]))
    return solved


_PHASED_LADDER_FAILED = 1e3  # km/s; above any real ladder cost


@dataclass(frozen=True)
class _LadderPricing:
    """What it costs to fly an inner ladder against real planet positions.

    Attributes:
        departure_burn: Oberth burn at 200 km buying the first leg's excess
            (km/s).
        node_total: Sum of the flyby node burns (km/s).
        node_burns: Per-node burns in ladder order, one per intermediate body
            (km/s).
        arrival_excess: Speed of the planet-relative excess at the final ladder
            body, where the Jovian leg begins (km/s).
        arrival_excess_vector: That excess as a heliocentric 3-vector (km/s).
        arrival_time: Seconds from the longitudes0 reference at the final body.
        arrival_longitude: Heliocentric longitude of the final body then (rad).
    """

    departure_burn: float
    node_total: float
    node_burns: List[float]
    arrival_excess: float
    arrival_excess_vector: npt.NDArray[np.float64]
    arrival_time: float
    arrival_longitude: float


def _phased_ladder_burn(
    epoch: float,
    leg_times: Sequence[float],
    ladder: Tuple[_AssistBody, ...],
    longitudes0: Sequence[float],
    params: _AssistChainParams,
    powered_nodes: bool = False,
) -> Optional[_LadderPricing]:
    """Burn to fly ``ladder`` from ``epoch`` against real planet positions.

    The phasing-free model puts each planet wherever the trajectory needs it;
    this puts the planets where they actually are and charges for the
    difference. Each leg becomes a Lambert arc between the two bodies at their
    true positions, and each node is charged for whatever the flyby cannot
    supply. That total is what ``ASSIST_CHAIN_PHASING_BUDGET`` is a stand-in
    for.

    Args:
        epoch: Launch time, seconds from the longitudes0 reference.
        leg_times: Duration of each leg (s), one per ladder hop.
        ladder: Bodies in order, departure body first; the Jovian leg is not
            included (this prices the inner ladder only).
        longitudes0: Heliocentric longitude of each ladder body at time zero
            (rad), parallel to ``ladder``.
        params: The assist-chain parameter block.
        powered_nodes: Charge each node with :func:`_powered_node_burn`, the
            honest split-hyperbola model, instead of ADR 0007's retired
            :func:`_flyby_mismatch_burn`. Defaults to False, which reproduces
            ADR 0007's numbers.

    Returns:
        The :class:`_LadderPricing`, or None if any leg's Lambert fails.
    """
    mu = params.flyby.mu_sun
    times = [float(t) for t in leg_times]
    node_burns: List[float] = []
    excess_in: Optional[npt.NDArray[np.float64]] = None
    departure_burn = 0.0
    clock = epoch
    lon_there = 0.0
    for index in range(len(ladder) - 1):
        body, target = ladder[index], ladder[index + 1]
        tof = times[index]
        if tof <= _ASSIST_MIN_LEG_TIME:
            return None
        lon_here = longitudes0[index] + _mean_motion(body) * clock
        lon_there = longitudes0[index + 1] + _mean_motion(target) * (clock + tof)
        r0, v_body = _body_state(body.orbit_radius, lon_here, body.v_circ)
        r1, v_target = _body_state(target.orbit_radius, lon_there, target.v_circ)
        separation = float(np.linalg.norm(np.cross(r0, r1)))
        if separation < 1e-6 * body.orbit_radius * target.orbit_radius:
            # Collinear endpoints leave the transfer plane undefined.
            return None
        try:
            v_depart, v_arrive = izzo(mu, r0, r1, tof, 0, True, True, 35, 1e-8)
        except (ValueError, RuntimeError):
            return None
        excess_out = v_depart - v_body
        if excess_in is None:
            # Departure: no flyby to turn anything, so the whole excess is
            # bought with the Oberth burn at 200 km, starting from whatever the
            # mass already carries there (escape, or the cycle orbit's periapsis
            # speed once the growth loop is closed).
            speed = float(np.linalg.norm(excess_out))
            departure_burn = (
                float(np.sqrt(params.flyby.v_esc_leo**2 + speed * speed))
                - params.flyby.v_depart_from
            )
        elif powered_nodes:
            node_burns.append(_powered_node_burn(excess_in, excess_out, body)[0])
        else:
            node_burns.append(_flyby_mismatch_burn(excess_in, excess_out, body))
        excess_in = v_arrive - v_target
        clock += tof
    if excess_in is None:
        return None
    return _LadderPricing(
        departure_burn=departure_burn,
        node_total=float(sum(node_burns)),
        node_burns=node_burns,
        arrival_excess=float(np.linalg.norm(excess_in)),
        arrival_excess_vector=excess_in,
        arrival_time=clock,
        arrival_longitude=lon_there,
    )


def _body_state(
    radius: float, longitude: float, v_circ: float
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cartesian position and velocity of a body on its circular orbit.

    The chain's own model is planar and circular, so the third component is
    always zero; Lambert wants 3-vectors regardless.

    Args:
        radius: Heliocentric orbit radius (km).
        longitude: Heliocentric longitude (rad).
        v_circ: Circular speed (km/s).

    Returns:
        (position km, velocity km/s) as 3-vectors.
    """
    c, s = float(np.cos(longitude)), float(np.sin(longitude))
    position = np.array([radius * c, radius * s, 0.0])
    velocity = np.array([-v_circ * s, v_circ * c, 0.0])
    return position, velocity


def _powered_node_burn(
    excess_in: npt.NDArray[np.float64],
    excess_out: npt.NDArray[np.float64],
    body: _AssistBody,
) -> Tuple[float, float]:
    """Burn at a *powered* gravity assist supplying the required turn (km/s).

    The honest node model, replacing :func:`_flyby_mismatch_burn`. An impulsive
    tangential burn at periapsis joins an inbound hyperbola of eccentricity
    ``e_in`` to an outbound one of ``e_out``, so the total bend is
    ``asin(1/e_in) + asin(1/e_out)`` -- the same split-hyperbola geometry
    :func:`_powered_flyby_leg` uses at Jupiter. The unpowered formula
    ``sin(delta/2) = 1/e`` does **not** apply across a burn.

    Periapsis radius and burn magnitude are two knobs against two demands (the
    outgoing excess *speed* and the *turn*), so the node is solved, not
    estimated. Both eccentricities rise with ``r_p``, so the bend falls
    monotonically in it and the solve is a 1-D root find: the turn alone fixes
    the geometry, and the burn follows from it.

    The consequence is that **a node that must turn hard is cheap**: a bigger
    turn forces a lower periapsis, which deepens the well and so deepens the
    Oberth discount on the speed change. Cost is therefore *non-monotonic* in
    the turn. A 5 -> 7 km/s node at Venus costs the full 2.0 km/s at zero turn
    (nothing to bend around, so no Oberth), falls to ~1.17 km/s by 60 degrees,
    and jumps to ~3.10 km/s at 90 degrees, past the 72.9-degree bend limit.
    Against :func:`_flyby_mismatch_burn` the comparison cuts both ways: this is
    cheaper in the middle of the range, equal at zero turn, and dearer past the
    limit, because the retired model grants ``2*asin(1/e_in)`` of bend -- both
    halves at the *incoming* eccentricity -- and so overstates the limit itself
    (84.4 vs 72.9 degrees for that node).

    Args:
        excess_in: Incoming body-relative excess velocity (km/s, 3-vector).
        excess_out: Excess velocity the next leg needs (km/s, 3-vector).
        body: The flyby body.

    Returns:
        (burn magnitude km/s, periapsis radius km). When even the periapsis
        floor cannot supply the turn, the flyby bends as far as it can and the
        residual angle is bought with a deep-space kick, which makes the result
        an upper bound in that regime only.
    """
    speed_in = float(np.linalg.norm(excess_in))
    speed_out = float(np.linalg.norm(excess_out))
    if speed_in < 1e-9 or speed_out < 1e-9:
        return abs(speed_out - speed_in), body.min_periapsis

    def bend(periapsis: float) -> float:
        ecc_in = 1.0 + periapsis * speed_in * speed_in / body.mu
        ecc_out = 1.0 + periapsis * speed_out * speed_out / body.mu
        return float(np.arcsin(1.0 / ecc_in) + np.arcsin(1.0 / ecc_out))

    def burn(periapsis: float) -> float:
        v_in = float(np.sqrt(speed_in * speed_in + 2.0 * body.mu / periapsis))
        v_out = float(np.sqrt(speed_out * speed_out + 2.0 * body.mu / periapsis))
        return abs(v_out - v_in)

    cos_turn = float(
        np.clip(np.dot(excess_in, excess_out) / (speed_in * speed_out), -1.0, 1.0)
    )
    turn = float(np.arccos(cos_turn))
    floor = body.min_periapsis
    bend_max = bend(floor)
    if turn > bend_max:
        # Even a grazing pass cannot point the excess where it must go. Bend as
        # far as the floor allows, fix the speed there (with Oberth), then buy
        # the leftover angle with a deep-space kick.
        residual = turn - bend_max
        kick = 2.0 * speed_out * float(np.sin(0.5 * residual))
        return burn(floor) + kick, floor
    # Bend falls monotonically in periapsis radius; bracket upward and root it.
    high = floor
    for _ in range(200):
        if bend(high) <= turn:
            break
        high *= 2.0
    else:
        # The bend is strictly positive at every finite periapsis, so a turn at
        # or near zero exhausts the bracket. There is no flyby geometry that
        # bends this little: the pass is irrelevant and the speed change is
        # bought out at infinity, with no Oberth discount. Falling back to
        # burn(floor) here would hand the caller the *deepest* discount for the
        # turn that has earned none of it.
        return abs(speed_out - speed_in), high
    if high == floor:
        return burn(floor), floor
    periapsis = float(brentq(lambda r: bend(r) - turn, floor, high, xtol=1e-6))
    return burn(periapsis), periapsis


def _flyby_mismatch_burn(
    excess_in: npt.NDArray[np.float64],
    excess_out: npt.NDArray[np.float64],
    body: _AssistBody,
) -> float:
    """Burn needed at a flyby that cannot supply the turn unaided (km/s).

    The **retired** node model, kept to reproduce ADR 0007's numbers and to
    document the overcharge. It charges the excess-velocity change *at
    infinity*, so it collects no Oberth leverage, and it bounds the bend with
    the unpowered ``sin(delta/2) = 1/e`` -- which CONTEXT.md warns does not hold
    across a burn. Prefer :func:`_powered_node_burn`.


    An unpowered flyby rotates the excess velocity but never rescales it, and
    can only rotate it so far. Phasing the *next* leg generally demands both a
    different magnitude and a turn beyond the bend limit, and the gap is what a
    deep-space maneuver has to pay for.

    The bend limit is evaluated at the incoming speed. Charging the shortfall
    this way is the standard MGA node model: turn as far as the flyby allows
    for free, then buy the rest. It is an estimate, not an optimum -- burning
    and bending together (the burn changes the speed, which changes the limit)
    can beat it -- so treat the result as an upper bound on the node's cost.

    Args:
        excess_in: Incoming body-relative excess velocity (km/s, 3-vector).
        excess_out: Excess velocity the next leg needs (km/s, 3-vector).
        body: The flyby body.

    Returns:
        Burn magnitude, zero when the flyby can do the whole job unaided.
    """
    speed_in = float(np.linalg.norm(excess_in))
    speed_out = float(np.linalg.norm(excess_out))
    if speed_in < 1e-9 or speed_out < 1e-9:
        return abs(speed_out - speed_in)
    ecc = 1.0 + body.min_periapsis * speed_in * speed_in / body.mu
    bend_limit = 2.0 * float(np.arcsin(min(1.0, 1.0 / ecc)))
    cos_turn = float(
        np.clip(np.dot(excess_in, excess_out) / (speed_in * speed_out), -1.0, 1.0)
    )
    turn = float(np.arccos(cos_turn))
    if turn <= bend_limit:
        # The flyby can point the excess where it must go; only the speed is
        # wrong, and a colinear burn is the cheapest way to fix that.
        return abs(speed_out - speed_in)
    # Bend as far as allowed, then close the remaining angle by burning.
    residual_turn = turn - bend_limit
    return float(
        np.sqrt(
            max(
                0.0,
                speed_in * speed_in
                + speed_out * speed_out
                - 2.0 * speed_in * speed_out * np.cos(residual_turn),
            )
        )
    )


@dataclass(frozen=True)
class _ChainTerminal:
    """Float summary of the chain's Jovian leg and retrograde return.

    Attributes:
        leg_tof: Last chain body to Jupiter's orbit radius (s).
        outbound_arrival: Whether the Jovian leg arrives moving outward.
        bend_angle: Unpowered Jovian bend applied to the excess velocity (rad).
        arrival_direction: Direction of the incoming Jupiter-relative excess
            velocity in the (tangential, radial-outward) basis (rad). The bend
            is applied relative to this, so it is what bounds which outgoing
            directions -- and hence which return legs -- are reachable.
        v_infinity_jupiter: Jupiter-relative excess speed (km/s).
        return_leg: The scored retrograde return.
        total_time: Departure to first retrograde 1 AU crossing (s).
    """

    leg_tof: float
    outbound_arrival: bool
    bend_angle: float
    arrival_direction: float
    v_infinity_jupiter: float
    return_leg: _ReturnLeg
    total_time: float


@dataclass(frozen=True)
class _PoweredJovianTerminal:
    """A *powered* Jovian flyby and the retrograde return it produces.

    Attributes:
        leg_tof: Last chain body to Jupiter's orbit radius (s).
        flyby_burn: Impulsive burn at perijove (km/s).
        periapsis_radius: Perijove radius, center-based (km).
        v_infinity_in: Jupiter-relative excess speed before the burn (km/s).
        v_infinity_out: Jupiter-relative excess speed after the burn (km/s).
        turn_angle: Total split-hyperbola bend, asin(1/e_in) + asin(1/e_out)
            (rad) -- note this is a function of the *burn*, not of geometry
            alone.
        return_leg: The scored retrograde return.
        total_time: Departure to first retrograde 1 AU crossing (s).
    """

    leg_tof: float
    flyby_burn: float
    periapsis_radius: float
    v_infinity_in: float
    v_infinity_out: float
    turn_angle: float
    return_leg: _ReturnLeg
    total_time: float


def _powered_jovian_terminal(
    v_t0: float,
    v_r0: float,
    r0: float,
    elapsed: float,
    periapsis_radius: float,
    flyby_burn: float,
    bend_sign: float,
    params: _AssistChainParams,
) -> Optional[_PoweredJovianTerminal]:
    """Retrograde return from a *powered* Jovian flyby.

    :func:`_jovian_terminal` flies Jupiter unpowered, where perijove radius *is*
    the bend: one knob against two demands (the return's ``v_b`` and its timing),
    which is over-determined -- the "v_b lottery". Powering the flyby adds a
    second knob and breaks it, at the price of propellant. That is why the burn
    is load-bearing here rather than an optional extra.

    The burn changes the *turn*, not only the speed. It fires at perijove, so the
    outgoing hyperbola has its own eccentricity ``e_out`` computed from the
    post-burn excess, and the total bend is ``asin(1/e_in) + asin(1/e_out)``.
    Speeding up (``e_out > e_in``) therefore *costs* bend, and slowing down buys
    it. The unpowered ``sin(delta/2) = 1/e`` does not apply across a burn.

    Args:
        v_t0: Tangential heliocentric speed at r0 (km/s).
        v_r0: Radial-outward heliocentric speed at r0 (km/s).
        r0: Current heliocentric radius (km).
        elapsed: Chain time already spent (s).
        periapsis_radius: Perijove radius, center-based (km).
        flyby_burn: Impulsive perijove burn, >= 0 (km/s).
        bend_sign: +1 rotates the excess from tangential toward radial-outward,
            -1 the other way (which side Jupiter is passed on).
        params: The assist-chain parameter block.

    Returns:
        The earliest-arriving :class:`_PoweredJovianTerminal` whose return meets
        the v_b target, or None if none does.
    """
    flyby = params.flyby
    if periapsis_radius < flyby.periapsis_floor * (1.0 - 1e-12) or flyby_burn < 0.0:
        return None
    mu_j = flyby.mu_jupiter
    best: Optional[_PoweredJovianTerminal] = None
    for leg_tof, v_t1, v_r1, _, _ in _conic_radius_crossings(
        flyby.mu_sun, r0, v_t0, v_r0, flyby.r_jupiter_orbit
    ):
        rel_t = v_t1 - flyby.v_jupiter_orbit
        w_in = float(np.hypot(rel_t, v_r1))
        if w_in < 1e-6:
            continue
        ecc_in = 1.0 + periapsis_radius * w_in * w_in / mu_j
        v_peri_in = float(np.sqrt(w_in * w_in + 2.0 * mu_j / periapsis_radius))
        v_peri_out = v_peri_in + flyby_burn
        w_out_sq = v_peri_out * v_peri_out - 2.0 * mu_j / periapsis_radius
        if w_out_sq <= 0.0:
            continue
        w_out = float(np.sqrt(w_out_sq))
        ecc_out = 1.0 + periapsis_radius * w_out * w_out / mu_j
        turn = float(np.arcsin(1.0 / ecc_in) + np.arcsin(1.0 / ecc_out))
        angle = float(np.arctan2(v_r1, rel_t)) + bend_sign * turn
        return_leg = _flyby_return_leg(
            flyby.v_jupiter_orbit + w_out * float(np.cos(angle)),
            w_out * float(np.sin(angle)),
            flyby,
        )
        if return_leg is None:
            continue
        if return_leg.collision_speed < params.target_collision_speed:
            continue
        if return_leg.collision_speed <= flyby.v_rf:
            continue
        total_time = elapsed + leg_tof + return_leg.tof
        if total_time > params.max_trip_time:
            continue
        if best is None or total_time < best.total_time:
            best = _PoweredJovianTerminal(
                leg_tof=leg_tof,
                flyby_burn=flyby_burn,
                periapsis_radius=periapsis_radius,
                v_infinity_in=w_in,
                v_infinity_out=w_out,
                turn_angle=turn,
                return_leg=return_leg,
                total_time=total_time,
            )
    return best


def _jovian_terminal(
    v_t0: float, v_r0: float, r0: float, elapsed: float, params: _AssistChainParams
) -> Optional[_ChainTerminal]:
    """Best unpowered Jovian bend into a qualifying retrograde return.

    From a heliocentric state at radius ``r0`` (after ``elapsed`` seconds of
    chain), follows each crossing of Jupiter's orbit radius, scans the
    unpowered bend of the Jupiter-relative excess velocity within the
    periapsis-floor limit, and keeps the earliest-arriving return whose
    collision speed meets the target.

    Args:
        v_t0: Tangential heliocentric speed at r0 (km/s).
        v_r0: Radial-outward heliocentric speed at r0 (km/s).
        r0: Current heliocentric radius (km).
        elapsed: Chain time already spent (s).
        params: The assist-chain parameter block.

    Returns:
        The best :class:`_ChainTerminal`, or None if no bend qualifies.
    """
    flyby = params.flyby
    best: Optional[_ChainTerminal] = None
    for leg_tof, v_t1, v_r1, outbound, _ in _conic_radius_crossings(
        flyby.mu_sun, r0, v_t0, v_r0, flyby.r_jupiter_orbit
    ):
        rel_t = v_t1 - flyby.v_jupiter_orbit
        rel_r = v_r1
        w = float(np.hypot(rel_t, rel_r))
        if w < 1e-6:
            continue
        ecc = 1.0 + flyby.periapsis_floor * w * w / flyby.mu_jupiter
        bend_limit = 2.0 * float(np.arcsin(1.0 / ecc))
        phi = float(np.arctan2(rel_r, rel_t))
        for bend in np.linspace(-bend_limit, bend_limit, _ASSIST_JOVIAN_BEND_SAMPLES):
            angle = phi + float(bend)
            return_leg = _flyby_return_leg(
                flyby.v_jupiter_orbit + w * float(np.cos(angle)),
                w * float(np.sin(angle)),
                flyby,
            )
            if return_leg is None:
                continue
            if return_leg.collision_speed < params.target_collision_speed:
                continue
            if return_leg.collision_speed <= flyby.v_rf:
                continue
            total_time = elapsed + leg_tof + return_leg.tof
            if total_time > params.max_trip_time:
                continue
            if best is None or total_time < best.total_time:
                best = _ChainTerminal(
                    leg_tof=leg_tof,
                    outbound_arrival=outbound,
                    bend_angle=float(bend),
                    arrival_direction=phi,
                    v_infinity_jupiter=w,
                    return_leg=return_leg,
                    total_time=total_time,
                )
    return best


@dataclass(frozen=True)
class _ChainDecision:
    """One recorded decision of the beam search, replayable exactly.

    Attributes:
        body_index: Body the decision is taken at (index into params.bodies).
        rotation: Unpowered rotation applied to the excess velocity (rad;
            0 for the departure state, whose aim angle is recorded separately).
        next_body_index: Body the following leg targets, or -1 when the Jovian
            leg follows instead.
        outbound_arrival: Arrival branch at the next body (unused when
            next_body_index is -1).
    """

    body_index: int
    rotation: float
    next_body_index: int
    outbound_arrival: bool


# A beam state: (body index, excess tangential, excess radial, elapsed s,
# departure aim angle rad, decisions so far).
_ChainState = Tuple[int, float, float, float, float, Tuple[_ChainDecision, ...]]


def _assist_chain_search(
    v_infinity_earth: float, params: _AssistChainParams
) -> Optional[Tuple[float, Tuple[_ChainDecision, ...], _ChainTerminal]]:
    """Beam-search the flyby chains for the earliest qualifying return.

    Explores free-aimed departures from Earth, unpowered flyby rotations at
    Venus/Earth/Mars, and conic legs between their orbit radii, terminating
    each state through :func:`_jovian_terminal`. Deterministic: no randomness,
    stable orderings throughout.

    Args:
        v_infinity_earth: Departure hyperbolic excess speed at Earth (km/s).
        params: The assist-chain parameter block.

    Returns:
        (departure aim angle rad, recorded decisions, Jovian terminal) of the
        minimum-total-time qualifying chain, or None if none exists.
    """
    best: Optional[Tuple[float, Tuple[_ChainDecision, ...], _ChainTerminal]] = None
    states: List[_ChainState] = [
        (
            params.earth_index,
            v_infinity_earth * float(np.cos(aim)),
            v_infinity_earth * float(np.sin(aim)),
            0.0,
            float(aim),
            (),
        )
        for aim in np.linspace(-np.pi, np.pi, _ASSIST_DEPARTURE_AIM_SAMPLES)
    ]
    time_floor = (
        params.max_trip_time - _ASSIST_JOVIAN_LEG_RESERVE_YEARS * _SECONDS_PER_YEAR
    )
    for depth in range(params.max_flybys + 1):
        expanded: Dict[Tuple[int, int, int, int], _ChainState] = {}
        for body_index, rel_t, rel_r, elapsed, aim, decisions in states:
            body = params.bodies[body_index]
            w = float(np.hypot(rel_t, rel_r))
            phi = float(np.arctan2(rel_r, rel_t))
            if depth == 0:
                rotations = np.array([0.0])
            else:
                bend_limit = 2.0 * float(
                    np.arcsin(1.0 / (1.0 + body.min_periapsis * w * w / body.mu))
                )
                rotations = np.linspace(
                    -bend_limit, bend_limit, _ASSIST_ROTATION_SAMPLES
                )
            for rotation in rotations:
                angle = phi + float(rotation)
                v_t0 = body.v_circ + w * float(np.cos(angle))
                v_r0 = w * float(np.sin(angle))
                terminal = _jovian_terminal(
                    v_t0, v_r0, body.orbit_radius, elapsed, params
                )
                if terminal is not None and (
                    best is None or terminal.total_time < best[2].total_time
                ):
                    final = decisions + (
                        _ChainDecision(body_index, float(rotation), -1, True),
                    )
                    best = (aim, final, terminal)
                if depth >= params.max_flybys:
                    continue
                for next_index, next_body in enumerate(params.bodies):
                    for leg_tof, v_t1, v_r1, outbound, _ in _conic_radius_crossings(
                        params.flyby.mu_sun,
                        body.orbit_radius,
                        v_t0,
                        v_r0,
                        next_body.orbit_radius,
                    ):
                        new_elapsed = elapsed + leg_tof
                        if new_elapsed > time_floor:
                            continue
                        state: _ChainState = (
                            next_index,
                            v_t1 - next_body.v_circ,
                            v_r1,
                            new_elapsed,
                            aim,
                            decisions
                            + (
                                _ChainDecision(
                                    body_index, float(rotation), next_index, outbound
                                ),
                            ),
                        )
                        key = (
                            next_index,
                            round(state[1] / _ASSIST_DEDUP_SPEED_BIN),
                            round(state[2] / _ASSIST_DEDUP_SPEED_BIN),
                            round(
                                new_elapsed
                                / _SECONDS_PER_YEAR
                                / _ASSIST_DEDUP_TIME_BIN_YEARS
                            ),
                        )
                        if key not in expanded or new_elapsed < expanded[key][3]:
                            expanded[key] = state
        # Time-bucketed pruning: per (body, time bucket) keep the top states by
        # excess speed, with an overall per-body cap allocated to earlier
        # buckets first.
        buckets: Dict[Tuple[int, int], List[_ChainState]] = {}
        for state in expanded.values():
            bucket_key = (
                state[0],
                int(state[3] / _SECONDS_PER_YEAR / _ASSIST_BUCKET_WIDTH_YEARS),
            )
            buckets.setdefault(bucket_key, []).append(state)
        kept_per_body: Dict[int, int] = {}
        states = []
        for bucket_key in sorted(buckets, key=lambda k: k[1]):
            ranked = sorted(
                buckets[bucket_key], key=lambda s: -float(np.hypot(s[1], s[2]))
            )[:_ASSIST_BEAM_PER_BUCKET]
            already = kept_per_body.get(bucket_key[0], 0)
            taken = ranked[: max(0, _ASSIST_BEAM_CAP_PER_BODY - already)]
            kept_per_body[bucket_key[0]] = already + len(taken)
            states += taken
    return best


@dataclass(frozen=True)
class AssistChainStep:
    """One node of an assist chain: a rotation at a body, then a coast leg.

    Attributes:
        body: Body the step happens at.
        rotation_angle: Unpowered rotation of the planet-relative excess
            velocity applied by the flyby (deg; 0 at the departure step).
        v_infinity: Planet-relative excess speed at this body (km/s) -- the
            Tisserand invariant an unpowered flyby cannot change.
        target: Body (or Jupiter) the following leg coasts to.
        outbound_arrival: Whether the leg arrives at the target's orbit radius
            moving outward (True) or inward (False).
        leg_time: Coast time of the leg (yr).
        elapsed: Cumulative trip time at the end of the leg (yr).
    """

    body: str
    rotation_angle: u.Quantity
    v_infinity: u.Quantity
    target: str
    outbound_arrival: bool
    leg_time: u.Quantity
    elapsed: u.Quantity


@dataclass(frozen=True)
class AssistChainReturn:
    """An unpowered V/E/M assist chain ending in the retrograde return.

    Like :class:`PoweredJovianFlybyReturn` this is not a
    :class:`PuffSatScenario`: it blends the (small) departure burn and phasing
    budget with the collision mass ratio, so it is reported on its own.

    Attributes:
        departure_burn: Oberth burn above escape at 200 km (km/s).
        v_infinity_earth: Departure hyperbolic excess speed (km/s).
        aim_angle: Free-aim angle of the departure excess velocity off Earth's
            orbital velocity (deg).
        steps: The chain nodes in order, departure first, Jovian leg last.
        sequence: Compact chain string, e.g. "E-V-E-V-V-E-J".
        flyby_count: Number of inner-planet gravity assists (excludes the
            departure node and the Jovian bend).
        v_infinity_jupiter: Jupiter-relative excess speed at arrival (km/s).
            Tisserand-fixed: the unpowered bend rotates this vector but cannot
            change its magnitude, so it caps the reachable collision speed.
        jovian_bend_angle: Unpowered bend applied at Jupiter (deg).
        jovian_arrival_direction: Direction of the incoming Jupiter-relative
            excess velocity off Jupiter's orbital velocity (deg). The bend is
            measured from this, so it fixes which returns are reachable; see
            :func:`jovian_return_phasing_envelope`.
        return_perihelion: Perihelion of the retrograde return orbit (AU),
            never reached -- the leg ends at 1 AU.
        closing_speed_1au: Earth-relative speed at the 1 AU crossing (km/s).
        collision_speed: The achieved v_b on the
            retrograde_jovian_hohmann_transfer convention (km/s).
        chain_time: Departure to Jupiter's orbit radius (yr).
        return_time: Jovian bend to the 1 AU crossing (yr).
        total_time: chain_time + return_time (yr).
        phasing_budget: Deep-space maneuver reserve charged as spent methalox
            (km/s).
        delivered_fraction: Mass fraction after the departure burn plus the
            phasing budget.
        payload_puffsat_mass_ratio: Mass ratio at the achieved collision speed
            against the sec:jupiter_only_growth push.
        end_to_end_mass_ratio: delivered_fraction x payload_puffsat_mass_ratio.
    """

    departure_burn: u.Quantity
    v_infinity_earth: u.Quantity
    aim_angle: u.Quantity
    steps: Tuple[AssistChainStep, ...]
    sequence: str
    flyby_count: int
    v_infinity_jupiter: u.Quantity
    jovian_bend_angle: u.Quantity
    jovian_arrival_direction: u.Quantity
    return_perihelion: u.Quantity
    closing_speed_1au: u.Quantity
    collision_speed: u.Quantity
    chain_time: u.Quantity
    return_time: u.Quantity
    total_time: u.Quantity
    phasing_budget: u.Quantity
    delivered_fraction: float
    payload_puffsat_mass_ratio: float
    end_to_end_mass_ratio: float


def _chain_to_result(
    departure_burn: float,
    v_infinity_earth: float,
    aim_angle: float,
    decisions: Tuple[_ChainDecision, ...],
    terminal: _ChainTerminal,
    params: _AssistChainParams,
) -> AssistChainReturn:
    """Replay recorded beam decisions into the public Quantity-valued result.

    Re-simulates every leg from the recorded decisions (rather than trusting
    the search's intermediate floats), so a bookkeeping bug in the beam search
    surfaces here as a replay failure instead of a silently wrong table.

    Args:
        departure_burn: Oberth burn above escape at 200 km (km/s).
        v_infinity_earth: Departure hyperbolic excess speed (km/s).
        aim_angle: Departure free-aim angle (rad).
        decisions: The recorded chain decisions, terminal rotation last.
        terminal: The Jovian terminal the search scored.
        params: The assist-chain parameter block.

    Returns:
        The :class:`AssistChainReturn`.

    Raises:
        ValueError: If a recorded leg fails to replay or the replayed total
            time disagrees with the search's.
    """
    kms = u.km / u.s
    body_index = params.earth_index
    rel_t = v_infinity_earth * float(np.cos(aim_angle))
    rel_r = v_infinity_earth * float(np.sin(aim_angle))
    elapsed = 0.0
    steps: List[AssistChainStep] = []
    symbols: List[str] = []
    for decision in decisions:
        if decision.body_index != body_index:
            raise ValueError("recorded assist-chain decision is at the wrong body")
        body = params.bodies[body_index]
        w = float(np.hypot(rel_t, rel_r))
        angle = float(np.arctan2(rel_r, rel_t)) + decision.rotation
        v_t0 = body.v_circ + w * float(np.cos(angle))
        v_r0 = w * float(np.sin(angle))
        symbols.append(body.symbol)
        if decision.next_body_index < 0:
            steps.append(
                AssistChainStep(
                    body=body.name,
                    rotation_angle=np.degrees(decision.rotation) * u.deg,
                    v_infinity=w * kms,
                    target="Jupiter",
                    outbound_arrival=terminal.outbound_arrival,
                    leg_time=(terminal.leg_tof * u.s).to(u.year),
                    elapsed=((elapsed + terminal.leg_tof) * u.s).to(u.year),
                )
            )
            break
        next_body = params.bodies[decision.next_body_index]
        candidates = [
            crossing
            for crossing in _conic_radius_crossings(
                params.flyby.mu_sun,
                body.orbit_radius,
                v_t0,
                v_r0,
                next_body.orbit_radius,
            )
            if crossing[3] == decision.outbound_arrival
        ]
        if not candidates:
            raise ValueError("recorded assist-chain leg did not replay")
        leg_tof, v_t1, v_r1, _, _ = candidates[0]
        elapsed += leg_tof
        steps.append(
            AssistChainStep(
                body=body.name,
                rotation_angle=np.degrees(decision.rotation) * u.deg,
                v_infinity=w * kms,
                target=next_body.name,
                outbound_arrival=decision.outbound_arrival,
                leg_time=(leg_tof * u.s).to(u.year),
                elapsed=(elapsed * u.s).to(u.year),
            )
        )
        rel_t = v_t1 - next_body.v_circ
        rel_r = v_r1
        body_index = decision.next_body_index
    chain_seconds = elapsed + terminal.leg_tof
    replayed_total = chain_seconds + terminal.return_leg.tof
    if abs(replayed_total - terminal.total_time) > 1.0:
        raise ValueError("replayed assist-chain total time disagrees with search")
    phasing_budget = float(ASSIST_CHAIN_PHASING_BUDGET.to_value(kms))
    delivered = float(
        np.exp(-(departure_burn + phasing_budget) / params.flyby.exhaust_speed)
    )
    collision = terminal.return_leg.collision_speed
    mass_ratio = float(
        2.0 * STD_FUDGE_FACTOR / np.log(collision / (collision - params.flyby.v_rf))
    )
    return AssistChainReturn(
        departure_burn=departure_burn * kms,
        v_infinity_earth=v_infinity_earth * kms,
        aim_angle=np.degrees(aim_angle) * u.deg,
        steps=tuple(steps),
        sequence="-".join(symbols) + "-J",
        flyby_count=len(steps) - 1,
        v_infinity_jupiter=terminal.v_infinity_jupiter * kms,
        jovian_bend_angle=np.degrees(terminal.bend_angle) * u.deg,
        jovian_arrival_direction=np.degrees(terminal.arrival_direction) * u.deg,
        return_perihelion=(terminal.return_leg.perihelion * u.km).to(u.AU),
        closing_speed_1au=terminal.return_leg.closing_speed * kms,
        collision_speed=collision * kms,
        chain_time=(chain_seconds * u.s).to(u.year),
        return_time=(terminal.return_leg.tof * u.s).to(u.year),
        total_time=(replayed_total * u.s).to(u.year),
        phasing_budget=phasing_budget * kms,
        delivered_fraction=delivered,
        payload_puffsat_mass_ratio=mass_ratio,
        end_to_end_mass_ratio=delivered * mass_ratio,
    )


def venus_reach_departure_floor() -> u.Quantity:
    """Minimum 200 km Oberth burn whose transfer can touch Venus's orbit.

    At a fixed hyperbolic excess speed, the transfer perihelion is minimized
    by aiming the excess velocity retrograde (anti-tangential): that choice
    minimizes both the orbit energy and the angular momentum at once. The
    floor is the burn at which even that best aim only just brings the
    perihelion down to Venus's orbit radius -- below it Venus (and with it the
    whole assist chain) is unreachable, and *at* it the arrival is tangent to
    Venus's orbit, which is the Tisserand-locked dead end: a tangential
    arrival leaves nothing for a Venus flyby to rotate.

    Returns:
        The floor burn (astropy Quantity, km/s), ~0.2794 km/s.
    """
    params = _powered_flyby_params()
    r_venus = float(VENUS_A.to_value(u.km))
    mu = params.mu_sun
    r_earth = params.r_earth_orbit

    def perihelion_gap(burn: float) -> float:
        v_infinity = float(
            np.sqrt((params.v_esc_leo + burn) ** 2 - params.v_esc_leo**2)
        )
        v_t = params.v_earth_orbit - v_infinity
        energy = v_t * v_t / 2.0 - mu / r_earth
        h = r_earth * v_t
        p = h * h / mu
        ecc = float(np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / (mu * mu))))
        return p / (1.0 + ecc) - r_venus

    floor = brentq(perihelion_gap, 1e-6, 2.0)
    return float(floor) * u.km / u.s


def assist_chain_return(
    departure_burn: u.Quantity,
    target_collision_speed: Optional[u.Quantity] = None,
    max_trip_time: u.Quantity = ASSIST_CHAIN_MAX_TRIP_TIME,
    max_flybys: int = ASSIST_CHAIN_MAX_FLYBYS,
) -> Optional[AssistChainReturn]:
    """Best unpowered V/E/M assist chain for a given departure burn.

    Starting from C3 = 0 (PuffSat-provided escape), spends ``departure_burn``
    at 200 km, then reaches the powered flyby's retrograde return using only
    unpowered gravity assists at Venus, Earth, and Mars plus one unpowered
    Jovian bend. "Best" is the earliest-arriving chain whose collision speed
    meets the target; see CONTEXT.md ("Unpowered assist chain") for why small
    margins above the Venus-reach floor unlock the chain (square-root escape
    from the Tisserand lock).

    Args:
        departure_burn: Oberth burn above escape at 200 km (astropy Quantity).
        target_collision_speed: Minimum acceptable v_b; defaults to the
            powered-flyby optimum's collision speed (which triggers that
            optimization -- pass it explicitly when it is already at hand).
        max_trip_time: Cap on total trip time (astropy Quantity).
        max_flybys: Cap on the number of inner-planet flybys.

    Returns:
        The :class:`AssistChainReturn`, or None if no chain within the caps
        reaches the target (the beam search is a feasibility *witness*: a None
        is "not found at these beam settings", not a proof of infeasibility).

    Raises:
        ValueError: If the departure burn is not positive.
    """
    burn = float(departure_burn.to_value(u.km / u.s))
    if burn <= 0.0:
        raise ValueError("departure_burn must be positive")
    if target_collision_speed is None:
        target_collision_speed = powered_jovian_flyby_return().collision_speed
    params = _assist_chain_params(
        target_collision_speed=float(target_collision_speed.to_value(u.km / u.s)),
        max_trip_time=max_trip_time,
        max_flybys=max_flybys,
    )
    v_infinity = float(
        np.sqrt((params.flyby.v_esc_leo + burn) ** 2 - params.flyby.v_esc_leo**2)
    )
    found = _assist_chain_search(v_infinity, params)
    if found is None:
        return None
    aim_angle, decisions, terminal = found
    return _chain_to_result(burn, v_infinity, aim_angle, decisions, terminal, params)


@dataclass(frozen=True)
class JovianReturnPhasingEnvelope:
    """Which retrograde returns the chain's Jovian arrival can still reach.

    The chain's search keeps only the earliest-arriving return
    (``_jovian_terminal`` breaks ties on ``total_time``), which hides a knob:
    the same Tisserand-fixed arrival can reach a continuous span of *arrival
    times*, and retuning the bend to land Earth is free. Walking the bend from
    one limit to the other traces a single connected curve from a slow outbound
    return (the craft leaves Jupiter still climbing, coasts to aphelion and
    falls back), through full reversal, to a fast inbound one.

    The bend is **one** knob with **two** outputs, so ``v_b`` cannot be held
    while the arrival time is swept -- they move together along that curve.
    Phasing therefore picks the bend, and the bend *dictates* ``v_b``: a
    lottery within ``collision_speed_min..max``, not a choice. It is benign
    only because every ticket in that band is an acceptable growth loop. See
    ADR 0006-unpowered-jupiter-and-free-return-phasing.

    Attributes:
        v_infinity: Jupiter-relative excess speed, Tisserand-fixed (km/s).
        arrival_direction: Incoming excess-velocity direction (deg).
        bend_limit: Largest unpowered bend available at the periapsis floor
            (deg). Reachable outgoing directions are arrival_direction +/- this.
        collision_speed_min: Lowest v_b phasing may force on you (km/s).
        collision_speed_max: Highest v_b phasing may force on you (km/s); with
            no time cap this is the Tisserand ceiling, hit at full reversal.
        return_time_min: Shortest reachable return (yr).
        return_time_max: Longest reachable return (yr).
        phasing_time_spread: Reachable arrival-time span (d).
        largest_gap: Widest step between consecutive sampled arrival times (d).
            This is resolution-limited, *not* a physical hole: it falls as
            1/``samples`` (~11.9 d at 968, ~0.7 d at 16000) because the curve is
            connected. It is reported because a genuine hole would refuse to
            shrink, and without that check a span quoted as max-minus-min over a
            disconnected set would silently count the hole as authority -- the
            error this function exists to prevent.
        earth_phase_coverage: Heliocentric longitude Earth sweeps over the
            span (deg).
        wraps: earth_phase_coverage / 360 deg.
        closes: Whether coverage reaches 360 deg, i.e. every launch phase
            admits a phased Earth intercept with no propellant.
    """

    v_infinity: u.Quantity
    arrival_direction: u.Quantity
    bend_limit: u.Quantity
    collision_speed_min: u.Quantity
    collision_speed_max: u.Quantity
    return_time_min: u.Quantity
    return_time_max: u.Quantity
    phasing_time_spread: u.Quantity
    largest_gap: u.Quantity
    earth_phase_coverage: u.Quantity
    wraps: float
    closes: bool


def jovian_return_phasing_envelope(
    chain: AssistChainReturn,
    max_total_time: Optional[u.Quantity] = None,
    samples: int = _ASSIST_JOVIAN_BEND_SAMPLES * 8,
) -> JovianReturnPhasingEnvelope:
    """Measure the free Earth-phasing authority of the chain's Jovian bend.

    Sweeps every bend the periapsis floor allows about ``chain``'s actual
    arrival direction -- so every state scored here is one the unpowered flyby
    can really produce -- and reports the span of arrival times reachable, the
    worst hole in it, and the v_b range that span drags along. Converting the
    span at Earth's mean motion gives the phase coverage: at or above 360 deg
    (``closes``), a phased Earth intercept exists at *every* launch phase, with
    no propellant and no powered flyby.

    ``largest_gap`` is what makes the span meaningful rather than decorative: a
    span quoted as max-minus-min over a *disconnected* set would count holes as
    authority. Report it alongside the coverage, always. It is resolution-
    limited here (it falls as 1/``samples``), which is precisely the evidence
    that the reachable set is connected and the span is really walkable.

    Periapsis is not an extra knob here: it *is* the bend (``e = 1 +
    r_p v_inf^2 / mu``, ``sin(delta/2) = 1/e``), so the sweep below is the
    periapsis sweep. Nor does it lift ``collision_speed_max``, which is set by
    the Tisserand-fixed ``v_infinity`` at full reversal.

    Args:
        chain: The chain whose Jovian arrival is being examined.
        max_total_time: Optional cap on chain_time + return_time; returns
            exceeding it are dropped. None applies no cap.
        samples: Bend samples across the reachable arc.

    Returns:
        The :class:`JovianReturnPhasingEnvelope`.

    Raises:
        ValueError: If no retrograde return is reachable from this arrival.
    """
    params = _powered_flyby_params()
    kms = u.km / u.s
    w = float(chain.v_infinity_jupiter.to_value(kms))
    phi = float(chain.jovian_arrival_direction.to_value(u.rad))
    ecc = 1.0 + params.periapsis_floor * w * w / params.mu_jupiter
    bend_limit = 2.0 * float(np.arcsin(1.0 / ecc))
    chain_seconds = float(chain.chain_time.to_value(u.s))
    cap = None if max_total_time is None else float(max_total_time.to_value(u.s))

    speeds: List[float] = []
    times: List[float] = []
    for bend in np.linspace(-bend_limit, bend_limit, samples):
        angle = phi + float(bend)
        leg = _flyby_return_leg(
            params.v_jupiter_orbit + w * float(np.cos(angle)),
            w * float(np.sin(angle)),
            params,
        )
        if leg is None:
            continue
        if cap is not None and chain_seconds + leg.tof > cap:
            continue
        speeds.append(leg.collision_speed)
        times.append(leg.tof)
    if not speeds:
        raise ValueError("no retrograde return reachable from this Jovian arrival")

    vb = np.array(speeds)
    tof = np.sort(np.array(times))
    spread_days = float(tof.max() - tof.min()) / 86400.0
    # The span is only authority if it has no holes: a bend sweep that skipped
    # a range of arrival times could not be walked onto an arbitrary Earth
    # position, and max-minus-min would silently count the hole as coverage.
    gap_days = float(np.diff(tof).max()) / 86400.0 if tof.size > 1 else 0.0
    # Earth's mean motion from Kepler, like assist_chain_window_cadence -- the
    # phase coverage is just how far Earth walks during the reachable spread.
    earth_period_days = float(
        (2.0 * np.pi * np.sqrt(EARTH_A.to(u.km) ** 3 / Sun.k)).to_value(u.day)
    )
    coverage = spread_days * 360.0 / earth_period_days
    return JovianReturnPhasingEnvelope(
        v_infinity=w * kms,
        arrival_direction=np.degrees(phi) * u.deg,
        bend_limit=np.degrees(bend_limit) * u.deg,
        collision_speed_min=float(vb.min()) * kms,
        collision_speed_max=float(vb.max()) * kms,
        return_time_min=(float(tof.min()) * u.s).to(u.year),
        return_time_max=(float(tof.max()) * u.s).to(u.year),
        phasing_time_spread=spread_days * u.day,
        largest_gap=gap_days * u.day,
        earth_phase_coverage=coverage * u.deg,
        wraps=coverage / 360.0,
        closes=coverage >= 360.0,
    )


@dataclass(frozen=True)
class AssistChainBurnScan:
    """Result of scanning departure burns for the smallest feasible one.

    Attributes:
        target_collision_speed: The v_b floor every probe was held to (km/s).
        infeasible_burns: Probes below the found minimum where the beam search
            found no chain (km/s Quantities, ascending).
        minimum: The chain at the smallest feasible probe, or None if every
            probe failed.
    """

    target_collision_speed: u.Quantity
    infeasible_burns: Tuple[u.Quantity, ...]
    minimum: Optional[AssistChainReturn]


def minimum_departure_burn_assist_chain(
    burn_candidates: Tuple[float, ...] = ASSIST_CHAIN_BURN_CANDIDATES,
    target_collision_speed: Optional[u.Quantity] = None,
    max_trip_time: u.Quantity = ASSIST_CHAIN_MAX_TRIP_TIME,
    max_flybys: int = ASSIST_CHAIN_MAX_FLYBYS,
) -> AssistChainBurnScan:
    """Scan departure burns ascending and return the first feasible chain.

    The scan answers "how little delta-v beyond Earth escape does the assist
    chain need?" -- the unpowered counterpart of the powered flyby's 4.45 km/s
    departure burn. Because the beam search is a heuristic witness, the result
    is an upper bound on the true minimum; the probes should bracket the
    analytic Venus-reach floor (:func:`venus_reach_departure_floor`).

    Args:
        burn_candidates: Departure burns to probe, km/s (default
            ASSIST_CHAIN_BURN_CANDIDATES).
        target_collision_speed: Minimum acceptable v_b; defaults to the
            powered-flyby optimum's collision speed (computed once for the
            whole scan).
        max_trip_time: Cap on total trip time (astropy Quantity).
        max_flybys: Cap on the number of inner-planet flybys.

    Returns:
        The :class:`AssistChainBurnScan`.
    """
    if target_collision_speed is None:
        target_collision_speed = powered_jovian_flyby_return().collision_speed
    infeasible: List[u.Quantity] = []
    for burn in sorted(burn_candidates):
        result = assist_chain_return(
            departure_burn=burn * u.km / u.s,
            target_collision_speed=target_collision_speed,
            max_trip_time=max_trip_time,
            max_flybys=max_flybys,
        )
        if result is not None:
            return AssistChainBurnScan(
                target_collision_speed=target_collision_speed,
                infeasible_burns=tuple(infeasible),
                minimum=result,
            )
        infeasible.append(burn * u.km / u.s)
    return AssistChainBurnScan(
        target_collision_speed=target_collision_speed,
        infeasible_burns=tuple(infeasible),
        minimum=None,
    )


@dataclass(frozen=True)
class AssistChainWindowCadence:
    """Synodic launch-window cadence of the assist-chain family (ADR 0005).

    The phasing-free chain model cannot say *when* a chain flies, but the
    window structure follows from synodic periods: every chain starts E->V,
    so windows open at the Earth-Venus synodic cadence, gated secondarily by
    Jupiter's phase; resonant-rev stretching of the interior ladder (quantized
    Venus/Earth revolutions) plus the DSM phasing budget absorb the residual.

    Attributes:
        venus_window: Earth-Venus synodic period -- the base cadence at which
            chain launch windows open (yr).
        jupiter_window: Earth-Jupiter synodic period -- how often Jupiter's
            phase gate recurs (yr).
        earth_venus_cycle: The 5-synodic Earth-Venus near-cycle (~8 yr) after
            which the inner-ladder geometry nearly repeats (yr).
        triple_realignment: Approximate recurrence of the full
            Venus-Earth-Jupiter geometry: 15 Earth-Venus synodics, which is
            also ~22 Earth-Jupiter synodics and ~2 Jupiter years (yr).
        effective_cycle_floor: Chain trip time with zero relaunch wait (yr).
        effective_cycle_ceiling: Trip time plus one full Venus window of
            relaunch wait -- the worst case for a returning cohort (yr).
        doubling_time_floor: Fleet doubling time at the chain's end-to-end
            mass ratio and the floor cycle (yr).
        doubling_time_ceiling: The same at the ceiling cycle (yr).
    """

    venus_window: u.Quantity
    jupiter_window: u.Quantity
    earth_venus_cycle: u.Quantity
    triple_realignment: u.Quantity
    effective_cycle_floor: u.Quantity
    effective_cycle_ceiling: u.Quantity
    doubling_time_floor: u.Quantity
    doubling_time_ceiling: u.Quantity


def assist_chain_window_cadence(
    total_time: u.Quantity, end_to_end_mass_ratio: float
) -> AssistChainWindowCadence:
    """Launch-window cadence and growth-cycle timing of the assist chain.

    Computes the synodic scaffolding behind "how often can an E-V ladder
    fly": Kepler periods of Venus, Earth, and Jupiter give the Earth-Venus
    window cadence (~1.6 yr), the Earth-Jupiter phase gate (~1.1 yr), the
    ~8 yr Earth-Venus near-cycle, and the ~24 yr full realignment. The
    effective growth cycle is the chain's trip time plus between zero and one
    Venus window of relaunch wait; doubling times follow from the end-to-end
    mass ratio per cycle. This is a cadence *estimate*, not an ephemeris
    search: which specific calendar windows work, and their trip-time spread,
    would need Lambert arcs against real planet positions (ADR 0005).

    Args:
        total_time: Chain trip time, e.g. an AssistChainReturn.total_time
            (astropy Quantity).
        end_to_end_mass_ratio: Mass multiplier per cycle, e.g. an
            AssistChainReturn.end_to_end_mass_ratio (> 1).

    Returns:
        The :class:`AssistChainWindowCadence`.

    Raises:
        ValueError: If end_to_end_mass_ratio is not greater than 1 (no growth,
            so no doubling time exists).
    """
    if end_to_end_mass_ratio <= 1.0:
        raise ValueError("end_to_end_mass_ratio must exceed 1 for a growth cycle")
    mu_sun = Sun.k

    def kepler_period(semi_major_axis: u.Quantity) -> u.Quantity:
        return (2.0 * np.pi * np.sqrt(semi_major_axis.to(u.km) ** 3 / mu_sun)).to(
            u.year
        )

    def synodic_period(period_a: u.Quantity, period_b: u.Quantity) -> u.Quantity:
        return abs(1.0 / (1.0 / period_a - 1.0 / period_b)).to(u.year)

    period_venus = kepler_period(VENUS_A)
    period_earth = kepler_period(EARTH_A)
    period_jupiter = kepler_period(JUPITER_A)
    venus_window = synodic_period(period_venus, period_earth)
    jupiter_window = synodic_period(period_earth, period_jupiter)
    floor_cycle = total_time.to(u.year)
    ceiling_cycle = floor_cycle + venus_window
    doublings_per_cycle = float(np.log(end_to_end_mass_ratio) / np.log(2.0))
    return AssistChainWindowCadence(
        venus_window=venus_window,
        jupiter_window=jupiter_window,
        earth_venus_cycle=5.0 * venus_window,
        triple_realignment=15.0 * venus_window,
        effective_cycle_floor=floor_cycle,
        effective_cycle_ceiling=ceiling_cycle,
        doubling_time_floor=floor_cycle / doublings_per_cycle,
        doubling_time_ceiling=ceiling_cycle / doublings_per_cycle,
    )
