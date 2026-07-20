""" "Sorry, I Don't Need ISRU" solar-dive Earth re-intercept
(paper Appendix sec:earth_reintercept; CONTEXT.md "Heliocentric re-intercept
(solar-dive return)").

A boosted solar-dive projectile leaves periapsis on an escaping hyperbola and
crosses 1 AU only once, far from where Earth has moved to. Crossing 1 AU is not
reaching Earth: the return must be phased to an Earth resonance, and that
phasing -- not a 6-month dive -- sets the payload-doubling cycle. The functions
below prove the appendix's cited figures from the repo's own primitives and
supply the derived cycle floor used by the growth-rate estimate.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy import units as u
from boinor.bodies import Sun
from boinor.twobody import Orbit
from scipy.optimize import brentq

from src.astro_constants import (
    EARTH_A,
    SOLAR_DIVE_PERIAPSIS_BURN,
    SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
    TARGET_LAUNCH_CAPACITY_MULTIPLE,
    TWO_IMPULSE_DIP_PERIAPSIS,
)
from src.orbit_utils import (
    apoapsis_speed,
    elliptic_time_of_flight,
    escape_velocity,
    get_period,
    hyperbolic_eccentricity,
    hyperbolic_time_of_flight,
    orbit_from_rp_ra,
    periapsis_speed,
    speed_around_attractor,
    speed_at_distance,
    true_anomaly_at_radius,
)

SOLAR_DIVE_PERIAPSIS = SOLAR_DIVE_PERIAPSIS_SOLAR_RADII * Sun.R


def launch_capacity_time(
    capacity_multiple_per_loop: float,
    one_loop_elapsed_time: u.Quantity,
    target_launch_capacity_multiple: float = TARGET_LAUNCH_CAPACITY_MULTIPLE,
) -> u.Quantity:
    """Time to scale launch capacity by a target multiple at a fixed per-loop growth.

    Shared by every growth-loop flavor (the solar-dive cycle, the apoapsis-raise
    cycle): given a capacity multiple achieved per loop and the loop's elapsed
    time, how long to reach ``target_launch_capacity_multiple``.

    Args:
        capacity_multiple_per_loop: Launch-capacity multiple gained per loop.
        one_loop_elapsed_time: Elapsed time of one loop (astropy Quantity).
        target_launch_capacity_multiple: Target multiple (default
            TARGET_LAUNCH_CAPACITY_MULTIPLE).

    Returns:
        The time to reach the target multiple (astropy Quantity, years).
    """
    time_elapsed: u.Quantity = (
        np.log(target_launch_capacity_multiple)
        / np.log(capacity_multiple_per_loop)
        * one_loop_elapsed_time
    )
    return time_elapsed.to(u.year)


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
    apoapsis_radius: u.Quantity = EARTH_A,
) -> u.Quantity:
    """Hyperbolic-excess speed left after a periapsis boost at the solar dive.

    The projectile arrives from the ellipse defined by ``apoapsis_radius`` and
    ``periapsis_radius``. A tangential ``periapsis_burn`` is added to that
    ellipse's actual periapsis speed, then the resulting hyperbolic excess is
    ``sqrt(v_boosted**2 - v_esc**2)``. For the default minimum-energy dive, the
    ~306.0 km/s arrival plus a 34.5 km/s burn leaves ~143.4 km/s to spare. Using
    local escape speed as the arrival speed would inject energy the ellipse does
    not carry and overstate the result as ~150 km/s.

    Args:
        periapsis_radius: Periapsis distance from the Sun's center (astropy
            Quantity, default 4 solar radii).
        periapsis_burn: Speed increase from the PuffSat boost at periapsis
            (astropy Quantity, default SOLAR_DIVE_PERIAPSIS_BURN).
        apoapsis_radius: Aphelion of the incoming dive ellipse (astropy Quantity,
            default EARTH_A).

    Returns:
        The hyperbolic-excess (escape-to-spare) speed (astropy Quantity, km/s).

    Raises:
        ValueError: If the boosted periapsis speed does not exceed local escape
            speed, so the post-burn orbit is not hyperbolic.
    """
    v_incoming: u.Quantity = solar_dive_periapsis_speed(
        periapsis_radius=periapsis_radius,
        apoapsis_radius=apoapsis_radius,
    )
    v_escape: u.Quantity = escape_velocity(Sun, altitude=periapsis_radius - Sun.R)
    v_boosted: u.Quantity = v_incoming + periapsis_burn
    if v_boosted <= v_escape:
        raise ValueError("periapsis burn does not put the solar dive on a hyperbola")
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
        periapsis_radius=periapsis_radius,
        apoapsis_radius=apoapsis_radius,
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
        periapsis_radius=periapsis_radius,
        apoapsis_radius=apoapsis_radius,
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
    earth_speed: u.Quantity = speed_around_attractor(a=launch_radius, attractor=Sun)

    def climb_geometry(aphelion: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
        """Return the climb-out anomaly and time for an incoming trial ellipse."""
        v_infinity: u.Quantity = boosted_solar_dive_v_infinity(
            periapsis_radius=periapsis_radius,
            apoapsis_radius=aphelion,
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
        return climb_true_anomaly, climb_time

    def closure_residual_deg(aphelion_au: float) -> float:
        """Heliocentric gap (deg) between Earth and the 1 AU re-crossing for a trial aphelion."""
        aphelion: u.Quantity = aphelion_au * u.AU
        climb_true_anomaly, climb_time = climb_geometry(aphelion)
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
    _, climb_time = climb_geometry(closing_aphelion)
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
