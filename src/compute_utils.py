import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy import units as u
from poliastro.bodies import Body, Earth, Moon, Saturn, Sun
from poliastro.maneuver import Maneuver
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

STD_FUDGE_FACTOR: float = 0.8


def get_burn(impulse: Tuple[npt.ArrayLike, npt.ArrayLike]) -> u.Quantity:
    """Compute the magnitude of the second impulse vector in a maneuver.

    Args:
        impulse: Tuple containing two impulse vectors (numpy arrays).

    Returns:
        The magnitude of the second impulse (astropy Quantity).
    """
    return as_scalar(impulse[1])


def get_hohmann_burns(h: Maneuver) -> List[u.Quantity]:
    """Compute the burn magnitudes for a Hohmann transfer maneuver.

    Args:
        h: A Maneuver object representing a Hohmann transfer.

    Returns:
        A list containing the magnitudes of the two burns (astropy Quantities).
    """
    i_1, i_2 = h.impulses
    return [get_burn(i_1), get_burn(i_2)]


def hohmann_transfer(
    r_i: u.Quantity, r_f: u.Quantity, attractor: Body = Sun
) -> Maneuver:
    """Compute the Hohmann transfer maneuver between two circular orbits.

    Args:
        r_i: Initial orbit radius (astropy Quantity).
        r_f: Final orbit radius (astropy Quantity).
        attractor: The central body (poliastro Body, default Sun).

    Returns:
        A Maneuver object representing the Hohmann transfer.
    """
    initial_orbit: Orbit = Orbit.circular(attractor, r_i)
    return Maneuver.hohmann(initial_orbit, r_f)


def body_speed(body: Body, altitude: u.Quantity) -> u.Quantity:
    """Compute the orbital speed at a given altitude above a body's surface.

    Args:
        body: A poliastro Body instance
        altitude: Altitude above the body's surface (astropy Quantity)

    Returns:
        The orbital speed at the given altitude (astropy Quantity, m/s)
    """

    orbit: Orbit = Orbit.circular(body, altitude)
    _, velocity_vector = orbit.rv()
    speed = np.linalg.norm(velocity_vector.value) * velocity_vector.unit
    return speed.to(u.km / u.s)


def speed_around_attractor(a: u.Quantity, attractor: Body = Sun) -> u.Quantity:
    """Compute the orbital speed at a given altitude above an attractor's surface.

    Args:
        a: Altitude above the attractor's surface (astropy Quantity).
        attractor: The central body (poliastro Body, default Sun).

    Returns:
        The orbital speed at the given altitude (astropy Quantity, km/s).
    """
    orbit: Orbit = Orbit.circular(attractor, a - attractor.R)
    _, velocity_vector = orbit.rv()
    speed = np.linalg.norm(velocity_vector.value) * velocity_vector.unit
    return speed.to(u.km / u.s)


def payload_mass_ratio(
    v_rf: u.Quantity,
    v_b: u.Quantity,
    v_ri: u.Quantity = 0 * u.km / u.s,
    fudge_factor: float = STD_FUDGE_FACTOR,
) -> float:
    """Compute the ratio of payload mass to balloon propulsion mass.
    Args:
        v_rf: Final velocity of the rocket (astropy Quantity, km/s).
        v_b: Velocity of the balloon  (astropy Quantity, km/s).
        v_ri: Initial velocity of the rocket (astropy Quantity, km/s, default 0 km/s).
        fudge_factor: Fudge factor for the mass calculation (default 0.8).

    Returns:
        The payload mass to balloon propulsion mass ratio.
    """

    denom: float = np.log((v_b - v_ri) / (v_b - v_rf)).item()
    return 2 * fudge_factor / denom


def escape_velocity(body: Body, altitude: u.Quantity = 0 * u.km) -> u.Quantity:
    """Compute the escape velocity from a body's surface or at a given altitude.

    Args:
        body: A poliastro Body instance.
        altitude: Altitude above the body's surface (astropy Quantity, default 0 km).

    Returns:
        The escape velocity at the given altitude (astropy Quantity, km/s).
    """
    # Distance from the center of the body
    r: u.Quantity = body.R + altitude
    v_esc: u.Quantity = np.sqrt(2 * body.k / r)
    return v_esc.to(u.km / u.s)


def retrograde_jovian_hohmann_transfer() -> u.Quantity:
    """Compute the reverse Hohmann transfer maneuver from Jupiter to Earth.

    Args:
        None

    Returns:
        the Earth crossing velocity in km/s
    """

    prograde_hohmann_speed = get_hohmann_burns(hohmann_transfer(JUPITER_A, EARTH_A))[1]
    earth_speed = speed_around_attractor(EARTH_A)
    # since we are retrograde, we add twice the Earth's speed
    retrograde_speed = prograde_hohmann_speed + 2 * earth_speed
    # add the poential energy of the Earth's orbit, which equals the kinetic energy of Earth's escape velocity
    earth_escape_velocity = escape_velocity(Earth)
    return np.sqrt(retrograde_speed**2 + earth_escape_velocity**2)


def get_period(body: Body, a: u.Quantity) -> u.Quantity:
    """Compute the orbital period for a given semi-major axis around a body.

    Args:
        body: A poliastro Body instance.
        a: Semi-major axis (astropy Quantity).

    Returns:
        The orbital period (astropy Quantity, seconds).
    """
    T = (2 * np.pi / np.sqrt(body.k)) * (a**1.5)
    return T.to(u.second)


def get_semimajor_axis(body: Body, T: u.Quantity) -> u.Quantity:
    """Compute the semi-major axis for a given orbital period around a body.

    Args:
        body: A poliastro Body instance.
        T: Orbital period (astropy Quantity).

    Returns:
        The semi-major axis (astropy Quantity, km).
    """
    a_cubed = (T**2 * body.k) / (4 * np.pi**2)
    a = a_cubed ** (1 / 3)
    return a.to(u.km)


def distance_to_center(altitude: u.Quantity, body: Body) -> u.Quantity:
    """Compute the distance from the center of a body given altitude and body radius.

    Args:
        altitude: Altitude above the body's surface (astropy Quantity).
        body: The celestial body (poliastro Body).

    Returns:
        The distance from the center of the body (astropy Quantity, km).
    """
    return (body.R + altitude).to(u.km)


def orbit_from_rp_ra(
    apoapsis_radius: u.Quantity,
    periapsis_radius: u.Quantity,
    attractor_body: Body = Sun,
) -> Orbit:
    """
    Generates a poliastro Orbit object aligned with the y-axis (periapsis on +y)
    and no z-component of motion (orbit in the XY-plane).

    Parameters
    ----------
       apoapsis_radius : astropy.units.Quantity
        The radius of the apoapsis (farthest point from the attractor).
        Must be an astropy Quantity with units of length (e.g., 10000 * u.km).
    periapsis_radius : astropy.units.Quantity
        The radius of the periapsis (closest point to the attractor).
        Must be an astropy Quantity with units of length (e.g., 6678 * u.km).
    attractor_body : poliastro.bodies.Body
        The central celestial body (e.g., Earth, Sun, Mars).

    Returns
    -------
    poliastro.twobody.Orbit
        The generated poliastro Orbit object.

    Raises
    ------
    ValueError
        If periapsis_radius is greater than or equal to apoapsis_radius.
    """

    if periapsis_radius >= apoapsis_radius:
        raise ValueError(
            "Periapsis radius must be less than apoapsis radius for a valid elliptical orbit."
        )

    # Calculate semi-major axis (a)
    semimajor_axis: u.Quantity = (apoapsis_radius + periapsis_radius) / 2

    # Calculate eccentricity (ecc)
    eccentricity: float = (apoapsis_radius - periapsis_radius) / (
        apoapsis_radius + periapsis_radius
    )

    # Set classical orbital elements for desired alignment
    # Inclination: 0 degrees for orbit in the XY-plane (no Z component)
    inclination: u.Quantity = 0 * u.deg
    # Right Ascension of the Ascending Node (RAAN): 0 degrees
    raan: u.Quantity = 0 * u.deg
    # Argument of Periapsis: 90 degrees to align periapsis with the positive Y-axis
    argp: u.Quantity = 90 * u.deg
    # True Anomaly: 0 degrees to start at periapsis
    true_anomaly: u.Quantity = 0 * u.deg

    # Create the Orbit object
    orbit: Orbit = Orbit.from_classical(
        attractor_body,
        semimajor_axis,
        eccentricity,
        inclination,
        raan,
        argp,
        true_anomaly,
    )

    return orbit


def periapsis_velocity(orbit: Orbit) -> u.Quantity:
    """Return the velocity vector at periapsis for a given poliastro Orbit.

    Args:
        orbit: A poliastro Orbit object.

    Returns:
        The velocity vector at periapsis (astropy Quantity, km/s).
    """
    # Create a new orbit at true anomaly = 0 deg (periapsis)
    orbit_at_periapsis = Orbit.from_classical(
        orbit.attractor,
        orbit.a,
        orbit.ecc,
        orbit.inc,
        orbit.raan,
        orbit.argp,
        0 * u.deg,
    )
    _, v_vec = orbit_at_periapsis.rv()
    return as_scalar(v_vec)


def apoapsis_velocity(orbit: Orbit) -> u.Quantity:
    """Return the velocity vector at apoapsis for a given poliastro Orbit.

    Args:
        orbit: A poliastro Orbit object.

    Returns:
        The velocity vector at apoapsis (astropy Quantity, km/s).
    """
    # Create a new orbit at true anomaly = 180 deg (apoapsis)
    orbit_at_apoapsis = Orbit.from_classical(
        orbit.attractor,
        orbit.a,
        orbit.ecc,
        orbit.inc,
        orbit.raan,
        orbit.argp,
        180 * u.deg,
    )
    _, v_vec = orbit_at_apoapsis.rv()
    return as_scalar(v_vec)


def velocity_at_distance(
    radius_periapsis: u.Quantity,
    velocity_periapsis: u.Quantity,
    distance: u.Quantity,
    attractor_body: Body = Sun,
) -> u.Quantity:
    """
    Compute the scalar orbital velocity at a given distance from the central body, given the periapsis radius and velocity.

    Parameters
    ----------
    radius_periapsis : astropy.units.Quantity
        The radius at periapsis (with length units).
    velocity_periapsis : astropy.units.Quantity
        The scalar velocity at periapsis (with velocity units).
    distance : astropy.units.Quantity
        The distance from the center of the attractor at which to compute the velocity (with length units).
    attractor_body : poliastro.bodies.Body
        The central celestial body (e.g., Earth, Sun).

    Returns
    -------
    astropy.units.Quantity
        The scalar orbital velocity at the given distance (with velocity units).

    Raises
    ------
    ValueError
        If the computed velocity is not real (e.g., for unphysical parameters).
    """
    mu = attractor_body.k
    r_p = radius_periapsis
    v_p = velocity_periapsis
    # Compute semi-major axis from vis-viva at periapsis
    # v_p^2 = mu * (2/r_p - 1/a)  => 1/a = 2/r_p - v_p^2/mu
    one_over_a = 2 / r_p - v_p**2 / mu

    # Handle parabolic orbit case (1/a ≈ 0)
    if np.isclose(one_over_a, 0, atol=1e-15):
        # For parabolic orbit: v = sqrt(2*mu/r)
        v2 = 2 * mu / distance
    else:
        a = 1 / one_over_a
        # Now compute velocity at the given distance
        v2 = mu * (2 / distance - 1 / a)
    if v2 < 0 * v2.unit:
        raise ValueError("No real velocity at this distance for the given orbit.")
    return np.sqrt(v2).to(u.km / u.s)


def as_scalar(vec: npt.ArrayLike) -> u.Quantity:
    """Return the norm of a vector as an astropy Quantity in km/s if possible, else as a float."""
    norm = np.linalg.norm(vec)
    # If input is an astropy Quantity, norm will be a Quantity; else, it's a float
    if isinstance(norm, u.Quantity):
        return norm.to(u.km / u.s)
    else:
        return norm * u.dimensionless_unscaled


def find_periapsis_radius_from_apoapsis_and_velocity(
    apoapsis_radius: u.Quantity,
    periapsis_velocity: u.Quantity,
    attractor_body: Body = Sun,
) -> u.Quantity:
    """
    Compute the periapsis radius of an orbit given the apoapsis radius, the scalar velocity at periapsis, and the central attractor.

    The function solves the vis-viva equation for the periapsis radius, assuming an elliptical orbit
    aligned with the y-axis and no z-component (orbit in the XY-plane).

    Parameters
    ----------
    apoapsis_radius : astropy.units.Quantity
        The radius of the apoapsis (farthest point from the attractor), with length units.
    periapsis_velocity : astropy.units.Quantity
        The scalar velocity at periapsis, with velocity units.
    attractor_body : poliastro.bodies.Body, optional
        The central celestial body (default: Sun).

    Returns
    -------
    astropy.units.Quantity
        The computed periapsis radius (with length units).

    Raises
    ------
    ValueError
        If the input parameters do not yield a real, positive periapsis radius.
    """
    # The quadratic equation is: A * rp^2 + B * rp + C = 0
    A = periapsis_velocity**2
    B = periapsis_velocity**2 * apoapsis_radius
    C = -2 * attractor_body.k * apoapsis_radius

    # Calculate the discriminant
    discriminant = B**2 - 4 * A * C

    # Ensure the discriminant is non-negative for a real solution
    if discriminant < 0:
        raise ValueError(
            "Invalid parameters: No real solution for periapsis radius. Check input values."
        )

    # Calculate the two possible solutions for periapsis radius
    rp1 = (-B + np.sqrt(discriminant)) / (2 * A)
    rp2 = (-B - np.sqrt(discriminant)) / (2 * A)
    rp1 = rp1.to(u.km)
    rp2 = rp2.to(u.km)

    # In orbital mechanics, radius must be positive.
    # We take the positive root. If both are positive, the problem context implies a physically
    # meaningful solution. In this case, the larger velocity at periapsis implies a smaller
    # periapsis radius, so we take the positive root.
    if rp1 > 0 and rp2 > 0:
        # Both roots are positive. Choose the one that makes physical sense.
        # Since Vp is given as the periapsis velocity, it implies rp < ra.
        # The equation derived earlier directly yields the correct physical radius.
        return rp1 if rp1 < apoapsis_radius else rp2
    elif rp1 > 0:
        return rp1
    elif rp2 > 0:
        return rp2
    else:
        raise ValueError("No positive solution for periapsis radius found.")


def get_orbital_velocity_at_radius(orbit: Orbit, radius: u.Quantity) -> u.Quantity:
    """
    Calculates the scalar orbital velocity at a given radial distance from the attractor body.

    Parameters
    ----------
    orbit : poliastro.twobody.orbit.Orbit
        The poliastro Orbit object, containing the orbital elements and attractor body.
    radius : astropy.units.Quantity
        The radial distance from the attractor body at which to calculate the velocity.
        Must be a scalar astropy Quantity with units of length.

    Returns
    -------
    astropy.units.Quantity
        The scalar orbital velocity at the given radius, with units of velocity.

    Raises
    ------
    ValueError
        If the provided radius is outside the valid range for the given orbit
        (e.g., negative for elliptical/parabolic/hyperbolic orbits, or
        greater than apoapsis for elliptical orbits if not handled correctly
        for specific velocity calculations).
    """
    if not isinstance(radius, u.Quantity) or not radius.unit.is_equivalent(u.km):
        raise TypeError("Radius must be an astropy Quantity with units of length.")
    if radius.size != 1:
        raise ValueError("Radius must be a scalar quantity.")

    # Gravitational parameter of the attractor body
    mu = orbit.attractor.k

    # Semimajor axis
    a = orbit.a

    # Specific energy (epsilon) for the orbit
    # epsilon = -mu / (2 * a)

    # Velocity formula for any conic section (vis-viva equation):
    # v = sqrt(mu * (2/r - 1/a))
    # For parabolic orbit, a approaches infinity, so 1/a approaches 0.
    # For hyperbolic orbit, a is negative, so 1/a is also negative.

    # Check for valid radius range depending on orbit type
    if orbit.ecc < 1:  # Elliptical orbit
        if radius < orbit.r_p or radius > orbit.r_a:
            # For an elliptical orbit, the radius must be between periapsis and apoapsis.
            # However, the Vis-Viva equation itself doesn't strictly break outside this range;
            # it would just give an imaginary velocity, indicating an invalid physical point.
            # We can allow the calculation as long as 2/r - 1/a > 0.
            pass  # The formula will handle it by returning a NaN if the sqrt is of a negative number
    elif orbit.ecc == 1:  # Parabolic orbit
        if radius < 0 * u.km:  # Radius must be non-negative
            raise ValueError("Radius cannot be negative for a parabolic orbit.")
        # For parabolic orbit, 1/a term becomes 0, Vis-Viva simplifies to sqrt(2*mu/r)
        # However, the general formula still works as a approaches infinity.
        pass
    else:  # Hyperbolic orbit (ecc > 1)
        if radius < 0 * u.km:  # Radius must be non-negative
            raise ValueError("Radius cannot be negative for a hyperbolic orbit.")
        # For hyperbolic orbits, 'a' is negative, but the Vis-Viva equation accounts for this.
        pass

    # Vis-viva equation
    # Make sure units are consistent before calculation
    velocity_squared = mu * (2 / radius - 1 / a)

    # Ensure the term inside the square root is non-negative
    if velocity_squared.value < 0:
        raise ValueError(
            f"The provided radius ({radius}) is not physically reachable for this orbit. "
            "Velocity would be imaginary. Ensure radius is within the orbit's bounds."
        )

    velocity = np.sqrt(velocity_squared)

    return velocity


def retrograde_orbit(orbit: Orbit) -> Orbit:
    """Return a new Orbit with the same shape as the input but with retrograde velocity.

    Args:
        orbit: A poliastro Orbit object.

    Returns:
        A new Orbit object with the same position but velocity reversed (retrograde).
    """
    r_vec, v_vec = orbit.rv()
    retrograde_v_vec = -v_vec
    return Orbit.from_vectors(
        orbit.attractor, r_vec, retrograde_v_vec, epoch=orbit.epoch
    )


def rocket_equation(delta_v: u.Quantity, exhaust_v: u.Quantity) -> u.Quantity:
    """
    Compute the fractional propellant mass required for a given delta-v using the Tsoilkovksy trocket equation.

    Parameters
    ----------
    delta_v : astropy.units.Quantity
        The total change in velocity required (delta-v), with velocity units.
    exhaust_v : astropy.units.Quantity
        The effective exhaust velocity of the rocket, with velocity units.

    Returns
    -------
    astropy.units.Quantity
        The fractional propellant mass required (dimensionless, as a Quantity).
    """
    return 1 - np.exp(-delta_v / exhaust_v)


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
        prograde_dv_parker_burn = burn_for_v_infinity(prograde_v_infinity_earth_to_parker)
        retrograde_jovian_speed = retrograde_jovian_hohmann_transfer()
        desc = """Balloons approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe"""
        BalloonScenario(
            v_rf=prograde_dv_parker_burn, v_b=retrograde_jovian_speed, desc=desc
        ).append(scenario_table)
        desc = """Balloons approach Earth from Jupiter retrograde Hohmann trajectory and push the object to escape velocity and then to a periapsis near Parker Space probe but in a retrograde orbit around the Sun"""
        # v_infinity for retrograde transfer from Earth to Parker orbit apoapsis
        retrograde_v_infinity_earth_to_parker = earth_speed + parker_apoapsis_velocity
        # calculate burn needed to achieve this v_infinity from Earth
        retrograde_dv_parker_burn = burn_for_v_infinity(retrograde_v_infinity_earth_to_parker)
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


def burn_for_v_infinity(
    v_infinity: u.Quantity,
    body: Body = Earth,
    altitude: u.Quantity = LEO_ALTITUDE,
    initial_velocity: u.Quantity = 0 * u.km / u.s,
) -> u.Quantity:
    """Calculate the burn required to achieve a specific v_infinity.

    This function computes the delta-v needed to achieve a desired v_infinity
    (hyperbolic excess velocity) when starting from a given altitude above a celestial body.
    The burn is applied at the specified altitude to achieve the target v_infinity.

    Args:
        v_infinity: The desired hyperbolic excess velocity (astropy Quantity).
        body: The celestial body to escape from (poliastro Body, default Earth).
        altitude: Altitude above the body's surface where the burn occurs (astropy Quantity, default LEO_ALTITUDE).
        initial_velocity: Initial velocity at the burn altitude, defaults to 0 km/s (astropy Quantity).

    Returns:
        The required burn (delta-v) to achieve the v_infinity (astropy Quantity).

    Raises:
        ValueError: If v_infinity is less than or equal to zero.
    """
    if v_infinity <= 0 * u.km / u.s:
        raise ValueError("v_infinity must be positive.")

    # Distance from the center of the body at burn altitude
    burn_radius: u.Quantity = body.R + altitude

    # Escape velocity at the burn altitude
    escape_velocity_at_altitude: u.Quantity = escape_velocity(body, altitude)

    # For a hyperbolic orbit, the total velocity at the burn point is:
    # v_total^2 = v_escape^2 + v_infinity^2
    # This is derived from the vis-viva equation for hyperbolic orbits

    total_velocity_squared: u.Quantity = escape_velocity_at_altitude**2 + v_infinity**2
    total_velocity: u.Quantity = np.sqrt(total_velocity_squared)

    # The burn required is the difference between total velocity and initial velocity
    required_burn: u.Quantity = total_velocity - initial_velocity

    return required_burn.to(u.km / u.s)


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
