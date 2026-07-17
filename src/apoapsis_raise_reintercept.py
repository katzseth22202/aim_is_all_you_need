"""Apoapsis-raise Earth re-intercept (candidate sec:earth_reintercept option;
design doc apoapsis_raise_reintercept_design.md in the paper source repo;
CONTEXT.md "Heliocentric re-intercept (solar-dive return)").

A projectile leaves Earth at a 200 km altitude on a C3=0 escape, an Oberth
methalox burn (Leg 1) raises its heliocentric aphelion to Q with perihelion
pinned at 1 AU, a retrograde argon-SEP burn at apoapsis (Leg 2) lowers the
perihelion, and it falls back to intercept Earth at 1 AU on the inbound leg.
The single free knob Q is root-solved for phasing-exact Earth re-intercept, the
same closure method single_impulse_resonant_dive() uses. This is the lowest-
closing-speed member of the sec:earth_reintercept family: ~24 km/s instead of
~150 km/s, needing no solar dive, no gravity assist, and no off-Earth boost
node -- only onboard propellant. The functions below reproduce the design doc's
locked design point (Sec. 4), economics (Sec. 5), and finite-thrust check
(Sec. 3) from the repo's own primitives.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.bodies import Earth, Sun
from boinor.twobody import Orbit
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src import conic_kernel
from src.astro_constants import (
    APOAPSIS_RAISE_APHELION_BRACKET,
    APOAPSIS_RAISE_SEP_BURN_DURATION,
    APOAPSIS_RAISE_SEP_DV,
    ARGON_SEP_ISP,
    EARTH_A,
    LEO_ALTITUDE,
    METHALOX_VACUUM_ISP,
    TARGET_LAUNCH_CAPACITY_MULTIPLE,
)
from src.heliocentric_reintercept import launch_capacity_time
from src.orbit_utils import (
    apoapsis_speed,
    elliptic_time_of_flight,
    escape_velocity,
    get_period,
    orbit_from_rp_ra,
    periapsis_speed,
    speed_around_attractor,
    speed_at_distance,
    speed_with_escape_energy,
    true_anomaly_at_radius,
)
from src.propulsion import (
    burn_for_v_infinity,
    exhaust_velocity_from_isp,
    payload_mass_ratio,
    rocket_equation,
)


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
    semimajor = conic_kernel.semimajor_axis_from_energy(mu, specific_energy)
    angular_momentum = x * vy - y * vx
    ecc = conic_kernel.eccentricity_from_energy_and_h(
        mu, specific_energy, angular_momentum
    )
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
