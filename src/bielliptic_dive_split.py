"""Split the solar-dive injection across two nodes, and phase the second one.

The paper's no-ISRU growth loop pays for its solar dive at Earth: the
**single-impulse resonant dive** (``sec:earth_reintercept``,
:func:`heliocentric_reintercept.single_impulse_resonant_dive`) folds the phasing
and the dive into one 37.53 km/s heliocentric boost, 28.16 km/s of it charged
from the 20-day parking orbit.  That is the largest single number in the cycle.

This module asks whether the injection can be **split**: a small prograde push at
Earth that only raises aphelion, a long coast, and the dive injection paid out at
the far end where the vehicle is barely moving.  Raising 1 AU to a 4 solar-radii
perihelion is a 54:1 radius ratio, so the bi-elliptic route is cheaper than the
direct one by construction -- 16.94 km/s through a 3 AU aphelion against 24.09
direct, with a floor of 12.34 km/s as the aphelion runs to infinity.

Four results, in the order they matter (ADR
``0023-the-split-dive-buys-the-pad-not-the-clock.md``):

* **The delta-v saving is real and does not buy growth.**  Charged on the same
  nozzle and the same impactor-scarce currency as ADR 0019, the fully closed
  two-node cycle doubles in 0.523 yr against the paper's 0.305 yr.  Impulse
  enters growth logarithmically and the clock enters linearly, so 2.7x less slug
  buys 1.3x in e-foldings against 2.55x more clock.  What it does buy is the
  **pad**: 0.879 kg of launched slug per delivered kilogram against 2.373.
* **A partial split is free.**  Stopping short -- outbound perihelion ~0.49 AU
  rather than 1 AU, on the *existing* re-intercept resonance -- doubles in
  0.2974 yr against 0.3053 and spends 1.54 kg/kg against 2.373.  Better on both
  axes, and the only strictly dominating result here.
* **The far node can be fed by the beam that already fed Earth.**  The boosted
  climb-out leaves 4 solar radii almost radially (0.7-2.3 degrees), so its 1 AU
  crossing and its 3 AU crossing are the *same ray* -- 21 days and 1.5 degrees
  apart.  The outer node eats what Earth did not catch.  That is not free in
  geometry, though: it takes three conditions on three knobs and the solution
  set is **discrete**, indexed by the re-intercept revolution count and by the
  rational beam-reuse fraction whose denominator is the number of interleaved
  chains.
* **And this is the architecture that can fly a shallow dive at all.**  Backing
  the dive out weakens the return beam *and* the boost it has to deliver, so the
  direct route does not fail on impulse -- it fails on **conduction**.  The Earth
  departure runs nearly along the arriving stream, so it cools itself through the
  burn (148.3 to 125.0 km/s at 4 solar radii, 95.4 to 76.3 at 23), and it drops
  through ADR 0022's 79.56 km/s floor at **19.80 solar radii**.  The split's Earth
  burn is a twentieth the size and never does; its far node thrusts perpendicular
  to the stream and so cannot cool itself at all.  ADR 0022 already put the
  shallow end at 23 solar radii, which is past where the single-impulse
  architecture runs out.

The model is circular and coplanar, in the same spirit as
:mod:`src.jovian_solar_dive_cycle`, and all algebra is float (km, s, km/s, rad)
delegating to :mod:`src.conic_kernel`.  The Earth-node burn is not re-derived
here: it is :func:`jovian_solar_dive_cycle.departure_nozzle_ledger`, the same
device, so the rows compare directly.

Run with ``make split-dive``.
"""

import argparse
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from boinor.bodies import Sun
from scipy.optimize import brentq, fsolve, minimize_scalar
from tabulate import tabulate

from src.astro_constants import (
    EARTH_A,
    SOLAR_DIVE_PERIAPSIS_BURN,
    SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
)
from src.circular_resonance_impulse import impulse_per_impactor_kg
from src.conic_kernel import (
    ConicState,
    conic_state_at_radius,
    elliptic_time_from_periapsis,
    hyperbolic_eccentricity,
    hyperbolic_tof_seconds,
    speed_components_at_radius,
    true_anomaly_at_radius_rad,
)
from src.heliocentric_reintercept import single_impulse_resonant_dive
from src.jovian_solar_dive_cycle import (
    DEFAULT_JET_ENERGY_EFFICIENCY,
    DEFAULT_PERIAPSIS_SURVIVAL,
    DEFAULT_SLUG_RATIO,
    DepartureNozzleLedger,
    _vehicle_frame_impact,
    departure_nozzle_ledger,
    paper_resonant_dive_ledger,
)
from src.retrograde_return_legs import _FlybyParams, _powered_flyby_params
from src.solar_dive_depth_trade import (
    DEFAULT_EXPANSION_MARGIN,
    conduction_threshold_closing_speed,
    direct_retrograde_placement_excess,
)

_SECONDS_PER_YEAR = 365.25 * 86400.0
_MU_SUN = float(Sun.k.to_value(u.km**3 / u.s**2))
_AU = float(EARTH_A.to_value(u.km))
_SOLAR_RADIUS = float(Sun.R.to_value(u.km))
_DIVE_PERIAPSIS = SOLAR_DIVE_PERIAPSIS_SOLAR_RADII * _SOLAR_RADIUS
_PERIAPSIS_BURN = float(SOLAR_DIVE_PERIAPSIS_BURN.to_value(u.km / u.s))

# --- Search bounds.  Recorded here, not just in a scratch harness: ADR 0007's
#     numbers became unreproducible when its bounds were left out (CLAUDE.md).
# Outbound perihelion bracket, in AU.  The floor is the dive perihelion itself
# (where the split degenerates to the paper's single impulse and the outer burn
# is zero); the ceiling stops just short of 1 AU, where the departure excess goes
# to zero and the Earth node has nothing to do.
OUTBOUND_PERIHELION_BRACKET_AU = (_DIVE_PERIAPSIS / _AU, 0.995)
# Outbound aphelion bracket, in AU.  Wide enough to hold both re-intercept
# resonances the loop admits (~1.9 AU at zero extra revolutions, ~3.0 AU at one)
# with room on each side; 12 AU is past any aphelion whose fall time keeps the
# cycle under the ~9.2 yr disqualification bound (CONTEXT.md).
CLOSING_APHELION_BRACKET_AU = (1.02, 12.0)
# Dive-injection true anomaly bracket, in degrees on the outbound ellipse.  180
# is aphelion; the phasing knob is how far past it the burn is taken.  The
# bracket is deliberately wide -- the admissible roots sit within 15 degrees of
# aphelion, and anything outside this range is a degenerate root that has walked
# back onto the Earth node.
INJECTION_ANOMALY_BRACKET_DEG = (150.0, 240.0)
# A root whose coast to the outer node is shorter than this is the degenerate
# one: the "outer" node has collapsed back onto the 1 AU departure point, which
# is the paper's single-impulse dive wearing two burns' clothing.
_DEGENERATE_COAST_YEARS = 0.05
# Steps used to integrate the outer-node burn, matching the departure ledger's.
# Both the impact angle and the closing speed move through the burn, and a
# midpoint evaluation hides it.
_BURN_INTEGRATION_STEPS = 400
# Tangential-speed samples bounding the cheapest dive-injection impulse.  The
# lower end is the tangential speed that would put the burn radius exactly at the
# new orbit's apoapsis (below it the target orbit cannot reach the burn point);
# the upper end is the tangential speed the vehicle already has, since lowering
# perihelion never wants more angular momentum.
_INJECTION_SEARCH_TOLERANCE = 1e-10
# The 5/8 phased closure's own Earth departure, used as the split's stand-in
# whenever the depth dial is swept: holding it fixed while the depth moves is
# what isolates the conduction question from the phasing one.
_SPLIT_DEPARTURE_EXCESS = 10.093
_SPLIT_APHELION = 2.9400 * _AU
# ADR 0022's admissible pad-floor crossing, in solar radii: past this the cycle
# stops returning the fifteenth of liftoff it committed to, whichever
# architecture flies it.  Quoted rather than recomputed --
# solar_dive_depth_trade.admissible_pad_floor_depth() is a ~210 s bisection, and
# nothing here needs it to more precision than "where the dial ends".
PAD_ADMISSIBLE_DEPTH = 22.93
# Depths to bisect the direct departure's conduction crossing between, in solar
# radii.  The floor is the paper's own 4; the ceiling is past ADR 0020's 32 and
# ADR 0022's 22.93 crossing with room to spare.
DEPTH_CONDUCTION_BRACKET = (4.0, 48.0)
# Depths to bisect the plunger's thermally-equivalent head-on depth between, in
# solar radii.  Wide, because halving the closing speed is worth roughly a factor
# of two in radius and the shallow end of the dial already reaches 32.
PLUNGER_EQUIVALENT_DEPTH_BRACKET = (2.0, 200.0)
# Slug ratios reported in the node-geometry table.  3 is near the node's own
# exhaust-speed optimum, 8.5 is solar_dive_depth_trade's CONSERVATIVE_SLUG_RATIO,
# 30 is DEFAULT_PERIAPSIS_SLUG_RATIO -- the trade's answer changes sign of
# usefulness across them, so quoting one alone would mislead.
NODE_GEOMETRY_SLUG_RATIOS: Tuple[float, ...] = (1.0, 3.0, 8.5, 30.0)
# Depths reported in the conduction table.  16 is Seth's proposed first shallow
# dive, 23 is ADR 0022's admissible pad-floor crossing, 32 is ADR 0020's.
DEPTH_CONDUCTION_SAMPLES: Tuple[float, ...] = (4.0, 8.0, 12.0, 16.0, 23.0, 32.0)
# Beam-reuse fractions worth reporting: the numerator is how many node cadences
# separate a beam's Earth crossing from the outer-node burn it feeds, the
# denominator is the number of interleaved chains that must be in flight.
DEFAULT_REUSE_FRACTIONS: Tuple[Tuple[int, int], ...] = ((3, 5), (5, 8), (8, 13))
# Initial guesses for the three-condition solve, in (perihelion AU, aphelion AU,
# injection anomaly deg).  All are taken from the one-revolution branch, where
# the admissible family lives; the ladder exists because `fsolve` reports "not
# making good progress" on a converged root whenever the requested step
# tolerance is finer than the residuals' own floating-point floor, so the solve
# is judged on the residuals it actually reaches rather than on that flag.
CLOSURE_GUESSES: Tuple[Tuple[float, float, float], ...] = (
    (0.90, 2.95, 191.0),
    (0.75, 3.00, 190.7),
    (0.60, 3.04, 189.8),
    (0.97, 2.92, 191.4),
)
# Residual tolerance a three-condition root must meet: degrees on each closure,
# and dimensionless on the reuse fraction.  Roots land near 1e-9; 1e-6 is loose
# enough to survive the solver's own arithmetic and tight enough that a
# half-converged root cannot pass.
CLOSURE_RESIDUAL_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class DiveInjection:
    """The cheapest single impulse onto an orbit with a given perihelion.

    Attributes:
        delta_v: Magnitude of the impulse (km/s).
        v_tangential: Tangential speed after the burn (km/s).
        v_radial: Radial speed after the burn (km/s), positive outward.
        semi_major_axis: Semi-major axis of the dive ellipse (km).
        eccentricity: Eccentricity of the dive ellipse.
        true_anomaly: True anomaly of the burn point on the dive ellipse (rad).
        thrust_axis: Direction of the impulse, measured from prograde with
            radially outward at +90 degrees (rad).
    """

    delta_v: float
    v_tangential: float
    v_radial: float
    semi_major_axis: float
    eccentricity: float
    true_anomaly: float
    thrust_axis: float


@dataclass(frozen=True)
class SplitDiveGeometry:
    """One split-dive loop, leg by leg, with both closure residuals.

    Attributes:
        outbound_perihelion: Perihelion of the outbound ellipse (km).
        outbound_aphelion: Aphelion of the outbound ellipse (km).
        injection_anomaly: True anomaly at which the dive injection fires, on
            the outbound ellipse (rad); pi is aphelion.
        dive_perihelion: Perihelion the dive reaches (km).
        periapsis_burn: Tangential boost taken there (km/s).
        departure_excess: Earth-relative excess the departure must reach (km/s).
        departure_aim: Direction that excess must point, from Earth's prograde,
            radially outward at +90 degrees (deg).
        coast_years: Departure to outer node (yr).
        fall_years: Outer node to the solar perihelion (yr).
        climb_years: Solar perihelion back out to 1 AU (yr).
        coast_sweep: Heliocentric longitude swept on the coast (deg).
        fall_sweep: Longitude swept falling to perihelion (deg).
        climb_sweep: Longitude swept climbing back to 1 AU (deg).
        injection: The outer-node impulse.
        node_radius: Heliocentric radius of the outer node (km).
        return_excess: Heliocentric excess speed of the boosted climb-out (km/s).
        stream_excess: Earth-relative excess of that stream at 1 AU (km/s).
        stream_axis: Direction it travels at Earth, from prograde (deg).
        beam_transit_years: Beam time from its 1 AU crossing to the outer node.
        beam_sweep: Longitude the beam sweeps over that leg (deg).
        cycle_years: Departure to the 1 AU re-crossing (yr).
        reintercept_residual: Earth's advance minus the longitude swept, wrapped
            to (-180, 180] (deg).  Zero is the **Earth re-intercept** closure.
        colocation_residual: Same wrap, for the outer node sitting on a beam ray
            (deg).  Zero is the **outer-node co-location** closure.
        reuse_fraction: Beam-reuse offset divided by the cycle time.
    """

    outbound_perihelion: float
    outbound_aphelion: float
    injection_anomaly: float
    dive_perihelion: float
    periapsis_burn: float
    departure_excess: float
    departure_aim: float
    coast_years: float
    fall_years: float
    climb_years: float
    coast_sweep: float
    fall_sweep: float
    climb_sweep: float
    injection: DiveInjection
    node_radius: float
    return_excess: float
    stream_excess: float
    stream_axis: float
    beam_transit_years: float
    beam_sweep: float
    cycle_years: float
    reintercept_residual: float
    colocation_residual: float
    reuse_fraction: float


@dataclass(frozen=True)
class SplitDiveLedger:
    """What one split-dive cycle multiplies, on ADR 0019's currency.

    Both nodes are magnetic nozzles fed by the returning stream, and only the
    slug is drawn from the vehicle, so the rocket equation is charged against
    slug at each node and the impactor bill is the two nodes' slug divided by
    the **slug ratio**.  Scored exactly as
    :func:`jovian_solar_dive_cycle.cycle_growth_ledger` scores a one-node cycle,
    so the rows are comparable line for line.

    Attributes:
        geometry: The loop this scores.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        periapsis_survival: Mass fraction surviving the solar-periapsis pass.
        departure: The Earth node's ledger.
        outer_impact_angle_start: Impact angle as the outer burn lights (deg).
        outer_impact_angle_end: Impact angle as it finishes (deg).
        outer_closing_speed: Closing speed at the outer node (km/s).
        outer_delivered_fraction: Mass fraction surviving the outer burn.
        delivered_fraction: Mass fraction surviving both burns.
        slug_per_delivered_kg: Launched slug spent per kilogram delivered to the
            dive -- the pad-side currency.
        impactor_per_delivered_kg: The same in impactor kilograms.
        earth_caught_fraction: Share of one wave consumed at the Earth node, the
            rest flying on to the outer node.
        return_per_impactor_kg: Returning kilograms per impactor kilogram.
        growth_rate: Natural-log growth per year (e-foldings/yr).
        doubling_years: Payload doubling time (yr).
    """

    geometry: SplitDiveGeometry
    slug_ratio: float
    jet_energy_efficiency: float
    periapsis_survival: float
    departure: DepartureNozzleLedger
    outer_impact_angle_start: float
    outer_impact_angle_end: float
    outer_closing_speed: float
    outer_delivered_fraction: float
    delivered_fraction: float
    slug_per_delivered_kg: float
    impactor_per_delivered_kg: float
    earth_caught_fraction: float
    return_per_impactor_kg: float
    growth_rate: float
    doubling_years: float


@dataclass(frozen=True)
class TwoNodeClosure:
    """A fully phased split-dive cycle: both closures met, beam reuse rational.

    Attributes:
        reuse_numerator: Node cadences between a beam's Earth crossing and the
            outer-node burn it feeds.
        interleaved_chains: Denominator of the beam-reuse fraction -- how many
            payload chains must be in flight at once.
        revolutions: Extra whole turns Earth makes during the loop, the
            re-intercept resonance index.
        geometry: The closed loop.
        ledger: Its growth ledger.
        node_cadence_years: Cycle time divided by the chain count -- how often a
            beam reaches Earth in steady state (yr).
    """

    reuse_numerator: int
    interleaved_chains: int
    revolutions: int
    geometry: SplitDiveGeometry
    ledger: SplitDiveLedger
    node_cadence_years: float


def _wrap_degrees(angle: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return float((angle + 180.0) % 360.0 - 180.0)


def dive_injection_impulse(
    radius: float,
    v_tangential: float,
    v_radial: float,
    target_perihelion: float = _DIVE_PERIAPSIS,
) -> DiveInjection:
    """Cheapest single impulse onto an orbit with perihelion ``target_perihelion``.

    The post-burn tangential speed is the free parameter: fixing it fixes the
    angular momentum, the perihelion constraint then fixes the energy, and
    vis-viva gives the radial speed up to a sign.  Both signs are searched --
    inbound turns the vehicle straight down towards perihelion, outbound lets it
    coast to a new aphelion first -- and the cheaper one wins.  At aphelion, where
    ``v_radial`` is zero, this reduces to the tangential burn
    ``v_apo(r_p_old, r_a) - v_apo(r_p_new, r_a)``.

    Args:
        radius: Heliocentric radius of the burn (km).
        v_tangential: Tangential speed before the burn (km/s), positive prograde.
        v_radial: Radial speed before the burn (km/s), positive outward.
        target_perihelion: Perihelion the burn must place (km).

    Returns:
        The :class:`DiveInjection`.

    Raises:
        ValueError: If no orbit with that perihelion reaches ``radius`` at any
            tangential speed at or below the vehicle's own.
    """
    # An orbit with perihelion target_perihelion reaches `radius` only if its
    # apoapsis is at least `radius`; that fixes the floor on tangential speed.
    tangential_floor = float(
        np.sqrt(
            2.0 * _MU_SUN * target_perihelion / (radius * (target_perihelion + radius))
        )
    )
    if tangential_floor > v_tangential * (1.0 + 1e-9):
        raise ValueError(
            f"no orbit with perihelion {target_perihelion} km reaches {radius} km "
            f"at or below the vehicle's {v_tangential} km/s of tangential speed"
        )

    def cost(trial_tangential: float, sign: float) -> Tuple[float, float]:
        """Impulse magnitude and post-burn radial speed for one trial."""
        angular_momentum = radius * trial_tangential
        energy = 0.5 * (angular_momentum / target_perihelion) ** 2 - (
            _MU_SUN / target_perihelion
        )
        speed_squared = 2.0 * (energy + _MU_SUN / radius)
        radial_squared = speed_squared - trial_tangential * trial_tangential
        # Round-off clamp in the spirit of conic_kernel.speed_components_at_radius,
        # but scaled by the *differenced* terms rather than the result.  A dive to
        # 4 solar radii computes its energy as the difference of two ~4.8e4
        # km^2/s^2 perihelion terms to get ~-294, so the cancellation error is
        # about 1e-11 -- three orders above what scaling by the ~3.6 km^2/s^2
        # result would allow, which would reject the exactly-tangential apsis burn
        # that is the true optimum whenever the burn is taken at aphelion.
        tolerance = 1e-12 * max(
            1.0,
            abs(speed_squared),
            (angular_momentum / target_perihelion) ** 2,
            _MU_SUN / target_perihelion,
        )
        if radial_squared < -tolerance:
            return float("inf"), 0.0
        radial = sign * float(np.sqrt(max(0.0, radial_squared)))
        return (
            float(np.hypot(trial_tangential - v_tangential, radial - v_radial)),
            radial,
        )

    best: Optional[Tuple[float, float, float]] = None
    lower = tangential_floor
    upper = max(lower * (1.0 + 1e-9), v_tangential)
    for sign in (-1.0, 1.0):
        found = minimize_scalar(
            lambda trial, s=sign: cost(trial, s)[0],
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": _INJECTION_SEARCH_TOLERANCE},
        )
        # The endpoints are checked explicitly because the bounded search never
        # evaluates them, and at a burn taken exactly at aphelion the optimum
        # *is* the lower bound: the cheapest dive there is the tangential apsis
        # burn, which leaves no radial speed and so sits at the tangential floor.
        for trial in (float(found.x), lower, upper):
            magnitude, radial = cost(trial, sign)
            if np.isfinite(magnitude) and (best is None or magnitude < best[0]):
                best = (magnitude, trial, radial)
    if best is None:
        raise ValueError("no admissible dive-injection impulse found")

    delta_v, tangential, radial = best
    angular_momentum = radius * tangential
    energy = 0.5 * (angular_momentum / target_perihelion) ** 2 - (
        _MU_SUN / target_perihelion
    )
    semi_major_axis = -_MU_SUN / (2.0 * energy)
    eccentricity = 1.0 - target_perihelion / semi_major_axis
    semi_latus = semi_major_axis * (1.0 - eccentricity * eccentricity)
    principal = true_anomaly_at_radius_rad(semi_latus, eccentricity, radius)
    if principal is None:
        raise ValueError("dive ellipse does not reach the burn radius")
    true_anomaly = principal if radial >= 0.0 else 2.0 * np.pi - principal
    return DiveInjection(
        delta_v=delta_v,
        v_tangential=tangential,
        v_radial=radial,
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        true_anomaly=true_anomaly,
        thrust_axis=float(np.arctan2(radial - v_radial, tangential - v_tangential)),
    )


def returning_beam(
    dive_semi_major_axis: float,
    dive_perihelion: float = _DIVE_PERIAPSIS,
    periapsis_burn: float = _PERIAPSIS_BURN,
) -> ConicState:
    """The boosted climb-out, as a conic the caller can evaluate anywhere.

    The projectile arrives at ``dive_perihelion`` off the dive ellipse, takes a
    tangential ``periapsis_burn`` there, and leaves on a solar-escape hyperbola.
    Its angular momentum is only ``r_p * v_p`` at 4 solar radii, so the climb-out
    is very nearly radial -- which is the whole reason its 1 AU crossing and its
    outer-node crossing sit on one ray.

    Args:
        dive_semi_major_axis: Semi-major axis of the incoming dive ellipse (km).
        dive_perihelion: Perihelion of the dive (km).
        periapsis_burn: Tangential boost taken there (km/s).

    Returns:
        The climb-out's :class:`ConicState`.
    """
    arrival_speed = float(
        np.sqrt(_MU_SUN * (2.0 / dive_perihelion - 1.0 / dive_semi_major_axis))
    )
    return conic_state_at_radius(
        _MU_SUN, dive_perihelion, arrival_speed + periapsis_burn, 0.0
    )


def beam_leg(
    beam: ConicState, radius_from: float, radius_to: float
) -> Tuple[float, float]:
    """Time and swept longitude for the beam to climb between two radii.

    Args:
        beam: The climb-out conic.
        radius_from: Inner radius (km).
        radius_to: Outer radius (km).

    Returns:
        (seconds, swept longitude in degrees).

    Raises:
        ValueError: If either radius is unreachable on the climb-out.
    """
    semi_major_abs = beam.p / (beam.ecc * beam.ecc - 1.0)
    inner = true_anomaly_at_radius_rad(beam.p, beam.ecc, radius_from)
    outer = true_anomaly_at_radius_rad(beam.p, beam.ecc, radius_to)
    if inner is None or outer is None:
        raise ValueError("radius not reachable on the climb-out hyperbola")
    seconds = hyperbolic_tof_seconds(
        _MU_SUN, semi_major_abs, beam.ecc, outer
    ) - hyperbolic_tof_seconds(_MU_SUN, semi_major_abs, beam.ecc, inner)
    return seconds, float(np.degrees(outer - inner))


def split_dive_geometry(
    outbound_perihelion: float,
    outbound_aphelion: float,
    injection_anomaly_deg: float,
    dive_perihelion: float = _DIVE_PERIAPSIS,
    periapsis_burn: float = _PERIAPSIS_BURN,
    params: Optional[_FlybyParams] = None,
) -> Optional[SplitDiveGeometry]:
    """Fly one split-dive loop and report both closure residuals.

    The vehicle departs 1 AU outbound on the ``(outbound_perihelion,
    outbound_aphelion)`` ellipse, coasts to ``injection_anomaly_deg``, takes the
    cheapest impulse onto a ``dive_perihelion`` dive, falls, is boosted, and
    climbs back to 1 AU.  ``outbound_perihelion == dive_perihelion`` with the
    injection at aphelion is the paper's single-impulse dive: the outer burn is
    then zero and the loop reduces to
    :func:`heliocentric_reintercept.single_impulse_resonant_dive`.

    Args:
        outbound_perihelion: Perihelion of the outbound ellipse (km).
        outbound_aphelion: Aphelion of the outbound ellipse (km).
        injection_anomaly_deg: True anomaly of the outer burn (deg); 180 is
            aphelion.
        dive_perihelion: Perihelion the dive must reach (km).
        periapsis_burn: Tangential boost taken there (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitDiveGeometry`, or None if the geometry is inadmissible
        (1 AU off the outbound ellipse, or the burn point behind the departure).
    """
    p = params if params is not None else _powered_flyby_params()
    if outbound_aphelion <= outbound_perihelion:
        return None
    semi_major = 0.5 * (outbound_perihelion + outbound_aphelion)
    eccentricity = (outbound_aphelion - outbound_perihelion) / (
        outbound_aphelion + outbound_perihelion
    )
    semi_latus = semi_major * (1.0 - eccentricity * eccentricity)
    departure_anomaly = true_anomaly_at_radius_rad(
        semi_latus, eccentricity, p.r_earth_orbit
    )
    if departure_anomaly is None:
        return None
    injection_anomaly = float(np.radians(injection_anomaly_deg))
    if injection_anomaly <= departure_anomaly:
        return None

    outbound = ConicState(
        energy=-_MU_SUN / (2.0 * semi_major),
        h=float(np.sqrt(_MU_SUN * semi_latus)),
        p=semi_latus,
        ecc=eccentricity,
    )
    departure_tangential, departure_radial = speed_components_at_radius(
        outbound, _MU_SUN, p.r_earth_orbit
    )
    departure_excess = float(
        np.hypot(departure_tangential - p.v_earth_orbit, departure_radial)
    )
    departure_aim = float(
        np.degrees(np.arctan2(departure_radial, departure_tangential - p.v_earth_orbit))
    )

    coast_seconds = elliptic_time_from_periapsis(
        _MU_SUN, semi_major, eccentricity, injection_anomaly
    ) - elliptic_time_from_periapsis(
        _MU_SUN, semi_major, eccentricity, departure_anomaly
    )
    coast_sweep = float(np.degrees(injection_anomaly - departure_anomaly))
    node_radius = semi_latus / (1.0 + eccentricity * np.cos(injection_anomaly))
    node_tangential, node_radial_magnitude = speed_components_at_radius(
        outbound, _MU_SUN, node_radius
    )
    node_radial = (
        node_radial_magnitude if injection_anomaly < np.pi else -node_radial_magnitude
    )
    injection = dive_injection_impulse(
        node_radius, node_tangential, node_radial, dive_perihelion
    )

    dive_period = 2.0 * np.pi * float(np.sqrt(injection.semi_major_axis**3 / _MU_SUN))
    fall_seconds = dive_period - elliptic_time_from_periapsis(
        _MU_SUN,
        injection.semi_major_axis,
        injection.eccentricity,
        injection.true_anomaly,
    )
    fall_sweep = 360.0 - float(np.degrees(injection.true_anomaly))

    beam = returning_beam(injection.semi_major_axis, dive_perihelion, periapsis_burn)
    return_excess = float(np.sqrt(max(0.0, 2.0 * beam.energy)))
    climb_seconds, climb_sweep = beam_leg(beam, dive_perihelion, p.r_earth_orbit)
    stream_tangential, stream_radial = speed_components_at_radius(
        beam, _MU_SUN, p.r_earth_orbit
    )
    stream_excess = float(np.hypot(stream_radial, stream_tangential - p.v_earth_orbit))
    stream_axis = float(
        np.degrees(np.arctan2(stream_radial, stream_tangential - p.v_earth_orbit))
    )
    beam_seconds, beam_sweep = beam_leg(beam, p.r_earth_orbit, node_radius)

    coast_years = coast_seconds / _SECONDS_PER_YEAR
    fall_years = fall_seconds / _SECONDS_PER_YEAR
    climb_years = climb_seconds / _SECONDS_PER_YEAR
    beam_years = beam_seconds / _SECONDS_PER_YEAR
    cycle_years = coast_years + fall_years + climb_years
    swept = coast_sweep + fall_sweep + climb_sweep
    return SplitDiveGeometry(
        outbound_perihelion=outbound_perihelion,
        outbound_aphelion=outbound_aphelion,
        injection_anomaly=injection_anomaly,
        dive_perihelion=dive_perihelion,
        periapsis_burn=periapsis_burn,
        departure_excess=departure_excess,
        departure_aim=departure_aim,
        coast_years=coast_years,
        fall_years=fall_years,
        climb_years=climb_years,
        coast_sweep=coast_sweep,
        fall_sweep=fall_sweep,
        climb_sweep=climb_sweep,
        injection=injection,
        node_radius=node_radius,
        return_excess=return_excess,
        stream_excess=stream_excess,
        stream_axis=stream_axis,
        beam_transit_years=beam_years,
        beam_sweep=beam_sweep,
        cycle_years=cycle_years,
        reintercept_residual=_wrap_degrees(360.0 * cycle_years - swept),
        colocation_residual=_wrap_degrees(
            360.0 * (coast_years - beam_years) - (coast_sweep + beam_sweep)
        ),
        reuse_fraction=(coast_years - beam_years) / cycle_years,
    )


def _reintercept_residual_raw(
    outbound_perihelion: float,
    outbound_aphelion: float,
    injection_anomaly_deg: float,
    dive_perihelion: float = _DIVE_PERIAPSIS,
    periapsis_burn: float = _PERIAPSIS_BURN,
    params: Optional[_FlybyParams] = None,
) -> float:
    """Unwrapped Earth-advance-minus-swept residual, for root-finding on revolutions."""
    geometry = split_dive_geometry(
        outbound_perihelion,
        outbound_aphelion,
        injection_anomaly_deg,
        dive_perihelion,
        periapsis_burn,
        params,
    )
    if geometry is None:
        return 1.0e9
    swept = geometry.coast_sweep + geometry.fall_sweep + geometry.climb_sweep
    return 360.0 * geometry.cycle_years - swept


def reintercept_closing_aphelion(
    outbound_perihelion: float,
    revolutions: int = 0,
    injection_anomaly_deg: float = 180.0,
    dive_perihelion: float = _DIVE_PERIAPSIS,
    periapsis_burn: float = _PERIAPSIS_BURN,
    params: Optional[_FlybyParams] = None,
) -> float:
    """Aphelion whose split dive re-crosses 1 AU where Earth is.

    The **Earth re-intercept** closure alone, with the injection held at
    aphelion.  At ``outbound_perihelion`` equal to the dive perihelion this
    reproduces :func:`heliocentric_reintercept.single_impulse_resonant_dive`'s
    1.9259 AU at ``revolutions`` = 0.  Raising the perihelion barely moves the
    closing aphelion (1.887 to 1.973 AU across the whole family) and steadily
    lengthens the clock, because the coast to aphelion grows from a tenth of the
    loop to more than half of it.

    Args:
        outbound_perihelion: Perihelion of the outbound ellipse (km).
        revolutions: Extra whole turns Earth makes during the loop; 0 is the
            shortest resonance, 1 the next.
        injection_anomaly_deg: True anomaly of the outer burn (deg).
        dive_perihelion: Perihelion the dive must reach (km).
        periapsis_burn: Tangential boost taken there (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The closing aphelion (km).

    Raises:
        ValueError: If the residual does not change sign over
            CLOSING_APHELION_BRACKET_AU.
    """

    def residual(aphelion_au: float) -> float:
        return (
            _reintercept_residual_raw(
                outbound_perihelion,
                aphelion_au * _AU,
                injection_anomaly_deg,
                dive_perihelion,
                periapsis_burn,
                params,
            )
            - 360.0 * revolutions
        )

    lower = max(
        CLOSING_APHELION_BRACKET_AU[0], outbound_perihelion / _AU * 1.001 + 0.02
    )
    upper = CLOSING_APHELION_BRACKET_AU[1]
    if residual(lower) * residual(upper) > 0.0:
        raise ValueError(
            f"no re-intercept root at revolutions={revolutions} over "
            f"({lower}, {upper}) AU"
        )
    return float(brentq(residual, lower, upper)) * _AU


def split_dive_ledger(
    geometry: SplitDiveGeometry,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    periapsis_survival: float = DEFAULT_PERIAPSIS_SURVIVAL,
    params: Optional[_FlybyParams] = None,
) -> SplitDiveLedger:
    """Score a split-dive loop on returning mass per impactor kilogram.

    The Earth node is :func:`jovian_solar_dive_cycle.departure_nozzle_ledger`
    unchanged.  The outer node is the same device in heliocentric flight: the
    thrust axis is the injection impulse's own direction, the stream arrives
    nearly radially, and the burn is integrated because the impact angle walks
    through 90 degrees as the vehicle's tangential speed is cancelled.

    Args:
        geometry: The loop to score.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        periapsis_survival: Mass fraction surviving the solar-periapsis pass.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitDiveLedger`.

    Raises:
        ValueError: If either burn delivers nothing, so no growth exists.
    """
    p = params if params is not None else _powered_flyby_params()
    departure = departure_nozzle_ledger(
        geometry.departure_excess,
        geometry.departure_aim,
        geometry.stream_excess,
        geometry.stream_axis,
        slug_ratio,
        jet_energy_efficiency,
        p,
    )
    beam = returning_beam(
        geometry.injection.semi_major_axis,
        geometry.dive_perihelion,
        geometry.periapsis_burn,
    )
    node_tangential, node_radial = speed_components_at_radius(
        beam, _MU_SUN, geometry.node_radius
    )
    beam_speed = float(np.hypot(node_tangential, node_radial))
    beam_axis = float(np.degrees(np.arctan2(node_radial, node_tangential)))
    thrust_axis = float(np.degrees(geometry.injection.thrust_axis))

    # The vehicle's speed along its own thrust axis, before and after the burn.
    axis_unit = np.array(
        [np.cos(geometry.injection.thrust_axis), np.sin(geometry.injection.thrust_axis)]
    )
    incoming = np.array(
        [
            geometry.injection.v_tangential
            - (geometry.injection.delta_v * axis_unit[0]),
            geometry.injection.v_radial - (geometry.injection.delta_v * axis_unit[1]),
        ]
    )
    speed_start = float(np.dot(incoming, axis_unit))
    speed_end = speed_start + geometry.injection.delta_v

    steps = np.linspace(speed_start, speed_end, _BURN_INTEGRATION_STEPS + 1)
    log_fraction = 0.0
    for lower, upper in zip(steps[:-1], steps[1:]):
        angle, closing = _vehicle_frame_impact(
            beam_axis, thrust_axis, beam_speed, 0.5 * (float(lower) + float(upper))
        )
        beta = impulse_per_impactor_kg(angle * u.rad, slug_ratio, jet_energy_efficiency)
        if beta <= 0.0:
            raise ValueError("the outer-node burn delivers no usable fraction")
        log_fraction -= (float(upper) - float(lower)) / (beta * closing / slug_ratio)
    outer_delivered = float(np.exp(log_fraction))
    start_angle, start_closing = _vehicle_frame_impact(
        beam_axis, thrust_axis, beam_speed, speed_start
    )
    end_angle, _ = _vehicle_frame_impact(beam_axis, thrust_axis, beam_speed, speed_end)

    delivered = departure.delivered_fraction * outer_delivered
    if delivered <= 0.0 or delivered >= 1.0:
        raise ValueError("the split departure delivers no usable fraction")
    slug = (1.0 - departure.delivered_fraction) + departure.delivered_fraction * (
        1.0 - outer_delivered
    )
    growth = periapsis_survival * delivered * slug_ratio / slug
    return SplitDiveLedger(
        geometry=geometry,
        slug_ratio=slug_ratio,
        jet_energy_efficiency=jet_energy_efficiency,
        periapsis_survival=periapsis_survival,
        departure=departure,
        outer_impact_angle_start=float(np.degrees(start_angle)),
        outer_impact_angle_end=float(np.degrees(end_angle)),
        outer_closing_speed=start_closing,
        outer_delivered_fraction=outer_delivered,
        delivered_fraction=delivered,
        slug_per_delivered_kg=slug / delivered,
        impactor_per_delivered_kg=slug / (delivered * slug_ratio),
        earth_caught_fraction=(1.0 - departure.delivered_fraction) / slug,
        return_per_impactor_kg=growth,
        growth_rate=float(np.log(growth) / geometry.cycle_years),
        doubling_years=float(geometry.cycle_years * np.log(2.0) / np.log(growth)),
    )


def partial_split_optimum(
    revolutions: int = 0,
    dive_perihelion: float = _DIVE_PERIAPSIS,
    periapsis_burn: float = _PERIAPSIS_BURN,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    periapsis_survival: float = DEFAULT_PERIAPSIS_SURVIVAL,
    params: Optional[_FlybyParams] = None,
) -> SplitDiveLedger:
    """Fastest-growing split along the Earth re-intercept closure curve.

    The **Earth re-intercept** closure pins the aphelion for each outbound
    perihelion, leaving one free knob; this maximises the growth *rate* along it
    (CONTEXT.md's **growth rate**, not doubling time, because doubling has a pole
    where growth approaches one).  On the shortest resonance the optimum is
    interior at ~0.49 AU and beats the paper's own dive on both axes at once --
    a shorter doubling time *and* a third less launched slug.

    This point does **not** satisfy the outer-node co-location, so its outer node
    needs a dedicated delivery rather than the beam's leftovers; the phased
    cycles are :func:`two_node_closure`.

    Args:
        revolutions: Re-intercept resonance index.
        dive_perihelion: Perihelion the dive must reach (km).
        periapsis_burn: Tangential boost taken there (km/s).
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        periapsis_survival: Mass fraction surviving the solar-periapsis pass.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitDiveLedger` at the rate optimum.
    """

    def negative_rate(perihelion_au: float) -> float:
        perihelion = perihelion_au * _AU
        aphelion = reintercept_closing_aphelion(perihelion, revolutions, params=params)
        geometry = split_dive_geometry(perihelion, aphelion, 180.0, params=params)
        if geometry is None:
            return 1.0e9
        return -split_dive_ledger(
            geometry, slug_ratio, jet_energy_efficiency, periapsis_survival, params
        ).growth_rate

    found = minimize_scalar(
        negative_rate,
        bounds=(
            OUTBOUND_PERIHELION_BRACKET_AU[0] + 1.0e-4,
            OUTBOUND_PERIHELION_BRACKET_AU[1],
        ),
        method="bounded",
        options={"xatol": 1.0e-4},
    )
    perihelion = float(found.x) * _AU
    aphelion = reintercept_closing_aphelion(
        perihelion, revolutions, 180.0, dive_perihelion, periapsis_burn, params
    )
    geometry = split_dive_geometry(
        perihelion, aphelion, 180.0, dive_perihelion, periapsis_burn, params
    )
    assert geometry is not None
    return split_dive_ledger(
        geometry, slug_ratio, jet_energy_efficiency, periapsis_survival, params
    )


def two_node_closure(
    reuse_numerator: int,
    interleaved_chains: int,
    revolutions: int = 1,
    dive_perihelion: float = _DIVE_PERIAPSIS,
    periapsis_burn: float = _PERIAPSIS_BURN,
    guesses: Sequence[Tuple[float, float, float]] = CLOSURE_GUESSES,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    periapsis_survival: float = DEFAULT_PERIAPSIS_SURVIVAL,
    params: Optional[_FlybyParams] = None,
) -> Optional[TwoNodeClosure]:
    """Solve all three conditions at once, for a named beam-reuse fraction.

    The three conditions are **Earth re-intercept** (the return crosses 1 AU
    where Earth is), **outer-node co-location** (the outer burn happens on a ray
    a beam is flying down), and rational beam reuse (the offset between a beam's
    Earth crossing and the outer burn it feeds is ``reuse_numerator /
    interleaved_chains`` of a cycle, so the chains close into a repeating
    pattern).  Three conditions, three knobs -- outbound perihelion, aphelion and
    injection anomaly -- so the solution set is **discrete**, exactly as the
    Jovian cycle's **synodic closure** is.

    Args:
        reuse_numerator: Node cadences between a beam's Earth crossing and the
            outer burn it feeds.
        interleaved_chains: Number of payload chains in flight at once.
        revolutions: Re-intercept resonance index; the admissible family lives
            at 1.
        dive_perihelion: Perihelion the dive must reach (km).
        periapsis_burn: Tangential boost taken there (km/s).
        guesses: Initial (perihelion AU, aphelion AU, injection anomaly deg)
            triples, tried in order until one lands on an admissible root.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        periapsis_survival: Mass fraction surviving the solar-periapsis pass.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`TwoNodeClosure`, or None if the solve does not land on an
        admissible root -- one inside the recorded brackets whose coast to the
        outer node has not collapsed back onto the Earth node.
    """
    target = reuse_numerator / interleaved_chains

    def residuals(knobs: Sequence[float]) -> List[float]:
        perihelion, aphelion, anomaly = (
            float(knobs[0]) * _AU,
            float(knobs[1]) * _AU,
            float(knobs[2]),
        )
        geometry = split_dive_geometry(
            perihelion, aphelion, anomaly, dive_perihelion, periapsis_burn, params
        )
        if geometry is None:
            return [1.0e3, 1.0e3, 1.0e3]
        swept = geometry.coast_sweep + geometry.fall_sweep + geometry.climb_sweep
        return [
            360.0 * geometry.cycle_years - swept - 360.0 * revolutions,
            geometry.colocation_residual,
            geometry.reuse_fraction - target,
        ]

    geometry: Optional[SplitDiveGeometry] = None
    for guess in guesses:
        with warnings.catch_warnings():
            # `fsolve` reports slow progress on a root whose residuals have
            # already reached their floating-point floor; the residual check
            # below is what actually accepts or rejects the root.
            warnings.simplefilter("ignore", RuntimeWarning)
            solution = fsolve(residuals, list(guess), xtol=1.0e-12)
        perihelion_au, aphelion_au, anomaly_deg = (float(x) for x in solution)
        if not (
            OUTBOUND_PERIHELION_BRACKET_AU[0]
            < perihelion_au
            < OUTBOUND_PERIHELION_BRACKET_AU[1]
        ):
            continue
        if not (
            INJECTION_ANOMALY_BRACKET_DEG[0]
            < anomaly_deg
            < INJECTION_ANOMALY_BRACKET_DEG[1]
        ):
            continue
        candidate = split_dive_geometry(
            perihelion_au * _AU,
            aphelion_au * _AU,
            anomaly_deg,
            dive_perihelion,
            periapsis_burn,
            params,
        )
        if candidate is None or candidate.coast_years < _DEGENERATE_COAST_YEARS:
            continue
        if (
            abs(candidate.reintercept_residual) > CLOSURE_RESIDUAL_TOLERANCE
            or abs(candidate.colocation_residual) > CLOSURE_RESIDUAL_TOLERANCE
            or abs(candidate.reuse_fraction - target) > CLOSURE_RESIDUAL_TOLERANCE
        ):
            continue
        geometry = candidate
        break
    if geometry is None:
        return None
    ledger = split_dive_ledger(
        geometry, slug_ratio, jet_energy_efficiency, periapsis_survival, params
    )
    return TwoNodeClosure(
        reuse_numerator=reuse_numerator,
        interleaved_chains=interleaved_chains,
        revolutions=revolutions,
        geometry=geometry,
        ledger=ledger,
        node_cadence_years=geometry.cycle_years / interleaved_chains,
    )


@dataclass(frozen=True)
class DepthConduction:
    """Whether each node's plume still conducts, at one dive depth.

    The **overtaking-leg conduction floor** of ADR 0022 is a statement about the
    device, not about which leg drives it: below a threshold closing speed the
    merged blob does not expand enough to stay conducting, and the nozzle has
    nothing to grip.  This carries that floor onto the two *departure* burns.

    Attributes:
        dive_solar_radii: Perihelion distance in solar radii.
        expansion_margin: Headroom demanded above the **expansion floor**.
        threshold: Minimum closing speed the plume needs (km/s).
        stream_excess: Earth-relative excess the return arrives with (km/s).
        direct_delta_v: Burn the paper's single-impulse resonant dive needs from
            the parking orbit at this depth (km/s).
        direct_closing_start: Closing speed as that burn lights (km/s).
        direct_closing_end: Closing speed as it finishes (km/s) -- its *coldest*
            instant, because the departure runs with the stream.
        direct_clears: Whether the direct departure stays conducting throughout.
        split_delta_v: Burn the split's Earth node needs (km/s).
        split_closing_end: Closing speed as that burn finishes (km/s).
        split_clears: Whether the split's Earth node stays conducting.
        outer_closing_cold: Coldest closing speed at the split's far node (km/s).
        outer_clears: Whether the far node stays conducting.
    """

    dive_solar_radii: float
    expansion_margin: float
    threshold: float
    stream_excess: float
    direct_delta_v: float
    direct_closing_start: float
    direct_closing_end: float
    direct_clears: bool
    split_delta_v: float
    split_closing_end: float
    split_clears: bool
    outer_closing_cold: float
    outer_clears: bool


def resonant_dive_at_depth(
    dive_solar_radii: float,
    periapsis_burn: float = _PERIAPSIS_BURN,
) -> Tuple[float, float, float, float]:
    """The paper's single-impulse resonant dive, re-solved at another depth.

    Both the boost and the return weaken as the dive is backed out, which is why
    the direct architecture does not simply fail on impulse: at 16 solar radii it
    needs 32.66 km/s rather than 37.53, against a stream that has fallen from
    157 km/s to 114.  What does not scale with it is the **conduction floor**.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        periapsis_burn: Tangential boost taken at perihelion (km/s).

    The burn is now passed through to the closure rather than dropped: it enters
    via the climb-out, and moves the conduction crossing by at most 0.06 solar
    radii over 20-45 km/s, so no published figure changes at quoted precision.

    Returns:
        (closing aphelion km, Earth boost km/s, boost aim deg, cycle years).
    """
    perihelion = dive_solar_radii * _SOLAR_RADIUS
    dive = single_impulse_resonant_dive(
        periapsis_radius=perihelion * u.km,
        periapsis_burn=periapsis_burn * u.km / u.s,
    )
    retrograde = float(dive.retrograde_component.to_value(u.km / u.s))
    radial = float(dive.radial_component.to_value(u.km / u.s))
    return (
        float(dive.closing_aphelion.to_value(u.AU)) * _AU,
        float(dive.earth_boost.to_value(u.km / u.s)),
        float(np.degrees(np.arctan2(radial, -retrograde))),
        float(dive.reintercept_time.to_value(u.year)),
    )


def depth_conduction(
    dive_solar_radii: float,
    split_departure_excess: float = _SPLIT_DEPARTURE_EXCESS,
    split_aphelion: float = _SPLIT_APHELION,
    periapsis_burn: float = _PERIAPSIS_BURN,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> Optional[DepthConduction]:
    """Score both architectures' departures against the conduction floor at one depth.

    The asymmetry this exists to expose: **the Earth departure cools itself and
    the far node cannot.**  At Earth the vehicle accelerates roughly *along* the
    arriving stream (the paper's dive departs 31 degrees off it), so the closing
    speed falls through the burn -- 148 to 125 km/s at 4 solar radii, and the
    end of the burn is the coldest instant.  At the far node the thrust is
    within a few degrees of *perpendicular* to the stream, so the closing speed
    is essentially constant (153.50 to 153.17): a burn cannot cool a stream it
    is not running away from.  The 86-90 degree cant that costs the far node
    about a fifth of its ``beta`` is the same fact that makes it immune here.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        split_departure_excess: Earth-relative excess the split's small Earth
            burn must reach (km/s).
        split_aphelion: Aphelion the split coasts to (km).
        periapsis_burn: Tangential boost taken at the dive perihelion (km/s).
        expansion_margin: Headroom demanded above the **expansion floor**.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`DepthConduction`, or None if no closing-speed threshold
        exists at this reading of the **conduction reserve**.
    """
    p = params if params is not None else _powered_flyby_params()
    threshold = conduction_threshold_closing_speed(
        jet_energy_efficiency, expansion_margin
    )
    if threshold is None:
        return None
    perihelion = dive_solar_radii * _SOLAR_RADIUS

    aphelion, boost, aim, _ = resonant_dive_at_depth(dive_solar_radii, periapsis_burn)
    direct_beam = returning_beam(
        0.5 * (perihelion + aphelion), perihelion, periapsis_burn
    )
    tangential, radial = speed_components_at_radius(
        direct_beam, _MU_SUN, p.r_earth_orbit
    )
    direct_excess = float(np.hypot(radial, tangential - p.v_earth_orbit))
    direct_axis = float(np.degrees(np.arctan2(radial, tangential - p.v_earth_orbit)))
    direct = departure_nozzle_ledger(
        boost, aim, direct_excess, direct_axis, slug_ratio, jet_energy_efficiency, p
    )

    split_beam = returning_beam(
        0.5 * (perihelion + split_aphelion), perihelion, periapsis_burn
    )
    tangential, radial = speed_components_at_radius(
        split_beam, _MU_SUN, p.r_earth_orbit
    )
    split_excess = float(np.hypot(radial, tangential - p.v_earth_orbit))
    split_axis = float(np.degrees(np.arctan2(radial, tangential - p.v_earth_orbit)))
    departure = departure_nozzle_ledger(
        split_departure_excess,
        0.0,
        split_excess,
        split_axis,
        slug_ratio,
        jet_energy_efficiency,
        p,
    )

    # Far node: tangential speed is cancelled from the coast ellipse's value to
    # the dive's, with the beam arriving within a degree of radial.
    node_tangential, node_radial = speed_components_at_radius(
        split_beam, _MU_SUN, split_aphelion
    )
    beam_speed = float(np.hypot(node_tangential, node_radial))
    beam_axis = float(np.degrees(np.arctan2(node_radial, node_tangential)))
    arriving = float(
        np.sqrt(
            _MU_SUN * (2.0 / split_aphelion - 2.0 / (p.r_earth_orbit + split_aphelion))
        )
    )
    leaving = float(
        np.sqrt(_MU_SUN * (2.0 / split_aphelion - 2.0 / (perihelion + split_aphelion)))
    )
    outer_cold = min(
        _vehicle_frame_impact(beam_axis, 180.0, beam_speed, -speed)[1]
        for speed in (arriving, leaving)
    )

    return DepthConduction(
        dive_solar_radii=dive_solar_radii,
        expansion_margin=expansion_margin,
        threshold=threshold,
        stream_excess=direct_excess,
        direct_delta_v=direct.delta_v,
        direct_closing_start=direct.closing_speed_start,
        direct_closing_end=direct.closing_speed_end,
        direct_clears=direct.closing_speed_end >= threshold,
        split_delta_v=departure.delta_v,
        split_closing_end=departure.closing_speed_end,
        split_clears=departure.closing_speed_end >= threshold,
        outer_closing_cold=outer_cold,
        outer_clears=outer_cold >= threshold,
    )


def direct_departure_conduction_depth(
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    periapsis_burn: float = _PERIAPSIS_BURN,
    bracket: Tuple[float, float] = DEPTH_CONDUCTION_BRACKET,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> Optional[float]:
    """Depth at which the paper's direct departure stops conducting.

    Bisected on the constraint rather than sampled near it -- ADR 0022's lesson.
    The split's Earth node has no such crossing anywhere in the bracket, because
    its burn is a twentieth of the direct one's and so barely cools the stream.

    Args:
        expansion_margin: Headroom demanded above the **expansion floor**.
        periapsis_burn: Tangential boost taken at the dive perihelion (km/s).
        bracket: Depths to bisect between, in solar radii.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The crossing depth in solar radii, or None when the bracket does not
        straddle one.
    """

    def residual(depth: float) -> float:
        row = depth_conduction(
            float(depth),
            periapsis_burn=periapsis_burn,
            expansion_margin=expansion_margin,
            slug_ratio=slug_ratio,
            jet_energy_efficiency=jet_energy_efficiency,
            params=params,
        )
        if row is None:
            return 0.0
        return row.direct_closing_end - row.threshold

    low, high = bracket
    if residual(low) * residual(high) >= 0.0:
        return None
    return float(brentq(residual, low, high, xtol=1e-4))


@dataclass(frozen=True)
class OpposingStreamPlacement:
    """Placing the *other* half of the dive node, direct against split.

    The payload is only one of the two things that has to arrive at the solar
    perihelion.  The **opposing stream** meets it there head-on, and that leg
    behaves the opposite way under the depth dial: a shallower dive keeps more of
    the orbit's angular momentum, so *reversing* its sign costs more, not less.
    The payload's injection gets cheaper as the dive is backed out and the
    opposing stream's gets dearer, and past about 8 solar radii the opposing
    stream is the dominant Earth-side cost.

    A near-radial plunge is the third option: it costs Earth's own orbital speed
    flat, independent of depth, but it arrives across the payload's path rather
    than into it, so the node's closing speed falls by roughly a factor of root
    two and the **derived periapsis survival** falls with it.  That trade is
    priced here only in placement terms.

    Attributes:
        dive_solar_radii: Perihelion distance in solar radii.
        aphelion: Aphelion the split routes through (km).
        direct_excess: Earth-relative excess for a direct retrograde placement.
        direct_delta_v: That burn from the parking orbit (km/s).
        direct_closing_start: Closing speed as it lights (km/s).
        direct_closing_end: Closing speed as it finishes (km/s).
        direct_clears: Whether it stays conducting.  It does, at every depth --
            it aims retrograde against a radially outward stream, so unlike the
            payload's departure it never runs *along* what feeds it.
        radial_excess: Earth-relative excess for a near-radial plunge (km/s).
        split_raise_excess: Excess for the split's small prograde raise (km/s).
        split_flip_delta_v: The tangential reversal taken at the far node (km/s).
        split_total: Raise plus flip, heliocentric (km/s).
        split_flip_closing_start: Closing speed as the flip lights (km/s).
        split_flip_closing_end: Closing speed as it finishes (km/s).
        split_clears: Whether the flip stays conducting.
        saving: Direct heliocentric placement divided by the split's.
    """

    dive_solar_radii: float
    aphelion: float
    direct_excess: float
    direct_delta_v: float
    direct_closing_start: float
    direct_closing_end: float
    direct_clears: bool
    radial_excess: float
    split_raise_excess: float
    split_flip_delta_v: float
    split_total: float
    split_flip_closing_start: float
    split_flip_closing_end: float
    split_clears: bool
    saving: float


def opposing_stream_placement_trade(
    dive_solar_radii: float,
    aphelion: float = _SPLIT_APHELION,
    periapsis_burn: float = _PERIAPSIS_BURN,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> Optional[OpposingStreamPlacement]:
    """Price the opposing stream's placement, direct against bi-elliptic.

    The bi-elliptic saving is largest exactly where the direct cost is worst,
    because reversing tangential motion at a slow aphelion is cheap however much
    of it there is: the raise costs the same either way and only the flip term
    differs.  So the saving *grows* as the dive is backed out (1.71x at 4 solar
    radii, 1.86x at 32) while the payload leg's saving shrinks -- which is why
    the split's case at a shallow dive rests more on this leg than on the one
    ADR 0023's addendum first scored.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        aphelion: Aphelion the split routes through (km).
        periapsis_burn: Tangential boost taken at the dive perihelion (km/s).
        expansion_margin: Headroom demanded above the **expansion floor**.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`OpposingStreamPlacement`, or None if no closing-speed
        threshold exists at this reading of the **conduction reserve**.
    """
    p = params if params is not None else _powered_flyby_params()
    threshold = conduction_threshold_closing_speed(
        jet_energy_efficiency, expansion_margin
    )
    if threshold is None:
        return None
    perihelion = dive_solar_radii * _SOLAR_RADIUS

    def apsis_speed(inner: float, outer: float, radius: float) -> float:
        return float(np.sqrt(_MU_SUN * (2.0 / radius - 2.0 / (inner + outer))))

    direct_excess = direct_retrograde_placement_excess(dive_solar_radii, p)
    beam = returning_beam(0.5 * (perihelion + aphelion), perihelion, periapsis_burn)
    tangential, radial = speed_components_at_radius(beam, _MU_SUN, p.r_earth_orbit)
    stream_excess = float(np.hypot(radial, tangential - p.v_earth_orbit))
    stream_axis = float(np.degrees(np.arctan2(radial, tangential - p.v_earth_orbit)))
    direct = departure_nozzle_ledger(
        direct_excess,
        180.0,
        stream_excess,
        stream_axis,
        slug_ratio,
        jet_energy_efficiency,
        p,
    )

    raise_excess = (
        apsis_speed(p.r_earth_orbit, aphelion, p.r_earth_orbit) - p.v_earth_orbit
    )
    arriving = apsis_speed(p.r_earth_orbit, aphelion, aphelion)
    leaving = apsis_speed(perihelion, aphelion, aphelion)
    flip = arriving + leaving  # prograde to retrograde, so the two terms add

    node_tangential, node_radial = speed_components_at_radius(beam, _MU_SUN, aphelion)
    beam_speed = float(np.hypot(node_tangential, node_radial))
    beam_axis = float(np.degrees(np.arctan2(node_radial, node_tangential)))
    flip_speeds = [
        _vehicle_frame_impact(beam_axis, 180.0, beam_speed, -speed)[1]
        for speed in (arriving, -leaving)
    ]

    return OpposingStreamPlacement(
        dive_solar_radii=dive_solar_radii,
        aphelion=aphelion,
        direct_excess=direct_excess,
        direct_delta_v=direct.delta_v,
        direct_closing_start=direct.closing_speed_start,
        direct_closing_end=direct.closing_speed_end,
        direct_clears=min(direct.closing_speed_start, direct.closing_speed_end)
        >= threshold,
        radial_excess=p.v_earth_orbit,
        split_raise_excess=raise_excess,
        split_flip_delta_v=flip,
        split_total=raise_excess + flip,
        split_flip_closing_start=flip_speeds[0],
        split_flip_closing_end=flip_speeds[1],
        split_clears=min(flip_speeds) >= threshold,
        saving=(
            p.v_earth_orbit + apsis_speed(perihelion, p.r_earth_orbit, p.r_earth_orbit)
        )
        / (raise_excess + flip),
    )


@dataclass(frozen=True)
class NodeGeometryTrade:
    """Head-on opposing stream against a straight-down plunger, at one depth.

    ``dive_node()`` hardcodes 180 degrees, so the entire depth trade of ADR
    0020/0021/0022 is built on a head-on node -- which is exactly the arrival
    that needs the **opposing stream**'s retrograde placement, the leg that gets
    *dearer* as the dive is backed out.  This prices the alternative.

    The two effects run against each other.  A plunger arrives across the
    payload's path rather than into it, so the closing speed drops by 1/root 2 --
    but the **impact-angle impulse law**'s ``cos theta`` debit drops with it, from
    a full ``-1`` to ``-0.707``, and that term is the impactor's own momentum
    arriving backwards.  ``beta`` therefore goes *up* while ``w`` goes down, and
    which wins depends entirely on the **slug ratio**: the debit is
    ``k``-independent while the useful term grows as ``sqrt(k)``, so the relief
    is large where the node is slug-poor and negligible where it is slug-rich.

    The arrival is 135 degrees, not 90: payload and plunger reach perihelion at
    nearly the same speed (306.0 against 306.0 km/s at 4 solar radii), so their
    relative velocity bisects the two axes.

    Attributes:
        dive_solar_radii: Perihelion distance in solar radii.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        arrival_speed: Payload's tangential speed at perihelion (km/s).
        plunger_speed: Radial speed of a zero-angular-momentum drop (km/s).
        head_on_closing: Closing speed of the head-on arrival (km/s).
        plunger_closing: Closing speed of the plunger (km/s).
        plunger_angle: Impact angle of the plunger, from the thrust axis (deg).
        head_on_beta: Impulse per impactor kilogram, head-on.
        plunger_beta: The same for the plunger.
        head_on_exhaust: Effective exhaust speed head-on (km/s).
        plunger_exhaust: The same for the plunger (km/s).
        exhaust_ratio: Plunger exhaust speed over head-on -- what the geometry
            costs in Isp.
        head_on_survival: Mass fraction surviving the node, head-on.
        plunger_survival: The same for the plunger.
        head_on_thermalised: Collision energy the merge thermalises (MJ/kg).
        plunger_thermalised: The same for the plunger (MJ/kg).
        thermal_ratio: Plunger thermalised energy over head-on -- one half, since
            it goes as ``w**2``.
    """

    dive_solar_radii: float
    slug_ratio: float
    arrival_speed: float
    plunger_speed: float
    head_on_closing: float
    plunger_closing: float
    plunger_angle: float
    head_on_beta: float
    plunger_beta: float
    head_on_exhaust: float
    plunger_exhaust: float
    exhaust_ratio: float
    head_on_survival: float
    plunger_survival: float
    head_on_thermalised: float
    plunger_thermalised: float
    thermal_ratio: float


def node_geometry_trade(
    dive_solar_radii: float,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    boost: float = _PERIAPSIS_BURN,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    start_radius: float = _AU,
) -> NodeGeometryTrade:
    """Price the dive node's collision geometry, head-on against a plunger.

    Both arrivals fall from ``start_radius``: the payload onto an ellipse with
    perihelion at the dive, the plunger on a zero-angular-momentum drop.  Their
    speeds at perihelion are within a percent of each other at every depth, so
    the relative velocity sits at 135 degrees and the closing speed is the
    head-on one divided by root two.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        boost: PuffSat boost taken at perihelion (km/s), for the survival term.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        start_radius: Radius both arrivals fall from (km).

    Returns:
        The :class:`NodeGeometryTrade`.
    """
    perihelion = dive_solar_radii * _SOLAR_RADIUS
    arrival = float(
        np.sqrt(_MU_SUN * (2.0 / perihelion - 2.0 / (perihelion + start_radius)))
    )
    plunger = float(
        np.sqrt(max(0.0, 2.0 * _MU_SUN / perihelion - 2.0 * _MU_SUN / start_radius))
    )
    head_on_closing = 2.0 * arrival
    plunger_closing = float(np.hypot(arrival, plunger))
    angle = float(np.degrees(np.arccos(-arrival / plunger_closing)))

    head_on_beta = impulse_per_impactor_kg(
        180.0 * u.deg, slug_ratio, jet_energy_efficiency
    )
    plunger_beta = impulse_per_impactor_kg(
        angle * u.deg, slug_ratio, jet_energy_efficiency
    )
    head_on_exhaust = head_on_beta * head_on_closing / slug_ratio
    plunger_exhaust = plunger_beta * plunger_closing / slug_ratio

    def thermalised(closing: float) -> float:
        """Dissipated centre-of-mass energy per kilogram of blob (MJ/kg)."""
        return (
            closing
            * closing
            * slug_ratio
            / (2.0 * (1.0 + slug_ratio) * (1.0 + slug_ratio))
        )

    return NodeGeometryTrade(
        dive_solar_radii=dive_solar_radii,
        slug_ratio=slug_ratio,
        arrival_speed=arrival,
        plunger_speed=plunger,
        head_on_closing=head_on_closing,
        plunger_closing=plunger_closing,
        plunger_angle=angle,
        head_on_beta=head_on_beta,
        plunger_beta=plunger_beta,
        head_on_exhaust=head_on_exhaust,
        plunger_exhaust=plunger_exhaust,
        exhaust_ratio=plunger_exhaust / head_on_exhaust,
        head_on_survival=float(np.exp(-boost / head_on_exhaust)),
        plunger_survival=float(np.exp(-boost / plunger_exhaust)),
        head_on_thermalised=thermalised(head_on_closing),
        plunger_thermalised=thermalised(plunger_closing),
        thermal_ratio=thermalised(plunger_closing) / thermalised(head_on_closing),
    )


def plunger_equivalent_depth(
    dive_solar_radii: float,
    start_radius: float = _AU,
    bracket: Tuple[float, float] = PLUNGER_EQUIVALENT_DEPTH_BRACKET,
) -> Optional[float]:
    """Depth whose *head-on* collision is as gentle as this depth's plunger.

    The plunger halves the thermalised energy at the same perihelion, and the
    thermalised energy goes as the square of the closing speed while the closing
    speed goes roughly as the inverse square root of the radius -- so halving it
    is worth a factor of about two in depth.  Read the other way: the plunger
    lets a dive stay **twice as deep** for the same collision heat, which is the
    axis ADR 0020 shows is expensive (2.51x doubling time from 4 to 32 solar
    radii).

    Only the *collision* term moves.  The solar flux at the node still goes as
    ``1/r**2`` and is untouched by the arrival geometry.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        start_radius: Radius both arrivals fall from (km).
        bracket: Depths to bisect between, in solar radii.

    Returns:
        The equivalent head-on depth in solar radii, or None when the bracket
        does not straddle it.
    """
    target = node_geometry_trade(
        dive_solar_radii, start_radius=start_radius
    ).plunger_closing

    def residual(depth: float) -> float:
        return (
            node_geometry_trade(float(depth), start_radius=start_radius).head_on_closing
            - target
        )

    low, high = bracket
    if residual(low) * residual(high) >= 0.0:
        return None
    return float(brentq(residual, low, high, xtol=1e-4))


def bielliptic_injection_cost(
    aphelion: float,
    dive_perihelion: float = _DIVE_PERIAPSIS,
    launch_radius: float = _AU,
) -> Tuple[float, float, float]:
    """Heliocentric impulse to reach a solar perihelion via a raised aphelion.

    The textbook bi-elliptic ledger, stated in this cycle's units so the Delta-v
    claim can be checked without the growth machinery: raise, then drop.  A 1 AU
    to 4 solar-radii transfer is a 54:1 radius ratio, far past the ~15.6:1 where
    the bi-elliptic route always beats the direct one, so the total falls
    monotonically towards ``(sqrt(2) - 1) * v_earth`` = 12.34 km/s.

    Args:
        aphelion: Aphelion to raise to (km).
        dive_perihelion: Perihelion the dive must reach (km).
        launch_radius: Radius the transfer starts from (km).

    Returns:
        (raise impulse, drop impulse, total) in km/s.
    """

    def apsis_speed(perihelion: float, apoapsis: float, radius: float) -> float:
        return float(np.sqrt(_MU_SUN * (2.0 / radius - 2.0 / (perihelion + apoapsis))))

    circular = float(np.sqrt(_MU_SUN / launch_radius))
    raise_impulse = apsis_speed(launch_radius, aphelion, launch_radius) - circular
    drop_impulse = apsis_speed(launch_radius, aphelion, aphelion) - apsis_speed(
        dive_perihelion, aphelion, aphelion
    )
    return raise_impulse, drop_impulse, raise_impulse + drop_impulse


def _delta_v_table() -> str:
    """The bi-elliptic Delta-v ledger against the direct dive."""
    circular = float(np.sqrt(_MU_SUN / _AU))
    direct = circular - float(
        np.sqrt(_MU_SUN * (2.0 / _AU - 2.0 / (_DIVE_PERIAPSIS + _AU)))
    )
    rows = []
    for aphelion_au in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 20.0):
        raise_dv, drop_dv, total = bielliptic_injection_cost(aphelion_au * _AU)
        rows.append(
            [
                f"{aphelion_au:.1f}",
                f"{raise_dv:.3f}",
                f"{drop_dv:.3f}",
                f"{total:.3f}",
                f"{total / direct:.3f}",
            ]
        )
    escape = circular * (np.sqrt(2.0) - 1.0)
    rows.append(
        ["inf", f"{escape:.3f}", "0.000", f"{escape:.3f}", f"{escape / direct:.3f}"]
    )
    rendered = tabulate(
        rows,
        headers=[
            "aphelion AU",
            "raise",
            "drop",
            "total km/s",
            f"vs {direct:.2f} direct",
        ],
        tablefmt="github",
    )
    return str(rendered)


def _closure_curve_table(revolutions: int) -> str:
    """Growth along the Earth re-intercept closure curve, injection at aphelion."""
    rows = []
    for perihelion_au in (
        _DIVE_PERIAPSIS / _AU,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.99,
    ):
        perihelion = perihelion_au * _AU
        aphelion = reintercept_closing_aphelion(perihelion, revolutions)
        geometry = split_dive_geometry(perihelion, aphelion, 180.0)
        assert geometry is not None
        ledger = split_dive_ledger(geometry)
        rows.append(
            [
                f"{perihelion_au:.4f}",
                f"{aphelion / _AU:.4f}",
                f"{geometry.cycle_years:.4f}",
                f"{ledger.departure.delta_v:.3f}",
                f"{geometry.injection.delta_v:.3f}",
                f"{ledger.return_per_impactor_kg:.2f}",
                f"{ledger.doubling_years:.4f}",
                f"{ledger.slug_per_delivered_kg:.4f}",
                f"{geometry.colocation_residual:+.2f}",
            ]
        )
    rendered = tabulate(
        rows,
        headers=[
            "q AU",
            "r_a AU",
            "cycle yr",
            "dv Earth",
            "dv node",
            "growth",
            "doubling",
            "slug/kg",
            "co-loc deg",
        ],
        tablefmt="github",
    )
    return str(rendered)


def _closure_table(
    fractions: Sequence[Tuple[int, int]] = DEFAULT_REUSE_FRACTIONS,
) -> str:
    """The fully phased two-node cycles."""
    rows = []
    for numerator, denominator in fractions:
        closure = two_node_closure(numerator, denominator)
        if closure is None:
            rows.append([f"{numerator}/{denominator}"] + ["--"] * 9)
            continue
        g, l = closure.geometry, closure.ledger
        rows.append(
            [
                f"{numerator}/{denominator}",
                f"{closure.interleaved_chains}",
                f"{g.outbound_perihelion / _AU:.4f}",
                f"{g.outbound_aphelion / _AU:.4f}",
                f"{np.degrees(g.injection_anomaly) - 180.0:.3f}",
                f"{g.cycle_years:.4f}",
                f"{g.injection.delta_v:.3f}",
                f"{l.doubling_years:.4f}",
                f"{l.slug_per_delivered_kg:.4f}",
                f"{closure.node_cadence_years:.4f}",
            ]
        )
    rendered = tabulate(
        rows,
        headers=[
            "reuse",
            "chains",
            "q AU",
            "r_a AU",
            "past apo deg",
            "cycle yr",
            "dv node",
            "doubling",
            "slug/kg",
            "cadence yr",
        ],
        tablefmt="github",
    )
    return str(rendered)


def _conduction_table(
    depths: Sequence[float] = DEPTH_CONDUCTION_SAMPLES,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
) -> str:
    """Both architectures' departures against the conduction floor, by depth."""
    rows = []
    for depth in depths:
        row = depth_conduction(depth, expansion_margin=expansion_margin)
        if row is None:
            continue
        rows.append(
            [
                f"{row.dive_solar_radii:.0f}",
                f"{row.stream_excess:.1f}",
                f"{row.direct_delta_v:.2f}",
                f"{row.direct_closing_start:.1f} -> {row.direct_closing_end:.1f}",
                "yes" if row.direct_clears else "**COLD**",
                f"{row.split_delta_v:.2f}",
                f"{row.split_closing_end:.1f}",
                "yes" if row.split_clears else "**COLD**",
                f"{row.outer_closing_cold:.1f}",
                "yes" if row.outer_clears else "**COLD**",
            ]
        )
    rendered = tabulate(
        rows,
        headers=[
            "R_sun",
            "stream",
            "direct dv",
            "direct w",
            "conducts",
            "split dv",
            "split w",
            "conducts",
            "outer w",
            "conducts",
        ],
        tablefmt="github",
    )
    return str(rendered)


def _opposing_stream_table(
    depths: Sequence[float] = DEPTH_CONDUCTION_SAMPLES,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
) -> str:
    """The opposing stream's placement, direct against split, by depth."""
    rows = []
    for depth in depths:
        row = opposing_stream_placement_trade(depth, expansion_margin=expansion_margin)
        if row is None:
            continue
        rows.append(
            [
                f"{row.dive_solar_radii:.0f}",
                f"{row.direct_excess:.2f}",
                f"{row.direct_closing_start:.1f} -> {row.direct_closing_end:.1f}",
                "yes" if row.direct_clears else "**COLD**",
                f"{row.radial_excess:.2f}",
                f"{row.split_raise_excess:.2f}",
                f"{row.split_flip_delta_v:.2f}",
                f"{row.split_total:.2f}",
                "yes" if row.split_clears else "**COLD**",
                f"{row.saving:.2f}x",
            ]
        )
    rendered = tabulate(
        rows,
        headers=[
            "R_sun",
            "retro direct",
            "its closing speed",
            "conducts",
            "radial",
            "split raise",
            "split flip",
            "split total",
            "conducts",
            "saving",
        ],
        tablefmt="github",
    )
    return str(rendered)


def _node_geometry_table(
    dive_solar_radii: float = 4.0,
    slug_ratios: Sequence[float] = NODE_GEOMETRY_SLUG_RATIOS,
) -> str:
    """Head-on against plunger at the dive node, across the slug ratio."""
    rows = []
    for slug_ratio in slug_ratios:
        row = node_geometry_trade(dive_solar_radii, slug_ratio)
        rows.append(
            [
                f"{slug_ratio:g}",
                f"{row.head_on_beta:.4f}",
                f"{row.plunger_beta:.4f}",
                f"{row.plunger_beta / row.head_on_beta:.3f}x",
                f"{row.head_on_exhaust:.1f}",
                f"{row.plunger_exhaust:.1f}",
                f"{row.exhaust_ratio:.3f}x",
                f"{row.head_on_survival:.3f}",
                f"{row.plunger_survival:.3f}",
                f"{row.thermal_ratio:.3f}x",
            ]
        )
    rendered = tabulate(
        rows,
        headers=[
            "k",
            "beta head",
            "beta plunge",
            "beta gain",
            "v_e head",
            "v_e plunge",
            "Isp kept",
            "surv head",
            "surv plunge",
            "heat",
        ],
        tablefmt="github",
    )
    return str(rendered)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Split the solar-dive injection across two nodes (ADR 0023)."
    )
    parser.add_argument(
        "--revolutions",
        type=int,
        default=0,
        help="re-intercept resonance index for the closure-curve table",
    )
    parser.add_argument(
        "--expansion-margin",
        type=float,
        default=DEFAULT_EXPANSION_MARGIN,
        help="headroom demanded above the expansion floor in the conduction table",
    )
    return parser


def main() -> None:
    """Print the Delta-v ledger, the closure curve, and the phased cycles."""
    args = _parser().parse_args()
    paper = paper_resonant_dive_ledger(158.50, 98.48)

    print("Bi-elliptic dive injection: the Delta-v claim")
    print(_delta_v_table())
    print()
    print(
        "The baseline it has to beat -- the paper's single-impulse resonant dive,\n"
        "on this same nozzle (k = 30, eta_jet^2 = 0.60, node survival 0.60):"
    )
    print(
        f"  cycle {paper.cycle_years:.4f} yr, departure burn "
        f"{paper.nozzle.delta_v:.3f} km/s from the parking orbit, "
        f"delivered {paper.nozzle.delivered_fraction:.4f},"
    )
    print(
        f"  growth {paper.return_per_impactor_kg:.2f}x, doubling "
        f"{paper.doubling_years:.4f} yr, slug "
        f"{(1.0 - paper.nozzle.delivered_fraction) / paper.nozzle.delivered_fraction:.4f}"
        " kg per delivered kg"
    )
    print()
    print(
        f"Growth along the Earth re-intercept closure curve (revolutions = "
        f"{args.revolutions}); the co-location column is the gap the outer node\n"
        "would have to close to be fed by the beam, and it never reaches zero:"
    )
    print(_closure_curve_table(args.revolutions))
    print()
    optimum = partial_split_optimum(args.revolutions)
    print(
        f"Rate optimum on that curve: q = "
        f"{optimum.geometry.outbound_perihelion / _AU:.4f} AU, r_a = "
        f"{optimum.geometry.outbound_aphelion / _AU:.4f} AU, cycle "
        f"{optimum.geometry.cycle_years:.4f} yr,"
    )
    print(
        f"  doubling {optimum.doubling_years:.4f} yr and slug "
        f"{optimum.slug_per_delivered_kg:.4f} kg/kg -- better than the paper's dive "
        "on both axes."
    )
    print()
    print(
        "Fully phased two-node cycles: all three conditions met, so the beam that\n"
        "feeds Earth also feeds the outer node of a chain further round the pattern."
    )
    print(_closure_table())
    print()
    threshold = conduction_threshold_closing_speed(
        DEFAULT_JET_ENERGY_EFFICIENCY, args.expansion_margin
    )
    print(
        "The conduction floor, carried onto the departure burn.  The Earth push runs\n"
        "*with* the stream, so its closing speed FALLS through the burn and its coldest\n"
        "instant is the end; the far node's thrust is perpendicular to the stream, so it\n"
        f"cannot cool itself.  Floor at margin {args.expansion_margin}: "
        f"{threshold:.2f} km/s."
    )
    print(_conduction_table(expansion_margin=args.expansion_margin))
    print()
    print(
        "The other half of the dive node. The **opposing stream** has to be placed at\n"
        "the same perihelion, and that leg runs the other way under the depth dial: a\n"
        "shallower dive keeps more angular momentum, so REVERSING its sign costs more.\n"
        "It conducts at every depth -- it aims retrograde against a radial stream, so it\n"
        "never runs along what feeds it -- but it becomes the dominant Earth-side cost,\n"
        "and the split's saving on it GROWS with depth where the payload leg's shrinks."
    )
    print(_opposing_stream_table(expansion_margin=args.expansion_margin))
    print()
    trade = node_geometry_trade(4.0)
    equivalent = plunger_equivalent_depth(4.0)
    print(
        "And the node's own geometry. A straight-down plunger arrives at "
        f"{trade.plunger_angle:.2f} deg, not 90 --\n"
        "payload and plunger reach perihelion within a percent of the same speed, so the\n"
        "relative velocity bisects the axes and the closing speed falls by 1/root 2. But\n"
        "the impulse law's `cos theta` debit -- the impactor's own momentum arriving\n"
        "backwards -- falls from -1 to -0.707 with it, so `beta` RISES. Which wins is a\n"
        "question about the slug ratio, because the debit is k-independent while the\n"
        f"useful term grows as sqrt(k). At {trade.dive_solar_radii:.0f} solar radii:"
    )
    print(_node_geometry_table())
    if equivalent is not None:
        print()
        print(
            f"The heat halves at every k, so a plunger at {trade.dive_solar_radii:.0f} "
            f"solar radii collides as gently as a head-on node at "
            f"{equivalent:.2f} -- worth about a factor of two in depth on the\n"
            "collision term alone (the solar flux at the node is untouched by it)."
        )
    crossing = direct_departure_conduction_depth(args.expansion_margin)
    if crossing is not None:
        print()
        print(
            f"The direct single-impulse dive's departure goes cold at "
            f"{crossing:.2f} solar radii."
        )
        print(
            "The split's Earth node has no crossing anywhere in "
            f"{DEPTH_CONDUCTION_BRACKET} solar radii."
        )


if __name__ == "__main__":
    main()
