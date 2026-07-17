"""Powered Jovian flyby retrograde return (planned subsection under
sec:jupiter_only_growth; ADR 0002-jupiter-flyby-objective; CONTEXT.md
"Jupiter powered-flyby retrograde return").

The leg the catalog's Jovian-return rows assume but never derive: an Oberth
methalox burn at 200 km above Earth (from C3 = 0, PuffSat-provided), a coast
to Jupiter, and a powered gravity assist -- a second impulsive methalox burn
at Jovian periapsis -- that bends and pumps the trajectory into a retrograde
heliocentric return crossing 1 AU. Planar patched conic; circular coplanar
planet orbits; velocities in the local (tangential, radial-outward) basis.

"Minimize total delta-v" is ill-posed here (the retrograde-plunge degeneracy,
see the ADR), so the optimizer maximizes the end-to-end mass ratio instead:
delivered mass fraction x payload mass ratio at the achieved collision speed.
The leg-level algebra lives in retrograde_return_legs.py (the substrate this
module shares with assist_chain.py and nozzle_analysis.py); this module is
the public-facing optimizer built on top of it.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.bodies import Earth, Jupiter
from scipy.optimize import differential_evolution

from src.astro_constants import (
    JUPITER_FLYBY_MAX_TOF,
    JUPITER_FLYBY_VB_TRADE_TARGETS,
    LEO_ALTITUDE,
    LOW_JUPITER_ALTITUDE,
    METHALOX_VACUUM_ISP,
    PUFFSAT_CYCLE_ORBIT_PERIOD,
)
from src.propulsion import exhaust_velocity_from_isp, payload_mass_ratio
from src.retrograde_return_legs import (
    _FlybyLeg,
    _FlybyParams,
    _powered_flyby_leg,
    _powered_flyby_params,
)


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
