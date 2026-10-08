"""The 1500 t launch unit through one growth cycle, plate push to departure.

The flown chain (:mod:`src.two_wave_growth`) is priced at a 200 km periapsis
with impulsive burns.  The parent flies both collisions higher
(``sec:jovian_meeting_altitudes``): the growth push at 400 km, and the
departure at 600 km, where the rod's steering packages see almost no drag.
This module moves each cycle's burns and wave speeds to those altitudes by
energy conservation, then carries one launch unit through the cycle, charging
the methalox apoapsis reversal (ADR 0009) that every design pays.
"""

import argparse
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from astropy import units as u
from boinor.bodies import Earth
from scipy.optimize import minimize_scalar
from tabulate import tabulate

from src.astro_constants import LEO_ALTITUDE, PUFFSAT_CYCLE_ORBIT_PERIOD
from src.chamber_departure import (
    DepartureLedger,
    LossModel,
    best_departure,
    growth_per_cycle,
)
from src.chamber_isp import (
    GATE_THRUST_COST,
    HYDROGEN_5500K,
    METHANE_7000K,
    ROD_MASS,
    ChamberPairing,
    absolute_efficiency,
)
from src.finite_burn_loss import fixed_direction_loss, steered_loss
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.nozzle_analysis import apoapsis_reversal_dv
from src.two_wave_growth import VE_METHALOX, TwoWaveCycle, adaptive_two_wave_cycles
from src.water_plate import (
    ARGON_SLUG,
    NO_BONDS,
    NO_FILM,
    PLATE_MAX_SLUG_RATIO,
    WATER_BOND_ENERGY,
    WATER_SLUG,
    OptimalPlatePush,
    PlateSlug,
    chemistry_ceiling,
    optimal_plate_push,
)


def _mu() -> float:
    return float(Earth.k.to_value(u.km**3 / u.s**2))


def _radius(altitude: u.Quantity) -> float:
    return float((Earth.R + altitude).to_value(u.km))


@dataclass(frozen=True)
class Departure:
    """A departure burn moved to another periapsis altitude.

    Attributes:
        start_speed: The 20-day cycle orbit's periapsis speed there.
        burn: Burn that reaches the same excess speed from there.
    """

    start_speed: u.Quantity
    burn: u.Quantity


def departure_at_altitude(
    burn_at_200_km: u.Quantity,
    altitude: u.Quantity,
    period: u.Quantity = PUFFSAT_CYCLE_ORBIT_PERIOD,
) -> Departure:
    """Move a chain departure burn from the 200 km periapsis to ``altitude``.

    The chain's burn raises the cycle orbit's 200 km periapsis speed to the
    speed that leaves with some excess speed ``v_inf``.  From a higher
    periapsis the same ``v_inf`` needs ``sqrt(v_inf^2 + 2 mu / r)``, starting
    from that altitude's own cycle-orbit periapsis speed.  A higher burn gets
    less from the Oberth effect, so it costs more.

    The chain defines its burn from the 20-day orbit's 200 km periapsis speed
    (``two_wave_growth._cycle_periapsis_speed``), so the excess speed is taken
    from there; the departure then starts from the parking orbit actually flown.

    Args:
        burn_at_200_km: The chain's ``departure_burn``.
        altitude: Periapsis altitude to depart from.
        period: Period of the parking orbit the departure starts from.

    Returns:
        The burn's start speed and size at ``altitude``.
    """
    mu = _mu()
    start_200 = float(puffsat_cycle_periapsis_speed().to_value(u.km / u.s))
    final_200 = start_200 + float(burn_at_200_km.to_value(u.km / u.s))
    excess_sq = final_200**2 - 2.0 * mu / _radius(LEO_ALTITUDE)
    start = puffsat_cycle_periapsis_speed(period=period, altitude=altitude).to(
        u.km / u.s
    )
    final = np.sqrt(excess_sq + 2.0 * mu / _radius(altitude)) * u.km / u.s
    return Departure(start_speed=start, burn=final - start)


#: The fixed launch unit, as it reaches the 400 km intercept: plate, chambers,
#: spray slug and its tank, departure propellant and tanks, and payload.  Every
#: unit brings its own chambers and tanks, which are expended with the departure
#: (:mod:`src.chamber_departure`).
LAUNCH_UNIT = 1500.0 * u.t
#: The pusher plate, about a tenth of the craft at a few hertz and a few metres
#: of stroke (``sec:plate_reuse``).  Dropped before the departure, probably to be
#: reused, but the ledger takes no credit for reuse.
PLATE_MASS = 150.0 * u.t
#: The plate sprays argon onto ice PuffSats (user, 2026-09-30): argon has no
#: bonds to strand, and it beats water by 8-11% in doubling (make plate-slug).
DEFAULT_PLATE_SLUG = ARGON_SLUG
#: The report's split gap, which is also the parking orbit (CONTEXT.md): 20
#: days, the orbit the parent's 600 km figures are quoted on (user, 2026-09-30).
#: Flown consistently it beats 10 days: the reversal falls from 372.5 to
#: 233.9 m/s while the growth wave's early-arrival burn rises from 179 to
#: 464 m/s, a wash for the chambers (0-2%) and 9% for methalox (ADR 0033).
#: ``two_wave_growth.DEFAULT_SPLIT_DAYS`` stays 10 for that module's reports.
DEFAULT_PARKING_DAYS = 20.0
#: Altitudes of the two collisions (``sec:jovian_meeting_altitudes``).
PUSH_ALTITUDE = 400.0 * u.km
DEPARTURE_ALTITUDE = 600.0 * u.km
_HOLD_ITERATIONS = 100
_HOLD_TOLERANCE_T = 1.0e-9

#: The chain's burns are defined on this orbit; the ledger flies each cycle's
#: own parking orbit, which is its split gap (CONTEXT.md, "Split gap").
REFERENCE_PERIOD = PUFFSAT_CYCLE_ORBIT_PERIOD


def _orbit_loss(
    table: Callable[[u.Quantity, u.Quantity, u.Quantity], u.Quantity],
    period: u.Quantity,
) -> LossModel:
    """A tabulated finite-burn loss bound to one parking orbit."""
    return lambda burn, burn_time: table(burn, burn_time, period)


def parking_period(cycle: TwoWaveCycle) -> u.Quantity:
    """The orbit the pushed payload coasts through: the cycle's own split gap.

    The growth wave pushes the payload at periapsis, it coasts one full orbit
    while the departure wave catches up, and it departs at the next periapsis,
    so the split gap and the parking period are one number (CONTEXT.md).

    Args:
        cycle: The flown cycle.

    Returns:
        The parking-orbit period.
    """
    return (cycle.split_days * u.day).to(u.day)


def periapsis_raise(period: u.Quantity) -> u.Quantity:
    """Methalox to raise periapsis from the push's 400 km to the departure's 600 km.

    Burned at apoapsis, where the craft crawls: the parent's 1.7 m/s at the
    20-day orbit's 613 000 km apoapsis (``sec:jovian_meeting_altitudes``).

    Args:
        period: Period of the parking orbit after the push.

    Returns:
        The burn (m/s).
    """
    mu = _mu()
    low, high = _radius(PUSH_ALTITUDE), _radius(DEPARTURE_ALTITUDE)
    a = (mu * (float(period.to_value(u.s)) / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    apoapsis = 2.0 * a - low
    before = np.sqrt(mu * (2.0 / apoapsis - 1.0 / a))
    after = np.sqrt(mu * (2.0 / apoapsis - 2.0 / (apoapsis + high)))
    return float(after - before) * 1.0e3 * u.m / u.s


def wave_speed_at_altitude(
    speed_at_200_km: u.Quantity, altitude: u.Quantity
) -> u.Quantity:
    """A returning wave's speed at another altitude, by energy conservation.

    Args:
        speed_at_200_km: The chain's collision speed at 200 km.
        altitude: Altitude to move it to.

    Returns:
        The wave's speed there.
    """
    v = float(speed_at_200_km.to_value(u.km / u.s))
    drop = 2.0 * _mu() * (1.0 / _radius(altitude) - 1.0 / _radius(LEO_ALTITUDE))
    return float(np.sqrt(v * v + drop)) * u.km / u.s


@dataclass(frozen=True)
class CycleGrowth:
    """One launch unit carried through one flown cycle.

    Attributes:
        push: The plate push, per kilogram of launch unit.
        puffsats: Growth-wave PuffSats the push consumes.
        departing_stack: Stack at departure ignition, after the plate and the
            water tank are dropped.
        departure: The chamber departure at its best chamber count.
        growth: PuffSats out over PuffSats in, both waves counted.
        initial_water_price: Starting water price of the plate's schedule.
        cryostats: Cryostat mass dropped with the plate before departure.
        boiled_off: Gas lost over the parking-orbit hold.
    """

    push: OptimalPlatePush
    puffsats: u.Quantity
    departing_stack: u.Quantity
    departure: DepartureLedger
    growth: float
    initial_water_price: float
    cryostats: u.Quantity = 0.0 * u.t
    boiled_off: u.Quantity = 0.0 * u.t


def _launch_and_push(
    wave_speed_at_200_km: float,
    plate_efficiency: float,
    initial_water_price: float,
    max_slug_ratio: Optional[float],
    slug: PlateSlug,
    impactor_bond_energy: u.Quantity,
    period: u.Quantity,
    film_per_impulse: u.Quantity = NO_FILM,
) -> Tuple[OptimalPlatePush, u.Quantity]:
    """Push the launch unit at 400 km and park it, ready to depart from 600 km.

    The wave pushes it from rest to the parking orbit's 400 km periapsis speed.
    At apoapsis it raises periapsis to 600 km and reverses the ellipse (ADR 0009),
    both in methalox, and it drops the plate and the slug's empty drop tank.
    The plate's film, if carried, is part of the unit and burns off in the push.

    Returns:
        The push, and the mass left to depart (before any cryostats or boil-off).
    """
    unit = LAUNCH_UNIT.to(u.t)
    push = optimal_plate_push(
        wave_speed_at_altitude(wave_speed_at_200_km * u.km / u.s, PUSH_ALTITUDE),
        puffsat_cycle_periapsis_speed(period=period, altitude=PUSH_ALTITUDE),
        plate_efficiency,
        initial_water_price,
        max_slug_ratio=max_slug_ratio,
        slug=slug,
        impactor_bond_energy=impactor_bond_energy,
        film_per_impulse=film_per_impulse,
    )
    methalox = (periapsis_raise(period) + apoapsis_reversal_dv(period)).to_value(
        u.km / u.s
    )
    available = (
        push.delivered_fraction * unit * float(np.exp(-methalox / VE_METHALOX))
        - PLATE_MASS
        - slug.tank_fraction * push.slug_fraction * unit
    ).to(u.t)
    return push, available


def price_cycle_growth(
    cycle: TwoWaveCycle,
    plate_efficiency: float,
    pairing: ChamberPairing,
    departure_efficiency: float,
    initial_water_price: float,
    gate_thrust_cost: float = GATE_THRUST_COST,
    pitch_ratio: float = 0.0,
    loss_model: Optional[LossModel] = None,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    cryostat_fraction: float = 0.0,
    boil_off: float = 0.0,
    film_per_impulse: u.Quantity = NO_FILM,
    chambers: Optional[int] = None,
) -> CycleGrowth:
    """Carry one launch unit through a flown cycle at a given water schedule.

    Args:
        cycle: The flown cycle.
        plate_efficiency: The plate's energy efficiency ``eta``.
        pairing: The departure chamber.
        departure_efficiency: The chamber's energy efficiency.
        initial_water_price: Starting water price of the plate's schedule.
        gate_thrust_cost: As in :func:`src.chamber_isp.effective_isp`.
        pitch_ratio: As in :func:`src.chamber_isp.effective_isp`.
        loss_model: Finite-burn loss; None uses the fixed-direction table on
            the cycle's own parking orbit.
        max_slug_ratio: Cap on the plate's water loading; None leaves it free.
        slug: What the plate sprays; argon by default.
        impactor_bond_energy: The PuffSat's bond energy per kilogram.
        cryostat_fraction: Cryostat mass per kilogram of gas launched; zero
            for methane, which the parent holds passively.
        boil_off: Share of the launched gas lost over the parking-orbit hold.
        film_per_impulse: Plate film burned per unit of push impulse, carried
            as launched mass (ADR 0043); none by default.
        chambers: Fly exactly this many chambers; None picks the best count.

    Returns:
        The cycle's ledger.
    """
    push, available = _launch_and_push(
        cycle.growth_wave_v_b,
        plate_efficiency,
        initial_water_price,
        max_slug_ratio,
        slug,
        impactor_bond_energy,
        parking_period(cycle),
        film_per_impulse,
    )
    departure_burn = departure_at_altitude(
        cycle.onward_burn * u.km / u.s, DEPARTURE_ALTITUDE, parking_period(cycle)
    )
    model = loss_model or _orbit_loss(fixed_direction_loss, parking_period(cycle))

    def depart(stack: u.Quantity) -> DepartureLedger:
        return best_departure(
            stack,
            departure_burn.start_speed,
            departure_burn.burn,
            wave_speed_at_altitude(
                cycle.nozzle_wave_v_b * u.km / u.s, DEPARTURE_ALTITUDE
            ),
            pairing,
            departure_efficiency,
            gate_thrust_cost=gate_thrust_cost,
            pitch_ratio=pitch_ratio,
            loss_model=model,
            chambers=chambers,
        )

    # The gas launched is what the burn spends over (1 - boil-off), and its
    # cryostats and boil-off never depart, so the departing stack is solved
    # self-consistently with the burn it has to fly.
    def launched_gas(stack: u.Quantity, departure: DepartureLedger) -> u.Quantity:
        spent = 1.0 - departure.delivered_fraction
        plug = pairing.plug_ratio
        burned = spent - (plug + pitch_ratio) * departure.rod_mass_fraction
        return (burned * stack / (1.0 - boil_off)).to(u.t)

    stack = available
    departure = depart(stack)
    for _ in range(_HOLD_ITERATIONS):
        held = (cryostat_fraction + boil_off) * launched_gas(stack, departure)
        updated = (available - held).to(u.t)
        converged = abs(float((updated - stack).to_value(u.t))) < _HOLD_TOLERANCE_T
        stack = updated
        departure = depart(stack)
        if converged:
            break
    else:
        raise RuntimeError("cryostat and boil-off charge did not converge")
    gas = launched_gas(stack, departure)
    puffsats = (push.puffsat_fraction * LAUNCH_UNIT).to(u.t)
    # Each wave burned methalox on the way home, so the batch that left Jupiter
    # was larger than what arrives: the growth wave to arrive early, the nozzle
    # wave for its deep-space-maneuver proxy.  (The chain's own ledger lumps both
    # onto the growth wave.)
    growth_arrives = float(np.exp(-cycle.growth_wave_burn / VE_METHALOX))
    rods_arrive = float(np.exp(-cycle.nozzle_wave_dsm / VE_METHALOX))
    growth = growth_per_cycle(
        float((stack / puffsats).to_value(u.one)) * growth_arrives,
        departure.rod_mass_fraction / rods_arrive,
        departure.delivered_net,
    )
    return CycleGrowth(
        push,
        puffsats,
        stack,
        departure,
        growth,
        initial_water_price,
        cryostats=(cryostat_fraction * gas).to(u.t),
        boiled_off=(boil_off * gas).to(u.t),
    )


_Flown = TypeVar("_Flown", "CycleGrowth", "MethaloxCycle")


def _best_over_price(fly: Callable[[float], _Flown]) -> _Flown:
    """Maximise ``fly(price).growth`` over the plate's starting water price."""
    scanned = [fly(float(p)) for p in _PRICE_GRID]
    i = int(np.argmax([c.growth for c in scanned]))
    low = float(np.log(_PRICE_GRID[max(i - 1, 0)]))
    high = float(np.log(_PRICE_GRID[min(i + 1, _PRICE_GRID.size - 1)]))
    refined = minimize_scalar(
        lambda x: -fly(float(np.exp(x))).growth,
        bounds=(low, high),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    best = fly(float(np.exp(refined.x)))
    return best if best.growth >= scanned[i].growth else scanned[i]


#: Log-spaced starting water prices the schedule search scans before refining.
#: The optimum has sat between 0.01 and 0.2 on every cycle tried.
_PRICE_GRID = np.geomspace(1.0e-3, 1.0, 31)


def best_cycle_growth(
    cycle: TwoWaveCycle,
    plate_efficiency: float,
    pairing: ChamberPairing,
    departure_efficiency: float,
    gate_thrust_cost: float = GATE_THRUST_COST,
    pitch_ratio: float = 0.0,
    loss_model: Optional[LossModel] = None,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    cryostat_fraction: float = 0.0,
    boil_off: float = 0.0,
    film_per_impulse: u.Quantity = NO_FILM,
    chambers: Optional[int] = None,
) -> CycleGrowth:
    """Carry the launch unit through a cycle on the water schedule that grows it most.

    The plate's schedule is fixed by its starting water price
    (:func:`src.water_plate.optimal_plate_push`), so the search is over that one
    number: a log grid over :data:`_PRICE_GRID`, then a bounded refinement
    between the best node's neighbours.

    Args:
        cycle: The flown cycle.
        plate_efficiency: The plate's energy efficiency ``eta``.
        pairing: The departure chamber.
        departure_efficiency: The chamber's energy efficiency.
        gate_thrust_cost: As in :func:`src.chamber_isp.effective_isp`.
        pitch_ratio: As in :func:`src.chamber_isp.effective_isp`.
        loss_model: Finite-burn loss; None uses the fixed-direction table on
            the cycle's own parking orbit.
        max_slug_ratio: As in :func:`price_cycle_growth`.
        slug: As in :func:`price_cycle_growth`.
        impactor_bond_energy: As in :func:`price_cycle_growth`.
        cryostat_fraction: As in :func:`price_cycle_growth`.
        boil_off: As in :func:`price_cycle_growth`.
        film_per_impulse: As in :func:`price_cycle_growth`.
        chambers: As in :func:`price_cycle_growth`.

    Returns:
        The cycle's ledger at the best schedule.
    """

    def fly(price: float) -> CycleGrowth:
        return price_cycle_growth(
            cycle,
            plate_efficiency,
            pairing,
            departure_efficiency,
            price,
            gate_thrust_cost=gate_thrust_cost,
            pitch_ratio=pitch_ratio,
            loss_model=loss_model,
            max_slug_ratio=max_slug_ratio,
            cryostat_fraction=cryostat_fraction,
            boil_off=boil_off,
            slug=slug,
            impactor_bond_energy=impactor_bond_energy,
            film_per_impulse=film_per_impulse,
            chambers=chambers,
        )

    return _best_over_price(fly)


@dataclass(frozen=True)
class ChainSummary:
    """Growth over a flown chain, reported the ledger's ways.

    Attributes:
        rate_per_year: Continuous growth rate, ``ln(product of growth) / years``.
        doubling_years: ``ln 2 / rate``.
        annual_growth: ``exp(rate) - 1``.
        ten_year_stepwise: Growth of the cycles that finish within ten years
            of the first launch, which is what actually lands.
        ten_year_continuous: The continuous rate projected, ``exp(10 rate)``.
    """

    rate_per_year: float
    doubling_years: float
    annual_growth: float
    ten_year_stepwise: float
    ten_year_continuous: float


def summarize_chain(
    periods_years: Sequence[float], growths: Sequence[float]
) -> ChainSummary:
    """Summarise a chain's per-cycle growth.

    Args:
        periods_years: Each cycle's length, in order.
        growths: Each cycle's growth factor, in order.

    Returns:
        The chain's summary.
    """
    rate = float(np.sum(np.log(growths)) / np.sum(periods_years))
    finished = np.cumsum(periods_years) <= 10.0
    return ChainSummary(
        rate_per_year=rate,
        doubling_years=float(np.log(2.0) / rate),
        annual_growth=float(np.expm1(rate)),
        ten_year_stepwise=float(np.prod(np.asarray(growths)[finished])),
        ten_year_continuous=float(np.exp(10.0 * rate)),
    )


def chain_growth(
    cycles: Sequence[TwoWaveCycle],
    plate_efficiency: float,
    pairing: ChamberPairing,
    departure_efficiency: float,
    pitch_ratio: float = 0.0,
    loss_model: Optional[LossModel] = None,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    cryostat_fraction: float = 0.0,
    boil_off: float = 0.0,
    film_per_impulse: u.Quantity = NO_FILM,
    chambers: Optional[int] = None,
) -> List[CycleGrowth]:
    """Carry the launch unit through every flown cycle, each on its best schedule.

    Args:
        cycles: Flown cycles from :func:`src.two_wave_growth.adaptive_two_wave_cycles`.
        plate_efficiency: The plate's energy efficiency ``eta``.
        pairing: The departure chamber.
        departure_efficiency: The chamber's energy efficiency.
        pitch_ratio: As in :func:`src.chamber_isp.effective_isp`.
        loss_model: Finite-burn loss; None uses the fixed-direction table on
            the cycle's own parking orbit.
        max_slug_ratio: As in :func:`price_cycle_growth`.
        slug: As in :func:`price_cycle_growth`.
        impactor_bond_energy: As in :func:`price_cycle_growth`.
        cryostat_fraction: As in :func:`price_cycle_growth`.
        boil_off: As in :func:`price_cycle_growth`.
        film_per_impulse: As in :func:`price_cycle_growth`.
        chambers: As in :func:`price_cycle_growth`.

    Returns:
        One ledger per cycle, in order.
    """
    return [
        best_cycle_growth(
            cycle,
            plate_efficiency,
            pairing,
            departure_efficiency,
            pitch_ratio=pitch_ratio,
            loss_model=loss_model,
            max_slug_ratio=max_slug_ratio,
            cryostat_fraction=cryostat_fraction,
            boil_off=boil_off,
            slug=slug,
            impactor_bond_energy=impactor_bond_energy,
            film_per_impulse=film_per_impulse,
            chambers=chambers,
        )
        for cycle in cycles
    ]


#: Raptor 3, sea-level: 280 tf, 1525 kg (SpaceX, 2024-08-01,
#: https://x.com/SpaceX/status/1819772716339339664).  No vacuum Raptor 3 has
#: been published, so its mass and thrust are paired with the parent's 380 s
#: vacuum methalox Isp (``raptor_vacuum_isp``); a mixed assumption, stated.
RAPTOR3_MASS = 1.525 * u.t
RAPTOR3_THRUST = (280.0 * u.t * u.m / u.s**2 * 9.80665).to(u.MN)
#: Methalox tank per kilogram of propellant, density-scaled from the parent's
#: drop-tank figures for the oxygen-methane mix.
METHALOX_TANK_FRACTION = 0.018
_MAX_ENGINES = 200


@dataclass(frozen=True)
class MethaloxDeparture:
    """A Raptor 3 departure, per kilogram of stack at ignition.

    Attributes:
        delivered_net: Stack delivered net of propellant, tanks and engines.
        engines: Raptor 3s firing.
        burn_time: Propellant over the engines' combined flow.
        finite_burn_loss: Steered-burn loss, included in the burn priced.
    """

    delivered_net: float
    engines: int
    burn_time: u.Quantity
    finite_burn_loss: u.Quantity


def methalox_departure(
    stack: u.Quantity,
    burn: u.Quantity,
    engines: int,
    loss_model: LossModel = steered_loss,
) -> MethaloxDeparture:
    """Depart a stack on Raptor 3s at 380 s, the loss solved as a fixed point.

    Args:
        stack: Stack at ignition.
        burn: Impulsive burn needed.
        engines: Raptor 3s firing.
        loss_model: Finite-burn loss; steered, since engines can follow the
            velocity.

    Returns:
        The departure.
    """
    exhaust = VE_METHALOX * u.km / u.s
    flow = (engines * RAPTOR3_THRUST / exhaust).to(u.t / u.s)
    loss = 0.0 * u.m / u.s
    for _ in range(_HOLD_ITERATIONS):
        spent = 1.0 - float(np.exp(-((burn + loss) / exhaust).to_value(u.one)))
        burn_time = (spent * stack / flow).to(u.s)
        updated = loss_model(burn, burn_time).to(u.m / u.s)
        converged = abs(float((updated - loss).to_value(u.m / u.s))) < 1.0e-9
        loss = updated
        if converged:
            break
    spent = 1.0 - float(np.exp(-((burn + loss) / exhaust).to_value(u.one)))
    engines_share = float((engines * RAPTOR3_MASS / stack).to_value(u.one))
    return MethaloxDeparture(
        delivered_net=1.0 - spent - METHALOX_TANK_FRACTION * spent - engines_share,
        engines=engines,
        burn_time=(spent * stack / flow).to(u.s),
        finite_burn_loss=loss,
    )


@dataclass(frozen=True)
class MethaloxCycle:
    """One launch unit through a cycle with the methalox incumbent departing.

    Attributes:
        push: The plate push.
        puffsats: PuffSats the push consumes: the whole returning batch.
        departing_stack: Stack at ignition.
        delivered_net: Stack delivered net of propellant, tanks and engines.
        engines: Raptor 3s at the best count.
        finite_burn_loss: Steered-burn loss.
        growth: PuffSats out over PuffSats in.
        initial_water_price: Starting water price of the plate's schedule.
    """

    push: OptimalPlatePush
    puffsats: u.Quantity
    departing_stack: u.Quantity
    delivered_net: float
    engines: int
    finite_burn_loss: u.Quantity
    growth: float
    initial_water_price: float


def price_methalox_cycle(
    cycle: TwoWaveCycle,
    plate_efficiency: float,
    initial_water_price: float,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    loss_model: Optional[LossModel] = None,
    film_per_impulse: u.Quantity = NO_FILM,
) -> MethaloxCycle:
    """Carry the launch unit through a cycle with Raptor 3s departing.

    With no departure wave there is no split, so the whole batch comes home
    together at the return's speed and pays only its DSM proxy, and all of it
    pushes.  The engine count is the one that delivers the most.

    Args:
        cycle: The flown cycle; methalox flies three-synodic cycles only.
        plate_efficiency: The plate's energy efficiency ``eta``.
        initial_water_price: Starting water price of the plate's schedule.
        max_slug_ratio: As in :func:`price_cycle_growth`.
        slug: As in :func:`price_cycle_growth`.
        impactor_bond_energy: As in :func:`price_cycle_growth`.
        loss_model: Finite-burn loss; None uses the steered table on the
            cycle's own parking orbit.
        film_per_impulse: As in :func:`price_cycle_growth`.

    Returns:
        The cycle's ledger.
    """
    push, stack = _launch_and_push(
        cycle.nozzle_wave_v_b,
        plate_efficiency,
        initial_water_price,
        max_slug_ratio,
        slug,
        impactor_bond_energy,
        parking_period(cycle),
        film_per_impulse,
    )
    burn = departure_at_altitude(
        cycle.onward_burn * u.km / u.s, DEPARTURE_ALTITUDE, parking_period(cycle)
    )
    model = loss_model or _orbit_loss(steered_loss, parking_period(cycle))
    best: Optional[MethaloxDeparture] = None
    falls = 0
    for engines in range(1, _MAX_ENGINES + 1):
        try:
            flown = methalox_departure(stack, burn.burn, engines, model)
        except ValueError:
            continue
        if best is None or flown.delivered_net > best.delivered_net:
            best, falls = flown, 0
        else:
            falls += 1
            if falls == 2:
                break
    if best is None:
        raise RuntimeError("no engine count converged")
    puffsats = (push.puffsat_fraction * LAUNCH_UNIT).to(u.t)
    batch = puffsats / float(np.exp(-cycle.nozzle_wave_dsm / VE_METHALOX))
    growth = best.delivered_net * float((stack / batch).to_value(u.one))
    return MethaloxCycle(
        push,
        puffsats,
        stack,
        best.delivered_net,
        best.engines,
        best.finite_burn_loss,
        growth,
        initial_water_price,
    )


def best_methalox_cycle(
    cycle: TwoWaveCycle,
    plate_efficiency: float,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    film_per_impulse: u.Quantity = NO_FILM,
) -> MethaloxCycle:
    """:func:`price_methalox_cycle` on the water schedule that grows it most.

    Args:
        cycle: The flown cycle.
        plate_efficiency: The plate's energy efficiency ``eta``.
        max_slug_ratio: As in :func:`price_cycle_growth`.
        slug: As in :func:`price_cycle_growth`.
        impactor_bond_energy: As in :func:`price_cycle_growth`.
        film_per_impulse: As in :func:`price_cycle_growth`.

    Returns:
        The cycle's ledger at the best schedule.
    """
    return _best_over_price(
        lambda price: price_methalox_cycle(
            cycle,
            plate_efficiency,
            price,
            max_slug_ratio=max_slug_ratio,
            slug=slug,
            impactor_bond_energy=impactor_bond_energy,
            film_per_impulse=film_per_impulse,
        )
    )


#: The paper's injection ratio (``sec:water_injected_overtake``), the cap the
#: plate designs fly under (ADR 0041).  The schedule still falls below it.
PAPER_SLUG_RATIO = 8.52
#: One pulse of the spray cup: 134 t sprung at 4 Hz on a 2.8 m stroke
#: (parent ``eq:plate_pulse_size``).  Sets how film per pulse becomes film per
#: unit of impulse (ADR 0043).
PULSE_IMPULSE = 12.0 * u.MN * u.s


@dataclass(frozen=True)
class PlateDesign:
    """A plate the ledger and the cost model can fly (ADR 0041).

    Attributes:
        label: Short label.
        efficiency: The energy efficiency ``eta = eta_jet^2`` the ledger takes,
            net of whatever chemistry it charges per pulse.
        impactor_bond_j_kg: The PuffSat's bond energy charged per pulse, J/kg.
            Zero for the impact-sim's solved designs, whose ``eta_jet`` already
            charges the PuffSat's water as lost.
        max_slug_ratio: Cap on the per-pulse loading.
        film_per_pulse_kg: Plate film burned per :data:`PULSE_IMPULSE` pulse,
            carried as launched mass (ADR 0043).  Zero leaves it to the cost
            book's film line, as every design did before.
    """

    label: str
    efficiency: float
    impactor_bond_j_kg: float
    max_slug_ratio: float
    film_per_pulse_kg: float = 0.0

    @property
    def film_per_impulse(self) -> u.Quantity:
        """Film burned per newton-second of push."""
        return (self.film_per_pulse_kg * u.kg / PULSE_IMPULSE).to(u.kg / (u.N * u.s))

    @property
    def carries_film(self) -> bool:
        """Whether the ledger, rather than the cost book, carries the film."""
        return self.film_per_pulse_kg > 0.0

    @property
    def jet_efficiency(self) -> float:
        """``eta_jet``, the impact-sim's convention."""
        return float(np.sqrt(self.efficiency))

    @property
    def impactor_bond_energy(self) -> u.Quantity:
        """The PuffSat's bond energy as a quantity."""
        return self.impactor_bond_j_kg * u.J / u.kg


_WATER_BOND_J_KG = float(WATER_BOND_ENERGY.to_value(u.J / u.kg))
#: ADR 0033's plate, which every figure before ADR 0041 flew: eta 0.7 net of
#: the ice PuffSat's bonds, charged per pulse, k <= 10.
ADR_0033_PLATE = PlateDesign(
    "ADR 0033 (eta 0.7, bonds per pulse)", 0.7, _WATER_BOND_J_KG, PLATE_MAX_SLUG_RATIO
)
#: The impact-sim's spray-plate designs (its ADR-0055, handoff Draft 2,
#: 2026-10-05, P11), given as eta_jet: the spray cup flies first at 0.6, 0.57
#: unmixed is its downside, the plug reaches ~0.70 once perfected, and the
#: paper's 0.775 is a reference.  Each is all-in, so the PuffSat's bonds are
#: not charged again.
SPRAY_CUP = PlateDesign("spray cup 0.60", 0.60**2, 0.0, PAPER_SLUG_RATIO)
SPRAY_CUP_UNMIXED = PlateDesign("spray cup 0.57", 0.57**2, 0.0, PAPER_SLUG_RATIO)
PLUG = PlateDesign("plug 0.70", 0.70**2, 0.0, PAPER_SLUG_RATIO)
PAPER_PLATE = PlateDesign("paper 0.775", 0.775**2, 0.0, PAPER_SLUG_RATIO)
PLATE_DESIGNS = (SPRAY_CUP, SPRAY_CUP_UNMIXED, PLUG, PAPER_PLATE, ADR_0033_PLATE)
#: The spray cup's film per 12 MN s pulse (impact sim P5, parent S12): 4-6 kg
#: of pitch when its own vapor shields the face, 28-33 kg when it does not.
SPRAY_CUP_FILM_PER_PULSE = {"vapor-shielded": (4.0, 6.0), "unshielded": (28.0, 33.0)}
#: The spray cup with its film carried as launched mass, at the heavy end of
#: each band (ADR 0043).
SPRAY_CUP_SHIELDED = replace(
    SPRAY_CUP,
    label="spray cup 0.60, film 6 kg",
    film_per_pulse_kg=SPRAY_CUP_FILM_PER_PULSE["vapor-shielded"][1],
)
SPRAY_CUP_UNSHIELDED = replace(
    SPRAY_CUP,
    label="spray cup 0.60, film 33 kg",
    film_per_pulse_kg=SPRAY_CUP_FILM_PER_PULSE["unshielded"][1],
)
#: The designs by command-line name (``--plate``, ADR 0042, 0043).
PLATE_DESIGNS_BY_NAME = {
    "spray-cup": SPRAY_CUP,
    "spray-cup-shielded": SPRAY_CUP_SHIELDED,
    "spray-cup-unshielded": SPRAY_CUP_UNSHIELDED,
    "spray-cup-unmixed": SPRAY_CUP_UNMIXED,
    "plug": PLUG,
    "paper-0.775": PAPER_PLATE,
    "adr-0033": ADR_0033_PLATE,
}


#: The ledger's efficiencies.  The plate's are net of its chemistry toll,
#: charged pulse by pulse; the chambers' are shares of their chemistry ceiling
#: (:func:`src.chamber_isp.absolute_efficiency`), with each solved chamber added
#: at its own share.  1.0 is a theoretical ceiling on both legs.
PLATE_EFFICIENCIES = (0.50, 0.70, 1.00)
CEILING_SHARES = (0.50, 0.70, 0.90, 1.00)
SOLVED_EFFICIENCY = {HYDROGEN_5500K.name: 0.858, METHANE_7000K.name: 0.538}
DEPARTURE_PAIRINGS = (HYDROGEN_5500K, METHANE_7000K)
#: Methane's pitch per pulse: the range :func:`pitch_sweep` prints, and the
#: pessimistic end the matrix carries.
METHANE_PITCH_RANGE = (1.4 * u.kg, 5.6 * u.kg)
METHANE_PITCH = METHANE_PITCH_RANGE[1]


def pitch_ratio(pitch: u.Quantity, rod_mass: u.Quantity = ROD_MASS) -> float:
    """A pitch per pulse as a share of the rod, as :func:`chain_growth` takes it."""
    return float((pitch / rod_mass).to_value(u.one))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


#: Hydrogen's cryostat mass and boil-off, per kilogram launched.  No
#: flight-scale source exists; passive-MLI large tanks lose 0.1-0.5%/day, held
#: for one parking orbit, and cryostat mass is swept over 1-5%.  The matrix
#: carries the middle of both; :func:`_hold_sweep` shows the range.
HYDROGEN_CRYOSTAT = 0.03
HYDROGEN_BOIL_OFF_PER_DAY = 0.003


def hydrogen_boil_off(cycles: Sequence[TwoWaveCycle], per_day: float) -> float:
    """Hydrogen lost over the hold: one parking orbit, the chain's split gap."""
    return per_day * float(parking_period(cycles[0]).to_value(u.day))


def _chamber_row(
    cycles: Sequence[TwoWaveCycle],
    plate_eta: float,
    pairing: ChamberPairing,
    eta: float,
    label: str,
) -> Dict[str, object]:
    """One chamber row of the matrix: capped, with the uncapped doubling."""
    periods = [c.period_years for c in cycles]
    hydrogen = pairing is HYDROGEN_5500K
    pitch = 0.0 if hydrogen else pitch_ratio(METHANE_PITCH)
    cryostat = HYDROGEN_CRYOSTAT if hydrogen else 0.0
    boil_off = hydrogen_boil_off(cycles, HYDROGEN_BOIL_OFF_PER_DAY) if hydrogen else 0.0
    grown = chain_growth(
        cycles, plate_eta, pairing, eta, pitch,
        cryostat_fraction=cryostat, boil_off=boil_off,
    )  # fmt: skip
    free = chain_growth(
        cycles, plate_eta, pairing, eta, pitch, max_slug_ratio=None,
        cryostat_fraction=cryostat, boil_off=boil_off,
    )  # fmt: skip
    summary = summarize_chain(periods, [g.growth for g in grown])
    row: Dict[str, object] = {
        "plate": plate_eta,
        "departure": label,
        "eta": eta,
    }
    for multiple in (2, 3):
        picked = [g for c, g in zip(cycles, grown) if c.synodic_multiple == multiple]
        tag = f"{multiple}S"
        row[f"{tag} k"] = (
            f"{_mean([g.push.slug_ratio_start for g in picked]):.0f}"
            f"->{_mean([g.push.slug_ratio_end for g in picked]):.1f}"
        )
        row[f"{tag} push"] = _mean(
            [float((g.departing_stack / g.puffsats).to_value(u.one)) for g in picked]
        )
        row[f"{tag} net"] = _mean([g.departure.delivered_net for g in picked])
        row[f"{tag} growth"] = _mean([g.growth for g in picked])
    row["payload t"] = _mean(
        [g.departure.delivered_net * g.departing_stack.to_value(u.t) for g in grown]
    )
    row["doubling yr"] = summary.doubling_years
    row["uncapped yr"] = summarize_chain(
        periods, [g.growth for g in free]
    ).doubling_years
    row["annual"] = summary.annual_growth
    row["10yr step"] = summary.ten_year_stepwise
    row["10yr cont"] = summary.ten_year_continuous
    return row


def _methalox_row(
    cycles: Sequence[TwoWaveCycle], plate_eta: float
) -> Dict[str, object]:
    """The methalox incumbent on the three-synodic-only chain."""
    periods = [c.period_years for c in cycles]
    flown = [best_methalox_cycle(c, plate_eta) for c in cycles]
    free = [best_methalox_cycle(c, plate_eta, max_slug_ratio=None) for c in cycles]
    summary = summarize_chain(periods, [f.growth for f in flown])
    return {
        "plate": plate_eta,
        "departure": "methalox 380 s",
        "3S k": f"{_mean([f.push.slug_ratio_start for f in flown]):.0f}"
        f"->{_mean([f.push.slug_ratio_end for f in flown]):.1f}",
        "3S push": _mean(
            [float((f.departing_stack / f.puffsats).to_value(u.one)) for f in flown]
        ),
        "3S net": _mean([f.delivered_net for f in flown]),
        "engines": _mean([f.engines for f in flown]),
        "3S growth": _mean([f.growth for f in flown]),
        "payload t": _mean(
            [f.delivered_net * f.departing_stack.to_value(u.t) for f in flown]
        ),
        "doubling yr": summary.doubling_years,
        "uncapped yr": summarize_chain(
            periods, [f.growth for f in free]
        ).doubling_years,
        "annual": summary.annual_growth,
        "10yr step": summary.ten_year_stepwise,
        "10yr cont": summary.ten_year_continuous,
    }


def _hold_sweep(cycles: Sequence[TwoWaveCycle]) -> str:
    """The solved hydrogen chamber behind a 0.7 plate, across the hold's range."""
    periods = [c.period_years for c in cycles]
    rows = []
    for cryostat in (0.0, 0.01, 0.03, 0.05):
        row: Dict[str, object] = {"cryostat": cryostat}
        for per_day in (0.0, 0.001, 0.003, 0.005):
            grown = chain_growth(
                cycles, 0.7, HYDROGEN_5500K, 0.858, cryostat_fraction=cryostat,
                boil_off=hydrogen_boil_off(cycles, per_day),
            )  # fmt: skip
            row[f"{100 * per_day:g}%/day"] = summarize_chain(
                periods, [g.growth for g in grown]
            ).doubling_years
        rows.append(row)
    return tabulate(rows, headers="keys", floatfmt=".3g")


def pitch_sweep(
    cycles: Sequence[TwoWaveCycle], loss_model: Optional[LossModel] = None
) -> List[Dict[str, object]]:
    """Methane's doubling at each end of its pitch range, behind each plate.

    The matrix carries the pessimistic 5.6 kg; this is the sensitivity
    behind it, at the solved efficiency and at 50/70/100% of the ceiling.

    Args:
        cycles: Flown cycles from :func:`src.two_wave_growth.adaptive_two_wave_cycles`.
        loss_model: As in :func:`chain_growth`.

    Returns:
        One row per plate and departure: the doubling (yr) at each pitch, and
        what the lightest pitch saves against the heaviest.
    """
    periods = [c.period_years for c in cycles]
    solved = SOLVED_EFFICIENCY[METHANE_7000K.name]
    departures = sorted(
        [
            (absolute_efficiency(METHANE_7000K, s), f"CH4 {s:.0%}")
            for s in (0.5, 0.7, 1.0)
        ]
        + [(solved, f"CH4 solved ({solved / METHANE_7000K.chemistry_ceiling:.0%})")]
    )
    rows = []
    for plate_eta in PLATE_EFFICIENCIES:
        for eta, label in departures:
            doubling = [
                summarize_chain(
                    periods,
                    [
                        g.growth
                        for g in chain_growth(
                            cycles, plate_eta, METHANE_7000K, eta, pitch_ratio(pitch),
                            loss_model=loss_model,
                        )  # fmt: skip
                    ],
                ).doubling_years
                for pitch in METHANE_PITCH_RANGE
            ]
            row: Dict[str, object] = {"plate": plate_eta, "departure": label}
            for pitch, years in zip(METHANE_PITCH_RANGE, doubling):
                row[f"{pitch.to_value(u.kg):g} kg yr"] = years
            row["saved yr"] = doubling[-1] - doubling[0]
            rows.append(row)
    return rows


def _report(cycles: Sequence[TwoWaveCycle], three_only: Sequence[TwoWaveCycle]) -> str:
    """Tabulate the scenario matrix, the methalox incumbent and the hold sweep."""
    rows = []
    for plate_eta in PLATE_EFFICIENCIES:
        for pairing in DEPARTURE_PAIRINGS:
            gas = pairing.name.split()[0]
            solved = SOLVED_EFFICIENCY[pairing.name]
            departures = [
                (absolute_efficiency(pairing, share), f"{gas} {share:.0%}")
                for share in CEILING_SHARES
            ] + [(solved, f"{gas} solved ({solved / pairing.chemistry_ceiling:.0%})")]
            for eta, label in sorted(departures):
                rows.append(_chamber_row(cycles, plate_eta, pairing, eta, label))
    incumbent = [
        _methalox_row(three_only, plate_eta) for plate_eta in PLATE_EFFICIENCIES
    ]
    return (
        tabulate(rows, headers="keys", floatfmt=".3g")
        + "\n\nMethalox incumbent, three-synodic cycles only "
        + f"({len(three_only)} cycles), whole batch pushes:\n"
        + tabulate(incumbent, headers="keys", floatfmt=".3g")
        + "\n\nSolved hydrogen (0.858) behind a 0.7 plate: doubling (yr) across cryostat "
        + "mass (per kilogram launched) and boil-off rate over the hold:\n"
        + _hold_sweep(cycles)
        + "\n\nMethane's pitch per pulse: doubling (yr) at each end of "
        + f"{METHANE_PITCH_RANGE[0].to_value(u.kg):g}-"
        + f"{METHANE_PITCH_RANGE[1].to_value(u.kg):g} kg (the matrix carries "
        + f"{METHANE_PITCH.to_value(u.kg):g} kg):\n"
        + tabulate(pitch_sweep(cycles), headers="keys", floatfmt=".3f")
    )


#: The plate options: what it sprays, on what PuffSat.
PLATE_OPTIONS = (
    ("water on ice", WATER_SLUG, WATER_BOND_ENERGY),
    ("argon on ice", ARGON_SLUG, WATER_BOND_ENERGY),
    ("all argon", ARGON_SLUG, NO_BONDS),
)


def _slug_comparison(cycles: Sequence[TwoWaveCycle]) -> str:
    """Water against argon on the plate, behind each solved chamber."""
    periods = [c.period_years for c in cycles]
    rows = []
    for plate_eta in PLATE_EFFICIENCIES:
        for pairing in DEPARTURE_PAIRINGS:
            eta = SOLVED_EFFICIENCY[pairing.name]
            hydrogen = pairing is HYDROGEN_5500K
            pitch = 0.0 if hydrogen else pitch_ratio(METHANE_PITCH)
            held = HYDROGEN_CRYOSTAT if hydrogen else 0.0
            boil = (
                hydrogen_boil_off(cycles, HYDROGEN_BOIL_OFF_PER_DAY)
                if hydrogen
                else 0.0
            )
            for label, slug, impactor in PLATE_OPTIONS:
                runs = {
                    cap: chain_growth(
                        cycles, plate_eta, pairing, eta, pitch,
                        cryostat_fraction=held, boil_off=boil,
                        max_slug_ratio=cap, slug=slug, impactor_bond_energy=impactor,
                    )  # fmt: skip
                    for cap in (PLATE_MAX_SLUG_RATIO, None)
                }
                grown = runs[PLATE_MAX_SLUG_RATIO]
                ceilings = [
                    (
                        chemistry_ceiling(
                            wave_speed_at_altitude(c.growth_wave_v_b * u.km / u.s, PUSH_ALTITUDE),
                            g.push.slug_ratio_start, slug, impactor,
                        ),
                        chemistry_ceiling(
                            wave_speed_at_altitude(c.growth_wave_v_b * u.km / u.s, PUSH_ALTITUDE)
                            - puffsat_cycle_periapsis_speed(altitude=PUSH_ALTITUDE),
                            g.push.slug_ratio_end, slug, impactor,
                        ),
                    )
                    for c, g in zip(cycles, grown)
                ]  # fmt: skip
                rows.append(
                    {
                        "plate": plate_eta,
                        "departure": f"{pairing.name.split()[0]} {eta:g}",
                        "slug": label,
                        "k": f"{_mean([g.push.slug_ratio_start for g in grown]):.1f}"
                        f"->{_mean([g.push.slug_ratio_end for g in grown]):.1f}",
                        "eta_chem": f"{_mean([a for a, _ in ceilings]):.3f}"
                        f"->{_mean([b for _, b in ceilings]):.3f}",
                        "push": _mean(
                            [float((g.departing_stack / g.puffsats).to_value(u.one)) for g in grown]
                        ),
                        "doubling yr": summarize_chain(
                            periods, [g.growth for g in grown]
                        ).doubling_years,
                        "uncapped yr": summarize_chain(
                            periods, [g.growth for g in runs[None]]
                        ).doubling_years,
                    }
                )  # fmt: skip
    return tabulate(rows, headers="keys", floatfmt=".3g")


def plate_design_rows(
    cycles: Sequence[TwoWaveCycle], three_only: Sequence[TwoWaveCycle]
) -> List[Dict[str, object]]:
    """Each plate design behind the solved chambers and methalox (ADR 0041).

    Args:
        cycles: The flown chain.
        three_only: The three-synodic-only chain methalox flies.

    Returns:
        One row per plate and departure: doubling under the design's cap, and
        with k held to 10 instead.
    """
    periods = [c.period_years for c in cycles]
    rows: List[Dict[str, object]] = []
    for plate in PLATE_DESIGNS:
        for pairing in DEPARTURE_PAIRINGS:
            hydrogen = pairing is HYDROGEN_5500K
            eta = SOLVED_EFFICIENCY[pairing.name]
            runs = [
                chain_growth(
                    cycles, plate.efficiency, pairing, eta,
                    0.0 if hydrogen else pitch_ratio(METHANE_PITCH),
                    max_slug_ratio=cap,
                    impactor_bond_energy=plate.impactor_bond_energy,
                film_per_impulse=plate.film_per_impulse,
                    cryostat_fraction=HYDROGEN_CRYOSTAT if hydrogen else 0.0,
                    boil_off=(
                        hydrogen_boil_off(cycles, HYDROGEN_BOIL_OFF_PER_DAY)
                        if hydrogen else 0.0
                    ),
                )
                for cap in (plate.max_slug_ratio, PLATE_MAX_SLUG_RATIO)
            ]  # fmt: skip
            grown = runs[0]
            summary = summarize_chain(periods, [g.growth for g in grown])
            rows.append(
                {
                    "plate": plate.label,
                    "departure": f"{pairing.name.split()[0]} {eta:g}",
                    "k": f"{_mean([g.push.slug_ratio_start for g in grown]):.2f}"
                    f"->{_mean([g.push.slug_ratio_end for g in grown]):.1f}",
                    "push": _mean(
                        [float((g.departing_stack / g.puffsats).to_value(u.one)) for g in grown]
                    ),
                    "growth": _mean([g.growth for g in grown]),
                    "doubling yr": summary.doubling_years,
                    "k<=10 yr": summarize_chain(
                        periods, [g.growth for g in runs[1]]
                    ).doubling_years,
                    "10yr step": summary.ten_year_stepwise,
                }
            )  # fmt: skip
        three_periods = [c.period_years for c in three_only]
        flown = [
            best_methalox_cycle(
                c, plate.efficiency, max_slug_ratio=plate.max_slug_ratio,
                impactor_bond_energy=plate.impactor_bond_energy,
                film_per_impulse=plate.film_per_impulse,
            )
            for c in three_only
        ]  # fmt: skip
        summary = summarize_chain(three_periods, [f.growth for f in flown])
        rows.append(
            {
                "plate": plate.label,
                "departure": "methalox 380 s",
                "k": f"{_mean([f.push.slug_ratio_start for f in flown]):.2f}"
                f"->{_mean([f.push.slug_ratio_end for f in flown]):.1f}",
                "push": _mean(
                    [float((f.departing_stack / f.puffsats).to_value(u.one)) for f in flown]
                ),
                "growth": _mean([f.growth for f in flown]),
                "doubling yr": summary.doubling_years,
                "k<=10 yr": float("nan"),
                "10yr step": summary.ten_year_stepwise,
            }
        )  # fmt: skip
    return rows


def _plate_chamber_growth(
    cycles: Sequence[TwoWaveCycle],
    plate: PlateDesign,
    pairing: ChamberPairing,
    eta: float,
    cap: Optional[float],
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    efficiency: Optional[float] = None,
    impactor_bond_energy: Optional[u.Quantity] = None,
    cryostat_fraction: Optional[float] = None,
    boil_off_per_day: Optional[float] = None,
    pitch: u.Quantity = METHANE_PITCH,
) -> List[CycleGrowth]:
    """A chamber behind ``plate`` with the matrix's hold and pitch (ADR 0042)."""
    hydrogen = pairing is HYDROGEN_5500K
    if cryostat_fraction is None:
        cryostat_fraction = HYDROGEN_CRYOSTAT if hydrogen else 0.0
    if boil_off_per_day is None:
        boil_off_per_day = HYDROGEN_BOIL_OFF_PER_DAY if hydrogen else 0.0
    return chain_growth(
        cycles,
        plate.efficiency if efficiency is None else efficiency,
        pairing,
        eta,
        0.0 if hydrogen else pitch_ratio(pitch),
        max_slug_ratio=cap,
        slug=slug,
        impactor_bond_energy=(
            plate.impactor_bond_energy
            if impactor_bond_energy is None
            else impactor_bond_energy
        ),
        cryostat_fraction=cryostat_fraction if hydrogen else 0.0,
        boil_off=hydrogen_boil_off(cycles, boil_off_per_day) if hydrogen else 0.0,
        film_per_impulse=plate.film_per_impulse,
    )


def design_grid_rows(
    cycles: Sequence[TwoWaveCycle],
    three_only: Sequence[TwoWaveCycle],
    plate: PlateDesign,
) -> List[Dict[str, object]]:
    """The full matrix behind one plate design (ADR 0042).

    Every chamber share of its ceiling plus the solved chamber, and methalox,
    under the design's cap and with k <= 10 as the sensitivity.  The parent's
    ``tab:growth_ledger_doubling`` and ``tab:growth_ledger_ten_year``.

    Args:
        cycles: The flown chain on the 20-day orbit.
        three_only: The three-synodic-only chain methalox flies.
        plate: The design.

    Returns:
        One row per departure.
    """
    periods = [c.period_years for c in cycles]
    caps = (plate.max_slug_ratio, PLATE_MAX_SLUG_RATIO)
    rows: List[Dict[str, object]] = []
    for pairing in DEPARTURE_PAIRINGS:
        gas = pairing.name.split()[0]
        solved = SOLVED_EFFICIENCY[pairing.name]
        departures = [
            (absolute_efficiency(pairing, share), f"{gas} {share:.0%}")
            for share in CEILING_SHARES
        ] + [(solved, f"{gas} solved ({solved / pairing.chemistry_ceiling:.0%})")]
        for eta, label in sorted(departures):
            runs = [
                _plate_chamber_growth(cycles, plate, pairing, eta, cap) for cap in caps
            ]
            summary = summarize_chain(periods, [g.growth for g in runs[0]])
            rows.append(
                {
                    "departure": label,
                    "doubling yr": summary.doubling_years,
                    "k<=10 yr": summarize_chain(
                        periods, [g.growth for g in runs[1]]
                    ).doubling_years,
                    "annual": summary.annual_growth,
                    "10yr step": summary.ten_year_stepwise,
                    "payload t": _mean(
                        [
                            g.departure.delivered_net * g.departing_stack.to_value(u.t)
                            for g in runs[0]
                        ]
                    ),
                    "G_0": runs[0][0].growth,
                }
            )
    three_periods = [c.period_years for c in three_only]
    flown = [
        [
            best_methalox_cycle(
                c, plate.efficiency, max_slug_ratio=cap,
                impactor_bond_energy=plate.impactor_bond_energy,
                film_per_impulse=plate.film_per_impulse,
            )
            for c in three_only
        ]
        for cap in caps
    ]  # fmt: skip
    summary = summarize_chain(three_periods, [f.growth for f in flown[0]])
    rows.append(
        {
            "departure": "methalox 380 s",
            "doubling yr": summary.doubling_years,
            "k<=10 yr": summarize_chain(
                three_periods, [f.growth for f in flown[1]]
            ).doubling_years,
            "annual": summary.annual_growth,
            "10yr step": summary.ten_year_stepwise,
            "payload t": _mean(
                [f.delivered_net * f.departing_stack.to_value(u.t) for f in flown[0]]
            ),
            "G_0": flown[0][0].growth,
        }
    )
    return rows


#: The impact sim's unmixed water band on the cup (its P6), all-in like argon.
WATER_JET_EFFICIENCIES = (0.50, 0.53, 0.56)


def design_sensitivities(
    cycles: Sequence[TwoWaveCycle], plate: PlateDesign, split_days: float = 10.0
) -> str:
    """The ledger's smaller settings behind one plate design (ADR 0042).

    Hydrogen's hold, methane's pitch range, water at its all-in jet
    efficiency, and a shorter parking orbit flown consistently.

    Args:
        cycles: The flown chain on the 20-day orbit.
        plate: The design.
        split_days: The shorter orbit to compare.

    Returns:
        The tables.
    """
    periods = [c.period_years for c in cycles]

    def doubling(grown: Sequence[CycleGrowth], per: Sequence[float] = periods) -> float:
        return summarize_chain(per, [g.growth for g in grown]).doubling_years

    cap = plate.max_slug_ratio
    hold = []
    for cryostat in (0.0, 0.01, 0.03, 0.05):
        row: Dict[str, object] = {"cryostat": cryostat}
        for per_day in (0.0, 0.001, 0.003, 0.005):
            row[f"{100 * per_day:g}%/day"] = doubling(
                _plate_chamber_growth(
                    cycles, plate, HYDROGEN_5500K, SOLVED_EFFICIENCY[HYDROGEN_5500K.name],
                    cap, cryostat_fraction=cryostat, boil_off_per_day=per_day,
                )
            )  # fmt: skip
        hold.append(row)
    pitch = [
        {
            "pitch kg": p.to_value(u.kg),
            "CH4 solved yr": doubling(
                _plate_chamber_growth(
                    cycles, plate, METHANE_7000K, SOLVED_EFFICIENCY[METHANE_7000K.name],
                    cap, pitch=p,
                )
            ),
        }
        for p in METHANE_PITCH_RANGE
    ]  # fmt: skip
    water = []
    for eta_jet in WATER_JET_EFFICIENCIES:
        row = {"water eta_jet": eta_jet}
        for pairing in DEPARTURE_PAIRINGS:
            row[f"{pairing.name.split()[0]} solved yr"] = doubling(
                _plate_chamber_growth(
                    cycles, plate, pairing, SOLVED_EFFICIENCY[pairing.name], cap,
                    slug=WATER_SLUG, efficiency=eta_jet**2,
                    impactor_bond_energy=NO_BONDS,
                )
            )  # fmt: skip
        water.append(row)
    short = adaptive_two_wave_cycles(split_days=split_days)
    short_three = adaptive_two_wave_cycles(threshold_m_s=0.0, split_days=split_days)
    short_periods = [c.period_years for c in short]
    orbit = {
        f"{pairing.name.split()[0]} solved yr": doubling(
            _plate_chamber_growth(
                short, plate, pairing, SOLVED_EFFICIENCY[pairing.name], cap
            ),
            short_periods,
        )
        for pairing in DEPARTURE_PAIRINGS
    }
    orbit["methalox yr"] = summarize_chain(
        [c.period_years for c in short_three],
        [
            best_methalox_cycle(
                c, plate.efficiency, max_slug_ratio=cap,
                impactor_bond_energy=plate.impactor_bond_energy,
                film_per_impulse=plate.film_per_impulse,
            ).growth
            for c in short_three
        ],
    ).doubling_years  # fmt: skip
    return (
        "Solved hydrogen: doubling (yr) across cryostat mass and boil-off:\n"
        + str(tabulate(hold, headers="keys", floatfmt=".3f"))
        + "\n\nMethane's pitch per pulse:\n"
        + str(tabulate(pitch, headers="keys", floatfmt=".3f"))
        + "\n\nWater on the plate, all-in (the impact sim's unmixed band, P6), "
        + "water's tank:\n"
        + str(tabulate(water, headers="keys", floatfmt=".3f"))
        + f"\n\nA {split_days:g}-day parking orbit flown consistently:\n"
        + str(tabulate([orbit], headers="keys", floatfmt=".3f"))
    )


def film_rows(
    cycles: Sequence[TwoWaveCycle],
    three_only: Sequence[TwoWaveCycle],
    plate: PlateDesign = SPRAY_CUP,
) -> List[Dict[str, object]]:
    """Doubling with the plate's film carried as launched mass (ADR 0043).

    The parent's S12: each end of the spray cup's vapor-shielded and unshielded
    film bands, against the film left to the cost book.  The film is part of
    the 1500 t unit and burns off pulse by pulse, so it is paid for in what
    the push delivers.

    Args:
        cycles: The flown chain on the 20-day orbit.
        three_only: The three-synodic-only chain methalox flies.
        plate: The design; its own film is replaced by each band's.

    Returns:
        One row per film per pulse.
    """
    periods = [c.period_years for c in cycles]
    three_periods = [c.period_years for c in three_only]
    films = [("not carried", 0.0)] + [
        (band, kg) for band, ends in SPRAY_CUP_FILM_PER_PULSE.items() for kg in ends
    ]
    rows: List[Dict[str, object]] = []
    for band, kg in films:
        flown = replace(plate, film_per_pulse_kg=kg)
        row: Dict[str, object] = {"film": band, "kg/pulse": kg}
        for pairing in DEPARTURE_PAIRINGS:
            grown = _plate_chamber_growth(
                cycles, flown, pairing, SOLVED_EFFICIENCY[pairing.name],
                flown.max_slug_ratio,
            )  # fmt: skip
            gas = pairing.name.split()[0]
            if pairing is HYDROGEN_5500K:
                row["film t/push"] = _mean(
                    [g.push.film_fraction * LAUNCH_UNIT.to_value(u.t) for g in grown]
                )
            row[f"{gas} solved yr"] = summarize_chain(
                periods, [g.growth for g in grown]
            ).doubling_years
        row["methalox yr"] = summarize_chain(
            three_periods,
            [
                best_methalox_cycle(
                    c, flown.efficiency, max_slug_ratio=flown.max_slug_ratio,
                    impactor_bond_energy=flown.impactor_bond_energy,
                    film_per_impulse=flown.film_per_impulse,
                ).growth
                for c in three_only
            ],
        ).doubling_years  # fmt: skip
        rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the growth ledger's scenario matrix over the flown chain and print it.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--slugs",
        action="store_true",
        help="compare water and argon on the plate instead of the full matrix",
    )
    parser.add_argument(
        "--split-days",
        type=float,
        default=DEFAULT_PARKING_DAYS,
        help="split gap, which is also the parking orbit (default "
        f"{DEFAULT_PARKING_DAYS:g})",
    )
    parser.add_argument(
        "--designs",
        action="store_true",
        help="the impact-sim's plate designs (ADR 0041) instead of the full matrix",
    )
    parser.add_argument(
        "--designs-grid",
        choices=sorted(PLATE_DESIGNS_BY_NAME),
        help="the full matrix and sensitivities behind one plate design (ADR 0042)",
    )
    parser.add_argument(
        "--film",
        action="store_true",
        help="the spray cup's film carried as launched mass (ADR 0043)",
    )
    args = parser.parse_args(argv)
    cycles = adaptive_two_wave_cycles(split_days=args.split_days)
    if args.film:
        three = adaptive_two_wave_cycles(threshold_m_s=0.0, split_days=args.split_days)
        print(
            f"Spray cup ({SPRAY_CUP.label}, k <= {SPRAY_CUP.max_slug_ratio:g}) with "
            "its film carried as launched mass (ADR 0043, parent S12): film per "
            f"{PULSE_IMPULSE.to_value(u.MN * u.s):g} MN s pulse, burned pulse by "
            f"pulse from the {LAUNCH_UNIT:g} unit.  {args.split_days:g}-day orbit; "
            "solved chambers as in --designs-grid; methalox on "
            f"{len(three)} three-synodic cycles."
        )
        print(tabulate(film_rows(cycles, three), headers="keys", floatfmt=".4g"))
        return
    if args.designs_grid:
        plate = PLATE_DESIGNS_BY_NAME[args.designs_grid]
        three = adaptive_two_wave_cycles(threshold_m_s=0.0, split_days=args.split_days)
        print(
            f"Plate {plate.label} (ADR 0041/0042) over the flown chain "
            f"({len(cycles)} cycles, {args.split_days:g}-day orbit; methalox on "
            f"{len(three)} three-synodic cycles).  k capped at {plate.max_slug_ratio:g}; "
            "'k<=10 yr' lifts the cap to 10.  Chambers gated, methane at "
            f"{METHANE_PITCH.to_value(u.kg):g} kg pitch, hydrogen with cryostats and "
            "boil-off.  Shares are of each chamber's chemistry ceiling."
        )
        print(
            tabulate(
                design_grid_rows(cycles, three, plate), headers="keys", floatfmt=".4g"
            )
        )
        print()
        print(design_sensitivities(cycles, plate))
        return
    if args.designs:
        three = adaptive_two_wave_cycles(threshold_m_s=0.0, split_days=args.split_days)
        print(
            f"Plate designs (ADR 0041) over the flown chain ({len(cycles)} cycles; "
            f"methalox on {len(three)} three-synodic cycles).  Argon spray; the "
            "impact-sim's eta_jet is all-in, so its PuffSat bonds are not charged "
            f"again.  k capped at the paper's {PAPER_SLUG_RATIO:g} (ADR 0033's plate "
            "at 10); 'k<=10 yr' lifts the cap to 10.  Solved chambers, gated, "
            "methane at 5.6 kg pitch, hydrogen with cryostats and boil-off."
        )
        print(
            tabulate(plate_design_rows(cycles, three), headers="keys", floatfmt=".3g")
        )
        return
    if args.slugs:
        print(
            f"Plate slug comparison over the flown chain ({len(cycles)} cycles).  "
            "Plate efficiency is net of chemistry (eta_geom^2); eta_chem is charged "
            "per pulse from eq:eta_chem, first -> last pulse.  k held to "
            f"{PLATE_MAX_SLUG_RATIO:g}; 'uncapped yr' frees it.  Assumed: argon's "
            "ionisation recombines while pressure-coupled (99% by 10 400-15 700 K at "
            "1 MPa-1 GPa); water keeps its frozen toll (pessimistic on a plate).  "
            "Argon gives up water's role as the plate's heat sponge (unpriced)."
        )
        print(_slug_comparison(cycles))
        return
    three_only = adaptive_two_wave_cycles(threshold_m_s=0.0, split_days=args.split_days)
    print(
        f"Growth ledger: {LAUNCH_UNIT:g} launch unit, plate push at 400 km, departure "
        f"from 600 km after the methalox apoapsis reversal (ADR 0009) on each "
        f"cycle's {cycles[0].split_days:g}-day parking orbit, over "
        f"the flown chain ({len(cycles)} cycles)."
    )
    print(
        "Plate: argon on ice PuffSats, efficiency net of its chemistry toll.  "
        "Chambers: efficiency as a share of the chemistry ceiling (H2 0.978, "
        "CH4 0.669; eta = absolute).  "
        "k = plate loading, first -> last pulse (per-cycle optimal schedule), "
        f"held to {PLATE_MAX_SLUG_RATIO:g}; 'uncapped yr' frees it (sensitivity).  "
        "push = departing stack per growth PuffSat; net = stack delivered net of "
        "tanks and chambers.  Chambers gated; methane at "
        f"{METHANE_PITCH.to_value(u.kg):g} kg pitch/pulse; "
        f"hydrogen with {HYDROGEN_CRYOSTAT:g} cryostats and "
        f"{100 * HYDROGEN_BOIL_OFF_PER_DAY:g}%/day boil-off over the hold.  "
        "Efficiency 1.0 is a theoretical ceiling."
    )
    print(_report(cycles, three_only))


if __name__ == "__main__":
    main()
