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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from astropy import units as u
from boinor.bodies import Earth
from scipy.optimize import minimize_scalar
from tabulate import tabulate

from src.astro_constants import LEO_ALTITUDE
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
    PLUG_RATIO,
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
    burn_at_200_km: u.Quantity, altitude: u.Quantity
) -> Departure:
    """Move a chain departure burn from the 200 km periapsis to ``altitude``.

    The chain's burn raises the cycle orbit's 200 km periapsis speed to the
    speed that leaves with some excess speed ``v_inf``.  From a higher
    periapsis the same ``v_inf`` needs ``sqrt(v_inf^2 + 2 mu / r)``, starting
    from that altitude's own cycle-orbit periapsis speed.  A higher burn gets
    less from the Oberth effect, so it costs more.

    Args:
        burn_at_200_km: The chain's ``departure_burn``.
        altitude: Periapsis altitude to depart from.

    Returns:
        The burn's start speed and size at ``altitude``.
    """
    mu = _mu()
    start_200 = float(puffsat_cycle_periapsis_speed().to_value(u.km / u.s))
    final_200 = start_200 + float(burn_at_200_km.to_value(u.km / u.s))
    excess_sq = final_200**2 - 2.0 * mu / _radius(LEO_ALTITUDE)
    start = puffsat_cycle_periapsis_speed(altitude=altitude).to(u.km / u.s)
    final = np.sqrt(excess_sq + 2.0 * mu / _radius(altitude)) * u.km / u.s
    return Departure(start_speed=start, burn=final - start)


#: The fixed launch unit, as it reaches the 400 km intercept: plate, chambers,
#: spray water and its tank, departure propellant and tanks, and payload.
LAUNCH_UNIT = 1500.0 * u.t
#: The pusher plate, about a tenth of the craft at a few hertz and a few metres
#: of stroke (``sec:plate_reuse``).  Dropped before the departure, probably to be
#: reused, but the ledger takes no credit for reuse.
PLATE_MASS = 150.0 * u.t
#: The plate sprays argon onto ice PuffSats (user, 2026-09-30): argon has no
#: bonds to strand, and it beats water by 8-11% in doubling (make plate-slug).
DEFAULT_PLATE_SLUG = ARGON_SLUG
#: Altitudes of the two collisions (``sec:jovian_meeting_altitudes``).
PUSH_ALTITUDE = 400.0 * u.km
DEPARTURE_ALTITUDE = 600.0 * u.km
_HOLD_ITERATIONS = 100
_HOLD_TOLERANCE_T = 1.0e-9
#: Periapsis raise from 400 km to 600 km at the 613 000 km apoapsis, in methalox.
PERIAPSIS_RAISE = 1.7 * u.m / u.s
#: The apoapsis reversal every design pays (ADR 0009): the push leaves the craft
#: moving along the wave's axis, and the departure must be prograde, so the
#: ellipse is reversed where the craft crawls.  Sized on the 20-day orbit the
#: burns are priced on (234 m/s).  The chain's own nozzle ledger sizes it on the
#: 10-day split orbit instead (372 m/s), a known inconsistency in the chain.
APOAPSIS_REVERSAL = apoapsis_reversal_dv().to(u.m / u.s)


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
) -> Tuple[OptimalPlatePush, u.Quantity]:
    """Push the launch unit at 400 km and park it, ready to depart from 600 km.

    The wave pushes it from rest to the 400 km cycle-orbit speed.  At the
    613 000 km apoapsis it raises periapsis to 600 km and reverses the ellipse,
    both in methalox, and it drops the plate and the slug's empty drop tank.

    Returns:
        The push, and the mass left to depart (before any cryostats or boil-off).
    """
    unit = LAUNCH_UNIT.to(u.t)
    push = optimal_plate_push(
        wave_speed_at_altitude(wave_speed_at_200_km * u.km / u.s, PUSH_ALTITUDE),
        puffsat_cycle_periapsis_speed(altitude=PUSH_ALTITUDE),
        plate_efficiency,
        initial_water_price,
        max_slug_ratio=max_slug_ratio,
        slug=slug,
        impactor_bond_energy=impactor_bond_energy,
    )
    methalox = (PERIAPSIS_RAISE + APOAPSIS_REVERSAL).to_value(u.km / u.s)
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
    loss_model: LossModel = fixed_direction_loss,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    cryostat_fraction: float = 0.0,
    boil_off: float = 0.0,
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
        loss_model: As in :func:`src.chamber_departure.price_departure`.
        max_slug_ratio: Cap on the plate's water loading; None leaves it free.
        slug: What the plate sprays; argon by default.
        impactor_bond_energy: The PuffSat's bond energy per kilogram.
        cryostat_fraction: Cryostat mass per kilogram of gas launched; zero
            for methane, which the parent holds passively.
        boil_off: Share of the launched gas lost over the parking-orbit hold.

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
    )
    departure_burn = departure_at_altitude(
        cycle.departure_burn * u.km / u.s, DEPARTURE_ALTITUDE
    )

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
            loss_model=loss_model,
        )

    # The gas launched is what the burn spends over (1 - boil-off), and its
    # cryostats and boil-off never depart, so the departing stack is solved
    # self-consistently with the burn it has to fly.
    def launched_gas(stack: u.Quantity, departure: DepartureLedger) -> u.Quantity:
        spent = 1.0 - departure.delivered_fraction
        burned = spent - (PLUG_RATIO + pitch_ratio) * departure.rod_mass_fraction
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
    loss_model: LossModel = fixed_direction_loss,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    cryostat_fraction: float = 0.0,
    boil_off: float = 0.0,
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
        loss_model: As in :func:`src.chamber_departure.price_departure`.
        max_slug_ratio: As in :func:`price_cycle_growth`.
        slug: As in :func:`price_cycle_growth`.
        impactor_bond_energy: As in :func:`price_cycle_growth`.
        cryostat_fraction: As in :func:`price_cycle_growth`.
        boil_off: As in :func:`price_cycle_growth`.

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
    loss_model: LossModel = fixed_direction_loss,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = DEFAULT_PLATE_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    cryostat_fraction: float = 0.0,
    boil_off: float = 0.0,
) -> List[CycleGrowth]:
    """Carry the launch unit through every flown cycle, each on its best schedule.

    Args:
        cycles: Flown cycles from :func:`src.two_wave_growth.adaptive_two_wave_cycles`.
        plate_efficiency: The plate's energy efficiency ``eta``.
        pairing: The departure chamber.
        departure_efficiency: The chamber's energy efficiency.
        pitch_ratio: As in :func:`src.chamber_isp.effective_isp`.
        loss_model: As in :func:`src.chamber_departure.price_departure`.
        max_slug_ratio: As in :func:`price_cycle_growth`.
        slug: As in :func:`price_cycle_growth`.
        impactor_bond_energy: As in :func:`price_cycle_growth`.
        cryostat_fraction: As in :func:`price_cycle_growth`.
        boil_off: As in :func:`price_cycle_growth`.

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
    loss_model: LossModel = steered_loss,
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
        loss_model: As in :func:`methalox_departure`.

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
    )
    burn = departure_at_altitude(cycle.departure_burn * u.km / u.s, DEPARTURE_ALTITUDE)
    best: Optional[MethaloxDeparture] = None
    falls = 0
    for engines in range(1, _MAX_ENGINES + 1):
        try:
            flown = methalox_departure(stack, burn.burn, engines, loss_model)
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
) -> MethaloxCycle:
    """:func:`price_methalox_cycle` on the water schedule that grows it most.

    Args:
        cycle: The flown cycle.
        plate_efficiency: The plate's energy efficiency ``eta``.
        max_slug_ratio: As in :func:`price_cycle_growth`.
        slug: As in :func:`price_cycle_growth`.
        impactor_bond_energy: As in :func:`price_cycle_growth`.

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
        )
    )


#: The ledger's efficiencies.  The plate's are net of its chemistry toll,
#: charged pulse by pulse; the chambers' are shares of their chemistry ceiling
#: (:func:`src.chamber_isp.absolute_efficiency`), with each solved chamber added
#: at its own share.  1.0 is a theoretical ceiling on both legs.
PLATE_EFFICIENCIES = (0.50, 0.70, 1.00)
CEILING_SHARES = (0.50, 0.70, 0.90, 1.00)
SOLVED_EFFICIENCY = {HYDROGEN_5500K.name: 0.858, METHANE_7000K.name: 0.538}
DEPARTURE_PAIRINGS = (HYDROGEN_5500K, METHANE_7000K)
#: Methane's pitch per pulse, at the pessimistic end of 1.4-5.6 kg.
METHANE_PITCH = 5.6 * u.kg


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


#: Hydrogen's cryostat mass and boil-off over the hold, per kilogram launched.
#: No flight-scale source exists; passive-MLI large tanks lose 0.1-0.5%/day, so
#: 1-5% over the hold, and cryostat mass is swept over the same 1-5%.  The
#: matrix carries the middle of both; :func:`_hold_sweep` shows the range.
HYDROGEN_CRYOSTAT = 0.03
HYDROGEN_BOIL_OFF = 0.03


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
    pitch = 0.0 if hydrogen else float((METHANE_PITCH / ROD_MASS).to_value(u.one))
    cryostat = HYDROGEN_CRYOSTAT if hydrogen else 0.0
    boil_off = HYDROGEN_BOIL_OFF if hydrogen else 0.0
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
        for boil in (0.0, 0.01, 0.03, 0.05):
            grown = chain_growth(
                cycles, 0.7, HYDROGEN_5500K, 0.858,
                cryostat_fraction=cryostat, boil_off=boil,
            )  # fmt: skip
            row[f"boil-off {boil:g}"] = summarize_chain(
                periods, [g.growth for g in grown]
            ).doubling_years
        rows.append(row)
    return tabulate(rows, headers="keys", floatfmt=".3g")


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
        + "\n\nSolved hydrogen (0.858) behind a 0.7 plate: doubling (yr) across cryostat mass "
        + "and boil-off, per kilogram launched:\n"
        + _hold_sweep(cycles)
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
            pitch = (
                0.0 if hydrogen else float((METHANE_PITCH / ROD_MASS).to_value(u.one))
            )
            held = HYDROGEN_CRYOSTAT if hydrogen else 0.0
            boil = HYDROGEN_BOIL_OFF if hydrogen else 0.0
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
    args = parser.parse_args(argv)
    cycles = adaptive_two_wave_cycles()
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
    three_only = adaptive_two_wave_cycles(threshold_m_s=0.0)
    print(
        f"Growth ledger: {LAUNCH_UNIT:g} launch unit, water plate at 400 km, departure "
        f"from 600 km after a {APOAPSIS_REVERSAL:.0f} methalox apoapsis reversal, over "
        f"the flown chain ({len(cycles)} cycles)."
    )
    print(
        "Plate: argon on ice PuffSats, efficiency net of its chemistry toll.  "
        "Chambers: efficiency as a share of the chemistry ceiling (H2 0.978, "
        "CH4 0.669; eta = absolute).  "
        "k = plate loading, first -> last pulse (per-cycle optimal schedule), "
        f"held to {PLATE_MAX_SLUG_RATIO:g}; 'uncapped yr' frees it (sensitivity).  "
        "push = departing stack per growth PuffSat; net = stack delivered net of "
        "tanks and chambers.  Chambers gated; methane at 5.6 kg pitch/pulse; "
        f"hydrogen with {HYDROGEN_CRYOSTAT:g} cryostats and {HYDROGEN_BOIL_OFF:g} "
        "boil-off.  Efficiency 1.0 is a theoretical ceiling."
    )
    print(_report(cycles, three_only))


if __name__ == "__main__":
    main()
