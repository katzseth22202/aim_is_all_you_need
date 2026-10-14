"""The growth ledger's chain, reduced to what the growth cost model prices (ADR 0037).

:mod:`src.growth_cost` is pure arithmetic over :class:`~src.growth_cost.DesignInputs`.
This module builds those records from the flown chains: per return, what one
1500 t launch unit lofted for growth consumes, sprays, burns, expends and sends
onward (:func:`src.seed_cost.design_chain`'s ledgers), and the delivery
schedules a returning batch can push cargo to L1 on
(:func:`src.harvest.delivery_front`).  It takes minutes; results are cached.

**Indexing** follows :func:`src.harvest.chain_returns`: the batch at return
``n`` is ``prod(G[:n])`` seeds, and a unit lofted at return ``n`` is pushed by
that return's waves and grows by ``G_n``.  That matches how
:func:`src.growth_ledger.price_cycle_growth` defines ``G_n``.  (The ledger
departs that unit with cycle ``n``'s own outbound burn, while it actually
leaves on cycle ``n + 1``; ADR 0037 records this as open.)

**Rods.**  The mass a unit sends onward comes back as the next return's growth
PuffSats and departure rods, so it is split in the ratio the next cycle
consumes them, wrapping to cycle 0 as the chain repeats.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from astropy import units as u

from src.chamber_departure import chamber_unit_mass
from src.growth_cost import (
    ROD_MASS,
    CycleHardware,
    DeliveryOption,
    Departure,
    DesignInputs,
    delayed,
)
from src.growth_ledger import (
    ADR_0033_PLATE,
    LAUNCH_UNIT,
    METHALOX_TANK_FRACTION,
    RAPTOR3_MASS,
    CycleGrowth,
    MethaloxCycle,
    PlateDesign,
    hydrogen_boil_off,
)
from src.harvest import chain_returns, delivery_front, harvest_wave_speed
from src.seed_cost import (
    Design,
    DesignChain,
    chains,
    design_chain,
    seed_excess_speed,
    seed_price_range,
)
from src.two_wave_growth import VE_METHALOX, TwoWaveCycle
from src.water_plate import PlatePush


@dataclass(frozen=True)
class SeedFlight:
    """How the seed reaches Jupiter, relative to the direct route (ADR 0039).

    The route scales the seed's price per kilogram on each ship and delays the
    whole program, exactly as :func:`src.growth_cost.route_break_even` does.

    Attributes:
        label: The route.
        cheap_ratio: Its dollars per seed kilogram over direct's, 40 t ship.
        dear_ratio: The same on the 60 t ship.
        delay_years: How much later its first return comes.
    """

    label: str
    cheap_ratio: float
    dear_ratio: float
    delay_years: float


DIRECT = SeedFlight("direct (EJ)", 1.0, 1.0, 0.0)
#: ADR 0039's Earth loop with methalox, ADR 0057's headline.  Its ratios were
#: searched at a 51.134 km/s return floor with the hull dropped after
#: departure, both of which flatter it (ADR 0059).
EEJ = SeedFlight("EEJ, methalox (ADR 0039)", 0.6991, 0.6112, 2.184)
#: ADR 0059's two-year Lucy-style loop: 86.03 / 128.65 t on the 40 t ship and
#: 66.03 / 110.56 t on the 60 t ship (direct / loop), ship carried through
#: every burn, return at the chain's 59.29 km/s.  The loop returns 2.166 and
#: 2.180 yr after direct; the larger is charged to both.
LOOP_2YR = SeedFlight("2-yr Earth loop, DSM at 370 s (ADR 0059)", 0.6687, 0.5972, 2.180)
#: The route every report prices the seed on: direct (ADR 0060).  The loop
#: cuts the cheap-seed break-even only 3-7% for two years' delay, a cryogenic
#: coast and a powered flyby, so it is quoted as a sensitivity.
HEADLINE_SEED = DIRECT


def _kg(mass: u.Quantity) -> float:
    return float(mass.to_value(u.kg))


def _survives(burn_km_s: float) -> float:
    """Share of a wave left after a methalox correction of ``burn_km_s``."""
    return float(np.exp(-burn_km_s / VE_METHALOX))


def _consumption(ledger: CycleGrowth, cycle: TwoWaveCycle) -> Tuple[float, float]:
    """(growth PuffSats, rods) a chamber unit consumes, as they leave Jupiter."""
    rods = ledger.departure.rod_mass_fraction * _kg(ledger.departing_stack)
    return (
        _kg(ledger.puffsats) / _survives(cycle.growth_wave_burn),
        rods / _survives(cycle.nozzle_wave_dsm),
    )


def _film(push: PlatePush, plate: PlateDesign) -> Optional[float]:
    """The film a unit's push burns, if the plate carries it (ADR 0043)."""
    return _kg(push.film_fraction * LAUNCH_UNIT) if plate.carries_film else None


def _methalox_cycles(
    chain: DesignChain, plate: PlateDesign
) -> Tuple[CycleHardware, ...]:
    hardware = []
    for ledger, cycle in zip(chain.ledgers, chain.cycles):
        assert isinstance(ledger, MethaloxCycle)
        stack = _kg(ledger.departing_stack)
        onward = ledger.delivered_net * stack
        engines = _kg(ledger.engines * RAPTOR3_MASS)
        # delivered_net = 1 - spent (1 + tank fraction) - engines' share.
        propellant = (stack - onward - engines) / (1.0 + METHALOX_TANK_FRACTION)
        hardware.append(
            CycleHardware(
                consumed=_kg(ledger.puffsats) / _survives(cycle.nozzle_wave_dsm),
                growth=ledger.growth,
                onward_puffsats=onward,
                onward_rods=0.0,
                plate_puffsats=_kg(ledger.puffsats),
                argon=ledger.push.slug_fraction * _kg(LAUNCH_UNIT),
                departure_units=ledger.engines,
                tanks=METHALOX_TANK_FRACTION * propellant,
                gas=propellant,
                film=_film(ledger.push, plate),
            )
        )
    return tuple(hardware)


def _chamber_cycles(
    chain: DesignChain, plate: PlateDesign
) -> Tuple[CycleHardware, ...]:
    pairing = chain.design.pairing
    assert pairing is not None
    storage = pairing.storage
    pitch_ratio = chain.design.pitch_ratio
    chamber_mass = _kg(chamber_unit_mass(pairing))
    boil_off = (
        0.0
        if storage is None
        else hydrogen_boil_off(chain.cycles, storage.boil_off_per_day)
    )
    ledgers = chain.ledgers
    consumption = []
    for ledger, cycle in zip(ledgers, chain.cycles):
        assert isinstance(ledger, CycleGrowth)
        consumption.append(_consumption(ledger, cycle))
    hardware = []
    for n, ledger in enumerate(ledgers):
        assert isinstance(ledger, CycleGrowth)
        departure = ledger.departure
        stack = _kg(ledger.departing_stack)
        rods = departure.rod_mass_fraction * stack
        burned = (1.0 - departure.delivered_fraction) * stack - (
            pairing.plug_ratio + pitch_ratio
        ) * rods
        onward = departure.delivered_net * stack
        puffsats_next, rods_next = consumption[(n + 1) % len(ledgers)]
        rod_share = rods_next / (puffsats_next + rods_next)
        hardware.append(
            CycleHardware(
                consumed=sum(consumption[n]),
                growth=ledger.growth,
                onward_puffsats=(1.0 - rod_share) * onward,
                onward_rods=rod_share * onward,
                plate_puffsats=_kg(ledger.puffsats),
                argon=ledger.push.slug_fraction * _kg(LAUNCH_UNIT),
                departure_units=departure.chambers,
                tanks=departure.tank_share * stack,
                gas=burned / (1.0 - boil_off),
                cryostats=_kg(ledger.cryostats),
                plugs=pairing.plug_ratio * rods,
                pitch=pitch_ratio * rods,
                pulses=departure.pulses,
                film=_film(ledger.push, plate),
                chamber_mass=chamber_mass,
            )
        )
    return tuple(hardware)


def _deliveries(
    chain: DesignChain, plate: PlateDesign
) -> Tuple[Tuple[DeliveryOption, ...], ...]:
    return tuple(
        tuple(
            DeliveryOption(
                _kg(d.puffsats), _kg(d.cargo), _kg(d.slug), _film(d.push, plate)
            )
            for d in delivery_front(
                harvest_wave_speed(cycle),
                plate.efficiency,
                plate.impactor_bond_energy,
                plate.max_slug_ratio,
                plate.film_per_impulse,
            )
        )
        for cycle in chain.cycles
    )


def design_inputs(
    design: Design,
    plate: PlateDesign = ADR_0033_PLATE,
    seed: SeedFlight = HEADLINE_SEED,
) -> DesignInputs:
    """One design's chain, priced per seed of one launch unit.

    Args:
        design: The departure design.
        plate: The plate every growth unit and delivery flies (ADR 0041).
        seed: The seed's route; the program slides by its delay (ADR 0057).
            Pair it with :func:`seed_prices` on the same route.

    Returns:
        The inputs.
    """
    return delayed(_direct_inputs(design, plate), seed.delay_years)


@lru_cache(maxsize=32)
def _direct_inputs(design: Design, plate: PlateDesign) -> DesignInputs:
    """:func:`design_inputs` on the direct seed's schedule (cached)."""
    chain = design_chain(design, plate)
    returns = chain_returns(chain)
    if design.pairing is None:
        departure = Departure.METHALOX
        cycles = _methalox_cycles(chain, plate)
        seed_puffsats, seed_rods = _kg(chain.seed), 0.0
    else:
        departure = Departure.HYDROGEN if design.pairing.hydrogen else Departure.METHANE
        cycles = _chamber_cycles(chain, plate)
        first = chain.ledgers[0]
        assert isinstance(first, CycleGrowth)
        seed_puffsats, seed_rods = _consumption(first, chain.cycles[0])
    return DesignInputs(
        label=design.label,
        departure=departure,
        launch_unit=_kg(LAUNCH_UNIT),
        seed_puffsats=seed_puffsats,
        seed_rods=seed_rods,
        cycles=cycles,
        times=returns.times,
        arrivals=returns.arrivals,
        harvest=returns.harvest,
        deliveries=_deliveries(chain, plate),
        rod_mass=(
            _kg(design.pairing.rod_mass) if design.pairing is not None else ROD_MASS
        ),
    )


def seed_prices(seed: SeedFlight = HEADLINE_SEED) -> Tuple[float, float]:
    """The seed's cheap and dear ends, dollars per kilogram (ADR 0036, 0057).

    Args:
        seed: The seed's route; pair it with :func:`design_inputs`'s.

    Returns:
        Direct, ($2172, $9293): the 40 t stripped ship at a $5M hull and
        $150/kg flights, and the 60 t ship at a $20M hull and Morgan Stanley's
        $500/kg.  Each is scaled by the route's ratio on that ship.
    """
    flown, _ = chains()
    prices = seed_price_range(seed_excess_speed(flown[0]))
    return prices.low * seed.cheap_ratio, prices.high * seed.dear_ratio
