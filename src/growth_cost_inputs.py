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

from functools import lru_cache
from typing import Tuple

import numpy as np
from astropy import units as u

from src.chamber_isp import HYDROGEN_5500K, PLUG_RATIO, ROD_MASS
from src.growth_cost import CycleHardware, DeliveryOption, Departure, DesignInputs
from src.growth_ledger import (
    HYDROGEN_BOIL_OFF_PER_DAY,
    LAUNCH_UNIT,
    METHALOX_TANK_FRACTION,
    METHANE_PITCH,
    RAPTOR3_MASS,
    CycleGrowth,
    MethaloxCycle,
    hydrogen_boil_off,
)
from src.harvest import chain_returns, delivery_front, harvest_wave_speed
from src.seed_cost import (
    SEED_PLATE_EFFICIENCY,
    Design,
    DesignChain,
    chains,
    design_chain,
    seed_excess_speed,
    seed_price_range,
)
from src.two_wave_growth import VE_METHALOX, TwoWaveCycle


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


def _methalox_cycles(chain: DesignChain) -> Tuple[CycleHardware, ...]:
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
            )
        )
    return tuple(hardware)


def _chamber_cycles(chain: DesignChain) -> Tuple[CycleHardware, ...]:
    hydrogen = chain.design.pairing is HYDROGEN_5500K
    pitch_ratio = 0.0 if hydrogen else float((METHANE_PITCH / ROD_MASS).to_value(u.one))
    boil_off = (
        hydrogen_boil_off(chain.cycles, HYDROGEN_BOIL_OFF_PER_DAY) if hydrogen else 0.0
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
            PLUG_RATIO + pitch_ratio
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
                plugs=PLUG_RATIO * rods,
                pitch=pitch_ratio * rods,
                pulses=departure.pulses,
            )
        )
    return tuple(hardware)


def _deliveries(chain: DesignChain) -> Tuple[Tuple[DeliveryOption, ...], ...]:
    return tuple(
        tuple(
            DeliveryOption(_kg(d.puffsats), _kg(d.cargo), _kg(d.slug))
            for d in delivery_front(harvest_wave_speed(cycle), SEED_PLATE_EFFICIENCY)
        )
        for cycle in chain.cycles
    )


@lru_cache(maxsize=8)
def design_inputs(design: Design) -> DesignInputs:
    """One design's chain, priced per seed of one launch unit (cached).

    Args:
        design: The departure design.

    Returns:
        The inputs.
    """
    chain = design_chain(design)
    returns = chain_returns(chain)
    if design.pairing is None:
        departure = Departure.METHALOX
        cycles = _methalox_cycles(chain)
        seed_puffsats, seed_rods = _kg(chain.seed), 0.0
    else:
        departure = (
            Departure.HYDROGEN
            if design.pairing is HYDROGEN_5500K
            else Departure.METHANE
        )
        cycles = _chamber_cycles(chain)
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
        deliveries=_deliveries(chain),
    )


def seed_prices() -> Tuple[float, float]:
    """The seed's cheap and dear ends, dollars per kilogram (ADR 0036).

    Returns:
        ($337, $9293): the 40 t stripped ship at a $5M hull and Musk's flight
        price, and the 60 t ship at a $20M hull and Morgan Stanley's.
    """
    flown, _ = chains()
    prices = seed_price_range(seed_excess_speed(flown[0]))
    return prices.low, prices.high
