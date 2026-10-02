"""The walled chamber's departure burn with its hardware charged.

:mod:`src.chamber_isp` prices one pulse and integrates the burn.  This module
charges what rides that burn and is not payload: the gas tanks and the chambers
themselves, wall plus nozzle extension.  The tanks cost ``tank_fraction`` of
the gas charge only; the plug and the pitch ride without tanks, as in the
parent's ``tab:wall_pairing_doubling``.

The departure wave's dry mass -- chambers, nozzle extensions and tanks -- is
expended every cycle (user, 2026-09-30): it rides the burn, leaves with the
stack on its escape trajectory, and a fresh set flies in the next launch unit.
It is charged here as mass that is never payload.  Retrieving the chambers for
reuse (``sec:steel_chamber_service``: PuffSats brake them back into the parking
orbit) would free that mass at the cost of the braking PuffSats, and is not
modelled.  The parent's ``tab:h2_breakdown`` charged tanks only, taking the
chamber back for free.

The departure wave arrives along one fixed line, so the chamber cannot steer
the burn along the velocity as it turns.  It pays the fixed-direction loss of
:mod:`src.finite_burn_loss`, integrated for the cycle's own burn and burn time.
The burn time is the pulse count over the chambers' combined 4 Hz, and the loss
adds to the burn, which adds pulses, so the priced burn is solved as a fixed
point.  The parent instead scales one figure, 27 m/s over 320 s, as the burn
time squared (:func:`square_law_loss`, kept for comparison).  That law holds to
about 640 s on a 5.4 km/s burn, but it ignores the burn's size, which matters
as much: a 7 km/s two-synodic burn loses about 30% more at any length.
"""

import argparse
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from astropy import units as u
from tabulate import tabulate

from src.chamber_isp import (
    GATE_THRUST_COST,
    HYDROGEN_5500K,
    METHANE_7000K,
    PLUG_RATIO,
    ROD_MASS,
    ChamberBurn,
    ChamberPairing,
    absolute_efficiency,
    chamber_departure_burn,
)
from src.finite_burn_loss import fixed_direction_loss
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.two_wave_growth import TwoWaveCycle, adaptive_two_wave_cycles

#: Pulse rate of one chamber (``fig:methane_wall_detail``: a 250 ms cycle).
PULSE_RATE = 4.0 / u.s
#: The parent's one fixed-direction figure, and the burn length it was
#: integrated for (``sec:jovian_meeting_altitudes``), for :func:`square_law_loss`.
FIXED_DIRECTION_LOSS = 27.0 * u.m / u.s
FIXED_DIRECTION_BURN_TIME = 320.0 * u.s
_LOSS_TOLERANCE_M_S = 1.0e-9
_LOSS_ITERATIONS = 200
#: Search ceiling for the chamber count; far above any optimum found.
_MAX_CHAMBERS = 200

#: A finite-burn loss: from the impulsive burn and the burn time, the loss.
LossModel = Callable[[u.Quantity, u.Quantity], u.Quantity]

#: RL10B-2's carbon-carbon extension, taken as its 131 kg dry-mass excess over
#: the RL10A-4-2 (301 kg against 170 kg); an upper bound, since some of that is
#: gimbal hardware.  Exit 2.13 m, area ratio ~250 (AIAA 97-2672).
RL10B2_EXTENSION_MASS = 131.0 * u.kg
RL10B2_EXIT_DIAMETER = 2.13 * u.m
#: Exit diameter at A/A* = 300 for the wider of the two throats
#: (``tab:nozzle_area_ratio``: 7.5-8.7 m); the wider end is the heavier.
NOZZLE_EXIT_DIAMETER = 8.7 * u.m
#: A thin conical shell at fixed half-angle and areal density scales as the exit
#: diameter squared.  A rough scaling across a factor of four in size, and the
#: hydrogen extension's material (not bare carbon) is unsized in the parent.
NOZZLE_EXTENSION_MASS = (
    RL10B2_EXTENSION_MASS * (NOZZLE_EXIT_DIAMETER / RL10B2_EXIT_DIAMETER) ** 2
).to(u.t)


@dataclass(frozen=True)
class DepartureLedger:
    """One departure burn with its hardware charged, per kilogram of stack.

    Attributes:
        delivered_fraction: Mass after the burn over the stack at ignition.
        rod_mass_fraction: Rod mass the departure wave delivers, per kilogram
            of stack.
        tank_share: Gas tanks, per kilogram of stack.
        chamber_share: Chambers (wall and nozzle), per kilogram of stack.
        chamber_unit_mass: One chamber's wall plus nozzle extension.
        delivered_net: Stack delivered net of tanks and chambers.
        pulses: Rods the departure wave delivers, one per pulse.
        burn_time: Pulses over the chambers' combined pulse rate.
        finite_burn_loss: Fixed-direction loss, included in the burn priced.
        chambers: Chambers firing in parallel.
    """

    delivered_fraction: float
    rod_mass_fraction: float
    tank_share: float
    chamber_share: float
    chamber_unit_mass: u.Quantity
    delivered_net: float
    pulses: float
    burn_time: u.Quantity
    finite_burn_loss: u.Quantity
    chambers: int


def square_law_loss(burn: u.Quantity, burn_time: u.Quantity) -> u.Quantity:
    """The parent's fixed-direction loss, scaled as the burn time squared.

    Args:
        burn: Impulsive burn; unused, since the parent's law ignores it.
        burn_time: Length of a burn centred on periapsis.

    Returns:
        Speed lost against an impulsive burn (m/s).
    """
    ratio = float((burn_time / FIXED_DIRECTION_BURN_TIME).to_value(u.one))
    return (FIXED_DIRECTION_LOSS * ratio * ratio).to(u.m / u.s)


def price_departure(
    stack_mass: u.Quantity,
    start_speed: u.Quantity,
    burn: u.Quantity,
    impactor_speed: u.Quantity,
    pairing: ChamberPairing,
    efficiency: float,
    chambers: int = 1,
    gate_thrust_cost: float = 0.0,
    pitch_ratio: float = 0.0,
    loss_model: LossModel = fixed_direction_loss,
) -> DepartureLedger:
    """Price one departure burn with its tanks and chambers charged.

    Args:
        stack_mass: Stack at ignition: payload, chambers, tanks, propellant.
        start_speed: Craft speed at ignition, in the impactors' frame.
        burn: Speed the burn adds.
        impactor_speed: Departure-wave speed, same frame, head-on.
        pairing: Gas and wall pairing.
        efficiency: Energy efficiency ``eta``.
        chambers: Chambers firing in parallel.
        gate_thrust_cost: As in :func:`src.chamber_isp.effective_isp`.
        pitch_ratio: As in :func:`src.chamber_isp.effective_isp`.
        loss_model: Finite-burn loss; the integrated fixed-direction table
            by default.

    Returns:
        The departure's ledger.

    Raises:
        RuntimeError: If the loss runs away, which happens only when the
            chambers are far too few for the stack.
        ValueError: If the burn runs past what the loss model covers.
    """
    rods_per_stack = float((stack_mass / ROD_MASS).to_value(u.one))
    rate = float((chambers * PULSE_RATE).to_value(1 / u.s))

    def fly(loss: u.Quantity) -> ChamberBurn:
        return chamber_departure_burn(
            start_speed,
            start_speed + burn + loss,
            impactor_speed,
            pairing,
            efficiency,
            gate_thrust_cost=gate_thrust_cost,
            pitch_ratio=pitch_ratio,
        )

    loss = 0.0 * u.m / u.s
    for _ in range(_LOSS_ITERATIONS):
        result = fly(loss)
        burn_time = result.rod_mass_fraction * rods_per_stack / rate * u.s
        updated = loss_model(burn, burn_time).to(u.m / u.s)
        converged = (
            abs(float((updated - loss).to_value(u.m / u.s))) < _LOSS_TOLERANCE_M_S
        )
        loss = updated
        if converged:
            break
    else:
        raise RuntimeError("fixed-direction loss did not converge; add chambers")
    result = fly(loss)
    pulses = result.rod_mass_fraction * rods_per_stack
    spent = 1.0 - result.delivered_fraction
    gas = spent - (PLUG_RATIO + pitch_ratio) * result.rod_mass_fraction
    tank_share = pairing.tank_fraction * gas
    unit = (pairing.wall_mass + NOZZLE_EXTENSION_MASS).to(u.t)
    chamber_share = float((chambers * unit / stack_mass).to_value(u.one))
    return DepartureLedger(
        delivered_fraction=result.delivered_fraction,
        rod_mass_fraction=result.rod_mass_fraction,
        tank_share=tank_share,
        chamber_share=chamber_share,
        chamber_unit_mass=unit,
        delivered_net=result.delivered_fraction - tank_share - chamber_share,
        pulses=pulses,
        burn_time=(pulses / rate) * u.s,
        finite_burn_loss=loss,
        chambers=chambers,
    )


def best_departure(
    stack_mass: u.Quantity,
    start_speed: u.Quantity,
    burn: u.Quantity,
    impactor_speed: u.Quantity,
    pairing: ChamberPairing,
    efficiency: float,
    gate_thrust_cost: float = 0.0,
    pitch_ratio: float = 0.0,
    loss_model: LossModel = fixed_direction_loss,
) -> DepartureLedger:
    """Price the departure at the chamber count that delivers the most stack.

    Each chamber adds its wall and nozzle to the dry mass; each one removed
    lengthens the burn, and the fixed-direction loss grows as its square.  The
    net stack delivered therefore peaks at one count, found by walking up from
    one chamber until it has fallen for two counts in a row.

    Args:
        stack_mass: Stack at ignition.
        start_speed: Craft speed at ignition, in the impactors' frame.
        burn: Speed the burn adds.
        impactor_speed: Departure-wave speed, same frame, head-on.
        pairing: Gas and wall pairing.
        efficiency: Energy efficiency ``eta``.
        gate_thrust_cost: As in :func:`src.chamber_isp.effective_isp`.
        pitch_ratio: As in :func:`src.chamber_isp.effective_isp`.
        loss_model: As in :func:`price_departure`.

    Returns:
        The ledger at the best chamber count.

    Raises:
        RuntimeError: If no count up to the search ceiling converges.
    """
    best = None
    falls = 0
    for chambers in range(1, _MAX_CHAMBERS + 1):
        try:
            ledger = price_departure(
                stack_mass,
                start_speed,
                burn,
                impactor_speed,
                pairing,
                efficiency,
                chambers=chambers,
                gate_thrust_cost=gate_thrust_cost,
                pitch_ratio=pitch_ratio,
                loss_model=loss_model,
            )
        except (RuntimeError, ValueError):
            continue
        if best is None or ledger.delivered_net > best.delivered_net:
            best, falls = ledger, 0
        else:
            falls += 1
            if falls == 2:
                break
    if best is None:
        raise RuntimeError("no chamber count converged")
    return best


def growth_per_cycle(
    push_ratio: float, rod_mass_fraction: float, delivered_net: float
) -> float:
    """The fleet's growth over one cycle: the parent's ``tab:h2_breakdown`` ledger.

    The growth wave pushes ``push_ratio`` kilograms of stack per kilogram of
    PuffSat, and every kilogram of stack then needs ``rod_mass_fraction`` of
    departure-wave rods.  So of each returning batch the share
    ``1 / (1 + rod_mass_fraction * push_ratio)`` pushes, and growth is

        push ratio x share that pushes x stack delivered net of hardware

    Args:
        push_ratio: Stack pushed per kilogram of growth-wave PuffSat.
        rod_mass_fraction: Departure-wave rods per kilogram of stack.
        delivered_net: Stack delivered net of hardware.

    Returns:
        The batch's growth factor over the cycle.
    """
    share = 1.0 / (1.0 + rod_mass_fraction * push_ratio)
    return push_ratio * share * delivered_net


def doubling_time(growth: float, cycle: u.Quantity) -> u.Quantity:
    """Doubling time for a growth factor per cycle: ``cycle ln 2 / ln growth``.

    Args:
        growth: Growth factor per cycle; must exceed one.
        cycle: Cycle length.

    Returns:
        Doubling time, in the cycle's units.

    Raises:
        ValueError: If the fleet does not grow.
    """
    if growth <= 1.0:
        raise ValueError("a fleet that does not grow never doubles")
    return cycle * float(np.log(2.0) / np.log(growth))


def price_chain_departures(
    cycles: Sequence[TwoWaveCycle],
    stack_mass: u.Quantity,
    pairing: ChamberPairing,
    efficiency: float,
    gate_thrust_cost: float = 0.0,
    pitch_ratio: float = 0.0,
    loss_model: LossModel = fixed_direction_loss,
) -> List[DepartureLedger]:
    """Price every flown cycle's departure at its best chamber count.

    Each burn starts at the closed cycle's 200 km periapsis speed and adds the
    ``onward_burn`` of the window the payload leaves on (the next cycle's
    departure, ADR 0038), head-on into this return's nozzle wave, whose
    collision speed ``nozzle_wave_v_b`` is the impactor speed in the same frame.

    Args:
        cycles: Flown cycles from :func:`src.two_wave_growth.adaptive_two_wave_cycles`.
        stack_mass: Stack at ignition, the same every cycle.
        pairing: Gas and wall pairing.
        efficiency: Energy efficiency ``eta``.
        gate_thrust_cost: As in :func:`src.chamber_isp.effective_isp`.
        pitch_ratio: As in :func:`src.chamber_isp.effective_isp`.
        loss_model: As in :func:`price_departure`.

    Returns:
        One ledger per cycle, in order.
    """
    periapsis = puffsat_cycle_periapsis_speed()
    return [
        best_departure(
            stack_mass,
            periapsis,
            cycle.onward_burn * u.km / u.s,
            cycle.nozzle_wave_v_b * u.km / u.s,
            pairing,
            efficiency,
            gate_thrust_cost=gate_thrust_cost,
            pitch_ratio=pitch_ratio,
            loss_model=loss_model,
        )
        for cycle in cycles
    ]


#: Push ratio the report's growth column uses until the plate is re-optimised
#: (next stage): the parent's ``tab:h2_breakdown`` value.  A placeholder.
_PLACEHOLDER_PUSH_RATIO = 8.43
_DEFAULT_STACK = 1000.0
#: Report rows: each pairing at 50/70/90/100% of its chemistry ceiling, plus
#: the solved chamber at A/A* = 300 (H2 0.858, CH4 0.538 carbon in
#: equilibrium).  Methane spans its 1.4-5.6 kg of pitch per pulse.
_CEILING_SHARES = (0.50, 0.70, 0.90, 1.00)
_SOLVED = {HYDROGEN_5500K.name: 0.858, METHANE_7000K.name: 0.538}
_PITCH_PER_PULSE = {
    HYDROGEN_5500K.name: (0.0 * u.kg,),
    METHANE_7000K.name: (1.4 * u.kg, 5.6 * u.kg),
}


def _report(cycles: Sequence[TwoWaveCycle], stack: u.Quantity, push: float) -> str:
    """Tabulate every report row over the flown chain."""
    years = sum(c.period_years for c in cycles)
    rows = []
    for pairing in (HYDROGEN_5500K, METHANE_7000K):
        etas = sorted(
            [absolute_efficiency(pairing, share) for share in _CEILING_SHARES]
            + [_SOLVED[pairing.name]]
        )
        for eta in etas:
            for pitch in _PITCH_PER_PULSE[pairing.name]:
                pitch_ratio = float((pitch / ROD_MASS).to_value(u.one))
                ledgers = price_chain_departures(
                    cycles, stack, pairing, eta, GATE_THRUST_COST, pitch_ratio
                )
                growth = [
                    growth_per_cycle(push, d.rod_mass_fraction, d.delivered_net)
                    for d in ledgers
                ]
                total = float(np.prod(growth))
                row = {
                    "chamber": pairing.name,
                    "share": f"{eta / pairing.chemistry_ceiling:.0%}",
                    "eta": eta,
                    "pitch kg": pitch.to_value(u.kg),
                }
                for multiple in (2, 3):
                    picked = [
                        d
                        for c, d in zip(cycles, ledgers)
                        if c.synodic_multiple == multiple
                    ]
                    tag = f"{multiple}S"
                    row[f"{tag} n"] = f"{min(d.chambers for d in picked)}-" + str(
                        max(d.chambers for d in picked)
                    )
                    row[f"{tag} burn s"] = np.mean(
                        [d.burn_time.to_value(u.s) for d in picked]
                    )
                    row[f"{tag} loss m/s"] = np.mean(
                        [d.finite_burn_loss.to_value(u.m / u.s) for d in picked]
                    )
                    row[f"{tag} hw"] = np.mean(
                        [d.tank_share + d.chamber_share for d in picked]
                    )
                    row[f"{tag} net"] = np.mean([d.delivered_net for d in picked])
                row["chain growth"] = total
                row["doubling yr"] = (
                    years * float(np.log(2.0) / np.log(total))
                    if total > 1.0
                    else float("nan")
                )
                rows.append(row)
    return tabulate(rows, headers="keys", floatfmt=".3g")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Price the walled chamber's departure over the flown chain and print it.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stack-t",
        type=float,
        default=_DEFAULT_STACK,
        help=f"stack at ignition, tonnes (default {_DEFAULT_STACK:g}; a placeholder "
        "until the 1500 t launch unit is carried through)",
    )
    parser.add_argument(
        "--push-ratio",
        type=float,
        default=_PLACEHOLDER_PUSH_RATIO,
        help=f"growth-wave push ratio (default {_PLACEHOLDER_PUSH_RATIO}, the "
        "parent's tab:h2_breakdown value; a placeholder until the plate is re-optimised)",
    )
    args = parser.parse_args(argv)
    cycles = adaptive_two_wave_cycles()
    print(
        f"Walled-chamber departure over the flown chain ({len(cycles)} cycles), "
        f"{args.stack_t:g} t at ignition, push ratio {args.push_ratio:g}, gate charged."
    )
    print(
        "n = chambers (best count per cycle); hw = tanks + chambers per kg of stack; "
        "net = stack delivered net of hardware.  share = efficiency as a share of the chemistry ceiling; eta = absolute."
    )
    print(_report(cycles, args.stack_t * u.t, args.push_ratio))


if __name__ == "__main__":
    main()
