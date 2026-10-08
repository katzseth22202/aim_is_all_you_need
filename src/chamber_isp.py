"""Effective specific impulse of the head-on impact chamber, ``eq:eta_isp``.

A rod arrives head-on at closing speed ``w`` inside a walled chamber holding
``k`` kilograms of gas charge per kilogram of rod, sealed by a polyethylene plug
of ``P`` kilograms per kilogram of rod.  Everything leaves together through the
nozzle, and a share ``eta`` of the rod's kinetic energy ``w^2/2`` leaves as
directed exhaust.  The parent paper (``Balloon-Pulse-Propulsion``,
``templateArxiv.tex``, ``eq:eta_isp`` in ``sec:methane_7000_near_term``) gives

    v_e   = w sqrt(eta / (k + 1 + P))
    I_eff = ((k + 1 + P) v_e - w) / ((k + P) g0)

The ``-w`` is the head-on momentum debit: the rod arrives moving the wrong way.
The plug is carried propellant, so it sits in the denominator with the charge.

The port gate (``sec:port_gate``) slams shut behind the rod and costs 1% of
thrust.  That 1% comes off the exhaust momentum ``(k + 1 + P) v_e`` *before* the
head-on debit, so it costs the finished Isp about 1.3%, not 1%.  Taken that way
it reproduces the gated rows of ``tab:wall_pairing_doubling`` (1112 s and 922 s
for hydrogen); taken off the finished Isp it would give 1114 s and 924 s.

The methane chamber also loses 1.4-5.6 kg of its pitch lining per pulse
(``tab:wall_pairings``).  That pitch leaves with the exhaust and is resprayed
from the tanks, so it is carried propellant at the same ``eta``: it joins
``k + 1 + P`` in the exhaust and ``k + P`` in the carried mass.  With the gate,
that reproduces methane's gated ranges in ``tab:wall_pairing_doubling``
(762-777 s at A/A* = 300 with carbon in equilibrium).

The slug ratio ``k`` is not a fixed property of the chamber.  The table's values
(24.0 for hydrogen, 27.5 for methane) hold at the reference closing speed of 75
km/s, where ``(k + 1 + P) u(T) = w^2 / 2`` sizes the charge so the chamber
reaches its design temperature.  On both the growth and the departure wave the
closing speed changes from pulse to pulse as the craft speeds toward or away
from the impactors, so each pulse carries its own ``k``.  The rod and plug stay
fixed and the gas load varies (:func:`slug_ratio_at`).  ``eta`` is held at its
reference value across speeds, which is an assumption: the companion solved it
only at 75 km/s.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.constants import g0

#: Plug mass per kilogram of rod: the 4.4 kg polyethylene plug on the 2.5 kg rod.
PLUG_RATIO = 1.76
#: Share of exhaust momentum the port gate costs (``sec:port_gate``).
GATE_THRUST_COST = 0.01
#: Trapezoid nodes across the burn's speed gain.
BURN_STEPS = 401
#: Rod mass the per-pulse figures are quoted against.
ROD_MASS = 2.5 * u.kg
#: Pulse rate of one 20 m^3 chamber (``fig:methane_wall_detail``: a 250 ms cycle).
PULSE_RATE = 4.0 / u.s


@dataclass(frozen=True)
class ChamberPairing:
    """A gas and wall pairing from the parent's ``tab:wall_pairings``.

    Attributes:
        name: Short label.
        temperature: Chamber temperature the charge is sized to.
        reference_slug_ratio: Charge per kilogram of rod, ``k``, at the
            reference closing speed of 75 km/s.
        wall_mass: Autofrettaged wall of one 20 m^3 chamber for the 2.5 kg
            rod, sized to the pairing's peak pressure (``sec:steel_chamber_service``).
        tank_fraction: Tank mass per kilogram of gas charge, scaled from
            NASA's Mars reference drop tank by density (``sec:ntr_departure``).
        chemistry_ceiling: The most of the pulse any chamber of this pairing
            could turn into directed exhaust: one minus the energy held in
            bonds at peak that the A/A* = 300 nozzle never returns, even in
            equilibrium.  Efficiencies are quoted as a share of it.
        rod_mass: The rod one pulse fires; the 2.5 kg rod by default.
        plug_ratio: Plug mass per kilogram of rod, ``P``.
        pulse_rate: Pulses per second one chamber fires.
        extension_mass: The nozzle extension; None takes
            :data:`src.chamber_departure.NOZZLE_EXTENSION_MASS`, the RL10B-2's
            scaled to the 20 m^3 sphere's 8.7 m exit.
    """

    name: str
    temperature: u.Quantity
    reference_slug_ratio: float
    wall_mass: u.Quantity
    tank_fraction: float
    chemistry_ceiling: float
    rod_mass: u.Quantity = field(default_factory=lambda: ROD_MASS)
    plug_ratio: float = PLUG_RATIO
    pulse_rate: u.Quantity = field(default_factory=lambda: PULSE_RATE)
    extension_mass: Optional[u.Quantity] = None


#: Closing speed ``tab:wall_pairings`` sizes the charges at: the 2.5 kg rod's.
REFERENCE_CLOSING_SPEED = 75.0 * u.km / u.s


#: Hydrogen at 5500 K on a bare copper liner (``tab:wall_pairings``): 818 bar
#: peak, about a 32 t wall; 0.205 kg of tank per kg of liquid hydrogen.  36% of
#: the pulse is in bonds at peak and the A/A* = 300 nozzle returns 94% of them
#: (``tab:nozzle_area_ratio``), so the ceiling is 1 - 0.36 x 0.06 = 0.978.
HYDROGEN_5500K = ChamberPairing(
    "H2 5500 K", 5500.0 * u.K, 24.0, 32.0 * u.t, 0.205, 1.0 - 0.36 * (1.0 - 0.94)
)
#: Methane at 7000 K on Cr-Mo steel lined with 0.2 mm of pitch
#: (``tab:wall_pairings``): 496 bar peak, about a 19 t wall; 0.034 kg of tank
#: per kg of liquid methane.  69% of the pulse is in bonds at peak and even with
#: carbon in equilibrium the A/A* = 300 nozzle returns only 52% of them; the
#: carbon never re-bonds.  The ceiling is 1 - 0.69 x 0.48 = 0.669 (0.683 at the
#: A/A* = 1000 stretch nozzle, 54% returned).
METHANE_7000K = ChamberPairing(
    "CH4 7000 K", 7000.0 * u.K, 27.5, 19.0 * u.t, 0.034, 1.0 - 0.69 * (1.0 - 0.52)
)

#: The survivable methane chamber (parent S17, impact sim @ 69d1f40, its
#: ``walled_nozzle_chamber_blast.md`` section 3h, decided 2026-10-08).  The 20 m^3
#: sphere's bonded overwrap delaminates under the stopped rod's line blast, so
#: the chamber is bulged to 212 m^3 at ~94 bar and wrapped dry: 48.2 t of
#: Kevlar 49 over a 10 mm Cr-Mo shell.  It fires a 5 kg rod at 2 Hz (4 Hz cannot
#: refill) behind a 20 kg frozen-methane plug, ``P`` = 4.  The charge and plug
#: together are ~117 kg per rod, so ``k + P`` = 23.4 at 75 km/s: taken as given.
#: (Holding the old chamber's energy per kilogram would need 29.3.)
SURVIVABLE_ROD_MASS = 5.0 * u.kg
SURVIVABLE_PULSE_RATE = 2.0 / u.s
SURVIVABLE_PLUG_RATIO = 4.0
SURVIVABLE_CARRIED_PER_ROD = 117.0 / 5.0
SURVIVABLE_WALL_MASS = 48.2 * u.t
#: The impact sim's fallback wall: a maraging-steel band.
SURVIVABLE_FALLBACK_WALL_MASS = 68.8 * u.t
#: The impact sim's net impulse per pulse, head-on debit taken, gate charged.
SURVIVABLE_IMPULSE = {100: 906.0 * u.kN * u.s, 300: 971.0 * u.kN * u.s}
#: Its nozzle extensions: the 2.19 t scaled by exit area, 13.0 m and 22.5 m exits.
SURVIVABLE_EXTENSION_MASS = {100: 4.9 * u.t, 300: 14.7 * u.t}


def survivable_methane(
    area_ratio: int,
    wall_mass: u.Quantity = SURVIVABLE_WALL_MASS,
    rods_split: int = 1,
) -> ChamberPairing:
    """The survivable methane chamber at one nozzle area ratio.

    Args:
        area_ratio: 100 or 300.
        wall_mass: The wall of one 5 kg chamber, or of all ``rods_split``.
        rods_split: Split the 5 kg rod's flow across this many smaller
            chambers, each firing ``5 / rods_split`` kg at 2 Hz with the same
            ``k`` and ``P``, sharing ``wall_mass`` evenly, and each with an
            extension of its share of the exit area (parent S17 item 3).

    Returns:
        The pairing; its ``eta`` is :func:`efficiency_for_impulse` at
        :data:`SURVIVABLE_IMPULSE`.
    """
    share = 1.0 / rods_split
    return ChamberPairing(
        f"CH4 survivable AR{area_ratio}"
        + (f" x{rods_split}" if rods_split > 1 else ""),
        7000.0 * u.K,
        SURVIVABLE_CARRIED_PER_ROD - SURVIVABLE_PLUG_RATIO,
        (wall_mass * share).to(u.t),
        METHANE_7000K.tank_fraction,
        METHANE_7000K.chemistry_ceiling,
        rod_mass=SURVIVABLE_ROD_MASS * share,
        plug_ratio=SURVIVABLE_PLUG_RATIO,
        pulse_rate=SURVIVABLE_PULSE_RATE,
        extension_mass=(SURVIVABLE_EXTENSION_MASS[area_ratio] * share).to(u.t),
    )


def efficiency_for_impulse(
    impulse_per_pulse: u.Quantity,
    pairing: ChamberPairing,
    closing_speed: u.Quantity = REFERENCE_CLOSING_SPEED,
    gate_thrust_cost: float = GATE_THRUST_COST,
) -> float:
    """The ``eta`` that gives one pulse a net impulse, ``eq:eta_isp`` inverted.

    Without pitch, a pulse's net impulse per kilogram of rod is
    ``w ((1 - gate) sqrt(eta (k + 1 + P)) - 1)``.

    Args:
        impulse_per_pulse: Net impulse of one pulse, head-on debit taken.
        pairing: The chamber, at its reference charge.
        closing_speed: The closing speed the impulse was quoted at.
        gate_thrust_cost: Share of exhaust momentum lost at the port.

    Returns:
        The energy efficiency ``eta``.
    """
    w = float(closing_speed.to_value(u.m / u.s))
    per_rod = float((impulse_per_pulse / pairing.rod_mass).to_value(u.m / u.s))
    root = (per_rod / w + 1.0) / (1.0 - gate_thrust_cost)
    return float(root**2 / (pairing.reference_slug_ratio + 1.0 + pairing.plug_ratio))


def absolute_efficiency(pairing: ChamberPairing, share_of_ceiling: float) -> float:
    """Convert an efficiency quoted against the chemistry ceiling to ``eta``.

    A chamber that is "50% efficient" turns half of what its chemistry allows
    into directed exhaust, not half the pulse: 0.33 for methane, 0.49 for
    hydrogen.

    Args:
        pairing: Gas and wall pairing.
        share_of_ceiling: Efficiency as a share of the pairing's ceiling.

    Returns:
        The energy efficiency ``eta`` of ``eq:eta_isp``.
    """
    return share_of_ceiling * pairing.chemistry_ceiling


def slug_ratio_at(closing_speed: u.Quantity, pairing: ChamberPairing) -> float:
    """Gas charge per kilogram of rod that holds the pairing's chamber temperature.

    The chamber's energy per kilogram, ``u(T) = (w^2/2) / (k + 1 + P)``, is fixed
    by the reference charge at ``REFERENCE_CLOSING_SPEED``, so the charge scales as

        k(w) = (k_ref + 1 + P) (w / w_ref)^2 - 1 - P

    Args:
        closing_speed: Rod speed relative to the chamber on this pulse, ``w``.
        pairing: Gas and wall pairing whose temperature is held.

    Returns:
        Slug ratio ``k`` for this pulse.

    Raises:
        ValueError: If the rod and plug alone cannot reach the chamber
            temperature at this speed, so no charge would do.
    """
    ratio = float((closing_speed / REFERENCE_CLOSING_SPEED).to_value(u.one))
    plug = pairing.plug_ratio
    exhausted = (pairing.reference_slug_ratio + 1.0 + plug) * ratio * ratio
    slug_ratio = exhausted - 1.0 - plug
    if slug_ratio < 0.0:
        raise ValueError(
            f"{pairing.name} cannot reach {pairing.temperature} at {closing_speed}: "
            "the rod and plug alone take more energy than the pulse carries"
        )
    return slug_ratio


def momentum_debit_share(
    efficiency: float,
    slug_ratio: float,
    plug_ratio: float = PLUG_RATIO,
    pitch_ratio: float = 0.0,
) -> float:
    """Share of the gross exhaust momentum the head-on rod cancels.

    The rod arrives carrying ``w`` of momentum per kilogram the wrong way, and
    that debit is paid in full whatever ``eta`` is.  The exhaust's momentum,
    ``(k + 1 + P) v_e = w sqrt(eta (k + 1 + P))``, falls only as ``sqrt(eta)``.
    So the share is ``1 / sqrt(eta (k + 1 + P))``, and it grows as ``eta``
    falls: lost efficiency costs a head-on chamber more than ``sqrt(eta)``.  An
    overtaking push is the mirror image, since there the impactor's momentum is
    a credit, and lost efficiency costs it less.

    Args:
        efficiency: Energy efficiency ``eta``.
        slug_ratio: Gas charge per kilogram of rod, ``k``.
        plug_ratio: Plug mass per kilogram of rod, ``P``.
        pitch_ratio: Wall lining lost per kilogram of rod.

    Returns:
        The debit over the gross exhaust momentum, without the gate.
    """
    return float(
        1.0 / (efficiency * (slug_ratio + 1.0 + plug_ratio + pitch_ratio)) ** 0.5
    )


def effective_isp(
    closing_speed: u.Quantity,
    efficiency: float,
    slug_ratio: float,
    plug_ratio: float = PLUG_RATIO,
    gate_thrust_cost: float = 0.0,
    pitch_ratio: float = 0.0,
) -> u.Quantity:
    """Effective specific impulse of one pulse, ``eq:eta_isp``.

    Args:
        closing_speed: Rod speed relative to the chamber, ``w``.
        efficiency: Share of the rod's kinetic energy leaving as directed
            exhaust, ``eta``.
        slug_ratio: Gas charge per kilogram of rod, ``k``.
        plug_ratio: Plug mass per kilogram of rod, ``P``.
        gate_thrust_cost: Share of exhaust momentum lost at the port;
            ``GATE_THRUST_COST`` for the gated chamber, zero for the raw
            ``tab:nozzle_area_ratio`` figures.
        pitch_ratio: Wall lining lost per kilogram of rod (pitch per pulse
            over the rod); zero for hydrogen's bare copper.

    Returns:
        Effective specific impulse (astropy Quantity, s).
    """
    w = float(closing_speed.to_value(u.m / u.s))
    exhaust = _effective_exhaust(
        w, efficiency, slug_ratio, plug_ratio, gate_thrust_cost, pitch_ratio
    )
    return (float(exhaust) / float(g0.to_value(u.m / u.s**2))) * u.s


def _effective_exhaust(
    closing: npt.ArrayLike,
    efficiency: float,
    slug_ratio: npt.ArrayLike,
    plug_ratio: float,
    gate_thrust_cost: float,
    pitch_ratio: float,
) -> npt.NDArray[np.float64]:
    """``eq:eta_isp`` as an effective exhaust speed, in ``closing``'s units."""
    w = np.asarray(closing, dtype=np.float64)
    carried = np.asarray(slug_ratio, dtype=np.float64) + plug_ratio + pitch_ratio
    exhausted = carried + 1.0
    exhaust_speed = w * np.sqrt(efficiency / exhausted)
    impulse = (1.0 - gate_thrust_cost) * exhausted * exhaust_speed - w
    return np.asarray(impulse / carried, dtype=np.float64)


@dataclass(frozen=True)
class ChamberBurn:
    """Mass ledger of one head-on chamber departure burn.

    Attributes:
        delivered_fraction: Mass after the burn over mass before it.
        rod_mass_fraction: Rod mass the departure wave must deliver, per
            kilogram of craft at ignition.  Divide by the pairing's
            ``rod_mass`` for pulses.
    """

    delivered_fraction: float
    rod_mass_fraction: float


def chamber_departure_burn(
    start_speed: u.Quantity,
    final_speed: u.Quantity,
    impactor_speed: u.Quantity,
    pairing: ChamberPairing,
    efficiency: float,
    gate_thrust_cost: float = 0.0,
    pitch_ratio: float = 0.0,
) -> ChamberBurn:
    """Integrate a departure burn whose every pulse re-sizes its charge.

    The departure wave meets the craft head-on, so the closing speed is the
    impactor's speed plus the craft's own, ``w = v_imp + v``, and it rises as the
    craft accelerates.  Each pulse carries ``k(w)`` from :func:`slug_ratio_at`,
    and the rocket equation is integrated with ``I_eff(w)`` varying, following
    ``circular_resonance_impulse._integrate_burn``:

        ln(m0 / m1) = integral dv / (I_eff(w(v)) g0)

    Args:
        start_speed: Craft speed at ignition, in the frame the impactors are
            measured in (periapsis speed for a burn at periapsis).
        final_speed: Craft speed at cutoff.
        impactor_speed: Impactor speed, same frame, directly opposing the craft.
        pairing: Gas and wall pairing whose temperature every pulse holds.
        efficiency: Energy efficiency ``eta``, held across the burn.
        gate_thrust_cost: As in :func:`effective_isp`.
        pitch_ratio: As in :func:`effective_isp`.

    Returns:
        The burn's mass ledger.

    Raises:
        ValueError: If the burn ends slower than it starts, or a pulse is too
            slow to reach the pairing's temperature.
    """
    v0 = float(start_speed.to_value(u.km / u.s))
    v1 = float(final_speed.to_value(u.km / u.s))
    if v1 < v0:
        raise ValueError("a departure burn cannot end slower than it starts")
    speeds = np.linspace(v0, v1, BURN_STEPS)
    closing = float(impactor_speed.to_value(u.km / u.s)) + speeds
    # Closing speed only rises through the burn, so ignition is the one pulse
    # that can be too slow to reach the chamber temperature.
    slug_ratio_at(float(closing[0]) * u.km / u.s, pairing)
    reference = float(REFERENCE_CLOSING_SPEED.to_value(u.km / u.s))
    plug = pairing.plug_ratio
    slug_ratios = (pairing.reference_slug_ratio + 1.0 + plug) * (
        closing / reference
    ) ** 2 - (1.0 + plug)
    exhaust = _effective_exhaust(
        closing, efficiency, slug_ratios, plug, gate_thrust_cost, pitch_ratio
    )
    steps = np.diff(speeds)
    integrand = 1.0 / exhaust
    log_mass = np.concatenate(
        ([0.0], np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * steps))
    )
    # Each kilogram of propellant spent is (k + P + pitch) kilograms per rod.
    rod_rate = np.exp(-log_mass) * integrand / (slug_ratios + plug + pitch_ratio)
    rods = float(np.sum(0.5 * (rod_rate[:-1] + rod_rate[1:]) * steps))
    return ChamberBurn(
        delivered_fraction=float(np.exp(-log_mass[-1])), rod_mass_fraction=rods
    )
