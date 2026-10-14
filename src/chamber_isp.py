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

from dataclasses import dataclass, field, replace
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
class HydrogenStorage:
    """How a hydrogen chamber's charge is stored until the burn (ADR 0051).

    Attributes:
        name: Short label.
        density: Bulk density of the stored charge; the tank scales inversely.
        cryostat_fraction: Cryostat mass dropped just before the departure
            burn, per kilogram of hydrogen launched.
        boil_off_per_day: Hydrogen lost per day of the hold, per kilogram
            launched.
    """

    name: str
    density: u.Quantity
    cryostat_fraction: float
    boil_off_per_day: float


#: Normal-boiling-point liquid, the ledger's long-standing case.  No
#: flight-scale source exists; passive-MLI large tanks lose 0.1-0.5%/day, held
#: for one parking orbit, and cryostat mass is swept over 1-5%; the middle of
#: both is carried.
LIQUID_HYDROGEN = HydrogenStorage("liquid", 70.8 * u.kg / u.m**3, 0.03, 0.003)
#: Slush at 50% solid, made on the ground by freeze-thaw (NASA Lewis K-Site,
#: 1990-94: McNelis et al., NTRS 19950017766; Hardy and Whalen, NASA TM 105596).
#: 15% denser than normal-boiling-point liquid, and 18% more heat sink: melting
#: the solid and warming the liquid from the 13.8 K triple point to 20.3 K
#: absorbs about 81 kJ/kg, 18% of the 446 kJ/kg that boils it.  The same
#: insulation's 0.3%/day therefore boils nothing for about 60 days, so a
#: 20-day hold loses none.  Same cryostat as the liquid (no active cooling).
SLUSH_HYDROGEN = HydrogenStorage("slush 50%", 1.15 * 70.8 * u.kg / u.m**3, 0.03, 0.0)
#: Tank per kilogram of normal-boiling-point liquid hydrogen, scaled from NASA's
#: Mars reference drop tank (``sec:ntr_departure``).
LIQUID_HYDROGEN_TANK_FRACTION = 0.205
#: Every hydrogen chamber stores its charge as slush (ADR 0065), so its tank is
#: the liquid's shrunk by the density gained.  :func:`stored_as` with
#: :data:`LIQUID_HYDROGEN` gives the liquid back for comparison.
SLUSH_HYDROGEN_TANK_FRACTION = LIQUID_HYDROGEN_TANK_FRACTION * float(
    (LIQUID_HYDROGEN.density / SLUSH_HYDROGEN.density).to_value(u.one)
)


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
        storage: How a hydrogen charge is stored; None for methane.  A pairing
            is a hydrogen pairing exactly when this is set, which survives
            pickling where ``is HYDROGEN_5500K`` does not.
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
    storage: Optional[HydrogenStorage] = None

    @property
    def hydrogen(self) -> bool:
        """Whether the charge is hydrogen."""
        return self.storage is not None


#: Closing speed ``tab:wall_pairings`` sizes the charges at: the 2.5 kg rod's.
REFERENCE_CLOSING_SPEED = 75.0 * u.km / u.s


#: Hydrogen at 5500 K on a bare copper liner (``tab:wall_pairings``): 818 bar
#: peak, about a 32 t wall; stored as slush (ADR 0065).  36% of
#: the pulse is in bonds at peak and the A/A* = 300 nozzle returns 94% of them
#: (``tab:nozzle_area_ratio``), so the ceiling is 1 - 0.36 x 0.06 = 0.978.
HYDROGEN_5500K = ChamberPairing(
    "H2 5500 K",
    5500.0 * u.K,
    24.0,
    32.0 * u.t,
    SLUSH_HYDROGEN_TANK_FRACTION,
    1.0 - 0.36 * (1.0 - 0.94),
    storage=SLUSH_HYDROGEN,
)


#: Hydrogen at 3000 K, the temperature NTR fuel and walls were tested to (ADR
#: 0049, 0052).  A study point, not a solved chamber.  Equilibrium internal
#: energy at the chamber's own density is 34.33 MJ/kg (Cantera ``gri30``, via
#: :func:`src.orbit_turn_analysis.hydrogen_hot_energy`), so the 75 km/s rod
#: and its 1.76 plug size the charge at ``k = 79.2``: 198 kg in 20 m^3, 1220
#: bar peak.  The same solve gives 806 bar for the 5500 K pairing against the
#: table's 818.  A thin wall's mass goes as ``p V``, fixed by the charge
#: whatever the volume, so the copper-lined wall scales from 32 t by 1220/818.
#: Only 0.3% of the nuclei are atomic, about 2% of the energy in bonds, so the
#: ceiling is 1 - 0.02 x 0.06.  Its efficiency is a guess (ADR 0052).
HYDROGEN_3000K = ChamberPairing(
    "H2 3000 K",
    3000.0 * u.K,
    79.2,
    32.0 * 1220.0 / 818.0 * u.t,
    SLUSH_HYDROGEN_TANK_FRACTION,
    1.0 - 0.02 * (1.0 - 0.94),
    storage=SLUSH_HYDROGEN,
)


#: The bulged hydrogen chamber (ADR 0055; impact sim @ 7afdba3, its
#: ``walled_nozzle_hydrocode_asks.md`` section 6).  The 20 m^3 sphere's wall is
#: unresolved at ~2.95 GPa, so the methane chamber's r 3.0 m bulge is filled
#: with hydrogen: 106 m^3, the same 60 kg charge on the 2.5 kg rod (fixed
#: charge, so 4884 K at 142 bar), a 0.84 m^2 throat and a 37.0 t dry Kevlar
#: wrap, which passes on a thin margin (2.20% flung strain against 2.4%).  Its
#: cold residue lets a polymer film seal at 0.3 bar, so it cycles at ~2.2-2.4 Hz;
#: it is flown at 2 Hz.  Doubling its throat fails the wrap.  The extension is
#: the sphere's scaled by exit area, ~18 m at A/A* = 300.
HYDROGEN_BULGED = ChamberPairing(
    "H2 bulged 106 m^3",
    4884.0 * u.K,
    24.0,
    37.0 * u.t,
    SLUSH_HYDROGEN_TANK_FRACTION,
    HYDROGEN_5500K.chemistry_ceiling,
    pulse_rate=2.0 / u.s,
    extension_mass=(2.19 * u.t * 0.84 * 300.0 / 59.4468).to(u.t),
    storage=SLUSH_HYDROGEN,
)
#: Its net impulse per pulse at A/A* = 300, gate charged, with finite-rate
#: H + H + M recombination through the nozzle (impact sim @ ca07938, its
#: ADR-0058; ADR 0064): 683 kN s, 661 at the high wall-heat edge.  It replaces
#: ADR 0055's 657, whose sudden freeze understated it.  The sphere gave 697.
HYDROGEN_BULGED_IMPULSE = 683.0 * u.kN * u.s


def stored_as(pairing: ChamberPairing, storage: HydrogenStorage) -> ChamberPairing:
    """A hydrogen pairing with its charge stored another way (ADR 0051).

    The tank scales with the charge's volume, so its fraction falls in
    proportion to the density gained.  Nothing else about the chamber changes.

    Args:
        pairing: A hydrogen pairing.
        storage: The new storage.

    Returns:
        The pairing.

    Raises:
        ValueError: If ``pairing`` is not hydrogen.
    """
    if pairing.storage is None:
        raise ValueError("only a hydrogen pairing has a storage to change")
    ratio = float((pairing.storage.density / storage.density).to_value(u.one))
    return replace(
        pairing,
        name=f"{pairing.name}, {storage.name}",
        tank_fraction=pairing.tank_fraction * ratio,
        storage=storage,
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
#: the chamber is bulged to 212 m^3 at ~94 bar and wrapped dry in Kevlar 49 over
#: a 10 mm Cr-Mo shell.  It fires a 5 kg rod at 2 Hz (4 Hz cannot refill) behind
#: a 20 kg frozen-methane plug, ``P`` = 4.  The charge and plug together are
#: ~117 kg per rod, so ``k + P`` = 23.4 at 75 km/s: taken as given.  (Holding
#: the old chamber's energy per kilogram would need 29.3.)
#:
#: ADR 0055 (impact sim @ 7afdba3, its ADR-0056 and 2026-10-13 amendment): the
#: throat doubles from 1.33 to 2.66 m^2 so the 2 Hz cycle keeps 150-200 ms
#: spare.  The cone is 1 m shorter and the dry wrap falls to 46.2 t (the 1.33
#: m^2 chamber re-solved needs 52.4 t, not 48.2).  Net impulse per pulse is
#: 885 / 941 kN s with the 1% gate and finite-rate H + H + M recombination
#: charged (ADR 0056; ADR 0055's 868 / 899 froze it suddenly, and the old
#: 906 / 971 charged neither).  Each shot spends a 2.6 kg aramid throat
#: film, sealed at 0.3 bar, which leaves with the exhaust as the pitch does.
SURVIVABLE_ROD_MASS = 5.0 * u.kg
SURVIVABLE_PULSE_RATE = 2.0 / u.s
SURVIVABLE_PLUG_RATIO = 4.0
SURVIVABLE_CARRIED_PER_ROD = 117.0 / 5.0
SURVIVABLE_WALL_MASS = 46.2 * u.t
#: The impact sim's fallback wall: a maraging-steel band.
SURVIVABLE_FALLBACK_WALL_MASS = 68.8 * u.t
#: The impact sim's net impulse per pulse, head-on debit taken, gate charged,
#: on the 2.66 m^2 throat, with H + H + M recombining at its measured rate
#: through the nozzle and carbon trading atoms freely (ADR 0056, the sim's S26
#: answer @ df4bc6e).  A/A* = 150 is S24's option (a): the 22.5 m exit kept.
SURVIVABLE_IMPULSE = {
    100: 885.0 * u.kN * u.s,
    150: 908.0 * u.kN * u.s,
    300: 941.0 * u.kN * u.s,
}
#: The same with carbon frozen at the chamber state: the floor (ADR 0056).
SURVIVABLE_IMPULSE_CARBON_FROZEN = {
    100: 810.0 * u.kN * u.s,
    150: 827.0 * u.kN * u.s,
    300: 849.0 * u.kN * u.s,
}
#: ADR 0055's figures: hydrogen recombination frozen suddenly, which tripled
#: its loss.
SURVIVABLE_IMPULSE_SUDDEN_FREEZE = {100: 868.0 * u.kN * u.s, 300: 899.0 * u.kN * u.s}
#: ADR 0044's figures (S17), gate and freeze uncharged, 1.33 m^2 throat.
SURVIVABLE_IMPULSE_S17 = {100: 906.0 * u.kN * u.s, 300: 971.0 * u.kN * u.s}
#: The aramid throat film each shot spends (sealed at 0.3 bar, mass ~ p r^3).
SURVIVABLE_THROAT_FILM = 2.6 * u.kg
#: Its nozzle extensions, scaled by exit area from the 2.19 t at 8.7 m: the
#: doubled throat doubles them, ~18 m and ~32 m exits (4.9 / 14.7 t at 1.33 m^2).
#: A/A* = 150 keeps the 22.5 m exit, and so the 14.7 t extension.
SURVIVABLE_EXTENSION_MASS = {100: 9.8 * u.t, 150: 14.7 * u.t, 300: 29.4 * u.t}

#: The survivable chamber's volume and throat (parent ``tab:one_chamber``).
SURVIVABLE_VOLUME = 212.0 * u.m**3
SURVIVABLE_THROAT_AREA = 2.66 * u.m**2
#: The throat ADR 0053 timed: the narrowest that fitted 2 Hz.
SURVIVABLE_THROAT_AREA_S17 = 1.33 * u.m**2
#: The 20 m^3 sphere's throat at A/A* = 300 behind the 8.7 m exit.
SPHERE_VOLUME = 20.0 * u.m**3
SPHERE_THROAT_AREA = 59.4468 / 300.0 * u.m**2


@dataclass(frozen=True)
class BlowdownGas:
    """A chamber's gas at peak, for the blowdown estimate (ADR 0053).

    Cantera ``gri30`` equilibrium at the peak state (polynomials extrapolated
    above 5000 K).  ``gamma`` is the effective isentropic exponent of an
    equilibrium expansion to a tenth of the density, which counts the heat that
    recombination returns; it is lower than the frozen value.

    Attributes:
        name: Label.
        peak_pressure: Pressure at peak.
        gamma: Effective isentropic exponent over the blowdown.
        sound_speed: ``sqrt(gamma p / rho)`` at peak.
    """

    name: str
    peak_pressure: u.Quantity
    gamma: float
    sound_speed: u.Quantity


#: The survivable methane chamber: 122 kg of methane and rod in 212 m^3 at
#: 7000 K, 94 bar (the parent's ~94 bar).  The parent's own blowdown uses
#: 3.7 km/s, a slower bound kept beside this one.
METHANE_SURVIVABLE_GAS = BlowdownGas(
    "CH4 7000 K, 212 m^3", 94.0 * u.bar, 1.225, 4.48 * u.km / u.s
)
#: The 20 m^3 hydrogen sphere: 60 kg at 5500 K, 806 bar (the table's 818).
HYDROGEN_SPHERE_GAS = BlowdownGas(
    "H2 5500 K, 20 m^3", 806.0 * u.bar, 1.194, 5.66 * u.km / u.s
)


def blowdown_time(
    volume: u.Quantity,
    throat_area: u.Quantity,
    gas: BlowdownGas,
    mass_left: float,
    sound_speed: Optional[u.Quantity] = None,
) -> u.Quantity:
    """Time for a chamber to empty to ``mass_left`` of its peak charge.

    The quasi-steady adiabatic blowdown of ADR 0049: a choked throat, the
    gas expanding isentropically behind it, discharge coefficient one,

        m / m0 = (1 + q lambda t)^(-1/q),   q = (gamma - 1) / 2,
        lambda = (A* / V) a0 (2 / (gamma + 1))^((gamma + 1) / (2 (gamma - 1))).

    Pressure falls as ``(m / m0)^gamma``, so a pressure target is
    ``mass_left = (p / p0)^(1 / gamma)``.

    Args:
        volume: Chamber volume.
        throat_area: Throat area.
        gas: The gas at peak.
        mass_left: Share of the peak mass still inside, in (0, 1).
        sound_speed: Overrides ``gas.sound_speed``.

    Returns:
        The time.

    Raises:
        ValueError: If ``mass_left`` is outside (0, 1).
    """
    if not 0.0 < mass_left < 1.0:
        raise ValueError("mass_left must lie in (0, 1)")
    gamma = gas.gamma
    speed = gas.sound_speed if sound_speed is None else sound_speed
    q = 0.5 * (gamma - 1.0)
    choke = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    rate = (throat_area / volume * speed * choke).to(1 / u.s)
    return ((mass_left ** (-q) - 1.0) / (q * rate)).to(u.ms)


#: The impact sim's real-gas time to seal the throat film on ADR 0053's
#: 1.33 m^2 throat, by the pressure it is sealed at (bar): 1.11 bar is the
#: fill pressure, which leaves a ~1,100 K residue under the film; 0.3 bar keeps
#: a polymer film under its 400 C rating (its ``walled_nozzle_hydrocode_asks.md``
#: section 3).
SEAL_TIME_S17 = {1.11: 281.0 * u.ms, 0.3: 389.0 * u.ms}
#: The impact sim's mechanism times, stated by it as assumptions: both films
#: laid, the next charge, the lock before the rod arrives.
FILM_LAY_TIME = 30.0 * u.ms
FILL_MASS = 108.0 * u.kg
FILL_RATE = 1.0 * u.t / u.s
LOCK_TIME = 20.0 * u.ms


def survivable_cycle(
    throat_area: u.Quantity,
    seal_pressure: float = 0.3,
    fill_rate: u.Quantity = FILL_RATE,
) -> u.Quantity:
    """The shortest period the 212 m^3 methane chamber can fire at.

    Seal the throat film, lay both films, fill, lock (the impact sim's
    ``cycle.py``).  The seal time is its real-gas solve on the 1.33 m^2 throat
    scaled as ``V / A*``, which holds exactly because the gas state depends only
    on energy per volume.  The fill cannot overlap an open throat; the plug
    goes in during the fill.

    Args:
        throat_area: Throat area.
        seal_pressure: Pressure the throat film is sealed at, a key of
            :data:`SEAL_TIME_S17`.
        fill_rate: Injector flow while filling.

    Returns:
        The period.
    """
    seal = SEAL_TIME_S17[seal_pressure] * float(
        (SURVIVABLE_THROAT_AREA_S17 / throat_area).to_value(u.one)
    )
    fill = FILL_MASS / fill_rate
    return (seal + FILM_LAY_TIME + fill + LOCK_TIME).to(u.ms)


def survivable_methane(
    area_ratio: int,
    wall_mass: u.Quantity = SURVIVABLE_WALL_MASS,
    rods_split: int = 1,
    pulse_rate: u.Quantity = SURVIVABLE_PULSE_RATE,
) -> ChamberPairing:
    """The survivable methane chamber at one nozzle area ratio.

    Args:
        area_ratio: 100, 150 (S24's option (a)) or 300.
        wall_mass: The wall of one 5 kg chamber, or of all ``rods_split``.
        rods_split: Split the 5 kg rod's flow across this many smaller
            chambers, each firing ``5 / rods_split`` kg at 2 Hz with the same
            ``k`` and ``P``, sharing ``wall_mass`` evenly, and each with an
            extension of its share of the exit area (parent S17 item 3).
        pulse_rate: Pulses per second; :func:`survivable_cycle` says which fit.

    Returns:
        The pairing; its ``eta`` is :func:`efficiency_for_impulse` at
        :data:`SURVIVABLE_IMPULSE`.
    """
    share = 1.0 / rods_split
    return ChamberPairing(
        f"CH4 survivable AR{area_ratio}"
        + (f" x{rods_split}" if rods_split > 1 else "")
        + (
            f" {pulse_rate.to_value(1 / u.s):g} Hz"
            if pulse_rate != SURVIVABLE_PULSE_RATE
            else ""
        ),
        7000.0 * u.K,
        SURVIVABLE_CARRIED_PER_ROD - SURVIVABLE_PLUG_RATIO,
        (wall_mass * share).to(u.t),
        METHANE_7000K.tank_fraction,
        METHANE_7000K.chemistry_ceiling,
        rod_mass=SURVIVABLE_ROD_MASS * share,
        plug_ratio=SURVIVABLE_PLUG_RATIO,
        pulse_rate=pulse_rate,
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
