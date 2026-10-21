"""A reactor plus chemical rockets at Saturn, against the nuclear rocket (S51 item 5).

The parent's core comparison (Seth, 2026-10-21): (a) the water NTR of
:mod:`src.iapetus_cycle`, one single-use reactor per stage; against (b) one
reactor launched once, kept at Saturn for years, making methalox from ice and
CO2 for many single-use methalox stages.  Only the chemical engines and tanks
are thrown away.

**The plant.** A heat-only core drives a two-step ceria cycle (reduction near
1773 K, reoxidation by H2O and CO2 near 1073 K, methanation, LOX captured): 10%
of the heat to fuel (5% pessimistic).  Or an electric plant at ~1150 K: Brayton
at ~25% and solid-oxide co-electrolysis at ~80%, ~20% to fuel.  Per kg of
methalox at O/F 3.6 the plant makes 13.7 MJ of chemical energy from 0.60 kg of
CO2 (0.163 kg of carbon) and 0.49 kg of net water.

**Growth.** A 10 MWt ceria plant (1 MW of fuel) makes ~2300 t of methalox a
year from ~15 t of plant, so the plant is almost never what limits growth:
the Earth-sourced chemical engines, shipped tanks, kits and skins are.  The
ledger keeps plant capacity and per-stage consumables apart.  A stage leaves
with what the plants made since the last one left; a plant produces from the
day it lands, so a new plant gives only the stay in its first round, and the
ledger ships plants a round ahead whenever capacity would bind.  Plants retire
after their life.

**Sites.** Iapetus (with its dog-leg) and Phoebe (no dog-leg: it orbits at
12.95 million km already), each with the parent's coplanar delta-v and the
real-plane medians from :mod:`src.saturn_moon_legs`.

Everything else (the Earth push, the ledgers, the outbound stack, the gates
and attrition) is :mod:`src.iapetus_cycle`'s.  Prices and masses marked
"assumption" are labelled placeholders, not sourced.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from scipy.optimize import minimize_scalar
from tabulate import tabulate

from src.iapetus_cycle import (
    BAG_MASS,
    DOGLEG_APOAPSIS,
    ESTIMATE,
    G0,
    IAPETUS_A,
    LEDGER_A,
    LEDGER_B,
    LEDGERS,
    MU_SATURN,
    OUT_YEARS,
    OUTBOUND_BOOK,
    RETURN_YEARS,
    SATURN_PERIAPSIS,
    STAY_YEARS,
    CyclePrices,
    Ledger,
    Outbound,
    OutboundBook,
    PushLaw,
    best_k,
    build_cycle,
    dogleg_dvs,
    iapetus_gates,
    outbound,
    outbound_at,
    periapsis_acceleration,
    push_dollars,
    rate_rows,
    seed_ship_price,
    value_cycle,
)
from src.solar_dive_growth import MISS_FLOOR, Gate, GatedSchedule, miss_fraction

_YEAR = 365.25 * 86400.0

#: Methalox chemistry per kg at O/F 3.6 (parent): chemical energy, CO2, water.
METHALOX_ENERGY = 13.7e6  # J/kg
CO2_PER_METHALOX = 0.60
WATER_PER_METHALOX = 0.49
#: Tank walls made at Saturn from CO2 and water (polyethylene-class): ~50-100
#: MJ of electricity per kg (parent), and 0.857 kg of carbon, 3.14 kg of CO2.
TANK_ENERGY = 75.0e6  # J/kg, charged against the plant's fuel output
CO2_PER_TANK = 0.857 * 44.0 / 12.0
#: Carbon fed as elemental or organic carbon (the dark material's hypothesis):
#: 12/44 of the CO2 mass.  Gasified with steam (C + H2O -> CO + H2), shifted
#: and methanated; all the oxygen comes from water, so C + 4 H2O -> CH4 + 2 O2
#: + 2 H2 takes 1068 kJ against CO2 + 2 H2O -> CH4 + 2 O2's 890 (liquid water,
#: the surplus hydrogen not used): 1.20x the energy per kg of methalox.
CARBON_PER_METHALOX = CO2_PER_METHALOX * 12.0 / 44.0
CARBON_PER_TANK = 0.857
CARBON_ENERGY_FACTOR = 1068.0 / 890.0
#: Phoebe (Horizons): semi-major axis, landing and escape from its surface.
PHOEBE_A = 12_947_780.0
PHOEBE_LANDING = 0.10


def _half_period(rp: float, ra: float) -> float:
    a = 0.5 * (rp + ra)
    return math.pi * math.sqrt(a**3 / MU_SATURN) / _YEAR


#: Iapetus's dog-leg, each way: 150 000 km to 20e6 km, then down to Iapetus.
IAPETUS_DOGLEG = _half_period(SATURN_PERIAPSIS, DOGLEG_APOAPSIS) + _half_period(
    IAPETUS_A, DOGLEG_APOAPSIS
)
#: Phoebe's leg, each way: the fall between its orbit and 150 000 km.
PHOEBE_LEG = _half_period(SATURN_PERIAPSIS, PHOEBE_A)


@dataclass(frozen=True)
class Site:
    """A moon the stage loads at.

    Attributes:
        name: Row name.
        outbound_dv: Saturn-side outbound (capture through landing, km/s).
        home_dv: Moon to the Earth-bound periapsis burn, impulsive (km/s).
        departure_excess: Saturn excess of the return (km/s).
        leg_years: The Saturn-side leg each way.
        ice_fraction: Ice by mass in mined material.
        anchoring: Note on what holds the kit down (unpriced).
    """

    name: str
    outbound_dv: float
    home_dv: float
    departure_excess: float
    leg_years: float
    ice_fraction: float
    anchoring: str

    def round_years(self, stay: float = STAY_YEARS) -> float:
        """Out, leg, stay, leg, home."""
        return OUT_YEARS + 2.0 * self.leg_years + stay + RETURN_YEARS


def _coplanar_sites() -> Tuple[Site, Site]:
    d = dogleg_dvs()
    iapetus = Site(
        "Iapetus (coplanar)",
        d["capture"] + d["raise"] + d["match"] + d["land"],
        0.74 + d["drop"] + d["departure"],
        10.4,
        IAPETUS_DOGLEG,
        0.775,
        "gravity 0.22 m/s^2",
    )
    rp, ra = SATURN_PERIAPSIS, PHOEBE_A
    a = 0.5 * (rp + ra)
    vp = math.sqrt(MU_SATURN * (2.0 / rp - 1.0 / a))
    va = math.sqrt(MU_SATURN * (2.0 / ra - 1.0 / a))
    v_phoebe = math.sqrt(MU_SATURN / PHOEBE_A)
    capture = (math.sqrt(10.7**2 + 2.0 * MU_SATURN / rp) - vp) * 1.02
    match = v_phoebe - va
    departure = math.sqrt(10.4**2 + 2.0 * MU_SATURN / rp) - vp
    phoebe = Site(
        "Phoebe (coplanar)",
        capture + match + PHOEBE_LANDING,
        PHOEBE_LANDING + match + departure,
        10.4,
        PHOEBE_LEG,
        0.50,
        "hold-down thrusters (unpriced)",
    )
    return iapetus, phoebe


IAPETUS, PHOEBE = _coplanar_sites()


@lru_cache(maxsize=None)
def real_plane_sites() -> Tuple[Site, Site]:
    """The same sites on :mod:`src.saturn_moon_legs`' medians over 2035-2048."""
    from src.saturn_moon_legs import site_summary

    d = dogleg_dvs()
    i = site_summary("Iapetus")
    p = site_summary("Phoebe")
    iapetus = replace(
        IAPETUS,
        name="Iapetus (real planes)",
        outbound_dv=i["capture"] * 1.02 + i["apse"] + d["match"] + d["land"],
        home_dv=0.74 + i["home_apse"] + i["departure"],
        departure_excess=i["saturn_excess"],
    )
    phoebe = replace(
        PHOEBE,
        name="Phoebe (real planes)",
        outbound_dv=p["capture"] * 1.02 + p["apse"] + PHOEBE_LANDING,
        home_dv=PHOEBE_LANDING + p["home_apse"] + p["departure"],
        departure_excess=p["saturn_excess"],
    )
    return iapetus, phoebe


@dataclass(frozen=True)
class Plant:
    """A reactor that makes methalox at the moon and stays (assumptions marked).

    Attributes:
        name: Row name.
        efficiency: Heat to chemical energy in the fuel.
        unit_mwt: One plant's reactor power (MWt).
        mass_per_mwt: Plant mass per MWt (kg): reactor, the ceria or the
            converter and electrolysers, methanation, LOX capture, radiators or
            the ceramic wall (the parent's 10-20 t per MW of fuel for ceria at
            10%, 15-30 t for the electric plant at 20%).
        fixed_price: Per plant, whatever its power (assumption: the core and
            its criticality-limited U-235 inventory).
        price_per_mwt: Per MWt above that (assumption).
        life_rounds: Rounds a plant runs (assumption: a heat-only core at
            ~2000 K for years is undemonstrated; ~1150 K is the Kilopower /
            SP-100 class).
        waste_usable: Share of the waste heat warm enough to heat regolith.
    """

    name: str
    efficiency: float
    unit_mwt: float
    mass_per_mwt: float
    fixed_price: float
    price_per_mwt: float
    life_rounds: float
    waste_usable: float = 0.5

    @property
    def unit_fuel_mw(self) -> float:
        """One plant's fuel output (MW)."""
        return self.unit_mwt * self.efficiency

    @property
    def mass_per_mw(self) -> float:
        """Plant mass per MW of fuel output (kg)."""
        return self.mass_per_mwt / self.efficiency

    @property
    def price_per_mw(self) -> float:
        """Plant price per MW of fuel output, at this unit size."""
        return (self.fixed_price / self.unit_mwt + self.price_per_mwt) / self.efficiency


CERIA = Plant("ceria (heat-only core ~2000 K)", 0.10, 10.0, 1.5e3, 50.0e6, 15.0e6, 1.0)
CERIA_PESSIMISTIC = replace(CERIA, name="ceria at 5%", efficiency=0.05)
ELECTRIC = Plant(
    "electric (Brayton + SOEC, ~1150 K)", 0.20, 10.0, 4.5e3, 50.0e6, 30.0e6, 2.0
)


@dataclass(frozen=True)
class ChemStage:
    """The single-use methalox stage (assumptions marked).

    Attributes:
        isp: Specific impulse (s): 370 base, 380 optimistic (Raptor Vacuum).
        dry: Dry mass, engines included, per kg of propellant (8-12%).
        engine_mass: Per engine (kg): Raptor Vacuum class, ~2.5 MN.
        engine_thrust: Per engine (kN).
        engine_price: Per engine (assumption: $1M at scale).
        burn_hours: The periapsis burn; sets the engine count (1 h: ~0.4%
            loss, so the burns are taken impulsive).
        divert: The empty stage's divert off the Earth-bound arc (km/s).
        tanks_shipped: Tanks from Earth, or made at Saturn by the plant
            (base: shipped, they are ~0.22 kg per kg of PuffSat and hold
            growth near 1.3-1.9).
        reuse: Brake the empty stage back (2.89 km/s) and refly its engines
            and tanks for ``reuse_rounds``.
        reuse_rounds: Rounds an engine and tank fly when reused (assumption).
    """

    isp: float = 370.0
    dry: float = 0.10
    engine_mass: float = 1600.0
    engine_thrust: float = 2500.0
    engine_price: float = 1.0e6
    burn_hours: float = 1.0
    divert: float = 0.050
    tanks_shipped: bool = False
    reuse: bool = False
    reuse_rounds: float = 5.0


CHEM = ChemStage()


@dataclass(frozen=True)
class Kits:
    """Mining and carbon processing (assumptions marked).

    Attributes:
        ice_kit: Mining and launch kit per kg of material mined for water.
        carbon_kit: Carbon kit (dig, heat, gas cleanup) per kg of regolith
            handled (the mining kit's 0.3 kg/t to start, as the parent asks).
        carbon_form: ``carbon`` (base: elemental or organic carbon in the dark
            material, gasified with steam) or ``CO2`` (CO2 ice).
        carbon_grade: That feed by mass in mined material.
        regolith_heat: Heat per kg of regolith, or None for the form's own:
            ~0.4 MJ to release CO2 (~500 K at 0.8 kJ/kg/K), ~0.75 MJ to take it
            to ~1000 K for steam gasification (assumptions).
        skins: PuffSat skins and electronics per kg of PuffSat.
    """

    ice_kit: float = 0.3e-3
    carbon_kit: float = 0.3e-3
    carbon_form: str = "carbon"
    carbon_grade: float = 0.01
    regolith_heat: Optional[float] = None
    skins: float = 0.05

    @property
    def feed_per_methalox(self) -> float:
        """Feed (carbon or CO2) per kg of methalox."""
        return CARBON_PER_METHALOX if self.carbon_form == "carbon" else CO2_PER_METHALOX

    @property
    def feed_per_tank(self) -> float:
        """Feed per kg of tank wall made at Saturn."""
        return CARBON_PER_TANK if self.carbon_form == "carbon" else CO2_PER_TANK

    @property
    def energy(self) -> float:
        """Plant output per kg of methalox (J)."""
        return METHALOX_ENERGY * (
            CARBON_ENERGY_FACTOR if self.carbon_form == "carbon" else 1.0
        )

    @property
    def heat(self) -> float:
        """Heat per kg of regolith (J)."""
        if self.regolith_heat is not None:
            return self.regolith_heat
        return 0.75e6 if self.carbon_form == "carbon" else 0.4e6


KITS = Kits()


@dataclass(frozen=True)
class StageUnit:
    """Per kg of PuffSat released: the stage's masses (kg).

    Attributes:
        methalox: Propellant, divert (and brake, if reused) included.
        engines: Engine count.
        engine_mass: Their mass.
        tanks: The rest of the dry mass.
        water: Ice-water needed: methalox's net water plus the PuffSat.
        regolith: Material handled for carbon.
        tank_energy: Plant output spent making the tanks at Saturn (J).
    """

    methalox: float
    engines: float
    engine_mass: float
    tanks: float
    water: float
    regolith: float
    tank_energy: float = 0.0


def stage_unit(site: Site, stage: ChemStage = CHEM, kits: Kits = KITS) -> StageUnit:
    """Size the methalox stage per kg of PuffSat.

    The stack ``1 + D + M`` flies ``home_dv`` with the PuffSat; after it
    separates, the empty stage diverts (or brakes back, if reused).  Dry mass
    is ``dry`` of all propellant.  Engines: enough for a ``burn_hours``
    periapsis burn at its start acceleration.
    """
    ve = stage.isp * G0
    after = stage.divert if not stage.reuse else 2.89
    r_main = math.exp(site.home_dv / ve)
    r_after = math.exp(after / ve)
    # D = dry (M_main + M_after); M_after = D (r_after - 1);
    # 1 + D + M_after + M_main = r_main (1 + D + M_after)
    # => M_main = (r_main - 1)(1 + D r_after)
    f = stage.dry
    d = f * (r_main - 1.0) / (1.0 - f * (r_after - 1.0) - f * (r_main - 1.0) * r_after)
    if d <= 0.0:
        raise ValueError("the methalox stage does not close")
    m_after = d * (r_after - 1.0)
    m_main = (r_main - 1.0) * (1.0 + d * r_after)
    methalox = m_main + m_after
    a0 = periapsis_acceleration(stage.burn_hours, ve, site.departure_excess)
    at_peri = (1.0 + d + m_after) * math.exp(_periapsis_dv(site.departure_excess) / ve)
    engines = a0 * at_peri / stage.engine_thrust
    engine_mass = engines * stage.engine_mass
    tanks = max(d - engine_mass, 0.0)
    feed = kits.feed_per_methalox * methalox + (
        0.0 if stage.tanks_shipped else kits.feed_per_tank * tanks
    )
    return StageUnit(
        methalox=methalox,
        engines=engines,
        engine_mass=engine_mass,
        tanks=tanks,
        water=WATER_PER_METHALOX * methalox + 1.0,
        regolith=feed / kits.carbon_grade,
        tank_energy=0.0 if stage.tanks_shipped else TANK_ENERGY * tanks,
    )


def _periapsis_dv(excess: float) -> float:
    rp, ra = SATURN_PERIAPSIS, DOGLEG_APOAPSIS
    return math.sqrt(excess**2 + 2.0 * MU_SATURN / rp) - math.sqrt(
        MU_SATURN * (2.0 / rp - 2.0 / (rp + ra))
    )


def methalox_per_mw_year(plant: Plant, unit: StageUnit, kits: Kits = KITS) -> float:
    """Methalox (kg) a MW of fuel output makes in a year, its other work charged.

    Tanks made at Saturn take their energy out of the fuel.  Heating the
    regolith takes ``regolith_heat`` per kg; the plant's usable waste heat
    pays first, and any shortfall is drawn from the reactor.
    """
    chemical = kits.energy + unit.tank_energy / unit.methalox
    per_kg_heat = chemical / plant.efficiency
    waste = plant.waste_usable * (per_kg_heat - chemical)
    regolith = kits.heat * unit.regolith / unit.methalox
    shortfall = max(regolith - waste, 0.0)
    return 1.0e6 / plant.efficiency * _YEAR / (per_kg_heat + shortfall)


@dataclass(frozen=True)
class PlantCycle:
    """A plant-and-stage cycle at one site, ready to value.

    Attributes:
        site: The moon.
        plant: The plant.
        stage: The methalox stage.
        kits: Kits and carbon grade.
        law: The Earth push.
        k: Plug per kg of bag.
        ledger: Ledger A or B.
        prices: The book.
        out: Outbound from the eccentric orbit, per kg landed.
        out_leo: The seed's, from LEO.
        stay: The stay at the moon.
        seed_wait: Years the first stage waits at the moon while the seed
            plant stockpiles; every later event moves with it.
    """

    site: Site
    plant: Plant
    stage: ChemStage
    kits: Kits
    law: PushLaw
    k: float
    ledger: Ledger
    prices: CyclePrices
    out: Outbound
    out_leo: Outbound
    stay: float = STAY_YEARS
    seed_wait: float = 0.0

    @property
    def unit(self) -> StageUnit:
        """The stage per kg of PuffSat."""
        return stage_unit(self.site, self.stage, self.kits)

    @property
    def round_years(self) -> float:
        """The round."""
        return self.site.round_years(self.stay)

    @property
    def puffsat_per_mw_round(self) -> float:
        """PuffSat (kg) a MW of fuel output supports over a whole round."""
        u = self.unit
        return (
            methalox_per_mw_year(self.plant, u, self.kits)
            * self.round_years
            / u.methalox
        )

    def consumables(self) -> Dict[str, float]:
        """Earth-sourced kg per kg of PuffSat, by line."""
        u, s, k = self.unit, self.stage, self.kits
        reuse = 1.0 / s.reuse_rounds if s.reuse else 1.0
        return {
            "chemical engines": u.engine_mass * reuse,
            "tanks": (u.tanks if s.tanks_shipped else 0.0) * reuse,
            "ice kit": k.ice_kit * u.water / self.site.ice_fraction,
            "carbon kit": k.carbon_kit * u.regolith,
            "skins": k.skins,
        }

    @property
    def earth_per_puffsat(self) -> float:
        """Earth-sourced kg per kg of PuffSat."""
        return sum(self.consumables().values())

    @property
    def useful(self) -> float:
        """Kg pushed per kg of bag, less the plate and spring."""
        return self.law.leverage(self.k) * (1.0 - self.law.plate_share(self.k))

    def consumable_dollars(self, puffsat: float) -> Dict[str, float]:
        """Dollars of consumables, propellant and outbound for ``puffsat`` kg."""
        p, u, s, o = self.prices, self.unit, self.stage, self.out
        c = self.consumables()
        reuse = 1.0 / s.reuse_rounds if s.reuse else 1.0
        earth = sum(c.values()) * puffsat
        lines = {
            "chemical engines": s.engine_price * u.engines * reuse * puffsat,
            "tanks": p.stage_hardware * c["tanks"] * puffsat,
            "kits": p.kit * (c["ice kit"] + c["carbon kit"]) * puffsat,
            "skins": p.skin * c["skins"] * puffsat,
            "outbound hydrogen NTP engines": p.engine * o.ntp_engines * earth,
            "Earth departure propellant": self.earth_propellant_price(o)
            * o.earth_propellant
            * earth,
            "stage hardware": p.stage_hardware * (o.earth_dry + o.methalox_dry) * earth,
            "methalox (outbound)": p.methalox * o.methalox * earth,
        }
        if self.ledger.electronics is None:
            lines["electronics"] = (
                p.package.floor * puffsat / BAG_MASS / p.bags_per_package
            )
        return lines

    def earth_propellant_price(self, o: Outbound) -> float:
        """Per kg of the Earth departure's propellant."""
        return (
            self.prices.hydrogen
            if o.earth_stage == "hydrogen NTP"
            else self.prices.methalox
        )

    def plant_dollars(self, mw: float) -> Dict[str, float]:
        """Dollars for ``mw`` of new plant landed (plant and its outbound)."""
        p, o = self.prices, self.out
        earth = self.plant.mass_per_mw * mw
        return {
            "plant": self.plant.price_per_mw * mw,
            "plant outbound": p.engine * o.ntp_engines * earth
            + self.earth_propellant_price(o) * o.earth_propellant * earth
            + p.stage_hardware * (o.earth_dry + o.methalox_dry) * earth,
        }

    def growth(self) -> float:
        """Consumable-limited growth per round, miss at its floor."""
        return (
            (1.0 - MISS_FLOOR) * self.useful / (self.out.ratio * self.earth_per_puffsat)
        )


def build_plant_cycle(
    site: Site = IAPETUS,
    plant: Plant = CERIA,
    ledger: Ledger = LEDGER_A,
    k: float = 4.0,
    law: PushLaw = PushLaw(),
    stage: ChemStage = CHEM,
    kits: Kits = KITS,
    prices: CyclePrices = ESTIMATE,
    stay: float = STAY_YEARS,
    book: OutboundBook = OUTBOUND_BOOK,
    seed_wait: Optional[float] = None,
) -> PlantCycle:
    """Assemble a plant cycle, its outbound loss and seed wait chosen on the rNPV.

    The outbound stack carries the site's Saturn-side delta-v.  As for the
    water cycle, the hydrogen Earth departure's loss is picked from
    ``book.losses`` for the lowest gated break-even, and so is the first
    stage's wait at the moon (0, 1, 2, 4 or 9.5 years) unless given.  The
    seed's LEO departure flies the book's Earth stage (methalox: two stages).
    """
    prices_book = (prices.engine, prices.hydrogen, prices.stage_hardware)
    out_leo = outbound(True, book, prices.leo, *prices_book, saturn_dv=site.outbound_dv)
    losses: Sequence[Optional[float]] = (
        book.losses if book.earth_stage == "hydrogen NTP" else (None,)
    )
    waits = SEED_WAITS if seed_wait is None else (seed_wait,)
    best: Optional[Tuple[float, PlantCycle]] = None
    for loss in losses:
        try:
            out = outbound_at(False, book, loss, site.outbound_dv)
        except ValueError:
            continue
        for wait in waits:
            c = PlantCycle(
                site,
                plant,
                stage,
                kits,
                law,
                k,
                ledger,
                prices,
                out,
                out_leo,
                stay,
                wait,
            )
            if c.growth() <= 1.0 and best is not None:
                continue
            v = value_plant_cycle(c)
            score = v.break_even if v.break_even is not None else math.inf
            if best is None or score < best[0]:
                best = (score, c)
    assert best is not None
    return best[1]


#: The first stage's waits at the moon the ledger chooses among (years).
SEED_WAITS = (0.0, 1.0, 2.0, 4.0, 9.5)


def plant_gates(site: Site, stay: float, wait: float = 0.0) -> Tuple[Gate, ...]:
    """S51's gates with the plant's in place of the water stage's (90%, placeholder)."""
    capture = OUT_YEARS + site.leg_years
    departure = capture + stay + wait
    round_years = site.round_years(stay) + wait
    return (
        Gate("captures at Saturn, lands plant and kit", capture, 0.95),
        Gate(
            "plant runs at Saturn and fuels the first stage (placeholder)",
            departure,
            0.90,
        ),
        Gate(
            "PuffSats coast and home onto the Earth bag plate", round_years - 1e-6, 0.85
        ),
        Gate("first Earth push with the ship-carried plug", round_years, 0.90),
    )


@dataclass(frozen=True)
class PlantValuation:
    """A plant cycle valued on the gated schedule.

    Attributes:
        growth: Consumable-limited growth per round.
        round_years: The round.
        break_even: PV $ over PV kg at the eccentric orbit.
        steady_cost: Undiscounted $/kg in a level round.
        steady_lines: The same by line.
        stages_per_mw_life: 826 t-PuffSat stages a MW of fuel output fuels in
            its life.
        plant_per_kg: The plant line, $/kg pushed, in the level round.
        seed: Seed dollars.
    """

    growth: float
    round_years: float
    break_even: Optional[float]
    steady_cost: float
    steady_lines: Dict[str, float]
    stages_per_mw_life: float
    plant_per_kg: float
    seed: float


#: The water stage's 12-hour base, for counting stages of the same size.
WATER_STAGE_PUFFSAT = 826.0e3
REINVESTMENTS = 2
HORIZON_YEARS = 300.0


def _overlap(born: int, stage: int, c: PlantCycle) -> float:
    """Years a plant shipped at round ``born`` runs while round ``stage``'s
    stage is being fuelled (from the last departure to this one).

    The seed plant lands on schedule; the first stage leaves ``seed_wait``
    late, and every later shipment and departure moves with it.
    """
    T = c.round_years
    arrive = OUT_YEARS + c.site.leg_years
    lands = born * T + arrive + (c.seed_wait if born > 0 else 0.0)
    retires = lands + c.plant.life_rounds * T
    departs = stage * T + arrive + c.stay + c.seed_wait
    window = (departs - T if stage > 0 else -math.inf, departs)
    return max(min(window[1], retires) - max(window[0], lands), 0.0)


def value_plant_cycle(
    c: PlantCycle, rate: float = 0.10, reinvestments: int = REINVESTMENTS
) -> PlantValuation:
    """Grow from a seed plant, then hold level; break-even $/kg at the eccentric orbit.

    Whole rounds.  The seed (one plant, and the consumables for
    what it makes during the first stay) leaves LEO at year 0.  Round ``r``'s
    stage is fuelled by every plant's run since round ``r - 1``'s left
    (:func:`_overlap`).  At each return the useful pushed mass buys, in order:
    the consumables for this round's stage, capped by that fuel; plant enough
    that the next round's fuel covers the consumable-limited growth (or, held
    level, replaces what retires); then, while growing, more plant, and once
    held level, cargo.
    """
    schedule = GatedSchedule(rate, plant_gates(c.site, c.stay, c.seed_wait))
    T = c.round_years
    per_mw_year = c.puffsat_per_mw_round / T  # PuffSat per MW-year of fuel
    earth = c.earth_per_puffsat
    out_ratio = c.out.ratio
    plant_kg = c.plant.mass_per_mw
    g_cons = c.growth()
    plants: List[Tuple[int, float]] = [(0, c.plant.unit_fuel_mw)]

    def fuel(stage: int) -> float:
        return per_mw_year * sum(mw * _overlap(b, stage, c) for b, mw in plants)

    p0 = fuel(0)
    seed_lines = dict(c.consumable_dollars(p0))
    seed_lines.update(c.plant_dollars(c.plant.unit_fuel_mw))
    o = c.out_leo
    seed_earth = earth * p0 + plant_kg * c.plant.unit_fuel_mw
    if o.earth_stage == "seed ship":
        seed_lines["seed ship"] = (
            seed_ship_price(c.prices.dear_seed) * o.ratio * seed_earth
        )
        seed_lines["stage hardware"] = (
            c.prices.stage_hardware * o.methalox_dry * seed_earth
        )
        seed_lines["outbound hydrogen NTP engines"] = 0.0
        seed_lines["Earth departure propellant"] = 0.0
    else:
        seed_lines["outbound hydrogen NTP engines"] = (
            c.prices.engine * o.ntp_engines * seed_earth
        )
        seed_lines["Earth departure propellant"] = (
            c.earth_propellant_price(o) * o.earth_propellant * seed_earth
        )
        seed_lines["stage hardware"] = (
            c.prices.stage_hardware * (o.earth_dry + o.methalox_dry) * seed_earth
        )
        seed_lines["launch to LEO"] = c.prices.leo * seed_earth * o.ratio
    seed_lines["plant outbound"] = 0.0  # carried in the seed's own lines
    seed = sum(seed_lines.values())
    pv_cost, pv_cargo = seed, 0.0
    produced = p0
    level: Optional[float] = None
    cumulative = 0.0
    steady: Tuple[float, Dict[str, float]] = (math.inf, {})
    r = 1
    while r * T <= HORIZON_YEARS:
        cumulative += produced
        bags = produced * (1.0 - miss_fraction(cumulative, max(p0, 1e-9)))
        budget = bags * c.useful / out_ratio  # Earth kg it can land
        if level is None and r > reinvestments:
            level = produced
        # This round's stage (fuel already on site, consumables bought now)
        # and the plant that fuels the next one, solved together: the stage
        # ``x`` needs ``x earth`` and the next round's fuel ``g x`` (or the
        # level) needs ``coef (want - existing)`` of plant.
        plants.append((r, 0.0))
        existing = fuel(r + 1)
        per_new = per_mw_year * _overlap(r, r + 1, c)
        coef = plant_kg / per_new
        cap = fuel(r) if level is None else min(fuel(r), level)
        if level is None:
            x = (budget + coef * existing) / (earth + coef * g_cons)
            if g_cons * x <= existing:
                x = budget / earth
        else:
            x = (budget - coef * max(level - existing, 0.0)) / earth
        stage_now = max(min(x, cap), 0.0)
        want = stage_now * g_cons if level is None else level
        new_mw = min(max(want - existing, 0.0) / per_new, budget / plant_kg)
        budget -= new_mw * plant_kg
        consumable = min(stage_now * earth, max(budget, 0.0))
        stage_now = consumable / earth
        budget -= consumable
        cargo = 0.0
        if level is None:
            new_mw += budget / plant_kg  # surplus while growing: more plant
        else:
            cargo = budget * out_ratio
        plants[-1] = (r, new_mw)
        lines = dict(push_dollars(c.law, c.k, c.ledger, c.prices, bags))
        for group in (c.consumable_dollars(stage_now), c.plant_dollars(new_mw)):
            for key, value in group.items():
                lines[key] = lines.get(key, 0.0) + value
        factor = schedule.factor(r * T + c.seed_wait)
        pv_cost += factor * sum(lines.values())
        pv_cargo += factor * cargo
        if (
            level is not None
            and cargo > 0.0
            and r >= reinvestments + 4
            and not steady[1]
        ):
            steady = (
                sum(lines.values()) / cargo,
                {k_: v / cargo for k_, v in lines.items()},
            )
        produced = stage_now
        r += 1
    stages = c.puffsat_per_mw_round * c.plant.life_rounds / WATER_STAGE_PUFFSAT
    return PlantValuation(
        growth=g_cons,
        round_years=T,
        break_even=pv_cost / pv_cargo if pv_cargo > 0.0 else None,
        steady_cost=steady[0],
        steady_lines=steady[1],
        stages_per_mw_life=stages,
        plant_per_kg=steady[1].get("plant", 0.0),
        seed=seed,
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _m(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value) or value > 1.0e6:
        return "does not grow"
    return f"{value:,.0f}" if value >= 100 else f"{value:.2f}"


def stage_report() -> str:
    """The methalox stage per kg of PuffSat at each site."""
    rows = []
    for site in (IAPETUS, PHOEBE) + real_plane_sites():
        for stage in (
            CHEM,
            replace(CHEM, isp=380.0, dry=0.08),
            replace(CHEM, dry=0.12),
        ):
            u = stage_unit(site, stage)
            rows.append(
                [
                    site.name,
                    f"{stage.isp:.0f} s, {stage.dry:.0%}",
                    f"{site.outbound_dv:.2f}",
                    f"{site.home_dv:.2f}",
                    f"{u.methalox:.2f}",
                    f"{u.methalox * 826:,.0f}",
                    f"{KITS.feed_per_methalox * u.methalox * 826:,.0f}",
                    f"{u.methalox * 826e3 * KITS.energy / 1e12:.0f}",
                    f"{u.engines * 826e3:.1f}",
                    f"{u.tanks:.3f}",
                ]
            )
    return (
        "The methalox stage per kg of PuffSat (divert 50 m/s; engines for a 1 h periapsis burn).\n"
        "Parent: Iapetus 3.70 km/s, 1,620-1,860 t of methalox per 826 t; Phoebe 3.97, 1,850-2,160 t.\n"
        + tabulate(
            rows,
            headers=[
                "site",
                "stage",
                "Saturn out km/s",
                "home km/s",
                "methalox/kg",
                "t per 826 t",
                "carbon t",
                "TJ",
                "engines",
                "tanks/kg",
            ],
        )
    )


def _site_rows(
    sites: Sequence[Site],
    plants: Sequence[Plant],
    ledgers: Sequence[Ledger],
    book: OutboundBook = OUTBOUND_BOOK,
    stage: ChemStage = CHEM,
) -> List[List[str]]:
    rows = []
    for site in sites:
        for plant in plants:
            for ledger in ledgers:
                c = build_plant_cycle(site, plant, ledger, stage=stage, book=book)
                v = value_plant_cycle(c)
                loan = value_plant_cycle(c, 0.04)
                r = rate_rows(c.growth(), c.round_years)
                rows.append(
                    [
                        site.name
                        + ("" if not stage.tanks_shipped else ", tanks shipped"),
                        plant.name.split(" (")[0],
                        ledger.label[0],
                        f"{c.earth_per_puffsat:.3f}",
                        f"{c.growth():.2f}",
                        f"{c.round_years:.2f}",
                        f"{r['doubling']:.2f}",
                        _m(v.steady_cost),
                        _m(v.break_even),
                        _m(loan.break_even),
                        f"{v.stages_per_mw_life:.0f}",
                        (
                            f"{v.plant_per_kg:.2f}"
                            if math.isfinite(v.steady_cost)
                            else "-"
                        ),
                    ]
                )
    return rows


def comparison_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Item 5: water NTR against methalox at Iapetus and Phoebe, both ledgers."""
    rows = []
    for real in (False, True):
        for ledger in LEDGERS:
            k, v = best_k(ledger, book=book, real_planes=real)
            c = build_cycle(ledger, k, book=book, real_planes=real)
            loan = value_cycle(c, 0.04)
            r = rate_rows(c.growth(), 9.5)
            rows.append(
                [
                    "Iapetus (real planes)" if real else "Iapetus (coplanar)",
                    "water NTR",
                    ledger.label[0],
                    f"{1.0 / c.stage.puffsat_per_earth_kg:.3f}",
                    f"{c.growth():.2f}",
                    "9.50",
                    f"{r['doubling']:.2f}",
                    _m(v.steady_cost),
                    _m(v.break_even),
                    _m(loan.break_even),
                    "1 (single use)",
                    f"{v.steady_lines.get('water engine', 0.0):.2f}",
                ]
            )
    rows += _site_rows(
        (IAPETUS, PHOEBE) + real_plane_sites(), (CERIA, ELECTRIC), LEDGERS, book
    )
    rows += _site_rows(
        (IAPETUS, PHOEBE),
        (CERIA,),
        (LEDGER_A,),
        book,
        replace(CHEM, tanks_shipped=True),
    )
    return (
        "Item 5. Nuclear rocket against reactor plus chemical rockets, $/kg at the eccentric orbit\n"
        f"(Earth departure {book.earth_stage}; k 4 compact, 1% carbon gasified, tanks made at Saturn unless\n"
        "marked, gated rNPV 10% / 4% loan, whole rounds).\n"
        "Reactor $/kg is the water engine line, or the plant line in a level round.\n"
        + tabulate(
            rows,
            headers=[
                "site",
                "Saturn propulsion",
                "ledger",
                "Earth kg per kg PuffSat",
                "G",
                "T yr",
                "doubling",
                "steady",
                "rNPV 10%",
                "loan 4%",
                "826 t stages per plant MW life",
                "reactor $/kg",
            ],
        )
    )


def lines_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """The steady lines of the methalox base at Iapetus and Phoebe (Ledger A)."""
    out = []
    for site in (IAPETUS, PHOEBE):
        c = build_plant_cycle(site, CERIA, LEDGER_A, book=book)
        v = value_plant_cycle(c)
        u = c.unit
        lines = ", ".join(
            f"{k} {x:.2f}"
            for k, x in sorted(v.steady_lines.items(), key=lambda kv: -kv[1])
            if x >= 0.01
        )
        mw_year = methalox_per_mw_year(c.plant, u, c.kits) / 1e3
        out.append(
            f"{site.name}, ceria, Ledger A: steady ${v.steady_cost:.2f}/kg: {lines}\n"
            f"  consumables per kg PuffSat: "
            + ", ".join(f"{k} {x:.4f}" for k, x in c.consumables().items())
            + f"\n  a MW of fuel (10 MWt ceria) makes {mw_year:,.0f} t of methalox a year, "
            f"{c.puffsat_per_mw_round / 1e3:,.0f} t of PuffSat a round "
            f"({c.puffsat_per_mw_round / WATER_STAGE_PUFFSAT:.1f} stages of 826 t); "
            f"at 5-10 MWt that is {0.5 * c.puffsat_per_mw_round / WATER_STAGE_PUFFSAT:.0f}-"
            f"{c.puffsat_per_mw_round / WATER_STAGE_PUFFSAT:.0f} stages a round"
        )
    return "\n\n".join(out)


def carbon_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Carbon grade: the open gate, and where the kit stops paying."""
    rows = []
    for site in (IAPETUS, PHOEBE):
        for form in ("carbon", "CO2"):
            for grade in (0.001, 0.003, 0.01, 0.03, 0.10):
                for kit in (0.1e-3, 0.3e-3, 1.0e-3):
                    kits = replace(
                        KITS, carbon_form=form, carbon_grade=grade, carbon_kit=kit
                    )
                    c = build_plant_cycle(site, CERIA, LEDGER_A, kits=kits, book=book)
                    v = value_plant_cycle(c)
                    u = c.unit
                    rows.append(
                        [
                            site.name,
                            form,
                            f"{grade:.1%}",
                            f"{kit * 1e3:.1f}",
                            f"{u.regolith * 826e3 / 1e6:,.2f}",
                            f"{c.consumables()['carbon kit']:.3f}",
                            f"{methalox_per_mw_year(c.plant, u, kits) / 1e3:,.0f}",
                            f"{c.growth():.2f}",
                            _m(v.steady_cost),
                            _m(v.break_even),
                        ]
                    )
    return (
        "Carbon grade by mass, as carbon (gasified with steam: 0.163 kg a kg of methalox, 1.2x the\n"
        "energy) or as CO2 (0.60 kg), ceria, Ledger A. The water stage needs none.\n"
        + tabulate(
            rows,
            headers=[
                "site",
                "feed",
                "grade",
                "kit kg/t",
                "Mt regolith per 826 t",
                "carbon kit/kg PuffSat",
                "t/MW-yr",
                "G",
                "steady",
                "rNPV",
            ],
        )
    )


def power_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """One plant's power, 1-20 MWt and 100 MWt: ceria against electric, Iapetus and Phoebe.

    The plant is bought by the unit: a fixed price (the core and its
    criticality inventory) plus a price per MWt, so a larger unit dilutes
    the fixed part, and the seed is one unit.
    """
    rows = []
    for mwt in (1.0, 2.0, 5.0, 10.0, 20.0, 100.0):
        row = [f"{mwt:g}"]
        for plant in (CERIA, ELECTRIC):
            unit = replace(plant, unit_mwt=mwt)
            for site in (IAPETUS, PHOEBE):
                c = build_plant_cycle(site, unit, LEDGER_A, book=book)
                v = value_plant_cycle(c)
                if site is IAPETUS:
                    row.append(
                        f"{c.puffsat_per_mw_round * unit.unit_fuel_mw / WATER_STAGE_PUFFSAT:.1f}"
                    )
                row += [_m(v.steady_cost), _m(v.break_even)]
        rows.append(row)
    return (
        "One plant's power (Ledger A): fixed $50M a plant plus $15M/MWt ceria, $30M/MWt electric\n"
        "(assumptions); stages are 826 t-PuffSat equivalents a round at Iapetus.\n"
        + tabulate(
            rows,
            headers=[
                "MWt",
                "ceria stages/round",
                "ceria I steady",
                "ceria I rNPV",
                "ceria P steady",
                "ceria P rNPV",
                "electric stages/round",
                "electric I steady",
                "electric I rNPV",
                "electric P steady",
                "electric P rNPV",
            ],
        )
    )


@dataclass(frozen=True)
class Sensitivity:
    """One row.

    Attributes:
        label: What moves.
        kwargs: Arguments to :func:`build_plant_cycle`.
    """

    label: str
    kwargs: Dict[str, object] = field(default_factory=dict)


SENSITIVITIES = (
    Sensitivity("base (Iapetus coplanar, ceria, Ledger A)"),
    Sensitivity("Phoebe", {"site": PHOEBE}),
    Sensitivity("electric plant", {"plant": ELECTRIC}),
    Sensitivity("ceria at 5%", {"plant": CERIA_PESSIMISTIC}),
    Sensitivity(
        "plant $500M fixed, 10 MWt", {"plant": replace(CERIA, fixed_price=500.0e6)}
    ),
    Sensitivity("plant life 2 rounds", {"plant": replace(CERIA, life_rounds=2.0)}),
    Sensitivity("380 s, 8% dry", {"stage": replace(CHEM, isp=380.0, dry=0.08)}),
    Sensitivity("12% dry", {"stage": replace(CHEM, dry=0.12)}),
    Sensitivity(
        "tanks shipped from Earth", {"stage": replace(CHEM, tanks_shipped=True)}
    ),
    Sensitivity(
        "reuse: brake back, refly 5 rounds", {"stage": replace(CHEM, reuse=True)}
    ),
    Sensitivity("engine $5M", {"stage": replace(CHEM, engine_price=5.0e6)}),
    Sensitivity("Ledger B", {"ledger": LEDGER_B}),
    Sensitivity("shaped-plug ceiling (k 5.7)", {"law": PushLaw("shaped"), "k": 5.7}),
    Sensitivity("stay 0.1 yr (load and leave)", {"stay": 0.1}),
    Sensitivity("Phoebe, stay 0.1 yr", {"site": PHOEBE, "stay": 0.1}),
)


def sensitivity_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """One line moved at a time."""
    rows = []
    for s in SENSITIVITIES:
        c = build_plant_cycle(**s.kwargs, book=book)  # type: ignore[arg-type]
        v = value_plant_cycle(c)
        loan = value_plant_cycle(c, 0.04)
        rows.append(
            [
                s.label,
                f"{c.earth_per_puffsat:.3f}",
                f"{c.growth():.1f}",
                f"{c.round_years:.2f}",
                _m(v.steady_cost),
                _m(v.break_even),
                _m(loan.break_even),
            ]
        )
    return "Sensitivities (methalox):\n" + tabulate(
        rows,
        headers=[
            "row",
            "Earth kg/kg PuffSat",
            "G",
            "T",
            "steady",
            "rNPV 10%",
            "loan 4%",
        ],
    )


def rate_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Item 6: growth rate by site and propulsion, pipelined and whole rounds."""
    rows = []
    k, _ = best_k(LEDGER_A, book=book)
    water = build_cycle(LEDGER_A, k, book=book)
    water_real = build_cycle(LEDGER_A, k, book=book, real_planes=True)
    entries: List[Tuple[str, float, float]] = [
        ("water NTR, Iapetus (coplanar)", water.growth(), 9.5),
        ("water NTR, Iapetus (real planes)", water_real.growth(), 9.5),
    ]
    for site in (IAPETUS, PHOEBE) + real_plane_sites():
        for stay in (STAY_YEARS, 0.1):
            c = build_plant_cycle(site, CERIA, LEDGER_A, stay=stay, book=book)
            entries.append(
                (f"methalox, {site.name}, stay {stay:g}", c.growth(), c.round_years)
            )
    for label, g, t in entries:
        r = rate_rows(g, t)
        rows.append(
            [
                label,
                f"{g:.2f}",
                f"{t:.2f}",
                f"{r['rate']:.3f}",
                f"{r['doubling']:.2f}",
                f"x{r['pipelined_19']:,.0f}",
                f"x{r['whole_19']:,.0f} ({r['whole_rounds']:.0f})",
            ]
        )
    return (
        "Item 6. Growth rate ln(G)/T (Ledger A, compact k 4); the gated rNPV counts whole rounds.\n"
        + tabulate(
            rows,
            headers=[
                "route",
                "G",
                "T",
                "ln(G)/T",
                "doubling",
                "19 yr pipelined",
                "whole rounds in 20 yr",
            ],
        )
    )


def flip_report(book: OutboundBook = OUTBOUND_BOOK) -> str:
    """Where Phoebe's preference over Iapetus flips: carbon grade and ice fraction."""
    rows = []
    for grade in (0.001, 0.003, 0.01, 0.03, 0.1):
        for ice in (0.25, 0.5):
            kits = replace(KITS, carbon_grade=grade)
            i = build_plant_cycle(IAPETUS, CERIA, LEDGER_A, kits=kits, book=book)
            p = build_plant_cycle(
                replace(PHOEBE, ice_fraction=ice), CERIA, LEDGER_A, kits=kits, book=book
            )
            ri, rp = rate_rows(i.growth(), i.round_years), rate_rows(
                p.growth(), p.round_years
            )
            vi, vp = value_plant_cycle(i), value_plant_cycle(p)
            rows.append(
                [
                    f"{grade:.1%}",
                    f"{ice:.0%}",
                    f"{ri['rate']:.3f} / {rp['rate']:.3f}",
                    "Phoebe" if rp["rate"] > ri["rate"] else "Iapetus",
                    f"{_m(vi.break_even)} / {_m(vp.break_even)}",
                    (
                        "Phoebe"
                        if (vp.break_even or 1e9) < (vi.break_even or 1e9)
                        else "Iapetus"
                    ),
                ]
            )
    return (
        "Iapetus against Phoebe (same carbon grade at both; Phoebe's ice fraction varied):\n"
        + tabulate(
            rows,
            headers=[
                "carbon",
                "Phoebe ice",
                "ln(G)/T I / P",
                "rate prefers",
                "rNPV I / P",
                "rNPV prefers",
            ],
        )
    )


def target_report(target: float = 2.0, book: OutboundBook = OUTBOUND_BOOK) -> str:
    """What the cheapest methalox cell needs to reach the $1-2 target."""
    best: Optional[Tuple[float, str, PlantValuation]] = None
    for site in (IAPETUS, PHOEBE):
        for plant in (CERIA, ELECTRIC):
            c = build_plant_cycle(site, plant, LEDGER_A, book=book)
            v = value_plant_cycle(c, 0.04)
            if v.break_even is not None and (best is None or v.break_even < best[0]):
                best = (v.break_even, f"{site.name}, {plant.name}", v)
    assert best is not None
    v = best[2]
    lofted = sum(v.steady_lines.get(k, 0.0) for k in ("lob", "plug", "bullets"))
    return (
        f"Target ${target:g}: the cheapest methalox cell on the loan is {best[1]}: "
        f"${best[0]:.0f} (steady ${v.steady_cost:.1f}, of which lofting ${lofted:.1f}).\n"
        f"  As for the water stage, the lob alone exceeds the target until the lofted kilogram costs about $1."
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the item 5 and item 6 report under both Earth-departure scenarios."""
    from src.iapetus_cycle import SCENARIOS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario", choices=("both", "hydrogen", "chemical"), default="both"
    )
    args = parser.parse_args(argv)
    print(stage_report())
    print()
    for (name, book), key in zip(SCENARIOS, ("hydrogen", "chemical")):
        if args.scenario not in ("both", key):
            continue
        print("=" * 100 + f"\nScenario: {name}\n" + "=" * 100)
        for section in (
            lines_report(book),
            comparison_report(book),
            carbon_report(book),
            power_report(book),
            sensitivity_report(book),
            rate_report(book),
            flip_report(book),
            target_report(book=book),
        ):
            print(section)
            print()


if __name__ == "__main__":
    main()
