"""What the first batch of PuffSats costs to send toward Jupiter, and what it buys.

The growth ledger (:mod:`src.growth_ledger`) counts kilograms and prices no
money, and the parent's delivery ledger (``tab:delivery_ledger``) leaves out
seeding the cycle.  This module prices the seed on one route (ADR 0035): it
flies the chain's first, three-synodic cycle, which leaves Earth in November
2026, on a Starship refuelled in low orbit that burns straight to the cycle's
excess speed and is expended with its payload.

From a circular orbit of radius ``r`` an excess speed ``v_inf`` takes a burn of
``sqrt(v_inf^2 + 2 mu / r) - sqrt(mu / r)``.  The burn has finite length, so the
steered loss (:func:`src.finite_burn_loss.finite_burn_loss`) is charged on that
circular orbit.  With ``R`` the mass ratio the burn needs, a ship holding
``P`` of propellant drives ``P / (R - 1)`` to departure speed, of which its own
dry mass is charged and the rest is seed.

The seed ship is stripped by default (ADR 0036): it is expended, so it never
re-enters and carries no heat shield, flaps or landing propellant, and it flies
three vacuum Raptors rather than six engines.  The stock ship (85 t, six
engines) is kept as the comparison that reproduces ADR 0035's 50.5 t.

ADR 0035's original module (companion revision ``18773ba``) was never pushed;
this one was rebuilt from the parent's ``sec:mass_interest`` and reproduces its
``tab:seed_amortization`` to the digits printed there.
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from astropy import units as u
from boinor.bodies import Earth
from tabulate import tabulate

from src.astro_constants import LEO_ALTITUDE
from src.chamber_isp import (
    HYDROGEN_5500K,
    METHANE_7000K,
    ROD_MASS,
    ChamberPairing,
    absolute_efficiency,
)
from src.finite_burn_loss import finite_burn_loss
from src.growth_ledger import (
    ADR_0033_PLATE,
    DEFAULT_PARKING_DAYS,
    HYDROGEN_BOIL_OFF_PER_DAY,
    HYDROGEN_CRYOSTAT,
    METHANE_PITCH,
    PLATE_DESIGNS_BY_NAME,
    RAPTOR3_THRUST,
    SOLVED_EFFICIENCY,
    ChainSummary,
    CycleGrowth,
    MethaloxCycle,
    PlateDesign,
    best_methalox_cycle,
    chain_growth,
    hydrogen_boil_off,
    summarize_chain,
)
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.two_wave_growth import VE_METHALOX, TwoWaveCycle, adaptive_two_wave_cycles

#: The parent's Starship propellant load, filled in low orbit.
SEED_PROPELLANT = 1200.0 * u.t
#: Propellant per tanker flight (parent ``sec:heat_shield_bill``).
TANKER_LOAD = 100.0 * u.t
#: Twelve tankers fill the ship; its own ascent carries the PuffSats.
TANKER_FLIGHTS = int(round(float((SEED_PROPELLANT / TANKER_LOAD).to_value(u.one))))
SEED_LAUNCHES = TANKER_FLIGHTS + 1
#: The seed ship's parking orbit before the burn (circular).
SEED_PARKING_ALTITUDE = LEO_ALTITUDE
#: Fleet multiples are taken over this horizon (``tab:growth_ledger_ten_year``).
HORIZON_YEARS = 10.0
#: Plate efficiency of every seed and harvest row.
SEED_PLATE_EFFICIENCY = 0.7


@dataclass(frozen=True)
class SeedShip:
    """An expended Starship configuration.

    Attributes:
        name: Short label.
        dry_mass: Dry mass, charged in full because the ship is expended.
        engines: Raptor 3s firing on the departure burn.
    """

    name: str
    dry_mass: u.Quantity
    engines: int


#: The stock ship: 85 t dry (parent ``gunter_starship``), six engines.  It
#: reproduces ADR 0035's 50.5 t and is kept only as the comparison.
STOCK_SHIP = SeedShip("stock 85 t, 6 engines", 85.0 * u.t, 6)
#: The stripped, expended ship (user, 2026-10-01): no heat shield, flaps or
#: landing propellant, three vacuum Raptors.  Dry masses 40 t and 60 t are
#: unsourced; they are ADR 0035's sweep points.  The baseline (ADR 0036).
STRIPPED_SHIPS = (
    SeedShip("stripped 40 t, 3 vac", 40.0 * u.t, 3),
    SeedShip("stripped 60 t, 3 vac", 60.0 * u.t, 3),
)
SEED_SHIPS = STRIPPED_SHIPS + (STOCK_SHIP,)

#: Hull build price, a hypothesis: no citable source exists, and SpaceX's
#: filing gives none.  The stripped ship is swept at $5M, the user's guess for
#: a hull with no heat shield or flaps, and $20M (user, 2026-10-01: $50M and up
#: is unreasonably pessimistic for it).
SHIP_PRICES = (5.0e6, 20.0e6)
#: The stock ship keeps ADR 0035's $20-100M, so its $910-14 840/kg ends stay
#: reproducible, plus $5M to compare against the stripped ship at the same hull.
STOCK_SHIP_PRICES = (5.0e6, 20.0e6, 50.0e6, 100.0e6)
#: Price of one Starship flight, three ways the parent already quotes:
#: Musk's $2M a flight (``spacex_starship_cost``), and Goldman's $183/kg
#: (``pethokoukis2026_goldman``) and Morgan Stanley's $500/kg
#: (``investing2026_ms_spacex``), each over 100 t.
FLIGHT_PRICES: Dict[str, float] = {
    "Musk": 2.0e6,
    "Goldman": 183.0 * 100.0e3,
    "Morgan Stanley": 500.0 * 100.0e3,
}

#: Costs of capital that bracket the investor.  Annual compounding throughout,
#: ``(1 + r)^t`` with ``t`` in years from the seed's departure.
#: - 7.60%: Damodaran, US cost of capital by sector, aerospace/defense,
#:   January 2026 (79 firms; cost of equity 8.17%).  Parent ``damodaran2026_wacc``.
#: - 30%: median required IRR of venture capitalists who use IRR (216
#:   responses; mean 31%).  Gompers, Gornall, Kaplan and Strebulaev, J. Financial
#:   Economics 135 (2020) 169-190.  Parent ``gompers2020_vc_decisions``.
COSTS_OF_CAPITAL = (0.076, 0.30)


def _mu() -> float:
    return float(Earth.k.to_value(u.km**3 / u.s**2))


def _radius(altitude: u.Quantity) -> float:
    return float((Earth.R + altitude).to_value(u.km))


def seed_excess_speed(cycle: TwoWaveCycle) -> u.Quantity:
    """The excess speed the chain's departure burn leaves with.

    The chain defines its burn above the 20-day orbit's 200 km periapsis speed,
    so ``v_inf^2 = (v_p + burn)^2 - 2 mu / r``.

    Args:
        cycle: The cycle the seed flies.

    Returns:
        The excess speed.
    """
    mu, r = _mu(), _radius(LEO_ALTITUDE)
    final = (
        float(puffsat_cycle_periapsis_speed().to_value(u.km / u.s))
        + cycle.departure_burn
    )
    return float(np.sqrt(final * final - 2.0 * mu / r)) * u.km / u.s


def burn_from_circular(
    excess_speed: u.Quantity, altitude: u.Quantity = SEED_PARKING_ALTITUDE
) -> u.Quantity:
    """Impulsive burn from a circular orbit to an excess speed.

    Args:
        excess_speed: Excess speed to leave with.
        altitude: Circular orbit's altitude.

    Returns:
        ``sqrt(v_inf^2 + 2 mu / r) - sqrt(mu / r)``.
    """
    mu, r = _mu(), _radius(altitude)
    v = float(excess_speed.to_value(u.km / u.s))
    return float(np.sqrt(v * v + 2.0 * mu / r) - np.sqrt(mu / r)) * u.km / u.s


@dataclass(frozen=True)
class SeedPayload:
    """What one expended ship sends.

    Attributes:
        burn: Impulsive burn from the parking orbit.
        burn_time: Propellant over the engines' combined flow.
        finite_burn_loss: Steered loss for that burn length.
        mass_ratio: Start mass over end mass, loss included.
        payload: Seed mass sent, net of the expended ship.
    """

    burn: u.Quantity
    burn_time: u.Quantity
    finite_burn_loss: u.Quantity
    mass_ratio: float
    payload: u.Quantity


@lru_cache(maxsize=16)
def _payload(dry_t: float, engines: int, excess_km_s: float) -> SeedPayload:
    exhaust = VE_METHALOX * u.km / u.s
    burn = burn_from_circular(excess_km_s * u.km / u.s)
    flow = (engines * RAPTOR3_THRUST / exhaust).to(u.t / u.s)
    burn_time = (SEED_PROPELLANT / flow).to(u.s)
    mu, r = _mu(), _radius(SEED_PARKING_ALTITUDE)
    circular_period = 2.0 * np.pi * np.sqrt(r**3 / mu) * u.s
    loss = finite_burn_loss(
        burn,
        exhaust,
        burn_time,
        steered=True,
        altitude=SEED_PARKING_ALTITUDE,
        period=circular_period,
    ).to(u.m / u.s)
    ratio = float(np.exp(((burn + loss) / exhaust).to_value(u.one)))
    payload = (SEED_PROPELLANT / (ratio - 1.0) - dry_t * u.t).to(u.t)
    return SeedPayload(burn.to(u.km / u.s), burn_time, loss, ratio, payload)


def seed_payload(ship: SeedShip, excess_speed: u.Quantity) -> SeedPayload:
    """Seed one expended ship sends to ``excess_speed``, steered loss charged.

    Args:
        ship: The ship flown.
        excess_speed: Excess speed the seed leaves with.

    Returns:
        The payload ledger.
    """
    return _payload(
        float(ship.dry_mass.to_value(u.t)),
        ship.engines,
        float(excess_speed.to_value(u.km / u.s)),
    )


def seed_price_per_kg(
    payload: u.Quantity, flight_price: float, hull_price: float
) -> float:
    """Dollars per kilogram of seed: 13 launches and one hull, over the payload.

    Args:
        payload: Seed one ship sends.
        flight_price: Dollars per Starship flight.
        hull_price: Dollars for the expended hull.

    Returns:
        Dollars per kilogram sent toward Jupiter.
    """
    dollars = SEED_LAUNCHES * flight_price + hull_price
    return dollars / float(payload.to_value(u.kg))


def tanker_share(flight_price: float, hull_price: float) -> float:
    """Share of one seed ship's bill spent on its tankers.

    Args:
        flight_price: Dollars per Starship flight.
        hull_price: Dollars for the expended hull.

    Returns:
        Tanker dollars over the ship's whole bill.
    """
    return TANKER_FLIGHTS * flight_price / (SEED_LAUNCHES * flight_price + hull_price)


@dataclass(frozen=True)
class SeedPriceRange:
    """The seed's cost per kilogram at the two ends the valuation is quoted at.

    Attributes:
        low: Cheapest end, dollars per kilogram.
        high: Dearest end, dollars per kilogram.
        low_label: Which ship, hull and flight price give ``low``.
        high_label: Which give ``high``.
    """

    low: float
    high: float
    low_label: str
    high_label: str


#: The ends every harvest, steady-state and IRR output is quoted at (ADR 0036):
#: the stripped ship from (40 t, $5M hull, Musk) to (60 t, $20M hull, Morgan
#: Stanley): the stripped ship's whole sweep.
STRIPPED_LOW = (STRIPPED_SHIPS[0], SHIP_PRICES[0], "Musk")
STRIPPED_HIGH = (STRIPPED_SHIPS[1], SHIP_PRICES[1], "Morgan Stanley")
#: ADR 0035's ends on the stock ship, kept for the comparison.
STOCK_LOW = (STOCK_SHIP, STOCK_SHIP_PRICES[1], "Musk")
STOCK_HIGH = (STOCK_SHIP, STOCK_SHIP_PRICES[-1], "Morgan Stanley")


def seed_price_range(
    excess_speed: u.Quantity,
    low: Tuple[SeedShip, float, str] = STRIPPED_LOW,
    high: Tuple[SeedShip, float, str] = STRIPPED_HIGH,
) -> SeedPriceRange:
    """The seed's dollars per kilogram at two (ship, hull, flight) corners.

    Args:
        excess_speed: Excess speed the seed leaves with.
        low: The cheap corner.
        high: The dear corner.

    Returns:
        The two ends, labelled.
    """

    def price(corner: Tuple[SeedShip, float, str]) -> float:
        ship, hull, flight = corner
        return seed_price_per_kg(
            seed_payload(ship, excess_speed).payload, FLIGHT_PRICES[flight], hull
        )

    def label(corner: Tuple[SeedShip, float, str]) -> str:
        ship, hull, flight = corner
        return f"{ship.name}, ${hull / 1e6:g}M hull, {flight}"

    return SeedPriceRange(price(low), price(high), label(low), label(high))


@dataclass(frozen=True)
class Design:
    """A departure design the seed is valued for.

    Attributes:
        label: Short label.
        pairing: The chamber, or None for the methalox incumbent.
        efficiency: The chamber's absolute energy efficiency (None for methalox).
    """

    label: str
    pairing: Optional[ChamberPairing]
    efficiency: Optional[float]


DESIGNS = (
    Design("Methalox", None, None),
    Design("Methane, 50%", METHANE_7000K, absolute_efficiency(METHANE_7000K, 0.5)),
    Design("Methane, solved", METHANE_7000K, SOLVED_EFFICIENCY[METHANE_7000K.name]),
    Design("Hydrogen, 50%", HYDROGEN_5500K, absolute_efficiency(HYDROGEN_5500K, 0.5)),
    Design("Hydrogen, solved", HYDROGEN_5500K, SOLVED_EFFICIENCY[HYDROGEN_5500K.name]),
)


@dataclass(frozen=True)
class DesignChain:
    """One design flown over its chain behind the plate, per launch unit.

    Attributes:
        design: The design.
        cycles: The chain it flies: three-synodic only for methalox, which
            cannot fly the two-synodic cycles, the flown chain otherwise.
        growths: Each cycle's growth, PuffSats out over PuffSats in.
        units: Departure units each launch unit expends per cycle: Raptor 3s
            for methalox, chambers otherwise.
        seed: PuffSat mass one launch unit consumes on the first cycle, both
            waves counted as they leave Earth.
        summary: The chain's growth summary.
        ledgers: Each cycle's launch unit, as the growth ledger flew it:
            :class:`MethaloxCycle` for methalox, :class:`CycleGrowth`
            otherwise.  The growth cost model (ADR 0037) prices them.
    """

    design: Design
    cycles: Tuple[TwoWaveCycle, ...]
    growths: Tuple[float, ...]
    units: Tuple[int, ...]
    seed: u.Quantity
    summary: ChainSummary
    ledgers: Tuple[Union[CycleGrowth, MethaloxCycle], ...] = ()

    @property
    def periods_years(self) -> List[float]:
        """Each cycle's length."""
        return [c.period_years for c in self.cycles]

    @property
    def finished(self) -> int:
        """Cycles that finish within the horizon, the stepwise multiple's count."""
        return int(np.sum(np.cumsum(self.periods_years) <= HORIZON_YEARS))


def _arrival(burn_km_s: float) -> float:
    """Share of a batch that survives a methalox correction of ``burn_km_s``."""
    return float(np.exp(-burn_km_s / VE_METHALOX))


@lru_cache(maxsize=4)
def chains(
    split_days: float = DEFAULT_PARKING_DAYS,
) -> Tuple[Tuple[TwoWaveCycle, ...], Tuple[TwoWaveCycle, ...]]:
    """The flown chain and the three-synodic-only chain, cached.

    Args:
        split_days: Split gap, which is also the parking orbit.

    Returns:
        (flown, three-synodic only).
    """
    flown = tuple(adaptive_two_wave_cycles(split_days=split_days))
    three = tuple(adaptive_two_wave_cycles(threshold_m_s=0.0, split_days=split_days))
    return flown, three


@lru_cache(maxsize=16)
def design_chain(
    design: Design,
    plate: PlateDesign = ADR_0033_PLATE,
    split_days: float = DEFAULT_PARKING_DAYS,
) -> DesignChain:
    """Fly one design over its chain, as the growth ledger's matrix does (cached).

    Args:
        design: The design.
        plate: The plate; ADR 0033's by default (ADR 0041 adds the others).
        split_days: Split gap, which is also the parking orbit.

    Returns:
        The design's chain.
    """
    flown, three = chains(split_days)
    if design.pairing is None:
        ledgers = [
            best_methalox_cycle(
                c,
                plate.efficiency,
                max_slug_ratio=plate.max_slug_ratio,
                impactor_bond_energy=plate.impactor_bond_energy,
            )
            for c in three
        ]
        first = ledgers[0]
        seed = (first.puffsats / _arrival(three[0].nozzle_wave_dsm)).to(u.t)
        growths = tuple(m.growth for m in ledgers)
        return DesignChain(
            design,
            three,
            growths,
            tuple(m.engines for m in ledgers),
            seed,
            summarize_chain([c.period_years for c in three], growths),
            tuple(ledgers),
        )
    hydrogen = design.pairing is HYDROGEN_5500K
    pitch = 0.0 if hydrogen else float((METHANE_PITCH / ROD_MASS).to_value(u.one))
    assert design.efficiency is not None
    grown = chain_growth(
        flown,
        plate.efficiency,
        design.pairing,
        design.efficiency,
        pitch,
        max_slug_ratio=plate.max_slug_ratio,
        impactor_bond_energy=plate.impactor_bond_energy,
        cryostat_fraction=HYDROGEN_CRYOSTAT if hydrogen else 0.0,
        boil_off=(
            hydrogen_boil_off(flown, HYDROGEN_BOIL_OFF_PER_DAY) if hydrogen else 0.0
        ),
    )
    g0, c0 = grown[0], flown[0]
    rods = g0.departure.rod_mass_fraction * g0.departing_stack
    seed = (
        g0.puffsats / _arrival(c0.growth_wave_burn)
        + rods / _arrival(c0.nozzle_wave_dsm)
    ).to(u.t)
    growths = tuple(g.growth for g in grown)
    return DesignChain(
        design,
        flown,
        growths,
        tuple(g.departure.chambers for g in grown),
        seed,
        summarize_chain([c.period_years for c in flown], growths),
        tuple(grown),
    )


def units_built(chain: DesignChain) -> float:
    """Departure units bought by the cycles that finish within the horizon.

    Starting from one launch unit, the waves that come home at the end of each
    cycle push ``1, G_0, G_0 G_1, ...`` launch units, each with its own units.

    Args:
        chain: The design's chain.

    Returns:
        Units built.
    """
    built, scale = 0.0, 1.0
    for growth, units in zip(
        chain.growths[: chain.finished], chain.units[: chain.finished]
    ):
        built += scale * units
        scale *= growth
    return built


def wright_unit_cost(n: float, learning_rate: float = 0.8) -> float:
    """Cost of the ``n``th unit over the first's, ``n^log2(rate)`` (Wright 1936).

    Args:
        n: Cumulative unit number.
        learning_rate: Share of unit cost left each time production doubles.

    Returns:
        The ``n``th unit's cost relative to the first.
    """
    return float(n ** np.log2(learning_rate))


def fleet_present_value(ten_year_multiple: float, rate: float) -> float:
    """ADR 0035's column: ``M10 / (1+r)^10``, fleet mass valued at seed cost.

    Fleet mass, not dollars: :mod:`src.harvest` replaces it.

    Args:
        ten_year_multiple: Stepwise ten-year multiple.
        rate: Annual cost of capital.

    Returns:
        Present value per unit of seed.
    """
    return float(ten_year_multiple / (1.0 + rate) ** HORIZON_YEARS)


def _price_table(excess: u.Quantity) -> str:
    rows = []
    for ship in SEED_SHIPS:
        flown = seed_payload(ship, excess)
        hulls = STOCK_SHIP_PRICES if ship is STOCK_SHIP else SHIP_PRICES
        for hull in hulls:
            row: Dict[str, object] = {
                "ship": ship.name,
                "sent t": float(flown.payload.to_value(u.t)),
                "burn s": float(flown.burn_time.to_value(u.s)),
                "loss m/s": float(flown.finite_burn_loss.to_value(u.m / u.s)),
                "hull $M": hull / 1e6,
            }
            for name, flight in FLIGHT_PRICES.items():
                row[f"{name} $/kg"] = seed_price_per_kg(flown.payload, flight, hull)
            for name in ("Goldman", "Morgan Stanley"):
                row[f"tankers/{name}"] = tanker_share(FLIGHT_PRICES[name], hull)
            rows.append(row)
    return tabulate(rows, headers="keys", floatfmt=".4g")


def _amortization(
    designs: Sequence[DesignChain], ship: SeedShip, prices: SeedPriceRange
) -> str:
    payload = seed_payload(ship, seed_excess_speed(designs[0].cycles[0])).payload
    rows = []
    for chain in designs:
        m10 = chain.summary.ten_year_stepwise
        built = units_built(chain)
        rows.append(
            {
                "departure": chain.design.label,
                "seed t": float(chain.seed.to_value(u.t)),
                "ships": float((chain.seed / payload).to_value(u.one)),
                "annual": chain.summary.annual_growth,
                "10 yr": m10,
                "seed $/fleet kg": f"{prices.low / m10:.0f}-{prices.high / m10:.0f}",
                "built": built,
                "Wright last": wright_unit_cost(built),
                **{
                    f"PV r={r:.1%}": fleet_present_value(m10, r)
                    for r in COSTS_OF_CAPITAL
                },
            }
        )
    return tabulate(rows, headers="keys", floatfmt=".3g")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the seed's price per kilogram and ``tab:seed_amortization``.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--plate",
        choices=sorted(PLATE_DESIGNS_BY_NAME),
        help="tab:seed_amortization behind one plate design (ADR 0042)",
    )
    args = parser.parse_args(argv)
    flown, _ = chains()
    excess = seed_excess_speed(flown[0])
    if args.plate:
        plate = PLATE_DESIGNS_BY_NAME[args.plate]
        print(f"Seed behind {plate.label} (ADR 0041/0042), stripped baseline:")
        print(
            _amortization(
                [design_chain(d, plate) for d in DESIGNS],
                STRIPPED_SHIPS[0],
                seed_price_range(excess),
            )
        )
        return
    print(
        f"Seed: the chain's first cycle, v_inf {excess.to_value(u.km / u.s):.2f} km/s, "
        f"burned from a {SEED_PARKING_ALTITUDE.to_value(u.km):g} km circular orbit "
        f"({burn_from_circular(excess).to_value(u.km / u.s):.2f} km/s impulsive) at "
        f"{VE_METHALOX:.3f} km/s, steered loss charged.  {SEED_PROPELLANT:g} of "
        f"propellant on {TANKER_FLIGHTS} tankers; {SEED_LAUNCHES} launches and one "
        "expended hull per ship.  Hull prices are a hypothesis (no source)."
    )
    print(_price_table(excess))
    designs = [design_chain(d) for d in DESIGNS]
    stripped = seed_price_range(excess)
    stock = seed_price_range(excess, STOCK_LOW, STOCK_HIGH)
    print(
        f"\nStripped baseline (ADR 0036): ${stripped.low:.0f}/kg ({stripped.low_label}) "
        f"to ${stripped.high:.0f}/kg ({stripped.high_label})."
    )
    print(_amortization(designs, STRIPPED_SHIPS[0], stripped))
    print(
        f"\nStock comparison (ADR 0035's tab:seed_amortization): ${stock.low:.0f}/kg "
        f"({stock.low_label}) to ${stock.high:.0f}/kg ({stock.high_label})."
    )
    print(_amortization(designs, STOCK_SHIP, stock))


if __name__ == "__main__":
    main()
