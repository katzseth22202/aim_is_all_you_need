"""What the seed and the fleet really cost: the growth-charged valuation (ADR 0037).

ADR 0036 (:mod:`src.harvest`) values the seed as the cargo its fleet delivers to
Sun-Earth L1.  Its cash flows are the seed at ``t = 0`` and the harvest
deliveries, and it charges each delivery only its lob, plate and absorber.
Nothing is charged for **growth**.  Every launch unit lofted to grow the fleet
is a ground lob plus an expended plate, expended departure hardware, tanks,
consumables and the PuffSats it sends onward, and the steady state's
reinvested ``1/G_n`` is the same.  This module charges all of it.

It is pure arithmetic over :class:`DesignInputs`, a small typed record of what
one launch unit carries on each cycle.  :mod:`src.growth_cost_inputs` builds
those records from the growth ledger, which takes minutes.  Everything here
takes milliseconds, so the cost logic is tested on hand-built chains in the
fast suite.

**Discounting.**  Ask G2: 30% a year (Gompers et al., the venture target) until
the design's first growth cycle returns, then 10%, the author's judgment of a
proven system's cost of capital, a little above Damodaran's 7.6%.  The 10% is
unsourced.
"""

import enum
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

from scipy.optimize import brentq

from src.learning_curve import LearningCurve

#: The 150 t plate and its 6 t Vectran absorber, expended with every launch
#: unit (``growth_ledger.PLATE_MASS + harvest.ABSORBER_MASS``, pinned by a
#: slow test so the pure module need not import the ledger).
PLATE_AND_ABSORBER = 156.0e3
#: Years simulated for the steady state.  The perpetuity's tail beyond it is
#: under 1e-12 of its value at 10%.
HORIZON_YEARS = 300.0
#: One growth PuffSat: six 10 kg spheres on Kevlar spokes (parent figure
#: caption), steered by two balanced avionics packages on tethers (ASKS H6).
PUFFSAT_MASS = 60.0
PACKAGES_PER_PUFFSAT = 2
#: The departure rod (``chamber_isp.ROD_MASS``) and the three steering
#: packages lost with it on every pulse.
ROD_MASS = 2.5
PACKAGES_PER_ROD = 3
#: The chain lap, counted from the harvest, on which the steady-state cost per
#: kilogram is measured: far enough in that the program's learning has settled.
MEASURED_LAP = 3


class Departure(enum.Enum):
    """What sends the launch unit on to Jupiter."""

    METHALOX = "methalox"
    METHANE = "methane"
    HYDROGEN = "hydrogen"


@dataclass(frozen=True)
class CycleHardware:
    """What one launch unit lofted for growth on one return uses, in kilograms.

    Attributes:
        consumed: PuffSats the unit consumes, both waves counted as they leave
            Jupiter (the measure the seed is sized in).
        growth: ``G_n``, PuffSats out over PuffSats in.
        onward_puffsats: Growth PuffSats the unit sends onward, built new.
        onward_rods: Departure rods it sends onward, built new.
        plate_puffsats: PuffSats that strike its plate (sets the film).
        argon: Argon its plate sprays.
        departure_units: Raptor 3s, or chambers.
        tanks: Expended tank mass.
        gas: Departure propellant launched: methane, hydrogen, or methalox.
        cryostats: Hydrogen cryostats dropped with the plate.
        plugs: Foam port plugs the departure burns.
        pitch: Wall pitch the methane chamber spends.
        pulses: Rods the departure fires, one per pulse (reported, not priced:
            the rods are priced as fleet when they are built).
    """

    consumed: float
    growth: float
    onward_puffsats: float
    onward_rods: float
    plate_puffsats: float
    argon: float
    departure_units: int
    tanks: float
    gas: float
    cryostats: float = 0.0
    plugs: float = 0.0
    pitch: float = 0.0
    pulses: float = 0.0


@dataclass(frozen=True)
class DeliveryOption:
    """One plate schedule a returning batch can push cargo to L1 on, per unit.

    Attributes:
        puffsats: PuffSats the push consumes.
        cargo: Mass left at L1 for the customer.
        slug: Argon the plate sprays.
    """

    puffsats: float
    cargo: float
    slug: float


@dataclass(frozen=True)
class DesignInputs:
    """One design's chain, per seed of one launch unit.

    Attributes:
        label: The design.
        departure: Its departure.
        launch_unit: Mass each lob lofts.
        seed_puffsats: Growth PuffSats in the seed.
        seed_rods: Departure rods in the seed.
        cycles: Each return's growth push, per launch unit.
        times: Each return, in years from the seed's departure.
        arrivals: Each return's surviving share of a harvested batch.
        harvest: Index of the last return within ten years.
        deliveries: Each return's delivery schedules to choose from.
        start_years: When the chain's first cycle departs, in years from the
            seed's purchase: zero, unless the seed flew a slower route
            (:func:`delayed`).
    """

    label: str
    departure: Departure
    launch_unit: float
    seed_puffsats: float
    seed_rods: float
    cycles: Tuple[CycleHardware, ...]
    times: Tuple[float, ...]
    arrivals: Tuple[float, ...]
    harvest: int
    deliveries: Tuple[Tuple[DeliveryOption, ...], ...]
    start_years: float = 0.0

    @property
    def seed(self) -> float:
        """The seed's PuffSat mass."""
        return self.seed_puffsats + self.seed_rods

    @property
    def chain_years(self) -> float:
        """One lap of the chain, after which it repeats."""
        return self.times[-1] - self.start_years

    @property
    def proof_years(self) -> float:
        """When the first growth cycle returns: the stepped rate's switch."""
        return self.times[1]


@dataclass(frozen=True)
class HalvingPrice:
    """A unit price that halves every ``years`` from the first growth return.

    Attributes:
        first: Price at the first growth return.
        floor: Price it approaches.
        years: Halving time.
    """

    first: float
    floor: float
    years: float

    def unit(self, elapsed: float) -> float:
        """Price ``elapsed`` years after the first growth return."""
        span = self.first - self.floor
        return float(self.floor + span * 0.5 ** (max(elapsed, 0.0) / self.years))


PackagePrice = Union[LearningCurve, HalvingPrice]


@dataclass(frozen=True)
class PriceBook:
    """Every price the model charges, in dollars (ASKS H1-H9).

    The defaults are the **Estimate**: unsourced guesses from industrial
    analogues, except the lob, which is the paper's.

    Attributes:
        label: Name of the scenario.
        lob: Per kilogram lofted (paper ``tab:delivery_ledger``).
        plate: Per plate and absorber, by cumulative plates (H1).
        plate_spray: Argon injection, film resprayer and NIR mapping, per plate (H9).
        film_fraction: Ablative film per kilogram of PuffSat on the plate (H9).
        film: Per kilogram of film.
        methane_chamber: Per methane chamber, by cumulative chambers (H2).
        hydrogen_chamber: Per hydrogen chamber, by cumulative chambers (H2).
        chamber_spray: Graphite renewal and charge sprayer, per chamber (H9).
        raptor: Per Raptor 3 (H3).
        tanks: Per kilogram of expended tank (H4).
        cryostats: Per kilogram of hydrogen cryostat (H8).
        argon: Per kilogram (H5).
        methane: Per kilogram of chamber methane (H5).
        hydrogen: Per kilogram of liquid hydrogen (H5).
        methalox: Per kilogram of methalox departure propellant (H7's guess).
        plugs: Per kilogram of foam plug (H5).
        pitch: Per kilogram of wall pitch (H5).
        package: Per steering package (H6).
        puffsat_body: Bags, film, spokes and fill, per kilogram of PuffSat (H6).
        rod_body: Polyethylene rod and tethers, per kilogram of rod (H6).
        fleet_flat: The paper's flat $/kg for every PuffSat and rod, if set.
        charge_growth: False reproduces ADR 0036: no growth costs, no seed
            manufacture.
    """

    label: str = "Estimate"
    lob: float = 25.0
    plate: LearningCurve = LearningCurve(30.0e6, 6.0e6, 0.8)
    plate_spray: float = 3.0e6
    film_fraction: float = 0.001
    film: float = 10.0
    methane_chamber: LearningCurve = LearningCurve(40.0e6, 3.0e6, 0.8)
    hydrogen_chamber: LearningCurve = LearningCurve(60.0e6, 5.0e6, 0.8)
    chamber_spray: float = 2.0e6
    raptor: LearningCurve = LearningCurve.flat(1.0e6)
    tanks: float = 40.0
    cryostats: float = 300.0
    argon: float = 1.0
    methane: float = 1.0
    hydrogen: float = 6.0
    methalox: float = 0.2
    plugs: float = 5.0
    pitch: float = 2.0
    package: PackagePrice = LearningCurve(100.0, 10.0, 0.8, anchor=1.0e5)
    puffsat_body: float = 3.0
    rod_body: float = 4.0
    fleet_flat: Optional[float] = None
    charge_growth: bool = True

    def departure_hardware(self, departure: Departure) -> LearningCurve:
        """The learning curve for one departure unit."""
        return {
            Departure.METHALOX: self.raptor,
            Departure.METHANE: self.methane_chamber,
            Departure.HYDROGEN: self.hydrogen_chamber,
        }[departure]

    def propellant(self, departure: Departure) -> float:
        """Per kilogram of departure propellant."""
        return {
            Departure.METHALOX: self.methalox,
            Departure.METHANE: self.methane,
            Departure.HYDROGEN: self.hydrogen,
        }[departure]


#: The paper's own prices where it has them: the plate at $500/kg on its 80%
#: curve, a $2M chamber (expended, ADR 0032), $20/kg for every PuffSat.  The
#: lines the paper never priced stay at the Estimate.
PAPER = PriceBook(
    label="Paper's prices",
    plate=LearningCurve(500.0 * PLATE_AND_ABSORBER, 0.0, 0.8),
    methane_chamber=LearningCurve.flat(2.0e6),
    hydrogen_chamber=LearningCurve.flat(2.0e6),
    fleet_flat=20.0,
)
#: The headline: every unsourced guess at its central value.
ESTIMATE = PriceBook()
#: Every line at its worst at once.  A labeled bound, not a forecast.
PESSIMISTIC = PriceBook(
    label="Pessimistic",
    lob=50.0,
    plate=LearningCurve.flat(500.0 * PLATE_AND_ABSORBER),
    plate_spray=8.0e6,
    film_fraction=0.04,
    methane_chamber=LearningCurve.flat(200.0e6),
    hydrogen_chamber=LearningCurve.flat(200.0e6),
    chamber_spray=5.0e6,
    raptor=LearningCurve.flat(2.0e6),
    tanks=100.0,
    cryostats=1000.0,
    argon=5.0,
    hydrogen=10.0,
    methalox=0.5,
    plugs=10.0,
    pitch=5.0,
    package=LearningCurve.flat(1000.0),
    puffsat_body=5.0,
    rod_body=5.0,
)
SCENARIOS = (PAPER, ESTIMATE, PESSIMISTIC)


def legacy_prices(plate_per_kg: float) -> PriceBook:
    """ADR 0036's charges: each delivery's lob and plate, nothing else.

    Args:
        plate_per_kg: Dollars per kilogram of plate and absorber, flat.

    Returns:
        The price book.
    """
    return PriceBook(
        label=f"ADR 0036 (plate ${plate_per_kg:g}/kg)",
        plate=LearningCurve.flat(plate_per_kg * PLATE_AND_ABSORBER),
        plate_spray=0.0,
        film=0.0,
        argon=0.0,
        charge_growth=False,
    )


@dataclass(frozen=True)
class DiscountSchedule:
    """An annual cost of capital that steps once.

    Attributes:
        early: Rate until ``switch_years``.
        late: Rate after it.
        switch_years: When the rate steps, in years from the seed's departure.
    """

    early: float
    late: float
    switch_years: float

    @classmethod
    def flat(cls, rate: float) -> "DiscountSchedule":
        """One rate for the whole life.

        Args:
            rate: Annual cost of capital.

        Returns:
            The schedule.
        """
        return cls(rate, rate, 0.0)

    @classmethod
    def stepped(
        cls, early: float, late: float, switch_years: float
    ) -> "DiscountSchedule":
        """``early`` until ``switch_years``, ``late`` after.

        Args:
            early: Rate before the switch.
            late: Rate after it.
            switch_years: When it switches.

        Returns:
            The schedule.
        """
        return cls(early, late, switch_years)

    def factor(self, years: float) -> float:
        """Present value of one dollar paid at ``years``.

        Args:
            years: Time from the seed's departure.

        Returns:
            The discount factor.
        """
        if years <= self.switch_years:
            return float((1.0 + self.early) ** (-years))
        early = (1.0 + self.early) ** (-self.switch_years)
        return float(early * (1.0 + self.late) ** (-(years - self.switch_years)))


@dataclass
class Program:
    """A seed's cash flows and the hardware its program buys.

    Attributes:
        flows: ``(years, dollars)``, outlays negative.
        plates: Plates bought.
        departure_units: Chambers or Raptors bought.
        packages: Steering packages bought.
        lines: Dollars by line over the measured steady-state lap.
        cargo: Cargo delivered over that lap.
    """

    flows: List[Tuple[float, float]] = field(default_factory=list)
    plates: float = 0.0
    departure_units: float = 0.0
    packages: float = 0.0
    lines: Dict[str, float] = field(default_factory=dict)
    cargo: float = 0.0

    @property
    def seed_outlay(self) -> float:
        """Dollars paid at ``t = 0``."""
        return -self.flows[0][1]


def _tally(tally: Optional[Dict[str, float]], lines: Dict[str, float]) -> float:
    if tally is not None:
        for key, value in lines.items():
            tally[key] = tally.get(key, 0.0) + value
    return sum(lines.values())


def _plate_lines(
    prices: PriceBook, program: Program, units: float, puffsats: float
) -> Dict[str, float]:
    lines = {
        "plate": prices.plate.batch_cost(program.plates, units),
        "plate_spray": prices.plate_spray * units,
        "film": prices.film_fraction * prices.film * puffsats,
    }
    program.plates += units
    return lines


def _fleet_lines(
    prices: PriceBook,
    program: Program,
    puffsats: float,
    rods: float,
    elapsed: float,
) -> Dict[str, float]:
    """Build ``puffsats`` and ``rods`` new, ``elapsed`` years after the proof."""
    if prices.fleet_flat is not None:
        return {"fleet": prices.fleet_flat * (puffsats + rods)}
    count = (
        puffsats * PACKAGES_PER_PUFFSAT / PUFFSAT_MASS
        + rods * PACKAGES_PER_ROD / ROD_MASS
    )
    if isinstance(prices.package, HalvingPrice):
        packages = prices.package.unit(elapsed) * count
    else:
        packages = prices.package.batch_cost(program.packages, count)
    program.packages += count
    bodies = prices.puffsat_body * puffsats + prices.rod_body * rods
    return {"fleet": packages + bodies}


def _grow(
    inputs: DesignInputs,
    n: int,
    units: float,
    years: float,
    prices: PriceBook,
    program: Program,
    tally: Optional[Dict[str, float]],
) -> float:
    """Dollars for ``units`` launch units lofted on return ``n``'s growth push."""
    if not prices.charge_growth:
        return 0.0
    cycle = inputs.cycles[n]
    count = cycle.departure_units * units
    hardware = prices.departure_hardware(inputs.departure)
    lines = {
        "lob": prices.lob * inputs.launch_unit * units,
        "argon": prices.argon * cycle.argon * units,
        "departure_hw": hardware.batch_cost(program.departure_units, count),
        "tanks": prices.tanks * cycle.tanks * units,
        "propellant": prices.propellant(inputs.departure) * cycle.gas * units,
        "cryostats": prices.cryostats * cycle.cryostats * units,
        "pulse_consumables": (prices.plugs * cycle.plugs + prices.pitch * cycle.pitch)
        * units,
    }
    if inputs.departure is not Departure.METHALOX:
        lines["chamber_spray"] = prices.chamber_spray * count
    program.departure_units += count
    lines.update(_plate_lines(prices, program, units, cycle.plate_puffsats * units))
    lines.update(
        _fleet_lines(
            prices,
            program,
            cycle.onward_puffsats * units,
            cycle.onward_rods * units,
            years - inputs.proof_years,
        )
    )
    return _tally(tally, lines)


def _deliver(
    inputs: DesignInputs,
    n: int,
    puffsats: float,
    l1_price: float,
    prices: PriceBook,
    program: Program,
    tally: Optional[Dict[str, float]],
) -> Tuple[float, float]:
    """Push ``puffsats`` of returning fleet's cargo to L1: (net dollars, cargo)."""
    plate_now = prices.plate.unit(program.plates + 1.0)

    def unit_cost(option: DeliveryOption) -> float:
        return (
            prices.lob * inputs.launch_unit
            + plate_now
            + prices.plate_spray
            + prices.film_fraction * prices.film * option.puffsats
            + prices.argon * option.slug
        )

    best = max(
        inputs.deliveries[n],
        key=lambda o: (l1_price * o.cargo - unit_cost(o)) / o.puffsats,
    )
    units = puffsats / best.puffsats
    cargo = units * best.cargo
    lines = {
        "lob": prices.lob * inputs.launch_unit * units,
        "argon": prices.argon * best.slug * units,
    }
    lines.update(_plate_lines(prices, program, units, puffsats))
    return l1_price * cargo - _tally(tally, lines), cargo


def run_program(
    inputs: DesignInputs,
    prices: PriceBook,
    seed_price: float,
    l1_price: float,
    steady: bool,
) -> Program:
    """A seed's cash flows: grow to the harvest, then liquidate or hold level.

    Args:
        inputs: The design's chain.
        prices: What everything costs.
        seed_price: Dollars per seed kilogram sent toward Jupiter.
        l1_price: Sale price of cargo at L1, per kilogram.
        steady: Hold the fleet level from the harvest on, rather than deliver
            the whole batch at the harvest.

    Returns:
        The program.
    """
    program = Program()
    built = 0.0
    if prices.charge_growth:
        lines = _fleet_lines(
            prices, program, inputs.seed_puffsats, inputs.seed_rods, -inputs.proof_years
        )
        built = sum(lines.values())
    program.flows.append((0.0, -(seed_price * inputs.seed + built)))
    batch = 1.0
    h = inputs.harvest
    for n in range(h):
        units = batch * inputs.seed / inputs.cycles[n].consumed
        cost = _grow(inputs, n, units, inputs.times[n], prices, program, None)
        program.flows.append((inputs.times[n], -cost))
        batch *= inputs.cycles[n].growth
    if not steady:
        delivered = batch * inputs.arrivals[h] * inputs.seed
        net, _ = _deliver(inputs, h, delivered, l1_price, prices, program, None)
        program.flows.append((inputs.times[h], net))
        return program
    lap, n = 0, h
    while True:
        years = inputs.times[n] + lap * inputs.chain_years
        if years > HORIZON_YEARS:
            return program
        tally = program.lines if lap == MEASURED_LAP else None
        cycle = inputs.cycles[n]
        units = batch * inputs.seed / (cycle.growth * cycle.consumed)
        cost = _grow(inputs, n, units, years, prices, program, tally)
        delivered = (
            batch * (1.0 - 1.0 / cycle.growth) * inputs.arrivals[n] * inputs.seed
        )
        net, cargo = _deliver(inputs, n, delivered, l1_price, prices, program, tally)
        if tally is not None:
            program.cargo += cargo
        program.flows.append((years, net - cost))
        n += 1
        if n == len(inputs.cycles):
            n, lap = 0, lap + 1


def present_value(
    flows: Sequence[Tuple[float, float]], schedule: DiscountSchedule
) -> float:
    """Sum of ``(years, dollars)`` flows, discounted on ``schedule``.

    Args:
        flows: The cash flows.
        schedule: The cost of capital.

    Returns:
        Dollars today.
    """
    return float(sum(amount * schedule.factor(t) for t, amount in flows))


def steady_state_cost(
    inputs: DesignInputs, prices: PriceBook, l1_price: float = 500.0
) -> Tuple[float, Dict[str, float]]:
    """Undiscounted cost per kilogram at L1 over a lap well into the steady state.

    The seed is excluded.  ``l1_price`` matters only through the delivery
    schedule it selects.

    Args:
        inputs: The design's chain.
        prices: What everything costs.
        l1_price: Sale price the deliveries are scheduled for.

    Returns:
        The total, and the same by line.
    """
    program = run_program(inputs, prices, 0.0, l1_price, steady=True)
    lines = {key: value / program.cargo for key, value in program.lines.items()}
    return float(sum(lines.values())), lines


def value_per_seed_dollar(
    inputs: DesignInputs,
    prices: PriceBook,
    seed_price: float,
    l1_price: float,
    schedule: DiscountSchedule,
    steady: bool = True,
) -> float:
    """Present value of everything after the seed, over the seed's cost.

    Held level, it is a perpetuity, so it overstates a fleet whose size demand
    would cap; the break-even price is the robust output.

    Args:
        inputs: The design's chain.
        prices: What everything costs.
        seed_price: Dollars per seed kilogram sent toward Jupiter.
        l1_price: Sale price at L1.
        schedule: The cost of capital.
        steady: Hold the fleet level rather than liquidate at the harvest.

    Returns:
        The ratio; above one repays the seed.
    """
    program = run_program(inputs, prices, seed_price, l1_price, steady)
    return present_value(program.flows[1:], schedule) / program.seed_outlay


def break_even_price(
    inputs: DesignInputs,
    prices: PriceBook,
    seed_price: float,
    schedule: DiscountSchedule,
    steady: bool,
    cap: float = 3000.0,
) -> Optional[float]:
    """The L1 sale price at which the seed and every growth cost are repaid.

    Args:
        inputs: The design's chain.
        prices: What everything costs.
        seed_price: Dollars per seed kilogram sent toward Jupiter.
        schedule: The cost of capital.
        steady: Hold the fleet level rather than liquidate at the harvest.
        cap: Highest price searched.

    Returns:
        The price, or None above ``cap``.
    """

    def worth(price: float) -> float:
        program = run_program(inputs, prices, seed_price, price, steady)
        return present_value(program.flows, schedule)

    if worth(cap) < 0.0:
        return None
    return float(brentq(worth, 0.0, cap, xtol=1.0e-3))


def delayed(inputs: DesignInputs, years: float) -> DesignInputs:
    """The same program, every return ``years`` later; the seed still paid now.

    A seed flown on a slower route (ADR 0039) comes home later, and everything
    after it slides: growth, deliveries and the proof that steps the rate.

    Args:
        inputs: The design's chain.
        years: The delay.

    Returns:
        The delayed chain.
    """
    return replace(
        inputs,
        times=tuple(t + years for t in inputs.times),
        start_years=inputs.start_years + years,
    )
