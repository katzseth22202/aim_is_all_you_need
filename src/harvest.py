"""What the seed is worth when the fleet delivers cargo instead of growing.

ADR 0035 valued a fleet kilogram at what a seed kilogram cost.  A returning
PuffSat kilogram is worth the cargo it pushes to Sun-Earth L1, and a fleet can
stop growing and pay a dividend instead (ADR 0036).  This module prices both.

**Delivery.**  The returning batch pushes a 1500 t launch unit from rest at the
400 km intercept to the periapsis speed of a transfer whose apoapsis is at L1,
on the argon plate (:func:`src.water_plate.optimal_plate_push`) at the return's
own speed.  A harvested batch needs no split, so it pays only the return's
deep-space-maneuver proxy, not the growth wave's early-arrival burn.  The unit
leaves its 150 t plate at L1 and drops the argon's tank; everything else is
cargo, which pays its own 30 m/s halo insertion in methalox.  The plate's
loading is chosen pulse by pulse under the same k <= 10 cap as the growth
ledger, and its one free number, the starting slug price, is chosen to
maximise the dollars each returning PuffSat kilogram nets,
``P (p_L1 - c_delivery)``, where ``P`` is cargo per PuffSat and
``c_delivery`` is the lob, plate and absorber per cargo kilogram.  PuffSats are
the fleet, so they are not a delivery cost.

**Liquidation.**  The batch returning at the last return within ten years is
``M10 / G_last`` per seed kilogram, the stepwise ten-year multiple without the
last cycle's growth.  It is delivered whole.

**Steady state.**  Keeping ``1/G_n`` of each returning batch reinvested holds
the fleet level and pays ``(1 - 1/G_n)`` of it as cargo, every cycle, from the
same return on.  The chain's own cycles are used, and after its last one the
chain repeats.  With a constant ``G`` and ``T`` the steady state beats
liquidation exactly when ``G > (1+r)^T``.

Every valuation is per dollar of seed: annual compounding, ``(1+r)^t`` with
``t`` in years from the seed's departure.
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.bodies import Earth
from scipy.optimize import brentq
from tabulate import tabulate

from src.growth_ledger import (
    DEFAULT_PLATE_SLUG,
    LAUNCH_UNIT,
    PLATE_MASS,
    PUSH_ALTITUDE,
    wave_speed_at_altitude,
)
from src.seed_cost import (
    COSTS_OF_CAPITAL,
    DESIGNS,
    HORIZON_YEARS,
    SEED_PLATE_EFFICIENCY,
    DesignChain,
    SeedPriceRange,
    chains,
    design_chain,
    fleet_present_value,
    seed_excess_speed,
    seed_price_range,
)
from src.two_wave_growth import VE_METHALOX, TwoWaveCycle
from src.water_plate import (
    PLATE_MAX_SLUG_RATIO,
    WATER_BOND_ENERGY,
    PlatePush,
    optimal_plate_push,
    plate_push,
)

#: Sun-Earth L1's distance from Earth, the delivery transfer's apoapsis.
L1_DISTANCE = 1.5e6 * u.km
#: Halo-orbit insertion at L1, paid in methalox by the cargo (parent
#: ``sec:space_data_centers``).
HALO_INSERTION = 30.0 * u.m / u.s
#: The booster lob, per kilogram lofted (parent ``tab:delivery_ledger``).
LOB_PRICE = 25.0
#: The plate left at L1, per kilogram: after Wright's learning curve (the
#: hundredth plate, $11M) and early production ($50M), parent ``tab:delivery_ledger``.
PLATE_PRICES: Dict[str, float] = {"learned": 114.0, "early": 500.0}
#: The single-stage Vectran gas-bag absorber, priced at the plate's rate per
#: kilogram, as the parent does.
ABSORBER_MASS = 6.0 * u.t
#: L1 sale prices swept (user, 2026-10-01): $500/kg is the absolute maximum,
#: and lower is preferred.  $500 is the Starcloud/Johnston bar and sits inside
#: Johnston's $467-667 break-even; $200 is the Suncatcher bar (parent
#: ``sec:heat_shield_bill``).  Starship-to-L1 prices are the competitor's
#: cost, not a sale price.
L1_PRICES = (100.0, 200.0, 300.0, 400.0, 500.0)
L1_PRICE_CAP = 500.0
#: Starting slug prices the delivery schedule is chosen from; a constant k = 0
#: push stands in for the top of the range.
_START_PRICES = np.geomspace(1.0e-4, 1.0, 81)
#: Annual cost-of-capital bracket for the IRR solve, and the steady state's
#: floor (its perpetuity diverges at r = 0).
_RATE_BRACKET = (0.0, 5.0)
_STEADY_FLOOR = 1.0e-6
_DAYS_PER_YEAR = 365.25


def _mu() -> float:
    return float(Earth.k.to_value(u.km**3 / u.s**2))


def l1_transfer_speed(altitude: u.Quantity = PUSH_ALTITUDE) -> u.Quantity:
    """Periapsis speed of a transfer from ``altitude`` whose apoapsis is at L1.

    Args:
        altitude: Periapsis altitude.

    Returns:
        ``sqrt(2 mu (1/r_p - 1/(r_p + r_a)))``.
    """
    r_p = float((Earth.R + altitude).to_value(u.km))
    r_a = float(L1_DISTANCE.to_value(u.km))
    return float(np.sqrt(2.0 * _mu() * (1.0 / r_p - 1.0 / (r_p + r_a)))) * u.km / u.s


def harvest_wave_speed(cycle: TwoWaveCycle) -> u.Quantity:
    """A harvested batch's speed at the 400 km intercept: the return's own.

    Args:
        cycle: The cycle whose return is harvested.

    Returns:
        The wave's speed at the intercept.
    """
    return wave_speed_at_altitude(cycle.nozzle_wave_v_b * u.km / u.s, PUSH_ALTITUDE)


def harvest_arrival(cycle: TwoWaveCycle) -> float:
    """Share of a batch that leaves Earth and comes home on this cycle's return.

    Args:
        cycle: The cycle.

    Returns:
        ``exp(-dsm / v_e)``: the return's deep-space-maneuver proxy only.
    """
    return float(np.exp(-cycle.nozzle_wave_dsm / VE_METHALOX))


@dataclass(frozen=True)
class Delivery:
    """One launch unit pushed to the L1 transfer.

    Attributes:
        push: The plate push, per kilogram of launch unit.
        puffsats: PuffSats the push consumes.
        slug: Argon the plate sprays.
        cargo: Mass left at L1 after the plate, the slug's tank and the halo
            insertion.
        slug_ratio_start: Loading of the first pulse.
        slug_ratio_end: Loading of the last pulse.
    """

    push: PlatePush
    puffsats: u.Quantity
    slug: u.Quantity
    cargo: u.Quantity
    slug_ratio_start: float
    slug_ratio_end: float

    @property
    def cargo_per_puffsat(self) -> float:
        """``P``: cargo delivered per PuffSat kilogram."""
        return float((self.cargo / self.puffsats).to_value(u.one))

    @property
    def lob_per_kg(self) -> float:
        """The lob's dollars per cargo kilogram."""
        return LOB_PRICE * float((LAUNCH_UNIT / self.cargo).to_value(u.one))

    def cost_per_kg(self, plate_price: float) -> float:
        """``c_delivery``: lob, plate and absorber, per cargo kilogram.

        Args:
            plate_price: Dollars per kilogram of plate and absorber.

        Returns:
            Dollars per cargo kilogram.
        """
        hardware = plate_price * float((PLATE_MASS + ABSORBER_MASS).to_value(u.kg))
        return self.lob_per_kg + hardware / float(self.cargo.to_value(u.kg))

    def net_per_puffsat(self, l1_price: float, plate_price: float) -> float:
        """Dollars one returning PuffSat kilogram nets, ``P (p_L1 - c_delivery)``.

        Args:
            l1_price: Sale price at L1, dollars per kilogram.
            plate_price: Dollars per kilogram of plate and absorber.

        Returns:
            Dollars per PuffSat kilogram.
        """
        return self.cargo_per_puffsat * (l1_price - self.cost_per_kg(plate_price))


def deliver(push: PlatePush, k_start: float, k_end: float) -> Delivery:
    """Book a push's launch unit as cargo at L1.

    Args:
        push: The push, per kilogram of launch unit.
        k_start: Loading of the first pulse.
        k_end: Loading of the last pulse.

    Returns:
        The delivery.
    """
    unit = LAUNCH_UNIT.to(u.t)
    slug = push.slug_fraction * unit
    halo = float(np.exp(-HALO_INSERTION.to_value(u.km / u.s) / VE_METHALOX))
    cargo = (
        push.delivered_fraction * unit
        - DEFAULT_PLATE_SLUG.tank_fraction * slug
        - PLATE_MASS
    ) * halo
    return Delivery(
        push,
        (push.puffsat_fraction * unit).to(u.t),
        slug,
        cargo.to(u.t),
        k_start,
        k_end,
    )


def constant_k_delivery(
    wave_speed: u.Quantity,
    slug_ratio: float,
    plate_efficiency: float = SEED_PLATE_EFFICIENCY,
) -> Delivery:
    """Deliver at a constant loading (:func:`src.water_plate.plate_push`).

    Args:
        wave_speed: The returning wave's speed at the intercept.
        slug_ratio: Argon per kilogram of PuffSat, held over the push.
        plate_efficiency: The plate's efficiency net of chemistry.

    Returns:
        The delivery.
    """
    push = plate_push(
        wave_speed,
        l1_transfer_speed(),
        plate_efficiency,
        slug_ratio,
        slug=DEFAULT_PLATE_SLUG,
        impactor_bond_energy=WATER_BOND_ENERGY,
    )
    return deliver(push, slug_ratio, slug_ratio)


@lru_cache(maxsize=64)
def _front(wave_km_s: float, plate_efficiency: float) -> Tuple[Delivery, ...]:
    pushes = [constant_k_delivery(wave_km_s * u.km / u.s, 0.0, plate_efficiency)]
    for price in _START_PRICES:
        push = optimal_plate_push(
            wave_km_s * u.km / u.s,
            l1_transfer_speed(),
            plate_efficiency,
            float(price),
            max_slug_ratio=PLATE_MAX_SLUG_RATIO,
            slug=DEFAULT_PLATE_SLUG,
            impactor_bond_energy=WATER_BOND_ENERGY,
        )
        pushes.append(deliver(push, push.slug_ratio_start, push.slug_ratio_end))
    return tuple(pushes)


def delivery_front(
    wave_speed: u.Quantity, plate_efficiency: float = SEED_PLATE_EFFICIENCY
) -> Tuple[Delivery, ...]:
    """Capped Pontryagin schedules across starting slug prices, plus k = 0.

    Each schedule is optimal for some exchange rate between argon and
    PuffSats; which one a delivery should fly depends on the sale price.

    Args:
        wave_speed: The returning wave's speed at the intercept.
        plate_efficiency: The plate's efficiency net of chemistry.

    Returns:
        The candidate deliveries.
    """
    return _front(float(wave_speed.to_value(u.km / u.s)), plate_efficiency)


def optimal_delivery(
    wave_speed: u.Quantity,
    l1_price: float,
    plate_price: float,
    plate_efficiency: float = SEED_PLATE_EFFICIENCY,
) -> Delivery:
    """The schedule that nets the most per returning PuffSat at a sale price.

    Args:
        wave_speed: The returning wave's speed at the intercept.
        l1_price: Sale price at L1, dollars per kilogram.
        plate_price: Dollars per kilogram of plate and absorber.
        plate_efficiency: The plate's efficiency net of chemistry.

    Returns:
        The best delivery on :func:`delivery_front`.
    """
    front = delivery_front(wave_speed, plate_efficiency)
    return max(front, key=lambda d: d.net_per_puffsat(l1_price, plate_price))


@dataclass(frozen=True)
class Returns:
    """The returns a valuation can draw on, from the harvest on.

    Attributes:
        growths: Each cycle's growth ``G_n`` over one full chain.
        times: Each cycle's return, in years from the seed's departure.
        arrivals: Each return's surviving share of the batch that left.
        chain_years: Departure of the first cycle to return of the last.
        harvest: Index of the last return within the horizon.
        batch: Batch that leaves Earth on the harvest cycle, per seed kilogram:
            ``M10 / G_harvest``.
    """

    growths: Tuple[float, ...]
    times: Tuple[float, ...]
    arrivals: Tuple[float, ...]
    chain_years: float
    harvest: int
    batch: float

    @property
    def harvest_years(self) -> float:
        """``t_h``: the harvest return's date."""
        return self.times[self.harvest]


def chain_returns(chain: DesignChain) -> Returns:
    """The design chain's returns, dated from the seed's departure.

    Args:
        chain: The design's chain.

    Returns:
        The returns.
    """
    start = chain.cycles[0].departure_jd
    times = tuple((c.return_jd - start) / _DAYS_PER_YEAR for c in chain.cycles)
    harvest = chain.finished - 1
    if times[harvest] > HORIZON_YEARS:
        raise ValueError("no return falls within the horizon")
    return Returns(
        growths=chain.growths,
        times=times,
        arrivals=tuple(harvest_arrival(c) for c in chain.cycles),
        chain_years=times[-1],
        harvest=harvest,
        batch=float(np.prod(chain.growths[:harvest])),
    )


def liquidation_value(
    returns: Returns, net: Sequence[float], rate: float, seed_price: float
) -> float:
    """Harvest the whole batch at ``t_h``, per dollar of seed.

    Args:
        returns: The chain's returns.
        net: Dollars per returning PuffSat kilogram, per cycle.
        rate: Annual cost of capital.
        seed_price: Seed dollars per kilogram.

    Returns:
        Present value over seed cost.
    """
    h = returns.harvest
    dollars = returns.batch * returns.arrivals[h] * net[h]
    return float(dollars / ((1.0 + rate) ** returns.harvest_years * seed_price))


def _dividends(returns: Returns, net: Sequence[float]) -> npt.NDArray[np.float64]:
    """Dollars paid at each return of one chain, per seed kilogram."""
    g = np.asarray(returns.growths)
    return np.asarray(
        returns.batch * (1.0 - 1.0 / g) * np.asarray(returns.arrivals) * np.asarray(net)
    )


def steady_state_value(
    returns: Returns, net: Sequence[float], rate: float, seed_price: float
) -> float:
    """Hold the fleet level from the harvest return on, per dollar of seed.

    Each return reinvests ``1/G_n`` and delivers the rest.  The chain's cycles
    run to its end and then repeat, so the tail is a geometric series.

    Args:
        returns: The chain's returns.
        net: Dollars per returning PuffSat kilogram, per cycle.
        rate: Annual cost of capital; must be positive.
        seed_price: Seed dollars per kilogram.

    Returns:
        Present value over seed cost.

    Raises:
        ValueError: If ``rate`` is not positive (the perpetuity diverges).
    """
    if rate <= 0.0:
        raise ValueError("the steady state's perpetuity needs a positive rate")
    paid = _dividends(returns, net)
    discount = (1.0 + rate) ** (-np.asarray(returns.times))
    h = returns.harvest
    first = float(np.sum(paid[h:] * discount[h:]))
    repeat = (1.0 + rate) ** (-returns.chain_years)
    tail = float(np.sum(paid * discount)) * repeat / (1.0 - repeat)
    return float((first + tail) / seed_price)


def steady_over_liquidation(growth: float, period_years: float, rate: float) -> float:
    """``(1 - 1/G) (1+r)^T / ((1+r)^T - 1)`` for a constant cycle.

    Above one exactly when ``G > (1+r)^T``.

    Args:
        growth: Growth per cycle.
        period_years: Cycle length.
        rate: Annual cost of capital.

    Returns:
        Steady-state value over liquidation value.
    """
    step = (1.0 + rate) ** period_years
    return float((1.0 - 1.0 / growth) * step / (step - 1.0))


def solve_rate(
    value: Callable[[float], float], low: float = _RATE_BRACKET[0]
) -> Optional[float]:
    """The cost of capital at which ``value(r) = 1``: the seed's IRR.

    Args:
        value: Present value over seed cost, decreasing in ``r``.
        low: Bottom of the bracket.

    Returns:
        The IRR; None when the value is below the seed even at ``low``;
        ``inf`` when it is still above at the top of the bracket.
    """
    at_low = value(low) - 1.0
    if at_low == 0.0:
        return low
    if at_low < 0.0:
        return None
    if value(_RATE_BRACKET[1]) - 1.0 > 0.0:
        return float("inf")
    return float(brentq(lambda r: value(r) - 1.0, low, _RATE_BRACKET[1], xtol=1.0e-14))


def fleet_irr(ten_year_multiple: float, years: float) -> float:
    """The fleet-mass IRR: ``M10^(1/t) - 1``.

    Args:
        ten_year_multiple: Stepwise multiple.
        years: Years it took.

    Returns:
        The annual rate.
    """
    return float(ten_year_multiple ** (1.0 / years) - 1.0)


@dataclass(frozen=True)
class Valuation:
    """One design valued at one sale price and plate price.

    Attributes:
        chain: The design's chain.
        returns: Its returns.
        deliveries: The best delivery on each cycle's return.
        net: Dollars per returning PuffSat kilogram, per cycle.
        l1_price: Sale price at L1.
        plate_price: Dollars per kilogram of plate and absorber.
    """

    chain: DesignChain
    returns: Returns
    deliveries: Tuple[Delivery, ...]
    net: Tuple[float, ...]
    l1_price: float
    plate_price: float

    def liquidation(self, rate: float, seed_price: float) -> float:
        """Liquidation value over seed cost."""
        return liquidation_value(self.returns, self.net, rate, seed_price)

    def steady(self, rate: float, seed_price: float) -> float:
        """Steady-state value over seed cost."""
        return steady_state_value(self.returns, self.net, rate, seed_price)

    def liquidation_irr(self, seed_price: float) -> Optional[float]:
        """The seed's IRR when the batch is harvested at ``t_h``."""
        return solve_rate(lambda r: self.liquidation(r, seed_price))

    def steady_irr(self, seed_price: float) -> Optional[float]:
        """The seed's IRR when the fleet is held level from ``t_h`` on."""
        return solve_rate(lambda r: self.steady(r, seed_price), low=_STEADY_FLOOR)

    def payback_years(self, rate: float, seed_price: float) -> Optional[float]:
        """First steady-state return whose cumulative value covers the seed.

        Args:
            rate: Annual cost of capital.
            seed_price: Seed dollars per kilogram.

        Returns:
            The return's date in years, or None within two repeats of the chain.
        """
        paid = _dividends(self.returns, self.net)
        times = np.asarray(self.returns.times)
        h = self.returns.harvest
        total = 0.0
        for lap in range(3):
            for n in range(h if lap == 0 else 0, len(paid)):
                t = times[n] + lap * self.returns.chain_years
                total += float(paid[n]) * (1.0 + rate) ** (-t) / seed_price
                if total >= 1.0:
                    return float(t)
        return None


def value_design(
    chain: DesignChain,
    l1_price: float,
    plate_price: float,
    plate_efficiency: float = SEED_PLATE_EFFICIENCY,
) -> Valuation:
    """Value a design's seed at one sale and plate price, k optimised per return.

    Args:
        chain: The design's chain.
        l1_price: Sale price at L1, dollars per kilogram.
        plate_price: Dollars per kilogram of plate and absorber.
        plate_efficiency: The plate's efficiency net of chemistry.

    Returns:
        The valuation.
    """
    deliveries = tuple(
        optimal_delivery(harvest_wave_speed(c), l1_price, plate_price, plate_efficiency)
        for c in chain.cycles
    )
    return Valuation(
        chain,
        chain_returns(chain),
        deliveries,
        tuple(d.net_per_puffsat(l1_price, plate_price) for d in deliveries),
        l1_price,
        plate_price,
    )


def break_even_price(
    chain: DesignChain,
    rate: float,
    seed_price: float,
    plate_price: float,
    steady: bool,
    cap: float = L1_PRICE_CAP,
) -> Optional[float]:
    """The L1 price at which the seed is exactly repaid at ``rate``.

    Args:
        chain: The design's chain.
        rate: Annual cost of capital.
        seed_price: Seed dollars per kilogram.
        plate_price: Dollars per kilogram of plate and absorber.
        steady: Value the steady state rather than liquidation.
        cap: Highest sale price considered.

    Returns:
        The price, or None when it lies above ``cap``.
    """

    def gap(price: float) -> float:
        valued = value_design(chain, price, plate_price)
        worth = valued.steady if steady else valued.liquidation
        return worth(rate, seed_price) - 1.0

    if gap(cap) < 0.0:
        return None
    return float(brentq(gap, 0.0, cap, xtol=1.0e-6))


def cycles_clearing(chain: DesignChain, rate: float) -> float:
    """Share of cycles whose growth beats the cost of capital over their length.

    ``G_i > (1+r)^T_i`` is the grow-or-harvest rule; ``P`` cancels out of it.

    Args:
        chain: The design's chain.
        rate: Annual cost of capital.

    Returns:
        The fraction of cycles that pass.
    """
    passed = [g > (1.0 + rate) ** t for g, t in zip(chain.growths, chain.periods_years)]
    return float(np.mean(passed))


def _irr(rate: Optional[float]) -> str:
    if rate is None:
        return "none"
    if np.isinf(rate):
        return f">{_RATE_BRACKET[1]:.0%}"
    return f"{rate:.1%}"


def _price(price: Optional[float]) -> str:
    return f">{L1_PRICE_CAP:.0f}" if price is None else f"{price:.0f}"


def _delivery_table(cycles: Sequence[TwoWaveCycle], label: str) -> str:
    rows = []
    for c in cycles:
        w = harvest_wave_speed(c)
        row: Dict[str, object] = {
            "chain": label,
            "cycle": f"{c.index} ({c.synodic_multiple}S)",
            "wave km/s": float(w.to_value(u.km / u.s)),
        }
        for tag, flown in (
            ("k=0", constant_k_delivery(w, 0.0)),
            ("k=10", constant_k_delivery(w, PLATE_MAX_SLUG_RATIO)),
            ("opt $500", optimal_delivery(w, 500.0, PLATE_PRICES["learned"])),
            ("opt $200", optimal_delivery(w, 200.0, PLATE_PRICES["learned"])),
        ):
            row[f"{tag} k"] = (
                f"{flown.slug_ratio_start:.1f}->{flown.slug_ratio_end:.1f}"
            )
            row[f"{tag} argon t"] = float(flown.slug.to_value(u.t))
            row[f"{tag} cargo t"] = float(flown.cargo.to_value(u.t))
            row[f"{tag} P"] = flown.cargo_per_puffsat
            row[f"{tag} lob $/kg"] = flown.lob_per_kg
        rows.append(row)
    return tabulate(rows, headers="keys", floatfmt=".4g")


def _growth_table(designs: Sequence[DesignChain]) -> str:
    rows = []
    for chain in designs:
        returns = chain_returns(chain)
        m10 = chain.summary.ten_year_stepwise
        mean_g = float(np.exp(np.mean(np.log(chain.growths))))
        mean_t = float(np.mean(chain.periods_years))
        row: Dict[str, object] = {
            "departure": chain.design.label,
            "g": chain.summary.annual_growth,
            "M10": m10,
            "t_h yr": returns.harvest_years,
            "batch/seed": returns.batch * returns.arrivals[returns.harvest],
            "fleet IRR": fleet_irr(m10, returns.harvest_years),
            "1-1/G": 1.0 - 1.0 / mean_g,
        }
        for r in COSTS_OF_CAPITAL:
            tag = f"{r:.1%}"
            row[f"g-r {tag}"] = chain.summary.annual_growth - r
            row[f"G>(1+r)^T {tag}"] = cycles_clearing(chain, r)
            row[f"fleet PV {tag}"] = fleet_present_value(m10, r)
            row[f"steady/liq {tag}"] = steady_over_liquidation(mean_g, mean_t, r)
        rows.append(row)
    return tabulate(rows, headers="keys", floatfmt=".3g")


def _valuation_tables(
    designs: Sequence[DesignChain], seed: SeedPriceRange
) -> Tuple[str, str, str, str]:
    irr_rows, pv_rows, payback_rows, even_rows = [], [], [], []
    ends = (("lo", seed.low), ("hi", seed.high))
    for chain in designs:
        for price in L1_PRICES:
            irr: Dict[str, object] = {"departure": chain.design.label, "p_L1": price}
            for plate, plate_price in PLATE_PRICES.items():
                valued = value_design(chain, price, plate_price)
                h = valued.returns.harvest
                irr[f"{plate} margin"] = price - valued.deliveries[h].cost_per_kg(
                    plate_price
                )
                for end, c_seed in ends:
                    irr[f"{plate} liq {end}"] = _irr(valued.liquidation_irr(c_seed))
                    irr[f"{plate} steady {end}"] = _irr(valued.steady_irr(c_seed))
                if plate != "learned":
                    continue
                pv: Dict[str, object] = {
                    "departure": chain.design.label,
                    "p_L1": price,
                    "k": f"{valued.deliveries[h].slug_ratio_start:.1f}"
                    f"->{valued.deliveries[h].slug_ratio_end:.1f}",
                    "P": valued.deliveries[h].cargo_per_puffsat,
                }
                back: Dict[str, object] = {
                    "departure": chain.design.label,
                    "p_L1": price,
                }
                for r in COSTS_OF_CAPITAL:
                    for end, c_seed in ends:
                        tag = f"{r:.1%} {end}"
                        pv[f"liq {tag}"] = valued.liquidation(r, c_seed)
                        pv[f"steady {tag}"] = valued.steady(r, c_seed)
                        years = valued.payback_years(r, c_seed)
                        back[tag] = "never" if years is None else f"{years:.1f}"
                pv_rows.append(pv)
                payback_rows.append(back)
            irr_rows.append(irr)
        even: Dict[str, object] = {"departure": chain.design.label}
        for plate, plate_price in PLATE_PRICES.items():
            for r in COSTS_OF_CAPITAL:
                for end, c_seed in ends:
                    for steady in (False, True):
                        tag = f"{plate} {'steady' if steady else 'liq'} {r:.1%} {end}"
                        even[tag] = _price(
                            break_even_price(chain, r, c_seed, plate_price, steady)
                        )
        even_rows.append(even)
    return (
        tabulate(irr_rows, headers="keys", floatfmt=".0f"),
        tabulate(pv_rows, headers="keys", floatfmt=".3g"),
        tabulate(payback_rows, headers="keys"),
        tabulate(even_rows, headers="keys"),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the harvest and steady-state valuation of the seed.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args(argv)
    flown, three = chains()
    seed = seed_price_range(seed_excess_speed(flown[0]))
    print(
        f"Delivery: {LAUNCH_UNIT:g} unit from rest at 400 km to the L1 transfer "
        f"({l1_transfer_speed().to_value(u.km / u.s):.2f} km/s), plate "
        f"{SEED_PLATE_EFFICIENCY:g}, argon on ice PuffSats, {PLATE_MASS:g} plate left "
        f"at L1, {HALO_INSERTION:g} halo insertion on the cargo.  'opt' flies the "
        f"capped (k <= {PLATE_MAX_SLUG_RATIO:g}) Pontryagin schedule that nets the most "
        "per returning PuffSat at that L1 price, learned plate."
    )
    print(_delivery_table(flown, "flown"))
    print(_delivery_table(three, "3S only"))
    designs = [design_chain(d) for d in DESIGNS]
    print(
        "\nGrow or harvest: g against r, and the share of cycles with "
        "G_i > (1+r)^T_i (P cancels).  Fleet PV is ADR 0035's column, fleet mass "
        "not dollars.  steady/liq uses the chain's mean G and T."
    )
    print(_growth_table(designs))
    irr, pv, payback, even = _valuation_tables(designs, seed)
    print(
        f"\nSeed: ${seed.low:.0f}/kg (lo: {seed.low_label}) to ${seed.high:.0f}/kg "
        f"(hi: {seed.high_label}).  Plate learned ${PLATE_PRICES['learned']:g}/kg, "
        f"early ${PLATE_PRICES['early']:g}/kg, absorber {ABSORBER_MASS:g} at the "
        "plate's rate.  margin = p_L1 - c_delivery at the harvest return "
        "(negative: the delivery loses money)."
    )
    print("\nThe seed's IRR, liquidation at t_h and steady state from t_h:")
    print(irr)
    print("\nValue / seed cost, learned plate:")
    print(pv)
    print("\nSteady-state payback (years from the seed's departure), learned plate:")
    print(payback)
    print(f"\nBreak-even L1 price ($/kg; '>{L1_PRICE_CAP:.0f}' is above the cap):")
    print(even)


if __name__ == "__main__":
    main()
