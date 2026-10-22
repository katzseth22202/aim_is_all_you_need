"""Straw Way energy from the Ceres loop: the parent's S58.

The Ceres loop of :mod:`src.ceres_dive` (ADR 0083) priced as a power source
instead of a launcher.  The Earth-bound ships carry guided ice impactors in
place of film sheets, and the impactors fly down the Straw Way to Heaven, a
magnetically supported vacuum tube from space to the ground, into an MHD
converter at its foot.  There is no L1 delivery, so there is no delivery lob
and there are no Earth plates in the energy price.  The Ceres-bound ships and
their stream are ADR 0083's, unchanged.

What changes in the ledger:

1. **Cargo.** The Earth-bound remainder is ice at the water price, each
   impactor carrying one steering package (:data:`IMPACTOR_MASS`).
2. **S57's plate book.** Plates, disk plates, the Sun bowl, the bore collar
   and their bags are Ceres-made at :data:`CERES_PLATE` with 1% imported
   (:attr:`src.ceres_dive.CeresPrices.ceres_plate`).  Avionics and packages
   stay Earth-made.
3. **Freight.** The Earth hardware (avionics, packages, the 1% import) still
   has to reach Ceres.  The paper's Earth leg carries it there, and that leg
   is pushed at 400 km by film sheets.  So a small share of the Earth-bound
   ships still fly sheets, on ADR 0083's own Earth stream (137 km/s at 4
   R_sun, 99 at 9.86) whatever the impactors' speed.  The sheet
   stream, the impactor stream, the Ceres stream and the freight are one
   linear system.
4. **Energy.** A caught kilogram arrives at the ground at
   ``sqrt(w_400^2 + 2 mu (1/R_E - 1/(R_E + 400 km)))`` and is converted at
   :data:`MHD_EFFICIENCY`.

The straw, the MHD plant, its heat sink and the Ceres industry's capital are
not charged (:func:`uncharged_report`).

Run with ``make ceres-energy``.
"""

import argparse
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from scipy.optimize import minimize
from tabulate import tabulate

from src import ceres_dive as cd
from src.ceres_dive import (
    ARC_GRID,
    CERES_BOOK,
    CERES_BOUNDS,
    EARTH_A,
    MU_SUN,
    R_SUN,
    SUN_GRID,
    CeresDesign,
    CeresLever,
    CeresPrices,
    ShipLedger,
    StageLaws,
    earth_stream,
    fly_ship,
    freight_leg,
    routes_for,
    sheet_material,
    steady_loop,
)
from src.cruise_thermal import (
    ICE_DENSITY,
    SOLAR_CONSTANT,
    equilibrium_temperature,
    sublimation_flux,
)

_MU_EARTH = 398600.4418
_R_EARTH = 6378.137
_HOUR = 3600.0
_YEAR = 365.25 * 86400.0
_G0 = 9.80665e-3  # km/s^2
_MWH = 3.6e9  # J

#: Kinetic energy to electricity in the straw's MHD converter (Seth, S58).
MHD_EFFICIENCY = 0.5
#: Impactors a second down one straw (Seth, S58).
STRAW_RATE = 20.0
#: One impactor's mass (kg): 1 TW of electricity at 20 Hz and 137 km/s is
#: 10.7 kg a pulse; ADR 0083's Ceres pellets are 7.5 kg.
IMPACTOR_MASS = 10.0
#: One steering package's Earth-made mass (kg): the child's 5% guidance
#: mass on a 60 kg PuffSat, shared by its two packages.
PACKAGE_MASS = 1.5
#: S57's Ceres-made plate price ($/kg) and its Earth-made share.
CERES_PLATE = 10.0
PLATE_IMPORT = 0.01

#: The energy book: ADR 0083's best-case book on S57's plates, with one
#: package an impactor and every package's mass charged freight.
ENERGY_BOOK = replace(
    CERES_BOOK,
    ceres_plate=CERES_PLATE,
    plate_import=PLATE_IMPORT,
    package_mass=PACKAGE_MASS,
    earth_packages_per_kg=1.0 / IMPACTOR_MASS,
)
#: The parent's harness book: S57's plates, massless packages at the
#: sheets' rate, the freight held at the L1 loop's (``l1_freight``).
PARENT_BOOK = replace(CERES_BOOK, ceres_plate=CERES_PLATE)


def ground_speed(w_400: float) -> float:
    """Speed at the ground (km/s) of a stream moving at ``w_400`` at 400 km."""
    return math.sqrt(
        w_400**2 + 2.0 * _MU_EARTH * (1.0 / _R_EARTH - 1.0 / (_R_EARTH + 400.0))
    )


def electric_mwh(w_400: float, efficiency: float = MHD_EFFICIENCY) -> float:
    """Electricity per caught kilogram (MWh) at the ground speed."""
    v = ground_speed(w_400) * 1e3
    return efficiency * 0.5 * v**2 / _MWH


@dataclass(frozen=True)
class EnergyLoop:
    """The Ceres loop in the steady state, priced as electricity on Earth.

    Attributes:
        c_ceres: Dollars per kilogram of Ceres stream caught.
        c_energy: Dollars per kilogram of impactor caught at the straw.
        c_sheets: Dollars per kilogram of the film sheets that push the
            freight's Earth leg.
        freight: Earth to a depot, per kilogram.
        lines: ``c_energy`` by the Earth-bound ship's lines ($ per caught kg),
            the Ceres stream it spends included.
        mwh: Electricity per caught kilogram (MWh).
        w_400: The impactors' speed at 400 km (km/s).
        earth_kg: Earth-made kilograms landed at Ceres per caught kilogram,
            the Ceres side's share included.
        sheet_share: Film sheets flown per kilogram of impactor caught, to
            push that freight.
        growth: The Ceres loop's growth per loop.
        ceres_ship: The Ceres-bound ship's ledger.
        energy_ship: The Earth-bound ship's ledger.
    """

    c_ceres: float
    c_energy: float
    c_sheets: float
    freight: float
    lines: Dict[str, float]
    mwh: float
    w_400: float
    earth_kg: float
    sheet_share: float
    growth: float
    ceres_ship: ShipLedger
    energy_ship: ShipLedger

    @property
    def price(self) -> float:
        """Dollars per MWh of electricity."""
        return self.c_energy / self.mwh


def _reference_ve(design: CeresDesign) -> Tuple[float, float]:
    """The Earth stream's speed at 400 km and the Sun film's reference v_e."""
    flown = routes_for(design)
    _, w_e = earth_stream(flown.earth)
    spring = replace(
        design.spring,
        k_delivery=design.k_delivery,
        slow_delivery=not design.child_delivery,
    )
    ve = StageLaws.load().push_ve(
        design.k_delivery, w_e, sheet_material(w_e), spring.push_law
    )
    return w_e, ve


def energy_loop(
    design: CeresDesign,
    prices: CeresPrices = ENERGY_BOOK,
    cargo_price: Optional[float] = None,
    efficiency: float = MHD_EFFICIENCY,
    l1_freight: Optional[CeresPrices] = None,
    freight_stream: Optional[float] = None,
) -> EnergyLoop:
    """Solve the Ceres loop for impactors at the straw and price their electricity.

    Four linear conditions on the Ceres stream's price ``c_C``, the
    impactors' ``c_I``, the freight sheets' ``c_S`` and the freight ``F``:

        c_C net = O_C + H_C F
        c_I Y   = O_I + H_I F + S_E c_C
        c_S Y   = O_S + H_S F + S_E c_C
        F       = F_0 + F_S c_S + F_C c_C   (or F fixed by the book)

    ``O`` is a ship's dollars at zero freight, ``H`` its Earth-made kilograms,
    ``S`` the Ceres stream it spends and ``Y`` the stream it returns caught.

    Args:
        design: The knobs; ``earth_stream`` is the impactors' speed at 400 km.
        prices: The book.
        cargo_price: Per kilogram of impactor ice; None is ``prices.water``.
            ``prices.film`` turns the impactors back into sheets.
        efficiency: Kinetic energy to electricity.
        l1_freight: Hold the freight at the L1 loop's solved value on this
            book (the parent's harness) instead of solving it here.
        freight_stream: The freight sheets' speed at 400 km (km/s); None is
            the depth's :data:`src.ceres_dive.EARTH_STREAMS`, so the freight
            flies ADR 0083's Earth leg at every impactor speed.

    Returns:
        The loop.

    Raises:
        ValueError: If the design does not fly or the loop does not close.
    """
    flown = routes_for(design)
    w_e, reference_ve = _reference_ve(design)
    ice = prices.water if cargo_price is None else cargo_price
    sheets_book = replace(prices, earth_packages_per_kg=None)
    caught = 1.0 - prices.misses
    c_free = fly_ship(design, prices, "ceres", 0.0, reference_ve)
    i_free = fly_ship(design, prices, "earth", 0.0, reference_ve, ice)
    sheet_design = replace(
        design,
        earth_stream=(
            cd.EARTH_STREAMS.get(design.depth, design.earth_stream)
            if freight_stream is None
            else freight_stream
        ),
    )
    _, sheet_ve = _reference_ve(sheet_design)
    s_free = fly_ship(sheet_design, sheets_book, "earth", 0.0, sheet_ve)
    net = c_free.cargo * caught - c_free.stream
    if net <= 0.0:
        raise ValueError("the Ceres loop does not grow")
    y_i = i_free.cargo * caught
    y_s = s_free.cargo * caught
    if l1_freight is not None:
        f_0, f_s, f_c = steady_loop(design, l1_freight).freight, 0.0, 0.0
    elif prices.freight is not None:
        f_0, f_s, f_c = prices.freight, 0.0, 0.0
    else:
        leg = freight_leg(sheet_design, prices)
        f_0 = leg.earth_dollars + leg.capture_dollars
        f_s, f_c = leg.earth_sheets, leg.capture_stream
    matrix = np.array(
        [
            [net, 0.0, 0.0, -c_free.earth_hardware],
            [-i_free.stream, y_i, 0.0, -i_free.earth_hardware],
            [-s_free.stream, 0.0, y_s, -s_free.earth_hardware],
            [-f_c, 0.0, -f_s, 1.0],
        ]
    )
    rhs = np.array(
        [
            sum(c_free.dollars.values()),
            sum(i_free.dollars.values()),
            sum(s_free.dollars.values()),
            f_0,
        ]
    )
    c_c, c_i, c_s, freight = (float(x) for x in np.linalg.solve(matrix, rhs))
    if min(c_c, c_i, c_s) <= 0.0 or freight < 0.0:
        raise ValueError("the stream prices do not close")
    c_ship = fly_ship(design, prices, "ceres", freight, reference_ve)
    i_ship = fly_ship(design, prices, "earth", freight, reference_ve, ice)
    lines = {key: value / y_i for key, value in i_ship.dollars.items()}
    lines["ceres_stream"] = i_ship.stream * c_c / y_i
    # Earth kilograms per caught kilogram, the Ceres stream's share carried in.
    earth_kg = (
        i_ship.earth_hardware + i_ship.stream * c_free.earth_hardware / net
    ) / y_i
    return EnergyLoop(
        c_ceres=c_c,
        c_energy=float(sum(lines.values())),
        c_sheets=c_s,
        freight=freight,
        lines=lines,
        mwh=electric_mwh(w_e, efficiency),
        w_400=w_e,
        earth_kg=earth_kg,
        sheet_share=earth_kg * f_s,
        growth=c_ship.cargo * caught / c_ship.stream,
        ceres_ship=c_ship,
        energy_ship=i_ship,
    )


# --------------------------------------------------------------------------
# The optimizer and the sweep
# --------------------------------------------------------------------------

#: The impactors' speeds at 400 km swept (km/s), and the Ceres stream's
#: excesses searched at each (ADR 0083's grid, cut where it stopped paying).
ENERGY_SPEEDS = (80.0, 100.0, 120.0, 137.0, 150.0, 175.0, 200.0, 225.0, 250.0, 300.0)
ENERGY_EXCESSES = (60.0, 75.0, 90.0, 100.0, 110.0, 120.0, 130.0, 150.0, 175.0)
SUN_NAMES = ("k_sun", "arc_hours")
#: Parallel workers for the sweep (each holds the stage laws, ~300 MB).
WORKERS = 4


def optimize_energy(
    design: CeresDesign,
    prices: CeresPrices = ENERGY_BOOK,
    l1_freight: Optional[CeresPrices] = None,
) -> Optional[Tuple[CeresDesign, EnergyLoop]]:
    """The cheapest $/MWh_e over the Sun's ``k`` and arc, the rest held.

    The same search as :func:`src.ceres_dive.optimize_loop`: the best node of
    :data:`SUN_GRID` x :data:`ARC_GRID`, then Powell on :data:`CERES_BOUNDS`.

    Returns:
        The best design and its loop, or None if nothing near the grid flies.
    """
    best: List[Tuple[float, CeresDesign]] = []

    def price(x: Sequence[float]) -> float:
        trial = replace(design, **dict(zip(SUN_NAMES, (float(v) for v in x))))  # type: ignore[arg-type]
        try:
            value = energy_loop(trial, prices, l1_freight=l1_freight).price
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1e6
        if not best or value < best[0][0]:
            best[:] = [(value, trial)]
        return value

    grid = [([k, arc], price([k, arc])) for k in SUN_GRID for arc in ARC_GRID]
    start, value = min(grid, key=lambda row: row[1])
    if value >= 1e6:
        return None
    minimize(
        price,
        start,
        method="Powell",
        bounds=[CERES_BOUNDS[name] for name in SUN_NAMES],
        options={"xtol": 1e-3, "ftol": 1e-6},
    )
    return best[0][1], energy_loop(best[0][1], prices, l1_freight=l1_freight)


Cell = Tuple[float, float, float, CeresPrices, Dict[str, object]]


def _cell(args: Cell) -> Optional[Tuple[CeresDesign, EnergyLoop]]:
    depth, w, x_c, prices, overrides = args
    try:
        base = CeresDesign(
            depth=depth, ceres_excess=x_c, earth_stream=w, k_delivery=5.0, **overrides  # type: ignore[arg-type]
        )
        return optimize_energy(base, prices)
    except ValueError:
        return None


def speed_sweep(
    depths: Sequence[float] = cd.DEPTHS,
    speeds: Sequence[float] = ENERGY_SPEEDS,
    excesses: Sequence[float] = ENERGY_EXCESSES,
    prices: CeresPrices = ENERGY_BOOK,
    workers: int = WORKERS,
) -> Dict[Tuple[float, float, float], Optional[Tuple[CeresDesign, EnergyLoop]]]:
    """Every (depth, speed, X_C) cell, the Sun's ``k`` and arc re-optimized."""
    keys = [(d, w, x) for d in depths for w in speeds for x in excesses]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        found = list(pool.map(_cell, [(d, w, x, prices, {}) for d, w, x in keys]))
    return dict(zip(keys, found))


def best_by_speed(
    cells: Dict[Tuple[float, float, float], Optional[Tuple[CeresDesign, EnergyLoop]]],
    depth: float,
) -> Dict[float, Tuple[CeresDesign, EnergyLoop]]:
    """The cheapest X_C at each speed of one depth."""
    out: Dict[float, Tuple[CeresDesign, EnergyLoop]] = {}
    for (d, w, _), found in cells.items():
        if d != depth or found is None:
            continue
        if w not in out or found[1].price < out[w][1].price:
            out[w] = found
    return dict(sorted(out.items()))


def flyable_limit(
    depth: float, x_c: float, low: float, high: float, prices: CeresPrices = ENERGY_BOOK
) -> float:
    """Bisect the fastest impactor stream any Sun ``k`` and arc still flies (km/s)."""
    for _ in range(6):
        mid = 0.5 * (low + high)
        if _cell((depth, mid, x_c, prices, {})) is None:
            high = mid
        else:
            low = mid
    return low


def _sun_floor(design: CeresDesign) -> str:
    """Why the fastest stream fails: the Sun bowl's survival floor against the ship it pushes.

    The same arithmetic as :func:`src.ceres_dive.fly_ship`'s Sun leg, per
    kilogram of ship, at the Sun ``k`` and arc given.
    """
    flown = routes_for(design)
    _, reference_ve = _reference_ve(design)
    laws = StageLaws.load()
    w_sun = cd.sun_closing(flown.earth, flown.earth_disks)
    ve_sun = laws.sun_ve(design.k_sun, w_sun)
    tau = design.depth * R_SUN / flown.earth.periapsis_speed / _HOUR
    loss = (design.arc_hours / tau) ** 2 * (1.0 / 96.0 + 1.0 / 24.0)
    boost = flown.earth.boost / (1.0 - loss)
    mr_sun = math.exp(boost / ve_sun)
    film_per = (
        cd.FILM_PER_IMPULSE
        * ve_sun
        / reference_ve
        * (1.0 - 1.0 / mr_sun)
        * ve_sun
        * 1e3
    )
    ship = design.ship_mass / (1.0 + film_per)
    disks = ship * (1.0 - 1.0 / mr_sun) / design.k_sun
    final = ship / mr_sun
    pulse_disk = disks / (cd.SUN_PULSE_RATE * design.arc_hours * _HOUR)
    spring = cd.gas_spring(
        pulse_disk * design.k_sun * ve_sun * 1e3 * cd.SUN_PULSE_RATE,
        cd.SUN_PULSE_RATE,
        final,
        design.spring,
        cd._plate_for(
            pulse_disk,
            cd.SUN_PLATE_PULSE,
            cd.SUN_PLATE_AREA,
            w_sun / cd.SUN_PLATE_SPEED,
        ),
    )
    return (
        f"Sun boost {boost:.1f} km/s at v_e {ve_sun:.1f} (disks close at {w_sun:.0f} km/s):"
        f" mass ratio {mr_sun:.1f}, {final / 1e3:.0f} t left of 1000 t against a Sun bowl"
        f" floor of {spring.plate / 1e3:.0f} t"
    )


# --------------------------------------------------------------------------
# Item 6: ice impactors on the Earth-bound dive
# --------------------------------------------------------------------------


def _outbound_radial_speed(depth: float, x_e: float, r: float) -> float:
    r_p = depth * R_SUN
    v_p = math.sqrt(x_e**2 + 2.0 * MU_SUN / r_p)
    h = r_p * v_p
    return math.sqrt(max(x_e**2 + 2.0 * MU_SUN / r - (h / r) ** 2, 1e-12))


#: A pellet's own sunward multilayer cap, held by its package: the share of
#: sunlight that still reaches the ice (an MLI effective emittance), as the
#: PuffSat's detachable shade of ``sec:lox_puffsat`` (src.cruise_thermal).
CAPPED = 0.01
#: The cap's areal mass (kg/m^2): ten layers of 25 um Ceres film.
CAP_AREAL = 0.35


def flight_days(
    depth: float, x_e: float, release_au: float, arrive_au: float = 1.0
) -> float:
    """Days from ``release_au`` out to ``arrive_au`` on the boosted hyperbola."""
    radii = np.geomspace(release_au * cd.AU, arrive_au * cd.AU, 400)
    rates = np.array([1.0 / _outbound_radial_speed(depth, x_e, r) for r in radii])
    return float(np.trapezoid(rates, radii)) / 86400.0


def sublimated_share(
    depth: float,
    x_e: float,
    release_au: float,
    mass: float = IMPACTOR_MASS,
    absorptivity: float = 1.0,
    steps: int = 120,
    arrive_au: float = 1.0,
) -> float:
    """Share of an ice sphere lost between release and ``arrive_au``.

    The sphere sits at the radiation-plus-sublimation balance of
    :mod:`src.cruise_thermal` at each radius (Hertz-Knudsen, an upper bound),
    and shrinks as it goes.  ``absorptivity`` 1 is bare ice; :data:`CAPPED`
    is a pellet behind its own cap.
    """
    radii = np.geomspace(release_au * cd.AU, arrive_au * cd.AU, steps)
    m = mass
    for r0, r1 in zip(radii[:-1], radii[1:]):
        r = math.sqrt(r0 * r1)
        au = r / cd.AU
        absorbed = absorptivity * SOLAR_CONSTANT / (4.0 * au**2)
        t = equilibrium_temperature(absorbed, bounds=(80.0, 600.0))
        flux = float(sublimation_flux(t).to_value(u.kg / u.m**2 / u.s))
        radius = (
            3.0 * m / (4.0 * math.pi * ICE_DENSITY.to_value(u.kg / u.m**3))
        ) ** (1.0 / 3.0)
        dt = (r1 - r0) / _outbound_radial_speed(depth, x_e, r)
        m -= 4.0 * math.pi * radius**2 * flux * dt
        if m <= 0.0:
            return 1.0
    return 1.0 - m / mass


def train_spread(
    depth: float,
    x_e: float,
    release_au: float,
    cargo: float,
    mass: float = IMPACTOR_MASS,
) -> float:
    """Along-track release speed spread (km/s) that strings one ship's cargo at the straw's rate.

    The cargo arrives over ``cargo / (mass * rate)`` seconds; a speed change
    ``dv`` at release moves the arrival by about ``(1 AU - r) dv / v^2``.
    It scales with the ship's cargo.
    """
    span = cargo / (mass * STRAW_RATE)
    seconds = flight_days(depth, x_e, release_au) * 86400.0
    return (EARTH_A - release_au * cd.AU) * span / seconds**2


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

#: The parent's sweep, reproduced on its own harness's conventions.
PARENT_SPEEDS = (100.0, 137.0, 175.0, 200.0)
PLATE_RATES = cd.PLATE_RATES
PLATE_HOURS = cd.PLATE_HOURS
#: Where the ship lets its capped pellets go, having kept them behind its own
#: shade since the Sun; the bare-ice rows' releases; just after the boost.
RELEASE_AU = 0.3
EARLY_AU = 0.05
BARE_RELEASES = (0.5, 0.8, 0.95)


def _f(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def reproduction_report() -> str:
    """The parent's harness: L1 freight held, massless packages, plates at $10 or Earth-made."""
    rows = []
    for plate in (CERES_PLATE, None):
        book = replace(CERES_BOOK, ceres_plate=plate)
        cells = [f"${plate:g} Ceres" if plate else "S56 Earth-made"]
        for w in PARENT_SPEEDS:
            base = CeresDesign(
                depth=4.0, ceres_excess=110.0, earth_stream=w, k_delivery=5.0
            )
            found = optimize_energy(base, book, l1_freight=CERES_BOOK)
            cells.append("-" if found is None else _f(found[1].price))
        rows.append(cells)
    base = CeresDesign(
        depth=4.0,
        ceres_excess=110.0,
        earth_stream=137.0,
        k_sun=29.907,
        arc_hours=2.4,
        k_delivery=5.0,
    )
    fixed = energy_loop(base, PARENT_BOOK, l1_freight=CERES_BOOK)
    return (
        "4 R_sun, X_C 110, $/MWh_e on the parent's conventions (L1 freight held,"
        " massless packages at the sheets' rate)\n"
        + tabulate(
            rows,
            headers=["plates"] + [f"{w:g}" for w in PARENT_SPEEDS],
            tablefmt="github",
        )
        + f"\nAt k_sun 29.907, arc 2.4 h: c_C ${fixed.c_ceres:.2f}, c_E ${fixed.c_energy:.2f},"
        f" {fixed.mwh:.3f} MWh_e/kg, ${fixed.price:.2f}/MWh_e (parent: 4.35 / 5.83 / 1.304 / 4.47)"
    )


def sweep_report(
    cells: Dict[Tuple[float, float, float], Optional[Tuple[CeresDesign, EnergyLoop]]],
) -> Tuple[str, Dict[float, Tuple[CeresDesign, EnergyLoop]]]:
    """Item 3: the speed sweep, X_C, k_sun and arc re-optimized, at both depths."""
    out = []
    optima: Dict[float, Tuple[CeresDesign, EnergyLoop]] = {}
    for depth in cd.DEPTHS:
        best = best_by_speed(cells, depth)
        rows = []
        for w, (design, loop) in best.items():
            by_x = {
                x: found[1].price
                for (d, s, x), found in cells.items()
                if d == depth and s == w and found is not None
            }
            ship = loop.energy_ship.mass
            g = ship["boost"] / (design.arc_hours * _HOUR) / _G0
            rows.append(
                [
                    f"{w:g}",
                    _f(routes_for(design).earth_excess, 1),
                    f"{design.ceres_excess:g}",
                    _f(design.k_sun, 1),
                    _f(design.arc_hours, 2),
                    _f(ship["boost"], 1),
                    _f(g, 2),
                    _f(loop.c_ceres),
                    _f(loop.c_energy),
                    _f(loop.freight),
                    _f(loop.mwh, 3),
                    f"**{loop.price:.2f}**",
                    f"{min(by_x.values()):.2f}-{max(by_x.values()):.2f}",
                ]
            )
        name, (design, loop) = min(best.items(), key=lambda kv: kv[1][1].price)
        optima[depth] = (design, loop)
        flyable = [w for w in best]
        failing = [w for w in ENERGY_SPEEDS if w not in best and w > max(flyable)]
        limit = (
            flyable_limit(
                depth, best[max(flyable)][0].ceres_excess, max(flyable), min(failing)
            )
            if failing
            else float("nan")
        )
        flat = [w for w, (_, lp) in best.items() if lp.price <= 1.05 * loop.price]
        out.append(
            f"\n{depth:g} R_sun: minimum ${loop.price:.2f}/MWh_e at {name:g} km/s"
            f" (X_C {design.ceres_excess:g}); within 5% from {min(flat):g} to {max(flat):g} km/s;"
            f" fastest flyable {limit:.0f} km/s; unflyable: {', '.join(f'{w:g}' for w in failing) or 'none'}"
        )
        out.append(
            tabulate(
                rows,
                headers=[
                    "w_400",
                    "X_E",
                    "X_C",
                    "k_sun",
                    "arc h",
                    "Sun boost",
                    "g",
                    "c_C",
                    "c_E $/kg",
                    "freight",
                    "MWh_e/kg",
                    "$/MWh_e",
                    "range over X_C",
                ],
                tablefmt="github",
            )
        )
        if failing:
            edge = replace(best[max(flyable)][0], earth_stream=min(failing))
            try:
                out.append(f"At {min(failing):g} km/s: {_sun_floor(edge)}")
            except ValueError as exc:
                out.append(f"At {min(failing):g} km/s: {exc}")
    return "\n".join(out), optima


def line_report(optima: Dict[float, Tuple[CeresDesign, EnergyLoop]]) -> str:
    """The minimum's line items, $ per caught kg and $/MWh_e."""
    out = []
    for depth, (design, loop) in optima.items():
        rows = sorted(loop.lines.items(), key=lambda kv: -kv[1])
        out.append(
            f"\n{depth:g} R_sun, {loop.w_400:.0f} km/s, X_C {design.ceres_excess:g},"
            f" k_sun {design.k_sun:.2f}, arc {design.arc_hours:.2f} h;"
            f" Ceres loop x{loop.growth:.2f} per loop"
        )
        out.append(
            tabulate(
                [[k, _f(v, 3), _f(v / loop.mwh, 3)] for k, v in rows if v >= 5e-4]
                + [["**total**", f"**{loop.c_energy:.3f}**", f"**{loop.price:.3f}**"]],
                headers=["line", "$/kg caught", "$/MWh_e"],
                tablefmt="github",
            )
        )
    return "\n".join(out)


def freight_report(optima: Dict[float, Tuple[CeresDesign, EnergyLoop]]) -> str:
    """Item 4: the freight re-solved for the energy loop."""
    rows = []
    for depth, (design, loop) in optima.items():
        held = energy_loop(design, ENERGY_BOOK, l1_freight=CERES_BOOK)
        free = energy_loop(design, replace(ENERGY_BOOK, freight=0.0))
        rows.append(
            [
                f"{depth:g}",
                _f(loop.freight),
                _f(held.freight),
                _f(loop.earth_kg, 4),
                _f(loop.sheet_share, 4),
                _f(loop.c_sheets),
                _f(loop.price),
                _f(held.price),
                _f(free.price),
            ]
        )
    return tabulate(
        rows,
        headers=[
            "R_sun",
            "freight $/kg",
            "L1 loop's (ADR 0083 book)",
            "Earth kg / kg caught",
            "sheets / kg caught",
            "c_S $/kg",
            "$/MWh_e",
            "at that freight",
            "freight free",
        ],
        tablefmt="github",
    )


def plate_report(optima: Dict[float, Tuple[CeresDesign, EnergyLoop]]) -> str:
    """Item 5: the Ceres plate's policy, Hz and push length on the energy objective."""
    design, _ = optima[4.0]
    rows = []
    best: Tuple[float, str] = (1e9, "")
    for rate in PLATE_RATES:
        for hours in PLATE_HOURS:
            cells = [f"{rate:g}", f"{hours:g}"]
            for policy in ("recover", "expend", "sun_bowl"):
                trial = replace(design, push_rate=rate, push_hours=hours, plate=policy)
                try:
                    value = energy_loop(trial).price
                except ValueError:
                    cells.append("-")
                    continue
                cells.append(_f(value, 3))
                if value < best[0]:
                    best = (value, f"{policy} at {rate:g} Hz, {hours:g} h")
            rows.append(cells)
    return (
        "4 R_sun minimum, design held, $/MWh_e by what is done with the plate\n"
        + tabulate(
            rows,
            headers=["Hz", "hours", "recover", "expend", "Sun bowl"],
            tablefmt="github",
        )
        + f"\nCheapest: {best[1]}, ${best[0]:.3f}/MWh_e"
    )


def impactor_report(
    cells: Dict[Tuple[float, float, float], Optional[Tuple[CeresDesign, EnergyLoop]]],
    optima: Dict[float, Tuple[CeresDesign, EnergyLoop]],
) -> str:
    """Item 6: the ice impactors through the dive and to the straw."""
    rows = []
    for depth in cd.DEPTHS:
        for w, (design, loop) in best_by_speed(cells, depth).items():
            x_e = routes_for(design).earth_excess
            cargo = loop.energy_ship.cargo
            ship = loop.energy_ship.mass
            rows.append(
                [f"{depth:g}", f"{w:g}", _f(x_e, 0)]
                + [
                    f"{100 * sublimated_share(depth, x_e, r):.0f}%"
                    for r in BARE_RELEASES
                ]
                + [
                    f"{100 * sublimated_share(depth, x_e, EARLY_AU, absorptivity=CAPPED):.0f}%",
                    f"{100 * sublimated_share(depth, x_e, RELEASE_AU, absorptivity=CAPPED):.2f}%",
                    _f(flight_days(depth, x_e, RELEASE_AU), 1),
                    _f(train_spread(depth, x_e, RELEASE_AU, cargo), 2),
                    _f(ship["boost"] / (design.arc_hours * _HOUR) / _G0, 2),
                ]
            )
    radius = (
        3.0 * IMPACTOR_MASS / (4.0 * math.pi * ICE_DENSITY.to_value(u.kg / u.m**3))
    ) ** (1.0 / 3.0)
    cap = CAP_AREAL * math.pi * radius**2 * 1.5
    ceres = []
    for depth, (design, _) in optima.items():
        x_c = design.ceres_excess
        ceres.append(
            f"{depth:g} R_sun, X_C {x_c:g}: bare {100 * sublimated_share(depth, x_c, BARE_RELEASES[0], arrive_au=cd.CERES_A / cd.AU):.0f}%"
            f" from {BARE_RELEASES[0]:g} AU,"
            f" capped {100 * sublimated_share(depth, x_c, RELEASE_AU, absorptivity=CAPPED, arrive_au=cd.CERES_A / cd.AU):.2f}%"
            f" over {flight_days(depth, x_c, RELEASE_AU, cd.CERES_A / cd.AU):.0f} days"
        )
    return (
        f"A 10 kg ice sphere: share sublimated by 1 AU if released bare at r (AU), or behind its own"
        f" cap ({CAPPED:g} of sunlight reaching the ice) from {EARLY_AU:g} and {RELEASE_AU:g} AU;"
        f" days from {RELEASE_AU:g} AU; the release speed spread (km/s, full width) that strings one"
        " 1000 t ship's impactors at 20 Hz; the Sun boost's mean g\n"
        + tabulate(
            rows,
            headers=["R_sun", "w_400", "X_E"]
            + [f"bare {r:g}" for r in BARE_RELEASES]
            + [
                f"capped {EARLY_AU:g}",
                f"capped {RELEASE_AU:g}",
                "days",
                "dv 20 Hz",
                "g",
            ],
            tablefmt="github",
        )
        + f"\nThe cap: {cap * 1e3:.0f} g a pellet ({100 * cap / IMPACTOR_MASS:.2f}%, a 1.5x-oversized disk"
        f" of {CAP_AREAL:g} kg/m^2), ${CERES_BOOK.film * cap / IMPACTOR_MASS:.3f} per kg at the Ceres film price"
        "\nThe Ceres stream's pellets, the same test to Ceres' orbit: "
        + "; ".join(ceres)
    )


#: Item 7: the parent's levers, then ADR 0083's risk rows.
ENERGY_SENSITIVITIES = (
    CeresLever("polyethylene (Sun charges) $0.20", prices={"polyethylene": 0.2}),
    CeresLever("Sun disks $10", prices={"disk": 10.0}),
    CeresLever("Earth hardware $10", prices={"earth_hardware": 10.0}),
    CeresLever("Ceres structure $3", prices={"structure": 3.0}),
    CeresLever("avionics halved (0.5%)", design={"avionics": 0.005}),
    CeresLever("misses 0", prices={"misses": 0.0}),
    CeresLever(
        "packages free and massless", prices={"package": 0.0, "package_mass": 0.0}
    ),
    CeresLever("package 0.3 kg", prices={"package_mass": 0.3}),
    CeresLever("impactor 7.5 kg", prices={"earth_packages_per_kg": 1.0 / 7.5}),
    CeresLever("impactor 11 kg", prices={"earth_packages_per_kg": 1.0 / 11.0}),
    CeresLever("Ceres plates $30", prices={"ceres_plate": 30.0}),
    CeresLever("S56 Earth-made plates", prices={"ceres_plate": None}),
    CeresLever("unlisted dead mass 15%", design={"reserve": 0.15}),
    CeresLever("Earth hardware $100", prices={"earth_hardware": 100.0}),
    CeresLever("Ceres film $10", prices={"film": 10.0}),
    CeresLever("freight $3", prices={"freight": 3.0}),
    CeresLever("freight $10", prices={"freight": 10.0}),
)
#: The parent's "all of the above": its seven levers together.
OPTIMISTIC = CeresLever(
    "all the parent's levers together",
    design={"avionics": 0.005},
    prices={
        "polyethylene": 0.2,
        "disk": 10.0,
        "earth_hardware": 10.0,
        "structure": 3.0,
        "misses": 0.0,
        "package": 0.0,
        "package_mass": 0.0,
    },
)


#: The same with the Ceres plates at the Ceres structure price.
OPTIMISTIC_PLATES = CeresLever(
    "and Ceres plates $3",
    design=OPTIMISTIC.design,
    prices={**OPTIMISTIC.prices, "ceres_plate": 3.0},
)
#: Where the optimistic book is re-optimized: around each depth's minimum.
OPTIMISTIC_EXCESSES = {4.0: (90.0, 110.0, 130.0), 9.86: (60.0, 75.0, 90.0, 110.0)}
OPTIMISTIC_SPEEDS = {4.0: (120.0, 137.0, 150.0, 175.0), 9.86: (80.0, 100.0, 120.0)}


def sensitivity_report(
    optima: Dict[float, Tuple[CeresDesign, EnergyLoop]], workers: int = WORKERS
) -> str:
    """Item 7: one at a time on the re-optimized design, then all together re-optimized."""
    rows = [["minimum"] + [_f(optima[d][1].price) for d in cd.DEPTHS]]
    for lever in ENERGY_SENSITIVITIES + (OPTIMISTIC, OPTIMISTIC_PLATES):
        cells = [lever.name]
        for depth in cd.DEPTHS:
            design, _ = optima[depth]
            d, p = cd._apply(design, ENERGY_BOOK, lever)
            try:
                cells.append(_f(energy_loop(d, p).price))
            except ValueError:
                cells.append("does not close")
        rows.append(cells)
    out = [
        "One at a time, design held ($/MWh_e)",
        tabulate(
            rows,
            headers=["lever"] + [f"{d:g} R_sun" for d in cd.DEPTHS],
            tablefmt="github",
        ),
    ]
    for lever in (OPTIMISTIC, OPTIMISTIC_PLATES):
        _, prices = cd._apply(optima[4.0][0], ENERGY_BOOK, lever)
        for depth in cd.DEPTHS:
            jobs: List[Cell] = [
                (depth, w, x_c, prices, dict(lever.design))
                for x_c in OPTIMISTIC_EXCESSES[depth]
                for w in OPTIMISTIC_SPEEDS[depth]
            ]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                found = [r for r in pool.map(_cell, jobs) if r is not None]
            d, loop = min(found, key=lambda r: r[1].price)
            top = sorted(loop.lines.items(), key=lambda kv: -kv[1])[:4]
            out.append(
                f"\n{depth:g} R_sun, {lever.name}, re-optimized (X_C, w, k_sun, arc):"
                f" ${loop.price:.2f}/MWh_e at {loop.w_400:.0f} km/s, X_C {d.ceres_excess:g};"
                f" largest lines ($/MWh_e) "
                + ", ".join(f"{k} {v / loop.mwh:.2f}" for k, v in top)
            )
    return "\n".join(out)


def package_report(optima: Dict[float, Tuple[CeresDesign, EnergyLoop]]) -> str:
    """Item 8: the packages' count and mass, and what their freight costs."""
    rows = []
    for depth, (design, loop) in optima.items():
        massless = energy_loop(design, replace(ENERGY_BOOK, package_mass=0.0))
        l1 = steady_loop(cd.HEADLINE[depth])
        l1_mass = steady_loop(
            cd.HEADLINE[depth], replace(CERES_BOOK, package_mass=PACKAGE_MASS)
        )
        rows.append(
            [
                f"{depth:g}",
                _f(1.0 / IMPACTOR_MASS, 3),
                _f(CERES_BOOK.packages_per_kg, 4),
                _f(loop.lines["packages"], 3),
                _f(loop.price),
                _f(massless.price),
                f"{l1.price:.3f} -> {l1_mass.price:.3f}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "R_sun",
            "packages / kg impactor",
            "/ kg Ceres pellet",
            "packages $/kg caught",
            "$/MWh_e",
            "massless",
            "ADR 0083 $/kg at L1 with 1.5 kg packages",
        ],
        tablefmt="github",
    )


def uncharged_report(optima: Dict[float, Tuple[CeresDesign, EnergyLoop]]) -> str:
    """Item 9: what is left out, sized in $/MWh_e where it can be."""
    design, loop = optima[4.0]
    # Freight lobs: $1M of operations per 1500 t unit lofted.
    leg = freight_leg(replace(design, earth_stream=cd.EARTH_STREAMS[4.0]), ENERGY_BOOK)
    ops = 1.0e6 / cd.LAUNCH_UNIT * leg.lofted * loop.earth_kg / loop.mwh
    caught_per_twyr = 1.0e12 * _YEAR / (loop.mwh * _MWH)
    ships = caught_per_twyr / (loop.energy_ship.cargo * (1.0 - ENERGY_BOOK.misses))
    seed = loop.ceres_ship.stream / loop.ceres_ship.cargo
    rows = [
        ["the straw, its magnetic support, the MHD plant and heat sink", "-"],
        ["the Ceres industry's capital (mining, reactors, plate and film plants)", "-"],
        [
            "the seed: the first Ceres stream",
            f"{seed:.2f} kg of pellets per kg a Ceres ship returns",
        ],
        ["the depot tug's reactor and tanks (its water is charged)", "-"],
        [
            "booster operations on the freight lobs, per $1M a lob",
            _f(ops, 3),
        ],
        [
            "the release ejector that strings each ship's impactors at 20 Hz",
            "see item 6",
        ],
        [
            f"one straw at {STRAW_RATE:g} Hz of {IMPACTOR_MASS:g} kg impactors",
            f"{STRAW_RATE * IMPACTOR_MASS * loop.mwh * _MWH / 1e12:.2f} TW electric",
        ],
        [
            "Earth-bound ships per TW-year of electricity",
            f"{ships:,.0f} ships of 1000 t",
        ],
        ["ship operations (unpriced per ship)", "-"],
    ]
    return tabulate(rows, headers=["not charged", "size ($/MWh_e)"], tablefmt="github")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Print the S58 answers.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--save", help="pickle the sweep's cells here")
    args = parser.parse_args(argv)
    print("Reproduction of the parent's harness\n" + reproduction_report())
    cells = speed_sweep(workers=args.workers)
    if args.save:
        with open(args.save, "wb") as handle:
            pickle.dump(cells, handle)
    sweep, optima = sweep_report(cells)
    print("\nItems 1-3. The energy price swept over arrival speed" + sweep)
    print("\nThe minimum's lines" + line_report(optima))
    print("\nItem 4. Freight re-solved for the energy loop\n" + freight_report(optima))
    print("\nItem 5. The Ceres plate\n" + plate_report(optima))
    print("\nItem 6. Ice impactors\n" + impactor_report(cells, optima))
    print("\nItem 7. Sensitivities\n" + sensitivity_report(optima, args.workers))
    print("\nItem 8. Packages\n" + package_report(optima))
    print("\nItem 9. Not charged\n" + uncharged_report(optima))


if __name__ == "__main__":
    main()
