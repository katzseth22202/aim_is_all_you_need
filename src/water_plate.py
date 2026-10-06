"""The water-injected pusher plate on the overtake leg.

The growth wave catches the craft from behind at closing speed ``w``.  The
craft sprays ``k`` kilograms of carried water per kilogram of PuffSat into the
gas ahead of the plate, and the plate turns the mixture into backward exhaust.
The parent (``sec:water_injected_overtake``) gives the impulse per kilogram of
PuffSat as ``beta w``, with

    beta = eta_jet sqrt(1 + k) + 1

where ``eta_jet = sqrt(eta)`` for an energy efficiency ``eta``, and the ``+1``
is the PuffSat's own momentum, which an overtake keeps as a credit.  As the
craft speeds up it runs away from the wave, so ``w`` falls through the push.
Per kilogram of PuffSat the craft spends ``k`` of water, so with ``v`` the
speed gained and ``M`` the craft's mass,

    dm/dv = M / (beta w),   dM/dv = -k M / (beta w).

At constant ``k`` this integrates to the parent's ``eq:water_plate_delivery``,
``M_f / m_in = k / ((w0/w1)^(k/beta) - 1)``.

The loading need not be constant: the spray can change pulse by pulse.  Water
and PuffSats are both costs -- PuffSats are the fleet, and water is launched
mass that is not payload -- so the push should minimise ``m_in + lambda W`` for
some exchange rate ``lambda`` the growth ledger sets.  Pontryagin's principle
turns that into a pointwise rule.  With ``c`` the running price of water, each
pulse takes the ``k`` that minimises ``(1 + c k) / beta(k)``, which with
``s = sqrt(1 + k)`` and ``a = eta_jet`` is the root of

    c a s^2 + 2 c s + a (c - 1) = 0,

and the price rises through the push as ``dc/dv = (1 + c k) / (beta w)``,
ending at ``lambda``.  So one number, the starting price, fixes the whole
schedule, and ``k`` falls pulse by pulse: water is sprayed most freely early,
when the wave is fastest.

The plate's sprayed film is launched mass too (parent S12, ADR 0043).  It is
burned at a fixed mass per pulse, and pulses are a fixed impulse, so it leaves
at ``phi`` kilograms per newton-second of impulse delivered:
``dM/dv`` gains ``-phi M``.  The loss does not depend on ``k``, so the
pointwise rule is unchanged; the price gains ``c phi``, since a kilogram of
craft carried further burns film on the way.

Left free, the schedule opens at k = 50-100.  The ideal-ceiling ``beta`` is
unvalidated that far above the parent's k = 9-10 candidates, and a cooler,
heavier mixture probably converts worse, so the schedule is clamped at
``PLATE_MAX_SLUG_RATIO`` by default.  The per-pulse cost is single-peaked in
``k``, so the clamped rule is still the Pontryagin optimum under that bound.
Holding k to 10 costs the growth ledger 4-5% on doubling time; the uncapped
schedule is kept as a sensitivity.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from scipy.integrate import solve_ivp

from src.plume_thermal import WATER_ATOMISATION_ENTHALPY, WATER_MOLAR_MASS

#: Largest water loading the plate flies: the top of the parent's k = 9-10
#: candidate operating points (``sec:water_injected_overtake``).
PLATE_MAX_SLUG_RATIO = 10.0
#: A PuffSat or slug with no bonds to pay: argon, or the parent's ideal ceiling.
NO_BONDS = 0.0 * u.J / u.kg
#: A plate whose film is not carried: every push before ADR 0043.
NO_FILM = 0.0 * u.kg / (u.N * u.s)
#: Water's atomisation enthalpy per kilogram, the bond energy ``eq:eta_chem``
#: charges (50.9 MJ/kg), from :mod:`src.plume_thermal`'s constants.
WATER_BOND_ENERGY = (WATER_ATOMISATION_ENTHALPY / WATER_MOLAR_MASS).to(u.J / u.kg)


@dataclass(frozen=True)
class PlateSlug:
    """What the plate sprays.

    Attributes:
        name: Short label.
        bond_energy: Energy per kilogram locked in bonds the frozen plume
            never returns (``eq:eta_chem``).
        tank_fraction: Drop tank per kilogram of slug.  The parent's tanks all
            scale as ~14.6 kg of tank per cubic metre (0.205 x 71, 0.034 x 423,
            0.015 x 1000 kg/m^3), cryogenic ones included.
    """

    name: str
    bond_energy: u.Quantity
    tank_fraction: float


#: Water: 50.9 MJ/kg of bonds, 0.015 kg of tank per kg (``sec:ntr_departure``).
WATER_SLUG = PlateSlug("water", WATER_BOND_ENERGY, 0.015)
#: Argon: no bonds.  Liquid at 1395 kg/m^3, so the density-scaled tank is
#: 14.6/1395 = 0.0105; held passively near 87 K like the parent's LOX.
#:
#: Its ionisation is assumed to come back while the gas still pushes on the
#: plate.  That is not "ionisation is minor": the merge thermalises ~100 MJ/kg
#: of blob at 50 km/s and k = 10, so at peak argon is fully singly ionised
#: (38.1 MJ/kg) and partly doubly (+~67 MJ/kg), well over half its energy.  It
#: is that all of it returns.  By the Saha equation argon is 99% recombined by
#: 10 400-15 700 K across 1 MPa-1 GPa (the parent's water plate peaks at
#: 5.79 GPa), and at 10 000 K only 0.02-0.65% ionised (<= 0.25 MJ/kg); at plate
#: densities three-body recombination is fast, so the gas tracks equilibrium.
#: Gas that spills past the rim freezes ionised, but it is already the capture
#: loss; recombination radiation belongs in eta_geom, as the parent defines it.
#: Water's bonds return only near 3000-4000 K, later in the expansion, which is
#: why water keeps the parent's frozen toll here; that toll is pessimistic on a
#: plate (the companion returned 23-31% of the store).  Argon also gives up
#: water's role as the plate's heat sponge, unpriced.
ARGON_SLUG = PlateSlug("argon", 0.0 * u.J / u.kg, 14.6 / 1395.0)

_RTOL = 1.0e-10
_ATOL = 1.0e-12


def chemistry_ceiling(
    closing_speed: u.Quantity,
    slug_ratio: float,
    slug: PlateSlug,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
) -> float:
    """``eq:eta_chem`` for a blob of impactor and slug of different materials.

    Only the impactor brings energy, ``w^2/2`` per kilogram, and every kilogram
    of the blob pays its own bonds, so

        eta_chem = sqrt(1 - 2 (E_impactor + k E_slug) / w^2).

    With water on an ice PuffSat this is the parent's ``(1 + k) E_a`` form.

    Args:
        closing_speed: Impactor speed relative to the plate, ``w``.
        slug_ratio: Slug per kilogram of impactor, ``k``.
        slug: The slug's material.
        impactor_bond_energy: The PuffSat's bond energy per kilogram; water
            ice by default.

    Returns:
        The chemistry ceiling on the jet efficiency, zero where the bonds
        take the whole budget.
    """
    w = float(closing_speed.to_value(u.m / u.s))
    toll = float(impactor_bond_energy.to_value(u.J / u.kg)) + slug_ratio * float(
        slug.bond_energy.to_value(u.J / u.kg)
    )
    return float(np.sqrt(max(1.0 - 2.0 * toll / (w * w), 0.0)))


@dataclass(frozen=True)
class PlatePush:
    """One overtake push, per kilogram of craft at the start of the push.

    Attributes:
        delivered_fraction: Craft mass after the push over mass before it.
        puffsat_fraction: Growth-wave PuffSat mass the push consumes.
        slug_fraction: Slug (water or argon) the push sprays.
        film_fraction: Plate film the push burns (ADR 0043).
    """

    delivered_fraction: float
    puffsat_fraction: float
    slug_fraction: float
    film_fraction: float

    @property
    def delivered_per_puffsat(self) -> float:
        """Craft mass delivered per kilogram of PuffSat, ``M_f / m_in``."""
        return self.delivered_fraction / self.puffsat_fraction

    @property
    def slug_per_puffsat(self) -> float:
        """Mean loading over the push, ``W / m_in``."""
        return self.slug_fraction / self.puffsat_fraction


def _impulse_factor(
    k: npt.ArrayLike,
    closing_km_s: float,
    jet: float,
    bonds: Tuple[float, float],
) -> npt.NDArray[np.float64]:
    """``beta = eta_geom eta_chem sqrt(1 + k) + 1``, vectorised over ``k``.

    ``bonds`` is the impactor's and the slug's bond energy, in J/kg.
    """
    loading = np.asarray(k, dtype=np.float64)
    w = closing_km_s * 1.0e3
    toll = bonds[0] + loading * bonds[1]
    remaining = np.maximum(1.0 - 2.0 * toll / (w * w), 0.0)
    return np.asarray(jet * np.sqrt((1.0 + loading) * remaining) + 1.0)


def _bonds(slug: PlateSlug, impactor_bond_energy: u.Quantity) -> Tuple[float, float]:
    return (
        float(impactor_bond_energy.to_value(u.J / u.kg)),
        float(slug.bond_energy.to_value(u.J / u.kg)),
    )


def _film_per_km_s(film_per_impulse: u.Quantity) -> float:
    """Film burned per kilogram of craft per km/s gained."""
    return float(film_per_impulse.to_value(u.kg / (u.N * u.s))) * 1.0e3


def _push_ode(
    closing_km_s: float,
    gain: float,
    rhs: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    start: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        solve_ivp(rhs, (0.0, gain), start, rtol=_RTOL, atol=_ATOL).y[:, -1],
        dtype=np.float64,
    )


def plate_push(
    initial_closing_speed: u.Quantity,
    speed_gain: u.Quantity,
    efficiency: float,
    slug_ratio: float,
    slug: PlateSlug = WATER_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    film_per_impulse: u.Quantity = NO_FILM,
) -> PlatePush:
    """Integrate an overtake push at a constant loading.

    Args:
        initial_closing_speed: Wave speed relative to the craft as the push
            starts, ``w0``.
        speed_gain: Speed the push adds; the push ends at ``w0 - speed_gain``.
        efficiency: Energy efficiency net of chemistry, ``eta_geom^2``; the
            chemistry ceiling is charged pulse by pulse.
        slug_ratio: Slug per kilogram of PuffSat, ``k``.
        slug: What the plate sprays.
        impactor_bond_energy: The PuffSat's bond energy per kilogram.
        film_per_impulse: Plate film burned per unit of impulse delivered.

    Returns:
        The push's ledger.
    """
    w0 = float(initial_closing_speed.to_value(u.km / u.s))
    gain = float(speed_gain.to_value(u.km / u.s))
    jet = float(np.sqrt(efficiency))
    bonds = _bonds(slug, impactor_bond_energy)
    phi = _film_per_km_s(film_per_impulse)

    def rhs(v: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        beta = float(_impulse_factor(slug_ratio, w0 - v, jet, bonds))
        rate = y[0] / (beta * (w0 - v))
        film = phi * y[0]
        return np.array([-slug_ratio * rate - film, rate, slug_ratio * rate, film])

    end = _push_ode(w0, gain, rhs, np.array([1.0, 0.0, 0.0, 0.0]))
    return PlatePush(
        delivered_fraction=float(end[0]),
        puffsat_fraction=float(end[1]),
        slug_fraction=float(end[2]),
        film_fraction=float(end[3]),
    )


@dataclass(frozen=True)
class OptimalPlatePush(PlatePush):
    """A push flown on the Pontryagin-optimal loading schedule.

    Attributes:
        slug_ratio_start: Loading of the first pulse.
        slug_ratio_end: Loading of the last pulse.
        final_water_price: The price ``c`` the schedule ends on, which is the
            exchange rate ``lambda`` it is optimal for.
    """

    slug_ratio_start: float
    slug_ratio_end: float
    final_water_price: float


#: Loading grid the per-pulse optimum is found on before a parabolic refine,
#: and the ceiling that stands in for "uncapped".
_LOADING_NODES = 257
_UNCAPPED_LOADING = 400.0


def optimal_slug_ratio(
    water_price: float,
    closing_speed: u.Quantity,
    efficiency: float,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = WATER_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
) -> float:
    """The loading that minimises ``(1 + c k) / beta(k, w)`` at price ``c``.

    With a chemistry toll ``beta`` depends on ``w`` too, so there is no closed
    form; the minimum is taken on a grid and refined by a parabola.

    Args:
        water_price: Price of a kilogram of slug in kilograms of PuffSat, ``c``.
        closing_speed: This pulse's closing speed, ``w``.
        efficiency: Energy efficiency net of chemistry.
        max_slug_ratio: Upper bound on ``k``; None leaves it free.
        slug: What the plate sprays.
        impactor_bond_energy: The PuffSat's bond energy per kilogram.

    Returns:
        The optimal ``k`` within the bound.
    """
    return _best_loading(
        water_price,
        float(closing_speed.to_value(u.km / u.s)),
        float(np.sqrt(efficiency)),
        max_slug_ratio,
        _bonds(slug, impactor_bond_energy),
    )


def _best_loading(
    price: float,
    closing_km_s: float,
    jet: float,
    max_slug_ratio: Optional[float],
    bonds: Tuple[float, float],
) -> float:
    top = _UNCAPPED_LOADING if max_slug_ratio is None else max_slug_ratio
    ks = np.linspace(0.0, top, _LOADING_NODES)
    cost = (1.0 + price * ks) / _impulse_factor(ks, closing_km_s, jet, bonds)
    i = int(np.argmin(cost))
    if i in (0, ks.size - 1):
        return float(ks[i])
    left, mid, right = cost[i - 1], cost[i], cost[i + 1]
    curvature = left - 2.0 * mid + right
    if curvature <= 0.0:
        return float(ks[i])
    step = ks[1] - ks[0]
    return float(ks[i] + 0.5 * step * (left - right) / curvature)


def optimal_plate_push(
    initial_closing_speed: u.Quantity,
    speed_gain: u.Quantity,
    efficiency: float,
    initial_water_price: float,
    max_slug_ratio: Optional[float] = PLATE_MAX_SLUG_RATIO,
    slug: PlateSlug = WATER_SLUG,
    impactor_bond_energy: u.Quantity = WATER_BOND_ENERGY,
    film_per_impulse: u.Quantity = NO_FILM,
) -> OptimalPlatePush:
    """Integrate an overtake push on the Pontryagin-optimal loading schedule.

    Args:
        initial_closing_speed: Wave speed relative to the craft as the push
            starts, ``w0``.
        speed_gain: Speed the push adds.
        efficiency: Energy efficiency net of chemistry, ``eta_geom^2``.
        initial_water_price: The slug price ``c`` at the first pulse.
        max_slug_ratio: Upper bound on ``k``; None leaves it free.
        slug: What the plate sprays.
        impactor_bond_energy: The PuffSat's bond energy per kilogram.
        film_per_impulse: Plate film burned per unit of impulse delivered.

    Returns:
        The push's ledger, with its schedule's end points.
    """
    w0 = float(initial_closing_speed.to_value(u.km / u.s))
    gain = float(speed_gain.to_value(u.km / u.s))
    jet = float(np.sqrt(efficiency))
    bonds = _bonds(slug, impactor_bond_energy)
    phi = _film_per_km_s(film_per_impulse)

    def loading(price: float, closing: float) -> float:
        return _best_loading(price, closing, jet, max_slug_ratio, bonds)

    def rhs(v: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        mass, _, _, price, _ = y
        k = loading(price, w0 - v)
        beta = float(_impulse_factor(k, w0 - v, jet, bonds))
        rate = 1.0 / (beta * (w0 - v))
        return np.array(
            [
                -k * mass * rate - phi * mass,
                mass * rate,
                k * mass * rate,
                (1.0 + price * k) * rate + price * phi,
                phi * mass,
            ]
        )

    end = _push_ode(w0, gain, rhs, np.array([1.0, 0.0, 0.0, initial_water_price, 0.0]))
    return OptimalPlatePush(
        delivered_fraction=float(end[0]),
        puffsat_fraction=float(end[1]),
        slug_fraction=float(end[2]),
        film_fraction=float(end[4]),
        slug_ratio_start=loading(initial_water_price, w0),
        slug_ratio_end=loading(float(end[3]), w0 - gain),
        final_water_price=float(end[3]),
    )
