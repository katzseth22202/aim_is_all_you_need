"""How deep the solar dive has to be: pricing a shallower perihelion.

ADR 0019 flies the Jovian solar-dive cycle to the paper's own 4 solar radii and
scores it at a *stated* 60 percent survival across the periapsis collision.  Two
things about that node are uncomfortable and neither is a trajectory question:
it sits in 3.9 MW/m^2 of sunlight, and the opposing streams meet at 617 km/s.
Backing the perihelion out trades growth for a node a spacecraft can plausibly
be built for.  This module prices that trade.

Four things it does that :mod:`src.jovian_solar_dive_cycle` does not:

* **The dive node is derived, not stated.**  ``periapsis_survival`` there is a
  constant; here it comes from the same **impact-angle impulse law** the Earth
  departure already uses, evaluated head-on at the opposing streams' closing
  speed (:func:`dive_node`).  At 4 solar radii and ``k_p`` = 30 that gives
  0.5977 against the module's stated 0.60 -- so the two are on one device, and
  the shallow rows are not being scored on a different footing from the deep one.
* **The plume has to survive its own expansion, not merely ignite.**  The
  **plume ignition window** is a condition at the instant of the merge; a
  magnetic nozzle does its work by letting the plume expand, which cools it.
  Requiring the plume to still clear the ignition bill *after* the jet is drawn
  gives the **expansion floor**, a window on ``k`` strictly inside the ignition
  window on both sides -- [1.93, 16.03] against [0.021, 47.88] at a 91.78 km/s
  closing speed (:func:`expansion_limited_slug_ratio_window`).  The efficiency
  ceiling it implies (:func:`maximum_jet_efficiency`) is **not** new: ADR 0016
  put a frozen-chemistry ceiling on the same parameter and implemented it as
  ``plume_thermal.chemistry_efficiency``.  This one is strictly tighter, because
  it asks that the plume still *conduct* rather than that the jet stay positive.
* **The slug ratio has three ceilings, and they pull opposite ways.**  Ignition
  and expansion cap ``k`` from above because thin slug cannot hold a plasma;
  the **launch ledger** caps it from above too, because every kilogram of slug
  was lifted.  They are far apart and the binding one is never the obvious one
  (:func:`slug_ratio_ceilings`).
* **The opposing stream is placed, not assumed.**  Backing the dive out makes
  the *prograde* placement cheaper and the *retrograde* placement dearer, which
  is the one direction nothing else in the trade runs
  (:func:`opposing_stream_placement`).

The headline, against ADR 0019's reference cycle:

* **Three synodic periods still closes at 32 solar radii**, prograde and
  unpowered, with +54.48 degrees of Jovian bend margin against +58.80.
* **Growth falls about 12x** at each depth's own constrained optimum, 97.26 to
  8.20 kg returned per impactor kilogram -- but **doubling time only 2.18x**,
  0.4961 yr against 1.0795.  A 64x cooler node costs a factor of two in the
  clock, not a factor of ten.
* **The cap on the slug ratio is the expansion floor, and at the shallow node
  the launch ledger is close behind.**  Neither is the plume ignition window,
  which is the only one the repository had: a search given only that constraint
  picks ``k`` = 23.75, where the jet would have to carry 0.60 of a merge energy
  that can spare 0.369.

ADR ``0020-the-nozzle-cap-is-the-expansion-not-the-ignition``.
Run with ``make dive-depth``.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from scipy.optimize import brentq, minimize_scalar
from tabulate import tabulate

from src.astro_constants import SOLAR_DIVE_PERIAPSIS_SOLAR_RADII
from src.circular_resonance_impulse import impulse_per_impactor_kg
from src.jovian_solar_dive_cycle import (
    DEFAULT_JET_ENERGY_EFFICIENCY,
    DEFAULT_RETURN_EXCESS,
    DEFAULT_SLUG_RATIO,
    CycleGrowthLedger,
    LaunchLedgerVerdict,
    SynodicCycleClosure,
    _dive_periapsis_radius,
    cycle_growth_ledger,
    departure_nozzle_ledger,
    dive_placement_excess_floor,
    launch_ledger_verdict,
    solve_synodic_closure,
)
from src.plume_thermal import (
    NOZZLE_FLOOR_TEMPERATURE,
    plume_ignition_energy,
    slug_ratio_window,
    specific_thermal_energy,
)
from src.retrograde_return_legs import _FlybyParams, _powered_flyby_params
from src.two_leg_nozzle_sweep import RETURN_FLOOR

# Sun's luminosity and the Stefan-Boltzmann constant, for the node's radiation
# environment.  Imported lazily through astropy.constants so the module keeps
# one source of physical constants.
from astropy import constants as const  # isort: skip

_SOLAR_LUMINOSITY = float(const.L_sun.to_value(u.W))
_STEFAN_BOLTZMANN = float(const.sigma_sb.to_value(u.W / u.m**2 / u.K**4))

# --- The conservative operating point this module exists to score. --------
# Perihelion radius (solar radii).  32 puts the node at 45x Earth's insolation
# rather than 2890x, which is the whole reason to consider it.
CONSERVATIVE_DIVE_SOLAR_RADII = 32.0
# Solar hyperbolic-excess speed the boosted climb-out keeps (km/s) -- the same
# quantity DEFAULT_RETURN_EXCESS names, halved.  Note this is *not* the Earth
# closing speed: at 75 the cycle closes at Earth at 85.07 km/s, because Earth's
# 29.8 km/s of transverse motion matters more at low speed than at 150/157.8.
CONSERVATIVE_RETURN_EXCESS = 75.0
# Departure slug ratio (kg of slug per kg of arriving impactor).  Proposed as a
# plasma-retention cap; see slug_ratio_ceilings() for what actually binds.
CONSERVATIVE_SLUG_RATIO = 8.5
# Thermal headroom demanded above the **expansion floor** at the design point.
# 1.0 is the floor itself, where the plume reaches nozzle exit at exactly the
# ignition temperature and the caloric model's own ~3% disagreement with the
# solved surface is the whole margin.  1.5 is a stated design margin, not a
# derived one, chosen to sit below both reference cycles: ADR 0019's k = 30 at
# 158 km/s carries 1.76 and the 32 R_sun k = 8.5 point carries 1.55.
DEFAULT_EXPANSION_MARGIN = 1.5

# Slug ratio at the *dive* node.  Held at the departure module's own default so
# the deep and shallow rows are scored on one device -- and because at 30 it
# reproduces ADR 0019's stated 0.60 survival at 4 solar radii to 0.4%.
DEFAULT_PERIAPSIS_SLUG_RATIO = DEFAULT_SLUG_RATIO

# --- Search bounds.  Recorded here rather than in a scratch harness, per
#     CLAUDE.md after ADR 0007's numbers became unreproducible. -------------
# Dive perihelia swept by depth_trade_table() (solar radii).  The floor is the
# paper's own 4; the ceiling is where the 3S closure's returned mass per pad
# kilogram has clearly stopped paying.
DEPTH_GRID: Tuple[float, ...] = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 48.0)
# Return excesses swept by boost_trade_table() (km/s).  The floor is well below
# any boost worth flying; the ceiling is past the paper's own 150.
RETURN_EXCESS_GRID: Tuple[float, ...] = (
    40.0,
    50.0,
    62.9,
    75.0,
    90.0,
    110.0,
    150.0,
)
# Grid the constrained optimum searches (km/s of climb-out excess, then slug
# ratio).  The excess floor is the slowest return that still closes at every
# depth swept; the ceiling is past where the launch ledger has clearly stopped
# paying at any slug ratio.  The slug-ratio step is fine enough to land within
# 0.3 of the plume ignition window's upper root, which is where the optimum
# actually sits.
OPTIMUM_EXCESS_GRID = tuple(20.0 + 2.5 * n for n in range(73))
OPTIMUM_SLUG_RATIO_STEP = 0.25
OPTIMUM_SLUG_RATIO_FLOOR = 1.0

# Bracket for the growth-optimal climb-out excess (km/s).  The floor is the
# slowest return that still closes at every depth swept; the ceiling is past the
# paper's own 150, so an optimum resting on either edge is reported as absent
# rather than as an interior result.
RETURN_EXCESS_BRACKET = (20.0, 200.0)
# Slug ratios swept by slug_ratio_table().  Spans from below the launch ledger's
# optimum to the plume ignition window's upper root at the shallow node.
SLUG_RATIO_GRID: Tuple[float, ...] = (
    2.0,
    4.0,
    6.0,
    8.5,
    12.0,
    20.0,
    30.0,
    47.0,
)
# Bracket for the launch-ledger ceiling root.  The floor is above the ignition
# window's lower root at every depth swept (the shallowest is 0.021); the
# ceiling is above the window's upper root at every depth swept (47.88 at 32
# solar radii), so the root search never runs off a physical edge.
SLUG_RATIO_SEARCH_BRACKET = (0.05, 60.0)
# Bracket for locating the launch ledger's interior maximum in k.  Delivered
# mass peaks where the nozzle's exhaust speed does, which at these impact angles
# is between k = 1 and k = 3, comfortably inside.
_LAUNCH_PEAK_BRACKET = (0.5, 20.0)
# Bracket for the tangential Earth departure that reaches a given Jupiter
# arrival excess (km/s).  The floor is below the Hohmann departure (8.79); the
# ceiling is past any arrival excess a dive placement asks for.
PLACEMENT_DEPARTURE_BRACKET = (9.0, 30.0)


# --------------------------------------------------------------------------
# The dive node, priced rather than stated
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiveNode:
    """The solar-periapsis collision, with its survival derived from the impulse law.

    Attributes:
        dive_solar_radii: Perihelion distance in solar radii.
        periapsis_radius: The same as a radius from the Sun's centre (km).
        arrival_speed: Speed the payload reaches perihelion with, falling from
            Jupiter's orbit radius (km/s).
        boost: PuffSat boost taken there (km/s).
        departure_speed: Speed leaving perihelion (km/s).
        opposing_closing_speed: Speed the opposing streams meet at, twice
            ``arrival_speed`` for two near-parabolic falls (km/s).
        slug_ratio: Kilograms of slug per kilogram of arriving impactor, ``k_p``.
        impulse_per_impactor: The impulse law's ``beta`` head-on.
        exhaust_speed: Slug-charged effective exhaust speed (km/s).
        specific_impulse: That speed as an Isp (s).
        propellant_fraction: Slug spent as a fraction of the mass at the node.
        survival: Mass fraction surviving the node, ``1 - propellant_fraction``
            read through the rocket equation.
        opposing_impactor_fraction: Opposing-projectile kilograms the node
            consumes per kilogram of vehicle arriving at it.
        thermalised_energy: Collision energy the merge thermalises (MJ/kg).
        solar_flux: Irradiance at the node (W/m^2).
        equilibrium_temperature: Isothermal-sphere equilibrium temperature (K).
    """

    dive_solar_radii: float
    periapsis_radius: float
    arrival_speed: float
    boost: float
    departure_speed: float
    opposing_closing_speed: float
    slug_ratio: float
    impulse_per_impactor: float
    exhaust_speed: float
    specific_impulse: float
    propellant_fraction: float
    survival: float
    opposing_impactor_fraction: float
    thermalised_energy: float
    solar_flux: float
    equilibrium_temperature: float

    @property
    def solar_constants(self) -> float:
        """Irradiance at the node in units of Earth's 1361 W/m^2."""
        return self.solar_flux / 1361.0


def near_parabolic_perihelion_speed(
    dive_solar_radii: float, params: Optional[_FlybyParams] = None
) -> float:
    """Speed at perihelion for a fall from Jupiter's orbit radius.

    Every dive this architecture flies starts from Jupiter's orbit, so its
    perihelion speed is fixed by the depth alone: an ellipse with aphelion at
    Jupiter and perihelion at the node.  It is within 1.4 percent of the local
    escape speed at every depth swept, which is why the node's exhaust speed
    tracks the depth so directly.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        Perihelion speed (km/s).
    """
    p = params if params is not None else _powered_flyby_params()
    r_p = _dive_periapsis_radius(dive_solar_radii)
    semimajor = 0.5 * (p.r_jupiter_orbit + r_p)
    return float(np.sqrt(2.0 * (p.mu_sun / r_p - p.mu_sun / (2.0 * semimajor))))


def opposing_stream_closing_speed(
    dive_solar_radii: float, params: Optional[_FlybyParams] = None
) -> float:
    """Speed the prograde payload and the opposing projectile stream meet at.

    Both arrive on near-parabolic falls from Jupiter's orbit radius, so both
    carry :func:`near_parabolic_perihelion_speed` and the head-on closing speed
    is twice it.  This is the ``w`` the node's impulse law consumes, and it is
    what collapses when the dive is backed out: 616.6 km/s at 4 solar radii
    against 215.3 at 32.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        Closing speed (km/s).
    """
    return 2.0 * near_parabolic_perihelion_speed(dive_solar_radii, params)


def dive_node(
    dive_solar_radii: float,
    boost: float,
    slug_ratio: float = DEFAULT_PERIAPSIS_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> DiveNode:
    """Price the periapsis collision on the same nozzle the Earth departure flies.

    The node is a magnetic nozzle catching an opposing stream exactly head-on,
    so the **impact-angle impulse law** reduces to ADR 0009's
    ``sqrt(eta*(1+k)) - 1`` and the effective exhaust speed is ``beta*w/k``.
    Survival is then the rocket equation on the boost, not a stated fraction --
    which matters because backing the dive out cuts ``w`` faster than it cuts
    the boost, so the node gets *less* efficient as it gets more survivable.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        boost: PuffSat boost taken at perihelion (km/s).
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`DiveNode`.
    """
    p = params if params is not None else _powered_flyby_params()
    r_p = _dive_periapsis_radius(dive_solar_radii)
    arrival = near_parabolic_perihelion_speed(dive_solar_radii, p)
    closing = 2.0 * arrival
    beta = impulse_per_impactor_kg(180.0 * u.deg, slug_ratio, jet_energy_efficiency)
    exhaust = beta * closing / slug_ratio
    survival = float(np.exp(-boost / exhaust))
    radius_metres = r_p * 1000.0
    flux = _SOLAR_LUMINOSITY / (4.0 * np.pi * radius_metres * radius_metres)
    return DiveNode(
        dive_solar_radii=dive_solar_radii,
        periapsis_radius=r_p,
        arrival_speed=arrival,
        boost=boost,
        departure_speed=arrival + boost,
        opposing_closing_speed=closing,
        slug_ratio=slug_ratio,
        impulse_per_impactor=beta,
        exhaust_speed=exhaust,
        specific_impulse=exhaust / 9.80665e-3,
        propellant_fraction=1.0 - survival,
        survival=survival,
        opposing_impactor_fraction=(1.0 - survival) / slug_ratio,
        thermalised_energy=float(
            specific_thermal_energy(closing * u.km / u.s, slug_ratio).to_value(
                u.MJ / u.kg
            )
        ),
        solar_flux=float(flux),
        equilibrium_temperature=float((flux / (4.0 * _STEFAN_BOLTZMANN)) ** 0.25),
    )


def rendezvous_timing_tolerance(node: DiveNode, miss_metres: float = 100.0) -> float:
    """Relative timing error the two streams may carry, for a given along-track miss.

    Both streams must be at the node at the same instant, and the miss they
    accumulate is the closing speed times the timing error.  This is the
    precision argument for backing the dive out, and it scales as ``1/w``.

    Args:
        node: The dive node.
        miss_metres: Along-track miss the nozzle can absorb (m).

    Returns:
        Permissible relative timing error (microseconds).
    """
    return 1.0e6 * miss_metres / (node.opposing_closing_speed * 1000.0)


# --------------------------------------------------------------------------
# What the plume can actually give back: the expansion floor
# --------------------------------------------------------------------------


def frozen_ignition_energy() -> float:
    """The part of the ignition bill that never comes back (MJ/kg).

    Vaporisation, atomisation of the water and ionisation of the seed --
    :func:`plume_thermal.plume_ignition_energy` evaluated at 0 K, so the
    translational term drops out.  ADR 0016 established that this store defaults
    rather than recombining: the plume crosses ``Da = 1`` at the first station
    past the nozzle lip with 90-100 percent of it still held.  So it is paid once
    and stays paid, and only what is left is available to push anything.

    Returns:
        Frozen chemical and seed-ionisation energy per kilogram of blob (MJ/kg).
    """
    return float(plume_ignition_energy(0.0 * u.K).to_value(u.MJ / u.kg))


def ignition_bill() -> float:
    """Specific energy to reach a conducting plume at the nozzle floor temperature.

    Returns:
        The ignition bill per kilogram of blob (MJ/kg).
    """
    return float(plume_ignition_energy().to_value(u.MJ / u.kg))


def jet_energy_fraction(
    slug_ratio: float,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
) -> float:
    """Fraction of the merge's internal energy the jet actually carries off.

    Not ``eta_jet**2`` itself, and the difference is worth a paragraph because
    getting it wrong is lenient rather than conservative.  ``eta_jet**2`` is
    defined against the *ideal one-axis* budget ``w**2 / 2`` per impactor
    kilogram, while the energy the merge really has to give is
    ``w**2 k / (2(1+k))``.  The jet's share of what exists is therefore

        eta * (1 + k) / k

    which is 0.671 at ``k`` = 8.5 rather than 0.600, and diverges as ``k``
    falls -- below ``k`` = ``eta / (1 - eta)`` the impulse law asks for more
    than the collision dissipates, which is why the **expansion floor** has a
    lower root as well as an upper one.

    Args:
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].

    Returns:
        The jet's share of the internal energy, clipped at 1.0.

    Raises:
        ValueError: If the slug ratio is not positive.
    """
    if slug_ratio <= 0.0:
        raise ValueError("slug_ratio must be positive")
    return float(min(1.0, jet_energy_efficiency * (1.0 + slug_ratio) / slug_ratio))


def expansion_residual(
    closing_speed: float,
    slug_ratio: float,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
) -> float:
    """Heat left in the plume once the nozzle has taken its work out (MJ/kg).

    ``eps_th * (1 - jet share)``.  The frozen chemistry sits inside both this
    and the **ignition bill** it is compared against, so nothing is
    double-charged: requiring the residual to clear the whole bill is the same
    statement as requiring the residual *translational* energy to clear the
    bill's 30.94 MJ/kg translational term.

    Args:
        closing_speed: Impactor speed relative to the vehicle at the coldest
            instant of the burn (km/s).
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].

    Returns:
        Residual specific internal energy (MJ/kg); zero when the jet would take
        everything.
    """
    thermalised = float(
        specific_thermal_energy(closing_speed * u.km / u.s, slug_ratio).to_value(
            u.MJ / u.kg
        )
    )
    return max(
        0.0,
        thermalised * (1.0 - jet_energy_fraction(slug_ratio, jet_energy_efficiency)),
    )


def maximum_jet_efficiency(closing_speed: float, slug_ratio: float) -> float:
    """Largest ``eta_jet**2`` that leaves a conducting plume at nozzle exit.

    From ``eps_th * (1 - eta*(1+k)/k) >= bill``:

        eta_jet**2  <=  (k / (1+k)) * (1 - bill / eps_th)

    **This is not a new bound, it is a stricter one.**  ADR 0016 already put a
    frozen-chemistry ceiling on the same parameter and implemented it as
    :func:`plume_thermal.chemistry_efficiency`, which charges the 50.9 MJ/kg
    atomisation toll against the ideal one-axis budget.  That bound asks only
    that the jet energy stay positive after the chemistry is paid; this one
    asks that the plume still be *conducting* when the expansion is done, which
    is what a magnetic nozzle actually needs.  The two agree on verdicts and
    differ on margin -- at 396.7 MJ/kg ADR 0016's ceiling is 0.885 and this one
    is 0.704; at the 84.7 MJ/kg point where an ignition-only search parks, ADR
    0016 gives 0.423 and this gives 0.003, and both are far under a stated 0.60.

    Args:
        closing_speed: Impactor speed relative to the vehicle at the coldest
            instant of the burn (km/s).
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.

    Returns:
        The largest available ``eta_jet**2``, clipped to [0, 1].
    """
    thermalised = float(
        specific_thermal_energy(closing_speed * u.km / u.s, slug_ratio).to_value(
            u.MJ / u.kg
        )
    )
    if thermalised <= 0.0:
        return 0.0
    ratio = slug_ratio / (1.0 + slug_ratio) * (1.0 - ignition_bill() / thermalised)
    return float(min(1.0, max(0.0, ratio)))


def expansion_limited_slug_ratio_window(
    closing_speed: float,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    margin: float = 1.0,
) -> Optional[Tuple[float, float]]:
    """Slug ratios whose plume survives its own expansion at this closing speed.

    A magnetic nozzle does its work by letting the plume expand, which cools it.
    The **plume ignition window** is a condition at the *start* of that
    expansion, so a blob on its upper root lights and then drops out of
    conduction the moment the field takes any work out of it -- the nozzle would
    decouple partway down and the quoted ``eta_jet**2`` would never be reached.
    Requiring the plume to still clear the **ignition bill** after the jet is
    drawn, with ``P = w**2 / 2`` and ``B`` the bill at the demanded margin:

        B k**2 + (2B - P(1 - eta)) k + (B + P eta)  <=  0

    a closed interval like the ignition window and strictly inside it, with a
    lower root that is *not* small: below about ``k`` = 2 the jet's share of the
    internal energy (:func:`jet_energy_fraction`) is what excludes it, not the
    merge energy.  At a 91.78 km/s closing speed this gives [1.93, 16.03]
    against the ignition window's [0.021, 47.88].

    Args:
        closing_speed: Impactor speed relative to the vehicle at the *coldest*
            instant of the burn (km/s).
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1).
        margin: Headroom demanded above the floor.

    Returns:
        ``(k_min, k_max)``, or None when no slug ratio clears the floor at this
        closing speed -- a real outcome, not an error.

    Raises:
        ValueError: If the efficiency is not in (0, 1).
    """
    if not 0.0 < jet_energy_efficiency < 1.0:
        raise ValueError("jet_energy_efficiency must lie in (0, 1)")
    peak = 0.5 * (closing_speed * 1000.0) ** 2
    bill = margin * ignition_bill() * 1.0e6
    linear = 2.0 * bill - peak * (1.0 - jet_energy_efficiency)
    discriminant = linear * linear - 4.0 * bill * (bill + peak * jet_energy_efficiency)
    if discriminant < 0.0:
        return None
    root = float(np.sqrt(discriminant))
    lower = (-linear - root) / (2.0 * bill)
    upper = (-linear + root) / (2.0 * bill)
    if upper <= 0.0:
        return None
    return max(lower, 0.0), upper


# --------------------------------------------------------------------------
# Two ceilings on the departure slug ratio, pulling opposite ways
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlugRatioCeilings:
    """What actually caps the departure slug ratio, read two ways.

    Attributes:
        coldest_closing_speed: Vehicle-frame closing speed at the coldest
            instant of the departure burn (km/s).  On this cycle the impact is
            past 90 degrees, so the vehicle's own acceleration *raises* the
            closing speed and the coldest instant is the burn's start.
        plasma_floor: Lower root of the **plume ignition window**.
        plasma_ceiling: Upper root -- too much slug spreads the collision energy
            below the ignition bill and the nozzle has no plasma to grip.
        expansion_ceiling: The same root against the **expansion floor** rather
            than the bare ignition bill, so the plume is still conducting when
            the nozzle has finished taking its work out.  Roughly half the
            plasma ceiling, and None when no slug ratio clears the floor.
        thermalised_energy: Merge energy at the design slug ratio (MJ/kg).
        expansion_margin: Heat left once the jet is drawn, over the **ignition
            bill**.  Below 1.0 the quoted ``eta_jet**2`` is not reachable and
            the cycle is not merely inefficient but unphysical.
        maximum_jet_efficiency: Largest ``eta_jet**2`` energy conservation
            allows at that merge energy.
        launch_peak_slug_ratio: Slug ratio maximising kilograms returned per
            kilogram off the pad.
        launch_peak_margin: That maximum over the committed 1/15 **return floor**.
            Below 1.0 the floor is failed at *every* slug ratio, which is a
            different statement from having a ceiling and must not be read as
            one -- see :attr:`committed_floor_reachable`.
        launch_ceiling: Slug ratio above which the committed floor fails.  None
            means there is no such crossing, which happens for two opposite
            reasons; read it with ``committed_floor_reachable``.
        rescaled_floor: The committed floor restated for what a kilogram
            returning at this cycle's closing speed is worth
            (``jovian_solar_dive_cycle.return_value_ratio``).
        rescaled_ceiling: The ceiling against that restated floor.
        binding: Which ceiling actually caps the slug ratio -- "expansion",
            "plasma", "launch", or "launch (fails at every k)".
    """

    coldest_closing_speed: float
    plasma_floor: float
    plasma_ceiling: float
    expansion_ceiling: Optional[float]
    thermalised_energy: float
    expansion_margin: float
    maximum_jet_efficiency: float
    launch_peak_slug_ratio: float
    launch_peak_margin: float
    launch_ceiling: Optional[float]
    rescaled_floor: float
    rescaled_ceiling: Optional[float]
    binding: str

    @property
    def committed_floor_reachable(self) -> bool:
        """Whether any slug ratio clears the committed 1/15 **return floor**."""
        return self.launch_peak_margin >= 1.0

    @property
    def rescaled_peak_margin(self) -> float:
        """Best returned-per-pad kilogram over the speed-rescaled floor."""
        return self.launch_peak_margin * RETURN_FLOOR / self.rescaled_floor

    @property
    def rescaled_floor_reachable(self) -> bool:
        """Whether any slug ratio clears the speed-rescaled floor."""
        return self.rescaled_peak_margin >= 1.0


def _returned_per_pad_kg(
    closure: SynodicCycleClosure,
    node: DiveNode,
    slug_ratio: float,
    jet_energy_efficiency: float,
    params: _FlybyParams,
) -> float:
    """Kilograms returning per kilogram off the pad at one departure slug ratio.

    Args:
        closure: The solved cycle.
        node: Its dive node, whose survival the round trip is charged.
        slug_ratio: Departure slug ratio to score.
        jet_energy_efficiency: The paper's ``eta_jet**2``.
        params: Float parameter block.

    Returns:
        Returned kilograms per pad kilogram, or 0.0 where no growth exists.
    """
    try:
        growth = cycle_growth_ledger(
            "probe",
            closure.departure_excess,
            closure.departure_aim_deg,
            closure.total_tof_years,
            closure.return_excess,
            closure.push_axis_deg,
            slug_ratio,
            jet_energy_efficiency,
            node.survival,
            params,
        )
    except ValueError:
        return 0.0
    return launch_ledger_verdict(growth, params).returned_per_pad_kg


def slug_ratio_ceilings(
    closure: SynodicCycleClosure,
    node: DiveNode,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    temperature: u.Quantity = NOZZLE_FLOOR_TEMPERATURE,
    params: Optional[_FlybyParams] = None,
) -> SlugRatioCeilings:
    """Find both ceilings on the departure slug ratio and say which one binds.

    The two run in opposite directions and it is easy to assume the wrong one
    is in charge.  **Plasma** caps ``k`` because a fixed collision energy shared
    over more slug eventually cannot reach the ignition bill -- but the cap
    scales as ``w^2``, so a fast closing speed puts it far out of reach.
    **Expansion** caps it harder than plasma does and for a reason the
    ignition window cannot see, since that window is a condition at the *start*
    of the expansion (:func:`expansion_limited_slug_ratio_window`).
    **Launch** caps ``k`` because slug is lifted from Earth: more slug per
    impactor buys more growth per impactor and less return per pad kilogram, so
    the two ledgers disagree about which direction is better.

    Args:
        closure: The solved cycle.
        node: Its dive node.
        slug_ratio: The design departure slug ratio, at which the merge energy
            and expansion margin are reported.  It does not affect either
            ceiling, since the slug ratio does not enter the burn's kinematics.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        temperature: Plume temperature the nozzle requires.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SlugRatioCeilings`.

    Raises:
        ValueError: If no slug ratio ignites a plume at this closing speed.
    """
    p = params if params is not None else _powered_flyby_params()
    probe = departure_nozzle_ledger(
        closure.departure_excess,
        closure.departure_aim_deg,
        closure.return_excess,
        closure.push_axis_deg,
        slug_ratio,
        jet_energy_efficiency,
        p,
    )
    # The slug ratio does not enter the kinematics, so either probe ratio gives
    # the same closing-speed profile; the coldest instant is what the window
    # must be evaluated at (plume_thermal.slug_ratio_window).
    coldest = min(probe.closing_speed_start, probe.closing_speed_end)
    window = slug_ratio_window(coldest * u.km / u.s, temperature)
    if window is None:
        raise ValueError("no slug ratio ignites a plume at this closing speed")
    plasma_floor, plasma_ceiling = window
    expansion = expansion_limited_slug_ratio_window(coldest, jet_energy_efficiency)
    expansion_ceiling = None if expansion is None else expansion[1]
    thermalised = float(
        specific_thermal_energy(coldest * u.km / u.s, slug_ratio).to_value(u.MJ / u.kg)
    )

    peak = minimize_scalar(
        lambda k: -_returned_per_pad_kg(
            closure, node, float(k), jet_energy_efficiency, p
        ),
        bounds=_LAUNCH_PEAK_BRACKET,
        method="bounded",
        options={"xatol": 1e-8},
    )
    peak_ratio = float(peak.x)
    peak_value = _returned_per_pad_kg(
        closure, node, peak_ratio, jet_energy_efficiency, p
    )

    def ceiling_above(floor: float) -> Optional[float]:
        """Slug ratio where returned-per-pad falls through ``floor``, above the peak."""
        upper = min(SLUG_RATIO_SEARCH_BRACKET[1], plasma_ceiling)
        if peak_value < floor or upper <= peak_ratio:
            return None
        if (
            _returned_per_pad_kg(closure, node, upper, jet_energy_efficiency, p)
            >= floor
        ):
            return None
        return float(
            brentq(
                lambda k: _returned_per_pad_kg(
                    closure, node, float(k), jet_energy_efficiency, p
                )
                - floor,
                peak_ratio,
                upper,
                xtol=1e-8,
            )
        )

    launch_ceiling = ceiling_above(RETURN_FLOOR)
    growth_at_peak = cycle_growth_ledger(
        "probe",
        closure.departure_excess,
        closure.departure_aim_deg,
        closure.total_tof_years,
        closure.return_excess,
        closure.push_axis_deg,
        peak_ratio,
        jet_energy_efficiency,
        node.survival,
        p,
    )
    rescaled_floor = launch_ledger_verdict(growth_at_peak, p).rescaled_floor
    rescaled_ceiling = ceiling_above(rescaled_floor)
    caps = {"plasma": plasma_ceiling}
    if expansion_ceiling is not None:
        caps["expansion"] = expansion_ceiling
    if launch_ceiling is not None:
        caps["launch"] = launch_ceiling
    if peak_value < RETURN_FLOOR:
        binding = "launch (fails at every k)"
    elif expansion_ceiling is None:
        binding = "expansion (no k works)"
    else:
        binding = min(caps, key=lambda name: caps[name])
    return SlugRatioCeilings(
        coldest_closing_speed=coldest,
        plasma_floor=plasma_floor,
        plasma_ceiling=plasma_ceiling,
        expansion_ceiling=expansion_ceiling,
        thermalised_energy=thermalised,
        expansion_margin=(
            expansion_residual(coldest, slug_ratio, jet_energy_efficiency)
            / ignition_bill()
        ),
        maximum_jet_efficiency=maximum_jet_efficiency(coldest, slug_ratio),
        launch_peak_slug_ratio=peak_ratio,
        launch_peak_margin=peak_value / RETURN_FLOOR,
        launch_ceiling=launch_ceiling,
        rescaled_floor=rescaled_floor,
        rescaled_ceiling=rescaled_ceiling,
        binding=binding,
    )


# --------------------------------------------------------------------------
# Placing the opposing stream, which is the one thing depth makes harder
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OpposingStreamPlacement:
    """Whether the projectile stream can reach the node, and what it costs.

    Attributes:
        dive_solar_radii: Perihelion distance in solar radii.
        prograde_floor: Minimum Jupiter arrival excess placing a prograde dive
            to this perihelion (km/s).
        radial_floor: The same for a radial plunge.
        retrograde_floor: The same for a retrograde dive -- the placement the
            head-on collision actually needs.
        cycle_arrival_excess: What the payload's own closure arrives with (km/s).
        retrograde_rides_the_cycle: Whether that arrival is hot enough to place
            the opposing stream too, so one departure energy serves both.
        retrograde_tangential_departure: Earth departure excess a *separate*,
            one-way tangential launch needs to reach ``retrograde_floor`` at
            Jupiter (km/s).
        direct_from_earth_excess: Earth departure excess for the cheapest
            retrograde-perihelion dive placed from 1 AU without Jupiter (km/s).
    """

    dive_solar_radii: float
    prograde_floor: float
    radial_floor: float
    retrograde_floor: float
    cycle_arrival_excess: float
    retrograde_rides_the_cycle: bool
    retrograde_tangential_departure: float
    direct_from_earth_excess: float


def direct_retrograde_placement_excess(
    dive_solar_radii: float, params: Optional[_FlybyParams] = None
) -> float:
    """Cheapest Earth departure excess placing a retrograde perihelion from 1 AU.

    Closed form, not a search.  The minimum-energy orbit with a retrograde
    perihelion at ``r_p`` and reaching 1 AU has 1 AU as its *aphelion* -- any
    radial speed there only adds energy, and for a fixed perihelion more energy
    means a larger angular momentum and therefore a larger retrograde tangential
    speed, so both terms of the excess grow together.  The excess is then
    Earth's own orbital speed plus that aphelion speed, because the two point
    opposite ways.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        Earth-relative hyperbolic-excess speed (km/s).
    """
    p = params if params is not None else _powered_flyby_params()
    r_p = _dive_periapsis_radius(dive_solar_radii)
    aphelion_speed = np.sqrt(
        2.0 * p.mu_sun * r_p / (p.r_earth_orbit * (p.r_earth_orbit + r_p))
    )
    return p.v_earth_orbit + float(aphelion_speed)


def tangential_arrival_excess(
    departure_excess: float, params: Optional[_FlybyParams] = None
) -> Optional[float]:
    """Jupiter arrival excess for a tangential Earth departure at this excess.

    A tangential departure is the cheapest way to reach a given arrival excess,
    so this is the right yardstick for a *one-way* projectile launch whose aim
    is not pinned by a **synodic closure**.

    Args:
        departure_excess: Earth-relative departure excess speed (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        Jupiter-relative arrival excess speed (km/s), or None if the transfer's
        aphelion does not reach Jupiter's orbit.
    """
    p = params if params is not None else _powered_flyby_params()
    v_t = p.v_earth_orbit + departure_excess
    energy = 0.5 * v_t * v_t - p.mu_sun / p.r_earth_orbit
    if energy + p.mu_sun / p.r_jupiter_orbit <= 0.0:
        return None
    speed = np.sqrt(2.0 * (energy + p.mu_sun / p.r_jupiter_orbit))
    v_t_jupiter = v_t * p.r_earth_orbit / p.r_jupiter_orbit
    v_r_jupiter = np.sqrt(max(0.0, speed * speed - v_t_jupiter * v_t_jupiter))
    return float(np.hypot(v_t_jupiter - p.v_jupiter_orbit, v_r_jupiter))


def opposing_stream_placement(
    dive_solar_radii: float,
    closure: SynodicCycleClosure,
    params: Optional[_FlybyParams] = None,
) -> OpposingStreamPlacement:
    """Ask whether the head-on projectile stream can be placed at this depth.

    The **dive-placement floor** is not symmetric about depth: a shallower
    perihelion needs *less* of Jupiter's 13.06 km/s cancelled for a prograde
    dive and *more* reversed for a retrograde one, so backing the dive out
    spreads the two floors apart.  Every other quantity in this trade gets
    easier as the dive gets shallower; this one does not.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        closure: The payload's solved cycle, whose arrival excess is the test.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`OpposingStreamPlacement`.
    """
    p = params if params is not None else _powered_flyby_params()
    r_p = _dive_periapsis_radius(dive_solar_radii)
    retrograde = dive_placement_excess_floor(-1.0, r_p, p)
    tangential = float(
        brentq(
            lambda x: (tangential_arrival_excess(float(x), p) or 0.0) - retrograde,
            *PLACEMENT_DEPARTURE_BRACKET,
            xtol=1e-9,
        )
    )
    return OpposingStreamPlacement(
        dive_solar_radii=dive_solar_radii,
        prograde_floor=dive_placement_excess_floor(1.0, r_p, p),
        radial_floor=dive_placement_excess_floor(0.0, r_p, p),
        retrograde_floor=retrograde,
        cycle_arrival_excess=closure.jupiter_arrival_excess,
        retrograde_rides_the_cycle=closure.jupiter_arrival_excess >= retrograde,
        retrograde_tangential_departure=tangential,
        direct_from_earth_excess=direct_retrograde_placement_excess(
            dive_solar_radii, p
        ),
    )


# --------------------------------------------------------------------------
# One row of the trade
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthTradeRow:
    """One dive depth, closed and scored end to end.

    Attributes:
        label: Name for the row.
        closure: The solved **synodic closure**.
        node: Its **dive node**, survival derived rather than stated.
        growth: What the cycle multiplies, impactor kilograms the scarce input.
        launch: The same charged for launched mass, three ways.
        ceilings: What caps the departure slug ratio, and which cap binds.
        placement: Whether the opposing stream can reach the node.
    """

    label: str
    closure: SynodicCycleClosure
    node: DiveNode
    growth: CycleGrowthLedger
    launch: LaunchLedgerVerdict
    ceilings: SlugRatioCeilings
    placement: OpposingStreamPlacement


def depth_trade_row(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    return_excess: float = CONSERVATIVE_RETURN_EXCESS,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    periapsis_slug_ratio: float = DEFAULT_PERIAPSIS_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    synodic_multiple: int = 3,
    label: Optional[str] = None,
    params: Optional[_FlybyParams] = None,
) -> Optional[DepthTradeRow]:
    """Close the cycle at one depth and score every ledger on it.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        slug_ratio: Departure slug ratio.
        periapsis_slug_ratio: Slug ratio at the dive node.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        synodic_multiple: Earth--Jupiter synodic periods per cycle.
        label: Name for the row; derived from the inputs when omitted.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`DepthTradeRow`, or None when no closure exists at this depth
        or the departure burn delivers nothing.
    """
    p = params if params is not None else _powered_flyby_params()
    closure = solve_synodic_closure(
        synodic_multiple, p, dive_solar_radii, return_excess, 1.0
    )
    if closure is None:
        return None
    node = dive_node(
        dive_solar_radii,
        closure.periapsis_boost,
        periapsis_slug_ratio,
        jet_energy_efficiency,
        p,
    )
    name = (
        label
        or f"{dive_solar_radii:g} R_sun / v_inf {return_excess:g} / k {slug_ratio:g}"
    )
    try:
        growth = cycle_growth_ledger(
            name,
            closure.departure_excess,
            closure.departure_aim_deg,
            closure.total_tof_years,
            closure.return_excess,
            closure.push_axis_deg,
            slug_ratio,
            jet_energy_efficiency,
            node.survival,
            p,
        )
    except ValueError:
        return None
    return DepthTradeRow(
        label=name,
        closure=closure,
        node=node,
        growth=growth,
        launch=launch_ledger_verdict(growth, p),
        ceilings=slug_ratio_ceilings(
            closure, node, slug_ratio, jet_energy_efficiency, params=p
        ),
        placement=opposing_stream_placement(dive_solar_radii, closure, p),
    )


def depth_trade_table(
    depths: Sequence[float] = DEPTH_GRID,
    return_excess: float = CONSERVATIVE_RETURN_EXCESS,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> List[DepthTradeRow]:
    """Sweep the dive depth at a fixed climb-out excess.

    Holding ``return_excess`` rather than the boost is what makes the sweep
    readable: the Earth end barely moves (the closing speed varies 89.8 to
    85.1 km/s across the whole grid) so essentially all the variation is at the
    dive node, which is the thing being traded.

    Args:
        depths: Perihelion distances to sweep (solar radii).
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        slug_ratio: Departure slug ratio.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        One :class:`DepthTradeRow` per depth that closes.
    """
    p = params if params is not None else _powered_flyby_params()
    rows = [
        depth_trade_row(
            float(depth),
            return_excess,
            slug_ratio,
            params=p,
            jet_energy_efficiency=jet_energy_efficiency,
        )
        for depth in depths
    ]
    return [row for row in rows if row is not None]


def boost_trade_table(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    return_excesses: Sequence[float] = RETURN_EXCESS_GRID,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> List[DepthTradeRow]:
    """Sweep how hard to boost at a fixed depth.

    The result is the module's least expected: at a shallow node the boost is a
    *pure cost*.  The departure burn is nearly fixed at 5.65 km/s whatever the
    stream arrives at, so a hotter return buys almost nothing at Earth, while
    the node's propellant bill is exponential in the boost.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        return_excesses: Climb-out excess speeds to sweep (km/s).
        slug_ratio: Departure slug ratio.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        One :class:`DepthTradeRow` per excess that closes.
    """
    p = params if params is not None else _powered_flyby_params()
    rows = [
        depth_trade_row(
            dive_solar_radii,
            float(excess),
            slug_ratio,
            params=p,
            jet_energy_efficiency=jet_energy_efficiency,
        )
        for excess in return_excesses
    ]
    return [row for row in rows if row is not None]


def growth_optimal_return_excess(
    dive_solar_radii: float,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> Optional[float]:
    """Climb-out excess maximising returned mass per impactor kilogram at one depth.

    The two terms pull against each other.  A hotter return raises the Earth
    closing speed, so the departure burn is flown at a better exhaust speed and
    more mass reaches Jupiter; but the boost that bought it is charged against
    the **dive node**'s exhaust speed, which is fixed by the depth.  Deep, the
    node runs at 68 km/s and the boost is nearly free, so the optimum sits above
    the top of ``RETURN_EXCESS_BRACKET``.  Shallow, the node runs at 24 km/s and
    the optimum collapses to the bracket floor: **at 32 solar radii the boost is
    a pure cost**, which is the single most counter-intuitive result here.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        slug_ratio: Departure slug ratio.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The optimising climb-out excess (km/s), or None when it lies on a
        bracket edge and is therefore not an interior optimum this can report.
    """
    p = params if params is not None else _powered_flyby_params()

    def growth_at(excess: float) -> float:
        """Returned mass per impactor kilogram, negated for minimisation."""
        row = depth_trade_row(
            dive_solar_radii,
            excess,
            slug_ratio,
            jet_energy_efficiency=jet_energy_efficiency,
            params=p,
        )
        return 0.0 if row is None else -row.growth.return_per_impactor_kg

    result = minimize_scalar(
        growth_at,
        bounds=RETURN_EXCESS_BRACKET,
        method="bounded",
        options={"xatol": 1e-4},
    )
    best = float(result.x)
    edge = 1e-2 * (RETURN_EXCESS_BRACKET[1] - RETURN_EXCESS_BRACKET[0])
    if best - RETURN_EXCESS_BRACKET[0] < edge or RETURN_EXCESS_BRACKET[1] - best < edge:
        return None
    return best


@dataclass(frozen=True)
class ConstrainedOptimum:
    """The fastest-growing cycle at one depth that both ceilings permit.

    Attributes:
        row: The winning cycle, fully scored.
        return_excess: Its climb-out excess (km/s).
        slug_ratio: Its departure slug ratio.
        limited_by: Which ceiling the winner sits on -- "expansion" or "launch".
        plasma_ceiling: The **expansion floor**'s upper root on ``k`` there, at
            the margin the search was given.
    """

    row: DepthTradeRow
    return_excess: float
    slug_ratio: float
    limited_by: str
    plasma_ceiling: float


def constrained_growth_optimum(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    return_floor: float = RETURN_FLOOR,
    params: Optional[_FlybyParams] = None,
) -> Optional[ConstrainedOptimum]:
    """Maximise growth over climb-out excess and slug ratio, honouring both caps.

    The two knobs are coupled and neither can be set alone, which is why this
    exists rather than a pair of one-dimensional sweeps.  Boosting less at the
    node lowers the Earth closing speed, which *narrows* the **plume ignition
    window** and lowers the slug ratio the nozzle may carry -- but it also spends
    far less of the node's propellant, which is what the **launch ledger**
    charges.  So the boost buys slug-ratio headroom in one direction and spends
    launch margin in the other, and the optimum sits where those meet.

    The cap that binds is the **expansion floor**, not the ignition window: a
    search given only the latter parks on its upper root, which is exactly where
    the plume ignites and then decouples the instant the field takes work out of
    it.  ``expansion_margin`` exists because an optimiser will sit on whatever
    boundary it is handed, and a boundary is not an operating point.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        expansion_margin: Thermal headroom demanded above the **expansion
            floor**.  1.0 puts the winner exactly on it.
        return_floor: Kilograms returned per pad kilogram a cycle must clear.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`ConstrainedOptimum`, or None if no cycle on the grid closes
        and clears the floor.
    """
    p = params if params is not None else _powered_flyby_params()
    winner: Optional[Tuple[float, float, float, float]] = None
    for excess in OPTIMUM_EXCESS_GRID:
        closure = solve_synodic_closure(3, p, dive_solar_radii, excess, 1.0)
        if closure is None:
            continue
        node = dive_node(
            dive_solar_radii,
            closure.periapsis_boost,
            DEFAULT_PERIAPSIS_SLUG_RATIO,
            jet_energy_efficiency,
            p,
        )
        # The slug ratio does not enter the burn's kinematics, so one probe
        # fixes the closing-speed profile and therefore both windows.
        probe = departure_nozzle_ledger(
            closure.departure_excess,
            closure.departure_aim_deg,
            closure.return_excess,
            closure.push_axis_deg,
            CONSERVATIVE_SLUG_RATIO,
            jet_energy_efficiency,
            p,
        )
        coldest = min(probe.closing_speed_start, probe.closing_speed_end)
        window = expansion_limited_slug_ratio_window(
            coldest, jet_energy_efficiency, expansion_margin
        )
        if window is None:
            continue
        ceiling = window[1]
        ratio = max(OPTIMUM_SLUG_RATIO_FLOOR, window[0])
        while ratio < ceiling:
            held = ratio
            ratio += OPTIMUM_SLUG_RATIO_STEP
            try:
                growth = cycle_growth_ledger(
                    "probe",
                    closure.departure_excess,
                    closure.departure_aim_deg,
                    closure.total_tof_years,
                    closure.return_excess,
                    closure.push_axis_deg,
                    held,
                    jet_energy_efficiency,
                    node.survival,
                    p,
                )
            except ValueError:
                continue
            if launch_ledger_verdict(growth, p).returned_per_pad_kg < return_floor or (
                winner is not None and growth.return_per_impactor_kg <= winner[0]
            ):
                continue
            winner = (growth.return_per_impactor_kg, excess, held, ceiling)
    if winner is None:
        return None
    _, excess, held, ceiling = winner
    row = depth_trade_row(
        dive_solar_radii,
        excess,
        held,
        jet_energy_efficiency=jet_energy_efficiency,
        params=p,
    )
    if row is None:
        return None
    return ConstrainedOptimum(
        row=row,
        return_excess=excess,
        slug_ratio=held,
        limited_by=(
            "expansion" if held >= ceiling - 2.0 * OPTIMUM_SLUG_RATIO_STEP else "launch"
        ),
        plasma_ceiling=ceiling,
    )


def slug_ratio_table(
    row: DepthTradeRow,
    slug_ratios: Sequence[float] = SLUG_RATIO_GRID,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> List[Tuple[float, CycleGrowthLedger, LaunchLedgerVerdict]]:
    """Score one closed cycle across departure slug ratios.

    Args:
        row: A solved row whose closure and dive node are held fixed.
        slug_ratios: Departure slug ratios to score.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        ``(slug_ratio, growth, launch)`` for each ratio that delivers mass.
    """
    p = params if params is not None else _powered_flyby_params()
    out: List[Tuple[float, CycleGrowthLedger, LaunchLedgerVerdict]] = []
    for ratio in slug_ratios:
        try:
            growth = cycle_growth_ledger(
                f"k = {ratio:g}",
                row.closure.departure_excess,
                row.closure.departure_aim_deg,
                row.closure.total_tof_years,
                row.closure.return_excess,
                row.closure.push_axis_deg,
                float(ratio),
                jet_energy_efficiency,
                row.node.survival,
                p,
            )
        except ValueError:
            continue
        out.append((float(ratio), growth, launch_ledger_verdict(growth, p)))
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _format_ceiling(value: Optional[float], reachable: bool) -> str:
    """Render an optional slug-ratio ceiling for a table cell.

    A missing ceiling is ambiguous on its own -- it means the floor is cleared
    at every slug ratio, or failed at every slug ratio -- so the two are spelled
    out rather than both printed as "none".

    Args:
        value: The ceiling, or None when there is no crossing.
        reachable: Whether any slug ratio clears the floor at all.

    Returns:
        A short cell string.
    """
    if value is not None:
        return f"{value:.2f}"
    return "cleared throughout" if reachable else "FAILS AT EVERY k"


def describe_row(row: DepthTradeRow) -> str:
    """Render one depth as the block report the ADR quotes.

    Args:
        row: The row to render.

    Returns:
        A multi-line report.
    """
    c, n, g, v, k = row.closure, row.node, row.growth, row.launch, row.ceilings
    lines = [
        f"=== {row.label}",
        "",
        f"  cycle                          {c.synodic_multiple}S, {c.total_tof_years:.4f} yr"
        f"   ({c.outbound_tof_years:.3f} out / {c.dive_tof_years:.3f} dive"
        f" / {c.climb_tof_days:.2f} d climb)",
        f"  Earth departure excess         {c.departure_excess:.4f} km/s"
        f"   aim {c.departure_aim_deg:+.2f} deg",
        f"    speed needed at 200 km       {c.departure_speed_200km:.4f} km/s",
        f"  Jupiter arrival excess         {c.jupiter_arrival_excess:.4f} km/s",
        f"  bend required / available      {c.bend_required_deg:.2f} / "
        f"{c.bend_available_deg:.2f} deg   margin {-c.bend_deficit_deg:+.2f}",
        f"  Earth re-intercept miss        {c.intercept_miss_deg:.4f} deg",
        "",
        f"  DIVE NODE at {n.dive_solar_radii:g} R_sun",
        f"    arrive / boost / leave       {n.arrival_speed:.2f} / {n.boost:.2f}"
        f" / {n.departure_speed:.2f} km/s",
        f"    opposing closing speed       {n.opposing_closing_speed:.1f} km/s"
        f"   (k_p = {n.slug_ratio:g}, beta = {n.impulse_per_impactor:.3f})",
        f"    effective exhaust speed      {n.exhaust_speed:.2f} km/s"
        f"   (Isp {n.specific_impulse:.0f} s)",
        f"    propellant fraction          {n.propellant_fraction:.3f}"
        f"   -> survival {n.survival:.4f}",
        f"    opposing impactors consumed  {100 * n.opposing_impactor_fraction:.2f}%"
        " of the vehicle at the node",
        f"    solar flux / T_eq            {n.solar_flux:.3e} W/m^2"
        f" ({n.solar_constants:.0f}x Earth) / {n.equilibrium_temperature:.0f} K",
        f"    100 m rendezvous miss        {rendezvous_timing_tolerance(n):.1f} us"
        " of relative timing",
        "",
        f"  EARTH DEPARTURE NOZZLE k = {g.nozzle.slug_ratio:g}",
        f"    cant (aim separation)        {c.aim_separation_deg:.2f} deg",
        f"    impact angle                 {g.nozzle.impact_angle_start_deg:.1f}"
        f" -> {g.nozzle.impact_angle_end_deg:.1f} deg",
        f"    closing speed                {g.nozzle.closing_speed_start:.2f}"
        f" -> {g.nozzle.closing_speed_end:.2f} km/s",
        f"    effective exhaust speed      {g.nozzle.effective_exhaust_speed:.2f} km/s"
        f"   (Isp {g.nozzle.specific_impulse:.0f} s)",
        f"    burn                         {g.nozzle.delta_v:.4f} km/s",
        f"    MASS FRACTION TO JUPITER     {g.nozzle.delivered_fraction:.4f}",
        f"    slug / impactor per kg       {g.nozzle.slug_per_delivered_kg:.3f}"
        f" / {g.nozzle.impactor_per_delivered_kg:.4f}",
        "",
        f"  SLUG-RATIO CEILINGS (coldest closing speed {k.coldest_closing_speed:.2f} km/s)",
        f"    plume ignition window        [{k.plasma_floor:.3f}, {k.plasma_ceiling:.2f}]",
        f"    expansion ceiling            "
        f"{_format_ceiling(k.expansion_ceiling, False)}"
        f"   (plume still conducting at nozzle exit)",
        f"    merge energy at this k       {k.thermalised_energy:.1f} MJ/kg",
        f"    heat left after the jet      "
        f"{k.expansion_margin * ignition_bill():.1f} MJ/kg"
        f"   = {k.expansion_margin:.2f}x the ignition bill",
        f"    eta_jet^2 available          {k.maximum_jet_efficiency:.3f}",
        f"    launch ledger peaks at k     {k.launch_peak_slug_ratio:.3f}"
        f"   ({k.launch_peak_margin:.3f}x the committed 1/15 floor)",
        f"    committed-floor ceiling      "
        f"{_format_ceiling(k.launch_ceiling, k.committed_floor_reachable)}",
        f"    rescaled-floor ceiling       "
        f"{_format_ceiling(k.rescaled_ceiling, k.rescaled_floor_reachable)}",
        f"    BINDING CEILING              {k.binding}",
        "",
        f"  GROWTH   round trip {g.round_trip_fraction:.4f}"
        f"   = departs {g.nozzle.delivered_fraction:.4f}"
        f" x survives {n.survival:.4f}",
        f"    per impactor kg              {g.return_per_impactor_kg:.2f} kg",
        f"    e-folds / yr                 {g.growth_rate:.4f}",
        f"    doubling / millionfold       {g.doubling_years:.4f} yr"
        f" / {g.millionfold_years:.2f} yr",
        "",
        f"  LAUNCH LEDGER  returned per pad kg {v.returned_per_pad_kg:.4f}",
        f"    vs committed 1/15            {v.stated_margin:.2f}x",
        f"    vs rescaled 1/{1 / v.rescaled_floor:.1f}"
        f"{'':>13} {v.rescaled_margin:.2f}x"
        f"   (a {v.return_value_ratio:.2f}x more valuable returned kg)",
        f"    time-normalised              {v.returned_per_pad_kg_per_year:.4f}"
        " kg per pad kg per year",
        "",
        f"  OPPOSING STREAM at {row.placement.dive_solar_radii:g} R_sun",
        f"    placement floors             prograde {row.placement.prograde_floor:.4f}"
        f" / radial {row.placement.radial_floor:.4f}"
        f" / retrograde {row.placement.retrograde_floor:.4f} km/s",
        f"    this cycle arrives with      {row.placement.cycle_arrival_excess:.4f} km/s"
        f"   -> retrograde rides the cycle: "
        f"{'yes' if row.placement.retrograde_rides_the_cycle else 'NO'}",
        f"    one-way tangential departure {row.placement.retrograde_tangential_departure:.3f}"
        " km/s of Earth excess",
        f"    direct from 1 AU instead     {row.placement.direct_from_earth_excess:.2f}"
        " km/s of Earth excess",
    ]
    return "\n".join(lines)


def depth_table_text(rows: Sequence[DepthTradeRow]) -> str:
    """Render a depth sweep as a table.

    Args:
        rows: Rows to render.

    Returns:
        The formatted table.
    """
    return str(
        tabulate(
            [
                [
                    f"{r.node.dive_solar_radii:g}",
                    f"{r.node.arrival_speed:.1f}",
                    f"{r.node.boost:.2f}",
                    f"{r.closure.earth_closing_speed:.2f}",
                    f"{-r.closure.bend_deficit_deg:+.2f}",
                    f"{r.closure.aim_separation_deg:.1f}",
                    f"{r.node.exhaust_speed:.1f}",
                    f"{r.node.survival:.4f}",
                    f"{r.growth.nozzle.delivered_fraction:.4f}",
                    f"{r.growth.return_per_impactor_kg:.2f}",
                    f"{r.growth.doubling_years:.3f}",
                    f"{r.launch.stated_margin:.2f}",
                    f"{r.node.equilibrium_temperature:.0f}",
                ]
                for r in rows
            ],
            headers=[
                "R_sun",
                "v_peri",
                "boost",
                "v_b",
                "bend",
                "cant",
                "v_e node",
                "survive",
                "departs",
                "kg/imp",
                "dbl yr",
                "1/15 x",
                "T_eq K",
            ],
            tablefmt="github",
        )
    )


def slug_ratio_table_text(
    entries: Sequence[Tuple[float, CycleGrowthLedger, LaunchLedgerVerdict]],
) -> str:
    """Render a slug-ratio sweep as a table.

    Args:
        entries: ``(slug_ratio, growth, launch)`` triples.

    Returns:
        The formatted table.
    """
    return str(
        tabulate(
            [
                [
                    f"{ratio:g}",
                    f"{growth.nozzle.specific_impulse:.0f}",
                    f"{growth.nozzle.delivered_fraction:.4f}",
                    f"{growth.return_per_impactor_kg:.2f}",
                    f"{growth.doubling_years:.4f}",
                    f"{verdict.returned_per_pad_kg:.4f}",
                    f"{verdict.stated_margin:.2f}",
                    f"{verdict.rescaled_margin:.2f}",
                ]
                for ratio, growth, verdict in entries
            ],
            headers=[
                "k",
                "Isp s",
                "departs",
                "kg/imp",
                "dbl yr",
                "kg/pad kg",
                "1/15 x",
                "rescaled x",
            ],
            tablefmt="github",
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description="Price a shallower solar dive against ADR 0019's 4 R_sun cycle."
    )
    parser.add_argument(
        "--dive-solar-radii",
        type=float,
        default=CONSERVATIVE_DIVE_SOLAR_RADII,
        help="Perihelion of the conservative cycle, in solar radii.",
    )
    parser.add_argument(
        "--return-excess",
        type=float,
        default=CONSERVATIVE_RETURN_EXCESS,
        help="Solar hyperbolic-excess speed of its climb-out (km/s).",
    )
    parser.add_argument(
        "--slug-ratio",
        type=float,
        default=CONSERVATIVE_SLUG_RATIO,
        help="Departure slug ratio for the conservative cycle.",
    )
    parser.add_argument(
        "--jet-efficiency",
        type=float,
        default=DEFAULT_JET_ENERGY_EFFICIENCY,
        help="The paper's eta_jet**2, in (0, 1].",
    )
    parser.add_argument(
        "--optimum",
        action="store_true",
        help=(
            "Also search each depth's growth optimum over climb-out excess and "
            "slug ratio, honouring the expansion floor and the launch ledger. "
            "Off by default because it re-solves the closure at every point on "
            "OPTIMUM_EXCESS_GRID and takes a few minutes."
        ),
    )
    parser.add_argument(
        "--expansion-margin",
        type=float,
        default=DEFAULT_EXPANSION_MARGIN,
        help="Thermal headroom demanded above the expansion floor by --optimum.",
    )
    return parser


def main() -> None:
    """Print the depth trade: the two named cycles and the sweeps behind them."""
    args = _build_parser().parse_args()
    params = _powered_flyby_params()

    deep = depth_trade_row(
        SOLAR_DIVE_PERIAPSIS_SOLAR_RADII,
        DEFAULT_RETURN_EXCESS,
        DEFAULT_SLUG_RATIO,
        jet_energy_efficiency=args.jet_efficiency,
        label="ADR 0019 reference: 4 R_sun, 150 km/s climb-out, k = 30",
        params=params,
    )
    shallow = depth_trade_row(
        args.dive_solar_radii,
        args.return_excess,
        args.slug_ratio,
        jet_energy_efficiency=args.jet_efficiency,
        label=(
            f"conservative: {args.dive_solar_radii:g} R_sun,"
            f" {args.return_excess:g} km/s climb-out, k = {args.slug_ratio:g}"
        ),
        params=params,
    )
    if deep is None or shallow is None:
        raise SystemExit("no closure at one of the two reference depths")

    print(describe_row(deep))
    print()
    print(describe_row(shallow))
    print()
    print(
        f"Backing the dive out costs"
        f" {deep.growth.return_per_impactor_kg / shallow.growth.return_per_impactor_kg:.2f}x"
        " the growth per impactor kilogram and"
        f" {shallow.growth.doubling_years / deep.growth.doubling_years:.2f}x"
        " the doubling time."
    )
    print()

    print(f"Dive depth at a fixed {args.return_excess:g} km/s climb-out")
    print(
        depth_table_text(
            depth_trade_table(
                return_excess=args.return_excess,
                slug_ratio=args.slug_ratio,
                jet_energy_efficiency=args.jet_efficiency,
                params=params,
            )
        )
    )
    print()

    print(f"How hard to boost at {args.dive_solar_radii:g} R_sun")
    print(
        depth_table_text(
            boost_trade_table(
                args.dive_solar_radii,
                slug_ratio=args.slug_ratio,
                jet_energy_efficiency=args.jet_efficiency,
                params=params,
            )
        )
    )
    print()

    if args.optimum:
        print(
            f"Growth optimum at each depth, expansion margin"
            f" {args.expansion_margin:g}x and the committed 1/15 return floor"
        )
        for depth in (SOLAR_DIVE_PERIAPSIS_SOLAR_RADII, args.dive_solar_radii):
            optimum = constrained_growth_optimum(
                depth,
                jet_energy_efficiency=args.jet_efficiency,
                expansion_margin=args.expansion_margin,
                params=params,
            )
            if optimum is None:
                print(f"  {depth:g} R_sun: nothing clears both floors")
                continue
            best = optimum.row
            print(
                f"  {depth:>5g} R_sun: climb-out {optimum.return_excess:6.1f} km/s,"
                f" k = {optimum.slug_ratio:5.2f} (limited by {optimum.limited_by}),"
                f" boost {best.node.boost:5.2f}, v_b {best.closure.earth_closing_speed:6.2f}"
            )
            print(
                f"{'':>16} {best.growth.return_per_impactor_kg:6.2f} kg per impactor kg,"
                f" doubling {best.growth.doubling_years:.4f} yr,"
                f" millionfold {best.growth.millionfold_years:.2f} yr,"
                f" launch {best.launch.stated_margin:.2f}x"
            )
        print()

    for row in (shallow, deep):
        print(f"Departure slug ratio on the {row.node.dive_solar_radii:g} R_sun cycle")
        print(
            slug_ratio_table_text(
                slug_ratio_table(
                    row, jet_energy_efficiency=args.jet_efficiency, params=params
                )
            )
        )
        print()


if __name__ == "__main__":
    main()
