"""How deep the solar dive has to be: pricing a shallower perihelion.

ADR 0019 flies the Jovian solar-dive cycle to the paper's own 4 solar radii and
scores it at a *stated* 60 percent survival across the periapsis collision.  Two
things about that node are uncomfortable and neither is a trajectory question:
it sits in 3.9 MW/m^2 of sunlight, and the opposing streams meet at 617 km/s.
Backing the perihelion out trades growth for a node a spacecraft can plausibly
be built for.  This module prices that trade.

Five things it does that :mod:`src.jovian_solar_dive_cycle` does not:

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
* **The departure is charged from the pad.**  ``cycle_growth_ledger`` starts the
  burn at Earth escape; the **launch ledger** in the same module assumes a
  4.09 km/s ballistic lob, about 3.6 km/s at the burn point, and nothing charged
  the gap.  Charging it splits the burn in two, because the overtaking half and
  the canted half want opposite geometries (:func:`split_push_ledger`).
* **The opposing stream is placed, not assumed.**  Backing the dive out makes
  the *prograde* placement cheaper and the *retrograde* placement dearer, which
  is the one direction nothing else in the trade runs
  (:func:`opposing_stream_placement`).

The headline, against ADR 0019's reference cycle:

* **Three synodic periods still closes at 32 solar radii**, prograde and
  unpowered, with +54.48 degrees of Jovian bend margin against +58.80.
* **Growth falls about 6.6x** once the departure is charged from the pad rather
  than from a free parking orbit, 23.0 to 3.49 kg returned per impactor kilogram
  -- but **doubling time only 2.51x**, 0.724 yr against 1.816, and millionfold
  14.4 yr against 36.2.  A 64x cooler node costs a factor of 2.5 in the clock,
  not a factor of seven (:func:`split_push_ledger`).
* **The cap on the slug ratio is the expansion floor, and at the shallow node
  the launch ledger is close behind.**  Neither is the plume ignition window,
  which is the only one the repository had: a search given only that constraint
  picks ``k`` = 23.75, where the jet would have to carry 0.60 of a merge energy
  that can spare 0.369.
* **Charged from the pad, 4 solar radii pays for its launch and 32 does not.**
  The **launch ledger** was run before the departure was, so it charged 1/4 of
  liftoff to reach the intercept point and then let the payload appear at Earth
  escape for free.  Charging the chain between costs about a third of the pad
  return at both depths: the deep cycle clears the committed 1/15 floor at
  1.043x and the shallow one fails at 0.618x -- at *every* slug ratio, and under
  the speed-rescaled floor too (:func:`split_push_launch_ledger`).
* **The frontier is bisected to its constraint, not stepped past it.**  Every
  optimum on the pad frontier hugs the climb-out at which the **overtaking
  leg**'s plume stops conducting, and that threshold moves with the **conduction
  reserve** while a grid does not (:func:`conduction_threshold_excess`,
  :func:`pad_frontier_optimum`).  Scored that way the shallow node reaches
  1.009x at the hard end of the bracket rather than 0.979x -- it *pays*, and
  what rules it out is that it doubles in 2.07 yr against the Jupiter-only
  chain's 1.74.  The depth crossing moves with it, from 21.09 solar radii on a
  dial that conducts nowhere to **22.93** on one that does
  (:func:`admissible_pad_floor_depth`).

ADR ``0020-the-nozzle-cap-is-the-expansion-not-the-ignition``,
ADR ``0021-the-shallow-dive-does-not-pay-for-its-launch`` and
ADR ``0022-the-conduction-bracket-was-read-off-a-grid``.
Run with ``make dive-depth``; ``--pad-frontier`` adds the scan that rules the
shallow node out rather than merely scoring it badly.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from boinor.bodies import Earth
from scipy.optimize import brentq, minimize_scalar
from tabulate import tabulate

from src.astro_constants import SOLAR_DIVE_PERIAPSIS_SOLAR_RADII
from src.circular_resonance_impulse import impulse_per_impactor_kg
from src.conic_kernel import (
    half_turn_angle,
    hyperbolic_eccentricity,
    speed_with_escape_energy,
)
from src.heliocentric_reintercept import single_impulse_resonant_dive
from src.jovian_solar_dive_cycle import (
    _BURN_RADIUS,
    _MU_EARTH,
    DEFAULT_JET_ENERGY_EFFICIENCY,
    DEFAULT_PERIAPSIS_SURVIVAL,
    DEFAULT_RETURN_EXCESS,
    DEFAULT_SLUG_RATIO,
    CycleGrowthLedger,
    LaunchLedgerVerdict,
    SynodicCycleClosure,
    _dive_periapsis_radius,
    _vehicle_frame_impact,
    cycle_growth_ledger,
    departure_nozzle_ledger,
    dive_placement_excess_floor,
    launch_ledger_verdict,
    return_value_ratio,
    solve_synodic_closure,
)
from src.plume_thermal import (
    NOZZLE_FLOOR_TEMPERATURE,
    plume_ignition_energy,
    slug_ratio_window,
    specific_thermal_energy,
)
from src.retrograde_return_legs import _FlybyParams, _powered_flyby_params
from src.two_leg_nozzle_sweep import PAYLOAD_FRACTION_AT_INTERCEPT, RETURN_FLOOR

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


def conduction_reserve(
    temperature: u.Quantity = NOZZLE_FLOOR_TEMPERATURE,
    reserve_thermal: bool = True,
) -> float:
    """Merge energy the jet may not spend, if the plume is to still conduct (MJ/kg).

    **A bracket, not a number.**  Two readings are defensible and they differ by
    the translational term:

    * ``reserve_thermal=True`` (the conservative end, and the default) reserves
      the whole **ignition bill**, so the plume reaches nozzle exit still at
      ``temperature``.  Sufficient, certainly, but not necessary.
    * ``reserve_thermal=False`` reserves only the frozen chemistry -- the *hard*
      end, energy that cannot come back under any assumption.  It is defensible
      because ADR 0016's own finding is that recombination **freezes** past the
      nozzle lip.  If dissociation does not recombine on the expansion clock,
      neither does the seed's ionisation: the electrons survive the cooling,
      Spitzer conductivity falls only as ``T**1.5``, and the field's grip
      degrades smoothly rather than cliff-edging at the ignition temperature.

    ``temperature`` is a design choice rather than a constant of nature -- a
    potassium-seeded plume is generally taken to stay workable near 6,000 K,
    which drops the bill from 84.41 to 65.85 MJ/kg.

    What closes the bracket is the magnetic Reynolds number at nozzle exit,
    ``Rm = mu0 * sigma * v * L >~ 1``, which nobody in this repository has
    computed.  It needs the exit density, velocity, scale length and the frozen
    ionisation fraction; ``puffsat_impact_simulation`` has the machinery, having
    produced the ``Da = 1`` crossing for ADR 0016.

    Args:
        temperature: Plume temperature the nozzle requires.
        reserve_thermal: Whether to reserve the translational term as well as
            the frozen chemistry.

    Returns:
        The reserved specific energy (MJ/kg).
    """
    if reserve_thermal:
        return float(plume_ignition_energy(temperature).to_value(u.MJ / u.kg))
    return frozen_ignition_energy()


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


def maximum_jet_efficiency(
    closing_speed: float,
    slug_ratio: float,
    reserve: Optional[float] = None,
) -> float:
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
        reserve: Merge energy the jet may not spend (MJ/kg); defaults to the
            conservative end of :func:`conduction_reserve`.  Pass
            ``conduction_reserve(reserve_thermal=False)`` for the hard end.

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
    held = conduction_reserve() if reserve is None else reserve
    ratio = slug_ratio / (1.0 + slug_ratio) * (1.0 - held / thermalised)
    return float(min(1.0, max(0.0, ratio)))


def expansion_limited_slug_ratio_window(
    closing_speed: float,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    margin: float = 1.0,
    reserve: Optional[float] = None,
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
        reserve: Merge energy the jet may not spend (MJ/kg); defaults to the
            conservative end of :func:`conduction_reserve`.

    Returns:
        ``(k_min, k_max)``, or None when no slug ratio clears the floor at this
        closing speed -- a real outcome, not an error.

    Raises:
        ValueError: If the efficiency is not in (0, 1).
    """
    if not 0.0 < jet_energy_efficiency < 1.0:
        raise ValueError("jet_energy_efficiency must lie in (0, 1)")
    peak = 0.5 * (closing_speed * 1000.0) ** 2
    held = conduction_reserve() if reserve is None else reserve
    bill = margin * held * 1.0e6
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
    parser.add_argument(
        "--pad-frontier",
        action="store_true",
        help=(
            "Also scan the climb-out excess at each depth for a slug ratio that "
            "both pays for its launch and leaves the plume conducting on both "
            "legs of the split push. Off by default because it re-solves the "
            "closure at every point on PAD_FRONTIER_EXCESS_GRID."
        ),
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
        print(
            f"Departure slug ratio on the {row.node.dive_solar_radii:g} R_sun cycle"
            " -- FREE PARKING ORBIT, superseded by the table below it"
        )
        print(
            slug_ratio_table_text(
                slug_ratio_table(
                    row, jet_energy_efficiency=args.jet_efficiency, params=params
                )
            )
        )
        print()

    print("=" * 78)
    print("THE LAUNCH LEDGER RE-SCORED, DEPARTURE CHARGED FROM THE PAD")
    print("=" * 78)
    print()
    for row, survival in (
        (deep, PROPOSED_NODE_SURVIVAL[SOLAR_DIVE_PERIAPSIS_SOLAR_RADII]),
        (
            shallow,
            PROPOSED_NODE_SURVIVAL.get(args.dive_solar_radii, shallow.node.survival),
        ),
    ):
        ledger = split_push_from_state(
            row.label,
            row.closure.departure_excess,
            row.closure.departure_aim_deg,
            row.closure.total_tof_years,
            row.closure.return_excess,
            row.closure.push_axis_deg,
            survival,
            row.growth.nozzle.slug_ratio,
            jet_energy_efficiency=args.jet_efficiency,
            params=params,
        )
        if ledger is None:
            print(f"  {row.label}: no split push flies this cycle")
            continue
        print(
            describe_split_push(
                ledger,
                split_push_launch_ledger(ledger, params),
                split_push_pad_ceilings(
                    row.closure,
                    survival,
                    jet_energy_efficiency=args.jet_efficiency,
                    expansion_margin=args.expansion_margin,
                    label=row.label,
                    params=params,
                ),
            )
        )
        print()
        print(
            f"Departure slug ratio on the {row.node.dive_solar_radii:g} R_sun cycle,"
            " charged from the pad"
        )
        print(
            split_push_slug_ratio_text(
                split_push_slug_ratio_table(
                    row.closure,
                    survival,
                    jet_energy_efficiency=args.jet_efficiency,
                    params=params,
                )
            )
        )
        print()

    paper = paper_resonant_dive_split_push(
        deep.closure.return_excess,
        deep.closure.push_axis_deg,
        slug_ratio=DEFAULT_SLUG_RATIO,
        jet_energy_efficiency=args.jet_efficiency,
        params=params,
    )
    if paper is not None:
        print(describe_split_push(paper, split_push_launch_ledger(paper, params)))
        print()

    print(
        f"Dive depth charged from the pad, {args.return_excess:g} km/s climb-out,"
        f" k = {args.slug_ratio:g}, node survival derived at every depth"
    )
    rows = split_push_depth_table(
        return_excess=args.return_excess,
        slug_ratio=args.slug_ratio,
        jet_energy_efficiency=args.jet_efficiency,
        params=params,
    )
    print(split_push_depth_text(rows))
    crossing = pad_floor_depth(
        return_excess=args.return_excess,
        slug_ratio=args.slug_ratio,
        jet_energy_efficiency=args.jet_efficiency,
        params=params,
    )
    if crossing is not None:
        print()
        print(
            f"The committed 1/15 floor is lost at {crossing:.2f} R_sun on this"
            " dial: a shallower dive than that does not return the fifteenth of"
            " liftoff it committed to.  SUPERSEDED as a recommendation (ADR"
            f" 0022): a {args.return_excess:g} km/s climb-out does not conduct on"
            " the overtaking leg at any depth, so this crossing sits on a dial"
            " nothing may fly.  --pad-frontier reports the admissible one."
        )
    print()

    if args.pad_frontier:
        for depth in (SOLAR_DIVE_PERIAPSIS_SOLAR_RADII, args.dive_solar_radii):
            print(
                f"Can {depth:g} R_sun be flown at all? Best pad return at each"
                f" climb-out, expansion margin {args.expansion_margin:g}x on both legs"
            )
            print(
                pad_frontier_text(
                    pad_return_frontier(
                        depth,
                        jet_energy_efficiency=args.jet_efficiency,
                        expansion_margin=args.expansion_margin,
                        params=params,
                    )
                )
            )
            print()
            print(
                f"Does that verdict survive the conduction reserve bracket at"
                f" {depth:g} R_sun? The reserve does not touch the pad return at"
                " any k; it moves the lowest climb-out the overtaking plume"
                " survives, and the pad return is won at low climb-outs."
            )
            print(
                conduction_bracket_text(
                    conduction_bracket_frontier(
                        depth,
                        jet_energy_efficiency=args.jet_efficiency,
                        params=params,
                    )
                )
            )
            print()
        admissible = admissible_pad_floor_depth(
            jet_energy_efficiency=args.jet_efficiency,
            expansion_margin=args.expansion_margin,
            params=params,
        )
        if admissible is not None:
            print(
                f"Flying each depth at its own best conducting climb-out, the"
                f" committed 1/15 floor is lost at {admissible:.2f} R_sun."
                " That is the dial a mission designer has, and the crossing on"
                " it is the shallowest dive worth proposing."
            )
            print()


# --------------------------------------------------------------------------
# Two pushes, not one: where the payload actually starts from
# --------------------------------------------------------------------------

# Speed a payload carries at the 200 km burn point after the launch ledger's own
# ballistic lob.  LAUNCH_PROPELLANT_FRACTION = 2/3 at 380 s is exactly a 4.09
# km/s lob, and a 4.09 km/s lob is nowhere near orbital: flown vertically it
# arrives at 200 km with about 3.6 km/s.  Approximate -- a real lofted trajectory
# trades some of that for downrange velocity -- but it is the right *scale*, and
# the point is that it is 3.6 and not the 11.0 km/s the growth ledger assumed.
LOB_DELTA_V = 4.09
_LOB_GRAVITY = 9.5e-3  # km/s^2, mean over the first 200 km

# Parking-orbit period between the two pushes (days).  5 days captures 95
# percent of the split's benefit while advancing Earth only 4.93 degrees during
# the coast, so the second wave's trajectory differs from the first's by very
# little.  20 days is worth 5 percent more and moves Earth 19.7 degrees.
DEFAULT_PARKING_PERIOD_DAYS = 5.0

# Node survival proposed for each reference depth, rounded from the impulse-law
# derivation (0.5977 at 4 solar radii, 0.3537 at 32) to the nearest defensible
# fraction and deliberately on the conservative side of both.
PROPOSED_NODE_SURVIVAL = {4.0: 0.5, 32.0: 1.0 / 3.0}

# Parking-orbit periods swept by split_push_period_sweep() (days).  The floor is
# below where the apoapsis re-aim destroys the split's advantage; the ceiling is
# past where the growth has saturated.
PARKING_PERIOD_GRID: Tuple[float, ...] = (
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    40.0,
)
# Steps used to integrate each leg of the split.  Both the closing speed and the
# impact angle move as the vehicle accelerates, and the legs are long (7-13 km/s
# against the 4-6 km/s the single-leg ledger integrates), so a midpoint
# evaluation is not good enough.
_LEG_INTEGRATION_STEPS = 3000


def lob_arrival_speed(
    lob_delta_v: float = LOB_DELTA_V, altitude: float = 200.0
) -> float:
    """Speed at the burn point after the launch ledger's own ballistic lob (km/s).

    Args:
        lob_delta_v: Ground-rocket delta-v (km/s).
        altitude: Burn altitude above the surface (km).

    Returns:
        Arrival speed (km/s); zero when the lob does not reach the altitude.
    """
    residual = lob_delta_v * lob_delta_v - 2.0 * _LOB_GRAVITY * altitude
    return float(np.sqrt(residual)) if residual > 0.0 else 0.0


def _burn_leg(
    start_speed: float,
    end_speed: float,
    stream_axis_offset_deg: float,
    stream_speed: float,
    slug_ratio: float,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
) -> float:
    """Mass fraction surviving one slug-nozzle burn between two periapsis speeds.

    Args:
        start_speed: Vehicle speed when the burn lights (km/s).
        end_speed: Speed when it finishes (km/s).
        stream_axis_offset_deg: Angle from the vehicle's thrust axis to the
            direction the stream travels.  Zero is a pure overtaking push.
        stream_speed: Impactor speed at the burn radius (km/s).
        slug_ratio: Kilograms of slug per kilogram of arriving impactor.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].

    Returns:
        The delivered mass fraction, or 0.0 if the impulse ever goes non-positive.
    """
    speeds = np.linspace(start_speed, end_speed, _LEG_INTEGRATION_STEPS + 1)
    log_fraction = 0.0
    for lower, upper in zip(speeds[:-1], speeds[1:]):
        angle, closing = _vehicle_frame_impact(
            stream_axis_offset_deg,
            0.0,
            stream_speed,
            0.5 * (float(lower) + float(upper)),
        )
        beta = impulse_per_impactor_kg(angle * u.rad, slug_ratio, jet_energy_efficiency)
        if beta <= 0.0:
            return 0.0
        log_fraction -= (float(upper) - float(lower)) / (beta * closing / slug_ratio)
    return float(np.exp(log_fraction))


def _leg_coldest_closing(
    stream_axis_offset_deg: float,
    stream_speed: float,
    start_speed: float,
    end_speed: float,
) -> float:
    """Coldest closing speed reached anywhere on one burn leg (km/s).

    The closing speed is monotone in the vehicle's speed at fixed geometry --
    ``|v_stream - v_vehicle|`` resolved along and across the thrust axis -- so
    the extremes are the endpoints and no scan is needed.  Which endpoint is
    colder flips at 90 degrees: below it the vehicle chases the stream and the
    closing speed *falls*, above it the vehicle runs into the stream and it
    rises.

    Args:
        stream_axis_offset_deg: Angle from the thrust axis to the stream.
        stream_speed: Impactor speed at the burn radius (km/s).
        start_speed: Vehicle speed when the burn lights (km/s).
        end_speed: Speed when it finishes (km/s).

    Returns:
        The coldest closing speed on the leg (km/s).
    """
    return min(
        _vehicle_frame_impact(stream_axis_offset_deg, 0.0, stream_speed, speed)[1]
        for speed in (start_speed, end_speed)
    )


def parking_orbit_periapsis_speed(period_days: float, altitude: float = 200.0) -> float:
    """Periapsis speed of a bound ellipse of this period (km/s).

    Dominated by ``2*mu/r_p``, which is why the split survives a *short* parking
    orbit far better than intuition suggests: a 6-hour ellipse still reaches
    9.87 km/s, 90 percent of the 11.01 km/s escape speed.

    Args:
        period_days: Orbital period (days).
        altitude: Periapsis altitude above the surface (km).

    Returns:
        Periapsis speed (km/s).
    """
    r_p = float(Earth.R.to_value(u.km)) + altitude
    seconds = period_days * 86400.0
    semi_major = float(np.cbrt(_MU_EARTH * seconds * seconds / (4.0 * np.pi**2)))
    return float(np.sqrt(_MU_EARTH * (2.0 / r_p - 1.0 / semi_major)))


def apoapsis_reaim_cost(
    period_days: float, cant_deg: float, altitude: float = 200.0
) -> float:
    """Delta-v to rotate the parking orbit by ``cant_deg`` at apoapsis (km/s).

    The **split push**'s enabling trick and its binding cost at the same time.
    Rotating a velocity vector by ``theta`` costs ``2 v sin(theta/2)``, and at
    apoapsis ``v`` is small -- 0.117 km/s on a 20-day ellipse, so a 125 degree
    turn costs 0.207 km/s.  But ``v_apo`` grows fast as the period shortens:
    2.41 km/s on a 6-hour ellipse, where the same turn costs 4.27 km/s and
    destroys the split it was meant to enable.

    Args:
        period_days: Parking-orbit period (days).
        cant_deg: Angle the periapsis velocity must be rotated through (deg).
        altitude: Periapsis altitude above the surface (km).

    Returns:
        The re-aim delta-v (km/s).
    """
    r_p = float(Earth.R.to_value(u.km)) + altitude
    seconds = period_days * 86400.0
    semi_major = float(np.cbrt(_MU_EARTH * seconds * seconds / (4.0 * np.pi**2)))
    v_p = parking_orbit_periapsis_speed(period_days, altitude)
    r_a = 2.0 * semi_major - r_p
    v_apo = r_p * v_p / r_a
    return 2.0 * v_apo * float(np.sin(0.5 * np.radians(cant_deg)))


@dataclass(frozen=True)
class SplitPushLedger:
    """Growth when the payload is pushed from the pad, not from a free parking orbit.

    The correction ADR 0020's first two versions needed.  ``cycle_growth_ledger``
    starts the departure burn at ``v_depart_from`` = 11.0086 km/s -- already at
    Earth escape -- while the **launch ledger** in the same module assumes the
    payload arrives on a 4.09 km/s ballistic lob, about 3.6 km/s at the burn
    point.  Nothing charged the ~7.4 km/s between them.

    Charging it changes the *architecture*, not just the number, because the two
    halves want opposite geometries:

    * The **overtaking push** from the lob to a parking orbit runs at
      ``theta`` = 0, where the impactor's own momentum *adds*:
      ``beta = 1 + sqrt(eta(1+k))`` = 3.387 at ``k`` = 8.5.
    * The **departure push** must leave along the aim the **synodic closure**
      demands, which the stream does not point at, so it runs canted --
      ``beta`` = 1.67 at this cycle's 124.8 degrees.

    A single push must fly the whole thing in the canted geometry.  Splitting it
    buys the cheap geometry for the larger half, and pays for the re-aim at
    apoapsis where velocity is small.

    Attributes:
        label: Which cycle this is.
        cant_deg: Angle from the thrust axis to the stream at the departure burn.
        stream_speed: Earth-relative collision speed of the returning stream (km/s).
        lob_speed: Speed the payload carries at the burn point off the lob (km/s).
        departure_speed: Speed the departure needs at 200 km (km/s).
        parking_period_days: Period of the ellipse between the two pushes.
        parking_periapsis_speed: Its periapsis speed (km/s).
        reaim_delta_v: Apoapsis rotation the split costs (km/s).
        reaim_fraction: Mass surviving that rotation, flown on methalox because
            there is no stream at apoapsis.
        overtaking_fraction: Mass fraction surviving the first push.
        departure_fraction: Mass fraction surviving the second.
        overtaking_coldest_closing: Coldest closing speed on the overtaking leg
            (km/s).  It falls through that burn -- at ``theta`` = 0 the vehicle's
            own speed subtracts directly -- so the coldest instant is the leg's
            *end*, at the parking orbit's periapsis.  **This is the coldest
            instant of the whole departure**, and it is a leg the single-push
            ledger never had, so no **expansion floor** was ever asked of it.
        departure_coldest_closing: Coldest closing speed on the canted leg
            (km/s).  Past 90 degrees the vehicle's acceleration *raises* the
            closing speed, so this is the leg's start -- the same parking-orbit
            periapsis, seen at the worse angle.
        single_push_fraction: Mass fraction if one canted push does it all.
        node_survival: Mass fraction surviving the solar-periapsis collision.
        split_growth: Kilograms returned per impactor kilogram, split.
        single_growth: The same for one canted push.
        free_parking_growth: The same when the parking orbit is not charged --
            what ADR 0019 and ADR 0020's first version reported.
        cycle_years: Departure-to-return time (yr).
    """

    label: str
    cant_deg: float
    stream_speed: float
    lob_speed: float
    departure_speed: float
    parking_period_days: float
    parking_periapsis_speed: float
    reaim_delta_v: float
    reaim_fraction: float
    overtaking_fraction: float
    departure_fraction: float
    overtaking_coldest_closing: float
    departure_coldest_closing: float
    single_push_fraction: float
    node_survival: float
    split_growth: float
    single_growth: float
    free_parking_growth: float
    cycle_years: float

    def _doubling(self, growth: float) -> float:
        """Doubling time for a per-cycle multiplier (yr); inf when it does not grow."""
        return (
            float(self.cycle_years * np.log(2.0) / np.log(growth))
            if growth > 1.0
            else float("inf")
        )

    @property
    def split_doubling_years(self) -> float:
        """Doubling time on the split push (yr)."""
        return self._doubling(self.split_growth)

    @property
    def single_doubling_years(self) -> float:
        """Doubling time on one canted push (yr)."""
        return self._doubling(self.single_growth)

    @property
    def free_parking_doubling_years(self) -> float:
        """Doubling time with the parking orbit uncharged (yr)."""
        return self._doubling(self.free_parking_growth)

    @property
    def split_advantage(self) -> float:
        """How much the split beats a single canted push."""
        return self.split_growth / self.single_growth

    @property
    def chain_to_departure(self) -> float:
        """Fraction of the lob's payload that reaches the Jupiter transfer.

        The three charges between the top of the ballistic lob and the departure
        hyperbola, multiplied: the overtaking push, the methalox apoapsis
        re-aim, and the canted departure leg.  This is the quantity the
        **launch ledger** needs and the one ``cycle_growth_ledger`` set to 1.
        """
        return self.overtaking_fraction * self.reaim_fraction * self.departure_fraction

    @property
    def coldest_closing_speed(self) -> float:
        """Coldest closing speed anywhere in the departure (km/s)."""
        return min(self.overtaking_coldest_closing, self.departure_coldest_closing)


def split_push_from_state(
    label: str,
    departure_excess: float,
    departure_aim_deg: float,
    cycle_years: float,
    stream_excess: float,
    stream_axis_deg: float,
    node_survival: float,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> Optional[SplitPushLedger]:
    """Fly a **split push** from any departure state, not just a solved closure.

    The same arguments :func:`jovian_solar_dive_cycle.cycle_growth_ledger` takes,
    so the paper's own single-impulse resonant dive can be scored on this device
    rather than only the Jovian closures -- which is what
    :func:`paper_resonant_dive_split_push` does.

    Args:
        label: Name for the row.
        departure_excess: Earth-relative departure excess (km/s).
        departure_aim_deg: Direction it must point, from Earth's prograde.
        cycle_years: Departure-to-return time (yr).
        stream_excess: Earth-relative excess of the arriving stream (km/s).
        stream_axis_deg: Direction the stream travels, from Earth's prograde.
        node_survival: Mass fraction surviving the solar-periapsis collision.
        slug_ratio: Departure slug ratio, used on both pushes.
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitPushLedger`, or None when a leg delivers nothing.
    """
    p = params if params is not None else _powered_flyby_params()
    stream = speed_with_escape_energy(stream_excess, p.v_esc_surface)
    lob = lob_arrival_speed()
    v_end = speed_with_escape_energy(departure_excess, p.v_esc_leo)
    v_park = parking_orbit_periapsis_speed(parking_period_days)
    if not lob < v_park < v_end:
        return None
    # The departure-hyperbola mirror puts the thrust axis off the outgoing aim;
    # the cant is measured from that axis to the stream (CONTEXT.md, and the
    # correction in ADR 0019's addendum -- the SOI aim separation is the wrong
    # angle to feed the impulse law).  The sign is free, exactly as in
    # departure_nozzle_ledger, so both are flown and the better one kept.
    mirror = float(
        np.degrees(
            half_turn_angle(
                hyperbolic_eccentricity(_MU_EARTH, _BURN_RADIUS, departure_excess)
            )
        )
    )
    best: Optional[SplitPushLedger] = None
    for sign in (1.0, -1.0):
        cant = abs(stream_axis_deg - (departure_aim_deg + sign * mirror))
        reaim = apoapsis_reaim_cost(parking_period_days, cant)
        reaim_fraction = float(np.exp(-reaim / p.exhaust_speed))
        f1 = _burn_leg(lob, v_park, 0.0, stream, slug_ratio, jet_energy_efficiency)
        f2 = _burn_leg(v_park, v_end, cant, stream, slug_ratio, jet_energy_efficiency)
        f_one = _burn_leg(lob, v_end, cant, stream, slug_ratio, jet_energy_efficiency)
        if min(f1, f2, f_one) <= 0.0 or max(f1, f2, f_one) >= 1.0:
            continue
        # Impactors consumed per kilogram placed on the Jupiter trajectory: the
        # departure leg's own, plus the overtaking leg's, scaled up because that
        # mass still has to survive the re-aim and the departure.
        impactors = (1.0 / f2 - 1.0) / slug_ratio + (1.0 / (f2 * reaim_fraction)) * (
            1.0 / f1 - 1.0
        ) / slug_ratio
        candidate = SplitPushLedger(
            label=label,
            cant_deg=cant,
            stream_speed=stream,
            lob_speed=lob,
            departure_speed=v_end,
            parking_period_days=parking_period_days,
            parking_periapsis_speed=v_park,
            reaim_delta_v=reaim,
            reaim_fraction=reaim_fraction,
            overtaking_fraction=f1,
            departure_fraction=f2,
            overtaking_coldest_closing=_leg_coldest_closing(0.0, stream, lob, v_park),
            departure_coldest_closing=_leg_coldest_closing(cant, stream, v_park, v_end),
            single_push_fraction=f_one,
            node_survival=node_survival,
            split_growth=node_survival / impactors,
            single_growth=node_survival * f_one * slug_ratio / (1.0 - f_one),
            free_parking_growth=node_survival * f2 * slug_ratio / (1.0 - f2),
            cycle_years=cycle_years,
        )
        if best is None or candidate.split_growth > best.split_growth:
            best = candidate
    return best


def split_push_ledger(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    return_excess: float = CONSERVATIVE_RETURN_EXCESS,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    node_survival: Optional[float] = None,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    label: Optional[str] = None,
    params: Optional[_FlybyParams] = None,
) -> Optional[SplitPushLedger]:
    """Score a cycle with the departure charged from the pad, split into two pushes.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        slug_ratio: Departure slug ratio, used on both pushes.
        node_survival: Mass fraction surviving the solar-periapsis collision;
            defaults to ``PROPOSED_NODE_SURVIVAL`` at a reference depth, or the
            impulse-law derivation otherwise.
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        label: Name for the row; derived from the inputs when omitted.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitPushLedger`, or None when no closure exists or a leg
        delivers nothing.
    """
    p = params if params is not None else _powered_flyby_params()
    closure = solve_synodic_closure(3, p, dive_solar_radii, return_excess, 1.0)
    if closure is None:
        return None
    node = dive_node(
        dive_solar_radii,
        closure.periapsis_boost,
        DEFAULT_PERIAPSIS_SLUG_RATIO,
        jet_energy_efficiency,
        p,
    )
    survival = (
        node_survival
        if node_survival is not None
        else PROPOSED_NODE_SURVIVAL.get(dive_solar_radii, node.survival)
    )
    return split_push_from_state(
        label
        or f"{dive_solar_radii:g} R_sun / k {slug_ratio:g} / {parking_period_days:g} d",
        closure.departure_excess,
        closure.departure_aim_deg,
        closure.total_tof_years,
        closure.return_excess,
        closure.push_axis_deg,
        survival,
        slug_ratio,
        parking_period_days,
        jet_energy_efficiency,
        p,
    )


def paper_resonant_dive_split_push(
    stream_excess: float,
    stream_axis_deg: float,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    periapsis_survival: float = DEFAULT_PERIAPSIS_SURVIVAL,
    params: Optional[_FlybyParams] = None,
) -> Optional[SplitPushLedger]:
    """Charge the paper's own single-impulse resonant dive from the pad too.

    The comparison row, and it carried the identical defect: ADR 0019 and ADR
    0020 both scored it with ``cycle_growth_ledger``, which starts its departure
    at Earth escape.  The paper's dive asks for a **37.53 km/s** Earth-relative
    excess -- it has to cancel most of Earth's orbital motion in one impulse --
    so it has far more of that burn to pay for than either Jovian cycle, and the
    correction is correspondingly larger.  Scoring it here rather than quoting
    it is what makes the ranking safe to state.

    Args:
        stream_excess: Earth-relative excess of the arriving stream (km/s).
        stream_axis_deg: Direction the stream travels, from Earth's prograde.
        slug_ratio: Departure slug ratio, used on both pushes.
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        periapsis_survival: Mass fraction surviving its solar-periapsis
            collision.  Stated rather than derived, matching
            :func:`jovian_solar_dive_cycle.paper_resonant_dive_ledger`.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitPushLedger`, or None when a leg delivers nothing.
    """
    dive = single_impulse_resonant_dive()
    retrograde = float(dive.retrograde_component.to_value(u.km / u.s))
    radial = float(dive.radial_component.to_value(u.km / u.s))
    return split_push_from_state(
        "paper single-impulse resonant dive",
        float(dive.earth_boost.to_value(u.km / u.s)),
        float(np.degrees(np.arctan2(radial, -retrograde))),
        float(dive.reintercept_time.to_value(u.year)),
        stream_excess,
        stream_axis_deg,
        periapsis_survival,
        slug_ratio,
        parking_period_days,
        jet_energy_efficiency,
        params,
    )


def split_push_period_sweep(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    return_excess: float = CONSERVATIVE_RETURN_EXCESS,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    periods: Sequence[float] = PARKING_PERIOD_GRID,
    params: Optional[_FlybyParams] = None,
) -> List[SplitPushLedger]:
    """Sweep the parking-orbit period, which is what the split trades against.

    Short periods keep the two waves close together -- Earth advances only 0.25
    degrees over a 6-hour coast -- but make the apoapsis re-aim ruinous.  Long
    periods make the re-aim nearly free and the phasing demanding.  The growth
    saturates well before the phasing becomes awkward.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        slug_ratio: Departure slug ratio.
        periods: Parking-orbit periods to sweep (days).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        One :class:`SplitPushLedger` per period that closes.
    """
    p = params if params is not None else _powered_flyby_params()
    out = [
        split_push_ledger(
            dive_solar_radii,
            return_excess,
            slug_ratio,
            parking_period_days=float(period),
            params=p,
        )
        for period in periods
    ]
    return [row for row in out if row is not None]


# --------------------------------------------------------------------------
# The launch ledger, re-scored from the pad
# --------------------------------------------------------------------------
#
# The paper's second open problem on this cycle.  Every cycle in the paper is
# charged for its own ground launch and must return a fifteenth of the mass
# lifted off the pad (`sec:two_leg_nozzle`), and that check was run on this
# cycle *before* the departure was charged from the lob.  The two ledgers now
# agree on where the payload starts -- PAYLOAD_FRACTION_AT_INTERCEPT is what a
# 4.09 km/s lob delivers, and `lob_arrival_speed()` is the speed it delivers it
# at -- so the correction is a substitution rather than a re-derivation: the
# free parking orbit's implicit factor of 1 becomes `chain_to_departure`.

#: Slug ratios the pad-return peak is searched over.  Deliberately wider than
#: the **expansion floor**'s window at every closing speed either reference
#: cycle reaches, so the peak is interior and not a bracket artefact.
_PAD_PEAK_BRACKET = (0.25, 20.0)
#: Bracket the committed floor's crossings are bisected in.  The upper end is
#: past the **plume ignition window**'s own ceiling at both cycles' closing
#: speeds, so a "no ceiling" verdict means cleared throughout, not truncated.
_PAD_CEILING_BRACKET = (0.25, 120.0)
#: Climb-out excesses scanned by :func:`pad_return_frontier` (km/s).  The floor
#: is below where the **expansion floor** vanishes on the overtaking leg at
#: either depth; the ceiling is past where the pad return has clearly collapsed.
PAD_FRONTIER_EXCESS_GRID: Tuple[float, ...] = (
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    100.0,
    120.0,
    150.0,
    187.5,
    200.0,
)
#: The **conduction reserve** bracket, as (label, MJ/kg) pairs, swept by
#: :func:`conduction_bracket_frontier`.  The conservative end reserves the whole
#: **ignition bill**; the hard end reserves only the frozen chemistry; 6,000 K is
#: the temperature a potassium-seeded plume is usually taken to stay workable at.
#: The three are ``conduction_reserve()`` at its two flags and at 6,000 K, stated
#: as literals so the sweep's axis is readable in one place.
CONDUCTION_RESERVE_BRACKET: Tuple[Tuple[str, float], ...] = (
    ("15000 K, all reserved", 84.41),
    ("6000 K", 65.85),
    ("frozen chemistry only", 53.47),
)
#: Expansion margins swept alongside it.  1.5 is ADR 0020's operating margin;
#: 1.0 puts the winner exactly on the floor, which is a boundary rather than a
#: design point, and is swept to show what the margin is carrying.
CONDUCTION_MARGIN_GRID: Tuple[float, ...] = (1.5, 1.0)
#: Depths :func:`pad_floor_depth` brackets the committed floor's crossing in.
PAD_FLOOR_DEPTH_BRACKET = (4.0, 48.0)
#: Closing speeds :func:`conduction_threshold_closing_speed` bisects between.
#: Wide enough to contain the threshold at every reading of the **conduction
#: reserve** bracket, which spans 51.7 to 79.6 km/s.
_CONDUCTION_THRESHOLD_BRACKET = (5.0, 200.0)
#: Climb-out excesses :func:`conduction_threshold_excess` bisects between (km/s).
#: Both ends sit inside where the three-synodic closure exists at every depth on
#: ``PAD_FLOOR_DEPTH_BRACKET`` -- at 32 solar radii it still closes at 170 km/s
#: and no longer does at 180 -- and they straddle the threshold at every reading
#: of the **conduction reserve** bracket, which lands between 38 and 82 km/s.
_CONDUCTION_EXCESS_BRACKET = (20.0, 150.0)
#: How far above the conduction threshold :func:`pad_frontier_optimum` searches
#: for the pad return's peak (km/s).  The peak sits within 1 km/s of the
#: threshold at 32 solar radii and up to 26 at 4, where the node is cheap enough
#: that the pinched window costs more than the boost does; 40 keeps it interior
#: at both, which the tests assert rather than assume.
PAD_OPTIMUM_EXCESS_SPAN = 40.0
#: Climb-out tolerance of that search (km/s).  The pad return is flat to about
#: one part in a million across 0.02 km/s at the peak, so this is far inside the
#: precision any figure is quoted to.
_PAD_OPTIMUM_EXCESS_XATOL = 0.02
#: Depths :func:`admissible_pad_floor_depth` brackets its crossing in.  Narrower
#: than ``PAD_FLOOR_DEPTH_BRACKET`` because each end costs a full climb-out
#: optimisation: 16 solar radii clears the committed floor with room and 32
#: fails at every reading of the **conduction reserve**, so the crossing is
#: interior.
ADMISSIBLE_PAD_FLOOR_DEPTH_BRACKET = (16.0, 32.0)
#: Depth tolerance of that bisection (solar radii).  The dial moves the pad
#: margin by about 0.05 per solar radius here, so this pins the crossing to
#: about a part in four hundred of the floor -- past what the floor's own
#: calibration supports.
_ADMISSIBLE_PAD_FLOOR_XTOL = 0.05


@dataclass(frozen=True)
class SplitPushLaunchLedger:
    """What a **split push** returns per kilogram off the pad.

    The same three readings as
    :class:`jovian_solar_dive_cycle.LaunchLedgerVerdict`, with the one
    substitution that ADR 0020's addendum forced.  There, ``returned_per_pad_kg``
    was ``PAYLOAD_FRACTION_AT_INTERCEPT * delivered * survival``: the payload
    appeared at Earth escape for free and only the departure burn was charged
    against it.  Here the lob's payload has to fly the overtaking push, the
    methalox apoapsis re-aim and the canted departure leg before the departure
    burn's mass fraction means anything, so the chain carries three factors
    where it carried one.

    Attributes:
        label: Which cycle this is.
        returned_per_pad_kg: Kilograms returning per kilogram off the pad.
        chain_to_departure: What survives the lob-to-departure chain.
        node_survival: Mass fraction surviving the solar-periapsis collision.
        stated_floor: The committed 1/15 **return floor**.
        stated_margin: ``returned_per_pad_kg`` over that floor.
        free_parking_returned_per_pad_kg: The same quantity read the way ADR
            0019 and ADR 0020 read it, with the parking orbit given away.
            Reported so the size of the correction is visible rather than
            inferred, **not** as an alternative answer.
        free_parking_margin: That reading over the committed floor.
        return_value_ratio: How much more a kilogram returning at this cycle's
            closing speed is worth than one at 60 km/s, measured as the payload
            each pushes to a common target.
        rescaled_floor: The committed floor restated at that value.
        rescaled_margin: ``returned_per_pad_kg`` over the rescaled floor.
        returned_per_pad_kg_per_year: The same currency, time-normalised.
        cycle_years: Departure-to-return time (yr).
    """

    label: str
    returned_per_pad_kg: float
    chain_to_departure: float
    node_survival: float
    stated_floor: float
    stated_margin: float
    free_parking_returned_per_pad_kg: float
    free_parking_margin: float
    return_value_ratio: float
    rescaled_floor: float
    rescaled_margin: float
    returned_per_pad_kg_per_year: float
    cycle_years: float

    @property
    def clears_committed_floor(self) -> bool:
        """Whether the cycle returns the fifteenth of liftoff it committed to."""
        return self.returned_per_pad_kg >= self.stated_floor

    @property
    def clears_rescaled_floor(self) -> bool:
        """Whether it clears the floor restated for a faster returned kilogram."""
        return self.returned_per_pad_kg >= self.rescaled_floor

    @property
    def correction_factor(self) -> float:
        """How much charging the lob-to-departure chain cost the pad return."""
        return self.returned_per_pad_kg / self.free_parking_returned_per_pad_kg


def split_push_launch_ledger(
    ledger: SplitPushLedger, params: Optional[_FlybyParams] = None
) -> SplitPushLaunchLedger:
    """Charge a **split push** for the mass it lifted off the pad.

    Args:
        ledger: The split push to charge.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitPushLaunchLedger`.
    """
    p = params if params is not None else _powered_flyby_params()
    returned = (
        PAYLOAD_FRACTION_AT_INTERCEPT * ledger.chain_to_departure * ledger.node_survival
    )
    free = (
        PAYLOAD_FRACTION_AT_INTERCEPT * ledger.departure_fraction * ledger.node_survival
    )
    value = return_value_ratio(ledger.stream_speed, p)
    rescaled = RETURN_FLOOR / value
    return SplitPushLaunchLedger(
        label=ledger.label,
        returned_per_pad_kg=returned,
        chain_to_departure=ledger.chain_to_departure,
        node_survival=ledger.node_survival,
        stated_floor=RETURN_FLOOR,
        stated_margin=returned / RETURN_FLOOR,
        free_parking_returned_per_pad_kg=free,
        free_parking_margin=free / RETURN_FLOOR,
        return_value_ratio=value,
        rescaled_floor=rescaled,
        rescaled_margin=returned / rescaled,
        returned_per_pad_kg_per_year=returned / ledger.cycle_years,
        cycle_years=ledger.cycle_years,
    )


@dataclass(frozen=True)
class SplitPushPadCeilings:
    """What the corrected pad ledger and the **expansion floor** jointly allow.

    Two caps on the departure slug ratio, and the **split push** changed both.

    The pad cap moved because the chain it charges got longer.  The expansion
    cap moved because the split *added a leg*: the overtaking push runs at
    ``theta`` = 0, where the vehicle's own speed subtracts from the closing
    speed directly, so it is colder than the canted departure leg throughout and
    colder at its end than anywhere else in the departure.  ``slug_ratio_ceilings``
    never saw that leg, because in the single-push ledger it did not exist.

    Attributes:
        label: Which cycle this is.
        overtaking_coldest_closing: Coldest closing speed on the first leg (km/s).
        departure_coldest_closing: Coldest closing speed on the second (km/s).
        overtaking_expansion_window: **Expansion floor** window on ``k`` at the
            first leg's coldest instant, at the margin asked for; None when no
            slug ratio clears the floor there.
        departure_expansion_window: The same for the second leg.
        expansion_margin: Headroom above the **ignition bill** that was demanded.
        peak_slug_ratio: Slug ratio maximising kilograms returned per pad kilogram.
        peak_returned_per_pad_kg: That maximum.
        launch_window: Slug ratios clearing the committed 1/15 **return floor**,
            or None when none do.  A closed interval, not a ceiling: the pad
            return peaks in ``k`` and falls away on both sides.
        rescaled_window: The same against the speed-rescaled floor.
        return_value_ratio: What sets that rescale.
        binding: Which constraint decides -- "expansion", "launch",
            "launch (fails at every k)", "expansion (no k works)", or
            "disjoint" when both are satisfiable and never together.
    """

    label: str
    overtaking_coldest_closing: float
    departure_coldest_closing: float
    overtaking_expansion_window: Optional[Tuple[float, float]]
    departure_expansion_window: Optional[Tuple[float, float]]
    expansion_margin: float
    peak_slug_ratio: float
    peak_returned_per_pad_kg: float
    launch_window: Optional[Tuple[float, float]]
    rescaled_window: Optional[Tuple[float, float]]
    return_value_ratio: float
    binding: str

    @property
    def peak_margin(self) -> float:
        """Best returned-per-pad kilogram over the committed floor."""
        return self.peak_returned_per_pad_kg / RETURN_FLOOR

    @property
    def committed_floor_reachable(self) -> bool:
        """Whether any slug ratio at all clears the committed floor."""
        return self.launch_window is not None

    @property
    def expansion_window(self) -> Optional[Tuple[float, float]]:
        """Slug ratios both legs' plumes still conduct at, or None."""
        return _intersect(
            self.overtaking_expansion_window, self.departure_expansion_window
        )

    @property
    def admissible_window(self) -> Optional[Tuple[float, float]]:
        """Slug ratios that clear the pad floor *and* keep both plumes conducting.

        The quantity the cycle actually has to have.  None is the interesting
        verdict, and it does not mean "no slug ratio works": it means the two
        constraints are individually satisfiable and never at the same ``k``.
        """
        return _intersect(self.expansion_window, self.launch_window)


def _intersect(
    first: Optional[Tuple[float, float]], second: Optional[Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """Intersect two optional closed intervals, returning None when empty.

    Args:
        first: The first interval, or None.
        second: The second, or None.

    Returns:
        The intersection, or None when either is missing or they do not overlap.
    """
    if first is None or second is None:
        return None
    low, high = max(first[0], second[0]), min(first[1], second[1])
    return (low, high) if low < high else None


def split_push_pad_ceilings(
    closure: SynodicCycleClosure,
    node_survival: float,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    label: Optional[str] = None,
    params: Optional[_FlybyParams] = None,
) -> Optional[SplitPushPadCeilings]:
    """Find the slug ratios a **split push** may actually be flown at.

    The corrected counterpart to :func:`slug_ratio_ceilings`, which reads both
    caps off the single-push ledger and therefore reads them off a departure
    that was never charged and a leg that did not exist.

    Args:
        closure: The solved cycle.
        node_survival: Mass fraction surviving the solar-periapsis collision.
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        expansion_margin: Thermal headroom demanded above the **expansion floor**.
        label: Name for the row; taken from the closure when omitted.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`SplitPushPadCeilings`, or None when no slug ratio in
        ``_PAD_PEAK_BRACKET`` flies the cycle at all.
    """
    p = params if params is not None else _powered_flyby_params()

    def ledger_at(slug_ratio: float) -> Optional[SplitPushLedger]:
        """The split push at one departure slug ratio."""
        return split_push_from_state(
            label or "split push",
            closure.departure_excess,
            closure.departure_aim_deg,
            closure.total_tof_years,
            closure.return_excess,
            closure.push_axis_deg,
            node_survival,
            slug_ratio,
            parking_period_days,
            jet_energy_efficiency,
            p,
        )

    def pad_at(slug_ratio: float) -> float:
        """Kilograms returned per pad kilogram, zero where the cycle does not fly."""
        row = ledger_at(float(slug_ratio))
        return (
            0.0
            if row is None
            else PAYLOAD_FRACTION_AT_INTERCEPT
            * row.chain_to_departure
            * row.node_survival
        )

    peak = minimize_scalar(
        lambda k: -pad_at(float(k)),
        bounds=_PAD_PEAK_BRACKET,
        method="bounded",
        options={"xatol": 1e-8},
    )
    peak_ratio = float(peak.x)
    peak_value = pad_at(peak_ratio)
    probe = ledger_at(peak_ratio)
    if probe is None or peak_value <= 0.0:
        return None

    def window_above(floor: float) -> Optional[Tuple[float, float]]:
        """Slug ratios clearing ``floor``, bisected either side of the peak."""
        if peak_value < floor:
            return None
        low, high = _PAD_CEILING_BRACKET
        crossings = []
        for bound in (low, high):
            if pad_at(bound) >= floor:
                crossings.append(bound)
                continue
            lower, upper = sorted((bound, peak_ratio))
            crossings.append(
                float(
                    brentq(lambda k: pad_at(float(k)) - floor, lower, upper, xtol=1e-8)
                )
            )
        return (crossings[0], crossings[1])

    value = return_value_ratio(probe.stream_speed, p)
    launch_window = window_above(RETURN_FLOOR)
    rescaled_window = window_above(RETURN_FLOOR / value)
    overtaking = expansion_limited_slug_ratio_window(
        probe.overtaking_coldest_closing, jet_energy_efficiency, expansion_margin
    )
    departure = expansion_limited_slug_ratio_window(
        probe.departure_coldest_closing, jet_energy_efficiency, expansion_margin
    )
    expansion = _intersect(overtaking, departure)
    # Both can fail independently and the caller has to be told when both did,
    # because "the plume will not conduct" and "the cycle does not pay for its
    # launch" are separate verdicts with separate fixes.
    if expansion is None and launch_window is None:
        binding = "launch and expansion (neither admits any k)"
    elif launch_window is None:
        binding = "launch (fails at every k)"
    elif expansion is None:
        binding = "expansion (no k works)"
    elif _intersect(expansion, launch_window) is None:
        binding = "disjoint"
    else:
        binding = "expansion" if expansion[1] < launch_window[1] else "launch"
    return SplitPushPadCeilings(
        label=label or "split push",
        overtaking_coldest_closing=probe.overtaking_coldest_closing,
        departure_coldest_closing=probe.departure_coldest_closing,
        overtaking_expansion_window=overtaking,
        departure_expansion_window=departure,
        expansion_margin=expansion_margin,
        peak_slug_ratio=peak_ratio,
        peak_returned_per_pad_kg=peak_value,
        launch_window=launch_window,
        rescaled_window=rescaled_window,
        return_value_ratio=value,
        binding=binding,
    )


def split_push_slug_ratio_table(
    closure: SynodicCycleClosure,
    node_survival: float,
    slug_ratios: Sequence[float] = SLUG_RATIO_GRID,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> List[Tuple[float, SplitPushLedger, SplitPushLaunchLedger]]:
    """Score one closed cycle across departure slug ratios, charged from the pad.

    The corrected counterpart to :func:`slug_ratio_table`.  The two columns move
    in opposite directions and that opposition is the whole content of the
    **launch ledger**: growth per *impactor* kilogram rises with ``k`` without
    limit, because slug is free in that currency, while return per *pad*
    kilogram peaks low and falls, because it is not.

    Args:
        closure: The solved cycle, held fixed across the sweep.
        node_survival: Mass fraction surviving the solar-periapsis collision.
        slug_ratios: Departure slug ratios to score.
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        ``(slug_ratio, ledger, launch)`` for each ratio the cycle flies at.
    """
    p = params if params is not None else _powered_flyby_params()
    out: List[Tuple[float, SplitPushLedger, SplitPushLaunchLedger]] = []
    for ratio in slug_ratios:
        row = split_push_from_state(
            f"k = {ratio:g}",
            closure.departure_excess,
            closure.departure_aim_deg,
            closure.total_tof_years,
            closure.return_excess,
            closure.push_axis_deg,
            node_survival,
            float(ratio),
            parking_period_days,
            jet_energy_efficiency,
            p,
        )
        if row is None:
            continue
        out.append((float(ratio), row, split_push_launch_ledger(row, p)))
    return out


def split_push_depth_table(
    depths: Sequence[float] = DEPTH_GRID,
    return_excess: float = CONSERVATIVE_RETURN_EXCESS,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> List[Tuple[float, SplitPushLedger, SplitPushLaunchLedger]]:
    """Sweep the dive depth with the departure charged from the pad.

    Node survival is **derived** at every depth rather than taken from
    ``PROPOSED_NODE_SURVIVAL``, whose rounded 1/2 and 1/3 are calibrated at the
    two reference depths only; mixing rounded and derived rows would put a step
    in an otherwise smooth sweep and make the crossing depth an artefact of the
    rounding.

    Args:
        depths: Perihelion distances to sweep (solar radii).
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        slug_ratio: Departure slug ratio.
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        ``(dive_solar_radii, ledger, launch)`` for each depth that closes.
    """
    p = params if params is not None else _powered_flyby_params()
    out: List[Tuple[float, SplitPushLedger, SplitPushLaunchLedger]] = []
    for depth in depths:
        closure = solve_synodic_closure(3, p, float(depth), return_excess, 1.0)
        if closure is None:
            continue
        node = dive_node(
            float(depth),
            closure.periapsis_boost,
            DEFAULT_PERIAPSIS_SLUG_RATIO,
            jet_energy_efficiency,
            p,
        )
        row = split_push_from_state(
            f"{depth:g} R_sun",
            closure.departure_excess,
            closure.departure_aim_deg,
            closure.total_tof_years,
            closure.return_excess,
            closure.push_axis_deg,
            node.survival,
            slug_ratio,
            parking_period_days,
            jet_energy_efficiency,
            p,
        )
        if row is None:
            continue
        out.append((float(depth), row, split_push_launch_ledger(row, p)))
    return out


def pad_floor_depth(
    return_excess: float = CONSERVATIVE_RETURN_EXCESS,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    bracket: Tuple[float, float] = PAD_FLOOR_DEPTH_BRACKET,
    params: Optional[_FlybyParams] = None,
) -> Optional[float]:
    """How deep the dive has to stay for the cycle to pay for its own launch.

    The single number the depth dial owes the **launch ledger**: shallower than
    this the cycle does not return the fifteenth of liftoff it committed to, at
    this climb-out and this slug ratio.

    Both of those are held fixed, and at the defaults both are *inadmissible*:
    a 75 km/s climb-out leaves the **overtaking leg** below the **expansion
    floor** at every depth, so the 21.09 solar radii this reports is a crossing
    on a dial nothing may fly.  :func:`admissible_pad_floor_depth` flies each
    depth at its own best conducting climb-out instead, and puts the crossing
    shallower (ADR 0022).  Keep this one for the comparison it makes and quote
    that one for the recommendation.

    Args:
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        slug_ratio: Departure slug ratio.
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        bracket: Depths to bisect between (solar radii).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The crossing depth in solar radii, or None when the floor is cleared or
        failed across the whole bracket and there is no crossing to report.
    """
    p = params if params is not None else _powered_flyby_params()

    def margin(depth: float) -> float:
        """Pad return at one depth, less the committed floor."""
        rows = split_push_depth_table(
            (float(depth),),
            return_excess,
            slug_ratio,
            parking_period_days,
            jet_energy_efficiency,
            p,
        )
        return (
            -RETURN_FLOOR if not rows else rows[0][2].returned_per_pad_kg - RETURN_FLOOR
        )

    low, high = bracket
    if margin(low) * margin(high) >= 0.0:
        return None
    return float(brentq(margin, low, high, xtol=1e-4))


@dataclass(frozen=True)
class PadFrontierRow:
    """The best a cycle can do on the pad ledger at one climb-out excess.

    Built to answer one question: is the shallow cycle *rescuable* by moving the
    knob the trade leaves free?  The two constraints pull opposite ways on it.
    Boosting less at the node spends less of the node's propellant, so the pad
    return rises -- but it also lowers the Earth closing speed, and the
    **expansion floor** needs that speed to leave the plume conducting once the
    jet is drawn.  This row reports both ends of that squeeze at one excess.

    Attributes:
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        earth_closing_speed: Where the stream arrives at Earth (km/s).
        overtaking_coldest_closing: Coldest closing speed on the first leg (km/s).
        departure_coldest_closing: Coldest closing speed on the second (km/s).
        node_survival: Derived mass fraction surviving the node.
        expansion_window: Slug ratios both legs' plumes still conduct at, or
            None when the overtaking leg admits none.
        best_slug_ratio: The pad-optimal slug ratio inside that window.
        best_returned_per_pad_kg: What it returns per pad kilogram.
        best_margin: That over the committed 1/15 **return floor**.
        doubling_years: Payload doubling time at that operating point (yr).
        growth_slug_ratio: The *largest* admissible slug ratio -- the fastest
            growth this excess allows while still paying for its launch, since
            growth per impactor kilogram rises with ``k`` throughout.  None when
            nothing here is admissible.
        growth_doubling_years: Doubling time there (yr).
        growth_margin: Its pad return over the committed floor.  Sits on 1.0
            when the pad floor is what stopped ``k``, and above it when the
            **expansion floor** stopped ``k`` first.
    """

    return_excess: float
    earth_closing_speed: float
    overtaking_coldest_closing: float
    departure_coldest_closing: float
    node_survival: float
    expansion_window: Optional[Tuple[float, float]]
    best_slug_ratio: Optional[float]
    best_returned_per_pad_kg: Optional[float]
    best_margin: Optional[float]
    doubling_years: Optional[float]
    growth_slug_ratio: Optional[float]
    growth_doubling_years: Optional[float]
    growth_margin: Optional[float]

    @property
    def clears_committed_floor(self) -> bool:
        """Whether any admissible slug ratio pays for the launch here."""
        return self.best_margin is not None and self.best_margin >= 1.0


def pad_return_frontier(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    excesses: Sequence[float] = PAD_FRONTIER_EXCESS_GRID,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    reserve: Optional[float] = None,
    params: Optional[_FlybyParams] = None,
) -> List[PadFrontierRow]:
    """Scan the climb-out excess for a point that pays for its launch and conducts.

    Maximises the pad return over the slug ratio at each excess, subject to the
    **expansion floor** on *both* legs of the **split push**, and reports what
    the best admissible point returns.  A depth whose every row fails is a depth
    the corrected **launch ledger** rules out, not one that merely scores badly
    at the reference operating point.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        excesses: Climb-out excesses to scan (km/s).
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        expansion_margin: Thermal headroom demanded above the **expansion floor**.
        reserve: Merge energy the jet may not spend (MJ/kg); defaults to the
            conservative end of the **conduction reserve** bracket.  It does not
            enter the pad return at any slug ratio -- that is a mass quantity and
            this is a plume-thermodynamics one -- but it moves the *climb-out* at
            which the overtaking plume stops conducting, and the pad return is
            won at low climb-outs.  So the two are coupled through this scan and
            nowhere else.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        One :class:`PadFrontierRow` per excess that closes.
    """
    p = params if params is not None else _powered_flyby_params()
    out: List[PadFrontierRow] = []
    for excess in excesses:
        closure = solve_synodic_closure(3, p, dive_solar_radii, float(excess), 1.0)
        if closure is None:
            continue
        node = dive_node(
            dive_solar_radii,
            closure.periapsis_boost,
            DEFAULT_PERIAPSIS_SLUG_RATIO,
            jet_energy_efficiency,
            p,
        )

        def ledger_at(slug_ratio: float) -> Optional[SplitPushLedger]:
            """The split push at one departure slug ratio, this excess held."""
            return split_push_from_state(
                f"{dive_solar_radii:g} R_sun / v_inf {excess:g}",
                closure.departure_excess,
                closure.departure_aim_deg,
                closure.total_tof_years,
                closure.return_excess,
                closure.push_axis_deg,
                node.survival,
                slug_ratio,
                parking_period_days,
                jet_energy_efficiency,
                p,
            )

        probe = ledger_at(CONSERVATIVE_SLUG_RATIO)
        if probe is None:
            continue
        window = _intersect(
            expansion_limited_slug_ratio_window(
                probe.overtaking_coldest_closing,
                jet_energy_efficiency,
                expansion_margin,
                reserve,
            ),
            expansion_limited_slug_ratio_window(
                probe.departure_coldest_closing,
                jet_energy_efficiency,
                expansion_margin,
                reserve,
            ),
        )

        def pad_chain(slug_ratio: float) -> float:
            """Round-trip fraction of the lob's payload, negated for minimising."""
            row = ledger_at(float(slug_ratio))
            return 0.0 if row is None else -row.chain_to_departure * row.node_survival

        def pad_at(slug_ratio: float) -> float:
            """Kilograms returned per pad kilogram at one slug ratio."""
            row = ledger_at(float(slug_ratio))
            return (
                0.0
                if row is None
                else PAYLOAD_FRACTION_AT_INTERCEPT
                * row.chain_to_departure
                * row.node_survival
            )

        best_ratio: Optional[float] = None
        best_value: Optional[float] = None
        doubling: Optional[float] = None
        growth_ratio: Optional[float] = None
        growth_doubling: Optional[float] = None
        growth_margin: Optional[float] = None
        if window is not None:
            found = minimize_scalar(
                pad_chain,
                bounds=window,
                method="bounded",
                options={"xatol": 1e-6},
            )
            best_ratio = float(found.x)
            winner = ledger_at(best_ratio)
            if winner is not None:
                best_value = split_push_launch_ledger(winner, p).returned_per_pad_kg
                doubling = winner.split_doubling_years
            # Growth per impactor kilogram rises with k throughout, so the
            # fastest admissible cycle sits on whichever cap comes first: the
            # expansion window's ceiling, or the slug ratio where the pad return
            # falls through the floor.
            if best_value is not None and best_value >= RETURN_FLOOR:
                growth_ratio = (
                    window[1]
                    if pad_at(window[1]) >= RETURN_FLOOR
                    else float(
                        brentq(
                            lambda k: pad_at(float(k)) - RETURN_FLOOR,
                            best_ratio,
                            window[1],
                            xtol=1e-6,
                        )
                    )
                )
                fastest = ledger_at(growth_ratio)
                if fastest is not None:
                    growth_doubling = fastest.split_doubling_years
                    growth_margin = (
                        split_push_launch_ledger(fastest, p).returned_per_pad_kg
                        / RETURN_FLOOR
                    )
        out.append(
            PadFrontierRow(
                return_excess=float(excess),
                earth_closing_speed=probe.stream_speed,
                overtaking_coldest_closing=probe.overtaking_coldest_closing,
                departure_coldest_closing=probe.departure_coldest_closing,
                node_survival=node.survival,
                expansion_window=window,
                best_slug_ratio=best_ratio,
                best_returned_per_pad_kg=best_value,
                best_margin=None if best_value is None else best_value / RETURN_FLOOR,
                doubling_years=doubling,
                growth_slug_ratio=growth_ratio,
                growth_doubling_years=growth_doubling,
                growth_margin=growth_margin,
            )
        )
    return out


@dataclass(frozen=True)
class ConductionBracketRow:
    """The best pad return a depth can reach at one reading of the plume physics.

    The **conduction reserve** does not enter the pad return at any slug ratio --
    it is a plume-thermodynamics quantity and the ledger is a mass quantity.
    What it moves is the *climb-out* at which the overtaking plume stops
    conducting, and since the pad return rises as the node boost falls, that
    threshold is exactly what decides whether the paying end of the dial is
    reachable.  So the two are coupled through the frontier and not through any
    single cycle, which is why this has to be swept rather than argued.

    Each cell's optimum is found by :func:`pad_frontier_optimum`, which bisects
    to the cell's own conduction threshold and searches upward from there.  It
    was originally read off ``PAD_FRONTIER_EXCESS_GRID``, and since every cell's
    optimum sits within about a kilometre per second of a threshold that moves
    with the reserve, a grid stepping 5 to 15 km/s understated all six of them
    (ADR 0022).

    Attributes:
        reserve_label: Which end of the bracket this is.
        reserve: Merge energy the jet may not spend (MJ/kg).
        expansion_margin: Headroom demanded above the **expansion floor**.
        threshold_closing_speed: Coldest closing speed below which no slug ratio
            conducts at this reading (km/s), or None if the search found none.
        threshold_return_excess: The same threshold as a climb-out excess at this
            depth (km/s) -- the slowest climb-out the cycle may be flown at.
        best_return_excess: Climb-out excess of the best admissible point (km/s).
        best_slug_ratio: Its slug ratio.
        best_returned_per_pad_kg: What it returns per pad kilogram.
        best_margin: That over the committed 1/15 **return floor**.
        best_doubling_years: Its payload doubling time (yr).
    """

    reserve_label: str
    reserve: float
    expansion_margin: float
    threshold_closing_speed: Optional[float]
    threshold_return_excess: Optional[float]
    best_return_excess: Optional[float]
    best_slug_ratio: Optional[float]
    best_returned_per_pad_kg: Optional[float]
    best_margin: Optional[float]
    best_doubling_years: Optional[float]

    @property
    def clears_committed_floor(self) -> bool:
        """Whether any admissible point pays for its launch at this reading."""
        return self.best_margin is not None and self.best_margin >= 1.0


def conduction_threshold_closing_speed(
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    reserve: Optional[float] = None,
    bracket: Tuple[float, float] = _CONDUCTION_THRESHOLD_BRACKET,
) -> Optional[float]:
    """Coldest closing speed below which no slug ratio keeps the plume conducting.

    The one number that connects the plume physics to the **launch ledger**:
    below it the **expansion floor** admits nothing at any ``k``, so the whole
    cheap-boost end of the climb-out dial -- the end where the pad return is
    good -- is unavailable.

    Args:
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1).
        expansion_margin: Headroom demanded above the **expansion floor**.
        reserve: Merge energy the jet may not spend (MJ/kg); defaults to the
            conservative end of the **conduction reserve**.
        bracket: Closing speeds to bisect between (km/s).

    Returns:
        The threshold in km/s, or None when the bracket does not contain one.
    """

    def admits(closing_speed: float) -> float:
        """+1 where some slug ratio conducts, -1 where none does."""
        window = expansion_limited_slug_ratio_window(
            float(closing_speed), jet_energy_efficiency, expansion_margin, reserve
        )
        return 1.0 if window is not None else -1.0

    low, high = bracket
    if admits(low) * admits(high) >= 0.0:
        return None
    return float(brentq(admits, low, high, xtol=1e-4))


def overtaking_coldest_closing_at(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    return_excess: float = CONSERVATIVE_RETURN_EXCESS,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> Optional[float]:
    """Coldest instant of the **overtaking leg** at one climb-out excess.

    The bridge between the two halves of the squeeze.
    :func:`conduction_threshold_closing_speed` says how cold the plume may get;
    this says how cold the first push actually gets at a given climb-out, so the
    two can be equated for the slowest climb-out the cycle may be flown at.

    Slug ratio does not enter -- the coldest instant is
    ``_leg_coldest_closing(0, stream, lob, v_park)``, which is set by the stream
    speed and the two ends of the burn -- and neither does the node survival.
    Depth does, through the closure: the same climb-out excess arrives at Earth
    faster from a deeper perihelion.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The coldest closing speed in km/s, or None when no closure exists there.
    """
    p = params if params is not None else _powered_flyby_params()
    closure = solve_synodic_closure(3, p, dive_solar_radii, float(return_excess), 1.0)
    if closure is None:
        return None
    node = dive_node(
        dive_solar_radii,
        closure.periapsis_boost,
        DEFAULT_PERIAPSIS_SLUG_RATIO,
        jet_energy_efficiency,
        p,
    )
    row = split_push_from_state(
        f"{dive_solar_radii:g} R_sun / v_inf {return_excess:g}",
        closure.departure_excess,
        closure.departure_aim_deg,
        closure.total_tof_years,
        closure.return_excess,
        closure.push_axis_deg,
        node.survival,
        CONSERVATIVE_SLUG_RATIO,
        parking_period_days,
        jet_energy_efficiency,
        p,
    )
    return None if row is None else row.overtaking_coldest_closing


def conduction_threshold_excess(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    reserve: Optional[float] = None,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    bracket: Tuple[float, float] = _CONDUCTION_EXCESS_BRACKET,
    params: Optional[_FlybyParams] = None,
) -> Optional[float]:
    """Slowest climb-out whose **overtaking leg** still leaves the plume conducting.

    The same threshold :func:`conduction_threshold_closing_speed` states as a
    closing speed, carried back onto the knob the trade actually leaves free.
    This is the number the **conduction bracket sweep** has to be anchored on:
    the pad return rises as the climb-out falls, so wherever the paying end of
    the dial is, it is just above this.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        expansion_margin: Headroom demanded above the **expansion floor**.
        reserve: Merge energy the jet may not spend (MJ/kg); defaults to the
            conservative end of the **conduction reserve**.
        parking_period_days: Period of the ellipse between the two pushes.
        bracket: Climb-out excesses to bisect between (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The threshold climb-out in km/s, or None when the bracket does not
        contain one -- either because no closing-speed threshold exists at this
        reading, or because the bracket does not straddle it.
    """
    p = params if params is not None else _powered_flyby_params()
    threshold = conduction_threshold_closing_speed(
        jet_energy_efficiency, expansion_margin, reserve
    )
    if threshold is None:
        return None

    def excess_of(excess: float) -> float:
        """Coldest instant at this climb-out, less the threshold."""
        coldest = overtaking_coldest_closing_at(
            dive_solar_radii,
            float(excess),
            parking_period_days,
            jet_energy_efficiency,
            p,
        )
        return -threshold if coldest is None else coldest - threshold

    low, high = bracket
    if excess_of(low) * excess_of(high) >= 0.0:
        return None
    return float(brentq(excess_of, low, high, xtol=1e-4))


def pad_frontier_optimum(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    reserve: Optional[float] = None,
    span: float = PAD_OPTIMUM_EXCESS_SPAN,
    params: Optional[_FlybyParams] = None,
) -> Optional[PadFrontierRow]:
    """Best pad return a depth can reach, found by bisection rather than off a grid.

    :func:`pad_return_frontier` reports what a *stated* list of climb-outs
    returns, and ``PAD_FRONTIER_EXCESS_GRID`` steps 5 to 15 km/s.  That is fine
    for showing the shape of the squeeze and wrong for reporting its optimum,
    because at the shallow node the optimum sits within about a kilometre per
    second of the conduction threshold, and that threshold moves with every
    reading of the **conduction reserve** while the grid does not -- so each cell
    of the bracket sweep was being scored at whatever grid point happened to fall
    above its own threshold, always understating it.  ADR 0022.

    The peak is interior and the search is one-dimensional because the two
    effects pull opposite ways in a single knob.  Dropping the climb-out raises
    the pad return, since less of the node's propellant is spent; but it also
    narrows the **expansion floor**'s window, which pinches shut at the
    threshold, and a pinched window forces a slug ratio well above the one the
    pad return peaks at.  So the pad return rises as the climb-out falls until
    the pinch costs more than the boost saves, and that crossing is the optimum.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        expansion_margin: Headroom demanded above the **expansion floor**.
        reserve: Merge energy the jet may not spend (MJ/kg); defaults to the
            conservative end of the **conduction reserve**.
        span: How far above the threshold to search (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`PadFrontierRow` at the optimising climb-out, or None when no
        climb-out at this depth conducts at all.
    """
    p = params if params is not None else _powered_flyby_params()
    threshold = conduction_threshold_excess(
        dive_solar_radii,
        jet_energy_efficiency,
        expansion_margin,
        reserve,
        parking_period_days,
        params=p,
    )
    if threshold is None:
        return None

    def row_at(excess: float) -> Optional[PadFrontierRow]:
        """The frontier row at one climb-out, or None where nothing closes."""
        rows = pad_return_frontier(
            dive_solar_radii,
            (float(excess),),
            parking_period_days,
            jet_energy_efficiency,
            expansion_margin,
            reserve,
            p,
        )
        return rows[0] if rows else None

    def negative_pad_return(excess: float) -> float:
        """Pad return at one climb-out, negated for minimising."""
        row = row_at(float(excess))
        return (
            0.0
            if row is None or row.best_returned_per_pad_kg is None
            else -row.best_returned_per_pad_kg
        )

    found = minimize_scalar(
        negative_pad_return,
        bounds=(threshold, threshold + span),
        method="bounded",
        options={"xatol": _PAD_OPTIMUM_EXCESS_XATOL},
    )
    return row_at(float(found.x))


def admissible_pad_floor_depth(
    parking_period_days: float = DEFAULT_PARKING_PERIOD_DAYS,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    expansion_margin: float = DEFAULT_EXPANSION_MARGIN,
    reserve: Optional[float] = None,
    bracket: Tuple[float, float] = ADMISSIBLE_PAD_FLOOR_DEPTH_BRACKET,
    span: float = PAD_OPTIMUM_EXCESS_SPAN,
    params: Optional[_FlybyParams] = None,
) -> Optional[float]:
    """How shallow the dive may go when each depth flies its own best climb-out.

    :func:`pad_floor_depth` answers the same question down a dial that holds the
    climb-out at ``CONSERVATIVE_RETURN_EXCESS`` = 75 km/s and the slug ratio at
    8.5.  Both are inadmissible: 75 km/s leaves the **overtaking leg** at 74.21
    km/s at 32 solar radii and 78.91 at 4, and the **expansion floor** wants
    79.57 at the operating margin, so that dial is barred at *every* depth.  Its
    21.09 crossing is therefore a crossing at an operating point nothing may fly.

    Here every depth is flown at the climb-out that maximises its own pad return
    subject to conducting -- :func:`pad_frontier_optimum` -- and the slug ratio
    that maximises it there.  That is the dial a mission designer actually has,
    and the crossing on it is the shallowest dive that pays for its launch.
    ADR 0022.

    Args:
        parking_period_days: Period of the ellipse between the two pushes.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        expansion_margin: Headroom demanded above the **expansion floor**.
        reserve: Merge energy the jet may not spend (MJ/kg); defaults to the
            conservative end of the **conduction reserve**.
        bracket: Depths to bisect between (solar radii).
        span: How far above each threshold to search for the peak (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The crossing depth in solar radii, or None when the bracket does not
        straddle it.
    """
    p = params if params is not None else _powered_flyby_params()

    def margin(depth: float) -> float:
        """Best admissible pad return at one depth, less the committed floor."""
        row = pad_frontier_optimum(
            float(depth),
            parking_period_days,
            jet_energy_efficiency,
            expansion_margin,
            reserve,
            span,
            p,
        )
        return (
            -RETURN_FLOOR
            if row is None or row.best_returned_per_pad_kg is None
            else row.best_returned_per_pad_kg - RETURN_FLOOR
        )

    low, high = bracket
    if margin(low) * margin(high) >= 0.0:
        return None
    return float(brentq(margin, low, high, xtol=_ADMISSIBLE_PAD_FLOOR_XTOL))


def conduction_bracket_frontier(
    dive_solar_radii: float = CONSERVATIVE_DIVE_SOLAR_RADII,
    reserves: Sequence[Tuple[str, float]] = CONDUCTION_RESERVE_BRACKET,
    margins: Sequence[float] = CONDUCTION_MARGIN_GRID,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    span: float = PAD_OPTIMUM_EXCESS_SPAN,
    params: Optional[_FlybyParams] = None,
) -> List[ConductionBracketRow]:
    """Ask whether the pad verdict survives every reading of the plume physics.

    ADR 0020 left the **conduction reserve** as an explicit bracket because the
    magnetic Reynolds number at nozzle exit has never been computed, and every
    reading of it cleared the efficiency the cycles are scored at -- so nothing
    depended on where in the bracket the truth sat.  Once the **launch ledger**
    is charged from the pad that stops being true at the shallow node, because
    the reserve sets the lowest climb-out the overtaking plume survives and the
    pad return is won at low climb-outs.

    Each cell is scored by :func:`pad_frontier_optimum`, which bisects to that
    cell's own conduction threshold rather than reading the best of a fixed
    climb-out grid.  The grid reading -- ``pad_return_frontier`` on
    ``PAD_FRONTIER_EXCESS_GRID``, which is still what plots the shape of the
    squeeze -- understated every cell, because each optimum sits within about a
    kilometre per second of a threshold that the grid steps past in 5 to 15
    (ADR 0022).

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        reserves: ``(label, MJ/kg)`` readings to sweep.
        margins: Expansion margins to sweep.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        span: How far above each threshold to search for the peak (km/s).
        params: Float parameter block; built with defaults when omitted.

    Returns:
        One :class:`ConductionBracketRow` per (reserve, margin) cell.
    """
    p = params if params is not None else _powered_flyby_params()
    out: List[ConductionBracketRow] = []
    for label, reserve in reserves:
        for margin in margins:
            best = pad_frontier_optimum(
                dive_solar_radii,
                jet_energy_efficiency=jet_energy_efficiency,
                expansion_margin=float(margin),
                reserve=float(reserve),
                span=span,
                params=p,
            )
            if best is not None and best.best_returned_per_pad_kg is None:
                best = None
            out.append(
                ConductionBracketRow(
                    reserve_label=label,
                    reserve=float(reserve),
                    expansion_margin=float(margin),
                    threshold_closing_speed=conduction_threshold_closing_speed(
                        jet_energy_efficiency, float(margin), float(reserve)
                    ),
                    threshold_return_excess=conduction_threshold_excess(
                        dive_solar_radii,
                        jet_energy_efficiency,
                        float(margin),
                        float(reserve),
                        params=p,
                    ),
                    best_return_excess=None if best is None else best.return_excess,
                    best_slug_ratio=None if best is None else best.best_slug_ratio,
                    best_returned_per_pad_kg=(
                        None if best is None else best.best_returned_per_pad_kg
                    ),
                    best_margin=None if best is None else best.best_margin,
                    best_doubling_years=None if best is None else best.doubling_years,
                )
            )
    return out


# --------------------------------------------------------------------------
# Reporting the corrected pad ledger
# --------------------------------------------------------------------------


def describe_split_push(
    ledger: SplitPushLedger,
    launch: SplitPushLaunchLedger,
    ceilings: Optional[SplitPushPadCeilings] = None,
) -> str:
    """Render one **split push** and its pad ledger as a block report.

    Args:
        ledger: The split push.
        launch: Its launch ledger.
        ceilings: What the pad floor and the **expansion floor** jointly allow,
            when it has been computed.

    Returns:
        A multi-line report.
    """
    lines = [
        f"=== {ledger.label}: departure charged from the pad",
        "",
        f"  lob -> departure               {ledger.lob_speed:.3f}"
        f" -> {ledger.departure_speed:.3f} km/s"
        f"   ({ledger.departure_speed - ledger.lob_speed:.3f} km/s to fly)",
        f"  cant at the departure burn     {ledger.cant_deg:.2f} deg",
        f"  stream speed at Earth          {ledger.stream_speed:.2f} km/s",
        f"  parking orbit                  {ledger.parking_period_days:g} d,"
        f" periapsis {ledger.parking_periapsis_speed:.3f} km/s",
        "",
        f"  overtaking leg f1              {ledger.overtaking_fraction:.4f}"
        f"   (coldest closing {ledger.overtaking_coldest_closing:.2f} km/s)",
        f"  apoapsis re-aim                {ledger.reaim_delta_v:.3f} km/s"
        f"   -> {ledger.reaim_fraction:.4f} on methalox",
        f"  departure leg f2               {ledger.departure_fraction:.4f}"
        f"   (coldest closing {ledger.departure_coldest_closing:.2f} km/s)",
        f"  CHAIN TO THE TRANSFER          {ledger.chain_to_departure:.4f}"
        "   (the free parking orbit called this 1)",
        f"  node survival                  {ledger.node_survival:.4f}",
        "",
        f"  growth per cycle               {ledger.split_growth:.2f} kg per impactor kg",
        f"  doubling / millionfold         {ledger.split_doubling_years:.4f} yr"
        f" / {ledger.split_doubling_years * np.log(1.0e6) / np.log(2.0):.2f} yr",
        "",
        f"  LAUNCH LEDGER  returned per pad kg {launch.returned_per_pad_kg:.4f}",
        f"    vs committed 1/15            {launch.stated_margin:.3f}x"
        f"   {'CLEARS' if launch.clears_committed_floor else 'FAILS'}",
        f"    vs rescaled 1/{1 / launch.rescaled_floor:<25.1f}"
        f"{launch.rescaled_margin:.3f}x"
        f"   {'CLEARS' if launch.clears_rescaled_floor else 'FAILS'}"
        f"   (a {launch.return_value_ratio:.2f}x more valuable returned kg)",
        f"    time-normalised              {launch.returned_per_pad_kg_per_year:.4f}"
        " kg per pad kg per year",
        f"    free-parking reading was     "
        f"{launch.free_parking_returned_per_pad_kg:.4f}"
        f" ({launch.free_parking_margin:.3f}x)"
        f"   -- charging the lob costs {1.0 - launch.correction_factor:.1%}",
    ]
    if ceilings is not None:
        lines += [
            "",
            f"  SLUG RATIOS ADMITTED (expansion margin {ceilings.expansion_margin:g}x)",
            f"    expansion, overtaking leg    "
            f"{_format_window(ceilings.overtaking_expansion_window)}"
            f"   at {ceilings.overtaking_coldest_closing:.2f} km/s",
            f"    expansion, departure leg     "
            f"{_format_window(ceilings.departure_expansion_window)}"
            f"   at {ceilings.departure_coldest_closing:.2f} km/s",
            f"    pad return peaks at k        {ceilings.peak_slug_ratio:.3f}"
            f"   ({ceilings.peak_margin:.3f}x the committed 1/15 floor)",
            f"    committed floor cleared for  "
            f"{_format_window(ceilings.launch_window)}",
            f"    rescaled floor cleared for   "
            f"{_format_window(ceilings.rescaled_window)}",
            f"    ADMISSIBLE                   "
            f"{_format_window(ceilings.admissible_window)}",
            f"    BINDING                      {ceilings.binding}",
        ]
    return "\n".join(lines)


def _format_window(window: Optional[Tuple[float, float]]) -> str:
    """Render an optional slug-ratio window for a report line.

    Args:
        window: The interval, or None when it is empty.

    Returns:
        A short cell string.
    """
    return "none" if window is None else f"[{window[0]:.3f}, {window[1]:.3f}]"


def split_push_depth_text(
    rows: Sequence[Tuple[float, SplitPushLedger, SplitPushLaunchLedger]],
) -> str:
    """Render a pad-charged depth sweep as a table.

    Args:
        rows: ``(depth, ledger, launch)`` triples.

    Returns:
        The formatted table.
    """
    return str(
        tabulate(
            [
                [
                    f"{depth:g}",
                    f"{row.node_survival:.4f}",
                    f"{row.cant_deg:.1f}",
                    f"{row.overtaking_fraction:.4f}",
                    f"{row.departure_fraction:.4f}",
                    f"{row.chain_to_departure:.4f}",
                    f"{row.split_growth:.2f}",
                    f"{row.split_doubling_years:.3f}",
                    f"{launch.returned_per_pad_kg:.4f}",
                    f"{launch.stated_margin:.2f}",
                    f"{launch.rescaled_margin:.2f}",
                ]
                for depth, row, launch in rows
            ],
            headers=[
                "R_sun",
                "survive",
                "cant",
                "f1",
                "f2",
                "chain",
                "kg/imp",
                "dbl yr",
                "kg/pad kg",
                "1/15 x",
                "rescaled x",
            ],
            tablefmt="github",
        )
    )


def split_push_slug_ratio_text(
    entries: Sequence[Tuple[float, SplitPushLedger, SplitPushLaunchLedger]],
) -> str:
    """Render a pad-charged slug-ratio sweep as a table.

    Args:
        entries: ``(slug_ratio, ledger, launch)`` triples.

    Returns:
        The formatted table.
    """
    return str(
        tabulate(
            [
                [
                    f"{ratio:g}",
                    f"{row.overtaking_fraction:.4f}",
                    f"{row.departure_fraction:.4f}",
                    f"{row.chain_to_departure:.4f}",
                    f"{row.split_growth:.2f}",
                    f"{row.split_doubling_years:.4f}",
                    f"{launch.returned_per_pad_kg:.4f}",
                    f"{launch.stated_margin:.2f}",
                    f"{launch.rescaled_margin:.2f}",
                ]
                for ratio, row, launch in entries
            ],
            headers=[
                "k",
                "f1",
                "f2",
                "chain",
                "kg/imp",
                "dbl yr",
                "kg/pad kg",
                "1/15 x",
                "rescaled x",
            ],
            tablefmt="github",
        )
    )


def conduction_bracket_text(rows: Sequence[ConductionBracketRow]) -> str:
    """Render a **conduction reserve** sweep of the pad verdict as a table.

    Args:
        rows: The bracket rows.

    Returns:
        The formatted table.
    """
    return str(
        tabulate(
            [
                [
                    r.reserve_label,
                    f"{r.reserve:.2f}",
                    f"{r.expansion_margin:g}",
                    (
                        "--"
                        if r.threshold_closing_speed is None
                        else f"{r.threshold_closing_speed:.2f}"
                    ),
                    (
                        "--"
                        if r.threshold_return_excess is None
                        else f"{r.threshold_return_excess:.2f}"
                    ),
                    (
                        "--"
                        if r.best_return_excess is None
                        else f"{r.best_return_excess:.2f}"
                    ),
                    "--" if r.best_slug_ratio is None else f"{r.best_slug_ratio:.2f}",
                    (
                        "--"
                        if r.best_returned_per_pad_kg is None
                        else f"{r.best_returned_per_pad_kg:.4f}"
                    ),
                    "--" if r.best_margin is None else f"{r.best_margin:.3f}",
                    (
                        "--"
                        if r.best_doubling_years is None
                        else f"{r.best_doubling_years:.3f}"
                    ),
                    "CLEARS" if r.clears_committed_floor else "fails",
                ]
                for r in rows
            ],
            headers=[
                "conduction reserve",
                "MJ/kg",
                "margin",
                "dies below",
                "slowest v_inf",
                "best v_inf",
                "k",
                "kg/pad kg",
                "1/15 x",
                "dbl yr",
                "",
            ],
            tablefmt="github",
        )
    )


def pad_frontier_text(rows: Sequence[PadFrontierRow]) -> str:
    """Render a climb-out scan of the pad ledger as a table.

    Args:
        rows: The frontier rows.

    Returns:
        The formatted table.
    """
    return str(
        tabulate(
            [
                [
                    f"{r.return_excess:g}",
                    f"{r.earth_closing_speed:.2f}",
                    f"{r.overtaking_coldest_closing:.2f}",
                    f"{r.node_survival:.4f}",
                    _format_window(r.expansion_window),
                    "--" if r.best_slug_ratio is None else f"{r.best_slug_ratio:.2f}",
                    (
                        "--"
                        if r.best_returned_per_pad_kg is None
                        else f"{r.best_returned_per_pad_kg:.4f}"
                    ),
                    "--" if r.best_margin is None else f"{r.best_margin:.2f}",
                    ("--" if r.doubling_years is None else f"{r.doubling_years:.3f}"),
                    (
                        "--"
                        if r.growth_slug_ratio is None
                        else f"{r.growth_slug_ratio:.2f}"
                    ),
                    (
                        "--"
                        if r.growth_doubling_years is None
                        else f"{r.growth_doubling_years:.3f}"
                    ),
                    "--" if r.growth_margin is None else f"{r.growth_margin:.2f}",
                ]
                for r in rows
            ],
            headers=[
                "v_inf",
                "v_b",
                "coldest",
                "survive",
                "expansion window",
                "best k",
                "kg/pad kg",
                "1/15 x",
                "dbl yr",
                "fastest k",
                "its dbl",
                "its 1/15 x",
            ],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
