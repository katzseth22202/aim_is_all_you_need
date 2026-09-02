"""Charge the opposing stream's placement, the dive node's second arrival.

The solar **dive node** is a head-on collision, so *two* things have to reach the
same perihelion at the same instant: the payload falling in prograde, and the
**opposing stream** arriving retrograde to meet it.  Every ledger in this
repository charges the first and none charges the second.
:func:`jovian_solar_dive_cycle.cycle_growth_ledger` takes the payload's
departure, the stream it eats, the clock and the node's survival; the two fields
that know the opposing stream exists --
:attr:`solar_dive_depth_trade.DiveNode.opposing_impactor_fraction` and
:attr:`solar_dive_depth_trade.OpposingStreamPlacement.retrograde_rides_the_cycle`
-- are read only by print statements and tests.

This module prices that second arrival (ADR
``0024-the-dive-node-s-second-arrival-is-cheap.md``).  Three results:

* **The Jovian route is flyable, and co-location is nearly free.**  A one-way
  tangential launch reaches the **dive-placement floor** at Jupiter, an unpowered
  flyby bends it retrograde with 1.6x the turn it needs, and it falls to the
  node.  Making it arrive at the payload's perihelion *at the payload's instant*
  costs **0.12 to 0.32 percent** over the floor -- 13 to 38 m/s -- because
  arrival excess above the floor sweeps the perihelion-longitude gap through a
  full 360 degrees over about 50 m/s, so a root always exists and the solutions
  are discrete, exactly as the **synodic closure** is
  (:func:`opposing_placement_route`).
* **The charge is a rounding error, in both currencies.**  Folded into the growth
  ledger it costs 0.1-1.1 percent of doubling time; folded into ADR 0021's
  pad-charged ledger it costs 0.45-2.00 percent of returned mass and flips no
  verdict (:func:`opposing_charge`).  Neither reading dominates: the two charges
  stand in the exact ratio ``(1 - delivered) / (1 - placed)``, so the pad reading
  is the stricter one only where the placement burn delivers a *smaller* fraction
  than the payload's own departure -- true for the paper's dive at every depth
  and for the shallow Jovian rows, false for the deep ones.  The reason is mass, not impulse: the node's slug
  ratio of 30 means one impactor kilogram vaporises 30 kg of slug, so the
  opposing stream is only 0.17-0.49 kg per impactor kilogram of payload.
* **The bi-elliptic architecture phases itself exactly, and Jupiter is not
  available to it.**  Injected at one far node, payload and opposing stream fly
  *the same ellipse* in opposite senses: same semi-major axis, so the same
  half-period, and an ellipse's perihelion is antipodal to its aphelion whatever
  direction it is flown, so both perihelia are the same point.  They arrive
  together, head-on, with **no tuning knob at all**, and neither leg ever goes
  closer to the Sun than the node (:func:`bielliptic_coplacement`).

The last point matters for what this module does *not* settle.  The split dive of
:mod:`src.bielliptic_dive_split` is Earth-only -- it raises aphelion and cancels
velocity out there for both arrivals -- so the cheap Jupiter route is not open to
it, and both of its far-node burns need impactors delivered to ~1.96 AU.  That
delivery is unpriced (worklist item S1 in the paper repository), so the split's
own opposing-stream charge cannot be closed here and is not reported.

Model is circular and coplanar and all algebra is float (km, s, km/s, rad), in
the same spirit as :mod:`src.jovian_solar_dive_cycle`, whose private leg
primitives ``_outbound_leg`` and ``_dive_leg`` this module reuses rather than
re-deriving.

Run with ``make opposing-stream``.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from .bielliptic_dive_split import _MU_SUN, _SOLAR_RADIUS
from .jovian_solar_dive_cycle import (
    DEFAULT_PERIAPSIS_SURVIVAL,
    DEFAULT_SLUG_RATIO,
    CycleGrowthLedger,
    SynodicCycleClosure,
    _dive_leg,
    _dive_periapsis_radius,
    _FlybyParams,
    _outbound_leg,
    _powered_flyby_params,
    departure_nozzle_ledger,
    paper_resonant_dive_ledger,
    solve_synodic_closure,
)
from .solar_dive_depth_trade import (
    CONSERVATIVE_RETURN_EXCESS,
    CONSERVATIVE_SLUG_RATIO,
    DEFAULT_JET_ENERGY_EFFICIENCY,
    DEFAULT_PERIAPSIS_SLUG_RATIO,
    depth_trade_row,
    direct_retrograde_placement_excess,
    opposing_stream_placement,
)
from .two_leg_nozzle_sweep import PAYLOAD_FRACTION_AT_INTERCEPT, RETURN_FLOOR

_YEAR_SECONDS = 365.25 * 86400.0
_AU = 1.495978707e8

#: Depths swept by the command-line tables (solar radii).  Spans ADR 0020's dial
#: from the paper's 4 to the shallow end ADR 0022 recommends starting at.
DEPTH_GRID: Sequence[float] = (4.0, 8.0, 16.0, 23.0, 32.0)

#: The **dive-placement floor** is a *tangency*: at exactly the floor the
#: post-flyby state is purely tangential at Jupiter, so ``_solve_dive_state``
#: sits on the edge of its bracket with ``v_radial = 0`` and returns None on
#: floating-point luck (it does so at 23 solar radii and not at 4, 8 or 32).
#: Every leg built here is therefore nudged this far above the floor.  The
#: nudge is 0.1 m/s and moves no reported figure.
TANGENCY_NUDGE = 1.0e-4

#: Bracket for the co-location root, as km/s of departure excess *above* the
#: floor departure.  Recorded per the ADR 0007 lesson.  The perihelion-longitude
#: gap sweeps a full 360 degrees over roughly 50 m/s here -- the dive time of
#: flight moves 0.25 yr for 0.05 km/s, and the gap carries that at 360 deg/yr --
#: so this window holds exactly the *first* root above the floor at every depth
#: in ``DEPTH_GRID``.  Widening it would return a later wrap instead, which is a
#: different (and dearer) solution, not a better-converged one.
COLOCATION_BRACKET = (1.0e-4, 0.05)

#: Aim angle of the opposing stream's one-way launch, degrees from Earth's
#: prograde.  Tangential is the cheapest way to a given arrival excess, which is
#: the right yardstick for a launch whose aim is not pinned by a closure.
OPPOSING_LAUNCH_AIM_DEG = 0.0


def _wrap_signed_degrees(angle: float) -> float:
    """Fold an angle into (-180, 180].

    Args:
        angle: Angle in degrees.

    Returns:
        The same angle folded into (-180, 180] degrees.
    """
    return (angle % 360.0 + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# Does the opposing stream's Jupiter route exist?
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementRoute:
    """The opposing stream's flight to the dive node, audited leg by leg.

    Attributes:
        dive_solar_radii: Perihelion distance in solar radii.
        placement_floor: Minimum Jupiter arrival excess that can place a
            retrograde dive to this perihelion (km/s).
        floor_departure: Earth departure excess whose tangential transfer
            arrives exactly at ``placement_floor`` (km/s).
        colocation_departure: Departure excess that *also* puts the stream at
            the payload's perihelion at the payload's instant (km/s).
        colocation_premium: ``colocation_departure / floor_departure``.
        payload_departure_excess: What the payload's own cycle departs at
            (km/s), for scale.
        bend_required_deg: Turn the Jovian flyby must deliver (deg).
        bend_available_deg: Turn it can deliver at the perijove floor (deg).
        feasible: Whether the unpowered flyby can deliver the required bend.
        outbound_tof_years: Earth-to-Jupiter flight time (yr).
        dive_tof_years: Jupiter-to-perihelion fall time (yr).
        lead_time_days: How much earlier the opposing stream must leave Earth
            than the payload; negative means it leaves later.
        dive_swept_deg: Heliocentric longitude swept by the dive (deg,
            negative because the dive is retrograde).
    """

    dive_solar_radii: float
    placement_floor: float
    floor_departure: float
    colocation_departure: float
    colocation_premium: float
    payload_departure_excess: float
    bend_required_deg: float
    bend_available_deg: float
    feasible: bool
    outbound_tof_years: float
    dive_tof_years: float
    lead_time_days: float
    dive_swept_deg: float


def _payload_arrival(
    dive_solar_radii: float,
    closure: SynodicCycleClosure,
    params: _FlybyParams,
) -> Optional[tuple[float, float]]:
    """When and where the payload reaches its dive perihelion.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        closure: The payload's solved **synodic closure**.
        params: Float parameter block.

    Returns:
        (time from Earth departure in yr, heliocentric longitude swept in deg),
        or None if the payload's own legs do not rebuild.
    """
    radius = _dive_periapsis_radius(dive_solar_radii)
    outbound = _outbound_leg(
        closure.departure_excess, closure.departure_aim_deg, params
    )
    if outbound is None:
        return None
    dive = _dive_leg(outbound, 1.0, params, radius)
    if dive is None:
        return None
    return (
        (outbound.tof + dive.tof) / _YEAR_SECONDS,
        float(np.degrees(outbound.swept + dive.swept)),
    )


def _colocation_gap(
    departure_excess: float,
    dive_radius: float,
    payload_years: float,
    payload_swept_deg: float,
    params: _FlybyParams,
) -> Optional[tuple[float, float, float, float, float]]:
    """Perihelion-longitude gap between the two arrivals, at one departure.

    Both vehicles leave Earth, so the opposing stream's launch is offset by
    whatever makes the two arrivals simultaneous; Earth has moved 360 deg/yr in
    the meantime, and the residual is the gap driven to zero here.

    Args:
        departure_excess: Opposing stream's Earth departure excess (km/s).
        dive_radius: Target perihelion radius (km).
        payload_years: Payload's Earth-to-perihelion time (yr).
        payload_swept_deg: Longitude the payload sweeps getting there (deg).
        params: Float parameter block.

    Returns:
        (gap in deg, outbound tof yr, dive tof yr, dive swept deg, bend margin
        deg), or None when no retrograde dive state exists at this excess.
    """
    outbound = _outbound_leg(departure_excess, OPPOSING_LAUNCH_AIM_DEG, params)
    if outbound is None:
        return None
    dive = _dive_leg(outbound, -1.0, params, dive_radius)
    if dive is None:
        return None
    years = (outbound.tof + dive.tof) / _YEAR_SECONDS
    swept = float(np.degrees(outbound.swept + dive.swept))
    gap = _wrap_signed_degrees(
        payload_swept_deg - swept - 360.0 * (payload_years - years)
    )
    return (
        gap,
        outbound.tof / _YEAR_SECONDS,
        dive.tof / _YEAR_SECONDS,
        swept,
        float(np.degrees(dive.bend_available - dive.bend_required)),
    )


def opposing_placement_route(
    dive_solar_radii: float,
    closure: SynodicCycleClosure,
    params: Optional[_FlybyParams] = None,
) -> Optional[PlacementRoute]:
    """Audit the opposing stream's Earth-Jupiter-dive route, and co-locate it.

    Five things have to hold for the second arrival to exist: a tangential
    launch, an arrival at or above the **dive-placement floor**, an unpowered
    bend onto the retrograde dive, the fall to perihelion, and co-location with
    the payload in both longitude and time.  The first four are geometry; the
    fifth is a root solve on the departure excess, which is the only knob that
    is not already spent.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        closure: The payload's solved **synodic closure**, whose perihelion the
            opposing stream has to meet.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`PlacementRoute`, or None when the payload's legs do not
        rebuild or no co-locating root exists in :data:`COLOCATION_BRACKET`.
    """
    p = params if params is not None else _powered_flyby_params()
    radius = _dive_periapsis_radius(dive_solar_radii)
    arrival = _payload_arrival(dive_solar_radii, closure, p)
    if arrival is None:
        return None
    payload_years, payload_swept = arrival
    placement = opposing_stream_placement(dive_solar_radii, closure, p)
    floor_departure = placement.retrograde_tangential_departure

    def gap(extra: float) -> float:
        probe = _colocation_gap(
            floor_departure + extra, radius, payload_years, payload_swept, p
        )
        return float("nan") if probe is None else probe[0]

    low, high = COLOCATION_BRACKET
    if not np.isfinite(gap(low)) or not np.isfinite(gap(high)):
        return None
    if gap(low) * gap(high) > 0.0:
        return None
    extra = float(brentq(gap, low, high, xtol=1e-12))
    solved = _colocation_gap(
        floor_departure + extra, radius, payload_years, payload_swept, p
    )
    if solved is None:
        return None
    _, outbound_years, dive_years, dive_swept, bend_margin = solved
    nudged = _outbound_leg(
        floor_departure + max(extra, TANGENCY_NUDGE), OPPOSING_LAUNCH_AIM_DEG, p
    )
    assert nudged is not None
    leg = _dive_leg(nudged, -1.0, p, radius)
    assert leg is not None
    return PlacementRoute(
        dive_solar_radii=dive_solar_radii,
        placement_floor=placement.retrograde_floor,
        floor_departure=floor_departure,
        colocation_departure=floor_departure + extra,
        colocation_premium=(floor_departure + extra) / floor_departure,
        payload_departure_excess=closure.departure_excess,
        bend_required_deg=float(np.degrees(leg.bend_required)),
        bend_available_deg=float(np.degrees(leg.bend_available)),
        feasible=bool(leg.bend_required <= leg.bend_available),
        outbound_tof_years=outbound_years,
        dive_tof_years=dive_years,
        lead_time_days=(outbound_years + dive_years - payload_years) * 365.25,
        dive_swept_deg=dive_swept,
    )


# --------------------------------------------------------------------------
# What the second arrival costs, in both currencies
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OpposingCharge:
    """Placing the opposing stream, charged in the growth and pad ledgers.

    The two ledgers divide differently and that is the point of reporting both.
    The growth ledger's currency is the impactor kilogram, so the opposing
    stream's mass is discounted by the departure **slug ratio** on its way in.
    ADR 0021's pad ledger counts raw kilograms off the pad and applies no such
    discount.  They are not ordered: ``pad_charge / growth_charge`` is exactly
    ``(1 - delivered) / (1 - placed)``, so whichever burn wastes more of its
    vehicle sets which reading bites harder.

    Attributes:
        label: Name of the cycle charged.
        dive_solar_radii: Perihelion distance in solar radii.
        placement_excess: Earth departure excess placing the opposing stream
            (km/s).
        placement_delivered_fraction: Mass fraction surviving that departure.
        opposing_kg_per_impactor_kg: Opposing projectile kilograms the node
            consumes, per impactor kilogram spent on the payload.
        growth_charge: Extra impactor kilograms per impactor kilogram.
        growth_uncharged: Returning mass per impactor kilogram, as published.
        growth_charged: The same with the placement charged.
        doubling_uncharged: Fleet doubling time as published (yr).
        doubling_charged: The same with the placement charged (yr).
        doubling_penalty: ``doubling_charged / doubling_uncharged``.
        pad_charge: Extra pad kilograms per pad kilogram.
        returned_per_pad_uncharged: Returned mass per kilogram off the pad.
        returned_per_pad_charged: The same with the placement charged.
        pad_margin_uncharged: That against ADR 0021's 1/15 floor.
        pad_margin_charged: The same with the placement charged.
        pad_verdict_flips: Whether charging moves the cycle across the floor.
    """

    label: str
    dive_solar_radii: float
    placement_excess: float
    placement_delivered_fraction: float
    opposing_kg_per_impactor_kg: float
    growth_charge: float
    growth_uncharged: float
    growth_charged: float
    doubling_uncharged: float
    doubling_charged: float
    doubling_penalty: float
    pad_charge: float
    returned_per_pad_uncharged: float
    returned_per_pad_charged: float
    pad_margin_uncharged: float
    pad_margin_charged: float
    pad_verdict_flips: bool


def opposing_charge(
    label: str,
    dive_solar_radii: float,
    growth: CycleGrowthLedger,
    placement_excess: float,
    placement_aim_deg: float,
    stream_excess: float,
    stream_axis_deg: float,
    node_survival: float,
    slug_ratio: float,
    periapsis_slug_ratio: float = DEFAULT_PERIAPSIS_SLUG_RATIO,
    jet_energy_efficiency: float = DEFAULT_JET_ENERGY_EFFICIENCY,
    params: Optional[_FlybyParams] = None,
) -> OpposingCharge:
    """Charge a cycle for placing the second arrival at its dive node.

    The node consumes ``(1 - survival) / periapsis_slug_ratio`` kilograms of
    opposing projectile per kilogram of vehicle it processes, because the vehicle
    loses ``1 - survival`` kilograms of slug and each impactor kilogram vaporises
    ``periapsis_slug_ratio`` of it.  Those projectiles are pushed to
    ``placement_excess`` on the same nozzle the payload's departure uses, so the
    charge is that mass divided by what one impactor kilogram delivers.

    Args:
        label: Name for the row.
        dive_solar_radii: Perihelion distance in solar radii.
        growth: The cycle to charge, already scored without the placement.
        placement_excess: Earth departure excess placing the stream (km/s).
        placement_aim_deg: Direction that excess must point, from Earth's
            prograde; 0 for a tangential Jovian launch, 180 for a direct
            retrograde placement from 1 AU.
        stream_excess: Earth-relative excess of the stream that pushes it
            (km/s) -- the same returning beam the payload's departure eats.
        stream_axis_deg: Direction that stream travels, from Earth's prograde.
        node_survival: Mass fraction surviving the node.
        slug_ratio: Departure slug ratio the cycle was scored at.
        periapsis_slug_ratio: Slug ratio at the dive node.
        jet_energy_efficiency: The paper's ``eta_jet**2``, in (0, 1].
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`OpposingCharge`.
    """
    p = params if params is not None else _powered_flyby_params()
    placement = departure_nozzle_ledger(
        placement_excess,
        placement_aim_deg,
        stream_excess,
        stream_axis_deg,
        slug_ratio,
        jet_energy_efficiency,
        p,
    )
    delivered = growth.nozzle.delivered_fraction
    placed = placement.delivered_fraction
    vehicle_per_impactor = slug_ratio * delivered / (1.0 - delivered)
    opposing_fraction = (1.0 - node_survival) / periapsis_slug_ratio
    opposing_mass = vehicle_per_impactor * opposing_fraction

    # Growth ledger: the currency is the impactor kilogram, and one of them
    # delivers ``slug_ratio * placed / (1 - placed)`` kilograms through the
    # placement burn, so the charge inverts that.
    growth_charge = opposing_mass * (1.0 - placed) / (slug_ratio * placed)
    charged_growth = growth.return_per_impactor_kg / (1.0 + growth_charge)
    doubling_charged = (
        growth.cycle_years * np.log(2.0) / np.log(charged_growth)
        if charged_growth > 1.0
        else float("inf")
    )

    # Pad ledger: the currency is the kilogram off the pad.  Both streams pay
    # the same lob to the intercept point, so that factor cancels and no
    # slug-ratio discount survives.  Against the growth charge this lands at
    # exactly (1 - delivered) / (1 - placed), which is above or below one
    # depending on which of the two burns is the more wasteful.
    pad_charge = delivered * (1.0 - node_survival) / (periapsis_slug_ratio * placed)
    returned = PAYLOAD_FRACTION_AT_INTERCEPT * growth.round_trip_fraction
    returned_charged = returned / (1.0 + pad_charge)
    return OpposingCharge(
        label=label,
        dive_solar_radii=dive_solar_radii,
        placement_excess=placement_excess,
        placement_delivered_fraction=placed,
        opposing_kg_per_impactor_kg=opposing_mass,
        growth_charge=growth_charge,
        growth_uncharged=growth.return_per_impactor_kg,
        growth_charged=charged_growth,
        doubling_uncharged=growth.doubling_years,
        doubling_charged=doubling_charged,
        doubling_penalty=doubling_charged / growth.doubling_years,
        pad_charge=pad_charge,
        returned_per_pad_uncharged=returned,
        returned_per_pad_charged=returned_charged,
        pad_margin_uncharged=returned / RETURN_FLOOR,
        pad_margin_charged=returned_charged / RETURN_FLOOR,
        pad_verdict_flips=(returned >= RETURN_FLOOR)
        != (returned_charged >= RETURN_FLOOR),
    )


def jovian_cycle_charge(
    dive_solar_radii: float,
    return_excess: float = CONSERVATIVE_RETURN_EXCESS,
    slug_ratio: float = CONSERVATIVE_SLUG_RATIO,
    params: Optional[_FlybyParams] = None,
) -> Optional[OpposingCharge]:
    """Charge the Jovian dive cycle, which has Jupiter to place its stream with.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        return_excess: Solar hyperbolic-excess speed of the climb-out (km/s).
        slug_ratio: Departure slug ratio.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`OpposingCharge`, or None when no closure exists at this
        depth.
    """
    p = params if params is not None else _powered_flyby_params()
    row = depth_trade_row(
        dive_solar_radii=dive_solar_radii,
        return_excess=return_excess,
        slug_ratio=slug_ratio,
        params=p,
    )
    if row is None:
        return None
    route = opposing_placement_route(dive_solar_radii, row.closure, p)
    excess = (
        route.colocation_departure
        if route is not None
        else opposing_stream_placement(
            dive_solar_radii, row.closure, p
        ).retrograde_tangential_departure
    )
    return opposing_charge(
        label=f"Jovian 3S / {dive_solar_radii:g} R_sun",
        dive_solar_radii=dive_solar_radii,
        growth=row.growth,
        placement_excess=excess,
        placement_aim_deg=OPPOSING_LAUNCH_AIM_DEG,
        stream_excess=row.closure.return_excess,
        stream_axis_deg=row.closure.push_axis_deg,
        node_survival=row.node.survival,
        slug_ratio=slug_ratio,
        params=p,
    )


def direct_dive_charge(
    dive_solar_radii: float,
    stream_excess: float = 158.50,
    stream_axis_deg: float = 98.48,
    slug_ratio: float = DEFAULT_SLUG_RATIO,
    node_survival: float = DEFAULT_PERIAPSIS_SURVIVAL,
    params: Optional[_FlybyParams] = None,
) -> OpposingCharge:
    """Charge the paper's single-impulse resonant dive, which has no Jupiter.

    Its opposing stream must be placed retrograde from 1 AU, which is the
    dearest of the routes here -- 35.48 km/s at 4 solar radii rising to 44.94 at
    32, against the Jovian cycle's ~11.5 flat.

    Args:
        dive_solar_radii: Perihelion distance in solar radii.
        stream_excess: Earth-relative excess of the arriving stream (km/s).
        stream_axis_deg: Direction the stream travels, from Earth's prograde.
        slug_ratio: Departure slug ratio.
        node_survival: Mass fraction surviving the node.
        params: Float parameter block; built with defaults when omitted.

    Returns:
        The :class:`OpposingCharge`.
    """
    p = params if params is not None else _powered_flyby_params()
    growth = paper_resonant_dive_ledger(
        stream_excess,
        stream_axis_deg,
        slug_ratio,
        DEFAULT_JET_ENERGY_EFFICIENCY,
        node_survival,
        p,
    )
    return opposing_charge(
        label=f"paper dive / {dive_solar_radii:g} R_sun",
        dive_solar_radii=dive_solar_radii,
        growth=growth,
        placement_excess=direct_retrograde_placement_excess(dive_solar_radii, p),
        placement_aim_deg=180.0,
        stream_excess=stream_excess,
        stream_axis_deg=stream_axis_deg,
        node_survival=node_survival,
        slug_ratio=slug_ratio,
        params=p,
    )


# --------------------------------------------------------------------------
# The bi-elliptic node, which phases itself
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BiellipticCoplacement:
    """Both arrivals injected at one far node, on the same ellipse.

    This is the geometry the split dive already flies, read for its *phasing*
    rather than its delta-v.  The payload cuts its tangential speed to drop
    perihelion and stays prograde; the opposing stream reverses the same speed
    and flies the identical ellipse retrograde.  Identical ellipse means
    identical semi-major axis, hence identical half-period, and an ellipse's
    perihelion is antipodal to its aphelion regardless of travel sense -- so both
    reach the same point at the same instant, moving exactly opposite.  There is
    no knob to tune and no residual to drive to zero.

    Attributes:
        node_radius_au: Radius of the shared far node (AU).
        dive_solar_radii: Target perihelion in solar radii.
        semi_major_axis_au: Semi-major axis of the shared dive ellipse (AU).
        dive_tof_years: Far node to perihelion, the same for both senses (yr).
        arriving_speed: Tangential speed at the far node on the raise ellipse
            (km/s).
        payload_cut: Tangential speed the payload sheds to drop perihelion
            (km/s).
        opposing_flip: Tangential speed the opposing stream reverses (km/s).
        closest_approach_solar_radii: Minimum heliocentric radius either vehicle
            reaches, which is the node depth itself -- nothing goes deeper, so
            the shallow dive's reason for existing survives.
        head_on_degrees: Angle between the two arrivals at perihelion.
    """

    node_radius_au: float
    dive_solar_radii: float
    semi_major_axis_au: float
    dive_tof_years: float
    arriving_speed: float
    payload_cut: float
    opposing_flip: float
    closest_approach_solar_radii: float
    head_on_degrees: float


def bielliptic_coplacement(
    dive_solar_radii: float, node_radius_au: float = 1.9649
) -> BiellipticCoplacement:
    """Phase both arrivals off a single far node, exactly and for free.

    Args:
        dive_solar_radii: Target perihelion in solar radii.
        node_radius_au: Radius of the shared far node (AU).

    Returns:
        The :class:`BiellipticCoplacement`.
    """
    node = node_radius_au * _AU
    perihelion = dive_solar_radii * _SOLAR_RADIUS
    axis = 0.5 * (node + perihelion)

    def apsis_speed(inner: float, outer: float, radius: float) -> float:
        return float(np.sqrt(_MU_SUN * (2.0 / radius - 2.0 / (inner + outer))))

    arriving = apsis_speed(_AU, node, node)
    leaving = apsis_speed(perihelion, node, node)
    return BiellipticCoplacement(
        node_radius_au=node_radius_au,
        dive_solar_radii=dive_solar_radii,
        semi_major_axis_au=axis / _AU,
        dive_tof_years=float(np.pi * np.sqrt(axis**3 / _MU_SUN) / _YEAR_SECONDS),
        arriving_speed=arriving,
        payload_cut=arriving - leaving,
        opposing_flip=arriving + leaving,
        closest_approach_solar_radii=dive_solar_radii,
        head_on_degrees=180.0,
    )


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _route_table(depths: Sequence[float] = DEPTH_GRID) -> str:
    """The Jovian placement route, audited at each depth.

    Args:
        depths: Dive depths to sweep (solar radii).

    Returns:
        A markdown-ish fixed-width table.
    """
    p = _powered_flyby_params()
    lines = [
        f"{'depth':>7} {'floor dep':>10} {'co-loc dep':>11} {'premium':>8} "
        f"{'payload dep':>12} {'bend req':>9} {'bend avl':>9} {"earlier d":>9}",
        "-" * 78,
    ]
    for depth in depths:
        closure = solve_synodic_closure(3, p, depth, CONSERVATIVE_RETURN_EXCESS, 1.0)
        if closure is None:
            lines.append(f"{depth:7.1f}   no closure")
            continue
        route = opposing_placement_route(depth, closure, p)
        if route is None:
            lines.append(f"{depth:7.1f}   no co-locating root in the bracket")
            continue
        lines.append(
            f"{depth:7.1f} {route.floor_departure:10.3f} "
            f"{route.colocation_departure:11.3f} "
            f"{100.0 * (route.colocation_premium - 1.0):7.2f}% "
            f"{route.payload_departure_excess:12.2f} "
            f"{route.bend_required_deg:9.1f} {route.bend_available_deg:9.1f} "
            f"{route.lead_time_days:8.1f}"
        )
    return "\n".join(lines)


def _charge_table(depths: Sequence[float] = DEPTH_GRID) -> str:
    """Both architectures, charged in both ledgers.

    Args:
        depths: Dive depths to sweep (solar radii).

    Returns:
        A markdown-ish fixed-width table.
    """
    lines = [
        f"{'architecture':>20} {'depth':>6} {'place':>7} {'growth x':>9} {'doubling':>17} "
        f"{'pad margin':>18} {'flips':>6}",
        "-" * 88,
    ]
    for depth in depths:
        row = jovian_cycle_charge(depth)
        if row is None:
            continue
        lines.append(
            f"{'Jovian 3S':>20} {depth:6.1f} {row.placement_excess:7.2f} "
            f"{1.0 / (1.0 + row.growth_charge):9.4f} "
            f"{row.doubling_uncharged:7.4f} -> {row.doubling_charged:6.4f} "
            f"{row.pad_margin_uncharged:8.3f} -> {row.pad_margin_charged:6.3f} "
            f"{str(row.pad_verdict_flips):>6}"
        )
    for depth in depths:
        row = direct_dive_charge(depth)
        lines.append(
            f"{'paper single-impulse':>17} {depth:6.1f} "
            f"{row.placement_excess:7.2f} "
            f"{1.0 / (1.0 + row.growth_charge):9.4f} "
            f"{row.doubling_uncharged:7.4f} -> {row.doubling_charged:6.4f} "
            f"{row.pad_margin_uncharged:8.3f} -> {row.pad_margin_charged:6.3f} "
            f"{str(row.pad_verdict_flips):>6}"
        )
    return "\n".join(lines)


def _coplacement_table(depths: Sequence[float] = DEPTH_GRID) -> str:
    """The bi-elliptic far node, which needs no phasing knob.

    Args:
        depths: Dive depths to sweep (solar radii).

    Returns:
        A markdown-ish fixed-width table.
    """
    lines = [
        f"{'far node':>9} {'depth':>7} {'a (AU)':>9} {'both TOF':>10} "
        f"{'cut':>8} {'flip':>8} {'closest':>9} {'angle':>7}",
        "-" * 74,
    ]
    for node in (1.9649, 3.0):
        for depth in depths:
            row = bielliptic_coplacement(depth, node)
            lines.append(
                f"{row.node_radius_au:9.4f} {depth:7.1f} "
                f"{row.semi_major_axis_au:9.5f} {row.dive_tof_years:10.6f} "
                f"{row.payload_cut:8.3f} {row.opposing_flip:8.3f} "
                f"{row.closest_approach_solar_radii:7.1f} R "
                f"{row.head_on_degrees:6.1f}"
            )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Price the dive node's second arrival: the opposing stream nobody "
            "charged for placing."
        )
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=list(DEPTH_GRID),
        help="dive depths to sweep, in solar radii",
    )
    return parser


def main() -> None:
    """Print the placement audit, the charge, and the bi-elliptic phasing."""
    args = _parser().parse_args()
    depths: List[float] = list(args.depths)

    print("The dive node needs TWO arrivals. Only one has ever been charged.\n")
    print(
        "1. Does the opposing stream's Jupiter route exist, and can it be\n"
        "   co-located with the payload in both longitude and time?\n"
    )
    print(_route_table(depths))
    print(
        "\n   Co-location is bought with departure excess above the placement\n"
        "   floor, which sweeps the perihelion-longitude gap through a full\n"
        "   360 degrees over ~50 m/s. A root therefore always exists, and the\n"
        "   solutions are discrete -- the same shape as the synodic closure."
    )

    print("\n\n2. What does placing it cost, in each ledger?\n")
    print(_charge_table(depths))
    print(
        "\n   The pad ledger is the stricter reading: it counts raw kilograms\n"
        "   off the pad, with no slug-ratio discount. It still flips no\n"
        "   verdict. The charge is small because of mass, not impulse -- the\n"
        f"   node's slug ratio of {DEFAULT_PERIAPSIS_SLUG_RATIO:g} means the "
        "opposing stream is well under a\n   kilogram per impactor kilogram of "
        "payload."
    )

    print("\n\n3. The bi-elliptic far node, which phases itself exactly.\n")
    print(_coplacement_table(depths))
    print(
        "\n   Both vehicles fly the SAME ellipse in opposite senses, so the\n"
        "   times of flight are identical and both perihelia are antipodal to\n"
        "   the shared far node -- the same point. No knob, no residual.\n"
        "   Closest approach is the node depth for both, so nothing goes\n"
        "   deeper than the dive the shallow depth was chosen to avoid.\n\n"
        "   Jupiter is NOT available to this architecture, so its own opposing\n"
        "   charge needs a price for delivering impactors to the far node.\n"
        "   That is unpriced (worklist S1) and is not reported here."
    )


if __name__ == "__main__":
    main()
