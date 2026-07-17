"""Unpowered V/E/M assist chain to the same retrograde return (CONTEXT.md
"Unpowered assist chain").

Same phasing-free, coplanar-circular patched-conic model as the powered
Jovian flyby (jovian_flyby.py), but after the LEO departure burn no propellant
is spent en route: every velocity change comes from unpowered gravity assists
at Venus, Earth, and Mars, then one unpowered Jovian bend into the retrograde
return. A fixed phasing budget (ASSIST_CHAIN_PHASING_BUDGET) is charged as
spent methalox so the mass accounting carries the cost of making real
ephemerides line up.

The search is a deterministic beam search over flyby chains. States live at
a body's orbit radius as planet-relative excess velocities; an unpowered
flyby rotates the excess velocity within the bend limit its periapsis floor
allows (it can never change the excess *speed* -- the Tisserand invariant),
and legs between bodies are conic arcs between orbit radii. Pruning is
time-bucketed: per (body, 0.2 yr time-of-flight bucket) the top states by
excess speed are kept, which keeps slow-but-closable chains alive alongside
fast high-energy ones (global top-by-speed pruning made feasibility
non-monotone in the burn). The sample counts below were calibrated so the
feasibility edge sits at ~0.29 km/s (a finer beam moves it to ~0.285 km/s,
still above the ~0.2794 km/s analytic Venus-reach floor) at ~10 s per burn
probe.
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from astropy import units as u
from boinor.bodies import Sun
from scipy.optimize import brentq

from src import conic_kernel
from src.astro_constants import (
    ASSIST_CHAIN_BURN_CANDIDATES,
    ASSIST_CHAIN_MAX_FLYBYS,
    ASSIST_CHAIN_MAX_TRIP_TIME,
    ASSIST_CHAIN_PHASING_BUDGET,
    EARTH_A,
    JUPITER_A,
    STD_FUDGE_FACTOR,
    VENUS_A,
)
from src.jovian_flyby import powered_jovian_flyby_return
from src.retrograde_return_legs import (
    _ASSIST_MIN_LEG_TIME,
    _ASSIST_ROTATION_SAMPLES,
    _assist_chain_params,
    _AssistChainParams,
    _flyby_return_leg,
    _powered_flyby_params,
    _ReturnLeg,
)

_ASSIST_DEPARTURE_AIM_SAMPLES = 181  # free-aim samples at Earth departure
_ASSIST_JOVIAN_BEND_SAMPLES = 121  # bend samples at the Jovian terminal
_ASSIST_BEAM_PER_BUCKET = 80  # states kept per (body, time bucket)
_ASSIST_BEAM_CAP_PER_BODY = 1500  # states kept per body per search depth
_ASSIST_BUCKET_WIDTH_YEARS = 0.2  # width of the pruning time buckets
_ASSIST_DEDUP_SPEED_BIN = 0.05  # km/s; velocity grid for state deduplication
_ASSIST_DEDUP_TIME_BIN_YEARS = 0.05  # yr; time grid for state deduplication
_ASSIST_JOVIAN_LEG_RESERVE_YEARS = 0.3  # yr kept free for the Jovian leg
_SECONDS_PER_YEAR = float((1.0 * u.year).to_value(u.s))


class _HeliocentricState(NamedTuple):
    """Heliocentric velocity state the Jovian-terminal search evaluates from.

    Attributes:
        v_tangential: Tangential heliocentric speed (km/s).
        v_radial: Radial-outward heliocentric speed (km/s).
        radius: Current heliocentric radius (km).
        elapsed: Chain time already spent reaching this state (s).
    """

    v_tangential: float
    v_radial: float
    radius: float
    elapsed: float


@dataclass(frozen=True)
class _ChainTerminal:
    """Float summary of the chain's Jovian leg and retrograde return.

    Attributes:
        leg_tof: Last chain body to Jupiter's orbit radius (s).
        outbound_arrival: Whether the Jovian leg arrives moving outward.
        bend_angle: Unpowered Jovian bend applied to the excess velocity (rad).
        arrival_direction: Direction of the incoming Jupiter-relative excess
            velocity in the (tangential, radial-outward) basis (rad). The bend
            is applied relative to this, so it is what bounds which outgoing
            directions -- and hence which return legs -- are reachable.
        v_infinity_jupiter: Jupiter-relative excess speed (km/s).
        return_leg: The scored retrograde return.
        total_time: Departure to first retrograde 1 AU crossing (s).
    """

    leg_tof: float
    outbound_arrival: bool
    bend_angle: float
    arrival_direction: float
    v_infinity_jupiter: float
    return_leg: _ReturnLeg
    total_time: float


@dataclass(frozen=True)
class _PoweredJovianTerminal:
    """A *powered* Jovian flyby and the retrograde return it produces.

    Attributes:
        leg_tof: Last chain body to Jupiter's orbit radius (s).
        flyby_burn: Impulsive burn at perijove (km/s).
        periapsis_radius: Perijove radius, center-based (km).
        v_infinity_in: Jupiter-relative excess speed before the burn (km/s).
        v_infinity_out: Jupiter-relative excess speed after the burn (km/s).
        turn_angle: Total split-hyperbola bend, asin(1/e_in) + asin(1/e_out)
            (rad) -- note this is a function of the *burn*, not of geometry
            alone.
        return_leg: The scored retrograde return.
        total_time: Departure to first retrograde 1 AU crossing (s).
    """

    leg_tof: float
    flyby_burn: float
    periapsis_radius: float
    v_infinity_in: float
    v_infinity_out: float
    turn_angle: float
    return_leg: _ReturnLeg
    total_time: float


def _powered_jovian_terminal(
    state: _HeliocentricState,
    periapsis_radius: float,
    flyby_burn: float,
    bend_sign: float,
    params: _AssistChainParams,
) -> Optional[_PoweredJovianTerminal]:
    """Retrograde return from a *powered* Jovian flyby.

    :func:`_jovian_terminal` flies Jupiter unpowered, where perijove radius *is*
    the bend: one knob against two demands (the return's ``v_b`` and its timing),
    which is over-determined -- the "v_b lottery". Powering the flyby adds a
    second knob and breaks it, at the price of propellant. That is why the burn
    is load-bearing here rather than an optional extra.

    The burn changes the *turn*, not only the speed. It fires at perijove, so the
    outgoing hyperbola has its own eccentricity ``e_out`` computed from the
    post-burn excess, and the total bend is ``asin(1/e_in) + asin(1/e_out)``.
    Speeding up (``e_out > e_in``) therefore *costs* bend, and slowing down buys
    it. The unpowered ``sin(delta/2) = 1/e`` does not apply across a burn.

    Args:
        state: Heliocentric velocity state to search from.
        periapsis_radius: Perijove radius, center-based (km).
        flyby_burn: Impulsive perijove burn, >= 0 (km/s).
        bend_sign: +1 rotates the excess from tangential toward radial-outward,
            -1 the other way (which side Jupiter is passed on).
        params: The assist-chain parameter block.

    Returns:
        The earliest-arriving :class:`_PoweredJovianTerminal` whose return meets
        the v_b target, or None if none does.
    """
    flyby = params.flyby
    if periapsis_radius < flyby.periapsis_floor * (1.0 - 1e-12) or flyby_burn < 0.0:
        return None
    mu_j = flyby.mu_jupiter
    best: Optional[_PoweredJovianTerminal] = None
    for leg_tof, v_t1, v_r1, _, _ in conic_kernel.conic_radius_crossings(
        flyby.mu_sun,
        state.radius,
        state.v_tangential,
        state.v_radial,
        flyby.r_jupiter_orbit,
        min_leg_time=_ASSIST_MIN_LEG_TIME,
    ):
        rel_t = v_t1 - flyby.v_jupiter_orbit
        w_in = float(np.hypot(rel_t, v_r1))
        if w_in < 1e-6:
            continue
        v_peri_in = float(np.sqrt(w_in * w_in + 2.0 * mu_j / periapsis_radius))
        v_peri_out = v_peri_in + flyby_burn
        w_out_sq = v_peri_out * v_peri_out - 2.0 * mu_j / periapsis_radius
        if w_out_sq <= 0.0:
            continue
        w_out = float(np.sqrt(w_out_sq))
        ecc_in = conic_kernel.hyperbolic_eccentricity(mu_j, periapsis_radius, w_in)
        ecc_out = conic_kernel.hyperbolic_eccentricity(mu_j, periapsis_radius, w_out)
        turn = conic_kernel.powered_bend_angle(ecc_in, ecc_out)
        angle = float(np.arctan2(v_r1, rel_t)) + bend_sign * turn
        return_leg = _flyby_return_leg(
            flyby.v_jupiter_orbit + w_out * float(np.cos(angle)),
            w_out * float(np.sin(angle)),
            flyby,
        )
        if return_leg is None:
            continue
        if return_leg.collision_speed < params.target_collision_speed:
            continue
        if return_leg.collision_speed <= flyby.v_rf:
            continue
        total_time = state.elapsed + leg_tof + return_leg.tof
        if total_time > params.max_trip_time:
            continue
        if best is None or total_time < best.total_time:
            best = _PoweredJovianTerminal(
                leg_tof=leg_tof,
                flyby_burn=flyby_burn,
                periapsis_radius=periapsis_radius,
                v_infinity_in=w_in,
                v_infinity_out=w_out,
                turn_angle=turn,
                return_leg=return_leg,
                total_time=total_time,
            )
    return best


def _jovian_terminal(
    state: _HeliocentricState, params: _AssistChainParams
) -> Optional[_ChainTerminal]:
    """Best unpowered Jovian bend into a qualifying retrograde return.

    From a heliocentric state, follows each crossing of Jupiter's orbit
    radius, scans the unpowered bend of the Jupiter-relative excess velocity
    within the periapsis-floor limit, and keeps the earliest-arriving return
    whose collision speed meets the target.

    Args:
        state: Heliocentric velocity state to search from.
        params: The assist-chain parameter block.

    Returns:
        The best :class:`_ChainTerminal`, or None if no bend qualifies.
    """
    flyby = params.flyby
    best: Optional[_ChainTerminal] = None
    for leg_tof, v_t1, v_r1, outbound, _ in conic_kernel.conic_radius_crossings(
        flyby.mu_sun,
        state.radius,
        state.v_tangential,
        state.v_radial,
        flyby.r_jupiter_orbit,
        min_leg_time=_ASSIST_MIN_LEG_TIME,
    ):
        rel_t = v_t1 - flyby.v_jupiter_orbit
        rel_r = v_r1
        w = float(np.hypot(rel_t, rel_r))
        if w < 1e-6:
            continue
        ecc = conic_kernel.hyperbolic_eccentricity(
            flyby.mu_jupiter, flyby.periapsis_floor, w
        )
        bend_limit = conic_kernel.unpowered_bend_angle(ecc)
        phi = float(np.arctan2(rel_r, rel_t))
        for bend in np.linspace(-bend_limit, bend_limit, _ASSIST_JOVIAN_BEND_SAMPLES):
            angle = phi + float(bend)
            return_leg = _flyby_return_leg(
                flyby.v_jupiter_orbit + w * float(np.cos(angle)),
                w * float(np.sin(angle)),
                flyby,
            )
            if return_leg is None:
                continue
            if return_leg.collision_speed < params.target_collision_speed:
                continue
            if return_leg.collision_speed <= flyby.v_rf:
                continue
            total_time = state.elapsed + leg_tof + return_leg.tof
            if total_time > params.max_trip_time:
                continue
            if best is None or total_time < best.total_time:
                best = _ChainTerminal(
                    leg_tof=leg_tof,
                    outbound_arrival=outbound,
                    bend_angle=float(bend),
                    arrival_direction=phi,
                    v_infinity_jupiter=w,
                    return_leg=return_leg,
                    total_time=total_time,
                )
    return best


@dataclass(frozen=True)
class _ChainDecision:
    """One recorded decision of the beam search, replayable exactly.

    Attributes:
        body_index: Body the decision is taken at (index into params.bodies).
        rotation: Unpowered rotation applied to the excess velocity (rad;
            0 for the departure state, whose aim angle is recorded separately).
        next_body_index: Body the following leg targets, or -1 when the Jovian
            leg follows instead.
        outbound_arrival: Arrival branch at the next body (unused when
            next_body_index is -1).
    """

    body_index: int
    rotation: float
    next_body_index: int
    outbound_arrival: bool


class _ChainState(NamedTuple):
    """One state of the assist-chain beam search.

    Attributes:
        body_index: Index into ``params.bodies`` of the body this state is at.
        excess_tangential: Tangential component of the body-relative excess
            velocity (km/s).
        excess_radial: Radial-outward component of the body-relative excess
            velocity (km/s).
        elapsed: Chain time already spent reaching this state (s).
        departure_aim: Free-aim angle of the departure that started this
            branch (rad); carried unchanged through every state on the branch.
        decisions: The recorded chain decisions so far, terminal rotation last.
    """

    body_index: int
    excess_tangential: float
    excess_radial: float
    elapsed: float
    departure_aim: float
    decisions: Tuple[_ChainDecision, ...]


def _assist_chain_search(
    v_infinity_earth: float, params: _AssistChainParams
) -> Optional[Tuple[float, Tuple[_ChainDecision, ...], _ChainTerminal]]:
    """Beam-search the flyby chains for the earliest qualifying return.

    Explores free-aimed departures from Earth, unpowered flyby rotations at
    Venus/Earth/Mars, and conic legs between their orbit radii, terminating
    each state through :func:`_jovian_terminal`. Deterministic: no randomness,
    stable orderings throughout.

    Args:
        v_infinity_earth: Departure hyperbolic excess speed at Earth (km/s).
        params: The assist-chain parameter block.

    Returns:
        (departure aim angle rad, recorded decisions, Jovian terminal) of the
        minimum-total-time qualifying chain, or None if none exists.
    """
    best: Optional[Tuple[float, Tuple[_ChainDecision, ...], _ChainTerminal]] = None
    states: List[_ChainState] = [
        _ChainState(
            params.earth_index,
            v_infinity_earth * float(np.cos(aim)),
            v_infinity_earth * float(np.sin(aim)),
            0.0,
            float(aim),
            (),
        )
        for aim in np.linspace(-np.pi, np.pi, _ASSIST_DEPARTURE_AIM_SAMPLES)
    ]
    time_floor = (
        params.max_trip_time - _ASSIST_JOVIAN_LEG_RESERVE_YEARS * _SECONDS_PER_YEAR
    )
    for depth in range(params.max_flybys + 1):
        expanded: Dict[Tuple[int, int, int, int], _ChainState] = {}
        for body_index, rel_t, rel_r, elapsed, aim, decisions in states:
            body = params.bodies[body_index]
            w = float(np.hypot(rel_t, rel_r))
            phi = float(np.arctan2(rel_r, rel_t))
            if depth == 0:
                rotations = np.array([0.0])
            else:
                ecc = conic_kernel.hyperbolic_eccentricity(
                    body.mu, body.min_periapsis, w
                )
                bend_limit = conic_kernel.unpowered_bend_angle(ecc)
                rotations = np.linspace(
                    -bend_limit, bend_limit, _ASSIST_ROTATION_SAMPLES
                )
            for rotation in rotations:
                angle = phi + float(rotation)
                v_t0 = body.v_circ + w * float(np.cos(angle))
                v_r0 = w * float(np.sin(angle))
                terminal = _jovian_terminal(
                    _HeliocentricState(v_t0, v_r0, body.orbit_radius, elapsed),
                    params,
                )
                if terminal is not None and (
                    best is None or terminal.total_time < best[2].total_time
                ):
                    final = decisions + (
                        _ChainDecision(body_index, float(rotation), -1, True),
                    )
                    best = (aim, final, terminal)
                if depth >= params.max_flybys:
                    continue
                for next_index, next_body in enumerate(params.bodies):
                    for (
                        leg_tof,
                        v_t1,
                        v_r1,
                        outbound,
                        _,
                    ) in conic_kernel.conic_radius_crossings(
                        params.flyby.mu_sun,
                        body.orbit_radius,
                        v_t0,
                        v_r0,
                        next_body.orbit_radius,
                        min_leg_time=_ASSIST_MIN_LEG_TIME,
                    ):
                        new_elapsed = elapsed + leg_tof
                        if new_elapsed > time_floor:
                            continue
                        state = _ChainState(
                            next_index,
                            v_t1 - next_body.v_circ,
                            v_r1,
                            new_elapsed,
                            aim,
                            decisions
                            + (
                                _ChainDecision(
                                    body_index, float(rotation), next_index, outbound
                                ),
                            ),
                        )
                        key = (
                            next_index,
                            round(state.excess_tangential / _ASSIST_DEDUP_SPEED_BIN),
                            round(state.excess_radial / _ASSIST_DEDUP_SPEED_BIN),
                            round(
                                new_elapsed
                                / _SECONDS_PER_YEAR
                                / _ASSIST_DEDUP_TIME_BIN_YEARS
                            ),
                        )
                        if key not in expanded or new_elapsed < expanded[key].elapsed:
                            expanded[key] = state
        # Time-bucketed pruning: per (body, time bucket) keep the top states by
        # excess speed, with an overall per-body cap allocated to earlier
        # buckets first.
        buckets: Dict[Tuple[int, int], List[_ChainState]] = {}
        for state in expanded.values():
            bucket_key = (
                state.body_index,
                int(state.elapsed / _SECONDS_PER_YEAR / _ASSIST_BUCKET_WIDTH_YEARS),
            )
            buckets.setdefault(bucket_key, []).append(state)
        kept_per_body: Dict[int, int] = {}
        states = []
        for bucket_key in sorted(buckets, key=lambda k: k[1]):
            ranked = sorted(
                buckets[bucket_key],
                key=lambda s: -float(np.hypot(s.excess_tangential, s.excess_radial)),
            )[:_ASSIST_BEAM_PER_BUCKET]
            already = kept_per_body.get(bucket_key[0], 0)
            taken = ranked[: max(0, _ASSIST_BEAM_CAP_PER_BODY - already)]
            kept_per_body[bucket_key[0]] = already + len(taken)
            states += taken
    return best


@dataclass(frozen=True)
class AssistChainStep:
    """One node of an assist chain: a rotation at a body, then a coast leg.

    Attributes:
        body: Body the step happens at.
        rotation_angle: Unpowered rotation of the planet-relative excess
            velocity applied by the flyby (deg; 0 at the departure step).
        v_infinity: Planet-relative excess speed at this body (km/s) -- the
            Tisserand invariant an unpowered flyby cannot change.
        target: Body (or Jupiter) the following leg coasts to.
        outbound_arrival: Whether the leg arrives at the target's orbit radius
            moving outward (True) or inward (False).
        leg_time: Coast time of the leg (yr).
        elapsed: Cumulative trip time at the end of the leg (yr).
    """

    body: str
    rotation_angle: u.Quantity
    v_infinity: u.Quantity
    target: str
    outbound_arrival: bool
    leg_time: u.Quantity
    elapsed: u.Quantity


@dataclass(frozen=True)
class AssistChainReturn:
    """An unpowered V/E/M assist chain ending in the retrograde return.

    Like :class:`PoweredJovianFlybyReturn` this is not a
    :class:`PuffSatScenario`: it blends the (small) departure burn and phasing
    budget with the collision mass ratio, so it is reported on its own.

    Attributes:
        departure_burn: Oberth burn above escape at 200 km (km/s).
        v_infinity_earth: Departure hyperbolic excess speed (km/s).
        aim_angle: Free-aim angle of the departure excess velocity off Earth's
            orbital velocity (deg).
        steps: The chain nodes in order, departure first, Jovian leg last.
        sequence: Compact chain string, e.g. "E-V-E-V-V-E-J".
        flyby_count: Number of inner-planet gravity assists (excludes the
            departure node and the Jovian bend).
        v_infinity_jupiter: Jupiter-relative excess speed at arrival (km/s).
            Tisserand-fixed: the unpowered bend rotates this vector but cannot
            change its magnitude, so it caps the reachable collision speed.
        jovian_bend_angle: Unpowered bend applied at Jupiter (deg).
        jovian_arrival_direction: Direction of the incoming Jupiter-relative
            excess velocity off Jupiter's orbital velocity (deg). The bend is
            measured from this, so it fixes which returns are reachable; see
            :func:`jovian_return_phasing_envelope`.
        return_perihelion: Perihelion of the retrograde return orbit (AU),
            never reached -- the leg ends at 1 AU.
        closing_speed_1au: Earth-relative speed at the 1 AU crossing (km/s).
        collision_speed: The achieved v_b on the
            retrograde_jovian_hohmann_transfer convention (km/s).
        chain_time: Departure to Jupiter's orbit radius (yr).
        return_time: Jovian bend to the 1 AU crossing (yr).
        total_time: chain_time + return_time (yr).
        phasing_budget: Deep-space maneuver reserve charged as spent methalox
            (km/s).
        delivered_fraction: Mass fraction after the departure burn plus the
            phasing budget.
        payload_puffsat_mass_ratio: Mass ratio at the achieved collision speed
            against the sec:jupiter_only_growth push.
        end_to_end_mass_ratio: delivered_fraction x payload_puffsat_mass_ratio.
    """

    departure_burn: u.Quantity
    v_infinity_earth: u.Quantity
    aim_angle: u.Quantity
    steps: Tuple[AssistChainStep, ...]
    sequence: str
    flyby_count: int
    v_infinity_jupiter: u.Quantity
    jovian_bend_angle: u.Quantity
    jovian_arrival_direction: u.Quantity
    return_perihelion: u.Quantity
    closing_speed_1au: u.Quantity
    collision_speed: u.Quantity
    chain_time: u.Quantity
    return_time: u.Quantity
    total_time: u.Quantity
    phasing_budget: u.Quantity
    delivered_fraction: float
    payload_puffsat_mass_ratio: float
    end_to_end_mass_ratio: float


def _chain_to_result(
    departure_burn: float,
    v_infinity_earth: float,
    aim_angle: float,
    decisions: Tuple[_ChainDecision, ...],
    terminal: _ChainTerminal,
    params: _AssistChainParams,
) -> AssistChainReturn:
    """Replay recorded beam decisions into the public Quantity-valued result.

    Re-simulates every leg from the recorded decisions (rather than trusting
    the search's intermediate floats), so a bookkeeping bug in the beam search
    surfaces here as a replay failure instead of a silently wrong table.

    Args:
        departure_burn: Oberth burn above escape at 200 km (km/s).
        v_infinity_earth: Departure hyperbolic excess speed (km/s).
        aim_angle: Departure free-aim angle (rad).
        decisions: The recorded chain decisions, terminal rotation last.
        terminal: The Jovian terminal the search scored.
        params: The assist-chain parameter block.

    Returns:
        The :class:`AssistChainReturn`.

    Raises:
        ValueError: If a recorded leg fails to replay or the replayed total
            time disagrees with the search's.
    """
    kms = u.km / u.s
    body_index = params.earth_index
    rel_t = v_infinity_earth * float(np.cos(aim_angle))
    rel_r = v_infinity_earth * float(np.sin(aim_angle))
    elapsed = 0.0
    steps: List[AssistChainStep] = []
    symbols: List[str] = []
    for decision in decisions:
        if decision.body_index != body_index:
            raise ValueError("recorded assist-chain decision is at the wrong body")
        body = params.bodies[body_index]
        w = float(np.hypot(rel_t, rel_r))
        angle = float(np.arctan2(rel_r, rel_t)) + decision.rotation
        v_t0 = body.v_circ + w * float(np.cos(angle))
        v_r0 = w * float(np.sin(angle))
        symbols.append(body.symbol)
        if decision.next_body_index < 0:
            steps.append(
                AssistChainStep(
                    body=body.name,
                    rotation_angle=np.degrees(decision.rotation) * u.deg,
                    v_infinity=w * kms,
                    target="Jupiter",
                    outbound_arrival=terminal.outbound_arrival,
                    leg_time=(terminal.leg_tof * u.s).to(u.year),
                    elapsed=((elapsed + terminal.leg_tof) * u.s).to(u.year),
                )
            )
            break
        next_body = params.bodies[decision.next_body_index]
        candidates = [
            crossing
            for crossing in conic_kernel.conic_radius_crossings(
                params.flyby.mu_sun,
                body.orbit_radius,
                v_t0,
                v_r0,
                next_body.orbit_radius,
                min_leg_time=_ASSIST_MIN_LEG_TIME,
            )
            if crossing.outbound == decision.outbound_arrival
        ]
        if not candidates:
            raise ValueError("recorded assist-chain leg did not replay")
        leg_tof, v_t1, v_r1, _, _ = candidates[0]
        elapsed += leg_tof
        steps.append(
            AssistChainStep(
                body=body.name,
                rotation_angle=np.degrees(decision.rotation) * u.deg,
                v_infinity=w * kms,
                target=next_body.name,
                outbound_arrival=decision.outbound_arrival,
                leg_time=(leg_tof * u.s).to(u.year),
                elapsed=(elapsed * u.s).to(u.year),
            )
        )
        rel_t = v_t1 - next_body.v_circ
        rel_r = v_r1
        body_index = decision.next_body_index
    chain_seconds = elapsed + terminal.leg_tof
    replayed_total = chain_seconds + terminal.return_leg.tof
    if abs(replayed_total - terminal.total_time) > 1.0:
        raise ValueError("replayed assist-chain total time disagrees with search")
    phasing_budget = float(ASSIST_CHAIN_PHASING_BUDGET.to_value(kms))
    delivered = float(
        np.exp(-(departure_burn + phasing_budget) / params.flyby.exhaust_speed)
    )
    collision = terminal.return_leg.collision_speed
    mass_ratio = float(
        2.0 * STD_FUDGE_FACTOR / np.log(collision / (collision - params.flyby.v_rf))
    )
    return AssistChainReturn(
        departure_burn=departure_burn * kms,
        v_infinity_earth=v_infinity_earth * kms,
        aim_angle=np.degrees(aim_angle) * u.deg,
        steps=tuple(steps),
        sequence="-".join(symbols) + "-J",
        flyby_count=len(steps) - 1,
        v_infinity_jupiter=terminal.v_infinity_jupiter * kms,
        jovian_bend_angle=np.degrees(terminal.bend_angle) * u.deg,
        jovian_arrival_direction=np.degrees(terminal.arrival_direction) * u.deg,
        return_perihelion=(terminal.return_leg.perihelion * u.km).to(u.AU),
        closing_speed_1au=terminal.return_leg.closing_speed * kms,
        collision_speed=collision * kms,
        chain_time=(chain_seconds * u.s).to(u.year),
        return_time=(terminal.return_leg.tof * u.s).to(u.year),
        total_time=(replayed_total * u.s).to(u.year),
        phasing_budget=phasing_budget * kms,
        delivered_fraction=delivered,
        payload_puffsat_mass_ratio=mass_ratio,
        end_to_end_mass_ratio=delivered * mass_ratio,
    )


def venus_reach_departure_floor() -> u.Quantity:
    """Minimum 200 km Oberth burn whose transfer can touch Venus's orbit.

    At a fixed hyperbolic excess speed, the transfer perihelion is minimized
    by aiming the excess velocity retrograde (anti-tangential): that choice
    minimizes both the orbit energy and the angular momentum at once. The
    floor is the burn at which even that best aim only just brings the
    perihelion down to Venus's orbit radius -- below it Venus (and with it the
    whole assist chain) is unreachable, and *at* it the arrival is tangent to
    Venus's orbit, which is the Tisserand-locked dead end: a tangential
    arrival leaves nothing for a Venus flyby to rotate.

    Returns:
        The floor burn (astropy Quantity, km/s), ~0.2794 km/s.
    """
    params = _powered_flyby_params()
    r_venus = float(VENUS_A.to_value(u.km))
    mu = params.mu_sun
    r_earth = params.r_earth_orbit

    def perihelion_gap(burn: float) -> float:
        v_infinity = float(
            np.sqrt((params.v_esc_leo + burn) ** 2 - params.v_esc_leo**2)
        )
        v_t = params.v_earth_orbit - v_infinity
        state = conic_kernel.conic_state_at_radius(mu, r_earth, v_t, 0.0)
        return conic_kernel.periapsis_radius_of_conic(state.p, state.ecc) - r_venus

    floor = brentq(perihelion_gap, 1e-6, 2.0)
    return float(floor) * u.km / u.s


def assist_chain_return(
    departure_burn: u.Quantity,
    target_collision_speed: Optional[u.Quantity] = None,
    max_trip_time: u.Quantity = ASSIST_CHAIN_MAX_TRIP_TIME,
    max_flybys: int = ASSIST_CHAIN_MAX_FLYBYS,
) -> Optional[AssistChainReturn]:
    """Best unpowered V/E/M assist chain for a given departure burn.

    Starting from C3 = 0 (PuffSat-provided escape), spends ``departure_burn``
    at 200 km, then reaches the powered flyby's retrograde return using only
    unpowered gravity assists at Venus, Earth, and Mars plus one unpowered
    Jovian bend. "Best" is the earliest-arriving chain whose collision speed
    meets the target; see CONTEXT.md ("Unpowered assist chain") for why small
    margins above the Venus-reach floor unlock the chain (square-root escape
    from the Tisserand lock).

    Args:
        departure_burn: Oberth burn above escape at 200 km (astropy Quantity).
        target_collision_speed: Minimum acceptable v_b; defaults to the
            powered-flyby optimum's collision speed (which triggers that
            optimization -- pass it explicitly when it is already at hand).
        max_trip_time: Cap on total trip time (astropy Quantity).
        max_flybys: Cap on the number of inner-planet flybys.

    Returns:
        The :class:`AssistChainReturn`, or None if no chain within the caps
        reaches the target (the beam search is a feasibility *witness*: a None
        is "not found at these beam settings", not a proof of infeasibility).

    Raises:
        ValueError: If the departure burn is not positive.
    """
    burn = float(departure_burn.to_value(u.km / u.s))
    if burn <= 0.0:
        raise ValueError("departure_burn must be positive")
    if target_collision_speed is None:
        target_collision_speed = powered_jovian_flyby_return().collision_speed
    params = _assist_chain_params(
        target_collision_speed=float(target_collision_speed.to_value(u.km / u.s)),
        max_trip_time=max_trip_time,
        max_flybys=max_flybys,
    )
    v_infinity = float(
        np.sqrt((params.flyby.v_esc_leo + burn) ** 2 - params.flyby.v_esc_leo**2)
    )
    found = _assist_chain_search(v_infinity, params)
    if found is None:
        return None
    aim_angle, decisions, terminal = found
    return _chain_to_result(burn, v_infinity, aim_angle, decisions, terminal, params)


@dataclass(frozen=True)
class JovianReturnPhasingEnvelope:
    """Which retrograde returns the chain's Jovian arrival can still reach.

    The chain's search keeps only the earliest-arriving return
    (``_jovian_terminal`` breaks ties on ``total_time``), which hides a knob:
    the same Tisserand-fixed arrival can reach a continuous span of *arrival
    times*, and retuning the bend to land Earth is free. Walking the bend from
    one limit to the other traces a single connected curve from a slow outbound
    return (the craft leaves Jupiter still climbing, coasts to aphelion and
    falls back), through full reversal, to a fast inbound one.

    The bend is **one** knob with **two** outputs, so ``v_b`` cannot be held
    while the arrival time is swept -- they move together along that curve.
    Phasing therefore picks the bend, and the bend *dictates* ``v_b``: a
    lottery within ``collision_speed_min..max``, not a choice. It is benign
    only because every ticket in that band is an acceptable growth loop. See
    ADR 0006-unpowered-jupiter-and-free-return-phasing.

    Attributes:
        v_infinity: Jupiter-relative excess speed, Tisserand-fixed (km/s).
        arrival_direction: Incoming excess-velocity direction (deg).
        bend_limit: Largest unpowered bend available at the periapsis floor
            (deg). Reachable outgoing directions are arrival_direction +/- this.
        collision_speed_min: Lowest v_b phasing may force on you (km/s).
        collision_speed_max: Highest v_b phasing may force on you (km/s); with
            no time cap this is the Tisserand ceiling, hit at full reversal.
        return_time_min: Shortest reachable return (yr).
        return_time_max: Longest reachable return (yr).
        phasing_time_spread: Reachable arrival-time span (d).
        largest_gap: Widest step between consecutive sampled arrival times (d).
            This is resolution-limited, *not* a physical hole: it falls as
            1/``samples`` (~11.9 d at 968, ~0.7 d at 16000) because the curve is
            connected. It is reported because a genuine hole would refuse to
            shrink, and without that check a span quoted as max-minus-min over a
            disconnected set would silently count the hole as authority -- the
            error this function exists to prevent.
        earth_phase_coverage: Heliocentric longitude Earth sweeps over the
            span (deg).
        wraps: earth_phase_coverage / 360 deg.
        closes: Whether coverage reaches 360 deg, i.e. every launch phase
            admits a phased Earth intercept with no propellant.
    """

    v_infinity: u.Quantity
    arrival_direction: u.Quantity
    bend_limit: u.Quantity
    collision_speed_min: u.Quantity
    collision_speed_max: u.Quantity
    return_time_min: u.Quantity
    return_time_max: u.Quantity
    phasing_time_spread: u.Quantity
    largest_gap: u.Quantity
    earth_phase_coverage: u.Quantity
    wraps: float
    closes: bool


def jovian_return_phasing_envelope(
    chain: AssistChainReturn,
    max_total_time: Optional[u.Quantity] = None,
    samples: int = _ASSIST_JOVIAN_BEND_SAMPLES * 8,
) -> JovianReturnPhasingEnvelope:
    """Measure the free Earth-phasing authority of the chain's Jovian bend.

    Sweeps every bend the periapsis floor allows about ``chain``'s actual
    arrival direction -- so every state scored here is one the unpowered flyby
    can really produce -- and reports the span of arrival times reachable, the
    worst hole in it, and the v_b range that span drags along. Converting the
    span at Earth's mean motion gives the phase coverage: at or above 360 deg
    (``closes``), a phased Earth intercept exists at *every* launch phase, with
    no propellant and no powered flyby.

    ``largest_gap`` is what makes the span meaningful rather than decorative: a
    span quoted as max-minus-min over a *disconnected* set would count holes as
    authority. Report it alongside the coverage, always. It is resolution-
    limited here (it falls as 1/``samples``), which is precisely the evidence
    that the reachable set is connected and the span is really walkable.

    Periapsis is not an extra knob here: it *is* the bend (``e = 1 +
    r_p v_inf^2 / mu``, ``sin(delta/2) = 1/e``), so the sweep below is the
    periapsis sweep. Nor does it lift ``collision_speed_max``, which is set by
    the Tisserand-fixed ``v_infinity`` at full reversal.

    Args:
        chain: The chain whose Jovian arrival is being examined.
        max_total_time: Optional cap on chain_time + return_time; returns
            exceeding it are dropped. None applies no cap.
        samples: Bend samples across the reachable arc.

    Returns:
        The :class:`JovianReturnPhasingEnvelope`.

    Raises:
        ValueError: If no retrograde return is reachable from this arrival.
    """
    params = _powered_flyby_params()
    kms = u.km / u.s
    w = float(chain.v_infinity_jupiter.to_value(kms))
    phi = float(chain.jovian_arrival_direction.to_value(u.rad))
    ecc = conic_kernel.hyperbolic_eccentricity(
        params.mu_jupiter, params.periapsis_floor, w
    )
    bend_limit = conic_kernel.unpowered_bend_angle(ecc)
    chain_seconds = float(chain.chain_time.to_value(u.s))
    cap = None if max_total_time is None else float(max_total_time.to_value(u.s))

    speeds: List[float] = []
    times: List[float] = []
    for bend in np.linspace(-bend_limit, bend_limit, samples):
        angle = phi + float(bend)
        leg = _flyby_return_leg(
            params.v_jupiter_orbit + w * float(np.cos(angle)),
            w * float(np.sin(angle)),
            params,
        )
        if leg is None:
            continue
        if cap is not None and chain_seconds + leg.tof > cap:
            continue
        speeds.append(leg.collision_speed)
        times.append(leg.tof)
    if not speeds:
        raise ValueError("no retrograde return reachable from this Jovian arrival")

    vb = np.array(speeds)
    tof = np.sort(np.array(times))
    spread_days = float(tof.max() - tof.min()) / 86400.0
    # The span is only authority if it has no holes: a bend sweep that skipped
    # a range of arrival times could not be walked onto an arbitrary Earth
    # position, and max-minus-min would silently count the hole as coverage.
    gap_days = float(np.diff(tof).max()) / 86400.0 if tof.size > 1 else 0.0
    # Earth's mean motion from Kepler, like assist_chain_window_cadence -- the
    # phase coverage is just how far Earth walks during the reachable spread.
    earth_period_days = float(
        (2.0 * np.pi * np.sqrt(EARTH_A.to(u.km) ** 3 / Sun.k)).to_value(u.day)
    )
    coverage = spread_days * 360.0 / earth_period_days
    return JovianReturnPhasingEnvelope(
        v_infinity=w * kms,
        arrival_direction=np.degrees(phi) * u.deg,
        bend_limit=np.degrees(bend_limit) * u.deg,
        collision_speed_min=float(vb.min()) * kms,
        collision_speed_max=float(vb.max()) * kms,
        return_time_min=(float(tof.min()) * u.s).to(u.year),
        return_time_max=(float(tof.max()) * u.s).to(u.year),
        phasing_time_spread=spread_days * u.day,
        largest_gap=gap_days * u.day,
        earth_phase_coverage=coverage * u.deg,
        wraps=coverage / 360.0,
        closes=coverage >= 360.0,
    )


@dataclass(frozen=True)
class AssistChainBurnScan:
    """Result of scanning departure burns for the smallest feasible one.

    Attributes:
        target_collision_speed: The v_b floor every probe was held to (km/s).
        infeasible_burns: Probes below the found minimum where the beam search
            found no chain (km/s Quantities, ascending).
        minimum: The chain at the smallest feasible probe, or None if every
            probe failed.
    """

    target_collision_speed: u.Quantity
    infeasible_burns: Tuple[u.Quantity, ...]
    minimum: Optional[AssistChainReturn]


def minimum_departure_burn_assist_chain(
    burn_candidates: Tuple[float, ...] = ASSIST_CHAIN_BURN_CANDIDATES,
    target_collision_speed: Optional[u.Quantity] = None,
    max_trip_time: u.Quantity = ASSIST_CHAIN_MAX_TRIP_TIME,
    max_flybys: int = ASSIST_CHAIN_MAX_FLYBYS,
) -> AssistChainBurnScan:
    """Scan departure burns ascending and return the first feasible chain.

    The scan answers "how little delta-v beyond Earth escape does the assist
    chain need?" -- the unpowered counterpart of the powered flyby's 4.45 km/s
    departure burn. Because the beam search is a heuristic witness, the result
    is an upper bound on the true minimum; the probes should bracket the
    analytic Venus-reach floor (:func:`venus_reach_departure_floor`).

    Args:
        burn_candidates: Departure burns to probe, km/s (default
            ASSIST_CHAIN_BURN_CANDIDATES).
        target_collision_speed: Minimum acceptable v_b; defaults to the
            powered-flyby optimum's collision speed (computed once for the
            whole scan).
        max_trip_time: Cap on total trip time (astropy Quantity).
        max_flybys: Cap on the number of inner-planet flybys.

    Returns:
        The :class:`AssistChainBurnScan`.
    """
    if target_collision_speed is None:
        target_collision_speed = powered_jovian_flyby_return().collision_speed
    infeasible: List[u.Quantity] = []
    for burn in sorted(burn_candidates):
        result = assist_chain_return(
            departure_burn=burn * u.km / u.s,
            target_collision_speed=target_collision_speed,
            max_trip_time=max_trip_time,
            max_flybys=max_flybys,
        )
        if result is not None:
            return AssistChainBurnScan(
                target_collision_speed=target_collision_speed,
                infeasible_burns=tuple(infeasible),
                minimum=result,
            )
        infeasible.append(burn * u.km / u.s)
    return AssistChainBurnScan(
        target_collision_speed=target_collision_speed,
        infeasible_burns=tuple(infeasible),
        minimum=None,
    )


@dataclass(frozen=True)
class AssistChainWindowCadence:
    """Synodic launch-window cadence of the assist-chain family (ADR 0005).

    The phasing-free chain model cannot say *when* a chain flies, but the
    window structure follows from synodic periods: every chain starts E->V,
    so windows open at the Earth-Venus synodic cadence, gated secondarily by
    Jupiter's phase; resonant-rev stretching of the interior ladder (quantized
    Venus/Earth revolutions) plus the DSM phasing budget absorb the residual.

    Attributes:
        venus_window: Earth-Venus synodic period -- the base cadence at which
            chain launch windows open (yr).
        jupiter_window: Earth-Jupiter synodic period -- how often Jupiter's
            phase gate recurs (yr).
        earth_venus_cycle: The 5-synodic Earth-Venus near-cycle (~8 yr) after
            which the inner-ladder geometry nearly repeats (yr).
        triple_realignment: Approximate recurrence of the full
            Venus-Earth-Jupiter geometry: 15 Earth-Venus synodics, which is
            also ~22 Earth-Jupiter synodics and ~2 Jupiter years (yr).
        effective_cycle_floor: Chain trip time with zero relaunch wait (yr).
        effective_cycle_ceiling: Trip time plus one full Venus window of
            relaunch wait -- the worst case for a returning cohort (yr).
        doubling_time_floor: Fleet doubling time at the chain's end-to-end
            mass ratio and the floor cycle (yr).
        doubling_time_ceiling: The same at the ceiling cycle (yr).
    """

    venus_window: u.Quantity
    jupiter_window: u.Quantity
    earth_venus_cycle: u.Quantity
    triple_realignment: u.Quantity
    effective_cycle_floor: u.Quantity
    effective_cycle_ceiling: u.Quantity
    doubling_time_floor: u.Quantity
    doubling_time_ceiling: u.Quantity


def assist_chain_window_cadence(
    total_time: u.Quantity, end_to_end_mass_ratio: float
) -> AssistChainWindowCadence:
    """Launch-window cadence and growth-cycle timing of the assist chain.

    Computes the synodic scaffolding behind "how often can an E-V ladder
    fly": Kepler periods of Venus, Earth, and Jupiter give the Earth-Venus
    window cadence (~1.6 yr), the Earth-Jupiter phase gate (~1.1 yr), the
    ~8 yr Earth-Venus near-cycle, and the ~24 yr full realignment. The
    effective growth cycle is the chain's trip time plus between zero and one
    Venus window of relaunch wait; doubling times follow from the end-to-end
    mass ratio per cycle. This is a cadence *estimate*, not an ephemeris
    search: which specific calendar windows work, and their trip-time spread,
    would need Lambert arcs against real planet positions (ADR 0005).

    Args:
        total_time: Chain trip time, e.g. an AssistChainReturn.total_time
            (astropy Quantity).
        end_to_end_mass_ratio: Mass multiplier per cycle, e.g. an
            AssistChainReturn.end_to_end_mass_ratio (> 1).

    Returns:
        The :class:`AssistChainWindowCadence`.

    Raises:
        ValueError: If end_to_end_mass_ratio is not greater than 1 (no growth,
            so no doubling time exists).
    """
    if end_to_end_mass_ratio <= 1.0:
        raise ValueError("end_to_end_mass_ratio must exceed 1 for a growth cycle")
    mu_sun = Sun.k

    def kepler_period(semi_major_axis: u.Quantity) -> u.Quantity:
        return (2.0 * np.pi * np.sqrt(semi_major_axis.to(u.km) ** 3 / mu_sun)).to(
            u.year
        )

    def synodic_period(period_a: u.Quantity, period_b: u.Quantity) -> u.Quantity:
        return abs(1.0 / (1.0 / period_a - 1.0 / period_b)).to(u.year)

    period_venus = kepler_period(VENUS_A)
    period_earth = kepler_period(EARTH_A)
    period_jupiter = kepler_period(JUPITER_A)
    venus_window = synodic_period(period_venus, period_earth)
    jupiter_window = synodic_period(period_earth, period_jupiter)
    floor_cycle = total_time.to(u.year)
    ceiling_cycle = floor_cycle + venus_window
    doublings_per_cycle = float(np.log(end_to_end_mass_ratio) / np.log(2.0))
    return AssistChainWindowCadence(
        venus_window=venus_window,
        jupiter_window=jupiter_window,
        earth_venus_cycle=5.0 * venus_window,
        triple_realignment=15.0 * venus_window,
        effective_cycle_floor=floor_cycle,
        effective_cycle_ceiling=ceiling_cycle,
        doubling_time_floor=floor_cycle / doublings_per_cycle,
        doubling_time_ceiling=ceiling_cycle / doublings_per_cycle,
    )
