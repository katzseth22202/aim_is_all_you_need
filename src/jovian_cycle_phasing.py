"""Forward chain optimizer for the closed Jupiter-only growth cycle
(CONTEXT.md "Jovian cycle phasing"; ADR 0010-jovian-cycle-phasing-verifier).

``puffsat_cycle_growth`` (jovian_flyby.py) scores the Earth -> Jupiter flyby ->
retrograde-return -> relaunch loop on doubling time, but its clock is
``trip_time + PUFFSAT_CYCLE_ORBIT_PERIOD`` -- it assumes the mass can re-depart
to Jupiter *instantly* after the 20-day periapsis coast. That assumption is
unpaid: a fresh Earth departure needs Jupiter in position, and the return bend
is spent getting home (ADR 0006 proved the return itself phases for free, but
explicitly deferred whether the whole loop keeps closing cycle after cycle).

Crucially the mass cannot *wait* for a good launch: it returns when its
trajectory says it does, is consumed in the collision immediately, and that
collision launches the next payload one 20-day coast later. So each cycle's
departure time is pinned to the previous cycle's arrival. The only freedom is to
*steer* the arrival -- the Jupiter bend moves the return crossing across ~1130
days (ADR 0006), roughly three Earth-Jupiter synodic periods -- so the trajectory
is chosen to land the next departure on a Jupiter phase that is both reachable
and growth-viable.

That makes this a forward *chain* search over cycles, not a per-cycle solve.
Each state is a departure time; each branch is a closing trajectory (an ``izzo``
outbound arc onto Jupiter's true position, a phased perijove bend/burn, and a
retrograde return that intercepts Earth) that both earns a growth factor and
sets the next departure time. A generational beam over cycles (bucketing
near-identical departure times) threads the chain that maximizes compounded mass
launched off Earth over the horizon, run with and without the powered perijove
burn. All in the repo's circular, coplanar model;
launch times are a relative epoch, not calendar dates (ADR 0006's ephemeris
study is still deferred).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from astropy import units as u
from boinor.core.iod import izzo
from scipy.optimize import brentq

from src import conic_kernel
from src.astro_constants import (
    JUPITER_FLYBY_MAX_TOF,
    PUFFSAT_CYCLE_ORBIT_PERIOD,
    STD_FUDGE_FACTOR,
)
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.retrograde_return_legs import (
    _assist_chain_params,
    _AssistChainParams,
    _body_state,
    _earth_phase_mismatch,
    _phased_jovian_flyby,
    _ReturnLeg,
)

# Search grids for the per-cycle branch enumeration. Wide enough to bracket a
# closing geometry (ADR 0006's return coverage is > 3 wraps of Earth phase) yet
# cheap enough for a full 30-year chain search of both runs.
# A retrograde return needs a large Jupiter-relative excess (> Jupiter's ~13 km/s
# orbital speed), so the outbound leg must be *fast* and energetic -- the
# Hohmann transfer (~2.73 yr, v_inf ~5.6 km/s at Jupiter) is far too slow to
# reverse. The grid therefore reaches down to ~1.1 yr, where the incumbent
# powered flyby's 1.31 yr outbound lives.
_OUTBOUND_TOF_MIN = 1.1  # yr
_OUTBOUND_TOF_MAX = 5.8  # yr; leaves room under the 7 yr total-TOF cap
_OUTBOUND_TOF_SAMPLES = 26
_PERIAPSIS_RATIO_MAX = 50.0  # periapsis scanned from the floor up to 50x it
_PERIAPSIS_SAMPLES = 26
_POWERED_BURN_MAX = 4.0  # km/s; perijove burn axis for the powered run
_POWERED_BURN_SAMPLES = 5

# Chain-search discretization. The search is a generational beam over cycles:
# each generation keeps the ``_BEAM_WIDTH`` highest-growth departure states,
# bucketing near-identical departure times so phase duplicates collapse.
_STATE_BUCKET_DAYS = 7.0  # departure times within a bucket are treated as one
_SEED_SAMPLES = 16  # candidate seed launches across the first synodic period
_BEAM_WIDTH = 48  # departure states carried between generations
_EARTH_JUPITER_SYNODIC_YEARS = 1.0923  # Kepler synodic of Earth and Jupiter


@dataclass(frozen=True)
class ChainCycle:
    """One closed Earth -> Jupiter -> Earth cycle on the optimal chain.

    Attributes:
        cycle_index: Zero-based position along the chain.
        launch_time: Departure time from the epoch (astropy Quantity, years). A
            relative launch date -- absolute calendar dates would need an
            ephemeris.
        outbound_time: Earth-to-Jupiter time of flight (astropy Quantity, years).
        return_time: Jupiter-to-1 AU return time of flight (astropy Quantity,
            years).
        cycle_time: Departure to the next departure, ``outbound + return +
            20-day coast`` (astropy Quantity, years).
        departure_burn: Oberth burn above the cycle-orbit periapsis speed buying
            the outbound leg (astropy Quantity, km/s).
        flyby_burn: Impulsive perijove burn (astropy Quantity, km/s); zero on the
            unpowered chain.
        collision_speed: Achieved v_b of the returning PuffSat (astropy Quantity,
            km/s), dictated by the bend the phasing selects.
        net_growth: Payload multiplier for this cycle, ``delivered_fraction x
            payload_mass_ratio`` -- above 1 grows the launched mass, below 1
            shrinks it.
        cumulative_mass: Compounded launched-mass multiple through this cycle
            (product of ``net_growth`` up to and including it), relative to the
            seed mass.
    """

    cycle_index: int
    launch_time: u.Quantity
    outbound_time: u.Quantity
    return_time: u.Quantity
    cycle_time: u.Quantity
    departure_burn: u.Quantity
    flyby_burn: u.Quantity
    collision_speed: u.Quantity
    net_growth: float
    cumulative_mass: float


@dataclass(frozen=True)
class ChainResult:
    """Best growth-maximizing chain of cycles found over the horizon.

    Attributes:
        powered: Whether the perijove burn knob was free (True) or pinned to
            zero (False).
        cycles: The cycles of the optimal chain, in launch order. Empty if no
            self-sustaining chain of even one cycle exists.
        mass_multiple_10yr: Compounded launched-mass multiple of the cycles whose
            departure falls at or before 10 years, relative to the seed mass.
        mass_multiple_20yr: Same at or before 20 years.
        mass_multiple_30yr: Same over the full horizon (the objective maximized).
        all_growth_positive: Whether every cycle on the chain has net_growth > 1
            (a purely self-sustaining chain, no phase-repositioning losses).
    """

    powered: bool
    cycles: Tuple[ChainCycle, ...]
    mass_multiple_10yr: float
    mass_multiple_20yr: float
    mass_multiple_30yr: float
    all_growth_positive: bool


@dataclass(frozen=True)
class _Branch:
    """One closing trajectory leaving a chain state (all floats, s / km/s).

    Attributes:
        next_departure: Departure time of the following cycle (s), i.e. this
            cycle's Earth arrival plus the 20-day coast.
        log_growth: ``ln(net_growth)`` earned by flying this cycle.
        outbound_tof: Earth-to-Jupiter time of flight (s).
        return_tof: Jupiter-to-1 AU return time of flight (s).
        departure_burn: Oberth departure burn (km/s).
        flyby_burn: Perijove burn (km/s).
        collision_speed: Achieved v_b (km/s).
        net_growth: Payload multiplier for this cycle.
    """

    next_departure: float
    log_growth: float
    outbound_tof: float
    return_tof: float
    departure_burn: float
    flyby_burn: float
    collision_speed: float
    net_growth: float


def _outbound_arrival(
    launch_time: float,
    outbound_tof: float,
    earth_longitude0: float,
    jupiter_longitude0: float,
    params: _AssistChainParams,
) -> Optional[Tuple[float, npt.NDArray[np.float64], float]]:
    """Lambert Earth-to-Jupiter arc onto Jupiter's true position.

    Places Earth and Jupiter where the circular model actually puts them at the
    launch and arrival instants, and solves the transfer between them -- the
    same ``izzo`` call :func:`_phased_ladder_burn` uses. The departure excess is
    bought with the Oberth burn at 200 km, starting from the cycle-orbit
    periapsis speed; the arrival excess is handed to the Jovian flyby as a
    heliocentric vector.

    Args:
        launch_time: Departure time from the epoch (s).
        outbound_tof: Earth-to-Jupiter time of flight (s).
        earth_longitude0: Earth's heliocentric longitude at the epoch (rad).
        jupiter_longitude0: Jupiter's heliocentric longitude at the epoch (rad).
        params: The assist-chain parameter block (for the substrate constants).

    Returns:
        (departure_burn km/s, arrival excess km/s heliocentric 3-vector,
        Jupiter longitude at arrival rad), or None if the Lambert solve fails or
        the endpoints are collinear.
    """
    flyby = params.flyby
    n_earth = conic_kernel.mean_motion(flyby.v_earth_orbit, flyby.r_earth_orbit)
    n_jupiter = conic_kernel.mean_motion(flyby.v_jupiter_orbit, flyby.r_jupiter_orbit)
    lon_earth = earth_longitude0 + n_earth * launch_time
    lon_jupiter = jupiter_longitude0 + n_jupiter * (launch_time + outbound_tof)
    r0, v_earth = _body_state(flyby.r_earth_orbit, lon_earth, flyby.v_earth_orbit)
    r1, v_jupiter = _body_state(
        flyby.r_jupiter_orbit, lon_jupiter, flyby.v_jupiter_orbit
    )
    separation = float(np.linalg.norm(np.cross(r0, r1)))
    if separation < 1e-6 * flyby.r_earth_orbit * flyby.r_jupiter_orbit:
        return None
    try:
        v_depart, v_arrive = izzo(
            flyby.mu_sun, r0, r1, outbound_tof, 0, True, True, 35, 1e-8
        )
    except (ValueError, RuntimeError):
        return None
    v_inf_depart = float(np.linalg.norm(v_depart - v_earth))
    departure_burn = (
        float(np.sqrt(flyby.v_esc_leo**2 + v_inf_depart**2)) - flyby.v_depart_from
    )
    excess_arrival = np.asarray(v_arrive - v_jupiter, dtype=np.float64)
    return departure_burn, excess_arrival, lon_jupiter


def _closing_returns(
    excess_arrival: npt.NDArray[np.float64],
    jupiter_longitude: float,
    flyby_time: float,
    earth_longitude0: float,
    flyby_burn: float,
    bend_sign: float,
    max_tof: float,
    params: _AssistChainParams,
) -> List[Tuple[_ReturnLeg, float]]:
    """Perijove passes whose retrograde return intercepts Earth.

    Sweeps the flyby periapsis (which *is* the bend: a larger radius bends less)
    and roots the Earth-phase mismatch, the same scan-for-sign-change-then-
    ``brentq`` pattern as :func:`_phased_leg_rotations`. Branch-cut jumps in the
    wrapped residual are rejected so phantom roots are never taken.

    Args:
        excess_arrival: Jupiter-relative arrival excess (km/s, heliocentric
            3-vector).
        jupiter_longitude: Jupiter's longitude at the flyby (rad).
        flyby_time: Flyby time from the epoch (s).
        earth_longitude0: Earth's longitude at the epoch (rad).
        flyby_burn: Perijove burn (km/s); zero on the unpowered run.
        bend_sign: Which side Jupiter is passed on (+1 or -1).
        max_tof: Cap on outbound + return time of flight (s).
        params: The assist-chain parameter block.

    Returns:
        One (return leg, residual) per rooted closure; residual is ~0 by
        construction. Possibly empty.
    """
    flyby = params.flyby
    floor = flyby.periapsis_floor
    grid = np.geomspace(floor, floor * _PERIAPSIS_RATIO_MAX, _PERIAPSIS_SAMPLES)

    def evaluate(periapsis: float) -> Optional[Tuple[float, _ReturnLeg]]:
        leg = _phased_jovian_flyby(
            excess_arrival, jupiter_longitude, periapsis, flyby_burn, bend_sign, params
        )
        # ``max_tof`` is the return budget the caller already reduced by the
        # outbound leg, so the cap is on the return time of flight alone.
        if leg is None or leg.tof > max_tof:
            return None
        mismatch = _earth_phase_mismatch(
            leg, jupiter_longitude, flyby_time, earth_longitude0, flyby
        )
        return mismatch, leg

    def residual(periapsis: float) -> float:
        found = evaluate(periapsis)
        return float("nan") if found is None else found[0]

    values = [residual(float(r)) for r in grid]
    solved: List[Tuple[_ReturnLeg, float]] = []
    for i in range(len(grid) - 1):
        lo, hi = values[i], values[i + 1]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if lo == 0.0:
            root = float(grid[i])
        elif lo * hi > 0.0:
            continue
        elif abs(lo - hi) > np.pi:
            continue
        else:
            try:
                root = float(brentq(residual, grid[i], grid[i + 1], xtol=1.0))
            except (ValueError, RuntimeError):
                continue
        found = evaluate(root)
        if found is None:
            continue
        solved.append((found[1], found[0]))
    return solved


def _net_growth(
    departure_burn: float,
    flyby_burn: float,
    collision_speed: float,
    params: _AssistChainParams,
) -> Optional[float]:
    """Payload multiplier for one cycle, ``delivered_fraction x mass_ratio``.

    The same growth arithmetic as :func:`puffsat_cycle_growth`, inlined in
    floats for the chain hot loop: the two methalox burns erode the mass by
    ``exp(-dv / exhaust)``, and the collision at ``collision_speed`` mints new
    payload per ``eq:PuffSat_ratio`` against the cycle-orbit push target
    ``params.flyby.v_rf``.

    Args:
        departure_burn: Oberth departure burn (km/s).
        flyby_burn: Perijove burn (km/s).
        collision_speed: Achieved v_b (km/s).
        params: The assist-chain parameter block.

    Returns:
        The net growth (> 1 grows the launched mass), or None if the collision
        does not exceed the push target (no payload minted).
    """
    v_rf = params.flyby.v_rf
    if collision_speed <= v_rf:
        return None
    delivered = float(
        np.exp(-(departure_burn + flyby_burn) / params.flyby.exhaust_speed)
    )
    mass_ratio = (
        2.0
        * STD_FUDGE_FACTOR
        / float(np.log(collision_speed / (collision_speed - v_rf)))
    )
    return delivered * mass_ratio


def _cycle_branches(
    departure_time: float,
    powered: bool,
    earth_longitude0: float,
    jupiter_longitude0: float,
    max_tof: float,
    coast: float,
    params: _AssistChainParams,
) -> List[_Branch]:
    """All closing trajectories leaving a cycle departing at ``departure_time``.

    Enumerates the outbound time of flight (each an ``izzo`` arc onto Jupiter's
    true position), the perijove bend, and -- on the powered run -- the perijove
    burn, keeping every retrograde return that intercepts Earth and mints
    payload. Branches landing in the same next-departure bucket are collapsed to
    the single highest-growth one, since the future value depends only on the
    next state, not on how it was reached.

    Args:
        departure_time: This cycle's departure time from the epoch (s).
        powered: Whether the perijove burn may be nonzero.
        earth_longitude0: Earth's longitude at the epoch (rad).
        jupiter_longitude0: Jupiter's longitude at the epoch (rad).
        max_tof: Cap on outbound + return time of flight (s).
        coast: The post-collision coast to the next departure (s).
        params: The assist-chain parameter block.

    Returns:
        One best :class:`_Branch` per reachable next-departure bucket.
    """
    burns = (
        [float(b) for b in np.linspace(0.0, _POWERED_BURN_MAX, _POWERED_BURN_SAMPLES)]
        if powered
        else [0.0]
    )
    year_s = float((1.0 * u.year).to_value(u.s))
    tof_grid = np.linspace(
        _OUTBOUND_TOF_MIN * year_s, _OUTBOUND_TOF_MAX * year_s, _OUTBOUND_TOF_SAMPLES
    )
    bucket_s = _STATE_BUCKET_DAYS * 86400.0
    best_by_bucket: Dict[int, _Branch] = {}
    for outbound_tof in tof_grid:
        arrival = _outbound_arrival(
            departure_time,
            float(outbound_tof),
            earth_longitude0,
            jupiter_longitude0,
            params,
        )
        if arrival is None:
            continue
        departure_burn, excess_arrival, lon_jupiter = arrival
        return_cap = max_tof - float(outbound_tof)
        if return_cap <= 0.0:
            continue
        for flyby_burn in burns:
            for bend_sign in (1.0, -1.0):
                for leg, _residual in _closing_returns(
                    excess_arrival,
                    lon_jupiter,
                    departure_time + float(outbound_tof),
                    earth_longitude0,
                    flyby_burn,
                    bend_sign,
                    return_cap,
                    params,
                ):
                    net = _net_growth(
                        departure_burn, flyby_burn, leg.collision_speed, params
                    )
                    if net is None or net <= 0.0:
                        continue
                    next_departure = (
                        departure_time + float(outbound_tof) + leg.tof + coast
                    )
                    branch = _Branch(
                        next_departure=next_departure,
                        log_growth=float(np.log(net)),
                        outbound_tof=float(outbound_tof),
                        return_tof=leg.tof,
                        departure_burn=departure_burn,
                        flyby_burn=flyby_burn,
                        collision_speed=leg.collision_speed,
                        net_growth=net,
                    )
                    key = int(round(next_departure / bucket_s))
                    incumbent = best_by_bucket.get(key)
                    if incumbent is None or branch.log_growth > incumbent.log_growth:
                        best_by_bucket[key] = branch
    return list(best_by_bucket.values())


def optimize_jovian_cycle_chain(
    years: float = 30.0,
    powered: bool = False,
    earth_longitude0: float = 0.0,
    jupiter_longitude0: float = 0.0,
) -> ChainResult:
    """Best growth-maximizing chain of Jupiter-only cycles over the horizon.

    The mass cannot wait: each cycle's departure is pinned to the previous
    cycle's Earth arrival plus the 20-day coast, and the only freedom is the
    trajectory itself (which also steers the arrival, hence the next departure).
    This threads a forward, time-bucketed dynamic program over cycles -- each
    state a departure time, each edge a closing trajectory that earns a growth
    factor -- to maximize the compounded mass launched off Earth by the horizon.
    The seed launch is free within the first Earth-Jupiter synodic period;
    every later departure is pinned. Run with the perijove burn free
    (``powered=True``) or pinned to zero to isolate the burn's value.

    Args:
        years: Horizon to optimize over (years).
        powered: Whether the perijove burn knob is free.
        earth_longitude0: Earth's heliocentric longitude at the epoch (rad).
        jupiter_longitude0: Jupiter's heliocentric longitude at the epoch (rad).

    Returns:
        The :class:`ChainResult` for the optimal chain, with 10/20/30-year
        compounded-mass milestones.
    """
    cycle_speed = puffsat_cycle_periapsis_speed()
    params = _assist_chain_params(
        target_collision_speed=float(cycle_speed.to_value(u.km / u.s)),
        cycle_periapsis_speed=cycle_speed,
    )
    kms = u.km / u.s
    year_s = float((1.0 * u.year).to_value(u.s))
    coast_s = float(PUFFSAT_CYCLE_ORBIT_PERIOD.to_value(u.s))
    max_tof = float(JUPITER_FLYBY_MAX_TOF.to_value(u.s))
    horizon = years * year_s
    bucket_s = _STATE_BUCKET_DAYS * 86400.0
    synodic_s = _EARTH_JUPITER_SYNODIC_YEARS * year_s

    # Generational beam over cycles. Each node is a departure state
    # (time, compounded log-growth, the branch that produced it, parent index).
    # Every edge advances time by >= one outbound leg (~1.1 yr), so cycles form
    # generations; each generation keeps only the _BEAM_WIDTH best-growth states,
    # bucketing near-identical departure times so phase duplicates collapse.
    node_time: List[float] = []
    node_value: List[float] = []
    node_branch: List[Optional[_Branch]] = []
    node_parent: List[int] = []

    def add_node(
        time: float, value: float, branch: Optional[_Branch], parent: int
    ) -> int:
        node_time.append(time)
        node_value.append(value)
        node_branch.append(branch)
        node_parent.append(parent)
        return len(node_time) - 1

    frontier: List[int] = []
    seen_seeds: set[int] = set()
    for seed in np.linspace(0.0, synodic_s, _SEED_SAMPLES):
        key = int(round(float(seed) / bucket_s))
        if key not in seen_seeds:
            seen_seeds.add(key)
            frontier.append(add_node(float(seed), 0.0, None, -1))

    best_terminal = -1
    for idx in frontier:
        if best_terminal < 0 or node_value[idx] > node_value[best_terminal]:
            best_terminal = idx

    while frontier:
        # bucket -> (value, time, branch, parent_idx): best successor per bucket.
        successors: Dict[int, Tuple[float, float, _Branch, int]] = {}
        for idx in frontier:
            depart = node_time[idx]
            if depart > horizon:
                continue
            base = node_value[idx]
            for branch in _cycle_branches(
                depart,
                powered,
                earth_longitude0,
                jupiter_longitude0,
                max_tof,
                coast_s,
                params,
            ):
                nb = int(round(branch.next_departure / bucket_s))
                value = base + branch.log_growth
                incumbent = successors.get(nb)
                if incumbent is None or value > incumbent[0]:
                    successors[nb] = (value, branch.next_departure, branch, idx)
        ranked = sorted(successors.values(), key=lambda item: item[0], reverse=True)
        frontier = []
        for value, next_time, branch, parent_idx in ranked[:_BEAM_WIDTH]:
            child = add_node(next_time, value, branch, parent_idx)
            frontier.append(child)
            if next_time <= horizon and value > node_value[best_terminal]:
                best_terminal = child

    if node_branch[best_terminal] is None:
        # Not one cycle could be flown from any seed.
        return ChainResult(
            powered=powered,
            cycles=(),
            mass_multiple_10yr=1.0,
            mass_multiple_20yr=1.0,
            mass_multiple_30yr=1.0,
            all_growth_positive=False,
        )

    # Walk parent pointers back to the seed and reverse into launch order.
    chain: List[_Branch] = []
    node = best_terminal
    while node >= 0 and node_branch[node] is not None:
        branch_here = node_branch[node]
        assert branch_here is not None
        chain.append(branch_here)
        node = node_parent[node]
    chain.reverse()

    cycles: List[ChainCycle] = []
    cumulative = 1.0
    for index, branch in enumerate(chain):
        cumulative *= branch.net_growth
        launch = branch.next_departure - (
            branch.outbound_tof + branch.return_tof + coast_s
        )
        cycles.append(
            ChainCycle(
                cycle_index=index,
                launch_time=(launch / year_s) * u.year,
                outbound_time=(branch.outbound_tof / year_s) * u.year,
                return_time=(branch.return_tof / year_s) * u.year,
                cycle_time=(
                    (branch.outbound_tof + branch.return_tof + coast_s) / year_s
                )
                * u.year,
                departure_burn=branch.departure_burn * kms,
                flyby_burn=branch.flyby_burn * kms,
                collision_speed=branch.collision_speed * kms,
                net_growth=branch.net_growth,
                cumulative_mass=cumulative,
            )
        )

    def mass_by(year_mark: float) -> float:
        milestone = 1.0
        for cycle in cycles:
            if float(cycle.launch_time.to_value(u.year)) <= year_mark:
                milestone = cycle.cumulative_mass
        return milestone

    return ChainResult(
        powered=powered,
        cycles=tuple(cycles),
        mass_multiple_10yr=mass_by(10.0),
        mass_multiple_20yr=mass_by(20.0),
        mass_multiple_30yr=cycles[-1].cumulative_mass if cycles else 1.0,
        all_growth_positive=all(c.net_growth > 1.0 for c in cycles),
    )
