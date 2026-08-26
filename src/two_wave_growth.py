"""Real-orbit two-wave nozzle growth over a 30-year adaptive 2S/3S cadence.

Joins two machines that were built separately:

- ``src/real_orbit_resonance.py``'s conditional two-or-three-synodic cadence
  (ADR 0011): real Astropy Earth and Jupiter states, zero-revolution Lambert
  arcs, a 2S return flown whenever its deep-space-maneuver proxy stays inside
  50 m/s and a 3S return otherwise.  Each selected return is the next cycle's
  departure, so the flown chain is a single connected trajectory.
- ``src/nozzle_analysis.py``'s two-wave same-cycle nozzle pricing (ADR 0009):
  the departing batch splits at Jupiter into a growth wave that arrives early
  and pushes the next payload, and a nozzle wave that supplies the head-on
  departure-burn projectiles for the payload the growth wave parked.

The split is re-solved here per flown cycle against the same real ephemeris,
as a second Earth-hit return leaving the same departure epoch and arriving
``split_days`` earlier than the nozzle wave.  That gives the growth wave its
own collision speed from geometry rather than by extrapolation.

Run it with ``make two-wave``; it is not imported by ``src.main`` and is not
part of ``make all``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.constants import g0
from astropy.time import Time
from scipy.optimize import minimize_scalar

from src.astro_constants import METHALOX_VACUUM_ISP
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.nozzle_analysis import (
    NozzlePricing,
    apoapsis_reversal_dv,
    same_cycle_nozzle,
)
from src.plume_thermal import (
    NOZZLE_FLOOR_TEMPERATURE,
    NOZZLE_GATE_TEMPERATURE,
    chemistry_efficiency,
    slug_ratio_window,
)
from src.real_orbit_resonance import (
    _DAYS_PER_YEAR,
    _EPHEMERIS,
    _FIXED_TWO_SYNODIC_DAYS,
    _ManeuverSolution,
    _resonance_epochs,
    _window_maneuver_solutions,
)

_DEFAULT_START = "2026-08-11"
_DEFAULT_HORIZON_YEARS = 30.0
_DEFAULT_THRESHOLD_M_S = 50.0
# The split gap is the parking-orbit period: the growth wave pushes the payload
# up from 200 km, the payload coasts one full orbit, and the nozzle wave departs
# it at the next periapsis.  So this same number sizes the apoapsis reversal.
# ADR 0013 decided the gap is 10 days, so that is the default here and the one
# ``two_leg_nozzle_sweep`` imports; the two must not drift apart, because the
# chain a caller gets silently depends on it (ADR 0015, item 4).
DEFAULT_SPLIT_DAYS = 10.0
# Methalox vacuum exhaust speed, the currency every correction burn is paid in.
VE_METHALOX = float((METHALOX_VACUUM_ISP * g0).to_value(u.km / u.s))
# Outer slug-ratio search box for the chain optimum, recorded per the ADR 0007
# lesson.  It is only the *box*: the admissible set is this intersected with
# the fleet ignition window (:func:`fleet_ignition_windows`), which is a closed
# interval on both sides because the dissipated energy ``w^2 k/(2(1+k)^2)``
# peaks at ``k = 1`` and falls away on both.  The old bare ceiling of 80 ran
# the optimiser six times past where the plume stops being a plasma the field
# can grip -- ``k_max`` is 36.88 even at the 75 km/s head-on leg, and 12.29 on
# the overtake's cold end (``puffsat_impact_simulation``, ``make analysis-toll``).
_K_SEARCH_MIN = 0.2
_K_SEARCH_MAX = 80.0
_K_SAMPLES = 60
# Sweep grids.  Recovery spans a bare dish (~0.25 of the ideal collimated
# impulse at k = 3) up to a near-ideal tamped plume; fudge spans the paper's
# f = 0.8 down to half-elastic.
_DEFAULT_RECOVERIES = (0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_DEFAULT_FUDGES = (0.5, 0.6, 0.7, 0.8)
_DEFAULT_SPLIT_OPTIONS = (10.0, 20.0, 30.0, 60.0)
# Operating point the split gap and the per-cycle table are reported at.
_REFERENCE_RECOVERY = 0.6
_REFERENCE_FUDGE = 0.8
# Horizon the continuous-rate projection is run out to, independent of the span
# the chain actually flies (which lands wherever whole cycles happen to end).
_PROJECTION_YEARS = 30.0


@lru_cache(maxsize=1)
def _cycle_periapsis_speed() -> float:
    """Closed-cycle 200 km periapsis speed the departure burn starts from (km/s)."""
    return float(puffsat_cycle_periapsis_speed().to_value(u.km / u.s))


@dataclass(frozen=True)
class TwoWaveCycle:
    """One flown cycle of the adaptive cadence, with both waves priced.

    Attributes:
        index: Position in the flown chain, counting from zero.
        departure_jd: TDB Julian date the batch leaves Earth.
        return_jd: TDB Julian date the nozzle wave reaches Earth, which is
            also the next cycle's departure.
        synodic_multiple: Two or three, whichever the policy selected.
        period_years: Departure-to-departure cycle length.
        departure_burn: Periapsis speed increment above the closed-cycle
            200 km speed (km/s).
        nozzle_wave_v_b: Collision speed of the wave that supplies the
            departure-burn projectiles (km/s at 200 km).
        nozzle_wave_dsm: Deep-space-maneuver proxy the selected return needs
            (km/s); the policy holds this under its threshold.
        split_days: Gap by which the growth wave leads the nozzle wave, equal
            to the parking-orbit period the pushed payload coasts through.
        growth_wave_arrival_jd: TDB Julian date the growth wave reaches Earth.
        growth_wave_v_b: Collision speed of the wave that pushes the next
            payload (km/s at 200 km).
        growth_wave_burn: Delta-v the growth wave spends to arrive early
            (km/s), the price of the split.
    """

    index: int
    departure_jd: float
    return_jd: float
    synodic_multiple: int
    period_years: float
    departure_burn: float
    nozzle_wave_v_b: float
    nozzle_wave_dsm: float
    split_days: float
    growth_wave_arrival_jd: float
    growth_wave_v_b: float
    growth_wave_burn: float


def adaptive_two_wave_cycles(
    start: str = _DEFAULT_START,
    years: float = _DEFAULT_HORIZON_YEARS,
    threshold_m_s: float = _DEFAULT_THRESHOLD_M_S,
    split_days: float = DEFAULT_SPLIT_DAYS,
) -> List[TwoWaveCycle]:
    """Fly the adaptive 2S/3S cadence and return its cycles.

    Each cycle first prices a return exactly two mean Earth-Jupiter synodic
    periods after its actual departure.  If that deep-space-maneuver proxy
    exceeds ``threshold_m_s`` the cycle instead flies a three-synodic return,
    which shifts the whole later lattice rather than being scored on its own.

    Args:
        start: First date a departure may occur (ISO date).
        years: Length of the study horizon (Julian years).
        threshold_m_s: Two-synodic maneuver proxy above which the cycle falls
            back to a three-synodic return.
        split_days: Gap by which the growth wave must lead the nozzle wave.

    Returns:
        The flown cycles, in order, whose returns fit inside the horizon.

    Raises:
        ValueError: If the horizon, threshold or split gap is negative.
        RuntimeError: If a window admits no maneuver solution at all.
    """
    if years <= 0.0:
        raise ValueError("years must be positive")
    if threshold_m_s < 0.0:
        raise ValueError("threshold_m_s must be nonnegative")
    if split_days <= 0.0:
        raise ValueError("split_days must be positive")
    start_epoch = Time(start, scale="tdb")
    departure_jd = float(_resonance_epochs(start_epoch, 5.0)[0])
    horizon_end_jd = float(start_epoch.tdb.jd) + years * _DAYS_PER_YEAR
    one_synodic_days = _FIXED_TWO_SYNODIC_DAYS / 2.0
    cycles: List[TwoWaveCycle] = []
    while departure_jd + 2.0 * one_synodic_days <= horizon_end_jd:
        two_return_jd = departure_jd + 2.0 * one_synodic_days
        selected = _cheapest_return(departure_jd, two_return_jd)
        multiple = 2
        return_jd = two_return_jd
        if 1000.0 * selected.dsm > threshold_m_s:
            multiple = 3
            return_jd = departure_jd + 3.0 * one_synodic_days
            if return_jd > horizon_end_jd:
                break
            selected = _cheapest_return(departure_jd, return_jd)
        growth_arrival_jd = return_jd - split_days
        growth = _cheapest_return(departure_jd, growth_arrival_jd)
        cycles.append(
            TwoWaveCycle(
                index=len(cycles),
                departure_jd=departure_jd,
                return_jd=return_jd,
                synodic_multiple=multiple,
                period_years=(return_jd - departure_jd) / _DAYS_PER_YEAR,
                departure_burn=(
                    selected.earth_departure_periapsis_speed - _cycle_periapsis_speed()
                ),
                nozzle_wave_v_b=selected.earth_return_collision_speed,
                nozzle_wave_dsm=selected.dsm,
                split_days=split_days,
                growth_wave_arrival_jd=growth_arrival_jd,
                growth_wave_v_b=growth.earth_return_collision_speed,
                growth_wave_burn=growth.total_dv,
            )
        )
        departure_jd = return_jd
    if not cycles:
        raise RuntimeError("study horizon contains no complete cycle")
    return cycles


def price_cycle(
    cycle: TwoWaveCycle,
    recovery: float,
    fudge: float,
    slug_ratio: Optional[float] = None,
    reversal_period: Optional[u.Quantity] = None,
    geometric_efficiency: Optional[float] = None,
) -> NozzlePricing:
    """Price one flown cycle on the two-wave same-cycle nozzle ledger.

    Both of the cycle's real-orbit correction burns are charged as methalox:
    the growth wave's cost of arriving early, and the nozzle wave's
    deep-space-maneuver proxy.  The parking orbit defaults to the split gap,
    since the payload coasts exactly one of its periods between the two waves.

    Args:
        cycle: The flown cycle to price.
        recovery: Nozzle impulse recovery fraction, ``e``.
        fudge: Elasticity of the growth-push collision, ``f``.
        slug_ratio: Fix ``k``; None optimizes it.
        reversal_period: Override the parking-orbit period; defaults to the
            cycle's own split gap.
        geometric_efficiency: Charge the frozen-dissociation toll, sweeping
            this as the *remaining* jet efficiency.  None (the default) leaves
            ``eta_jet = 1`` and reproduces the published arithmetic, with
            ``recovery`` doing all the derating from outside the momentum
            debit.  Given a value, the head-on nozzle is priced at
            ``eta_jet = eta_chem * geometric_efficiency`` acting on the gross
            jet, and callers sweeping the ledger's way should pass
            ``recovery = 1.0`` so the derating is not applied twice.

    Returns:
        The cycle's pricing, including its growth factor and mass fractions.

    Note:
        ``eta_chem`` is evaluated at the *coldest instant* of the head-on burn,
        which is its start -- the vehicle accelerates into the stream, so ``w``
        and hence ``eta_chem`` only rise from there.  Holding it there through
        the burn understates ``beta`` and so overstates the slug spent, which
        is the conservative direction and the same anchor
        :func:`fleet_ignition_windows` already uses.
    """
    period = cycle.split_days * u.day if reversal_period is None else reversal_period
    jet = 1.0
    if geometric_efficiency is not None:
        jet = geometric_efficiency * chemistry_efficiency(
            coldest_closing_speeds(cycle)[1] * u.km / u.s,
            1.0 if slug_ratio is None else slug_ratio,
        )
    return same_cycle_nozzle(
        growth_collision_speed=cycle.growth_wave_v_b,
        growth_wave_burn=cycle.growth_wave_burn + cycle.nozzle_wave_dsm,
        nozzle_collision_speed=cycle.nozzle_wave_v_b,
        departure_dv=cycle.departure_burn,
        cycle=cycle.period_years,
        exhaust_speed=VE_METHALOX,
        recovery=recovery,
        slug_ratio=slug_ratio,
        fudge=fudge,
        reversal_period=period,
        jet_efficiency=jet,
    )


def coldest_closing_speeds(cycle: TwoWaveCycle) -> Tuple[float, float]:
    """The closing speed at the coldest instant of each of the cycle's burns.

    The overtaking push starts fast and slows as the vehicle runs away from the
    stream, so its cold end is the *finish*.  The head-on burn does the
    opposite -- the vehicle accelerates into the stream -- so its cold end is
    the *start*.  Those are the two speeds the ignition window must hold at.

    Args:
        cycle: The flown cycle.

    Returns:
        ``(growth_push_end, departure_burn_start)`` in km/s.
    """
    v_rf = _cycle_periapsis_speed()
    return cycle.growth_wave_v_b - v_rf, cycle.nozzle_wave_v_b + v_rf


def fleet_ignition_windows(
    cycles: Sequence[TwoWaveCycle],
    temperature: u.Quantity = NOZZLE_FLOOR_TEMPERATURE,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Slug-ratio windows a single fleet-wide loading must satisfy every cycle.

    One nozzle design flies the whole chain (``price_chain``'s convention), so
    the admissible set is the *intersection* of the per-cycle windows.  The
    binding cycles are the three-synodic ones: they close slowest, so they run
    coldest and set the ceiling for everyone.

    Args:
        cycles: The flown cycles.
        temperature: Plume temperature the nozzles require.

    Returns:
        ``(leg1_window, leg2_window)``; either is None if some cycle admits no
        slug ratio at all.
    """
    legs: List[Optional[Tuple[float, float]]] = []
    for index in (0, 1):
        low, high = 0.0, float("inf")
        for cycle in cycles:
            window = slug_ratio_window(
                coldest_closing_speeds(cycle)[index] * u.km / u.s, temperature
            )
            if window is None:
                low, high = 1.0, 0.0
                break
            low, high = max(low, window[0]), min(high, window[1])
        legs.append((low, high) if low < high else None)
    return legs[0], legs[1]


def _best_slug_ratio(
    score: Callable[[float], float],
    bounds: Tuple[float, float] = (_K_SEARCH_MIN, _K_SEARCH_MAX),
) -> float:
    """Slug ratio maximizing ``score``, by coarse grid then continuous refinement.

    The growth-versus-``k`` curve is smooth and single-peaked, so a log-spaced
    bracket followed by a bounded scalar search costs a fraction of a dense
    grid at the same resolution.

    Args:
        score: Chain growth as a function of the slug ratio.
        bounds: Slug ratios to search between.  Defaults to the bare recorded
            box; callers that know which leg they are pricing should pass the
            fleet ignition window intersected with it, since the box alone
            admits ratios whose plume the nozzle cannot grip.

    Returns:
        The maximizing slug ratio, inside the supplied bounds.
    """
    grid = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), _K_SAMPLES)
    values = [score(float(k)) for k in grid]
    peak = int(np.argmax(values))
    low = float(grid[max(0, peak - 1)])
    high = float(grid[min(len(grid) - 1, peak + 1)])
    if high <= low:
        return float(grid[peak])
    refined = minimize_scalar(
        lambda k: -score(float(k)), bounds=(low, high), method="bounded"
    )
    return float(refined.x) if refined.success else float(grid[peak])


@dataclass(frozen=True)
class ChainGrowth:
    """What a whole flown chain multiplies the launched mass by.

    Attributes:
        recovery: Nozzle impulse recovery fraction the chain was priced at.
        fudge: Growth-push elasticity the chain was priced at.
        split_days: Gap between the two waves, and the parking-orbit period.
        slug_ratio: The single ``k`` every cycle's nozzle is built to.
        total_growth: Product of the per-cycle growth factors.
        horizon_years: Departure of the first cycle to return of the last.
        rate: ``ln(total_growth) / horizon_years``, e-foldings per year.
        doubling: Years to double at that rate.
        cycles: Number of flown cycles.
        two_synodic_cycles: How many flew a two-synodic return.
        three_synodic_cycles: How many fell back to three synodic.
    """

    recovery: float
    fudge: float
    split_days: float
    slug_ratio: float
    total_growth: float
    horizon_years: float
    rate: float
    doubling: float
    cycles: int
    two_synodic_cycles: int
    three_synodic_cycles: int

    @property
    def annual_increase(self) -> float:
        """Fractional mass gain per year as a continuous exponential process.

        ``exp(rate) - 1``, so 0.5 means the launched mass grows by 50% a year.
        The real chain is lumpy -- mass arrives in 2.18 and 3.28 year steps --
        so this is the smooth idealization of it, not a per-cycle figure.
        """
        return float(np.expm1(self.rate))

    def mass_after(self, years: float) -> float:
        """Mass multiple after ``years`` at this chain's continuous rate.

        Args:
            years: Length of the projection.

        Returns:
            ``exp(rate * years)``.  At ``years = horizon_years`` this is
            exactly the compounded ``total_growth``; beyond the flown span it
            extrapolates the rate rather than flying more cycles, so it
            assumes the cadence keeps finding equally good windows.
        """
        return float(np.exp(self.rate * years))


def headon_slug_ratio_bounds(
    cycles: Sequence[TwoWaveCycle],
    gate: u.Quantity = NOZZLE_GATE_TEMPERATURE,
) -> Tuple[float, float]:
    """Slug ratios the head-on departure nozzle may actually be built to.

    The recorded search box is a *search* convenience; the physics is the fleet
    ignition window, and the admissible set is their intersection.  The window
    is closed on both sides because the dissipated energy ``w^2 k/(2(1+k)^2)``
    peaks at ``k = 1``: too much slug spreads the energy thin, too little
    dissipates nothing in the merge, and a magnetic nozzle needs a plasma
    either way.

    **Why the gate is looser than the design floor, which reads backwards at
    first.** :data:`NOZZLE_FLOOR_TEMPERATURE` is 15 000 K, the temperature the
    nozzle is *designed* around, and ``two_leg_nozzle_sweep`` holds its legs to
    it.  The gate here is 10 000 K, and it answers a different question --
    below what stagnation temperature does the plume stop being something the
    toll calculation even applies to.  ``puffsat_impact_simulation`` answered
    it by solving the surface: what fails below 10 000 K is *dissociation*, not
    conduction, because the potassium seed keeps supplying electrons long after
    the water stops.  Using the gate rather than the floor here therefore
    bounds the search by what is physical rather than by what is intended.

    Args:
        cycles: The flown cycles.  One nozzle flies all of them, so the window
            is the intersection over the fleet, and the three-synodic cycles
            bind because they close slowest and so run coldest.
        gate: Stagnation temperature the plume must reach at the coldest
            instant of the head-on burn.

    Returns:
        ``(k_min, k_max)`` to search between.

    Raises:
        ValueError: If no slug ratio serves every cycle at this gate.
    """
    window = fleet_ignition_windows(cycles, gate)[1]
    if window is None:
        raise ValueError(f"no slug ratio reaches {gate} on every cycle's head-on burn")
    low, high = max(_K_SEARCH_MIN, window[0]), min(_K_SEARCH_MAX, window[1])
    if low >= high:
        raise ValueError(
            f"ignition window {window} does not meet the search box "
            f"[{_K_SEARCH_MIN}, {_K_SEARCH_MAX}]"
        )
    return low, high


def price_chain(
    cycles: List[TwoWaveCycle],
    recovery: float,
    fudge: float,
    slug_ratio: Optional[float] = None,
    reversal_period: Optional[u.Quantity] = None,
    gate: u.Quantity = NOZZLE_GATE_TEMPERATURE,
    geometric_efficiency: Optional[float] = None,
) -> ChainGrowth:
    """Compound a flown chain's cycles into one growth figure.

    One slug ratio is used for every cycle, because the fleet builds one
    nozzle; when ``slug_ratio`` is None the ratio maximizing the *chain*
    is searched rather than each cycle's own optimum.

    Args:
        cycles: The flown cycles, in order.
        recovery: Nozzle impulse recovery fraction, ``e``.
        fudge: Elasticity of the growth-push collision, ``f``.
        slug_ratio: Fix ``k``; None searches the chain optimum.
        reversal_period: Override the parking-orbit period; defaults per cycle
            to its own split gap.
        gate: Stagnation temperature the head-on plume must reach, which bounds
            the search (see :func:`headon_slug_ratio_bounds`).
        geometric_efficiency: Charge the frozen-dissociation toll and sweep
            this as the remaining jet efficiency; see :func:`price_cycle`.
            **Pass ``recovery = 1.0`` when sweeping it.**  ``recovery`` derates
            from *outside* the momentum debit and this derates from inside, so
            passing the same number to both charges it twice and silently
            returns a much smaller growth -- 7282 rather than 6.289e4 on the
            flown chain at 0.8.  The tolled grid holds ``recovery`` at 1.0 and
            sweeps only this.

    Returns:
        The chain's compounded growth and its per-year rate.

    Raises:
        ValueError: If no cycles are supplied.
    """
    if not cycles:
        raise ValueError("cycles must not be empty")

    def total(k: float) -> float:
        product = 1.0
        for cycle in cycles:
            product *= price_cycle(
                cycle,
                recovery,
                fudge,
                slug_ratio=k,
                reversal_period=reversal_period,
                geometric_efficiency=geometric_efficiency,
            ).growth
        return product

    if slug_ratio is None:
        slug_ratio = _best_slug_ratio(total, headon_slug_ratio_bounds(cycles, gate))
    product = total(slug_ratio)
    horizon = sum(cycle.period_years for cycle in cycles)
    rate = float(np.log(product) / horizon)
    return ChainGrowth(
        recovery=recovery,
        fudge=fudge,
        split_days=cycles[0].split_days,
        slug_ratio=slug_ratio,
        total_growth=product,
        horizon_years=horizon,
        rate=rate,
        doubling=float("inf") if rate <= 0.0 else float(np.log(2.0) / rate),
        cycles=len(cycles),
        two_synodic_cycles=sum(1 for c in cycles if c.synodic_multiple == 2),
        three_synodic_cycles=sum(1 for c in cycles if c.synodic_multiple == 3),
    )


def _cheapest_return(departure_jd: float, return_jd: float) -> _ManeuverSolution:
    """Best maneuver-proxy return joining two fixed Earth epochs.

    Args:
        departure_jd: TDB Julian date of Earth departure.
        return_jd: TDB Julian date the wave must reach Earth.

    Returns:
        The cheapest solution found for the window.

    Raises:
        RuntimeError: If the window admits no solution.
    """
    solutions = _window_maneuver_solutions(
        departure_jd, return_jd, None, cases=("dsm_only",)
    )
    if not solutions:
        raise RuntimeError(f"no return solution for window {departure_jd}->{return_jd}")
    return solutions[0]


@dataclass(frozen=True)
class TwoWaveGrowthAnalysis:
    """Tables behind the real-orbit two-wave growth comparison.

    Attributes:
        cycles: One row per flown cycle, at the selected split gap.
        sweep: One row per ``(recovery, fudge)`` pair, with the chain optimum
            slug ratio and the growth it delivers.  This is the *published*
            column: ``e`` derates the net impulse from outside the momentum
            debit and no chemistry is charged.
        tolled: The same grid re-priced with the frozen-dissociation toll of
            ADR 0016, so the swept axis is ``eta_geom`` and the jet efficiency
            actually flown is ``eta_chem * eta_geom``.  ``eta_chem`` is
            reported per row because it is the *ceiling* on ``eta_jet``: a row
            asking for ``eta_geom = 0.9`` is really asking the nozzle for
            ``0.9 eta_chem``, which is what tells a reader whether the row is
            reachable.  This is what ``tab:space_mortgage_growth`` regenerates
            to, since its growth push keeps a plate and a plate owes no
            chemistry.
        split_trade: One row per candidate split gap, showing the perijove
            burn it costs against the apoapsis reversal it saves.
        split_days: The split gap the ``cycles`` and ``sweep`` tables use.
        horizon_years: Departure of the first cycle to return of the last.
        ephemeris: Ephemeris provider behind the real-orbit states.
    """

    cycles: pd.DataFrame
    sweep: pd.DataFrame
    tolled: pd.DataFrame
    split_trade: pd.DataFrame
    split_days: float
    horizon_years: float
    ephemeris: str


def analyze_two_wave_growth(
    start: str = _DEFAULT_START,
    years: float = _DEFAULT_HORIZON_YEARS,
    threshold_m_s: float = _DEFAULT_THRESHOLD_M_S,
    recoveries: Sequence[float] = _DEFAULT_RECOVERIES,
    fudges: Sequence[float] = _DEFAULT_FUDGES,
    split_options: Sequence[float] = _DEFAULT_SPLIT_OPTIONS,
) -> TwoWaveGrowthAnalysis:
    """Fly the adaptive cadence and price it across recovery, elasticity and split.

    The split gap is chosen first, on growth at the mid-sweep operating point,
    because it is a hardware decision rather than a per-cell one: it sets both
    the perijove burn the growth wave pays and the parking orbit the apoapsis
    reversal is sized from.

    Args:
        start: First date a departure may occur (ISO date).
        years: Length of the study horizon (Julian years).
        threshold_m_s: Two-synodic maneuver proxy above which a cycle falls
            back to three synodic.
        recoveries: Nozzle impulse recovery fractions to sweep.
        fudges: Growth-push elasticities to sweep.
        split_options: Candidate split gaps, in days.

    Returns:
        The per-cycle, sweep and split-trade tables.
    """
    trade_rows = []
    chains = {}
    for split in split_options:
        cycles = adaptive_two_wave_cycles(start, years, threshold_m_s, split)
        chains[split] = cycles
        reference = price_chain(cycles, _REFERENCE_RECOVERY, _REFERENCE_FUDGE)
        trade_rows.append(
            {
                "split_days": split,
                "growth_wave_burn_km_s": float(
                    np.mean([c.growth_wave_burn for c in cycles])
                ),
                "apoapsis_reversal_m_s": float(
                    apoapsis_reversal_dv(split * u.day).to_value(u.m / u.s)
                ),
                "slug_ratio": reference.slug_ratio,
                "total_growth": reference.total_growth,
                "e_foldings_per_year": reference.rate,
                "doubling_years": reference.doubling,
            }
        )
    split_trade = pd.DataFrame(trade_rows)
    best_split = float(
        split_trade.loc[split_trade["e_foldings_per_year"].idxmax(), "split_days"]
    )
    cycles = chains[best_split]

    sweep_rows = []
    for fudge in fudges:
        for recovery in recoveries:
            chain = price_chain(cycles, recovery, fudge)
            sweep_rows.append(
                {
                    "recovery": recovery,
                    "fudge": fudge,
                    "slug_ratio": chain.slug_ratio,
                    "total_growth": chain.total_growth,
                    "annual_increase_pct": 100.0 * chain.annual_increase,
                    "e_foldings_per_year": chain.rate,
                    "doubling_years": chain.doubling,
                    "mass_after_30_yr": chain.mass_after(_PROJECTION_YEARS),
                }
            )

    tolled_rows = []
    for fudge in fudges:
        for geometric in recoveries:
            chain = price_chain(cycles, 1.0, fudge, geometric_efficiency=geometric)
            ceiling = min(
                chemistry_efficiency(
                    coldest_closing_speeds(cycle)[1] * u.km / u.s, chain.slug_ratio
                )
                for cycle in cycles
            )
            tolled_rows.append(
                {
                    "eta_geom": geometric,
                    "fudge": fudge,
                    "eta_chem_floor": ceiling,
                    "eta_jet": ceiling * geometric,
                    "slug_ratio": chain.slug_ratio,
                    "total_growth": chain.total_growth,
                    "e_foldings_per_year": chain.rate,
                    "doubling_years": chain.doubling,
                    "mass_after_30_yr": chain.mass_after(_PROJECTION_YEARS),
                }
            )

    cycle_rows = []
    for cycle in cycles:
        priced = price_cycle(cycle, _REFERENCE_RECOVERY, _REFERENCE_FUDGE)
        cycle_rows.append(
            {
                "cycle": cycle.index,
                "departure_tdb": Time(
                    cycle.departure_jd, format="jd", scale="tdb"
                ).isot[:10],
                "return_tdb": Time(cycle.return_jd, format="jd", scale="tdb").isot[:10],
                "synodic": cycle.synodic_multiple,
                "period_years": cycle.period_years,
                "departure_burn_km_s": cycle.departure_burn,
                "growth_wave_v_b_km_s": cycle.growth_wave_v_b,
                "growth_wave_burn_km_s": cycle.growth_wave_burn,
                "nozzle_wave_v_b_km_s": cycle.nozzle_wave_v_b,
                "nozzle_wave_dsm_m_s": 1000.0 * cycle.nozzle_wave_dsm,
                "growth": priced.growth,
            }
        )
    return TwoWaveGrowthAnalysis(
        cycles=pd.DataFrame(cycle_rows),
        sweep=pd.DataFrame(sweep_rows),
        tolled=pd.DataFrame(tolled_rows),
        split_trade=split_trade,
        split_days=best_split,
        horizon_years=sum(c.period_years for c in cycles),
        ephemeris=_EPHEMERIS,
    )


def _format(frame: pd.DataFrame, formats: Dict[str, str]) -> str:
    """Render a frame with per-column formatting and no index.

    Args:
        frame: The table to render.
        formats: Format spec per column name.

    Returns:
        The rendered table.
    """
    shown = frame.copy()
    for column, spec in formats.items():
        shown[column] = shown[column].map(lambda v, s=spec: format(v, s))
    return str(shown.to_string(index=False))


def main() -> None:
    """Print the real-orbit two-wave growth tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=_DEFAULT_START, help="first departure date")
    parser.add_argument(
        "--years", type=float, default=_DEFAULT_HORIZON_YEARS, help="study horizon"
    )
    parser.add_argument(
        "--threshold-m-s",
        type=float,
        default=_DEFAULT_THRESHOLD_M_S,
        help="2S maneuver proxy above which a cycle falls back to 3S",
    )
    parser.add_argument("--csv", default=None, help="write the sweep table here")
    args = parser.parse_args()

    analysis = analyze_two_wave_growth(
        start=args.start, years=args.years, threshold_m_s=args.threshold_m_s
    )
    print("Real-orbit two-wave nozzle growth on the adaptive 2S/3S cadence")
    print(
        f"start {args.start}; horizon {args.years:.0f} yr; ephemeris "
        f"{analysis.ephemeris}; 3S fallback above {args.threshold_m_s:.0f} m/s; "
        f"methalox v_e {VE_METHALOX:.4f} km/s"
    )
    print(
        f"slug-ratio search box k in [{_K_SEARCH_MIN}, {_K_SEARCH_MAX}], "
        f"{_K_SAMPLES} log-spaced starts then bounded refinement\n"
    )

    print("=== Split-gap trade (priced at e = 0.6, f = 0.8) ===")
    print(
        "The gap is the parking-orbit period, so it buys a cheaper apoapsis "
        "reversal\nand pays for it in the growth wave's perijove burn."
    )
    print(
        _format(
            analysis.split_trade,
            {
                "split_days": ".0f",
                "growth_wave_burn_km_s": ".4f",
                "apoapsis_reversal_m_s": ".1f",
                "slug_ratio": ".2f",
                "total_growth": ".4g",
                "e_foldings_per_year": ".4f",
                "doubling_years": ".3f",
            },
        )
    )
    print(f"\n-> flying the {analysis.split_days:.0f} d split\n")

    print(f"=== Flown chain, {analysis.split_days:.0f} d split (e = 0.6, f = 0.8) ===")
    print(
        _format(
            analysis.cycles,
            {
                "period_years": ".4f",
                "departure_burn_km_s": ".4f",
                "growth_wave_v_b_km_s": ".3f",
                "growth_wave_burn_km_s": ".4f",
                "nozzle_wave_v_b_km_s": ".3f",
                "nozzle_wave_dsm_m_s": ".3f",
                "growth": ".4f",
            },
        )
    )
    print(f"\nflown span {analysis.horizon_years:.4f} yr\n")

    print(
        "=== Growth with the ADR 0016 dissociation toll charged, by eta_geom "
        "and elasticity f ==="
    )
    print(
        "    eta_jet = eta_chem * eta_geom; eta_chem is the ceiling, so a row's\n"
        "    eta_geom is only reachable if the nozzle can deliver eta_jet."
    )
    print(
        _format(
            analysis.tolled,
            {
                "eta_geom": ".2f",
                "fudge": ".2f",
                "eta_chem_floor": ".4f",
                "eta_jet": ".4f",
                "slug_ratio": ".2f",
                "total_growth": ".4g",
                "e_foldings_per_year": ".4f",
                "doubling_years": ".3f",
                "mass_after_30_yr": ".4g",
            },
        )
    )

    print("\n=== Growth over the flown chain, by recovery e and elasticity f ===")
    print(
        _format(
            analysis.sweep,
            {
                "recovery": ".2f",
                "fudge": ".2f",
                "slug_ratio": ".2f",
                "total_growth": ".4g",
                "annual_increase_pct": "+.2f",
                "e_foldings_per_year": ".4f",
                "doubling_years": ".3f",
                "mass_after_30_yr": ".4g",
            },
        )
    )
    print(
        "\ntotal_growth is the flown chain compounded over its own span; "
        f"mass_after_30_yr runs the\ncontinuous rate out to {_PROJECTION_YEARS:.0f} "
        "years, which assumes the cadence keeps finding\nequally good windows "
        "past the flown span."
    )
    if args.csv is not None:
        analysis.sweep.to_csv(args.csv, index=False)
        print(f"\nsweep written to {args.csv}")


if __name__ == "__main__":
    main()
