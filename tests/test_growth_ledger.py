"""Tests for src/growth_ledger.py: the 1500 t launch unit through one cycle."""

import dataclasses
from typing import Optional

import numpy as np
import pytest
from astropy import units as u
from boinor.bodies import Earth

from src.chamber_departure import best_departure, growth_per_cycle, square_law_loss
from src.chamber_isp import GATE_THRUST_COST, HYDROGEN_5500K, METHANE_7000K, PLUG_RATIO
from src.growth_ledger import (
    ADR_0033_PLATE,
    DEFAULT_PARKING_DAYS,
    LAUNCH_UNIT,
    METHALOX_TANK_FRACTION,
    METHANE_PITCH,
    METHANE_PITCH_RANGE,
    PAPER_SLUG_RATIO,
    PLATE_DESIGNS,
    PLATE_MASS,
    PLUG,
    RAPTOR3_MASS,
    SPRAY_CUP,
    best_cycle_growth,
    chain_growth,
    departure_at_altitude,
    methalox_departure,
    periapsis_raise,
    pitch_ratio,
    pitch_sweep,
    price_cycle_growth,
    price_methalox_cycle,
    summarize_chain,
)
from src.jovian_flyby import puffsat_cycle_periapsis_speed
from src.nozzle_analysis import apoapsis_reversal_dv
from src.two_wave_growth import VE_METHALOX, TwoWaveCycle, adaptive_two_wave_cycles
from src.water_plate import (
    ARGON_SLUG,
    NO_BONDS,
    WATER_BOND_ENERGY,
    WATER_SLUG,
    optimal_plate_push,
)

KM_S = u.km / u.s


@pytest.mark.parametrize("burn_200, premium_m_s", [(5.32, 107.0), (5.54, 110.0)])
def test_departing_from_600_km_costs_the_parents_premium(
    burn_200: float, premium_m_s: float
) -> None:
    """``sec:jovian_meeting_altitudes``: for the same excess speed, the three-synodic
    departures need 107-110 m/s more at 600 km than at 200 km (less Oberth)."""
    burn = departure_at_altitude(burn_200 * KM_S, 600.0 * u.km).burn
    assert (burn.to_value(u.m / u.s) - 1e3 * burn_200) == pytest.approx(
        premium_m_s, abs=1.0
    )


def _cycle(
    multiple: int,
    burn: float,
    nozzle_v_b: float,
    growth_v_b: float,
    onward: Optional[float] = None,
) -> TwoWaveCycle:
    """A cycle whose payload departs on ``onward`` (default: a repeat of ``burn``)."""
    period = {2: 2.184, 3: 3.276}[multiple]
    return TwoWaveCycle(
        index=0, departure_jd=0.0, return_jd=0.0, synodic_multiple=multiple,
        period_years=period, departure_burn=burn, nozzle_wave_v_b=nozzle_v_b,
        nozzle_wave_dsm=0.0, split_days=20.0, growth_wave_arrival_jd=0.0,
        growth_wave_v_b=growth_v_b, growth_wave_burn=0.0,
        onward_burn=burn if onward is None else onward,
    )  # fmt: skip


def _at(speed_200: float, altitude_km: float) -> float:
    """A wave's speed at another altitude, by energy conservation."""
    mu = float(Earth.k.to_value(u.km**3 / u.s**2))
    r200, r = (float((Earth.R + h * u.km).to_value(u.km)) for h in (200.0, altitude_km))
    return float(np.sqrt(speed_200**2 + 2.0 * mu * (1.0 / r - 1.0 / r200)))


def test_the_launch_unit_is_pushed_at_400_km_and_departs_from_600_km() -> None:
    """1500 t is pushed from rest to the 400 km cycle-orbit speed by the growth wave.
    At the 613 000 km apoapsis it raises periapsis to 600 km (1.7 m/s) and reverses
    the ellipse so the departure is prograde (ADR 0009, 234 m/s), both in methalox.
    It drops the 150 t plate and the
    empty water tank, and departs the rest into the nozzle wave at 600 km through
    the gated chamber.  The plate sprays argon onto ice PuffSats by default."""
    cycle = _cycle(3, 5.33, 55.4, 57.4)
    grown = price_cycle_growth(
        cycle, 0.7, METHANE_7000K, 0.7, initial_water_price=0.05,
        loss_model=square_law_loss,
    )  # fmt: skip
    push = optimal_plate_push(
        _at(57.4, 400.0) * KM_S, puffsat_cycle_periapsis_speed(altitude=400.0 * u.km),
        0.7, 0.05, slug=ARGON_SLUG,
    )  # fmt: skip
    unit = LAUNCH_UNIT.to_value(u.t)
    assert grown.puffsats.to_value(u.t) == pytest.approx(
        push.puffsat_fraction * unit, rel=1e-9
    )
    methalox = (periapsis_raise(20.0 * u.day) + apoapsis_reversal_dv()).to_value(
        u.km / u.s
    )
    raised = push.delivered_fraction * unit * np.exp(-methalox / VE_METHALOX)
    stack = (
        raised
        - PLATE_MASS.to_value(u.t)
        - ARGON_SLUG.tank_fraction * push.slug_fraction * unit
    )
    assert grown.departing_stack.to_value(u.t) == pytest.approx(stack, rel=1e-9)
    departure = departure_at_altitude(5.33 * KM_S, 600.0 * u.km)
    alone = best_departure(
        grown.departing_stack, departure.start_speed, departure.burn,
        _at(55.4, 600.0) * KM_S, METHANE_7000K, 0.7,
        gate_thrust_cost=GATE_THRUST_COST, loss_model=square_law_loss,
    )  # fmt: skip
    assert grown.departure == alone
    expected = growth_per_cycle(
        stack / grown.puffsats.to_value(u.t),
        alone.rod_mass_fraction,
        alone.delivered_net,
    )
    assert grown.growth == pytest.approx(expected, rel=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize(
    "cycle", [_cycle(3, 5.33, 55.4, 57.4), _cycle(2, 7.17, 61.4, 63.6)]
)
def test_each_cycle_flies_the_water_schedule_that_grows_it_most(
    cycle: TwoWaveCycle,
) -> None:
    """One number, the starting water price, fixes the plate's whole schedule, so the
    search is one-dimensional; no nearby price grows the cycle more."""
    best = best_cycle_growth(cycle, 0.7, METHANE_7000K, 0.7, loss_model=square_law_loss)
    start = best.initial_water_price
    for other in np.geomspace(start / 3.0, start * 3.0, 25):
        rival = price_cycle_growth(
            cycle, 0.7, METHANE_7000K, 0.7, float(other), loss_model=square_law_loss
        )
        assert rival.growth <= best.growth + 1e-9
    assert best.push.slug_ratio_start > best.push.slug_ratio_end


def test_the_chain_summary_reports_ten_years_both_ways() -> None:
    """Three cycles of 3.0, 2.0 and 6.0 yr growing 4x, 2x and 3x: 24x over 11 yr.
    Stepwise, only the first two finish inside ten years (8x); the continuous rate
    projects exp(10 ln 24 / 11)."""
    summary = summarize_chain([3.0, 2.0, 6.0], [4.0, 2.0, 3.0])
    rate = np.log(24.0) / 11.0
    assert summary.rate_per_year == pytest.approx(rate, rel=1e-12)
    assert summary.doubling_years == pytest.approx(np.log(2.0) / rate, rel=1e-12)
    assert summary.annual_growth == pytest.approx(np.expm1(rate), rel=1e-12)
    assert summary.ten_year_stepwise == pytest.approx(8.0, rel=1e-12)
    assert summary.ten_year_continuous == pytest.approx(np.exp(10.0 * rate), rel=1e-12)


@pytest.mark.slow
def test_the_flown_chain_grows_every_cycle() -> None:
    """Methane at eta 0.7 behind a 0.7 plate grows the fleet on all eleven cycles."""
    cycles = adaptive_two_wave_cycles(split_days=DEFAULT_PARKING_DAYS)
    grown = chain_growth(cycles, 0.7, METHANE_7000K, 0.7)
    assert len(grown) == 11 and all(g.growth > 1.0 for g in grown)


@pytest.mark.slow
def test_the_plate_is_held_to_k_10_and_freeing_it_is_the_sensitivity() -> None:
    cycle = _cycle(2, 7.17, 61.4, 63.6)
    capped = best_cycle_growth(
        cycle, 0.7, METHANE_7000K, 0.7, loss_model=square_law_loss
    )
    assert capped.push.slug_ratio_start <= 10.0
    free = best_cycle_growth(
        cycle, 0.7, METHANE_7000K, 0.7, loss_model=square_law_loss, max_slug_ratio=None
    )
    assert free.push.slug_ratio_start > 10.0
    assert free.growth >= capped.growth


def test_each_wave_pays_its_own_correction_burn_in_methalox() -> None:
    """The growth wave burns methalox to arrive early and the nozzle wave to correct
    its return (the chain's DSM proxy), so the batch that left Jupiter was larger
    than what arrives: growth-wave PuffSats by ``exp(burn / v_e)``, rods by
    ``exp(dsm / v_e)``."""
    base = _cycle(3, 5.33, 55.4, 57.4)
    corrected = dataclasses.replace(base, growth_wave_burn=0.30, nozzle_wave_dsm=0.04)
    args = (0.7, METHANE_7000K, 0.7)
    free = price_cycle_growth(base, *args, 0.05, loss_model=square_law_loss)
    paid = price_cycle_growth(corrected, *args, 0.05, loss_model=square_law_loss)
    d_growth, d_rods = np.exp(-0.30 / VE_METHALOX), np.exp(-0.04 / VE_METHALOX)
    stack = paid.departing_stack.to_value(u.t)
    arrived = paid.puffsats.to_value(u.t)
    rods = paid.departure.rod_mass_fraction * stack
    expected = (
        paid.departure.delivered_net * stack / (arrived / d_growth + rods / d_rods)
    )
    assert paid.growth == pytest.approx(expected, rel=1e-12)
    assert paid.growth < free.growth


def test_the_pushed_unit_departs_on_the_next_windows_burn() -> None:
    """Return n's waves push a payload that leaves on cycle n + 1, so it flies
    the onward burn, not the burn this cycle's own batch left Earth on."""
    three_to_two = _cycle(3, 5.33, 55.4, 57.4, onward=7.17)
    args = (0.7, METHANE_7000K, 0.7, 0.05)
    flown = price_cycle_growth(three_to_two, *args, loss_model=square_law_loss)
    repeat = _cycle(3, 7.17, 55.4, 57.4)
    expected = price_cycle_growth(repeat, *args, loss_model=square_law_loss)
    assert flown.growth == pytest.approx(expected.growth, rel=1e-12)
    methalox = price_methalox_cycle(three_to_two, 0.7, 0.05, loss_model=square_law_loss)
    expected_methalox = price_methalox_cycle(
        repeat, 0.7, 0.05, loss_model=square_law_loss
    )
    assert methalox.growth == pytest.approx(expected_methalox.growth, rel=1e-12)


def test_hydrogen_pays_its_cryostats_and_boil_off_before_it_departs() -> None:
    """The hydrogen is launched in cryostats and held through the parking orbit,
    losing ``b`` of itself; the cryostats (``c`` per kilogram launched) drop with
    the plate.  Neither departs, so they come out of the stack, and the hydrogen
    launched is what the departure burns over ``1 - b``."""
    cycle = _cycle(3, 5.33, 55.4, 57.4)
    args = (cycle, 0.7, HYDROGEN_5500K, 0.858, 0.05)
    bare = price_cycle_growth(*args, loss_model=square_law_loss)
    held = price_cycle_growth(
        *args, loss_model=square_law_loss, cryostat_fraction=0.03, boil_off=0.03
    )
    dep = held.departure
    stack = held.departing_stack.to_value(u.t)
    burned = (1.0 - dep.delivered_fraction - PLUG_RATIO * dep.rod_mass_fraction) * stack
    launched = burned / (1.0 - 0.03)
    assert held.cryostats.to_value(u.t) == pytest.approx(0.03 * launched, rel=1e-6)
    assert held.boiled_off.to_value(u.t) == pytest.approx(0.03 * launched, rel=1e-6)
    before = stack + held.cryostats.to_value(u.t) + held.boiled_off.to_value(u.t)
    assert before == pytest.approx(bare.departing_stack.to_value(u.t), rel=1e-6)
    assert held.growth < bare.growth


def test_the_methalox_incumbent_pushes_with_the_whole_batch() -> None:
    """No departure wave, so the whole returning batch pushes (at the full return's
    speed, paying only its DSM proxy), and Raptor 3s depart the stack on a steered
    burn.  The engine count delivers more than one fewer or one more."""
    cycle = dataclasses.replace(_cycle(3, 5.33, 55.4, 57.4), nozzle_wave_dsm=0.04)
    flown = price_methalox_cycle(cycle, 0.7, 0.05, loss_model=square_law_loss)
    stack = flown.departing_stack.to_value(u.t)
    batch = flown.puffsats.to_value(u.t) / np.exp(-0.04 / VE_METHALOX)
    assert flown.growth == pytest.approx(flown.delivered_net * stack / batch, rel=1e-12)
    burn = departure_at_altitude(5.33 * KM_S, 600.0 * u.km).burn
    for engines in (flown.engines - 1, flown.engines + 1):
        rival = methalox_departure(
            flown.departing_stack, burn, engines, square_law_loss
        )
        assert rival.delivered_net < flown.delivered_net
    spent = 1.0 - np.exp(
        -(burn + flown.finite_burn_loss).to_value(u.km / u.s) / VE_METHALOX
    )
    hardware = (
        METHALOX_TANK_FRACTION * spent
        + flown.engines * RAPTOR3_MASS.to_value(u.t) / stack
    )
    assert flown.delivered_net == pytest.approx(1.0 - spent - hardware, rel=1e-9)


def test_an_argon_plate_pays_argons_tank_and_no_slug_bonds() -> None:
    cycle = _cycle(3, 5.33, 55.4, 57.4)
    grown = price_cycle_growth(
        cycle, 0.7, METHANE_7000K, 0.7, 0.05, loss_model=square_law_loss,
        slug=ARGON_SLUG,
    )  # fmt: skip
    push = optimal_plate_push(
        _at(57.4, 400.0) * KM_S, puffsat_cycle_periapsis_speed(altitude=400.0 * u.km),
        0.7, 0.05, slug=ARGON_SLUG,
    )  # fmt: skip
    assert grown.push == push
    unit = LAUNCH_UNIT.to_value(u.t)
    methalox = (periapsis_raise(20.0 * u.day) + apoapsis_reversal_dv()).to_value(
        u.km / u.s
    )
    stack = (
        push.delivered_fraction * unit * np.exp(-methalox / VE_METHALOX)
        - PLATE_MASS.to_value(u.t)
        - 14.6 / 1395.0 * push.slug_fraction * unit
    )
    assert grown.departing_stack.to_value(u.t) == pytest.approx(stack, rel=1e-9)


def test_raising_periapsis_at_apoapsis_costs_the_parents_1_7_m_s_on_the_20_day_orbit() -> (
    None
):
    """``sec:jovian_meeting_altitudes``: 400 km -> 600 km at the 613 000 km apoapsis of
    the 20-day orbit costs 1.7 m/s.  A 10-day orbit's apoapsis is lower and faster, so
    the same raise costs more there."""
    assert periapsis_raise(20.0 * u.day).to_value(u.m / u.s) == pytest.approx(
        1.7, abs=0.05
    )
    assert periapsis_raise(10.0 * u.day) > periapsis_raise(20.0 * u.day)


@pytest.mark.parametrize("split", [10.0, 20.0])
def test_the_parking_orbit_is_the_cycles_own_split(split: float) -> None:
    """CONTEXT.md: the split gap *is* the parking-orbit period -- the payload is pushed
    at periapsis, coasts one orbit while the departure wave catches up, and departs at
    the next periapsis.  So the push target, the raise, the reversal and the departure's
    start all come from the cycle's own ``split_days``, not from a fixed 20 days."""
    cycle = dataclasses.replace(_cycle(3, 5.33, 55.4, 57.4), split_days=split)
    period = split * u.day
    grown = price_cycle_growth(
        cycle, 0.7, METHANE_7000K, 0.7, 0.05, loss_model=square_law_loss
    )
    push = optimal_plate_push(
        _at(57.4, 400.0) * KM_S,
        puffsat_cycle_periapsis_speed(period=period, altitude=400.0 * u.km),
        0.7, 0.05, slug=ARGON_SLUG,
    )  # fmt: skip
    assert grown.push == push
    methalox = (periapsis_raise(period) + apoapsis_reversal_dv(period)).to_value(
        u.km / u.s
    )
    unit = LAUNCH_UNIT.to_value(u.t)
    stack = (
        push.delivered_fraction * unit * np.exp(-methalox / VE_METHALOX)
        - PLATE_MASS.to_value(u.t)
        - ARGON_SLUG.tank_fraction * push.slug_fraction * unit
    )
    assert grown.departing_stack.to_value(u.t) == pytest.approx(stack, rel=1e-9)
    departure = departure_at_altitude(5.33 * KM_S, 600.0 * u.km, period)
    assert departure.start_speed == puffsat_cycle_periapsis_speed(
        period=period, altitude=600.0 * u.km
    ).to(u.km / u.s)


@pytest.mark.slow
def test_the_pitch_sweep_brackets_the_pitch_the_matrix_carries() -> None:
    """The paper's "1.4 to 5.6 kg per pulse moves its doubling by 0.01 to 0.02 yr"
    comes from this sweep; its heavy end is the pitch the matrix carries.

    No sign is asserted on ``saved yr``: on the flown chain the CH4 50% row
    behind a 1.0 plate comes out 0.006 yr *slower* at the lighter pitch.
    """
    assert METHANE_PITCH == METHANE_PITCH_RANGE[1]
    cycles = [_cycle(3, 5.33, 55.4, 57.4), _cycle(2, 7.17, 61.4, 63.6)]
    rows = pitch_sweep(cycles, loss_model=square_law_loss)
    assert len(rows) == 12
    for row in rows:
        assert row["saved yr"] == pytest.approx(row["5.6 kg yr"] - row["1.4 kg yr"])
    solved = next(r for r in rows if r["plate"] == 0.7 and "solved" in r["departure"])
    grown = chain_growth(
        cycles, 0.7, METHANE_7000K, 0.538, pitch_ratio(METHANE_PITCH),
        loss_model=square_law_loss,
    )  # fmt: skip
    periods = [c.period_years for c in cycles]
    expected = summarize_chain(periods, [g.growth for g in grown]).doubling_years
    assert solved["5.6 kg yr"] == pytest.approx(expected, rel=1e-12)


def test_the_impact_sims_plates_are_all_in_and_capped_at_the_papers_k() -> None:
    """ADR 0041: the handoff's eta_jet already charges the PuffSat's bonds, so
    the ledger takes eta = eta_jet^2 with no bonds; ADR 0033's plate keeps its
    0.7 net of bonds charged per pulse, at k <= 10."""
    assert SPRAY_CUP.efficiency == pytest.approx(0.36)
    assert PLUG.efficiency == pytest.approx(0.49)
    for plate in PLATE_DESIGNS:
        if plate is ADR_0033_PLATE:
            continue
        assert plate.impactor_bond_energy == NO_BONDS
        assert plate.max_slug_ratio == PAPER_SLUG_RATIO
    assert ADR_0033_PLATE.efficiency == 0.7
    assert ADR_0033_PLATE.impactor_bond_energy == WATER_BOND_ENERGY
    assert ADR_0033_PLATE.max_slug_ratio == 10.0


def test_the_spray_cup_grows_slower_than_the_plug() -> None:
    cycle = _cycle(3, 5.33, 55.4, 57.4)

    def growth(plate) -> float:  # type: ignore[no-untyped-def]
        return price_cycle_growth(
            cycle, plate.efficiency, METHANE_7000K, 0.538, 0.05,
            loss_model=square_law_loss, max_slug_ratio=plate.max_slug_ratio,
            impactor_bond_energy=plate.impactor_bond_energy,
        ).growth  # fmt: skip

    assert 1.0 < growth(SPRAY_CUP) < growth(PLUG)


def test_every_plate_design_has_a_command_line_name() -> None:
    """ADR 0042: ``--plate`` and ``--designs-grid`` reach every design."""
    from src.growth_ledger import PLATE_DESIGNS, PLATE_DESIGNS_BY_NAME

    assert set(PLATE_DESIGNS_BY_NAME.values()) == set(PLATE_DESIGNS)
    assert PLATE_DESIGNS_BY_NAME["spray-cup"].jet_efficiency == pytest.approx(0.60)
