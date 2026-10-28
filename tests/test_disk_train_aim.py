"""Tests for aiming the Sun-wave disk train (src/disk_train_aim.py)."""

import math
from dataclasses import replace

import pytest

from src import solar_dive_growth as sdg
from src.disk_train_aim import (
    CME_DENSEST_N,
    CME_TILTS,
    DEEP_CLOSURE,
    HEADLINE,
    HEADLINE_X12,
    PROPOSAL,
    THICK_X13,
    THIN_MATCHED,
    THIN_X12,
    THIN_X13,
    Disk,
    blind_miss,
    cme_drift,
    dispersion_budget,
    dispersion_requirement,
    edge_sunlight_share,
    hold_fraction,
    kick_knowledge,
    loop_gain,
    mean_coupling_loss,
    price,
    short_loop_window,
    simulate_blind_miss,
    struck_share,
    sunlight_pressure,
)

# The simulation's own disk (cme_drift.py @ 3581fbd): 1 kg, 0.5 m, 2.8 mm.
_SIM_DISK = Disk("sim", 1.0, 2.0, 0.5, 2.8e-3, THIN_MATCHED)
_SIM_PERIAPSIS_SPEED = 195846.1074478761


def test_blind_miss_matches_the_monte_carlo() -> None:
    kicks = [3.45] * 600
    for blind in (10.0, 60.0):
        formula = blind_miss(0.01, 3.45, 1.0, blind)
        simulated, _ = simulate_blind_miss(0.01, kicks, 1.0, blind, hold=False)
        assert simulated == pytest.approx(formula, rel=0.1)


def test_holding_spends_the_mean_side_impulse() -> None:
    kicks = [2.0] * 2000
    _, spent = simulate_blind_miss(0.01, kicks, 1.0, 60.0, hold=True)
    assert spent == pytest.approx(0.01 * math.sqrt(2.0 / math.pi) * 4000.0, rel=0.03)


def test_the_parents_one_milliradian_wander_is_reproduced() -> None:
    # sec:sun_wave_ranging: 1 mrad pulse to pulse, 2.8 m/s^2 at 2 Hz, a minute
    # out: "about half a meter".
    assert blind_miss(1e-3, 2.8, 0.5, 60.0) == pytest.approx(0.53, abs=0.03)


def test_the_parents_blind_minute_is_unstable_at_centimetres() -> None:
    assert loop_gain(PROPOSAL, 3.77, 60.0) > 100.0
    window = short_loop_window(PROPOSAL, 3.77)
    assert loop_gain(PROPOSAL, 3.77, window) <= 0.5
    assert loop_gain(PROPOSAL, 3.77, window + 1.0) > 0.5
    assert window <= 2.0


def test_requirement_is_the_bore_clip() -> None:
    sigma = dispersion_requirement(THICK_X13)
    assert sigma == pytest.approx(0.05981 / math.sqrt(2.0 * math.log(1e3)), rel=1e-3)
    assert math.exp(-(THICK_X13.clearance**2) / (2.0 * sigma**2)) == pytest.approx(1e-3)


def test_matched_charge_loses_its_lens() -> None:
    assert mean_coupling_loss(THICK_X13, 0.02) == 0.0
    # 2 d / (pi R) at d = 1.5 cm on 0.19 m is Result 23's 5% line.
    d = 0.015
    sigma = d / math.sqrt(math.pi / 2.0)
    loss = mean_coupling_loss(replace(THIN_MATCHED, lens_radius=0.19), sigma)
    assert loss == pytest.approx(0.050, abs=0.001)


def test_overhang_reproduces_result_24s_lone_lens() -> None:
    # thin_disk_side.csv, x1.20: caught whole at 4 cm, 0.98489 struck at 5 cm.
    r, c = THIN_X12.disk_radius, THIN_X12.charge_radius
    assert struck_share(0.04, r, c) == pytest.approx(0.99998, abs=2e-5)
    assert struck_share(0.05, r, c) == pytest.approx(0.98489, rel=1e-4)
    assert struck_share(0.0, r, c) == 1.0


def test_the_x12_charge_keeps_its_coupling_at_the_required_dispersion() -> None:
    # Whole to 4.0 cm against a 1.36 cm sigma: the overhang costs under 0.01%.
    sigma = dispersion_requirement(THIN_X12)
    assert sigma == pytest.approx(0.0136, abs=1e-4)
    assert 0.0 < mean_coupling_loss(THIN_X12, sigma) < 1e-4
    assert mean_coupling_loss(THIN_X13, 0.02) == 0.0
    assert THIN_X12.coupling > THIN_X13.coupling


def test_hold_fraction_sums_side_and_torque() -> None:
    sigma = 0.016
    mean = sigma * math.sqrt(math.pi / 2.0)
    assert hold_fraction(PROPOSAL, sigma, lever=13.0) == pytest.approx(
        mean * (THICK_X13.side_slope + 1.0 / 13.0)
    )


def test_kick_knowledge_inverts_the_walk() -> None:
    sigma = kick_knowledge(0.01, 1.0, 20.0)
    assert blind_miss(sigma, 1.0, 1.0, 20.0) == pytest.approx(0.01)


def test_cme_law_reproduces_the_simulations_table() -> None:
    # cme_drift_answers_for_parent.md, 9.86 R_sun, densest x 2.0 at 2500 km/s.
    for tilt, dv, drift in (
        (0.0, 0.0461527192496872, 207.6872366235924),
        (23.43864527899927, 2.6092108683253286, 11741.448907463979),
    ):
        d = cme_drift(_SIM_DISK, CME_DENSEST_N, 2500.0e3, tilt, _SIM_PERIAPSIS_SPEED)
        assert d.dv == pytest.approx(dv, rel=1e-3)
        assert d.drift == pytest.approx(drift, rel=1e-3)


def test_edge_sunlight_reproduces_the_simulation() -> None:
    assert edge_sunlight_share(_SIM_DISK, 9.86) == pytest.approx(0.04776, rel=1e-3)
    assert sunlight_pressure(9.86) == pytest.approx(2.16e-3, rel=0.01)


def test_the_thick_disk_drifts_less_in_a_tilted_sheath() -> None:
    tilt = CME_TILTS[-1]
    thin = cme_drift(HEADLINE, CME_DENSEST_N, 2500.0e3, tilt, _SIM_PERIAPSIS_SPEED)
    thick = cme_drift(PROPOSAL, CME_DENSEST_N, 2500.0e3, tilt, _SIM_PERIAPSIS_SPEED)
    assert thick.drift < 0.5 * thin.drift


def test_network_alone_nearly_fills_the_proposals_budget() -> None:
    budget = dispersion_budget(PROPOSAL, 0.0)
    assert budget.closes
    assert budget.ship == pytest.approx(math.sqrt(0.0161**2 - 0.013**2), rel=0.02)


def test_ledger_defaults_are_the_headline() -> None:
    design = sdg.HEADLINE_STARTS[(9.86, 60.0)]
    assert design.sun_rate == sdg.SUN_PULSE_RATE
    assert design.sun_coupling == 1.0 and design.sun_hold == 0.0


def test_hold_and_coupling_move_the_ledger_as_signed() -> None:
    departure = sdg.departure_route(9.86, 60.0)
    push = sdg.push_route(9.86, 60.0, sdg.PARKING_DAYS)
    assert departure is not None and push is not None
    base = sdg.HEADLINE_STARTS[(9.86, 60.0)]
    flown = sdg.fly_unit(departure, push, departure, base).lines
    held = sdg.fly_unit(departure, push, departure, replace(base, sun_hold=0.003)).lines
    assert flown["sun_hold"] == 0.0
    boost = departure.periapsis_boost / (1.0 - sdg.arc_loss(base.arc_hours, departure))
    burn = boost * 0.003 / (sdg.STORABLE_ISP * 9.80665e-3)
    assert held["sun_hold"] == pytest.approx(
        flown["ship_at_sun"] * (1.0 - math.exp(-burn)), rel=1e-9
    )
    stronger = sdg.fly_unit(
        departure, push, departure, replace(base, sun_coupling=1.1)
    ).lines
    assert stronger["ve_sun"] == pytest.approx(1.1 * flown["ve_sun"])
    assert stronger["disks"] < flown["disks"]


@pytest.mark.slow
def test_the_proposal_is_priced() -> None:
    headline = price(HEADLINE, 0.0, offset_loss=False)
    assert headline.optimum.dear_break_even == pytest.approx(158.2, abs=0.3)
    proposal = price(
        PROPOSAL, hold_fraction(PROPOSAL, dispersion_requirement(THICK_X13))
    )
    assert proposal.lines[0]["pulse_disk"] == pytest.approx(2.5, rel=1e-3)
    assert proposal.optimum.dear_break_even > headline.optimum.dear_break_even


@pytest.mark.slow
def test_result_24s_x12_charge_is_priced_at_both_depths() -> None:
    hold = hold_fraction(HEADLINE_X12, dispersion_requirement(THIN_X12))
    wide = price(HEADLINE_X12, hold)
    assert wide.optimum.dear_break_even == pytest.approx(156.1, abs=0.3)
    deep_headline = price(HEADLINE, 0.0, offset_loss=False, closure=DEEP_CLOSURE)
    assert deep_headline.optimum.dear_break_even == pytest.approx(88.7, abs=0.3)
    deep = price(HEADLINE_X12, hold, closure=DEEP_CLOSURE)
    assert deep.optimum.dear_break_even == pytest.approx(89.1, abs=0.3)
