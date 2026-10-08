"""Tests for src/survivable_chamber.py: the parent's S17 cases (ADR 0044).

The chains themselves are the report (``make survivable-chamber``); these pin
the designs it flies.
"""

import pytest
from astropy import units as u

from src.chamber_isp import HYDROGEN_5500K, METHANE_7000K
from src.growth_cost import ESTIMATE, Program, _fleet_lines
from src.growth_ledger import METHANE_PITCH
from src.seed_cost import DESIGNS, Design
from src.survivable_chamber import (
    SURVIVABLE_PITCH_RANGE,
    ledger_cases,
    survivable_design,
    survivable_efficiency,
)


def test_the_pitch_is_charged_per_kilogram_of_each_chambers_rod() -> None:
    assert survivable_design(100, 24.0 * u.kg).pitch_ratio == pytest.approx(4.8)
    assert survivable_design(100, 3.5 * u.kg).pitch_ratio == pytest.approx(0.7)
    split = survivable_design(100, 24.0 * u.kg, rods_split=2)
    assert split.pitch_ratio == pytest.approx(4.8)
    assert Design("x", METHANE_7000K, 0.5).pitch_ratio == pytest.approx(
        (METHANE_PITCH / (2.5 * u.kg)).to_value(u.one)
    )
    assert Design("x", HYDROGEN_5500K, 0.8).pitch_ratio == 0.0
    assert DESIGNS[0].pitch_ratio == 0.0


def test_the_area_ratios_fly_the_impact_sims_eta() -> None:
    assert survivable_efficiency(100) == pytest.approx(0.488, abs=0.001)
    assert survivable_efficiency(300) == pytest.approx(0.539, abs=0.001)
    assert survivable_design(300).efficiency == survivable_efficiency(300)


def test_every_s17_item_has_its_cases() -> None:
    cases = ledger_cases()
    labels = [c.label for c in cases]
    assert len(set(labels)) == len(labels)
    assert sum(c.group.startswith("1.") for c in cases) == 2 * len(
        SURVIVABLE_PITCH_RANGE
    )
    redundancy = [c.design for c in cases if c.group.startswith("3.")]
    assert {d.pairing.rod_mass.to_value(u.kg) for d in redundancy} == {2.5, 5.0}
    assert any(c.steered for c in cases)


def test_a_heavier_rod_carries_fewer_packages_per_kilogram() -> None:
    """Three packages ride each rod, so 5 kg rods need half as many per kg."""
    light = _fleet_lines(ESTIMATE, Program(), 0.0, 1000.0, 0.0, rod_mass=2.5)
    heavy = _fleet_lines(ESTIMATE, Program(), 0.0, 1000.0, 0.0, rod_mass=5.0)
    bodies = ESTIMATE.rod_body * 1000.0
    assert heavy["fleet"] - bodies < 0.6 * (light["fleet"] - bodies)
