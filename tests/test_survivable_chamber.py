"""Tests for src/survivable_chamber.py: the parent's S17 cases (ADR 0044).

The chains themselves are the report (``make survivable-chamber``); these pin
the designs it flies.
"""

import pytest
from astropy import units as u

from src.chamber_isp import (
    HYDROGEN_5500K,
    METHANE_7000K,
    SURVIVABLE_IMPULSE_CARBON_FROZEN,
)
from src.growth_cost import ESTIMATE, Program, _fleet_lines
from src.growth_ledger import METHANE_PITCH, SPRAY_CUP_SHIELDED
from src.seed_cost import DESIGNS, Design
from src.survivable_chamber import (
    HYDROGEN_BULGED_DESIGN,
    PLATE,
    SURVIVABLE_PITCH_RANGE,
    cost_designs,
    ledger_cases,
    paper_cost_designs,
    paper_grid_designs,
    s24_rates,
    survivable_design,
    survivable_efficiency,
)


def test_the_pitch_and_throat_film_are_charged_per_kilogram_of_rod() -> None:
    # ADR 0055: the 2.6 kg throat film leaves in the exhaust with the pitch.
    assert survivable_design(100, 24.0 * u.kg).pitch_ratio == pytest.approx(5.32)
    assert survivable_design(100, 3.5 * u.kg).pitch_ratio == pytest.approx(1.22)
    split = survivable_design(100, 24.0 * u.kg, rods_split=2)
    assert split.pitch_ratio == pytest.approx(5.32)
    assert Design("x", METHANE_7000K, 0.5).pitch_ratio == pytest.approx(
        (METHANE_PITCH / (2.5 * u.kg)).to_value(u.one)
    )
    assert Design("x", HYDROGEN_5500K, 0.8).pitch_ratio == 0.0
    assert DESIGNS[0].pitch_ratio == 0.0


def test_the_area_ratios_fly_the_impact_sims_eta() -> None:
    assert survivable_efficiency(100) == pytest.approx(0.472, abs=0.001)
    assert survivable_efficiency(150) == pytest.approx(0.490, abs=0.001)
    assert survivable_efficiency(300) == pytest.approx(0.515, abs=0.001)
    assert survivable_design(300).efficiency == survivable_efficiency(300)
    assert survivable_efficiency(
        300, SURVIVABLE_IMPULSE_CARBON_FROZEN
    ) == pytest.approx(0.446, abs=0.001)


def test_s24_flies_two_and_three_hertz_not_four() -> None:
    """ADR 0056: with a 2 t/s fill the doubled throat's period is 298 ms."""
    assert [r.to_value(1 / u.s) for r in s24_rates()] == [2.0, 3.0]
    s24 = [c for c in ledger_cases() if c.group.startswith("S24")]
    assert len(s24) == 2 * 2 * 3
    rates = {c.design.pairing.pulse_rate.to_value(1 / u.s) for c in s24}
    assert rates == {2.0, 3.0}
    # The film rides every rate: it is sealed at 0.3 bar whatever the period.
    assert {round(c.design.pitch_ratio, 2) for c in s24} == {5.32}


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


def test_every_row_carries_the_film_the_parents_tables_carry() -> None:
    # Parent S21: the S17 rows flew the film-free cup while the parent's
    # tables fly the shielded one, so their hydrogen rows disagreed by $2/kg.
    assert PLATE is SPRAY_CUP_SHIELDED
    assert PLATE.carries_film


def test_the_bulged_hydrogen_chamber_is_priced_beside_the_sphere() -> None:
    names = [d.label for d in cost_designs()]
    assert HYDROGEN_BULGED_DESIGN.label in names
    assert DESIGNS[4].label in names
    assert HYDROGEN_BULGED_DESIGN.pitch_ratio == 0.0
    assert HYDROGEN_BULGED_DESIGN.efficiency == pytest.approx(0.773, abs=0.002)


def test_the_wall_life_case_flies_two_chambers() -> None:
    forced = [c for c in ledger_cases() if "wall life" in c.label]
    assert {c.design.chambers for c in forced} == {2}


def test_the_paper_grid_flies_the_papers_methane_rows() -> None:
    """ADR 0056: the shares of tab:growth_ledger_doubling with the solved row
    in its place, all on the doubled throat at A/A* = 300."""
    rows = paper_grid_designs()
    assert [label for label, _ in rows][2] == "Methane, solved (77%)"
    assert len(rows) == 5
    assert all(d.pairing.extension_mass.to_value(u.t) == 29.4 for _, d in rows)
    plates = {plate.label for _, plate in paper_cost_designs()}
    assert PLATE.label in plates and len(plates) == 4
