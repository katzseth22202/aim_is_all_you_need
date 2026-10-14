"""Tests for the departure-lever ranking (ADR 0063, 0064)."""

import pytest
from astropy import units as u

from src.chamber_isp import (
    HYDROGEN_BULGED,
    LIQUID_HYDROGEN,
    SLUSH_HYDROGEN,
    efficiency_for_impulse,
)
from src.departure_levers import (
    CHOSEN_TEMPERATURE,
    SOLVED_HYDROGEN,
    cases,
    solved_hydrogen,
)
from src.survivable_chamber import HYDROGEN_BULGED_DESIGN


def test_a_hotter_charge_is_smaller_and_cycles_faster():
    rows = [solved_hydrogen(t)[0] for t in sorted(SOLVED_HYDROGEN)]
    slugs = [p.reference_slug_ratio for p in rows]
    rates = [p.pulse_rate.to_value(1 / u.s) for p in rows]
    assert slugs == sorted(slugs, reverse=True)
    assert rates == sorted(rates)


def test_todays_solved_row_is_the_flown_bulge_less_its_wall_heat():
    pairing, eta = solved_hydrogen(4884.0)
    assert pairing.reference_slug_ratio == pytest.approx(
        HYDROGEN_BULGED.reference_slug_ratio, abs=0.05
    )
    # 661 kN s with the wall heat debited, against the flown 683.
    assert eta < HYDROGEN_BULGED_DESIGN.efficiency
    assert eta == pytest.approx(
        efficiency_for_impulse(661.0 * u.kN * u.s, HYDROGEN_BULGED), rel=0.005
    )


def test_the_solved_rows_keep_the_bulged_chambers_tanks_and_ceiling():
    assert CHOSEN_TEMPERATURE == 4884.0
    pairing, _ = solved_hydrogen(5500.0)
    assert pairing.temperature == 5500.0 * u.K
    assert pairing.tank_fraction == HYDROGEN_BULGED.tank_fraction
    assert pairing.wall_mass == HYDROGEN_BULGED.wall_mass
    assert solved_hydrogen(7000.0)[0].wall_mass == 29.8 * u.t


def test_every_lever_is_listed_once():
    labels = [label for label, _ in cases()]
    assert len(labels) == len(set(labels))
    assert len([label for label in labels if "solved" in label]) == len(SOLVED_HYDROGEN)


def test_hydrogen_flies_as_slush_but_for_one_liquid_row():
    hydrogen = [
        (label, design.pairing)
        for label, design in cases()
        if design.pairing is not None and design.pairing.hydrogen
    ]
    liquid = [label for label, p in hydrogen if p.storage == LIQUID_HYDROGEN]
    assert liquid == ["hydrogen, baseline, liquid"]
    for label, pairing in hydrogen:
        if label not in liquid:
            assert pairing.storage == SLUSH_HYDROGEN
            assert pairing.tank_fraction == pytest.approx(0.205 / 1.15)
