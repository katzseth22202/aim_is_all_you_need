"""Tests for Straw Way energy from the Ceres loop (S58, ADR 0084)."""

from dataclasses import replace

import pytest

from src.ceres_dive import CERES_BOOK, CeresDesign
from src.ceres_energy import (
    CAPPED,
    ENERGY_BOOK,
    PARENT_BOOK,
    electric_mwh,
    energy_loop,
    flight_days,
    ground_speed,
    sublimated_share,
)
from src.solar_dive_growth import Design, unfloored

#: The parent's fixed design for its line items and sensitivities, flown
#: as ADR 0084 flew it: before the disk floor and its gas (ADR 0085).
PARENT_DESIGN = CeresDesign(
    depth=4.0,
    ceres_excess=110.0,
    earth_stream=137.0,
    k_sun=29.907,
    arc_hours=2.4,
    k_delivery=5.0,
    spring=unfloored(Design()),
)


def test_the_fall_from_400_km_adds_only_a_few_hundredths():
    assert ground_speed(137.0) - 137.0 == pytest.approx(0.03, abs=0.005)
    assert electric_mwh(137.0) == pytest.approx(1.304, abs=0.001)


def test_s57_plates_price_the_import_at_landed_hardware():
    assert ENERGY_BOOK.plate_price(35.0) == pytest.approx(0.99 * 10.0 + 0.01 * 35.0)
    assert ENERGY_BOOK.plate_earth_share == pytest.approx(0.01)
    # ADR 0083's book keeps its Earth-made plates.
    assert CERES_BOOK.plate_price(35.0) == 35.0
    assert CERES_BOOK.plate_earth_share == 1.0


def test_bare_ice_does_not_survive_release_inside_1_au_but_a_capped_pellet_does():
    assert sublimated_share(4.0, 128.0, 0.5, steps=30) > 0.99
    assert 0.3 < sublimated_share(4.0, 128.0, 0.8, steps=30) < 0.7
    assert sublimated_share(4.0, 128.0, 0.05, absorptivity=CAPPED, steps=30) > 0.2
    assert sublimated_share(4.0, 128.0, 0.3, absorptivity=CAPPED, steps=30) < 0.001
    assert flight_days(4.0, 128.0, 0.1) > flight_days(4.0, 128.0, 0.5)


@pytest.mark.slow
def test_the_parents_harness_reproduces():
    loop = energy_loop(PARENT_DESIGN, PARENT_BOOK, l1_freight=CERES_BOOK)
    assert loop.c_ceres == pytest.approx(4.35, abs=0.005)
    assert loop.c_energy == pytest.approx(5.83, abs=0.005)
    assert loop.price == pytest.approx(4.47, abs=0.005)


@pytest.mark.slow
def test_film_cargo_on_earth_plates_is_adr_0083s_ceres_stream():
    film = energy_loop(
        replace(PARENT_DESIGN, k_sun=29.907, arc_hours=3.14, k_delivery=5.62),
        CERES_BOOK,
        cargo_price=CERES_BOOK.film,
    )
    assert film.c_ceres == pytest.approx(5.69, abs=0.01)
    assert film.c_energy == pytest.approx(10.74, abs=0.01)


@pytest.mark.slow
def test_package_mass_and_solved_freight_raise_the_parents_figure():
    loop = energy_loop(PARENT_DESIGN)
    assert loop.price > 4.47
    assert loop.lines["packages"] > 0.5
    assert 0.0 < loop.sheet_share < 0.05


@pytest.mark.slow
def test_the_energy_headline_reproduces():
    """ADR 0084's minimum: 4 R_sun, 137 km/s, X_C 120 on the energy book."""
    design = CeresDesign(
        depth=4.0,
        ceres_excess=120.0,
        earth_stream=137.0,
        k_sun=29.907,
        arc_hours=2.421,
        k_delivery=5.0,
        spring=unfloored(Design()),
    )
    loop = energy_loop(design)
    assert loop.price == pytest.approx(5.03, abs=0.01)
    assert loop.c_ceres == pytest.approx(5.30, abs=0.01)
    assert loop.freight == pytest.approx(4.28, abs=0.01)
