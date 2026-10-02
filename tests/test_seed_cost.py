"""Tests for src/seed_cost.py: the expended seed ship and tab:seed_amortization."""

import pytest
from astropy import units as u

from src.seed_cost import (
    DESIGNS,
    FLIGHT_PRICES,
    SEED_LAUNCHES,
    SHIP_PRICES,
    STOCK_HIGH,
    STOCK_LOW,
    STOCK_SHIP,
    STRIPPED_SHIPS,
    TANKER_FLIGHTS,
    burn_from_circular,
    chains,
    design_chain,
    fleet_present_value,
    seed_excess_speed,
    seed_payload,
    seed_price_per_kg,
    seed_price_range,
    tanker_share,
    units_built,
    wright_unit_cost,
)

#: The first cycle's excess speed (the chain gives 11.9928 km/s).
EXCESS = 11.99276933 * u.km / u.s


def test_twelve_tankers_and_one_ship_launch() -> None:
    """1200 t of propellant at 100 t per tanker, plus the ship's own ascent."""
    assert TANKER_FLIGHTS == 12
    assert SEED_LAUNCHES == 13


def test_the_burn_from_200_km_is_the_parents() -> None:
    """``sec:mass_interest``: 8.50 km/s from 200 km to 11.99 km/s."""
    assert burn_from_circular(EXCESS).to_value(u.km / u.s) == pytest.approx(
        8.495, abs=1e-3
    )


def test_the_stock_ship_reproduces_adr_0035() -> None:
    """Six engines burn 271 s, lose 31 m/s steered, and send 50.5 t."""
    flown = seed_payload(STOCK_SHIP, EXCESS)
    assert flown.burn_time.to_value(u.s) == pytest.approx(271.4, abs=0.1)
    assert flown.finite_burn_loss.to_value(u.m / u.s) == pytest.approx(30.8, abs=0.1)
    assert flown.mass_ratio == pytest.approx(9.85, abs=0.01)
    assert flown.payload.to_value(u.t) == pytest.approx(50.53, abs=0.01)


def test_the_stock_ends_are_the_parents_910_to_14_800() -> None:
    low = seed_price_range(EXCESS, STOCK_LOW, STOCK_HIGH)
    assert low.low == pytest.approx(910.4, abs=0.5)
    assert low.high == pytest.approx(14843.0, abs=2.0)


def test_three_engines_double_the_burn_and_recharge_the_loss() -> None:
    """The stripped ship's longer burn is charged its own loss, not the 31 m/s."""
    stock = seed_payload(STOCK_SHIP, EXCESS)
    stripped = seed_payload(STRIPPED_SHIPS[0], EXCESS)
    assert float(stripped.burn_time / stock.burn_time) == pytest.approx(2.0)
    assert stripped.finite_burn_loss.to_value(u.m / u.s) == pytest.approx(
        116.8, abs=0.2
    )
    # Below the 95.5 t a 40 t ship would send on the stock ship's loss.
    assert stripped.payload.to_value(u.t) == pytest.approx(92.10, abs=0.01)
    assert seed_payload(STRIPPED_SHIPS[1], EXCESS).payload.to_value(
        u.t
    ) == pytest.approx(72.10, abs=0.01)


def test_the_stripped_ship_nearly_halves_the_price_per_kilogram() -> None:
    """At the same hull and flight price the payload, not the hull, moves $/kg."""
    stock = seed_payload(STOCK_SHIP, EXCESS).payload
    stripped = seed_payload(STRIPPED_SHIPS[0], EXCESS).payload
    for flight in FLIGHT_PRICES.values():
        ratio = seed_price_per_kg(stripped, flight, 5e6) / seed_price_per_kg(
            stock, flight, 5e6
        )
        assert ratio == pytest.approx(0.549, abs=0.001)


def test_under_bank_prices_the_tankers_are_most_of_the_bill() -> None:
    for name in ("Goldman", "Morgan Stanley"):
        assert tanker_share(FLIGHT_PRICES[name], SHIP_PRICES[0]) > 0.90
    assert tanker_share(FLIGHT_PRICES["Musk"], SHIP_PRICES[0]) < 0.8


def test_the_stripped_baseline_ends() -> None:
    """(40 t, $5M, Musk) to (60 t, $20M, Morgan Stanley)."""
    ends = seed_price_range(EXCESS)
    assert ends.low == pytest.approx(336.6, abs=0.5)
    assert ends.high == pytest.approx(9293.0, abs=2.0)


def test_wright_at_80_percent() -> None:
    """The forty-fourth unit costs 0.30 of the first."""
    assert wright_unit_cost(44.3) == pytest.approx(0.30, abs=0.005)
    assert wright_unit_cost(2.0) == pytest.approx(0.8)


def test_fleet_present_value_is_the_parents_column() -> None:
    assert fleet_present_value(5.09, 0.076) == pytest.approx(2.45, abs=0.01)
    assert fleet_present_value(5.09, 0.30) == pytest.approx(0.369, abs=0.001)


@pytest.mark.slow
def test_the_chain_seed_leaves_at_the_parents_excess_speed() -> None:
    flown, three = chains()
    assert seed_excess_speed(flown[0]).to_value(u.km / u.s) == pytest.approx(
        EXCESS.to_value(u.km / u.s), abs=1e-6
    )
    assert flown[0].departure_jd == three[0].departure_jd


@pytest.mark.slow
@pytest.mark.parametrize(
    "index, seed_t, annual, ten_year, built",
    [
        # ADR 0038 (onward burn); the parent printed 72.7 / 19% / 5.1 / 12.5,
        # 80.81 / 51% / 69.5 / 34.4 and 80.73 / 57% / 104.2 / 44.3.
        (0, 72.71, 0.185, 5.071, 10.70),
        (2, 83.08, 0.488, 49.66, 25.28),
        (4, 83.25, 0.548, 75.07, 32.85),
    ],
)
def test_tab_seed_amortization_reproduces(
    index: int, seed_t: float, annual: float, ten_year: float, built: float
) -> None:
    """The parent's ``tab:seed_amortization`` rows, from the rebuilt module."""
    chain = design_chain(DESIGNS[index])
    assert chain.seed.to_value(u.t) == pytest.approx(seed_t, abs=0.01)
    assert chain.summary.annual_growth == pytest.approx(annual, abs=0.001)
    assert chain.summary.ten_year_stepwise == pytest.approx(ten_year, rel=2e-3)
    assert units_built(chain) == pytest.approx(built, abs=0.05)
