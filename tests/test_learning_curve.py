"""Tests for src/learning_curve.py: Wright's law with a floor and an anchor."""

import math

import pytest

from src.learning_curve import LearningCurve


def test_the_first_unit_costs_the_first_price() -> None:
    plate = LearningCurve(first=30.0e6, floor=6.0e6, rate=0.8)
    assert plate.batch_cost(built=0.0, count=1.0) == pytest.approx(30.0e6, rel=0.03)


def test_integer_batches_match_the_discrete_sum() -> None:
    # Midpoint integration runs slightly low: under 2% from two units, under 1%
    # from six, which is where the near-term plate counts sit.
    plate = LearningCurve(first=1.0, floor=0.0, rate=0.8)
    for count, tolerance in ((2, 0.02), (6, 0.01), (50, 0.005), (1000, 0.001)):
        discrete = sum(n ** math.log2(0.8) for n in range(1, count + 1))
        assert plate.batch_cost(0.0, float(count)) == pytest.approx(
            discrete, rel=tolerance
        )


def test_a_batch_costs_the_same_bought_whole_or_in_pieces() -> None:
    chamber = LearningCurve(first=40.0e6, floor=3.0e6, rate=0.8)
    whole = chamber.batch_cost(3.0, 7.5)
    pieces = chamber.batch_cost(3.0, 2.5) + chamber.batch_cost(5.5, 5.0)
    assert whole == pytest.approx(pieces, rel=1e-12)


def test_the_price_never_crosses_the_floor() -> None:
    plate = LearningCurve(first=30.0e6, floor=6.0e6, rate=0.8)
    assert plate.unit(1e15) == pytest.approx(6.0e6, rel=1e-3)
    assert plate.batch_cost(1e9, 10.0) > 10 * 6.0e6


def test_the_price_holds_until_the_anchor() -> None:
    package = LearningCurve(first=100.0, floor=10.0, rate=0.8, anchor=1.0e5)
    assert package.unit(5.0e4) == 100.0
    assert package.unit(2.0e5) == pytest.approx(10.0 + 90.0 * 0.8)
    assert package.batch_cost(0.0, 1.0e3) == pytest.approx(1.0e5)


def test_a_flat_price_is_the_count_times_the_price() -> None:
    assert LearningCurve.flat(2.0e6).batch_cost(17.0, 3.25) == pytest.approx(6.5e6)


def test_units_to_reproduces_the_parents_plate_learning_table() -> None:
    # ASKS.md "Plate learning": plates before the price falls below $10M.
    assert LearningCurve(30.0e6, 6.0e6, 0.8).units_to(10.0e6) == pytest.approx(
        261, abs=1
    )
    assert LearningCurve(20.0e6, 6.0e6, 0.8).units_to(10.0e6) == pytest.approx(
        49, abs=1
    )
    assert LearningCurve(78.0e6, 6.0e6, 0.7).units_to(10.0e6) == pytest.approx(
        275, abs=1
    )
    # README: "about 600 plates" on the paper's own curve, no floor.
    assert LearningCurve(78.0e6, 0.0, 0.8).units_to(10.0e6) == pytest.approx(590, abs=5)


def test_units_to_is_none_at_or_below_the_floor() -> None:
    assert LearningCurve(30.0e6, 6.0e6, 0.8).units_to(6.0e6) is None
    assert LearningCurve.flat(2.0e6).units_to(1.0e6) is None
