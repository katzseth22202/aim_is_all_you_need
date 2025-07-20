"""Tests for the main module."""

import pytest
from src.main import greet, calculate_sum


class TestGreet:
    """Test cases for the greet function."""

    def test_greet_with_name(self) -> None:
        """Test that greet returns the expected greeting message."""
        result = greet("Alice")
        expected = "Hello, Alice! Welcome to AIM is all you need!"
        assert result == expected

    def test_greet_with_empty_string(self) -> None:
        """Test that greet works with empty string."""
        result = greet("")
        expected = "Hello, ! Welcome to AIM is all you need!"
        assert result == expected

    def test_greet_with_special_characters(self) -> None:
        """Test that greet works with special characters."""
        result = greet("John@Doe")
        expected = "Hello, John@Doe! Welcome to AIM is all you need!"
        assert result == expected


class TestCalculateSum:
    """Test cases for the calculate_sum function."""

    def test_calculate_sum_with_positive_numbers(self) -> None:
        """Test that calculate_sum correctly sums positive numbers."""
        result = calculate_sum([1, 2, 3, 4, 5])
        assert result == 15

    def test_calculate_sum_with_negative_numbers(self) -> None:
        """Test that calculate_sum correctly sums negative numbers."""
        result = calculate_sum([-1, -2, -3])
        assert result == -6

    def test_calculate_sum_with_mixed_numbers(self) -> None:
        """Test that calculate_sum correctly sums mixed positive and negative numbers."""
        result = calculate_sum([1, -2, 3, -4, 5])
        assert result == 3

    def test_calculate_sum_with_empty_list(self) -> None:
        """Test that calculate_sum returns 0 for empty list."""
        result = calculate_sum([])
        assert result == 0

    def test_calculate_sum_with_single_number(self) -> None:
        """Test that calculate_sum works with single number."""
        result = calculate_sum([42])
        assert result == 42

    def test_calculate_sum_with_zero(self) -> None:
        """Test that calculate_sum correctly handles zero."""
        result = calculate_sum([0, 1, 2, 3])
        assert result == 6


@pytest.mark.slow
class TestIntegration:
    """Integration tests that may be slower."""

    def test_greet_and_calculate_workflow(self) -> None:
        """Test a simple workflow using both functions."""
        # This is a simple integration test
        greeting = greet("TestUser")
        assert "TestUser" in greeting
        
        numbers = [10, 20, 30]
        total = calculate_sum(numbers)
        assert total == 60
        
        # Verify both results are as expected
        assert greeting == "Hello, TestUser! Welcome to AIM is all you need!"
        assert total == 60 