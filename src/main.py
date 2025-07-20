"""Main entry point for the application."""

from typing import List


def greet(name: str) -> str:
    """Return a greeting message for the given name.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting message
    """
    return f"Hello, {name}! Welcome to AIM is all you need!"


def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers.
    
    Args:
        numbers: List of integers to sum
        
    Returns:
        The sum of all numbers in the list
    """
    return sum(numbers)


def main() -> None:
    """Main function that runs the application."""
    print("=== AIM is all you need ===")
    print()
    
    # Example usage of our functions
    greeting = greet("Developer")
    print(greeting)
    
    numbers = [1, 2, 3, 4, 5]
    total = calculate_sum(numbers)
    print(f"Sum of {numbers} is: {total}")
    
    print()
    print("Application completed successfully!")


if __name__ == "__main__":
    main() 