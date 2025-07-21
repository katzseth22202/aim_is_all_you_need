.PHONY: help install install-dev clean test test-cov mypy lint format run all

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-simple:  ## Run tests without coverage (if pytest-cov not available)
	pytest --no-cov

mypy:  ## Run mypy type checking
	mypy src/

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

check-format:  ## Check if code is formatted correctly
	black --check src/ tests/
	isort --check-only src/ tests/

run:  ## Run the main script
	python -m src.main

all: mypy test run  ## Run mypy, tests, and then the main script 