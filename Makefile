.PHONY: help install install-dev clean test mypy format check-format run all export-env

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

export-env:  ## Export the current conda environment to environment.yml
	conda env export --no-builds --from-history > environment.yml

all: format mypy test run export-env  ## Format code, run mypy, tests, main script, and export environment 