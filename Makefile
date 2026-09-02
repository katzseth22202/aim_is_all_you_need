.PHONY: help install clean test test-slow test-all mypy format check-format run nozzle resonance resonance-impulse jovian-dive dive-depth split-dive opposing-stream shallow-dive two-wave two-leg bag-state nozzle-geom cruise-thermal plume-state bag-converge all export-env

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

test:  ## Run the fast tests (~1 min); deselects the 'slow' marker
	pytest -s -m "not slow"

test-slow:  ## Run only the slow tests (~12 min): optimiser sweeps and multi-minute searches
	pytest -s -m "slow"

test-all:  ## Run every test (~13 min). The gate before committing or quoting a number
	pytest -s

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

nozzle:  ## Run the ADR 0009 nozzle analysis (compute-intensive; not part of 'all')
	python -m src.nozzle_analysis

resonance:  ## Audit 200 years of real-orbit 2S windows and the 2S/3S fallback
	python -m src.real_orbit_resonance --years 200

two-wave:  ## Price the real-orbit adaptive 2S/3S cadence on the two-wave nozzle ledger
	python -m src.two_wave_growth

two-leg:  ## Compare a magnetic nozzle on both legs against the pusher plate (ADR 0014)
	python -m src.two_leg_nozzle_sweep

bag-state:  ## Reproduce tab:bag_state and close the leak bracket (ledger items 5-10)
	python -m src.bag_state

nozzle-geom:  ## Snowplow sweep, mirror trade and two-term nozzle mass (items 11-13)
	python -m src.nozzle_geometry

cruise-thermal:  ## Ice sublimation equilibrium for the projectile (ledger item 14)
	python -m src.cruise_thermal

plume-state:  ## Burn envelope, bag consequence and tab:seed_window (items 1, 3)
	python -m src.plume_state

bag-converge:  ## Iterate the bag loop to a fixed point and report the gap (rule 2)
	python -m src.bag_converge

resonance-impulse:  ## Score circular 2S/3S closures on departure-burn delivered mass (ADR 0012)
	python -m src.circular_resonance_impulse

jovian-dive:  ## Close Earth->Jupiter->4 Rsun->Earth on a synodic clock; 3S works, 2S does not (ADR 0019)
	python -m src.jovian_solar_dive_cycle

dive-depth:  ## Price a shallower solar dive against the 4 Rsun cycle, launch ledger charged from the pad (ADR 0020/0021/0022; --optimum and --pad-frontier add the searches)
	python -m src.solar_dive_depth_trade

split-dive:  ## Split the dive injection across two nodes and phase the far one (ADR 0023)
	python -m src.bielliptic_dive_split

opposing-stream:  ## Charge the dive node's second arrival, the opposing stream nobody priced (ADR 0024)
	python -m src.opposing_stream_ledger

shallow-dive:  ## Price a shallow dive for the direct architecture, node charged for its burn (ADR 0025)
	python -m src.shallow_dive_burn_trade

export-env:  ## Export the current conda environment to environment.yml
	conda env export --no-builds --from-history | grep -v "prefix:" | sed '1s/^name: .*/name: puffsat_math_env/' > environment.yml.tmp
	mv environment.yml.tmp environment.yml

all: format mypy test run  ## Format, mypy, FAST tests, main script (slow tests: 'make test-all'; export-env separately)
