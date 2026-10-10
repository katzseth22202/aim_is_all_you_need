.PHONY: help install clean test test-changed test-slow test-all mypy format check-format run nozzle resonance resonance-impulse orbit-turn fly-park jovian-dive dive-depth split-dive opposing-stream shallow-dive sep-split sep-split-10d dsm-bound two-wave two-leg bag-state nozzle-geom cruise-thermal plume-state bag-converge chamber-departure growth-ledger seed-cost seed-harvest growth-cost plate-slug plate-designs plate-designs-cost plate-grid plate-cost plate-seed plate-film lob-brake survivable-chamber survivable-ledger survivable-cost carbon-equilibrium all export-env

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

test-changed:  ## Run every test (slow ones too) that the uncommitted change can reach by imports; BASE=main to diff against a branch
	@tests=$$(python tests/changed_tests.py $(or $(BASE),HEAD)); \
	if [ -n "$$tests" ]; then pytest -s $$tests; else echo "No tests reach this change."; fi

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

sep-split:  ## Price the 20-day split's correction burns, methalox vs argon SEP (ADR 0026)
	python -m src.sep_split_correction

sep-split-10d:  ## The same at the 10-day gap the paper flies, the figures tab:cadence_propellant quotes (ADR 0026 addendum)
	python -m src.sep_split_correction --split-days 10

two-wave:  ## Price the real-orbit adaptive 2S/3S cadence on the two-wave nozzle ledger
	python -m src.two_wave_growth

chamber-departure:  ## Price the walled chamber's departure over the flown chain, tanks, chambers and loss charged
	python -m src.chamber_departure

growth-ledger:  ## The 1500 t launch unit over the flown chain: water plate (per-pulse k) and walled chamber, the scenario matrix
	python -m src.growth_ledger

carbon-equilibrium:  ## B' of graphite in hot hydrogen: the carbon a pitch coat loses to the gas (companion reply 2026-10-04)
	python -m src.carbon_hydrogen_equilibrium

seed-cost:  ## The expended seed ship's $/kg and tab:seed_amortization, stripped baseline and stock comparison (ADR 0035/0036)
	python -m src.seed_cost

seed-harvest:  ## The seed valued as delivered cargo: k-optimised delivery to L1, liquidation, steady state, IRR, break-even price (ADR 0036)
	python -m src.harvest

growth-cost:  ## The growth-charged seed valuation and steady-state $/kg at L1: 10%, 50% odds (ADR 0037/0040), seed routes (ADR 0039)
	python -m src.growth_cost_report

plate-designs:  ## The impact-sim's plate designs (spray cup 0.60/0.57, plug 0.70, paper 0.775) through the growth ledger (ADR 0041)
	python -m src.growth_ledger --designs

plate-designs-cost:  ## The same plates through the growth cost model, lob charged for climbing at 400 km (ADR 0041, baseline ADR 0042)
	python -m src.growth_cost_report --plates

plate-grid:  ## The full ledger matrix and sensitivities behind the spray cup and the plug (ADR 0042)
	python -m src.growth_ledger --designs-grid spray-cup
	python -m src.growth_ledger --designs-grid plug

plate-cost:  ## The whole cost report behind the spray cup; headline and odds behind the plug and ADR 0033's plate (ADR 0042)
	python -m src.growth_cost_report --plate spray-cup
	python -m src.growth_cost_report --plate plug --quick
	python -m src.growth_cost_report --plate adr-0033 --quick

plate-seed:  ## tab:seed_amortization, L1 deliveries and grow-or-harvest behind the spray cup (ADR 0042)
	python -m src.seed_cost --plate spray-cup
	python -m src.harvest --plate spray-cup

plate-film:  ## The spray cup's film carried as launched mass: ledger at each band's ends, cost headline shielded / unshielded (ADR 0043)
	python -m src.growth_ledger --film
	python -m src.growth_cost_report --plate spray-cup-shielded --quick
	python -m src.growth_cost_report --plate spray-cup-unshielded --quick

lob-brake:  ## The booster's brake and reserve at each climb rate, and the lob charge it sets (ADR 0043)
	python -m src.lob_rise

survivable-chamber: survivable-ledger survivable-cost  ## The survivable 5 kg methane chamber (parent S17): ledger, AR100/300, redundancy, cost book and seed (ADR 0044)

survivable-ledger:  ## The survivable chamber through the growth ledger: both area ratios and pitch edges, redundancy, sensitivities (ADR 0044)
	python -m src.survivable_chamber --ledger

survivable-cost:  ## The survivable chamber through the cost book and tab:seed_amortization behind the spray cup (ADR 0044)
	python -m src.survivable_chamber --cost

plate-slug:  ## Water against argon on the plate, chemistry toll charged per pulse, behind each solved chamber
	python -m src.growth_ledger --slugs

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

orbit-turn:  ## Earth occultation and a 20-minute burn on the circular 3S return (ADR 0045)
	python -m src.orbit_turn_analysis

fly-park:  ## Fly short and park to hold the clock; sweet phase, launch windows, the 2S/3S synodic locks and the chain check on them (ADR 0030/0031)
	python -m src.fly_and_park

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

dsm-bound:  ## Free the split's correction burn in time and place; it does not get cheaper (ADR 0028; ~25 min, --split-days 10 for the paper's gap)
	python -m src.free_dsm_bound

export-env:  ## Export the current conda environment to environment.yml
	conda env export --no-builds --from-history | grep -v "prefix:" | sed '1s/^name: .*/name: puffsat_math_env/' > environment.yml.tmp
	mv environment.yml.tmp environment.yml

all: format mypy test run  ## Format, mypy, FAST tests, main script (slow tests: 'make test-all'; export-env separately)
