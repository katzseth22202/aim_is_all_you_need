# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a Python scientific computing project implementing orbital mechanics calculations for PuffSat-based propulsion systems. The research paper it supports, "Aim Is All You Need" (DOI: [10.5281/zenodo.16741183](https://doi.org/10.5281/zenodo.16741183)), analyzes externally pulsed propulsion for missions including lunar transfer, interplanetary travel, and solar periapsis maneuvers.

## Reading the Paper

The paper is **not checked into this repository**. Its LaTeX source lives in the separate
[Balloon-Pulse-Propulsion](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
repository (which also has its own `CONTEXT.md` and `docs/adr/` with commit-level
rationale). To consult the paper's text, clone that repo and grep the `.tex` sources
directly — locate claims by their quoted wording, not by line numbers:

```bash
git clone https://github.com/katzseth22202/Balloon-Pulse-Propulsion /tmp/bpp
grep -rn "300 m/s to 600 m/s" /tmp/bpp --include="*.tex"
```

The published PDF is on Zenodo at the DOI above if the rendered version is needed.

**Caveat on stale pointers:** older scratch notes (e.g. `todos/citation_audit_findings.md`)
reference line numbers from a `/tmp/paper.txt` extracted from the now-removed PDF. Those
line numbers were never reproducible; locate claims by grep-ing the quoted wording in the
LaTeX source instead.

## Commands

```bash
# Run all checks (format → mypy → test → run → export-env)
make all

# Individual steps
make test           # pytest -s
make mypy           # mypy src/
make format         # black + isort
make check-format   # check only, no changes
make run            # python -m src.main

# Run a single test file
pytest tests/test_orbit_utils.py -s

# Conda environment
conda env create -f environment.yml
conda activate puffsat_math_env
```

## Architecture

The source modules form a strict dependency hierarchy — each layer imports only from layers below it:

```
astro_constants.py   ← physical constants and mission parameters (no src imports)
      ↓
orbit_utils.py       ← orbital mechanics primitives (boinor/astropy wrappers)
      ↓
propulsion.py        ← rocket equation, Hohmann transfers, mass ratio calculations
      ↓
scenario.py          ← PuffSatScenario dataclass, paper_scenarios(), find_best_lunar_return()
      ↓
main.py              ← entry point, prints scenario tables and analysis
```

**Key types used throughout:**
- `astropy.units.Quantity` — all physical values carry units; never use bare floats for physics
- `boinor.twobody.Orbit` — represents orbital state; constructed via `orbit_from_rp_ra()` or `Orbit.circular()`
- `numpy.typing.NDArray` — typed arrays for vectorized calculations

**`PuffSatScenario`** (frozen dataclass in `scenario.py`) holds `v_rf` (final velocity), `v_b` (PuffSat collision velocity), `v_ri` (initial velocity), and `desc`. The `paper_scenarios()` static method runs all scenarios from the paper and returns a pandas DataFrame.

**`BurnInfo`** (frozen dataclass in `scenario.py`) is the return type of `find_best_lunar_return()`, containing the optimal burn magnitude, combined mass ratio, and incoming velocity.

## Type Checking

mypy is configured in strict mode (`pyproject.toml`). Key settings:
- `disallow_untyped_defs = true` — all functions in `src/` must have type annotations
- `ignore_missing_imports = true` — stubs are missing for boinor/astropy but that's accepted
- Tests (`tests.*`) are exempt from annotation requirements

All new functions in `src/` must have full type annotations and Google-style docstrings (see `cursor-rules.toml`).

## Scratch / Todo Directory

`todos/` exists for Claude's scratch files, plans, notes, and todo lists. The directory is gitignored (`todos/*` is excluded, only `.gitkeep` is tracked), so write freely there without worrying about accidental commits. Use it for anything that would clutter the repo otherwise — intermediate analyses, draft notes, working scratchpads.

## Testing

Tests live in `tests/` with one file per source module. `tests/test_helpers.py` has shared utilities including floating-point comparison helpers for physics quantities. Use `pytest markers` `slow` and `integration` when appropriate (defined in `pyproject.toml`).
