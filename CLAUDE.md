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

`scenario.py` was split into six analysis modules (2026-07), one per CONTEXT.md
"Scenarios" section, plus a shared substrate. Almost all of it is still a strict
hierarchy — each layer imports only from layers below it:

```
astro_constants.py           ← physical constants and mission parameters (no src imports)
      ↓
conic_kernel.py               ← float two-body conic geometry (TOF, bend angles, radius crossings)
      ↓
orbit_utils.py                ← orbital mechanics primitives (boinor/astropy wrappers)
      ↓
propulsion.py                 ← rocket equation, Hohmann transfers, mass ratio calculations
      ↓
heliocentric_reintercept.py   ← solar-dive re-intercept leg; owns launch_capacity_time()
      ↓
apoapsis_raise_reintercept.py ← apoapsis-raise re-intercept leg
      ↓
scenario_catalog.py           ← PuffSatScenario, paper_scenarios(), find_best_lunar_return()
      ↓
retrograde_return_legs.py     ← float substrate: leg/body/state primitives shared below
      ↓
jovian_flyby.py                assist_chain.py imports from jovian_flyby.py too
      ↓
assist_chain.py
      ↓
main.py / nozzle_analysis.py  ← both import from every module above
```

One edge runs opposite the diagram: `retrograde_return_legs.py` imports
`lunar_transfer_periapsis_speed()` from `scenario_catalog.py` (the push target
the flyby/chain searches score against). This doesn't create a cycle —
`scenario_catalog.py` never imports back from `retrograde_return_legs.py` or
anything above it — but it means the "substrate" isn't literally at the bottom
of the import graph. Don't try to flatten this without checking the graph stays
acyclic; it's a known, load-bearing exception, not a leftover to clean up.

`conic_kernel.py` trades `astropy.units.Quantity` for plain floats (km, s, km/s, rad):
it is the shared substrate under optimizer hot loops (the powered Jovian flyby and
assist-chain searches, and `nozzle_analysis.py`), and `orbit_utils.py`'s
Quantity-valued time-of-flight/true-anomaly functions delegate to it rather than
duplicating the algebra. See CONTEXT.md, "Conic kernel".

`retrograde_return_legs.py` is a second, higher-level substrate in the same
spirit (see CONTEXT.md, "Retrograde-return legs"): it holds the leg-by-leg
body/radius/velocity/TOF state both the powered flyby and the unpowered chain
assemble, plus the primitives `nozzle_analysis.py` reuses directly. Its private
names are a real cross-module seam, not test-only scaffolding — two production
modules (`jovian_flyby.py`, `assist_chain.py`) and `nozzle_analysis.py` all
import from it. Re-inlining any of that state or algebra at a new call site
defeats the point of the split.

**Key types used throughout:**
- `astropy.units.Quantity` — all physical values carry units; never use bare floats for physics
- `boinor.twobody.Orbit` — represents orbital state; constructed via `orbit_from_rp_ra()` or `Orbit.circular()`
- `numpy.typing.NDArray` — typed arrays for vectorized calculations

**`PuffSatScenario`** (frozen dataclass in `scenario_catalog.py`) holds `v_rf` (final velocity), `v_b` (PuffSat collision velocity), `v_ri` (initial velocity), and `desc`. The `paper_scenarios()` function runs all scenarios from the paper and returns a pandas DataFrame.

**`BurnInfo`** (frozen dataclass in `scenario_catalog.py`) is the return type of `find_best_lunar_return()`, containing the optimal burn magnitude, combined mass ratio, and incoming velocity.

## Type Checking

mypy is configured in strict mode (`pyproject.toml`). Key settings:
- `disallow_untyped_defs = true` — all functions in `src/` must have type annotations
- `ignore_missing_imports = true` — stubs are missing for boinor/astropy but that's accepted
- Tests (`tests.*`) are exempt from annotation requirements

All new functions in `src/` must have full type annotations and Google-style docstrings (see `cursor-rules.toml`).

## Grill Sessions (grill-me / grill-with-docs)

When running an interview/grill session in this repo:

- **Explain why the question is being asked.** Before each question, give brief
  context: what decision hangs on the answer, and what changes downstream
  depending on which way it goes.
- **Justify recommendations.** When marking an option "(Recommended)", state the
  reasoning for the recommendation — don't just label it.
- **Clarify jargon first.** Any technical term used in the explanation preceding
  a question (e.g. v∞, Oberth effect, patched conic, C3) must be defined in
  plain language at first use, either inline or in a short glossary line. Don't
  assume the reader holds the whole orbital-mechanics vocabulary in their head.

## Scratch / Todo Directory

`todos/` exists for Claude's scratch files, plans, notes, and todo lists. The directory is gitignored (`todos/*` is excluded, only `.gitkeep` is tracked), so write freely there without worrying about accidental commits. Use it for anything that would clutter the repo otherwise — intermediate analyses, draft notes, working scratchpads.

**Never commit anything that depends on `todos/` or other gitignored state.** Checked-in files (src, tests, ADRs, CONTEXT.md) must stand alone: no imports from `todos/`, no load-bearing references to scratch files, and no unrecorded provenance — if a committed result came from a scratch search or derivation, record the inputs that produced it (bounds, settings, seeds) in the committed file or ADR itself, not just a pointer to the scratch. Before committing, check that the change would still make sense if `todos/` were emptied; historical mentions of scratch files are fine only when explicitly marked gitignored/disposable. (Precedent: ADR 0007's numbers became unreproducible when its scratch harness vanished; ADR 0009 / `src/nozzle_analysis.py` records its search box for exactly this reason.)

## Testing

Tests live in `tests/` with one file per source module. `tests/test_helpers.py` has shared utilities including floating-point comparison helpers for physics quantities. Use `pytest markers` `slow` and `integration` when appropriate (defined in `pyproject.toml`).
