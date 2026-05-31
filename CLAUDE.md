# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

This is a Python scientific computing project implementing orbital mechanics calculations for PuffSat-based propulsion systems. The research paper it supports (`paper/Aim_Is_All_You_Need.pdf`) analyzes externally pulsed propulsion for missions including lunar transfer, interplanetary travel, and solar periapsis maneuvers.

## Reading the Paper PDF

The paper is only shipped as `paper/Aim_Is_All_You_Need.pdf` (46 pages) — there is **no
checked-in plain-text copy**. To get text, use **`pypdf`** (the only PDF library that is
installed):

```bash
conda activate puffsat_math_env          # pypdf lives in this env
python - <<'PY'
from pypdf import PdfReader
r = PdfReader("paper/Aim_Is_All_You_Need.pdf")
parts = []
for i, page in enumerate(r.pages, 1):
    parts.append(f"===== PAGE {i} =====")
    parts.append(page.extract_text())
open("/tmp/paper.txt", "w").write("\n".join(parts))
PY
grep -n "300 m/s to 600 m/s" /tmp/paper.txt   # find a claim by its wording
```

What's actually available (verified, so don't waste time rediscovering):
- `pypdf` (v6.12.2) is the **only** PDF lib installed — `pdfminer`, `PyMuPDF`/`fitz`, and
  `PyPDF2` are **not** importable.
- **No PDF CLI tools** are on PATH or in the conda env: `pdftotext`, `pdfinfo`, `mutool`,
  and `gs` are all absent. Don't reach for `pdftotext`.
- The built-in **Read tool can open the PDF directly** via its `pages=` parameter — handy
  for a quick look (read in chunks; it's 46 pages).

**Important caveat on line numbers:** citation references like "paper.txt L691" in
`todos/citation_audit_findings.md` were taken from one specific older `/tmp/paper.txt`
extract, and the **exact line numbers are NOT reproducible** — a fresh `pypdf` extract
shifts them (e.g. the "300 m/s to 600 m/s" claim is at L691 in the audit's file but L675
in a fresh extract). So **locate claims by `grep`-ing the quoted wording, not by trusting
the LNNN**, and treat the audit's line numbers as approximate pointers only.

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
