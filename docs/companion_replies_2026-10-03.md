# Companion replies, 2026-10-03, for the parent paper

Answers two asks `Balloon-Pulse-Propulsion` raised on 2026-10-03 after checking
the paper against companion `22853ce` (post ADR 0038). Each needed new code,
because a plain rerun could not produce it. **Written to be copied into the
paper repo**, so the quoted paper text is restated here.

**Nothing has been written into `templateArxiv.tex`.**

## C1. Methane pitch sensitivity in the growth ledger

**Paper text** (`sec:growth_ledger`, "The smaller settings move the answer
little"): "Methane's pitch, from 1.4 to 5.6 kg per pulse, moves its doubling
by 0.01 to 0.02 yr."

**Reproduce:** `make growth-ledger`. The last table is the new pitch sweep,
`pitch_sweep()` in `src/growth_ledger.py`. The matrix still carries the
pessimistic 5.6 kg (`METHANE_PITCH`). The sweep's 5.6 kg column matches the
matrix rows.

Doubling time (yr) saved by flying 1.4 kg instead of 5.6 kg, flown chain,
10-day split:

| plate | CH4 50% | CH4 70% | CH4 solved (80%, eta 0.538) | CH4 100% |
|---|---|---|---|---|
| 0.5 | 0.033 | 0.018 | 0.015 | 0.011 |
| 0.7 | 0.007 | 0.011 | **0.009** | 0.008 |
| 1.0 | -0.006 | 0.005 | 0.005 | 0.005 |

Solved methane behind 0.7, absolute: 1.733 yr at 1.4 kg, 1.743 yr at 5.6 kg.

**What the paper should say.** "0.01 to 0.02 yr" fits solved methane behind
0.5 to 0.7 plates (0.009 to 0.015 yr), but not the full range. Two options:

- Narrow it to the headline case: "...moves solved methane's doubling behind a
  0.7 plate by 0.01 yr (1.74 to 1.73 yr)."
- Keep the range sentence but make it true: "...moves its doubling by under
  0.04 yr across the matrix, and by 0.01 yr for solved methane behind a 0.7
  plate."

**One cell is unexplained.** Behind a 1.0 plate at CH4 50%, the lighter pitch
doubles 0.006 yr *slower*. That is probably the per-cycle schedule optimiser's
resolution rather than physics, but it has not been checked. Don't quote that
cell as an effect.

## C2. The plate chain at f = 0.818 in a `make` target

**Paper text** (`sec:two_leg_nozzle`, chemistry-charged comparison): two
nozzles return 7.42e6 against 3.36e5 for the plate chain at f = 0.818
(eta_geom 0.9), and 5.83e5 against 5.66e4 at 0.8.

**Reproduce:** `make two-leg`. The ADR 0016 matched-diagonal table now has a
`measured_plate_growth` column at f = 0.818 (`MEASURED_PLATE_ELASTICITY` in
`src/two_leg_nozzle_sweep.py`), next to `plate_growth` at the stated f = 0.8.

| eta_geom | nozzle (two legs) | plate, f = 0.8 | plate, f = 0.818 |
|---|---|---|---|
| 0.9 | 7.419e6 | 2.815e5 | 3.364e5 |
| 0.8 | 5.825e5 | 4.762e4 | 5.662e4 |

The paper's four figures stand as written. The sweep's grid stops at eta_geom
0.9, so ADR 0038's 1.0 cell (1.199e6) is not printed. The paper does not quote
it. `tests/test_two_wave_growth.py` pins it.

## Not asked, for completeness

The other two items the paper session checked needed no companion code:

- **10-day parking-orbit sensitivity:** reproduce with
  `python -m src.growth_ledger --split-days 10`.
- **"About 2.9 times the mass" (`sec:split_tail`):** this is the stricter
  chain's mean growth per cycle, total^(1/10), from `make sep-split-10d`. The
  paper has already corrected it to 2.8.
