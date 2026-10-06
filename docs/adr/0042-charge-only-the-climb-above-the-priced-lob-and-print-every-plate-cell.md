# Charge only the climb above the priced lob, and print every cell the parent quotes behind a plate

Status: accepted

Amends: ADR 0041 §2-3 (the climbing lob's charge). Adds reproductions for the parent's
spray-cup tables.

Date: 2026-10-06

## Context

The parent applied ADR 0041 and the impact sim's Draft 2 (`7e1d8f7`) at its ADR 0025. Two things
came back.

**The climb was charged from the wrong baseline.** `lob_rise.booster_growth` returned
`lofted_mass(0) / lofted_mass(v)`, the lofted mass of a lob whose apex is at 400 km over one
still climbing at `v`. The parent's `sec:vertical_lob` lofts its 1250-1430 t to a top near
430 km, so that lob already passes 400 km at about 0.75 km/s. ADR 0037's 1.1-1.2x booster, and
the $25/kg lob it prices, were sized from those figures. Charging from an apex counts the first
0.75 km/s twice.

**The parent prints more than ADR 0041 ran.** ADR 0041 ran the solved chambers and methalox.
The parent's tables also quote every chamber share of its ceiling, the k <= 10 sensitivity, the
hold, pitch, water and 10-day sensitivities, the full cost report (odds, rates, matching prices,
plate learning, packages, line sensitivities, seed routes), `tab:seed_amortization`, and the L1
deliveries and grow-or-harvest table, all behind the spray cup. Those were first run paper-side
from a scratch driver over this repo's functions.

## Decision

1. **`booster_growth(v, isp, baseline=0.75 km/s)`.** The default baseline is the parent's lob
   (`BASELINE_RISE_SPEED`). `baseline=0` keeps ADR 0041's first measure. The operating climb is
   `OPERATING_RISE_SPEED` = 1.1 km/s, the one ADR 0041 found holds the unit within ~65 km of the
   stream.

   | climb at 400 km | 380 s | 350 s |
   |---|---|---|
   | 1.0 km/s | 1.029 | 1.032 |
   | **1.1 km/s** | **1.044** | 1.048 |
   | 1.2 km/s | 1.059 | 1.065 |

   The lob becomes $26.1/kg lofted at 1.1 km/s, against ADR 0041's $26.8-27.5. The booster is
   about 1.13-1.28x Super Heavy, against 1.18-1.33x. `make plate-designs-cost` now prints the
   smaller charge; ADR 0041's cost columns are superseded by about -$2 to -$4/kg steady.

2. **`--plate NAME` and `--designs-grid NAME`** (`PLATE_DESIGNS_BY_NAME`: `spray-cup`,
   `spray-cup-unmixed`, `plug`, `paper-0.775`, `adr-0033`).
   - `growth_ledger --designs-grid`: every chamber share plus solved, and methalox, at the
     design's cap with k <= 10 beside it; then the hold, pitch, all-in water (η_jet 0.50/0.53/0.56,
     `WATER_SLUG`, `NO_BONDS`) and 10-day sensitivities.
   - `growth_cost_report --plate`: sections 2-9 behind the plate, every book's lob climbed at
     1.1 km/s. The price books are now passed to each section instead of read from the module,
     so the default report is unchanged.
   - `seed_cost --plate` and `harvest --plate`: `tab:seed_amortization`, the L1 deliveries
     (lob climbed) and the grow-or-harvest table.
   - Targets: `make plate-grid`, `make plate-cost`, `make plate-seed`.

3. **Every new call passes the 20-day split.** `two_wave_growth.DEFAULT_SPLIT_DAYS` is 10 while
   `growth_ledger.DEFAULT_PARKING_DAYS` is 20. A bare `adaptive_two_wave_cycles()` silently flies
   the 10-day orbit. The parent's first scratch grid did exactly that. The defaults are left as
   they are, since other analyses rely on the 10-day one; the new code never calls it bare.

## Results behind the spray cup (η_jet 0.60, k <= 8.52, 20-day orbit)

Solved hydrogen doubles in 2.02 yr (2.00 at k <= 10), solved methane 2.28, methalox 7.82.
Steady $119 / $122 / $285 per kg at L1; break-even cheap / dear $189 / $370, $200 / $465,
$755 / >3000. Seed 102-112 t per launch unit (1.11-1.22 stripped ships). Full tables in
`docs/plate_grid_for_parent.md`.

## Not done

- The film's mass (parent ask S12): 5-9 t of pitch per push vapor-shielded, 34-50 t unshielded.
  Neither the ledger nor the cost book carries it; the book's film line is still 0.1% / 4%.
- The booster's brake at the faster climb (parent's `sec:vertical_lob`).
- The plug in the deep bowl (impact sim).
