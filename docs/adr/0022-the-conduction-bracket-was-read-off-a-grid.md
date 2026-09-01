# The conduction bracket was read off a grid, and the shallow end starts at 23 solar radii

Status: accepted

Date: 2026-09-01

## Context

ADR 0021 re-scored the **launch ledger** from the pad and ruled the shallow
Jovian dive out twice over: it does not pay for its launch at any slug ratio,
and its **overtaking leg** does not conduct. The paper absorbed that result in
Balloon-Pulse-Propulsion commit `6406f32`, *"Pay the launch ledger the Jovian
dive cycle owed, and move its shallow end to 23 solar radii"*, which moved the
open problem out of "What This Does Not Settle" and into `sec:depth_cost`.

Writing it up turned up two defects in what this repository had handed over.
The paper states them plainly:

> ADR 0021's headline that the node fails at every reading at the 1.5 margin is
> a grid artefact: each cell's optimum sits at its own conduction threshold and
> PAD_FRONTIER_EXCESS_GRID steps 5 to 15 km/s, so every cell was understated and
> the frozen-chemistry cell actually clears.

> pad_floor_depth() puts the crossing at 21.09 solar radii, but it holds the
> climb-out at 75 km/s, and 75 km/s arrives at Earth at 85.07 km/s at 32 solar
> radii and 89.77 at 4. The overtaking pulse needs 90.42. So the whole depth
> dial, 4 solar radii included, is barred at that climb-out, and 21.09 is a
> crossing at an operating point the same ADR rules out.

Both are the same mistake in two places: **a constraint that moves was scored on
a dial that does not.** The conduction threshold moves with every reading of the
**conduction reserve** and with the expansion margin; the climb-out grid does
not move with it. The depth dial's climb-out is fixed at
`CONSERVATIVE_RETURN_EXCESS` = 75 km/s; the threshold it has to clear is not.

Neither correction was reproducible from `make dive-depth` as it shipped — the
paper computed both on a finer grid than the repository carries, and recorded
that fact rather than quietly quoting them. This ADR absorbs them.

## Decision

Bisect to the constraint instead of stepping past it.

- `overtaking_coldest_closing_at()` — the coldest instant of the first push at
  one climb-out. Slug ratio does not enter it; depth does, through the closure.
- `conduction_threshold_excess()` — that map inverted: the **slowest climb-out
  whose overtaking plume still conducts**, at a given depth, reserve and margin.
- `pad_frontier_optimum()` — maximises the pad return over the climb-out above
  that threshold, which is where the answer always is.
- `conduction_bracket_frontier()` is rewired onto it, and gains
  `threshold_return_excess` per cell so the axis the optimum hugs is visible in
  the output.
- `admissible_pad_floor_depth()` — the depth crossing with every depth flown at
  its own best conducting climb-out, replacing `pad_floor_depth()` as the
  recommendation.

`pad_return_frontier()` and its `PAD_FRONTIER_EXCESS_GRID` are unchanged and
still what plots the shape of the squeeze — which is also what keeps the size of
this correction checkable, since the superseded reading is the maximum over that
same grid. `pad_floor_depth()` is likewise kept, with its docstring saying why
its answer is not the recommendation.

New search bounds are recorded in the module per CLAUDE.md's rule after ADR
0007: `_CONDUCTION_EXCESS_BRACKET`, `PAD_OPTIMUM_EXCESS_SPAN`,
`_PAD_OPTIMUM_EXCESS_XATOL`, `ADMISSIBLE_PAD_FLOOR_DEPTH_BRACKET`,
`_ADMISSIBLE_PAD_FLOOR_XTOL`.

Every table below comes from one call at the module's own defaults
(`eta_jet**2` = 0.60, a 5-day parking ellipse, node survival derived at each
point), so each is reproducible without the scratch it was found in:

| table | call |
| --- | --- |
| the pinch | `pad_return_frontier(32.0, (61.887, 61.9, 62.34, 63, 65, 70, 75), expansion_margin=1.5, reserve=53.47)` |
| the corrected bracket | `conduction_bracket_frontier(32.0)` |
| 4 solar radii | `conduction_bracket_frontier(4.0)` |
| the depth dial | `pad_frontier_optimum(d)` for each `d`, and `admissible_pad_floor_depth()` |

`python -m src.solar_dive_depth_trade --pad-frontier` prints the middle three
and the crossing; `make dive-depth` runs the same module without them.

## Consequences

### Why a grid could not have scored this sweep, at any spacing worth running

The mechanism is a pinch, and it is worth stating because it decides where the
optimum lives rather than merely how accurate it is.

Dropping the climb-out raises the pad return: less of the node's propellant is
spent, so more mass comes home. It also narrows the **expansion floor**'s window
on the slug ratio, and at the threshold that window closes to a *point* — near
`k` = 4.0 at 32 solar radii. The pad return peaks well below that — at `k` = 1.93
at ADR 0021's reference point, and below 2.3 everywhere on this frontier — so a
pinched window forces a loading the ledger does not want. The two effects meet
within about a kilometre per second of the threshold:

| climb-out | window | best `k` | kg/pad kg | vs 1/15 |
| ---: | --- | ---: | ---: | ---: |
| 61.887 (the threshold) | closed | -- | -- | -- |
| 61.90 | [3.912, 4.091] | 3.91 | 0.0667 | 1.001 |
| **62.34** | **[3.517, 4.598]** | **3.52** | **0.0673** | **1.009** |
| 63.00 | [3.286, 5.000] | 3.29 | 0.0670 | 1.005 |
| 65.00 | [2.924, 5.890] | 2.92 | 0.0653 | 0.979 |
| 70.00 | [2.507, 7.708] | 2.51 | 0.0596 | 0.894 |
| 75.00 | [2.287, 9.446] | 2.29 | 0.0536 | 0.804 |

(32 solar radii, frozen chemistry only, 1.5x margin.) The peak sits 0.45 km/s
above the threshold; the first grid point that clears the threshold is 65, a
further 2.7 km/s past the peak, and it scores the cell 3 percent low. **The grid
was not too coarse; it was on the wrong axis.** Every cell's optimum is anchored to
its own threshold, and the threshold moves 46.7 to 81.2 km/s across the six
cells while the grid stands still.

### The corrected bracket, and the headline it retires

At 32 solar radii, node survival derived at each point, both legs held to the
margin:

| conduction reserve | margin | dies below | slowest `v_inf` | best `v_inf` | `k` | kg/pad kg | vs 1/15 | doubling | ADR 0021 said | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 15000 K, all reserved | 1.5 | 79.57 | 81.22 | 81.43 | 3.69 | 0.0443 | **0.665** | 2.852 yr | 0.625 | fails |
| 6000 K | 1.5 | 70.28 | 70.34 | 70.66 | 3.60 | 0.0566 | **0.849** | 2.319 | 0.794 | fails |
| frozen chemistry only | 1.5 | 63.33 | 61.89 | 62.34 | 3.52 | 0.0673 | **1.009** | 2.066 | 0.979 | **CLEARS** |
| 15000 K, all reserved | 1.0 | 64.97 | 63.91 | 64.33 | 3.54 | 0.0646 | **0.970** | 2.118 | 0.966 | fails |
| 6000 K | 1.0 | 57.38 | 54.33 | 54.98 | 3.42 | 0.0773 | **1.159** | 1.914 | 1.159 | **CLEARS** |
| frozen chemistry only | 1.0 | 51.71 | 46.70 | 47.65 | 3.31 | 0.0874 | **1.312** | 1.812 | 1.291 | **CLEARS** |

Every cell moved up and the direction was never in doubt — a maximum found on a
subset can only be too low. **ADR 0021's "at the 1.5x operating margin the
shallow cycle fails at every reading of the bracket" is retired.** It fails at
two of three. The hard end — charging only the frozen chemistry, reserving
nothing thermal — clears by nine parts in a thousand.

Note which cell moved least: 6000 K at unit margin, 1.159 either way. Its
threshold happened to fall 0.7 km/s below a grid point. That is the whole
character of the defect — not a bias with a size, but a set of six cells each
scored at wherever its own grid neighbour happened to sit.

Two figures ADR 0021 derives from the top row move with it: the best admissible
point at 32 solar radii returns **1/22.5** rather than 1/24.0, and it doubles in
**2.852 yr** rather than 3.993. Both are quoted there in the sensitivity
argument about the floor's calibration, and both leave that argument standing —
1/22.5 is still a 50 percent relaxation of the committed floor, and 2.852 yr
still loses to the chain.

### The conclusion survives, on the clock rather than on the floor

This changes what carries the verdict at the loosest corner, and does not change
the verdict:

| cell | vs 1/15 | doubling |
| --- | ---: | ---: |
| frozen chemistry, 1.5x margin | 1.009 | **2.066 yr** |
| 6000 K, unit margin | 1.159 | **1.914** |
| frozen chemistry, unit margin | 1.312 | **1.812** |
| the Jupiter-only chain | — | **1.74** |
| 4 solar radii, `k` = 30 | 1.043 | **0.724** |

**Every reading of the bracket that lets the shallow node pay leaves it doubling
in 1.81 to 2.07 years, and the Jupiter-only chain the paper already has doubles
in 1.74.** So the shallow node is beaten by an existing cycle wherever it is
admissible at all, which is the same conclusion ADR 0021 reached, resting on a
comparison that does not depend on the 1/15 floor's calibration. ADR 0021 got
there by saying the node never pays at the operating margin; that argument is
gone and this one replaces it.

The paper quotes 0.67 / 0.85 / 1.01 and 0.97 / 1.16 / 1.31, and a 1.82 to 2.08
year spread on the cells that pay. It found them on its own finer grid; these
are the same numbers to the tolerance either search was run at.

### At 4 solar radii the grid was already right, which is the test of the diagnosis

If the defect were "the grid is too coarse" it would bite everywhere. It does
not:

| conduction reserve | margin | slowest `v_inf` | best `v_inf` | vs 1/15 | ADR 0021 said |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15000 K, all reserved | 1.5 | 75.78 | 83.50 | 2.422 | 2.421 |
| 6000 K | 1.5 | 64.39 | 75.97 | 2.488 | 2.487 |
| frozen chemistry only | 1.5 | 55.36 | 70.97 | 2.530 | 2.529 |
| 15000 K, all reserved | 1.0 | 57.54 | 72.09 | 2.520 | 2.519 |
| 6000 K | 1.0 | 47.08 | 67.15 | 2.561 | 2.560 |
| frozen chemistry only | 1.0 | 38.38 | 63.93 | 2.588 | 2.587 |

**One part in two thousand, on every cell.** The deep node's optima sit 8 to 26
km/s *above* their thresholds — its node is cheap enough that the pinch costs
more than the boost saves for a long way up — so they are interior peaks on a
broad ridge, and a 5 km/s grid finds those. The artefact needs an optimum pinned
against a constraint the grid does not track, and only the shallow node has one.

### The depth dial, flown at climb-outs it may actually fly

`pad_floor_depth()` holds the climb-out at 75 km/s. That climb-out leaves the
overtaking leg at 74.21 km/s at 32 solar radii, 75.81 at 21.09 and 78.91 at 4,
against a floor of 79.57 — **so no depth on that dial conducts**, and its 21.09
crossing is a crossing between two inadmissible rows. Flying each depth at the
climb-out that maximises its own pad return subject to conducting:

| R_sun | slowest `v_inf` | best `v_inf` | `k` | kg/pad kg | vs 1/15 | doubling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 79.20 | 79.64 | 3.56 | 0.0761 | 1.142 | 1.545 yr |
| 22 | 79.55 | 79.93 | 3.59 | 0.0695 | **1.043** | 1.677 |
| **22.93** | 79.71 | 80.06 | 3.60 | 0.0667 | **1.000** | **1.746** |
| 23 | 79.72 | 80.07 | 3.60 | 0.0664 | 0.997 | 1.751 |
| 24 | 79.89 | 80.22 | 3.61 | 0.0635 | 0.952 | 1.832 |
| 26 | 80.23 | 80.51 | 3.64 | 0.0580 | 0.870 | 2.015 |

**The committed floor is lost at 22.93 solar radii** (`admissible_pad_floor_depth()`),
not 21.09. The paper describes this dial as flying "the slowest climb-out that
does conduct", which is a simplification of the same construction — at the
threshold itself the window is pinched and the best point is a fraction of a
kilometre per second above it. Its per-depth digits are the optimised ones and
match these exactly; its crossing doubling of 1.742 yr against this 1.746 is the
two searches' resolution, not a disagreement. The correction is small in depth and large in what it licenses,
because the paper's recommendation was "fly a shallow version first" and the
shallowest version that pays for itself is what that sentence names. **It starts
at 23 solar radii rather than at 32.**

Backing out from 4 to 23 solar radii still divides the solar flux by 33 and the
equilibrium temperature by 2.4, so most of the thermal relief the shallow
version was for survives. What is lost is the coolest end of it — the end that
made the first flight the easy one.

And note where the crossing lands on the clock: **the cycle at 22.93 solar radii
doubles in 1.746 years, within half a percent of the Jupiter-only chain's
1.74.** The depth at which this architecture stops paying for its own rocket and
the depth at which it stops beating a cycle the paper already has are, to the
precision either is known to, the same depth. Nothing was arranged to make that
happen, and nothing in the derivation connects them.

### What does not change

- **32 solar radii at the reference operating point still fails**, 0.618x at
  `k` = 8.5 and 0.761x at the best `k`. ADR 0021's reference-point verdict is
  untouched; only the frontier's optima moved.
- **4 solar radii still clears at 1.043x** at `k` = 30 and 1.326x at `k` = 8.5.
- **The ranking against the paper's own resonant dive is untouched.** Nothing
  here re-scores it: the correction moves frontier optima, and the comparison
  row is a reference operating point.
- The **overtaking-leg conduction floor** itself is unchanged — same threshold,
  79.57 km/s at the operating margin. What changed is that it is now *solved
  for* rather than stepped past.

## What this does not settle

- **The conduction reserve is still a bracket, and its hard end now clears.**
  ADR 0021 could say the shallow node failed at the operating margin whatever
  the reserve. It cannot: at frozen chemistry and 1.5x it pays, by 0.9 percent.
  The verdict at that corner now rests on the doubling comparison (2.07 yr
  against the chain's 1.74) rather than on the pad floor. What would close it is
  the exit magnetic Reynolds number, still uncomputed — and it is now decisive
  for a *reason*, rather than merely load-bearing.
- **22.93 inherits the 1/15 floor's calibration.** It is a crossing of one round
  number that ADR 0021 already flagged as the weakest link, and it moves roughly
  a solar radius per 5 percent of floor. Quote the depth as "about 23", which is
  what the paper does, and not as 22.93.
- **The crossing was found at the conservative reserve and the 1.5x margin.**
  A looser reading of either moves it shallower, on the same dial as the
  bracket above; it has not been swept, because the recommendation should sit on
  the conservative reading.
- **`pad_floor_depth()` is kept**, labelled, because ADR 0021 quotes 21.09 and
  the size of this correction should stay checkable — the same treatment ADR
  0021 gave `slug_ratio_ceilings()`. Anything quoting a depth recommendation
  should use `admissible_pad_floor_depth()`.
- **It costs about four times what the grid did**: the bracket sweep goes from
  roughly one minute per depth to four, because every evaluation now lands on an
  admissible point and pays for the inner slug-ratio optimisation, where most
  grid points were barred and returned early. The depth crossing adds about
  three and a half minutes on top, each of its bisection steps being a full
  climb-out optimisation. Both stay behind `--pad-frontier` and behind the
  `slow` test marker, which is where the budget for them is.
- **Circular and coplanar**, as ADR 0019, ADR 0020 and ADR 0021.
