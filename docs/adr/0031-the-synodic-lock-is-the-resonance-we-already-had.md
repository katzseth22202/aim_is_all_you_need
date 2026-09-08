# The 2S synodic lock is the resonance we already had, and the coast is what proves it

Status: accepted

Amends: ADR `0030-the-chain-search-was-reading-its-own-search-box`. Its decisions
stand; this ADR discharges the second of its N4 caveats (that fly-and-park was a
single-cycle comparison and not a chain run), corrects one constant it shipped,
and answers asks **S5-S8** from the paper repository's
`docs/fly_and_park_asks_for_aim_repo.md`.

Date: 2026-09-08

## Context

The paper repository read ADR 0030's fly-and-park result and asked for four
things. The largest, **S5**, proposed a **2.00 S synodic lock**: pad a sub-2.00 S
flight up to exactly two synodic periods and the cycle becomes a departure-phase
fixed point by construction, removing the drift that stops the phased chain from
ever sustaining a 2S cadence. It came with a worked construction -- departure
phase 0.781, flight 1.978 S, park 0.022 S, `dv` 7.25 km/s, `v_b` 63.2 km/s,
doubling **0.84 yr** at Isp 2214 -- and with the right question attached: *that
park is nine days, and the cycle orbit's period is twenty. Is a nine-day coast
admissible?*

It is not, and the reason turns out to settle the whole item.

The **park is the coast, lengthened**. Every cycle in this architecture already
ends with one full period of the bound near-escape orbit: the returning wave
pushes the payload up at periapsis, and the departure burn lights at the *next*
periapsis, which is the aim reversal that turns a retrograde push into a prograde
departure. `optimize_jovian_cycle_chain()` encodes exactly this -- each departure
is pinned to the previous arrival plus `PUFFSAT_CYCLE_ORBIT_PERIOD`. Parking
lengthens that orbit. It cannot shorten it.

`src/fly_and_park.py` shipped with `MINIMUM_PARK = 0.02` synodic periods, about
eight days: **2.5x under the coast it was standing in for**. Nothing published in
ADR 0030 turned on the difference, because the winning 3S cycles park for 83 to
106 days. The 2S lock was constructed precisely on the difference.

## Decision

**1. `MINIMUM_PARK` is the coast, and nothing looser.** It is now
`COAST_SYNODICS` -- `PUFFSAT_CYCLE_ORBIT_PERIOD` expressed in synodic periods,
0.05013 S. A candidate is admissible when `flight + COAST_SYNODICS <= target`.

**2. The synodic target is an argument, not a constant.**
`fly_and_park_comparison`, `perfect_retrograde_premium`, `phase_reaches` and
`hottest_reachable` all take `target_synodics`, and `synodic_lock()` returns the
best locking cycle at any target. This is what S5 asked for.

**3. The chain check is run, not argued** (`sustainable_chain_optimum()`). The
existing beam search structurally *cannot* park -- it pins departure to arrival
plus a fixed coast -- so a new chain was needed rather than a rerun of the old
one. It searches the same enumerated branches over policies that may park any
length, as a maximum-ratio-cycle over the departure-phase graph, and reports the
best rate any repeating policy can hold.

**4. The paper may quote the 2S lock, but not as a new cycle.** It is ADR 0011's
two-synodic resonance, which this repository already publishes.

## What the numbers say

**The nine-day lock cannot be flown.** Its park is 8.7 days against a 20-day
coast. Priced at Isp 2214 it doubles in 0.843 yr, which is the figure S5 carried.

**The admissible 2S lock exists, and it is not new.** Charge the coast and the
best 2.00 S lock moves to departure phase **0.8082**, flight **1.9497 S**, park
**20.06 days** -- the coast plus about ninety minutes -- `dv` **8.613 km/s**,
`v_b` **63.35 km/s**. That is, to every digit, the two-synodic fixed point
`fixed_points()` already reported and ADR 0030's N4 already published. The test
asserts cycle identity, not resemblance.

**So fly-and-park does not produce the 2S operating point.** The lock's park *is*
the mandatory coast, so there is nothing left for the construction to add. What
fly-and-park buys at 2S is nothing; what it buys at 3S is what ADR 0030 already
recorded.

| Isp | 2.00 S lock | 3.00 S lock |
| ---: | :--- | :--- |
| 380 (methalox) | **none admissible** | 3.641 yr, park 21 d |
| 1200 | 1.082 yr, phase 0.8082, park 20 d | 1.377 yr, park 83 d |
| 2214 | **0.873 yr**, phase 0.8082, park 20 d | 1.189 yr, park 106 d |

Charging the coast costs the 2S lock **+3.5%** on the clock (0.843 -> 0.873 yr at
Isp 2214, 1.001 -> 1.082 at 1200) and two departure phases (28 -> 26 of 73 at
2214, 13 -> 11 at 1200). The correction is small; that it lands on a cycle we
already had is the part worth having.

**The chain check agrees with the single cycle.** At every exhaust speed the best
sustainable policy is a **single cycle returning to its own phase** -- a lock --
and it is the same cycle `synodic_lock()` picks, to four digits:

| Isp | loop | length | phase | doubling |
| ---: | ---: | ---: | ---: | ---: |
| 380 | 1 cycle | 3.0000 S | 0.7397 | 3.641 yr |
| 1200 | 1 cycle | 2.0000 S | 0.8082 | 1.082 yr |
| 2214 | 1 cycle | 2.0000 S | 0.8082 | 0.873 yr |

So ADR 0030's N4.2 caveat -- that a single-cycle optimum need not survive a chain
lookahead -- is discharged for this result rather than merely restated. The
methalox row is the same verdict ADR 0030 reached by a different route: the 2S
point costs an 8.613 km/s departure burn, which at 3.727 km/s of exhaust delivers
9.9% of the vehicle, so the chain's refusal of it is correct.

**The branch dedup was audited and is innocent.** `_cycle_branches` collapses
branches landing in one 7-day bucket, keeping the highest growth *as scored on
methalox*, which in principle could hide a branch that only wins at Isp 2214.
Re-enumerating the whole grid with the dedup scored at Isp 2214 instead returns
every figure above unchanged to four digits.

**The usable phases form one window, not several** (S7). Cyclic run-length over
the same 73-phase grid, at every exhaust speed tested:

| Isp (s) | 380 | 700 | 1000 | 1200 | 1500 | 1800 | 1900 | 2214 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| usable phases | 13 | 26 | 36 | 44 | 56 | 69 | 73 | 73 |
| windows | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| widest (d of 399) | 71 | 142 | 197 | 240 | 306 | 377 | 399 | 399 |

The arc must be measured cyclically or the Isp 1800 row, which wraps through
phase 0, reports two windows. So the operational claim is **one window, 3.4x
wider at Isp 1200**, which is both simpler and stronger than a set of windows.

**The flown chain's 2S cycles are already exact locks** (S8). The real-ephemeris
adaptive cadence builds every return on an exact synodic multiple of the circular
synodic period, so each of the eleven flown cycles has a departure-phase drift
below 1e-4 S. The seven 2S cycles carry `dv` 6.84-7.17 km/s and growth-wave `v_b`
61.83-65.13; the circular lock's 63.35 km/s arrival lands inside that range and
its 8.613 km/s burn sits about 20% above the dearest. Different models, so this
is a family resemblance rather than an identity -- but the paper's claim is
"lock a cycle we already fly", and it is the correct one.

**What stops a clean 2S cadence is therefore not phase drift.** The flown chain
still takes four 3S returns out of eleven, and ADR 0011 says why: only 45 of 91
windows over 200 years clear the 4,000 km perijove floor. The lock removes a
problem the architecture did not have.

## The doubling ladder, and why the rungs do not compare (S6)

`sec:jupiter_only_growth` publishes 4.0 yr, 3.0 yr, 1.74 yr and 1.45 yr. Adding
1.189 and 0.873 to that list unlabelled would put six numbers side by side that
appear to disagree at matched efficiency and do not measure the same thing.

**They are not at matched `f`.** The paper's "1.74 yr at `f` = 0.6 and 1.45 at
`f` = 0.8" quotes the **nozzle impulse recovery** `e` of
`two_wave_growth.price_chain`, which derates the whole impulse from outside the
momentum debit. The elasticity `f` is held at **0.8 in both**, and it is
`STD_FUDGE_FACTOR`, the collision elasticity in `eq:PuffSat_ratio`. The paper
already makes exactly this distinction for the dive family in `sec:depth_cost`
-- "the nozzle efficiencies are not the same parameter and should not be read as
one" -- and it applies here unchanged.

**And they are not at matched scope or accounting.** Three axes separate the new
rungs from the 1.45:

| | fly-and-park lock | two-wave chain |
| --- | --- | --- |
| model | circular coplanar, relative epoch | real ephemeris, calendar epochs |
| scope | best single cycle at its best phase | 11 flown cycles including the dear ones |
| departure charged as | `exp(-dv/v_e)` at Isp 2214 | two-currency nozzle ledger, recovery `e` |
| projectile stream | **not charged at all** | 20-29% of the batch on the nozzle bend |

The last row is the one that matters. `fly_and_park` charges the slug against the
payload and mints `M(v_b)` from the whole arriving wave; the two-wave ledger
sources the projectiles explicitly by splitting the batch -- 19.5-23.6% of it onto
the nozzle bend at `e` = 0.8 across the flown chain, 24.2-29.0% at `e` = 0.6. CONTEXT.md's
**departure-burn accounting seam** already records that nothing is charged for
delivering the head-on stream. So **0.873 yr is systematically optimistic against
1.45 yr, and is not a refinement of it.** The paper should state it as a
different measurement, with its scorer attached, or not at all.

`doubling_ladder()` emits this module's rungs carrying `scorer`, `model`, `scope`
and `efficiency` strings, and deliberately does *not* restate the other rungs:
copying another module's ledger figure into this one is how they drift.

## Consequences

- One published figure moves: fly-and-park beats the plain 3S cycle at **16** of
  30 phases at Isp 1200, not 17. The Isp 380 (1 of 11) and Isp 2214 (25 of 30)
  counts are unchanged, as are the exchange rate, the sweet phase, the
  usable-phase fractions and the 5%/27% gain shape.
- `perfect_retrograde_premium()` now requires the hot cycle to be at least as hot
  as the cycle it is measured against. Charging the coast shrinks the parkable
  set enough that one phase's hottest survivor was *colder* than its pure-3S
  cycle (67.16 against 69.03 at phase 0.452), and calling that difference a
  premium would price a downgrade. The published range is unchanged and the
  median moves 1.036 -> **1.037** km/s over 29 phases rather than 30.
- The paper's held 0.84 yr figure does not lift. **0.873 yr** does, at Isp 2214,
  with the ladder caveats above attached.

## Considered and rejected

**Shrinking the parking orbit so a nine-day park becomes legal.** A 8.7-day
cycle orbit is a different orbit with a different periapsis speed, so it changes
`v_rf`, the mass ratio, the aim reversal and the split gap together. That is an
architecture change priced nowhere in this repository, not a park length. If the
paper wants it, it is a new question and not this one.

**Reporting the 0.84 yr lock with a caveat instead of correcting it.** The park
is a hard constraint of the closed cycle, not a modelling preference. A figure
that needs an inadmissible coast is wrong rather than optimistic.

**Re-pricing the fly-and-park locks through the two-wave nozzle ledger** to put
a matched rung on the doubling ladder. The two-wave split geometry does not exist
in the circular phased model -- there is no second Earth-hit return to solve --
so a matched rung would have to be assembled from an assumed split fraction. The
mapping between `e` and `eta_jet` is unworked, and this ADR is not the place to
work it; naming the scorers is the honest answer until it is.

## Reproduction

`make fly-park` (~18 s) prints the launch-window layout, the synodic locks with
both the inadmissible and the admissible 2S row, the chain check, and the ladder.
`make two-wave` prints the flown chain with its new `period_synodics` column.

Search box, unchanged from ADR 0030: 73 departure phases over one synodic period,
`_cycle_branches` at the committed settings with `_OUTBOUND_TOF_MIN` freed to
0.70 yr, growth re-scored as `M(v_b) * exp(-dv/v_e)` with `f` = 0.8 and `v_rf` =
10.9503 km/s. The chain search adds one parameter of its own: the maximum-ratio
bisection is bracketed on [0, 3] e-foldings per synodic period and run 60 steps,
which puts the rate inside float noise.

Pinned by `tests/test_fly_and_park.py` (the coast identity, the inadmissible
lock, the admissible lock's identity with `fixed_points()`, the +3.5% cost of
charging the coast, every lock padding exactly to its target, the chain optimum
being that same lock, the 3S-on-methalox / 2S-above-it verdict, the one-window
layout and its 3.4x widening, and the ladder carrying its scorer) and by
`tests/test_two_wave_growth.py` (every flown cycle an exact lock; the 2S family's
burn and arrival ranges bracketing the circular lock; the projectile-bend
fraction the fly-and-park scorer omits).
