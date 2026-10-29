# ADR 0088: Milestone 1 on methalox barely grows; a smaller seed costs dollars, not the loop

Date: 2026-10-29

Status: Accepted. It answers the parent's S63 (`docs/milestone_one_asks_for_aim_repo.md` in
`Balloon-Pulse-Propulsion` @ `e30114b`) and the S29 it re-raises. Parent note:
`docs/milestone_one_answers_for_parent.md`. Reproduce with `make methalox-loop` (~1 min) and
`make small-seed` (~1 min).

## Context

The child (`puffsats_for_datacenters` @ `480194d`) prints the cycle as three milestones. The
first is the Jupiter-only loop with Kapton sheets overtaking the push plate and a **methalox**
departure in place of the bowl. Its only priced ancestor is `growth_ledger`'s methalox
incumbent, behind the spray cup: 7.92 / 9.66 yr doubling (20- / 10-day orbit), $353 steady,
$2323 / >$5000. S63 asks for:

1. that departure on ADR 0075's plates;
2. a sweep of slower departures;
3. S29 again: is anything indivisible below one launch unit, and what do both headlines give
   on a 1- or 2-Starship seed;
4. the headline seed's tonnage. The paper prints 622 t / 725 t; the parent's scratch on
   `c559cc7` gets 567 t / 674 t.

## Decision

### Milestone 1 (`src/methalox_jupiter_loop.py`)

- **The Earth half is shared, not copied.** `solar_dive_growth.fly_earth` is split.
  `push_and_turn` holds the 400 km push and the apoapsis turn, and `fly_earth` adds the bowl.
  The split leaves every solar-dive ledger line byte-identical.
- **The departure is the incumbent's.** `growth_ledger.best_methalox_departure`, split out of
  `price_methalox_cycle`: Raptor 3 at 380 s, 0.018 tank fraction, engines optimized, steered
  loss on the 10-day orbit from 600 km. Engines and tanks are expended and priced (the cost
  book's `departure_stage`, $1M a Raptor and $40/kg of tank).
- **The push plate is left at apoapsis.** With no bowl there is nothing to share it with, so
  `METHALOX_START` is ADR 0075's start with `shared_spring=False`. The plate and its bags are
  68.7 t a unit.
- **No role split.** Every ship flies the chain's growth trajectory, which returns
  `PARKING_DAYS` before the next departure. `role_growth` gains the degenerate case with no
  rods spent (`x = 1`, `g = O a / P`).
- **The burn axis** is the departure asymptote turned back by the hyperbola's half-turn at
  600 km, on the side that turns the orbit least at apoapsis.
- **The gates** are ADR 0075's. The last is renamed to the push and the Raptor burn and kept
  at 90%, since the parent gives no split of it.
- **`plate_jupiter_loop`'s valuation and optimizer take the unit pair as a `fly` argument**
  (`bowl_pair` by default), plus `knobs` and `gates`. Milestone 1 tunes only `k_push` and
  `k_delivery`.

| Chain (dear-seed optimum) | k push / delivery | growth per 3S | doubling | steady $/kg | break-even cheap / dear | seed |
|---|---|---|---|---|---|---|
| 3S only (9 windows) | 5.00 / 5.19 | **1.08** | 30.6 yr | 1344 | **10559 / 34376** | 119 t at 11.99 km/s, 1.29 / 1.65 ships |
| flown 2S/3S (11 windows) | 5.11 / 5.20 | 0.82 | does not grow | 506 | 94803 / 336109 | 77 t |

- **Per window on 3S only:**
  - push stream 53.9-57.6 km/s at 400 km;
  - departure excess 12.13-13.22 km/s, a 5.6-6.5 km/s burn;
  - 2 Raptors for 7 minutes, with a 36-47 m/s steered loss;
  - 601-647 t of methalox;
  - 113-148 t sent on to Jupiter;
  - growth 0.96-1.23.
- **The 2S windows fly but shrink.** Methalox can fly every window. On the flown chain the
  2S windows need 14.3-14.6 km/s (7.4-7.6 km/s burns), send on 76-88 t and grow 0.58-0.83.
- **The optimum is at the Kapton table's floor.** `k_push` = 5.0 is the overtake law's lowest
  simulated slug ratio (`K_PUSH`). Growth still rises toward it: one 3S window gives 1.22 at
  k = 5, 1.17 at 6.5 and 0.95 at 10. The table cannot say what a heavier sheet would give.
- **The lob is the price:** $1098 of the $1344, because 1500 t is lofted per ~125 t sent on and
  the loop barely reinvests.

### Slower departures (item 2)

`slow_departures` takes the first 3S window's departure epoch. It returns 3, 4 or 5 synodic
periods later, less the parking gap, and keeps the best member of `return_family` per 0.5 km/s
band (corrections up to 2 km/s, each flown every cycle and pushed by its own return).

- **Nothing slower closes.** The slowest member with a correction of 2 km/s or less departs at
  11.5 km/s on 4S. On 3S and 5S the floor is 12.0 km/s.
- **The slowest at any correction** departs at 10.0 km/s on 3S with a 45 km/s correction, and
  at 9.1 km/s on 4S and 5S with 55 and 10.5 km/s.
- **The best members are the near-zero-correction ones:** 12.17 km/s on 3S, giving 1.29
  one-batch; 11.70 on 4S, giving 1.14; 12.82 on 5S, giving 1.15.
- **The return stream stays fast.** Return excess is 49-58 km/s, a 50-60 km/s stream at
  400 km, on every clock. These are the chain's retrograde Jupiter returns, so the slow
  departure does not return a slow ~14 km/s stream.
- **The parent's conclusion holds and its reason is corrected.** A slower departure does not
  pay because it needs corrections far larger than the methalox it saves. Its return stream
  is not slow.

### The one-unit seed (item 4, `src/small_seed.py`)

The seed at the adopted headline, as `value_design` forms it (one launch unit's departing ships
in the steady role mix):

| Closure | Design | Seed | Ships 40 / 60 t | Launch $B | $/kg sent |
|---|---|---|---|---|---|
| 9.86 R_sun, X 60 | ADR 0085 (matched, centred) | 567.5 t | 3.68 / 4.23 | 0.74 / 2.83 | 1297 / 4993 |
| 9.86 R_sun, X 60 | **x1.2, 1.36 cm, held** | **567.5 t** | 3.68 / 4.23 | 0.74 / 2.83 | 1297 / 4993 |
| 4 R_sun, X 90 | ADR 0085 | 673.7 t | 4.79 / 5.59 | 0.96 / 3.75 | 1423 / 5559 |
| 4 R_sun, X 90 | **x1.2, 1.36 cm, held** | **673.9 t** | 4.80 / 5.59 | 0.96 / 3.75 | 1423 / 5559 |

- **The parent's scratch is right.** The track hold is burned at the Sun, after the departure,
  so it does not change the departing ship.
- **The printed 622 t / 725 t comes from no aim commit.** `0aa5813` (ADR 0083, unfloored)
  gives 565.2 t and `7ce7628` (ADR 0085) gives 567.5 t. No ledger line is near 622 t.
- **The printed $/kg-sent column matches aim's.** Only the tonnes, ships and launch dollars
  are off, all by the same ~1.10x.

### The smaller seed (item 3, S29)

- **Seeds below a whole unit are now valued.** `value_design(seed_units=...)` flies a seed below
  one unit as one smaller unit (`solar_dive_growth.sub_unit`, `Design.unit_mass`). So does every
  generation until its sheets fill a whole one.
  - Its push, plates and disks are sized from it.
  - Its Sun arc is shortened onto the 1 kg disk floor (`floor_arc`).
  - Attrition learns from sheets caught, so a small seed keeps the first miss rate longer.
- **What is indivisible is the Sun plate.** The floor holds the pulse at 1 kg, so the
  paraboloid and its collar keep their full size on a smaller ship.
  - Its share of the departure ship doubles at half size: 2.8% → 5.6% at 9.86 R_sun, 4.3% →
    8.5% at 4 R_sun.
  - A 1-ship unit flies a 0.25 h arc against 0.93 h.
  - Growth falls to 2.56x (9.86 R_sun) and 2.64x (4 R_sun), against 3.24x and 4.84x.
  - Nothing else moves by more than 2% except the collar and, at a quarter unit, 0.5% on the
    bowl.

| Closure | End | Seed | Units | Tonnes | Up-front $B | Break-even | Fleet x | Seed $/fleet kg | Years to match |
|---|---|---|---|---|---|---|---|---|---|
| 9.86 R_sun | cheap | one unit | 1 | 568 | 0.77 | 130 | 70.6 | 36 | — |
| | | 1 ship | 0.273 | 154 | 0.21 | 142 | 11.2 | 64 | +5.1 |
| | | 2 ships | 0.544 | 308 | 0.42 | 133 | 32.7 | 43 | +2.1 |
| | dear | one unit | 1 | 568 | 2.87 | 156 | 70.6 | 136 | — |
| | | 1 ship | 0.238 | 134 | 0.68 | 198 | 8.7 | 266 | +5.8 |
| | | 2 ships | 0.474 | 268 | 1.36 | 167 | 27.1 | 169 | +2.7 |
| 4 R_sun | cheap | one unit | 1 | 674 | 1.00 | 86 | 376.4 | 8 | — |
| | | 1 ship | 0.210 | 141 | 0.21 | 91 | 28.0 | 23 | +5.4 |
| | | 2 ships | 0.418 | 281 | 0.42 | 87 | 117.2 | 10 | +2.4 |
| | dear | one unit | 1 | 674 | 3.79 | 89 | 376.4 | 29 | — |
| | | 1 ship | 0.180 | 121 | 0.68 | 110 | 15.6 | 131 | +6.6 |
| | | 2 ships | 0.359 | 241 | 1.36 | 92 | 92.0 | 43 | +2.9 |

How to read the table:

- Each end uses its own ship: 154 t a Starship cheap, 134 t dear, at 9.86 R_sun.
- The break-even keeps the 15-synodic growth phase, so a smaller seed harvests a smaller fleet.
- "Years to match" is `ln(fleet ratio) / ln(steady growth)` cycles of 3.276 yr, the extra growth
  before a smaller seed's fleet reaches the one-unit seed's harvest. In whole 3S cycles it is
  1 for two ships and 2-3 for one.
- The headline seed itself is unchanged: $129.5 / $156.1 and $85.6 / $89.1.

Results:

- **Two ships cost $3 cheap and $11 dear at Parker's depth, and $1-3 at 4 R_sun.**
- **One ship costs $12 / $42 and $5 / $21.** Most of that is the lighter first generations
  carrying a full Sun plate.

## Consequences

- `solar_dive_growth` gains:
  - `PushLeg` and `push_and_turn`;
  - `Design.unit_mass` (the `lofted` line, and the lob charged on it);
  - `sub_unit`;
  - `value_design(seed_units=...)`;
  - `Valuation.seed_mass` and `seed_outlay`;
  - the cost book's `departure_stage` line, zero on every bowl ledger.
- `growth_ledger.best_methalox_departure`; `chamber_departure.role_growth`'s no-rods case.
- `plate_jupiter_loop`: `Fly`, `bowl_pair`, and `fly` / `knobs` / `gates` on `window_growth`,
  `chain_rate`, `value_chain` and `optimize_loop`. ADR 0075's figures are unchanged.
- New modules `src/methalox_jupiter_loop.py` and `src/small_seed.py`, with `make methalox-loop`
  and `make small-seed`.

## Assumptions to revisit

- **Milestone 1's growth is pinned at k = 5, the floor of the Kapton table.** A run below it
  would say whether heavier sheets push better.
- **The departure's gate is kept whole at 90%,** although the bowl is not retired here. Splitting
  it raises milestone 1's value only a little, since the break-even is set by the lob.
- **The seed flies the arriving cycle's nozzle excess (11.99 km/s)** as ADR 0075's does, though
  its ships then fly the growth trajectory.
- **The slow-departure sweep is a screening measure.** It keeps the window's directions and
  flies each member every cycle.
- **A sub-unit generation is one smaller unit,** not a full unit lofted part-empty. Its booster
  is charged per kilogram lofted, as every lob is.
