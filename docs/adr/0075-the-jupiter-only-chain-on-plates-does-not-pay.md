# ADR 0075: The Jupiter-only chain on plates does not pay against the Sun dive

Date: 2026-10-19

Status: Accepted. Answers the parent's S41 (`docs/plate_jupiter_loop_asks.md` in
`Balloon-Pulse-Propulsion` @ `988a9e3`). Its S42 is for the impact simulation. Parent note:
`docs/plate_jupiter_loop_answers_for_parent.md`.

## Context

The author asked (2026-10-19) whether the Jupiter-only chain, flown on ADR 0073's plates
with no nozzle and no Sun dive, gives an attractive growth cycle. The plates are light,
they are not temperature-limited, and the cycle needs no $1000/kg disks. The parent's
ballpark was 2.49x per 3S cycle, 2.48 yr and $292 dear at the best exact-3S phase, against
the Sun dive's 3.81x, 1.70 yr and $134. Its 2S lock did not grow.

## Decision

`src/plate_jupiter_loop.py` (`make plate-jupiter-loop`) flies the child's chain on the
plates:

1. **The chain is the child's.** Real ephemerides on the adaptive 2S/3S cadence
   (`adaptive_two_wave_cycles`, 10-day split, 11 cycles), and the 3S-only chain (9 cycles).
   Each window's excess vectors are re-solved from its selected Lambert legs
   (`departure_visibility._excess_vectors`). Growth carriers depart on the next window's
   growth trajectory and rod carriers on its nozzle trajectory. Each pays its own wave's
   return burn in methalox.
2. **The Earth half is the solar dive's**, shared rather than copied: `solar_dive_growth.
   fly_earth`, split out of `fly_unit`. It covers the push on `push_ve` at 2 Hz and 1 g from
   the 2.43 km/s lob, the push plate's disposal, the apoapsis turn from the growth wave's
   direction onto the thrust axis (0.374-0.386 km/s, nearly the reversal), and the departure
   on `hole_ve` at 1 Hz on the **visible sign** (`visible_sign`, generalized from
   `visible_departure`). The bowl meets the nozzle wave at 160-180 deg and 66-76 km/s. The
   departure's finite-burn loss is the chain's stored **visibility-constrained** loss
   (ADR 0066), the same table the chamber ledger flies.
3. **Home:** the departing stack less 1% avionics returns as sheets. There is no Sun stage,
   disk, carrier, shade or Sun plate.
4. **Valuation:** the solar dive's gated rNPV with the Sun gate removed (95 / 85 / 90). The
   seed flies the chain's first cycle at 11.99 km/s, as the child's does (ADR 0060). Each
   window is its own ledger, and the chain repeats past its stored ~30 years. The 2 Hz
   delivery and its 25 g plate are ADR 0074's.
5. **The slug ratios are optimized for the dear break-even** (push, departure, delivery).

## Results (`make plate-jupiter-loop ARGS="--family"`, ~10 min)

| chain, departure | k push / dep / delivery | growth per 3S | doubling | steady | break-even cheap / **dear** |
|---|---|---|---|---|---|
| flown, 0.25 g | 6.77 / 12.50 / 6.61 | 2.01 | 3.26 yr | $263 | $315 / **$573** |
| flown, 0.5 g | 6.61 / 12.58 / 6.13 | 2.45 | 2.54 yr | $167 | $197 / **$279** |
| flown, 1 g | 6.51 / 12.66 / 5.96 | 2.59 | 2.39 yr | $153 | $179 / **$238** |
| 3S only, 0.25 g | 6.64 / 12.25 / 6.60 | 1.74 | 4.11 yr | $172 | $430 / $1026 |
| 3S only, 0.5 g | 6.59 / 12.36 / 6.38 | 2.00 | 3.28 yr | $141 | $278 / $560 |
| 3S only, 1 g | 6.54 / 12.44 / 6.32 | 2.07 | 3.12 yr | $134 | $257 / $499 |
| **Sun dive, 9.86 R_sun, X 60** (ADR 0074) | 7.06 / 12.54 / 5.57 | **3.80** | **1.70 yr** | $87 | $107 / **$118** |

"Growth per 3S" is the chain's continuous rate projected over three synodic periods. Each
window's own growth is 1.2-1.8x for a 2S window and 1.9-2.6x for a 3S window.

- **The Sun dive wins at every rate**, by 2x or more on the dear break-even. The plate chain's
  best (the flown chain at 1 g) is $238 against $118.
- **The departure rate matters most.** At 0.25 g a 6.3-9.4 km/s burn lasts 43-62 minutes and
  loses **0.8-2.0 km/s** with the stream kept in view. At 1 g the losses shrink, and growth
  rises from 2.01 to 2.59. The bowl's heavier pulse costs much less than the loss saves.
  The 2S burns stay inside the visibility table (7200 s) at every rate.
- **The flown chain beats 3S-only on plates.** Its 2S windows grow less per cycle but cycle
  faster, which also pays its seed sooner.
- **Why the ballpark was kinder** (2.49x at 3S): (a) the ballpark's fixed-direction loss.
  With it the 3S-only chain at 0.25 g rises from 1.74 to 1.85 and $1026 to $802. Keeping the
  stream in view costs 0.2-0.6 km/s. (b) Real 3S windows return at 52-56 km/s excess, not the
  best phase's 61.3, and depart at 11.9-13.2 km/s.

**Slug ratios (item 2).** The optimum is k 6.5-6.8 on the push, 12.3-12.7 on the departure
and 6.0-6.6 on the delivery. "Relatively high" k (10 / 20 / 10) barely grows: 1.06x on the
flown chain and $29 660 dear. At 55-75 km/s, a higher k lowers v_e faster than it saves
impactor, as the ballpark said.

**The seed (item 4).** The seed leaves at 11.99 km/s ($2172 / $9293 per kg on the stripped
ships). ADR 0059's two-year Earth loop (x0.6687 seed price, +2.18 yr) cuts the flown 0.25 g
break-even $315 / $573 to $300 / $513: 5% cheap, 10% dear. ADR 0060's verdict stands. The
seed flies direct, and the loop is a sensitivity. The seed is the child's seed: same chain,
same excess.

**A faster return (item 5).** At each window's fixed epochs (`return_family`), every
sampled encounter and path pair needing at most a 0.5 km/s correction was scored on the
plate ledger as one batch. The chain's least-correction return is also the best on the plate
ledger in both a 3S and a 2S window. The fastest member returns only 0.1 km/s faster (54.42
against 54.31; 60.50 against 60.37), pays 0.28-0.42 km/s of correction, and grows less (1.84
against 1.95; 1.51 against 1.64). The Jupiter-only geometry pins the return speed at fixed
epochs. A 69 km/s return at a 12.1 km/s departure is not on this chain.

## Consequences

- The paper's `tab:no_sun_loop` should be replaced by these figures. They are worse than the
  ballpark, and the conclusion that the Sun dive is needed is stronger: at best $238 dear and
  2.39 yr against $118 and 1.70 yr.
- If the plate loop is ever flown, fly it fast: 1 g on the bowl, if the bowl and the
  impactors' aim allow it.

## Limits

- Planar departure geometry: the excess vectors are projected onto the ecliptic. The stored
  loss tables assume the chamber's visible mirror, and the bowl's sign is chosen by
  occultation at periapsis. Both keep the stream in view.
- The bowl at 160-180 deg is interpolated between the 129.9 and 180 deg capture runs, and the
  push at 54-65 km/s uses the 100 km/s law scaled by `w / 100`. The parent's S42 asks the
  simulation to confirm both.
- The return family varies only the encounter and paths at the chain's fixed epochs. It does
  not move the departure date or the cadence.
- Each window's ledger applies that window's onward return burns. The child's
  `growth_ledger` applies the arriving cycle's.

## Reproduction

`make plate-jupiter-loop` (headline, per-window table, seed), `ARGS="--family"` for item 5.
Tests: `tests/test_plate_jupiter_loop.py`.
