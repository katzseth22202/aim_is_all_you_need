# Milestone 1 on methalox, slower departures, and the seed: aim's answers to S63 (for the parent)

Raised 2026-10-29 in `aim_is_all_you_need` (ADR 0088). **Written to be copied into
`Balloon-Pulse-Propulsion`.** It answers S63 (`docs/milestone_one_asks_for_aim_repo.md` @
`e30114b`) and the S29 that S63 re-raises. Reproduce with `make methalox-loop` (~1 min) and
`make small-seed` (~1 min).

## The short answers

1. **Milestone 1 barely grows on today's plates.** It grows 1.08x per 3.276 yr (30.6 yr
   doubling) on the three-synodic windows. Steady cost is $1344/kg at L1 and break-even is
   $10,559 cheap / $34,376 dear. On the flown 2S/3S chain it shrinks (0.82x). The parent's "it
   grows" holds only in sign, and barely. The child's shrink estimate is nearer the mark than
   the spray-cup rows suggested.
2. **No slower departure closes.** Within 2 km/s of correction, nothing departs below 11.5 km/s
   on a 3S, 4S or 5S clock. The return stream stays at 50-60 km/s, not ~14 km/s.
3. **Below one launch unit the Sun plate is indivisible.** A 2-ship seed costs $3 / $11 at
   Parker's depth and $1 / $3 at 4 R_sun. A 1-ship seed costs $12 / $42 and $5 / $21, and 5-7
   more years of growth.
4. **The headline seed is 567.5 t at Parker's depth and 673.9 t at 4 R_sun.** Your scratch is
   right. The printed 622 t / 725 t is not reproducible from aim.

## 1. Milestone 1's ledger

`src/methalox_jupiter_loop.py`. The setup:

- **Push and parking.** ADR 0075's push at 400 km on the flat push plate (2 Hz, 1 g, gas spring),
  Kapton on the overtake law, the 10-day orbit, and the apoapsis raise and turn.
- **Departure.** `growth_ledger`'s methalox departure in place of the bowl: Raptor 3 at 380 s,
  0.018 tank fraction, engines optimized and expended, steered loss from 600 km.
- **The push plate is left at apoapsis** (68.7 t with bags), since there is no bowl to share it
  with.
- **No role split.** Every ship flies the chain's growth trajectory and comes home as the next
  push wave.
- **Valuation.** Gated rNPV at 10%/yr, gates 95/85/90 with no Sun gate, 15 synodic periods of
  growth, dear-seed optimum.

| Chain | Growth per 3.276 yr | Doubling | Steady $/kg at L1 | Break-even cheap / dear |
|---|---|---|---|---|
| Three-synodic only (9 windows) | **1.08** | 30.6 yr | **$1344** | **$10,559 / $34,376** |
| Flown 2S/3S (11 windows) | 0.82 | does not grow | $506 | $94,803 / $336,109 |

**Push stream speed band** (three-synodic windows, at 400 km): **53.9-57.6 km/s**. On the flown
chain the 2S windows push at 56.5-65.1 km/s.

**Per unit, on the three-synodic windows:**

- Raptors: 2, burning about 7 minutes.
- Burn: 5.6-6.5 km/s at 12.1-13.2 km/s excess, with a 36-47 m/s steered loss.
- Methalox: 601-647 t.
- Sent on to Jupiter: 113-148 t.
- Growth: 0.96-1.23 by window.
- The apoapsis turn costs 0.35-0.38 km/s, as on the bowl.

**The windows methalox can fly.** It can fly every window, including the two-synodic ones.
There it needs 14.3-14.6 km/s of excess (7.4-7.6 km/s burns), sends on only 76-88 t and grows
0.58-0.83, so methalox should fly three-synodic only.

**The seed:** 119 t at 11.99 km/s, which is 1.29 / 1.65 stripped Starships (40 t / 60 t).

**Why it is so much weaker than the spray-cup rows (1.27x on the 10-day orbit):**

- The push plate is now dropped every cycle.
- The three-synodic stream is the slow end of the 54-65 km/s band.
- The optimum sits at **k = 5, the lowest slug ratio the Kapton table holds**. One window gives
  1.22 at k = 5, 1.17 at 6.5 and 0.95 at 10, so heavier sheets might push better. Only a
  simulation run below k = 5 can say.

The steady price is mostly the lob ($1098 of $1344): 1500 t is lofted per ~125 t sent on.

Suggested text for the child:

> On the sheets and spring plate, the Jupiter-only loop with a methalox departure only just
> grows: about 1.08 times per 3.28 years on the three-synodic windows, a 31-year doubling, and it
> breaks even at over \$10,000 per kilogram at L1. With the departure bowl it grows 1.65 times.

## 2. Slower departures

`slow_departures` starts from the first three-synodic window's departure and returns 3, 4 or 5
synodic periods later. On each of those clocks it keeps the best member of the window's return
family per 0.5 km/s of departure excess. "One-batch growth" assumes every cycle flies that
member, pushed by its own return.

| Clock | Departure excess | Return excess | Stream at 400 km | Correction | One-batch growth |
|---|---|---|---|---|---|
| 3S | 11.97 | 56.66 | 57.7 | 0.88 | 1.08 |
| 3S | **12.17** | 56.33 | 57.4 | 0.009 | **1.29** |
| 3S | 12.57 | 55.80 | 56.8 | 1.55 | 0.76 |
| 4S | 11.48 | 49.57 | 50.7 | 1.12 | 0.91 |
| 4S | 11.70 | 49.09 | 50.3 | 0.010 | 1.14 |
| 4S | 12.09 | 49.04 | 50.2 | 1.41 | 0.71 |
| 5S | 12.49 | 58.01 | 59.0 | 0.86 | 0.99 |
| 5S | 12.82 | 58.34 | 59.3 | 0.031 | 1.15 |
| 5S | 13.02 | 58.50 | 59.5 | 0.51 | 0.97 |

**Confirmed, with one correction to the reasoning.**

- **Nothing slower closes.** The slowest departure with a correction of 2 km/s or less is
  11.5 km/s, on the four-synodic clock.
- **Slower ones cost far too much to correct.** Below that, the cheapest needs 10.5 km/s of
  correction (9.1 km/s departure, 5S) and 45 km/s (10.0 km/s, 3S).
- **The best member on every clock is the near-zero-correction one** near 12 km/s.
- **The return is not slow.** A longer clock does not return a slow stream: every member
  returns at 49-58 km/s of excess, because these are the chain's retrograde Jupiter returns. A
  Hohmann-like 8.8 km/s departure is not in the family at all.

So "a slow departure does not pay" stands. The reason is the correction it needs, not a weak
next push.

## 3. The smaller seed (S29)

**Is anything indivisible below one 1500 t launch unit?**

- **Yes: the Sun plate.**
  - A smaller seed now flies as one smaller unit (`Design.unit_mass`): its push, plates and disks
    sized from it, every Sun pulse held at the 1 kg disk floor.
  - So the smaller ship flies fewer, heavier pulses on a shorter arc. A 1-ship unit flies
    0.25 h against 0.93 h.
  - It keeps the full-size paraboloid (19 t mean) and collar.
  - At half size the Sun plate's share of the departure ship doubles (2.8% → 5.6% at 9.86 R_sun).
  - A 1-ship generation grows 2.56x (9.86 R_sun) or 2.64x (4 R_sun), against 3.24x and 4.84x.
- **Nothing else is indivisible.** No other line moves by more than 2% of its share, except the
  1 t collar and 0.5% on the bowl at a quarter unit. The push plate, bowl, carrier and charges
  scale. Each generation stays one smaller unit until its sheets fill a whole one.

**Both headlines on 1- and 2-ship seeds.** These are on the adopted headline (x1.2 charge,
1.36 cm, track hold). The 15-synodic growth phase is kept, and each end uses its own ship
(cheap 40 t, dear 60 t).

| Closure | End | Seed | Tonnes | Up-front $B | Break-even | Seed $/fleet kg | Years to match |
|---|---|---|---|---|---|---|---|
| 9.86 R_sun | cheap | one unit | 568 | 0.77 | $130 | 36 | — |
| | | 2 ships | 308 | 0.42 | $133 | 43 | +2.1 |
| | | 1 ship | 154 | 0.21 | $142 | 64 | +5.1 |
| | dear | one unit | 568 | 2.87 | $156 | 136 | — |
| | | 2 ships | 268 | 1.36 | $167 | 169 | +2.7 |
| | | 1 ship | 134 | 0.68 | $198 | 266 | +5.8 |
| 4 R_sun | cheap | one unit | 674 | 1.00 | $86 | 8 | — |
| | | 2 ships | 281 | 0.42 | $87 | 10 | +2.4 |
| | | 1 ship | 141 | 0.21 | $91 | 23 | +5.4 |
| | dear | one unit | 674 | 3.79 | $89 | 29 | — |
| | | 2 ships | 241 | 1.36 | $92 | 43 | +2.9 |
| | | 1 ship | 121 | 0.68 | $110 | 131 | +6.6 |

How to read the table:

- **Up-front bill** is the seed's launch plus its contents.
- **Years to match** is the extra growth, at the steady rate, before the smaller seed's fleet
  reaches the one-unit seed's harvest. In whole 3.276 yr cycles that is one cycle for two ships
  and two to three for one.
- **S29 item 3, the dear seed's $/fleet kg:** it is in the table, $136 / $29 against the cheap
  $36 / $8 for one unit.

Suggested sentence for both papers:

> A seed of two stripped Starships instead of four starts the loop for half the up-front bill.
> It raises the break-even by \$3 to \$11 at Parker's depth and by \$1 to \$3 at 4 solar radii,
> and takes one more cycle to reach the same fleet. What cannot shrink is the Sun plate, which
> the 1 kg disk floor holds at full size.

## 4. The headline seed's tonnage

| Closure | Seed | Push share | Ships 40 / 60 t | Launch $B cheap / dear | $/kg sent |
|---|---|---|---|---|---|
| 9.86 R_sun, X 60 | **567.5 t** | 0.743 | 3.68 / 4.23 | 0.74 / 2.83 | 1297 / 4993 |
| 4 R_sun, X 90 | **673.9 t** | 0.696 | 4.80 / 5.59 | 0.96 / 3.75 | 1423 / 5559 |

- **This is the seed at the adopted headline.** The track hold is burned at the Sun, after the
  ship departs, so it does not change the departing mass. ADR 0085's matched design gives the
  same 567.5 t and 673.7 t.
- **622 t / 725 t comes from no aim commit.**
  - `0aa5813` (unfloored) gives 565.2 t and `7ce7628` (the floor) gives 567.5 t.
  - No ledger line on either ship is near 622 t.
  - Your $/kg-sent column already matches aim's, so tonnes, ships and launch dollars are all off
    by the same factor of ~1.10.
- **Corrected `tab:seed_amortization` row:**
  - 9.86 R_sun: 568 t, 3.7 / 4.2 ships, $0.74 / 2.83 B; fleet 70.6x, $36 / $136 per fleet kg.
  - 4 R_sun: 674 t, 4.8 / 5.6 ships, $0.96 / 3.75 B; fleet 376x, $8 / $29.

"About four stripped Starships" holds at Parker's depth. At 4 R_sun it is about five.

## Caveats

- Milestone 1's departure gate is kept whole at 90%. The break-even is set by the lob, so
  splitting it would barely move it.
- Its seed flies the arriving cycle's 11.99 km/s, as ADR 0075's does.
- The slow-departure sweep is a screening measure: the window's directions are kept and each
  member is flown every cycle.
- A sub-unit generation is one smaller unit whose booster is charged per kilogram lofted, not a
  full booster lofting a part-empty unit. Charging a whole flight would make the small seed
  dearer.
