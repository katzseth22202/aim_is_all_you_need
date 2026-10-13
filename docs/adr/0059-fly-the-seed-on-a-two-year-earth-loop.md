# ADR 0059: Fly the seed on a two-year Earth loop with a deep-space burn

Date: 2026-10-13

Status: Accepted. Supersedes ADR 0057's headline route (EEJ) and amends ADR 0039.

## Context

The author asked (2026-10-13) whether a Lucy-style trajectory makes the seed
cheaper by enough to pay for its delay at 10% a year (ADR 0040). In this
trajectory, also called a delta-v Earth gravity assist (ΔV-EGA):
- a modest Oberth burn leaves Earth slowly;
- a burn in deep space, usually retrograde near aphelion, pumps the orbit's
  eccentricity;
- Earth is met again at a much higher excess speed;
- a **powered** Earth flyby then boosts and bends the ship onto the arc to
  Jupiter.

The burns in deep space and at the flyby are methalox at 370 s, kept cold under
blankets and cryocoolers. The question is which loop (one, two or three years)
supplies the delta-v, with the flyby leaving the ship pointed on the arc to
Jupiter.

Answering it exposed two flaws in ADR 0039's EEJ, which ADR 0057 had just made the
headline:

1. **The return floor was too low.** ADR 0039 held every route's return to
   51.134 km/s (ADR 0007's powered-flyby optimum). The cost model credits the seed
   with the chain's first, three-synodic cycle, whose growth wave returns at
   **59.29 km/s** (nozzle wave 55.45). At 51.134 even "direct" left at only
   10.86 km/s, against the chain's 11.99.
2. **Hull staging was scored as route gain.** ADR 0039 flies the node burns on a
   separate light stage (8% tankage) and drops the 40-60 t hull after departure.
   That gain is staging, not trajectory, and the direct route could stage the same
   way. Under it, a one-year "loop" that simply departs in two burns wins at no
   delay.

## Decision

1. **The seed flies a two-year loop** (`growth_cost_inputs.LOOP_2YR`,
   `HEADLINE_SEED`). Per kilogram it is ×0.6687 on the 40 t ship and ×0.5972 on
   the 60 t ship, and it returns 2.180 years after direct. The flight (40 t ship)
   is:
   - launch 0.25 yr into the window at v∞ 5.22 km/s;
   - a 0.56 km/s burn 45% of the way round a 1.87-year loop;
   - back at Earth at 8.99 km/s;
   - a 2.71 km/s powered flyby at 300 km that turns 40.6° and leaves at 12.8 km/s;
   - 1.16 years to Jupiter, and a return at 60.2 km/s;
   - 128.65 t sent, against direct's 86.03 t.
2. **Seed routes are searched against the chain's floor**, `seed_loop.CHAIN_FLOOR`
   = 59.29 km/s, and `seed_route.best_route(..., floor=)` takes it too.
3. **The ship flies every burn itself, its dry mass carried**
   (`seed_loop.seed_carried`):
   `seed = P / (R_departure × R_burns − 1) − dry`. ADR 0039's staged accounting
   stays only as `--staged`, to show the artifact.
4. **The two-year loop, not the three-year one.** On the 40 t ship the three-year
   loop is marginally better (1.26 against 1.22); on the 60 t ship the two-year loop
   is better (1.36 against 1.29). The two-year loop has half the delay, so less
   rests on `delayed()`'s time shift (see Limits).

## The search (`src/seed_loop.py`, `make seed-loop`)

The model is the same as ADR 0039's:
- coplanar circular planets at their real longitudes from 2026-11-09;
- a launch window of 6 years;
- Jupiter's unpowered bend solved so the return lands on Earth.

The new leg works like this:
- **Launch.** v∞ `v0` (1-12 km/s) at angle `alpha` in the ecliptic.
- **Coast.** Kepler for `eta` (0.05-0.95) of the loop.
- **Deep-space burn.** Its magnitude is the gap between the coasting velocity and
  the start of the Lambert arc back to Earth (prograde, 0 or 1 revolution, both
  paths; the cheapest is kept).
- **Earth flyby.** Its excess speed is set by the Lambert arc to Jupiter's real
  position after `cruise` (0.8-5 yr). The flyby burn is the split-hyperbola node
  (`_powered_node_burn`), with periapsis at least 300 km.
- **Loop bands (yr):** 1 = (0.9, 1.6), 2 = (1.6, 2.6), 3 = (2.6, 3.6).
- **Score.** `ln(seed) − t_return ln 1.1`, plus 10 per km/s below the floor.
- **Optimiser.** pygmo SADE: 4 islands per bend side, 40 individuals,
  150 generations, 3 evolutions. Seed 7; island `i` on side `s` is seeded
  `7 + 2i + s`.
- **Re-pricing.** Champions are re-priced with `seed_payload`'s finite-burn
  departure loss for a full 1200 t burn, which slightly overstates it when
  propellant is kept back.

The direct comparator is `best_route("EJ", ..., rate=0.10, floor=59.29)` with the
same settings. It launches at the window's opening at 12.20 km/s and returns at
3.21 yr. The search is deterministic: a re-run of the two-year band reproduced
both champions exactly. The 40 t champion is
`x = (0.24774, 5.21552, -0.13610, 0.44741, 1.87364, 1.15915)`; the 60 t one is
`(0.27042, 5.21308, 0.14792, 0.44429, 1.85671, 1.16990)`
(launch yr, v0 km/s, alpha rad, eta, loop yr, cruise yr).

## Results

**Ship carried, floor 59.29 km/s (the decision).** k is seed over direct's;
worth is `k × 1.1^-dt`.

| Ship | Loop | v0 | DSM | Flyby burn | Earth v∞ in → out | Turn | Return (yr) | Seed (t) | k | **Worth** |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 t | direct | 12.20 | — | — | — | — | 3.21 | 86.0 | 1 | 1 |
| 40 t | 1 yr | 1.59 | 0.05 | 5.16 | 1.7 → 12.0 | 90° | 4.30 | 87.8 | 1.02 | **0.92** |
| 40 t | **2 yr** | 5.22 | 0.56 | 2.71 | 9.0 → 12.8 | 41° | 5.38 | 128.7 | 1.50 | **1.22** |
| 40 t | 3 yr | 7.15 | 0.74 | 0.99 | 13.0 → 14.2 | 28° | 7.58 | 165 | 1.92 | **1.26** |
| 60 t | direct | 12.20 | — | — | — | — | 3.21 | 66.0 | 1 | 1 |
| 60 t | 1 yr | 1.56 | 0.11 | 5.22 | 1.6 → 12.0 | 90° | 4.30 | 63.3 | 0.96 | **0.87** |
| 60 t | **2 yr** | 5.21 | 0.51 | 2.73 | 8.6 → 12.5 | 42° | 5.39 | 110.6 | 1.67 | **1.36** |
| 60 t | 3 yr | 7.56 | 0.47 | 1.27 | 11.5 → 13.2 | 32° | 7.58 | 130 | 1.96 | **1.29** |

**The artifacts.** These are the worths on the same search, scored ADR 0039's
ways:

| Accounting | Floor | 40 t: 1 / 2 / 3 yr | 60 t: 1 / 2 / 3 yr |
|---|---|---|---|
| Ship carried (decision) | 59.29 | 0.92 / 1.22 / 1.26 | 0.87 / 1.36 / 1.29 |
| Hull dropped (ADR 0039's) | 59.29 | 1.39 / 1.69 / 1.59 | 1.78 / 2.06 / 1.92 |
| Hull dropped | 51.134 | 1.43 / 1.61 / 1.62 | 1.55 / 1.96 / 1.82 |

Dropping the hull is worth 0.4-0.9 in the ratio, and it is what made a one-year
"loop" look free.

**What it does to the price.** Break-even at L1 with the seed on the two-year
loop. Spray cup, lob climbing at 1.1 km/s, 10% a year, 50% odds; entries are
cheap / dear seed:

Seed $1452 / $5550 per kilogram on the loop, against $2172 / $9293 direct.

| Design | Free seed | Direct | **2-yr loop (headline)** | Loop, Pessimistic book | Loop, 25% odds | EEJ (ADR 0057) |
|---|---|---|---|---|---|---|
| Methalox | 631 | 1566 / >3000 | 1402 / >3000 | 2385 / >3000 | 2408 / >3000 | 1437 / >3000 |
| CH4 AR100, 3.5 kg pitch | 289 | 463 / 1027 | **432 / 833** | 1273 / 1674 | 632 / 1430 | 439 / 845 |
| CH4 AR100, 24 kg | 304 | 492 / 1105 | **460 / 894** | 1338 / 1773 | 676 / 1542 | 467 / 908 |
| CH4 AR150, 24 kg | 289 | 463 / 1030 | 433 / 835 | 1269 / 1672 | 633 / 1434 | 439 / 848 |
| CH4 AR300, 24 kg | 288 | 472 / 1071 | 440 / 864 | 1273 / 1698 | 648 / 1495 | 447 / 878 |
| CH4 AR150, 24 kg, 3 Hz | 247 | 382 / 818 | **358 / 668** | 1078 / 1388 | 511 / 1127 | 363 / 677 |
| Hydrogen, solved | 185 | 231 / 377 | 223 / 327 | 733 / 837 | 281 / 486 | 225 / 330 |
| Hydrogen, bulged, 2 Hz | 329 | 469 / 925 | 445 / 768 | 1292 / 1616 | 625 / 1270 | 450 / 778 |

The `DESIGNS` headline (Estimate, ADR 0033's plate, loop seed) is: methalox
$522 / $1110, solved methane $151 / $199, solved hydrogen $144 / $178.

The loop lands within $5-13 of EEJ's break-evens. Its ratios happen to resemble
EEJ's (×0.67 / ×0.60 over 2.18 yr, against ×0.70 / ×0.61 over 2.18). The
difference is that these ratios are measured on the floor and the hull the cost
model assumes.

## Consequences

- **Yes, a Lucy-style loop justifies its delay,** by 1.2-1.36× at 10% a year.
  It does not repeat ADR 0039's 1.33x on the dear ship and 1.16x on the cheap
  one as stated: those figures rested on the low floor and on staging.
- **The 370 s deep-space burn is cheap leverage.** About 0.5 km/s, half-way round
  a two-year loop, returns the ship to Earth at 9 km/s, from where a 2.7 km/s
  perigee burn reaches 12.5-12.8 km/s. **The seed's real cost is still the
  departure:** 5.2 km/s of excess from 200 km is a 4.5 km/s burn on a full
  1200 t ship. That is the author's next study: a cheaper apogee raise near
  Earth, combined with this loop.
- **Staging is a separate lever and is not taken here.** A light kick stage
  would help direct and the loop alike. Pricing it needs a real stage design
  (engines, tanks and cryocoolers for a two-year coast), not an 8% fraction.

## Limits

- **The time shift.** `delayed()` slides the whole chain by 2.18 years as if the
  chain could start on any date, whereas its geometry repeats each synodic period
  (1.09 yr). This is ADR 0039's simplification too. 2.18 years is two synodic
  periods to within 0.2%, which is one more reason to prefer the two-year loop
  over the three-year one (4.4 yr, 4.0 periods).
- **One stream, not two waves.** The seed's return is one stream held to the
  growth wave's speed. The 20-day split into growth and nozzle waves, and its
  correction, are not flown.
- **Storage.** The deep-space and flyby burns are charged 370 s and nothing else.
  Cryocoolers and blankets for 1.9 years in space carry no mass, and there is no
  boil-off.
- **Model.** Coplanar circular orbits, an impulsive deep-space burn, and a
  departure loss taken from a full-propellant burn.

## Reproduction

`make seed-loop` (about 30 min, 2 workers on 4 GB). Options: `--staged` and
`--floor 51.134` for the comparison rows, `--bands "2 yr"` for the headline only.
Fast tests: `tests/test_seed_loop.py`. The slow test pins the floor to the chain.
The break-even table comes from the same script as ADR 0057's, with
`HEADLINE_SEED = LOOP_2YR`.
