# ADR 0060: Fly the seed direct; quote the Earth loop as a sensitivity

Date: 2026-10-14

Status: Accepted. Reverses ADR 0059's headline route (its search and results
stand). ADR 0057's prices and valuation stand.

## Context

ADR 0059 made a two-year Lucy-style Earth loop the headline seed route. Its
seed is worth 1.22× (40 t ship) and 1.36× (60 t ship) the direct seed after
its 2.18 years of delay at 10% a year. The author (2026-10-14) asked whether
that is worth the delay and the complexity. Here is what the loop actually
moves, as steady break-even at L1 ($/kg; spray cup, lob climbing at 1.1 km/s,
Estimate book, 10% a year, 50% odds; cheap / dear seed):

| Design | Direct | 2-yr loop | Saving |
|---|---|---|---|
| Methalox | 1566 / >3000 | 1402 / >3000 | 10% / — |
| CH4 AR100, 3.5 kg pitch | 463 / 1027 | 432 / 833 | 7% / 19% |
| CH4 AR150, 24 kg, 3 Hz | 382 / 818 | 358 / 668 | 6% / 18% |
| Hydrogen, solved | 231 / 377 | 223 / 327 | 3% / 13% |

The seed is about a third of the methane chamber's cheap-seed break-even, so
a 1.22× better seed moves the price 6-7%. Against that, the loop asks for
things its 1.22× does not charge:

- methalox held cold for a 1.9-year coast (cryocoolers and blankets, no mass
  in the model);
- a deep-space burn and a 2.7 km/s powered Earth flyby that must fly on time,
  or the seed is lost;
- the proof of the whole cycle two years later. The 10% rate charges the money;
  it does not charge the program waiting two more years to learn whether the
  cycle works.

## Decision

1. **The seed flies direct** (`growth_cost_inputs.HEADLINE_SEED = DIRECT`): one
   burn from 200 km to 12.2 km/s, 86.0 t on the 40 t ship, returning at the
   chain's 59.29 km/s. Every report that prices the seed moves with it.
2. **The loops are the quoted sensitivity.** `growth_cost_report.SEED_ROUTES`
   now lists ADR 0059's two- and three-year loops on each ship (×0.6687 /
   ×0.5972 over 2.180 yr; about ×0.521 / ×0.510 over 4.37 yr, from ADR 0059's
   rounded `k`). It drops ADR 0039's EEJ and EVEEJ, whose return floor and
   hull staging flatter them. The one-line claim: *a Lucy-style loop would cut
   the break-even 3-10% on the cheap seed and 13-19% on the dear one, two years
   later.*
3. `LOOP_2YR` and `EEJ` stay in `growth_cost_inputs` as `SeedFlight`s, so any
   report can be re-run on them.

## Results

The headline returns to ADR 0057's "Direct" column:
- CH4 AR100 (3.5 kg pitch) $463 / $1027;
- 3 Hz AR150 $382 / $818;
- solved hydrogen $231 / $377;
- methalox $1566 / >$3000.

The `DESIGNS` headline (Estimate book, ADR 0033's plate, direct seed) is
methalox $567 / $1396, solved methane
$155 / $222, solved hydrogen $146 / $194 (on the loop they were $522 / $1110,
$151 / $199 and $144 / $178).

## Consequences

- **The seed's real cost is still the departure:** 8.5 km/s from 200 km on a
  full 1200 t ship. Neither trajectory (ADR 0039, 0059) nor an argon tug
  (ADR 0058, 0061) changes that by enough to justify its complexity. The
  author's next question is the growth multiple, which sets how much a seed
  kilogram is worth in the first place.
- If the dear seed (60 t ship, Morgan Stanley's $500/kg) turns out to be the
  real price, the loop's 13-19% is worth revisiting.

## Reproduction

`make growth-cost` (section 9 prints the loops against direct). Slow tests:
`tests/test_growth_cost_inputs.py` pins the direct headline.
