# Charge the risk once: expected value at 10%, 50% odds the cycle works

Status: accepted

Amends: ADR 0037 (the stepped cost of capital). Its growth charge, prices and
cash flows stand; the discounting changes.

Date: 2026-10-03

## Context

ADR 0037 discounts at 30% a year (Gompers et al.'s venture target) until the
first growth cycle returns, then at 10%. A venture hurdle rate bundles three
things: the time value of money, the risk that the idea fails, and a haircut for
success-case projections. Compounding it every year treats each year of waiting
as carrying the venture's whole failure risk again.

The seed-route analysis (ADR 0039) showed why that matters. An Earth gravity
assist sends 1.64 times the seed of the direct route at no extra cost and comes
back 2.19 years later. At 30% that delay costs a factor of 1.78 and the route
loses; at 10% it costs 1.23 and the route wins. But waiting does not make the
physics likelier to fail: whether the chamber works is settled by the first
growth cycle, whichever route the seed flew. The rate, not the trajectory, was
deciding the answer.

## Decision

**Value the program as an expected value** (author, 2026-10-03):

1. **Time at 10% a year** throughout, about the yield of a risky bond (the
   author's judgment; Damodaran's 7.6% for aerospace is the comparison row).
2. **A 50% chance the cycle works**, settled once, at the proof (the design's
   first growth return, `DesignInputs.proof_years`). Every flow **from the
   proof on** (later growth launches, all deliveries and sales) is weighted by
   it. Flows **before** the proof (the seed, the growth launch at the seed's
   return) are spent whether or not it works. **After the proof the program
   is assumed to work:** no later cycle carries further risk. That is
   optimistic (wear, accidents or a bad batch could still stop it) and is not
   modelled.
3. `DiscountSchedule.risked(rate, success, proof_years)` implements it; the
   existing stepped and flat schedules are unchanged (the new fields default to
   certain success). `growth_cost_report.TIME_RATE` = 0.10, `SUCCESS` = 0.5.
4. **The odds are reported, not buried.** Break-evens are shown at 50%
   (headline), 25% (pessimistic) and 100% (the success case), with ADR 0037's
   stepped rates and the flat 7.6% and 30% as comparisons.

A 50% chance is the author's assumption, not a measurement. Most venture-backed
bets fail, so it is not conservative; a reader who doubts it reads the 25% row.

## Results (Estimate prices; `make growth-cost` section 4)

Steady-state break-even L1 price, $/kg, cheap / dear seed:

| Design | **10%, 50% odds** | 10%, 25% | 10%, 100% | 30%→10% (ADR 0037) | flat 7.6% | flat 30% |
|---|---|---|---|---|---|---|
| Methalox | **352 / 1396** | 479 / 2564 | 288 / 812 | 360 / 1919 | 255 / 594 | 831 / >3000 |
| Methane, 50% | **388 / 1138** | 502 / 1999 | 331 / 707 | 373 / 1306 | 294 / 538 | 886 / >3000 |
| Methane, solved | **137 / 222** | 151 / 317 | 130 / 173 | 135 / 240 | 121 / 150 | 231 / 679 |
| Hydrogen, 50% | **516 / 1454** | 682 / 2556 | 433 / 902 | 490 / 1657 | 381 / 686 | 1208 / >3000 |
| Hydrogen, solved | **134 / 194** | 145 / 262 | 128 / 159 | 132 / 206 | 120 / 140 | 220 / 533 |

Liquidation break-evens at 50% odds (cheap / dear): methalox 374 / 2168,
solved methane 162 / 400, solved hydrogen 159 / 334.

## Consequences

- **Cheap seed: almost nothing moves.** The seed is a small share of the bill.
- **Dear seed: the headline improves.** Solved hydrogen's steady break-even
  falls from $206 to **$194/kg**, back under Suncatcher's $200; solved methane
  from $240 to $222. Methalox improves ($1919 to $1396) and still clears
  neither target on the dear seed.
- **The odds matter on the dear seed.** At 25% solved hydrogen needs $262 and
  methalox $2564. The paper should print that row beside the headline.
- **Value per seed dollar now means expected value per seed dollar.**
- **Delays cost 10% a year.** The odds weight every route the same way, so a
  slower seed route wins when `k x 1.1^-dt > 1`. Under it the seed's Earth
  gravity assist wins (ADR 0039: 1.33x direct on the dear ship, 1.16x on the
  cheap one) and lowers the dear-seed break-evens further: solved hydrogen $179,
  solved methane $201, methalox $1129. The
  same holds for waiting for cheaper launches: ten years must make the seed
  2.6 times cheaper, not 13.8, and Morgan Stanley's forecast (3.3 times by
  2040) passes on the rate alone. The case against a long wait has to rest on
  competition, overhead or the window closing, not on the discount rate.
- `tab:seed_return`'s IRR and the parent's flat 7.6% and 30% rows remain as
  comparisons; the parent's `sec:mass_interest` should lead with this schedule.

## Considered and rejected

- **Keeping 30% a year until proof** (ADR 0037). It charges venture risk
  again for every year of waiting, which decided the seed-route question by
  assumption.
- **A one-time 30% markup** (÷1.3, about 77% odds). Too optimistic for an
  unproven propulsion concept.
- **Weighting every flow, including the seed, by the odds.** Money spent before
  the proof is spent either way.

## Reproduction

`make growth-cost` (sections 3 and 4). Fast:
`test_risk_is_charged_once_at_the_proof_and_time_at_the_bond_rate`
(`tests/test_growth_cost.py`).
