# ADR 0057: Price the seed at $150 a kilogram, fly it on the Earth loop, retire the 30% rows

Date: 2026-10-13

Status: Accepted; its headline route is superseded by ADR 0059 (the two-year
Earth loop replaces EEJ, and ADR 0059 carries the current table). Its prices and
valuation stand. Amends ADR 0036 (the seed's price ends), ADR 0037 and ADR 0040
(the comparison rates), and ADR 0039 (EEJ becomes the headline, not a sensitivity).

## Context

The seed's cheap end was Musk's $2M a flight: $337 per kilogram sent on the 40 t
stripped ship. At that price the seed was a rounding error in every break-even,
and the question "how cheap can the seed get?" never came up. The author
(2026-10-13) calls it misleading. The comparison the program faces is SpaceX
selling $150/kg to low orbit and $500/kg to high orbit. The analysts' figures
already quoted (Goldman $183/kg, Morgan Stanley $500/kg) are worse than that.

The reports also carried ADR 0037's stepped schedule (30% a year until the first
growth return) and a flat 30% beside ADR 0040's expected value. The author's
view: a 30% venture rate prices speculative science *and* a speculative market.
Here the physics and the demand are reasonably clear, but the risk of failure is
high. ADR 0040 already says exactly that: time at 10% a year, and a 50% chance the
cycle works, settled once when the first impactors come back.

## Decision

1. **Seed flights are $150/kg × 100 t = $15M** (`seed_cost.BASELINE_FLIGHT`). The
   ends become **(40 t ship, $5M hull, $150/kg) = $2172/kg** and **(60 t, $20M,
   Morgan Stanley $500/kg) = $9293/kg**. Musk's price is gone from
   `FLIGHT_PRICES`, and the stock comparison's cheap end moves to the baseline
   ($4255/kg, was $910). Goldman stays in the price table.
2. **The headline seed flies EEJ** (ADR 0039). It reaches Jupiter later but more
   cheaply, which the author accepts. `growth_cost_inputs.SeedFlight` carries
   the route: ×0.6991 per kilogram on the 40 t ship, ×0.6112 on the 60 t ship,
   2.184 years later. `design_inputs` and `seed_prices` apply
   `HEADLINE_SEED = EEJ` by default, so the growth-cost, survivable-chamber,
   hardware-recovery and slush reports all move together. Section 9 (seed routes)
   and the tab:seed_return validation fly `DIRECT` explicitly. The ratios are
   ADR 0039's (node burns at 380 s); ADR 0059 re-searched the loop at 370 s and replaced it.
3. **Valuation is ADR 0040's alone.** `growth_cost_report` drops the stepped
   30%→10/15/20% columns and the flat 30%. It keeps the flat 7.6% (Damodaran)
   and the 25% and 100% odds rows. The 25% row is the pessimistic valuation.
   `DiscountSchedule.stepped` stays in `growth_cost` (tested arithmetic).
4. **The pessimistic price book is unchanged.** It already prices every line at
   the analysts' worst, and its seed is the dear end.

## Results (spray cup, lob climbing at 1.1 km/s, Estimate, 10% a year, 50% odds)

Steady-state break-even at L1, $/kg. "Free" is the seed at $0. Entries are
cheap / dear seed.

| Design | Free | Direct | **EEJ (headline)** | EEJ, Pessimistic book | EEJ, 25% odds |
|---|---|---|---|---|---|
| Methalox | 631 | 1566 / >3000 | 1437 / >3000 | 2420 / >3000 | 2478 / >3000 |
| CH4 AR100, 3.5 kg pitch | 289 | 463 / 1027 | **439 / 845** | 1280 / 1687 | 645 / 1456 |
| CH4 AR100, 24 kg | 304 | 492 / 1105 | **467 / 908** | 1345 / 1787 | 690 / 1570 |
| CH4 AR150, 24 kg | 289 | 463 / 1030 | 439 / 848 | 1276 / 1685 | 646 / 1460 |
| CH4 AR300, 24 kg | 288 | 472 / 1071 | 447 / 878 | 1280 / 1712 | 662 / 1522 |
| CH4 AR150, 24 kg, 3 Hz | 247 | 382 / 818 | **363 / 677** | 1083 / 1398 | 521 / 1147 |
| Hydrogen, solved | 185 | 231 / 377 | 225 / 330 | 735 / 840 | 284 / 492 |
| Hydrogen, bulged, 2 Hz | 329 | 469 / 925 | 450 / 778 | 1297 / 1626 | 636 / 1291 |

For comparison, the methane chamber at AR100 with 3.5 kg pitch was $316 on
Musk's seed.

The `DESIGNS` headline, Estimate book with ADR 0033's plate on EEJ:
- methalox $532 / $1129;
- solved methane $152 / $201;
- solved hydrogen $144 / $179.

The tab:seed_return validation on the direct seed (IRR %, liquidation and
steady):
- methalox 17.9 / 1.7 and 18.9 / 8.7;
- solved hydrogen 49.8 / 29.2 and 51.1 / 33.1.

## Consequences

- **The seed is now about a third of the methane chamber's break-even** ($150 of
  $439), against 9% on Musk's price. On the dear seed it is two thirds. Cutting
  the seed's cost is worth real money for the first time.
- **No methane row clears Suncatcher's $200 even with a free seed.** The floor
  is growth cost. The 3 Hz AR150 chamber clears Starcloud's $500 on the cheap
  seed ($363), and the 2 Hz chambers do too ($439-467).
- **The parent's `sec:mass_interest`, `tab:seed_amortization` and
  `tab:seed_return` were printed on Musk's price** and need re-deriving. The
  for-parent docs (`docs/growth_cost_for_parent.md`) still quote the old ends.
  They are left as dated hand-off records, and the parent owes the update.

## Considered and rejected

- **Keeping Musk's price as an optimistic bound.** It is a single quote with no
  delivered-cost evidence, and it hides the seed's share of the bill.
- **A 50%-a-year rate until the first return.** It charges the failure risk
  again every year (ADR 0040's argument against 30%) and is harsher still.

## Reproduction

`make growth-cost`, `make survivable-chamber` (cost report), `make seed-cost`.
Slow tests: `tests/test_growth_cost_inputs.py` pins the headline and the
validation. The table above came from `survivable_chamber.cost_designs()` and
`_priced`, with `break_even_price` under `growth_cost_report.risked` on
`climbing_lob(book, OPERATING_RISE_SPEED)`. Direct is `delayed(inputs, -2.184)`.
