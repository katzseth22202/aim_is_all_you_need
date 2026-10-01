# The seed flies an expended, refuelled Starship

Status: accepted; amended by ADR 0036 (the stripped ship is the baseline, and
the present-value column is replaced). Its route stands.

Builds on: ADR 0033 and ADR 0034 (the growth ledger), ADR 0032 (the chamber
departure).

Date: 2026-10-01 (rebuilt the same day; see "Provenance")

## Context

The growth ledger counts kilograms and prices no money. The parent's
`tab:delivery_ledger` leaves out "seeding the Jupiter cycle". The parent's
`sec:mass_interest` (parent `9f63126`) prices the seed on one route and cites
`make seed-cost` at companion revision `18773ba` for its new `tab:seed_amortization`.

## Provenance

Revision `18773ba` was never pushed. It is not in this repository, on GitHub,
or in any checkout we could reach, and the module behind it was lost with it.
`src/seed_cost.py` was **rebuilt from the parent's prose** in `9f63126`. It was
accepted only because it reproduces every printed figure of `tab:seed_amortization`
and its text, to the digits printed:

| | Seed (t) | Ships | Annual | 10 yr | Seed per fleet kg ($) | Built | PV 7.6% | PV 30% |
|---|---|---|---|---|---|---|---|---|
| Methalox | 72.7 | 1.44 | 19% | 5.09 | 179-2915 | 12.5 | 2.45 | 0.369 |
| Methane, 50% | 83.3 | 1.65 | 23% | 8.21 | 111-1809 | 10.5 | 3.95 | 0.595 |
| Methane, solved | 80.8 | 1.60 | 51% | 69.5 | 13-214 | 34.4 | 33.4 | 5.04 |
| Hydrogen, 50% | 83.8 | 1.66 | 22% | 7.28 | 125-2038 | 10.2 | 3.50 | 0.528 |
| Hydrogen, solved | 80.7 | 1.60 | 57% | 104 | 9-142 | 44.3 | 50.1 | 7.56 |

The ship: 50.53 t sent, a 271 s burn, a 31 m/s steered loss, a mass ratio of
9.85, $910-14 843/kg, and Wright's 0.30 at the forty-fourth chamber.

One aside in the parent's prose was **not** reproduced. "Two full ships in series
… sends 56.5 t" matches no staging model we tried, which gave 43-179 t. It is
prose only, appears in no table, and nothing here depends on it. The parent
should drop it or re-derive it on the hand-back.

## Decision

1. **The seed flies the chain's first cycle.** That is the three-synodic cycle
   leaving Earth in November 2026 at v_inf = 11.99 km/s. The excess speed is
   taken from the chain's burn above the 20-day orbit's 200 km periapsis speed.
2. **The ship is refuelled in low orbit and burns straight to that speed from a
   200 km circular orbit,** `sqrt(v_inf^2 + 2 mu / r) - sqrt(mu / r)` = 8.50 km/s,
   at 380 s. The finite burn's steered loss is charged with `finite_burn_loss`
   on the circular orbit. The ship is expended, so its dry mass is charged in full.
3. **Twelve 100 t tankers fill 1200 t.** The ship's own ascent carries the
   PuffSats, so one ship costs 13 launches and one hull.
4. **Flight prices are the parent's three:** Musk's $2M a flight, and Goldman's
   $183/kg and Morgan Stanley's $500/kg, each over 100 t. **The hull is a
   hypothesis.** No citable build price exists.
5. **The seed per launch unit** is the PuffSat mass one 1500 t unit consumes on
   cycle 0, with both waves counted as they leave Earth. Each wave's methalox
   correction is grossed back up. Methalox flies the three-synodic-only chain,
   the chambers the flown chain.
6. **Built** counts the departure units bought by the cycles that finish within
   ten years: `sum_i units_i prod_{j<i} G_j`, Raptors for methalox and chambers
   otherwise.
7. **Present value** was `M10 / (1+r)^10` at r = 7.6% (Damodaran, aerospace and
   defense, January 2026) and 30% (Gompers et al. 2020, median VC hurdle), with
   annual compounding (`COSTS_OF_CAPITAL`).

## Consequences

- `make seed-cost` reproduces `tab:seed_amortization` as the "stock comparison" block.
- ADR 0036 changes the baseline ship and replaces decision 7, which values a fleet
  kilogram at what a seed kilogram cost.

## Reproduction

`make seed-cost` at the committing revision; `tests/test_seed_cost.py`, whose
slow tests pin the table rows.
