# The seed is worth the cargo it delivers, and k is optimised for dollars

Status: accepted

Amends: ADR 0035. Its route stands. Its baseline ship becomes the stripped
ship, and its present-value column is replaced.

Builds on: ADR 0033/0034 (the growth ledger and its plate schedule).

Date: 2026-10-01

## Context

ADR 0035 divided the ten-year *fleet* multiple by `(1+r)^10`, which values a
fleet kilogram at what a seed kilogram cost. That is wrong in two ways:

1. The last wave need not go back to Jupiter. A returning PuffSat kilogram
   pushes `P` kilograms of cargo to Sun-Earth L1, so the stake is worth
   delivered cargo, not fleet mass.
2. A fleet can stop growing and pay a dividend instead (a steady state).

ADR 0035 also flew a stock 85 t, six-engine Starship and priced its hull at
$20-100M. The user judged both wrong for a ship that is thrown away.

## Decision

### The seed ship is stripped (user, 2026-10-01)

It is expended, so it carries no heat shield, flaps or landing propellant. It
flies **three vacuum Raptors**, so thrust halves and the burn doubles to 543 s.
The steered loss is **recharged** for that burn on the same circular orbit:
**117 m/s**, not 31. The dry masses, 40 t and 60 t, are unsourced; they are
ADR 0035's sweep points. The stripped hull is swept at **$5M**, the user's
hypothesis (unsourced), and **$20M** (`SHIP_PRICES`). $50M and above is
unreasonably pessimistic for a hull with no heat shield or flaps (user,
2026-10-01). ADR 0035's $20/50/100M stay only on the stock ship
(`STOCK_SHIP_PRICES`), so its ends remain reproducible.

| Ship | Sent (t) | Musk $/kg ($5M hull) | Goldman | Morgan Stanley |
|---|---|---|---|---|
| Stripped 40 t | 92.1 | 337 | 2637 | 7112 |
| Stripped 60 t | 72.1 | 430 | 3369 | 9085 |
| Stock 85 t (comparison) | 50.5 | 614 | 4807 | 12 963 |

The stripped ship sends 92.1 t, below the 95.5 t a hand check gets by reusing
the stock ship's 31 m/s. **Under the bank prices the twelve tankers are 90-92%
of a $5M-hull ship's bill** (85-90% at $20M), so the hull price matters little.
The payload matters more: at a fixed hull and flight price the 40 t ship costs
0.55 of the stock ship per kilogram.

**The valuation is quoted at two seed-price ends:** $337/kg (40 t, $5M hull,
Musk) to $9293/kg (60 t, $20M hull, Morgan Stanley). These are the corners of the
stripped ship's whole sweep (`STRIPPED_LOW`, `STRIPPED_HIGH`). The stock ship is a comparison row only. Its figures in
the parent ($910-14 840/kg, 1.4-1.7 ships per launch unit) are replaced, not
kept beside the new ones. Per launch unit the stripped ship carries the seed in
0.79-0.91 ships, against the stock ship's 1.44-1.66.

### Delivery to L1 (`src/harvest.py`)

- **The launch unit is the growth ledger's:** 1500 t at the 400 km intercept,
  with a 150 t plate left at L1 and the argon's tank at 0.0105 kg per kg dropped.
  Everything else delivered is cargo, and the cargo pays its own 30 m/s halo
  insertion in methalox. This differs from the parent's `tab:delivery_ledger`,
  which books 500 t of cargo and leaves the rest as unspecified room.
- **The push target** runs from rest at 400 km to the periapsis speed of a
  transfer with apoapsis at L1 (1.5e6 km): 10.82 km/s.
- **The harvest wave arrives at the return's own speed** (`nozzle_wave_v_b` at
  400 km) and pays only the return's DSM proxy. A harvested batch needs no
  split, so it pays no early-arrival burn.
- **The plate's loading is optimised under the k <= 10 cap** (user, 2026-10-01),
  as in the growth ledger, and argon on ice PuffSats stays the slug (user,
  2026-10-01). The schedule is the capped Pontryagin one
  (`optimal_plate_push`). Its starting price is chosen to **maximise the dollars
  each returning PuffSat kilogram nets, `P (p_L1 - c_delivery)`**. It is picked
  from 81 log-spaced starting prices (1e-4 to 1) plus a constant k = 0 push
  (`_START_PRICES`). PuffSats are the fleet, so they are not a delivery cost.
  Argon displaces cargo, and that is how it is charged; its own dollar price is
  unpriced. The optimum depends on the sale price. Constant k = 0 and k = 10 are
  reported as references.

| Return | k = 0: P, lob $/kg | k = 10 held: P, cargo t, lob $/kg | Optimised at $500: k, P, cargo t, lob $/kg | at $200 |
|---|---|---|---|---|
| Cycle 0 (3S), 55.42 km/s | 7.48, 28.0 | 10.15, 674, 55.7 | 10->2.4, 10.31, 796, 47.1 | 10->1.2, 9.80, 936, 40.1 |
| Cycle 1 (2S), 61.35 km/s | 8.39, 28.0 | 11.98, 729, 51.4 | 10->3.3, 12.04, 810, 46.3 | 10->2.0, 11.67, 899, 41.7 |

**Optimising k beats holding it at 10 on both counts.** The schedule opens at
the cap and tapers, so it keeps 80-160 t more cargo per lob *and* a slightly
higher `P`. At $500 it cuts the lob from $50-59 to $46-47 per cargo
kilogram. At $200 it goes further toward k = 0 (lob about $40), giving up `P` for
cargo. Across both chains `P` runs 9.3-12.6 at $500 and 8.9-12.2 at $200, against
6.9-8.7 at k = 0.

`c_delivery` is the lob at $25 per lofted kilogram, plus the plate and the
6 t absorber at $114/kg learned or $500/kg early. At the harvest return it is
$68-69/kg learned and $128-132/kg early, so the margin `p_L1 - c_delivery` is
positive at every swept price: $52-55 learned and $14 early at $100/kg.

### Liquidation and the steady state

- **Liquidation:** at the last return within ten years (9.83 yr on both chains),
  the batch is `M10 / G_last` per seed kilogram. That is the stepwise multiple
  without the last cycle's growth (pinned in a test). The whole batch is
  delivered, with its DSM proxy charged.
- **Steady state from the same return:** each return reinvests `1/G_n` of its
  batch and delivers the rest. The dividend uses each cycle's own `G_n`, date
  and delivery. After the chain's last cycle the chain repeats, so the tail is
  a geometric series. With a constant `G` and `T`, steady/liquidation =
  `(1 - 1/G)(1+r)^T / ((1+r)^T - 1)`, which is above one exactly when
  `G > (1+r)^T`. That is the same test as g > r. `P` cancels out of it, so it
  holds for any delivery model.
- Annual compounding throughout, `t` in years from the seed's departure
  (2026-11-09 TDB). r = 7.6% and 30% are ADR 0035's anchors.

### Results (learned plate, k optimised; `make seed-harvest`)

**Grow or harvest.** Every cycle of every design clears 7.6%, except
hydrogen at 50%, where 82% do. At 30% only the solved chambers clear on every
cycle. Methalox and hydrogen at 50% clear none, and methane at 50% clears 18%.
The fleet-mass IRR, `M10^(1/9.83) - 1`, trails `annual_growth` by the stepwise
gap (-1.4 points for methalox).

**The seed's IRR**, at the cheap / dear seed end:

| Design | Liquidation, $500 | Liquidation, $200 | Steady, $500 | Steady, $200 |
|---|---|---|---|---|
| Methalox | 45% / 3.4% | 29% / none | 39% / 9.5% | 26% / 4.3% |
| Methane, 50% | 54% / 9.8% | 37% / none | 46% / 13.1% | 32% / 6.6% |
| Methane, solved | 81% / 29% | 61% / 15% | 78% / 32% | 60% / 21% |
| Hydrogen, 50% | 53% / 8.9% | 36% / none | 44% / 12.2% | 31% / 6.0% |
| Hydrogen, solved | 87% / 33% | 66% / 18% | 85% / 36% | 65% / 24% |

**Break-even L1 price** ($/kg), cheap / dear seed end, liquidation then steady:

| Design | r = 7.6% | r = 30% |
|---|---|---|
| Methalox | 72 / >500, 56 / 368 | 215 / >500, 264 / >500 |
| Methane, 50% | 60 / 422, 50 / 232 | 145 / >500, 174 / >500 |
| Methane, solved | 45 / 134, 42 / 70 | 65 / >500, 60 / 423 |
| Hydrogen, 50% | 61 / 452, 51 / 258 | 152 / >500, 193 / >500 |
| Hydrogen, solved | 44 / 112, 42 / 61 | 59 / 410, 54 / 316 |

Reading:

- **The seed price still decides the outcome.** On the cheap seed every design
  repays a venture hurdle below $300/kg. On the dear one only the solved
  chambers clear 30% below the cap, from $316/kg (hydrogen, steady state).
- **The steady state beats liquidation exactly where growth beats r.** At 30%
  methalox would rather liquidate (steady/liquidation 0.76), while the solved
  chambers would rather hold the fleet level (1.3-1.4).
- **At the dear seed end, methalox's liquidation never repays 7.6% below $500,**
  while its steady state does from $368.

## Consequences

- The parent's `tab:seed_amortization` PV columns become the seed's IRR
  (liquidation and steady state, at $500 and $200, both seed ends) plus the
  break-even `p_L1` at 7.6% and 30%. The caption states the k <= 10 optimised
  schedule, plate 0.7 and argon. Its stock-ship figures are replaced.
- The parent's "two ships in series" aside is unreproduced (ADR 0035).
- `fleet_present_value` is kept for continuity, labelled "fleet mass, not dollars".

## Not priced

- k above 10 (`P` keeps rising, to about 15 at k = 30, outside the validated range).
- Argon's own dollar price. The optimum here charges argon only for the cargo
  it displaces; pricing it in dollars would push k down.
- A falling launch price over time (`p_L1` and the lob are held fixed).
- Demand capping the fleet. When g > r the value is unbounded, so in practice
  the steady state is set by demand, not by finance.
- Plate reuse, chamber recovery, and Wright's law on the plate within the
  harvest (the plate price is held at one of the two ends).

## Considered and rejected

- **Constant k = 10 for delivery** (the todo's first assumption). It is
  dominated: the optimised schedule beats it on cargo and `P` at every sale price.
- **Starship-to-L1 prices ($640-2300/kg) as the sale price.** They are the
  competitor's cost, and the user rejected them as a sale price. $500/kg is the cap.
- **Using the chain's mean `G` and `T` for the steady state.** The report keeps
  them only for the steady/liquidation ratio column. The valuation uses each
  cycle's own `G_n`, date and delivery.

## Reproduction

`make seed-harvest` (about 3.5 min) and `make seed-cost` (about 2.5 min) at the
committing revision. The plate is 0.7 and the parking orbit 20 days. The
schedule is chosen from `_START_PRICES` = 81 log-spaced starting prices on
[1e-4, 1] plus k = 0. Break-even prices are solved by `brentq` on [0, 500] to
1e-6 $/kg, and IRRs on [0, 5] (steady state from 1e-6). Tests:
`tests/test_harvest.py` and `tests/test_seed_cost.py`.
