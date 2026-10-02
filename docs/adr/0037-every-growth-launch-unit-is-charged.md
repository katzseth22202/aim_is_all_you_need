# Every growth launch unit is charged, and the cost of capital steps once proven

Status: accepted

Amends: ADR 0036. Its delivery model, its seed prices and its valuation stand;
its cash flows gain the growth phase, and its flat 7.6%/30% gain a stepped rate.

Builds on: ADR 0032 (chambers expended), ADR 0033 (the argon plate),
ADR 0035/0036 (the seed and the harvest).

Date: 2026-10-01

## Context

The parent paper's scratch model (`Balloon-Pulse-Propulsion`, gitignored
`todos/growth_cost_model/`, at companion `00d78c9`) raised asks G1-G4. ADR 0036
charges the seed at `t = 0` and each harvest delivery's lob, plate and
absorber. It charges **nothing for growth**: each 1500 t launch unit lofted
before the harvest, and each one the steady state reinvests, is a $37.5M lob
plus an expended plate, expended departure hardware, tanks, consumables and new
PuffSats. Over ten years the growth bill is about 90 times the cheap seed. ADR
0036 also discounts at a flat 7.6% or 30% for the whole life.

## Decision

**1. The growth charge (G1).** At every return before the harvest, and at every
steady-state return, each launch unit lofted for growth pays its lob; plate,
absorber, spray system and film; argon; departure hardware (chambers or
Raptor 3s) and chamber sprayers; tanks; departure propellant; hydrogen
cryostats; foam plugs and pitch; and the manufacture of the growth PuffSats
and rods it sends onward. The seed's own manufacture is paid at `t = 0`, on
top of the ship. Delivery pushes also pay their argon, spray system and film,
which ADR 0036 never priced.

- **Launch units per return:** `batch x seed / consumed_n`; the steady state
  reinvests `batch x seed / (G_n x consumed_n)`.
- **Plates, chambers and packages ride Wright's law** over the cumulative count
  the program builds, from a seed of one launch unit. Growth and delivery
  plates share one tally.
- **Wright's law is integrated at the unit midpoint** (`src/learning_curve.py`):
  unit `n` costs the curve's integral over `[n - 1/2, n + 1/2]`. The scratch
  model integrated from zero, which charges about one extra first unit (+47% on
  unit 1 at 80%). This is the main reason the break-evens below sit a few
  dollars under the scratch model's.
- **Each delivery flies the schedule that nets most at the prices it actually
  pays** (current plate unit price, spray, argon, film), chosen from
  `harvest.delivery_front`. The scratch model fixed the schedules chosen at
  $200 or $500 with a $114/kg plate.
- **Rods sent onward** are split from growth PuffSats in the ratio the *next*
  cycle consumes them (the scratch model used cycle 0's ratio throughout), and
  **pulses are read per cycle** from `DepartureLedger.pulses` (G1 output 7),
  not scaled from cycle 0.
- **Methalox departure propellant** is charged at $0.2/kg (H7's guess). The
  scratch model left it out. It adds $0.2/kg.

**2. The stepped cost of capital (G2).** 30% a year (Gompers et al.) until the
design's first growth return, `times[1]` (5.46 yr for every chamber design, 6.55
yr for methalox, which flies three-synodic cycles only), then **10%**. The 10%
is the user's judgment of a proven system, a little above Damodaran's 7.6%,
**unsourced**. Because one measured cycle is a thin proof, the late rate is quoted
as a **10-20% band** (user, 2026-10-01, on the condition that it does not change
the conclusion; it does not). Across the band the solved chambers break even at
$132-178 / $206-427 (cheap / dear seed), under $500 on both seeds and under $200
on the cheap one. Methalox breaks even at $360-574 / $1919 to over $3000. Outputs are break-even `p_L1` and value per seed
dollar, not IRR, since growth costs give the cash flows more than one sign
change.

**3. The prices (H1-H9)** are the parent's, every one an unsourced guess except
the lob ($25/kg, paper). Three books: **Paper's prices** (plate $500/kg first on
an 80% curve, $2M chamber, $20/kg fleet, new lines at the Estimate),
**Estimate** (the headline), **Pessimistic** (every line at its worst at once, a
labeled bound). The package is on Wright's law by default: $100 until 100 000
are built, 80% per doubling, $10 floor.

**4. The lifting rocket is assumed big enough to loft 1500 t to 400 km**
(user, 2026-10-01). Today's Super Heavy does not quite do it. The parent's
`sec:vertical_lob` lofts 1250-1430 t from a ~4900 t liftoff, with the braking
reserve, under 4 g, at 380 s (1070-1250 t at 350 s). The user's judgment is that
a booster **10-20% bigger** closes the gap, and the arithmetic backs it. At the
same thrust-to-weight and propellant fraction, lofted mass scales with the
vehicle, so the braked range becomes 1375-1716 t at 1.1-1.2 times. A vertical
integration without the brake or drag (5000 t liftoff, 250 t dry, 75 MN, all
scaled together) lofts 1564 t to 400 km at 1.1 and 1706 t at 1.2. **At the
pessimistic 350 s average, only the full 20% reaches 1500 t.** The lob stays
priced per kilogram lofted, $25/kg, so the bigger booster is assumed to cost
proportionally more per flight ($37.5M per unit). A vacuum second stage above
about 20 km was checked and does not help (about 395 km at 1500 t on today's
booster): the lob's loss is gravity loss, set by liftoff thrust, not by Isp.

**5. The seam.** `src/growth_cost.py` is pure arithmetic over `DesignInputs`
(floats in kilograms and dollars, like `conic_kernel`). It is tested on
hand-built chains in the fast suite, including the closed forms for
liquidation, the ADR 0036 perpetuity and the `C_g / (G - 1)` steady-state
overhead. `src/growth_cost_inputs.py` builds the inputs from the growth ledger
(slow). `src/growth_cost_report.py` prints the report. `DesignChain` now keeps
its per-cycle ledgers, so the chain is flown once.

## Results (Estimate, stepped 30% -> 10%; `make growth-cost`)

**Amended by ADR 0038.** Each payload now departs on the next window's burn.
The tables below carry the corrected figures; ADR 0038 lists what they replaced
(for example solved hydrogen's steady break-even, 129 / 186 before the fix).

Steady-state cost per kilogram at L1 is undiscounted, a lap into the steady
state, seed excluded. Break-even is the sale price that repays the seed and
every growth cost, cheap / dear seed ($337 / $9293 per kilogram).

| Design | Steady $/kg | Lob | Plate + spray | Departure hw | Fleet | BE steady | BE liquidation | Value at $500 |
|---|---|---|---|---|---|---|---|---|
| Methalox | 169 | 115 | 46 | 4 | 1 | 360 / 1919 | 388 / >3000 | 3.3 / 0.1 |
| Methane, 50% | 184 | 110 | 42 | 21 | 4 | 373 / 1306 | 379 / 2022 | 4.3 / 0.2 |
| Methane, solved | 100 | 69 | 22 | 6 | 1 | **135 / 240** | 157 / 452 | 88 / 3.5 |
| Hydrogen, 50% | 228 | 117 | 45 | 37 | 4 | 490 / 1657 | 481 / 2422 | 1.2 / 0.0 |
| Hydrogen, solved | 100 | 66 | 20 | 7 | 1 | **132 / 206** | 153 / 371 | 127 / 5.0 |

Paper's prices: methalox 194 $/kg (BE 501 / 2061), solved methane 99 (144 /
250), solved hydrogen 95 (132 / 207). Pessimistic: 410-1035 $/kg, and no design
repays the seed under $500; solved hydrogen misses by least, at 509 / 582.

**Validation (G1).** Before ADR 0038, with the growth charge off and the plate
flat at $114/kg, the seed's IRRs at $500 reproduced the parent's
`tab:seed_return` exactly: methalox 45 / 3 (liquidation) and 39 / 10 (steady),
solved hydrogen 87 / 33 and 84 / 36. With ADR 0038 they are 42 / 2 and 38 / 9,
and 81 / 29 and 80 / 33. The test pins the corrected values.

**Readings.** The parent's draft holds. Methalox does not pay for itself at the
bars that matter ($360 cheap, $1919 dear, against the $500 cap). A solved chamber
brings the break-even under Suncatcher's $200 on the cheap seed and to just above
it on the dear one ($206-240). Chamber efficiency decides, not chamber price. The lob is two
thirds of the solved chambers' cost.

## G3: the growth index

The cost model's indexing matches `harvest.chain_returns` and the growth
ledger's own definition of `G_n`. A unit lofted at return `n` is pushed by
return `n`'s waves and grows by `G_n`. Answering G3 exposed an off-by-one
underneath it: every ledger departed that unit on cycle `n`'s own outbound burn,
though it leaves on cycle `n + 1`. ADR 0038 fixes it.

## G4: rod size against package count

Not traded. A 2.5 kg rod carries three expended packages, so rods cost about
$124/kg at a $100 package against about $6/kg for growth PuffSats. They are
still $1-3 per delivered kilogram for the solved chambers. A 10 kg rod's 130 t
wall (`tab:layered_wall_mass`) is the other side of the trade.

## Not priced

- Plate and chamber recovery; a falling lob or sale price over time.
- Demand capping the fleet. Value per seed dollar is a perpetuity and
  overstates a demand-limited fleet; the break-even prices are the robust output.
- A program larger than one launch unit of seed, which would learn faster.
- Argon supply. A fleet near 50 launch units per cycle sprays about 30 000 t a
  cycle.

## Considered and rejected

- **Charging every delivery the 100th plate's price** (`tab:delivery_ledger`'s
  "paper's figures" column). It assumes 99 plates are already paid for. The
  program charges each plate at its own count instead.
- **Integrating Wright's law from zero** (scratch model). It overcharges the
  first unit by 47% at 80%, which matters at the 6-50 plates a decade builds.
- **IRR under the stepped schedule.** Not well defined.

## Reproduction

`make growth-cost` (about 3.5 min, nearly all of it building the chains;
`--quick` prints the headline tables only) at the committing revision. Plate 0.7, parking orbit 20 days, delivery
schedules from `harvest._START_PRICES`, a 300-year horizon (the tail beyond it
is under 1e-12 at 10%), and the steady-state cost measured on chain lap 3 after
the harvest. Break-evens are solved by `brentq` on [0, 3000] to 1e-3 $/kg. Tests:
`tests/test_learning_curve.py`, `tests/test_growth_cost.py` (fast) and
`tests/test_growth_cost_inputs.py` (slow, including the `tab:seed_return`
validation and the headline pins).
