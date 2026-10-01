# Growth cost model: assumptions, costs and results, for the parent paper

Raised 2026-10-01 in `aim_is_all_you_need`, answering the parent's asks G1-G4
(`Balloon-Pulse-Propulsion`, scratch `todos/growth_cost_model/ASKS.md`).
**Written to be copied into the paper repo**, so it repeats context the paper
already has. Backing decision: `docs/adr/0037-every-growth-launch-unit-is-charged.md`.
Reproduce with `make growth-cost` (about 3.5 min). Code: `src/growth_cost.py`
(pure cost model), `src/growth_cost_inputs.py` (the chain it prices),
`src/learning_curve.py`, `src/growth_cost_report.py`.

Ground rules as usual: neither repo is the source of truth. Where a number here
disagrees with the scratch model, the reason is given.

**Nothing has been written into `templateArxiv.tex`.**

**Bottom line.** With every growth cost charged, a *solved* pulsed chamber meets
the cost targets and methalox probably does not. Under the Estimate prices, a
solved chamber delivers to L1 for $98/kg in steady state and repays the seed at
$129-174/kg on the cheap seed and $186-364 on the dear one, across a late rate
of 10-20%. That is under Starcloud's $500 cap on both seeds and under
Suncatcher's $200 on the cheap one. Methalox's steady-state cost is $163/kg,
under both bars, but it grows too slowly to repay its seed in time. It needs $346-564/kg on
the cheap seed and $1694 to over $3000 on the dear one. A chamber at half its
ceiling does no better than methalox, so the target is met by chamber
*efficiency*, not by having a chamber. With every price at its worst at once
(the Pessimistic book), nothing repays the seed under $500.

---

## 1. Verdict on the scratch model

**The numbers match.** Every scratch figure is reproduced to within a few
dollars per kilogram, and the scratch model's conclusions stand. The concrete
implementation differs from it in five places. All are corrections or
refinements, and none changes a ranking:

| Change | Effect |
|---|---|
| Wright's law integrated at the unit midpoint, not from zero. Scratch charged about one extra first plate or chamber per program (+47% on unit 1 at 80%). | Steady break-evens fall $2-16/kg (methalox 359 / 1710 to 346 / 1694; solved hydrogen 133 / 188 to 129 / 186) |
| Each delivery picks its plate schedule at the prices it actually pays (plate, spray, argon, film). Scratch used the schedules optimised at $200/$500 with a $114/kg plate. | Pessimistic $/kg falls 3-4% (more cargo per lob when the plate is dear) |
| Rods sent onward split by the *next* cycle's consumption; pulses read per cycle, not scaled from cycle 0 | Under $0.5/kg |
| Methalox departure propellant charged at $0.2/kg (scratch: free) | +$0.2/kg |
| Hydrogen and methane charged on gas *launched*, including boil-off | Under $0.5/kg |

**Validation (G1):** with the growth charge off and the plate flat at $114/kg,
the code returns `tab:seed_return` exactly: methalox 45 / 3 and 39 / 10, solved
hydrogen 87 / 33 and 84 / 36 (IRR %, liquidation and steady, cheap / dear seed,
$500). This is a pinned test.

---

## 2. Every assumption

### Physics, from the companion (not guesses)

| Item | Value | Source |
|---|---|---|
| Launch unit | 1500 t lofted to the 400 km intercept, on a lifting rocket 10-20% bigger than Super Heavy (**assumed**, §4 item 2) | ADR 0033 (`LAUNCH_UNIT`), ADR 0037 |
| Plate + absorber, expended every unit | 150 t + 6 t | ADR 0033, ADR 0036 |
| Plate efficiency, slug | 0.7, argon on ice PuffSats, k <= 10 | ADR 0033 |
| Growth PuffSat | 60 kg, two steering packages | parent figure caption, ASKS H6 |
| Departure rod | 2.5 kg, three packages, expended per pulse | `sec:rod_2p5kg` |
| Pulses per departure | 5261-7569, per cycle | `DepartureLedger.pulses` |
| Seed ship, seed price | stripped expended Starship, $337 / $9293 per kg | ADR 0035/0036 |
| Delivery | to the L1 transfer (10.82 km/s from 400 km), 30 m/s halo insertion in methalox | ADR 0036 |

Per launch unit, cycle 0:

| Design | Departure units | Tanks (t) | Propellant (t) | Argon (t) | Sent onward (t) | Consumed (t) | G_0 |
|---|---|---|---|---|---|---|---|
| Methalox | 2 Raptor 3 | 9.8 | 543 | 581 | 150 | 72.7 | 2.07 |
| Methane, 50% | 1 chamber | 13.8 | 406 | 588 | 187 | 83.3 | 2.25 |
| Methane, solved | 1 | 11.3 | 333 | 577 | 286 | 80.8 | 3.54 |
| Hydrogen, 50% | 1 | 71.0 | 368 | 576 | 196 | 83.8 | 2.34 |
| Hydrogen, solved | 1 | 54.7 | 284 | 564 | 317 | 80.7 | 3.93 |

### Prices: every one is a guess unless marked "paper"

| Line | Estimate (headline) | Pessimistic | Paper's prices |
|---|---|---|---|
| Lob, per kg lofted | **$25 (paper)**; author's band $10-25 | $50 | $25 |
| Plate + absorber (156 t) | $30M first, 80%, $6M floor | $78M flat | $78M first ($500/kg), 80%, no floor |
| Plate spray system, per plate | $3M | $8M | $3M |
| Ablative film | 0.1% of PuffSat mass on the plate, $10/kg | 4% (Orion) | as Estimate |
| Methane chamber (21.2 t) | $40M first, 80%, $3M floor | $200M flat | $2M flat, expended (ADR 0032) |
| Hydrogen chamber (34.2 t) | $60M first, 80%, $5M floor | $200M flat | $2M flat, expended |
| Chamber sprayers, per chamber | $2M | $5M | $2M |
| Raptor 3 | $1M | $2M | $1M |
| Tanks | $40/kg | $100/kg | $40/kg |
| Hydrogen cryostats | $300/kg | $1000/kg | $300/kg |
| Argon | $1/kg | $5/kg | $1/kg |
| Methane / hydrogen / methalox | $1 / $6 / $0.2 per kg | $1 / $10 / $0.5 | as Estimate |
| Plugs / pitch | $5 / $2 per kg | $10 / $5 | as Estimate |
| Steering package | $100 to 100 000 built, then 80% per doubling, $10 floor | $1000 flat | — |
| PuffSat / rod bodies | $3 / $4 per kg | $5 / $5 | — |
| Fleet, flat | — | — | **$20/kg (paper)** for every PuffSat and rod |

### Money

| Item | Value | Status |
|---|---|---|
| Cost of capital | **30% until the first growth return, then 10%** | 30%: Gompers et al. 10%: the author's judgment, **unsourced** |
| First growth return | 5.46 yr (chambers), 6.55 yr (methalox) | `times[1]` |
| Sale price at L1 | $500 cap (Starcloud), $200 (Suncatcher) | parent `sec:heat_shield_bill` |
| Horizon | 300 yr; steady-state $/kg measured on chain lap 3 after the harvest | modelling choice |

---

## 3. Results

### Headline: Estimate prices, stepped rate

"Steady $/kg" is the undiscounted cost per kilogram at L1, a lap into the
steady state, seed excluded. Break-even (BE) is the sale price that repays the
seed and every growth cost. Each pair is cheap / dear seed.

| Design | Steady $/kg | Lob | Plate + spray | Departure hw + spray | Fleet | Other | BE steady | BE liquidation | Value per seed $ at $500 | at $200 |
|---|---|---|---|---|---|---|---|---|---|---|
| Methalox | 163 | 112 | 44 | 3 | 1 | 3 | 346 / 1694 | 357 / 2637 | 4.0 / 0.1 | -1.8 / -0.1 |
| Methane, 50% | 167 | 103 | 37 | 20 | 3 | 3 | 346 / 972 | 322 / 1374 | 7.0 / 0.3 | -4.6 / -0.2 |
| **Methane, solved** | **98** | 68 | 21 | 6 | 1 | 2 | **132 / 211** | 148 / 368 | 119 / 4.6 | 22 / 0.9 |
| Hydrogen, 50% | 199 | 107 | 39 | 32 | 3 | 18 | 445 / 1168 | 394 / 1534 | 2.8 / 0.1 | -7.1 / -0.3 |
| **Hydrogen, solved** | **98** | 65 | 20 | 8 | 1 | 5 | **129 / 186** | 147 / 310 | 171 / 6.7 | 32 / 1.2 |

"Other" is argon, tanks, propellant, cryostats, plugs, pitch and film.

### The other price books

| Design | Paper's: steady $/kg | Paper's: BE steady | Pessimistic: steady $/kg | Pessimistic: BE steady |
|---|---|---|---|---|
| Methalox | 185 | 480 / 1828 | 488 | 840 / 2194 |
| Methane, 50% | 166 | 366 / 993 | 831 | 1376 / 2002 |
| Methane, solved | 96 | 139 / 220 | 426 | 556 / 635 |
| Hydrogen, 50% | 188 | 429 / 1152 | 916 | 1579 / 2302 |
| Hydrogen, solved | 93 | 129 / 186 | 400 | 511 / 566 |

With every line at its worst at once, no design repays the seed under $500.

### By cost of capital (Estimate, steady break-even)

| Design | 30%->10% | 30%->15% | 30%->20% | Flat 7.6% | Flat 30% | 30%->10%, growth uncharged |
|---|---|---|---|---|---|---|
| Methalox | 346 / 1694 | 449 / 2601 | 564 / >3000 | 250 / 541 | 832 / >3000 | 108 / 1468 |
| Methane, solved | 132 / 211 | 152 / 282 | 174 / 364 | 119 / 140 | 224 / 568 | 45 / 141 |
| Hydrogen, solved | 129 / 186 | 149 / 241 | 169 / 303 | 118 / 133 | 215 / 456 | 44 / 114 |

The growth charge roughly triples the solved chambers' cheap-seed break-even
($44-45 to $129-132). **It is the largest change to `tab:seed_return`.**

### Sensitivity: one line at a time from the Estimate (steady $/kg; BE steady)

| Case | Methalox | Methane, solved | Hydrogen, solved |
|---|---|---|---|
| Estimate | 163; 346 / 1694 | 98; 132 / 211 | 98; 129 / 186 |
| Lob $5/kg | 74; 209 / 1555 | 45; 74 / 151 | 47; 76 / 130 |
| Lob $10/kg | 96; 243 / 1590 | 58; 89 / 166 | 60; 90 / 144 |
| Lob $15/kg | 119; 278 / 1624 | 72; 103 / 181 | 73; 103 / 158 |
| Lob $50/kg | 270; 517 / 1868 | 161; 202 / 285 | 158; 195 / 253 |
| Argon $5/kg | 169; 357 / 1705 | 102; 135 / 215 | 101; 132 / 189 |
| Package $300 | 164; 350 / 1698 | 107; 145 / 224 | 106; 141 / 197 |
| Package $1000 | 170; 364 / 1712 | 130; 182 / 259 | 126; 173 / 228 |
| Package $3000 | 185; 405 / 1752 | 196; 283 / 359 | 185; 261 / 314 |
| Plate spray $8M | 177; 369 / 1717 | 107; 141 / 221 | 106; 138 / 195 |
| Paper's $20/kg fleet | 166; 355 / 1702 | 99; 132 / 212 | 99; 130 / 187 |

Film, chamber sprayers, cryostats, plugs and pitch each move a solved chamber by
$0-3/kg. The package's learning curve is worth $2/kg; even free packages save
under $3.

### Thresholds

- **Chamber price at which a chamber matches methalox's steady $/kg:** solved
  methane $125M, solved hydrogen $149M, methane at 50% $8.4M, hydrogen at 50%
  never.
- **Package price at which a solved chamber matches methalox:** $2567 (methane),
  $3006 (hydrogen), 26-30 times the at-volume estimate.
- **Plates built by the harvest (about 9.8 yr):** methalox 6, the 50% chambers
  11, solved methane 38, solved hydrogen 50. Plates to fall under $10M
  ($6M floor): 49 at a $20M first unit and 80%, 261 at $30M, 1717 at $50M, 7929
  at $78M. **The user's view that the plate falls below $10M quickly holds only
  if the first plate is near $20M.**

---

## 4. Pushback: flawed reasoning and doubtful prices

In order of how much they matter. **All eight were accepted by the author on
2026-10-01.** Items 2 and 3 are settled as stated. Item 1 is accepted as open
work: the off-by-one is to be measured in the companion before the paper quotes
new growth figures. The rest are changes owed to the paper (§5).

1. **The growth ledger pairs the wrong outbound burn with each push (answer to
   G3).** The cost model's indexing is consistent with the ledger. The unit
   lofted at return `n` is pushed by return `n`'s waves and grows by `G_n`. But
   `growth_ledger.price_cycle_growth(cycle n)` departs that unit with cycle
   `n`'s **own** outbound burn. That payload actually leaves on cycle `n + 1`,
   with cycle `n + 1`'s burn. This is an off-by-one in the companion's growth
   ledger, upstream of every `G_n` the paper quotes (ADRs 0033-0037). It is
   **unmeasured**. 2S and 3S cycles alternate, so the burns differ cycle to
   cycle. **Recommend measuring it before the paper quotes new growth figures.**
2. **The 1500 t launch unit needs a booster 10-20% bigger than Super Heavy.
   Settled: we assume one** (author's judgment, 2026-10-01). `sec:vertical_lob`
   lofts 1250-1430 t with the braking reserve under 4 g at 380 s (1070-1250 t at
   350 s), short of the growth ledger's 1500 t. Lofted mass scales with the
   vehicle at fixed thrust-to-weight and propellant fraction, so a booster 10-20%
   bigger lofts 1375-1716 t braked. A vertical integration without brake or drag
   lofts 1564-1706 t to 400 km at those sizes. At the pessimistic 350 s average,
   only the full 20% reaches 1500 t. A vacuum second stage above 20 km does not
   help: the lob's loss is gravity loss set by liftoff thrust, not Isp. The lob
   stays at $25 per kilogram lofted, so the bigger booster costs proportionally
   more per flight and the cost per kilogram does not move. **The paper should
   say plainly that it assumes a lifting rocket big enough for 1500 t** (draft
   in §5, item 10).
3. **"Proven" after one cycle is a thin proof for a 10% rate. Settled: the late
   rate is quoted as a 10-20% band** (author, 2026-10-01). The parent's own
   `sec:mass_interest` says each cycle returns one measurement of `eta_geom` and
   permits one redesign. **The band does not change the conclusion.** Across
   10-20% the solved chambers break even at $129-174 on the cheap seed and
   $186-364 on the dear one. That clears Starcloud's $500 on both seeds and
   Suncatcher's $200 on the cheap seed. Methalox breaks even at $346-564 and
   $1694 to over $3000. It never clears $200, never clears $500 on the dear
   seed, and on the cheap seed clears $500 only up to a 15% late rate. The 10%
   end stays labeled as the author's assumption.
4. **`tab:delivery_ledger`'s "paper's figures" column charges every delivery
   the 100th plate's price** ($11.4M). That assumes 99 plates are already paid
   for. By the harvest the program has built 6-50 plates. Charging each plate
   at its own count, as this model does, is the consistent reading.
5. **`tab:delivery_ledger` charges PuffSats at $45/kg** (built, plus their own
   lob). A returning PuffSat's cost to the business is not its pad price. It
   is the growth it gives up, which is what the growth charge prices. The
   ledger's $45-56/kg (paper's figures, 500 t) is therefore a delivery-only
   cost. The all-in steady-state figure is **$98/kg for a solved chamber** under
   the Estimate. That is what the data-center comparison should quote.
6. **The $2M chamber "flown five times" in `tab:delivery_ledger`** is a
   recovered chamber whose recovery PuffSats are not charged. The companion
   expends it (ADR 0032). The Estimate's $40-60M first unit is a hypothesis.
   But solved chambers stay ahead of methalox up to $125-149M a chamber, so the
   chamber price does not decide the ranking. **Chamber efficiency does.**
7. **The package price.** $100 for a phone-class board, cameras, gyro, radio,
   cold-gas valves, tank and battery, built by the million, is plausible only
   at volume. Radiation is **not** the objection: the PuffSat's own ice and
   argon shield it, tens of g/cm². The cost risk is acceptance testing at
   million-unit scale. It does not decide anything: solved chambers match
   methalox only near $2600-3000 a package.
8. **Argon supply, not argon price.** $1/kg bulk is plausible. But one launch
   unit sprays 564-590 t, and a fleet of 50 units per cycle sprays about
   30 000 t per cycle, and the fleet keeps growing. Argon is a by-product of air
   separation for oxygen, so its supply follows oxygen demand, not ours. The
   paper should check world argon output against that figure, beside ADR 0033's
   argon choice. This model does not.

### Small items in the published text (checked while reproducing)

- `tab:delivery_ledger`: the paper's-figures total is $45.2-56.2M at the
  unrounded 100th plate ($11.35M) and absorber ($0.68M). That is $90-112/kg at
  500 t, not $91-113. The table sums lines it has already rounded.
- `tab:l1_comparison`: from the unrounded multipliers (3.52-4.56), Morgan
  Stanley at 20.5 kg/kW is 82.0-107.6 cents/kWh, not 107.8. Goldman at 15 kg/kW
  tops out at 28.6-28.8 depending on which end is rounded, not 28.7. The
  Miller floor's top is 53.8, not 53.9.
- The paper's `CONTEXT.md` "Delivery ledger" entry still says the paper's
  column prices the plate at "$5M at 10x steel" and totals $76-98/kg. The tex
  now uses Wright's law ($11M, $91-113). The glossary is stale.

---

## 5. What comes back to the paper

Revised from ASKS.md "What comes back to the paper", with this model's numbers.

**`sec:mass_interest`:**
1. Say that the growth phase is charged: every launch unit's lob, plate and
   spray, departure hardware, tanks, propellant, cryostats, consumables and
   fleet manufacture, and the seed's own manufacture.
2. Replace `tab:seed_return`'s IRR columns with stepped-rate break-even and
   value per seed dollar (tables in §3). Keep flat 7.6% and 30% as a comparison
   row.
3. State the stepped rate and its reason. Quote the late rate as a 10-20% band (§4,
   item 3).
4. Near-term choice: methalox breaks even at $346 / $1694; the solved chambers
   at $129-132 / $186-211. A half-ceiling chamber is no better than methalox.
5. Steady-state overhead goes as `C_g / (G - 1)`: methalox, at G near 2,
   carries about one reinvested launch unit per delivered one.

**`sec:heat_shield_bill`:**
6. Quote the all-in steady-state cost ($98/kg for a solved chamber, Estimate;
   $58-98 across the $10-25 lob band) beside the delivery-only ledger.
7. Replace the $2M chamber with the H2 hypothesis, and the $50M-per-100 t
   plate with H1 plus the spray system. Label both as hypotheses.
8. Fleet price: $20/kg holds for growth PuffSats. Rods cost about $124/kg at a
   $100 package, and the fleet line is $1-3 per delivered kilogram.
9. Fix the rounding items in §4.

**Elsewhere:**
10. `sec:vertical_lob`: state the assumption (§4, item 2). Draft: "The growth
    ledger's launch unit is 1500 t. Super Heavy as flown lofts 1250 t to 1430 t
    with its braking reserve, so we assume a lifting rocket 10 to 20 percent
    larger, big enough to loft 1500 t to the 400 km intercept. At the same
    thrust-to-weight and propellant fraction, lofted mass grows in proportion
    to the vehicle. We price the lob per kilogram lofted, so the larger booster
    costs proportionally more per flight and the cost per kilogram is unchanged."
11. Argon supply beside ADR 0033's argon choice (§4, item 8).
12. `sec:batch_filled_steel`: extend its cost hypothesis to the ledger's
    chamber walls, which are mostly IM7 carbon, not steel.

**Citations still owed:** the 10% late rate, the argon price and supply, and
any analogue for the chamber, plate, sprayers, cryostats and package. Until
then each is a labeled hypothesis.
