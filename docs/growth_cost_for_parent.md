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

**Every figure below includes ADR 0038's fix** (each payload departs on the next
window's burn, not its own cycle's) **and ADR 0040's valuation** (10% a year, a
50% chance the cycle works, settled at the first growth return), which replaced
ADR 0037's 30%-then-10% rate on 2026-10-03. The seed flies an Earth gravity
assist (ADR 0039). §4 item 1 lists what that moved, including
tables the parent has already published.

**Bottom line.** With every growth cost charged and the program valued as an
expected value (10% a year, 50% odds the cycle works), a *solved* pulsed chamber
meets the cost targets and methalox does not. Under the Estimate prices, a
solved chamber delivers to L1 for $100/kg in steady state and repays the seed at
$134-137/kg on the cheap seed and $194-222 on the dear one, flying the seed
direct. On the Earth gravity assist the seed now takes (ADR 0039), the dear seed
falls to **$179** (solved hydrogen) and **$201** (solved methane). That is under
Starcloud's $500 on both seeds and under Suncatcher's $200 on the cheap seed and,
for solved hydrogen, on the dear one. Methalox's steady-state cost is $169/kg,
under both bars, but it grows too slowly to repay its seed: $352/kg on the
cheap seed and $1129-1396 on the dear one. A chamber at half its ceiling does no
better than methalox, so the target is met by chamber *efficiency*, not by having
a chamber. At 25% odds the dear-seed figures rise sharply (solved hydrogen $262
direct); with every price at its worst at once (the Pessimistic book), nothing
repays the dear seed under $500.

---

## 1. Verdict on the scratch model

**The numbers match.** Every scratch figure is reproduced to within a few
dollars per kilogram, and the scratch model's conclusions stand. The concrete
implementation differs from it in five places. All are corrections or
refinements, and none changes a ranking:

| Change | Effect |
|---|---|
| Wright's law integrated at the unit midpoint, not from zero. Scratch charged about one extra first plate or chamber per program (+47% on unit 1 at 80%). | Steady break-evens fall $2-16/kg before ADR 0038 (methalox 359 / 1710 to 346 / 1694; solved hydrogen 133 / 188 to 129 / 186) |
| Each delivery picks its plate schedule at the prices it actually pays (plate, spray, argon, film). Scratch used the schedules optimised at $200/$500 with a $114/kg plate. | Pessimistic $/kg falls 3-4% (more cargo per lob when the plate is dear) |
| Rods sent onward split by the *next* cycle's consumption; pulses read per cycle, not scaled from cycle 0 | Under $0.5/kg |
| Methalox departure propellant charged at $0.2/kg (scratch: free) | +$0.2/kg |
| Hydrogen and methane charged on gas *launched*, including boil-off | Under $0.5/kg |
| **ADR 0038: each payload departs on the next window's burn** (a companion bug, upstream of the scratch model too) | Steady $/kg +$2-6; break-evens +$3-14 cheap seed, +$20-225 dear seed (§4 item 1) |

**Validation (G1):** with the growth charge off and the plate flat at $114/kg,
the code returned `tab:seed_return` exactly before ADR 0038: methalox 45 / 3
and 39 / 10, solved hydrogen 87 / 33 and 84 / 36 (IRR %, liquidation and steady,
cheap / dear seed, $500). With ADR 0038 the same calculation gives methalox
42 / 2 and 38 / 9, solved hydrogen 81 / 29 and 80 / 33. **Those are the values
`tab:seed_return` should now print.** Both are pinned tests.

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
| Pulses per departure | 4410-8426, per cycle | `DepartureLedger.pulses` |
| Seed ship, seed price | stripped expended Starship, $337 / $9293 per kg | ADR 0035/0036 |
| Delivery | to the L1 transfer (10.82 km/s from 400 km), 30 m/s halo insertion in methalox | ADR 0036 |

Per launch unit, cycle 0:

| Design | Departure units | Tanks (t) | Propellant (t) | Argon (t) | Sent onward (t) | Consumed (t) | G_0 |
|---|---|---|---|---|---|---|---|
| Methalox | 2 Raptor 3 | 10.3 | 572 | 581 | 121 | 72.7 | 1.66 |
| Methane, 50% | 1 chamber | 15.9 | 468 | 587 | 113 | 85.7 | 1.31 |
| Methane, solved | 1 | 13.5 | 398 | 580 | 205 | 83.1 | 2.46 |
| Hydrogen, 50% | 1 | 84.4 | 438 | 576 | 106 | 86.7 | 1.22 |
| Hydrogen, solved | 1 | 67.4 | 350 | 567 | 230 | 83.3 | 2.76 |

Cycle 0 is a 3S return feeding the 2S cycle 1, so its unit flies the dearer
7.17 km/s departure. That is why `G_0` is the chain's weakest step.

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
| Valuation (ADR 0040) | **10% a year; 50% chance the cycle works, settled at the first growth return; flows before it certain** | Both the author's judgment (2026-10-03), **unsourced**; 10% is about a risky bond's yield. Reported also at 25% and 100% odds |
| Old rate (comparison) | 30% until the first growth return, then 10-20% (ADR 0037) | 30%: Gompers et al. |
| First growth return | 5.46 yr (chambers), 6.55 yr (methalox) | `times[1]` |
| Sale price at L1 | $500 cap (Starcloud), $200 (Suncatcher) | parent `sec:heat_shield_bill` |
| Horizon | 300 yr; steady-state $/kg measured on chain lap 3 after the harvest | modelling choice |

---

## 3. Results

### Headline: Estimate prices (ADR 0040: 10%, 50% odds)

"Steady $/kg" is the undiscounted cost per kilogram at L1, a lap into the
steady state, seed excluded. Break-even (BE) is the sale price that repays the
seed and every growth cost. Value per seed dollar is the expected present value
of everything after the seed over the seed's cost. Each pair is cheap / dear
seed, flown direct.

| Design | Steady $/kg | Lob | Plate + spray | Departure hw + spray | Fleet | Other | BE steady | BE liquidation | Value per seed $ at $500 | at $200 |
|---|---|---|---|---|---|---|---|---|---|---|
| Methalox | 169 | 115 | 46 | 4 | 1 | 3 | 352 / 1396 | 374 / 2168 | 4.7 / 0.2 | -2.7 / -0.1 |
| Methane, 50% | 184 | 110 | 42 | 25 | 4 | 4 | 388 / 1138 | 406 / 1726 | 4.6 / 0.2 | -5.0 / -0.2 |
| **Methane, solved** | **100** | 69 | 22 | 7 | 1 | 2 | **137 / 222** | 162 / 400 | 109 / 4.3 | 19 / 0.7 |
| Hydrogen, 50% | 228 | 117 | 45 | 41 | 4 | 20 | 516 / 1454 | 524 / 2084 | 0.6 / 0.0 | -7.0 / -0.3 |
| **Hydrogen, solved** | **100** | 66 | 20 | 8 | 1 | 5 | **134 / 194** | 159 / 334 | 158 / 6.2 | 28 / 1.1 |

"Other" is argon, tanks, propellant, cryostats, plugs, pitch and film. The
steady $/kg columns do not depend on the valuation.

### With the seed on the Earth gravity assist (ADR 0039; `make growth-cost` §9)

Steady break-even, Estimate, 50% odds. The route makes the seed 1.43x (cheap)
and 1.64x (dear) cheaper per kilogram and returns 2.18 years later.

| Design | Direct, cheap / dear | EEJ, cheap | EEJ, dear | EVEEJ, cheap | EVEEJ, dear |
|---|---|---|---|---|---|
| Methalox | 352 / 1396 | 346 | **1129** | 346 | 1357 |
| Methane, solved | 137 / 222 | 136 | **201** | 136 | 219 |
| Hydrogen, solved | 134 / 194 | 134 | **179** | 134 | 192 |

### The other price books (ADR 0040, seed direct)

| Design | Paper's: steady $/kg | Paper's: BE steady | Pessimistic: steady $/kg | Pessimistic: BE steady |
|---|---|---|---|---|
| Methalox | 194 | 500 / 1545 | 506 | 861 / 1911 |
| Methane, 50% | 184 | 416 / 1166 | 910 | 1466 / 2216 |
| Methane, solved | 99 | 146 / 232 | 438 | 563 / 647 |
| Hydrogen, 50% | 215 | 499 / 1438 | 1035 | 1712 / 2650 |
| Hydrogen, solved | 95 | 134 / 195 | 410 | 515 / 574 |

With every line at its worst at once, nothing repays the dear seed under $500.

### By valuation (Estimate, steady break-even, seed direct)

| Design | **10%, 50% odds** | 10%, 25% | 10%, 100% | 30%->10% (ADR 0037) | Flat 7.6% | Flat 30% | 10%, 50%, growth uncharged |
|---|---|---|---|---|---|---|---|
| Methalox | **352 / 1396** | 479 / 2564 | 288 / 812 | 360 / 1919 | 255 / 594 | 831 / >3000 | 94 / 1154 |
| Methane, solved | **137 / 222** | 151 / 317 | 130 / 173 | 135 / 240 | 121 / 150 | 231 / 679 | 46 / 147 |
| Hydrogen, solved | **134 / 194** | 145 / 262 | 128 / 159 | 132 / 206 | 120 / 140 | 220 / 533 | 44 / 118 |

The growth charge roughly triples the solved chambers' cheap-seed break-even
($44-46 to $134-137). **It is the largest change to `tab:seed_return`.** The odds
matter most on the dear seed.

### Sensitivity: one line at a time from the Estimate (steady $/kg; BE steady, seed direct)

| Case | Methalox | Methane, solved | Hydrogen, solved |
|---|---|---|---|
| Estimate | 169; 352 / 1396 | 100; 137 / 222 | 100; 134 / 194 |
| Lob $5/kg | 77; 209 / 1251 | 46; 79 / 161 | 48; 80 / 137 |
| Lob $10/kg | 101; 245 / 1288 | 60; 93 / 176 | 62; 94 / 152 |
| Lob $15/kg | 123; 280 / 1324 | 74; 108 / 191 | 75; 107 / 166 |
| Lob $50/kg | 280; 530 / 1577 | 165; 209 / 296 | 161; 200 / 262 |
| Argon $5/kg | 175; 363 / 1408 | 104; 140 / 226 | 104; 137 / 197 |
| Package $300 | 170; 356 / 1400 | 110; 150 / 234 | 108; 145 / 204 |
| Package $1000 | 176; 369 / 1413 | 133; 187 / 270 | 129; 177 / 235 |
| Package $3000 | 192; 408 / 1452 | 201; 289 / 370 | 189; 266 / 323 |
| Plate spray $8M | 184; 376 / 1421 | 109; 146 / 232 | 109; 143 / 203 |
| Paper's $20/kg fleet | 172; 360 / 1404 | 102; 137 / 222 | 101; 135 / 194 |

Film, chamber sprayers, cryostats, plugs and pitch each move a solved chamber by
$0-5/kg. The package's learning curve is worth $2-3/kg; even free packages save
only about $3 more.

### Thresholds

- **Chamber price at which a chamber matches methalox's steady $/kg:** solved
  methane $126M, solved hydrogen $151M, methane at 50% $3.6M, hydrogen at 50%
  never.
- **Package price at which a solved chamber matches methalox:** $2653 (methane),
  $3109 (hydrogen), 27-31 times the at-volume estimate.
- **Plates built by the harvest (about 9.8 yr):** methalox 5, the 50% chambers
  7, solved methane 29, solved hydrogen 38. Plates to fall under $10M
  ($6M floor): 49 at a $20M first unit and 80%, 261 at $30M, 1717 at $50M, 7929
  at $78M. **The user's view that the plate falls below $10M quickly holds only
  if the first plate is near $20M.**

---

## 4. Pushback: flawed reasoning and doubtful prices

In order of how much they matter. **All eight were accepted by the author on
2026-10-01.** Items 2 and 3 are settled as stated. Item 1 is accepted as open
work: the off-by-one is to be measured in the companion before the paper quotes
new growth figures. The rest are changes owed to the paper (§5).

1. **The growth ledger paired the wrong outbound burn with each push (answer to
   G3). Fixed: ADR 0038.** The waves of return `n` push a payload that leaves on
   cycle `n + 1`, but every ledger departed it on cycle `n`'s own burn. A 3S
   outbound needs about 5.3 km/s and a 2S one about 7.0, so cycle 0 (a 3S
   return feeding the 2S cycle 1) was undercharged by 1.84 km/s. The fix moves
   **published** figures:
   - `tab:mass_interest_growth`: the f = 0.8 column becomes 37 / 3290 / 4.76e4 /
     2.82e5 for `eta_geom` 0.6-0.9 (was 61 / 4674 / 6.29e4 / 3.55e5).
     **`eta_geom` = 0.60 at f = 0.5 falls to 0.95, so it now loses mass** (was
     1.5). Full grid in ADR 0038.
   - `tab:seed_amortization`: solved hydrogen 54.8% a year and 75 in ten years
     (was 57%, 104); solved methane 48.8%, 49.7 (was 51%, 69.5); methalox 18.5%,
     5.07 (was 19%, 5.1). Seeds 83.1-83.3 t for the chambers (was 80.7-80.8).
   - `tab:seed_return`: see §1.
   - The plate column at f = 0.818, ADR 0013's rate (0.3989 to 0.3900 /yr) and
     ADR 0015's matched-recovery table, whose verdict holds and sharpens (11x at
     e = 0.6, was 8.5x).
   **The headline conclusion survives.** One claim weakens: solved hydrogen
   no longer clears $200 on the dear seed ($206, was $186) under
   ADR 0037's rate. ADR 0040's valuation restores it ($194 direct, $179 on the
   Earth gravity assist).
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
3. **"Proven" after one cycle is a thin proof for a 10% rate. Superseded by
   ADR 0040** (author, 2026-10-03). The 30%-then-10% rate charged venture risk
   again for every year of waiting, which decided the seed-route question by
   assumption. The program is now valued as an expected value: 10% a year and a
   50% chance the cycle works, settled once at the first growth return. The
   odds, not a rate band, carry the doubt, and the paper should print 25% and
   100% beside 50% (§3, "By valuation").
4. **`tab:delivery_ledger`'s "paper's figures" column charges every delivery
   the 100th plate's price** ($11.4M). That assumes 99 plates are already paid
   for. By the harvest the program has built 5-38 plates. Charging each plate
   at its own count, as this model does, is the consistent reading.
5. **`tab:delivery_ledger` charges PuffSats at $45/kg** (built, plus their own
   lob). A returning PuffSat's cost to the business is not its pad price. It
   is the growth it gives up, which is what the growth charge prices. The
   ledger's $45-56/kg (paper's figures, 500 t) is therefore a delivery-only
   cost. The all-in steady-state figure is **$100/kg for a solved chamber** under
   the Estimate. That is what the data-center comparison should quote.
6. **The $2M chamber "flown five times" in `tab:delivery_ledger`** is a
   recovered chamber whose recovery PuffSats are not charged. The companion
   expends it (ADR 0032). The Estimate's $40-60M first unit is a hypothesis.
   But solved chambers stay ahead of methalox up to $126-151M a chamber, so the
   chamber price does not decide the ranking. **Chamber efficiency does.**
7. **The package price.** $100 for a phone-class board, cameras, gyro, radio,
   cold-gas valves, tank and battery, built by the million, is plausible only
   at volume. Radiation is **not** the objection: the PuffSat's own ice and
   argon shield it, tens of g/cm². The cost risk is acceptance testing at
   million-unit scale. It does not decide anything: solved chambers match
   methalox only near $2650-3100 a package.
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
2. Replace `tab:seed_return`'s IRR columns with ADR 0040's break-even and
   expected value per seed dollar (tables in §3). Keep ADR 0037's stepped rate
   and the flat 7.6% and 30% as comparison rows.
3. State the valuation and its reason: time at 10% a year (about a risky
   bond's yield), and a 50% chance the cycle works, settled at the first growth
   return, applied to every flow from then on; money spent before it is spent
   either way. A venture rate compounded each year would charge the same risk
   again for every year of waiting. Print the 25% and 100% rows; label both
   numbers as the author's assumptions.
4. Near-term choice: methalox breaks even at $352 / $1396 (seed direct); the
   solved chambers at $134-137 / $194-222, and $179-201 on the dear seed with
   the Earth gravity assist. A half-ceiling chamber is no better than methalox.
5. Steady-state overhead goes as `C_g / (G - 1)`: methalox, at G near 2,
   carries about one reinvested launch unit per delivered one.

**`sec:heat_shield_bill`:**
6. Quote the all-in steady-state cost ($100/kg for a solved chamber, Estimate;
   $60-100 across the $10-25 lob band) beside the delivery-only ledger.
7. Replace the $2M chamber with the H2 hypothesis, and the $50M-per-100 t
   plate with H1 plus the spray system. Label both as hypotheses.
8. Fleet price: $20/kg holds for growth PuffSats. Rods cost about $124/kg at a
   $100 package, and the fleet line is about $1 per delivered kilogram.
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

**ADR 0038 (published tables):**
13. Replace `tab:mass_interest_growth`, `tab:seed_amortization` and
    `tab:seed_return` with ADR 0038's figures, and re-quote every growth figure
    drawn from ADRs 0013, 0015 and 0032-0037 (§4 item 1). State that a 3S return
    feeding a 2S departure flies the 2S burn.

**A path that does not depend on Starship** (author, 2026-10-02; for
`sec:mass_interest`, beside the seed's cost):
14. Every seed price above assumes an expended Starship refuelled in low
    orbit. Say what happens without it, whether through competition,
    technical setbacks or access (author, 2026-10-03: the strongest reason to
    plan for it). The direct route is demanding: the seed leaves Earth at
    11.3 km/s of excess speed, because it must come back from Jupiter at the
    cycle's collision speed. Conventional expendable launchers deliver little
    useful mass at that speed. Chemical gravity assists cut it: an Earth loop
    leaves at 6.8 km/s, Venus routes at 2.8-4.4 km/s. That is why Galileo
    (Venus-Earth-Earth) and Juno (an Earth loop) flew them. Without a
    refuelled Starship, the comparison is not seed per dollar against a cheap
    direct route but flying with assists or not flying, so the plan is a few
    conventional launches on gravity-assist routes. Make the business case
    plainly: a program that can only start on one provider's refuelled
    vehicle depends on that provider, and this is the hedge. These routes are
    flight-proven and the companion models them correctly (ADR 0039). SEP
    could add to them but is not shown to fly as modelled; leave it as a
    possible addition, not part of the case. Not priced: no launcher other
    than Starship is in the companion.

**Citations still owed:** the 10% late rate, the argon price and supply, and
any analogue for the chamber, plate, sprayers, cryostats and package. Until
then each is a labeled hypothesis.

---

## 6. Draft owed: cheaper seeds if launch stays dear (for `sec:mass_interest`)

Requested by the author on 2026-10-01: a short section on how to cut the seed's
cost if the bank flight prices hold. It now carries the companion's results
(ADR 0039 for the routes, ADR 0040 for the valuation, 2026-10-03): **fly the
seed on an Earth gravity assist.** An earlier version claimed the Venus route
sends 2.9 times the mass; that used ADR 0008's phasing-free trajectories and is
withdrawn. Solar-electric propulsion is out of scope (ADR 0039).

> **Draft text.**
>
> The seed is paid once, so it can be bought cheaper by spending time. A choice
> that makes the seed $k$ times cheaper per kilogram but delays the first return
> by $\Delta t$ years repays itself when $k(1+r)^{-\Delta t} > 1$. Because the
> chance that the cycle works is the same whichever way the seed travels, it
> cancels, and $r$ is the ordinary time value of money, 10\% a year. The program's
> later costs and revenues slide together, so the test does not depend on what
> the fleet is worth. A two-year delay has to buy a seed 1.21 times cheaper, a
> four-year delay 1.46 times.
>
> A gravity assist passes. Venus and Earth flybys lower the burn the seed ship
> makes from low orbit, from an excess speed of 11.3 km/s for the direct route
> to 6.8 km/s for one Earth loop. With the planets where they really are, the
> maneuvers between flybys still cost something, but the Earth loop sends 1.64
> times the seed of the direct route on the dear seed ship (1.43 times on the
> cheap one) and returns 2.2 years later. That delay needs 1.23 times, so the
> route is worth 1.33 times the direct seed per dollar (1.16 times on the cheap
> ship) \cite{Katz_aim_is_all_you_need_2025}. A Venus--Earth--Earth route, the
> one Galileo flew, does about as well on the cheap ship (1.19 times) and barely
> pays on the dear one (1.04 times); routes with burns of several km/s between
> flybys lose. On the dear seed the Earth loop lowers solved hydrogen's
> break-even from \$194 to \$179 per kilogram, and solved methane's from \$222
> to \$201.
>
> Waiting for cheaper launches passes the same test, narrowly. Under the bank
> prices the twelve tankers are 85 to 92\% of the seed's bill, so the seed's
> price follows the flight price. Morgan Stanley's \$500 per kilogram in 2030
> becomes less than \$150 by 2040 \cite{investing2026_ms_spacex}, 3.3 times
> cheaper for ten years of waiting against the 2.6 times that ten years at 10\%
> demands. The case against a decade's wait is not the discount rate but what
> the rate leaves out: competitors, a team carried for ten years, and the chance
> the opportunity closes.
>
> A smaller first launch unit cuts the capital at risk but not the price. The
> seed is one launch unit's first-cycle consumption, and every later cost and
> delivery scales with it. Halving the unit halves the seed and the fleet
> together, and leaves every break-even price where it is. It costs one doubling
> time to catch up.

Not in the draft, deliberately: lunar-sourced tanker propellant (`sec:isru`).
The tankers are most of the bill, so it would matter, but nothing in either repo
prices it.

**Companion result (ADR 0039).** `src/seed_route.py` searched direct, EEJ, EVEJ,
EVVEJ and EVEEJ with methalox burns between flybys on both seed ships, each
route scored on seed per dollar with the delay charged at 10% a year (the odds
cancel). Seed per dollar against direct:

| Route | $670M ship | $31M ship |
|---|---|---|
| **EEJ** (one Earth loop) | **1.33x** | 1.16x |
| EVEEJ (Galileo's) | 1.04x | **1.19x** |
| EVEJ | 0.60x | 0.51x |
| EVVEJ | 0.59x | 0.51x |

At ADR 0037's 30% a year the same routes all lost (EEJ 0.92x and 0.81x): the
old rate, not the trajectory, had decided the question.

**SEP, out of scope (ADR 0039).** An impulsive model credited SEP routes with
wins of up to 1.5x on the dear seed, but it treats a leg's thrust as one burn at
the next flyby. Flown with continuous thrust (pykep's zero-order-hold legs,
checked against an independent integration), EEJ at 2 W/kg needed twice its
array's thrust and EEJ at 4 W/kg could not fly once its thrust fell with
distance from the Sun. Deciding SEP needs a trajectory designed for low thrust
from end to end, which this analysis does not build.

---

## 7. Status (2026-10-03)

**Answered in the companion:**
- Asks G1-G4: ADR 0037, with ADR 0038 fixing the off-by-one that G3 exposed.
- All ten requested outputs: `make growth-cost`.
- All eight pushback items in §4: accepted by the author on 2026-10-01.
- Validation against `tab:seed_return`: §1.
- The valuation: ADR 0040 (10% a year, 50% odds the cycle works).
- Seed routes: ADR 0039 (fly an Earth gravity assist; SEP out of scope).

**Open:**
1. **Nothing is written into the parent's `.tex`.** §5 lists 14 changes.
2. **Citations owed:** the 10% time value and the 50% odds (both the author's
   assumptions), the argon price, and analogues for the chamber, plate,
   sprayers, cryostats and package. Until then, each is a labeled hypothesis.
3. **Argon supply** against about 30 000 t per cycle (§4 item 8). This is for
   the paper to check; the companion does not model it.
4. **A path that does not depend on Starship** (§5 item 14): a few
   conventional launches on chemical gravity-assist routes. A direction for the
   paper; not priced, since the companion has no launcher other than Starship.
5. **Not priced, by decision** (ADR 0037): plate and chamber recovery, a
   falling lob or sale price over time, demand capping the fleet, a seed
   larger than one launch unit, and the trade between rod size and package
   count (G4).
