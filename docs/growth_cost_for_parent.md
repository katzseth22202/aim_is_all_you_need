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
window's burn, not its own cycle's). §4 item 1 lists what that moved, including
tables the parent has already published.

**Bottom line.** With every growth cost charged, a *solved* pulsed chamber meets
the cost targets and methalox probably does not. Under the Estimate prices, a
solved chamber delivers to L1 for $100/kg in steady state and repays the seed at
$132-178/kg on the cheap seed and $206-427 on the dear one, across a late rate
of 10-20%. That is under Starcloud's $500 cap on both seeds and under
Suncatcher's $200 on the cheap one; on the dear seed it sits just above $200.
Methalox's steady-state cost is $169/kg, under both bars, but it grows too
slowly to repay its seed in time. It needs $360-574/kg on the cheap seed and
$1919 to over $3000 on the dear one. A chamber at half its
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
| Cost of capital | **30% until the first growth return, then 10-20%** | 30%: Gompers et al. 10%: the author's judgment, **unsourced**; the band per §4 item 3 |
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
| Methalox | 169 | 115 | 46 | 4 | 1 | 3 | 360 / 1919 | 388 / >3000 | 3.3 / 0.1 | -1.6 / -0.1 |
| Methane, 50% | 184 | 110 | 42 | 25 | 4 | 4 | 373 / 1306 | 379 / 2022 | 4.3 / 0.2 | -3.4 / -0.1 |
| **Methane, solved** | **100** | 69 | 22 | 7 | 1 | 2 | **135 / 240** | 157 / 452 | 88 / 3.5 | 16 / 0.6 |
| Hydrogen, 50% | 228 | 117 | 45 | 41 | 4 | 20 | 490 / 1657 | 481 / 2422 | 1.2 / 0.0 | -4.9 / -0.2 |
| **Hydrogen, solved** | **100** | 66 | 20 | 8 | 1 | 5 | **132 / 206** | 153 / 371 | 127 / 5.0 | 23 / 0.9 |

"Other" is argon, tanks, propellant, cryostats, plugs, pitch and film.

### The other price books

| Design | Paper's: steady $/kg | Paper's: BE steady | Pessimistic: steady $/kg | Pessimistic: BE steady |
|---|---|---|---|---|
| Methalox | 194 | 501 / 2061 | 506 | 855 / 2421 |
| Methane, 50% | 184 | 398 / 1332 | 910 | 1400 / 2333 |
| Methane, solved | 99 | 144 / 250 | 438 | 555 / 659 |
| Hydrogen, 50% | 215 | 475 / 1643 | 1035 | 1625 / 2793 |
| Hydrogen, solved | 95 | 132 / 207 | 410 | 509 / 582 |

With every line at its worst at once, no design repays the seed under $500.

### By cost of capital (Estimate, steady break-even)

| Design | 30%->10% | 30%->15% | 30%->20% | Flat 7.6% | Flat 30% | 30%->10%, growth uncharged |
|---|---|---|---|---|---|---|
| Methalox | 360 / 1919 | 463 / 2903 | 574 / >3000 | 255 / 594 | 831 / >3000 | 117 / 1688 |
| Methane, solved | 135 / 240 | 156 / 327 | 178 / 427 | 121 / 150 | 231 / 679 | 47 / 169 |
| Hydrogen, solved | 132 / 206 | 151 / 271 | 172 / 346 | 120 / 140 | 220 / 533 | 45 / 134 |

The growth charge roughly triples the solved chambers' cheap-seed break-even
($45-47 to $132-135). **It is the largest change to `tab:seed_return`.**

### Sensitivity: one line at a time from the Estimate (steady $/kg; BE steady)

| Case | Methalox | Methane, solved | Hydrogen, solved |
|---|---|---|---|
| Estimate | 169; 360 / 1919 | 100; 135 / 240 | 100; 132 / 206 |
| Lob $5/kg | 77; 222 / 1779 | 46; 77 / 179 | 48; 79 / 150 |
| Lob $10/kg | 101; 256 / 1814 | 60; 92 / 195 | 62; 92 / 164 |
| Lob $15/kg | 123; 291 / 1849 | 74; 106 / 210 | 75; 106 / 178 |
| Lob $50/kg | 280; 533 / 2094 | 165; 206 / 314 | 161; 197 / 274 |
| Argon $5/kg | 175; 370 / 1930 | 104; 138 / 244 | 104; 135 / 210 |
| Package $300 | 170; 364 / 1923 | 110; 148 / 252 | 108; 143 / 217 |
| Package $1000 | 176; 378 / 1937 | 133; 185 / 288 | 129; 175 / 247 |
| Package $3000 | 192; 419 / 1978 | 201; 286 / 388 | 189; 263 / 334 |
| Plate spray $8M | 184; 383 / 1942 | 109; 144 / 250 | 109; 141 / 215 |
| Paper's $20/kg fleet | 172; 368 / 1927 | 102; 135 / 241 | 101; 133 / 207 |

Film, chamber sprayers, cryostats, plugs and pitch each move a solved chamber by
$0-4/kg. The package's learning curve is worth $2-3/kg; even free packages save
only about $1 more.

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
   no longer clears $200 on the dear seed ($206, was $186).
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
   10-20% the solved chambers break even at $132-178 on the cheap seed and
   $206-427 on the dear one. That clears Starcloud's $500 on both seeds and
   Suncatcher's $200 on the cheap seed. Methalox breaks even at $360-574 and
   $1919 to over $3000. It never clears $200, never clears $500 on the dear
   seed, and on the cheap seed clears $500 only up to a 15% late rate. The 10%
   end stays labeled as the author's assumption.
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
2. Replace `tab:seed_return`'s IRR columns with stepped-rate break-even and
   value per seed dollar (tables in §3). Keep flat 7.6% and 30% as a comparison
   row.
3. State the stepped rate and its reason. Quote the late rate as a 10-20% band (§4,
   item 3).
4. Near-term choice: methalox breaks even at $360 / $1919; the solved chambers
   at $132-135 / $206-240. A half-ceiling chamber is no better than methalox.
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

**Citations still owed:** the 10% late rate, the argon price and supply, and
any analogue for the chamber, plate, sprayers, cryostats and package. Until
then each is a labeled hypothesis.

---

## 6. Draft owed: cheaper seeds if launch stays dear (for `sec:mass_interest`)

Requested by the author on 2026-10-01: a short section on how to cut the seed's
cost if the bank flight prices hold. The rule in the first paragraph and the
waiting arithmetic are exact. **The route paragraphs are placeholders** until
the companion's seed-route analysis (gravity assists plus SEP, queued after ADR
0038) lands. An earlier version of this draft claimed the Venus route sends
2.9 times the mass. That used ADR 0008's phasing-free trajectories. ADR 0007's
real-ephemeris search shows chemical phasing costs about 3.6 km/s, which erases
the gain, so the claim is withdrawn.

> **Draft text.**
>
> The seed is paid once, so it can be bought cheaper by spending time. A choice
> that makes the seed $k$ times cheaper per kilogram but delays the first return
> by $\Delta t$ years repays itself when $k(1+r)^{-\Delta t} > 1$, with $r$ the
> rate charged before the cycle is proven. The program's later costs and revenues
> slide together, so the test does not depend on what the fleet is worth. At 30%,
> a two-year delay has to buy a seed 1.7 times cheaper and a three-year delay
> one 2.2 times cheaper.
>
> Waiting for cheaper launches fails this test. Under the bank prices the twelve
> tankers are 85 to 92% of the seed's bill, so the seed's price follows the flight
> price. Morgan Stanley's \$500 per kilogram in 2030 becomes less than \$150 by
> 2040 \cite{investing2026_ms_spacex}. That is 3.3 times cheaper for ten years of
> waiting, against the 13.8 times that ten years at 30% demands.
>
> A slower route can pass only with help. Venus and Earth gravity assists
> lower the burn the seed ship makes from low orbit, but with the planets
> where they really are, the Venus--Earth--Jupiter sequence needs about
> \SI{3.6}{\kilo\meter\per\second} of maneuvers between flybys. Chemically, that
> brings its total to \SI{4.57}{\kilo\meter\per\second}, no better than the
> direct route \cite{Katz_aim_is_all_you_need_2025}. The route pays only if
> something cheaper than methalox flies those maneuvers.
>
> Solar-electric propulsion can do part of the work instead, and the Venus
> route needs some of it to phase its encounters. It is charged twice. The
> array, power processing, thrusters and argon tankage ride with the seed and
> displace PuffSats, and they are bought. Pushing a stack of mass $m$ through
> $\Delta v$ in time $t$ takes a jet power of roughly
> $m\,\Delta v\,v_e/(2\eta t)$. At a \SI{2000}{\second} argon exhaust and 50%
> thruster efficiency, \SI{3}{\kilo\meter\per\second} on \SI{200}{\tonne} over
> two years takes about \SI{190}{\kilo\watt}. At \SIrange{10}{20}{\kilo\gram} per
> kilowatt that is \SIrange{2}{4}{\tonne} of hardware, plus about \SI{28}{\tonne}
> of argon and \SI{4}{\tonne} of tankage. At a hypothetical \$50 to \$1000 per
> watt, the hardware costs \$10 million to \$190 million. That is small against a
> dear seed ship, about \$670 million (thirteen launches and the hull), and
> large against a cheap one, about \$31 million, so electric propulsion is a dear-seed option. Heliocentric
> delta-v also earns no Oberth effect, so a kilometer per second of it replaces
> less than a kilometer per second of the departure burn.
>
> A smaller first launch unit cuts the capital at risk but not the price. The
> seed is one launch unit's first-cycle consumption, and every later cost and
> delivery scales with it. Halving the unit halves the seed and the fleet
> together, and leaves every break-even price where it is. It costs one doubling
> time to catch up.

**Electric propulsion assumptions** (author's request, 2026-10-01: charge its
mass and its cost). The mass figures are ADR 0026's: argon at 2000 s, thruster
efficiency 0.5, 10/15/20 kg per kW at 1 AU for array, PPU, thrusters and gimbals,
argon tankage 0.15 kg/kg, power falling as 1/r^2. The **cost of $50/$200/$1000
per watt at 1 AU** is a new, unsourced hypothesis. It spans a mass-produced
Starlink-class array to a science-mission system, and the companion analysis
sweeps it. ADR 0026 found argon SEP cannot pay for the *growth wave's* split
corrections. That verdict was on a recurring cost per cycle; a one-time seed is a
different test, which is why it is back here.

Not in the draft, deliberately: lunar-sourced tanker propellant (`sec:isru`).
The tankers are most of the bill, so it would matter, but nothing in either repo
prices it.

**Queued companion work (after ADR 0038):** a seed-route option in
`src/seed_cost.py`. It prices each route's actual burns from low orbit (direct,
E-V-E-J, VVEJGA, VEEGA, each with and without SEP), carries the SEP hardware's
mass and dollars as above, takes each route's return speed, re-flies the chain
from the later start, and reruns the cheap- and dear-seed break-evens with the
proof date moved. A 15-year harvest is swept separately.
