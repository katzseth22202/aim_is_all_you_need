# The spray cup and the plug replace the 0.7 plate, and the lob climbs at 400 km

Status: accepted

Amends: ADR 0033 (the plate's efficiency) and ADR 0037 §4 (the lifting rocket).
Every earlier figure keeps ADR 0033's plate; this ADR adds rows, it does not
rewrite them.

Date: 2026-10-06

## Context

`puffsat_impact_simulation`'s spray-plate study (its ADR-0055; handoff
`docs/argon_plate_owed_to_companions.md`, Draft 2, commit `7e1d8f7`,
2026-10-05) solved the overtake plate with real-physics 1-D columns scaled by
2-D effective-γ containment. Neither of its designs reaches the paper's
η_jet = 0.775:

- **Spray cup**, the design that flies first: η_jet = **0.60** baseline, 0.57
  unmixed downside (0.67-0.71 fully mixed). It is a deep bowl, d/D 0.25-0.30,
  with a 2 m skirt.
- **Plug**, the later design once it is proven on the nozzle: η_jet ~**0.70**
  (an estimate carried from 2-D).

It asks this repo to recompute the growth ledger and the cost model for both
designs, at the paper's k = 8.52, before the parent paper changes (handoff,
"Order of work", step 1). It names three conventions, and all three were
checked:

1. The ledger's `plate_efficiency` is the energy efficiency, η = η_jet². So
   0.60 is 0.36, 0.57 is 0.325, 0.70 is 0.49 and 0.775 is 0.60.
2. The impact-sim's η_jet already charges the PuffSat's water bonds as lost,
   so these designs run with `NO_BONDS`. ADR 0033's "argon on ice" option
   would charge the same bonds twice.
3. Its plate is 150 t (`PLATE_MASS`), which is what the study was re-sized to.

The handoff also points out that the push is impulsive in the ledger. At
12 MN·s and 4 Hz (48 MN on 1500 t) the push lasts minutes, and a plate cannot
tilt its thrust to hold altitude (ADR 0009). **The author's ruling
(2026-10-06): the lob must still be climbing at 400 km, so the lifting rocket
is bigger again.**

## Decision

**1. Four plate designs, plus ADR 0033's plate as the comparison**
(`growth_ledger.PlateDesign`, `PLATE_DESIGNS`). The four are spray cup 0.60,
spray cup 0.57, plug 0.70 and paper 0.775, all with argon spray and no
PuffSat bonds. Their per-pulse loading is capped at the paper's
**k = 8.52** (`PAPER_SLUG_RATIO`). The Pontryagin schedule still falls below
the cap pulse by pulse, since the impact-sim finds η_jet flat in k (its P13).
Lifting the cap to 10 is printed as a sensitivity and moves doubling by 1-2%.
`seed_cost.design_chain`, `harvest.delivery_front` and
`growth_cost_inputs.design_inputs` take the plate design, so a delivery to L1
flies the same plate as growth. Defaults are ADR 0033's plate, so every
existing figure and test is unchanged.

**2. The lob arrives at 400 km climbing at 1.0-1.2 km/s** (`src/lob_rise.py`).
Integrating the push in the orbit plane (thrust along the stream, 48 MN, mass
falling with speed to ~0.80 of its start) shows that a unit arriving at an
apex falls ~220 km over the ~300 s push. Meanwhile the stream, nearly straight
at ~60 km/s, climbs **~185 km** as the Earth curves away beneath it. The craft
would end ~400 km under the stream's path. Arriving at 1.1 km/s keeps it within
**about ±65 km** of the path (−61 / +71 km on the first three cycles), and the
final periapsis is ~460 km. That residual is closed by aiming each PuffSat at
the craft's predicted position, a fraction of a m/s days out (estimate).

The handoff's figures differ: ~250 s, and a stream climb of ~70 km. Its 1-1.2
km/s climb rate agrees with this integration.

**3. The climb costs the booster 7-11%.** A vertical ascent on ADR 0037's
scaled vehicle (5000 t, 250 t dry, 75 MN, no drag, no brake) loses 7.0% of its
lofted mass at 1.0 km/s and 10.1% at 1.2 km/s (380 s). At 350 s the losses are
7.6% and 11.0%. Lofted mass scales with the vehicle, so 1500 t now needs a
booster **about 1.18-1.33x Super Heavy**, up from ADR 0037's 1.1-1.2x. ADR 0037
prices a flight in proportion to the booster, so the lob's $/kg lofted rises by
the same ratio: **$25 becomes $26.8-27.5** on the Estimate. The cost model
charges this on every lob, growth and delivery alike. The seed is unaffected
(it departs from LEO on a refuelled Starship). This integration lofts 1575 t
at scale 1 where ADR 0037 quoted 1422 t, so only ratios are taken from it.

## Results

`make plate-designs` (ledger, ~11 min) and `make plate-designs-cost` (Estimate
book, ADR 0040's 10% a year and 50% odds, direct seed, ~20 min). Doubling is
under the k = 8.52 cap. Costs are with the lob climbing at 1.0 | 1.2 km/s.
Break-evens (BE) are cheap / dear seed.

| Plate | Departure | Doubling yr | 10-yr multiple | Steady $/kg (apex lob → climbing) | BE steady | BE liquidation |
|---|---|---|---|---|---|---|
| **spray cup 0.60** | H2 solved | **2.02** | 28.6 | 116 → 121-123 | **191 / 372** - 193 / 375 | 227 / 700 - 230 / 702 |
| | CH4 solved | 2.28 | 19.0 | 118 → 124-126 | 203 / 468 - 206 / 470 | 239 / 880 - 242 / 883 |
| | methalox | 7.82 | 2.3 | 277 → 289-295 | 763 / >3000 | 571 / >3000 |
| spray cup 0.57 | H2 solved | 2.10 | 25.0 | 119 → 123-125 | 201 / 414 - 204 / 417 | 239 / 781 - 242 / 784 |
| | CH4 solved | 2.38 | 16.7 | 122 → 127-129 | 215 / 528 - 218 / 531 | 253 / 988 - 256 / 991 |
| | methalox | 8.93 | 2.1 | 309 → 323-329 | 880 / >3000 | 605 / >3000 |
| **plug 0.70** | H2 solved | **1.80** | 43.9 | 107 → 112-114 | **162 / 272** - 164 / 274 | 193 / 497 - 196 / 499 |
| | CH4 solved | 2.01 | 29.1 | 108 → 113-115 | 169 / 327 - 171 / 330 | 200 / 613 - 203 / 616 |
| | methalox | 5.54 | 3.3 | 209 → 218-223 | 515 / 2542 - 522 / 2550 | 472 / >3000 |
| paper 0.775 (reference) | H2 solved | 1.67 | 59.3 | 101 → 105-107 | 146 / 223 - 148 / 225 | 173 / 395 - 175 / 397 |
| | CH4 solved | 1.85 | 39.1 | 102 → 106-108 | 150 / 261 - 152 / 264 | 178 / 481 - 180 / 484 |
| | methalox | 4.57 | 4.2 | 178 → 187-190 | 407 / 1763 - 413 / 1769 | 411 / 2555 - 416 / 2561 |
| ADR 0033 (comparison) | H2 solved | 1.59 | 75.1 | 100 → 105-107 | 139 / 198 - 141 / 201 | 163 / 339 - 166 / 341 |
| | CH4 solved | 1.74 | 49.7 | 100 → 105-107 | 142 / 227 - 144 / 229 | 167 / 406 - 170 / 408 |
| | methalox | 4.08 | 5.1 | 169 → 177-180 | 364 / 1409 - 370 / 1415 | 386 / 2180 - 391 / 2185 |

ADR 0033's apex-lob row reproduces `docs/growth_cost_for_parent.md`'s $100 and
$169 steady state.

**What changes.**

- **The spray cup costs a solved chamber about a quarter of its growth rate.**
  Solved hydrogen's doubling goes from 1.59 to 2.02 yr, and its 10-year
  multiple from 75 to 29. Steady $/kg rises only ~16% (100 → 116, 121-123 with
  the climbing lob), because the plate and lob dominate steady cost per
  kilogram whatever the growth. The dear-seed break-even nearly doubles (198 →
  372), because slower growth stretches the seed's repayment.
- **The solved chambers still meet both targets on the cheap seed** with the
  spray cup: under Starcloud's $500, and against Suncatcher's $200 hydrogen
  passes ($191-193/kg steady break-even) while methane just misses ($203-206). **On the dear seed** the spray cup's hydrogen
  ($372-375) and methane ($468-470) stay under $500 and miss $200. The 0.57
  downside puts methane at $528-531, over $500.
- **Methalox no longer repays any seed** behind the spray cup (value per seed
  dollar below zero at $500). Its 7.8-yr doubling leaves the chamber's lead
  larger than before.
- **The plug recovers most of the loss.** Solved hydrogen doubles in 1.80 yr
  and breaks even at $162 / $272, under both bars on the cheap seed.
- **The climbing lob costs $5-7/kg steady** on every design and $2-3 on
  break-evens. It does not change any ranking.

Not recomputed here: the seed routes (ADR 0039's Earth-assist figures), the
Pessimistic and Paper books, and the `make growth-cost` sensitivity sweeps for
the new plates. All of them use ADR 0033's plate until a follow-up.

## Consequences

- `tab:mass_interest_growth` and the cost tables in the parent should quote
  the spray cup (0.60) and plug (0.70) rows, with 0.775 as a reference.
  Handoff: `docs/plate_designs_for_parent.md`.
- The lifting rocket assumption in `growth_cost_for_parent.md` §4 item 2 and
  the parent's `sec:vertical_lob` grows from 10-20% to roughly 18-33% bigger
  than Super Heavy.
- The push is still impulsive in the ledger. The climb is charged on the
  booster, not as a gravity loss on the plate. That is right if the climb
  rate is chosen to track the stream, as here. A finite-push ledger is open.
