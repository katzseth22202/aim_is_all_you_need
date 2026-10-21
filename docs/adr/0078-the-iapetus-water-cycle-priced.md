# ADR 0078: The Iapetus water cycle priced: a nuclear rocket to the eccentric orbit

Date: 2026-10-21

Status: Accepted. Answers the parent's S51 items 1-4 and 7 for the water NTR, and item 6's
growth rate at Iapetus (`docs/iapetus_cycle_ledger_asks_for_aim_repo.md`, carried from
`Balloon-Pulse-Propulsion` @ `ec5995d`). Item 5 (the reactor plus chemical rockets) and
the rest of item 6 (Phoebe, the propagator) follow in their own ADR.

## Context

S51 asks for the parent's new nuclear scenario. Each round takes 9.5 years:

- Earth hardware is pushed to the eccentric orbit (10.916 km/s at 200 km, which is the
  10-day orbit) and flies to Saturn.
- A methalox stage dog-legs it onto Iapetus.
- A single-use water NTR stage (1300 K, 200 s, 560 MW, SiC barrier) loads ice and
  returns 25 kg water bags to Earth at 51.1 km/s.
- The bags push the next round's ship on the Earth bag plate. They strike a ship-carried
  ice plug of `k` kg per kg of bag (the impact sim's Result 20; its S52 is unrun).

The parent wants:

- the mass ledger and a cost ledger, as two whole ledgers: A, reuse at scale, and B, no
  reuse;
- the gated rNPV at 10% and a 4% government loan;
- `k` solved on the whole ledger;
- the comparison with the paper's current route;
- the $1-2/kg target;
- sensitivities.

The author settled two framing questions (2026-10-21):

- **The headline is $/kg to the eccentric orbit, with L1 beside it.**
- **The gated rNPV counts whole rounds; the pipeline is printed too.**

## Decision

`src/iapetus_cycle.py`, `make iapetus-cycle` (~20 s). The ledger is kept per 560 MW
engine, since every mass in the chain scales with thrust.

1. **Finite burns on point masses, 2-D, thrust along velocity.** One integrator serves
   Saturn's periapsis burn (start optimised), Iapetus's spiral and the Earth departure.
   - It reproduces the parent's Oberth losses (1.3 / 2.8 / 8.5 / 20.3 / 27.2%).
   - At the parent's own 1% tankage it reproduces the 214 / 315 / 554 / 826 / 919 t of
     PuffSat per engine and the 32.6 h run.
   - The parent's Iapetus loading orbit is **800 km radius** (65 km up), not 800 km
     altitude. That is what gives its 0.74 km/s impulsive and 1.37 km/s low-thrust
     figures.
2. **The new 0.2% tank base carries 1,251 t of PuffSat per engine, not 826.**
   - The 826 t is the old 1%-tankage figure.
   - With the brake-only reactor charged, PuffSat per Earth kg is 13.2, not 4.8.
   - At 0.2% the growth optimum is near the **6 h** burn, not 3 h: 3 h is now the worst
     of 3/6/12 h on growth.
3. **The outbound stack is 16.1 kg per kg landed, not the parent's 8.2.**
   - Saturn side: methalox at S51's own 370 s and 10% dry gives 5.1 kg per kg landed.
     The parent's 8.2 needs about 5% dry.
   - Earth side: the outbound hydrogen NTP stage is a single-use nuclear engine, the
     same $40-100M NASA engine.
   - At the parent's 5%-loss rule it needs 0.36 engines per landed tonne, $245/kg of
     cargo. The ledger picks the engine count on the gated break-even instead: a 60%
     finite loss, 0.019 engines per landed tonne.
   - Its full table is printed. A methalox Earth departure (28.9 kg per kg landed) is a
     sensitivity, and it is worse.
4. **The Earth push uses Result 20's table through the parent's overtake law.**
   - `B` is fitted per tabled `k` and interpolated in `ln k`, only inside the table.
   - Backup pulses enter at J/mv 1.74 with the plug spent (`p` = 5%, placeholder).
   - The liner's graphite carried as charge costs 2%.
   - Plate and spring are 4.4% of each 654 t push and are not cargo.
   - 60 / 68 km/s scale by the k = 8.5 gains.
5. **The ledgers.**
   - A: $200K a plate flight, $1 per kg pushed for electronics, 12 h burn.
   - B: $5M, the existing skin and package prices, 6 h burn.
   - Lob $25 x 1.0648 = $26.6 on every kilogram lofted: cargo, ship, plate, plug and
     liner.
   - Engine: U-235 (71.3 kg of a 19.75% LEU cermet at $112.5k/kg) fixed, the rest per
     MW, $60M at 560 MW.
   - Gates as S51: 95 / 90 / 85 / 90%.
   - Two returns reinvested (9.5 and 19 years), held level after. Bag homing on the solar
     dive's 20%-to-3% attrition curve.
6. **The solar dive gains a delivery target** (`Design.delivery_target`,
   `make solar-dive ARGS=--eccentric`). Its headline closures are re-optimised to deliver
   to the 10-day orbit at 400 km: **$143 / $86** dear, against $146 / $87 at L1.

## Results (Ledger A unless noted; $/kg at the eccentric orbit)

| ledger, plug | k | leverage | G / round | steady | rNPV 10% | loan 4% | rNPV / loan pipelined |
|---|---|---|---|---|---|---|---|
| **A, compact (simulated)** | **4** | 7.83 | **5.95** | **$90** | **$309** | **$131** | $281 / $126 |
| A, shaped ceiling (unrun) | 5.7 | 8.40 | 6.38 | $92 | $278 | $127 | $256 / $124 |
| B, compact | 4 | 7.83 | 6.09 | $113 | $345 | $158 | $317 / $153 |
| B, shaped ceiling (unrun) | 5.7 | 8.40 | 6.53 | $111 | $308 | $150 | $286 / $146 |
| solar dive 9.86 R_sun, X 60 | | | 3.36 / 3S | $93 | $143 | | |
| solar dive 4 R_sun, X 90 | | | 5.00 / 3S | $72 | $86 | | |

- **Against the current route, the cycle loses on the gated rNPV and roughly ties on
  steady cost.**
  - Steady: $90 against the solar dive's $93 / $72.
  - Gated rNPV: $309 against $143 / $86. A 9.5-year round with two reinvestments puts
    the first cargo at 28.5 years.
  - The 4% loan brings it to $131, which still loses to the solar dive at 10%.
  - The optimistic mass rows give $142, a tie with 9.86 R_sun.
- **Where Ledger A's $90 goes:**
  - the lob on the pushed mass, $34;
  - the plug, $17;
  - the outbound hydrogen NTP engines, $14;
  - chemical stage hardware, $11;
  - the water engine, $8 (9%; 12% on Ledger B);
  - redundancy, $1.75;
  - plate, liner and gun, $0.4 each on A ($9.6 each on B).
- **k: low wins.** The compact plug's optimum is its lowest run, k = 4. The shaped
  ceiling's is 5.7. Miss rows at k = 4: p = 1% $301, 20% $344.
- **Plates (A):** 15 pushes of 3,340 pulses per engine a round. A plate flies 6.1 pushes
  in the half-year stream at a 30-day turnaround (assumption), so 2.5 plates per engine
  per round. Ledger B expends 9.1. The ledger's own plate curve at 26-32 t gives
  $2.5-3.0M first and $0.5-0.6M at its floor, both under Seth's $5M.
- **Residual loss is not small.** About 167 gun firings a push at `q` = 1e-3 lose a
  plate and its ~$200M ship 17% of pushes. That adds **$64/kg**. The row reaches 1% of
  $/kg at **`q` = 1.4e-5**.
- **Growth (item 6, Iapetus):** G 5.95 per 9.5 yr round, ln G / T = 0.188, doubling
  3.7 yr. Over 19 years: x35, the same pipelined or in whole rounds. The frozen edge
  gives 5.34, the shaped ceiling 6.38.
- **The $1-2 target is out of reach on this architecture.**
  - Each kg of cargo lofts 1.26 kg pushed and 0.64 kg of plug: $51 at $26.6.
  - Everything else is $39.
  - With everything else free, the lofted kilogram must cost **$1.04** (x26 cheaper),
    for the ship and cargo as well as the plug.
- **Sensitivities (Ledger A, rNPV 10%):**

  | row | rNPV |
  |---|---|
  | 1800 K | $249 |
  | engine $40M / $100M | $265 / $389 |
  | 60 km/s | $309 (flat) |
  | 68 km/s | $562 (the Saturn burn grows) |
  | 6 h / 16 h / 3 h burn | $319 / $320 / $397 |
  | frozen edge | $388 |
  | second engine set / no redundancy | $346 / $294 |
  | graphite at $1000/kg | $335 |
  | 1% tankage | **$3,681** (G 2.6) |
  | pessimistic masses | **$8,801** (G 2.1) |
  | optimistic masses | $142 |

  Power per engine barely matters ($334 at 280 MW, $297 at 1120 MW), because only the
  U-235 is fixed.
- **Barrier:** the parent's coat table reproduces: 0.33-0.61 mm at 1400 K and 10 atm
  over 32.5 h.
- **Saturn-made tank walls:** a 40 kWe unit makes 16.8 t a year, the walls of 6.7
  stages a round. It pays if landing it costs under $130M a round.

## Consequences

- The cycle is a **steady-cost tie and an rNPV loss** to the solar dive. Its
  disadvantage is the 9.5-year round, not the Saturn side.
- **Tankage is the cycle's knife-edge.** The parent should not print the 826 t without
  its 1% label, nor the 0.2% base without the 1% row beside it.
- **The outbound engine is a cost line the parent did not show.** At $60M, the
  single-use hydrogen NTP engines cost more per kg of cargo than the water engine.
- Labelled placeholders and assumptions, none sourced here:
  - `p`, `q`;
  - the shaped-plug ceiling;
  - 90% plant odds (item 5);
  - the 71.3 kg U-235 (from a garbled table);
  - LH2 $6/kg, stage hardware $500/kg, kit $1000/kg, graphite $20/kg, gun $5M;
  - the plate's 30-day turnaround;
  - the brake reactor at 10% power.
- Not done here: the methalox comparison and Phoebe (item 5 and the rest of item 6); the
  12-hour arc's ring clearance on a real propagator; the most power a core reaches at
  the 1400 K wall; a multi-perigee Earth departure, which would cut the hydrogen
  engines' 60% loss.
