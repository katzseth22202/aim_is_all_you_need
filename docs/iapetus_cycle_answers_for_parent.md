# The Iapetus water cycle priced: answers to S51 (for the parent)

Raised 2026-10-21 in `aim_is_all_you_need` (ADR 0078, ADR 0079). **Written to be copied
into `Balloon-Pulse-Propulsion`.** Answers S51 items 1-7
(`docs/iapetus_cycle_ledger_asks_for_aim_repo.md` @ `ec5995d`).

Reproduce with:

- `make iapetus-cycle` (~1 min): the water NTR, both ledgers, both Earth-departure
  scenarios.
- `make saturn-legs` (~20 s): the real arcs and planes.
- `make saturn-methalox` (~3 min): the reactor plus methalox.
- `make solar-dive ARGS=--eccentric` (~2 min): the solar dive at the eccentric orbit.

All figures are $/kg to the eccentric orbit (the 10-day orbit, 10.916 km/s at 200 km).
They use Ledger A, the compact plug at k = 4, the gated rNPV at 10% and the 4% loan in
whole rounds, unless noted.

## Headline for the paper (Seth, 2026-10-21; ADR 0081)

**On the paper's headline metric, the gated rNPV at 10%, the solar dive beats every Saturn
scenario computed, and it needs no nuclear approvals.** Every Saturn cycle needs a fission
reactor: a water NTR, an outbound hydrogen NTP, or a methalox plant at Saturn. That holds
even with no nuclear near Earth, where the reactors are launched cold as cargo. The solar
dive is PuffSats and methalox only.

All $/kg are to the eccentric orbit (the 10-day orbit), Ledger A, k = 4, rNPV 10% / 4%
loan, dear seed for the solar dive.

| scenario | best Saturn cell | rNPV 10% | 4% loan | steady |
|---|---|---|---|---|
| ADR 0078-0079 base, H2 NTP | methalox at Phoebe (coplanar) / water NTR at Iapetus | $308 / $309 | $129 / $131 | $81 / $90 |
| base, no nuclear near Earth | methalox at Phoebe (coplanar) | $338 | $129 | $84 |
| reactor and plates reused, H2 NTP | methalox at Phoebe, 0.1 yr stay | $206 | $99 | $72 |
| reactor and plates reused, no nuclear near Earth | the same | $235 | $103 | $73 |
| aerocapture, H2 NTP | water NTR at Iapetus (coplanar; real planes $198) | **$172** | $100 | $79 |
| aerocapture, no nuclear near Earth | water NTR at Iapetus (coplanar; real planes $299) | $223 | $108 | $80 |
| **solar dive, 9.86 R_sun / 4 R_sun** | | **$143 / $86** | $105 / $76 | $93 / $72 |

- **The 10% rNPV is a clean sweep.** The best Saturn cell ($172) is 1.2x the shallow dive
  and 2.0x the deep one. The Saturn rows on the real planes are all worse than their
  coplanar rows.
- **Say this precisely in the paper.** The deep dive (4 R_sun) wins on every metric. The
  shallow dive (9.86 R_sun) wins on the 10% rNPV, but on the 4% loan the best Saturn
  cells tie or edge it ($99-100 against $105), and on steady cost they undercut it ($72-80
  against $93). Those cells are coplanar or a reuse idealisation. On the real planes,
  aerocapture gives $106 against $105.
- **Why Saturn loses: distance, not hardware.** A 7.7-9.5 year round puts first cargo near
  year 24-30, and every kilogram still rides the same lofting floor (lob plus plug, about
  $49 of the steady cost). Reuse and aerocapture each cut the Saturn cost a lot and leave
  both untouched.
- **Not computed:** the head-on-impactor Saturn cycle Seth has queued (overtake PuffSats to
  the eccentric orbit, head-on PuffSats onto the Saturn arc about 20 days later,
  aerocapture for all but the seed). If it is run, the headline should be re-checked.
  Bounds that zero every price but the lob are not scenarios: they reach $71-95 only by
  making the single-use nuclear engines free.

**Suggested paper text:** "We priced a Saturn water cycle (S51) in every variant we could
construct: a nuclear water rocket or a kept reactor making methalox, at Iapetus or Phoebe,
with or without nuclear propulsion near Earth, with the reactor and plates reused, and with
aerocapture at Saturn. At a 10% gated rNPV, none beats the solar dive. The Saturn cycle's
long round and the lofting floor dominate its cost, not its hardware. The solar dive
also needs no fission reactor, and so no nuclear launch or operating approval."

## Decisions Seth made during the work (2026-10-21)

- The headline is $/kg to the eccentric orbit, with L1 beside it. Whole rounds is the
  headline count, with the pipeline printed beside it.
- **No nuclear near Earth is a second scenario, compared on everything.** The concern is
  that a nuclear thermal rocket may not get approval to run inside the magnetosphere.
  - An accident after startup scatters fission products; a cold core is far less of a
    hazard.
  - The seed leaves LEO in one burn, on the refuelled, expended Starship the ledger
    already prices.
  - Recurring ships are pushed to the eccentric orbit by the bags, then burn methalox at
    perigee.
  - Reactors ride out cold, as cargo.
- **The methalox feed is carbon, not necessarily CO2.** The dark material is thought to
  carry carbon. It is gasified with steam, shifted and methanated. The ceria cycle, or an
  ordinary electric plant with electrolysis, supplies the extra hydrogen and the LOX.

## Short answers

1. **The mass ledger reproduces the parent's tables, and the parent's 826 t is
   superseded.**
   - Your 1.3 / 2.8 / 8.5 / 20.3 / 27.2% periapsis losses and the 214 / 315 / 554 /
     826 / 919 t of PuffSat per engine reproduce to the tonne at the old 1% tankage. So
     does the 32.6 h run.
   - Your loading orbit is **800 km radius** (65 km up), not altitude.
   - **At your new 0.2% tank base the engine carries 1,251 t, not 826.** PuffSat per
     Earth kg is 13.2, not 4.8. Growth peaks near the **6 h** burn, not 3 h.
   - Tankage is the knife-edge: 1% drops growth to 2.6.
2. **Your outbound 8.2 is light.**
   - S51's own methalox book (370 s, 10% dry) gives 5.1 on the Saturn side alone.
   - The hydrogen NTP engines are single-use $40-100M reactors. At your 5% loss rule
     they cost $245/kg of cargo, so the ledger buys fewer and takes a 60% finite loss.
     The stack is then 16.1.
3. **The water NTR: $90 steady, $309 rNPV, $131 on the loan** (Ledger B: $113 / $345 /
   $158). Against the current route at the same orbit, the solar dive is $143 (9.86 R_sun)
   / $86 (4 R_sun) dear-seed. The Iapetus cycle **ties on steady cost and loses on the
   rNPV**, because the 9.5-year round puts its first cargo at year 28.5.
4. **Lines (Ledger A):** lob on the pushed mass $34, plug $17, outbound NTP engines $14,
   stage hardware $11, water engine $8 (9%), redundancy $1.75, plate, liner and gun
   $0.4 each.
   - k: low wins: 4 on the compact plug (its lowest run), 5.7 on the shaped ceiling.
   - The ledger's plate curve, $2.5-3.0M a plate, undercuts your $5M.
5. **The residual plate loss is not small.** At q = 1e-3 it adds **$64/kg**: 167
   firings a push lose a ~$200M ship 17% of the time. q must be under **1.4e-5** to stay
   under 1%.
6. **No nuclear near Earth costs the water NTR 1.7x:** $532 rNPV, against $309. On the
   real planes it costs 2.5x: $1,173 against $475. Steady barely moves ($95).
7. **The core comparison (item 5).** A kept reactor plus methalox **at Phoebe** ties the
   water NTR with hydrogen NTP allowed ($308 / $309) and **beats it without** ($338 /
   $532; real planes $426 / $1,173).
   - The uranium is not the expensive line: the water engine is $8/kg, the plant $5-9.
   - Methalox wins on Earth mass per kg of PuffSat (0.071 against 0.076) and on Phoebe's
     7.7-year round.
   - It needs carbon at 0.3% or better and tanks made at Saturn. Shipped tanks stall it
     (G 0.9-2.3).
   - Electric edges ceria everywhere. Ceria wins nowhere.
8. **Sites (item 6).** Phoebe beats Iapetus for methalox at every carbon grade and ice
   fraction tried, on rate and on rNPV.
   - 1% carbon: ln G / T is 0.253 at Phoebe against 0.200 at Iapetus (doubling 2.7 against
     3.5 years). Chemical only: 0.197 against 0.138.
   - Whole rounds fit two in 20 years at either site, so the gated rNPV (whole rounds)
     gains from Phoebe only through its earlier dates.
   - A 0.1-year stay (load and leave) gives T = 7.34 at Phoebe and $312.
9. **The real planes cost little turn but faster arcs.**
   - Plane turns at apoapsis are under 0.02 km/s, because the apse is free on a cone.
   - Real capture is 2.81 km/s (median), not 2.55. Saturn-side outbound is 5.10 at
     Iapetus (not 4.78) and 4.47 at Phoebe (not ~4.1).
   - Earth departure is 5.19 km/s, not 4.90.
   - Phoebe's e = 0.164 is worth 0.22 km/s each way if met near apoapsis.
   - No 150 000 km orbit crosses the ring plane inside the F ring. Some cross at 150-175
     thousand km, two inside the G ring.
10. **$1-2/kg is out of reach on any cell.**
    - The best is Phoebe, electric, on the loan: $129. Of its $81 steady, $49 is lofting.
    - Each kg of cargo lofts about 1.3 kg pushed and 0.65 kg of plug. With everything else
      free, the lofted kilogram must cost **about $1** (x26 cheaper), ship and cargo as
      well as plug.

## Tables

**Headline.** Steady / rNPV 10% / loan 4%. Methalox uses the electric plant, 1% carbon
and tanks made at Saturn.

| Saturn side | planes | hydrogen NTP Earth departure | chemical only |
|---|---|---|---|
| water NTR, Iapetus | coplanar | $90 / **$309** / $131 | $95 / **$532** / $165 |
| water NTR, Iapetus | real | $121 / $475 / $188 | $119 / $1,173 / $270 |
| methalox, Iapetus | coplanar | $86 / $478 / $148 | $91 / $666 / $167 |
| methalox, Iapetus | real | $93 / $603 / $167 | $101 / $947 / $206 |
| **methalox, Phoebe** | coplanar | $81 / **$308** / $129 | $84 / **$338** / $129 |
| **methalox, Phoebe** | real | $87 / **$357** / $140 | $91 / **$426** / $147 |

**Water NTR, both ledgers** (coplanar):

| Earth departure | ledger | outbound | G / round | steady | rNPV | loan | pipelined rNPV | seed (10 engines) |
|---|---|---|---|---|---|---|---|---|
| hydrogen NTP | A | 16.1 | 5.95 | $90 | $309 | $131 | $281 | $16.2B |
| hydrogen NTP | B | 16.1 | 6.09 | $113 | $345 | $158 | $317 | $10.0B |
| chemical | A | 28.9 | 3.31 | $95 | $532 | $165 | $453 | $9.7B |
| chemical | B | 28.9 | 3.38 | $122 | $586 | $200 | $505 | $6.1B |

**Water NTR sensitivities** (hydrogen NTP, Ledger A, rNPV):

| row | rNPV |
|---|---|
| 1800 K | $249 |
| engine $40M / $100M | $265 / $389 |
| 60 / 68 km/s | $309 / $562 |
| 6 / 16 / 3 h burn | $319 / $320 / $397 |
| frozen edge | $388 |
| shaped ceiling | $278 |
| p 1% / 20% | $301 / $344 |
| second engine set / none | $346 / $294 |
| graphite at $1000/kg | $335 |
| **1% tankage** | **$3,681** |
| optimistic / pessimistic masses | $142 / $8,801 |

The pessimistic 4.5 t engine is replaced by a sourced 3.54 t LEU cermet ("Design
comparison of nuclear thermal rocket concepts").

**Carbon grade** (methalox, Iapetus, hydrogen NTP, kit 0.3 kg/t): G and rNPV.

| feed | 0.1% | 0.3% | 1% | 3% | 10% |
|---|---|---|---|---|---|
| carbon | 2.6, $5,289 | 4.9, $905 | 6.7, $478 | 7.9, $406 | 8.5, $388 |
| CO2 | does not grow | 2.3, $8,115 | 4.7, $954 | 6.3, $476 | 7.9, $377 |

**Plant power** (electric, Phoebe, hydrogen NTP). The fixed $50M a plant is diluted by
unit size:

| | 1 MWt | 2 MWt | 5 MWt | 10 MWt | 20 MWt | 100 MWt |
|---|---|---|---|---|---|---|
| rNPV | $424 | $361 | $322 | $308 | $302 | $297 |

## Labelled assumptions (none sourced here)

- **Push placeholders:** miss rate p = 5%, backup failure q = 1e-3, the shaped-plug
  ceiling (unrun, S52).
- **Water engine:**
  - $40/60/100M per engine;
  - 71.3 kg U-235 for a LEU cermet core, read from a garbled table (verify);
  - the brake-only reactor at 10% power;
  - the most power a core reaches at the 1400 K wall (unsolved).
- **Prices:** LH2 $6/kg; stage hardware $500/kg; kits $1000/kg; graphite $20/kg (Indian
  B2B listings, ~$6-36); gun $5M.
- **Plates:** a 30-day turnaround and a half-year arrival stream.
- **Plants:**
  - fixed $50M plus $15M/MWt (ceria) or $30M/MWt (electric);
  - 1.5 / 4.5 t per MWt;
  - a life of 1 / 2 rounds;
  - the 90% gate.
- **Carbon dig:** regolith heat 0.4-0.75 MJ/kg; the kit at 0.3 kg per tonne handled.
- **Saturn-made tank walls** at 75 MJ/kg.
- **Phoebe:** anchoring is unpriced.

## Suggested paper text (when S52 lands)

- Print the nuclear Iapetus cycle as a tie on steady cost with the solar dive, and a loss
  on the gated rNPV.
- Print the 826 t only with its 1% tankage label.
- If the paper takes the position that nuclear rockets do not run near Earth, the Saturn
  scenario should be **a kept reactor making methalox at Phoebe**. The water NTR at
  Iapetus is the simpler, carbon-free fallback, at about 1.6x the rNPV.

## Follow-up (2026-10-21): reuse the reactor and the plates

Seth asked: if the reactor stays at Iapetus or Phoebe making methalox (and tanks) until its
cost is small, and the Earth plates are reused, does the Saturn cycle still lose? **Yes.**
ADR 0079 addendum; `make saturn-methalox` prints the table (`reuse_report`).

| row (rNPV 10%, loan 4%) | Phoebe, H2 NTP | Phoebe, no nuclear | Iapetus, H2 NTP | Iapetus, no nuclear |
|---|---|---|---|---|
| ADR 0079 base | $308 ($129) | $338 ($129) | $478 ($148) | $666 ($167) |
| reactor free, 10-round life | $213 ($101) | $248 ($106) | $378 ($125) | $531 ($144) |
| + plate and gun free | $212 ($100) | $247 ($105) | $377 ($124) | $530 ($143) |
| + stay 0.1 yr | **$206** ($99) | **$235** ($103) | $357 ($122) | $498 ($141) |

- **Against the solar dive** ($143 at 9.86 R_sun, $86 at 4 R_sun), the best cell is
  about 1.5x the first. Steady cost ($72-73 at Phoebe) ties the 4 R_sun dive's $72.
- **Plates were already reused** on Ledger A (25 flights, $0.4/kg), so freeing them moves
  $1. Reflying the methalox stage makes it worse: braking it back costs growth.
- **What is expensive when everything is reused:**
  - **Lofting from the ground.** Each kg of cargo lofts about 1.3 kg pushed and 0.65 kg
    of plug at $26.6/kg: $32 + $16 of the $72. The $25 lob already assumes a reusable
    booster.
  - **Time.** The first cargo is sold in year 23-24 at Phoebe, after the seed and two
    reinvested rounds. At 10% a dollar then is worth about a ninth, and the gates
    multiply to 0.65. The rNPV is 2.9-3.4x the steady cost, against 1.5x for the solar
    dive.
- **Even with every price but the lob zeroed**, the no-nuclear Phoebe cell is $178. Only
  the hydrogen NTP bound beats the dive ($71), and only because its single-use outbound
  nuclear engines are free in that row.

**Suggested paper text:** the Saturn cycle's handicap is distance, not hardware. Reusing
the reactor and plates narrows the gap to the solar dive but does not close it, because
the lofting floor and the 7.7-9.5 year round remain.

## Follow-up (2026-10-21): aerocapture at Saturn

Seth asked for aerocapture at Saturn to be modelled, rings included (ADR 0080,
`make saturn-aerocapture`, about 5 min). Every recurring ship flies one pass through
Saturn's upper atmosphere instead of the capture burn. The seed keeps its burn.

- **The rings allow it in every window, 2035-2047, at both moons.**
  - The trick is to free the apse line round the asymptote's cone, rather than pin it to
    the moon's plane (ADR 0079's rule, which left only blocked geometries), and to accept
    a non-tangent match at the moon.
  - The inbound ring-plane crossing then falls at 61 000-65 000 km, in the gap under the D
    ring that Cassini's Grand Finale flew. The exit crossing is beyond the G ring.
- **The pass is mild.**
  - It sheds about 1.9 km/s about 400 km above 1 bar.
  - About 30 km/s relative to the atmosphere (prograde), 1.4 g, 172 W/cm^2 peak, 38
    kJ/cm^2.
  - The aeroshell is 24% of entry mass (HEEET fit plus 10% structure).
  - NASA's study (Putnam et al. 2023) reports 1400 W/cm^2 for a deeper pass to Titan's
    orbit.
- **It halves the outbound stack.** Water NTR: 16.1 -> 8.7 kg leaving the eccentric orbit
  per kg landed (H2 NTP), 28.9 -> 15.6 chemical only.

| rNPV 10% (4% loan) | H2 NTP: burn -> aerocapture | chemical only: burn -> aerocapture |
|---|---|---|
| water NTR, Iapetus, coplanar | $309 -> **$172** ($100) | $532 -> **$223** ($108) |
| water NTR, Iapetus, real planes | $475 -> **$198** ($106) | $1,173 -> **$299** ($123) |
| methalox, Phoebe, real planes | $357 -> $308 | $426 -> $296 |
| solar dive 9.86 / 4 R_sun | $143 / $86 | |

- **The water NTR at Iapetus becomes the preferred Saturn cell** with H2 NTP. Chemical
  only, it ties methalox at Phoebe on the real planes.
- **It still loses to the solar dive.** The round and the lofting are unchanged.
- **Largest levers:**
  - the aeroshell price: $5000/kg gives $291 / $419;
  - letting the seed aerocapture too: Phoebe $269 / $257.
- **The open gate is the corridor:** 0.17 deg of entry angle, 33 km of periapsis, at L/D
  0.25; 53 km at 0.4. A factor-two error in Saturn's upper-atmosphere density moves the
  needed periapsis by about 35 km. Guidance and dispersions are not run.

**Suggested paper text:** if the Saturn cycle stays, its recurring ships aerocapture. The
rings constrain the geometry but forbid no window. The corridor, which needs better
knowledge of Saturn's upper atmosphere, is the remaining feasibility question.

**Asked in conversation (not computed):** can one nuclear core serve both a hydrogen and
a water engine?
- Not with one coating. The tungsten cermet that suits 2700-3000 K hydrogen oxidizes to
  volatile hydrated oxides in steam above about 1000 K. The SiC that suits steam, via its
  silica skin, is attacked by hot hydrogen.
- A shared core would mean running the hydrogen mode at about 1300-1400 K, about 550-600
  s, behind SiC, or an oxide topcoat demonstrated in neither gas.
- It would save at most the water engine's ~$8/kg.
