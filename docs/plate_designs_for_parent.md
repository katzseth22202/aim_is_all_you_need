# Spray cup and plug: the growth ledger and cost model, for the parent paper

Raised 2026-10-06 in `aim_is_all_you_need`. This answers step 1 of
`puffsat_impact_simulation`'s spray-plate handoff
(`docs/argon_plate_owed_to_companions.md`, Draft 2, `7e1d8f7`): recompute the
growth ledger and the cost model for both plate designs before the parent
applies P1-P15. **Written to be copied into the paper repo.** Backing
decision: `docs/adr/0041-the-spray-cup-and-the-plug-replace-the-07-plate.md`.
Reproduce with `make plate-designs` (~11 min) and `make plate-designs-cost`
(~20 min).

**Nothing has been written into `templateArxiv.tex`.**

## What was run

- **Plates.** η_jet 0.60 (spray cup), 0.57 (its unmixed downside), 0.70 (plug)
  and 0.775 (the paper, reference only). The ledger takes η = η_jet². Argon
  spray, and the PuffSat's bonds are not charged again, because the
  impact-sim's η_jet already counts them as lost.
- **Loading.** k is capped at the paper's 8.52 and falls pulse by pulse on the
  optimal schedule (8.52 → 1.4-4.2 by the last pulse). A cap of 10 changes
  doubling by 1-2%.
- **Plate and unit.** 150 t plate, 1500 t launch unit, push at 400 km, 20-day
  parking orbit, departure from 600 km. Solved chambers (H2 0.858, CH4 0.538)
  and the methalox incumbent, as in `tab:growth_ledger_ten_year`.
- **Costs.** Estimate book, 10% a year, 50% odds the cycle works (ADR 0040),
  seed flown direct, $337 / $9293 per kg (cheap / dear).
- **The lob climbs.** It arrives at 400 km still rising at 1.0-1.2 km/s, so the
  unit tracks the stream through the push (below). Every lob, growth and
  delivery alike, costs 7.0-10.1% more.

## Results

| Plate | Departure | Doubling yr | 10-yr multiple | Steady $/kg at L1 | BE steady, cheap / dear | BE liquidation, cheap / dear |
|---|---|---|---|---|---|---|
| **Spray cup 0.60** | H2 solved | **2.02** | 28.6 | 121-123 | **191-193 / 372-375** | 227-230 / 700-702 |
| | CH4 solved | 2.28 | 19.0 | 124-126 | 203-206 / 468-470 | 239-242 / 880-883 |
| | Methalox | 7.82 | 2.3 | 289-295 | 763-773 / >3000 | 571-578 / >3000 |
| Spray cup 0.57 | H2 solved | 2.10 | 25.0 | 123-125 | 201-204 / 414-417 | 239-242 / 781-784 |
| | CH4 solved | 2.38 | 16.7 | 127-129 | 215-218 / 528-531 | 253-256 / 988-991 |
| | Methalox | 8.93 | 2.1 | 323-329 | 880-892 / >3000 | 605-612 / >3000 |
| **Plug 0.70** | H2 solved | **1.80** | 43.9 | 112-114 | **162-164 / 272-274** | 193-196 / 497-499 |
| | CH4 solved | 2.01 | 29.1 | 113-115 | 169-171 / 327-330 | 200-203 / 613-616 |
| | Methalox | 5.54 | 3.3 | 218-223 | 515-522 / 2542-2550 | 472-478 / >3000 |
| Paper 0.775 (ref.) | H2 solved | 1.67 | 59.3 | 105-107 | 146-148 / 223-225 | 173-175 / 395-397 |
| | CH4 solved | 1.85 | 39.1 | 106-108 | 150-152 / 261-264 | 178-180 / 481-484 |
| | Methalox | 4.57 | 4.2 | 187-190 | 407-413 / 1763-1769 | 411-416 / 2555-2561 |
| Previous (ADR 0033) | H2 solved | 1.59 | 75.1 | 105-107 | 139-141 / 198-201 | 163-166 / 339-341 |

Ranges in the cost columns are the 1.0 to 1.2 km/s climb. With the lob at an
apex as before, the previous plate gives the published $100 (chambers) and
$169 (methalox) steady state. The climb alone adds $5-7/kg steady and $2-3 to
break-evens.

## What it means for the paper

1. **`tab:mass_interest_growth`**: replace the 0.775 requirement with the spray
   cup (0.60) and plug (0.70) rows, keeping 0.775 as a reference (handoff P11).
2. **Growth**: behind the spray cup a solved hydrogen chamber doubles in about
   two years (2.02), not 1.6. Its ten-year multiple falls from 75 to 29. The
   plug gets back to 1.80 years.
3. **Costs**: steady state at L1 stays near $120/kg for a solved chamber.
   On the cheap seed, the spray cup's hydrogen chamber repays under $200 and
   methane just over ($203-206). On the dear seed both stay under $500
   ($372-375 / $468-470) but neither reaches $200. The 0.57 downside puts dear-seed
   methane over $500 ($528-531). The plug brings dear-seed hydrogen to
   $272-274.
4. **Methalox** behind the spray cup doubles in 7.8 years and repays no seed at
   $500. The case for the chamber is stronger than before.
5. **The lob** (`sec:vertical_lob`, and §4 item 2 of `growth_cost_for_parent.md`).
   The unit must arrive at 400 km still climbing. Draft: "The growth push lasts
   about five minutes. Unsupported, the launch unit would fall about 220 km
   while the stream, nearly straight at 60 km/s, rises about 185 km as the
   Earth curves away beneath it. A plate cannot tilt its thrust to hold
   altitude, so the lob arrives at 400 km still climbing at 1.0 to 1.2 km/s,
   which keeps the unit within about 65 km of the stream's path. Each PuffSat
   is aimed at the unit's predicted position. The climb costs the lob 7 to 11
   percent of its payload, so we assume a lifting rocket about 18 to 33
   percent bigger than Super Heavy, priced in proportion."

## Back to `puffsat_impact_simulation`

- The handoff estimates the push at ~250 s and the stream's climb at ~70 km.
  The in-plane integration here gives **~300 s and ~185 km**, with ~220 km of
  fall unsupported. Its 1-1.2 km/s climb rate is confirmed.
- About ±65 km between the craft and a single stream hyperbola remains at the
  best climb rate. The stream is assumed to be aimed per PuffSat (an estimate,
  not solved here).

## Not yet recomputed for the new plates

The seed routes (ADR 0039's Earth gravity assist, which lowered the dear-seed
figures before), the Paper and Pessimistic books, and the sensitivity sweeps in
`make growth-cost`. Each still uses the previous plate.
