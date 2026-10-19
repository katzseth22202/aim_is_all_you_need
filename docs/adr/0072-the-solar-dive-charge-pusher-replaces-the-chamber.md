# ADR 0072: The solar-dive charge pusher beats the chamber, everything charged

Date: 2026-10-16

Status: Accepted (author, 2026-10-16: "This is a vastly superior architecture"; keep
the gated valuation and the 15-synodic growth phase, and explain the seed's cost).
Answers `puffsat_impact_simulation` a6742a2, `docs/solar_dive_cycle_ask_for_calcs.md`
(D1-D7, gathering G1-G4 and G10). Reply: `docs/solar_dive_cycle_reply_for_sim.md`.
Supersedes ADR 0040's single 50% odds for the reports that quote this cycle, and adds a
15-synodic growth phase; the child's existing reports are unchanged.
**Superseded in part by ADR 0073** (2026-10-19): the canted plate's departure sign hides
the stream behind Earth; the cycle now flies two plates, the visible sign and a 1 g push.
This ADR's figures reproduce at commit 127162f.

## Context

The impact simulation proposes a different growth loop (its Results 10-12): the
returning **Earth wave** of Kapton sheets pushes and departs ships onto the three-synodic
Earth -> Jupiter -> Sun -> Earth closure (ADR 0019) on a flat **canted plate**; Sun-wave
graphite disks, split from the ship at Jupiter onto the mirror-image retrograde dive,
boost it head-on at perihelion; its remainder returns as the next Earth wave. No
chamber, rod, plug or pitch is expended. Its own model doubles in ~1.1 yr (7-8x per 3S),
but counts only impactor and leaves out the plates, dry mass, the push into the parking
orbit, launch and the seed. The child leads with the chamber cycle (ADR 0063: the
departure is where growth is lost) and asked whether this one should replace it.

## Decision

1. **Fly it through the ledger** (`src/solar_dive_growth.py`, `make solar-dive`). The
   stage laws are the simulation's, vendored (`data/solar_dive_stage_laws.csv`); the
   ledger reproduces its 8.4x / 7.0x from its own inputs (a slow test).
2. **Charge what its figure leaves out** (author, 2026-10-16, each a decision):
   - **The Earth wave pushes the 1500 t unit at 400 km** into the 10-day orbit, on the
     simulation's overtake law, like the child's ledger.
   - **The role split.** The push and the departure are ten days apart and a wave
     arrives as one pulse, so the push wave's ships return 10 days early. Their route is
     solved on three knobs (excess, aim, climb-out excess) against the clock, the Earth
     intercept and **Jupiter's phase**, which the plain closure lets float. They come home
     early on a faster climb (60 -> 87 km/s at 9.86 R_sun).
   - **One flat plate** does the push and the departure, sized for the larger pulse at
     30 kg/m^2 scaled as pulse energy^(2/3), checked against a 100 g cap over a 2 m
     absorber (the cargo is robust), then **diverted ~20 m/s into Jupiter**. Priced at half
     the spray cup's per-kg curve.
   - **The Sun paraboloid** is dead mass (cutting it into sheets is an option, not taken),
     priced on the spray cup's full curve. Its pulses are capped at **2 Hz**, so a longer
     arc means smaller pulses and a lighter plate; the arc's Oberth and off-head-on losses
     are charged (`T^2/96 tau^2 + T^2/24 tau^2`).
   - **Both plates carry film** at the spray cup's vapor-shielded rate per unit impulse.
   - **Dry mass:** avionics 1%, shade 0.5% (Parker-scaled), carrier 10% of the disks on a
     330 s stage at the highest perijove that turns the disks (<= 6.5 R_J), collar 1 t,
     mounts 1%.
   - **Disks at $1000/kg** (author's baseline). **Sheets** on the child's PuffSat book,
     coasting furled and unfurling in the last minutes; misses and failed unfurls an
     attrition curve, 20% falling on an 80% curve to 3%.
   - **Growth is counted in launch units.** The disks never return (unlike the child's
     rods, which come back as the nozzle wave), so they cost launch mass and dollars, not
     returned mass. The simulation's "per total impactor" is reported beside it.
3. **Value it on the gated rNPV** (the drug-development convention): 10% a year, each
   flow weighted by the odds of every milestone before it. Seed departs and the carrier
   splits at Jupiter 95%; first Sun boost 75%; sheets arrive 85%; first push and canted
   departure 90% (~55% in all). **15 synodic periods of growth** (author: a drug
   programme's ~15 years, in the cycle's own unit) before any mass is delivered, then
   held level. The child's rows are recomputed on the same footing (its chamber's first
   departure is its 75% gate).
4. **Optimize for the dear-seed break-even** (author: the more realistic end): the four
   slug ratios, the Sun arc and the departure pulse, by Nelder-Mead inside the simulated
   ranges (`_BOUNDS`), from the recorded starts (`HEADLINE_STARTS`).

## Results (`make solar-dive ARGS="--sweep --child"`, ~3 min on 4 workers)

| design | growth per cycle | doubling | steady $/kg at L1 | break-even cheap / **dear** |
|---|---|---|---|---|
| **solar dive, 9.86 R_sun**, X 60 | 3.60 (3.23 per total impactor) | 1.77 yr | $69 | $88 / **$104** |
| **solar dive, 4 R_sun**, X 90 (shade unsolved) | 4.92 (4.35) | 1.43 yr | $55 | $63 / **$67** |
| bulged hydrogen, 2 Hz | | | $165 | $302 / **$430** |
| methane AR100, 24 kg, 3 Hz | | | $163 | $363 / **$634** |
| methane AR100, 24 kg | | | $194 | $509 / **$967** |
| methalox | | | $353 | $2323 / **>$5000** |

The seed ends are ADR 0057's at the 10.43 km/s excess: $1297 / $4993 per kg sent.

- **The seed is dear per kilogram and cheap per fleet kilogram.** After 15 synodic
  periods it has grown 110x (9.86 R_sun) or 401x (4 R_sun): $21 or $7 per fleet kg. At
  10 synodic periods it is $264 and the dear break-even $224; at 21, $85.
- **The lob is the dominant cost**: about $40-45 of every L1 kilogram (mostly the
  delivery lob). At $50/kg the dear break-even is $166 / $116.
- **Of the charges, the g cap moves growth most**: at 20 g the plates grow and 9.86 R_sun
  falls to 2.92x and $155 dear. Nothing else tried moves the break-even more than ~$6
  (guidance 2-10%, film scaling, bonds, shade scaled with sunlight, carrier perijove, plate
  prices 25-100%, disks $500-2000/kg, optimistic odds).
- **Flat 30%, no odds, 9 synodic periods** (ADR 0037's rate as a bound): $355 / $982 at
  9.86 R_sun and $174 / $421 at 4 R_sun, against $1158 / $3098 for hydrogen. The ranking
  holds on every schedule tried.

### Depth (dear-seed break-even, each closure optimized)

| R_sun | best return excess | growth | doubling | dear break-even |
|---|---|---|---|---|
| 4 | 90-110 km/s | 4.9-5.0 | 1.41-1.43 yr | $66-67 |
| 6 | 90 | 4.36 | 1.54 | $77 |
| 8 | 75 | 3.93 | 1.66 | $89 |
| 9.86 | 60 | 3.60 | 1.77 | $104 |
| 12 | 60 | 3.33 | 1.89 | $121 |
| 15 | 75 | 2.43 | 2.56 | $334 |
| 20 | none | | | |

Deeper is cheaper: a fixed return excess needs less boost where perihelion is faster
(`X^2 ≈ 2 v_p dv`). **The cycle ends on geometry, not cost**: beyond ~12-15 R_sun the disks
cannot make their Jupiter turn onto the mirror dive at any perijove.

## Corrections returned to the simulation

- The departure from the 10-day orbit's 600 km periapsis is 4.34 km/s, not 4.15 (which
  starts from parabolic speed at 200 km); 4.42 at 4 R_sun.
- The disks need the **true mirror excess**, 15.07 km/s, not the 14.78 floor (14.34 against
  14.16 at 4 R_sun), and cannot turn above 6.72 R_J (6.90); the ship's flyby is pinned at
  7.86 R_J.
- G1 and G4 confirmed.

## Limits

- **No development cost** (R&D, test flights, factories, tracking) in either
  architecture. The comparison stands; every absolute figure is flattered.
- **The optima sit on the simulated tables' edges** (k_E 5-6, k_S 20, delivery k 5), so
  the true optimum may be cheaper. The canted law is scaled off its two closures; the
  overtake law is scaled by closing speed from 100 km/s; the canted plate is assumed to
  load like the overtake one per kg of sheet. Asks 1-5 of the reply.
- **The 4 R_sun shade is unsolved** (the parent's E-items); 9.86 R_sun has Parker's flight.
- Circular, coplanar closures; the two ship types' different aims charged by magnitude.
- Lofting many units inside one wave's passage, and the 10-day orbit's lunar perturbation,
  are not modelled (as in the child's ledger).

## Reproduction

`make solar-dive` (headline, corrections and sensitivities, ~2 min);
`make solar-dive ARGS="--sweep --child"` adds the depth sweep and the child's rows.
Tests: `tests/test_solar_dive_growth.py` (the closures, the true mirror excess, the
push-wave route and the reproduction of the simulation's growth are slow).
