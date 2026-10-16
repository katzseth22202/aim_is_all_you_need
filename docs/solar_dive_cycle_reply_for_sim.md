# The solar-dive charge-pusher cycle, charged and priced: reply to D1-D7

Raised 2026-10-16 in `aim_is_all_you_need` (ADR 0072). It answers
`puffsat_impact_simulation` @ `a6742a2`, `docs/solar_dive_cycle_ask_for_calcs.md` (D1-D7,
gathering G1-G4 and G10). Reproduce with `make solar-dive` (~2 min) and
`make solar-dive ARGS="--sweep --child"` (~3 min on 4 workers). The stage laws are the simulation's,
vendored in `data/solar_dive_stage_laws.csv` (see `data/README.md`).

## The answer

**Lead with it.** With the plates, the push into the parking orbit, the role split, dry mass,
the carrier, launch, the impactors and the seed all charged, the cycle beats every chamber row
by about 3x or more on every valuation tried:

| design | doubling | steady $/kg at L1 | break-even, cheap / **dear** seed |
|---|---|---|---|
| **solar dive, 9.86 R_sun** (return excess 60 km/s) | 1.77 yr | $69 | $88 / **$104** |
| **solar dive, 4 R_sun** (90 km/s; shade unsolved) | 1.43 yr | $55 | $63 / **$67** |
| bulged hydrogen chamber, 2 Hz | | $165 | $302 / **$430** |
| methane AR100, 24 kg, 3 Hz | | $163 | $363 / **$634** |
| methane AR100, 24 kg | | $194 | $509 / **$967** |
| methalox | | $353 | $2323 / **>$5000** |

The valuation is the **gated rNPV** (ADR 0072): 10% a year, each flow weighted by the odds of
every milestone before it, and 15 synodic periods of growth before any mass is delivered. The
chamber rows are recomputed on exactly that footing (same odds structure, same growth phase,
the same lob climbing at 1.1 km/s). On a flat 30%, no odds and 9 synodic periods, the solar
dive is $355 / $982 (9.86 R_sun) and $174 / $421 (4 R_sun) against $1158 / $3098 for hydrogen.

**Growth falls from your 7-8x to 3.6x per 3S at 9.86 R_sun (4.9x at 4 R_sun).** The ledger
reproduces your 8.4x / 7.0x exactly from your own inputs, so the drop is the charges, not the
stage laws. Most of it is two terms your figure leaves out (D2-D5 move it far less):
1. **Pushing the launch unit into the parking orbit.** The Earth wave does it, at 400 km, on
   your overtake law: ~53 t of sheet per 1500 t unit, about as much as the departure spends.
2. **The role split.** A wave arrives as one pulse, so the push and the departure, ten days
   apart, need two waves. The push wave's ships must come home 10 days early. Solved on three
   knobs (excess, aim, their own climb-out excess) against the clock, the Earth intercept and
   Jupiter's phase, they do it on a faster climb: a 60 km/s climb-out needs 87 km/s.

**The seed is dear per kilogram and cheap per fleet kilogram.** Sent at the 10.43 km/s
excess on an expended, refuelled Starship it costs $1297 (cheap) to $4993 (dear) per kg. After
15 synodic periods it has grown 110x (9.86 R_sun) or 401x (4 R_sun), so it is $21 or $7 per
kilogram of fleet. A shorter growth phase is what makes it bite: at 10 synodic periods it is
$264 per fleet kg and the dear break-even is $224.

## Corrections to what was copied (G1, G4, G10)

| closure | departure excess | burn | Earth wave | disks' excess | ship perijove | disks' highest turning perijove |
|---|---|---|---|---|---|---|
| 9.86 R_sun, X 90 | 10.428 km/s | **4.15 -> 4.340 km/s** | 100.85 km/s | **14.78 -> 15.067 km/s** | 7.86 R_J | **6.72 R_J** |
| 4 R_sun, X 150 | 10.539 | **4.23 -> 4.417** | 157.42 | **14.16 -> 14.343** | 7.75 | **6.90** |

1. **G1: the closures are confirmed** (10.43 km/s at -13.5 degrees, 3.276 yr, 20.45 km/s boost,
   stream at ~101 degrees and ~103 km/s). X = 90 km/s is the right stand-in for a 100 km/s
   crossing: 99.4 km/s heliocentric at 1 AU, 100.8 km/s Earth-relative.
2. **The departure is 4.34 km/s, not 4.15.** The 4.15 is `departure_nozzle_ledger`'s convention,
   which starts the burn at parabolic speed at 200 km. From the 10-day orbit's 600 km periapsis
   it is 4.34 (4.42 at 4 R_sun), before the finite-burn loss (now charged: 40-60 m/s over a
   7-9 minute burn). About 3.5% more Earth-wave sheet per kilogram departing.
3. **G4: confirmed.** 5.3 km/s from near-escape is a ~12.0 km/s excess.
4. **G10: the disks need the true mirror excess, not the floor.** The mirror image of the
   ship's post-flyby state does meet it (on the symmetry line, 1.7 degrees past perihelion and
   1.7 degrees off head-on; a Lambert solve agrees). But the ship leaves Jupiter falling inward
   at 2.9 km/s, so the mirror needs `|(-v_t - V_J, v_r)|` = 15.07 km/s, not the 14.78 floor.
   The burn is ~10% larger.
5. **G10: the disks cannot turn above ~6.7 R_J.** They need a 73.7 degree turn. The ship's own
   unpowered flyby is pinned at 7.86 R_J, so the carrier must fly its own, lower perijove; your
   10 R_J rows do not turn. The ledger flies 6.5 R_J (just outside Io's orbit): 1.67 km/s, the
   propellant about 40% of carrier plus disks, ~1% of the ship. Lower perijoves are cheaper but
   deeper in the belts; at 2 R_J the break-even moves $1.
6. **D4's wording:** 22.7 kg is the mass *after* the burn. The parking orbit holds `MR_E` times
   that. `growth_cycle.csv` has it right.

## D1-D7

- **D1, the ledger.** Done, on this repository's closures, your canted law for the departure
  and your `v_e(k)` at the Sun, with the 400 km push on your overtake law. Above.
- **D2, the plates.** Both are sized by the survival floor (30 kg/m^2) scaled with pulse
  energy at fixed peak pressure (`m ∝ E^(2/3)`), against your ratings (the flat plate with skirt
  takes ~12.5 kg of 100 km/s sheet; the 0.5 paraboloid a 1 kg disk at 400 km/s), and checked for
  the inertia a 100 g cap needs over a 2 m absorber.
  - **The flat plate does the push and the departure, sized for the larger pulse, then diverts
    ~20 m/s into Jupiter's atmosphere** (author): 40-50 t, about 3% of the unit.
  - **The Sun plate is kept as dead mass** (G5 open: cutting it into sheets is an option, not
    taken). At 2 Hz over a 2-5 hour arc its pulses are small, so it is only 6-19 t, about 1-2.5%
    of the ship, against the 16% your fixed 20 m bowl implies.
  - **Both carry ablative film** at the spray cup's vapor-shielded rate per unit impulse, the
    Sun's scaled by its gas's energy per unit impulse. It costs well under 1%.
- **D3, dry mass.** Avionics and attitude control 1%, shade 0.5% (Parker-scaled at 9.86 R_sun;
  unsolved below), carrier dry 10% of the disks, collar 1 t, charge mounts 1% of the charges.
  Together they cost ~$3/kg at L1 and ~2% of growth.
- **D4, launch.** Every unit is lofted at the child's $25/kg, climbing at 1.1 km/s (x1.065), and
  pushed at 400 km. **The lob is the dominant cost: about $40-45 of every L1 kilogram, mostly the
  delivery lob.** At $50/kg the dear break-even is $166 (9.86 R_sun) and $116 (4 R_sun). The
  departure "charged from the pad" (G3) is what the ledger does: nothing starts in orbit free.
- **D5, impactors.** Disks at $1000/kg (author's baseline) plus their launch, push, departure
  and carrier: $2-4/kg at L1; $500-2000/kg moves the break-even $3-6. Sheets on the child's
  PuffSat book (body $3/kg plus steering package), guidance mass bracketed 2-10%: under $1/kg.
  Misses and failed unfurls are an attrition curve: 20% on the first return, falling on an 80%
  learning curve in cumulative sheets to a 3% floor.
- **D6, the seed.** Above: $1297-4993 per kg sent, $7-21 per fleet kilogram after 15 synodic
  periods. The first return is at 3S; there is no Earth loop on this route.
- **D7, the comparison.** Above. **Of D2-D5, the g cap moves doubling most** (at 20 g the plates
  get heavy: 3.6 -> 2.9x, dear break-even $104 -> $155). **The lob moves cost most.** Nothing
  else moves the break-even by more than ~$6.

**Depth.** Every depth from 4 to ~12 R_sun breaks even under $200/kg on the dear seed; deeper is
cheaper, because the same return speed needs less boost there. Beyond ~12-15 R_sun the disks
cannot make their Jupiter turn onto the mirror dive at any perijove, so the cycle ends on
geometry, not cost (`--sweep`, table in ADR 0072).

## New asks for this repository's inputs

1. **Extend the stage tables to lower k.** The optimum sits on their edges: the departure at
   k_E 5-6 (table from 5), the Sun at k_S 20 (table from 20) at 9.86 R_sun, the delivery push at
   k 5. The true optimum is probably a little cheaper.
2. **The canted plate's peak pressure per pulse.** The ledger assumes the canted sheet loads the
   flat plate like the overtake sheet, per kilogram of sheet at the same energy. The departure
   pulse (~4 MN s, ~60 kg of sheet at k 6) is the one that sizes the plate.
3. **Ablation per unit impulse against k,** at both closing speeds, for both plates.
4. **The canted law off the two closures.** The headline's Earth wave arrives at 75-100 km/s and
   ~95-101 degrees; the ledger scales your two rows by closing speed and interpolates the angle.
5. **The Sun boost's arc.** Over a 2-5 hour arc the disks strike up to ~10-30 degrees off
   head-on (the mirror orbit is head-on only on the symmetry line). The ledger charges it as
   `T^2/(24 tau^2)` of the boost plus Oberth's `T^2/(96 tau^2)`; your 10 h arc does not.

## Not modelled

- Development cost (R&D, test flights, factories, tracking). Neither architecture carries it,
  so the comparison stands but every absolute figure is flattered.
- The real ephemeris: the closures are circular and coplanar, and the two ship types' slightly
  different aims are charged by magnitude only.
- Operations at scale: lofting many 1500 t units inside a wave's passage, and the 10-day orbit's
  lunar perturbation (the child's ledger carries the same orbit).
- The sheets' late unfurl (they coast furled and must open in the last minute or two, ~10,000 km
  out, to land within tens of cm) is priced only as guidance mass and attrition.
