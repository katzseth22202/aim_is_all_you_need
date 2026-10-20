# ADR 0077: Results 16-18 in the ledger, and the Moon's capture band

Date: 2026-10-20

Status: Accepted. Answers the parent's S47
(`docs/consolidation_asks_for_aim_repo.md` in `Balloon-Pulse-Propulsion` @ `94f8808`).
Amends ADR 0076 (the sheet rule, the bags' gas, the bowl at 117 deg and the plate loop's
steady column). Parent note: `docs/consolidation_answers_for_parent.md`.

## Context

The impact simulation's Results 16-18 (`puffsat_impact_simulation` @ `1bed1c6`, `9bcc8be`)
landed after ADR 0076:

- **Result 16:** Kapton holds at 115-122 km/s where it holds at 99. The departure bowl
  with its charge at 2.5 F, which a bare impactor needs below the rim line at 117 deg,
  keeps 0.907-0.919 of its 1.5 F capture.
- **Result 17:** the spring law holds nonlinearly. Its bags should be nitrogen, which
  warms 65-74 K over a push; helium warms 140-152 K and must radiate. They need a
  bladder, and the sheet should be as wide as the 9.5 m floor.
- **Result 18:** on that floor-wide sheet the computed backward plume fits u = 8 km/s at
  99 km/s, and Kapton holds at every printed closure through 137.1 km/s.

The parent reran ADR 0076's ledger with three run-time overrides and printed 3.36x / $146
(9.86 R_sun) and 4.99x / $87 (4 R_sun). S47 asks us to adopt those three as real
switches, and to answer four questions: theta = pi with nitrogen, 60 against 75 km/s at
9.86 R_sun, the plate loop's steady column, and a launch-window rule for the Moon.

The parent's script, run against `efe6121`, reproduces its printed figures exactly. With
no flags it reproduces ADR 0076's 3.38x / $144 and 5.31x / $91.

## Decision

1. **Sheets are Kapton to 140 km/s, carbon fabric above** (`KAPTON_LIMIT`,
   `Design.kapton_limit`). One limit sets both the overtake law's impactor
   (`sheet_material`) and the sheet's price (`SolarDivePrices.sheet_price`).
   - The 100-130 km/s band switch (`band_carbon`, `KAPTON_HOLDS`) is retired. Result 16
     ran the band, and Result 18 holds Kapton above it.
   - The 4 R_sun headline's 137.1 km/s growth sheets are Kapton.
   - Result 18 also holds Kapton at 208 km/s on maraging only. We keep fabric there, as
     the parent does.
   - Result 15's narrower-sheet limit is a sensitivity row (130 km/s).
2. **The bags are nitrogen, with a bladder** (`BAG_GASES`, `Design.bag_gas`,
   `BLADDER_PER_PV`, `Design.bladder`).
   - gamma 1.4 and R 296.8 J/(kg K) move both the mean `PV` and the gas mass.
   - The bladder is charged per joule of `PV`. It is calibrated to the simulation's
     whole-floor 52 kPa bag: 1.3 t on its 63 t, 8 m spring's 259.6 MJ
     (`gas_spring_design.csv`).
   - That comes to **5.4 t** per Earth plate, against the simulation's 5.7 t
     (1.5 fibre + 2.9 gas + 1.3 bladder) and the parent's 5.5 t. The parent scaled the
     bladder by gas mass instead; on nitrogen the two differ by 3%.
   - Helium with no bladder (ADR 0076) and the simulation's 1 MPa four-cell bladder
     (0.13 t) are sensitivity rows.
3. **The bowl's charge sits at 2.5 F below the rim line** (`RIM_LINE_DEG` = 122 deg,
   `StageLaws.far_focus`, `Design.far_focus`, `data/far_bowl_capture.csv`).
   - Result 16's `band_far_bowl.csv` is vendored unchanged. Its 1.5 F captures are this
     repository's `departure_hole_law.csv` rows to every digit. Moving the charge changes
     only the capture: `hole_ve` falls by the capture's ratio to four digits.
   - The 117 deg ratio (0.907 / 0.915 / 0.919 at k 10 / 20 / 30) is applied to the
     interpolated 1.5 F capture, log-linearly in k and held flat outside 10-30. The
     optima sit at k_E 12.2-13.7.
   - The parent's override used 0.909 / 0.912 / 0.922, read from the rounded `hole_ve`
     columns. It lands on the same dollars.
4. **theta = pi with nitrogen** costs +$4 dear at 9.86 R_sun ($150) and +$1 at 4 R_sun
   ($88), with the headline design held. This confirms the parent's +$4 / +$1. Result 17
   puts the gas column at 8.3 m there, against 18.6 m at 2 pi / 3.
5. **9.86 R_sun keeps the 60 km/s return.**
   - On these switches X 75 is $145 dear against X 60's $146, at equal growth (3.36x).
     Its cheap break-even is $1 lower too ($123 / $124).
   - X 60's growth stream is 99.2 km/s, against X 75's 114.8. That leaves more Kapton
     margin: Result 16's 100 um face peaks 100 K under the rating at 99 km/s and 26 K
     under it at 121.8.
   - So the code keeps X 60 as the headline, and the tie is printed.
   - At 4 R_sun, X 90 stays the optimum ($87, against X 110's $90 and X 75's $94).
6. **The plate loop's steady column was a definition effect, plus a fleet that bled.**
   `value_chain` took the steady cost from one cycle, six after the harvest, on a chain
   whose windows grow 0.97-2.41x.
   - At the 0.5 g optimum, the per-cycle cost runs **$125-1466/kg** across the windows.
   - The tallied cycle was the one after window 0's 0.97-0.98x, whose surplus is thin.
     So it read $1466 where the chain as a whole averages ~$190.
   - Separately, the harvest held the fleet with `min(G, 1)`. A sub-1x window shrank it
     and nothing rebuilt it: 3.56 to 3.32 units over three chains.
   - Fix: the fleet is held at its harvest size, so later windows rebuild it before
     delivering a surplus. The steady cost is now one whole chain's cost over its cargo.
   - The solar dive's `value_design` is unchanged. Its cycles are equal and grow 3-5x,
     so one cycle is the steady state there.
7. **The Moon: the ship's own parking orbit, flown** (`plate_disposal.parking_drift`,
   `make plate-disposal ARGS="--drift"`).
   - The ship is pushed at 400 km and raised two-body to 600 km at the two-body apoapsis
     time. It flies through Earth, the Moon (circular, at 0, 5.1 or 28.6 deg to the
     orbit) and the Sun's tide (two directions), and is not corrected.
   - Phases run 0-355 deg on a 5 deg grid, measured at the apoapsis time.
   - **10-day gap: the Moon captures the pass at phases 335-30 deg**, about 1/6 of
     cycles. Capture means passing inside its 61 500 km Hill sphere, where it rivals
     Earth's pull. The band is the same at all three tilts.
     - The ship flies both halves of the orbit, so the band is wider than ADR 0076's
       ±20 deg for the plate, which flies only the inbound half.
     - Outside the band the departure perigee still lands anywhere from below the
       surface to 7 800 km, and up to 30 h off its time. The extremes are at the band's
       edges.
   - **Even with no Moon, the Sun's tide alone moves the 10-day departure by -1.5 to
     +3.1 h.** So the 10-day orbit must be navigated at every phase, not only near the
     Moon.
   - **5-day gap: no phase enters the Hill sphere.** The apoapsis is at 240 000 km. The
     perigee drifts -356 to +1 673 km and up to 3.1 h. **3-day:** 398-806 km and up to
     0.4 h.
   - **The rule.** A cycle's waves arrive on dates the closure fixes years ahead, so the
     phase is known when its push wave is aimed.
     - Fly 10 days outside 335-30 deg. Inside the band, aim the push wave for 5 days.
     - The 5-day gap costs $1 more on the dear break-even at 9.86 R_sun ($147, 3.20x
       against 3.36x), and $2 less at 4 R_sun ($85 against $87).
     - So the rule costs the cadence under $1 on the dear break-even (a sixth of the
       cycles at ±$2), and about 1% of growth at 9.86 R_sun (a sixth of 5%).
     - Flying 5 days always is the simpler alternative: it is free of the Moon, at a
       quarter of the drift.
   - **Not solved: the correction itself.** Two single-burn targetings were tried, and
     neither closes.
     - A burn at the apoapsis time has two in-plane components for three in-plane
       targets (radius, time, apse angle).
     - Freeing the burn time solves it with the Moon far, at ~25 m/s, but only by moving
       the burn ±47 h from apoapsis. With a push-speed trim added, it hits the perigee
       point on time at the wrong speed (up to 1.7 km/s off).
     - A navigated orbit needs a multi-burn, minimum-Δv targeting.
     - First-order scales, for sizing only (est.): at the 400 km push, 1 m/s moves the
       10-day period by ~3.8 h. At apoapsis, 1 m/s moves the perigee ~77 km. That
       suggests a few m/s for timing and ~10-100 m/s of methalox across the 10-day
       drifts, much less on 5 days.

## Results

Each closure is optimized for the dear-seed break-even. "X" is the return excess, the
Sun-relative hyperbolic excess speed the Sun boost leaves.

| closure | growth | doubling | steady | break-even cheap / dear | ADR 0076 |
|---|---|---|---|---|---|
| **9.86 R_sun, X 60** | **3.36** | **1.87 yr** | $94.1 | $124 / **$146** | 3.38x, $144 |
| **4 R_sun, X 90** (Kapton) | **5.00** | **1.41 yr** | $73.5 | $84 / **$87** | 5.31x, $91 |
| 9.86 R_sun, X 90 | 3.10 | 2.01 | $100.7 | $144 / $182 | 3.12x, $188 |
| 4 R_sun, X 150 | 4.19 | 1.58 | $90.2 | $111 / $119 | 4.22x, $118 |

Each switch, held at the headline design (dear break-even, from the sensitivity table):

| closure | headline | Kapton only to 130 | helium, no bladder | 1 MPa cells | bowl at 1.5 F | theta = pi |
|---|---|---|---|---|---|---|
| 9.86, X 60 | 3.36x, $146 | $146 | 3.38x, $144 | $145 | $146 | $150 |
| 4, X 90 | 5.00x, $87 | $94 | 5.02x, $87 | $87 | 5.32x, $85 | $88 |

- The nitrogen bags cost $2 at 9.86 R_sun and under $1 at 4 R_sun.
- The 2.5 F bowl costs 0.32x of growth at 4 R_sun, the only closure that departs below
  the rim line, but only $2 of dear break-even.
- Kapton at 137 km/s saves $7 there.

Gaps (each re-optimized):

| gap | 9.86 R_sun, X 60 | 4 R_sun, X 90 |
|---|---|---|
| 3 d | 2.97x, $162 | 4.55x, $87 |
| 5 d | 3.20x, $147 | 4.83x, $85 |
| 10 d | 3.36x, $146 | 5.00x, $87 |

Depth sweep, dear optimum per depth: 4 R_sun X 90 $87; 6 R_sun X 90 $99; 8 R_sun X 75
$118; 9.86 R_sun X 75 $145 (X 60 $146); 12 R_sun X 60 $182; 15 R_sun X 75 $520.

**The Jupiter-only plate loop** (the steady column now a whole chain):

| departure | ADR 0076, dear | parent's rerun: steady, cheap / dear | now: growth, steady, cheap / dear |
|---|---|---|---|
| 0.25 g | 1.27x, $9436 | no steady deliveries | 1.24x, $270, $3326 / $10505 |
| 0.5 g | 1.68x, $1654 | $1474, $761 / $1881 | 1.65x, $189, $759 / $1873 (best) |
| 1 g | 1.59x, $2408 | | 1.56x, $214, $1017 / $2627 |

Holding the fleet level moves the break-evens by under $10 against the parent's rerun.
ADR 0075's verdict stands: the Sun dive is needed.

## Consequences

- `make solar-dive` reproduces every solar-dive figure above in ~1 min, and
  `ARGS="--sweep --gaps"` in ~3 min. `make plate-jupiter-loop` reproduces the loop in
  ~20 min, and `make plate-disposal ARGS="--drift"` the drift in ~10 s.
- Sensitivities replace the band switch with "Kapton only to 130 km/s". They add helium
  bags, the 1 MPa bladder and the 1.5 F bowl.
- Still open:
  - the navigated parking orbit's correction budget (item 7);
  - whether the paper flies the 5-day gap always;
  - the 18.6 m gas column (theta = pi or a reservoir);
  - the lossless spring's drift under pulse scatter (the parent's S48 to the simulation).
