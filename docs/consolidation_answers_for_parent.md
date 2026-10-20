# Results 16-18 in the ledger, and the Moon: answers to S47 (for the parent)

Raised 2026-10-20 in `aim_is_all_you_need` (ADR 0077). **Written to be copied into
`Balloon-Pulse-Propulsion`.** Answers S47 items 1-7
(`docs/consolidation_asks_for_aim_repo.md` @ `94f8808`).

Reproduce with:

- `make solar-dive` (~1 min) for the headlines and sensitivities;
- `ARGS="--sweep --gaps"` (~3 min) for the depth sweep and the gap;
- `make plate-jupiter-loop` (~20 min) for the plate loop;
- `make plate-disposal ARGS="--drift"` (~10 s per gap; `--days 5` for the 5-day orbit)
  for item 7.

Your override script, run against `efe6121`, reproduces your printed $146 / $87 exactly.
With no flags it reproduces ADR 0076's $144 / $91. The script can be retired: the three
overrides are now real switches, on by default.

## Short answers

1. **Adopted:** Kapton to 140 km/s, carbon fabric above. One limit sets both the law's
   impactor and the price. The band switch is gone.
2. **Adopted:** nitrogen bags with a bladder, **5.4 t** per Earth plate. The bladder is
   calibrated per joule of `PV` to the sim's whole-floor bag. You had 5.5 t.
3. **Adopted:** the bowl at 2.5 F below the 122 deg rim line, from the sim's
   `band_far_bowl.csv`. Its exact capture ratio is 0.907 / 0.915 / 0.919 at k 10 / 20 /
   30, against your 0.909 / 0.912 / 0.922. Same dollars.
4. **theta = pi with nitrogen: +$4 / +$1 dear** ($150 / $88), as you found.
5. **9.86 R_sun: keep 60 km/s.** 75 km/s is $1 cheaper on both seeds at equal growth,
   which is a tie. 60 has the slower growth stream (99 against 115 km/s) and so more
   Kapton margin. The code headlines 60.
6. **The plate loop's steady column was a definition effect, plus a fleet that bled.**
   It came from one cycle, and that cycle fell right after the chain's 0.97x window. The
   chain as a whole delivers at ~$190/kg. Fixed: now **$189** at 0.5 g and **$270** at
   0.25 g, both below the break-evens. The break-evens barely move.
7. **The Moon:**
   - On the 10-day orbit, **phases 335-30 deg** put the ship inside the Moon's Hill
     sphere, about 1/6 of cycles.
   - **Rule:** fly those cycles on the 5-day gap, which never enters it and costs within
     $2 ($147 / $85 against $146 / $87). That costs the cadence under $1.
   - New: **even with the Moon far, the Sun's tide moves the 10-day departure by hours.**
     The orbit must be navigated every cycle. Its correction budget is unsolved.

## The headline

| closure | k push / dep / Sun / delivery, arc | growth stream | growth | doubling | steady | break-even cheap / **dear** |
|---|---|---|---|---|---|---|
| **9.86 R_sun, 60 km/s return** | 6.50 / 12.50 / 20.0 / 5.00, 4.97 h | 99.2 (Kapton) | **3.36** | **1.87 yr** | $94.1 | $124 / **$146** |
| **4 R_sun, 90 km/s return** | 7.55 / 12.25 / 20.0 / 5.00, 2.19 h | 137.1 (Kapton) | **5.00** | **1.41 yr** | $73.5 | $84 / **$87** |
| 9.86 R_sun, 90 km/s | 12.94 / 13.65 / 20.0 / 6.67, 5.26 h | 132.1 (Kapton) | 3.10 | 2.01 yr | $100.7 | $144 / $182 |
| 4 R_sun, 150 km/s | 25.91 / 13.04 / 20.0 / 12.53, 2.39 h | 207.8 (carbon) | 4.19 | 1.58 yr | $90.2 | $111 / $119 |

These are your printed figures (3.36x / $146; 4.99x / $87). 4 R_sun's optimum is flat,
and it lands at 5.00x from the recorded start. Your rerun's other rows should also stand
within $1, since the code is the same arithmetic. One change against your script: the bags
are 5.4 t, not 5.5 t, and the dear break-even rounds the same.

Each switch at the headline design (dear):

| closure | headline | Kapton only to 130 | helium, no bladder | 1 MPa cells | bowl at 1.5 F | theta = pi |
|---|---|---|---|---|---|---|
| 9.86, 60 | 3.36x, $146 | $146 | 3.38x, $144 | $145 | $146 | $150 |
| 4, 90 | 5.00x, $87 | $94 | 5.02x, $87 | $87 | 5.32x, $85 | $88 |

## Item 1: the sheet rule

- `Design.kapton_limit` is 140 km/s. The overtake law charges graphite above it, and the
  price book charges fabric above it.
- Result 15's narrow-sheet limit is a sensitivity row (130 km/s). It costs $7 at 4 R_sun,
  where the growth stream is 137.1 km/s, and nothing at 9.86.
- Result 18 also holds Kapton at 207.8 km/s on maraging only. We keep fabric there, as you
  do.
- The every-sheet-$100/kg row (your S43 table) stays as a comparison.

## Item 2: the bags

- nitrogen: gamma 1.4, R 296.8 J/(kg K) at 293 K;
- fibre wall at the peak pressure, as ADR 0076;
- bladder 1.3 t per 259.6 MJ of mean `PV` (5.0e-6 kg/J): the sim's whole-floor 52 kPa
  bag on its 63 t, 8 m, 2 pi / 3 spring.

That gives 5.4 t per Earth plate: about 3.0 t of nitrogen, 1.3 t of bladder and 1.2 t of
wall.
The sim has 5.7 t (2.9 gas + 1.3 bladder + 1.5 fibre); its fibre is the heavier.

Charging the bladder by `PV`, not by gas mass as your script did, keeps it the right size
on helium. The sim's helium bag carries 1.5 t of bladder, and `PV` scaling gives 1.55 t.
On nitrogen the two methods differ by 3%. The sim's 1 MPa four-cell bladder (0.13 t) is a
sensitivity row: $1 cheaper at 9.86 R_sun.

## Item 3: the bowl at 2.5 F

- `data/far_bowl_capture.csv` is the sim's `band_far_bowl.csv`, unchanged.
- Its 1.5 F captures equal our `departure_hole_law.csv` rows to every digit.
  `hole_ve_25 / hole_ve_15 = capture_25 / capture_15` to four digits, so the move changes
  only the capture.
- The ratio is applied below 122 deg (only 4 R_sun's 117 deg). It is log-linear in k and
  held flat outside 10-30; the optima sit at k_E 12.2.
- It costs 0.32x of growth at 4 R_sun (5.32 to 5.00x) but only $2 of dear break-even.
  Fewer, bigger cycles are partly offset by the cheaper Kapton.
- Your 0.909 / 0.912 / 0.922 came from the rounded `hole_ve` columns. Ours are the exact
  capture ratios, 0.907 / 0.915 / 0.919.

## Item 4: theta = pi

$150 against $146 at 9.86 R_sun, and $88 against $87 at 4 R_sun, with the headline design
held: +$4 / +$1 dear, as you found. Result 17 puts the gas column at 8.3 m at pi, against
18.6 m at 2 pi / 3.

## Item 5: 60 or 75 km/s at 9.86 R_sun

| return | growth stream | growth | doubling | steady | cheap / dear |
|---|---|---|---|---|---|
| 60 km/s | 99.2 km/s | 3.36 | 1.87 yr | $94 | $124 / $146 |
| 75 km/s | 114.8 | 3.36 | 1.87 | $92 | $123 / $145 |

A $1 tie. The code keeps 60. Its growth stream is 15 km/s slower, which is where Result 16
found the most margin on a 100 um Kapton face: 100 K under the rating at 99 km/s, against
26 K at 121.8. "Ties with a 75 km/s return" is the accurate sentence.

The depth sweep's dear optima: 4 R_sun X 90 $87; 6 R_sun X 90 $99; 8 R_sun X 75 $118;
9.86 R_sun X 75 $145 (X 60 $146); 12 R_sun X 60 $182; 15 R_sun X 75 $520.

## Item 6: the Jupiter-only loop's steady column

The loop's valuation took its steady cost from one cycle, six after the harvest. The loop's
windows are unequal: at the 0.5 g optimum they grow 0.97-0.98, 1.03-1.04, 1.27, 1.5 x 5,
1.68, 2.25 and 2.41x. Traced cycle by cycle at that optimum, the cost per kg delivered
runs **$125 to $1466** by window.

The tallied cycle was the one after window 0 (0.97x), whose surplus is thin, so it read
$1466. Averaged over a whole chain, cost over cargo is ~$190.

The trace also showed a small policy flaw. The harvest held the fleet with `min(G, 1)`, so
every sub-1x window shrank it and nothing rebuilt it: 3.56 to 3.32 units over three chains.

Both are fixed in `value_chain`:

- the fleet is held at its harvest size, and later windows rebuild it before delivering;
- the steady cost is one whole chain's cost over its cargo, from six cycles after the
  harvest.

The solar dive's valuation is unchanged. Its cycles are equal and grow 3-5x, so one cycle
is the steady state there.

| departure | growth per 3S | steady $/kg | cheap / dear |
|---|---|---|---|
| 0.25 g | 1.24 | $270 (was: no deliveries) | $3326 / $10505 |
| **0.5 g** | **1.65** | **$189** (was $1474) | **$759 / $1873** |
| 1 g | 1.56 | $214 | $1017 / $2627 |

Your rerun printed $761 / $1881 at 0.5 g. The fleet held level moves the break-evens by
under $10.

The steady column is now below the break-evens, as it should be: the break-evens carry
the seed and 15 synodic periods of growth. The verdict stands. At its best the loop is
$1873 against the Sun dive's $146.

## Item 7: the Moon and the push

We flew the ship's own parking orbit (`plate_disposal.parking_drift`), not the plate's:

- pushed at 400 km;
- raised two-body to 600 km at the two-body apoapsis time;
- flown through Earth, the Moon (circular; 0, 5.1 or 28.6 deg to the orbit) and the Sun's
  tide (two directions), with no correction.

The phase is the Moon's angle from the apoapsis at the apoapsis time, swept 0-355 deg in
5 deg steps.

| gap | apoapsis | phases inside the Moon's Hill sphere | departure perigee (uncorrected), elsewhere | timing |
|---|---|---|---|---|
| **10 d** | 385 000 km | **335-30 deg (~1/6)** | below the surface to 7 800 km | up to 30 h |
| **5 d** | 240 000 km | **none** | -356 to 1 673 km | up to 3.1 h |
| 3 d | 169 000 km | none | 398-806 km | up to 0.4 h |

"Inside the Hill sphere" means within 61 500 km of the Moon, where its pull rivals Earth's
and it, not Earth, sets the orbit. The band is the same at all three tilts. The ship flies
both halves of the orbit, so its band is wider than ADR 0076's ±20 deg for the plate, which
flies only the inbound half.

**The rule.** A cycle's waves arrive on dates the closure fixes years ahead, so the Moon's
phase at that cycle's apoapsis is known when its push wave is aimed.

- Fly the 10-day gap outside 335-30 deg.
- Inside the band, aim the push wave for 5 days. Its apoapsis comes 2.5 days later and
  145 000 km lower, and the Moon never captures it.
- The 5-day gap costs $1 more on the dear break-even at 9.86 R_sun ($147, 3.20x against
  3.36x) and $2 less at 4 R_sun ($85 against $87).
- So the rule costs the cadence under $1 and ~1% of growth at 9.86 R_sun: a sixth of the
  cycles lose 5%.

Flying 5 days always is the simpler alternative: free of the Moon, at a quarter of the
drift. That is your call; the paper flies 10 (ADR 0071's role split).

**New: the 10-day orbit must be navigated at every phase.** With the Moon switched off, the
Sun's tide alone moves the departure perigee's time by -1.5 to +3.1 h. With the Moon far
from apoapsis (90-270 deg), the perigee lands at 42 to 1 197 km against 600, and up to
5.4 h off. The departure wave arrives on its own clock, so these must be corrected each
cycle. A navigated orbit is ordinary practice. The cost is the open item.

**Not solved: the correction budget.**

- A single burn at the apoapsis time cannot close it. Its two in-plane components face
  three in-plane targets: perigee radius, time and apse angle.
- Freeing the burn's time closes it with the Moon far, at ~25 m/s, but only by moving the
  burn ±47 h from apoapsis.
- Adding a push-speed trim hits the perigee point on time at the wrong speed, up to
  1.7 km/s off.
- A navigated orbit needs a multi-burn, minimum-Δv targeting. We have not built it.

First-order scales, for sizing only (est.):

- at the 400 km push, 1 m/s moves the 10-day period by ~3.8 h, so the timing costs a
  few m/s of push trim;
- at apoapsis, 1 m/s moves the perigee ~77 km, so the 10-day drifts take ~10-100 m/s of
  methalox, the high end near the band's edges;
- the 5-day orbit needs far less.

## Suggested changes (for the parent to make)

- Drop "our rerun of" from the ADR 0076 figures that now come out of the code unchanged:
  3.36x / $146, 5.00x / $87 (your 4.99x; a flat optimum), the sweep, the gaps and the sensitivities. Cite ADR 0077.
  The seed table and plate-loop rows should be re-read from `make solar-dive` /
  `make plate-jupiter-loop`. The loop's steady column changes (item 6).
- `sec:charge_pusher_plume` (or wherever the bags are sized): "The bags are nitrogen, about
  5.4 t per Earth plate with their bladder."
- The 9.86 R_sun headline: "a 60 km/s return, which ties with 75 km/s at $146 and $145; we
  keep 60 for its slower growth stream, where Kapton has the most margin."
- The Moon paragraph. Replace "the schedule is unsolved" with: "On the 10-day orbit the
  Moon captures the ship's pass in about one cycle in six, when it is within 335-30 deg of
  the apoapsis. Those cycles fly the 5-day gap, chosen when the push wave is aimed, within
  $2 of the same break-even. Even away from the Moon the Sun's tide moves the departure by hours, so
  the parking orbit is navigated every cycle; its correction budget is not yet computed."
- `tab:no_sun_loop`: the steady column is now a whole chain's ($189 at 0.5 g), and "no
  steady deliveries" at 0.25 g becomes $270.
