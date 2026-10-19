# The two-plate cycle on one cadence: answers to S39 (for the parent)

Raised 2026-10-19 in `aim_is_all_you_need` (ADR 0074). **Written to be copied into
`Balloon-Pulse-Propulsion`.** Answers S39 items 1-8
(`docs/two_plates_followup_asks_for_aim_repo.md` @ `988a9e3`). Reproduce with
`make solar-dive` (~1 min), `ARGS="--gaps"` (~5 min) and `ARGS="--sweep"` (~25 min).

## Item 1: the rerun is confirmed, then moved by item 2

Before any change, your paper-side rerun reproduces to every printed digit (3.81x, 1.70 yr,
$98.6, $122 / $134 at 9.86 R_sun, X 60; 5.72x, 1.30 yr, $78.2, $88 / $89 at 4 R_sun, X 90; and
both faster rows). The 2 Hz delivery is now the default. `HEADLINES` are X 60 and X 90, and
X 90 and X 150 are the comparison.

With the delivery plate resized (item 2), the figures to print are:

| closure | k push / dep / Sun / delivery, arc | growth stream / dep stream | growth | doubling | steady | break-even cheap / **dear** |
|---|---|---|---|---|---|---|
| **9.86 R_sun, X 60** | 7.06 / 12.54 / 20.0 / 5.57, 4.89 h | 99.2 / 76.2 km/s | **3.80** | **1.70 yr** | $86.7 | $107 / **$118** |
| **4 R_sun, X 90** | 6.62 / 12.07 / 20.45 / 5.01, 2.16 h | 137.1 / 102.6 | **5.71** | **1.30 yr** | $68.9 | $77 / **$79** |
| 9.86 R_sun, X 90 | 7.94 / 13.90 / 20.0 / 6.83, 5.12 h | 132.1 / 101.4 | 3.27 | 1.92 yr | $90.1 | $122 / $151 |
| 4 R_sun, X 150 | 8.23 / 13.71 / 20.0 / 7.33, 2.32 h | 207.8 / 157.8 | 3.85 | 1.68 yr | $77.5 | $98 / $110 |

Depth sweep, dear optimum per depth: 4 -> X 90 (5.71, $79; X 110 ties); 6 -> X 75 (4.82,
$90; X 90 ties); 8 -> X 60 (4.13, $107); 9.86 -> X 60 (3.81, $118); 12 -> X 60 (3.43, $139);
15 -> X 75 (2.49, $337).

## Item 2: the delivery plate

It is the absorber bound `I^2 / (2 s M a)`, the push plate's law, at the cargo's cap.
`M` is the pushed stack at the end, `s` = 2 m, and `a` = **25 g** (your rack-server rating,
read as a peak).

- **Plate 44-51 t** (9.86 R_sun, X 60: 50.9 t), not 156 t.
- **Peak 25 g at the push's end, 18-21 g at its start, over 24-28 ms a pulse.** The 156 t
  plate would give your 6-8 g over ~85 ms. The lighter plate is the cheaper use of the cap.
- The child's 4 Hz, 12 MN s pulse needs 135 t at the same cap. The child's 150 t was near it.
- **The cost falls $10-21 dear**: $134 -> $118 and $89 -> $79.
- Caveat: a 24-28 ms pulse is longer than the 10 ms rating, so it drives a server's low modes
  harder. At a 10 g cap the break-even is $130 dear, and at 50 g it is $115.

## Item 3: both streams

Both are printed in `headline_report` and the sweep. Your figures are confirmed: growth
stream at 400 km 99.2 / 132.1 km/s at 9.86 R_sun (X 60 / 90) and 137.1 / 207.8 at 4 R_sun
(X 90 / 150). The departure streams at 600 km are 76.2 / 101.4 and 102.6 / 157.8.

## Item 4: graphite growth sheets

`GRAPHITE_SHEET_BODY = $20/kg`. Flexible graphite foil is listed at Rs 350-1000/kg for
0.35-3 mm rolls and Rs 1800/kg at 0.5 mm (IndiaMART asking prices, about $4-21/kg). A 100 um
film will be dearer per kg. At 9.86 R_sun, X 60 the dear break-even moves $118 -> $121
($20/kg) and $134 ($100/kg). At 4 R_sun, X 90 it moves $79 -> $80 and $88. Your +$8 bound at
X 150 agrees. The heavier point-blast push plate waits on S40.

## Item 5: push-plate disposal

The 10-day orbit's apoapsis is 385 000 km. A **4.2 m/s** nudge there drops the perigee from
400 km to 100 km (5.0 m/s from 460 km). It costs 14-16 kg of storable, now charged. That is
6.6 m/s on a 5-day orbit and 9.2 m/s on a 3-day orbit. This is two-body only: the Moon
perturbs a lunar-distance apoapsis strongly, so the 100 km perigee is not propagated.

## Item 6: the sensitivity rows

The delivery now flies its own cadence and lob. It reads neither the growth push's `push_g`
nor its `lob_climb`. At 9.86 R_sun, X 60 the rows are: "growth lob climb not charged" $81
steady / $109 dear; "growth push at 3.26 g" $85 / $125 (growth 3.47); "deliveries at the
child's 4 Hz / 3.26 g" $76 / $110.

## Item 7: `corrections_report`

It now prints the flown headline. At 9.86 R_sun, X 60: 10.239 km/s excess, **4.209 km/s**
burn, 75.5 km/s Earth excess, 9.79 km/s Sun boost, **15.052 km/s** true mirror excess,
7.97 R_J ship perijove, **6.79 R_J** for the disks. At 4 R_sun, X 90: 4.444, 14.322, 7.06.

## Item 8: the gap

The push route carries its gap (`DiveRoute.early_days`), and every use reads it from there.
`make solar-dive ARGS="--parking-days N"` flies any gap, and `--gaps` sweeps 3 / 5 / 10 days,
each re-optimized:

| closure | gap | turn | growth stream | growth | doubling | steady | break-even cheap / dear |
|---|---|---|---|---|---|---|---|
| 9.86 R_sun, X 60 | 10 d | 0.332 | 99.2 | 3.80 | 1.70 | $86.7 | $107 / $118 |
| | 5 d | 0.535 | 87.1 | 3.71 | 1.73 | $84.6 | $104 / $115 |
| | 3 d | 0.759 | 82.6 | 3.52 | 1.81 | $85.7 | $107 / $120 |
| 4 R_sun, X 90 | 10 d | 0.311 | 137.1 | 5.71 | 1.30 | $68.9 | $77 / $79 |
| | 5 d | 0.501 | 119.0 | 5.60 | 1.32 | $67.5 | $75 / $77 |
| | 3 d | 0.712 | 112.3 | 5.35 | 1.35 | $67.6 | $75 / $77 |

Your turns, streams and growth are confirmed. The dollars are lower by item 2. **5 days is
the cheaper gap at both depths by ~$2 dear, for 2-3% less growth.** It brings the 4 R_sun
growth stream to 119 km/s, where S40 may keep Kapton viable.
