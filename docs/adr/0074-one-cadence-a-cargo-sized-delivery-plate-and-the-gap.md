# ADR 0074: One cadence, a cargo-sized delivery plate, and the gap as a parameter

Date: 2026-10-19

Status: Accepted. Answers the parent's S39 (`docs/two_plates_followup_asks_for_aim_repo.md`
in `Balloon-Pulse-Propulsion` @ `988a9e3`). Amends ADR 0073: its headlines become the
comparison, its delivery moves to 2 Hz and 1 g, and its 156 t delivery plate is resized. Its
push, bowl, visible sign, Sun stage, seed and valuation are unchanged. Parent note:
`docs/two_plates_followup_answers_for_parent.md`.

## Context

The parent took ADR 0073 with three choices:

- every Earth push, delivery included, flies 2 Hz at 1 g;
- the headline is the cheapest closure at each depth: 9.86 R_sun at X 60 and 4 R_sun at X 90;
- the faster growth waves probably need graphite sheets.

It printed a rerun of this code and asked for eight things. They were: make the 2 Hz delivery
the default and confirm the rerun; size the delivery plate for the 7.35 MN s pulse at the
cargo's g cap; print both stream speeds; price graphite sheets; dispose of the push plate; fix
two sensitivity rows; print `corrections_report` for the flown headline; and make the
growth-to-departure gap a real parameter.

## Decision

1. **The paper's rerun reproduces exactly.** Before any change, optimizing from
   `replace(HEADLINE_STARTS[key], slow_delivery=True)` gives every figure in S39's table. At
   9.86 R_sun, X 60 that is 3.81x, 1.70 yr, $98.6 steady and $122 / $134. At 4 R_sun, X 90 it
   is 5.72x, 1.30 yr, $78.2 and $88 / $89. The X 90 and X 150 rows reproduce too.
   `Design.slow_delivery` now defaults to True. `HEADLINES` are (9.86, 60) and (4, 90), and
   ADR 0073's (9.86, 90) and (4, 150) are `COMPARISONS`.
2. **The delivery plate is the absorber bound at the cargo's cap.** It uses
   `I^2 / (2 s M a)`, the same law as the push plate, with `M` the pushed stack at the end, a
   2 m stroke and `a` = 25 g. The 25 g is the parent's cited 10 ms rack-server shock rating,
   read as a peak (`CARGO_G_CAP`, `Design.delivery_g_cap`). At 7.35 MN s the plate is
   **44-51 t**. The cargo peaks at 25 g at the push's end and 18-21 g at its start, over
   **24-28 ms** a pulse. The parent's 6-8 g estimate held the 156 t plate. At 25 g the child's
   4 Hz, 12 MN s pulse needs 135 t, so the child's 150 t was close to this cap already.
3. **The delivery flies its own cadence.** `delivery_unit` takes 2 Hz, 1 g from its own lob
   climb when `slow_delivery` is set (x1.488, on the L1 transfer's end speed), and the child's
   4 Hz, 3.26 g from the 1.1 km/s lob otherwise. It no longer reads `Design.push_g` or the
   growth unit's `lob_climb`. That fixes item 6: the "growth lob climb not charged" and
   "growth push at 3.26 g" rows now change only the growth push.
4. **Push-plate disposal is charged.** `plate_disposal_dv` nudges the plate at apoapsis to a
   100 km perigee. That is **4.2 m/s** from 400 km (5.0 m/s from 460 km) on the 10-day orbit,
   6.6 m/s at 5 days and 9.2 m/s at 3 days. It costs 14-16 kg of storable, carried up and
   priced. It rounds to zero.
5. **Graphite sheets are a price sensitivity.** `GRAPHITE_SHEET_BODY = $20/kg`. Flexible
   graphite foil is listed at Rs 350-1000/kg (0.35-3 mm) and Rs 1800/kg (0.5 mm), IndiaMART
   asking prices, about $4-21/kg. A 100 um film should be dearer per kg than either. The
   parent's $100/kg is the dear row. The heavier point-blast push plate waits on S40.
6. **The gap is carried by the push route.** `DiveRoute.early_days` is set by `push_route`
   (which refuses a non-positive gap). `fly_unit` and `value_design` read it, and pass it
   explicitly to `departure_burn`, `apoapsis_methalox`, `departure_loss`, the push's end speed
   and the schedule. `make solar-dive ARGS="--parking-days N"` flies any gap, and `--gaps`
   sweeps 3 / 5 / 10 days. The `lru_cache` on `push_route` keys on the gap, so an explicit
   call works. The bug was overriding the module default.
7. `corrections_report` prints all four closures. The flown headline has no simulation figure
   to check against: 4.209 km/s, 15.052 km/s and 6.79 R_J at 9.86 R_sun, X 60.
8. Both stream speeds are printed in `headline_report` and the sweep.

## Results (`make solar-dive`, ~1 min; `--gaps` ~5 min; `--sweep` ~25 min)

| closure | k push / dep / Sun / delivery, arc | streams growth / dep | growth | doubling | steady | break-even cheap / **dear** |
|---|---|---|---|---|---|---|
| **9.86 R_sun, X 60** | 7.06 / 12.54 / 20.0 / 5.57, 4.89 h | 99.2 / 76.2 | **3.80** | 1.70 yr | $86.7 | $107 / **$118** |
| **4 R_sun, X 90** | 6.62 / 12.07 / 20.45 / 5.01, 2.16 h | 137.1 / 102.6 | **5.71** | 1.30 yr | $68.9 | $77 / **$79** |
| 9.86 R_sun, X 90 | 7.94 / 13.90 / 20.0 / 6.83, 5.12 h | 132.1 / 101.4 | 3.27 | 1.92 yr | $90.1 | $122 / $151 |
| 4 R_sun, X 150 | 8.23 / 13.71 / 20.0 / 7.33, 2.32 h | 207.8 / 157.8 | 3.85 | 1.68 yr | $77.5 | $98 / $110 |

Growth streams are at 400 km and departure streams at 600 km, both in km/s. Against the
parent's rerun, the lighter delivery plate takes **$12-16/kg** off the steady cost and
**$10-21** off the dear break-even. Growth moves by 0.01 at most: the disposal propellant,
and k re-optimized.

**The gap** (each re-optimized):

| closure | gap | turn | growth stream | growth | doubling | steady | break-even cheap / dear |
|---|---|---|---|---|---|---|---|
| 9.86 R_sun, X 60 | 10 d | 0.332 | 99.2 | 3.80 | 1.70 | $86.7 | $107 / $118 |
| | 5 d | 0.535 | 87.1 | 3.71 | 1.73 | $84.6 | $104 / $115 |
| | 3 d | 0.759 | 82.6 | 3.52 | 1.81 | $85.7 | $107 / $120 |
| 4 R_sun, X 90 | 10 d | 0.311 | 137.1 | 5.71 | 1.30 | $68.9 | $77 / $79 |
| | 5 d | 0.501 | 119.0 | 5.60 | 1.32 | $67.5 | $75 / $77 |
| | 3 d | 0.712 | 112.3 | 5.35 | 1.35 | $67.6 | $75 / $77 |

This confirms the parent's paper-side run (its turns, streams and growth to the second
decimal). The dollars are lower by the delivery plate. **A 5-day gap costs 2-3% of growth
and is $2 cheaper dear.** It brings the 4 R_sun growth stream to 119 km/s.

**Depth sweep** (dear optimum per depth): 4 -> X 90 (5.71, $79; X 110 ties at $79),
6 -> X 75 (4.82, $90; X 90 also $90), 8 -> X 60 (4.13, $107), 9.86 -> X 60 (3.81, $118),
12 -> X 60 (3.43, $139), 15 -> X 75 (2.49, $337). The parent's sweep picked X 90 at 6 R_sun
and X 75 at 8. With the lighter plate, the slower return now ties or wins there.

### Sensitivities (9.86 R_sun, X 60, design held; dear break-even)

| case | growth | steady | dear |
|---|---|---|---|
| headline | 3.80 | $87 | $118 |
| growth lob climb not charged (fixed) | 3.80 | $81 | $109 |
| growth push at 3.26 g (fixed) | 3.47 | $85 | $125 |
| deliveries at the child's 4 Hz / 3.26 g (135 t plate, no climb) | 3.80 | $76 | $110 |
| delivery g cap 10 / 50 | 3.80 | $96 / $84 | $130 / $115 |
| graphite sheets $20/kg / $100/kg | 3.80 | $88 / $97 | $121 / $134 |
| departure at 0.5 g | 3.94 | $86 | $115 |

At 4 R_sun the graphite rows give $80 / $88 dear, against the headline's $79. That matches
the parent's bound of +$8 at 4 R_sun, X 150. The delivery's lob is now the largest line at
both depths ($52-58 of each L1 kg). The growth lob is $10-18.

## Consequences

- The paper's figures move by the delivery plate alone: $134 -> **$118** and $89 -> **$79**
  dear. Everything else it printed stands.
- Graphite growth sheets would cost $3-16 dear per L1 kg at $20-100/kg. They are not decisive.
  The open question is the plate (S40).
- A 5-day gap is the cheaper setting at both depths, by about $2 dear. At 4 R_sun it may keep
  Kapton viable (119 km/s). Taking it would also move the child's ledger off ADR 0071's
  10 days, which is the author's call.

## Limits

- A 25 g peak over 24-28 ms is a longer pulse than the 10 ms rating. It drives the cargo's low
  modes harder than the rated half-sine does. The g cap is a stated assumption, and the 10 g
  row shows what a stricter reading costs (+$12 dear).
- The delivery plate's area and survival floor are not checked. A 45-51 t plate takes the
  same pulse as the 11-12 t push plate, so its pressure margin is larger.
- Disposal is two-body. The Moon perturbs a 385 000 km apoapsis strongly, and the 100 km
  perigee is not propagated.

## Reproduction

`make solar-dive` (headlines, comparisons, sensitivities), `ARGS="--gaps"`, `ARGS="--sweep"`.
Tests: `tests/test_solar_dive_growth.py` covers the delivery plate's absorber bound, the
disposal nudge, the gap carried by the push route and the 3/5/10-day streams and turns.
