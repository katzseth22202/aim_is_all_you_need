# Plates on springs, the overtake law at speed, carbon fabric: answers to S45 and S43 (for the parent)

Raised 2026-10-19 in `aim_is_all_you_need` (ADR 0076). **Written to be copied into
`Balloon-Pulse-Propulsion`.** Answers S45 items 1-4 and S43 items 1-5
(`docs/two_plates_followup_asks_for_aim_repo.md` @ `5d1379c`).

Reproduce with:

- `make solar-dive` (~1 min) for the headlines and sensitivities;
- `ARGS="--sweep --gaps"` (~2 min) for the depth sweep and the gap;
- `make plate-disposal` (~5 min) for S43 item 5;
- `make plate-jupiter-loop` (~10 min) for the plate loop.

## Terminology: "X 60" means a 60 km/s return

Our reports label each closure by its depth and **X**, the **return excess**. That is the
**hyperbolic excess speed relative to the Sun** that the perihelion boost leaves the ship
with as it climbs back out of the Sun's well. "9.86 R_sun, X 60" is a 9.86 R_sun dive with
a 60 km/s return.

X is easy to misread in two ways:

- **It is Sun-relative, not Earth-relative.** The returning wave meets Earth at a
  different, larger excess: 75.5 km/s for X 60 at 9.86 R_sun, and 102 km/s for X 90 at
  4 R_sun. Earth's gravity then speeds it up further, to the growth stream at 400 km
  (99 / 137 km/s) and the departure stream at 600 km (76 / 103 km/s).
- **It is the return leg, not the outbound one.** The outbound Earth departure excess is
  10.2-10.6 km/s.

Your CONTEXT.md already keeps "X 60" out of the prose ("a 60 km/s return"). Two places
need a sentence:

1. `sec:charge_pusher_cycle` defines the return as "the speed the ship has left after
   climbing back out of the Sun's well". We suggest naming it there as the hyperbolic
   excess speed relative to the Sun. The paper already defines hyperbolic excess speed
   where it first quotes the Jupiter departure speeds. Add that it is
   not the stream's speed at Earth, and give the 75.5 km/s example.
2. Where `sec:charge_pusher_depth` writes "A return speed $X$ costs a boost that grows as
   $X^2$", define $X$ at that first use as that Sun-relative hyperbolic excess speed.

If any table or caption uses "X 60" as a column label, define it in the caption the same
way.

## The headline (S45 + S43 together)

Each closure is optimized for the dear-seed break-even.

| closure | k push / dep / Sun / delivery, arc | growth stream / delivery stream | growth | doubling | steady | break-even cheap / **dear** |
|---|---|---|---|---|---|---|
| **9.86 R_sun, 60 km/s return** | 6.50 / 12.53 / 20.0 / 5.00, 5.44 h | 99.2 / 76.3 km/s (Kapton) | **3.38** | **1.86 yr** | $93.4 | $123 / **$144** |
| **4 R_sun, 90 km/s return** | 8.28 / 11.16 / 20.0 / 5.00, 2.17 h | 137.1 (carbon) / 102.6 | **5.31** | **1.36 yr** | $76.5 | $88 / **$91** |
| 9.86 R_sun, 90 km/s | 12.99 / 13.57 / 20.0 / 6.72, 5.23 h | 132.1 (carbon) / 101.4 | 3.12 | 2.00 yr | $104.7 | $150 / $188 |
| 4 R_sun, 150 km/s | 25.93 / 12.95 / 20.0 / 12.55, 2.41 h | 207.8 (carbon) / 157.8 | 4.22 | 1.58 yr | $89.6 | $110 / $118 |

Against ADR 0074's printed 3.80x / $118 and 5.71x / $79 ($88 on fabric), the dear
break-even moves step by step, each step re-optimized:

| closure | ADR 0074 | + spring plates (S45) | + the sim's law (S43.1) | + carbon fabric (S43.2) |
|---|---|---|---|---|
| 9.86, 60 | 3.80x, $118 | 3.50x, $134 | 3.38x, $144 | 3.38x, $144 |
| 4, 90 | 5.71x, $79 | 5.36x, $83 | 5.31x, $84 | 5.31x, $91 |
| 9.86, 90 | 3.27x, $151 | 3.05x, $176 | 3.12x, $179 | 3.12x, $188 |
| 4, 150 | 3.85x, $110 | 3.66x, $121 | 4.22x, $112 | 4.22x, $118 |

## S45 item 1: the spring law

Every plate now rides your law. We confirm it by direct integration, including a craft
only six times the plate, starting from your half-size first pulse:

- the peak/mean ratio is `(theta/2)/sin(theta/2)`, 1.209 at 2 pi / 3;
- the swing is `F tan(theta/4) / (2 m theta nu^2)`;
- the spring force never leaves compression below theta = pi;
- the mean force on the craft is `F M / (m + M)`.

One refinement. **`m x stroke` does not depend on the craft's mass**, because the relative
velocity step is `J / m` whatever the craft weighs. The reduced mass enters only the
stiffness, `k = mu (theta nu)^2`, and through it the bags.

At your 8 m stroke and theta = 2 pi / 3:

| plate | m x stroke | plate | bags | stroke flown | peak on the craft |
|---|---|---|---|---|---|
| growth push (2 Hz, 14.7 MN) | 507 t m | 63.4 t | 1.9 t | 8 m | 1.56-1.74 g (end of push) |
| departure bowl (1 Hz, 1.9-2.3 MN) | 260-310 t m | the shared 63.4 t | shared | 4.2-4.9 m | 0.34-0.38 g |
| delivery (2 Hz, 14.7 MN) | 507 t m | 63.4 t | 1.9 t | 8 m | 1.21 -> 1.52-1.66 g |
| Sun plate (2 Hz, 0.3-1.9 MN) | 17-66 t m | survival floor, 5-29 t | < 0.05 t | 1.7-1.9 m | — |

- The bowl's point-blast floor (12-19 t) is kept as a minimum. It never binds against the
  shared 63 t.
- The Sun plate's absorber bound had the same flaw. On the spring its survival floor
  binds, so its mass is unchanged.

The stroke is now the strongest single lever:

| | 4 m | 8 m (headline) | 12 m |
|---|---|---|---|
| 9.86, 60 | 3.01x, $192 | 3.38x, $144 | 3.50x, $133 |
| 4, 90 | 4.82x, $104 | 5.31x, $91 | 5.47x, $87 |

The stiffest spring (theta = pi, peak 1.57x) costs $5 / $1.

## S45 item 2: one shared spring

The shared assembly is the heavier of the two legs' plates: 63.4 t, on which the
departure needs only 4.2-4.9 m of the stroke, as you estimated (4.3 m). It flies through
the push, the turn and the departure, then diverts into Jupiter, as the bowl did.
Re-optimized against two springs (the push plate left in the parking orbit):

| | shared | two springs |
|---|---|---|
| 9.86, 60 | 3.38x, $144 | 3.41x, $144 |
| 4, 90 | 5.31x, $91 | 5.31x, $92 |
| 9.86, 90 | $188 | $189 |
| 4, 150 | $118 | $120 |

On cost it is a wash. The shared assembly is the default because it leaves nothing in an
orbit the Moon perturbs (S43 item 5), and it builds one plate instead of two.

## S45 item 3: the bags

We use your form: mean `PV = gamma (F M/(m+M))^2 / k` with helium (gamma 5/3).

- The fibre wall is `3 P V rho / sigma` at the peak pressure (x1.21), on 2.5 GPa,
  1600 kg/m^3 fibre at a safety factor of 2.
- The helium fill is at 293 K.
- That comes to **1.9 t** per Earth plate, priced and carried with the plate.
- Your paper's liner, felt, rings and guides (5 t on a 1.7e8 J bag, scaled with PV) are a
  sensitivity row. They cost $5 dear at 9.86 R_sun and $1 at 4 R_sun.

## S45 item 4: the g-cap rows

They are dropped. On the spring the cargo feels the mean thrust times 1.21, whatever the
stroke: 1.21 g at the start of a 1 g push and 1.52-1.66 g at its end. That is far under
the 25 g rack rating, so the cap no longer sizes anything. The ship's 100 g cap is gone
too.

## S43 item 1: the sim's overtake law

`fast_wave_law.csv` has only the five closures. So we regenerated the sim's gitignored
2-D overtake runs and its CH2 table (`f7cce80`), and ran its own
`push_ve(k, w, IMPACTOR_TOLL[...])` on a grid: k = 5-40, w = 50-330 km/s, for Kapton and
graphite. The sim's `law_rows()` on that rerun reproduce its committed CSV to every
digit, and our interpolated grid reproduces them to 0.13%. It is vendored as
`data/fast_wave_push_law.csv`.

How far it moves things is the third column of the step table above: **$10 dear and 0.12x
of growth at 9.86, 60; $1 and 0.05x at 4, 90.** It is not a uniform haircut:

- at k ~ 7-8 it is 1-4% below the linear law at 99-137 km/s and 9% below at 208;
- **at high k and high w it is above the linear law, by up to 10-20%**, because a heavier
  charge keeps its rebound while the bond toll shrinks. At 4 R_sun, 150 km/s the optimum
  moves to k_push ~26 and growth *rises*, 3.66x -> 4.22x. That k is outside where Result 15
  checked sheet heating, so treat that row as provisional;
- at the plate loop's 55-65 km/s it is 10-20% below.

At both headlines k_delivery sits on 5.0, the sim's lowest k.

## S43 item 2: sheet material

Each wave's sheets are priced by the stream they fly in at 400 km:

- up to 130 km/s, Kapton at $3/kg;
- above 130 km/s, dry woven carbon fabric at **$100/kg**.

The push wave goes by the growth stream. The departure wave's returns go by their
delivery stream at 400 km, the faster of their two uses. The 100-130 km/s band is Kapton
by default, and a switch flies carbon fabric there.

**No citable bulk price for dry fabric was found.** Retail 532 g/m^2 dry triaxial fabric
is $85-99 a yard (Rock West, 2026-10), about $145-167/kg, which matches your retail end.
So $100/kg stays your hypothesis. The $34 and $165 rows at 4 R_sun, 90 give $87 / $95
dear.

**Your table, confirmed, then corrected.**

- On the code you ran (`e18dc45`), re-optimizing each point with every sheet at $100/kg
  reproduces all 13 rows exactly.
- That policy also charges the departure wave fabric. At 4 R_sun, 90 its delivery stream
  is 102.6 km/s, which is Kapton under your 130 km/s rule. On the new code that policy is
  the row "every sheet carbon fabric $100/kg": $162 dear at 9.86, 60 and $95 at 4, 90.

The corrected sweep, dear optimum per depth:

| depth | Kapton in the 100-130 band | carbon fabric in the band |
|---|---|---|
| 4 R_sun | **75 km/s, $89** (110: $90, 90: $91) | 110 km/s, $90 (90: $91) |
| 6 R_sun | 75, $100 | 90, $105 |
| 8 R_sun | 75, $117 | 75, $125 |
| 9.86 R_sun | 75, $143 (60: $144) | 60, $144 |
| 12 R_sun | 60, $179 | 60, $179 |
| 15 R_sun | 75, $504 | 75, $519 |

With Kapton in the band the 4 R_sun headline would move to the 75 km/s return. That
turns on your S44 to the sim (Kapton at 115-122 km/s).

## S43 item 3: the 8 R_sun optimum

Our ADR 0074 summary was wrong. On its own code the 75 km/s return was $96 / $102,
below 60's $99 / $107. You printed the right one. On the new code it is 75 again ($117).

## S43 item 4: one number

The sweep's headline points now start from `HEADLINE_STARTS`, so the headline and the
sweep print the same figures. Starting every closure from `_SWEEP_START` instead lands
within $1 and 0.01x, so the optima are not start-dependent.

## S43 item 5: the disposal with the Moon

Your 385 000 km apoapsis is at the Moon's distance. We flew the plate from the apoapsis
to its next perigee with Earth, the Moon (circular, 0 and 28.6 deg to the orbit) and the
Sun's tide, over 24 lunar phases and two Sun directions. Then we solved the nudge that
reaches 100 km.

- **With the Moon more than 20 deg from the apoapsis, the nudge is 0-15 m/s** (median
  2.7).
- In 34 of 84 geometries the Moon alone brings the plate down.
- The two-body 4.2 m/s lands within 50 km of 100 km in only 6 of 84, and leaves the
  perigee as high as 897 km. The nudge must be targeted.
- The ledger now charges 15 m/s (~0.3 t on a 65 t plate). It only matters to the
  two-spring row, since the shared assembly leaves nothing behind.

**A new finding the paper may need.** Within ~20 deg of the apoapsis the Moon is within
~100 000 km of it, and the parking orbit itself meets the Moon (36-137 m/s, or no
solution). The ship shares that orbit until apoapsis, so roughly a ninth of lunar phases
put the Moon in the ship's way. The Moon also moves a 400-600 km perigee by hundreds of km
over one half-orbit. The apoapsis methalox burn can absorb that for up to ~15 m/s. This is
not solved here. It constrains when the push may happen relative to the Moon.

## The gap

| gap | 9.86, 60 | 4, 90 |
|---|---|---|
| 3 d | 2.99x, $159 | 4.81x, $83 |
| 5 d | 3.21x, $144 | 5.14x, $82 |
| 10 d | 3.38x, $144 | 5.31x, $91 |

At 4 R_sun a 5-day gap brings the growth stream to 119 km/s, keeping Kapton. That is now
worth $9, not ~$2.

## The Jupiter-only plate loop (ADR 0075)

It gets much worse and still does not pay.

| departure | ADR 0075 | now |
|---|---|---|
| 0.25 g | 2.01x, $573 | 1.27x, $9436 |
| 0.5 g | 2.45x, $279 | 1.68x, $1654 (now its best) |
| 1 g | 2.59x, $238 | 1.59x, $2408 |

The spring plates alone give 1.44x / $3705 at 0.25 g. The sim's law does the rest at the
loop's slow 55-65 km/s streams, where it runs 10-20% below the linear law. Any paper text
quoting 2.01x / 2.59x needs these figures.

## Still open

- **S46 (sim):** the bag spring's loss and heating over 2000 cycles. These figures assume
  it lossless.
- **S44 (sim):** Kapton at 115-122 km/s, which picks the 4 R_sun headline.
- Sheet heating at the high k the 150 km/s optimum now picks.
- The parking orbit's lunar encounters (S43 item 5 above).
