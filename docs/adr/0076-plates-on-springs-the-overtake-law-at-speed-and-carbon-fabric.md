# ADR 0076: Plates on springs, the overtake law at speed, and carbon fabric

Date: 2026-10-19

Status: Accepted. Answers the parent's S45 and S43
(`docs/two_plates_followup_asks_for_aim_repo.md` in `Balloon-Pulse-Propulsion` @ `5d1379c`).
Amends ADR 0073 (the push plate's and bowl's sizing), ADR 0074 (the delivery plate, the
overtake law, the graphite rows and the disposal nudge) and ADR 0075 (its figures; its
verdict stands). Parent note: `docs/spring_plates_and_fast_law_for_parent.md`.

## Context

S45: `_inertia_plate` sized every plate by `m = I^2 / (2 s M a)`, a plate brought to rest
within each pulse by a constant-force absorber. That absorber must dissipate the plate's
kinetic energy `I^2 / (2 m)` every pulse. Over a push this comes to 4.4 TJ, or 363 MJ per
kg of the 12.2 t push plate, and steel vaporizes at ~7 MJ/kg. The paper's own
`sec:pusher_plate_damping` already flies a nearly lossless gas spring tuned across pulses.
The parent asked for that law at an 8 m stroke on every plate, for one shared push-and-bowl
assembly, for the bags to be charged, and for the delivery g-cap rows to be dropped.

S43 asked for five things:

- adopt the impact simulation's re-derived overtake law (Result 15);
- make the sheet material a function of the stream, with carbon fabric at $100/kg above
  130 km/s and the 100-130 km/s band as a switch;
- settle the 8 R_sun optimum;
- start the sweep's headline points from `HEADLINE_STARTS`;
- propagate the push plate's disposal nudge with the Moon.

## Decision

1. **Every plate rides the spring law** (`gas_spring`, `spring_plate_stroke`,
   `peak_to_mean`). In the steady state each pulse `J = F / nu` flips the plate's velocity
   relative to the craft from `-J/2m` to `+J/2m`. Between pulses the relative motion is an
   arc `theta` of a cosine. So the peak force is `(theta/2) / sin(theta/2)` times the mean,
   1.21 at `theta = 2 pi / 3`. The plate's swing is `F tan(theta/4) / (2 m theta nu^2)`.
   - A direct integration with the craft only six times the plate reproduces both to
     1e-3, starting from the paper's half-size first pulse
     (`test_the_spring_law_matches_a_direct_integration`).
   - The relative velocity step is `J / m` whatever the craft weighs, so `m x stroke` does
     not depend on the craft's mass. The craft enters only the stiffness `mu (theta nu)^2`,
     through the reduced mass. The spring carries the craft's share of the thrust,
     `F M / (m + M)`.
   - `Design.spring_stroke` is 8 m (Seth) and `spring_theta` is 2 pi / 3. That gives
     **63.4 t** for the push plate and for the delivery plate (507 t m at 2 Hz, 1 g). The
     bowl keeps its point-blast floor (12-19 t) as a minimum. The Sun plate is held at its
     survival floor and swings 1.7-1.9 m.
   - The absorber bound and the ship's and cargo's g caps (`G_CAP`, `CARGO_G_CAP`,
     `Design.g_cap`, `Design.delivery_g_cap`) are gone. On the spring the ship peaks at
     1.56-1.74 g at the end of the push and 0.34-0.38 g on the bowl. The delivery's cargo
     sees 1.21 g at the start and 1.52-1.66 g at the end.
2. **The bags are charged** (`bag_mass`). The mean `PV` is `gamma (F M/(m+M))^2 / k`, with
   helium at `gamma = 5/3`. The fibre wall is `3 P V rho / sigma` at the peak pressure,
   using a 2.5 GPa, 1600 kg/m^3 fibre at half strength. The helium fill is at 293 K. This
   comes to **1.9 t** per Earth plate and under 0.05 t on the Sun plate.
   - Bags are priced on the plate's per-kg curve and carried with it.
   - The paper's liner, felt, rings and guides (5 t on its 1.7e8 J bag) are a sensitivity
     (`Design.bag_fittings`). It costs $5 dear at 9.86 R_sun and $1 at 4 R_sun.
3. **The push plate and the bowl are one shared assembly** (`Design.shared_spring`,
   default True, Seth's suggestion).
   - It is the heavier of the two legs' plates: 63.4 t, with the bowl swinging 4.2-4.9 m of
     the same stroke. Its bags take the larger of the two fills.
   - It flies through the push, the apoapsis turn and the departure, then diverts into
     Jupiter.
   - Against two springs, each re-optimized, it is a wash. At 9.86 R_sun, X 60 both are
     $144 dear, with two springs growing 3.41x against 3.38x. At 4 R_sun, X 90 the shared
     assembly is $1 cheaper. It is cheaper at both comparison closures too, by $1-2.
   - It is taken because nothing is left in the parking orbit (item 7). Two springs are a
     sensitivity row.
4. **The overtake law is the simulation's at speed** (`StageLaws.push_ve`,
   `data/fast_wave_push_law.csv`, `Design.push_law`).
   - The simulation commits only its five closures (`fast_wave_law.csv`). Its
     `push_ve(k, w, toll)` was therefore run here on a k = 5-40, w = 50-330 km/s grid,
     after regenerating its gitignored 2-D overtake runs (`data/README.md` has the steps).
     On that rerun it reproduces its committed rows to every digit, and the grid
     reproduces them to 0.13%.
   - The law is no longer linear in `w`. At k ~ 7-8 it is 1-4% below ADR 0074's law at
     99-137 km/s and 9% below at 208. At high k and high w it is up to 10-20% *above* it,
     because a heavier charge keeps its rebound while the bond toll shrinks. At the plate
     loop's 55-65 km/s it is 10-20% below.
   - The impactor's atomization is charged by material (Kapton or graphite). It moves
     `v_e` under 0.1%.
5. **Sheets are priced by the stream they fly in at 400 km** (`SolarDivePrices.sheet_price`,
   `KAPTON_HOLDS`, `KAPTON_FAILS`, `sheet_material`).
   - Up to 130 km/s they are Kapton at $3/kg. Above it they are dry woven carbon fabric at
     **$100/kg**. That is the parent's hypothesis: no citable bulk price was found.
     Retail 532 g/m^2 dry triaxial fabric is $85-99 a yard (Rock West, 2026-10), about
     $145-167/kg.
   - The unrun 100-130 km/s band is Kapton by default, with `band_carbon` as a switch. The
     sweep prints both.
   - The push wave's sheets are priced by the growth stream. The departure wave's returns
     are priced by their delivery stream at 400 km, the faster of their two uses.
   - This differs from the parent's S43 table, which charged *every* sheet $100/kg wherever
     the growth stream passed 100 km/s. That table reproduces exactly on the pre-ADR code
     (all 13 rows) and is now the row "every sheet carbon fabric $100/kg".
   - `GRAPHITE_SHEET_BODY` ($20/kg graphite foil) is retired. The simulation's answer is
     fabric, not film.
6. **The optimizer starts are settled.**
   - ADR 0074's sweep summary was wrong at 8 R_sun. On its own code X 75 is $96 / $102,
     below X 60's $99 / $107, so X 75 was the optimum.
   - The sweep's headline points now start from `HEADLINE_STARTS`, so the headline and the
     sweep print one number.
   - Every closure was also re-optimized from `_SWEEP_START`. It lands within $1 and 0.01x.
7. **The left plate's nudge is flown with the Moon** (`src/plate_disposal.py`,
   `make plate-disposal`). The 10-day orbit's 385 000 km apoapsis is at the Moon's
   distance. Earth, the Moon (circular, 0 or 28.6 deg to the orbit) and the Sun's tide
   were flown from the apoapsis to the next perigee, over 24 lunar phases and two Sun
   directions.
   - With the Moon more than 20 deg from the apoapsis, the nudge to a 100 km perigee is
     **0-15 m/s** (median 2.7). In 34 of 84 geometries the Moon alone brings the plate
     down.
   - The two-body 4.2 m/s lands within 50 km of the target in only 6 of 84, and leaves the
     perigee as high as 897 km.
   - `plate_disposal_dv` now charges 15 m/s, or the two-body nudge where that is larger.
     It is ~0.3 t of storable on a 65 t plate.
   - Within 20 deg the parking orbit itself meets the Moon (36-137 m/s, or no solution).
     That is a constraint on the ship's own parking orbit and the departure's timing, not
     on the plate. It is not solved here.

## Results

Each closure is optimized for the dear-seed break-even. "X" is the return excess, the
Sun-relative hyperbolic excess speed the Sun boost leaves.

| closure | growth | doubling | steady | break-even cheap / dear | ADR 0074 |
|---|---|---|---|---|---|
| **9.86 R_sun, X 60** | **3.38** | **1.86 yr** | $93.4 | $123 / **$144** | 3.80x, $118 |
| **4 R_sun, X 90** (carbon growth sheets) | **5.31** | **1.36 yr** | $76.5 | $88 / **$91** | 5.71x, $79 ($88 on fabric) |
| 9.86 R_sun, X 90 | 3.12 | 2.00 | $104.7 | $150 / $188 | 3.27x, $151 |
| 4 R_sun, X 150 | 4.22 | 1.58 | $89.6 | $110 / $118 | 3.85x, $110 |

The dear break-even moves step by step, each step re-optimized:

| closure | ADR 0074 | + spring plates | + overtake law | + carbon fabric |
|---|---|---|---|---|
| 9.86, X 60 | 3.80x, $118 | 3.50x, $134 | 3.38x, $144 | 3.38x, $144 |
| 4, X 90 | 5.71x, $79 | 5.36x, $83 | 5.31x, $84 | 5.31x, $91 |
| 9.86, X 90 | 3.27x, $151 | 3.05x, $176 | 3.12x, $179 | 3.12x, $188 |
| 4, X 150 | 3.85x, $110 | 3.66x, $121 | 4.22x, $112 | 4.22x, $118 |

- The spring plates cost $4-25. The plates grow from 12 t to 63 t, and the delivery plate
  from 51 t to 63 t.
- The law costs $1-10 where k stays near 7. At X 150 it lets k_push rise to ~26 and
  *gains* growth: the law runs above linear there. That k is outside the range Result 15
  checked for sheet heating.
- Fabric costs $6-9 where the growth stream passes 130 km/s.
- At both headlines `k_delivery` sits on its 5.0 lower bound, the simulation's lowest k.

The spring stroke is the largest new lever. At 9.86 R_sun, X 60 (dear) a 4 m stroke gives
$192 and a 12 m stroke gives $133. At 4 R_sun the same strokes give $104 and $87.
`theta = pi` costs $5 / $1.

Depth sweep, dear optimum per depth, with Kapton in the band (carbon fabric in the band in
brackets):

| depth | Kapton in the band | (carbon fabric in the band) |
|---|---|---|
| 4 R_sun | X 75, $89 (X 110 $90, X 90 $91) | X 110, $90 |
| 6 R_sun | X 75, $100 | X 90, $105 |
| 8 R_sun | X 75, $117 | X 75, $125 |
| 9.86 R_sun | X 75, $143 (X 60 $144) | X 60, $144 |
| 12 R_sun | X 60, $179 | X 60, $179 |
| 15 R_sun | X 75, $504 | X 75, $519 |

So with Kapton in the band, the 4 R_sun headline would move to X 75. Whether Kapton holds
at 115-122 km/s is the parent's S44 to the simulation.

| gap | 9.86 R_sun, X 60 | 4 R_sun, X 90 |
|---|---|---|
| 3 d | 2.99x, $159 | 4.81x, $83 |
| 5 d | 3.21x, $144 | 5.14x, $82 |
| 10 d | 3.38x, $144 | 5.31x, $91 |

At 4 R_sun the shorter gaps keep the growth sheets on Kapton (112-119 km/s), which is what
makes them $8-9 cheaper.

**The Jupiter-only plate loop (ADR 0075) gets worse, and its verdict stands.**

| departure | ADR 0075 | now |
|---|---|---|
| 0.25 g | 2.01x, $573 | 1.27x, $9436 |
| 0.5 g | 2.45x, $279 | 1.68x, $1654 (now its best) |
| 1 g | 2.59x, $238 | 1.59x, $2408 |

The spring alone gives 1.44x / $3705 at 0.25 g and 1.82x / $950 at 1 g. The overtake law
does the rest at the loop's slow 55-65 km/s streams. Its `k_push` sits on the 5.0 bound.

## Consequences

- `make solar-dive` reproduces every figure above in ~1 min. `ARGS="--sweep --gaps"`
  takes ~2 min, `make plate-disposal` ~5 min and `make plate-jupiter-loop` ~10 min.
- The sensitivity table replaces its g-cap rows with stroke, theta, bag-fitting and
  two-spring rows. The graphite rows become carbon-fabric price rows ($34, $165), the
  band switch and the parent's every-sheet policy.
- Still open:
  - the gas spring's loss and heating over 2000 cycles (the parent's S46 to the
    simulation);
  - Kapton in the 100-130 km/s band (S44);
  - sheet heating at the high k the X 150 optimum now picks;
  - lunar encounters of the 10-day parking orbit itself.
