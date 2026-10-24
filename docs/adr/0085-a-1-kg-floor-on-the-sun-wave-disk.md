# ADR 0085: A 1 kg floor on the Sun-wave disk: the harder boost, priced

Date: 2026-10-24

Status: Accepted. Answers the parent's S59 (`docs/sun_disk_floor_asks_for_aim_repo.md`,
copied from `Balloon-Pulse-Propulsion` @ `baa3a23`). Reproduce with `make sun-disk-floor`
(~1 min); the headlines with `make solar-dive`, the Ceres loop with `make ceres-dive` and
`make ceres-energy`.

## Context

Each Sun-wave disk now carries 85-130 g of guidance hardware plus 25 g of cold gas (the
parent's `sec:sun_wave_ranging`). The author: a disk below 1 kg is implausible
miniaturization. The ledger flew `disks / (2 Hz x arc)` of disk a pulse: 0.20 kg at the
9.86 R_sun headline and 0.55 kg at 4 R_sun. Of three ways up (lower `k_sun`, stacking
units into one Sun ship, a harder boost), the author chose the **harder boost**: a
shorter arc. The parent's scratch capped the arc until the pulse reached 1 kg.

## Decision

**1. The floor is a constraint** (`Design.disk_floor`, `DISK_FLOOR = 1.0` kg, headline).

- One disk strikes a pulse. The floor is on the struck mass; the gas rides on top.
- `optimize_design` projects any arc longer than `floor_arc` onto it and penalises the cut
  in log space, as it does a bound. Every design it scores holds the floor, and so does the
  optimum. The arc is solved by the secant in log space (`arc_on_floor`), because the pulse
  falls almost as 1/arc.
- The floor binds on the **lighter ship**, the departure-wave one (`pulse_disks` takes the
  minimum). The push-wave ship then flies 1.70 kg a pulse on the same arc. The parent's
  harness checked the departure-wave unit only, which happens to be the binding one.
- The optimizer restarts Nelder-Mead from its last answer, up to four times. Without that
  one simplex stalled on `k_delivery` after the floor moved the arc: $162 instead of $158
  at 9.86 R_sun. The unfloored headline reproduces ADR 0077 exactly either way.

**2. The headlines under the floor** (dear-seed optimum; the floor and gas charged):

| Closure | arc (h) | kg/pulse dep / push | growth | doubling (y) | steady $/kg | BE cheap / dear |
|---|---|---|---|---|---|---|
| 9.86, X 60, ADR 0077 (no floor) | 4.97 | 0.200 / 0.339 | 3.361 | 1.873 | 94.1 | 124 / 146 |
| 9.86, X 60, **1 kg floor** | 0.99 | 1.000 / 1.698 | 3.226 | 1.939 | 96.5 | 131 / 158 |
| 4, X 90, ADR 0077 (no floor) | 2.19 | 0.546 / 0.913 | 4.995 | 1.412 | 73.5 | 84 / 87 |
| 4, X 90, **1 kg floor** | 1.17 | 1.000 / 1.678 | 4.880 | 1.433 | 74.1 | 85 / 89 |

This confirms the parent's scratch to $0.2 ($130.8 / $158.0 and $85.6 / $88.9). The floor
costs $12 dear at 9.86 R_sun and $1.5 at 4 R_sun, where the unfloored pulse was already
half the floor. A 0.5 kg floor costs $4 at 9.86 and nothing at 4 R_sun. A 2 kg floor is
out of reach at 9.86 (the 0.5 h arc carries 1.96 kg) and costs $7 at 4 R_sun.

The faster returns: 9.86, X 90 binds (1.83 h, against 5.26 h). It goes from 3.10x and
$144 / $182 to 2.95x and $153 / $204, the dearest hit. Its pulse was 0.34 kg, and the
dear seed amplifies the growth it loses. 4, X 150 clears the floor (1.02 kg at 2.39 h) and is unchanged ($111 / $119).

Elsewhere:

- **The gap** (3 / 5 / 10 days, re-optimized): 9.86 R_sun $177 / $158 / $158 dear,
  4 R_sun $89 / $87 / $89. The 5-day gap now ties the 10-day one at 9.86 R_sun.
- **The depth dial** (`--sweep`): 4 R_sun, X 90 is still the cheapest closure. 4 R_sun,
  X 60 cannot carry the floor at all: 0.92 kg a pulse even at k_sun 20 on the 0.5 h arc.
- **The eccentric-orbit delivery** (S51's comparison): $156 dear at 9.86 R_sun and $87 at
  4 R_sun, $2 under L1.

**3. The Sun plate at 1 kg pulses** (S59 item 3). It is inside what it was fit to:

- `SUN_PLATE_PULSE` is *rated* at a 1 kg disk at 400 km/s. `_plate_for` scales the survival
  floor as the pulse's energy to the 2/3, peak pressure held. At 9.86 R_sun (391.7 km/s)
  a 1 kg pulse is 0.96 of the rating: the anchor itself. The old 0.2 kg pulse was the
  extrapolation, at 0.19 of it.
- At 4 R_sun (616.6 km/s) a 1 kg pulse is 2.38 times the rated energy, against 1.30 before.
  That is an extrapolation in energy on a scaling law, not a fit; it was one before too.
- The survival floor binds, not the spring: the strokes are 2.9 m and 2.2 m, under the 8 m
  stroke the spring law would size. `spring_plate_stroke` is exact for any pulse.
- The Sun plate, its bags and the collar grow from 6.3 to 16.7 t per unit at 9.86 R_sun,
  and from 20.1 to 29.7 t at 4 R_sun. The ship's peak on its spring rises from 0.08 to 0.40 g
  and from 0.26 to 0.46 g.
- The stage law (`sun_boost`) is per kilogram of charge and does not depend on pulse size.
  The parent's central hardware lump in the charge is the simulation's to fly.

**4. The disk's hardware and gas** (S59 item 4).

- The 85-130 g of hardware is inside the struck kilogram: impactor mass, not dead mass. The
  ledger already treats it that way, since the stage law is per kg of disk.
- **The 25 g of cold gas is carried** (`Design.disk_gas`, `DISK_GAS = 0.025` kg a disk,
  headline). It is launched, carried to Jupiter and turned by the carrier (its dry mass and
  storable propellant scale with it), and priced as storable propellant. It never strikes.
  It costs $0.1-0.3 of break-even and 0.001-0.003 of growth. Worth carrying because it is
  honest, not because it moves anything.

**5. Stacking, as a comparison only** (`Design.sun_stack`; not adopted).

Five launch units' disks feed one Sun ship. Its plate, bags, collar and spring are shared,
so each unit carries a fifth, and the pulse is five times the unit's:

| Closure | case | arc (h) | kg/pulse dep | Sun plate+collar (t/unit) | growth | BE cheap / dear |
|---|---|---|---|---|---|---|
| 9.86, X 60 | ADR 0077's design held, 5 units | 4.97 | 1.00 | 3.35 | 3.401 | 122 / 143 |
| 9.86, X 60 | 5 units, re-optimized | 4.11 | 1.21 | 3.77 | 3.402 | 122 / 142 |
| 4, X 90 | ADR 0077's design held, 5 units | 2.19 | 2.73 | 11.47 | 5.153 | 83 / 85 |
| 4, X 90 | 5 units, re-optimized | 1.87 | 3.18 | 12.68 | 5.154 | 83 / 85 |

**Stacking is not ledger-neutral; it pays.** The parent's first-order argument holds for
the disks, charge and film, which scale with the units. It misses the plate and collar. The
survival floor goes as the pulse to the 2/3, so five units share a plate 2.9 times one
unit's, and the 1 t collar is split five ways. At 9.86 R_sun that is $3 dear under the
*unfloored* headline and $15 under the floored one. At 4 R_sun it is $2 and $4.

Still not adopted, as asked: docking is not designed or charged, and the seed is one unit.
The program is linear in units, so a five-unit seed leaves the break-even unchanged but
quintuples the seed outlay.

**6. The Ceres loop** (ADR 0083, 0084), on the author's instruction (2026-10-24).

The Ceres ships fly the same Sun leg, and they were under the floor too: 0.60 / 0.73 kg a
pulse at 4 R_sun and 0.33-0.35 at 9.86. `fly_ship` now carries the gas (pushed from Ceres
with the disks, priced as water) and records `pulse_disk`. `optimize_loop` and
`optimize_energy` hold the arc to the floor through `ceres_dive.on_floor`.

| Ceres loop | ADR 0083 (no floor) | 1 kg floor |
|---|---|---|
| 4 R_sun, X_C 110: arc, k_sun | 3.14 h, 29.9 | 2.00 h, 26.6 |
| 4 R_sun: steady $/kg at L1 | 1.463 | 1.488 |
| 9.86 R_sun, X_C 90: arc, k_sun | 7.38 h, 27.8 | 3.00 h, 21.4 |
| 9.86 R_sun: steady $/kg at L1 | 1.696 | 1.741 |

The Ceres loop still doubles in 5.8 and 7.4 months. Its $1 line is unchanged from the
3000 t ships on ($0.997 at 4 R_sun): three times the ship triples the pulse, and the floor
stops binding. The two steps before it each cost $0.02-0.05 more.

The Straw Way energy optimizer (`optimize_energy`, ADR 0084) holds the floor the same way.
Its rerun under the floor (`make ceres-energy`) is not recorded here yet; ADR 0084's
reproduction tests fly unfloored springs.

Along the floor the Ceres price is flat in `k_sun` to $0.001 (k 26-30 at 4 R_sun), so the
optimum's `k_sun` is loosely determined.

## Consequences

- `HEADLINE_STARTS` are now the floored optima. ADR 0077's are kept as `UNFLOORED_STARTS`,
  and `unfloored()` gives a design without the floor or gas. The ADR 0082 book test is held
  on them, as it was run.
- `ceres_dive.HEADLINE` holds the floored optima; ADR 0083's are in its comment. The ADR 0084
  reproduction tests fly unfloored springs.
- Every solar-dive report re-optimizes under the floor. Sensitivities still hold the
  headline design. A row that changes the disk count (the film exponent, the carrier
  perijove) can sit a hair under the floor; it is not re-optimized.
- The Jupiter-only plate loop (ADR 0075) flies no disks and is unchanged. Its gap to the
  solar dive narrows by the floor's $12.

## Found on the way: the bowl law's interpolation seam (not changed)

`StageLaws.hole_ve` interpolates the capture and the rebound/mixing factor separately,
each log-linear in `k` between the table's nodes (k = 5, 10, 20, 30, 40, 60), and
multiplies them. The product humps between nodes: at the 9.86 R_sun departure (84.0 km/s,
129.9 deg) `v_e` is 12.95 km/s at k = 7.5, against 11.9 from interpolating `v_e` itself.
At 12.5 it is 10.41 against 9.72. Between k = 5 and 5.5 it even rises.

So the optimizer sees two basins either side of the k = 10 node. ADR 0077's headline sits
in the k_E ≈ 12.5 one. The k_E ≈ 7.4 basin is **$7 cheaper dear** at 9.86 R_sun ($138.6
against $145.7, unfloored) and $2 at 4 R_sun. Both basins lean on the same artefact.
This ADR keeps the recorded basin, so the floor's cost is measured like for like. Whether to
re-interpolate (`v_e` directly, or the product in log) is open. It moves every headline
and wants the simulation's k = 7.5 and 15 nodes.
