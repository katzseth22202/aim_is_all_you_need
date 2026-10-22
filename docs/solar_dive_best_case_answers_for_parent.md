# The solar dive's best long-term price: answers to S53 (for the parent)

Raised 2026-10-22 in `aim_is_all_you_need` (ADR 0082). **Written to be copied into
`Balloon-Pulse-Propulsion`.** It answers S53 items 1-6
(`docs/solar_dive_best_case_asks_for_aim_repo.md` @ `c27de29`).

Reproduce with `make solar-dive-best` (~8 min). `ARGS=--sweep` adds the deep closures'
scan at a $1 lob (~10 min).

All figures are the steady $/kg at L1, re-optimized on the **steady** objective over both
spring layouts. "Mature" is the author's book: disk $10, carrier $30, shield and avionics
$100/kg, and the growth push plate and the delivery plate each flown 10 times. Lob prices
are `SolarDivePrices.lob`, before the climb factors.

## Headline (for the paper, if it wants one)

**$3/kg sits at the chemical lob's propellant floor, not above it.**

- **At 4 R_sun, X 90, it reaches $2.81, but only if all of this holds:**
  - the booster is charged its methalox alone;
  - operations cost nothing;
  - the plates fly 100 times;
  - packages cost $1;
  - the Earth-side charges are the author's snow foam, flying polyethylene's stage law.
- **At 9.86 R_sun it is not reachable.** The best is $3.41, and that needs the unsolved
  4 Hz delivery as well.
- **The honest long-term figures** are about $5.5 / $3.7 at a propellant-only lob, or
  $7.1 / $5.0 at a $1 lob.

## Item 1. The table, and the method

The table is confirmed to within $0.05. Cheap Sun hardware with reuse in the ledger gives
$85.34 / $67.84, against your $85.29 / $67.81. The paper's book reproduces exactly
($94.15 / $73.52).

The match is for the wrong reason, though. **The headline's `push_plate` line is zero.**
Under the shared assembly (ADR 0076) the push plate *is* the bowl and departs with the ship,
so dividing it by 10 did nothing. Your gain is the cheap Sun hardware plus the delivery
plate.

The method misses three things:

- **Re-optimizing.** On the steady objective it takes another $1.6-1.9 off ($83.74 /
  $65.98). `HEADLINE_STARTS` are dear-seed break-even optima, not steady optima.
- **The spring layout.** The re-optimized optimum flips to two springs, which leave the
  push plate in the parking orbit where it can be reused.
- **Slower learning.** Fewer plates are bought, so each one sits higher on the curve
  (+$0.05).

## Item 2. The best-case book

The book is `MATURE`. The use counts are `SolarDivePrices.push_plate_uses` and
`delivery_plate_uses`; each use buys `1 / uses` of a plate on the curve.

**The plates' recovery is not charged.** That covers the push plate's re-entry from the
10-day orbit and the delivery plate's return from L1.

Steady $/kg at a $25 lob:

| closure | 1 use | 10 uses | 100 uses |
|---|---|---|---|
| 9.86 R_sun, X 60 | 85.45 | 83.74 | 83.46 |
| 4 R_sun, X 90 | 68.54 | 65.98 | 65.81 |
| 9.86 R_sun, X 90 | 80.82 | 79.26 | 79.43 |
| 4 R_sun, X 150 | 76.19 | 74.83 | 74.66 |

**Cheap disks do not move the optimum. A cheap lob moves it a little.**

- **Depth.** 4 R_sun stays cheapest in the scan over 4/6/8/9.86 R_sun, and at 4 R_sun
  X 90 stays best ($5.03 at a $1 lob).
- **Return excess.** At 9.86 R_sun the best return excess moves from 60 to 75 km/s ($6.57
  against $7.12 at a $1 lob).
- **`k_delivery` does not move.** It is pinned at 5 except at the carbon-fabric closures.

**Wanted from the impact simulation.** The steady optimum sits on the stage laws' lower
bounds: k_push = k_delivery = 5, and k_sun = 20. The optimum wants lighter slugs than the
simulation has flown. Laws below those bounds are the largest lever left unpriced.

## Item 3. The lob sweep and its floor

**The booster's propellant per kg lofted** comes from `lob_rise.braked_lob`: the 5000 t
vehicle, with the brake and landing held back.

| climb at 400 km | kg / kg | at $0.20 (book) | at $0.12-0.27 (sourced) |
|---|---|---|---|
| 0.75 km/s (the `lob` axis) | 2.46 | **$0.49** | $0.30-0.66 |
| 1.1 km/s | 2.69 | $0.54 | $0.32-0.73 |
| 2.43 km/s | 4.31 | $0.86 | $0.52-1.16 |

Your figure checks out. 3,400 t over 1,500 t is 2.27 kg/kg, against this model's 3,378 t
over 1,372 t.

The sourced price range builds methalox at Raptor's 3.6 O/F from two inputs:

- **Methane, ~$0.32/kg.** EIA's July 2026 STEO puts Henry Hub at $3.67/MMBtu, which is
  $0.19/kg. About $2.5/MMBtu of liquefaction comes on top.
- **Bulk oxygen, $0.065-0.25/kg.** These are Expert Market Research's and IMARC's US Q4
  2025 indices. They disagree by four times.

The floor is propellant only. It leaves out the booster's amortization, refurbishment, the
pad and the operations.

**The sweep.** Each cell is as asked / with the S27 stage (one Raptor, tanks) and a 40 m/s
deep-space reserve charged:

| lob $/kg | 9.86, X 60 | 4, X 90 | 9.86, X 90 | 4, X 150 |
|---|---|---|---|---|
| 25 | 83.74 / 84.63 | 65.98 / 66.38 | 79.26 / 80.76 | 74.83 / 75.52 |
| 10 | 35.83 / 36.45 | 27.94 / 28.20 | 34.33 / 36.19 | 34.87 / 35.33 |
| 5 | 19.88 / 20.44 | 15.20 / 15.45 | 19.84 / 19.94 | 21.34 / 21.73 |
| 2 | 10.31 / 10.79 | 7.57 / 7.80 | 10.48 / 11.04 | 13.17 / 13.30 |
| 1 | 7.12 / 7.57 | 5.03 / 5.25 | 7.02 / 7.51 | 10.26 / 10.62 |
| **0.49 (floor)** | **5.50 / 5.94** | **3.74 / 3.95** | 5.75 / 6.27 | 8.77 / 9.13 |

## Item 4. What sets lofted kg per kg sold

At a $1 lob, kg per kg at L1 (cargo = 1). "Booster-eq." multiplies each lob by its climb
factor over the 1.1 km/s lob: x1.49 for a delivery, x1.48 for growth.

| closure | delivery charge | plate + bags | halo methalox | growth lob | lofted | booster-eq. |
|---|---|---|---|---|---|---|
| 9.86, X 60 | 0.401 | 0.068 | 0.008 | 0.545 | 2.02 | 3.00 |
| 4, X 90 | 0.274 | 0.062 | 0.008 | 0.267 | 1.61 | 2.39 |
| 9.86, X 90 | 0.286 | 0.062 | 0.008 | 0.565 | 1.92 | 2.85 |
| 4, X 150 | 0.371 | 0.066 | 0.008 | 0.403 | 1.85 | 2.75 |

**A faster return does not lower the floor.**

- **At 9.86 R_sun:** X 90 cuts the delivery charge by 29%, and the lofted mass by 5%. But
  its growth falls (2.87 against 3.29), so it buys more growth units. At the floor X 90
  costs $5.75 against X 60's $5.50.
- **At 4 R_sun:** X 150's stream is past Kapton's 140 km/s, so its sheets are carbon
  fabric, and they alone cost $3.81/kg.

## Item 5. Lines that matter at $5/kg

At a $5 lob, mature book, S27 charged. These are every line at $0.10 or more, in $/kg:

| line | 9.86, X 60 | 4, X 90 |
|---|---|---|
| delivery_lob | 11.67 | 10.64 |
| lob | 4.41 | 2.10 |
| delivery_charge | 0.80 | 0.55 |
| polyethylene | 0.69 | 0.28 |
| bowl | 0.63 | 0.35 |
| **apoapsis_stage (S27)** | **0.40** | **0.19** |
| sun_plate | 0.24 | 0.40 |
| sheets | 0.36 | 0.22 |
| **packages** | **0.30** | **0.18** |
| delivery_plate | 0.24 | 0.17 |
| avionics | 0.21 | 0.13 |
| mounts / shield | 0.17 / 0.11 | under 0.10 |
| **operations, per $1M a lob** | **1.36** | **1.07** |

Your four candidates:

- **The S27 stage: yes, $0.19-0.40.** It is one Raptor 3 at $1M plus tanks at 0.018 kg per
  kg of methalox, dropped after the burn (`Design.apoapsis_stage`).
- **The deep-space reserve: no dollar line.** It costs about 1.1% of the returned sheets,
  and that is inside the S27 column (`Design.deep_space_reserve`).
- **The steering package: yes, $0.18-0.30.** It is set by the curve's slope, not its $10
  floor: at the steady count a package still costs about $15. It is now its own line,
  `packages`.
- **Operations: the biggest unpriced line.** Each 1000 t at L1 takes 0.18-0.37 growth lobs
  and 0.90-0.98 delivery lobs. So $1M of operations per lob costs more than the whole
  floor-priced lob.

## Item 6. What must fall to reach $3/kg

The steps are cumulative, each re-optimized, with S27 charged and operations at $0:

| step | 9.86, X 60 | 4, X 90 | 9.86, X 90 | 4, X 150 |
|---|---|---|---|---|
| mature book, $5 lob | 20.44 | 15.45 | 19.94 | 21.73 |
| lob at its propellant floor ($0.49) | 5.94 | 3.95 | 6.27 | 9.13 |
| polyethylene $1.2/kg | 5.34 | 3.62 | 5.69 | 8.51 |
| plates flown 100 times | 5.06 | 3.43 | 5.11 | 8.28 |
| steering package $1 flat | 4.77 | 3.26 | 4.88 | 7.99 |
| Earth charges as snow foam, $0.062/kg (law unsolved) | 3.79 | **2.81** | 4.45 | 7.11 |
| the child's 4 Hz delivery from a 1.1 km/s lob (S37 unsolved) | 3.41 | 2.45 | 4.08 | 6.67 |

What must fall, ordered by size at 4 R_sun, X 90:

1. the lob to its propellant floor (-$11.50 from a $5 lob);
2. the Earth charges from $2 polyethylene to snow foam (-$0.78);
3. the 4 Hz delivery (-$0.36);
4. the plates to 100 uses (-$0.19);
5. the packages to $1 (-$0.17);
6. operations held under ~$0.18M a lob, which is all the $0.19 of headroom buys.

**Said plainly:**

- **At 9.86 R_sun, $3 is below the floor of any chemical-booster lob.**
- **At 4 R_sun, $3 needs a booster flown for its propellant alone, with nothing else
  charged.**

**The snow foam** is the author's idea (2026-10-22): snow in a thin polyethylene skin, a
light foam charge. Only its price is in the ledger: 5% skin at $1.2 and water at $0.002,
so $0.062/kg. It is priced on polyethylene's stage law, which is a bound. It applies only
to the push, departure and delivery charges. The Sun charge stays polyethylene, because
snow would not survive a 4 R_sun dive.

**Wanted from the impact simulation:** the foam's own exhaust-speed law. Water's bonds
re-form late (ADR 0033), and the foam's density will change how the slug stagnates.

## Asks raised

- **Impact simulation:**
  - the stage laws below k = 5 and k_S = 20;
  - a snow-foam charge in a polyethylene skin;
  - the 4 Hz delivery's sheet survival (S37).
- **Unpriced:**
  - plate recovery;
  - booster amortization and operations per lob.

Nothing here touches `tab:delivery_ledger`'s headline. Every lever defaults off, and
`make solar-dive` is unchanged.
