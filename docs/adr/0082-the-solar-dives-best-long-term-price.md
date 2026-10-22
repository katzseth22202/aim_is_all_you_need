# ADR 0082: The solar dive's best long-term price, and why $3/kg sits at the propellant floor

Date: 2026-10-22

Status: Accepted. Answers the parent's S53 (`docs/solar_dive_best_case_asks_for_aim_repo.md`
@ `c27de29`). Reproduce with `make solar-dive-best` (~8 min); `ARGS=--sweep` adds the
deep closures' scan at a $1 lob (~10 min more, 3 workers).

## Context

The author asked how low the charge pusher's steady $/kg at L1 can go in the long term,
against a $3/kg target. The levers: Sun-wave hardware at about $30/kg blended, the growth
push plate and the delivery plate reused, and the departure bowl and Sun plate single-use.
The parent's scratch held `HEADLINE_STARTS`, read `steady_lines`, and divided the plate
lines by 10 afterwards. It got $85 / $68 at a $25 lob and $7.1 / $5.2 at a $1 lob.

## Decision

**1. The ledger now carries the levers.**

- `SolarDivePrices.push_plate_uses` and `delivery_plate_uses` are use counts. Each use buys
  `1 / uses` of a plate on the spray cup's curve. The two plates share one learning count,
  so reuse also slows the learning (a use costs 0.10-0.13 of a new plate, not 0.10). The
  plates' recovery is **not charged**: the push plate's re-entry from the 10-day orbit
  (`plate_disposal` already lowers its perigee to 100 km) and the delivery plate's trip
  back from L1.
- `MATURE` is the author's book: disk $10, carrier $30, shield and avionics $100 per kg,
  both plates flown 10 times. The bowl and Sun plate stay single-use, and the lob stays at
  the Estimate's $25. The report sweeps the lob.
- `Design.apoapsis_stage` charges the apoapsis turn's engine and tanks (the parent's S27):
  one Raptor 3 (1525 kg, $1M) and tanks at 0.018 kg per kg of methalox ($40/kg), dropped
  after the burn. `Design.deep_space_reserve` burns a methalox correction reserve before
  Jupiter (the report uses 40 m/s). Both are off in the headline, which does not move.
- `SolarDivePrices.operations` charges each lob, growth or delivery. It is $0 by default,
  and the report quotes it per $1M.
- `SolarDivePrices.earth_charge` prices the push, departure and delivery charges apart
  from the Sun charge. This is for the author's snow foam (2026-10-22): snow in a thin
  polyethylene skin. Only the price changes. The stage laws stay polyethylene's.
- The steering packages are now their own line (`packages`), split from `sheets`.
- `Valuation.steady_lofted` gives the kilograms lofted per kilogram at L1, by what they are.

**2. The parent's method is a fair stand-in, but for the wrong reason.** On the mature book
at a $25 lob it gives $85.29 / $67.81, against $85.34 / $67.84 with reuse in the ledger.
But the headline's `push_plate` line is **zero**. Under the shared assembly (ADR 0076) the
push plate *is* the bowl and departs with the ship, so dividing it by 10 did nothing. The
whole gain is the cheap Sun hardware plus the delivery plate. Re-optimizing on the steady
objective takes another $1.6-1.9 off ($83.74 / $65.98). It also flips the optimum to **two
springs**, which leave the push plate in the parking orbit to be reused. The method misses
three things: that spring choice, the re-optimization (`HEADLINE_STARTS` are dear-seed
break-even optima, not steady optima), and the slower learning (+$0.05).

**3. Reuse is worth under $3 at a $25 lob.** Steady $/kg, re-optimized:

| closure | 1 use | 10 uses | 100 uses |
|---|---|---|---|
| 9.86 R_sun, X 60 | 85.45 | 83.74 | 83.46 |
| 4 R_sun, X 90 | 68.54 | 65.98 | 65.81 |
| 9.86 R_sun, X 90 | 80.82 | 79.26 | 79.43 |
| 4 R_sun, X 150 | 76.19 | 74.83 | 74.66 |

(9.86/90's 100-use cell sits $0.17 above its 10-use cell. That is Nelder-Mead noise
across the spring switch, not a real rise.)

**4. Cheap disks do not move the optimum. A cheap lob moves it a little.**

- **Depth.** 4 R_sun stays cheapest.
- **Return excess.** At 9.86 R_sun a $1 lob moves the best return excess from 60 to
  75 km/s ($6.57 against $7.12). At 4 R_sun, X 90 stays best ($5.03).
- **The knobs sit on their bounds.** k_push and k_delivery are at 5, the low edge of
  their range, at every Kapton closure, and k_sun is at 20, its low edge. The optimum wants
  slug ratios lighter than the simulation has flown, and the stage laws refuse to
  extrapolate. That is the clearest lever left unpriced.

**5. The lob's propellant floor is $0.49/kg on the book's lob axis.** The figure comes
from `src.lob_rise.braked_lob`: the 5000 t vehicle, brake and landing held back, the whole
load less the payload. Per kilogram lofted:

| climb at 400 km | kg propellant / kg | at $0.20/kg (book) | at $0.12-0.27/kg (sourced) |
|---|---|---|---|
| 0.75 km/s (ADR 0037's lob, the `lob` axis) | 2.46 | $0.49 | $0.30-0.66 |
| 1.1 km/s (the child's) | 2.69 | $0.54 | $0.32-0.73 |
| 2.43 km/s (the 1 g push's) | 4.31 | $0.86 | $0.52-1.16 |

The sourced range builds methalox at Raptor's 3.6 O/F from two inputs:

- **Liquid methane, ~$0.32/kg.** EIA's July 2026 STEO puts Henry Hub at $3.67/MMBtu for
  2026, which is $0.19/kg at 19.0 kg per MMBtu. About $2.5/MMBtu of liquefaction comes on
  top.
- **Bulk oxygen, $0.065-0.25/kg.** These are Expert Market Research's and IMARC's US Q4
  2025 indices. They differ by four times, and neither says whether it is liquid.

The parent's ~$0.5 stands. Its 3,400 t of propellant over a 1500 t lob is 2.27 kg/kg,
against this model's 3,378 t over 1,372 t. The floor is propellant only. It leaves out the
booster's amortization, its refurbishment, the pad and the operations.

**6. The lob sweep**, mature book, re-optimized. Each cell is as asked / with the S27 stage
and a 40 m/s reserve charged:

| lob $/kg | 9.86, X 60 | 4, X 90 | 9.86, X 90 | 4, X 150 |
|---|---|---|---|---|
| 25 | 83.74 / 84.63 | 65.98 / 66.38 | 79.26 / 80.76 | 74.83 / 75.52 |
| 10 | 35.83 / 36.45 | 27.94 / 28.20 | 34.33 / 36.19 | 34.87 / 35.33 |
| 5 | 19.88 / 20.44 | 15.20 / 15.45 | 19.84 / 19.94 | 21.34 / 21.73 |
| 2 | 10.31 / 10.79 | 7.57 / 7.80 | 10.48 / 11.04 | 13.17 / 13.30 |
| 1 | 7.12 / 7.57 | 5.03 / 5.25 | 7.02 / 7.51 | 10.26 / 10.62 |
| 0.49 (floor) | 5.50 / 5.94 | 3.74 / 3.95 | 5.75 / 6.27 | 8.77 / 9.13 |

At the floor the lob lines are only $1.25-1.56. The rest of the bill now dominates:
charges, bowl, Sun plate, sheets and packages.

**7. Lofted kg per kg at L1**, at a $1 lob. "Booster-equivalent" multiplies each lob by its
climb factor over the 1.1 km/s lob (x1.49 delivery, x1.48 growth):

| closure | delivery charge | plate + bags | halo methalox | growth lob | lofted | booster-eq. |
|---|---|---|---|---|---|---|
| 9.86, X 60 | 0.401 | 0.068 | 0.008 | 0.545 | 2.02 | 3.00 |
| 4, X 90 | 0.274 | 0.062 | 0.008 | 0.267 | 1.61 | 2.39 |
| 9.86, X 90 | 0.286 | 0.062 | 0.008 | 0.565 | 1.92 | 2.85 |
| 4, X 150 | 0.371 | 0.066 | 0.008 | 0.403 | 1.85 | 2.75 |

Cargo is 1. A faster return does cut the delivery charge: at 9.86 R_sun, X 90 has 29%
less than X 60, and 5% less lofted mass in all. But its growth falls (2.87 against 3.29),
so it buys more growth units. At the floor it costs $5.75 against $5.50. **A faster return
does not lower the floor.** At 4 R_sun, X 150's stream is above Kapton's 140 km/s, and its
carbon-fabric sheets alone are $3.81/kg.

**8. Lines at $0.10 or more** at a $5 lob, with S27 charged:

| line | 9.86, X 60 | 4, X 90 |
|---|---|---|
| delivery_lob | 11.67 | 10.64 |
| lob | 4.41 | 2.10 |
| delivery_charge | 0.80 | 0.55 |
| polyethylene | 0.69 | 0.28 |
| bowl | 0.63 | 0.35 |
| apoapsis_stage (S27) | 0.40 | 0.19 |
| sun_plate | 0.24 | 0.40 |
| sheets | 0.36 | 0.22 |
| packages | 0.30 | 0.18 |
| delivery_plate | 0.24 | 0.17 |
| avionics | 0.21 | 0.13 |
| mounts | 0.17 | <0.10 |
| shield | 0.11 | <0.10 |
| **operations, per $1M a lob** | **1.36** | **1.07** |

- **The deep-space reserve** is not a dollar line. It costs about 1.1% of the returned
  sheets, and that is inside the S27 column.
- **The steering packages** are set by the curve's slope, not its $10 floor. The curve
  still sits near $15 a package at the steady count.
- **Operations.** Each 1000 t at L1 takes 0.18-0.37 growth lobs and 0.90-0.98 delivery
  lobs. So $1M of operations per lob adds more than the whole floor-priced lob does.

**9. What must fall to reach $3/kg.** The steps are cumulative, each re-optimized, with S27
charged and operations at $0:

| step | 9.86, X 60 | 4, X 90 | 9.86, X 90 | 4, X 150 |
|---|---|---|---|---|
| mature book, $5 lob | 20.44 | 15.45 | 19.94 | 21.73 |
| lob at its propellant floor ($0.49) | 5.94 | 3.95 | 6.27 | 9.13 |
| polyethylene $1.2/kg | 5.34 | 3.62 | 5.69 | 8.51 |
| plates flown 100 times | 5.06 | 3.43 | 5.11 | 8.28 |
| steering package $1 flat | 4.77 | 3.26 | 4.88 | 7.99 |
| Earth charges as snow foam, $0.062/kg (polyethylene's law) | 3.79 | 2.81 | 4.45 | 7.11 |
| the child's 4 Hz, 3.26 g delivery from a 1.1 km/s lob (S37 unsolved) | 3.41 | 2.45 | 4.08 | 6.67 |

**At 9.86 R_sun, $3/kg is below the floor of any chemical-booster lob.** It is $3.41
with every lever pulled, the unsolved 4 Hz delivery included.

**At 4 R_sun, X 90, $3 is reached ($2.81) only if all of this holds:**

- the booster costs its propellant alone ($0.49/kg lofted);
- operations are free;
- the polyethylene is at $1.2;
- the plates fly 100 times;
- the packages cost $1;
- the Earth charges are snow foam that flies polyethylene's stage law.

The $0.19 left over buys about $0.18M of operations per lob. **$3 is at the floor, not
above it.** Two of the levers are unsolved physics: the snow foam's law and the 4 Hz
delivery (S37, which reaches $2.45).

What must fall, ordered by size at 4 R_sun, X 90 (each step as it fell in the cascade):

1. the lob to its propellant floor (-$11.50 from a $5 lob);
2. the Earth charges from $2 polyethylene to snow foam (-$0.78: -$0.33 to $1.2
   polyethylene, then -$0.45 to the foam);
3. the 4 Hz delivery (-$0.36);
4. the plates to 100 uses (-$0.19);
5. the steering package to ~$1 (-$0.17);
6. operations held under ~$0.2M a lob.

The lines left are the bowl and Sun plate ($0.36 + $0.32 on the spray cup's curve), the
sheets ($0.22) and S27's stage ($0.20).

## Consequences

- The headline (ADR 0077, `make solar-dive`) is unchanged. Every new lever defaults off.
- `plate_jupiter_loop` prices its deliveries through the shared `delivery_dollars`.
- **Open, for the impact simulation (via the parent):**
  - the stage laws below k = 5 (push, delivery) and k_S = 20, where the steady optimum
    is pinned;
  - a snow-foam charge in a thin polyethylene skin, flown for its own law;
  - the sheets' survival at the child's 4 Hz delivery (S37).
- **Not charged:**
  - the plates' recovery;
  - the reserve's tankage;
  - the booster's hardware and operations above propellant.
