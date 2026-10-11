# Hydrogen departure and growth cost: exploratory record, 2026-10-10

## Status and scope

These are conditional analysis results, not a replacement for the production
growth ledger. The accompanying [JSON snapshot](../data/hydrogen_curvature_cost_2026-10-10.json)
records results, trajectories, complete financial input records and price books.
Source revision: `63db991a966d5eaab8c52035bf36637d96f109f4`.
No production physics or financial defaults were changed.

The requested horizon is **15 Earth–Jupiter synodic periods = 16.380584 years**.
Two different comparisons must be kept separate:

1. A controlled change of harvest horizon using the original production routes.
2. An exploratory comparison on a common circular three-synodic-period route,
   with signed impact momentum and Earth visibility constraints. This also
   changes the route and early-wave correction, so it does not isolate the
   benefit of curvature. Applying only the geometry change to the original
   mixed 2S/3S routes remains unfinished.

## Why propellant fraction changes price so much

At equal 1,200 s burn duration, hydrogen burned is 38.06% of the ignition
stack at 5,500 K versus 53.14% at 3,000 K. Including consumed plugs, total
onboard expenditure is 41.70% versus 53.14%. These fractions exclude earlier
boiloff and externally supplied projectiles; they are not fractions of the
initial 1,500 t launch unit.

The selected financial cases use different burn durations:

| Quantity | 5,500 K hydrogen | 3,000 K hydrogen |
|---|---:|---:|
| Burn duration | 2,500 s | 1,200 s |
| Ignition stack | 769.35 t | 779.83 t |
| Hydrogen burned | 304.48 t | 414.37 t |
| Hydrogen loaded, including storage losses | 323.91 t | 440.82 t |
| Tank | 62.42 t | 84.95 t |
| Chamber and nozzle | 34.19 t | 34.19 t |
| Onward payload | 336.03 t | 246.33 t |
| Growth multiplier per 3S cycle, G | 2.4384 | 1.8485 |
| Fraction reinvested to sustain the fleet, 1/G | 41.01% | 54.10% |
| Fleet multiplier over four growth cycles, G⁴ | 35.36 | 11.68 |
| Initial seed | 137.81 t | 133.26 t |

The cooler case carries about 27% less onward payload. Four growth cycles
amplify the difference to a factor of 3.03 in fleet multiplication. Lower
manufacturing volume also leaves hardware higher on its learning curve.
Buying hydrogen itself accounts for only $1.23 versus $2.93 per delivered kg.
Other learned operating charges include lob launches ($71.19 versus $90.21/kg),
chamber hardware ($8.13 versus $16.88/kg), and plates ($15.13 versus $21.67/kg).
Thus reduced output and slower growth dominate the financial difference.
This selected-case comparison is not a temperature-only experiment.

## Financial definitions and results

All prices below are dollars per kg delivered to L1. Operating cost excludes
the initial seed and is measured on steady-operation lap 3, with hardware
learning included. Program break-even is the selling price making expected
discounted net present value zero, including seed, growth and operations.
It is not a market-price forecast. The valuation uses 10% annual discounting
and 50% probability of cycle success, resolved at the first growth return.

The harvest horizon selects when growth ends and harvesting begins; it is
**not a payback deadline**. The steady-operation valuation extends cash flows
to year 300 as an approximation to a perpetuity. Liquidation instead sells
the whole final batch and ends the program. The JSON separately records a
strict 15S payback calculation with no later revenues or terminal fleet value.

### Common circular 3S route

Ranges represent cheap and expensive seed transport assumptions.
Thermal incumbents were searched at 1,200 and 2,500 s and on both mirror
branches, selecting best growth; this is not a global duration optimization.
The main 3,000 K case stays fixed at 1,200 s. Methalox uses two Raptors.

| Case | Operating, 10 y | Operating, 15S | Program break-even, 10 y | Program break-even, 15S | Liquidation, 15S |
|---|---:|---:|---:|---:|---:|
| H₂, 3,000 K | 159.2 | 149.9 | 360.8–1,186.6 | 339.7–792.0 | 361.2–1,133.7 |
| H₂, 5,500 K | 115.8 | 108.4 | 214.0–587.7 | 177.7–297.6 | 201.8–461.3 |
| CH₄, survivable AR300 | 143.4 | 131.4 | 335.3–997.8 | 294.3–609.4 | 330.9–911.9 |
| Methalox | 159.9 | 155.1 | 328.6–1,295.4 | 328.5–911.4 | 329.9–1,267.9 |

The fifth return occurs at 15S, after four growth reinvestments following the
seed's first return. Relative to the discussed $200/kg target, both hydrogen
cases have lower modeled operating cost; only the cheap-seed 5,500 K case
has lower program break-even in this exploratory comparison. The target's
scope would need matching before claiming economic equivalence.

### Original routes: horizon-only rerun

Physics and price assumptions are held fixed; only the selected harvest changes.

| Case | Operating, 10 y → 15S | Program break-even, 10 y | Program break-even, 15S |
|---|---:|---:|---:|
| Current H₂, 5,500 K | 121.58 → 114.98 | 192.46–376.97 | 167.46–221.59 |
| CH₄, AR300 mass-priced | 161.74 → 149.18 | 340.00–809.73 | 300.76–511.35 |
| Methalox | 293.64 → 286.29 | 776.43–>3,000 | 998.61–>3,000 |

The mixed thermal chain's last completed return before 15S is at 14S
(15.28854 years). Methalox returns at 15S. Delaying harvest need not improve
value: additional growth can be worth less than the income postponed.

## Initial seed bill: common-route 3,000 K case

The 133.26055 t seed comprises 122.28509 t of growth PuffSats and 10.97546 t
of rods. Manufacturing assumes $3/kg for growth bodies, $4/kg for rod bodies,
and 17,246.72 packages at $100 each.

| Item | Cheap seed, million dollars | Expensive seed, million dollars |
|---|---:|---:|
| Expended ship hulls | 6.327 | 31.243 |
| Seed ascent flights | 2.531 | 78.108 |
| Tanker flights | 30.371 | 937.301 |
| Growth bodies | 0.367 | 0.367 |
| Rod bodies | 0.044 | 0.044 |
| Packages | 1.725 | 1.725 |
| Total | **41.365** | **1,048.788** |

Each seed mission requires one ascent and twelve 100 t tanker deliveries.
Cheap/dear assumptions are 40/60 t stripped ship dry mass, $5M/$20M hull,
and $2M/$50M per flight. Useful seed payload is 105.30489/85.30489 t per
mission, giving 1.26547/1.56217 normalized missions. These are fractional
cost allocations, not integer procurement plans. Transport is
$294.383/$7,854.181 per seed kg before manufacturing. These prices are
assumptions, not supplier quotes, and exclude later growth launches.

## Physical assumptions and limits

- H₂ tanks retain the existing 20.5% mass allowance, with a separate 3%
  cryostat and 6% storage loss over 20 days. No composite tank mass saving
  is credited.
- The 3,000 K case has zero chamber ablation and zero plugs, as requested.
  Its thermal survival is assumed, not established. Pusher-plate film remains.
- The 34.1855 t chamber/nozzle and its price are borrowed from the hotter
  case; they are not a new thermal design for 3,000 K.
- The 3,000 K result also removes the entire $2M chamber-spray allowance.
  That item bundles graphite renewal with charge-sprayer equipment, so this
  is optimistic. Retaining it gives approximately $345/$797 program
  break-even instead of $340/$792, without introducing ablative mass.
- The common route adds a 0.699240252 km/s early-wave outbound-vector
  correction outside Earth's gravity well, a conservative proxy. It retains
  the existing apoapsis-reversal charge and changes the periapsis-raise charge.
  It does **not** close the parking orbit's apsidal orientation against the
  optimized departure. An end-to-end architecture remains to be demonstrated.
- Optimized impacts remain broadly head-on on this route. Signed momentum
  does not imply forward impacts or elimination of projectile recoil.

## Method and provenance

The circular reference uses `analyze_circular_resonant_impacts` with 91 phase
samples and 121 encounter samples: Earth–Jupiter phase −93.40825486 degrees,
outbound time 1.20000958 years, period 3.276116757 years, outgoing/incoming
Earth-relative asymptotic speeds 11.56427943/55.17834131 km/s, and asymptote
separation 144.89995040 degrees. The early wave is 20 days earlier, solved
using circular planet states and the existing `_cheapest_return` search.

The vector departure law uses relative speed w in km/s, a velocity-aligned
nozzle, and constant onboard expenditure rate:

```
c(w) = A (w / 75)^2 - 1 + pitch
ve = w sqrt(eta / (c + 1)) (1 - gate)
impulse per onboard kg = (1 + 1/c) ve axis + relative_velocity_vector / c
rod_flow = onboard_flow / c
gas_spent = onboard_spent - (plug + pitch) rods
```

For 3,000 K, A = 26.76 × 5500/3000, eta = 0.85², and plug, pitch and gate
are zero. For 5,500 K, A = 26.76 with the source's solved efficiency,
1.76 plug ratio and gate. Methane uses `survivable_design(300)` with pitch
24/5 and plug ratio 4. Peak cadence is constrained to 4 Hz H₂ and 2 Hz CH₄.

SLSQP varies orientation, fraction of burn before reference periapsis,
spent fraction, and altitude/10,000 km, with respective bounds
[−π,π], [0,1], [0.05,0.9], [0.06,2]. Settings: ftol 1e−10, maxiter 150.
Constraints enforce the outgoing velocity vector, Earth visibility and
missed-projectile periapsis at least 600 km. Optimization uses 121 samples,
20 s maximum integration step, and relative/absolute tolerance 3e−8.
Verification uses 1,201 samples, 2 s maximum step, relative tolerance 1e−10
and absolute tolerance 1e−9, requiring endpoint residual below 1e−6 and
clearance within 0.001 km. Both mirrored trajectories are evaluated.
Launch-push pricing scans 31 logarithmic points from 0.001 to 1 and refines
the top three intervals to account for integer chamber counts.

For the selected 3,000 K solution, reference periapsis altitude is
2,670.03069 km, all 1,200 s occur before reference periapsis, onboard spent
fraction is 0.5313599343, rod fraction is 0.01407413574, and integrated
propulsive velocity change is 5.79806 km/s.

## Verification recorded during the study

- Growth-cost tests: 20 passed.
- Orbit-turn tests excluding slow cases: 9 passed, 6 deselected.
- Vector propulsion law reproduces the existing head-on effective-Isp law
  at 60, 70 and 80 km/s to relative tolerance 1e−12 for H₂ and CH₄.
- Independent 3,000 K integration reproduces the existing reference spent
  fraction within 1e−5 and altitude within 0.3 km.
- Trajectory endpoint, clearance, cadence and mass-accounting checks passed;
  mass closure tolerance was 1e−6 kg.
- Seed line-item total equals the financial model's seed outlay.

These checks support the recorded calculation under its assumptions; they
do not establish chamber survival or resolve the parking-interface limitation.
