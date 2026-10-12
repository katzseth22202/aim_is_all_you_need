# Hydrogen follow-ups: selected designs and comparison limits

> **2026-10-12:** the 3,000 K and 3,500 K figures here use a hot energy scaled
> linearly from 5,500 K, which ADR 0049 shows overstates it by up to two-thirds.
> Treat them as void; the 5,500 K and methane figures are unaffected.

These exploratory results extend ADR 0046 at source revision `930083e`.
The durable inputs are `data/hydrogen_followups_2026-10-10.json`: 16 financial
cases, 46 finite-burn profiles, physical design summaries, and the adaptive
ephemeris route solutions/vectors. They require no ignored caches.

Reproduce selected-design accounting and independently integrate departures:

```sh
PYTHONPATH=. .venv/bin/python docs/hydrogen_followup_analysis.py --verify-trajectories
```

This is a replay of selected designs, not an end-to-end rerun of all searches.
The snapshot preserves full financial inputs and price books rather than only
headlines. The companion trajectory module also exposes `optimize(geometry)`.

## Assumptions and search provenance

1500 t launch units; 90% tank and 60% chamber/nozzle recovery; each independent
stack carries a 5 t/$1M recovery machine. Recovered material incurs ordinary
PuffSat body/steering costs. No free machine recovery credit. Hydrogen burns
last 1200 s at 85% exhaust-velocity efficiency, without ablation or the old
bundled chamber-spray allowance. At least one 34.1855 t chamber/nozzle per
stack, with additional units required above 4 Hz. The launch push retains
the spray-cup, argon, and film assumptions. Valuation: 10% discount, 50% success,
harvest at the last complete return within 15S, revenues through year 300.

Departure search: both mirrors independently; SLSQP minimizes hydrogen spent.
Variables are rotation angle [-pi,pi], pre-periapsis burn fraction [0,1], spent
fraction [.05,.9], and periapsis altitude [600,20000] km. Initial values and
the full signed projectile momentum law are in `hydrogen_followup_trajectory.py`.
Constraints enforce target outgoing asymptote, Earth clearance, and a 600 km
missed-projectile floor. SLSQP uses 150 iterations and ftol 1e-10; final
integration uses 1201 samples, <=2 s steps, rtol 1e-10 and atol 1e-9.

Early-wave launch price search: 41 logarithmic points from .001 to 1, local
bounded refinements with log-price tolerance 1e-6. Adaptive: 31 points from
.001 to .2, same refinement tolerance. These are conditional local searches,
not proof of a global optimum. Both refine the three best coarse intervals
and retain coarse candidates. Split studies enumerate one to three chambers
per stack; adaptive stream ratios use damped fixed-point iteration (200 steps,
1e-10 convergence tolerance). Saved financial records define the selected
temperature designs and their exact settings. Temperature retuning maximizes
growth on 31 logarithmic launch-price points from .001 to 1 with bounded local
refinement (log-price tolerance 1e-7), rather than claiming a global cost minimum.

## Temperature

The 3000 K control reproduces $169.63/$269.55 cheap/dear break-even and
$107.07/kg operating cost. At 3500 K, keeping the launch setting would require
4.49 Hz with one chamber: adding the second yields $286.19 dear break-even.
Retuning to one chamber at 4 Hz yields $168.60/$251.27 and $107.92 operating.
Neither reaches $200 dear at 3500 K.

## Earlier growth wave

At Earth-minus-Jupiter phase -93.40825486 degrees, the earlier return is
2.735995S, 68.617042 km/s at 200 km, departure excess 13.830550 km/s.
The route search's impulsive departure increment is 6.726636 km/s above its
20-day parking-orbit periapsis speed at 200 km, not above circular LEO speed.
It uses the source revision's circular `jovian_cycle_phasing._cycle_branches`
with zero flyby burn, the fixed phase above, a 7-year total-time cap, and a
lowered outbound-time minimum of 0.7 years (maximum 5.8 years). It selects the
lowest departure-burn branch among surface collision speeds 67.5--69 km/s
and return times 2.6--2.85S, then converts speed to 200 km altitude. Remaining
branch-search settings are those at the recorded source revision.
It precedes the 56.265787 km/s later wave by 105.042032 days. Hydrogen spent
fractions are .637970442 and .528300986. Both stacks consume the later
projectile wave; the output split balances future growth and rod consumption.

| Seed transport treatment | Existing 31.5126% storage loss | Optimistic 6% loss |
|---|---:|---:|
| Actual trajectories | 821.02 | 455.15 |
| Grant original 3S seed price | 472.47 | 297.04 |
| Free transport, manufacturing charged | 285.32 | 195.47 |

Values are dear-seed program break-even $/kg, separately optimized. The 6%
case assumes better storage without charging additional equipment. Granting
the old seed price does not solve the bootstrap transition. Early-route
cheap/dear seed transport is $656.98/$24645.38 per kg, versus ordinary
$294.38/$7854.18. Actual-route operating costs are $142.84/$121.10 for the
two loss assumptions; chamber counts are [1,1]/[2,1].

A 105-day ellipse with 400 km periapsis has apogee radius 1,874,013 km,
beyond the roughly 1,496,559 km Earth Hill radius: parking is not validated.
Independent departure orientations do not close the common parking interface.

## Adaptive 2S/3S with a 20-day wave split

Real ephemeris routes starting at TDB JD 2461354.1393931713 over the recorded
30-year horizon, each wave's correction
and seed price charged. A cyclic fixed point balances the two output streams.
Exactly two $1M machines are charged per flight, independent of chamber count.

| 2S acceptance rule | Sequence | Cheap/dear break-even | Operating |
|---|---|---:|---:|
| Later wave correction <50 m/s | 3,2,2,2,2,3,2,2,2,3,3 | 298.84/489.13 | 158.97 |
| Both corrections <50 m/s | 3,3,2,3,3,3,2,3,3 | 234.23/401.30 | 130.78 |

Harvest begins at 14S (15.288545 years). The first policy admits earlier-wave
corrections of 1.585, 2.089, and 1.426 km/s; these are charged, not overlooked.
One common launch price is optimized. Burn duration, cycle-specific tuning,
and shared parking-plane closure are not optimized. Financial repetition of
the sampled ledger to year 300 is not 300 years of trajectory validation.

The saved departures pass normalized endpoint residual <1e-6 and corridor
checks to 1 m. Financial replay checks mass closure, cost-line sums, and all
16 recorded prices. This supports the limited conclusion in ADR 0047:
ordinary 3S is preferred among the configurations tested, not universally best.

Verification for this record: full `make test-all` passed with 877 passed and
1 skipped (23 warnings); `make test-changed` selected no affected tests because
production source is unchanged. The standalone replay passed all 16 financial
cases, hardware/stream checks, and all 46 integrated departure profiles.
