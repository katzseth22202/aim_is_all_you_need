# The two-wave split is nearly free in real orbits, and it is the architecture that pays

Status: accepted

Date: 2026-08-17

Amends: ADR 0009 (its point 4 concluded the two-wave split "must be bought"
with a 0.326-0.626 km/s perijove burn; that holds only in the
circular-coplanar model, where the bend is the sole knob). Everything else in
ADR 0009 stands. Builds on ADR 0011 (the adaptive 2S/3S cadence) and ADR 0012
(the impulse law and its head-on endpoint).

## Context

ADR 0009 left the architecture choice open between two ways of running the
growth loop, and the gap between them turned out to be the single largest
number in the whole accounting:

- **One-wave parked.** One PuffSat wave arrives per cycle and does two jobs:
  push the newly-lofted payload into the parking orbit, and be the head-on
  nozzle projectiles for the payload that is departing right now. Those cannot
  be the same payload, so a mass pushed this cycle departs with *next* cycle's
  wave. Push, depart, return spans two cycles and the steady per-cycle growth
  solves `g^2 (1+sigma)/(r M) + (sigma/k) g = 1` -- a square-root law.
- **Two-wave same-cycle.** The departing batch splits at Jupiter so a growth
  wave arrives early, pushes the payload, and a nozzle wave arrives one
  parking-orbit period later to depart it. Growth is then linear in the same
  quantities: `g [ (1+sigma)/(r M d1) + sigma/k ] = 1`.

At the same inputs the two differ by roughly the square: 1.93 against 3.81 per
cycle at `e = 0.6, k = 7`. ADR 0009 priced the split in the circular-coplanar
model, where the only knob is perijove radius, found its Earth-hit roots about
a year apart, and concluded the 10-20 d gap had to be bought with a perijove
burn. That conclusion was never tested against real ephemerides, where the
Lambert pair also has a free encounter time.

Two further things were unresolved. The split gap and the parking-orbit period
are the *same number* -- the payload coasts one full orbit between the two
waves -- but ADR 0009 priced a `dt = 10 d` split against a 20-day apoapsis
reversal. And the growth-push elasticity `f` had never been swept: every
result in the repo assumes the paper's 0.8.

## Decision

**Fly the two-wave split, on a 10-day gap, and score it on real orbits.**
`src/two_wave_growth.py` is committed as the reproducible harness
(`make two-wave`; not part of `make all`). It flies ADR 0011's adaptive
cadence -- a two-synodic return whenever its deep-space-maneuver proxy stays
inside 50 m/s, three-synodic otherwise, each selected return becoming the next
departure -- and prices every flown cycle on ADR 0009's two-wave ledger.

The split is **re-solved per cycle against the same ephemeris**, as a second
Earth-hit return leaving the same departure epoch and arriving `split_days`
before the nozzle wave. The growth wave therefore gets its own collision speed
from geometry rather than by extrapolation, and both real-orbit correction
burns -- the growth wave's cost of arriving early and the nozzle wave's DSM
proxy -- are charged as methalox. ADR 0011 reported the DSM but never priced
it.

Search box and inputs, recorded per the ADR 0007 lesson: start 2026-08-11,
30-year horizon, 50 m/s fallback threshold, Astropy built-in analytical
ephemeris, zero-revolution Lambert arcs on both legs, 4,000 km perijove
altitude floor, fixed mean two-synodic lattice of 797.734430 d, methalox
`v_e` = 3.7265 km/s (380 s vacuum). Slug ratio searched over `k` in [0.2, 80],
60 log-spaced starts then bounded scalar refinement, one `k` for the whole
chain because the fleet builds one nozzle. Swept: recovery `e` in
{0.25, 0.3, 0.4, ..., 0.9}, elasticity `f` in {0.5, 0.6, 0.7, 0.8}, split gap
in {10, 20, 30, 60} d with the apoapsis reversal resized to match.

## The algebra, in one place

Every formula below is closed-form and reproduces from
`src/nozzle_analysis.py` and `src/circular_resonance_impulse.py`. It is
recorded here because the derivations previously lived only in a gitignored
scratch note.

**The impulse law.** An impactor of mass 1 arrives at closing speed `w`
relative to the vehicle and vaporises a carried slug of mass `k`. The merge is
inelastic, so the blob masses `1+k` and its centre of mass drifts at
`V = w/(1+k)`. The impact energy re-expands as a collimated jet whose speed
follows from momentum-out-of-energy, `p = sqrt(2 m E)`: spreading the same
energy over `1+k` kilograms gives a jet speed `u = w/sqrt(1+k)` relative to
the blob. The mass scales as the square root because the energy is fixed and
the momentum is not.

For a **head-on** impact the CM drift is backward, away from the useful
direction, so it is subtracted:

    impulse per impactor kg = (1+k) u - w = w (sqrt(1+k) - 1)

For an **overtaking** impact the drift is forward and adds:

    impulse per impactor kg = (1+k) u + w = w (1 + sqrt(1+k))

Both are endpoints of ADR 0012's single angle-dependent law
`beta(theta, k) = sqrt(1 + k - sin^2 theta) + cos theta`, at 180 and 0 degrees.
At `k = 0` the head-on device yields nothing (no slug, no jet) while the
overtaking one yields `2 w`, the paper's elastic plate -- which is the
convergence check that licenses reusing `f` as the recovery factor.

**Effective exhaust speed.** Charging the impulse against the propellant
actually spent -- the slug, `k` kilograms per impactor kilogram -- and derating
by a uniform recovery `e`:

    v_e = e w (sqrt(1+k) - 1) / k

Equivalently `e (u - V) (1+k)/k`, which is how the two terms above appear when
the algebra is written as jet-speed-minus-recoil times blob-mass-over-slug-mass.
Note `sqrt(1+k)`, not `sqrt(k)`: the re-expanding mass is impactor *plus* slug.
At `k = 3` the difference is 31%.

**Slug per craft kilogram.** The vehicle accelerates into the stream during the
burn, so `w = v_b + v` climbs. Integrating `dv/(v_b + v) = e beta (-dm/m)`
across the burn with `Lambda = ln(w_final / w_initial)`:

    sigma = exp( Lambda k / (e (sqrt(1+k) - 1)) ) - 1

so the departure delivers `1/(1+sigma)` of the parked payload as craft.
(`_sigma()` in `src/nozzle_analysis.py`.)

**The growth push.** The mint side is unchanged from the paper: a payload
pushed from rest to `v_rf` by collisions at `v_b` yields

    M = 2 f / ln( v_b / (v_b - v_rf) )

with `v_rf` = 10.9503 km/s, the closed cycle's 200 km periapsis speed -- not
escape. The push target is sub-escape because of aim, not propulsion
(ADR 0009). (`payload_mass_ratio()` in `src/propulsion.py`.)

**The two ledgers.** Writing `r` for the apoapsis-reversal factor
`exp(-dv_reversal / v_e_methalox)` and `d1` for the growth wave's correction
factor `exp(-dv_split / v_e_methalox)`:

    two-wave same-cycle:  g [ (1+sigma)/(r M d1) + sigma/k ] = 1
    one-wave parked:      g^2 (1+sigma)/(r M) + (sigma/k) g = 1

The `(1+sigma)` denominator is the parked payload splitting into craft and
slug; the `sigma/k` term is the impactors the nozzle consumes, charged against
the arriving wave at its own opportunity cost. The parked form's `g^2` is the
one-cycle parking delay, and it is worth roughly a square root of the growth.

## Consequences

- **The split is nearly free in real orbits, and ADR 0009's "it must be
  bought" does not survive the move off the circular model.** Eight of the
  eleven flown cycles buy their 10-day-early growth wave for under 1 m/s; only
  three pay at all, at 0.3985, 0.5560 and 1.0113 km/s, for a chain mean of
  0.1791 km/s. The circular-coplanar model had one knob (perijove radius *is*
  the bend) and its Earth-hit roots sit about a year apart; real ephemerides
  add a free encounter time, and that second knob buys ten days for almost
  nothing. This is most of why the doubling times below beat ADR 0009's
  1.69-1.80 yr.

- **The split gap is the parking-orbit period, and 10 days wins -- narrowly.**
  Priced at `e = 0.6, f = 0.8`, with the apoapsis reversal resized to the gap:

  | split | growth-wave burn | reversal | k* | growth over 28.39 yr | e-fold/yr | doubling |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 10 d | 0.1791 km/s | 372.5 m/s | 8.53 | x83,070 | 0.3989 | 1.737 yr |
  | 20 d | 0.4641 | 233.9 | 8.51 | x76,820 | 0.3962 | 1.749 |
  | 30 d | 0.8441 | 178.3 | 8.26 | x45,720 | 0.3779 | 1.834 |
  | 60 d | 3.1440 | 112.1 | 6.18 | x411 | 0.2120 | 3.270 |

  10 and 20 days are within 0.7% of each other: the cheaper reversal a longer
  orbit buys is very nearly cancelled by the larger burn needed to pull the
  growth wave that much further ahead. The trade only collapses at 60 days.
  ADR 0009's `dt = 10 d` row had the right gap attached to the wrong reversal.

- **The flown chain is 11 cycles, 7 two-synodic and 4 three-synodic, spanning
  28.3930 yr** from 2026-11-09 to 2055-04-02. Departure burns run 5.32-7.17
  km/s, nozzle-wave collision speeds 54.49-63.31 km/s, growth-wave collision
  speeds 56.53-65.13 km/s -- the growth wave always closes faster, because
  arriving early means arriving hotter. Maximum DSM across the chain is
  18.479 m/s, comfortably inside the 50 m/s policy trigger.

- **Growth over the flown chain, by recovery and elasticity** (e-foldings/yr,
  doubling in years, and the chain-optimal slug ratio):

  | `e` | f = 0.5 | f = 0.6 | f = 0.7 | f = 0.8 |
  | ---: | ---: | ---: | ---: | ---: |
  | 0.25 | -0.083 / inf / 4.53 | -0.041 / inf / 5.00 | -0.007 / inf / 5.42 | 0.021 / 32.70 / 5.82 |
  | 0.30 | 0.009 / 79.70 / 4.94 | 0.053 / 13.00 / 5.47 | 0.090 / 7.72 / 5.97 | 0.120 / 5.76 / 6.42 |
  | 0.40 | 0.130 / 5.33 / 5.55 | 0.179 / 3.87 / 6.19 | 0.219 / 3.16 / 6.79 | 0.253 / 2.74 / 7.35 |
  | 0.50 | 0.208 / 3.34 / 5.98 | 0.260 / 2.67 / 6.70 | 0.303 / 2.29 / 7.38 | 0.339 / 2.05 / 8.02 |
  | 0.60 | 0.262 / 2.65 / 6.31 | 0.316 / 2.20 / 7.09 | 0.361 / 1.92 / 7.83 | 0.399 / 1.74 / 8.53 |
  | 0.70 | 0.301 / 2.30 / 6.56 | 0.357 / 1.94 / 7.39 | 0.404 / 1.72 / 8.18 | 0.444 / 1.56 / 8.93 |
  | 0.80 | 0.332 / 2.09 / 6.76 | 0.389 / 1.78 / 7.63 | 0.437 / 1.59 / 8.46 | 0.478 / 1.45 / 9.25 |
  | 0.90 | 0.356 / 1.95 / 6.92 | 0.415 / 1.67 / 7.82 | 0.464 / 1.50 / 8.69 | 0.505 / 1.37 / 9.52 |

- **The same sweep as a continuous exponential**: annual mass increase, and the
  multiple reached if that rate is run out to a full 30 years. The chain
  actually flies 28.3930 yr, so the projection extrapolates the last 1.6 yr
  rather than flying it, and assumes the cadence keeps finding equally good
  windows.

  | `e` | f = 0.5 | f = 0.6 | f = 0.7 | f = 0.8 |
  | ---: | ---: | ---: | ---: | ---: |
  | 0.25 | -7.95% / x0.083 | -4.03% / x0.292 | -0.71% / x0.806 | +2.14% / x1.889 |
  | 0.30 | +0.87% / x1.298 | +5.48% / x4.951 | +9.39% / x14.78 | +12.79% / x37.03 |
  | 0.40 | +13.90% / x49.59 | +19.61% / x215.3 | +24.52% / x719.9 | +28.83% / x1,995 |
  | 0.50 | +23.07% / x506.2 | +29.64% / x2,409 | +35.32% / x8,734 | +40.34% / x2.60e4 |
  | 0.60 | +29.88% / x2,551 | +37.13% / x1.30e4 | +43.44% / x5.01e4 | +49.03% / x1.58e5 |
  | 0.70 | +35.15% / x8,408 | +42.95% / x4.52e4 | +49.76% / x1.83e5 | +55.82% / x6.00e5 |
  | 0.80 | +39.35% / x2.10e4 | +47.60% / x1.18e5 | +54.83% / x4.96e5 | +61.28% / x1.69e6 |
  | 0.90 | +42.77% / x4.36e4 | +51.40% / x2.53e5 | +58.98% / x1.10e6 | +65.76% / x3.84e6 |

  The annual figure is the honest headline for a process this lumpy: mass
  arrives in 2.18 and 3.28 year steps, so "49% a year" at the reference point
  is a smoothed rate, not something observable at any single epoch.

- **The architecture dies below `e` about 0.3, and that is the real design
  requirement.** At `e = 0.25` -- a bare paraboloid's free-molecular capture at
  `k = 3` -- the chain is flat or shrinking at every `f` below 0.8. Everything
  above `e = 0.4` doubles inside 5.5 years and the top corner reaches 1.37 yr.
  So the question the hardware has to answer is not "how good is the nozzle"
  but "does it clear one third of the ideal collimated impulse".

- **`f` is worth about as much as `e`, and it was never swept before.**
  Dropping the growth push from the paper's `f = 0.8` to 0.5 costs 0.11-0.15
  e-foldings/yr, comparable to a 0.2 swing in nozzle recovery. Any claim about
  nozzle hardware should carry the `f` it assumed.

- **The chain-optimal slug ratio rises with both**, 4.5 to 9.5 across the
  table, and is interior to the search box everywhere. It sits near ADR 0009's
  two-currency `k*` of 6-12 and above the bare-dish Isp optimum of 7.06,
  because the chain charges impactor throughput as well as exhaust speed.

- **Inherited caveats, unchanged.** The split cost is priced with ADR 0011's
  DSM proxy: an exact velocity match at the Jupiter patched-conic seam, not a
  finite-location interplanetary deep-space maneuver. A true DSM optimization
  could move it, and "the split is nearly free" is exactly the claim most
  exposed to that. The ephemeris is Astropy's built-in analytical model, and
  any claim past 2100 needs DE-kernel verification -- not in play for this
  2026-2055 horizon. The recovery `e` remains a single lumped factor standing
  in for collimation, capture and plasma coupling, with no source calibrating
  any of them.

- `same_cycle_nozzle()` gains `fudge` and `reversal_period` parameters, both
  defaulting to previous behaviour, so ADR 0009's pinned numbers are unchanged
  and act as the regression check.
