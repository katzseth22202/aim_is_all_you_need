# Freeing the burn does not move the split's price, and the residue is a bend deficit

Status: accepted

Date: 2026-09-04

Builds on ADR 0026 (the split's tail and the SEP verdict), ADR 0013 (the
two-wave split and its chain) and ADR 0011 (the adaptive 2S/3S cadence and the
DSM proxy). Amends nothing: every number in those ADRs stands. It closes ADR
0026's first open item and CONTEXT.md's flagged ambiguity **"the two-wave
split's cost is a DSM proxy, not a deep-space maneuver"**.

## Context

ADR 0026 found the 20-day split's cost bimodal — eight of eleven cycles buy
their early arrival for under 2 m/s, three pay 1.4–2.1 km/s — and concluded
that argon solar-electric propulsion cannot reach the tail: the worst cycle
needs 2089.4 m/s where the array delivers 254.5, short by 8.2×. (The paper
flies a 10-day gap and the same three cycles pay 0.40–1.01 km/s there; this ADR
runs both, and the answer is the same at each. See ADR 0026's 2026-09-04
addendum for the 10-day column.) Then it said, in its own words, what the
finding rested on:

> The impulse integral is a **budget of thrust-seconds, not a low-thrust
> trajectory solution**: whether a burn distributed over 365+ days achieves the
> same endpoint change as the impulsive proxy it is compared against is
> unsolved, and it is the single largest open item here — a real low-thrust
> optimisation could move the required delta-v in either direction.

That impulsive proxy is ADR 0011's **seam value**: an exact velocity match at
the Jupiter patched-conic boundary, taken instantaneously. It is the most
constrained place in the whole trajectory that a burn could possibly go, and
ADR 0011 offered it as "an upper-bound-like architecture comparison" rather
than as a price. The suspicion is therefore specific and reasonable: a burn
free to happen somewhere else — earlier, later, split in two — might be far
cheaper, and the tail might be affordable after all.

Two ways to settle it. Build a Sims-Flanagan or collocation transcription of
the finite-thrust problem, which is weeks of work and carries its own
convergence risk; or **relax** the problem instead of transcribing it.

## Decision

**Bound it.** `src/free_dsm_bound.py` is committed as the reproducible harness
(`make dsm-bound`; not part of `make all`), and it solves a relaxation rather
than the thing itself, on one observation:

> The minimum-delta-v **impulsive** trajectory, with its burns free in time and
> place, is a lower bound on the minimum-delta-v **finite-thrust** trajectory
> between the same boundary conditions — because the impulsive problem is the
> finite-thrust problem with the thrust-magnitude constraint deleted, and a
> relaxation's optimum cannot be worse than the constrained problem's.

Both problems measure cost the same way, as the integral of thrust
acceleration, which is what the rocket equation charges. So a finite-thrust
solution is a feasible point of the impulsive problem at the same cost, and the
impulsive optimum can only be lower.

The trajectory posed is ADR 0013's boundary-value problem exactly: the growth
wave leaves Earth at the cycle's departure epoch and must reach Earth
`split_days` before the nozzle wave. Between them it flies an MGA-nDSM shape —
up to five maneuvers on the outbound leg, an unpowered Jupiter flyby at a free
epoch with a free aim point, up to five more on the return. Every maneuver is
free in time and direction and unbounded in magnitude.

**The flyby is left unpowered deliberately.** A perijove burn is an Oberth
device, and a solar-electric stage cannot fire one — it has no thrust to spend
in the few hours that matter. Allowing one would relax a constraint the
hardware really has, and the bound would stop bounding the question asked.

**The departure excess is free but uncharged**, because the growth wave's
departure is provided by the head-on nozzle rather than by propellant. It is
boxed to 8–20 km/s rather than left unbounded, or the optimiser buys the whole
correction at Earth for nothing and answers a different question.

Search box and settings, recorded per the ADR 0007 lesson: start 2026-08-11,
30-year horizon, 50 m/s fallback threshold, 20-day split gap, Astropy built-in
analytical ephemeris, zero-revolution Lambert arcs (Izzo, 350 iterations,
rtol 1e-8), Farnocchia coasts, 4,000 km perijove altitude floor. Free
parameters per cycle: departure excess speed 8–20 km/s with free direction;
Jupiter encounter within ±400 days of the seam solution's own; perijove 1–60×
the floor with a free B-plane angle; five maneuver epochs per leg spanning
1e-4 to 0.98 of the leg, each free maneuver carried as (magnitude ≤ 5 km/s,
right ascension, declination). Forty parameters in all. Engine: monotonic
basin hopping on the unit box, 60 hops from the seam-derived seed plus 5
restructured and 6 nudged seeds at 20 hops each and 4 random starts at 15,
local method L-BFGS-B (maxiter 400, finite-difference step 1e-7 in normalised
coordinates), seed 20260902. Blind check: 2,000 uniform samples of the same box
per cycle.

Three choices in that list were arrived at by getting them wrong first, and are
recorded in the module because each one silently produced a *worse* answer than
the trajectory being bounded:

- **Maneuvers are carried in polar form, not Cartesian components.** The
  objective sums `norm(dv)`, which has a kink at the origin; in Cartesian
  coordinates the finite-difference gradient there is about 10 in normalised
  units and swamps the small smooth gradients in timing and B-plane direction.
  In polar form the cost is linear in magnitude and `d(cost)/d(magnitude)` at
  zero is exactly the primer-vector test for whether an impulse is worth adding.
- **Extra maneuvers are seeded slightly off zero.** Seeded at exactly zero they
  sit on that kink, where every direction raises the cost to first order while
  any benefit is second order; L-BFGS-B stops immediately and a five-maneuver
  search returns more than its own seed.
- **The maneuver-epoch floor is 1e-4 of a leg, not 0.05.** The seam burns *at*
  Jupiter. A floor 36 days into a two-year return leg cannot express the very
  trajectory the bound has to be seeded with.

Differential evolution was tried first as the global engine and is kept only
behind a flag: unseeded it returned 3,367–4,432 m/s on a cycle whose seeded
value is 2,127, and worse at 40 parameters. MGA-nDSM landscapes are many narrow
basins rather than one broad one, which is what basin hopping is for.

## Consequences

- **Nothing helps.** On all three expensive cycles, at both split gaps, the
  free-burn optimum is the seam trajectory: a single burn just after Jupiter,
  at the perijove floor, at the seam's own encounter epoch, with every other
  maneuver left at zero magnitude and nothing at all spent outbound.

  At the 20-day gap ADR 0013 and ADR 0026 price:

  | departure | seam | free-burn optimum | ratio | spent outbound | perijove | encounter moved |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 2030-02-18 | 1584.955 | 1585.191 | 1.00015 | 0.003 m/s | 1.05595 R_J | +0.04 d |
  | 2036-09-07 | 2089.377 | 2089.704 | 1.00016 | 0.006 m/s | 1.05595 R_J | +0.07 d |
  | 2042-02-22 | 1425.619 | 1425.819 | 1.00014 | 0.002 m/s | 1.05595 R_J | +0.01 d |

  At the 10-day gap the paper flies (`--split-days 10`):

  | departure | seam | free-burn optimum | ratio | spent outbound | perijove | encounter moved |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 2030-02-18 | 556.029 | 556.112 | 1.00015 | 0.0005 m/s | 1.05595 R_J | +0.0004 d |
  | 2036-09-07 | 1011.273 | 1011.429 | 1.00015 | 0.005 m/s | 1.05595 R_J | +0.02 d |
  | 2042-02-22 | 398.470 | 398.828 | 1.00090 | 0.004 m/s | 1.05595 R_J | +0.10 d |

  Burns in m/s. The optimum comes back **0.015–0.09% above** the seam charge
  rather than exactly on it, and that residual is not a finding: it is this
  module rebuilding the seam trajectory in its own flyby model — a swept
  B-plane angle, a maneuver placed 1e-4 of a leg after the encounter — instead
  of importing it. The seeded starting point is 1.3% above the seam, so the
  search closes 98.9% of its own reconstruction gap and then stops. What
  matters is the sign: after freeing ten maneuvers, the encounter date, the aim
  point and the departure excess, there was nothing below.

- **The residue is an angle, and no placement buys an angle.** These cycles
  demand **107.6–109.6 degrees of turn and Jupiter supplies 104.2–107.2** even
  scraping the 4,000 km floor:

  | gap | departure | v∞ in (km/s) | turn needed | available at the floor | bend deficit | seam charge |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | 20 d | 2030-02-18 | 20.622 | 109.56° | 105.85° | **3.71°** | 1585.0 m/s |
  | 20 d | 2036-09-07 | 21.183 | 108.83° | 104.20° | **4.63°** | 2089.4 m/s |
  | 20 d | 2042-02-22 | 20.643 | 109.13° | 105.79° | **3.34°** | 1425.6 m/s |
  | 10 d | 2030-02-18 | 20.167 | 108.57° | 107.21° | **1.36°** | 556.0 m/s |
  | 10 d | 2036-09-07 | 20.802 | 107.65° | 105.32° | **2.33°** | 1011.3 m/s |
  | 10 d | 2042-02-22 | 20.190 | 108.12° | 107.14° | **0.97°** | 398.5 m/s |

  The chord that closes the deficit, `2 v∞ sin(deficit/2)`, runs 0.35–0.37 km/s
  per degree and accounts for **82–86%** of the seam charge; the rest is the
  speed error the exact match also pays. Halving the head start halves the
  deficit and so halves the bill, which is the whole difference between the two
  gaps. A maneuver anywhere on either leg can change where the wave is and how
  fast it is going. It cannot make Jupiter bend it further.

- **Slowing down before the flyby loses.** It is the one trade a multi-impulse
  solution could exploit and a single-impulse one cannot: turn authority rises
  as the arrival slows, since `e = 1 + r_p v∞² / mu`. The optimiser has five
  free maneuvers on the outbound leg with which to try it and leaves every one
  of them at zero — buying the turn costs more than the turn returns, because
  the same slowing has to be undone to reach Earth on time.

- **The perijove stays pinned at the floor.** The box allows 1–60× it and the
  optimum never leaves 1.0×. Every kilometre of extra perijove is turn
  authority given away, and these cycles have none to give.

- **So ADR 0026's verdict is now bounded rather than assumed** — and the
  distance from "bounded" to "proved" is worth stating exactly, because it is
  the whole epistemic content of this ADR. The relaxation argument is
  structural and holds. What the numerics supply is an *upper* bound on the
  relaxation's own optimum: a heuristic global search cannot certify that
  nothing cheaper exists, so the chain "finite-thrust ≥ true impulsive optimum
  ≤ what we found" does not close by itself. The verdict follows unless the
  true impulsive optimum lies well below everything a 40-dimensional
  multi-start search and 2,000 blind samples of the box turned up. How far
  below depends on which cycle and which gap: against the 254.5 m/s the array
  delivers, the cycle that sizes the hardware is over by **8.2× at the 20-day
  gap and 4.0× at the paper's 10-day gap**, and the least expensive of the
  three at 10 days is over by 1.6×. Two things make a miss that large unlikely
  rather than merely unproven: those are factors and not percentages, and the
  mechanism is visible in closed form — a bend deficit at the perijove floor,
  which no arrangement of maneuvers addresses. The 1.6× row is the one to watch
  if the ephemeris or the cadence ever moves.

- **Blind sampling says the seeded basin is not one of a crowd.** ADR 0008's
  rule is that an optimiser table means nothing until the box has been sampled
  blind, because an empty feasible set and a broken search look identical. The
  check runs the other way round here — the risk with a seeded search is that
  it never leaves the basin it started in — and the answer is the same
  reassurance:

  | gap | departure | blind draws flying a complete trajectory | cheapest of them | × the seam |
  | --- | --- | ---: | ---: | ---: |
  | 20 d | 2030-02-18 | 210 / 2000 | 96.31 km/s | 60.8× |
  | 20 d | 2036-09-07 | 253 / 2000 | 112.38 km/s | 53.8× |
  | 20 d | 2042-02-22 | 212 / 2000 | 98.36 km/s | 69.0× |
  | 10 d | 2030-02-18 | 221 / 2000 | 92.40 km/s | 166× |
  | 10 d | 2036-09-07 | 265 / 2000 | 110.47 km/s | 109× |
  | 10 d | 2042-02-22 | 219 / 2000 | 94.43 km/s | 237× |

  A tenth of the box flies, so "nothing cheaper was found" is a statement about
  cost and not about feasibility; and nothing random comes within fifty times
  the seam.

- **The two-wave split's headline is no longer exposed to a trajectory
  optimisation.** CONTEXT.md flagged ADR 0013's "8 of 11 cycles buy their split
  gap for under 1 m/s" as the claim in that ADR most exposed to a real
  optimisation, and load-bearing for the whole two-wave architecture. The
  exposure was one-sided in a way that flatters nobody: freeing the burn could
  only have made the *expensive* cycles cheaper, and it does not.

- **The paper's own caveat can go.** `sec:split_tail` closes with "a
  finite-thrust optimization could shift the requirement either way, and it is
  the one piece of work that could overturn the verdict". It cannot shift it
  downward past this floor, so the 0.79 MW that section quotes is a lower bound
  on what an electric stage would need rather than an estimate of it. Written
  up for the paper, self-contained, in
  `docs/paper_corrections_split_tail_2026-09-04.md`.

- **What this does not settle** is whether a low-thrust trajectory exists at
  all. The bound speaks about delta-v only. A finite-thrust solution that
  delivered 1.0-2.1 km/s over a 440-day leg would also have to keep the wave on a
  trajectory that still meets Jupiter and still reaches Earth on the day it is
  wanted, and nothing here says it can. The bound is useful because it points
  the wrong way for the optimist: it rules the tail out without needing that
  question answered.

## Reproducing

```bash
make dsm-bound                                    # the 20-day table, ~25 min
python -m src.free_dsm_bound --split-days 10      # the 10-day one, the paper's gap
pytest tests/test_free_dsm_bound.py -s            # 22 fast, 16 slow
make sep-split-10d                                # ADR 0026's 10-day column
```

Inputs, so this does not depend on scratch state: the three cycles come from
`expensive_cycles()`, which flies ADR 0013's own cadence (start 2026-08-11,
30-year horizon, 50 m/s nozzle-wave fallback threshold, 10- or 20-day split,
fixed mean two-synodic lattice of 797.734430 d) and keeps the cycles charged
more than `DEFAULT_MINIMUM_DV` — set at 254.5 m/s, what the array itself
delivers, rather than at a round number, because below that low thrust already
pays and there is nothing to test. The same three departures are selected at
both gaps. The seam prices are
`real_orbit_resonance._window_maneuver_solutions(..., cases=("dsm_only",))`,
unchanged from ADR 0026. The deliverable 254.5 m/s is ADR 0026's measured
impulse budget for the adaptive cadence's worst cycle at the stated 2e-5 m/s²
characteristic acceleration — an operating point the paper deliberately does
not assert (ADR 0026's 2026-09-04 addendum), so the paper-facing statement in
`docs/paper_corrections_split_tail_2026-09-04.md` is written against the
required *power* instead and quotes no shortfall ratio. The search box is the
one listed under **Decision** and lives in the module's constants, seed
20260902.

The slow tests run a reduced hop budget — bracketed around the answer rather
than searched from scratch, per the CLAUDE.md rule — because the CLI's own
budget is minutes per cycle. They assert what a cheaper search still has to
reproduce: that the optimum sits at the seam and not below it, that nothing is
spent outbound, that the perijove stays at the floor, and that blind sampling
finds nothing near.

## Open items

- **The search is heuristic and the bound is therefore empirical.** See the
  fifth consequence. A deterministic global method — interval branch-and-bound
  on the primer-vector conditions, say — would turn the argument into a proof
  and is not attempted.
- **The impulsive family is restricted in one way that is not obviously
  harmless**: both legs are zero-revolution Lambert arcs with a single Jupiter
  encounter. Multi-revolution arcs and repeated flybys are outside the box.
  Primer-vector theory covers the *number* of impulses (five per leg is
  generous), not the arc topology.
- **The other side of ADR 0026's open item stands.** This bounds the delta-v a
  finite-thrust stage would have to spend. It does not construct the
  finite-thrust trajectory, and ADR 0026's remaining open items — the nozzle
  wave's uncharged array, the uncharged Earth-departure spread, the reusable
  tug — are untouched.
- **The cheap cycles are not bounded**, only the three expensive ones. Nothing
  hangs on them: they are already bought for under 2 m/s, which is 1% of what
  the array delivers.
