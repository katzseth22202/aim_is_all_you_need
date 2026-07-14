# The powered Jovian flyby optimizes end-to-end mass ratio, not minimum delta-v

Status: accepted

Date: 2026-07-14

## Context

The catalog's three Jovian-return rows (`sec:jupiter_gravity_initial` x2,
`sec:jupiter_only_growth`) all assume PuffSats arrive at Earth on a retrograde
Jupiter-to-Earth Hohmann ellipse, closing at ~69 km/s
(`retrograde_jovian_hohmann_transfer()`). Nothing in the repo modelled how a PuffSat
gets *onto* that retrograde trajectory. The missing leg is: an Oberth methalox burn at
200 km above Earth (starting from C3 = 0, PuffSat-provided), a coast to Jupiter, and a
powered gravity assist there — a second methalox burn at Jovian periapsis — that bends
and pumps the trajectory into a retrograde heliocentric return.

The natural-sounding formulation, "choose the two burns to minimize total propellant
subject to a retrograde 1 AU crossing," turned out to be **ill-posed**. The Jupiter
burn cost is set by the outgoing hyperbolic excess `|v_inf_out| = |v_helio_out -
v_Jupiter|`, which is minimized by making the post-flyby heliocentric velocity as
small as possible: the optimizer drives the tangential component to 0- (a
near-radial plunge from 5.2 AU, `|v_inf_out| -> 13.1 km/s`) instead of the retrograde
Hohmann (`|v_inf_out| ~ 20.5 km/s`). The boundary `v_t = 0` is excluded by the
retrograde requirement, so no minimizer exists, and every near-optimum is a
barely-retrograde plunge that crosses 1 AU at ~50 km/s closing speed instead of
~69 km/s — cheaper propellant, weaker collision. Propellant cost and collision
velocity `v_b` are in direct tension; an objective that ignores `v_b` degenerates.

A second trap: unconstrained, the search wanders into extreme apoapsis-raise
trajectories (outbound aphelion far beyond Jupiter, lazy multi-year returns) that are
numerically fine but operationally absurd.

## Decision

1. **Optimize the end-to-end mass ratio, a single well-posed scalar.** Maximize
   (delivered mass fraction after both methalox burns, via the rocket equation) x
   (payload/PuffSat mass ratio at the achieved Earth-closing `v_b`). This folds
   propellant and collision strength into the repo's existing currency — payload
   pushed per unit mass launched — and has an interior optimum: pushing from a
   plunge-like return toward the retrograde Hohmann costs ~2 km/s more at Jupiter
   (delivered mass x ~0.59) while the mass-ratio reward grows only ~1.4x, so the
   optimum sits strictly between plunge and Hohmann.

2. **Score against the `sec:jupiter_only_growth` push.** The mass-ratio factor uses
   `v_rf` = lunar-transfer periapsis speed (~10.9 km/s), the exponential
   launch-growth loop this leg physically closes. The two Parker rows are *reported*
   at the resulting `v_b`, not optimized for (a higher `v_rf` rewards `v_b` more
   steeply and would shift the optimum).

3. **Cap total time of flight at 7 years.** Outbound Earth-to-Jupiter plus return
   Jupiter-to-1 AU heliocentric time of flight must not exceed
   `JUPITER_FLYBY_MAX_TOF = 7 yr` (time inside Jupiter's sphere of influence is
   days and is ignored). This excludes the extreme apoapsis-raise family. Baseline
   Hohmann-out/Hohmann-back is ~5.5 yr, so the cap leaves ~1.5 yr of slack.

4. **Model the powered flyby as a split hyperbola.** An impulsive tangential burn at
   Jovian periapsis joins two hyperbolas with different eccentricities sharing one
   periapsis; the total bend is the sum of the two asymptote half-angles
   `asin(1/e_in) + asin(1/e_out)`. A powered assist is *not* "unpowered bend plus a
   tangent kick" — higher outgoing speed means less bend per unit periapsis depth,
   and the two are solved together.

5. **Publish as a standalone analysis, leave the catalog rows untouched.** The
   optimizer's `v_b` will differ from the published ~69 km/s. Following the
   resonant-dive and apoapsis-raise precedents, the result is a typed function +
   frozen dataclass (`powered_jovian_flyby_return()`), printed as its own report in
   `main.py` under a planned-paper-section label; the three existing rows keep the
   paper's numbers until the paper itself adopts the new ones.

6. **Report the optimum plus a `v_b` trade curve.** A sweep of minimum total delta-v
   against target `v_b` (45-70 km/s) exposes how flat the optimum is, pre-answering
   the sensitivity question a referee will ask when the paper adopts the numbers.

## Considered options

- **Minimize total delta-v alone — rejected (ill-posed).** No minimizer exists; see
  Context. Recorded in `CONTEXT.md` as the *retrograde-plunge degeneracy*.
- **Constrain the return to the retrograde Hohmann state — rejected as the
  objective, kept as the boundary.** It fixes `v_b` at the published ~69 km/s and
  reduces the problem to two decoupled Oberth burns, but pre-judges the answer the
  end-to-end metric is supposed to find. It survives as the expensive end of the
  trade curve.
- **Real phased Earth intercept ("opportunities") — deferred.** Requiring Earth to
  be at the crossing point makes this an ephemeris/synodic-windows problem. The repo
  convention (and the catalog rows this supports) is geometry-only, "outbound
  injection only"; phasing is scored separately when needed.
- **Tangential-only departure (1 knob) — rejected by choice of generality.** The
  departure burn's cost depends only on the excess *speed*; aiming the excess
  velocity vector is free (rotate the escape hyperbola). Restricting to tangential
  departures would cost nothing at the burn but forecloses arrival-direction
  control; the full free-aim search keeps 4 continuous knobs and is still cheap.

## Consequences

- The repo gains the first genuinely multi-dimensional optimization in `src/`
  (`scipy.optimize.differential_evolution`, seeded for determinism), alongside the
  existing 1-D `brentq` root solves.
- The achieved `v_b` is expected to land strictly between ~50 and ~69 km/s. The
  catalog's three Jovian rows keep the paper's ~69 km/s and therefore *overstate*
  the collision speed of the propellant-optimal leg; the discrepancy is deliberate
  and lives in the new report until the paper catches up.
- A new Jovian periapsis floor constant (`LOW_JUPITER_ALTITUDE = 4000 km`,
  Juno-class perijove) enters `astro_constants.py`. Sensitivity is negligible
  (~40 m/s of delta-v between 200 km and 4000 km floors), which the trade-curve
  report makes checkable.
- The domain language (powered Jovian flyby, retrograde-plunge degeneracy,
  end-to-end mass ratio, free-aim departure, seven-year cap) is recorded in
  `CONTEXT.md`.
