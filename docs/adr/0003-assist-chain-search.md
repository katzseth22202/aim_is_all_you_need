# The assist chain solves for minimum departure burn with a feasibility-witness beam search

Status: accepted

Date: 2026-07-14

## Context

ADR 0002 established the powered Jovian flyby: ~4.45 km/s of methalox at 200 km
buys the ~13.6 km/s hyperbolic excess that an unpowered Jovian bend turns into the
end-to-end-optimal retrograde return (v_b ~51 km/s). The follow-up question: can
unpowered Venus/Earth/Mars gravity assists replace that departure burn, within a
10-year trip and a small correction budget beyond Earth?

Three facts shape the design (all verified numerically in `scenario.py` and its
tests):

1. **The Venus-reach floor.** Below ~279.4 m/s above escape at 200 km
   (`venus_reach_departure_floor()`), even the best (anti-tangential) free aim
   leaves the transfer perihelion above Venus's orbit: no assist body is
   reachable at all. The originally hoped-for 250 m/s is below the floor.
2. **The Tisserand lock.** An unpowered flyby rotates the planet-relative excess
   velocity but never changes its magnitude. At *exactly* the floor, the Venus
   arrival is tangent — the excess velocity is aligned with Venus's motion — so
   rotations only re-aim and the reachable Earth-relative excess is frozen at
   its launch value forever. The floor is a dead end, not a minimum.
3. **Square-root escape.** The arrival misalignment grows as the square root of
   the margin above the floor (~40 degrees at +20 m/s), so feasibility switches
   on within a few m/s of the floor: at ~290 m/s an E-V pump ladder reaches
   Jupiter at ~15 km/s of excess in ~2 years of chain time.

Two formulation traps were avoided. "Minimize total delta-v to any retrograde
crossing" is ill-posed exactly as in ADR 0002; here the search instead holds the
return to the *powered flyby optimum's* collision speed (v_b >= ~51.1 km/s), so
both legs answer the same question and their end-to-end mass ratios compare
directly. And the model is phasing-free (coplanar circular planets, each body
wherever the trajectory needs it), which would make the mass accounting
misleadingly rosy — so a fixed 300 m/s deep-space-maneuver reserve
(`ASSIST_CHAIN_PHASING_BUDGET`) is charged as spent methalox alongside the
departure burn.

## Decision

- **Objective: minimum departure burn**, scanned over an explicit probe grid
  (`ASSIST_CHAIN_BURN_CANDIDATES`, bracketing the analytic floor), each probe
  required to reach v_b >= the powered-flyby optimum's collision speed within
  `ASSIST_CHAIN_MAX_TRIP_TIME` (10 yr) and `ASSIST_CHAIN_MAX_FLYBYS` (5)
  inner-planet assists. All flybys are strictly unpowered (300 km floors at
  V/E/M, the ADR 0002 4000 km floor at Jupiter).
- **Search: deterministic beam search over flyby chains.** States are
  planet-relative excess velocities at a body's orbit radius; expansions rotate
  the excess within the periapsis-floor bend limit and coast conic legs between
  orbit radii; every state is also terminated through the unpowered Jovian bend
  scored by the ADR 0002 return leg (`_flyby_return_leg`, same v_b convention).
  No randomness anywhere.
- **Pruning is time-bucketed**: per (body, 0.2 yr time-of-flight bucket) the top
  states by excess speed are kept. Global top-by-speed pruning was tried first
  and made feasibility *non-monotone in the burn* (300 m/s feasible, 350
  "infeasible"): junk high-speed states crowded out closable slower paths.
- **Results are feasibility witnesses.** The search records every decision and
  the public result is built by *replaying* them leg by leg
  (`_chain_to_result`), so bookkeeping bugs surface as replay failures. A
  `None` means "not found at these beam settings", never "infeasible"; the
  reported minimum burn is an upper bound on the true minimum.
- **Beam settings are calibrated, not maximal**: 181 departure aims, 31
  rotations per flyby, 121 Jovian bends, 80 states per bucket, 1500 per body.
  This closes at 290 m/s in ~10 s per probe; a finer beam closes at 285 m/s in
  ~40 s. The gap is documented rather than paid for on every test run.

## Consequences

- Headline: **~0.29–0.30 km/s of departure burn replaces the powered flyby's
  4.45 km/s.** At 300 m/s the chain is E-V-E-V-V-E-J, total ~3.46 yr, v_b
  ~51.5 km/s; even with the 300 m/s phasing budget charged, the end-to-end mass
  ratio is ~5.7 versus the powered flyby's ~2.0. The user's original 250 m/s
  (+300 m/s corrections) split inverts: ~300 m/s must go to the departure burn,
  leaving the rest for phasing.
- The phasing reserve is an *estimate charged as propellant*, not a phasing
  solution: real ephemerides, launch windows, and arrival timing remain
  unmodelled, like every heliocentric analysis in this repo.
- Minimum-burn numbers quote the calibrated beam. Anyone tightening the probe
  grid below 290 m/s must either accept the beam's upper-bound semantics or
  recalibrate the beam constants (and the pinned tests) together.
- The catalog rows are untouched, as in ADR 0002: this is a standalone analysis
  (`assist_chain_return`), reported next to the powered flyby in `main.py`.
