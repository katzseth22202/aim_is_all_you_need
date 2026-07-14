# The V/E pump ladder stands; Cassini-style VVE only opens at ~1 km/s

Status: accepted

Date: 2026-07-14

## Context

After ADR 0003 landed the E-V-E-V-V-E-J minimum-burn chain, two challenge
questions were put to it: would a Cassini-style V-V-E sequence (VVEJGA order)
be feasible at similar low burns with a shorter trip, and is the chain's
15.4 km/s Jupiter arrival *too fast* for the unpowered retrograde bend?

Both were answered with the production physics itself: a sequence-constrained
variant of the ADR 0003 beam search (identical parameters, floors, target
v_b >= 51.13 km/s, and Jovian terminal; only the next-body choice pinned to a
prescribed order), cross-checked against analytic Tisserand ceilings computed
from the same `_conic_radius_crossings` legs.

**Why VVE cannot pump at low burn.** An unpowered flyby preserves the
planet-relative excess *speed* (the Tisserand invariant), so a V->V leg leaves
the Venus-relative excess unchanged: extra same-body flybys only re-aim. A
chain's pumping power is therefore counted by its *body alternations* — each
V<->E hop converts misalignment into heliocentric energy — and E-V...V-E-J of
any length contains exactly two. The Earth-return excess of every
Venus-only-interior sequence is capped by the single-Venus-pump ceiling:

| departure burn | launch excess w_E0 | max w_V | VVE ceiling on w_E |
|---|---|---|---|
| 300 m/s | 2.59 km/s | 3.35 | **5.24** |
| 550 m/s | 3.52 km/s | 7.20 | **13.10** |
| 1000 m/s | 4.80 km/s | 10.48 | 17.84 |

Reaching Jupiter at all needs w_E >= 8.79 km/s (the Hohmann excess); a
retrograde return needs the Jovian arrival excess to exceed Jupiter's own
13.06 km/s orbital speed, ~14–16 km/s in practice for v_b >= 51.13. So at
300 m/s VVE is *provably* impossible (ceiling 5.24), and at 550 m/s it is
marginal on paper (13.10) and not found by the search. The constrained runs:

| sequence | 300 m/s | 550 m/s | 1000 m/s | 4000 m/s |
|---|---|---|---|---|
| E-V-E-J | infeasible | infeasible | infeasible | infeasible |
| E-V-V-E-J (Cassini VVE) | infeasible | infeasible | 2.93 yr | 2.42 yr |
| E-V-E-V-E-J | infeasible | infeasible | infeasible | 5.75 yr |
| E-V-E-V-V-E-J (ADR 0003) | 3.46 yr | 3.66 yr | 3.71 yr | 5.73 yr |

The real Cassini is the consistency check, not a counterexample: it launched
at C3 = 16.6 km^2/s^2 (~0.74 km/s above escape from LEO) *and* burned a
~470 m/s deep-space maneuver between the two Venus flybys — the DSM is exactly
what breaks the V-V Tisserand freeze. This repo's chain model keeps flybys
strictly unpowered, so VVE only closes once the departure burn alone carries
the pump (~1 km/s). Note the ADR 0003 winner already *ends* V-V-E-J — the
Cassini tail is in the chain; its V-V hop is a free re-aim, and the extra
leading V-E rung is what replaces Cassini's launch energy and DSM.

**Why 15.4 km/s is not too fast at Jupiter.** The unpowered turning limit is
delta_max = 2*asin(1/e), e = 1 + r_p w^2 / mu_J. At the ADR 0002 perijove
floor (4000 km altitude, r_p = 75,492 km) and w = 15.37 km/s, delta_max ~ 122
degrees; the chain's terminal uses 95.9, leaving ~26 degrees of margin —
equivalently the same bend is available from a perijove as high as ~2.6 R_J.
Jupiter's mu is so large that even 25 km/s of excess still turns ~93 degrees
at the floor. The dangerous direction is too *slow*, not too fast: below
13.06 km/s no bend of any size yields retrograde (a Hohmann arrival at
5.6 km/s fails absolutely), which is the same hard floor that makes the VVE
ceiling binding. The floor-level pass itself is Juno territory (~60 km/s
perijove speed, single pass).

## Decision

- **Keep the unconstrained ADR 0003 search and its E-V-E-V-V-E-J result.**
  Sequence families are not restricted: the search already contains VVE-type
  subchains and picks the ladder because the physics does.
- **Record VVE's actual envelope**: impossible at <= ~550 m/s (analytic
  ceiling at 300 m/s; witness-not-found at 550), available from ~1 km/s where
  it is the *time*-optimal family (2.93 yr vs 3.46), at an end-to-end cost
  (delivered fraction 0.71 vs 0.85 with the phasing budget charged;
  end-to-end ~4.7 vs ~5.7).
- **Pin the two load-bearing facts in fast tests**: the single-Venus-pump
  ceiling at 300 m/s sits below the Hohmann Jupiter-reach excess, and the
  300 m/s chain's Jovian bend fits inside the periapsis-floor turning limit
  with margin, above the retrograde hard floor.

## Consequences

- The ~3.5 yr phasing-free floor makes the Jupiter-only growth cycle
  attractive on its own terms: at the 300 m/s chain's end-to-end mass ratio of
  ~5.7 per cycle, a working cycle of 3.46 yr doubles mass every ~1.4 yr
  (millionfold in ~27 yr), and even granting a full 4 yr cycle for real-world
  slack still doubles every ~1.6 yr (millionfold in ~32 yr) — competitive with
  the phased solar-dive loop's ~17 yr without any solar dive, and well ahead
  of the apoapsis-raise loop's ~54 yr. Caveat: cycle time is conic flight time
  plus nothing; real phasing waits beyond the charged 300 m/s DSM budget are
  unmodelled, which is why the 4 yr allowance is quoted alongside the floor.
- If a mission ever weighs trip time above delivered mass, the documented buy
  is VVE at ~1 km/s: about half a year faster for an ~18% end-to-end hit.
- The bend-margin result also retires the "arrival too hot?" worry for both
  legs of this program: the powered flyby (13.6 km/s, 67 degrees used, chose a
  7.7 R_J perijove) and the chain (15.4 km/s, 96 of ~122 degrees at the floor)
  both sit well inside the unpowered envelope, from opposite sides.
