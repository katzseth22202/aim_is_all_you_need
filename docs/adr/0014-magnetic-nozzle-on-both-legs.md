# A magnetic nozzle on the growth push needs ~0.6 recovery just to draw level with the plate

Status: accepted; verdict superseded by ADR 0015, which ADR 0016 then re-grounded
on a like-for-like comparison against the plate's measured f = 0.818 (the
arithmetic of all three stands; only 0015's reasoning was replaced)

Date: 2026-08-20

Builds on ADR 0012 (the impact-angle impulse law and its two endpoints) and ADR
0013 (the two-wave same-cycle ledger and the real-orbit chain it is flown on).
Amends nothing: the plate remains the baseline, and ADR 0013's published table
still reproduces bit-for-bit wherever the new ground-launch floor is slack.

## Context

Every result in the repo prices the growth push with a **pusher plate** and only
the head-on departure burn with a **magnetic nozzle**. That asymmetry was never
a decision — the plate is what the paper's `eq:PuffSat_ratio` describes, and the
nozzle arrived later for the leg the plate cannot serve. Nobody had asked what a
nozzle would do on the overtaking leg too.

The question is sharp because the two devices sit at opposite ends of ADR 0012's
single impulse law, `beta(theta, k) = sqrt(1 + k - sin^2 theta) + cos theta`. On
the overtaking push the arriving PuffSat's momentum *adds* to the exhaust rather
than subtracting from it, so `beta = 1 + sqrt(1+k)` against the head-on burn's
`sqrt(1+k) - 1`. Read as engines: the plate has infinite Isp and a hard cap of
`2 f w` impulse per impactor kilogram, while the nozzle trades finite Isp
(~975 s equivalent at `k1 = 15`) for impulse that grows as `sqrt(k)`. Since a
PuffSat kilogram costs a whole Jupiter round trip and a slug kilogram costs only
a ground launch, spending the cheap currency to save the dear one looks free.

It is not free, and two limits say why.

## Decision

**Price the overtaking leg as a nozzle on ADR 0013's own ledger, subject to a
plume-ignition window and a ground-launch floor, and keep the plate.**

The ledger change is one substitution. ADR 0013's

    g [ (1+sigma)/(r M d1) + sigma/k ] = 1

keeps its shape; only `M`, the parked mass per arriving impactor kilogram, moves
from the plate's `2 f / Lambda` to the overtaking nozzle's `k1 / sigma1`, with

    sigma1 = exp( Lambda1 k1 / (e1 (1 + sqrt(1+k1))) ) - 1

and `Lambda1 = ln(v_b / (v_b - v_rf))`. `same_cycle_nozzle()` takes both forms,
so nothing is re-derived (`_sigma_overtaking()` in `src/nozzle_analysis.py`).

### The algebra, in one place

**The k -> 0 identity, and why it is only formal.** As `k1 -> 0`,
`k1/sigma1 -> 2 e1 / Lambda1`, which is `payload_mass_ratio()` with `e1` in the
role of `f`. The plate is therefore the vanishing-slug limit of the nozzle
*arithmetically*. It is not one physically: a magnetic nozzle steers a
conductor, and a vanishing slug dissipates nothing in the merge, so there is no
plasma to steer. See the window below.

**Plume ignition.** The merge is inelastic, so of the impactor's vehicle-frame
kinetic energy `w^2/2` per impactor kilogram only the fraction `k/(1+k)` is
thermalised; the rest stays as bulk drift. Spread over `1+k` kilograms of blob:

    eps_th(w, k) = w^2 k / (2 (1+k)^2)

This peaks at `k = 1` and falls on **both** sides, so requiring a floor
temperature gives a closed **window** in `k`, not a ceiling — the roots of
`eps_th = eps_req`. `k -> 0` is outside it.

**The ignition bill.** The slug is water with a 1% potassium seed. The seed
supplies the free electrons, so the water is never charged for ionisation, and
the bill at 15,000 K comes to 84.4 MJ/kg:

| term | MJ/kg |
| --- | ---: |
| vaporisation + sensible heat | 2.97 |
| **atomisation, H2O -> 2H + O** | **50.39** |
| translational, 3 atoms per molecule | 30.94 |
| seed ionisation + electron thermal | 0.11 |

Breaking the water up is 60% of it. Ionisation, the thing one expects to
dominate, is 0.13%. `src/plume_thermal.py`.

**The ground-launch ledger.** Both legs' slug is lofted from Earth, so they
compete for the same launched kilograms. Charging 2/3 of liftoff as propellant
(exactly a 4.09 km/s lob at 380 s, which covers a 200 km zero-velocity toss with
losses) and a quarter of the remainder as launcher dry mass (structural
coefficient 11%) leaves 1/4 of liftoff reaching the intercept point. Requiring
1/15 of liftoff to return is then

    0.25 * r * d1 / [(1 + sigma1)(1 + sigma2)] >= 1/15

applied per flown cycle, since `r` and `d1` vary across the chain.

## Consequences

- **The overtaking nozzle needs roughly its own `f` back before it wins, and
  more.** Converting each `(e1, e2)` cell to the plate elasticity that would
  match it at the same `e2` — the incumbent's own axis, where `f > 1` is
  physically impossible:

  | `e1` \ `e2` | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0.40 | 0.399 | 0.434 | 0.450 | 0.459 | 0.464 | 0.467 | 0.469 |
  | 0.50 | 0.497 | 0.557 | 0.593 | 0.613 | 0.626 | 0.634 | 0.639 |
  | 0.60 | 0.594 | 0.682 | 0.742 | 0.778 | 0.801 | 0.817 | 0.828 |
  | 0.70 | 0.690 | 0.806 | 0.897 | 0.952 | 0.989 | 1.015 | 1.033 |
  | 0.80 | 0.785 | 0.931 | 1.056 | 1.135 | 1.188 | 1.226 | 1.254 |
  | 0.90 | 0.879 | 1.055 | 1.214 | 1.324 | 1.397 | 1.450 | 1.486 |

  Against the paper's `f = 0.8` plate the crossover is `e1` between 0.5 and 0.6
  at a good head-on leg, rising to `e1 ~ 0.7` when `e2` is mediocre. Against a
  *perfectly elastic* plate it takes `e1 >= 0.8` at `e2 >= 0.5`. ADR 0013
  records `e ~ 0.3` as the architecture's survival threshold and a bare
  paraboloid at 0.31, so the overtaking nozzle must be roughly twice as good as
  the leg that already exists merely to replace a device that carries no
  propellant. **That is the finding, and it is why the plate stays.**
  ADR 0015 revisits this against a plate at matched recovery rather than at
  `f = 0.8` and reaches the opposite verdict from the same numbers. The
  equivalent-plate table above is unchanged and is what both ADRs turn on.

- **The ground-launch floor binds almost everywhere; the plume floor almost
  nowhere.** Across the 8x8 grid the returned fraction sits pinned at exactly
  1/15 in 56 of 64 cells, while the ignition ceiling is active in one. The
  intuition that plasma temperature would be the limiting physics is wrong at
  these closing speeds: 15,000 K is a low bar for a 49 km/s impact, and what
  actually stops `k1` is that the water has to be launched.

- **The two legs compete, and at low `e2` the competition is total.** When the
  head-on nozzle is poor its `sigma2` consumes the whole launch budget, `k1`
  collapses onto the ignition window's lower root (~0.1), and the equivalent
  plate elasticity converges to `e1` itself — the device degenerates toward the
  plate but cannot reach it, because below that root the plume will not light.

- **The fleet-wide slug ratio is set by the three-synodic cycles.** One nozzle
  loading flies the whole chain (`price_chain`'s convention), so the admissible
  window is the intersection over cycles. The 3S returns close slowest
  (growth-wave `v_b` 56.5-57.4 against the 2S cycles' 61.8-65.1) and therefore
  run coldest, capping `k1` at 10.21 where a 2S cycle alone would allow 15.32.
  Letting each
  cycle pick its own `k` recovers only 0.6-3.2% of rate, which does not justify
  a per-cycle loading.

- **ADR 0013 still reproduces, but only where the new floor is slack.** At
  `e = 0.6, f = 0.8` the plate returns 1/10.8 of liftoff and the floor does not
  bind, so 0.3989 e-foldings/yr stands unchanged. At `e2 <= 0.3` it bites the
  incumbent too, and hard enough to change its sign: at `f = 0.8, e2 = 0.25` the
  plate goes from +0.0212 e-foldings/yr to **-0.0994**, growing to shrinking.
  The ground-launch charge is new information about the plate as well.

### Considered and rejected

- **Tapering `k1` through the burn to ride the ignition ceiling.** The window's
  ceiling moves with `w`, which falls 59.8 -> 48.8 km/s across the push at
  ADR 0009's reference point, so a schedule `k(w)` could start at 18.9 and end
  at 11.9. Integrating the variable-
  `k` push gives +1% to +6% parked mass, bought with a 15% larger mass
  multiplier — and under a binding launch floor that trade is negative. Not
  worth the loss of a closed form.

- **A lighter or heavier slug.** The impulse law is momentum-out-of-energy and
  carries no molar-mass dependence, so the model would always prefer whatever
  is cheapest to ionise per kilogram (Xe at ~12 MJ/kg, ceiling `k1 ~ 99`) and
  reject the species a real magnetic nozzle wants (Li at ~150 MJ/kg, ceiling
  `k1 ~ 5.8`). Water was fixed by choice rather than by optimisation, precisely
  because the model cannot make that choice honestly.

- **Charging the launcher's dry mass against liftoff rather than against the
  non-propellant remainder.** At 25% of liftoff the payload reaching intercept
  falls to 1/12, the plate incumbent itself returns only 1/22.4, and the 1/15
  bar is unreachable by any architecture including the one already in the paper.
  That reading makes the constraint a wall rather than a test.

### Assumptions worth revisiting

- Both impactor and slug are given water's caloric properties. Exact nowhere;
  close wherever the answer matters, since the blob is over 75% slug by mass for
  `k >= 3`. Below `k ~ 1` the blob is mostly impactor and the window's lower
  root should be read as indicative rather than pinned.

- `e1` and `e2` are swept independently with no imposed relation. Two effects
  point opposite ways and neither is derivable here: the wrong-way bulk drift an
  overtaking nozzle must reverse is only `1/(1+k1)` of the blob energy (8.9% at
  the fleet ceiling `k1 = 10.21`, arguing `e1 ~ e2`), but the overtaking geometry
  puts impactor
  entry and exhaust exit at the *same* end of the vehicle, so each impactor
  flies up the previous shot's plume (arguing `e1 < e2`). The equivalent-plate
  table is published instead of a penalty factor so a reader can place their own
  belief about `e1`.

- 15,000 K is an operational proxy for "conductive enough to magnetise", not a
  derived threshold. `plume_ignition_energy()` takes the temperature as an
  argument for exactly this reason.

## Reproducing

`make two-leg` (`src/two_leg_nozzle_sweep.py`). Search box, recorded per the
ADR 0007 lesson: 32 log-spaced points per axis clipped to the fleet-wide
ignition window, 4 rounds re-spanning both grids between the neighbours of the
current best point; the equivalent-plate elasticity is a `brentq` over `f` in
[0.02, 4.0] at `xtol = 1e-3`. Flown on ADR 0013's 11-cycle, 28.3930 yr adaptive
2S/3S chain at a 10-day split.
