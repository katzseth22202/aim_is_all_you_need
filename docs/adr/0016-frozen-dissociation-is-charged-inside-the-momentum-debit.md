# The frozen-dissociation toll is charged inside the momentum debit, not through recovery

Status: accepted

Date: 2026-08-26

## Context

The growth tables assume the 50.9 MJ/kg spent pulling water apart comes back as
the plume cools and expands. `sec:watering_it_down` argues for it in one line --
"They do, and quickly" -- and names the condition it rests on: recombination
freezes below about 0.01 kg/m^3, "and we have not computed the density a real
pulse produces."

`puffsat_impact_simulation` has now computed it (`make analysis-fireball`,
`make analysis-toll`). **The loan defaults.** The plume crosses `Da = 1` at the
*first station past the nozzle lip*, at 1.1e-2 to 2.4e-2 kg/m^3, with 90-100%
of the store still held.

The paper's rate check is not sloppy, it is evaluated at the wrong station. At
1 kg/m^3 three-body recombination really is ~0.01 us against a hundreds-of-us
expansion -- but at 1 kg/m^3 the plume is fully atomised and has nothing to give
back yet. It only has something to return once it has cooled, and by then it
sits at ~0.02 kg/m^3 and past the lip, where the local expansion clock steps
down 8x in one step.

**This is not a new term.** `sec:jet_efficiency` already defines `eta_jet^2` to
include "frozen ionization or dissociation energy". What changes is that one of
its five contributions is now *computed* rather than swept, and it puts a
ceiling on a parameter the growth tables run to 0.90. Charging it as a separate
energy debit would double-count.

## Decision

**Split the one efficiency knob into the two places the physics actually acts,
and charge the chemistry at the earlier one.**

`nozzle_analysis._sigma` and `_sigma_overtaking` multiplied `recovery` into
`beta` as a whole, which scales the merged blob's bulk-drift term as well as
the jet:

    beta_headon   = recovery * (sqrt(1+k) - 1) / k
    beta_overtake = recovery * (1 + sqrt(1+k)) / k

The `-1` and `+1` are pure momentum conservation and no efficiency of any kind
can touch them (Seth, 2026-08-25). They now read

    beta_headon   = (eta_jet sqrt(1+k) - 1) / k
    beta_overtake = (1 + eta_jet sqrt(1+k)) / k

with `eta_jet = eta_chem * eta_geom`, and `recovery` retained as a separate,
independent factor on the *net* impulse after the debit -- which is what the
paper's `e_2` was defined to absorb, and which correctly has no floor.

**The outer form contradicted the paper's own `eq:ve_general`**, which states
that effective exhaust velocity is what remains *after* debiting the incoming
momentum. That equation also names the floor this restores: forward thrust
vanishes at `eta_jet = sqrt(m_rp)`, here `1/sqrt(1+k)`, **0.324 at `k = 8.5`**.

`eta_chem` is the closed form

    eta_chem = sqrt(1 - 2 E_a phi (1+k) / w^2)      E_a = 50.90 MJ/kg

in `plume_thermal.chemistry_efficiency`. At `phi = 1` it is a **floor** that can
never overstate the solved surface, because `phi <= 1` and `eta_chem` falls with
`phi`; across the impact sim's 74 conducting nodes it understates by at most
0.057. Fed the solved `bond_fraction` it reproduces that surface to four
figures (0.7540 against 0.753759 at the cold anchor; 0.9100 against 0.909910 at
75 km/s), which is an independent cross-repository check, since the surface is
solved on `eos_water` through the expansion and this is algebra.

**Defaults preserve every published number.** `jet_efficiency` defaults to 1.0
and `geometric_efficiency` to None, so ADR 0013/0014/0015 reproduce bit-identically;
the tolled arithmetic is opt-in.

## Consequences

**The chain is priced against a 28.39-year, 11-cycle flown chain at a 10-day
split.** Two changes are in play and they must not be conflated -- moving `e`
inside the debit is a correction independent of the chemistry:

| `e` | A: `e` outside (published) | B: `e` inside, no toll | C: `e` inside + toll |
| ---: | ---: | ---: | ---: |
| 0.50 | 1.51e4 | 48.4 | 0.0102 |
| 0.60 | 8.31e4 | 3 897 | 60.9 |
| 0.70 | 2.94e5 | 5.74e4 | 4 674 |
| 0.80 | 7.83e5 | 3.46e5 | 6.29e4 |
| 0.90 | 1.71e6 | 1.24e6 | 3.55e5 |

- **The toll differentially penalises the two-leg nozzle, which is the option
  `sec:two_leg_nozzle` concludes is the better one.** A plate owes no chemistry,
  so `tab:space_mortgage_growth` is reached only through the head-on leg at
  65-74 km/s where `eta_chem` is a mild 0.880-0.908. `tab:two_leg_growth` is
  reached through both, and the growth push's cold end runs 45.58-54.18 km/s
  where `eta_chem` is **0.731-0.819**. Matched at `e1 = e2 = e`, the nozzle
  falls 36.7x at `e = 0.80` against the plate chain's 12.4x, and 35.2x against
  4.8x at 0.90.
- **The nozzle still wins the matched diagonal, by much less.** 12.3x at
  `e = 0.80` against ADR 0015's published 36.2x. Below `e = 0.7` the two-leg
  chain becomes inadmissible outright -- it cannot pay for its own launch.
- **The optimal slug ratio falls**, because `eta_chem` falls with `k`: more slug
  is more water to pull apart per unit of collision energy. `k2` goes 9.25 ->
  7.77 at `e = 0.8`; on the two-leg chain `k1` goes 8.19 -> 4.81.
- **Which plate the growth push flies is now settled** (Seth, 2026-08-26): the
  **heavy block**, not the paper's 5 m plate. So `STD_FUDGE_FACTOR = 0.8` is
  defensible unchanged -- the impact sim measures `f = 0.817-0.820` on the 15 m
  heavy plate across 45-63 km/s -- and no plate-side `f` revision is owed.
- **ADR 0015 is exposed twice over and is not revised here.** Its stated premise
  ("nothing rebounds elastically at 46 km/s") was measured false before it was
  written, and `f` and `e_1` cannot be swept as a matched pair now that they
  carry different physical ceilings: `f` is a plate restitution carrying the
  impact sim's ADR-0026 frozen bracket, `e_1` is a nozzle recovery carrying the
  `eta_chem` ceiling above. Matching them numerically compares the plate at a
  value it can reach against a nozzle at one it may not.
- **The search box was wrong and the answer was not.** `two_wave_growth`'s bare
  `_K_SEARCH_MAX = 80` ran six times past the ignition window's ceiling (26.9
  across the flown fleet at the 10 000 K gate, 36.88 at the 75 km/s anchor). It
  is now intersected with that window in `headon_slug_ratio_bounds`. The
  optimum never lived out there -- it sits at 7.3-9.5 either way, and the
  untolled chain reproduces to every digit -- so this is the ADR 0007 hygiene
  lesson rather than a moved number.
- **The gate is 10 000 K and is deliberately *looser* than the 15 000 K design
  floor**, which reads backwards until named. The floor is what the nozzle is
  designed around; the gate is where the plume stops dissociating at all. What
  fails below it is dissociation, not conduction -- the potassium seed holds
  `sigma` at 1480 S/m there and 403 S/m at 4500 K.

### What this does not cover

- **`eta_geom` is entirely unmeasured.** `eta_chem` bounds one of the five
  contributions to `eta_jet`; nothing in either repository bounds divergence,
  exhaust-speed spread, radiative escape, or mass the field fails to grip.
- **It is not a stability gate.** The impact sim's ADR-0038 flips the
  electrothermal verdict between `T_0` = 19 710 K and 15 170 K, above this gate
  *and* above the paper's own flown cold anchor. That is its Q-O, undecided.
- **`eta_chem` is held at the coldest instant of each burn** rather than varied
  through it. That understates `beta` and so overstates the slug spent, which
  is the conservative direction, and it is the anchor
  `fleet_ignition_windows` already uses.
- **The bag is assumed resized with `k`**, holding plume density at the flown
  0.323 kg/m^3, matching the paper's treatment of bag radius as a live design
  variable (`tab:bag_sizing`).

## Reproducing

`price_chain(cycles, 1.0, 0.8, geometric_efficiency=e)` for the plate-on-growth
chain and `price_chain_two_leg(cycles, 1.0, 1.0, geometric_efficiency=e)` for
the two-leg chain, against the same calls with `geometric_efficiency=None` and
`recovery=e` for the published columns. Bounds: `headon_slug_ratio_bounds`
intersects the recorded box `[0.2, 80.0]` with the fleet ignition window at
`NOZZLE_GATE_TEMPERATURE`. The input surface is `data/results/eta_chem.csv` in
`puffsat_impact_simulation` (81 nodes, 74 conducting), regenerated there with
`make analysis-toll`; `chemistry_efficiency`'s `phi = 1` default needs no part
of it.

---

## Glossary

Three terms this ADR leans on that appear nowhere in the paper.

**`eta_jet`** -- the paper's own (`sec:jet_efficiency`): the fraction of the
*ideal one-directional exhaust momentum* a real nozzle delivers, where "ideal"
means every joule of collision energy placed on one axis pointing backwards.
The paper says `eta_jet^2` bundles five losses: plume divergence (the exhaust
sprays in a cone), exhaust-speed spread, radiative escape (energy leaving as
light, which a magnetic field cannot steer and which pushes nothing), frozen
ionisation or dissociation energy, and mass the field fails to grip.

**`eta_chem`** -- the fourth of those five, computed by this ADR.

**`eta_geom`** -- the other four, lumped, so `eta_jet = eta_chem * eta_geom`.
**The name is loose** ("mass the field fails to grip" is not geometry) and it is
**entirely unmeasured** -- nothing in either repository bounds it, which is why
it stays a swept axis rather than becoming a number.

**Rigid front** -- the snowplow assumption that the sweeping front keeps the
radius it arrived with for the whole column. The slug bag is a puffed cloud
(0.323 kg/m^3, about a quarter of sea-level air density), the projectile plows
through it, and the mass it plows up *is* `k`. So the question the arrival
radius asks is simply how wide the plow blade is, and "rigid" is the assumption
that it never widens. It is the pessimistic bound, and every number in the
addendum's sensitivity table is computed under it.

**`c_exp`** -- how fast the blade widens (km/s), radially. Material at the nose
is shock-heated to tens of thousands of kelvin and expands sideways; `c_exp` is
essentially a sound speed in the shocked gas, hence the 3-8 km/s range swept.
Since the front advances at `w` and widens at `c_exp`, the cone half-angle is
`arctan(c_exp/w)` -- **3.8 to 10.0 degrees over 45.58 km/s**, which closes the
0.6 m gap from `r/R` = 0.8 in 3.4 to 9.1 m of a 23.8 m column. That is why
spreading swamps the arrival radius, and why `c_exp` rather than `r/R` is the
quantity that decides whether the arrival radius is load-bearing at all.
**`c_exp` cannot come from the snowplow model**, which is one-dimensional along
the axis while this is radial by definition. It needs a 2D hydro solve, in
`puffsat_impact_simulation`, and has not been run.

## Addendum, 2026-08-26: two design assumptions fixed, and the ceiling they give

Seth settled the two inputs this ADR had left open.

### 1. Recombination on escape is assumed unsound and zero (`phi = 1`)

The `phi = 1` default was introduced above as a *conservative floor* on a solved
surface that reports `phi` = 0.86-1.00. It is now the **design assumption**: the
store is taken to stay locked, and no credit is taken for the water that
re-forms. This is deliberately more pessimistic than the solve at the cold
anchor, where the plume is cool enough at the lip that `phi` falls to 0.927.

Nothing in `src/` changes -- `chemistry_efficiency` already defaults this way,
so every tolled number already published in this ADR was computed under it.
What changes is its status: the numbers are the **design ceiling**, not a bound.

**Maximum `eta_jet` under that assumption**, i.e. what the architecture can
reach with perfect geometry (`eta_geom` = 1, no divergence, no spread, no
radiative escape, nothing ungripped):

| `k` | overtake leg (45.58 km/s) | head-on leg (65.44 km/s) |
| ---: | ---: | ---: |
| 2 | 0.9236 | 0.9637 |
| 4 | 0.8689 | 0.9387 |
| **5.563** | **0.8237** | **0.9187** |
| 6.750 | 0.7998 | 0.9032 |
| 8.5 | 0.7311 | 0.8798 |
| 12 | 0.6025 | 0.8312 |

The **absolute ceiling on the chain** (plate on the growth push, `eta_geom` = 1)
is `k*` = **6.750**, growth **1.218e6** over the 28.39-year chain, doubling
**1.405 yr**. Every real `eta_geom` below 1 comes off that.

### 2. The projectile arrives spanning 0.8 of the bore

`k` was never a free design variable; it is an output of the snowplow geometry,
and the relation is exact for a rigid front:

    k_eff / k_full = (r_arrival / R_bore)^2      k_full = 8.692

Now in `src/nozzle_geometry.py` (ledger item 11; items 12 and 13 are still
owed). **At 0.8 of the bore a rigid front delivers `k` = 5.563**, and 36% of the
bag is launched but never swept.

**The sensitivity, which is the reason this parameter deserves naming in the
paper.** Chain growth at the `k` each arrival radius delivers, rigid front:

| `r/R` | `k` | growth, `eta_geom` = 1 | growth, `eta_geom` = 0.8 |
| ---: | ---: | ---: | ---: |
| 0.60 | 3.129 | 5.562e5 (0.46) | 8 257 (0.13) |
| 0.70 | 4.259 | 9.153e5 (0.75) | 2.659e4 (0.42) |
| **0.80** | **5.563** | **1.156e6 (0.95)** | **4.827e4 (0.77)** |
| 0.90 | 7.041 | **1.214e6 (1.00)** | 6.143e4 (0.98) |
| 0.95 | 7.845 | 1.178e6 (0.97) | **6.288e4 (1.00)** |
| 1.00 | 8.692 | 1.108e6 (0.91) | 6.100e4 (0.97) |

### Why a wider sweep is not simply better

Stated plainly, because the shape of this trade is not obvious and the paper
never says it.

Picture pushing a boat along by throwing rocks off the back. Each throw has a
fixed amount of arm energy -- here, the projectile's kinetic energy, which the
closing speed fixes and the slug cannot change. Throwing a *heavier* rock with
that fixed energy does move the boat more, because the push is mass times speed
and the speed only falls as the square root of the mass: doubling the rock costs
you a factor 1.41 in speed and gains you a factor 2 in mass. That is the
`sqrt(1+k)` in the impulse law, and it is why carrying slug helps at all.

Two things stop it.

**You had to haul the rocks.** Every kilogram of slug was launched from Earth
and is charged against the same mass budget as the payload. The push grows like
`sqrt(k)`; the bill grows like `k`. Past some load the boat is mostly rocks.

**The rocks are frozen together and must be melted apart before they can be
thrown, out of the same arm energy.** That is the dissociation toll, and it is a
flat 50.9 MJ per kilogram of slug, whatever the closing speed. The energy
available per kilogram of blob, meanwhile, is `w^2/(2(1+k))` -- it *falls* as
slug is added. So the tax per kilogram is constant while the income per kilogram
shrinks, and the fraction eaten grows without limit. **At 45.58 km/s, `k` = 19.4
is where the toll consumes the entire one-axis budget and the plume produces no
exhaust at all.** The optimum sits well below that, because the hauling bill
binds first.

So there are three curves -- push rising as `sqrt(k)`, launch cost rising as
`k`, and an energy tax whose bite rises as `k` -- and one peak where they
balance. Here that peak is `k` = 6.75-7.77, which is `r/R` = 0.88-0.95. Sweeping
more of the bore than that is buying slug past the point where it pays.

**Three things fall out of this that were not obvious.**

- **The optimum is interior, at `r/R` = 0.88-0.95.** Sweeping the *whole* bore
  is worse than sweeping 0.9 of it, by 3% at `eta_geom` = 0.8 and 9% at 1.0,
  because `k_full` = 8.69 overshoots the tolled optimum of 6.75-7.77. So a
  design need not chase the last few percent of the bore -- that is improbable
  precision *and* slightly counterproductive.
- **The toll made this requirement easier, not harder.** Before the toll the
  optimum wanted `k` ~ 9.25, which needs `r/R` = 0.97. Charging the chemistry
  drops the optimum to 6.75-7.77, which needs **0.88-0.95**. The two findings
  push in opposite directions and the geometry one is the beneficiary.
- **At 0.8 the geometry binds the optimiser**, costing 5% of the ceiling at
  `eta_geom` = 1 and 23% at 0.8. At 0.9 it very nearly does not. **That is the
  whole case for stating the arrival radius as a design parameter**: the
  difference between 0.8 and 0.9 is a quarter of the delivered growth at
  realistic efficiency, and it is currently unwritten anywhere.

**The caveat that decides how much any of this matters.** All of the above is
the **rigid-front** bound. Letting the front grow as `dr/dx = c_exp/v` with
momentum shared inelastically, it has 23.8 m of column to close a 1.2 m gap:

| `r/R` | rigid | `c_exp` = 3 km/s | 5 | 8 |
| ---: | ---: | ---: | ---: | ---: |
| 0.60 | 3.129 | 7.259 | 7.721 | 8.026 |
| 0.80 | 5.563 | 8.277 | 8.414 | 8.503 |
| 0.90 | 7.041 | 8.572 | 8.613 | 8.640 |

With any spreading at all the arrival radius stops mattering and `k` returns to
~8.3-8.6 regardless. **So the honest statement is not "the front must arrive
wide" but "the arrival radius matters exactly insofar as the front is rigid"** --
and whether a real front spreads at 3, 5 or 8 km/s is a 2D hydro question
neither repository solves (`puffsat_impact_simulation`, Q-Q). Note also that
spreading pushes `k` back toward 8.5, which is *past* the tolled optimum, so a
strongly spreading front wants a smaller bag rather than a wider one.

## Reproducing the addendum

`nozzle_geometry.swept_slug_ratio` and `arrival_fraction_for` for the geometry;
`price_chain(cycles, 1.0, 0.8, slug_ratio=k, geometric_efficiency=g)` for each
row, with `k = full_bore_slug_ratio() * (r/R)**2`.

---

## Addendum 2, 2026-08-26: `c_exp` is computable, and it retires the rigid front

**Addendum 1's sensitivity table is the rigid-front bound, and that bound is now
known to be the wrong branch.** It is kept above because it is a genuine bound
and because the quadratic relation it rests on is exact -- but it should not be
read as the design case.

### The number nobody had computed

`c_exp` was swept at 3-8 km/s because "whether a real front spreads at 3, 5 or 8
km/s is a 2D hydro question" neither repository solves. That framing was wrong
about what it takes. The **magnitude** does not need a 2D solve, only the
detailed shape does: the freshly shocked layer at the front takes `v^2/2` of
specific energy -- the snowplow's own inelastic-accretion assumption -- and
`eos_water` inverts that to a temperature through the full dissociation and
`O+ .. O8+` ladder. Its sound speed is `c_exp`.

| front speed | shocked `T` | `c_s` | `dr/dx` | half-angle |
| ---: | ---: | ---: | ---: | ---: |
| 45.58 km/s | 94 630 K | **21.1 km/s** | 0.464 | **24.9 deg** |
| 25.00 | 34 620 | 10.5 | 0.420 | 22.8 |
| 12.00 | 10 226 | 4.7 | 0.389 | 21.3 |
| 7.00 | 4 398 | 1.9 | 0.274 | 15.3 |

**The sound speed of the shocked plume is roughly half the closing speed**, so
the front widens at 15-25 degrees for the whole transit. Swept values of 3-8
km/s were low by a factor of 3-7 at the entry, which is exactly where widening
matters most.

**Two checks that this is not an artefact.** Venting the shocked layer sideways
until its pressure balances the cold cloud's ram pressure, `sqrt(P/rho_amb)`,
gives **1.6-1.9x faster** than `c_s` at every station -- so taking `c_s` is the
conservative choice, not a flattering one. And the shock compression ratio, the
one weakly-known input, moves `c_s` by **9% across 2x-16x**.

### The front is self-widening, so the arrival radius is very nearly irrelevant

| `r/R` | `k`, rigid front | `k`, self-widening | fills bore at |
| ---: | ---: | ---: | ---: |
| 0.062 (compact ice rod) | 0.034 | **7.24** | 6.2 m |
| 0.30 | 0.782 | 7.79 | 4.7 m |
| 0.60 | 3.129 | 8.36 | 2.7 m |
| **0.80** | **5.563** | **8.60** | **1.3 m** |
| 0.90 | 7.041 | 8.67 | 0.7 m |
| 1.00 | 8.692 | 8.692 | -- |

The column is 23.8 m. From every plausible arrival radius the front reaches the
wall in the first few metres, and `k` lands in **7.2-8.7** instead of the rigid
model's 0.03-8.7. **Q-Q's "the projectile must arrive spanning 74-97% of the
bore" is retired**, and so is Addendum 1's finding that 0.8 costs 23% of the
chain: at 0.8 the front sweeps **99.0%** of the bag.

Seth's 0.8 assumption stands and is now comfortable rather than marginal.

### The consequence points at the bag, not the projectile

A self-widening front at 0.8 delivers `k` = **8.603**, and ADR 0016 puts the
tolled chain optimum at 6.75-7.77. **The flown bag therefore carries more slug
than the chain wants** -- the "spreading dominates" branch this ADR flagged as
wanting a smaller bag rather than a wider front.

| `eta_geom` | optimum `k` | growth at optimum | growth as flown | gain | slug saved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 6.750 | 1.218e6 | 1.116e6 | 9.1% | 21.5% |
| 0.90 | 7.187 | 3.547e5 | 3.341e5 | 6.2% | 16.5% |
| 0.80 | 7.773 | 6.289e4 | 6.133e4 | 2.5% | 9.6% |
| 0.70 | 8.598 | 4 674 | 4 674 | 0.0% | 0.1% |

**The flown 213 kg bag is exactly optimal at `eta_geom` = 0.70**, and 10-22%
oversized above that. Since `eta_geom` is unmeasured, the honest reading is that
the current bag is defensible rather than wrong -- but every improvement in
nozzle quality argues for shrinking it, which is the opposite of the intuition
that a better nozzle should carry more propellant. At `eta_geom` = 1 the saving
is 213 kg -> 167 kg of slug per pulse **and** 9% more growth.

### What still needs the 2D solve

The **magnitude** of `c_exp` is settled well enough that the arrival radius no
longer decides anything. What a 2D solve would still add: whether the widening
front stays a coherent plow or breaks into jets and fingers, and whether the
material that reaches the bag wall stays in the flow or is lost past its edge.
Both bear on how much of the swept mass really couples -- but the impact sim's
Study 2 already found coupling holds with two orders of margin, so neither is
positioned to overturn this.

## Reproducing addendum 2

`nozzle_geometry.shocked_sound_speed` and `self_consistent_slug_ratio`. The
sound-speed table's provenance -- the `puffsat_impact_simulation` commit and the
exact three-line call -- is recorded on `_SOUND_SPEED_TABLE` in that module,
because this repository cannot regenerate it.
