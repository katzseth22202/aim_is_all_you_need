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
