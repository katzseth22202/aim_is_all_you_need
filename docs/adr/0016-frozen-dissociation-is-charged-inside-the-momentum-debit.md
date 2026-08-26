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
