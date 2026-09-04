# The melting gate is not the warming ladder, and the cold bag does boil

Status: accepted

Date: 2026-09-04

## Context

Balloon-Pulse-Propulsion commit `1d42c91`, *"Let the cold slug finish melting,
and bank the shell's load in the payload"*, corrected the enthalpy of bringing a
Jupiter slug from 122 K storage to room-temperature liquid, and asked this
repository for the half it could not do by hand:

> One half of the correction is not propagated yet. `tab:bag_state`'s Jupiter
> column still encodes the retired claim in six rows, from the -0.73 MJ entry
> through the zero vapor fraction to the 0 kg pressure vessel, and the prose at
> the row now contradicts it. Regenerating it needs the saturation-curve solve
> from the companion repo rather than a hand edit, so it is left for the next
> pass.

Two errors, and the paper names both. The smaller one is arithmetic: warming
122 K ice to the melting point was charged at a constant heat capacity, and
ice Ih's falls by a factor of twelve between melting and 30 K. Integrating the
measured curve gives **0.236 MJ/kg** rather than 0.26, and warming the liquid
afterwards costs 0.084 rather than 0.14. The ladder is 0.653 MJ/kg, not 0.73.

The larger one is structural, and it is this repository's to fix. The ladder has
three terms and **only the first two gate melting**:

| term | | gates melting? |
| --- | ---: | --- |
| warming 122 K ice to 273 K | 0.236 MJ/kg | yes |
| heat of fusion | 0.334 MJ/kg | yes |
| warming the liquid to room temperature | 0.084 MJ/kg | **no** |
| **ladder** | **0.653 MJ/kg** | |
| **gate** | **0.570 MJ/kg** | |

`BagState` subtracted the whole ladder before asking whether anything boiled.
Against 0.646 MJ/kg of waste heat that came out negative, so the Jupiter column
printed *"0, the slug never finishes melting"* and a 0 kg pressure vessel.
Against the real gate the same waste heat clears melting with **0.076 MJ/kg to
spare**, and the slug boils.

This was foreseen. `docs/paper_corrections_bag_state_2026-08-28.md` §C already
recorded that the Jupiter column's "no vapour at all" was *"a slightly stronger
claim than the model supports"* and recommended not acting on it, because at the
time the gap was an inconsistency in where the liquid warming was charged rather
than a demonstrated error. The paper's ice-Ih correction turned it into one.

## Decision

**Separate the gate from the ladder, and let the ice column solve its own
warming row.**

1. `MELTING_GATE` is a first-class constant: `WARMING_ICE_TO_MELT +
   HEAT_OF_FUSION` from ice, and **zero** from Earth's 278 K liquid — water in
   vacuum needs no threshold to start evaporating. `BagState.melts` and
   `boil_onset_leak()` both key off it.
2. `WARMING_ICE_TO_MELT` becomes **0.236 MJ/kg**. The integration itself is the
   paper's: `tab:payload_sink` prints 570 kJ/kg for "warming and melting from
   122 K" against 334 of latent heat, which leaves 236. It is consumed here as
   a published input, the way `WARMING_TO_LIQUID` already was.
3. **The Jupiter column's warming row becomes an output.** Its surplus over the
   gate is 0.076 MJ/kg against a 0.653 ladder, so *where* the liquid warming is
   charged decides the whole column: charge it at an assumed room temperature
   and 0.076 vanishes; charge it at the mist temperature the column actually
   reaches and most of it survives as vapour. So that column runs
   `melting_fixed_point` — the same saturation-curve solve `bag_converge`'s
   cut 4 has reported against since 2026-08-28 — and its book balances exactly:
   waste heat = warming + plug + boiling.
4. **The Earth column does not change.** Its 0.21 MJ/kg is a published input the
   paper still prints, it has no gate, and cut 4 goes on measuring the 0.5 kg of
   film that closing *its* loop would cost. Only the ice column moved.
5. `superseded_bag_state()` is pinned to the superseded *model* as well as the
   superseded leak, via a `published_warming` override. Its whole job is to
   reproduce the 4.4% table cell for cell so that 306 K and 328 K stay greppable
   downstream; a reproduction that drifted with the correction would stop
   finding them.

## Consequences

### `tab:bag_state`'s Jupiter column, regenerated

`make bag-state`. Every cell is an output; nothing is hand-entered.

| row | printed | **regenerated** |
| --- | ---: | ---: |
| waste heat | 0.65 MJ/kg | 0.646 (unchanged) |
| warming and melting up to liquid | −0.73 MJ/kg | **−0.594** |
| left over to boil | 0, never finishes melting | **0.052 MJ/kg** |
| vapour fraction `x` | 0 | **0.022** |
| vapour mass | — | **4.68 kg** |
| mist temperature | no vapour | **278.8 K** |
| pressure | 0 | **0.91 kPa** |
| film as a pressure vessel | 0 kg | **0.51 kg, 0.24%** |
| film once the handling floor applies | 2.0–8.4 kg | 2.0–8.4 kg (unchanged) |

The prose the paper already wrote at that row is confirmed to the digit: "a few
percent of the flow" is 2.2%, "a pressure of order 1 kPa against the Earth leg's
8.7" is 0.91 against 8.70, and "a fraction of a kilogram" is 0.51.

**The conclusion survives; the margin does not.** Half a kilogram is far under
the handling floor — 2.0–8.4 kg over the table's own 5.4 m sphere, 2.4–10.0 kg
over the flown 23 m column — so cold storage still removes the
*pressure vessel* and the cold bag is still a containment membrane. But boiling
onset moves from 2.93% to **2.19%** and the cold leg leaks 2.54%, so the leg is
**past** that line rather than 1.2x short of it. The leak model has moved once
and it moved in the direction that costs.

### Three figures downstream of the correction, which the paper does not have yet

These follow from the corrected ladder rather than from the gate, and the paper
was not in a position to compute them:

- **The plug's effect on the Earth column moved.** The plug is charged the
  ladder, so it now removes 0.653 × 37.5 = **24.5 MJ** rather than 27.4 — which
  is the figure the paper already prints. But the *consequences* of that smaller
  sink were not propagated: the Earth column's vapour falls to **28.9 kg** and
  its mist to **310.2 K**, not the 27.7 kg and 309 K the paper prints, and the
  flown 23 m pressure film is **4.4 kg**, not 4.2. The polyethylene margin is
  **113 K**, not 114.
- **"Six to ten times over" is a pre-correction figure.** The three hot legs
  clear their onset **4.9x, 6.5x and 7.5x**, because onset fell with the gate.
- **The plug's share of the bill is 17.8%**, between the paper's "a sixth" and
  the retired "a fifth", and closer to the former.

### One reversal worth stating

**The plug now keeps the cold leg dry, where it used to buy nothing there.**
Cold storage on its own melts through by 0.076 MJ/kg; the plug's 24.5 MJ is
0.115 MJ/kg of slug, more than that surplus, so it puts the column back under
the gate. `tab:bag_state` is the no-plug case and stays so, but "from cold
storage the slug never finishes melting with or without the plug" (ADR 0018,
2026-08-28 addendum) is now false in both halves — without the plug it melts,
and with it the plug is the reason it does not.

### What did not move

- The Earth column, in every row.
- `tab:bag_sizing` and `tab:axial_bag`, whose Pressure columns are quoted at the
  Earth-storage vapour state. `tab:axial_bag`'s note that cold storage gives
  "0 kg at every length" is the one line in them that is now wrong: the cold
  column runs 0.51-0.67 kg, a tenth of the Earth one, still under the floor at
  every length.
- The superseded 4.4% table, deliberately.
- Every leg but the cold end of the growth push: the three hot legs still never
  finish melting, so the film is 0 kg on all of them.

## Reproducing

```bash
make bag-state      # the regenerated table, the leak bracket, the plug section
make bag-converge   # cut 4, now the Earth column alone
pytest tests/test_bag_state.py tests/test_bag_converge.py -s
```

Inputs, so this does not depend on scratch state: the table is the 45.58 km/s
cold end of the growth push at the solved equilibrium leak of 2.54% against
21.43 MJ/kg of stored field energy (`SOLVED_LEAK_FRACTIONS`, `LEG_PLUME_STATES`,
both unchanged); the melting gate is 0.236 + 0.334 MJ/kg; the latent heat is the
2.36 MJ/kg back-solved from the paper's own two columns; the bag is 213 kg in
659.6 m³ and the saturation curve is Magnus-Tetens.

## Open items

- **The ice-Ih integration is consumed, not performed.** 0.236 MJ/kg is read off
  the paper's `tab:payload_sink` rather than integrated here. Doing it properly
  needs a measured `c_p(T)` table for ice Ih with real provenance; the paper's
  three anchors (2110 J/kg/K at melting, 790 at 100 K, 175 at 30 K) reproduce
  its 236 and 60 kJ/kg spans on a linear interpolation but miss its 290 kJ/kg
  from 30 K by 2%, which is not good enough to publish as a reproduction.
- **The Earth column's own loop is still cut** (cut 4). Closing it would move
  that column 2 K and half a kilogram of film. Unchanged recommendation: leave
  it, the paper prints the row as an input.
- **`sec:needle_through_fog`'s Earth-with-plug figures are owed an edit**, per
  the downstream list above. Written up self-contained in
  `docs/paper_corrections_bag_state_2026-09-04.md`.
