# Corrections owed to the paper: the bag-state rerun R14 asked for

**Written to be copied into
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
and worked there**, on `templateArxiv.tex`. Self-contained: every number needed
to make every edit is stated here, because an agent in the paper repository
cannot run the code that produced them.

Generated 2026-09-05 in `katzseth22202/aim_is_all_you_need` at the commit that
carries this file, by rerunning `make bag-state`, `make nozzle-geom` and
`make bag-converge` against the paper at commit `85f29ec`.

**This file supersedes `docs/paper_corrections_bag_state_2026-09-04.md`**, which
was written against the paper at `1d42c91` and has not been applied. Every item
in it is carried forward here with its numbers refreshed; do not work the two
side by side. It is the single pass reply R14 asked for:

> The generator has **not** been rerun, so the paper's table and its generator
> now disagree. That sits alongside the `tab:bag_state` recompute already owed
> for the ice-enthalpy defects. **Both want one pass.**

The decisions behind it are `docs/adr/0027-the-melting-gate-is-not-the-warming-ladder.md`
(the ice-enthalpy half) and `docs/adr/0029-one-standoff-volume-and-it-is-the-columns.md`
(the geometry half).

Same ground rules as `docs/paper_corrections.md`: **the paper is not the source
of truth**; locate claims by grepping their quoted wording, not by line number;
if a number here looks wrong, say so rather than working around it.

---

## Summary

| | edit | what changes |
| --- | --- | --- |
| **A1** | `tab:axial_bag`'s **Pressure** column, four of five cells | the hand recompute is high by the volume ratio; 4.8 / 5.9 / 6.2 / 6.3 / 6.4 |
| **A2** | `tab:bag_state`'s Earth pressure and film | **8.5 kPa**, not 8.7; **4.8 kg**, not 4.9 |
| **A3** | `tab:bag_state`'s caption, "in a 5.4 m bag" | it is the **5.44 m** equal-volume sphere now |
| **A4** | `tab:bag_state`'s Jupiter column, six rows | the regenerated column below (was C1) |
| **A5** | "Ice at 122 K absorbs 0.65 MJ/kg ... never reaches the boiling step" | **0.59**, and the rest does reach it (was C2) |
| **A6** | `tab:axial_bag`'s caption, "that column is zero throughout" | **0.51-0.68 kg**, a tenth of the Earth column (was C3) |
| **A7** | `sec:needle_through_fog`'s plug figures | 27.7 kg → **28.9 kg**, 309 K → **310 K**, 4.2 kg → **4.4 kg**, 114 K → **113 K** (was C4) |
| **A8** | "the three hot legs clear theirs six to ten times over" | **five to seven and a half** (was C5) |
| **A9** | "the slug barely finishes melting either way" | it finishes **without** the plug and does not **with** it (was C6) |
| **A10** | `sec:needle_through_fog`, "5.1 kg over the column's 437 m²" | **5.2 kg over 449 m²** |
| **A11** | `sec:minimum_nozzle`, the mirror's "7.6 T is enough" | **7.5 T** |
| **A12** | `sec:needle_through_fog`, "ends up sweeping k = 7.2" | **7.1** |
| **A13** | the paper repo's `CONTEXT.md` plug bullet | superseded twice over; replacement below (was C7) |

**Confirmed, no edit needed:** `tab:axial_bag`'s Bore, Conductor, `F`, Area and
Handling columns — all twenty-five cells reproduce exactly at the adopted
672.9 m³, including the sphere row's 10.9 m / 5.44 m / 371 m². The mirror's
ratio of 1.17 and its 23 MPa. `k` = 8.52 with 8.5 as the round figure, which the
generator now returns *identically* as 213/25. Bag density 0.32. The 449 m²,
the 7.5 kg/m² areal density, and every "23.8 m" R14 put into the prose.

---

## A. What the rerun changed in the model, in two paragraphs

**The geometry (ADR 0029).** This repository had both of the numbers R14 was
choosing between, and in one module it had them at once: `nozzle_geometry`
carried the sim's 3.0 m bore and 23.8 m column *and* a 659.6 m³ bag, so its
full-bore sweep returned `k` = 8.69 for a quantity whose definition is
213/25 = 8.52. `BAG_VOLUME` is now derived — `pi r^2 l` = 672.9 m³ — and the
sphere radius is derived from *it*, 5.4361 m, so the two cannot drift again.
The mist that fills the bag is solved in the same volume, because adopting the
geometry and not the state would rebuild the split R14 was closing.

**The melting gate (ADR 0027), unchanged from the 09-04 file.** The ladder from
122 K ice to room-temperature liquid has three terms — 0.236 warming the ice on
the measured capacity of ice Ih, 0.334 fusing it, 0.084 warming the liquid — and
only the **first two gate melting**. The code was subtracting all three,
0.653 MJ/kg, before asking whether anything boiled. The real gate is
**0.570 MJ/kg**, and 0.646 MJ/kg of waste heat clears it with **0.076 to spare**.
Where that surplus lands is fixed by solving the energy balance and the
saturation curve together. The cold column's book then balances exactly:

```
 0.646  waste heat
-0.570  melting gate (0.236 warm ice + 0.334 fusion)
-0.023  warming the liquid 5.6 K, to the mist temperature it actually reaches
-0.052  boiling 2.2% of the slug
=    0
```

---

## A1. `tab:axial_bag`'s Pressure column is high by the volume ratio

**Locate:** `grep -n "Pouring the same" templateArxiv.tex`, then the table below it.

`85f29ec` recomputed the table by hand at 672.9 m³ and moved the Pressure column
up by 0.1 kg in four rows. It should not have moved at all, except downward by a
hundredth.

| length | printed | **replace with** |
| ---: | ---: | ---: |
| 10.9 m (sphere) | `\SI{4.9}{\kilogram}` | `\SI{4.8}{\kilogram}` |
| 16 m | `\SI{6.0}{\kilogram}` | `\SI{5.9}{\kilogram}` |
| 23.8 m | `\SI{6.3}{\kilogram}` | `\SI{6.2}{\kilogram}` |
| 32 m | `\SI{6.4}{\kilogram}` | `\SI{6.3}{\kilogram}` |
| 50 m | `\SI{6.5}{\kilogram}` | `\SI{6.4}{\kilogram}` |

Unrounded: 4.844, 5.878, 6.162, 6.275, 6.367 kg.

**Why.** `eq:bag_film_mass` is `F x rho_f R_g T / (M sigma)`, and **volume does
not appear in it** — the paper says so itself two paragraphs earlier ("The radius
does not appear, which makes the film the second quantity in this section a
bigger bag leaves alone"). The film sizes a vessel holding `PV = n R_g T`, and a
bigger bag drops `P` by exactly what it adds to `V`. Multiplying the film by the
new volume charges the change twice: once in `V`, and once already inside the
`P` the old volume produced. The gap is the volume ratio, 672.93/659.6 = 1.0202:
multiplying the five modelled figures by it gives 4.94, 6.00, 6.29, 6.40 and
6.50, which is every printed cell to the digit.

The change the new volume *does* make runs the other way and is a tenth of the
size: the same vapour spread 2% more thinly saturates 0.4 K colder, and the film
is linear in `T`. That is why the sphere row falls from 4.850 to 4.844 kg.

**Consequence in the prose.** `grep -n "moves the bag from" templateArxiv.tex`:
*"The shape factor moves the bag from \SI{4.9}{\kilogram} to
\SI{6.3}{\kilogram}, which is \SI{2.9}{\percent} of the slug"* becomes **4.8 to
6.2**, still 2.9% of the slug (6.162/213 = 2.89%). *"For a fifth more conductor
and \SI{1.3}{\kilogram} more film"* is **confirmed and needs no edit**: the gap
is 6.162 − 4.844 = 1.32 kg, and it was 1.31 before the volume moved. This is
also the item `docs/paper_changes_owed.md` P6 chased, closed for the second
time.

**Reproduce:** `make bag-state`, section "tab:axial_bag (item 9)".

---

## A2. `tab:bag_state`'s Earth pressure and film move with the volume

**Locate:** `grep -n "Bag film as a pressure vessel" templateArxiv.tex`

| row | printed | **replace with** |
| --- | ---: | ---: |
| Pressure (Earth) | `\SI{8.7}{\kilo\pascal}` | `\SI{8.5}{\kilo\pascal}` |
| Bag film as a pressure vessel (Earth) | `2.3\%, \SI{4.9}{\kilogram}` | `2.3\%, \SI{4.8}{\kilogram}` |

Unrounded: 8.519 kPa and 4.844 kg, 2.274% of the slug. The mist temperature is
**315.87 K** and still prints as `\SI{316}{\kelvin}`; do not change it.

**Two other places quote the 8.7.** `grep -n "8.7}{\kilo\pascal}"`: the
polyethylene paragraph's *"the bag holds at most \SI{8.7}{\kilo\pascal} for the
seconds between venting and impact"* becomes **8.5**, and
`sec:needle_through_fog`'s *"a pressure of order \SI{1}{\kilo\pascal} against
the Earth leg's \SI{8.7}{\kilo\pascal}"* becomes **8.5**.

**Worth knowing while you are in there.** `sec:watering_it_down` already prints
*"against the \SI{8.5}{\kilo\pascal} of the steam bag"* for this same row. After
this edit that sentence is right and needs no change, and an internal
disagreement the paper was carrying closes on its own. We are not claiming the
8.5 was derived this way — it has one commit of provenance and may have been a
slip — only that it now agrees.

**Also affected by the pressure fall**, if it is quoted anywhere we have not
found: the Earth-with-plug pressure is **6.15 kPa** (was 6.28).

**Reproduce:** `make bag-state`, first section.

---

## A3. `tab:bag_state`'s caption: the bag is 5.44 m, not 5.4

**Locate:** `grep -n "in a \\\\SI{5.4}{\\\\meter} bag" templateArxiv.tex`

The caption reads *"Per kilogram of slug, for the \SI{213}{\kilogram} slug in a
\SI{5.4}{\meter} bag"*, and the paragraph above it says *"\autoref{tab:bag_state}
runs that chain for the \SI{5.4}{\meter} row"*.

Once `tab:axial_bag` pours 672.9 m³, the sphere of that volume is **5.4361 m** —
which is why R14's own table row is now 10.9 m / 5.44 m rather than 10.8 / 5.40.
`tab:bag_state` is the mist filling the flown bag, so it is solved in 672.9 m³
too. **Replace both "5.4 m" with "5.44 m"**, or reword the caption to *"for the
\SI{213}{\kilogram} slug in the \SI{672.9}{\cubic\meter} the column encloses"*,
which is the more robust form and does not have to move again.

`tab:bag_sizing`'s **5.4 m row is a sweep row and does not move.** All 30 of its
cells still reproduce, including its 316 K mist and its 4.11 T. Its row is a
candidate bag; 5.44 m is the flown one.

---

## A4. `tab:bag_state`'s Jupiter column

**Locate:** `grep -n "the slug never finishes melting" templateArxiv.tex`

Six rows. Everything above the "Waste heat" row is unchanged.

| row | printed now | **replace with** |
| --- | ---: | ---: |
| Warming and melting the slug up to liquid | `\SI{-0.73}{\mega\joule}` | `\SI{-0.59}{\mega\joule}` |
| Left over to boil water | `0, the slug never finishes melting` | `\SI{0.05}{\mega\joule}` |
| Vapor fraction $x$ | `0` | `\num{0.022}` |
| Temperature, from the saturation curve | `no vapor` | `\SI{279}{\kelvin}` |
| Pressure | `0` | `\SI{0.91}{\kilo\pascal}` |
| Bag film as a pressure vessel | `\SI{0}{\kilogram}` | `0.24\%, \SI{0.51}{\kilogram}` |

Unrounded: warming 0.5932, left to boil 0.05248, $x$ 0.02224, vapour mass
4.736 kg, mist 278.70 K, pressure 0.9054 kPa, film 0.5148 kg, 0.242% of the slug.

The last row, "Bag film once the handling floor of `tab:axial_bag` is applied",
becomes **`\SIrange{2.0}{8.5}{\kilogram}`** — the sphere row's floor rose with
its area, 366 m² to 371 m², so the printed 2.0–8.4 is now 2.0–8.5. The Earth
column's version of that row becomes **`\SIrange{4.8}{8.5}{\kilogram}`**, since
its lower end is the pressure film of A2.

**A caption sentence is worth adding**, because this column is now the only one
in the paper whose warming row is an output rather than an input: *"The Jupiter
column's warming row is solved rather than assumed. Only warming the ice and
fusing it gate melting; the liquid warming that follows competes with boiling
for the same surplus, so it is charged at the mist temperature the column
reaches rather than at room temperature."*

**Reproduce:** `make bag-state`, first section.

---

## A5. Cold storage absorbs 0.59 before the boiling step, not 0.65

**Locate:** `grep -n "never reaches the boiling step" templateArxiv.tex`

**Now:** *"Ice at \SI{122}{\kelvin} absorbs \SI{0.65}{\mega\joule\per\kilogram}
on its way to liquid, which is heat that never reaches the boiling step and so
never turns into pressure."*

0.65 is the whole ladder, and the waste heat never pays all of it: 0.593 goes
into warming and melting and the remaining 0.052 turns into the vapour the same
paragraph now says exists. **Suggested replacement:** *"Ice at \SI{122}{\kelvin}
takes \SI{0.59}{\mega\joule\per\kilogram} of the waste heat into warming and
melting, and only the \SI{0.05}{\mega\joule\per\kilogram} left over turns into
pressure."*

Strictly the split is simultaneous rather than sequential — the liquid warms
while it boils, which is why the column has to be solved — so "before the
boiling step" is the phrase to avoid, not just the number.

The two sentences after it are unaffected: the shading window still recovers the
full \SI{0.65}{\mega\joule\per\kilogram} and still rejects \SI{139}{\mega\joule}
from a 213 kg slug (0.653 × 213 = 139.1).

---

## A6. `tab:axial_bag`'s caption: the cold column is no longer zero

**Locate:** `grep -n "that column is zero throughout" templateArxiv.tex`

**Now:** *"...and from cold Jupiter storage the slug never finishes melting and
that column is zero throughout."*

**Replace with:** *"...and from cold Jupiter storage the slug boils about an
eighth as much water into a colder mist, so that column runs
\SIrange{0.51}{0.68}{\kilogram} across the same lengths -- a tenth of the
Earth one -- and never leaves the handling floor."*

Per length: 0.515 kg at 10.9 m, 0.625 at 16 m, 0.655 at 23.8 m, 0.667 at 32 m,
0.677 at 50 m. They scale with $F$ exactly as the Earth column does.

---

## A7. The plug's effect on the Earth column moved with the ladder

**Locate:** `grep -n "The vapor mass falls to" templateArxiv.tex`

The plug is charged the ladder, and the ladder fell from 0.73 to 0.653. The
**24.5 MJ** the paper already prints is correct — it is the *consequences* of
that smaller sink that were computed at 0.73 and did not move with it.

| | printed | **now** |
| --- | ---: | ---: |
| vapour mass, Earth with plug | \SI{27.7}{\kilogram} | **\SI{28.9}{\kilogram}** |
| mist, Earth with plug | \SI{309}{\kelvin} | **\SI{310}{\kelvin}** |
| pressure film at the flown column | \SI{4.2}{\kilogram} | **\SI{4.4}{\kilogram}** |
| polyethylene margin | \SI{114}{\kelvin} | **\SI{113}{\kelvin}** |

Unrounded: 28.935 kg, 309.79 K, 3.496 kg at $F = 1.5$ and 4.447 kg at the
23.8 m column's $F = 1.908$, margin 423 − 309.79 = 113.2 K.

**None of it changes a conclusion.** 4.4 kg is still under the **5.2 kg**
handling floor of that column (A10), so the plug still retires the pressure
vessel on the Earth leg, and `sec:two_leg_nozzle`'s "the flown bag is
handling-governed at both ends of the burn" still holds. The sentence *"Working
as a heat sink it removes enough of the boiling load to bring the pressure film
to \SI{4.2}{\kilogram}"* in `sec:two_leg_nozzle` carries the same 4.2 and needs
the same edit.

One share is worth a word: 24.5 of 137.5 MJ is **17.8%**, which the paper calls
"a sixth". Closer to a sixth than to a fifth, so the sentence stands, but 17.8%
is the number.

**Reproduce:** `make bag-state`, section "sec:needle_through_fog: the plug as a
heat sink".

---

## A8. "Six to ten times over" was computed against the old gate

**Locate:** `grep -n "six to ten times over" templateArxiv.tex`

Onset scales with the gate, and the gate fell from 0.73 to 0.570, so every leg's
margin fell with it. Against the same solved leaks:

| leg | $E_B$ | leak | onset | margin |
| ---: | ---: | ---: | ---: | ---: |
| 75 km/s | 12.2 GJ | 0.11% | 0.82% | **7.5x** |
| 65 km/s | 9.06 GJ | 0.17% | 1.10% | **6.5x** |
| 56.53 km/s | 6.96 GJ | 0.29% | 1.43% | **4.9x** |
| 45.58 km/s | 4.57 GJ | 2.54% | 2.19% | **0.86x** — past onset |

**Replace** *"while the three hot legs clear theirs six to ten times over"* with
*"while the three hot legs clear theirs five to seven and a half times over"*.

The cold leg's row is the one the paper already states correctly: it is past its
line, not short of it. Onset is 2.187%, which the paper prints as 2.19%.

---

## A9. "Barely finishes melting either way" is two different verdicts

**Locate:** `grep -n "barely finishes melting either way" templateArxiv.tex`

It does not go the same way with and without the plug. Without, the surplus over
the gate is 0.076 MJ/kg and the slug melts through. With, the plug takes 0.115
MJ/kg of slug — more than the surplus — and the melt does not complete: that
column is dry, 0 kg of film, no mist at all.

That is a *better* fact for the paragraph than the one it replaces, because it
makes the plug do something on the Jupiter leg instead of nothing. **Suggested
replacement:** *"From cold storage the plug is what keeps the bag dry rather
than what lightens it: the bare slug melts through by
\SI{0.076}{\mega\joule\per\kilogram} and the plug absorbs more than that, so the
column boils nothing at all. The figures below are therefore the Earth-storage
case, the one every film mass in this section is quoted at."*

---

## A10. The handling floor rose with the column's area

**Locate:** `grep -n "that \\\\SI{12.7}{\\\\micro\\\\meter} of polyethylene weighs" templateArxiv.tex`

R14 already moved the membrane area from 437 m² to **449 m²** everywhere it
appears, and moved the handling range on that row to 2.5–10.3 kg. One figure did
not come with it: *"That is under the \SI{5.2}{\kilogram} that
\SI{12.7}{\micro\meter} of polyethylene weighs over the column's
\SI{449}{\square\meter}"* is already correct at **5.242 kg** — confirmed, no
edit. But `tab:bag_state`'s handling row still prints the sphere's old
2.0–8.4 kg; see A4.

For checking, the whole floor at the adopted volume:

| length | area | 6 µm | 12.7 µm | 25 µm |
| ---: | ---: | ---: | ---: | ---: |
| 10.9 m | 371 m² | 2.05 kg | 4.34 kg | 8.54 kg |
| 16 m | 368 m² | 2.03 kg | 4.30 kg | 8.46 kg |
| 23.8 m | 449 m² | 2.48 kg | 5.24 kg | 10.32 kg |
| 32 m | 520 m² | 2.87 kg | 6.08 kg | 11.96 kg |
| 50 m | 650 m² | 3.59 kg | 7.60 kg | 14.96 kg |

---

## A11. The mirror's field falls to 7.5 T

**Locate:** `grep -n "7.6}{\\\\tesla} is enough" templateArxiv.tex`

R14 changed that sentence's *"spread through \SI{660}{\cubic\meter}"* to 672.9
without moving what depends on it. `B^2/2\mu_0` has to stand off a pressure that
falls as the volume rises, so the field falls as `V^{-1/2}`:

| | printed | **now** |
| --- | ---: | ---: |
| ram/static ratio | 1.17 | 1.17 — **unchanged**, it is a ratio |
| wall pressure | \SI{23}{\mega\pascal} | 22.7 MPa, still **23** — no edit |
| field | \SI{7.6}{\tesla} | **\SI{7.5}{\tesla}** |

The ship-end case (6.7, 1.26 GPa, 56 T) is worked at a bore cross-section rather
than the bag volume and does not move.

**Reproduce:** `make nozzle-geom`, "Item 12: mirror stagnation pressure versus
plug position".

---

## A12. The self-widening front delivers 7.1, not 7.2

**Locate:** `grep -n "sweeping \$k = 7.2\$" templateArxiv.tex`

The snowplow sweeps the bag's own density, and that density fell 2% with the
volume, so every delivered `k` falls with it: the design 0.15 m arrival gives
**7.07** rather than 7.21, and a front arriving at 0.8 of the bore gives 8.43
rather than 8.60.

**The argument is unchanged and slightly stronger.** ADR 0016's tolled optimum
band is 6.75–7.77; the compact arrival now sits 0.67 inside its upper edge where
it used to sit 0.56 inside, and the wide front still overshoots. *"ends up
sweeping $k = 7.2$ rather than 0.02"* becomes **7.1 rather than 0.02** (the
rigid-front figure is 0.021).

R14's own `k` = 8.52 is confirmed and is now exact rather than approximate: with
the density and the swept volume finally the same bag, the full-bore sweep
returns `BAG_SLUG_MASS / IMPACTOR_MASS` = 213/25 identically.

**Reproduce:** `make nozzle-geom`, "Item 11".

---

## A13. `CONTEXT.md`'s plug bullet, which `1d42c91` also flagged

**Locate (in the paper repo's `CONTEXT.md`, not the paper):**
`grep -n "The plug is thermally free" CONTEXT.md`

**Now:** *"it absorbs 0.73 MJ/kg on the way to liquid. 37.5 kg soaks 27 MJ of
the 211 MJ waste-heat bill. Vapor ends at 24.7 kg vs 23.4 kg baseline,
saturation 307 K vs 306 K. PE keeps 116 K of melt margin."*

Every figure in it is superseded twice over — once by the 2026-08-28 leak solve
(the 211 MJ bill and the 306/307 K pair are the 4.4% table's) and once by this
correction. **Replace with:** *"it absorbs 0.653 MJ/kg on the way to liquid.
37.5 kg soaks 24.5 MJ of the 138 MJ waste-heat bill, a sixth of it. From Earth
storage vapor ends at 28.9 kg against a 39.3 kg baseline and the mist falls from
316 K to 310 K; PE keeps 113 K of melt margin. From cold storage the plug is
what keeps the bag dry at all."*

---

## B. Do not change

- **`tab:axial_bag`'s Bore, Conductor, `F`, Area and Handling columns.** All
  twenty-five cells reproduce at 672.9 m³, sphere row included: 10.872 m prints
  as 10.9, its bore is 5.4361 m, its area 371.4 m². R14's adoption is confirmed
  in full; only the Pressure column (A1) is off.
- **`tab:bag_sizing`.** All 30 cells. Its 5.4 m row is a sweep row and is not
  the flown bag; see A3.
- **`tab:bag_state`'s Earth column above the double rule**, and its `x` = 0.18:
  waste heat 0.6457, warming 0.21, left to boil 0.4357, $x$ 0.1846. Only the
  pressure and film rows move (A2).
- **The `1d42c91` prose at the Jupiter row**, in full. "A few percent of the
  flow" is 2.2%; "a fraction of a kilogram" is 0.51 kg; "melting completes at a
  leak of 2.19% and the cold leg runs at 2.54%" reproduces to 2.187%.
- **The 4.4% comparison table**, wherever it is still quoted: 306 K and 328 K
  remain its fingerprint, and this repository still reproduces it cell for cell
  under its own superseded model.
- **Bag density 0.32**, the areal density 7.5 kg/m², and every 23.8 m and
  449 m² R14 put into the prose.

---

## C. Two things this does not settle

**The ice-Ih integration is consumed here, not performed.** The 0.236 MJ/kg for
warming 122 K ice comes from the paper's own `tab:payload_sink` — 570 kJ/kg for
"warming and melting from 122 K" less 334 of latent heat — rather than from an
integration in this repository. The paper's three published anchors (2110 J/kg/K
at melting, 790 at 100 K, 175 at 30 K) reproduce its 236 and 60 kJ/kg spans on a
linear interpolation but come out 2% high on its 290 kJ/kg from 30 K, which is
not close enough to call a reproduction. The gate is not especially sensitive to
it: the cold leg would still melt through at 0.26 and would still be past onset.

**The companion sim's runs are 2% off the adopted density.** This is an ask on
`katzseth22202/puffsat_impact_simulation` rather than an edit to the paper, and
it is recorded here so the paper knows the shape of it. The plume state
(`data/plume_state.csv`), the conductivity fits behind `tab:seed_window`, the
conductivity cliff and the shocked sound speeds behind the snowplow were all
solved at **0.323 kg/m³**, which is 213 kg in the *old* 659.6 m³. The adopted
bag is **0.3165**. Nothing suggests it matters — the sound-speed table moves 9%
across an eightfold span in shock compression, so 2% in density is a fortieth of
a known-small sensitivity — but the mismatch is real and is not assumed away
here: `nozzle_geometry.SIM_SOLVE_DENSITY` names that density and the cross-repo
pins are evaluated at it, so they stay exact and stay honest about which bag they
are. **Nothing in the paper needs to wait for it.**

**Reproduce everything above:** `make bag-state`, `make nozzle-geom`,
`make bag-converge`,
`pytest tests/test_bag_state.py tests/test_nozzle_geometry.py tests/test_bag_converge.py -s`.
