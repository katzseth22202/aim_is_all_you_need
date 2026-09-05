# Corrections owed to the paper: `tab:bag_state`'s Jupiter column, regenerated

> **SUPERSEDED 2026-09-05 by `paper_corrections_bag_state_2026-09-05.md`. Do not
> work this file.** It was written against the paper at `1d42c91` and never
> applied. The paper then adopted `puffsat_impact_simulation`'s 23.8 m column
> and its 672.9 m³ (reply R14, commit `85f29ec`), which moves the third digit of
> every figure below and adds four items of its own. Every C-item here is
> carried forward into the 09-05 file, renumbered A4-A9 and A13, with its
> numbers refreshed. Kept only because ADR 0027 cites it.

**Written to be copied into
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
and worked there**, on `templateArxiv.tex`. Self-contained: every number needed
to make every edit is stated here, because an agent in the paper repository
cannot run the code that produced them.

Generated 2026-09-04 in `katzseth22202/aim_is_all_you_need` at the commit that
carries this file, by rerunning `make bag-state` and `make bag-converge` against
the paper at commit `1d42c91`. It answers the request that commit left open:

> `tab:bag_state`'s Jupiter column still encodes the retired claim in six rows,
> from the -0.73 MJ entry through the zero vapor fraction to the 0 kg pressure
> vessel, and the prose at the row now contradicts it. Regenerating it needs the
> saturation-curve solve from the companion repo rather than a hand edit, so it
> is left for the next pass.

Same ground rules as `docs/paper_corrections.md`: **the paper is not the source
of truth**; locate claims by grepping their quoted wording, not by line number;
if a number here looks wrong, say so rather than working around it.

The decision behind it is `docs/adr/0027-the-melting-gate-is-not-the-warming-ladder.md`.

---

## Summary

| | edit | what changes |
| --- | --- | --- |
| **C1** | `tab:bag_state`'s Jupiter column, six rows | the regenerated column below |
| **C2** | "Ice at 122 K absorbs 0.65 MJ/kg ... never reaches the boiling step" | **0.59**, and the rest does reach it |
| **C3** | `tab:axial_bag`'s caption, "that column is zero throughout" | **0.51-0.67 kg**, a tenth of the Earth column, still under the floor |
| **C4** | `sec:needle_through_fog`'s plug figures | 27.7 kg → **28.9 kg**, 309 K → **310 K**, 4.2 kg → **4.4 kg**, 114 K → **113 K** |
| **C5** | "the three hot legs clear theirs six to ten times over" | **five to seven and a half** |
| **C6** | `sec:needle_through_fog`, "the slug barely finishes melting either way" | it finishes **without** the plug and does not **with** it |
| **C7** | the paper repo's `CONTEXT.md` plug bullet | superseded twice over; full replacement below |

**Confirmed, no edit needed:** every claim the `1d42c91` prose already makes at
the row. "A few percent of the flow" is **2.2%**; "a pressure of order 1 kPa
against the Earth leg's 8.7 kPa" is **0.91** against **8.70**; "a fraction of a
kilogram" is **0.51 kg**; "melting completes at a leak of 2.19% and the cold leg
runs at 2.54%" reproduces to **2.187%**; the plug's **24.5 MJ** of the pulse's
**138 MJ** is exact.

**Do not change:** `tab:bag_state`'s Earth column (every row reproduces),
`tab:bag_sizing` (all 30 cells), `tab:axial_bag`'s numeric columns.

---

## A. What the rerun changes in the model, in one paragraph

The ladder from 122 K ice to room-temperature liquid has three terms — 0.236
warming the ice on the measured heat capacity of ice Ih, 0.334 fusing it, 0.084
warming the liquid — and only the **first two gate melting**. The code was
subtracting all three, 0.653 MJ/kg, before asking whether anything boiled. The
real gate is **0.570 MJ/kg**, and 0.646 MJ/kg of waste heat clears it with
**0.076 to spare**. That surplus does not all become vapour, because the liquid
still warms; where it lands is fixed by solving the energy balance and the
saturation curve together, which is what the paper asked for. It lands at
**278.8 K**, and the column's book then balances exactly:

```
0.646  waste heat
-0.570  melting gate (0.236 warm ice + 0.334 fusion)
-0.024  warming the liquid 5.7 K, to the mist temperature it actually reaches
-0.052  boiling 2.2% of the slug
= 0
```

---

## C1. `tab:bag_state`'s Jupiter column

**Locate:** `grep -n "the slug never finishes melting" templateArxiv.tex`

Six rows. Everything above the "Waste heat" row is unchanged, and the Earth
column is unchanged throughout.

| row | printed now | **replace with** |
| --- | ---: | ---: |
| Warming and melting the slug up to liquid | `\SI{-0.73}{\mega\joule}` | `\SI{-0.59}{\mega\joule}` |
| Left over to boil water | `0, the slug never finishes melting` | `\SI{0.05}{\mega\joule}` |
| Vapor fraction $x$ | `0` | `\num{0.022}` |
| Temperature, from the saturation curve | `no vapor` | `\SI{279}{\kelvin}` |
| Pressure | `0` | `\SI{0.91}{\kilo\pascal}` |
| Bag film as a pressure vessel | `\SI{0}{\kilogram}` | `0.24\%, \SI{0.51}{\kilogram}` |

The last row, "Bag film once the handling floor of `tab:axial_bag` is applied",
stays `\SIrange{2.0}{8.4}{\kilogram}`: 0.51 kg is far under that floor, so the
floor still governs and the printed range is still right.

Unrounded, for checking: warming 0.5938, left to boil 0.05191, $x$ 0.02199,
vapour mass 4.685 kg, mist 278.83 K, pressure 0.9140 kPa, film 0.5094 kg,
0.2392% of the slug.

**A caption sentence is worth adding**, because this column is now the only one
in the paper whose warming row is an output rather than an input: *"The Jupiter
column's warming row is solved rather than assumed. Only warming the ice and
fusing it gate melting; the liquid warming that follows competes with boiling
for the same surplus, so it is charged at the mist temperature the column
reaches rather than at room temperature."*

**Reproduce:** `make bag-state`, first section.

---

## C2. Cold storage absorbs 0.59 before the boiling step, not 0.65

**Locate:** `grep -n "never reaches the boiling step" templateArxiv.tex`

**Now:** *"Ice at \SI{122}{\kelvin} absorbs \SI{0.65}{\mega\joule\per\kilogram}
on its way to liquid, which is heat that never reaches the boiling step and so
never turns into pressure."*

0.65 is the whole ladder, and the waste heat never pays all of it: 0.594 goes
into warming and melting and the remaining 0.052 turns into the vapour the same
paragraph now says exists. **Suggested replacement:** *"Ice at \SI{122}{\kelvin}
takes \SI{0.59}{\mega\joule\per\kilogram} of the waste heat into warming and
melting, and only the \SI{0.05}{\mega\joule\per\kilogram} left over turns
into pressure."*

Strictly the split is simultaneous rather than sequential — the liquid warms
while it boils, which is why the column has to be solved — so "before the
boiling step" is the phrase to avoid, not just the number.

The two sentences after it are unaffected: the shading window still recovers the
full \SI{0.65}{\mega\joule\per\kilogram} and still rejects \SI{139}{\mega\joule}
from a 213 kg slug (0.653 × 213 = 139.1).

---

## C3. `tab:axial_bag`'s caption: the cold column is no longer zero

**Locate:** `grep -n "that column is zero throughout" templateArxiv.tex`

**Now:** *"...and from cold Jupiter storage the slug never finishes melting and
that column is zero throughout."*

**Replace with:** *"...and from cold Jupiter storage the slug boils about an
eighth as much water into a colder mist, so that column runs
\SIrange{0.51}{0.67}{\kilogram} across the same lengths -- a tenth of the
Earth one -- and never leaves the handling floor."*

Per length, if the exact figures are wanted: 0.51 kg at 10.8 m, 0.62 at 16 m,
0.65 at 23 m, 0.66 at 32 m, 0.67 at 50 m. They scale with $F$ exactly as the
Earth column does.

---

## C4. The plug's effect on the Earth column moved with the ladder

**Locate:** `grep -n "The vapor mass falls to" templateArxiv.tex`

The plug is charged the ladder, and the ladder fell from 0.73 to 0.653. The
**24.5 MJ** the paper already prints is correct — it is the *consequences* of
that smaller sink that were computed at 0.73 and did not move with it.

| | printed | **now** |
| --- | ---: | ---: |
| vapour mass, Earth with plug | \SI{27.7}{\kilogram} | **\SI{28.9}{\kilogram}** |
| mist, Earth with plug | \SI{309}{\kelvin} | **\SI{310}{\kelvin}** |
| pressure film at the 23 m column | \SI{4.2}{\kilogram} | **\SI{4.4}{\kilogram}** |
| polyethylene margin | \SI{114}{\kelvin} | **\SI{113}{\kelvin}** |

Unrounded: 28.935 kg, 310.18 K, 3.500 kg at $F = 1.5$ and 4.443 kg at the 23 m
column's $F = 1.90$, margin 423 − 310.18 = 112.8 K.

**None of it changes a conclusion.** 4.4 kg is still under the 5.1 kg handling
floor of that column, so the plug still retires the pressure vessel on the Earth
leg, and `sec:two_leg_nozzle`'s "the flown bag is handling-governed at both ends
of the burn" still holds. The sentence *"Working as a heat sink it removes
enough of the boiling load to bring the pressure film to \SI{4.2}{\kilogram}"*
in `sec:two_leg_nozzle` carries the same 4.2 and needs the same edit.

One share is worth a word: 24.5 of 137.5 MJ is **17.8%**, which the paper calls
"a sixth". Closer to a sixth than to a fifth, so the sentence stands, but 17.8%
is the number.

**Reproduce:** `make bag-state`, section "sec:needle_through_fog: the plug as a
heat sink".

---

## C5. "Six to ten times over" was computed against the old gate

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
line, not short of it.

---

## C6. "Barely finishes melting either way" is two different verdicts

**Locate:** `grep -n "barely finishes melting either way" templateArxiv.tex`

**Now:** *"From cold storage the slug barely finishes melting either way, so
this is the Earth-storage case, the one every film mass in this section is
quoted at."*

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

## C7. `CONTEXT.md`'s plug bullet, which `1d42c91` also flagged

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

## D. Do not change

- **`tab:bag_state`'s Earth column.** Every row reproduces: waste heat 0.6457,
  warming 0.21, left to boil 0.4357, $x$ 0.1846, mist 316.28 K, 8.70 kPa,
  4.85 kg and 2.28% of the slug.
- **`tab:bag_sizing`.** All 30 cells reproduce; its mist column is held at the
  Earth-storage vapour fraction, which did not move.
- **`tab:axial_bag`'s numeric columns.** Bore, conductor, $F$ and the Pressure
  column are all the Earth-storage case.
- **The `1d42c91` prose at the row**, in full. It was written ahead of this
  regeneration and every figure in it checks out.
- **The 4.4% comparison table**, wherever it is still quoted: 306 K and 328 K
  remain its fingerprint, and this repository still reproduces it cell for cell
  under its own superseded model.

---

## E. One thing this does not settle

**The ice-Ih integration is consumed here, not performed.** The 0.236 MJ/kg for
warming 122 K ice comes from the paper's own `tab:payload_sink` — 570 kJ/kg for
"warming and melting from 122 K" less 334 of latent heat — rather than from an
integration in this repository. The paper's three published anchors (2110 J/kg/K
at melting, 790 at 100 K, 175 at 30 K) reproduce its 236 and 60 kJ/kg spans on a
linear interpolation but come out 2% high on its 290 kJ/kg from 30 K, which is
not close enough to call a reproduction. If `tab:payload_sink`'s ice rows are
ever challenged, that integration is the thing to go and get.

Everything downstream of it in `tab:bag_state` is only as good as that number,
and it is worth knowing that the gate is not especially sensitive to it: the
cold leg would still melt through at 0.26 (gate 0.594, surplus 0.052) and would
still be past onset. What the ice-Ih correction changed was the size of the
surplus, not its sign.

**Reproduce everything above:** `make bag-state`, `make bag-converge`,
`pytest tests/test_bag_state.py tests/test_bag_converge.py -s`.
