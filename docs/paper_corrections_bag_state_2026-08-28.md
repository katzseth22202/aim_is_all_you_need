# Corrections owed to the paper: the bag-state rerun

**Written to be copied into
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
and worked there**, on `templateArxiv.tex`. Self-contained: every number needed
to make every edit is stated here, because an agent in the paper repository
cannot run the code that produced them.

Generated 2026-08-28 in `katzseth22202/aim_is_all_you_need` at the commit that
carries this file, by rerunning `make bag-state` against the paper at commit
`7c56c5d`. It answers `todos/bag_state_rerun_2026-08-28.md`, which asked for
four numbers and turned up eight.

Same ground rules as `docs/paper_corrections.md`: **the paper is not the source
of truth**; locate claims by grepping their quoted wording, not by line number;
if a number here looks wrong, say so rather than working around it.

---

## Summary

`tab:bag_state` itself is **correct as printed and reproduces to every cell**.
What is wrong is a set of numbers computed from the *superseded* version of that
table, before the field leak was solved. They are identifiable: **306 K and
328 K are the fingerprint.** Those were the mist temperatures of the 4.4%-leak
table; the table now prints 316 K in one column and no vapour at all in the
other.

| | edit | what changes |
| --- | --- | --- |
| **B1** | `sec:needle_through_fog`'s vapour pair | 24.5/23.4 kg → **27.7/39.3 kg**, and the direction reverses |
| **B2** | the same sentence's mist temperatures | 306→307 K → **316→309 K** |
| **B3** | the plug's share of the bill | 13% implied → **20%**, stated |
| **B4** | the plug's effect on the film | "moves it to 6.4 kg" → **moves it to 4.2 kg** |
| **B5** | the polyethylene margin paragraph | "306 K on Jupiter and 328 K on Earth" → no mist on Jupiter, **316 K** on Earth |
| **B6** | the self-regulation sentence | "at the same 306 K" → **316 K** |
| **B7** | `sec:needle_through_fog`'s film delta | "0.8 kg more film" → **1.3 kg** |
| **B8** | the 116 K polyethylene margin | → **107 K** without the plug, **114 K** with it |

**Do not change:** `tab:bag_state`, `tab:bag_sizing`, `tab:axial_bag`. All three
reproduce cell for cell. See section D.

---

## A. What the rerun confirms

`make bag-state` at current HEAD reproduces the published `tab:bag_state`
exactly. The table is one leg — the 45.58 km/s cold end of the growth push, at
the solved 2.54% leak against 21.43 MJ/kg of stored field energy:

| row | published | computed |
| --- | ---: | ---: |
| blackbody intercept | 0.10 MJ/kg | 0.1013 |
| field leak | 0.55 MJ/kg | 0.5444 |
| waste heat | 0.65 MJ/kg | 0.6457 |
| left to boil, Jupiter | never finishes melting | −0.084 |
| left to boil, Earth | 0.44 MJ/kg | 0.4357 |
| `x`, Earth | 0.18 | 0.1846 |
| mist, Earth | 316 K | 316.28 |
| pressure, Earth | 8.7 kPa | 8.70 |
| film, Earth | 2.3%, 4.9 kg | 2.277%, 4.85 kg |

The one wrinkle: the leak row reads "2.54% of 21.5 MJ/kg", and 21.5 is itself
the rounded 21.43 the cold leg actually stores, so the printed 0.55 is 0.3%
above the product of the unrounded inputs. **No edit needed** — it is the same
shape of rounding the superseded table had, and 0.55 is the right two-figure
answer.

The **per-pulse bill is 137.5 MJ**, which rounds to the 138 MJ already in the
text. That correction stands.

The **code was the thing that was stale**, not the paper: `bag_state.py` was
still reproducing the 4.4% table under the name `paper_bag_state`, and its
`tab:bag_sizing` mist column was still held at the superseded `x` = 0.11. Both
are fixed, and the mist column now reads its fraction off the table rather than
from a pinned constant, so it cannot go stale again. Nothing in the paper
depended on the old code path.

---

## B. The edits

### B1. The plug's vapour pair points the wrong way, and is not two configurations

**Locate:** `grep -n "The vapor mass lands at" templateArxiv.tex`

**Now:** *"The vapor mass lands at \SI{24.5}{\kilogram} against
\SI{23.4}{\kilogram} without the plug, and the saturation curve moves the mist
from \SI{306}{\kelvin} to \SI{307}{\kelvin}."*

**The problem is physical, not arithmetic.** The plug is 37.5 kg of extra
condensed mass absorbing heat the pulse was going to deliver anyway. At a fixed
waste-heat bill it can only **remove** energy from the boiling step, so adding
it must **lower** the vapour mass and **cool** the mist. The sentence has both
rising, which contradicts the sentence immediately before it.

**Where the old pair came from:** it was never two configurations. On the
pre-solve 211 MJ bill, `(211 − 213 × 0.73)/2.26 = 24.6` and
`0.11 × 213 = 23.4` — the same energy pool divided by two different latent
heats, one at the boiling point and one near 300 K, and the 0.11 is the
superseded table's Jupiter vapour fraction. Neither was a with-plug or
without-plug case.

**Should be**, on the 137.5 MJ bill at Earth storage (the only column that still
boils):

| | without the plug | with the plug |
| --- | ---: | ---: |
| left to boil | 92.8 MJ | 65.4 MJ |
| vapour mass | **39.3 kg** | **27.7 kg** |
| `x` | 0.185 | 0.130 |
| mist | **316.3 K** | **309.3 K** |

**Suggested replacement sentence:** *"The vapor mass falls from
\SI{39.3}{\kilogram} to \SI{27.7}{\kilogram}, and the saturation curve moves
the mist from \SI{316}{\kelvin} down to \SI{309}{\kelvin}."*

**Reproduce:** `make bag-state`, section "sec:needle\_through\_fog: the plug as
a heat sink".

---

### B2. The Jupiter leg has no mist for the plug to cool

Same paragraph. The plug's heat-sink argument is framed on the Jupiter
departure, and **on that leg it buys nothing**: from 122 K storage the waste
heat is 0.65 MJ/kg against 0.73 to melt, so the slug never finishes melting with
or without the plug. Both columns are dry.

The heat sink is worth something only from **Earth storage**, which is the case
the section already says every film mass is quoted at. The numbers in B1 are
that case. One clause is enough: *"From cold storage the slug never finishes
melting either way, so this is the Earth-storage case, which is the one the film
is quoted at throughout."*

---

### B3. The plug's share of the bill grew, and it is worth saying

**Now:** *"Over \SI{37.5}{\kilogram} that is \SI{27}{\mega\joule} of the pulse's
\SI{138}{\mega\joule} waste-heat bill removed before the boiling step ever sees
it."*

**The 27 MJ survives** — it is `37.5 × 0.73 = 27.4` and does not depend on the
leak. What changed is its **share**: it was 13% of the 211 MJ bill and is
**20%** of the 138 MJ one. Worth stating, because it makes the plug a better
heat sink than the text implies, not a worse one.

---

### B4. The plug lightens the bag by 2 kg, it does not add 0.2 kg

**Locate:** `grep -n "the plug described next moves it to" templateArxiv.tex`

**Now:** *"The shape factor moves the bag from \SI{4.9}{\kilogram} to
\SI{6.2}{\kilogram}, and the plug described next moves it to
\SI{6.4}{\kilogram}, which is \SI{3.0}{\percent} of the slug."*

The 4.9 → 6.2 half is right. The 6.4 is the same reversal as B1: it came from
the plug appearing to raise the vapour mass 5% and the mist 1 K.

**Should be:** the plug takes the 23 m column's pressure vessel from **6.2 kg to
4.2 kg**. And that crosses a threshold the section already establishes — 4.2 kg
is **below the 5.1 kg handling floor** at 12.7 µm over the column's 437 m².

**So the flown bag is handling-governed, not pressure-governed.** This is the
more interesting statement and it strengthens the section: on the cold leg
storage retires the pressure vessel, and on the warm leg **the plug does**.

**Suggested replacement:** *"The shape factor moves the bag from
\SI{4.9}{\kilogram} to \SI{6.2}{\kilogram}, and the plug described next takes it
back down to \SI{4.2}{\kilogram} — under the \SI{5.1}{\kilogram} handling floor
of \autoref{tab:axial_bag}, so the flown column is sized by handling rather than
by pressure at both ends of the burn."*

**Note for whoever applies this:** `tab:axial_bag`'s Pressure column is the
**no-plug** case and should stay that way, since its caption ties it to
`tab:bag_state`'s vapour state and that table has no plug in it. Only the prose
changes.

---

### B5. The polyethylene paragraph quotes two superseded mist temperatures

**Locate:** `grep -n "normally disqualified for the one reason" templateArxiv.tex`

**Now:** *"The mist runs at \SI{306}{\kelvin} on the Jupiter departure and
\SI{328}{\kelvin} on the warmer Earth boost."*

Both are pre-solve. There is **no mist at all** on the Jupiter departure, and
the Earth boost runs at **316 K** without the plug and **309 K** with it.

**Suggested replacement:** *"The mist runs at \SI{316}{\kelvin} on the Earth
boost, and on the Jupiter departure the slug never finishes melting, so there is
no mist to speak of."*

---

### B6. The self-regulation sentence quotes 306 K

**Locate:** `grep -n "answers with a larger vapor fraction at the same" templateArxiv.tex`

**Now:** *"the bag answers with a larger vapor fraction at the same
\SI{306}{\kelvin}"* → **316 K**. The paragraph is about `tab:bag_state`, so it
should carry that table's own Earth-storage temperature.

---

### B7. The 23 m column costs 1.3 kg more film, not 0.8 kg

**Locate:** `grep -n "for a fifth more conductor and" templateArxiv.tex`

0.8 kg was the sphere-to-23 m delta on the superseded Jupiter column
(2.8 → 3.6 kg). On the published Earth column it is **4.9 → 6.2 kg = 1.3 kg**.

---

### B8. The polyethylene margin follows the mist temperature

**Now:** *"Polyethylene keeps \SI{116}{\kelvin} of margin below its melting
point."* That is 423 − 307 from B1's superseded 307 K.

**Should be:** 423 − 316 = **107 K** without the plug, 423 − 309 = **114 K**
with it. The paragraph is about the plug, so **114 K** is the figure it wants.

The other margin claim in this area — 95 K, from 423 − 328 — belongs to B5's
sentence and becomes **107 K** on the same arithmetic.

---

## C. One thing the rerun found that is not an edit

**`tab:bag_state`'s warming row is an output of its own mist row.** "Warming and
melting the slug up to liquid" warms liquid water up to whatever temperature the
mist settles at, and the published 0.73 and 0.21 MJ/kg were closed against
**306 K and 328 K** — the superseded mist. Read back, they imply 4.10 and
4.14 kJ/kg/K, which is liquid water; the temperatures are the part that moved.

Solving the two rows together instead gives:

| | printed | loop closed |
| --- | ---: | ---: |
| Earth warming | 0.21 MJ/kg | 0.168 |
| Earth `x` | 0.185 | 0.202 |
| Earth mist | 316.3 K | 318.2 K |
| Earth film | 4.85 kg | 5.35 kg |
| Jupiter | no vapour | slush at 275.5 K, 3.8 kg of vapour, 0.4 kg of film |

**Recommendation: do not apply this.** The paper prints the warming row as an
input and the reader can follow it by hand, which is the point of cutting a loop
in the first place. Both gaps are under a kilogram of film, and neither
overturns "cold storage removes the pressure vessel", which is what the section
rests on. It is recorded because the 0.73 row is the last place in this cascade
where a superseded number is still doing work, and because the Jupiter column's
"no vapour at all" is a slightly stronger claim than the model supports — the
honest version is "a slush at the freezing point holding well under a kilogram
of film".

If the paper ever wants to state that: the Jupiter bag holds **0.7 kPa** and
**0.4 kg** of pressure film, against a 2.4–10.0 kg handling floor, so the
conclusion is unchanged and the sentence merely gets more defensible.

**Reproduce:** `make bag-converge`, section "Cut 4".

---

## D. Do not change

- **`tab:bag_state`.** Every cell reproduces. See section A.
- **`tab:bag_sizing`.** All 30 cells reproduce, including the Coldest mist
  column at 405/346/316/290/259/243 K — that column was already updated to the
  Earth-storage fraction and is right. (The 28 m row computes 242.5 K, which
  rounds either way and is outside the usable band the caption already flags.)
- **`tab:axial_bag`.** All 35 cells reproduce, Pressure column included. It is
  the no-plug case by construction and should stay so — see B4.
- **The 138 MJ bill.** Computes 137.5 MJ. The correction already applied at
  `7c56c5d` is right.
- **The cold-leg headroom claim.** 2.54% against a 2.93% onset, 15% of
  headroom, hot legs clearing 6.6–10×. Untouched by any of this.
- **The plug itself, its 37.5 kg sizing, and the needle-through-fog
  conclusion.** All correct. B1–B4 change what the plug is worth, not whether it
  is there.

---

## E. If something here looks wrong

Say so rather than working around it. The numbers here come from a reproduction
pass, not a re-derivation, so any that disagree with the paper by a small amount
are more likely to be mine than the paper's; any that disagree by a factor, or
by a **sign**, are more likely to be the paper's — B1 is exactly that case, and
it was found because two sentences in the same paragraph disagreed about which
way a heat sink pushes.
