# Deferred back to the companion repos

Things found while applying `paper_corrections.md` that could not be settled in
the paper repository. Each says what is owed, which repo owes it, what the paper
did in the meantime — and, now, **how it was resolved and what the paper should
do about it**.

**This document is written to be copied into
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
and worked there,** exactly like `paper_corrections.md` before it. Section
["What the paper should change"](#what-the-paper-should-change) is self-contained:
every number needed to make every edit is stated in it, because an agent in the
paper repository cannot run the code that produced them.

Raised 2026-08-26 (D1–D9). Worked 2026-08-26 in
`katzseth22202/aim_is_all_you_need`.

---

## Status

| # | owed by | status | does the paper change? |
| --- | --- | --- | --- |
| D1 | `aim_is_all_you_need` | [x] **closed, won't fix** — the diagonal is the right object | no |
| D2 | `aim_is_all_you_need` | [x] **fixed** in `docs/paper_corrections.md` | no — the paper already prints 0.731 |
| D3 | `aim_is_all_you_need` | [x] **fixed** — warning lifted into both wrappers, column pinned by test | **yes, one clause** (P2) |
| D4 | `aim_is_all_you_need` | [x] **modelled** — `handling_film_mass()` in `src/bag_state.py` | **yes, a requote** (P3) |
| D5 | `aim_is_all_you_need` | [x] **diagnosed and closed** — a rounded input, not a wrong source | no — the paper already prints 0.069 / 7.9 |
| D6 | `aim_is_all_you_need` | [x] **recorded** — both columns below | no |
| D7 | `aim_is_all_you_need` | [x] **fixed** in `docs/paper_corrections.md` | no — landed as 50.9 |
| D8 | `puffsat_impact_simulation` | [x] **regenerated** — the document's row is stale, the paper's range is not | no |
| D9 | `puffsat_impact_simulation` + `aim_is_all_you_need` | [x] **solved properly** — the cliff is 2450 K, not 2570 K | **yes, a number** (P1) |

Three paper edits fall out, **P1 to P3 below**. Nothing else here needs the
paper touched; the rest is recorded so the same ground is not re-walked.

---

## What the paper should change

### P1. The conductivity cliff is at ~2450 K, and the gap above it is ~1350 K

**Priority: do this one first.** It is the only item here where the paper
currently prints a wrong number.

**Locate:** `grep -n "falls to 1 near" templateArxiv.tex`

**Now:** "At the stated $vL$, $Rm$ falls to 1 near \SI{2570}{\kelvin} … The leak
limit above it sits at \SI{3800}{\kelvin}, some twelve hundred kelvin earlier".

**Should be:** **\SI{2450}{\kelvin}**, and **"some thirteen hundred kelvin
earlier"** (3800 − 2450 = 1350).

**Why:** 2570 was obtained by log-interpolating the six tabulated conductivities
of `tab:seed_window`, which sample every 1000 K while `sigma` climbs 60× between
the first two. The shape of the steepest part of the curve is simply not in that
table, and interpolating it guesses high by about 120 K. The crossing is an
*output of the conductivity model*, and that model solves it directly:
`puffsat_impact_simulation` @ `0216a09`,
`puffsat.conductivity.cliff_temperature(rho=0.3229, x_k=0.01, v_l=7.4e4)`, which
bisects `Rm(T) = 1` on the continuous `sigma`. Run there 2026-08-26:

| `v L` [m²/s] | 1.81e4 (retired) | 5.5e4 | **7.4e4 (stated)** | 9.7e4 |
| --- | ---: | ---: | ---: | ---: |
| cliff, solved | 2859 K | 2524 K | **2450 K** | 2386 K |
| cliff, interpolated | 2911 K | 2640 K | 2568 K | 2502 K |

**The argument gets slightly stronger, not weaker.** B3's point was that the
leak binds well before the field loses grip; the gap is 1350 K rather than 1200.

**If the paper wants the sensitivity in one clause:** across the whole solved
`v L` band of 5.5e4–9.7e4 the cliff moves only 2386–2524 K, so nothing rests on
the middle of the band being exactly right.

**Reproduce:** `make plume-state` in `aim_is_all_you_need` (prints the solved
crossing beside the interpolated one), or
`make analysis-conductivity` in `puffsat_impact_simulation` after `REF_V_L` is
moved off 1.81e4 — see D9 below.

---

### P2. Say which elasticity the plate column is quoted at

**Locate:** `grep -n "tab:two_leg_growth" templateArxiv.tex`, then the paragraph
around the plate comparison C1 rewrote.

**Now:** the plate column is quoted without an `f`.

**Should be:** add a clause — the plate column is priced at the plate's full
measured elasticity **`f` = 0.818**, not at the 0.800 the companion target
defaults to.

**Why:** both are defensible and they answer different questions, but they do
not give the same table: at `eta_geom` = 0.9 the plate's growth is 4.244e5 at
`f` = 0.818 against 3.547e5 at 0.800, a fifth larger, and the nozzle-over-plate
ratio moves 22× against 27×. A reader who reruns the target as committed gets
the second and has no way to tell which the paper meant. At `f` = 0.818 the chain's growth is 1.464e6 /
4.244e5 / 7.486e4 at `eta_geom` = 1.0 / 0.9 / 0.8, which is exactly the column
C1 printed (1.46e6 / 4.24e5 / 7.49e4).

**Reproduce:** `make two-wave`; pinned by
`tests/test_two_wave_growth.py::test_the_plate_column_reproduces_at_the_measured_elasticity`.

---

### P3. Requote the bag film on the handling floor, not on the pressure vessel

**Locate:** `grep -n "bag film\|tab:axial_bag" templateArxiv.tex`, and the
shape-factor sentence that reads "1.7% of the slug".

**Now:** after A3, the film is quoted from Earth storage — the only case left
that still boils — because `eq:bag_film_mass` returns **0 kg** from cold
storage, where the solved leak boils nothing.

**Should be:** state that the film is `max(pressure vessel, handling floor)`,
and quote the floor. For the flown 23 m column, at a 12.7 µm film:

| length | area | 6 µm | **12.7 µm** | 25 µm | pressure (Earth, cold leg) | governs |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10.8 m | 366 m² | 2.0 kg | **4.3 kg** | 8.4 kg | 4.9 kg | pressure |
| 16.0 m | 364 m² | 2.0 kg | **4.3 kg** | 8.4 kg | 5.9 kg | pressure |
| **23.0 m** | 437 m² | 2.4 kg | **5.1 kg** | 10.0 kg | 6.2 kg | pressure |
| 32.0 m | 515 m² | 2.8 kg | **6.0 kg** | 11.8 kg | 6.3 kg | pressure |
| 50.0 m | 644 m² | 3.6 kg | **7.5 kg** | 14.8 kg | 6.4 kg | handling |

The pressure column is Earth storage on the coldest leg -- 45.58 km/s, the only
case the solved leak still boils anything in. **From cold storage that column is
0 kg in every row**, and the 12.7 µm column is then the whole of the bag.

**Why it matters beyond one number.** The two models scale differently, and the
paper currently uses the wrong one for a bag that holds no pressure:

- A **pressure vessel**'s film mass is independent of bag radius and rises only
  through the shape factor `F`, which saturates at 2.0. From 16 m to 50 m, `F`
  rises 8%.
- A **handling-gauge** bag pays for *area*. A capsule's area is exactly
  `2 pi r L`, and with the volume fixed `r ∝ 1/sqrt(L)`, so area `∝ sqrt(L)` and
  does not saturate. From 16 m to 50 m the area rises **77%**.

So stretching the bag into a longer column costs film much faster than `F`
suggests, which *strengthens* the launch-envelope argument for 23 m over 50 m
rather than weakening it.

**Consequential rewrites in the same passages:** "1.7% of the slug" becomes
**about 2.4%** (5.1 kg of 213 kg) at the quoted gauge, or 1.1–4.7% across the
gauge band. The polyethylene-against-polyester choice and the 4.9 kPa liner
argument both survive — but they are now arguments about a *membrane*, not about
a vessel, so whichever of them is phrased in terms of hoop stress needs
rephrasing in terms of gauge and handling.

**The one thing to check before printing:** the 12.7 µm gauge is anchored on
Echo 1, which flew a 30 m sphere of half-mil metallised PET in 1960, packed,
deployed and inflated on orbit. **That claim needs a citation the paper repo
verifies itself** — the arithmetic above does not depend on it, only the choice
of gauge does. If no citation can be verified, print the 6–25 µm band instead of
a single gauge; every conclusion here holds across it.

**What is still not modelled:** seams, ripstop, metallisation and inflation
hardware are all excluded, so this is a floor, not a bag design.

**Reproduce:** `make bag-state`, section "D4: the handling floor under the film".

---

## The items, and how each was settled

### D1. `tab:two_leg_growth` has no tolled two-axis sweep

**Owed by:** `aim_is_all_you_need` · **Raised by:** A1

A1 asks for the swept axis of *both* growth tables to become `eta_geom`. Only
one of the two can be regenerated. `make two-wave` prints the full 8×4 tolled
grid behind `tab:space_mortgage_growth`, and that correction lands. `make
two-leg` does not print the matching 8×5 grid over `e_1` × `e_2`; what it prints
is a tolled *matched diagonal*, a single shared `eta_geom` on both legs, plus the
**untolled** `e_1` × `e_2` grids.

**CLOSED 2026-08-26 — won't fix, and the reason is physical rather than
budgetary.** The change is small: `_score()` already computes `jet1` and `jet2`
separately and multiplies both by one shared `eta_geom`, so making it a per-leg
pair is about forty lines through `price_chain_two_leg()` and the grid loop. It
should not be done. **One nozzle flies both legs**, so an 8×5 grid over two
independently swept geometric efficiencies is a table the design cannot occupy
anywhere off its own diagonal — it would add a dimension no one can locate the
vehicle in. The matched diagonal is not a fallback; it is the honest object.

If leg-to-leg capture really does differ — different closing speed, different
arrival radius, different plume — the answer is to **compute** `eta_geom` per leg
from `nozzle_geometry.py`'s snowplow sweep (ADR 0016, ledger item 11), not to
sweep it as a second free axis. That would be a new result, not a re-tabulation.

**Paper:** nothing. C1's replacement of the comparison with the matched diagonal
is the right treatment and has landed.

---

### D2. A1's `eta_chem` row was labelled with the wrong slug ratio

**Owed by:** `aim_is_all_you_need` (`docs/paper_corrections.md`) · **Raised by:** A1

A1 printed a row of `eta_chem` "at `k` = 8.5" whose 45.58 km/s entry was 0.754,
the `k` = 7.77 value — the tolled optimum rather than the flown ratio.

**FIXED 2026-08-26.** Confirmed by computation:

| `w` [km/s] | 45.58 | 56.53 | 61.83 | 65.13 | 75.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `chemistry_efficiency` at `k` = 8.5 | **0.7311** | 0.8351 | 0.8643 | 0.8786 | 0.9100 |
| at `k` = 7.77 | 0.7551 | 0.8489 | 0.8755 | 0.8886 | 0.9172 |

Only the first cell was wrong; the other four were already `k` = 8.5 values.
`docs/paper_corrections.md` in this repository now prints 0.731 with a note.

**Paper:** nothing — it landed 0.731, which is what the flown design pays. The
copy of `paper_corrections.md` in the paper repo still carries the old 0.754 in
its A1 table; it is superseded and worth striking so it is not re-applied.

---

### D3. C1's plate column is at `f` = 0.818, not the target's default 0.800

**Owed by:** `aim_is_all_you_need` · **Raised by:** C1

**RESOLVED 2026-08-26 while working C1**, and hardened since. The document was
right and the target's default is simply not what it quotes:
`price_chain(cycles, 1.0, 0.818, geometric_efficiency=eta_geom)` reproduces C1's
plate column exactly — 1.464e6 / 4.244e5 / 7.486e4 at `eta_geom` = 1.0 / 0.9 /
0.8, against the document's 1.46e6 / 4.24e5 / 7.49e4.

**The trap that cost the time is now documented and pinned.** `price_chain`'s
first positional argument is `recovery`, which derates from *outside* the
momentum debit, while `geometric_efficiency` derates from *inside* it. The tolled
grid holds `recovery` at 1.0 and sweeps only `geometric_efficiency`; passing
`eta_geom` to both charges it twice and silently returns a much smaller number —
7282 rather than 6.289e4 at `eta_geom` = 0.8 and the target's own `f` = 0.800.
Both figures verified 2026-08-26. Fixed by:

- lifting the warning out of `price_cycle`'s docstring, where it already was,
  into `price_chain` and `price_chain_two_leg`, where callers actually look;
- `test_double_charging_the_toll_silently_shrinks_the_chain` (fast) and
  `test_the_plate_column_reproduces_at_the_measured_elasticity` (slow), which
  pin both the trap and the published column.

**Paper:** P2 above — one clause naming the elasticity.

---

### D4. A3 zeroes the film everywhere, and section F does not say so

**Owed by:** `aim_is_all_you_need` · **Raised by:** A3

`eq:bag_film_mass` sizes a pressure vessel, so with the solved leak boiling
nothing from cold storage it returns zero *wherever* it is evaluated, not only in
`tab:bag_state`. Section F's "every one of its 20 cells reproduces exactly.
Nothing is owed" is true of reproduction and false once A3 lands. The interim
treatment requoted the bag at Earth's 278 K storage (every film mass ×1.725,
because the film goes as `F x T` and only `x T` moves) and presented cold storage
as removing the requirement outright. What was owed was a **handling-sized
membrane model**: with no pressure to hold, the real bag is not 0 kg.

**MODELLED 2026-08-26.** `src/bag_state.py` gains `bag_surface_area()`,
`handling_film_mass()` and `governing_film_mass()`, and `make bag-state` prints
the floor beside the pressure vessel. Numbers and their consequences are in P3
above. Two findings worth keeping separate from the requote:

1. **A capsule's membrane area is exactly `2 pi r L`** — the hemispherical caps
   put back precisely what shortening the cylinder removed — so with volume fixed
   the area goes as `sqrt(L)` and never saturates, while `F` saturates at 2.0.
   The film column of `tab:axial_bag` is therefore on the wrong scaling for a bag
   that holds no pressure.
2. **The two models are the same size at the flown geometry** (5.1 kg handling
   against 6.2 kg pressure at 23 m), which is why nothing downstream breaks --
   from Earth storage. From cold storage the pressure term is 0 kg and the floor
   is the whole answer.

**Paper:** P3 above.

---

### D5. A7 says two numbers reproduce that do not

**Owed by:** `aim_is_all_you_need` (`docs/paper_corrections.md`) · **Raised by:** A7

A7 closed with "194 K equilibrium, 0.064 Pa, 7.4 kg/m²/day …" while
`make cruise-thermal` gives 0.0688 Pa and 7.91 kg/m²/day — both about 7% low, in
the same direction, suggesting one input rather than two errors.

**DIAGNOSED AND CLOSED 2026-08-26. Neither source is wrong; the input was
rounded.** The paper's two figures are the Marti–Mauersberger correlation
evaluated at exactly **194.000 K**, and the solved balance sits at **194.42 K**:

| T | vapour pressure | free-evaporation rate |
| ---: | ---: | ---: |
| 194.000 K (rounded) | 0.0642 Pa | 7.40 kg/m²/day |
| **194.42 K (solved)** | **0.0688 Pa** | **7.91 kg/m²/day** |

`d ln P / dT` is `2663.5 ln 10 / T²`, about **16% per kelvin** here, so 0.42 K of
rounding is the whole 7%. Recorded in `vapour_pressure`'s docstring ("round the
answer, never the input"), printed by `make cruise-thermal`, and pinned by
`test_rounding_the_equilibrium_before_using_it_moves_the_answer_7_percent`.

**Paper:** nothing. It already prints 0.069 Pa, 7.9 kg/m²/day and "near 194 K",
which is self-consistent, and the five-day survival follows from the numbers
printed beside it (41.1 / 7.91 = 5.2 days).

---

### D6. B1's growth comparison is quoted at `eta_geom` = 1 without saying so

**Owed by:** `aim_is_all_you_need` (`docs/paper_corrections.md`) · **Raised by:** B1

B1 closed with "99.4% of achievable growth against 91.7%" for the compact
arrival against the wide one. Repricing on the flown chain reproduces the first
and very nearly the second, but only at `eta_geom` = 1:

| | `eta_geom` = 1 | `eta_geom` = 0.8 |
| --- | ---: | ---: |
| optimum `k` | 6.75 | 7.45 |
| `k` = 7.21, compact | **99.4%** | 99.7% |
| `k` = 8.69, full bore | **91.0%** | 93.4% |

Two things the sentence did not carry: the comparison is against a **full-bore**
arrival at `k` = 8.69, not the `k` = 8.60 it names (which is `r/R` = 0.8); and the
spread narrows as `eta_geom` falls, so the argument is strongest exactly where
the nozzle is most optimistic.

**RECORDED 2026-08-26 — no further work.** The paper's rounding to "91% against
99%" without attaching an `eta_geom` is the right treatment: the ordering holds
across the range and the precise gap does not.

**Paper:** nothing.

---

### D7. The paper's atomisation enthalpy was 50.4 MJ/kg, and nothing flagged it

**Owed by:** `aim_is_all_you_need` (`docs/paper_corrections.md`) · **Raised by:** B2

A2 and A1 both introduce `E_a` = 50.9 MJ/kg while B2 quoted the paper's own
"Breaking water into its atoms takes 50.4 MJ/kg" without asking for it to change,
so applying the corrections as written would have left the paper stating two
values for one quantity two paragraphs apart.

**FIXED 2026-08-26** in `docs/paper_corrections.md`, where B2 now says plainly
that the quoted sentence is itself an edit. 50.9 is right:
`plume_thermal.WATER_ATOMISATION_ENTHALPY` is 917 kJ/mol, and 917 / 18.015 g/mol
= 50.90 MJ/kg; 50.4 implies 908 kJ/mol, 1.8% low.

**Paper:** nothing — it landed unified at 50.9 MJ/kg, stated as 917 kJ/mol so the
conversion is checkable, cited to `crc_handbook`. Verified present at four sites
(`sec:watering_it_down`, the ice-versus-polyethylene paragraph, the nozzle
arithmetic, and `eq:eta_chem`). Both downstream claims survive: 59.8% ≈ "about
60%", and 9.0% ≈ "about nine percent".

---

### D8. B2's cold-leg row is divergence-sensitive and the document quotes one angle

**Owed by:** `puffsat_impact_simulation` · **Raised by:** B2

**REGENERATED 2026-08-26** by running `python -m puffsat.fireball` at
`0216a09`. The document's cold-leg row matches **no** current cell, and the
reason is not the angle alone — the sweep has a second axis the document does not
name, the recombination **partner**: `H` (an OH is always available, optimistic,
the shipped default) against `OH` (equilibrium OH, conservative).

| 45.58 km/s | `rho` at freeze | `T` | held | stranded |
| --- | ---: | ---: | ---: | ---: |
| document | 1.01e-2 | 3908 K | 91.7% | 47.7% |
| 15°, partner H | 4.60e-3 | 3803 K | 83.1% | 43.3% |
| **45°, partner H** | **1.13e-2** | **4292 K** | **90.0%** | **46.9%** |
| 60°, partner H | 1.78e-2 | 4667 K | 93.7% | 48.8% |
| any angle, partner OH | 1.87e-1 | 12 658 K | 99.9% | 52.0% |

The document's row sits between 15° and 45° in density and temperature but
between 45° and 60° in held fraction, so it is a **stale run** rather than an
unlabelled angle. What is owed in `puffsat_impact_simulation` is to replace that
row with the 45°/H one and label both axes on it.

**Paper:** nothing. Its printed ranges — 0.011–0.024 kg/m³, 90–100% held, 19–47%
stranded — are exactly the current 45°/partner-H column across all four legs, so
the text is right even though the document behind it is stale.

---

### D9. B3's conductivity cliff is quoted at the old `vL`

**Owed by:** `puffsat_impact_simulation` and `aim_is_all_you_need` ·
**Raised by:** B3

B3 asked the paper to state "`Rm` = 1 at ~2845 K", computed at the retired
`vL` = 1.81e4; A9 has the paper state `vL` = 7.4e4 from the solved expansion.
The interim treatment moved the paper to ~2570 K.

**SOLVED PROPERLY 2026-08-26, and 2570 was also wrong.** Both figures came from
interpolating the six tabulated conductivities rather than from the model that
produces them. The solved crossings are in P1 above; at the stated `vL` the cliff
is **2450 K**, and the interpolation error runs +52 to +118 K across the band.
The vendored `sigma` values themselves are fine — they agree with the model to
0.4% — so this is purely a tabulation being asked a question it cannot answer.

Changed here: `src/plume_state.py` gains `CLIFF_TEMPERATURE` (solved values, with
provenance and an explicit "do not interpolate this" note), `cliff_temperature()`
and, deliberately kept, `_interpolated_cliff()` so `make plume-state` prints the
wrong answer beside the right one and nobody re-derives it. Four tests pin it.

**Still owed by `puffsat_impact_simulation`:** `python/puffsat/conductivity.py`
has `REF_V_L = 1.81e4`, the retired expansion speed, so `make
analysis-conductivity` regenerates `tab:seed_window` and its cliff at a value the
paper no longer states. Move it to 7.4e4 (or take it as an argument and report
the 5.5e4–9.7e4 band), and the numbers this repository has vendored will follow
from a run there rather than from a paste.

**Paper:** P1 above.

---

## Provenance

Every companion figure quoted here was produced on 2026-08-26 by running the
code, not by recalling it:

| run | covers |
| --- | --- |
| `python -m src.plume_state` | P1, D9 |
| `python -m src.two_wave_growth` via `price_chain` on the flown chain | P2, D3 |
| `python -m src.bag_state` | P3, D4 |
| `python -m src.cruise_thermal` | D5 |
| `python -m src.two_wave_growth.chemistry_efficiency` | D2 |
| `puffsat.conductivity.cliff_temperature` @ `0216a09` | P1, D9 |
| `python -m puffsat.fireball` @ `0216a09` | D8 |

D6's table is the one exception: it is carried unchanged from B1's own
repricing, which raised the item, and was not re-run here because nothing in the
paper now depends on it.
