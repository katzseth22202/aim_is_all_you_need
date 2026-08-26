# Corrections owed to the paper

**This document is written to be copied into
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
and worked there.** It is self-contained: every number needed to make every edit
is stated here, because an agent in the paper repository cannot run the code
that produced them.

Generated 2026-08-26 from a full reproduction pass over the paper's computed
tables, in `katzseth22202/aim_is_all_you_need`.

---

## Before you start

### The three repositories

| repo | holds | can you run it from the paper repo? |
| --- | --- | --- |
| `Balloon-Pulse-Propulsion` | `templateArxiv.tex`, `references.bib` | yes -- this is where you are |
| `aim_is_all_you_need` | trajectories, growth chain, bag and nozzle models | no |
| `puffsat_impact_simulation` | EOS, expansion, opacity, hydro solvers | no |

**Every number in this document is already stated in full.** The `make` targets
named under "Reproduce" are provenance, not instructions — you are not expected
to run them. If a number here seems wrong, say so rather than recomputing it.

### The ground rule

**The paper is not the source of truth.** Seth, 2026-08-26: *"The paper is not a
source of truth. If we should modify the paper based on calculations, let's do
that."* Where a calculation disagrees with the text, the calculation wins and
the text changes. Several of these corrections replace claims that were argued
rather than computed.

### What you can verify here

- **That the claim exists and reads as quoted.** Locate every claim by grepping
  its quoted wording; line numbers are not given because they drift.
- **That the paper still builds.** `./build.sh --clean --quiet`.
- **That every `\cite` key resolves.** Check `references.bib` before adding one.
- **Internal consistency.** Several corrections below were found precisely
  because two numbers in the same paragraph disagreed — that check is available
  to you and is worth repeating as you edit.

### What you cannot verify here

Any physical quantity. If an edit seems to need a number this document does not
give, stop and ask rather than deriving one.

---

## Priority

**Do these first — they are wrong, not merely unclear.** Each is an arithmetic
or factual error, and two of them are self-contradictory within the paper:

1. **A7** — ice rod areal density (55 → 41 kg/m²); contradicts the next sentence
2. **A5** — `tab:bag_state` leak row; the row's own arithmetic disagrees
3. **A9** — `tab:seed_window`'s `Rm` column cannot be reproduced at all
4. **A3** — the bag's field leak, and the bag film that follows from it

**Then the numbers that move:** A2, A4, A10, A1.

**Then the arguments:** B1, B2, B3.

**Then framing and mechanism:** C1, C2, C3, A6, A8, A11.

**Then citations:** section D, and the reproduction lines in section E.

---

## A. Numbers that are wrong or that move

### A1. Jet-efficiency axes are `eta_geom`, not `e`

**Locate:** `grep -n "tab:space_mortgage_growth\|tab:two_leg_growth" templateArxiv.tex`

**Now:** both growth tables sweep a bare `e` (recovery).

**Should be:** relabel the swept axis **`\eta_{\mathrm{geom}}`**, and state that
the jet efficiency actually flown is `eta_jet = eta_chem * eta_geom`.

**Why:** `sec:jet_efficiency` already defines `eta_jet^2` to include "frozen
ionization or dissociation energy". That contribution is now *computed* rather
than swept, so the axis is no longer the whole of `eta_jet`. Charging the
chemistry as a separate energy term instead would double-count.

**The ceiling to show beside the axis** — `eta_chem` at `k = 8.5`, which is what
caps `eta_jet`:

| `w` [km/s] | 45.58 | 56.53 | 61.83 | 65.13 | 75.00 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `eta_chem` | 0.754 | 0.835 | 0.864 | 0.879 | 0.910 |

**Closed form**, if the paper wants to print one:

    eta_chem = sqrt(1 - 2 E_a (1+k) / w^2),   E_a = 50.9 MJ/kg

**Reproduce:** `make two-wave`, `make two-leg` in `aim_is_all_you_need`.

---

### A2. The gas expands at 14.1 km/s, not 17

**Locate:** `grep -n "sqrt{2u}" templateArxiv.tex`

**Now:** "a gas converting all of that to directed motion expands at
$\sqrt{2u} = \SI{17}{\kilo\meter\per\second}$" (`sec:two_leg_nozzle`).

**Should be:** **`\SI{14.1}{\kilo\meter\per\second}`**, a 19% reduction.

**Why:** of the 150 MJ/kg dissipated at that closing speed, 50.9 MJ/kg is spent
atomising the water and does not come back (see B2). The gas converts
99.6 MJ/kg, not 150.

**Knock-on in the same sentence:** "so material reaches the field at roughly
\SIrange{11}{23}{\kilo\meter\per\second}" becomes roughly **8 to 20 km/s**
(5.9 km/s of drift ± 14.1).

---

### A3. The bag's field leak is 0.11–2.54%, not 4.4% — and the bag film goes to zero

**Locate:** `grep -n "Field leaking through the plume" templateArxiv.tex`

**Now:** `tab:bag_state` row "Field leaking through the plume (4.4\% of
\SI{20.8}{\mega\joule\per\kilogram})", giving \SI{0.89}{\mega\joule}.

**Should be:** the residence-weighted `1/Rm` over the solved cooling history:

| `w` [km/s] | 75 | 65 | 56.53 | 45.58 |
| ---: | ---: | ---: | ---: | ---: |
| leak | **0.11%** | **0.17%** | **0.29%** | **2.54%** |

**The consequence is larger than the row, and it is the point.** At those leak
fractions the waste heat no longer finishes melting the slug from 122 K storage,
so **nothing boils, the bag holds no pressure, and the film is not a mass item
at all** — 0 kg rather than 2.8 kg. The bag becomes a containment membrane sized
by handling rather than a pressure vessel.

**Full replacement cascade, per kg of slug, Jupiter 122 K storage:**

| row | now | should be (45.58 km/s, the worst leg) |
| --- | ---: | ---: |
| blackbody intercept | 0.10 MJ | 0.10 MJ |
| field leak | 0.89 MJ | **0.55 MJ** |
| waste heat | 0.99 MJ | **0.65 MJ** |
| warming and melting | −0.73 MJ | −0.73 MJ |
| left to boil | 0.26 MJ | **0 — the slug does not finish melting** |
| vapour fraction `x` | 0.11 | **0** |
| bag film | 1.3%, 2.8 kg | **0 kg** |

**Two things worth adding as text**, because they are not obvious from the
table:

- **The margin is thin on one leg only.** The cold leg's 2.54% sits against a
  2.93% onset — **15% of margin** — while the hot legs clear theirs 9–10× over.
  If the leak model moves at all it moves there.
- **Cold storage is doing work the bag would otherwise do.** From Earth's 278 K
  the same leg *does* boil (`x` = 0.18) and needs a **4.9 kg** film. This makes
  the paper's existing "scheduling choice rather than a hardware one" framing
  more load-bearing than it currently reads.

**Reproduce:** `make bag-state`.

---

### A4. The nozzle mass floor is ~8 t, not 3.7 t

**Locate:** `grep -n "virial floor of the paragraphs above turns that into" templateArxiv.tex`

**Now:** "the virial floor of the paragraphs above turns that into
\SIrange{3.7}{11}{\tonne} and \SIrange{10}{30}{\tonne} respectively", and later
"That is a tenth to a third of a \SI{100}{\tonne} craft".

**Note what is already right:** the same passage says *"Neither figure is a
design. The two-term mass model of \autoref{sec:minimum_nozzle}, virial
structure plus conductor, is what has to settle it, and it has not been run at
this pulse."* **It has now been run.** The virial term reproduces exactly; the
conductor term is what was missing.

**Should be:**

| `E_B` | field | structure (virial) | **conductor** | **two-term total** |
| ---: | ---: | ---: | ---: | ---: |
| 4.43 GJ | 4.11 T | 3.7–11.1 t | **4.4 t** | **8.0–15.4 t** |
| 12.2 GJ | 6.80 T | 10.1–30.4 t | **7.2 t** | **17.3–37.6 t** |

- "a tenth to a third of a \SI{100}{\tonne} craft" becomes **8% to 38%**.
- "Call it tens of tonnes" **survives unchanged**.
- **The optimistic end is not reachable**: the conductor alone exceeds the 3.7 t
  structure floor.

**Also owed, and it is a real gap:** the paper never states a **tape operating
current**, and the conductor term is exactly inversely proportional to it —
300 A gives 7.3 t of tape, 500 A gives 4.4 t, 1000 A gives 2.2 t. That is a
bigger lever than the structure's pre-compression band, which the paper does
discuss. Name an operating current.

**Conductor model, if the paper wants to print it:** `NI = B l / mu_0`
ampere-turns, each turn `2 pi r` of tape, at 1.5 g/m for thin-substrate REBCO.
Substituting `eq:bore_from_length` makes the mass run as `B sqrt(V l / pi)` —
which the paper already derives at `sec:needle_through_fog`.

**Reproduce:** `nozzle_geometry.two_term_nozzle_mass`.

---

### A5. `tab:bag_state`'s leak row contradicts its own arithmetic

**Locate:** same row as A3.

**Now:** the row reads "4.4\% of \SI{20.8}{\mega\joule\per\kilogram}" and prints
**\SI{0.89}{\mega\joule}**. But `0.044 × 20.8 = 0.915`, and `0.89 / 20.8 =
4.28%`.

**Should be:** consistent. **The printed 0.89 is the load-bearing one** — every
downstream digit in the table (waste heat 0.99, left to boil 0.26, `x` = 0.11,
film 2.8 kg) follows from 0.89 and not from 0.915.

**If A3 is applied this row is replaced wholesale** and this correction lapses.
It is recorded separately because it is an independent error and because it
shows the table was assembled from a spreadsheet whose inputs are not all
printed.

---

### A6. `sec:two_leg_nozzle`'s mirror passage cannot be checked as written

**Locate:** `grep -n "the wall sees" templateArxiv.tex`

**Now:** "Their ratio is $Mv^2/(\gamma-1)E$ … The ratio is 6.7, the wall sees
\SI{1.26}{\giga\pascal}, and it would take \SI{56}{\tesla}."

**The numbers are right** — both cases reproduce to 1% (ship end 6.67 /
1.276 GPa / 56.6 T; throat end 1.17 / 23.1 MPa / 7.6 T).

**The problem is that two inputs are not stated**, and both are needed:

- **`\gamma = 1.2`**, not the monatomic 5/3. Back-solves from the ratio of 6.7.
  Worth a clause of its own: dissociation and ionisation are absorbing energy
  that a monatomic gas would put into translation.
- **The closing speed is \SI{56}{\kilo\meter\per\second}.** Recoverable only
  from the momentum the two cases share: `62.5 × 22.4 = 238 × 5.9 = 1400`
  kg·km/s.

**Also worth adding — the mechanism, which the passage states only as an
outcome:** a throat-end plug lets the fireball snowplow the whole column first,
which dissipates **more** energy (35.1 GJ against 23.5) yet needs **a seventh of
the field**, because the ram term falls faster than the static term rises. That
is the non-obvious part and it is what decides the plug position.

**Reproduce:** `nozzle_geometry.mirror_stagnation`.

---

### A7. The ice rod's areal density is 41 kg/m², not 55

**Locate:** `grep -n "carries about" templateArxiv.tex` (`sec:needle_through_fog`)

**Now:** "A \SI{25}{\kilogram} rod carries about \SI{55}{\kilogram\per\square\meter}.
It would be gone in a week."

**Should be:** **\SI{41}{\kilogram\per\square\meter}**, and **gone in about five
days**.

**Why, and the paper proves it itself:** a 25 kg ice rod at 0.1 m radius is
0.868 m long — the paper's own \SI{0.87}{\meter}, stated two paragraphs earlier
— with **0.608 m²** of total surface. `25 / 0.608 = 41.1`.

**The next sentence is the proof.** It says the shaded cruise loses "about
\SI{1.4}{\percent} of the rod", and 0.58 kg/m² comes to 1.4% of 25 kg **only if
the area is 0.608 m²**. So the two figures are inconsistent two sentences apart,
and the 55 is the one that does not fit.

**The conclusion is unaffected and slightly stronger:** under a week rather than
a week. Everything else in the paragraph reproduces — 194 K equilibrium,
0.064 Pa, 7.4 kg/m²/day, 0.58 kg/m², 1.4%.

**Reproduce:** `make cruise-thermal`.

---

### A8. `tab:bag_sizing`'s mist column has no stated model

**Locate:** `grep -n "Coldest mist" templateArxiv.tex`

**Now:** a column running 372 / 332 / 306 / 281 / 250 / 228 K, with no statement
of how it is computed.

**Reproduced:** holding the vapour fraction at `tab:bag_state`'s 0.11 and
changing only the volume gives **333 / 306 / 282 / 253** across the four middle
rows — within 1.5 K — but **388 K at 1.8 m against the printed 372** and **237 K
at 28 m against 228**.

**Likely cause, physical rather than arithmetic:** the paper notes the plume
"stops being optically thick past about \SI{7}{\meter}", and a bag that has gone
thin reabsorbs less of its own radiation, so the vapour fraction should *fall*
below 0.11 at large radii rather than hold. That fixes the 28 m row. The 1.8 m
row deviates the other way and is not explained by it.

**Should be:** state the model. Either state the optical-depth correction or
mark the two outer rows as outside the band it is calibrated for.

**Not load-bearing:** the paper's own usable band is 3.5–7 m, entirely inside
the region that reproduces.

**Note:** if A3 is applied, `x` is no longer 0.11 and this column needs
regenerating rather than annotating. Do A3 first and then revisit.

---

### A9. `tab:seed_window`'s `Rm` column is not one expansion

**Locate:** `grep -n "tab:seed_window" templateArxiv.tex`

**Now:** an `Rm` column running 0.1 / 9.2 / 76.5 / 238 / 400 / 361 at
2000–15 000 K, with **neither `sigma` nor `v` nor `L` stated**.

**The problem:** `Rm = mu_0 sigma v L`, and `v` and `L` enter only as their
product. Taking `sigma(T)` from the solved conductivity model, the `v L` each
row implies is:

| `T` [K] | 2000 | 3000 | 4000 | 5000 | 6000 | 15000 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sigma` [S/m] | 1.05 | 63.3 | 286 | 492 | 600 | 6987 |
| implied `v L` | 7.6e4 | 1.2e5 | 2.1e5 | 3.9e5 | **5.3e5** | **4.1e4** |

**A factor of 12.9, and non-monotone** — rising through the window then
collapsing at the hot end. If the column were one flow sampled at several
temperatures, every row would imply the same product. **No single `v L`
reproduces it.**

**Should be:** regenerated from a stated `sigma(T)` at a stated `v L`. The
solved expansion now supplies one: **`v L` = 5.5e4 to 9.7e4 m²/s**, the first
time either quantity has been an output rather than a guess. At the middle of
that band (7.4e4) the column becomes:

| `T` [K] | 2000 | 3000 | 4000 | 5000 | 6000 | 15000 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Rm` | 0.10 | 5.9 | 26.6 | 45.8 | 55.8 | 650 |
| leak `~1/Rm` | 100% | 17% | 3.8% | 2.2% | 1.8% | 0.15% |

**Why this reaches past the table:** the paper itself says this column "prices
the field leak that \autoref{tab:bag_state} charges the slug for … Those are the
same number". So an `Rm` column that cannot be reproduced is also a leak
fraction that cannot be — which is exactly what A3 replaces.

**The potassium-ionised column also moves**, though less consequentially:
0.02% / 2.0% / 18.5% / 56.3% / 85.6% / 98.8% against the printed 0.01% / 1.1% /
11.0% / 38.0% / 70.1% / 99.9%.

**Reproduce:** `make plume-state`.

---

### A10. The radiated-loss column holds the plume at 15 000 K across the whole burn

**Locate:** `grep -n "Energy radiated" templateArxiv.tex`

**Now:** `tab:bag_sizing`'s "Energy radiated" column, and the 3.5–7 m usable band
derived from it, are worked at 15 000 K.

**Should be:** the plume runs 15 165–26 521 K across the burn, and radiative
loss goes as `T^4`. At the flown 5.4 m bag:

| `w` [km/s] | `T` solved | tabulated | actual | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 75 | 26 521 K | 1.17% | **3.64%** | **3.1×** |
| 65 | 22 685 K | 1.17% | 2.58% | 2.2× |
| 56.53 | 19 708 K | 1.17% | 1.93% | 1.7× |
| 45.58 | 15 165 K | 1.17% | 1.03% | 0.9× |

**Stated fairly:** a factor of three in a term that is small either way. It does
not move the bag out of the optically thick band and does not reach the film.
**The correction is to say the column is a cold-pulse figure rather than a
burn-wide one** — the cut was defensible where it was made and misleading where
it was carried across the burn.

**Reproduce:** `make bag-converge`.

---

### A11. The optical-thickness convention is not stated

**Locate:** `grep -n "optically thick past about" templateArxiv.tex`

**Now:** "the plume stops being optically thick past about \SI{7}{\meter}".

**The problem:** `kappa rho r = 1` gives **5.04 m**, which would make the flown
5.4 m bag already thin. `kappa rho D = 1`, across the **diameter**, gives
**7.13 m** and reproduces the stated limit.

**Should be:** say which. The flown design sits at `tau = 1.74` on the diameter
convention and below 1 on the radius convention, so as written a reader cannot
tell whether the baseline is inside its own band.

---

## B. Arguments that reach the right answer by the wrong route

### B1. The aperture argument is about neutrals, not open area

**Locate:** `grep -n "not a mirror at all" templateArxiv.tex`

**Now:** "A head-on nozzle is a chamber with a hole in the end the projectile
came through … **Anything left open leaks it.**"

**The problem:** that is an open-**area** argument, and it does not hold for a
*magnetic* mirror, whose leak is a loss-cone property set by the **mirror
ratio** rather than by how much of the end wall is physically open.

**Should be:** the mechanism is the **un-ionised fraction**, which the field
cannot steer and which does leave ballistically through any physical hole. At
the cold leg that is most of the plume (`f` = 0.06 at 15 165 K).

**The conclusion — keep the projectile compact — survives unchanged**, and there
is now a second, independent argument for it the paper does not use: **the
compact arrival also wins on chain growth.** Because the snowplow front is
self-widening (see B4 below), a 0.15 m arrival still sweeps most of the bag and
delivers `k` = **7.21**, which sits inside the tolled optimum band of 6.75–7.77,
where the 8.60 a wide arrival delivers overshoots it — 99.4% of achievable
growth against 91.7%.

---

### B2. The recombination argument is checked at the wrong station

**Locate:** `grep -n "They do, and quickly" templateArxiv.tex`

**Now:** "Breaking water into its atoms takes \SI{50.4}{\mega\joule\per\kilogram}
… It comes back only if the atoms find each other again as the plume cools and
expands. **They do, and quickly.** … The energy is a loan rather than a cost."

**The problem is not the rate, it is the station.** At 1 kg/m³ three-body
recombination really is ~0.01 µs against a hundreds-of-µs expansion. But at
1 kg/m³ the plume is fully atomised and has **nothing to give back yet**. It
only has something to return once it has cooled, and by then it sits at
~0.02 kg/m³ and **past the nozzle lip**, where the local expansion clock steps
down 8× in one step.

**Should be:** **the loan defaults.** The freeze is at **1.1e-2 to 2.4e-2
kg/m³** with **90–100% of the store still held**:

| `w` [km/s] | `rho` at freeze | `T` | store still held | stranded |
| ---: | ---: | ---: | ---: | ---: |
| 75 | 2.35e-2 | 16 063 K | 100% | 19.2% of the dissipated budget |
| 56.53 | 2.10e-2 | 11 271 K | 100% | 33.9% |
| 45.58 | 1.01e-2 | 3 908 K | 91.7% | 47.7% |

**The paper already flags the condition** — *"we have not computed the density a
real pulse produces"* — so the correction is that it has now been computed and
the answer is the unfavourable one. **The paper's own 0.01 kg/m³ threshold is
where the cold leg lands, to within 1%**, which reads as confirmation of where
the threshold was drawn.

**This is what A1 and A2 follow from.** It is not a new energy term: it is a
computed floor under `eta_jet`, one of the five contributions
`sec:jet_efficiency` already names.

**Also affects `sec:two_leg_nozzle`:** "Both tables also assume the dissociation
energy comes back during expansion, which \autoref{sec:watering_it_down} argues
for and which fails if the fireball is thinner than
\SI{0.01}{\kilogram\per\cubic\meter}" — that assumption now **fails**, and the
sentence should say so rather than conditioning on it.

---

### B3. `tab:seed_window`'s `Rm` and `tab:bag_state`'s leak are the same quantity

**Locate:** `grep -n "leak schedule written backwards" templateArxiv.tex`

**Status: the paper already makes this connection**, and makes it well —
*"Those are the same number, and the paper spent a while treating them as two."*

**What is still owed** is the consequence, in two parts:

- **The conductivity cliff should be stated where the leak is.** `Rm` = 1 at
  **~2845 K**, below which the field diffuses out faster than one expansion.
  The paper's revised window floor of ~3300 K sits just above it.
- **The two tables should now carry the same number.** They do not: A9 shows the
  `Rm` column is not reproducible and A3 replaces the leak row. Applying both
  makes the paper's own claim true, which it currently is not.

---

### B4. The snowplow front is self-widening, which the paper does not use

**Locate:** `grep -n "needle through fog" templateArxiv.tex`

**Status: nothing in the paper is wrong here.** This is an argument the paper
could make and does not, and it strengthens two existing conclusions.

**The finding:** the freshly shocked layer at the snowplow front takes `v²/2` of
specific energy, which at 45.58 km/s is **94 630 K**, whose sound speed is
**21.1 km/s** — very nearly half the closing speed, a **24.9° half-angle**. The
front widens fast enough to reach the bore wall within the first few metres of a
23.8 m column from any plausible arrival radius.

| arrival `r/R` | `k` if the front stays rigid | `k` self-widening | fills bore at |
| ---: | ---: | ---: | ---: |
| 0.05 (the 0.15 m aperture) | 0.02 | **7.21** | 6.2 m |
| 0.60 | 3.13 | 8.36 | 2.7 m |
| 1.00 | 8.69 | 8.69 | — |

**Cross-checked two ways:** venting the shocked layer sideways until its pressure
balances the cold cloud's ram pressure gives **1.6–1.9× faster** than the sound
speed, so the sound speed is the conservative estimate; and the shock
compression ratio, the one weakly-known input, moves it by only 9% across
2×–16×.

**What it buys the paper:** the "needle through fog" problem is real at the
*start* of the crossing and solves itself thereafter. The compact projectile the
paper insists on (rightly, see B1) does **not** pay a coupling penalty for being
compact.

---

## C. Framing written for arguments since retired

### C1. The `e1 ~ 0.6` crossover

**Locate:** `grep -n "crossover\|draw level" templateArxiv.tex`

**Now:** `sec:two_leg_nozzle` states the crossover against `f = 0.8`, so a
reader who believes the incumbent's number can reach the opposite conclusion
from the same tables.

**Should be:** the comparison no longer needs a matched-quality argument at all.
Compared at the same `eta_geom`, with the chemistry charged on whichever legs
carry a nozzle, and **granting the plate its full measured `f` = 0.818**:

| `eta_geom` | nozzle on both legs | plate on the growth push | ratio |
| ---: | ---: | ---: | ---: |
| 1.0 | 6.93e7 | 1.46e6 | **47×** |
| 0.9 | 9.52e6 | 4.24e5 | **22×** |
| 0.8 | 7.72e5 | 7.49e4 | **10×** |

**The nozzle wins outright against the incumbent's best measured number.** No
matched diagonal and no claim that 0.8 is indefensible.

**Note what does not change:** `STD_FUDGE_FACTOR = 0.8` is **correct as printed**
— `puffsat_impact_simulation` measures `f` = 0.817–0.820 on the heavy plate
across 45–63 km/s, and Seth confirmed 2026-08-26 that the heavy plate is what
the growth push flies. Do not weaken the plate's number.

**What the chemistry did take** is the size of the margin: 36.2× at matched 0.80
before the toll, 10.3× after. Worth stating rather than burying.

---

### C2. `eta_geom` is unmeasured and the text should say so

**Now:** nothing distinguishes the grounding of the two factors of `eta_jet`.

**Should be:** state plainly that `eta_chem` is computed while the remaining four
contributions — plume divergence, exhaust-speed spread, radiative escape, and
mass the field fails to grip — are **bounded by nothing in either repository**.

**Why it matters:** a reader seeing `eta_jet` split into a computed factor and a
swept one may read more rigour into the second than exists. This is the largest
remaining uncertainty in the whole growth chain and it currently reads as an
ordinary sweep parameter.

---

### C3. The swept grid runs below the forward-thrust floor

**Locate:** the recovery axis of both growth tables.

**Now:** the axis starts at 0.25.

**Should be:** trimmed, or the bottom rows marked unreachable. Forward thrust
vanishes at `eta_jet = 1/sqrt(1+k)`, which is **0.324 at `k` = 8.5**, so the
bottom two rows of the published axis (0.25 and 0.30) deliver **zero** growth
rather than small growth. The paper's own `eq:ve_general` already names this
floor at `eta_jet = sqrt(m_rp)`; the tables do not honour it.

---

## D. Citations needed

### Already in `references.bib` — cite these, do not add them

| key | use it for |
| --- | --- |
| `Katz_puffsat_impact_sim_2026` | **every number in A3, A9, A10, B2, B4** and the `f` = 0.818 plate measurement |
| `Katz_aim_is_all_you_need_2025` | every number in A1, A2, A4, A6, A7, C1, C3 |
| `marti1993ice` | ice vapour pressure (A7) — already cited there |
| `nist_webbook_water` | the saturation curve behind `tab:bag_state` (A3, A8) |
| `rebco_tapes` | the conductor term (A4) |
| `zeldovich_raizer` | shock and ionisation behaviour (A6, B4) |
| `hansen_kawaler_trimble_stellar_interiors` | Saha (A9) |
| `kerrebrock1964nonequilibrium`, `rosa1968mhd`, `messerle1995mhd` | seeded-plasma conductivity at 2000–3000 K (A9, B3). Each is already cited once, in the paragraph introducing the alkali seed. **A9 replaces a `sigma`-derived column with no source, so these are the measurements that should back the new one** — they cover exactly the 2000–3000 K regime where the conductivity cliff sits |
| `birkhoff1948lined` | the plug's penetration law — already cited |

### New entries that need adding

**I have not verified these bibliographic details — check each before
committing.** They are named so you know what to look for, not so you can paste
them.

| needed for | what to cite |
| --- | --- |
| the conductivity model behind A9 | Spitzer, *Physics of Fully Ionized Gases*, 2nd ed. (1962) — the standard reference for the fully-ionised limit the model reduces to |
| the sublimation flux in A7 | the **Hertz–Knudsen** relation, `Phi = P sqrt(M / 2 pi R T)`; Knudsen's 1915 *Annalen der Physik* paper is the usual primary cite, but a kinetic-theory textbook may serve the paper better |
| the saturation correlation in A3/A8 | the **Magnus–Tetens** form; Alduchov and Eskridge, *J. Appl. Meteorol.* (1996) is the standard modern fit. **Optional** — `nist_webbook_water` may cover it, and the paper already uses that key |

### Citations that are *not* needed

- **The `eta_chem` closed form (A1)** is algebra on quantities the paper already
  states; it needs the impact-sim cite for `phi`, not a literature cite.
- **The virial floor (A4)** is already argued in the paper's own text.
- **The self-widening front (B4)** rests on `eos_water`, so the impact-sim cite
  covers it.

---

## E. Reproduction lines (the ledger's rule 3)

Add a closing sentence to the caption of every computed table naming the target
that regenerates it. This is the convention ADR 0015 already uses internally
("Reproducing: `make two-leg` …"), extended into the paper.

| table | line to add |
| --- | --- |
| `tab:bag_sizing` | Reproduce with `make bag-state`. |
| `tab:bag_state` | Reproduce with `make bag-state`. |
| `tab:axial_bag` | Reproduce with `make bag-state`. |
| `tab:seed_window` | Reproduce with `make plume-state`. |
| `tab:space_mortgage_growth` | Reproduce with `make two-wave`. |
| `tab:two_leg_growth` | Reproduce with `make two-leg`. |

All targets are in `katzseth22202/aim_is_all_you_need`; say so once, in the first
such caption or in a footnote.

---

## F. Do not change

- **The 26 existing bare `\cite{Katz_aim_is_all_you_need_2025}` calls.** The
  reproduction-line convention in section E is additive and requires no change
  to them.
- **`STD_FUDGE_FACTOR = 0.8` and the plate's restitution.** Measured at
  0.817–0.820 on the heavy plate, which is the plate the growth push flies. See
  C1.
- **`tab:axial_bag`.** Every one of its 20 cells reproduces exactly. Nothing is
  owed.
- **`tab:bag_sizing`'s density, both field columns, and the radiative column as
  cold-pulse figures.** All 24 cells reproduce; only the *scope* of the
  radiative column is wrong (A10) and the mist column's model is unstated (A8).
- **The "needle through fog" conclusion, the plug, and the compact projectile.**
  All correct; B1 and B4 change the supporting argument, not the design.
- **The 194 K cruise equilibrium and everything else in that paragraph.**
  Reproduces exactly apart from A7.

---

## G. If something here looks wrong

Say so rather than working around it. Three of these corrections were found
because two numbers in the same paragraph disagreed, and that failure mode is
not unique to the paper — this document could carry one too. The numbers here
come from a reproduction pass, not from a re-derivation, so any that disagree
with the paper by a small amount are more likely to be mine than the paper's;
any that disagree by a factor are more likely to be the paper's.
