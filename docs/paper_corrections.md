# Corrections owed to the paper

Living list. The paper is `templateArxiv.tex` in
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion);
it is **not** checked into this repository, so each entry names the claim by its
quoted wording rather than by line number.

**The paper is not a source of truth.** Where a calculation here disagrees with
it, the calculation wins and the paper changes. Entries below are written that
way round.

Every entry states: what the paper says now, what it should say, and the `make`
target or function that reproduces the replacement. Nothing here is applied yet.

---

## A. Numbers that move

### A1. Jet-efficiency axes are `eta_geom`, not `e`
**Now:** `tab:space_mortgage_growth` and `tab:two_leg_growth` sweep a bare `e`.
**Should be:** `eta_geom`, with the chemistry ceiling `eta_chem` shown beside it,
because `eta_jet = eta_chem * eta_geom` and only the second is swept.
**Why:** ADR 0016. `sec:jet_efficiency` already defines `eta_jet^2` to include
"frozen ionization or dissociation energy", so one of its five contributions is
now computed rather than swept.
**Reproduce:** `make two-wave`, `make two-leg`.

### A2. The gas expands at 14.1 km/s, not 17
**Now:** "a gas converting all of that to directed motion expands at
`sqrt(2u)` = 17 km/s" (`sec:two_leg_nozzle`).
**Should be:** **14.1 km/s**, a 19% reduction, because 50.9 MJ/kg of the
dissipated budget is spent atomising the water and does not come back.
**Reproduce:** `plume_thermal.chemistry_efficiency`.

### A3. The bag's field leak is 0.11-2.54%, not 4.4%
**Now:** `tab:bag_state` row "Field leaking through the plume (4.4% of
20.8 MJ/kg)".
**Should be:** the residence-weighted `1/Rm` from the solved cooling history --
**0.11% / 0.17% / 0.29% / 2.54%** at 75 / 65 / 56.53 / 45.58 km/s.
**Consequence, which is larger than the row:** from 122 K storage the waste heat
no longer finishes melting the slug, so **nothing boils and the bag film is not
a mass item at all** -- 0 kg rather than 2.8 kg. The bag becomes a containment
membrane sized by handling rather than a pressure vessel.
**Reproduce:** `make bag-state`.

### A4. The nozzle mass floor is ~8 t, not 3.7 t
**Now:** "the virial floor ... turns that into 3.7-11 t and 10-30 t
respectively" and "a tenth to a third of a 100 t craft" (`sec:space_mortgages`).
**Should be:** the two-term model the same paragraph says is owed. Adding the
REBCO conductor gives **8.0-15.4 t** and **17.3-37.6 t**, so the optimistic end
is not reachable and the fraction of a 100 t craft is **8-38%**.
**Also owed:** the paper never states a tape operating current, and the
conductor term is exactly inversely proportional to it (300 A -> 7.3 t,
1000 A -> 2.2 t). That is a bigger lever than the structure's pre-compression
band and should be named.
**Reproduce:** `nozzle_geometry.two_term_nozzle_mass`.

### A5. `tab:bag_state`'s own leak row is internally inconsistent
**Now:** the row reads "4.4% of 20.8 MJ/kg" and prints **0.89** MJ/kg.
0.044 x 20.8 is 0.915, and 0.89/20.8 is 4.28%.
**Should be:** whichever is intended, stated consistently. The printed table's
downstream digits follow 0.89, so that is the load-bearing one.

---

## B. Arguments that reach the right answer by the wrong route

### B1. The aperture argument is about neutrals, not open area
**Now:** "A head-on nozzle is a chamber with a hole in the end the projectile
came through ... Anything left open leaks it" (`sec:needle_through_fog`).
**Problem:** that is an open-area argument, and it is wrong for a *magnetic*
mirror, whose leak is a loss-cone property set by the mirror ratio rather than
by how much of the end wall is physically open.
**Should be:** the mechanism is the **un-ionised fraction**, which the field
cannot steer and which does leave ballistically through any physical hole. At
the cold leg that is most of the plume (`f` = 0.06 at 15 170 K). The conclusion
-- keep the projectile compact -- survives unchanged.
**Independent support the paper does not use:** the compact arrival also wins on
chain arithmetic alone. Self-widening delivers `k` = 7.21 from a 0.15 m arrival
against 8.60 from 0.8 of the bore, and 7.21 sits inside ADR 0016's tolled
optimum band while 8.60 overshoots it.

### B2. The recombination argument is checked at the wrong station
**Now:** "Breaking water into its atoms takes 50.4 MJ/kg ... It comes back only
if the atoms find each other again as the plume cools and expands. **They do,
and quickly.**" (`sec:watering_it_down`).
**Problem:** the rate check is correct *at 1 kg/m^3*, but there the plume is
fully atomised and has nothing to give back yet. By the time it has cooled
enough to recombine it sits at ~0.02 kg/m^3 and past the nozzle lip, where the
expansion clock steps down 8x in one step.
**Should be:** the loan defaults. The freeze is at 1.1e-2 to 2.4e-2 kg/m^3 with
90-100% of the store still held.
**Note the paper already flags the condition** ("we have not computed the
density a real pulse produces") -- the correction is that it has now been
computed and the answer is the unfavourable one.

### B3. `tab:seed_window`'s `Rm` and `tab:bag_state`'s leak are the same quantity
**Now:** presented two pages apart as unrelated.
**Should be:** joined explicitly. `tau_d/t_exp = mu0 sigma L^2 / (L/v) = mu0 sigma v L`,
which *is* `Rm`, so the leak fraction is `~1/Rm` and the paper has been printing
the answer to its own open question. The conductivity cliff (`Rm` = 1 at ~2845 K)
should be stated where the leak is.

---

## C. Framing written for retired arguments

### C1. The `e1 ~ 0.6` crossover
**Now:** `sec:two_leg_nozzle` states the crossover against `f = 0.8`, so a reader
who believes the incumbent's number can reach the opposite conclusion.
**Should be:** the comparison no longer needs a matched-quality argument at all.
At the same `eta_geom`, granting the plate its measured `f` = 0.818, the nozzle
wins **47x / 22x / 10x** at `eta_geom` = 1.0 / 0.9 / 0.8. See ADR 0015
amendment 2.

### C2. `eta_geom` is unmeasured and the text should say so
**Now:** nothing distinguishes the grounding of the two factors of `eta_jet`.
**Should be:** state plainly that `eta_chem` is computed while the remaining four
contributions -- divergence, exhaust-speed spread, radiative escape, and mass the
field fails to grip -- are bounded by nothing in either repository. A reader
seeing `eta_jet` split into a computed factor and a swept one may read more rigour
into the second than exists.

### C3. The swept grid runs below the forward-thrust floor
**Now:** the recovery axis starts at 0.25.
**Should be:** trimmed, or marked. Forward thrust vanishes at
`eta_jet = 1/sqrt(1+k)`, which is 0.324 at `k` = 8.5, so the bottom two rows of
the published axis were never reachable and the table did not say so.

### A6. `sec:two_leg_nozzle`'s mirror passage cannot be checked as written
**Now:** "Their ratio is `Mv^2/(gamma-1)E` ... The ratio is 6.7, the wall sees
1.26 GPa, and it would take 56 T."
**Problem:** neither `gamma` nor the closing speed is stated, and both are needed
to reproduce it. They are recoverable -- `gamma` = **1.2** back-solves from the
6.7, and the closing speed is **56 km/s** from the momentum the two cases share
(62.5 x 22.4 = 238 x 5.9 = 1400 kg km/s) -- but a reader should not have to.
**Should be:** state both. `gamma` = 1.2 rather than the monatomic 5/3 is itself
worth a clause, since dissociation and ionisation are absorbing energy a
monatomic gas would put into translation.
**Also worth adding:** the mechanism. A throat-end plug dissipates *more* energy
(35.1 GJ against 23.5) yet needs a seventh of the field, because the ram term
falls faster than the static term rises. That is the non-obvious part and the
passage states only the outcome.
**Reproduce:** `nozzle_geometry.mirror_stagnation`.

---

## D. Citation mechanism (the ledger's rule 3)

Add "Reproduce with `make <target>`" to the caption of every computed table:
`tab:bag_sizing`, `tab:bag_state`, `tab:axial_bag`, `tab:seed_window`,
`tab:space_mortgage_growth`, `tab:two_leg_growth`. The 26 existing bare
`\cite{Katz_aim_is_all_you_need_2025}` calls need no change.
