# Companion replies, 2026-10-04 (updated 2026-10-05), for the parent paper

Answers the open questions on the hydrogen hot-band coat and its thermography, as
landed in `Balloon-Pulse-Propulsion` @ `9031d3c` ("Spray pitch on the hydrogen
chamber's hot spots and sense coats thermally" and the panel redraw after it).
**Written to be copied into the paper repo**, so the quoted paper text is
restated here and every number is given in full.

**This is the single reply to take back.** It folds in the hydrocode's answer of
2026-10-05 (`puffsat_impact_simulation` @ `8a4f80a`, "Cross-check the hydrocode
ask's B' on our own EOS (N18 revised)") and replaces the separate
`docs/hydrocode_asks_2026-10-04.md`. That ask's N18 revision now lives in the
last section below, updated with what the hydrocode did and did not do.

**Nothing has been written into `templateArxiv.tex`.**

## In short

- **The hydrocode's answer changes no paper-facing number.** It checked B′ on its
  own equation of state. On the species both codes carry, the two agree within
  7% at all five points asked. The hydrocode's lower hot-end values are missing
  species, mainly C₄H₂, not a difference in thermochemistry. The Cantera table
  below stands, now cross-checked.
- **The paper edits are the ones below.** H1: B′ is 0.7 at 2500 K (methane) and
  5.3 at 3900 K (acetylene), and the loss is 13 to 26 µm per pulse. The hot band
  spends 0.23 kg. The whole-wall coat costs at least 1.2 kg, or 2%, and must not
  yet be called worth it. H2: one thermography sentence. H3: drop Milos.
- **The deciding term is still open.** Pitch's thermal recession under hydrogen
  (N18 item 1) has not been started. The hydrocode gave specific reasons (below).
  Until it is solved, the hot band rests on methane's 120 µm, and the whole-wall
  decision is not made either way.

Two of the three answers needed numbers this repository had not computed: the
equilibrium carbon loading of hydrogen at the chamber's pressure, and the
transfer coefficient behind the companion's convective flux. The method for
each is recorded under its answer so that it stands without scratch files.

## H1. Hydrogen carries more carbon than the paper says, but far less gas reaches the wall

**Paper text** (the hydrogen-chamber paragraph after `tab:wall_pairings`, starting
"Hot hydrogen does attack carbon"): 0.5 kg of carbon per kg of gas at 2500 K
"mostly as acetylene", 3.5 kg at 3900 K, 14 to 189 MW/m² divided by 94 MJ/kg
giving 0.15 to 2.0 kg/m²/s, "turns 3.5 into 1.5", and losses of 1 to 25 µm at
2500 K and 5 to 90 µm at 3900 K, "the same order as the methane chamber's
thermal recession below".

### The equilibrium ratio, solved

B' is the kilograms of carbon held in the gas, per kilogram of hydrogen, at
equilibrium with graphite. It was solved on Cantera 3.2.0's NASA Glenn data:
every C/H species in `nasa_gas.yaml` plus `C(gr)` from `nasa_condensed.yaml`,
ideal gas. The ideal-gas error is a compressibility of about 1.07 for H2 at
818 bar and 3900 K. Each species CₐH_b sets its partial pressure
`p_i = K_i p_H2^(b/2)` against unit-activity graphite, with
`ln K_i = -(g°_i - a g°_C(gr) - (b/2) g°_H2)/RT`, and `p_H2` is root-found so
that the partial pressures sum to the total.

| T (K) | 10 bar | 30 bar | 62 bar | 100 bar | 300 bar | 818 bar |
|---|---|---|---|---|---|---|
| 2500 | 0.14 | 0.16 | 0.19 | 0.22 | 0.38 | 0.70 |
| 2800 | 0.40 | 0.42 | 0.44 | 0.46 | 0.57 | 0.81 |
| 3100 | 0.96 | 0.98 | 1.00 | 1.02 | 1.10 | 1.27 |
| 3400 | 2.00 | 2.02 | 2.03 | 2.05 | 2.11 | 2.23 |
| 3700 | 3.90 | 3.74 | 3.72 | 3.72 | 3.75 | 3.81 |
| 3900 | 6.50 | 5.55 | 5.36 | 5.31 | 5.27 | 5.28 |
| 4100 | 13.51 | 8.61 | 7.73 | 7.45 | 7.19 | 7.10 |

Reproduce: `make carbon-equilibrium` (`src/carbon_hydrogen_equilibrium.py`).

- **At 818 bar and 2500 K the gas holds 0.70 kg, mostly as methane** (9.5 mol%
  CH4, 1.0% C2H2; methane carries 74% of the carbon). The high pressure is what
  favours methane. At 62 bar acetylene carries two-thirds of the carbon instead,
  and at 1 bar the same wall gives 0.13. The peak value is an upper bound,
  because the blowdown only lowers it.
- **At 3900 K it holds 5.3 kg, mostly acetylene** (27 mol% C2H2, 5% C4H2; by
  carbon, 60% C2H2, 23% C4H2, 7% C6H2). This is flat from 818 bar down to about
  30 bar.

### Cross-check on the hydrocode's own equation of state (2026-10-05)

The hydrocode solved B′ independently. It uses its own JANAF-corrected harmonic
thermochemistry and a mass-action network in number-density form, and it pins
monatomic carbon at graphite's saturation density. It solves total pressure and
charge neutrality explicitly. The function is
`near_term.carbon_blowing_parameter(T, p)` in
`python/puffsat/walled_nozzle/near_term.py`. Five parametrised tests and a
flatness check pin it, and `make walled-nozzle-near-term` prints it.

Its species set is `eos_methane.MOLECULES` = CH4, CH3, CH2, CH, C2H2, C2H, C2,
C3, H2, plus the atoms and ions. It lacks C4H2, C6H2, C2H4 and the other heavier
hydrocarbons. To separate thermochemistry from species inventory, the last two
columns re-solve Cantera with only those 11 species (CH4, CH3, CH2, CH,
`C2H2,acetylene`, C2H, C2, C3, H2, H, C). Every other species in
`nasa_gas.yaml` is dropped from the partial-pressure sum, and the method is
otherwise unchanged:

| T (K) | p (bar) | Cantera, all species | hydrocode | ratio | Cantera, hydrocode's species | hydrocode / matched |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2500 | 62 | 0.187 | 0.184 | 0.98 | 0.180 | 1.02 |
| 2500 | 818 | 0.697 | 0.590 | 0.85 | 0.637 | 0.93 |
| 3700 | 300 | 3.75 | 3.01 | 0.80 | 2.90 | 1.04 |
| 3900 | 62 | 5.36 | 3.82 | 0.71 | 3.93 | 0.97 |
| 3900 | 818 | 5.28 | 3.86 | 0.73 | 3.74 | 1.03 |

- **On the same species, the two codes agree within 4% at four points and 7% at
  the fifth.** That meets the ask's 10% bar everywhere. The two thermochemistry
  sources (NASA Glenn and the hydrocode's JANAF fits) are independent, so the
  check retires the question.
- **The hot-end gap of 20-30% is species inventory.** C4H2 alone carries 19-23%
  of the carbon at 3700-3900 K, and C6H2 4-7% more. The hydrocode named C4H2,
  which is the larger part. Missing species can only remove carbon, so the
  hydrocode's value is a lower bound there and Cantera's is the one to use.
- **At 2500 K and 818 bar, the hydrocode attributes the agreement to CH4. That
  holds at that point only.** C2H4 carries 8% of the carbon there and the
  hydrocode lacks it. That accounts for most of the 15% gap. The remaining 7%
  is CH4's thermochemistry, or other small differences between the two codes.
  At 62 bar acetylene carries the carbon, so the 2% agreement there checks
  C2H2, not CH4.
- **Both codes find the 3900 K row flat in pressure** (hydrocode 3.82 against
  3.86, a 1% spread). The flatness comes from the mole-conserving reactions,
  not from which heavy species are carried.
- **The hydrocode also confirmed the hydrogen charge's `carbon_free` = 0.0720**
  (`near_term.solve_charge(near_term.PAIRINGS[0]).carbon_free` = 0.07205). It did
  not solve B′ for that real edge composition, carbon already in the gas. The
  increment is not defined yet: how much of the carbon at the wall came from
  upstream, and how much was freshly ablated. The pure-hydrogen table therefore
  remains the stated upper bound.

Reproduce the matched column: in `graphite_equilibrium`, keep only those 11
species in the sum (zero the others' partial pressures before the root-find). The
hydrocode's column is `puffsat_impact_simulation` @ `8a4f80a`.

### The gas brought to the wall, from the companion's own correlation

The paper divides the tabulated flux by an enthalpy. That inversion is not
consistent with how the flux was made. The companion's liner flux is Bartz's
`q = h_g (T_c - T_w)`, and `h_g` is proportional to `c_p`
(`puffsat_impact_simulation`, `walled_nozzle/near_term.py::bartz_liner`). The
low edge takes the frozen `c_p` (11.1 kJ/kg/K), and the high edge takes the
equilibrium `c_p` (54.8 kJ/kg/K, 4.9 times larger) together with Pr = 0.5 and
the small cross-section. The gas the boundary layer brings to the wall is
`g = h_g / c_p`, so **the c_p bracket cancels**. Only the viscosity and Prandtl
brackets remain, a factor of 1.83. Most of the 14-to-189 spread is a spread in
how much heat each kilogram brings, not in how many kilograms arrive.

The coat sits on the hot band at 3900 K, not on copper at 1000 K, so `g` is
taken against a 3900 K wall. Bartz's σ factor is about 1.3 times lower there
than against copper.

| throat (m²) | q at the liner (MW/m²) | g against a 3900 K wall (kg/m²/s) | mass-transfer clock (ms) |
|---|---|---|---|
| 0.149 | 4.3 to 39.1 | 0.244 to 0.446 | 40.7 |
| 0.198 | 5.4 to 49.0 | 0.306 to 0.560 | 30.6 |

The clock is `∫ (p/p0)^0.8 dt` over the companion's real-gas blowdown
(`near_term.blowdown`), held only while the gas is hotter than the 3900 K wall.
That ends 76 to 101 ms after peak, at 62 bar, which is inside the flat part of
the 3900 K row. The B' tail below 10 bar never comes into play.

For comparison, the same calculation against a 1000 K copper wall reproduces the
paper's flux to within the copper temperature range: 15.7 to 177 MW/m² against
the tabulated 14 to 189. The g there is 0.31 to 0.72 kg/m²/s.

Reproduce: run in `puffsat_impact_simulation` @ `a8f0e73`
(`PYTHONPATH=python uv run python <script>`) with `solve_charge` on
`hydrogen_5500`/janaf, the equilibrium branch, `bartz_liner` at `T_w` = 3900 K,
and `cp_frozen = 2.5 k_B / (ρ / n_total)`,
`cp_eq = wall.frozen_heat_capacity(...)` exactly as `bartz_liner` builds them.

### The loss per pulse

`m = g ln(1 + B') t_clock`, with pitch at 1300 kg/m³:

| wall | B' used | ln(1+B') | loss per pulse |
|---|---|---|---|
| 3900 K | 5.3 | 1.84 | **13 to 26 µm** |
| 2500 K | 0.70 (peak, an upper bound) | 0.53 | **at most 4 to 8 µm** |

The 2500 K row takes `g` with the σ factor for a 2500 K wall, 1.12 times the
3900 K value. Had the hydrocode's B′ been used at 3900 K, ln(1+B′) would be
about 15% lower (1.58 against 1.84). Use the full-species value.

### What the paper should say

Replace the equilibrium and loss sentences. Suggested text, in the paper's voice:

> At \SI{2500}{\kelvin} and the chamber's \SI{818}{\bar}, hydrogen in equilibrium
> with a carbon wall holds \SI{0.7}{\kilogram} of carbon per kilogram of gas,
> mostly as methane, and at \SI{3900}{\kelvin} it holds \SI{5.3}{\kilogram},
> mostly as acetylene.
> [... NERVA sentences unchanged ...]
> The same film that carries heat in carries carbon out. The convective flux of
> \autoref{tab:wall_pairings} is Bartz's correlation, whose coefficient divided
> by the gas's heat capacity is the mass of gas the film brings to the wall
> each second. Against a pitch surface at \SI{3900}{\kelvin} that is 0.24 to
> \SI{0.56}{\kilogram\per\square\meter\per\second}. Carbon vapor leaving the surface pushes that film outward
> and slows the exchange. The usual correction multiplies the gas flow by
> $\ln(1+B)$ rather than by the equilibrium ratio $B$ itself, which turns 5.3
> into 1.8. Over the 31 to \SI{41}{\milli\second} in which the gas stays hotter than the
> wall, a surface at \SI{3900}{\kelvin} loses 13 to \SI{26}{\micro\meter} of pitch per pulse, and
> one at \SI{2500}{\kelvin} at most \SI{8}{\micro\meter}. That is a fifth of the methane chamber's
> thermal recession below.

Keep the caveat sentence ("This is our estimate from a single transfer
coefficient, not a solved balance of heat and chemistry at the surface."). It is
still true. If the paper wants to say the equilibrium figures are checked, one
clause will do. Two independent thermochemistry sources agree on them within
7% on a common set of species.

### What follows from it

- **The hot band (`:2213`).** "Hydrogen's chemistry and the heat together remove
  at most about 0.2 mm" becomes **about 0.15 mm** (26 µm of chemistry plus the
  120 µm thermal bound borrowed from methane). With a hot spot doubling the load,
  the band spends 0.29 mm × 1300 kg/m³ × 0.594 m² = **0.23 kg**, or 0.4% of the
  59.9 kg charge. "Under 0.4 kg" stays true, and the 0.3 mm spray now covers
  even the doubled load.
- **The whole-wall coat (same paragraph as H1).** "At least 4 kg of pitch per
  pulse, 7% of the charge" was the chemical loss alone over 35.6 m². At 26 µm it
  is **at least 1.2 kg, 2% of the charge**. The argument for staying at 5500 K
  then rests on the thermal recession and the hotter gas's added flux, not on
  the chemistry. The thermal recession of pitch under *hydrogen* is not yet
  solved (N18 item 1 below, not started as of 2026-10-05). So the sentence
  should not yet claim that a whole-wall coat pays either.
- **"Mostly as acetylene" at 2500 K is wrong** at the chamber pressure. It is
  methane.

### The arithmetic note behind 94 MJ/kg

For the record: 94 MJ/kg assumed about 5% dissociation. At 5500 K and 818 bar
hydrogen is 18.3% dissociated in the companion's chamber state (18.5% by mass
on Cantera). That matches the table's 36% bond share. The equilibrium enthalpy
drop to a 1000 K wall is 125 MJ/kg, and the frozen drop is 83 MJ/kg. The point is
moot once `g = h_g / c_p` replaces the division, but don't reuse the 94.

## H2. Thermography: the square law stands, and radiation is a correction

**Paper text** (`:2217`): "The surface then cools by conducting heat down through
the remaining layer." The L²/α arithmetic (26 ms, 67 ms, 0.3 s) is correct.

At 3900 K the surface also radiates about 11 MW/m² (ε = 0.85), which is
comparable to the conduction into a thin layer. A 1-D implicit conduction model
was run. The pitch is k = 0.2 W/m/K at 1300 kg/m³ and 1600 J/kg/K, on copper
3 mm thick. The surface is held at 3900 K for 35 ms, then released to
radiation plus conduction. It gives these surface temperatures 26 ms after
release:

| coat | radiation + conduction | conduction only |
|---|---|---|
| 50 µm | 670 K | 710 K |
| 80 µm | 1320 K | 1520 K |
| 170 µm | 1950 K | 2340 K |

Radiation lowers every curve by 5 to 20% but leaves them far apart. Thickness
still sets when the surface cools, so the method stands. Suggested fix:

> The surface then cools by radiating and by conducting heat down through the
> remaining layer, and only the conduction depends on how thick the layer is.

The model holds the surface for hydrogen's convective pulse length. Methane's
radiative load lasts longer (89 to 119 ms weighted), which drives heat deeper
before release. That shifts the times but not the ordering. The hydrocode's
version of this run, on methane's release (N18 item 5), has not been done yet.
Nothing in the suggested sentence waits on it.

## H3. Citations

- **Pulsed thermography:** resolved. The paper cites `maldague2001_ir_ndt`.
- **Galileo probe / Milos et al. 1999:** never used in the paper. Drop it.

## N18 as it now stands, for `docs/nozzle_asks_for_impact_sim.md`

**Copy this section into the paper's register** in place of N18's estimate and
"What is wanted" sections, and from there into `puffsat_impact_simulation`.
N18 is "Does a renewed carbon coat survive hydrogen? B′ and a coupled surface
balance for the near-term chamber", @ `9031d3c`. This section revises N18 as of
2026-10-04 and records the hydrocode's reply of 2026-10-05 (@ `8a4f80a`). Item
numbers are the ones both sides have used.

### What changed since N18 was written

N18 asked for a paper-side estimate to be confirmed or broken:
`m = (q_conv / 94 MJ/kg) ln(1 + B′) τ`, with B′ = 0.5 at 2500 K and 3.5 at 3900 K,
giving up to 93 µm per pulse at 3900 K. Two of its three inputs have been
replaced (H1 above). B′ is solved and cross-checked: 0.70 at 2500 K and 5.3 at
3900 K, at 818 bar. The transfer coefficient is `g = h_g / c_p` from the
hydrocode's own Bartz flux. **The result is 13 to 26 µm per pulse at 3900 K.
The chemical term is no longer the large one. The thermal term is, and nobody
has solved it for hydrogen.**

### Status

| # | item | status, 2026-10-05 |
|---|---|---|
| 3 | B′ on the hydrocode's EOS | **done for pure H₂; agrees within 7% on matched species** (H1). Two parts remain open: the real edge composition, and methane's B′ at 496 bar |
| 1 | coupled surface balance under hydrogen | **not started.** It needs a structural change to `wall_layers.py`, plus substrate data |
| 2 | Lewis-number correction | **not started.** It needs a binary C₂H₂–H₂ cross-section from a cited source |
| 4 | strip threshold and Isp cost | **blocked on item 1** |
| 5 | thermography with radiation, on methane | **not started.** It does not depend on hydrogen, and it is the cheapest item left |
| 6 | the hydrogen chamber's infrared view | **not wanted.** It gates nothing |

### 1. The coupled surface balance under hydrogen (first)

Run the A5 wall solver (`make walled-nozzle-wall-layers`, `wall_layers.py`) for
**pitch under the hydrogen chamber's blowdown** (`near_term.PAIRINGS[0]`,
`hydrogen_5500`), with the chemical term added. So far it has run pitch only
under methane. The paper's hydrogen hot band borrows methane's 120 µm thermal
recession as its bound, and that is now the larger term by a factor of five.

- Chemical term: `m_chem = g ln(1 + B′(T_s, p(t)))`, with g from `h_g / c_p`.
  Apply the blowing correction to the heat flux as well. The `ln(1+B)/B` factor
  that cuts the mass flux cuts the convective heat too.
- **Use the full-species B′.** Use the H1 table, or `carbon_blowing_parameter`
  after C4H2, C6H2 and C2H4 are added to `eos_methane.MOLECULES`. As it stands,
  the function reads ln(1+B′) about 15% low at 3900 K.
- Report per pulse: recession in µm, **split chemical against thermal**; kg over
  a 3× throat-area hot band and over the whole 35.6 m²; the surface temperature
  history (does the surface sit at 3900 K under hydrogen's flux, or below it?);
  and the substrate's peak temperature. Cover both flux edges and both throats.

**Why it is not started (hydrocode, 2026-10-05).** There are two gaps.
(a) `wall_layers.py` removes mass only once the surface pins at a thermal cap.
That is the only way methane loses mass, since its boundary layer plates carbon.
Hydrogen's chemical ablation runs continuously, and the blowing factor enters
the heat-flux boundary condition of the implicit solve. That is a change to a
nonlinear solve, not a new parameter. (b) The repository carries graphite, pitch
and Cr-Mo steel, but not GRCop-84 (the liner) or A-286 (the throat inserts).
Inventing their properties would break its provenance rule.

**Agreed order.** Build it first on bare Cr-Mo steel, whose data is already in the
repository. Get the chemical/thermal split and the verdict below. Chase GRCop-84
and A-286 data only if those numbers say the substrate matters.

**What it decides.** The paper keeps hydrogen at 5500 K partly because a
whole-wall coat at a hotter charge would cost "at least 4 kg of pitch per pulse,
7% of the charge". That figure was the old chemical loss alone. At 26 µm it is
1.2 kg, or 2%. Whether a whole-wall coat pays now turns on the thermal recession
this item computes.

### 2. A Lewis-number correction

Both g and the `ln(1 + B′)` form assume the carbon species diffuse like heat
(Le = 1). C₂H₂ and C₄H₂ are heavy molecules diffusing through light H₂ and H.
The mass-transfer Stanton number is `St_m = St_h (Pr/Sc)^(2/3)`, so it could sit
some tens of percent either side of St_h. Wanted: Sc for C₂H₂ in the hydrogen
chamber's edge gas at 3900 K and 62–818 bar, and the resulting factor on g. One
number with a bracket is enough.

**Why it is not started (hydrocode, 2026-10-05).** `wall.chapman_enskog_viscosity`
is a single-species hard-sphere self-viscosity. A binary diffusion coefficient
needs a C₂H₂–H₂ collision cross-section with a citation, and the repository has
none. That objection is correct. A candidate source is Lennard-Jones parameters
for C₂H₂ and H₂ from a tabulated transport set, with the standard combining
rules. Svehla's NASA TR R-132 and the GRI-Mech 3.0 transport file are examples;
check both before use. This item can be done on either side of the boundary.

### 3. B′ from the hydrocode's EOS (done for pure hydrogen)

Done: see H1's cross-check. Still wanted:

- **The real edge composition.** The hydrogen charge carries 5.9 kg of rod and
  plug carbon (`carbon_free` = 0.0720, confirmed). Carbon already in the edge
  gas lowers B′, so the pure-H₂ table is an upper bound. First define the
  increment: the wall's ablation, net of carbon that arrives from upstream.
- **Methane's B′ at 496 bar.** It stays wanted for the reason N18 gave: methane
  is not exempt. The 2026-10-05 reply did not address it.

### 4. Strip threshold and specific-impulse cost (blocked on item 1)

As N18 asked: the high-edge flux multiplier at which a 0.1, 0.2 or 0.3 mm layer
is gone in one pulse, from item 1's coupled model; and the η cost of the
picked-up carbon through the A3 machinery. At 0.23 kg per pulse the carbon is
about 4% of the 5.9 kg the rod and plug already bring, so expect a small number.

### 5. Thermography, with radiation

The H2 check above found that radiation lowers the surface 26 ms after release
by 5 to 20%, but the thicknesses stay far apart. **Include surface radiation**
(ε = 0.85) in the cooling runs N18 asked for: +10, 25, 50, 100 and 150 ms;
20–200 µm; char k at 0.2, 1 and 5 W/m/K. Run them on **methane's** release.
The paper's camera argument is methane-only, and methane's radiation-weighted
load lasts 89 to 119 ms, which drives heat deeper than a 35 ms hold.
`wall_layers.run` already carries a radiation term. It is not yet wired to a
parametric cold-wall sweep.

### 6. Optional: the hydrogen chamber's infrared view

Only if the paper later reads the hydrogen hot band by camera. Gas leaving the
band carries up to 5 kg of carbon per kg of hydrogen at the wall, as C₂H₂ and
C₄H₂, and it can form soot as it cools. That would mean the in-band optical
depth of the emptying chamber 100 to 250 ms after impact. The paper does not
claim this view, so nothing is asked.

### What would settle it

N18's thresholds stand, restated against the new numbers:

- **The hot-band text stands if** item 1 gives a hydrogen hot band under about
  0.5 kg per pulse at the high edge with a 2× hot spot, and a 0.3 mm spray or
  thinner survives it. At 0.23 kg on the borrowed thermal bound, the hydrogen
  thermal term can come out at about twice methane's and still pass.
- **The whole-wall decision reopens if** item 1's hydrogen thermal recession is
  well under methane's 120 µm. A whole-wall coat would then cost a few percent
  of the charge, and the 100 s that 7000 K is worth would be back on the table.
- **The methane pitch figure moves if** its chemical term (item 3's 496 bar B′
  through item 1) is not small against the thermal one, as N18 said.

Item 3's cross-check moves none of these thresholds. It changes confidence in
the inputs to item 1, not the decision.

### Not impact-sim scope

As N18 said: pitch pyrolysis in 800 bar hydrogen, char conductivity and
emissivity, and the camera's view of the throat.
