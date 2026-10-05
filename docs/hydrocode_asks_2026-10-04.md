# N18, revised: what is still owed by `puffsat_impact_simulation` on the hydrogen hot-band coat

Raised in `aim_is_all_you_need` on 2026-10-04. It revises **N18** ("Does a renewed carbon coat
survive hydrogen? B′ and a coupled surface balance for the near-term chamber") in
`Balloon-Pulse-Propulsion`'s `docs/nozzle_asks_for_impact_sim.md` @ `9031d3c`. **Written to be
copied verbatim** into that register in place of N18's estimate and "What is wanted" sections,
and from there into `puffsat_impact_simulation`. It repeats the context it needs.

The paper-facing half of this work is `docs/companion_replies_2026-10-04.md` in
`aim_is_all_you_need`. That file has the full B′ table, the method and suggested paper text.

## What changed since N18 was written

N18 asked you to confirm or break a paper-side estimate:
`m = (q_conv / 94 MJ/kg) ln(1 + B′) τ`, with B′ = 0.5 at 2500 K and 3.5 at 3900 K, giving up to
93 µm per pulse at 3900 K. **Two of its three inputs have now been replaced, one of them using
your own code.**

1. **B′ is solved, for pure-hydrogen edge gas.** Cantera 3.2.0 on NASA Glenn data (every C/H
   species in `nasa_gas.yaml`, `C(gr)` from `nasa_condensed.yaml`) gives an ideal gas against
   unit-activity graphite. At 818 bar B′ is **0.70 at 2500 K, mostly CH₄**, and **5.3 at 3900 K,
   mostly C₂H₂ and C₄H₂**. The 3900 K row is flat from 818 bar down to 30 bar (5.3 to 5.6). The
   2500 K row falls with pressure (0.19 at 62 bar). N18 expected pressure to matter little
   because C + H₂ → C₂H₂ conserves moles. That holds at 3900 K, but at 2500 K methane carries
   the carbon, and its formation does not conserve moles.
2. **The transfer coefficient comes from your Bartz liner flux, without dividing by an enthalpy.**
   `near_term.bartz_liner` computes `q = h_g (T_c − T_w)` with `h_g ∝ c_p`. Its low edge takes the
   frozen c_p (11.1 kJ/kg/K) and its high edge the equilibrium c_p (54.8 kJ/kg/K). The gas the
   layer brings to the wall is `g = h_g / c_p`, so the c_p bracket cancels and only the
   viscosity and Prandtl brackets remain (a factor of 1.83). The paper's division by 94 MJ/kg
   used an enthalpy that does not match either c_p edge. Against a **3900 K** pitch surface, on
   your `a8f0e73`, `hydrogen_5500`/janaf, equilibrium branch:

   | throat (m²) | q at the liner (MW/m²) | g = h_g/c_p (kg/m²/s) | mass clock (ms) |
   |---|---|---|---|
   | 0.149 | 4.3 to 39.1 | 0.244 to 0.446 | 40.7 |
   | 0.198 | 5.4 to 49.0 | 0.306 to 0.560 | 30.6 |

   The mass clock is `∫ (p/p0)^0.8 dt` over `near_term.blowdown`, while the gas is above
   3900 K. The gas reaches 3900 K at 76 to 101 ms after peak and 62 bar.

**The result is 13 to 26 µm per pulse at 3900 K**, against N18's 70–93 µm. On the 3× throat hot
band, 26 µm of chemistry plus the 120 µm thermal bound borrowed from methane, doubled for a hot
spot, is 0.23 kg per pulse. The hot-band text very likely stands. **The chemical term is no
longer the large one. The thermal term is, and nobody has solved it for hydrogen.**

## What is still wanted, in order

### 1. The coupled surface balance under hydrogen (was N18 item 2, now first)

Run the A5 wall solver (`make walled-nozzle-wall-layers`, `wall_layers.py`) for **pitch under the
hydrogen chamber's blowdown**, with the chemical term added. So far it has run pitch only under
methane. The paper's hydrogen hot band borrows methane's 120 µm thermal recession as its bound,
and that is now the larger term by a factor of five.

- Chemical term: `m_chem = g ln(1 + B′(T_s, p(t)))`, with g from `h_g / c_p` above and B′ from the
  table below or from your own EOS (item 3). Apply the blowing correction to the heat flux as
  well. The `ln(1+B)/B` factor that cuts the mass flux cuts the convective heat too.
- Substrates: pitch on GRCop-84 (the liner) and on A-286 (the throat inserts).
- Both flux edges, both throats.
- Report per pulse: recession in µm, **split chemical against thermal**; kg over a 3× throat-area
  hot band and over the whole 35.6 m²; the surface temperature history (does the surface sit at
  3900 K under hydrogen's flux, or below it?); and the substrate's peak temperature.

**What it decides.** The paper keeps hydrogen at 5500 K partly because a whole-wall coat at a
hotter charge would cost "at least 4 kg of pitch per pulse, 7% of the charge". That figure was the
old chemical loss alone. At 26 µm it is 1.2 kg, or 2%. Whether a whole-wall coat pays now
turns on the thermal recession this item computes.

### 2. A Lewis-number correction (new)

Both g and the `ln(1 + B′)` form assume the carbon species diffuse like heat (Le = 1). C₂H₂ and
C₄H₂ are heavy molecules diffusing through light H₂ and H, so the mass-transfer Stanton number is
`St_m = St_h (Pr/Sc)^(2/3)` and could sit some tens of percent either side of St_h. You already
carry Chapman–Enskog transport (`wall.chapman_enskog_viscosity`). Wanted: Sc for C₂H₂ in the
hydrogen chamber's edge gas at 3900 K and 62–818 bar, and the resulting factor on g. One number
with a bracket is enough.

### 3. B′ from your own EOS, as a cross-check (was N18 item 1, now smaller)

The table below is for **pure hydrogen** edge gas. Your hydrogen charge already carries the rod and
plug's 5.9 kg of carbon (`chambers.csv`: `carbon_free` = 0.072). Carbon already in the edge gas
lowers B′, so the table is an upper bound. Wanted: B′ on the near-term EOS (`near_term.py`,
`graphite_chemical_potential`) with the real edge composition, at the five points that matter:
2500 K and 3900 K at 62 and 818 bar, plus 3700 K at 300 bar. Agreement within about 10% retires
the question. The methane chamber's B′ at 496 bar stays wanted for the reason N18 gave: methane is
not exempt.

B′ supplied (kg C per kg H, ideal gas, pure H₂ edge, Cantera 3.2.0 / NASA Glenn;
`make carbon-equilibrium` in `aim_is_all_you_need`):

| T (K) | 10 bar | 30 bar | 62 bar | 100 bar | 300 bar | 818 bar |
|---|---|---|---|---|---|---|
| 2500 | 0.14 | 0.16 | 0.19 | 0.22 | 0.38 | 0.70 |
| 2800 | 0.40 | 0.42 | 0.44 | 0.46 | 0.57 | 0.81 |
| 3100 | 0.96 | 0.98 | 1.00 | 1.02 | 1.10 | 1.27 |
| 3400 | 2.00 | 2.02 | 2.03 | 2.05 | 2.11 | 2.23 |
| 3700 | 3.90 | 3.74 | 3.72 | 3.72 | 3.75 | 3.81 |
| 3900 | 6.50 | 5.55 | 5.36 | 5.31 | 5.27 | 5.28 |
| 4100 | 13.51 | 8.61 | 7.73 | 7.45 | 7.19 | 7.10 |

Method: each CₐH_b sets `p_i = K_i p_H2^(b/2)` with
`ln K_i = −(g°_i − a g°_C(gr) − (b/2) g°_H2)/RT`, and `p_H2` is root-found so the partial
pressures sum to the total.

### 4. Strip threshold and specific-impulse cost (N18 items 3 and 4, unchanged)

As N18 asked: the high-edge flux multiplier at which a 0.1, 0.2 or 0.3 mm layer is gone in one
pulse, now from item 1's coupled model; and the η cost of the picked-up carbon through the A3
machinery. At 0.23 kg per pulse the carbon is about 4% of the 5.9 kg the rod and plug already
bring, so expect a small number.

### 5. Thermography (N18 item 5, with radiation added)

N18 said the surface "cools by conducting into the substrate". At 3900 K it also radiates about
11 MW/m², which is comparable. A paper-side 1-D model checked this: pitch at A5's properties on
3 mm of copper, held at 3900 K for 35 ms, then radiation (ε = 0.85) plus conduction. Radiation
lowers the surface 26 ms after release by 5 to 20%, but the thicknesses stay far apart: 670,
1320 and 1950 K at 50, 80 and 170 µm. **Please include surface radiation** in the cooling runs N18
asked for (+10, 25, 50, 100, 150 ms; 20–200 µm; char k at 0.2, 1 and 5 W/m/K). Run them on
**methane's** release, since the paper's camera argument is methane-only and methane's
radiation-weighted load lasts 89 to 119 ms, which drives heat deeper than the 35 ms hold above.

### 6. Optional: the hydrogen chamber's infrared view

Only if the paper later reads the hydrogen hot band by camera. Gas leaving the band carries up to
5 kg of carbon per kg of hydrogen at the wall, as C₂H₂ and C₄H₂, and it can form soot as it cools.
Wanted: the in-band optical depth of the emptying hydrogen chamber 100 to 250 ms after impact,
with that carbon in it. The paper does not currently claim this view, so this item gates nothing.

## What would settle it

N18's thresholds stand, restated against the new numbers:

- **The hot-band text stands if** item 1 gives a hydrogen hot band under about 0.5 kg per pulse
  at the high edge with a 2× hot spot, and a 0.3 mm spray or thinner survives it. At 0.23 kg on
  the borrowed thermal bound, there is room for the hydrogen thermal term to come out at about
  twice methane's.
- **The whole-wall decision reopens if** item 1's hydrogen thermal recession is well under
  methane's 120 µm. A whole-wall coat would then cost a few percent of the charge, and the
  100 s that 7000 K is worth would be back on the table.
- **The methane pitch figure moves if** its chemical term (item 3's 496 bar B′ through item 1)
  is not small against the thermal one, as N18 said.

## Not impact-sim scope

As N18 said: pitch pyrolysis in 800 bar hydrogen, char conductivity and emissivity, and the
camera's view of the throat.
