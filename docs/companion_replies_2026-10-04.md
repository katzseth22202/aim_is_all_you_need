# Companion replies, 2026-10-04, for the parent paper

Answers the open questions on the hydrogen hot-band coat and its thermography, as
landed in `Balloon-Pulse-Propulsion` @ `9031d3c` ("Spray pitch on the hydrogen
chamber's hot spots and sense coats thermally" and the panel redraw after it).
**Written to be copied into the paper repo**, so the quoted paper text is
restated here and every number is given in full.

**Nothing has been written into `templateArxiv.tex`.**

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
  CH4, 1.0% C2H2). The high pressure is what favours methane. At 1 bar the same
  wall gives 0.13. The peak value is an upper bound, because the blowdown only
  lowers it.
- **At 3900 K it holds 5.3 kg, mostly acetylene** (27 mol% C2H2, 5% C4H2). This
  is flat from 818 bar down to about 30 bar.

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

| throat (m²) | g against a 3900 K wall (kg/m²/s) | mass-transfer clock (ms) |
|---|---|---|
| 0.149 | 0.244 to 0.446 | 40.7 |
| 0.198 | 0.306 to 0.560 | 30.6 |

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
3900 K value.

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
still true.

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
  solved (see the companion ask of 2026-10-04 to `puffsat_impact_simulation`).
  So the sentence should not yet claim that a whole-wall coat pays either.
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
before release. That shifts the times but not the ordering.

## H3. Citations

- **Pulsed thermography:** resolved. The paper cites `maldague2001_ir_ndt`.
- **Galileo probe / Milos et al. 1999:** never used in the paper. Drop it.

## Not settled here

- **The surface balance under hydrogen.** This covers whether the hot band's
  surface sits at 3900 K under hydrogen's flux, and pitch's thermal recession
  there (the paper borrows methane's 120 µm). It also covers a Lewis-number
  correction for heavy C2H2 diffusing through H2. All three are asked of
  `puffsat_impact_simulation` in `docs/hydrocode_asks_2026-10-04.md`. That file
  is a revision of N18 in the paper's `docs/nozzle_asks_for_impact_sim.md`.
  It replaces N18's paper-side estimate table, which carried the old
  70–93 µm, and reorders what is wanted now that the chemical term is small.
- **A camera in the hydrogen chamber** would look through gas carrying carbon.
  The paper's `:2217` argues only the methane view, so nothing there needs to
  change unless the hydrogen band is also to be read by camera.
