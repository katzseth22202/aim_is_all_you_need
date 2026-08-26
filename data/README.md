# Vendored inputs

Files this repository consumes but cannot regenerate. Each is produced by
`puffsat_impact_simulation`, whose solvers need physics this repository does not
implement (a chemical-equilibrium water EOS, an expansion integrator, a
radiation-transport opacity model).

The ledger's routing rule is that `aim_is_all_you_need` may take "known
constants **or another repo's output**" as input. That repo commits these CSVs
to its tree rather than gitignoring them, precisely so a consumer elsewhere can
cite them -- but a consumer cannot run its `make`, so the file is copied here
with its provenance rather than fetched.

| file | source | regenerate with |
| --- | --- | --- |
| `plume_state.csv` | `puffsat_impact_simulation` @ `0216a09`, `data/results/plume_state.csv` | `make analysis-plume` there |

## `plume_state.csv`

171 rows: `w` = 44-76 km/s on a 2 km/s grid with the four quoted anchors
inserted exactly, crossed with `rho` = 0.05-2.0 kg/m^3. Solved on `eos_water`,
which carries dissociation by law of mass action and the full `O+ .. O8+` Saha
ladder.

**The table is two-dimensional and has to be.** Dissipated energy depends only
on `w` and `k`, never on density -- but Saha does, so the same budget lands at a
different temperature in a different bag. At 56.53 km/s, `rho` 0.05 -> 1.0 moves
the plume 16 857 -> 21 795 K. Since this repository sets `rho = m_slug / V` and
the enclosed volume is a live design variable, a single row would not serve.

**One trap.** `eos_water` references specific energy to bound molecular H2O at
`T -> 0`, so the bond energy is *already inside* `e`. Solve
`e(rho, T) = e_dissipated` with nothing subtracted. Subtracting an atomisation
enthalpy first double-charges the bond and lands 4 672 K instead of 15 165 K at
the cold anchor.
