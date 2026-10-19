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
| `departure_visibility.csv` | this repository (ADR 0066) | `make departure-visibility` |
| `solar_dive_stage_laws.csv` | `puffsat_impact_simulation` @ `a6742a2` (ADR 0072) | see below |
| `departure_hole_law.csv` | `puffsat_impact_simulation` @ `bfea495` (ADR 0073) | see below |

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

## `departure_visibility.csv`

Unlike the file above, this one is regenerable here; it is stored only because
building it takes about 6 minutes a process, and every cost report needs it.
One row per flown-chain cycle on the 20-day parking orbit: the stream its
onward departure meets (aim separation, both excess speeds) and the least
finite-burn loss (km/s) over burn placements that keep every incoming route in
view of Earth, at burn lengths 0-7,200 s on 400 s nodes
(`src/departure_visibility.py`). `stored_tables` refuses any cycle not in it.
`tests/test_departure_visibility.py` regenerates the streams and one node.
Rebuild it whenever the chain, the parking orbit or the loss integrator changes.

## `solar_dive_stage_laws.csv`

The impact simulation's exhaust speeds `v_e(k)` for the solar-dive charge-pusher cycle
(ADR 0072), carbon left atomic unless `bonds` says otherwise:

- `earth_departure`: the canted plate, square charge, at the two closures' mean impact
  angle and closing speed (9.86 R_sun: 101.15 deg, 103.2 km/s; 4 R_sun: 94.7 deg,
  158.35 km/s), k = 5-60. Copied from its `data/results/xray_collision/growth_earth_stage.csv`.
- `sun_boost`: the delta/D 0.5 paraboloid at 1.5 F, the real disk-into-charge collision,
  k = 20-100 on 17 log-spaced points, at closing speeds 250-700 km/s. Generated with its
  own code, `puffsat.xray_collision.growth.sun_ve(k, w)`, because its committed CSV holds
  only the optimum.
- `overtake_push`: the flat plate with a 4 m skirt, point fireball, grid x1, carbon frozen,
  at 100 km/s, k = 5-40. Copied from its `overtake_ledger.csv`.

## `departure_hole_law.csv`

The impact simulation's departure bowl (its Result 14, ADR-0065): compact impactors thread a hole
in a delta/D 0.5 paraboloid and stop in a charge at 1.5 F. Its `departure_hole.hole_ve(plate,
theta, k, w)` factors exactly as

    v_e = capture(theta, k) * w * speed(k, w)

because the only run-dependent terms are the capture and `confined_ratio`, and the latter is one
constant (1 / 0.571655). So the table has two kinds of row:

- `capture`: impulse per `m w` for "paraboloid 0.50 at 1.5 F" at theta = 89.2 / 117 / 129.9 / 180
  deg and k = 5-60, copied from its committed `data/results/xray_collision/departure_hole_law.csv`
  (its 2-D runs' JSONL is not committed).
- `speed`: `speed(k, w) = (1 + e(u_rms)) / k * mixing(k) * (1 - bond toll(k, w)) / confined_ratio`
  for k = 5-60 and w = 60-220 km/s, `bonds` = `carbon atomic` (toll charged) or `condensed` (none).
  Generated with its own code at `bfea495`, after building its polyethylene table
  (`PYTHONPATH=python python -m puffsat.tables --ch2`):

  ```python
  u_rms = w * sqrt(k) / (1 + k)
  column = (1 + k) * growth.REBOUND_IMPACTOR / (2 * pi * charge.PLATE_R**2)
  v = (1 + charge.rebound(u_rms, column)) / k * charge.mixing_factor(growth.SUN_PLATE, k)
  speed = 0.571655 * v * (1 - departure_hole.bond_loss(k, w))   # condensed: no bond term
  ```

`StageLaws.hole_ve` interpolates the capture linearly in `cos theta` (extrapolating past the run
angles, as the simulation does) and both factors log-linearly in k. It reproduces the simulation's
`hole_ve` columns to within 0.5% (the k-interpolation between its runs).
