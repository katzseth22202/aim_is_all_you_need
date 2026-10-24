# A 1 kg floor on the Sun-wave disk: answers to S59 (for the parent)

Raised 2026-10-24 in `aim_is_all_you_need` (ADR 0085). **Written to be copied into
`Balloon-Pulse-Propulsion`.** It answers S59 items 1-5
(`docs/sun_disk_floor_asks_for_aim_repo.md` @ `baa3a23`), and adds the Ceres loop at the
author's request.

Reproduce with `make sun-disk-floor` (~1 min). The headlines are `make solar-dive`
(`ARGS="--gaps"`, `"--sweep"`, `"--eccentric"`), and the Ceres loop is `make ceres-dive`.

## Headline (for the paper)

**Your floored rows are confirmed to $0.2 and can replace the unfloored optimum everywhere.**

| Closure | arc | growth | doubling | steady $/kg | BE cheap / dear |
|---|---|---|---|---|---|
| 9.86 R_sun, X 60 | 0.99 h | 3.23 | 1.94 y | 96.5 | 131 / 158 |
| 4 R_sun, X 90 | 1.17 h | 4.88 | 1.43 y | 74.1 | 85 / 89 |
| 9.86 R_sun, X 90 | 1.83 h | 2.95 | 2.10 y | 103.3 | 153 / 204 |
| 4 R_sun, X 150 (floor not binding) | 2.39 h | 4.19 | 1.58 y | 90.3 | 111 / 119 |

These include the disk's 25 g of cold gas. The Ceres loop under the floor: **$1.488/kg** at
4 R_sun (was $1.463) and **$1.741** at 9.86 R_sun (was $1.696). The $1 line ($0.997) is
unchanged.

## Item 1. The floor as a constraint

`Design.disk_floor` defaults to 1 kg (`DISK_FLOOR`). The optimizer projects any arc that
puts a ship under the floor onto the floor's arc, and penalises the cut. The floor binds on
the **departure-wave ship**, the one your harness checked. The push-wave ship then flies
1.70 kg a pulse on the same arc.

## Item 2. Everything rerun

- The headline table is above.
- **Gap, 3 / 5 / 10 days** (dear): 9.86 R_sun $177 / $158 / $158; 4 R_sun $89 / $87 / $89.
- **Depth dial:** 4 R_sun, X 90 is still the cheapest closure. 4 R_sun, X 60 cannot carry
  1 kg at any arc of 0.5 h or more.
- **Eccentric delivery:** $156 dear at 9.86 R_sun and $87 at 4 R_sun.
- **Jupiter-only loop:** unchanged, since it flies no disks.
- **Best case:** the steady optima land at k_E 5.5-6.2. That is an interpolation artefact
  of the bowl law (ADR 0085's last section), so the S53 figures are not yet restated here.

## Item 3. The Sun plate at 1 kg

It is inside what it was fit to. `SUN_PLATE_PULSE` is rated at 1 kg and 400 km/s.

- **At 9.86 R_sun** a 1 kg pulse is 0.96 of that rating: the anchor itself. The old
  0.2 kg pulse was the extrapolation.
- **At 4 R_sun** it is 2.4 times the rated energy, against 1.3 before. This is a scaling,
  not a fit.
- The survival floor binds, not the spring (strokes 2.9 and 2.2 m).
- Plate, bags and collar grow from 6.3 to 16.7 t per unit at 9.86 R_sun. The ship's peak on
  its spring goes from 0.08 to 0.40 g.

## Item 4. Hardware and gas

- The hardware is impactor mass, as you say. The stage law is already per kg of disk.
- **Carry the gas.** It is now charged: launched, turned by the carrier, priced as storable
  propellant. It costs $0.1-0.3 of break-even.

## Item 5. Stacking: not ledger-neutral

Five units on one Sun ship share its plate and collar. The survival floor goes as the pulse
to the 2/3, so each unit carries a smaller share of plate.

- At 9.86 R_sun that gives **$143 dear** with ADR 0077's design held, against $158 for the
  harder boost and $146 unfloored.
- At 4 R_sun it gives $85 against $89.

Not adopted: docking is not designed, and the seed must be five units (break-even
unchanged, outlay ×5).

## Open: the bowl law's two basins

The departure bowl's v_e is a product of two interpolations between k nodes 5, 10 and 20.
It humps between nodes, so the optimizer has two basins either side of k = 10.

- The headline sits at k_E ≈ 12.5, as ADR 0077 recorded.
- A k_E ≈ 7.4 optimum is $7 cheaper dear at 9.86 R_sun.

The figures above keep the recorded basin. Resolving it wants the simulation's k = 7.5 and
15 nodes.
