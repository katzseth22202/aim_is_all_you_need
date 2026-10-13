# ADR 0055: Double the methane throat, drop the gas cushion, bulge the hydrogen

Date: 2026-10-13

Status: Accepted. Takes the impact simulation's answers to
`docs/hydrocode_asks_2026-10-12.md` (B1-B6), `puffsat_impact_simulation` @ `7afdba3`: its
ADR-0056 with the 2026-10-13 amendment, `docs/hydrocode_asks_handoff.md` and
`docs/walled_nozzle_hydrocode_asks.md`. Amends ADR 0044 and ADR 0053; rejects ADR 0054.

## Context

ADR 0053 found that the 212 m3 survivable methane chamber fits 2 Hz with 164-222 ms to
spare, and that 4 Hz would need a smaller chamber. ADR 0054 proposed a gas-cushioned wall
that might let a 53-106 m3 chamber survive. Both went to the impact simulation, along
with the hydrogen chamber's geometry and rate, wall loads by zone, and impulse per pulse
against volume.

## What the impact simulation found

- **The blowdown clock holds at its fast end:** 281 ms to fill pressure, solved on the real
  equilibrium gas (ADR 0053: 278-336 ms).
- **The cycle does not close as ADR 0053 assumed.**
  - The throat film cannot be laid across a sonic jet. It seals only once the flow
    fades, and it traps a ~3 kg, ~4,100 K residue that joins the next charge.
  - A polymer film under its 400 °C rating must wait for ~0.3 bar. On the 1.33 m2 throat
    that runs 548 ms, 48 ms past the 2 Hz period.
  - The film weighs kilograms per shot, not grams: its mass goes as `p r^3`.
- **A smaller chamber fails the wall.** Unbulged, the spike beside the plug grows 5-8x
  (2.3-3.5 GPa) and no dry wrap survives. Stand-off is the only lever on the spike.
- **The gas cushion fails.**
  - The solved spike is 9-77 kPa s, not ADR 0054's 2-5, and it is uneven over a few
    centimetres.
  - The tiles bend at 0.9-8 GPa, and none of 390 cases lasts 4,200 pulses (the best ~190).
- **The parent's impulse was gross.** `tab:one_chamber`'s 906 / 971 kN s charged neither
  the 1% port gate nor the H + H + M nozzle freeze. Corrected, they are 861 / 892 kN s on
  the 1.33 m2 throat.
- **Doubling the throat closes the cycle.**
  - Blowdown time goes as `V / A*`. At 2.66 m2 the chamber reaches fill pressure in
    139 ms and leaves 149 ms of the 2 Hz period with the film sealed at 0.3 bar.
  - It recovers slightly more chemistry: 868 / 899 kN s, since a longer nozzle at the same
    area ratio gives recombination more time.
  - The cone is 1 m shorter and the dry wrap passes at 46.2 t. Re-solved, the 1.33 m2
    chamber needs 52.4 t.
  - The costs: a 2.6 kg aramid film every shot, and exits of ~18 m (A/A* = 100) and
    ~32 m (300).
- **The r 3.0 m bulge holds hydrogen.**
  - Charge and rod: the sphere's 60 kg charge on the 2.5 kg rod, 106 m3, 4,884 K,
    142 bar.
  - Wall: a 37.0 t dry wrap, which passes at 2.20% flung strain against Kevlar's 2.4%,
    a 1.09x margin.
  - Impulse: 656-658 kN s at A/A* = 300 on a 0.75-1.0 m2 throat, against the sphere's
    697.
  - Rate: ~2-3 Hz, not 4. A doubled hydrogen throat fails the wrap.

## Decision

1. **The methane chamber's throat doubles to 2.66 m2** (`chamber_isp.SURVIVABLE_THROAT_AREA`).
   It flies:
   - a 46.2 t wall;
   - 868 / 899 kN s per pulse, which gives `eta` 0.459 / 0.483;
   - extensions of 9.8 t and 29.4 t, scaled by exit area from 4.9 t and 14.7 t;
   - `SURVIVABLE_THROAT_FILM` = 2.6 kg a shot, charged with the pitch as carried mass that
     leaves in the exhaust.

   ADR 0053's 1.33 m2 throat and ADR 0044's impulses stay in the code as `_S17` constants
   for the record.
2. **ADR 0054 is rejected.** `src/wall_cushion.py` stays as the record of the estimate.
3. **`chamber_isp.HYDROGEN_BULGED`** is the bulged hydrogen chamber, with:
   - a 0.84 m2 throat and a 37.0 t wall;
   - a 9.3 t extension, the sphere's scaled by exit area;
   - 657 kN s per pulse (`eta` 0.773), flown at 2 Hz.

   It is the first hydrogen chamber with a solved wall. The sphere's row stays as the
   record, its wall unresolved at ~2.95 GPa. `make survivable-chamber`,
   `make hardware-recovery` and `make slush-hydrogen` fly it beside the sphere.
4. **The survivable ledger reports the most pulses any one chamber fires** (`max/chamber`),
   against the impact simulation's ~4,200-pulse wall life (`WALL_LIFE_PULSES`). It also
   flies two 5 kg chambers as a wall-life sensitivity.

## Results

Growth ledger, spray cup, 20-day orbit, fixed-direction loss integrated
(`make survivable-ledger`). Methane at 24 kg pitch, doubling time:

| step | A/A* = 100 | A/A* = 300 |
|---|---:|---:|
| a. ADR 0044 (906 / 971 kN s, 48.2 t, 4.9 / 14.7 t extension, no film) | 3.06 yr | 2.78 yr |
| b. impulse corrected on the 1.33 m2 throat (861 / 892) | 3.54 | 3.48 |
| c. the 1.33 m2 chamber as re-solved (52.4 t wall, 0.93 kg film) | 3.70 | 3.63 |
| d. 2.66 m2: 868 / 899, 46.2 t, 2.6 kg film, extension *not* rescaled | 3.41 | 3.35 |
| **e. ADR 0055 (d with the extension doubled)** | **3.57** | **3.86** |

- **Most of the loss is the impulse correction (a → b), 0.5-0.7 yr.** It owes nothing to the
  throat: the parent's 906 / 971 kN s were simply gross.
- **Against the 1.33 m2 chamber as re-solved (c), doubling the throat gains 0.13 yr at
  A/A* = 100.** It loses 0.24 yr at 300, through the extension alone. The 1.33 m2 chamber
  does not close 2 Hz in any case.
- **A/A* = 100 now beats 300, by 0.29 yr** (3.57 against 3.86). This reverses ADR 0044.
  At 300 the extension is 29.4 t of a 75.6 t chamber. The extension's mass is the
  scaling ADR 0044 used (a thin shell at fixed areal density, by exit area). Nobody has
  sized a 32 m carbon-carbon extension. Lighter construction would bring 300 back, since
  step d shows it 0.05 yr ahead without the extra mass.
- **Methane, 3.5 kg pitch:** 3.46 yr (A/A* = 100), 3.71 yr (300).
- **The maraging fallback wall** gives 4.62 / 5.09 yr. **The steered loss**, which a
  head-on chamber cannot fly, gives 3.37 / 3.61 yr.
- **Redundancy.** Two 2.5 kg chambers cost 0.06 yr at A/A* = 100 (3.62 yr). At 300 they
  come out 0.14 yr ahead (3.73 yr), flying one half-size chamber on some cycles.

**Burn length and wall life.**
- Methane burns run 25-41 min, 3,040-4,880 pulses on one chamber at 2 Hz. A 2,500 s burn
  is the longest cycle's.
- The longest cycles fire 4,628 (A/A* = 100) to 4,877 (300) pulses on one wall, 10-16%
  past the ~4,200 the impact simulation sized it for.
- That figure was "one departure of a 500-600 t stack". The ledger's departing stacks
  are now 790-860 t.
- Splitting the burn across two 5 kg chambers keeps each under the limit (2,950-3,190
  pulses), but costs 1.6-3.6 yr of doubling (5.15 / 7.42 yr): the second wall and
  extension outweigh the shorter burn.
- The wall's rating has to rise to ~5,000 pulses instead. That is an ask for the impact
  simulation.

**Cost** (Estimate book, lob climbing, 10% a year, 50% odds; `make survivable-cost`):
- Methane steady cost is $172 / $178 per kg at 24 kg pitch (A/A* = 100 / 300), against
  ADR 0044's $149 / $136.
- Priced by chamber mass against the sphere, it is $203 / $229, against $171 / $160.
- Dear-seed steady break-even is $1254 / $1461, against $886 / $710.

**Bulged hydrogen, liquid, 2 Hz** (`make survivable-cost`):
- **The burn reaches the loss table's one-hour edge.** G_0 is 1.56 at 2 Hz on one
  chamber, and the long cycles' burns sit at 3,598 s, the edge of the integrated loss
  table.
  - The best single-chamber stack is therefore bounded by the table, not found.
  - Those cycles fire 7,196 pulses on one wall.
- **Two chambers:** each fires ≤4,422 pulses on 20-37 min burns, and the chain doubles in
  3.28 yr. One chamber doubles in ~2.96 yr, still against the table edge.
- **Cost:** $175/kg steady on the Estimate book, against the sphere's $122.
- **Hydrogen's lead shrinks.** The bulge's 6% lower impulse and its 2 Hz, against the
  sphere's 4 Hz, leave liquid hydrogen level with methane before any recovery.

- **Stored as slush with recovered tanks** (`make slush-hydrogen`): growth is 2.236 a cycle
  at $135/kg, against the sphere's 2.877 at $105, or 2.597 at $116 at 2 Hz.
- **Recovered hardware** (`make hardware-recovery`, which now flies methane at A/A* = 100,
  the leading ratio):

| recovery | CH4 AR 100 (priced by mass) | H2 bulged | H2 sphere |
|---|---:|---:|---:|
| none | $203 | $175 | $122 |
| rods only | $161 | $147 | $110 |
| rods + plate | $152 | $126 | $105 |

So hydrogen's lead narrows from ~$50-80/kg to ~$10-25/kg once its wall is the one the
impact simulation can support.

## Consequences

- **The parent's methane rows need the corrected impulse.** Its chamber-count, burn and
  cost figures follow `docs/hydrocode_answers_for_parent.md`.
- **Lead with A/A* = 100** unless a lighter 32 m extension is argued.
- **Ask the impact simulation for a ~5,000-pulse wall life,** or a stack held near
  600 t. Ask too for the 0.25 cm check of the hydrogen bulge, whose fling margin is
  1.09x.
- **The hydrogen performance case** (ADR 0051, `docs/orbit_turn_for_parent.md` item 9)
  rested on the sphere at 4 Hz, whose wall is unresolved. With a solved wall, hydrogen
  leads methane by $10-25/kg with recovered hardware, not by $50-80.
- **Extending the loss table** past an hour would let the single bulged hydrogen chamber
  find its own optimum. It would also fire far past any wall life, so this is not urgent.

## Limitations

- The impulses are 0-D real-gas blowdowns. The wall loads are gamma-law 2-D solves whose
  peaks rose 15-25% per grid halving.
- The film, fill and lock times are the impact simulation's assumptions.
- The extensions scale with exit area at fixed areal density.
- The bulged hydrogen chamber's impulse is interpolated at 0.84 m2, between the solved
  0.75 and 1.0 m2. Its cycle rate at that throat is ~2.2-2.4 Hz, and 2 Hz is flown.

## Reproduction

- **Ledger and cost book:** `make survivable-chamber`.
- **Bulged hydrogen as slush and with recovered tanks:** `make slush-hydrogen`,
  `make hardware-recovery`.
- **Tests:** `pytest tests/test_chamber_isp.py tests/test_survivable_chamber.py` pins:
  - the 868 / 899 kN s inversion;
  - the halved blowdown;
  - the bulged hydrogen chamber;
  - the film in the carried mass.
- **Steps a-e** were a scratch script (gitignored). To redo them:
  - fly `survivable_design(ar, 24 kg)` through `ledger_row`;
  - substitute each step's wall, extension and `efficiency_for_impulse` impulse as in the
    table;
  - charge the film as pitch: 24 + 0.93 kg on step c, 24 + 2.6 kg on steps d and e.
