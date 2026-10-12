# ADR 0054: A gas-cushioned wall could take the spike

Date: 2026-10-12

Status: Proposed. A one-degree-of-freedom estimate; the impact simulation has to
confirm it.

## Context

Two things hold the survivable methane chamber at 2 Hz (ADR 0053).

- **Its bulge.** The chamber is bulged to 212 m3, and blowdown time goes as volume
  over throat area. Shrinking it to 106 or 53 m3 would open 4 Hz.
- **Why it is bulged.** The bulge exists so the wall survives the blast. The wall
  sees a spike of a few kPa s in 2-4 microseconds on top of the steady ~94 bar. The
  spike drives a hoop overshoot `D`, the solved shell's peak strain over its static
  strain, of 1.8 to 5.7 by zone (parent `tab:dry_wrap_zones`).

Thickening the wall locally is already the design: the dry Kevlar wrap varies from
110 to 207 mm. Thickening further costs mass, and it does nothing for the steel's
cyclic hoop stress. A thicker solid shell also loses the 3.4-microsecond wave
cancellation that protects the 10 mm one.

Orion carried its pulse loads through gas-bag shock absorbers. The same idea applied
to the chamber is an inner tile on a sealed gas cushion, backed by a strong backwall.

## Decision

Add `src/wall_cushion.py` (`make wall-cushion`).

**The model.**
- An inner tile of areal mass `mu` rides on an adiabatic gas layer of thickness
  `d`, precharged to `p0`. It sits on inner stops until chamber pressure beats the
  precharge.
- The spike only gives the tile a speed `I / mu`.
- The steady pressure rises over 1 ms and decays with the chamber's 64 ms blowdown
  e-fold time.
- Orifice damping `c v` between cushion cells also loads the backwall.

**Results** for a 10 mm steel tile (80 kg/m2):

| Cushion | Precharge | Spike | Damping, Pa s/m | Tile travel | Backwall peak / 94 bar |
|---:|---:|---:|---:|---:|---:|
| 6 cm | 80 bar | 2 kPa s | 0 | 1.6 cm | 1.33 |
| 6 cm | 80 bar | 2 kPa s | 1e5 | 0.8 cm | 1.12 |
| 6 cm | 80 bar | 5 kPa s | 0 | 1.9 cm | 1.43 |
| 6 cm | 80 bar | 5 kPa s | 1e5 | 1.3 cm | 1.52 |
| 8 cm | 60 bar | 2 kPa s | 1e5 | 2.5 cm | 1.13 |
| 8 cm | 60 bar | 5 kPa s | 0 | 3.1 cm | 1.26 |

**What it buys.**
- **Design load.** The backwall's design load drops from 1.8-5.7 times the static
  load to 1.1-1.5 times, with 1-3 cm of tile travel. The backwall sees milliseconds,
  not microseconds, so it neither spalls nor flings.
- **Hydrogen cracking.** The tiles carry no hoop tension, so the failure that ended
  solid Cr-Mo, crack growth in hydrogen under cyclic hoop tension, largely goes away.
- **Volume.** The cushion sits outside the gas volume and adds about 2% to the
  backwall's radius.

**Sizing still matters.** A 1 cm cushion at 20 bar slams the tile into the backwall
at over 100 times the steady load.

## Open, for the impact simulation

- **Spike shape.** The real spike is not uniform: it is a reflected-shock pattern by
  zone.
- **Seals.** Tile seals must keep hot gas out of the cushion and cushion gas in, over
  1-3 cm of travel and 4200 pulses. Candidates are sliding labyrinth seals, a
  corrugated foil skin, or a methane precharge whose leak joins the next charge.
- **Moving tiles.** The regen passages and the pitch layer now ride on moving tiles.
- **Smaller chambers.** The main question is whether a cushioned wall lets a 53-106
  m3 chamber survive, which would open 4 Hz (ADR 0053).

The request is written up in `docs/hydrocode_asks_2026-10-12.md`.

## Reproduction

`make wall-cushion`. `pytest tests/test_wall_cushion.py`.
