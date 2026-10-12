# ADR 0052: The 3000 K hydrogen chamber does not close on the paper's chain

Date: 2026-10-12

Status: Accepted. The 3000 K chamber stays a study point, not a design.

## Context

ADRs 0045 and 0049 studied a 3000 K hydrogen chamber. That is the temperature NTR
fuel and walls were tested to, and at 3000 K hydrogen is almost fully molecular.
Those studies flew it only on the circular 3S return at 5.8 km/s. They did not
charge tanks, chamber or cryostats, and they reported delivered fraction, not
growth. This ADR puts the chamber on the paper's flown chain, with every launch-unit
charge, with slush storage (ADR 0051) and with recovered tanks (ADR 0050).

## The pairing

`chamber_isp.HYDROGEN_3000K` is built as follows.

- **Energy.** The equilibrium internal energy at the charge's own density is
  34.33 MJ/kg (Cantera `gri30`).
- **Charge.** It is sized like every pairing, `(k + 1 + P) u = w^2/2` at 75 km/s
  with the 1.76 plug. That gives **k = 79.2**: 198 kg of hydrogen in 20 m3, at a
  peak of 1,220 bar. The same solve gives 806 bar for the 5500 K pairing, against
  the paper's 818.
- **Wall.** A thin wall's mass goes as `p V`, and that is fixed by the charge
  whatever the volume. So the 32 t copper-lined wall scales by 1220/818, to 47.7 t.
- **Ceiling.** It is 0.9988: about 2% of the energy is in bonds.
- **Efficiency.** It is a guess. The primary case is 0.7225 (85% of ideal exhaust
  velocity, the ADR 0045/0049 convention). The sensitivity uses 0.858, the 5500 K
  chamber's solved value.

The pulse rate is held at 4 Hz, as ADR 0049 required.

## Result

`make slush-hydrogen`, spray cup, lob climbing, Estimate book:

| Design | Growth, geometric mean | Weakest cycle |
|---|---:|---:|
| 3000 K, liquid | 0.830 | 0.549 |
| 3000 K, slush, tanks recovered | 1.028 | 0.700 |
| 3000 K, slush, eta 0.858, tanks recovered | 1.360 | 0.956 |
| 5500 K, slush, tanks recovered (for comparison) | 2.877 | 2.282 |

None of the 3000 K cases holds a fleet level. Every one has cycles below
`G = 1`, so it has no steady state to price.

## Why

Efficiency is not the cause. The cause is the energy per kilogram the temperature
cap allows. Effective specific impulse, with the gate charged:

| Chamber | Ieff at 60 / 68 / 75 km/s |
|---|---:|
| 5500 K, eta 0.858 | 1060 / 1090 / 1111 s |
| 3000 K, eta 0.7225 | 606 / 617 / 625 s |
| 3000 K, eta 0.858 | 671 / 682 / 690 s |
| 3000 K, eta 1.0 (perfect) | 734 / 744 / 752 s |

- **Energy per kilogram.** At 5500 K a kilogram of charge holds 105 MJ, about a
  third of it in dissociation, and the A/A* = 300 nozzle returns 94% of that store.
  At 3000 K a kilogram holds 34 MJ. That is a third of the energy, so about 57% of
  the exhaust speed.
- **The momentum debit.** The rod's head-on momentum debit `w / (k + P)` is smaller
  per kilogram at `k = 79`. It cannot make up the lost exhaust speed.
- **Even a perfect chamber falls short.** At eta = 1.0 a 3000 K chamber reaches
  about 750 s. That is the methane chamber's specific impulse, carried with
  hydrogen's tank: 0.18-0.205 kg of tank per kg against methane's 0.034, plus
  cryostats.
- **The chain's hardest cycles.** The two-synodic cycles need about 7 km/s at
  200 km, and there it does not grow.

The recombination store that 3000 K was chosen to avoid is a third of what makes
the 5500 K chamber work.

The ADR 0045 tables were not wrong. They answer a narrower question: a gentler
burn, with the tank, chamber and cryostat not charged.

## Consequences

- NTR materials heritage does not transfer into a competitive pulsed hydrogen
  chamber at NTR temperature. The paper should say so, briefly, where it raises the
  idea.
- If a cooler hydrogen chamber is wanted, the question becomes the lowest
  temperature at which the chain grows. A temperature sweep would answer it. It
  also needs the dissociation store to be recovered by a long nozzle again, so
  ADR 0049's short-nozzle and fast-blowdown benefit shrinks as temperature rises.
- `HYDROGEN_3000K` stays in `chamber_isp` as the study point, and
  `make slush-hydrogen` prints it.

## Reproduction

`make slush-hydrogen`. `pytest tests/test_chamber_isp.py` pins the charge
against `orbit_turn_analysis.hydrogen_hot_energy`.
