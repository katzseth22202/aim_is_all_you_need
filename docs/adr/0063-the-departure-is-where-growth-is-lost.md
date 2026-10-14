# ADR 0063: The departure is where growth is lost; rank its levers, and ask for hot hydrogen

Date: 2026-10-14

Status: Accepted (a ranking and an ask; no headline changes). The hot-hydrogen
rows are unsolved until the impact simulation answers
`docs/hydrocode_asks_2026-10-14_hot_hydrogen.md`.

## Context

The author asked (2026-10-14) why a push that puts about ten times its PuffSat
mass into the parking orbit grows the fleet only 1.3-3.0 times a cycle. Here is
the ledger's cycle 0, for the methane chamber (AR100, 24 kg pitch) behind the
spray cup:

| Step | Mass | Factor |
|---|---|---|
| Lofted to 400 km | 1500 t | |
| Push: 100 t of PuffSats, 471 t of argon | 1023 t | 10.2 t per PuffSat t |
| Plate, argon tank, methalox reversal | 805 t departing | ×0.79 |
| **Departure: 7.30 km/s plus 0.84 km/s loss at about 6.9 km/s exhaust, then 56 t of chamber and 12 t of tanks expended** | **178 t onward** | **×0.22** |
| Rods (22 t) also count against the PuffSats spent | | ×0.82 |
| Growth | 178 / 122 | 1.46 |

The departure is the largest loss by far. Its 7.3 km/s is the next window's
burn, two-synodic here. The chamber's effective exhaust is only about 7 km/s (720
s), because each 5 kg rod throws out 28-33 kg of charge, plug, pitch and film at
efficiency 0.47.

**Flying only three-synodic cycles does not help.** Their departures are 5.3-6.2
km/s, but the 3.28 yr clock cancels the gain:

| Chamber | Flown chain | Three-synodic only |
|---|---|---|
| Methane | 3.36 yr | 3.45 yr |
| Hydrogen | 2.84 yr | 2.92 yr |

## The ranking (`src/departure_levers.py`, `make departure-levers`)

Each lever is flown through `chain_growth` on the flown chain behind today's push.

| Lever | Doubling (yr) | Annual | 10-yr | Mean finite-burn loss | Burn |
|---|---|---|---|---|---|
| Methane, baseline | 3.36 | 22.9% | 6.5× | 0.67 km/s | 32 min |
| Methane, 3 Hz | 3.02 | 25.8% | 8.4× | 0.39 | 22 |
| Methane, two chambers | 4.70 | 15.9% | 3.4× | 0.34 | 20 |
| Methane, 3.5 kg pitch | 3.26 | 23.7% | 7.0× | 0.78 | 37 |
| **Methane, 3 Hz + 3.5 kg pitch** | **2.91** | 26.9% | 9.3× | 0.46 | 25 |
| Hydrogen (bulged, 4884 K), baseline | 2.84 | 27.7% | 9.6× | 1.05 | 48 |
| Hydrogen, 2.4 Hz | 2.60 | 30.6% | 12.2× | 1.01 | 46 |
| Hydrogen, two chambers | 3.01 | 25.9% | 8.1× | 0.64 | 31 |

**Hot hydrogen, unsolved.** This is the bulged chamber held at 7000 K or 10 000 K:
- **The charge.** Cantera `gri30` equilibrium at 0.57 kg/m3 gives 242 and about
  360 MJ/kg, against 111. That holds the temperature with k = 9.5 or 5.5,
  against 24.
- **The efficiency** is scaled by the chemistry ceiling, `1 - 0.60 (1 - returned)`.
- **Everything else** (wall, tanks per kg of hydrogen) is held at the bulged
  chamber's.

Doubling in years:

| Bonds returned | 7000 K at 2 / 3 / 4 Hz | 10 000 K at 2 / 3 / 4 Hz |
|---|---|---|
| 94% | 2.48 / 2.21 / 2.03 | 2.79 / 2.29 / 2.15 |
| 70% | 3.51 / 2.84 / 2.53 | 4.62 / 3.13 / 2.76 |
| 50% | 7.49 / 4.40 / 3.59 | 49.8 / 5.71 / 4.28 |

## Decision

1. **Pulse rate is the cheapest lever.** For methane, 3 Hz (the 2 t/s fill ADR
   0056 priced) takes doubling from 3.36 to 3.02 yr. Light pitch takes it to 2.91
   yr. A second chamber always loses: its 46-56 t outweighs the shorter burn.
2. **Hot hydrogen is the biggest prize and the least certain.** A pulse's energy
   is fixed by the rod and momentum goes as `sqrt(2 m E)`. So a hotter, thinner
   charge gives a faster exhaust but less impulse per pulse: about 280 kN s at
   10 000 K against 657. It pays only if it also cycles faster.
   - 7000 K beats 10 000 K at every rate and recovery tried.
   - 7000 K, 94% returned, 4 Hz doubles in **2.03 yr**.
   - At 50% returned, neither temperature pays at any rate tried.
3. **Ask the impact simulation**
   (`docs/hydrocode_asks_2026-10-14_hot_hydrogen.md`) for the hot chamber's:
   - peak state;
   - finite-rate recombination and impulse;
   - wall heat and strain;
   - sustainable pulse rate (as important as the recombination);
   - the best temperature, by impulse per kilogram carried.
4. **No headline changes.** The levers depend on the impact simulation's
   answers: 3 Hz methane needs its wall heat checked (ADR 0056), and the hot rows
   are unsolved.

## Limits

- The hot rows hold the wall mass and its survival at the bulged chamber's, and
  the plug at P = 1.76. Pressure is estimated to stay near today's because the
  charge shrinks, but it is not solved.
- Cantera's fits are extrapolated above about 6000 K. The 10 000 K energy takes
  the middle of 340-390 MJ/kg, checked against 216 MJ/kg of dissociation plus
  about 124 MJ/kg of atomic thermal energy.

## Reproduction

`make departure-levers` (about 40 min on 2 workers, 26 cases). The breakdown
table and the three-synodic comparison came from the same ledger
(`best_cycle_growth` on cycle 0; `chain_growth` on `chains()[1]`). Fast tests:
`tests/test_departure_levers.py`.
