# The parking orbit is the split, and twenty days it is

Status: accepted

Amends: ADR 0033. Its decisions stand; its figures mixed a 10-day split with
20-day burns and are replaced by the ones below.

Date: 2026-09-30

## Context

CONTEXT.md is explicit that the split gap *is* the parking-orbit period: the
growth wave pushes the payload at periapsis, the payload coasts one full orbit
while the departure wave catches up, and it departs at the next periapsis. The
growth ledger of ADR 0033 broke that rule. It flew the chain's default 10-day
split (so each growth wave paid only the 10-day early-arrival burn) but priced
the push target, the departure, the loss tables and the apoapsis reversal on the
20-day orbit (so it paid only the 20-day reversal). That took the cheaper side of
both trades, and its figures were optimistic.

## Decision

**1. The parking orbit is read from each cycle's own split** (`parking_period`).
The push target (the orbit's 400 km periapsis speed), the periapsis raise to
600 km (now computed; it reproduces the parent's 1.7 m/s at 20 days), the
apoapsis reversal, the departure's starting speed, the fixed-direction and
steered loss tables (now cached per orbit) and the hydrogen hold (0.3%/day of
boil-off over one parking orbit) all follow from that one number. The two cannot
drift apart again.

**2. The ledger flies twenty days** (`DEFAULT_PARKING_DAYS`, decided with the
user). Flown consistently, it beats ten:

| Plate 0.7 | 10 days, consistent | 20 days, consistent | ADR 0033 (mixed) |
|---|---|---|---|
| Apoapsis reversal | 372.5 m/s | 233.9 m/s | 233.9 |
| Periapsis raise | 2.72 m/s | 1.72 m/s | 1.7 |
| Growth wave's early-arrival burn (mean) | 179 m/s | 464 m/s | 179 |
| H2 at 50% of ceiling | 3.44 yr | **3.43** | 3.17 |
| H2 solved (88%) | 1.54 | **1.54** | 1.48 |
| CH4 at 50% | 3.38 | **3.30** | 3.14 |
| CH4 solved (80%) | 1.70 | **1.68** | 1.64 |
| Methalox 380 s | 4.30 | **3.91** | 3.91 |

For the chambers the cheaper reversal and the dearer early-arrival burn nearly
cancel (0-2%, as ADR 0013 found for its own nozzle ledger). Methalox has no
departure wave and so no split to buy, and keeps the whole reversal saving (9%).
Twenty days is also the orbit the parent's 600 km figures are quoted on
(10.785 km/s, 1.7 m/s, 613 000 km). `two_wave_growth.DEFAULT_SPLIT_DAYS` stays 10
for that module's own reports.

## What the numbers say

Doubling time in years at 20 days, k held to 10 (uncapped in parentheses).
Chambers at a share of their chemistry ceiling (ADR 0032), gated, methane at
5.6 kg of pitch per pulse, hydrogen with 3% cryostats and 6% boil-off over the
20-day hold. Plate efficiency net of chemistry, argon on ice PuffSats.

| Departure | Plate 0.5 | Plate 0.7 | Plate 1.0 |
|---|---|---|---|
| H2 50% | 4.5 (4.18) | 3.43 (3.13) | 2.72 (2.42) |
| H2 70% | 2.16 (2.08) | 1.87 (1.77) | 1.63 (1.51) |
| H2 solved (88%) | 1.73 (1.68) | 1.54 (1.47) | 1.37 (1.28) |
| H2 90% | 1.7 (1.65) | 1.51 (1.45) | 1.35 (1.26) |
| H2 100% | 1.58 (1.54) | 1.42 (1.36) | 1.27 (1.2) |
| CH4 50% | 4.28 (3.85) | 3.3 (2.92) | 2.65 (2.28) |
| CH4 70% | 2.21 (2.1) | 1.91 (1.78) | 1.67 (1.51) |
| CH4 solved (80%) | 1.91 (1.83) | 1.68 (1.58) | 1.49 (1.37) |
| CH4 90% | 1.75 (1.68) | 1.55 (1.46) | 1.38 (1.27) |
| CH4 100% | 1.62 (1.56) | 1.45 (1.38) | 1.3 (1.21) |
| Methalox 380 s (3S only) | 5.27 (4.97) | 3.91 (3.59) | 3.01 (2.68) |

1. **Both chambers still beat methalox at every efficiency swept**, now by 10-19%
   at 50% of ceiling (behind a 0.7 plate: 3.43 and 3.30 years against 3.91) and
   about twice as fast from 70% up. ADR 0033's mixed figures had the 50% margin at
   about 20%.
2. **At 50% of ceiling methane edges hydrogen by about 4%** (3.30 against 3.43
   years behind a 0.7 plate). It did so narrowly in ADR 0033's figures too
   (3.14 against 3.17), which that ADR called a tie, and at 10 days (3.38 against
   3.44). From 70% up hydrogen leads by 2-3%, and the solved chambers are 1.54
   (H2) against 1.68 (CH4) years.
3. **Argon still beats water on the plate by 8-11%** (`make plate-slug`, solved
   chambers):

   | Plate | Departure | Water on ice | Argon on ice | All argon |
   |---|---|---|---|---|
   | 0.5 | H2 0.858 | 1.88 | 1.73 | 1.71 |
   | 0.7 | H2 0.858 | 1.68 | 1.54 | 1.52 |
   | 1.0 | H2 0.858 | 1.50 | 1.37 | 1.35 |
   | 0.7 | CH4 0.538 | 1.87 | 1.68 | 1.66 |

4. **The hold matters a little more at twenty days.** The solved hydrogen chamber
   behind a 0.7 plate runs from 1.49 years (no cryostat, no boil-off) to 1.57
   (5% cryostats, 0.5%/day).

## Consequences

- ADR 0033's decisions and model stand; quote this ADR's figures.
- Tests pin the rule directly: `test_the_parking_orbit_is_the_cycles_own_split`
  checks the push, raise, reversal and departure at 10 and 20 days, and
  `test_raising_periapsis_at_apoapsis_costs_the_parents_1_7_m_s_on_the_20_day_orbit`
  pins the raise.
- The chain's burns are still *defined* from the 20-day orbit's 200 km periapsis
  speed (`two_wave_growth._cycle_periapsis_speed`), so the excess speed is taken
  from there and the departure then starts from the orbit actually flown.

## Considered and rejected

- **Ten days everywhere.** Consistent, but 0-2% slower for the chambers and 9%
  slower for methalox, and off the orbit the parent's 600 km figures use.
- **Keeping ADR 0033's mixed figures.** They took the cheap side of both trades.

## Reproduction

`make growth-ledger` and `make plate-slug` at the committing revision (both now
default to `--split-days 20`); `make growth-ledger` with `--split-days 10` gives
the 10-day column. The comparison table's 10- and 20-day columns come from the
same functions on `adaptive_two_wave_cycles(split_days=10 or 20)` with the
methalox chain at `threshold_m_s=0`. Settings otherwise as in ADR 0033.
