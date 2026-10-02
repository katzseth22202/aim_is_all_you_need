# Each payload departs on the next window's burn

Status: accepted

Amends: every ADR that prices a departure on the flown chain: 0013 and 0015
(the two-wave and two-leg ledgers), 0032 (the chamber departure), 0033 (the
growth ledger), 0035/0036 (the seed and the harvest), 0037 (the growth cost
model). Their structure stands; their growth figures move.

Date: 2026-10-01

## Context

A `TwoWaveCycle` describes one trajectory. The batch leaves Earth at
`departure_jd` with `departure_burn`, and its two waves come home around
`return_jd`, which is also the next cycle's `departure_jd`. The waves of cycle
`n` push a payload that therefore leaves on **cycle n + 1's** trajectory.

Every ledger that priced that payload's departure used **cycle n's own**
`departure_burn`: `growth_ledger.price_cycle_growth`,
`growth_ledger.price_methalox_cycle`, `chamber_departure.price_chain_departures`,
`two_wave_growth.price_cycle` and `two_leg_nozzle_sweep`. Found while answering
the parent's ask G3 (ADR 0037).

The error is large where the cadence changes. A three-synodic outbound needs
about 5.3 km/s above the closed-cycle speed, a two-synodic one about 7.0. So a 3S
return feeding a 2S departure was undercharged by up to 1.84 km/s (cycle 0
feeding cycle 1), and a 2S return feeding a 3S departure was overcharged by up
to 1.68 km/s.

## Decision

1. **`TwoWaveCycle.onward_burn`**: the departure burn the payload pushed by
   this cycle's waves must fly, which is the next window's `departure_burn`.
   `adaptive_two_wave_cycles` fills it with `link_onward_burns`. The last flown
   cycle takes **the window after the horizon** (`departure_burn_after`, the
   same 2S-unless-too-dear policy), not a wrap to cycle 0.
2. **Every departure is priced on `onward_burn`.** The waves' own speeds,
   corrections and parking orbit stay cycle `n`'s, because those are the waves
   that arrive.
3. `sep_split_correction` links its converted cycles the same way.
4. Indexing is unchanged. `G_n` is still the growth of the unit pushed at
   return `n`, and the batch at return `n` is still `prod(G[:n])`.

The **seed** is unaffected. It leaves on cycle 0's own burn
(`seed_excess_speed(flown[0])`), which was always right.

## What moved

Behind a 0.7 plate, flown chain unless noted. "Before" is what the ADRs and the
parent recorded.

| Figure | Before | After |
|---|---|---|
| Solved hydrogen: annual growth / 10-yr stepwise multiple / chambers built in 10 yr | 57.0% / 104.2 / 44.3 | 54.8% / 75.1 / 32.8 |
| Solved methane | 51.0% / 69.5 / 34.4 | 48.8% / 49.7 / 25.3 |
| Methalox (3S chain): annual / 10 yr / Raptors built | 19.4% / 5.09 / 12.5 | 18.5% / 5.07 / 10.7 |
| Seed, one launch unit (methalox / solved methane / solved hydrogen) | 72.7 / 80.8 / 80.7 t | 72.7 / 83.1 / 83.3 t |
| Cycle 0 growth `G_0` (methalox / CH4 solved / H2 solved / CH4 50% / H2 50%) | 2.07 / 3.54 / 3.93 / 2.25 / 2.34 | 1.66 / 2.46 / 2.76 / 1.31 / 1.22 |
| `tab:seed_return` at $500, IRR % (liquidation; steady), cheap / dear seed: methalox | 45 / 3; 39 / 10 | 42 / 2; 38 / 9 |
| Solved hydrogen | 87 / 33; 84 / 36 | 81 / 29; 80 / 33 |
| Fleet-mass IRR less `annual_growth` (methalox) | -1.37 pt | -0.57 pt |
| Plate column at f = 0.818, `eta_geom` 1.0 / 0.9 / 0.8 | 1.464e6 / 4.244e5 / 7.486e4 | 1.199e6 / 3.364e5 / 5.662e4 |
| ADR 0013 chain rate (e = 0.6, f = 0.8, 10-day split) | 0.3989 /yr | 0.3900 /yr |
| ADR 0015 matched recovery, nozzle / plate at e = 0.25 | 1.33e-4 / 3.09e-4 | 3.72e-4 / 4.35e-4 |
| at e = 0.30 | 2.95e-2 / 2.91e-2 (tie) | 2.45e-2 / 2.04e-2 (nozzle leads) |
| at e = 0.60 | 6.63e4 / 7.83e3 (8.5x) | 6.91e4 / 6.16e3 (11x) |
| Diverted to projectiles, e = 0.8 / 0.6 | 19.5-23.6% / 24.2-29.0% | 17.0-24.2% / 21.3-29.7% |

`tab:mass_interest_growth` (recovery 1, rows `eta_geom`, columns f = 0.5 / 0.6 /
0.7 / 0.8), after:

| `eta_geom` | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|
| 0.25, 0.30 | 0 | 0 | 0 | 0 |
| 0.40 | 1.7e-16 | 8.0e-16 | 2.8e-15 | 7.8e-15 |
| 0.50 | 1.2e-4 | 5.1e-4 | 1.7e-3 | 4.5e-3 |
| 0.60 | **0.95** | 4.1 | 14 | 37 |
| 0.70 | 78 | 347 | 1180 | 3290 |
| 0.80 | 1030 | 4730 | 1.66e4 | 4.76e4 |
| 0.90 | 5530 | 2.63e4 | 9.51e4 | 2.82e5 |

The published table had 1.5 / 6.7 / 22 / 61 in the 0.60 row and 3.55e5 in the
corner. **`eta_geom` = 0.60 at f = 0.5 now loses mass** (0.95 against 1.5).

**The growth cost model (ADR 0037), Estimate prices, stepped 30% -> 10%:**

| Design | Steady $/kg | BE steady | BE liquidation |
|---|---|---|---|
| Methalox | 163 -> **169** | 346 / 1694 -> **360 / 1919** | 357 / 2637 -> 388 / >3000 |
| Methane, solved | 98 -> **100** | 132 / 211 -> **135 / 240** | 148 / 368 -> 157 / 452 |
| Hydrogen, solved | 98 -> **100** | 129 / 186 -> **132 / 206** | 147 / 310 -> 153 / 371 |

Across the 10-20% late-rate band the solved chambers now break even at
$132-178 / $206-427, and methalox at $360-574 / $1919 to over $3000.

## Consequences

- **The headline conclusion stands.** Solved chambers clear Starcloud's $500 on
  both seeds and Suncatcher's $200 on the cheap seed. Methalox never clears
  $200, and clears $500 only on the cheap seed up to a 15% late rate. **One
  claim weakens:** solved hydrogen no longer clears $200 on the dear seed ($206).
- The cost lands early. Cycle 0 is the 3S return feeding the 2S cycle 1, so the
  first growth step falls by 20-48% depending on design. Later 2S->3S steps get
  easier, so the steady state moves by only $2-6/kg.
- Fewer plates and chambers are built in a decade (solved hydrogen 38 plates
  by the harvest, not 50), so learning is slower.
- ADR 0015's verdict holds and sharpens: the matched-recovery crossover moves
  below 0.30, and the nozzle's lead at 0.60 grows from 8.5x to 11x.
- **The parent owes updates** to `tab:mass_interest_growth`,
  `tab:seed_amortization`, `tab:seed_return` and every growth figure quoted from
  ADRs 0013, 0015 and 0032-0037. See `docs/growth_cost_for_parent.md`.

## Considered and rejected

- **Wrapping the last cycle to cycle 0**, as the steady state's repeating chain
  does. The window after the horizon is the payload's real departure, so it is
  flown instead. The steady state's repeat remains an approximation, as before.
- **Keeping the old pairing as a "same-cycle" convention.** The name
  `same_cycle_nozzle` refers to both waves of one return, not to the departure.
  No ADR chose the old pairing.

## Reproduction

Fast: `tests/test_two_wave_growth.py` (`link_onward_burns`),
`tests/test_growth_ledger.py` and `tests/test_chamber_departure.py` (the payload
flies the onward burn). Slow: the moved pins above, in the same files plus
`tests/test_two_leg_nozzle_sweep.py`, `tests/test_seed_cost.py`,
`tests/test_harvest.py` and `tests/test_growth_cost_inputs.py`; and
`test_the_flown_chains_onward_burns_are_its_next_departures`.
