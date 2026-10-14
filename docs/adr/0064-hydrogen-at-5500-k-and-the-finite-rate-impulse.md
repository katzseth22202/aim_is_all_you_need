# ADR 0064: Fly hydrogen at 5500 K, not 7000 K; and the bulged chamber's finite-rate impulse

Date: 2026-10-14

Status: Accepted. The temperature is chosen. The headline hydrogen chamber stays
at 4884 K until the impact simulation solves the 5500 K wrap. The impulse
correction is applied now.

## Context

ADR 0063 asked the impact simulation
(`docs/hydrocode_asks_2026-10-14_hot_hydrogen.md`) about the bulged hydrogen
chamber at 7000 K and 10 000 K. It answered at commit `ca07938` (its ADR-0058,
`docs/hot_hydrogen_handoff.md` and `docs/walled_nozzle_hot_hydrogen.md`), and
added rows at 5500, 6000, 8000 and 9000 K:

- **The bonds come back at 60% (7000 K) and 55% (10 000 K), not 94%.** The 94%
  was the dense 5500 K sphere at 818 bar; the bulge is a fifth as dense.
- **The charge stops shrinking.** The rod and plug store little energy, so the
  charge is 24.1 kg at 7000 K (k = 9.65) and 18.7 kg at 10 000 K, not 5.5.
- **10 000 K fails the wrap** (3.48% flung strain against 2.4%). 7000 K passes
  on a 29.8 t wrap.
- **The gas turns optically thick** above about 6000 K and radiates onto the
  wall.
- **Today's impulse was understated.** With finite-rate recombination it is 683
  kN s, not ADR 0055's 657, or 661 at the high wall-heat edge.

The sim's rows (wall heat debited, sealed at 0.3 bar on the 0.5 t/s fill):

| | k | Impulse per pulse | Per kg carried | Rate | Thrust | Wall heat per kg of charge |
|---|---|---|---|---|---|---|
| today, 4884 K | 24.0 | 661 kN s | 8.51 km/s | 2.27 Hz | 1.50 MN | 0.4-5.1 MJ/kg |
| **5500 K** | 16.1 | 485 | 9.02 | 2.66 | 1.29 | **7.9** |
| 6000 K | 12.7 | 402 | 9.30 | 2.94 | 1.18 | 9.6 |
| 7000 K | 9.65 | 327 | 9.57 | 3.28 | 1.07 | 9-14 |
| 10 000 K | 7.47 | 264 | 9.63 | 3.61 | 0.95 | 29-31 |

The hydrogen charge can carry 9.0 MJ/kg as regenerative coolant (77 K to 700 K).

## The ledger (`make departure-levers`)

Each solved row flown through `chain_growth` on the flown chain behind today's
push, at its own rate and the sim's wall-heat-debited impulse:

| Temperature | Wrap | Doubling |
|---|---|---|
| 4884 K | 37.0 t (solved) | 2.61 yr |
| 5500 K | 37.0 t (held) | 2.70 yr |
| 5500 K | 33.1 t (interpolated) | 2.62 yr |
| 6000 K | 37.0 t (held) | 2.82 yr |
| 6000 K | 31.7 t (interpolated) | 2.68 yr |
| 7000 K | 29.8 t (solved) | 2.79 yr |
| 5500 K at 2.34 Hz, 37.0 t | | 2.92 yr |

The interpolated wraps are linear in the charge between the two solved fills
(59.9 kg at 37.0 t, 24.1 kg at 29.8 t).

## Decision

1. **Fly the bulged hydrogen chamber at 5500 K** (`departure_levers.CHOSEN_TEMPERATURE`),
   the author's call (2026-10-14), on three grounds:
   - **7000 K does not pay.** Its 12% faster exhaust is lost to 29% less thrust.
     It doubles in 2.79 yr against 2.61-2.62.
   - **5500 K is the hottest charge that still cools its own wall.** Its 7.9
     MJ/kg is under the 9.0 MJ/kg its hydrogen can carry; 6000 K's 9.6 is not,
     and from there the thick gas's radiation needs a separate heat path.
   - **It keeps the most thrust margin of the hot rows** for 6% more impulse
     per kilogram carried than today.
2. **It ties today's 4884 K on growth; it does not beat it.** 2.62 against 2.61
   yr, and only if its wrap comes in near 33 t. Its edge is its faster cycle:
   flown at today's margin below its rate it doubles in 2.92 yr.
3. **The headline chamber stays at 4884 K** until the sim solves the 5500 K
   wrap. Switching it moves `survivable_chamber`, the cost report and the seed.
4. **`HYDROGEN_BULGED_IMPULSE` is 683 kN s** (finite rate), replacing 657. Its
   efficiency rises from 0.773 to 0.822. Figures below.
5. **10 000 K is dropped.** It fails the wrap and gains nothing per kilogram.

## The corrected impulse

**Not yet re-run.** Every bulged-hydrogen figure in ADRs 0055, 0056, 0057, 0059
and 0063 (growth, $/kg, break-even, slush and recovery rows) was computed on 657
kN s and is now stale. They move together and in hydrogen's favour: on the
sim's like-for-like rows the chamber doubles in 2.61 yr against ADR 0063's
2.84. Re-run `make survivable-cost`, `make hardware-recovery`,
`make slush-hydrogen` and `make departure-levers` before quoting any of them.

## Asks of the impact simulation

- The 5500 K fill's wrap, at 0.5 cm and the owed 0.25 cm check.
- Whether the 5500 K pulse rate holds at 2.66 Hz on the 0.3 bar seal.

## Limits

- The wraps at 5500 and 6000 K are interpolated, not solved.
- The temperature comparison uses the wall-heat-debited impulses for every row
  (661 for today), so it is like for like; the flown headline uses 683.
- The tanks per kilogram of hydrogen and the extension are held at the bulged
  chamber's.

## Reproduction

- `make departure-levers` flies the solved rows (`SOLVED_HYDROGEN`) at the
  sim's wraps. The interpolated-wrap and 2.34 Hz rows were a scratch run
  (gitignored): `solved_hydrogen(T)` with `wall_mass` replaced by 33.1 t (5500
  K) or 31.7 t (6000 K), or `pulse_rate` by 2.34 / s, flown through `lever_row`.
- Tests: `tests/test_departure_levers.py`, `tests/test_chamber_isp.py`,
  `tests/test_survivable_chamber.py`.
