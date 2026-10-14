# ADR 0065: Store every hydrogen charge as slush, charge its storage in the lever ranking, and fly 4884 K

Date: 2026-10-14

Status: Accepted. Amends ADR 0051 (slush becomes the default, not an option),
ADR 0063 and ADR 0064 (their hydrogen rows charged no storage; 5500 K is not
chosen). The hydrogen cost headlines and the paper note wait for the
departure-visibility charge (below, "Not yet re-run").

## Context

Three findings from one session (author, 2026-10-14):

1. **`departure_levers` charged hydrogen no storage.** `lever_row` called
   `chain_growth` without `cryostat_fraction` or `boil_off`, while
   `seed_cost` charges both. The liquid hydrogen rows of ADRs 0063 and 0064
   were about 0.12 yr optimistic against methane, which carries no such
   charge.
2. **Slush pays.** ADR 0051 preferred it but every pairing still flew liquid.
3. **5500 K does not beat today's fill.** On cycle 0, both as slush, at the
   impact sim's solved rows:

   | | 4884 K, 2.27 Hz | 5500 K, 2.66 Hz | 5500 K at 3.09 Hz (thrust matched) |
   |---|---|---|---|
   | Thrust per chamber | 1.50 MN | 1.29 MN | 1.50 MN |
   | Chambers flown | 1 | 2 | 1 |
   | Stack delivered net of hardware | 0.270 | 0.272 | 0.293 |
   | Rods per tonne of stack | 26.0 kg | 35.3 kg | 36.4 kg |
   | Share of the returning batch left to push | 0.828 | 0.783 | 0.775 |
   | Growth | **1.786** | 1.672 | 1.811 |

   - The hotter charge's exhaust gains only 6% per kilogram carried, because
     its bonds come back at 60-71% (ADR 0064).
   - Each rod buys 27% less impulse (661 to 485 kN s): a pulse's momentum
     goes as `sqrt(2 m E)` at fixed rod energy. The departure then takes 36%
     more rods out of the returning batch.
   - Those two nearly cancel (+8.5% delivered, -6.4% share pushing, at
     matched thrust). Thrust decides, and at its solved rate 5500 K needs a
     second 46 t chamber.

## Decision

1. **Every hydrogen pairing is stored as 50% slush.** `HYDROGEN_5500K`,
   `HYDROGEN_3000K` and `HYDROGEN_BULGED` carry `SLUSH_HYDROGEN` and
   `SLUSH_HYDROGEN_TANK_FRACTION` (0.205 / 1.15 = 0.178).
   `stored_as(p, LIQUID_HYDROGEN)` gives the liquid back for comparison;
   `make slush-hydrogen` and `make departure-levers` keep one liquid row each.
2. **The growth matrix's hold constants are slush's**: 3% cryostats, no
   boil-off. `_hold_sweep` still flies liquid, whose boil-off it brackets.
3. **Cryostats drop just before ignition**, not with the plate. They ride the
   reversal and the 10-day coast, keeping the insulation ADR 0051's zero
   boil-off needs. Nothing burns between the reversal and ignition, so the
   ledger's charge is unchanged; only the docstrings moved.
4. **`departure_levers` charges storage** for every hydrogen row, as
   `seed_cost` does.
5. **Fly the bulged chamber at 4884 K** (`CHOSEN_TEMPERATURE`), its fixed 60
   kg charge, reversing ADR 0064's 5500 K. 5500 K would pay only at about 3.1
   Hz, 17% above its solved 2.66 Hz.

## Results (`make departure-levers`)

Flown chain, spray cup, film carried, 20-day parking orbit.

| Lever | ADR 0064 (no storage) | Liquid, charged | **Slush, charged** | Annual | 10-yr |
|---|---|---|---|---|---|
| Bulged hydrogen, 2 Hz | 2.54 yr | 2.66 | **2.42** | 33.2% | 15.3x |
| 2.4 Hz | 2.38 | | **2.30** | 35.3% | 18.2x |
| Two chambers | 2.71 | | **2.60** | 30.5% | 12.3x |
| Solved 4884 K | 2.61 | | **2.49** | 32.1% | 14.0x |
| Solved 5500 K (37.0 t wrap) | 2.70 | | **2.59** | 30.7% | 12.7x |
| Solved 6000 K | 2.82 | | **2.68** | 29.5% | 11.7x |
| Solved 7000 K | 2.79 | | **2.69** | 29.4% | 11.5x |
| Methane AR100, baseline | 3.36 | | 3.36 | 22.9% | 6.5x |
| Methane, 3 Hz + 3.5 kg pitch | 2.91 | | 2.91 | 26.9% | 9.3x |

The methane rows are unchanged, as they should be.

**Pinned figures that moved.** The slow suite pins the legacy "Hydrogen,
solved" design (the 5500 K sphere, `DESIGNS[4]`), which the parent's tables
were built on. Stored as slush:

| Figure | Liquid | Slush |
|---|---|---|
| `tab:seed_amortization` row: seed, annual, 10-yr, units built | 83.25 t, 54.8%, 75.07x, 32.85 | 83.65 t, 58.2%, 95.52x, 38.36 |
| Rods as a share of the seed | 19.6% | 20.1% |
| `tab:seed_return` IRR, liquidation (cheap / dear seed) | 49.8 / 29.2% | 52.6 / 31.6% |
| The same, steady state | 51.1 / 33.1% | 54.0 / 35.5% |
| Steady $/kg at L1, Estimate book | $100 | $96 |
| Steady break-even, cheap / dear seed | $146 / $194 | $135 / $174 |

These move the parent's tables; the paper note waits for the visibility charge
(below), which moves them again.

## Not yet re-run

**The departure is not checked for Earth occultation at its real length.**
ADR 0045 cleared a 20-minute burn on the circular 3S closure. The ledger's
burns run 25-38 min (methane) and 41-60 min (hydrogen). Centred on periapsis,
Earth blocks the stream during the approach leg for part of the burn at most
aim separations swept (144.9-175 degrees). Three hydrogen cycles also sit at
the 3,600 s edge of the finite-burn loss table. The visibility-constrained
charge is ADR 0066. The hydrogen cost headlines (`make survivable-cost`,
`make hardware-recovery`, `make slush-hydrogen`) and the paper note on slush
and 4884 K wait for it.

## Reproduction

- `make departure-levers`.
- The cycle-0 temperature comparison was a scratch run (gitignored):
  `solved_hydrogen(T)` at 4884 and 5500 K through `chain_growth` on
  `flown[:1]` with the slush storage charged, and the 5500 K pairing with
  `pulse_rate = 661 x 2.27 / 485 / s`.
- Tests: `tests/test_chamber_isp.py`, `tests/test_departure_levers.py`;
  `make test-all`.
