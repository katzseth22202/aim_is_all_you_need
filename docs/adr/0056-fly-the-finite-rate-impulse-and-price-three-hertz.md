# ADR 0056: Fly the finite-rate impulse, and price 3 Hz on the doubled throat

Date: 2026-10-13

Status: Accepted. Answers items 1, 3 and 4 of the parent's S24
(`Balloon-Pulse-Propulsion/docs/throat_and_turn_asks_for_aim_repo.md`). It takes the
impact simulation's S26 answer, `puffsat_impact_simulation` @ `df4bc6e` (its
`docs/throat_kinetics_handoff.md`, second issue, and ADR-0057). Amends ADR 0055.

## Context

ADR 0055 flew the doubled 2.66 m2 throat on 868 / 899 kN s per pulse (A/A* = 100 / 300).
Those figures froze hydrogen recombination suddenly in the nozzle. The impact simulation
has since solved H + H + M at its measured rate, and the sudden freeze turned out to have
tripled the hydrogen loss. Carbon has no measured three-body rate, so the simulation
brackets it:
- **finite rate:** carbon trades atoms freely through two-body reactions;
- **floor:** carbon frozen at the chamber state.

Net impulse per 5 kg pulse on the 2.66 m2 throat, with the 1% gate and no wall heat:

| nozzle | exit | finite rate | carbon frozen | ADR 0055 |
|---|---:|---:|---:|---:|
| A/A* = 100 | 18.4 m | **885 kN s** | 810 | 868 |
| A/A* = 150, S24 option (a) | 22.5 m | **908** | 827 | — |
| A/A* = 300, S24 option (b) | 31.9 m | **941** | 849 | 899 |

The simulation's handoff quotes a ledger re-run of 3.36 / 3.33 yr and points to this
repo's `docs/companion_replies_2026-10-13.md` for it. That file did not hold those figures;
it held only a "near 2.9 yr" interpolation. This ADR produces the figures here.

S24 also asks which pulse rates the doubled throat's cycle fits (item 1), and for the
ledger at every rate that fits (item 3).

## Decision

1. **`chamber_isp.SURVIVABLE_IMPULSE` takes the finite-rate figures**, 885 / 908 / 941 kN s,
   which give `eta` 0.472 / 0.490 / 0.515. The carbon-frozen floor is
   `SURVIVABLE_IMPULSE_CARBON_FROZEN`. ADR 0055's figures are kept as
   `SURVIVABLE_IMPULSE_SUDDEN_FREEZE` for the record.
2. **A/A* = 150 is a flown nozzle.** Option (a) keeps the 22.5 m exit, so it carries ADR
   0044's 14.7 t extension.
3. **`chamber_isp.survivable_cycle()` times the cycle.** It uses the impact simulation's
   own cycle model:
   - the real-gas seal time on the 1.33 m2 throat (281 ms to fill pressure, 389 ms to
     0.3 bar), scaled as `V / A*`;
   - then 30 ms to lay the films, 108 kg filled at 1 t/s, and a 20 ms lock.

   It reproduces the simulation's 439 / 548 ms and 297 / 351 ms periods within 1.5 ms.
4. **A rate fits only when its margin clears the blowdown clock's own uncertainty.** That
   uncertainty is ±45 ms on the 1.33 m2 throat (wall heat, a discharge coefficient below
   one), so ±22.5 ms on 2.66 m2.
5. **The ledger flies 2 Hz and 3 Hz.** Every S24 row flies at both carbon limits
   (`make survivable-ledger`, group "S24"). At 3 Hz the fill is doubled to 2 t/s, and the
   polymer film is still sealed at 0.3 bar, so the 2.6 kg film is charged at every rate.

## The cycle (S24 item 1)

The 212 m3 methane chamber (`make survivable-ledger` prints this table). Options (a) and (b)
share the chamber and throat, so they share the cycle: the exit does not touch the
blowdown.

| throat | film sealed at | fill | period | fastest | 2 Hz | 3 Hz | 4 Hz |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.33 m2 | 0.3 bar (polymer) | 1 t/s | 547 ms | 1.83 Hz | −47 ms | — | — |
| 2.66 m2 | 0.3 bar (polymer) | 1 t/s | 352 ms | 2.84 Hz | +148 | −19 | −103 |
| 2.66 m2 | 0.3 bar (polymer) | **2 t/s** | **298 ms** | **3.35 Hz** | +202 | **+35** | −49 |
| 2.66 m2 | 1.11 bar (needs a ~1,100 K film) | 1 t/s | 298 ms | 3.35 Hz | +202 | +35 | −49 |
| 2.66 m2 | 1.11 bar (needs a ~1,100 K film) | 2 t/s | 245 ms | 4.09 Hz | +256 | +89 | +5.5 |

Margins are the period left over; the blowdown clock's own uncertainty is ±22.5 ms on the
2.66 m2 throat.

- **2 Hz fits, with ~150 ms to spare.** This confirms ADR 0055.
- **3 Hz fits only with one more change.**
  - At the impact simulation's 1 t/s fill and a polymer film, the cycle runs 352 ms. That
    is 19 ms past the 333 ms period.
  - Doubling the fill to 2 t/s brings the margin to +35 ms, clear of the ±22.5 ms clock.
    The simulation already treats a 2 t/s fill as the way to speed a smaller chamber.
  - A film rated to ~1,100 K and sealed at fill pressure gives the same +35 ms. That
    would have to be a metal foil, whose fragments then join the exhaust. It is not
    flown here.
- **4 Hz does not fit.** Even with both changes the margin is 5.5 ms, inside the clock's
  uncertainty. Each change alone leaves a deficit of about 49 ms.

## The ledger (S24 items 2-3)

Spray cup with its film carried, 20-day orbit, 24 kg pitch plus the 2.6 kg throat film,
fixed-direction loss integrated and solved with the burn (ADR 0032). One 5 kg chamber in
every row.

| nozzle | chamber t | finite rate, 2 Hz | finite rate, 3 Hz | carbon frozen, 2 Hz | carbon frozen, 3 Hz |
|---|---:|---:|---:|---:|---:|
| A/A* = 100 | 56.0 | 3.36 yr | 3.02 yr | 4.66 yr | 3.94 yr |
| A/A* = 150, option (a) | 60.9 | **3.26** | **2.94** | 4.54 | 3.85 |
| A/A* = 300, option (b) | 75.6 | 3.33 | 2.99 | 4.94 | 4.10 |
| *ADR 0055 (sudden freeze)* | | *3.57 / 3.86* | | | |

- **The impact simulation's 3.36 / 3.33 yr reproduce exactly.** So does its carbon-frozen
  floor of 4.66 / 4.94 yr.
- **All three nozzles fall within 0.10 yr of each other.**
  - Option (a) is ahead by 0.07-0.10 yr at 2 Hz and by 0.05-0.08 yr at 3 Hz.
  - A/A* = 300 returns the most impulse, but carries 29.4 t of extension against
    14.7 t.
  - That extension is a thin shell scaled by exit area (ADR 0044), and nobody has sized a
    32 m carbon-carbon one. A spread this small is inside the extension's own uncertainty.
- **3 Hz is worth 0.32-0.34 yr.** G_0 rises from 1.46-1.49 to 1.57-1.60. Burns fall from
  25-39 min to 16-26 min, and the finite-burn loss halves (7.3-12.3% of the burn down to
  3.9-7.5%).
- **The carbon-frozen floor costs 1.3-1.6 yr at 2 Hz.** At the floor, A/A* = 300 falls to
  last.
- **The companion reply's "near 2.9 yr" was wrong.** It interpolated the share-of-ceiling
  rows of `survivable_chamber_for_parent.md`. Those were flown before ADR 0055 added the
  throat film and doubled the extension. Re-flown, 70% of the ceiling now doubles in
  4.20 yr and 90% in 2.57 yr. The finite-rate 0.515 is 77.0% of the 0.669 ceiling, and it
  doubles in 3.33 yr.
- **Other rows at 2 Hz:**
  - **3.5 kg pitch:** 3.26 yr at A/A* = 100 and 3.21 yr at 300.
  - **Maraging fallback wall:** 4.25 / 4.13 yr.
  - **Steered loss** (which a head-on chamber cannot fly): 3.19 / 3.15 yr.
  - **Two 2.5 kg chambers:** 3.41 / 3.22 yr.
  - **Two 5 kg chambers** (the wall-life case): 4.70 / 5.52 yr.
- **Wall life is not relieved by rate.** The longest cycle fires the same pulses
  whatever the rate. The most on one wall is 4,569-4,708 at 2 Hz and 4,619-4,743 at 3 Hz,
  against the ~4,200 the wrap is sized for. ADR 0055's ask for a ~5,000-pulse wall stands.

**Cost** (Estimate book, lob climbing, 10% a year, 50% odds, `make survivable-cost`). Each
cell gives steady $/kg, then the steady break-even with cheap / dear seed:

| design, 24 kg pitch | 2 Hz | 3 Hz | priced by chamber mass, 2 / 3 Hz |
|---|---|---|---|
| A/A* = 100 | $163, $333 / $1105 | $146, $281 / $870 | $191 / $169 |
| A/A* = 150, option (a) | $158, $316 / $1030 | $142, $268 / $818 | $187 / $167 |
| A/A* = 300, option (b) | $157, $317 / $1071 | $141, $267 / $842 | $197 / $174 |
| *ADR 0055, A/A* = 100 / 300* | *$172 / $178* | | *$203 / $229* |
| H2 bulged, 2 Hz (unchanged) | $175, $351 / $925 | | |
| H2 sphere, wall unverified (unchanged) | $122, $192 / $377 | | |

- Priced by chamber mass against the sphere, option (a) is cheapest at 2 Hz, and
  A/A* = 100 and (a) are within $2 at 3 Hz.
- On the flat chamber price, A/A* = 300 is $1 cheaper than (a).

## The film (S24 item 4)

- **The 2.6 kg aramid film is charged in every row above, at both rates.** It is added to
  the 24 kg pitch as carried mass that leaves in the exhaust (`survivable_design`), which
  makes the pitch ratio 5.32 per kilogram of rod.
- **The film's mass already carries the wider throat.** The film holds `p r / (2 t)` over a
  cap of the throat's radius, so its mass goes as `p r^3`. That is how the simulation got
  2.6 kg on 2.66 m2 against 0.9 kg on 1.33 m2 at 0.3 bar.
- **The rate does not change the film.** The film is sealed at a fixed 0.3 bar whatever the
  period, and 3 Hz buys its time in the fill, not the seal.
- **Only the hot-film route would change it.** Sealing at fill pressure would load the film
  to ~2.9 bar and ~1,100 K. In aramid-equivalent `p r^3` that is ~4.2 kg a shot, and in
  practice it means a foil. That route is not flown.

## Consequences

- **The paper can print 3.36 / 3.33 yr** for A/A* = 100 / 300 on the doubled throat at
  2 Hz, with the carbon floor at 4.66 / 4.94 yr. Option (a) at A/A* = 150 is 3.26 yr.
- **The paper can print 3 Hz as conditional.** The rows are 3.02 / 2.94 / 2.99 yr, and
  they need the fill doubled to 2 t/s, or a ~1,100 K film. Without that condition, the
  doubled throat is a 2 Hz chamber.
- **No nozzle leads.** The three nozzles sit within 0.1 yr of each other, and that spread
  depends on an extension mass nobody has sized. ADR 0055's "lead with A/A* = 100" is
  withdrawn. Lead with whichever exit the paper argues on other grounds.
- **`hardware_recovery.py` flies A/A* = 100 on the new impulse** (`make hardware-recovery`,
  Estimate book, methane priced by chamber mass). The hydrogen rows do not move.

  | recovery | CH4 AR 100 | was (ADR 0055) | H2 bulged, 2 Hz |
  |---|---:|---:|---:|
  | none | $191 | $203 | $175 |
  | rods only | $154 | $161 | $147 |
  | rods + plate | $147 | $152 | $126 |

  With recovered hardware the bulged hydrogen chamber now leads methane by $7-21/kg at
  2 Hz, against ADR 0055's $10-25. At 3 Hz methane, which hydrogen's 2 Hz chamber has not
  been timed against, the lead would narrow further. That case is not flown.

## Limitations

- **The impulses are quasi-1D finite-rate solves.** Carbon is bracketed, not solved, and the
  2-12% wall-heat debit sits on top of every row.
- **The mechanism times are the impact simulation's assumptions:** film laying, the fill
  rate and the lock. The 2 t/s fill is sized, not designed. Its pump and accumulator mass
  is not charged.
- **3 Hz is unchecked on two counts.** Wall heat per second rises 1.5x at 3 Hz. Neither the
  wall's thermal state nor its fatigue at the higher rate has been checked.
- **The extension masses scale by exit area** at fixed areal density.

## Reproduction

- **Ledger, cycle table and cost book:** `make survivable-chamber`. The ledger alone, with
  the cycle table, is `make survivable-ledger` (~4 min on four workers). The cost book alone
  is `make survivable-cost` (~3 min).
- **The paper's methane cells** (`docs/throat_answers_for_parent.md`):
  - `make survivable-paper-grid` (~4 min) flies A/A* = 300 at every share of the ceiling,
    and as solved. It flies them behind the spray cup (film shielded and unshielded), the
    plug, and ADR 0033's plate, at both loading caps.
  - `python -m src.survivable_chamber --paper-cost` (~3 min) prices the solved row behind
    each plate, by chamber mass, with the odds, liquidation and Earth-loop columns.
- **Tests:** `pytest tests/test_chamber_isp.py tests/test_survivable_chamber.py` pins:
  - the 885 / 908 / 941 kN s inversions;
  - the ordering floor < sudden freeze < finite rate;
  - the simulation's four cycle periods;
  - 3 Hz needing the faster fill, and 4 Hz never fitting;
  - the film on every S24 row.
