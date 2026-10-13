# Companion reply, 2026-10-13: finite-rate recombination in the bulged chamber's nozzle

From `puffsat_impact_simulation` @ `2addc17`, answering the paper repo's S26
(`Balloon-Pulse-Propulsion/docs/throat_kinetics_asks_for_impact_sim.md`). The full answer is
that repo's `docs/throat_kinetics_handoff.md`, carried into the paper repo as
`docs/throat_kinetics_answers_from_impact_sim.md`. Evidence: the sim's
`docs/walled_nozzle_throat_kinetics.md` and ADR-0057.

**Why this repo needs it.** S24 (the wider throat) flies the chamber through this repo's
ledger, and the ledger takes the methane chamber as a share of the AR 300 chemistry ceiling
(0.669, `docs/survivable_chamber_for_parent.md`). That ceiling and the solved 0.539 behind its
80.5% share are both full-equilibrium figures. The sim's hydrocode reply of 2026-10-12
(`hydrocode_asks_handoff.md`) corrected them to 861 / 892 kN s with a sudden freeze of
hydrogen recombination. **That correction is itself superseded.** The sudden freeze tripled the
hydrogen loss.

## What changes

Net impulse per 5 kg pulse, 212 m^3 bulged chamber, 1% gate included, no wall heat:

| nozzle | full equilibrium | finite rate, carbon trades freely | carbon frozen (floor) | 2026-10-12 figure |
|---|---|---|---|---|
| 1.33 m^2, A/A* 100 | 893 kN s | **883** | 808 | 861 |
| 1.33 m^2, A/A* 300 (22.5 m exit) | 958 | **938** | 845 | 892 |
| 2.66 m^2, A/A* 150 (22.5 m exit) | 919 | **908** | 827 | 880 |
| 2.66 m^2, A/A* 300 (31.9 m exit) | 958 | **941** | 849 | 898 |

**As a share of the 0.669 ceiling** (the solved 0.539 scaled by the square of the
exhaust-speed ratio):

| A/A* 300 | eta-equivalent | share | was |
|---|---|---|---|
| 1.33 m^2, carbon trades freely | 0.523 | **78.2%** | 80.5% |
| 1.33 m^2, carbon frozen | 0.451 | 67.5% | |
| 2.66 m^2, carbon trades freely | 0.525 | **78.5%** | |
| 2.66 m^2, carbon frozen | 0.455 | 68.0% | |

- **Use ~78% for S24.** Interpolating the share rows of `survivable_chamber_for_parent.md`
  (70% at 3.61 yr, 80.5% at 2.76 yr) puts it near 2.9 yr. That is a rough read; the ledger
  should be re-run. **Superseded by ADR 0056:** re-flown on the finite-rate impulse,
  doubling is 3.36 / 3.33 yr (A/A* 100 / 300). The share rows predate ADR 0055's film and
  doubled extension, which is why the read was low.
- **The carbon-frozen floor falls below the 70% row**, so it would not grow faster than 3.6 yr.
  The sim judges the floor unlikely, though it cannot be ruled out. Keeping carbon in step needs
  a three-body coefficient about 100x below the only carbon measurement there is.
- **The wider throat is neutral to slightly positive on chemistry** (+0.3% at A/A* 300).
  Option (a) at A/A* 150 loses 3.5% to the shorter expansion. Its timing case stands on S24's
  own cycle analysis.
- **Wall heat is still extra.** The 2026-10-12 bracket (2-12% of impulse; less on the doubled
  throat, which empties sooner) applies on top of every row.
