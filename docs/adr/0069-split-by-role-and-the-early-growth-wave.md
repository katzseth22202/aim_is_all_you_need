# ADR 0069: Split by role, not at Jupiter; an earlier, faster growth wave is not worth it

Date: 2026-10-14

Status: Accepted as a finding (a scout, centred burn, not built into the
ledger). The early growth wave is declined (author, 2026-10-14: "At 2 percent,
it's not enough of a win though to really be worth the complexity"). The
ledger's default still flies ADR 0013's uncharged split (ADR 0068).

## Context

ADR 0068 found that the two-wave split has never been charged, and that
dividing one departing batch at Jupiter is dear. The split is needed only
because one batch supplies both waves. The fleet departs many launch units per
window, so each unit can instead be dedicated to a role:

- **Growth carriers** depart on the growth wave's own trajectory and come back
  as the next push.
- **Rod carriers** depart on the nozzle wave's trajectory and come back as the
  departure rods.

Nothing divides at Jupiter. Each unit pays only its own departure burn, through
the chamber, not in methalox.

The author also asked whether a 3S growth wave returning earlier and faster,
about 68 km/s at 2.8 synodic periods, pays when the nozzle wave stays at 3S.

## The balance

Per cycle, a share `x` of departing units are growth carriers. Every unit needs
`P` growth PuffSats for its push and rods for its own departure, `R_g` or
`R_n` by its burn. Each unit type sends `O_g` or `O_n` onward, and each wave
keeps `a = exp(-correction / v_e,methalox)` of its mass. The growth `g` and
`x` solve both supply balances:

    x O_g a_g = g P
    (1 - x) O_n a_n = g (x R_g + (1 - x) R_n)

With equal burns this is the ledger's `growth_per_cycle` exactly.

## Results

Flown chain, 20-day parking orbit, spray cup, film carried, burn centred (so
every row is like for like), hydrogen as slush with its cryostats scaled as
`0.03 x max(1, hold / 60 days)`. A hold past 20 days is taken as more laps of
the 20-day orbit, with one apoapsis reversal.

| How the split is flown | Methane AR100, 24 kg | Bulged hydrogen, 2 Hz |
|---|---|---|
| ADR 0013's batch, departure uncharged (today's figures) | 3.36 yr | 2.42 yr |
| Batch divided at Jupiter (ADR 0068) | 4.46 | 2.96 |
| **By role, 20-day lead** | **3.91** | **2.77** |
| By role, 3S growth wave 40 d early | 3.85 | 2.72 |
| By role, 3S growth wave 60 d early | 3.84 | 2.71 |
| By role, 3S growth wave 80 d early (2.8S, ~68 km/s) | 3.86 | 2.70 |
| By role, 3S growth wave 120 d early | 3.96 | 2.78 |

- **By role is the cheaper honest split.** It costs about 15% of doubling
  against the uncharged figures, and about half what dividing at Jupiter
  costs. Most units are growth carriers, and each departs a little harder:
  excess speed 12.36 against 11.99 km/s on 3S cycles, 14.8 against 14.4 on 2S.
- **The early growth wave gains about 2%, at 60-80 days.** On a 3S cycle the
  faster wave lifts that cycle's growth (methane cycle 0, 1.37 to 1.59). But
  the growth carriers leaving on that window depart about 0.9 km/s harder.
  That falls on the preceding cycle (cycle 4, 1.68 to 1.41), and the two
  nearly cancel.

## Decision

1. **Do not fly an early growth wave.** About 2% is not worth a hold of up to
   80 days, its cryostats or coolers, and the per-window bookkeeping.
2. **The honest split is by role.** It is recorded here, not built. The
   ledger's figures stay optimistic by about 15% of doubling until the role
   split replaces ADR 0013's uncharged one.

## Not in these numbers

- **Visibility for the growth carriers' departures.** They leave on a
  different excess vector, so they meet the stream at a different aim
  separation.
- **Choosing each growth trajectory for growth.** The solver takes the least
  correction burn, not the cheapest departure, which may give back part of the
  15%.
- **Putting the off-resonance return on the rod carriers instead.** Growth
  would come home on the resonant return and the rods later, which for 3S
  means a lower departure speed. That shifts the cycle clock.
- **Each unit type's own water schedule.** Both unit types reuse the batch's
  best starting water price.

## Reproduction (scratch, gitignored; inputs recorded here)

- **Chain.** `adaptive_two_wave_cycles(split_days=20.0, split="independent")`.
- **Trajectories.** Each window's growth trajectory is
  `window_waves(d, split_days=20.0, split="independent")[3]`. On 3S windows
  with a lead other than 20 days it is `_cheapest_return(d, return - lead)`
  instead. The window after the chain is included.
- **Per cycle `c`.**
  - The cycle carries its growth trajectory's arrival speed and correction.
  - The batch's best water price comes from `best_cycle_growth` with
    `loss_model=_orbit_loss(fixed_direction_loss, parking_period(c))`.
  - `price_cycle_growth` is flown at that price twice, with `onward_burn` set
    to the next window's growth and nozzle departure burns (each trajectory's
    `earth_departure_periapsis_speed` less `_cycle_periapsis_speed()`).
  - The balance above is solved for `g` with `brentq`.
- **Chain summary.** `summarize_chain` over the cycles' periods.
- **Leads.** 20, 40, 60, 80 and 120 days on 3S windows; 2S windows at 20.
