# ADR 0071: The ledger splits by role, on a 10-day parking orbit

Date: 2026-10-14

Status: Accepted (author, 2026-10-14: "yes, add the 20 days into the ledger";
then, after the split-gap sweep, "move to 10 days if its better"). Supersedes
ADR 0034's choice of 20 days for the ledger's parking orbit. Builds ADR 0069's
role split into the ledger.

## Context

ADR 0069 priced the honest two-wave split as a scout: launch units dedicated
as growth carriers or rod carriers, each departing on its own trajectory and
paying its own departure burn. It cost about 15% of doubling against ADR
0013's uncharged split, and was not in the ledger.

The parking orbit was 20 days (ADR 0034). The split gap *is* the parking
period. A longer gap buys a cheaper apoapsis reversal and pays for it in a
harder growth trajectory. Under the uncharged split that harder trajectory was
free, so the only cost of a short gap was the reversal. Once each growth
carrier pays its own departure, the trade changes.

## Decision

1. **The ledger splits by role.** `TwoWaveCycle` carries both onward burns,
   `onward_burn` for the rod carriers and `onward_growth_burn` for the growth
   carriers, linked from the next window (`link_onward_burns`).
   - `price_cycle_growth` prices both carriers, each with its own stream kept
     in view (the visibility tables now store both roles).
   - `chamber_departure.role_growth` solves ADR 0069's balance.
   - The returned `CycleGrowth` is the fleet-mean unit, weighted by
     `growth_share`, and carries both carriers' departures.
   - Synthetic cycles without a growth burn fly one batch, as before.
   - Methalox departs nothing through a chamber and is unchanged.
2. **`DEFAULT_PARKING_DAYS` is 10 days**, the best gap in the sweep below.
   The parent's 600 km figures are quoted on the 20-day orbit, and are not
   rescaled here.

Centred on periapsis on the 20-day orbit, the ledger reproduces ADR 0069's
3.91 yr for methane exactly.

## The split-gap sweep

Role split, stream kept in view, spray cup, film carried (`make split-by-role`
with `--split-sweep`; the growth share is the fraction of departing units
that are growth carriers):

| Split gap = parking orbit | Apoapsis reversal | Methane AR100, 24 kg | Bulged hydrogen, 2 Hz | Growth share |
|---|---|---|---|---|
| 4 d | 691 m/s | 4.84 yr | 3.12 yr | 0.85-0.87 |
| 5 d | 594 | 4.47 | 2.97 | |
| 7 d | 474 | 4.14 | 2.84 | |
| **10 d** | **372** | **3.98** | **2.77** | |
| 15 d | 284 | 4.00 | 2.79 | |
| 20 d | 234 | 4.16 | 2.87 | |

- **Below 10 days the reversal dominates.** It is methalox, so it is paid on
  the whole unit.
- **Above about 15 days the growth carriers' harder departure dominates.** It
  is a chamber burn, but it grows with how far ahead the growth wave must
  arrive.
- **The optimum is flat between 10 and 15 days.** 10 days is chosen, and it is
  also `two_wave_growth`'s own default.

## Results

Every figure below is the repo's before this ADR (one uncharged batch on the
20-day orbit, the stream kept in view) against after (split by role on the
10-day orbit).

| Figure | Before | After |
|---|---|---|
| Bulged hydrogen, 2 Hz: doubling (`make departure-levers`) | 2.61 yr | **2.77 yr** |
| Bulged hydrogen at 2.4 Hz | 2.44 | **2.58** |
| Methane AR100, 24 kg | 3.59 | **3.98** |
| Methane at 3 Hz | 3.10 | **3.38** |
| Bulged hydrogen: steady $/kg at L1 (`make survivable-cost`) | $154 | **$165** |
| Its steady break-even, cheap / dear seed | 369 / 690 | **404 / 765** |
| Methane AR100: steady $/kg | $177 | **$194** |
| Methane AR100 at 3 Hz | $152 | **$163** |
| Solved hydrogen (the sphere): steady $/kg (`make growth-cost`) | $98 | **$101** |
| Solved methane | $102 | **$106** |
| Methalox | $169 | **$183** |
| Bulged hydrogen, tanks and plate recovered (`make hardware-recovery`) | $121 | **$126** |

- **The ranking of levers is unchanged.** Pulse rate still pays more than a
  second chamber (hydrogen two chambers 2.95 yr against one at 2.77), and 4884
  K still beats 5500 K (2.86 against 2.94 yr, solved impulses).
- **Methalox loses most** because it pays the dearer 10-day reversal and has no
  chamber departure for the role split to relieve. It is the incumbent, not a
  flown design.
- **The shorter gap recovers about a third of what the role split costs.** On
  the 20-day orbit the role split lengthens methane's doubling by 16% (3.59 to
  4.16 yr) and hydrogen's by 10% (2.61 to 2.87). On the 10-day orbit it is 11%
  and 6%.

## Kept on the 20-day orbit

These reproduce earlier ADRs and fly their own orbit explicitly:

- `split_by_role`'s scout rows (`SCOUT_PARKING_DAYS`, ADRs 0068-0069).
- `departure_levers --centred` and `--temperatures --centred`
  (`EARLIER_PARKING_DAYS`, ADRs 0063-0065): one batch, burn centred.
- `seed_loop.CHAIN_FLOOR`, 59.29 km/s (ADR 0059). On the 10-day orbit the
  first growth wave arrives at 57.43, so the old floor is the stricter.
- `growth_ledger.design_sensitivities` now compares the 20-day orbit as the
  alternative.

## Limits

- **Each carrier's trajectory is still chosen for the least correction burn,**
  not for growth (ADR 0069). Choosing for growth may recover more.
- **The sweep's gaps are discrete.** Between 10 and 15 days the curve is flat
  to under 1%.
- **The parent's 600 km finite-burn figures** (22/226/486 and 27/101 m/s)
  remain the 20-day orbit's (`finite_burn_loss`, unchanged).

## Reproduction

`make departure-levers`, `make survivable-cost`, `make growth-cost`, `make
hardware-recovery`, `make slush-hydrogen`; the sweep with `python -m
src.split_by_role --split-sweep`. The visibility tables (both roles, 20 and 10
days, plus the sweep's 4, 5, 7 and 15) are rebuilt by `make
departure-visibility` (add `--sweep` for the sweep's orbits). Tests:
`tests/test_split_by_role.py` (the ledger reproducing 3.91 yr and the 10-day
optimum are slow), the moved pins in `tests/test_growth_cost_inputs.py` and
`tests/test_seed_cost.py`, with the old values kept in comments.
