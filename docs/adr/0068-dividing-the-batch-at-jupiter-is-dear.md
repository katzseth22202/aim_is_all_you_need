# ADR 0068: The two-wave split was never charged; dividing the batch at Jupiter is dear

Date: 2026-10-14

Status: Accepted as a finding. Amends ADR 0013. The chain's default stays the
independent re-solve until the split by role (the next ADR) replaces it, so the
headlines move once. Every growth figure since ADR 0013 omits the split's cost.

## Context

The author asked (2026-10-14) why the flown chain never splits a 3S cycle so
the growth wave returns faster. The answer turned up a gap in how every split
has been priced.

ADR 0013 re-solves each growth wave as its own Earth-Jupiter-Earth Lambert pair
from the departure epoch, with a free Jupiter encounter time, and charges only
its later correction burn. Both waves leave on one departure burn, yet the
growth wave's solution needs a different Earth departure excess vector. At the
20-day split the difference is **0.52-0.95 km/s** per cycle, and nothing pays
for it. CONTEXT.md says the batch "divides at Jupiter"; the code never solved
that.

## Decision

`two_wave_growth` now offers both ways:

- **`split="independent"`** is ADR 0013's re-solve, still the default.
- **`split="shared"`** divides the batch at Jupiter (`shared_split`).
  - Both waves fly the departure burn's one outbound leg, so they share the
    departure by construction.
  - Each takes its cheapest Jupiter option: unpowered, a perijove burn, or a
    deep-space maneuver.
  - The encounter maximises growth to first order: the growth wave's burn at
    the methalox exhaust speed, the nozzle wave's at a fifth of that (rods are
    about a fifth of consumption), and the departure burn at a chamber's ~8
    km/s.

`TwoWaveCycle.nozzle_wave_dsm` is now the nozzle wave's whole correction
(`total_dv`), which equals the old proxy under the independent split.

## Results

**Per cycle at the 20-day split** (flown chain; the policy's 2S/3S choices and
dates are unchanged):

| | Independent | Shared |
|---|---|---|
| Nozzle wave's correction | 0-18 m/s | **0.48-0.82 km/s** |
| Growth wave's correction | 0-2.09 km/s (cycles 1, 4, 6) | 0-2.23 km/s (the same cycles) |
| Onward departure burn | | **+0.14-0.56 km/s** |
| Nozzle wave's arrival speed | | 0.2-0.7 km/s slower |

The optimiser puts the cost where it hurts least, on the nozzle wave and the
departure burn. Valuing the rods at 0.6 or the departure burn at a 4 km/s
exhaust leaves the doubling unchanged. Loading the split onto the growth wave
instead is worse (4.63 yr for methane).

**Through the ledger** (the burn centred, so both columns are like for like):

| | Independent | Shared |
|---|---|---|
| Methane AR100, 24 kg | 3.36 yr | **4.46 yr** |
| Bulged hydrogen, 2 Hz | 2.42 yr | **2.96 yr** |

**An earlier, faster growth wave, divided at Jupiter** (3S cycles 0 and 9, the
nozzle wave at 3S):

| Growth return | Days early | Growth wave arrives | Its Jupiter burn |
|---|---|---|---|
| 2.95S | 20 | 59.5-59.6 km/s | 0.49-0.75 km/s |
| 2.9S | 40 | 62.8-63.4 | 1.25-1.41 |
| 2.85S | 60 | 65.4-66.4 | 1.96-1.97 |
| 2.8S | 80 | 67.3-68.6 | 2.39-2.43 |
| 2.7S | 120 | 69.2-70.5 | 2.90-2.97 |

Under the independent re-solve these returns look free, because the growth
wave's own departure is never charged. Divided at Jupiter, a 68 km/s growth
wave costs about 2.4 km/s of methalox, which is 47% of its mass, against about
32% more push per kilogram of PuffSat. It does not pay this way.

## Consequences

- **The chain's growth figures are optimistic** by however much the true split
  costs. Dividing at Jupiter sets an upper bound: 33% on methane's doubling
  and 22% on hydrogen's.
- **A cheaper architecture may exist.** The split is needed only because one
  departing batch supplies both waves. If launch units are dedicated by role,
  growth carriers and rod carriers each depart on their own trajectory, and
  nothing divides at Jupiter. Each pays only its own departure burn, through
  the chamber. The next ADR prices that, and the earlier, faster growth wave
  under it.
- **The visibility tables are for the independent chain.** A default that
  changes the departures must rebuild them (`make departure-visibility`).

## Reproduction

- `two_wave_growth.adaptive_two_wave_cycles(split_days=20.0, split=...)`.
- The ledger comparison and the weight sweep were scratch runs (gitignored):
  `chain_growth` on each chain with `loss_model=_orbit_loss(fixed_direction_loss,
  parking_period(...))`. The sweep set `_SPLIT_ROD_SHARE` to 0.2, 0.6 or 1.0 and
  `_SPLIT_CHAMBER_EXHAUST` to 8, 4 or 3 km/s.
- The early growth returns: `shared_split(departure, departure + 3S,
  departure + f S)` for the flown chain's cycles 0 and 9, where S is
  `_FIXED_TWO_SYNODIC_DAYS / 2`.
- Test: `tests/test_two_wave_growth.py::test_a_shared_split_flies_one_outbound_leg_and_pays_to_divide`
  (slow).
