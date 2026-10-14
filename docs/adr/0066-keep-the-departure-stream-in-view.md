# ADR 0066: Keep the departure stream in view; it costs about 0.2 yr of doubling

Date: 2026-10-14

Status: Accepted for the lever ranking (`make departure-levers-visible`). Not
yet the ledger's default: the cost book, seed and harvest reports still fly the
centred loss table (see "Not yet done").

## Context

ADR 0045 showed Earth does not occult the stream during a 20-minute burn on the
circular 3S closure, whose stream arrives 144.9 degrees from the departure. Two
things differ on the flown chain:

- **The burns are longer.** The ledger's burns run 25-38 min (methane) and
  41-60 min (hydrogen). Three hydrogen cycles sat at the 3,600 s edge of the
  finite-burn loss table.
- **The streams are more nearly head-on.** Re-solving each cycle's selected
  Lambert legs (`departure_visibility.onward_streams`) gives the angle between
  the onward departure's excess vector and the returning stream's:

  | Cycle | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Synodic | 3S | 2S | 2S | 2S | 2S | 3S | 2S | 2S | 2S | 3S | 3S |
  | Aim separation | 167.2 | 175.5 | 162.2 | 154.1 | 172.0 | 167.5 | 174.9 | 161.4 | 164.4 | 167.9 | 167.9 |

A burn centred on periapsis then loses the stream on its approach leg. The
PuffSats would have to pass through Earth to reach the vehicle. At 175 degrees
even a 20-minute centred burn is blocked for 27% of its length.

The other departure mirror does not help. At periapsis its routes are marginal
(6-326 km projectile periapsis) or blocked on every cycle, against 394-600 km
on the mirror flown.

## Decision

1. **A departure must keep every incoming route visible for its whole burn.**
   `src/departure_visibility.py` integrates the fixed-direction burn (ADR
   0045's integrator, on its visible mirror) and finds the placement with the
   least finite-burn loss among those that stay visible.
   - Blocking always falls at the start, on the approach leg, so the fix is to
     start later: less of the burn before periapsis, more after.
   - The latest admissible start is found by bisection. The loss falls toward
     the free optimum, so that start is the visible optimum.
2. **The loss is tabulated per cycle over burn length** (0-7,200 s, 400 s
   nodes; `visible_loss_table`). Every flown stream has a visible placement up
   to two hours. A burn past the longest visible one raises, and the chamber
   search adds a chamber.
3. **`chain_growth(..., loss_models=...)`** takes one loss model per cycle.
   `departure_levers --visible` builds the tables once and flies every lever.

## Results (`make departure-levers-visible`)

Doubling time, flown chain, spray cup, film carried, hydrogen as slush:

| Lever | Centred (ADR 0065) | **In view** | Mean loss, km/s | Burn, min |
|---|---|---|---|---|
| Methane AR100, baseline | 3.36 yr | **3.59** | 0.81 | 31.6 |
| Methane, 3 Hz | 3.02 | **3.10** | 0.44 | 21.2 |
| Methane, two chambers | 4.70 | 4.86 | 0.38 | 19.2 |
| Methane, 3.5 kg pitch | 3.26 | 3.52 | 0.96 | 36.1 |
| Methane, 3 Hz + 3.5 kg pitch | 2.91 | **3.01** | 0.55 | 24.1 |
| Hydrogen, bulged, 2 Hz, liquid | 2.66 | 2.91 | 1.42 | 51.6 |
| **Hydrogen, bulged, 2 Hz** | 2.42 | **2.61** | 1.47 | 53.4 |
| Hydrogen, 2.4 Hz | 2.30 | **2.44** | 1.21 | 44.0 |
| Hydrogen, two chambers | 2.60 | 2.72 | 0.72 | 29.0 |
| Hydrogen, two chambers, 2.4 Hz | | 2.58 | 0.55 | 24.1 |
| Solved 4884 K | 2.49 | 2.68 | 1.32 | 47.8 |
| Solved 5500 K | 2.59 | 2.77 | 1.57 | 57.4 |
| Solved 6000 K | 2.68 | 2.88 | 1.65 | 61.1 |
| Solved 7000 K | 2.69 | 2.86 | 1.06 | 39.8 |

- **Visibility costs 0.19 yr for the flown hydrogen chamber and 0.23 yr for
  methane.** The longer the burn, the more it costs: 3 Hz methane loses only
  0.08 yr.
- **Pulse rate is worth more than before.** Hydrogen at 2.4 Hz gains 0.17 yr
  over 2 Hz, against 0.12 yr centred.
- **A second chamber still does not pay.** Halving the burn cuts the loss from
  1.47 to 0.72 km/s, but the 46 t chamber costs more than the Oberth leverage
  recovers: 2.72 yr against 2.61. Two chambers at 2.4 Hz (2.58) only tie one
  chamber at 2 Hz.
- **The temperature ranking holds.** 4884 K still beats 5500 K (ADR 0065).

**The impact angle.** Earth's bending moves the impacts 2-27 degrees off
head-on during a burn. Charged at the actual angle and closing speed, it is
worth 0-1% of delivered mass for the bulged hydrogen chamber, within 0.04% at
the flown chain's 154-175 degrees. The ledger's exact head-on debit stands.
For the record, the bulged chamber's jet leaves at about 1,340 s; per kilogram
carried, gross is about 1,390 s and net of the head-on debit 1,060-1,085 s.

## Not yet done

- **The cost book, seed and harvest still fly the centred table.**
  `seed_cost.design_chain` and everything downstream of it (`make
  survivable-cost`, `make hardware-recovery`, `make slush-hydrogen`,
  `make growth-cost`) need `loss_models` threaded through. The tables cost
  about 6 min per process; vendoring them as data, with a test that
  regenerates a node, would keep the reports fast.
- **Misses are not priced.** Under the visible placement about half the stream
  would enter the atmosphere if it missed. The author accepts that: a
  polyethylene rod arriving at 55 km/s ablates.
- **Planar, two-body, one exhaust speed.** The parking orbit's plane is taken to
  contain both excess vectors, which its apoapsis burns can arrange. The loss is
  computed at `TABLE_EXHAUST` (11 km/s), as `finite_burn_loss`'s table is.

## Reproduction

- `make departure-levers-visible` (about 40 min on three workers). The
  centred column is `make departure-levers`.
- The impact-angle and Isp figures were a scratch run (gitignored): each
  cycle's ledger burn re-integrated with `orbit_turn_analysis._propagate_burn`,
  with the per-pulse exhaust from `chamber_isp._effective_exhaust` at the
  actual relative velocity, against the scalar head-on closing speed.
- Tests: `tests/test_departure_visibility.py` (one slow),
  `tests/test_departure_levers.py`.
