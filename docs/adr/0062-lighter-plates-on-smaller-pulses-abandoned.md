# ADR 0062: Lighter plates on smaller pulses: abandoned

Date: 2026-10-14

Status: Abandoned (negative result). The author stopped the study once it was
clear it did not move growth: the plate is not where growth is lost.

## Context

The spray cup pushes the 1500 t launch unit with 12 MN s pulses at 4 Hz on a
150 t plate: about 74-92 kg of PuffSat a pulse, 48 MN, 3.3 g at the start. The
author asked (2026-10-14) whether smaller pulses would deliver more. The parent's
`eq:plate_pulse_size`, `J = 8 m_s nu s`, scales the plate with its pulse, so a
smaller pulse means a lighter plate and a lower peak acceleration.

The question started as an all-rocket push, with the departure chamber taking
the push too. That was dropped in the grill. A pressure vessel's thrust-to-weight
stays near 3 at any size, against the plate's 33. The chamber must also carry
5-10 kg of charge per kilogram of impactor to hold its wall temperature, so it
delivers about half what the plate does (about 375 t against 750 t, on rough
estimates).

## The model (`src/light_plate_push.py`, `make light-plate-push`)

Settled in the grill (CONTEXT.md: **growth-wave floor**, **delivery acceleration
cap**, **delivery push objective**):
- **The push flies against gravity**, in the orbit plane, from the lob at 400 km.
- **Each PuffSat is aimed on its own.** It meets the stack at a flight-path angle
  that tilts the thrust, with its perigee at or above 400 km. The stack must
  therefore stay at or above 400 km for the whole push.
- **Deliveries** load k = 10 and are held to 4 g felt by the payload (a 3.64 g
  mean, under an assumed 10% spring ripple), with the pulses tapering past it.
- **The plate** is 150 t × J / 12 MN s. A sensitivity holds 20 t of it fixed.
- **The booster** is fixed in size: 1500 t at a 0.75 km/s climb, less for faster.
- **Search.** Pulse size and a 4-node aim profile are chosen by scipy
  differential evolution (seed 7, population 10 per parameter, 40 generations)
  at each climb speed, for the most data-centre tonnes per booster flight at the
  20-day orbit.

## Results (slowest three-synodic wave, 56.53 km/s at 400 km)

| Case | Lob climb | Plate | PuffSat a pulse | Peak g | Data centre per booster flight | Per PuffSat kg |
|---|---|---|---|---|---|---|
| Today: 12 MN s, as flown | 1.1 km/s | 150 t | 71-88 kg | 7.1 | 535 t | 7.5 |
| Today, with the 4 g cap | 1.1 km/s | 150 t | 71 → 45 kg | 3.64 | 535 t | 7.5 |
| Today's plate, aimed, on the parent's lob | 0.75 km/s | 150 t | | | 578.5 t | |
| **Best: 8.28 MN s** | 0.75 km/s | 104 t | 49 kg | 3.64 | **623.5 t** | 8.2 |
| 5.92 MN s | 1.0 km/s | 74 t | 35 kg | 3.42 | 623.2 t | 8.6 |
| 3.43 MN s | 1.5 km/s | 43 t | 20 kg | 2.25 | 573.1 t | 8.9 |
| 2.34 MN s | 2.0 km/s | 29 t | 14 kg | 1.84 | 484.4 t | 9.3 |

On the fastest two-synodic wave (65.13 km/s): today 610 t, best 701.7 t. With 20 t
of the plate held fixed, the best falls to 617.9 t.

The gain over today, 89 t (17%), is two halves:
- **Per-PuffSat aim** lets the stack use the parent's 0.75 km/s lob instead of
  1.1 km/s: +43.5 t.
- **The lighter plate:** +45 t.

The 4 g cap costs deliveries nothing: it stretches the push from about 230 to
300 s.

**A growth push cannot fly the delivery optimum's plate.** Growth flies the
ledger's lighter loading (k from 8.52 down to about 1, since PuffSats are its
currency), so the stack stays heavy, at 2.2-2.7 g. On the 8.28 MN s plate and the
0.75 km/s lob, it falls back to 400 km before reaching the parking orbit at every
loading price. The smallest plate a growth push can fly on that lob is about
**8.35 MN s** (a 104 t plate), barely above the delivery optimum. An 8.5 MN s
plate (106 t) would serve both, delivering 621 t. The growth chain behind it was
started but stopped before it finished, when the study was abandoned.

## Decision

**Keep the 150 t spray cup on the parent's 1.1 km/s lob.** The plate was about
44% above the smallest one growth can fly: that is a design margin, not waste. Taking it
out is worth about 8% of a delivery and does nothing for the departure, where
growth is lost (ADR 0063).

## Consequences

- **Per-PuffSat aim is the one finding worth keeping.** It holds the stack above
  400 km on the parent's 0.75 km/s lob, worth about 43 t a delivery (8%) with the
  plate unchanged. It has not been checked against the growth push's loading or
  the PuffSats' steering budget.
- The three CONTEXT.md terms stand as requirements for any future push model:
  the growth-wave floor, the delivery acceleration cap, and the delivery
  objective.
- `growth_ledger` takes no push hook; the impulsive push stays.

## Reproduction

`make light-plate-push` (about 20 min on 2 workers) prints today's plate and the
curve on both waves, with and without the fixed share. Each point prints its
pulse and aim. Tests: `tests/test_light_plate_push.py`, including that the growth
loading cannot hold 400 km on the 8.28 MN s plate.
