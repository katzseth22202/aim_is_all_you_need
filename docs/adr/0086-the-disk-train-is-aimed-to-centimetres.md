# ADR 0086: The Sun-wave disk train is aimed to centimetres, and the thick disk does not pay

Date: 2026-10-27

Status: Accepted. It answers the child's thick-disk aim request (items 1-4). Its item 2 rests
on the impact simulation's Result 23 (`puffsat_impact_simulation` @ `3581fbd`,
`docs/thick_disk_answers.md`, ADR-0074 there). Reproduce with `make disk-train-aim` (~1 min,
`ARGS="--no-price"` ~25 s). The proposed new headline, the 1 kg disk on the wider charge with
its track hold charged, is left to the author. `HEADLINE_STARTS` is unchanged.

## Context

The child proposed a thick Sun-wave disk: 2.5 kg, 1 cm and 38 cm of molded graphite, flown at
1 Hz. That replaces the headline's 1 kg, 3.6 mm, 40 cm disk at 2 Hz. It asked four questions at
the 9.86 R_sun baseline (head-on closing at 392 km/s):

1. Can each disk be re-aimed in time while the ship corrects its own drift? Take each pulse's
   impulse error as 1-5% of the kick.
2. What aim tolerance does the impact simulation's off-centre answer set?
3. How far do CMEs and sunlight push an edge-on disk?
4. What thrusters and cold gas does each disk need?

The parent's `sec:sun_wave_ranging` assumed a disk-to-charge tolerance of "meters" and a last
correction a minute or two out. The simulation's Result 23 replaced both.

- **The disk must clear the bore.** It is caught whole until its rim clips the bowl's 0.25 m
  bore. On a charge 1.3 times the disk's radius that happens 6 cm off centre, and on a 1.5x charge
  through a 0.285 m bore 9.5 cm off centre.
- **A matched charge loses coupling to offset.** On a charge as wide as the disk, an offset `d`
  leaves the lens `2d / (pi R)` of the disk unstruck. That is 5% at 1.5 cm.
- **An off-centre hit pushes the ship sideways.** The side impulse is 0.4% of the axial impulse
  at 6 cm on the matched charge, 0.8% at 6 cm on x1.3 and 1.3% at 9.5 cm on x1.5. It is odd in
  the offset, so it is linear near the axis.
- **The coupling changes with the charge,** over the thin disk on its matched charge at k = 20:
  0.94 for the 2.5 kg disk matched, 1.11 on x1.3 and 1.05 on x1.5. The thin disk spread to x1.3
  gets 1.08.

## Decision

**1. The tolerance is centimetres** (`dispersion_requirement`). With one disk in a thousand
allowed to clip the bore, each axis may carry `clearance / sqrt(2 ln 1000)`:

- 1.35 cm for the 1 kg disk (5 cm clearance);
- 1.61 cm for the 2.5 kg disk on x1.3;
- 2.56 cm for the 2.5 kg disk on x1.5.

The ranging network's sideways error is 1.3 cm at the train's ends: the parent's 1 cm ranges
times 1.3 on the 0.99 h train. It fills most of the budget. A matched charge flown at 1.35-1.61 cm
also loses 5.4-6.8% of its coupling to the unstruck lens. The ledger charges that loss
(`flown_coupling`).

**2. Predict-and-steer with a minute's blind window does not close** (`blind_miss`,
`loop_gain`). A disk steered on the network's prediction cannot see the kicks of its last `B`
seconds. Each kick carries the side impulse `g r` of its offset `r`. The miss on each axis is
`g sigma a sqrt(dt B^3/3)`, which the code verifies against a Monte Carlo
(`simulate_blind_miss`). The same formula gives the parent's own 1 mrad case its "half a meter".

At the required dispersion and `B` = 60 s, the miss is 0.5 m for the 1 kg train and 2.2 m for the
2.5 kg train on x1.3. Since the miss feeds the next offset, the dispersion has a steady value
only when the loop gain `G` is below 1. `G` is 35-134, so the loop is unstable.

**3. The ship holds its track** (`hold_fraction`, `Design.sun_hold`). It cancels each pulse's
side impulse and its torque `J r`, with thrusters at the plate's 13.0 m radius, before the next
pulse.

- **The cost is independent of pulse size and cadence.** With `E|r| = sigma sqrt(pi/2)`, the
  side impulses sum to `g E|r|` of the Sun boost and the torques to `E|r| / lever` of it.
- **The departure-wave ship spends** 12 + 13 m/s (1 kg, matched), 22 + 13 (1 kg, x1.3) and
  27 + 15 (2.5 kg, x1.3), as side plus torque. That is 4-7 t of storable propellant. The push-wave
  ship spends twice this.
- **It is charged as a dead line** (`sun_hold`), priced as storable propellant.
- **The alternative was not adopted.** In it the ship broadcasts each measured kick to disks a
  pulse or two out (`short_loop_window`). It holds `G <= 0.5` only with a 1-3.5 s window, and
  needs 13-480 mN thrusters on every disk.
- **At the request's 1-5% the hold would cost** `1.25 eps x boost`: 123-614 m/s on the departure
  ship. The simulation's side impulse at the required dispersion is 0.1-0.35%, so that range does
  not arise.

**4. The ship must know each kick to about 5e-5 of itself** (`kick_knowledge`). An error in a
measured kick is a velocity nobody corrects until the network sees the ship move, about 20 s
later. On the typical budget, what is left for the ship is:

- 0.34 cm on the 1 kg train, which needs 0.048 mm/s per pulse on 1.1-1.7 m/s kicks;
- 0.95 cm on the 2.5 kg train on x1.3, which needs 0.19 mm/s on 3.3-4.3 m/s kicks.

This is the binding open requirement. It falls on the ship's accelerometers or a faster ranging
of the ship.

**5. The prices** (9.86 R_sun, X 60, re-optimized for the dear seed):

| Train | arc (h) | kg/pulse dep / push | Sun plate + bag (t) | hold (t) | growth | BE cheap / dear |
|---|---|---|---|---|---|---|
| 1 kg, 2 Hz, matched: ADR 0085 (centred, no hold) | 0.986 | 1.00 / 1.70 | 15.9 / 22.8 | 0 / 0 | 3.226 | 130.7 / 158.2 |
| 1 kg, 2 Hz, matched, flown at 1.35 cm | 1.031 | 1.00 / 1.69 | 15.9 / 22.7 | 4.1 / 8.0 | 3.037 | 142.3 / 180.8 |
| **1 kg, 2 Hz, x1.3 charge** | 0.923 | 1.00 / 1.71 | 16.0 / 23.1 | 5.9 / 11.5 | **3.255** | **128.9 / 155.0** |
| 2.5 kg, 1 Hz, matched | 0.884 | 2.50 / 4.17 | 29.6 / 43.1 | 4.9 / 9.5 | 2.651 | 182.3 / 266.0 |
| 2.5 kg, 1 Hz, x1.3 (the proposal) | 0.722 | 2.50 / 4.30 | 32.9 / 56.3 | 7.1 / 13.7 | 2.953 | 150.0 / 195.0 |
| 2.5 kg, 1 Hz, x1.5 | 0.765 | 2.50 / 4.27 | 31.1 / 52.9 | 11.6 / 22.2 | 2.790 | 164.0 / 226.0 |

- **The headline flown at centimetres on its matched charge costs $23 dear.** Nearly all of it is
  the 5.4% lens loss. The hold is under $2.
- **The wider charge pays that back and $3 more.** The 1 kg disk on x1.3 breaks even at $155
  dear. We propose it as the new headline.
- **The thick disk costs $40 more than that** ($195). Its pulse is 2.5 times the energy, and the
  plate's survival floor rises with the pulse, so the Sun plate doubles. The push-wave ship flies
  4.3 kg pulses and its spring sits at the 8 m stroke limit.
- **Reconciliation with Result 23's bowl.** The simulation's 40-43 t bowl uses its own absolute
  impulse (2.32 MN s, J/mv = 2.39). The ledger flies the vendored stage law times the coupling
  ratio (1.83 MN s), which is how the simulation itself quotes ship dv. Bowl mass goes as J, so
  the two agree: 40-43 t / 1.27 = 32-34 t, against 32.9 t here.

**6. Drift** (`cme_drift`, `edge_sunlight_share`). The simulation's law, re-flown on each disk:

- **CME:** in the extreme case, the densest of 29 CMEs times Parker's shock compression at
  2500 km/s, the pressure is 3.7 mPa. The request's "pessimistic" 1e-3 Pa sits between that and the
  typical 0.08 mPa.
- **Tilt sets the CME push, not the edge.** Edge-on, the face takes the out-of-plane flow
  `sin(tilt_rel)`. In Parker's 23 deg sheath the thick disk drifts 0.64 m/s and 2.9 km over a
  2.5 h passage, against 1.69 m/s and 7.6 km for the 1 kg disk. Tumbled face-on it drifts 1.5 m/s
  and 6.7 km.
- **Sunlight is 2.16 mPa.** Edge-on, the limb still lights the faces, so the 1 cm disk feels 1/13
  of its face-on push, not the request's 1/30. That is 0.8 m/s summed over the inbound dive,
  modelled. Only its 1% modelling error needs gas.
- **The disk's thrusters absorb all of it.** The ship holds one track for the whole train and
  cannot chase single disks.
- **Fix latency is the gap.** Over the 20 s a fix takes to reach a disk, an extreme 23 deg sheath
  moves the 2.5 kg disk 1.4 cm and the 1 kg disk 3.8 cm. That is more than the budget leaves, so in
  such a sheath those disks are deliberate misses, as the parent already allows, unless the
  latency falls or the disk senses its own push.

**7. Thrusters** (`thruster`). The disk's final move, three network sigmas (3.9 cm) in 10 s,
sizes the thrust at 1.6 mN for the 1 kg disk and 3.9 mN for the 2.5 kg disk. The worst CME needs
less. The gas is 2.9 g at 9.86 R_sun. The 10 mN class is ample, and the 25 g carried (sized at
4 R_sun) stays.

## Consequences

- The ledger gains `Design.sun_rate`, `sun_coupling` and `sun_hold`, plus a `sun_hold` dead line
  priced as storable propellant. Their defaults reproduce ADR 0085 exactly.
- The parent's "meters" tolerance and its one-to-two-minute last correction are both wrong at
  the bowl's real geometry. The parent's own 0.1% tilt term was the right order, but at
  centimetres it no longer fits.
- **Proposed, not adopted:** a headline of 1 kg at 2 Hz on the x1.3 charge with its hold, at
  $155 dear. Two inputs are assumed:
  - Result 23 ran the thin disk on x1.3 only on the 0.025 m grid (1.08x, grids agree to 0.4%).
  - Its bore and side impulse there were not traced. This ADR takes a 5 cm clearance and the
    thick disk's x1.3 slope.
- **4 R_sun is not answered.** The simulation ran 392 km/s only.

## Assumptions to revisit

- Side impulse is linear in offset, from Result 23's single point per charge.
- Offsets are 2-D Gaussian. Clips are allowed at one in a thousand.
- The hold cancels every pulse (the upper bound) and is charged as if carried to the arc's end.
- The network's sideways error is 1.3 cm at the 0.99 h train's ends. The shorter 1 Hz train
  would do a little better.
