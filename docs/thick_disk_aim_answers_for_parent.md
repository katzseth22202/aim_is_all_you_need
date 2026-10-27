# Aiming the Sun-wave disk train: answers to the thick-disk aim request (for the parent)

Raised 2026-10-27 in `aim_is_all_you_need` (ADR 0086). **Written to be copied into
`Balloon-Pulse-Propulsion`**, then summarised in one line for the child. It answers items 1-4 of
the child's thick-disk aim request, and uses the impact simulation's Result 23
(`puffsat_impact_simulation` @ `3581fbd`, `docs/thick_disk_answers.md`) for item 2.

Reproduce with `make disk-train-aim` (~1 min).

## Note to the parent: the aim tolerance is centimetres, not meters

`sec:sun_wave_ranging` says the disk-to-charge tolerance "is known only as meters", and it lets
the last correction fall "a minute or two before impact". Result 23 replaces both.

- **The disk has to clear the bore.** It is caught whole only until its rim clips the bowl's
  0.25 m bore, 5-6 cm off centre. Allowing one clip in a thousand, each axis may carry
  **1.35 cm (1 kg disk) or 1.61 cm (2.5 kg disk on a 1.3x charge)**.
- **The network uses most of that budget.** Its own sideways error is about 1.3 cm at the train's
  ends.
- **A matched charge is worse.** On a charge as wide as the disk, a 1.5 cm offset already loses
  5% of the coupling. At the required dispersion the matched-charge headline loses 5.4%, which
  costs $23 dear.
- **Every off-centre hit kicks the ship sideways,** by 0.1-0.35% of the pulse. In the minute a
  disk cannot see, those kicks move the ship 0.5-2.2 m. The predict-and-steer loop is unstable
  at this tolerance.

Several sentences in `sec:sun_wave_ranging` therefore need rewriting:

- "against a requirement of meters";
- "Holding consecutive disks within 5 m of their slots";
- the 70 s for 5 m;
- "far inside the meters the disks need".

The answer is below: the ship holds its own track, and must know each kick to about 5e-5 of
itself.

## Headline

| Train (9.86 R_sun, X 60) | Sun plate + bag dep / push (t) | growth | BE cheap / dear |
|---|---|---|---|
| 1 kg, 2 Hz, matched, centred (ADR 0085, today's paper) | 15.9 / 22.8 | 3.23 | 131 / 158 |
| 1 kg, 2 Hz, matched, flown at 1.35 cm with its hold | 15.9 / 22.7 | 3.04 | 142 / 181 |
| **1 kg, 2 Hz, 1.3x charge, with its hold (proposed)** | 16.0 / 23.1 | **3.26** | **129 / 155** |
| 2.5 kg, 1 Hz, 1.3x charge, with its hold (the child's proposal) | 32.9 / 56.3 | 2.95 | 150 / 195 |

- **Keep the 1 kg disk.** Fly it on the wider charge. Charged for centimetre aim, it costs $3 less
  than today's headline.
- **The thick disk at 1 Hz costs $40 more.** Its Sun plate doubles, because the plate's survival
  floor rises with the pulse. The push-wave ship flies 4.3 kg pulses with its spring at the 8 m
  stroke limit.
- **What the thick disk gains is aim.** Its tolerance is 6 cm rather than 5. The ship's kick
  knowledge requirement is four times looser. A tilted CME sheath drifts it 2.6 times less.

The new headline is proposed, not adopted. The ledger's `HEADLINE_STARTS` is unchanged until the
author decides.

## Item 1. Re-aim

- **The parent's scheme does not close at centimetres.**
  - Steering on the network's prediction leaves the kicks of the last `B` seconds unseen. Each
    pulse's side impulse is `g r`, with `g` = 0.07-0.13% per cm of offset.
  - The miss is `g sigma a sqrt(dt B^3/3)`: 0.5 m (1 kg) and 2.2 m (2.5 kg) at `B` = 60 s.
  - The miss feeds the next offset with a loop gain of 35-134, so the loop is unstable.
- **The ship holds its track.** It cancels each pulse's side impulse and torque before the next
  pulse.
  - The departure-wave ship spends 25 m/s (1 kg, matched), 35 m/s (1 kg, 1.3x) and 42 m/s
    (2.5 kg, 1.3x), or 4-7 t of storable. The push-wave ship spends twice that.
  - The cost is the same share of the boost whatever the pulse size or cadence.
  - The ledger charges it.
- **At the request's 1-5% impulse error** the hold would cost 123-614 m/s on the departure ship.
  The simulation's error is 0.1-0.35%, so that range does not arise.
- **The other way is not recommended.** The ship can instead broadcast each measured kick to the
  disks a pulse or two out. That needs a 1-3.5 s last correction and 13-480 mN on every disk.
- **The open requirement is kick knowledge.** The network sees the ship move about 20 s late, so
  the ship must know each kick to 0.05 mm/s (1 kg, 2 Hz) or 0.19 mm/s (2.5 kg, 1 Hz). That is
  about 5e-5 of the kick.
- **Steering authority at lead time.** With the ship holding track, a disk only makes its final
  move: three network sigmas, 3.9 cm, in 10 s.

## Item 2. Tolerance against the budget (one axis)

| Train | required | network | CME drift in a 20 s fix, typical / extreme | left for the ship |
|---|---|---|---|---|
| 1 kg on its 5 cm clearance | 1.35 cm | 1.3 cm | 0.08 / 3.75 cm | 0.34 cm / none |
| 2.5 kg, 1.3x charge | 1.61 cm | 1.3 cm | 0.03 / 1.42 cm | 0.95 cm / none |
| 2.5 kg, 1.5x charge | 2.56 cm | 1.3 cm | 0.03 / 1.42 cm | 2.2 / 1.7 cm |

In an extreme, tilted CME sheath the disks inside it become deliberate misses, as the paper
already allows, unless the fix latency falls or each disk senses its own push.

## Item 3. Drift (9.86 R_sun, 2.5 h passage)

- **The extreme CME is 3.7 mPa.** That is the densest of 29 CMEs times 2.03 shock compression at
  2500 km/s. The request's 1e-3 Pa sits between that and the typical 0.08 mPa.
- **The out-of-plane tilt sets the push, not the edge.**
  - In Parker's 23 deg sheath the 2.5 kg disk drifts 0.64 m/s and 2.9 km.
  - The 1 kg, 40 cm disk drifts 1.69 m/s and 7.6 km.
  - Tumbled face-on, the 2.5 kg disk drifts 1.5 m/s and 6.7 km.
- **Sunlight is 2.16 mPa.** Edge-on, the limb still lights both faces, so the 1 cm disk feels
  1/13 of its face-on push, not 1/30. Over the inbound dive that sums to 0.8 m/s, modelled.
- **The disk's thrusters absorb the drift.** The ship holds a single track for the whole train.

## Item 4. Thrusters

- **Thrust:** 1.6 mN (1 kg) and 3.9 mN (2.5 kg), set by the final move. The 10 mN class is ample.
- **Gas:** 2.9 g per disk at 9.86 R_sun. The 25 g sized at 4 R_sun stays.

## Open

- **For the simulation:** the thin 1 kg disk on the 1.3x charge needs its bore clearance and
  side impulse traced, and a run at the 0.0125 m grid. This ADR assumes 5 cm and the thick
  disk's slope.
- **4 R_sun** is unanswered: Result 23 ran 392 km/s only.
- **The ship's kick knowledge** (5e-5) needs a design.

## One line for the child

The 1 Hz, 2.5 kg train closes on aim but costs $40/kg dear more than the 1 kg disk flown on the
same 1.3x-wide charge ($195 against $155): keep the 1 kg disk, and aim it to about 1.5 cm with the
ship holding its own track.
