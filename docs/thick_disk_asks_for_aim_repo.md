# Request to aim_is_all_you_need: aiming a 1 Hz train of thick Sun disks

For the companion repo, then the parent, then a one-line copy down here. Raised 2026-10-27.
The coupling questions are in `todos/thick_disk_impact_request.md`, for
`puffsat_impact_simulation`. Item 2 here needs that repo's off-center answer.

## The proposed disk and train

The Sun boost at the 9.86 R_sun baseline closes head-on at about 392 km/s. The proposal
replaces the 1 kg, 3.6 mm, 40 cm disk with a 2.5 kg, 1 cm, 38 cm molded-graphite disk. It
carries the parent's 4 R_sun per-pulse energy at nearly the parent's footprint. Each disk
keeps the 85-130 g pod behind its sunshade and flies edge-on to the Sun.

- 7.2 t of disks is 2,880 disks.
- At 1 Hz the train is 48 minutes long. Disks are 392 km apart, and the whole train spans
  about 1.1e6 km.
- Each disk gives the ship about 3.4 m/s, for a 0.35 g average. This assumes the impulse per
  kilogram of disk stays at the baseline's.
- The ship sweeps about 5 degrees of arc around the Sun during the burn, against about
  29 degrees for a 4.9 h burn.

The local paper's 2 per second over 4.9 h does not match its 7.2 t of 1 kg disks. The 1 Hz
train above replaces both.

## Ask

1. **Re-aim at 1 Hz.** Each pulse departs from its expected kick. Can each later disk be
   re-aimed in time, with the ship correcting its own drift by conventional rockets? Take a
   per-pulse impulse error as a parameter, 1% to 5% of 3.4 m/s along and across the axis.
   Report the ship's rocket Δv over the burn and the steering authority each disk needs at
   its lead time.
2. **Aim tolerance.** Turn the impact repo's off-center answer (its item 3) into an arrival
   dispersion requirement for a 38 cm disk at 392 km/s, and compare it with the dispersion the
   disk-train estimate now carries.
3. **Space weather drift.** Bound the trajectory push on an edge-on disk from a CME, and from
   sunlight, over the coast and the last hours before perihelion. Edge-on, the outward-moving
   wind and the sunlight both meet the 1 cm rim rather than the face, about 30 times less
   area. A pessimistic CME dynamic pressure is about 1e-3 Pa and sunlight about 2e-3 Pa. If the
   disk tumbles toward face-on, take the full face. Say whether the disk's thrusters or the
   ship's re-aim absorbs the drift.
4. **Thruster sizing.** From items 1 and 3, the cold gas thrust and propellant each disk needs.
   We assumed 10-100 mN class.

## Loads already cleared

Neither CME pressure nor thruster force threatens the graphite. Plate bending under 1e-3 Pa
is about 0.4 Pa, and 1 N through a 1 cm² mount is about 10 kPa, against a breaking stress of
tens of MPa. Those forces matter only for aim, which is what items 3 and 4 ask.

## Outcome

If items 1 and 2 close, the 1 Hz thick-disk train goes into the parent, and the Sun-boost row
of the local table moves to 2,880 disks over 48 minutes. If re-aim cannot keep up at 1 Hz,
the trade is a longer train or smaller disks, with a smaller kick per pulse.
