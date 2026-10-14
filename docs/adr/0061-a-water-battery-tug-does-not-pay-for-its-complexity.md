# ADR 0061: A water-battery argon tug does not pay for its complexity

Date: 2026-10-14

Status: Accepted (negative result). Builds on ADR 0058 (the perigee pump and
the array sold at L1), ADR 0059 (the two-year loop) and ADR 0060 (the seed flies
direct).

## Context

ADR 0058's argon tug failed on its duty cycle. It thrusts only near perigee,
and perigee comes round once an orbit, so the array idled most of the time. The
author proposed (2026-10-14) a fix that uses the seed's payload:

- **Electrolyse the payload water all round the orbit**, and recombine the
  hydrogen and oxygen in a fuel cell during the perigee arc. The water is a
  battery whose storage medium is already paid for: it comes back as water.
  The hydrogen and oxygen are never burned.
- **Spend the stored energy in a high-power thruster at perigee**, at a lower
  exhaust speed than ADR 0058's 2000 s, for more thrust per watt.
- **Count the array as resold**, to data centres at L1.
- **Zero methalox boil-off** while the ship waits: blankets, plus cryocoolers
  drawing about 10 kW off the tug's array (author, 2026-10-14).

## The model (`src/seed_battery_tug.py`, `make seed-battery-tug`)

- **Thruster.** Quasi-steady magnetoplasmadynamic (MPD) on argon: an arc whose
  own current and field push the plasma out, built for MW-to-GW pulses.
  - A thermal argon arcjet was rejected: heavy argon gives only 3-5 km/s,
    barely above methalox.
  - VASIMR was rejected: it loses most of its power ionising argon below about
    5000 s, and needs RF and superconducting magnets at the peak power.
  - Exhaust speed is free over 12-30 km/s at a fixed efficiency of 0.3.
- **Battery.** Each orbit the array (sunlit 0.87, belt-derated 0.9) feeds the
  thrusters directly during the arc and the electrolyser otherwise. The arc
  spends what comes back, up to the thrusters' peak rating. Round trip 0.48
  (electrolyser 0.8, fuel cell 0.6).
- **Heat.** The fuel cell's loss and half the thruster's loss go into the
  payload water during the arc, beyond what the radiators reject meanwhile.
  The radiators are sized for the orbit's average heat. Warmed liquid
  (290-340 K) or melted ice, whichever is lighter. The water a pass needs must
  not exceed the seed.
- **Climb.** Orbit-averaged Gauss equations for tangential thrust on an arc of
  half-width θ about perigee, stepped orbit by orbit with argon leaving.
- **Ship.** It takes over on the ellipse at semi-major axis `a_f` and burns at
  perigee onto its route, with the finite-burn loss charged for the propellant
  it burns. The routes are direct to 12.20 km/s (ADR 0060's headline) and ADR
  0059's two-year loop (its champions' launch excess and later burns, held
  fixed).
- **Money.**
  - Everything is lofted at $150/kg.
  - The climb is paid early: the seed's departure date is fixed, and the
    launch date is free (ADR 0059). Its cost is capital, `x 1.1^climb`.
  - The array is credited at L1: gross price at 0.8 of rating, plus $500/kg
    shipping on 7 kg/kW, 0.25 yr after hand-over.
  - Thrusters, bus, fuel cells and electrolysers are priced per W of rating and
    not resold. Tanks and radiators cost $1000/kg.
- **Worth.** Seed per net dollar over the same route flown without a tug.
  Above 1 the tug pays.
- **Search.** scipy differential evolution over θ (15-150°), `a_f`
  (9 000-90 000 km), exhaust speed, array rating (0.3-40 MW) and peak over
  array (1-300). Seed 7, population 12 per parameter, 60 generations, no
  polish. Climbs are capped at 2 years.

**Unsourced assumptions**, to be replaced if the design is reopened:

| Item | Assumption |
|---|---|
| MPD thrusters | efficiency 0.3; half the loss leaves in the plume; 1 kg/kW of peak |
| Fuel cells | efficiency 0.6, 1 kg/kW |
| Electrolysers | efficiency 0.8, 2 kg/kW |
| Gas tanks | 27 kg per kg of hydrogen held |
| Radiators | 5 kg/m², emissivity 0.85 |
| Array | 7 kg/kW |
| Prices | arrays $10 / $25 / $50 per W gross; burst hardware $0.25 / $1 / $3 per W |

## Results (40 t ship, $5M hull, $150/kg)

Without a tug, direct sends 86.1 t for $198M and the two-year loop sends
129.5 t for $204M.

| Route | Array $/W | Burst $/W | **Worth** | Seed (t) | Climb (yr) | `a_f` (km) | θ | Exhaust (km/s) | Array / peak (MW) | Tug km/s | Argon (t) | Hardware (t) | Water a pass (t) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | 10 | 0.25 | **1.31** | 240 | 2.00 | 30 900 | 50° | 14.4 | 5.8 / 24.7 | 2.77 | 350 | 180 | 137 |
| direct | 10 | 1 | 1.21 | 222 | 1.96 | 24 800 | 58° | 13.3 | 4.9 / 14.1 | 2.65 | 353 | 145 | 89 |
| direct | 10 | 3 | 1.04 | 159 | 1.95 | 12 500 | 66° | 12.0 | 2.1 / 4.0 | 1.79 | 236 | 70 | 24 |
| direct | 25 | 0.25 | 1.22 | 237 | 1.98 | 28 600 | 46° | 12.4 | 4.5 / 23.9 | 2.70 | 400 | 170 | 122 |
| direct | 25 | 1 | 1.13 | 210 | 1.96 | 20 700 | 55° | 12.4 | 3.9 / 12.0 | 2.46 | 345 | 126 | 70 |
| direct | 25 | 3 | 0.98 | 153 | 1.87 | 11 800 | 64° | 12.0 | 1.9 / 5.0 | 1.67 | 218 | 67 | 31 |
| direct | 50 | 0.25 | 1.10 | 201 | 2.00 | 17 800 | 47° | 12.0 | 2.9 / 14.8 | 2.25 | 321 | 118 | 76 |
| direct | 50 | 1 | 1.04 | 204 | 2.00 | 19 000 | 52° | 12.0 | 3.2 / 12.1 | 2.35 | 338 | 117 | 68 |
| direct | 50 | 3 | 0.93 | 149 | 1.86 | 11 200 | 62° | 12.0 | 1.7 / 4.7 | 1.56 | 201 | 62 | 27 |
| 2-yr loop | 10 | 0.25 | **1.25** | 350 | 1.93 | 30 400 | 64° | 15.5 | 6.5 / 20.4 | 2.85 | 358 | 182 | 159 |
| 2-yr loop | 10 | 1 | 1.17 | 326 | 1.92 | 25 000 | 73° | 14.7 | 5.7 / 12.3 | 2.77 | 354 | 151 | 108 |
| 2-yr loop | 10 | 3 | 1.05 | 276 | 1.81 | 16 800 | 77° | 12.3 | 3.7 / 6.0 | 2.33 | 340 | 108 | 47 |
| 2-yr loop | 25 | 0.25 | 1.16 | 345 | 1.97 | 28 100 | 61° | 13.0 | 4.9 / 18.3 | 2.77 | 415 | 166 | 134 |
| 2-yr loop | 25 | 1 | 1.05 | 270 | 1.81 | 15 700 | 71° | 12.6 | 2.9 / 10.0 | 2.19 | 308 | 107 | 82 |
| 2-yr loop | 25 | 3 | 0.99 | 252 | 1.74 | 13 900 | 74° | 12.2 | 2.9 / 5.1 | 2.02 | 286 | 89 | 36 |
| 2-yr loop | 50 | 0.25 | 1.06 | 331 | 1.99 | 24 000 | 56° | 12.0 | 3.8 / 18.4 | 2.61 | 416 | 153 | 120 |
| 2-yr loop | 50 | 1 | 1.01 | 297 | 1.98 | 18 600 | 65° | 12.1 | 3.2 / 9.4 | 2.40 | 362 | 116 | 70 |
| 2-yr loop | 50 | 3 | 0.93 | 226 | 1.88 | 11 600 | 69° | 12.0 | 2.0 / 3.7 | 1.64 | 225 | 66 | 22 |

The heat always went into liquid water. The 60 t ship was not searched.

## Decision

**Do not fly a tug.** The battery fixes ADR 0058's duty cycle, but the result
is the same: the tug pays only when its hardware is nearly free.

## Consequences

- **The battery works as intended, and the peaks are modest.** The search
  settles on low hand-overs (12 000-31 000 km semi-major axis) and wide arcs
  (±46-77°), at peaks of 4-25 MW: 2-5x the array, not gigawatts. The gas
  stored is under 0.4 t of hydrogen a pass.
- **The tug's gain is capped by the money, not the physics.** It doubles to
  triples the seed (240 t against 86 t direct), but it costs as much again as
  the ship ($420M net against $198M). It reaches 1.31x only with a $10/W array,
  which nets almost nothing after the L1 sale, *and* burst hardware at
  $0.25/W. At $3/W of burst hardware it loses at every array price. That makes
  the price of fuel cells, MPD thrusters and their bus the deciding unknown.
  No source is in hand.
- **Two bounds bind, and both cut against the author's hopes.**
  - Every best case climbs for about the full 2 years allowed, not 6 months. A
    six-month climb needs about four times the power: the hand-picked
    0.77-year climb on the direct route scored 0.82.
  - Every best case runs at or near the 12 km/s floor, so a lower exhaust speed
    does help, as the author expected. But argon MPD thrusters below about
    1200 s are not credible, so the floor stays.
- **It does not stack well with the loop.** The tug on the loop (1.25x the
  loop) is worth about the same as the tug on direct (1.31x direct). Loop and
  tug together are about 1.5x direct for roughly four years of climb and loop.
- **Heat matters as much as power.** Each pass dumps up to 160 t of water's
  worth of heat. It fits inside the seed, but only because the seed is the
  payload. A design that delivered something other than water would need its
  own heat sink.
- **Reopen it if** burst hardware (fuel cells, MPD thrusters and a
  high-current bus) is shown at well under $1/W, *and* arrays at L1 are worth
  their gross price there. Both are needed.

## Limits

Orbit-averaged and coplanar, with no drag, Moon or Sun, and no plane change.
The arc and the eclipse are not phased against each other. Thruster efficiency
does not vary with exhaust speed or power. The loop's trajectory is held at ADR
0059's champion rather than re-searched for a cheaper departure. The ship's
payload capacity is not checked against 240-350 t of seed. Cryocooler mass is
not charged.

## Reproduction

`make seed-battery-tug` (about 1 h per ship and route on 2 workers). The table
above is `--ships 0` with the default routes, seed 7. The search is
deterministic. The script now prints each champion's whole design vector. The
run behind this table printed only the first four entries; its peak ratings
are the "peak" column. Fast tests:
`tests/test_seed_battery_tug.py`.
