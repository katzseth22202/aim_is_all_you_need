# The solar dive grown at Ceres: answers to S56 (for the parent)

Raised 2026-10-22 in `aim_is_all_you_need` (ADR 0083). **Written to be copied into
`Balloon-Pulse-Propulsion`.** It answers S56 items 1-9
(`docs/ceres_isru_dive_asks_for_aim_repo.md` @ `75729e3`). Reproduce with `make ceres-dive`
(~5 min).

## Headline (for the paper, if it wants one)

**Ceres removes the growth lob. The steady price is $1.46/kg at L1 at 4 R_sun, and $1.70 at
9.86 R_sun.** That is half the solar dive's $2.81 extreme best case, on the same book. The
Ceres loop doubles every 6-8 months.

- **$1 is a delivery question, not a Ceres one.** The delivery lob alone is $1.01/kg: the
  2 Hz, 1 g delivery climbs at 2.43 km/s, and that climb is charged on the propellant floor.
- On the child's 4 Hz delivery (S37, unsolved) the price is $1.13. With Ceres film at $1,
  3000 t ships and a 2 Hz Ceres push it reaches **$0.997** at 4 R_sun.
- At 9.86 R_sun, $1 needs methalox at the sourced low end as well ($0.88).

## Where the parent's $0.99 moves

| line | parent | here (4 R_sun) |
|---|---|---|
| delivery lob | 0.622 (climb factors left off) | 1.006 |
| Earth stream | ~0.34 (c_E ~$6) | 0.426 (c_E $10.74) |
| charges + plate | 0.032 | 0.032 |
| **total** | **0.99** | **1.463** |

c_E is higher for three reasons:

- **The Sun bowl sits at its survival floor**: 23 t per 1000 t, not 0.74%. The solar dive's
  own is 19 t on 654 t.
- **The freight is priced** at $4.4/kg, against the $3 placeholder.
- **The hole surface needs 15-50% more pellets than law 1.**

## Item 1. Push law

- **The departure check reproduces:** law 1 gives 8.07 against the paper's 8.00. The
  simulation's own hole surface gives 9.67 there, so law 1 is pessimistic at 130 deg.
- **We did not use the ratio transplant** (the polyethylene surface times law 1's water over
  polyethylene). It diverges where polyethylene's toll reaches zero (k = 26 at 67 km/s).
- **Instead:** the surface's bonds-returned rows times water's frozen toll, the ice pellet's
  own toll charged. Results at 80 deg:

| w km/s | best k | v_e | pellets/kg | water/kg |
|---|---|---|---|---|
| 95 | 14.8 | 13.8 | 0.145 | 2.15 |
| 150 | 26.6 | 15.8 | 0.065 | 1.73 |
| 200 | 28.0 | 20.8 | 0.041 | 1.14 |
| 300 | 29.8 | 30.4 | 0.023 | 0.68 |

- **51 MJ/kg is the right frozen toll.** The ionization comes back on expansion, but the
  bonds re-form only at 3000-4000 K. Returning 31% of the toll is worth under $0.02.

## Item 2. Orbits

- **Every figure reproduces.** Dive Δv is 14.67 / 15.84 km/s prograde and 21.14 / 19.97
  retrograde. Stream speeds and angles at Ceres are tabulated for X 60-300.
- **Ceres' real orbit:**
  - Its eccentricity moves the direct dive from 17.1 km/s (Ceres' perihelion) to 14.6
    (aphelion), and the landing by ±6 deg.
  - The Earth-bound dive's worst plane tilt is ~15 deg, which costs 0.1 km/s.
  - **Coplanar circular is enough for the ledger.**

## Item 3. Phasing

**Your misses reproduce, and so does the 23.3 km/s retrograde inbound closure. But a single
co-orbital depot does not close the loop.**

- **Why one depot fails.** The depot holds its phase, but ships leaving it land their stream
  `miss` further on. The ships pushed there send the stream on again.
- **What closes: a ring of three depots 120 deg apart**, each depot's stream landing on the
  next.
  - Its dive costs within 0.22 km/s of the direct dive, and exactly the direct dive at
    4 R_sun, X 150.
  - The two depots off Ceres are fed by a 200 s steam tug: 0.61 km/s over 20 years, about
    $0.002/kg at L1.
- **The other options:**
  - **(b)** The single-depot retrograde closure costs 23.3 km/s against 14.7.
  - **(c)** Off-perihelion or turned boosts move the landing only 1-2 deg per 0.1 km/s.
  - **(d)** A Mars flyby turns the stream 1.5 deg.
- **The disks** dive head-on from the ship's own depot at 20.1 km/s.

## Item 4. Two clocks

- **One schedule feeds both.** Each depot's Earth window is open ~155 days of the 467-day
  synodic period for +0.5 km/s. Three depots cover 97-100% of the time. Between windows the
  stream pushes Ceres-bound ships, so the Ceres side never idles.
- **Drop the growth/departure split for deliveries.** Keep it only for the Earth hardware's
  freight, which flies the paper's own departure.

## Item 5. The steady ledger

| | 4 R_sun, X_C 110 | 9.86 R_sun, X_C 90 |
|---|---|---|
| $/kg at L1 | **1.463** | **1.696** |
| c_C / c_E | $5.69 / $10.74 | $8.07 / $10.56 |
| freight | $4.40 | $5.52 |
| Ceres loop | x4.44 per 1.06 yr | x2.72 per 0.92 yr |

- **The Ceres stream's speed is worth about a cent.** At 4 R_sun the price is $1.46-1.50
  from X_C 60 to 175. From X_C 200 the Sun boost eats the cargo.
- **The Earth stream's lines**, in $/kg at L1:
  - Ceres film: 0.12
  - Sun bowl: 0.08
  - Ceres stream: 0.06
  - disks and Sun charges: 0.045 each
  - avionics: 0.03

## Item 6. Earth push

k_delivery rises only from 5 to 5.6. The push is 259 t of charge, 0.040 kg of sheet per kg
and 1163 t delivered, not 318 / 0.053 / 1182. Your 318 t included the plate and halo
methalox, and your sheets were at the 99 km/s rate.

## Item 7. The Ceres plate

- **At 0.5 Hz over 8 h the plate is 66 t per 1000 t** (spring law, 8 m stroke), not 0.5%.
  One 63 t plate pushes 959 t.
- **Recover it:** the stream pushes it back, and it flies 100 times. Expending it costs
  +$0.29.
- **It can double as the Sun bowl** once it is no heavier than the bowl, at a push of 24 h or
  more, or 1 Hz or more. That is within $0.004 of recovery.

## Item 8. Sensitivities and the $1 line

| step (cumulative) | 4 R_sun | 9.86 R_sun |
|---|---|---|
| headline | 1.463 | 1.696 |
| child's 4 Hz delivery (S37) | 1.131 | 1.351 |
| Ceres film $1 | 1.047 | 1.226 |
| 3000 t ships | 1.011 | 1.200 |
| 2 Hz Ceres push | **0.997** | 1.164 |
| 31% of water's toll returned | 0.995 | 1.154 |
| methalox $0.12 | 0.726 | 0.875 |

**The floor is the delivery lob: $1.006 on the 2 Hz delivery, $0.68 on the child's.**

The largest single risks:

- the lob at $1 lofted: $2.51;
- unlisted dead mass at 15%: $1.97;
- Earth hardware at $100: $1.81;
- Ceres film at $10: $1.77.

## Item 9. Not charged

- **Booster operations: $0.86/kg per $1M a lob.**
- The Ceres industry's capital.
- The seed: one loop's pellets, ~120 t per 1000 t ship.
- The tug's reactor.
- The Earth plates' recovery.
- The Earth plates' idle time between windows.
- The pellets' guidance is charged at the sheets' rate, but 7.5 kg pellets at 0.5 Hz are an
  assumption.

## Asks raised

- **Impact simulation:**
  - water charges in the hole bowl, flown for their own law;
  - ice pellets at 80 deg;
  - S37 (the 4 Hz delivery), now the lever that decides $1.
