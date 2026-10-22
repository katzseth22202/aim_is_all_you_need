# ADR 0083: The solar dive grown at Ceres, and why the delivery lob keeps it above $1

Date: 2026-10-22

Status: Accepted. Answers the parent's S56 (`docs/ceres_isru_dive_asks_for_aim_repo.md`,
copied from `Balloon-Pulse-Propulsion` @ `75729e3`). Reproduce with `make ceres-dive`
(~5 min).

## Context

S53 (ADR 0082) found the lob is the floor on the solar dive's price. The author's idea
removes the growth lob with ISRU at Ceres:

- Ceres water costs $0.02/kg, and graphite, structure, polyethylene and film are made there
  on nuclear power. Only plates, bowls and avionics come from Earth.
- A returning **Ceres stream** of ice pellets pushes ships from Ceres' orbit: prograde dives
  for the ships, retrograde for the disks.
- At the Sun the ships are boosted as in the solar dive. Their remainder returns either to
  Ceres (pellets) or to Earth (Ceres-film sheets that push lofted payload to L1).

The parent's scratch gave $0.99/kg at L1 at 4 R_sun and $1.16 at 9.86, against a $1 target.

## Decision

**1. The Ceres push law** (`ceres_ve`, `best_push`).

- The parent's law 1 reproduces the departure check: 8.07 against the paper's 8.00 km/s. The
  simulation's own departure-hole surface gives 9.67 at that point (11.05 with every bond
  returned), so law 1 is pessimistic there.
- The parent's way of moving water onto the surface (the polyethylene surface times law 1's
  water/polyethylene ratio) is **not used**. Law 1's polyethylene toll reaches zero at
  `k = w^2/168`, which is 26 at 67 km/s. Near there the ratio diverges and gives 0.016 kg of
  pellets per kg.
- The headline instead takes the surface's **bonds-returned** rows (the bowl's capture,
  rebound and mixing) times law 1's frozen toll for water on an ice pellet. The pellet's own
  51 MJ/kg is charged too. Below 89.2 deg the surface is extrapolated in `cos theta`, as the
  simulation does.
- At 80 deg and 15.84 km/s, this needs 15-50% more pellets than law 1 below 150 km/s:

| w | best k | v_e | pellets/kg | water/kg | law 1 pellets |
|---|---|---|---|---|---|
| 95 | 14.8 | 13.8 | 0.145 | 2.15 | 0.112 |
| 150 | 26.6 | 15.8 | 0.065 | 1.73 | 0.048 |
| 200 | 28.0 | 20.8 | 0.041 | 1.14 | 0.031 |
| 300 | 29.8 | 30.4 | 0.023 | 0.68 | 0.018 |

- **Water's 51 MJ/kg is the right toll for frozen atomization.** At 95 km/s and k = 15 the
  blob carries ~280 MJ/kg, well past full atomization. The ionization is returned on
  expansion (ADR 0033's Saha argument). Water's bonds re-form only at 3000-4000 K, late in the
  expansion. The companion's 23-31% returned share moves the price by under $0.02.
- The k = 30 cap binds only above X_C ≈ 200. Lifting it to 60 saves under 0.005 kg/kg.

**2. Ceres-side orbits.** The coplanar circular orbit reproduces every parent figure:

- dive Δv 14.67 / 15.84 km/s prograde and 21.14 / 19.97 retrograde;
- Earth-bound excess X_E 128 / 87 for the paper's 137 / 99 km/s.

Ceres' real eccentricity moves the direct dive's Δv from 17.1 at perihelion to 14.6 at
aphelion, and its landing by ±6 deg about the mean. Both average out to the circular figure.
Tilting the dive 10.6 deg out of Ceres' plane costs 0.04 km/s; the Earth-bound dive's worst
tilt, about 15 deg, costs 0.08-0.11. **Coplanar circular is enough for the ledger.**

**3. Phasing.** The parent's misses reproduce: -115, -123, -102 and -107 deg prograde, and
-28 to -55 deg retrograde. So does its retrograde inbound closure at 9.86 R_sun, X 90 (23.32
km/s, aphelion 3.28 AU, 0.68 yr).

**(a) A single co-orbital depot does not close.** Ships leaving the depot land their stream
`miss` further on, and the ships it pushes there send the stream on again. A depot holds its
phase but the stream keeps moving. **A ring of `N` depots closes** when each depot's stream
lands `j` depots on (`miss = -360 j/N`). Three depots close within 0.22 km/s of the direct
dive at every headline closure (exactly at 4 R_sun, X 150). The depots off Ceres sit a mean
80 deg away. A water steam tug at 200 s feeds them: 0.61 km/s over 20 years, 0.365 kg of water
per kg moved, about $0.002/kg at L1.

**(b) The retrograde inbound closure** works at one depot, but at 23.3 km/s against 14.7 for
the ring's 3-depot dive. Prograde single-depot closures need 22.7-23.8 km/s and 4+ year loops.

**(c) An off-perihelion or turned boost** moves the landing 1-2 deg per 0.1 km/s. That is
fine for trimming the ring, useless for the 107 deg.

**(d)**
- A Mars flyby turns the stream 1.4-1.6 deg at v_inf 29-30 km/s, and its windows recur every
  3.2 years. Rejected.
- Diving from Ceres' aphelion saves 1.2 km/s on that departure, but a steady ring flies from
  every phase.

**Disks** dive head-on from the ship's own depot: 20.1 km/s against the direct mirror's
19.97, launched 5-9 days apart.

**4. Two clocks.**

- The Ceres loop returns in 0.92-1.06 yr. The Earth leg flies 0.85-0.88 yr, but Earth must
  be at the right phase: once per 467-day synodic period per depot.
- Raising the Earth-bound dive's aphelion within +0.5 km/s holds a depot's window open 151-155
  days. Three depots spaced 120 deg give 97-100% coverage.
- **One dive schedule feeds both.** The stream pushes Earth-bound ships whenever some depot
  is in its window and Ceres-bound ships otherwise, so the Ceres side never idles.
- **The growth/departure split is dropped for deliveries** (one push to L1). It is kept only
  for the freight to Ceres, which flies the paper's own Earth leg.

**5. The steady ledger** (`steady_loop`). Book: S53's best case (lob at its $0.49 propellant
floor, snow-foam Earth charges, plates flown 100 times, packages $1) plus the Ceres prices.

- The masses come from the solar dive's own laws: the Sun boost on the arc, the spring-law
  plates at their survival floors, shade, avionics and mounts.
- The ice pellets carry steering packages at the sheets' rate.
- The freight is priced, not assumed: the paper's Earth leg to the Hohmann excess (8.8 km/s),
  then a Ceres-stream capture at 4.9 km/s. It comes to **$4.4-5.5/kg** against the parent's
  $3 placeholder.
- `c_C`, `c_E` and the freight are one linear system, solved directly. Optimized over k_sun,
  arc and k_delivery:

| | 4 R_sun (X_C 110) | 9.86 R_sun (X_C 90) |
|---|---|---|
| delivery_lob | 1.006 | 1.057 |
| delivery_charge | 0.014 | 0.018 |
| delivery_plate | 0.018 | 0.019 |
| earth_stream | 0.426 | 0.603 |
| **$/kg at L1** | **1.463** | **1.696** |
| c_C / c_E ($/kg) | 5.69 / 10.74 | 8.07 / 10.56 |
| Ceres loop growth | x4.44 per 1.06 yr (doubles in 5.9 months) | x2.72 per 0.92 yr (7.6 months) |

- **The parent's $0.99 misses three charges:**
  - **The delivery lob's climb.** S53's $0.49 is the `lob` axis. The 2 Hz, 1 g delivery
    climbs at 2.43 km/s (x1.065 x 1.4875), so the line is $1.006, not $0.62.
  - **The Sun bowl at its survival floor.** It is 23 t per 1000 t at 4 R_sun, not the 0.74%
    the parent took. The solar dive's own is 19 t on 654 t.
  - **The Ceres pusher plate.** It is 66 t per 1000 t at 0.5 Hz over 8 h, not 0.5% (item 7).
- The Ceres stream's speed is worth about a cent, as the parent found. At 4 R_sun it is
  $1.461-1.504 from X_C 60 to 175, best at 110-120. At 9.86 R_sun the best is X_C 60-90.
  From X_C 200 at 4 R_sun (and 150 at 9.86) the Sun boost eats the cargo, and c_C rises
  steeply.

**6. The Earth push on c_E.** At $10.7/kg of stream, k_delivery moves off its floor of 5 only
to 5.6. The delivery is 259 t of charge, 0.040 kg of sheet per kg and 1163 t delivered, not
the parent's 318 / 0.053 / 1182. The parent counted the plate and halo methalox as charge and
priced sheets at the 99 km/s rate.

**7. The Ceres plate.** The spring law's `m s = F tan(theta/4)/(2 theta nu^2)` makes a
0.5 Hz, 8-hour push carry **66 t of plate per 1000 t**. One 63 t-class plate pushes 959 t.

| Hz | hours | plate t / 1000 t | t pushed by 63 t | recover | expend | Sun bowl |
|---|---|---|---|---|---|---|
| 0.5 | 8 | 65.7 | 959 | 1.463 | 1.748 | 1.680 |
| 0.5 | 24 | 22.6 | 2790 | 1.451 | 1.545 | 1.453 |
| 1 | 24 | 6.5 | 9713 | 1.447 | 1.477 | 1.450 |
| 2 | 8 | 5.8 | 10813 | 1.447 | 1.474 | 1.451 |

- **Recovering the plate is cheapest.** The stream pushes it back to its depot, and it flies
  100 times. Expending it costs $0.29 at 0.5 Hz / 8 h.
- **The Ceres plate can double as the Sun bowl** once it is no heavier than the Sun bowl's
  floor: at 24 h or more, or 1 Hz or more. That is within $0.004 of recovery. At 0.5 Hz /
  8 h it is $0.22 dearer.
- A longer or faster push saves only $0.02, because the recovered plate is already cheap.

**8. Sensitivities and the $1 line.** One at a time at 4 / 9.86 R_sun:

- Ceres film $1 / $10: 1.38 / 1.77 and 1.57 / 2.13.
- Earth hardware $100: 1.81 / 2.02.
- Freight $3 / $10: 1.46 / 1.49.
- Unlisted dead mass 5% / 15%: 1.58 / 1.97.
- Misses 10%: 1.50.
- Packages $15: 1.49.
- Earth plates 10 uses: 1.65.
- Lob $0.30 / $0.66 / $1: 1.07 / 1.81 / 2.51.
- Law 1: 1.445. Clamped angle: 1.473.
- Ship 3000 t: 1.43.
- Child's 4 Hz delivery: 1.135 / 1.351.

What must fall, cumulative and re-optimized:

| step | 4 R_sun | 9.86 R_sun |
|---|---|---|
| headline | 1.463 | 1.696 |
| the child's 4 Hz, 3.26 g delivery from the 1.1 km/s lob (S37 unsolved) | 1.131 | 1.351 |
| Ceres film $1/kg | 1.047 | 1.226 |
| 3000 t ships | 1.011 | 1.200 |
| Ceres push at 2 Hz | **0.997** | 1.164 |
| water's toll 31% returned | 0.995 | 1.154 |
| methalox $0.12/kg (lob $0.30) | 0.726 | 0.875 |

- **The floor is the delivery lob.** It is $1.006 at book methalox on the 2 Hz delivery, so
  that delivery alone keeps the price above $1. On the child's delivery the floor is $0.68.
- At 4 R_sun, $1 needs the 4 Hz delivery, Ceres film at $1, big ships and a 2 Hz Ceres push.
- At 9.86 R_sun, $1 needs methalox at the sourced low end as well.

**9. Not charged.**

- Booster operations and hardware: $0.86/kg at L1 per $1M a lob.
- The Ceres industry's capital.
- The seed: one loop's pellets, about 120 t per 1000 t ship, from some first push.
- The tug's reactor.
- The Earth plates' recovery.
- The idle time of the Earth plates between windows.

The pellets' guidance is charged at the sheets' package rate, but 7.5 kg pellets at 0.5 Hz
are an assumption, not a design.

## Consequences

- New module `src/ceres_dive.py`, `make ceres-dive`, `tests/test_ceres_dive.py`.
  `solar_dive_growth` is unchanged.
- Search bounds and starts are in `CERES_BOUNDS`, `SUN_GRID`, `ARC_GRID` and `HEADLINE`.
- **For the parent:**
  - the single-depot closure needs the ring;
  - $0.99 becomes $1.46 / $1.70;
  - $1 is a delivery question (S37), not a Ceres one.
- **Open, for the impact simulation:**
  - water charges in the hole bowl, flown for their own law;
  - ice pellets at 80 deg;
  - the sheets' survival at the 4 Hz delivery (S37).
