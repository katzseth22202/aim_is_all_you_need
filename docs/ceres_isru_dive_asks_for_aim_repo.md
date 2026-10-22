# The solar dive grown at Ceres: ask S56 (for `aim_is_all_you_need`)

Raised 2026-10-22 against aim `6a8390b` (S53's answer, ADR 0082). **Written to be copied verbatim
into `katzseth22202/aim_is_all_you_need`**, so it repeats context the paper repo already has.
Register entry: S56 in `docs/deferred_to_companion_repos.md`.

Same ground rules as your own handbacks. Neither document is the source of truth. If a number here
looks wrong, say so rather than working around it. Every figure below is a parent scratch estimate
on simple algebraic laws. None of it is a companion result yet.

## The question (Seth, 2026-10-22)

S53 found that the lob is the floor on the solar dive's long-term price. At 4 R_sun the extreme
best case is $2.81/kg at L1. Dropping the lob from $5 to its propellant floor takes $11.50 off, and
every other lever together takes $1.14. Much of the remaining lob is the **growth lob**: every
reinvested launch unit is lofted from Earth. Seth's idea is to remove the growth lob with ISRU at
Ceres. Iapetus is ruled out because the trip is too long (S51, `sec:saturn_cycle_priced`).

Seth's scenario, as he stated it:

1. **Ceres supplies the bulk.** Water comes from the polar ice at **$0.02/kg**, not free as on Earth.
   Graphite for the Sun disks and carbon foam for the heat shields are made at Ceres.
2. **A semi-mature ISRU industry with nuclear power.** Reactors on Ceres boil the water and make the
   graphite, the tank walls that store water, and the carbon-foam heat shield. Don't price reactor
   approval or development. It is assumed.
3. **Only plates and electronics come from Earth.** That means the Sun-dive plates (bowls), the Ceres
   pusher plates and the avionics. The Sun-dive impactors cost more than S53's $10 disks, **about
   $30/kg on average**.
4. **The Ceres plate pushes water with almost no lob.** Ceres' escape speed is 0.51 km/s. Its burn
   can last about 8 hours and still be effectively impulsive on a heliocentric orbit. So a small
   plate at low acceleration pushes a large mass.
5. **Geometry.** Ships dive from Ceres to 9.86 or 4 R_sun. At perihelion they are boosted (Oberth)
   onto a fast return. Returning ships either come back to Ceres or go to Earth. At Ceres the returning
   stream pushes the next ships directly onto prograde dives (the ships) and retrograde dives (the
   disks, the Sun wave). The retrograde wave is smaller but needs the larger burn. The exact
   perihelion is a tuning knob.
6. **Earth plates only push to high orbit.** They are reusable and push the payload into the
   high elliptical orbit. The Earth stream is held at the speed the paper already flies, so that the
   Kapton sheets survive.
7. **The Ceres stream is free in speed.** It is compact ice at about 0.5 Hz, so sheet survival is not
   a constraint. A fast stream suits the large delta-v needed at Ceres.
8. **The target is below $1/kg at L1**, with the mass from Earth being payload, plates and
   electronics.

## What the parent ran (scratch, not a companion change)

Scripts are in the parent's session scratch: `ceres_loop.py`, `ceres_phase.py`, `ceres_ledger.py`,
`ceres_speed.py`, `ceres_split_speed.py`. Everything they do is written out below so it can be
rebuilt in aim.

### 1. Impulse law at the Ceres push

The bowl law, per kilogram of impactor at closing speed `w`, slug ratio `k`, angle `theta` between the
thrust and the stream's direction of travel (0 = overtake):

    I/(m w) = eta_geom * eta_chem(k, w) * sqrt(1 + k) + cos(theta)
    eta_geom = 0.65    (mid of the bowl's 60-73% axial turn, departure-hole runs)
    eta_chem = sqrt(1 - 2 k E_b / w^2)
    E_b = 51 MJ/kg for water (atomization, H2O -> 2H + O, 9.5 eV per molecule)
    E_b = 84 MJ/kg for polyethylene

The charges ride on the ship, so the push is a rocket with exhaust speed `v_e = I/(k m)`:

    R = exp(dv / v_e);  charge per kg final = R - 1;  stream per kg final = (R - 1)/k

`k` is chosen to minimize stream per kilogram final, because water is nearly free and the stream is
not. **Check against the paper's departure:** polyethylene, w = 84 km/s, theta = 130 deg, k = 15 gives
v_e = 8.07 km/s. The paper's departure takes 1015 t to 566 t over 4.67 km/s (4.21 plus the 0.46
steering loss) on 29.5 t of impactor, which is v_e = 8.0 km/s at k = 15. Please replace this law
with the departure-hole simulation's own surface over (k, theta, w) wherever it covers the case.

### 2. Geometry at Ceres (2D, coplanar, Ceres circular at 2.767 AU, v_C = 17.9 km/s)

- Prograde dive delta-v from Ceres' circular orbit, Hohmann-like to perihelion r_p:
  **14.7 km/s to 9.86 R_sun, 15.8 km/s to 4 R_sun.** Retrograde (disks): **21.1 / 20.0 km/s.**
- The returning stream crosses Ceres' orbit almost radially. At return excess X = 90 km/s it closes
  at **95 km/s, about 80 deg from the retrograde thrust** (theta about 80 deg). That is a little
  better than the 89 deg bowl that was simulated.
- Perihelion boost is `sqrt(2GM/r_p + X^2)` minus the arriving perihelion speed. At 4 R_sun and
  X = 90 it is 13.9 km/s. At 9.86 R_sun and X = 90 it is 21.2 km/s.

Results of law 1 at Ceres (prograde dive):

| Depth | X | Ceres w | theta | best k | v_e | stream per kg final | water per kg final |
|---|---|---|---|---|---|---|---|
| 9.86 | 60 | 66.6 | 77 | 6.5 | 19.1 | 0.178 | 1.16 |
| 9.86 | 90 | 94.5 | 81 | 15.0 | 15.9 | 0.101 | 1.52 |
| 4 | 60 | 67.0 | 76 | 6.0 | 20.5 | 0.195 | 1.17 |
| 4 | 90 | 94.8 | 80 | 13.5 | 17.2 | 0.112 | 1.52 |

On the retrograde dive at 4 R_sun and X = 90 the best k is 10.0, with 0.161 kg of stream per kg.
Past k of about 15 at 95 km/s the stream needed rises again, because the charges are carried.

### 3. Phasing: the stream misses Ceres

Two-body conics. Ships depart Ceres at angle 0. They coast in on an ellipse to r_p and take a
tangential boost there to excess X. Then they coast out on the hyperbola to r = 2.767 AU. The miss
is the stream's polar angle at that crossing minus Ceres' angle at that time.

| Case | Direct dive from circular orbit, miss | Best inbound-branch closure |
|---|---|---|
| 9.86, X 60, prograde | -115 deg | does not close (-93 at 23.5 km/s) |
| 9.86, X 60, retrograde | -43 deg | closes at 28.0 km/s, aphelion 6 AU |
| 9.86, X 90, prograde | -123 deg | does not close |
| 9.86, X 90, retrograde | -28 deg | **closes at 23.3 km/s**, aphelion 3.28 AU, 0.68 yr |
| 4, X 60, prograde | -102 deg | does not close |
| 4, X 60, retrograde | -55 deg | -16 deg at 27.2 km/s |
| 4, X 90, prograde | -107 deg | does not close |
| 4, X 90, retrograde | -42 deg | -3 deg at 27.2 km/s |

Changing X moves the miss by only about ten degrees. The parent's way to close it is a
**co-orbital water depot** at the stream's crossing phase. Ceres' gravity is negligible away from
Ceres, so a depot holds a fixed phase in Ceres' orbit. Drifting water 100 deg along the orbit takes
`dv = v_C (2/3) dtheta / (n_C t)` over two burns: **1.52 km/s over 10 yr, 0.76 km/s over 20 yr.** A
nuclear steam tug that loses half its water adds under one cent per kg at L1, because water is $0.02.
The ships then depart from the depot. The ledger below charges the water twice to cover that tug and
charges nothing else for it.

### 4. Ceres ledger (per kg of ship mass after the Ceres push, before the dive)

| Line | Fraction of dive mass | Price |
|---|---|---|
| Ceres-bound stream for the push | 0.112 (4 R_sun, X 90) | c_C per kg, solved |
| Water charges | 1.52 | $0.02 x 2 (depot tug) |
| Avionics, from Earth | 1.0% | $33 ($30 hardware + $3 freight) |
| Sun bowl, from Earth (5 t on 672 t) | 0.74% | $33 |
| Ceres pusher plate, from Earth, expended with the ship | 0.5% | $33 |
| Graphite disks, Ceres-made, retrograde Sun wave | 1.3% at 4 R_sun / X 90 | $30 + 0.161 c_C for their push |
| Ceres-made structure (shade, carrier, mounts, carbon foam) | 1.6% | $10 |
| Sun charges, Ceres polyethylene | 5% at 4 R_sun / X 90 | $1 |
| Cargo that returns as stream | f_c = 0.75 - Sun charges (0.70 at 4 R_sun / X 90) | ice $0.04, or film $3 + $0.02 package |
| Misses | 3% | |

The disk and Sun-charge fractions scale with the boost and inversely with the perihelion closing
speed. They are anchored at 1.3% and 5% at 4 R_sun and X = 90:
`scale = (boost / 13.9) * (617 / (2 v_esc(r_p)))`.

The Earth-hardware freight of $3/kg is a placeholder. Hohmann departure to Ceres is v_inf 8.8 km/s,
inside the paper's 10.24 km/s departure. Arrival is v_inf about 4.9 km/s, caught by the Ceres
stream.

Ceres-bound stream cost solves the fixed point
`c_C * f_c * 0.97 = cost per kg dive (cargo = ice)`. Earth-bound stream cost is then
`c_E = cost per kg dive (cargo = film) / (f_c * 0.97)`. Its Sun boost is set by its own X_E.

### 5. Earth side (per kg at L1)

These are the paper's 4 R_sun push ratios per 1500 t launch unit: 318 t of charge, sheet = charge /
5.1, and 1182 t delivered. The plate is reused 100 times. S53's conventions apply: booster
propellant only, $0.49/kg lofted, and no operations.

| Line | $/kg at L1 |
|---|---|
| Delivery lob, 1500/1182 x $0.49 | 0.622 |
| Snow-foam charges, 318/1182 x $0.06 | 0.016 |
| Earth plate, 63 t / 1182 t / 100 x $30 | 0.016 |
| Earth stream, 0.053 x (137 / w_E) kg sheet x c_E | about 0.34 |
| Growth lob | **0** |

Above 130 km/s the overtake law is derated linearly, to 9% low at 208 km/s.

### 6. Results

**Same return speed for both streams** (film $3 below 140 km/s at Earth, carbon fabric $10 above):

| Depth | X | Earth w | c_E | $/kg at L1 |
|---|---|---|---|---|
| 9.86 | 90 | 101 | 7.12 | 1.16 |
| 9.86 | 100 | 110 | 7.62 | 1.16 |
| 9.86 | 150 | 157 | 19.41 | 1.58 |
| 4 | 90 | 103 | 5.60 | 1.05 |
| 4 | 120 | 130 | 6.11 | 1.00 |
| 4 | 150 | 158 | 14.12 | 1.33 |
| 4 | 200 | 206 | 16.22 | 1.28 |

**Split speeds, Seth's design.** The Earth stream is held at the paper's 137 km/s at 4 R_sun
(X_E = 128) and 99 km/s at 9.86 R_sun (X_E = 87). The Ceres stream X_C is swept:

| Depth | X_C | Ceres w | best k (cap 30) | stream per kg | Ceres-bound boost | c_C | c_E | $/kg at L1 |
|---|---|---|---|---|---|---|---|---|
| 4 | 90 | 95 | 13.5 | 0.112 | 14 | 2.53 | 6.41 | 0.993 |
| 4 | 120 | 124 | 24.5 | 0.070 | 24 | 3.04 | 6.30 | 0.988 |
| 4 | 150 | 153 | 38 (30) | 0.048 | 36 | 3.83 | 6.26 | 0.986 (0.985) |
| 4 | 200 | 202 | 68 (30) | 0.028 | 60 | 5.95 | 6.24 | 0.985 (0.983) |
| 4 | 300 | 301 | 80 (30) | 0.014 | 123 | 17.08 | 6.30 | 0.988 (0.993) |
| 9.86 | 90 | 95 | 15 | 0.101 | 21 | 4.04 | 6.98 | 1.166 |
| 9.86 | 120 | 123 | 26 | 0.063 | 35 | 5.89 | 6.92 | 1.162 |
| 9.86 | 150 | 153 | 41 (30) | 0.042 | 52 | 9.12 | 6.96 | 1.164 |
| 9.86 | 200 | 202 | 73 (30) | 0.025 | 85 | 22.79 | 7.27 | 1.187 |

At 9.86 R_sun with X_C of 250 or more, the fixed point blows up or goes negative. The Sun charges
eat the cargo fraction, and the loop stops closing.

**Sensitivities at 4 R_sun, X = 90, uniform speed** (base $0.95 under the earlier 137 km/s sheet
ratio): film $1 gives 0.84, film $10 gives 1.33, Earth hardware $100 gives 1.13, freight $10 gives
0.96, Ceres structure $30 gives 0.98, cargo fraction 0.6 gives 0.98, retrograde direct closure at
23.3 km/s gives about 0.99, and a lob of $1/kg lofted gives 1.60.

The parent's reading:

- **The lob is now two-thirds of the price.** The delivery lob's $0.62 is the floor while the
  payload is lofted from Earth. Ceres removes the growth lob, and the rest of the loop is about $0.35.
- **A faster Ceres stream is better physics but saves about a cent.** Stream per kg falls fourfold
  from X_C 90 to 200. Each Ceres-bound kg then needs a bigger Sun boost, and the push is only about
  5% of c_E. Film (about $2.10 per kg dive) and Earth-made hardware ($0.74) dominate c_E.
- **Best settings:** X_C 150-200 at 4 R_sun and about 120 at 9.86. The k-at-30 cap costs under
  $0.001, so the answer does not need the law extrapolated past the simulated range.

## S56. Wanted

Put this in code as a new scenario beside the solar dive, for example `ceres_dive.py` with a
`make ceres-dive` target. Reuse the solar-dive ledger's objects and price book where they fit. Use
S53's `MATURE` book (ADR 0082) for every shared line.

1. **The Ceres push law.** Implement the impulse law of section 1 for a water charge in the bowl.
   Use the departure-hole simulation's surface where it covers (k, theta, w), and law 1 outside it.
   Report best k, v_e, stream and water per kg at theta about 80 deg for w from 65 to 300 km/s.
   Reproduce the parent's departure check (8.0 km/s) first. Say whether water's 51 MJ/kg toll is
   the right bond charge, and how much of it is recovered as the gas expands.
2. **Ceres-side orbits.** Compute the prograde and retrograde dive delta-v from Ceres, the
   perihelion boost, and the stream's speed and angle at Ceres for both depths and X from 60 to 300.
   Use Ceres' real orbit (e = 0.0785, i = 10.6 deg) or say why coplanar circular is enough.
3. **Phasing closure.** Reproduce the section 3 misses. Then solve the closure in each of the
   following ways, priced:
   (a) a co-orbital depot at the crossing phase, fed by a water steam tug over 10 or 20 years, with
   the tug's water loss;
   (b) the retrograde inbound-branch departure (23.3 km/s at 9.86 R_sun and X 90);
   (c) a non-tangential or off-perihelion boost, if it rotates the asymptote enough;
   (d) anything better you find, such as a Mars flyby or a dive from Ceres' aphelion.
   Also fit the disks' retrograde dive so that it meets the ship head-on at perihelion, from the same
   departure point or depot.
4. **Two clocks.** The Ceres-bound waves return on the Ceres loop's own clock (about 0.6-1 yr). The
   Earth-bound waves need the Ceres-Earth synodic window (1.28 yr). Say whether one dive schedule
   can feed both, or what idle time it costs. Earth-bound ships need the paper's growth/departure
   split, with the push wave ten days early. Keep it or drop it, and say which.
5. **The steady ledger.** Implement sections 4 and 5 as named lines on the `MATURE` book, with
   these new prices: water $0.02, Ceres disks $30, Ceres structure $10, Ceres polyethylene $1,
   Ceres film for Earth sheets $3 (Kapton-grade, a hypothesis), Earth hardware $30 + freight.
   Replace the parent's $3/kg freight placeholder with a priced Earth-to-Ceres leg: the paper's
   departure to v_inf 8.8 km/s and a Ceres-stream capture at about 4.9 km/s. Solve c_C and c_E as
   fixed points. Report $/kg at L1 at both depths with the two streams' speeds as separate knobs.
   The Earth stream is held at the paper's 137 / 99 km/s at 400 km.
6. **Re-optimize the Earth push on c_E.** Earth stream now costs about $6/kg against $0.55/kg for a
   lofted snow-foam charge (lob plus foam). Re-solve the Earth push's k, and say whether the 318 t
   charge, the 0.053 kg sheet per kg delivered and the 1182 t delivered move.
7. **Plate sizing at Ceres.** Size the Ceres pusher plate for an 8-hour burn (mean about 0.06 g)
   on the spring model. Say how many tonnes of ship one 63 t-class plate pushes, and whether
   expending it with the ship (the parent's 0.5%) or recovering it is cheaper. Seth thinks the Ceres
   plate might double as the Sun bowl. Check whether that works; the parent has not priced it.
8. **Sensitivities and the $1 line.** Run film $1-10, Earth hardware $30-100, freight, the cargo
   fraction, misses, and the lob from its floor up to $1/kg lofted. Give an ordered list of what has
   to fall to get under $1, in the same form as S53. If the delivery lob alone keeps the price above
   some floor, give that floor.
9. **What is left out.** List every line that reaches $0.02/kg at L1 and is not charged. The parent
   knows of the Ceres industry's capital and seed, Earth plate recovery from the high orbit,
   operations, the booster's hardware, and the Ceres stream's guidance on compact ice at 0.5 Hz.

The paper does not wait on this. Nothing goes into the paper until it is answered.
