# ADR 0080: Aerocapture at Saturn, rings cleared

Date: 2026-10-21

Status: Accepted. Extends ADR 0078 and ADR 0079 at the author's request (2026-10-21):
"model aerocapture at Saturn ... it would not really be easy to do without hitting
Saturn's rings but we can model it." It replaces the Saturn-side capture burn of every
recurring ship. The seed keeps its burn. Nuclear electric propulsion was considered in
the same conversation and dropped by the author (see Context).

## Context

In ADRs 0078 and 0079 the Saturn-side outbound leg is all methalox:

- a capture burn at 150 000 km: 2.55 km/s coplanar, 2.81 median on the real arcs;
- the dog-leg's apoapsis raise and plane turn;
- the match and the landing.

At Iapetus that is 5.1 kg of stage per kg landed. The Earth departure then pushes all of
it: the outbound stack is 16.1 kg per kg landed with hydrogen NTP, and 28.9 chemical only.

Two questions:

- Can the capture be flown through Saturn's atmosphere instead?
- Can the pass and the legs after it cross the ring plane only where nothing is?

The author also asked about **nuclear electric propulsion**: methalox to escape, then
the plant's own reactor driving argon thrusters to Saturn, started outside the
magnetosphere. NASA's SR-1 Freedom (about 20 kWe, Brayton, Mars 2028) is the current
pathfinder. A sizing pass found two problems:

- **Power.** A 3-year trip needs about 10 kWe per tonne shipped. At NASA's MTAS design
  point (1.9 MWe at 11-13 kg/kWe), power is 10-20% of ship mass before argon, at
  plant-class prices.
- **Time.** A full low-thrust transfer takes 6-8 years, which lengthens every round.

The author dropped it. No code was written.

## Decision

`src/saturn_aerocapture.py`, `make saturn-aerocapture` (~5 min).

1. **The pass.**
   - A 3-DOF point mass in Saturn's inertial frame, with point-mass gravity.
   - The atmosphere co-rotates with the planet: 10.66 h, AMAT's figure.
   - The 1-bar ellipsoid is 60 268 / 54 364 km. The pass starts and ends at 1000 km
     above it.
   - Lift is steered in the vertical plane only; bank reversals are taken as perfect.
   - The corridor is the pair of vacuum periapses at which full lift up (steep) and full
     lift down (shallow) exit onto the target apoapsis. The nominal pass flies their
     midpoint, with the bank solved to hit the target.
2. **The atmosphere.**
   - Hydrostatic H2/He at 2.2 g/mol, over a labelled temperature profile: 134 K at 1
     bar, 84 K at 100 mbar, about 145 K in the stratosphere, about 400 K above a
     microbar.
   - It reproduces AMAT's 0.198 kg/m^3 at 1 bar.
   - NASA's own study (Putnam et al. 2023, NTRS 20230005179) bridges Yelle and Sayanagi
     instead, and is the anchor: near-equatorial, about 4.5 km/s shed to Titan's orbit,
     about 2.5 g, about 1400 W/cm^2.
3. **Heating and the aeroshell.**
   - Stagnation-point heating uses AMAT 2.4.0's Saturn coefficients: Sutton-Graves
     convective at 0.6356e-4 (SI), and the ESTEC Jupiter-probe radiative fit AMAT
     applies to Saturn.
   - Thermal protection comes from the worst heat load of the steep, shallow and nominal
     passes, on the HEEET fit (Girija et al. 2023, arXiv 2308.07005:
     `5.91 log10(Q kJ/cm^2) + 5` per cent). The legacy carbon-phenolic fit (Laub and
     Venkatapathy, `0.091 Q^0.51575`) is a row.
   - Structure, backshell and mechanisms add 10% of entry mass (assumption). The
     aeroshell is shed after the pass and priced at $1000/kg (assumption).
   - Vehicle: beta 500 kg/m^2, L/D 0.25, 500 t, a 28 m shield with a 14 m nose. All of
     these are assumptions.
4. **The rings.**
   - An equator crossing is **clear** inside 65 000 km (below the D ring, where Cassini's
     Grand Finale flew) or beyond 175 000 km. It is **dusty** in the F-G gap
     (140 220-166 000 km) and **blocked** anywhere else.
   - ADR 0079's rule (apse line in the moon's plane) leaves at most two apse lines.
     Both were blocked in the first window tried.
   - So the apse line is swept round the asymptote's cone (2 deg steps), and each line is
     ring-checked on the arrival hyperbola and the exit ellipse.
5. **The leg after the pass, methalox (`moon_leg`).**
   - The burn at the first apoapsis puts the ship on an ellipse whose plane holds the
     apse line and a point `d` of the moon's orbit. It meets the moon there and matches
     its velocity, so the match is not tangent unless `d` lies opposite the apse line.
     That opposite case is the parent's dog-leg, which a test reproduces.
   - `d` is swept round the moon's orbit (1 deg), in both senses.
   - The first apoapsis is the dog-leg's 20 million km at Iapetus. At Phoebe it is 1.05,
     1.2, 1.5 or 2.0x Phoebe's distance, with a cap on the leg's years. The ledger picks
     Phoebe's pair on the rNPV.
   - A 50 m/s targeting correction and the landing are added.
6. **The ledger.**
   - `Outbound` gains `aeroshell`: the stack is the post-pass stage over (1 - share).
     `Site` gains `aeroshell`, `seed_outbound_dv` and `arrival_years`.
   - `build_cycle` gains `aero`.
   - The seed keeps the capture burn (the first of its kind); "seed aerocaptures too" is a
     row.

## Results

**The pass, median over 13 windows (2035-2047), real arcs:**

| | Iapetus | Phoebe |
|---|---|---|
| ring-clear windows | 13 of 13 | 13 of 13 |
| atmosphere-relative entry | 30 km/s (prograde; retrograde windows 46) | 29.5 |
| drag delta-v | 1.86 km/s | 1.85 |
| peak heat / load | 172 W/cm^2 / 38 kJ/cm^2 | 160 / 36 |
| peak deceleration | 1.4 g | 1.4 g |
| aeroshell share of entry mass | 24.3% (worst 26.6%) | 24.2% (worst 26.8%) |
| methalox leg after the pass | 1.68 km/s (worst 1.98) | 1.71 |
| pass to moon | 1.16 yr (dog-leg 1.17) | 0.5 yr chosen (base 0.27) |
| corridor | 0.17 deg, 33 km of vacuum periapsis | 0.17 deg |

- **The rings do not forbid it.** Every window has a clear geometry.
  - The inbound crossing is at 61 000-65 000 km, in the Grand Finale gap.
  - The exit crossing is beyond the G ring, out to millions of km.
  - The periapsis sits within about 15 deg of latitude of the equator.
- **Prograde wins on heat, but not always on the stack.** Iapetus's first three windows
  and six of Phoebe's take a retrograde pass at 46 km/s and 88-99 kJ/cm^2 (26.6-26.8%),
  because its leg is cheaper.
- **Iapetus keeps the dog-leg's time.** Phoebe's pass must aim above its orbit, so the
  ledger trades leg time against delta-v. It chooses a 13.6 million km apoapsis with the
  leg held under 0.5 years, which adds 0.23 years to the round.

**The ledger (Ledger A, k 4 compact, rNPV 10% / 4% loan; methalox on the electric
plant):**

| Saturn side | planes | H2 NTP: burn -> aerocapture | chemical only: burn -> aerocapture |
|---|---|---|---|
| water NTR, Iapetus | coplanar | $309 -> **$172** ($131 -> $100) | $532 -> **$223** ($165 -> $108) |
| water NTR, Iapetus | real | $475 -> **$198** ($188 -> $106) | $1,173 -> **$299** ($270 -> $123) |
| methalox, Iapetus | real | $603 -> $367 | $947 -> $384 |
| methalox, Phoebe | coplanar | $308 -> $278 | $338 -> $256 |
| methalox, Phoebe | real | $357 -> $308 | $426 -> **$296** |
| solar dive 9.86 / 4 R_sun | | $143 / $86 | (no nuclear) |

- **Aerocapture halves the outbound stack and roughly doubles growth.**
  - Water NTR, coplanar: 16.1 -> 8.7 kg per kg landed, G 5.95 -> 11.0.
  - Chemical only: 28.9 -> 15.6, G 3.31 -> 6.13.
- **It reorders the Saturn options.**
  - The water NTR at Iapetus becomes the best Saturn cell with H2 NTP ($198 real, $172
    coplanar). Chemical only, it ties methalox at Phoebe on the real planes ($299 against
    $296) and wins coplanar ($223 against $256).
  - Methalox gains less, because its plant cycle already landed less Earth mass per kg of
    PuffSat.
  - The no-nuclear-near-Earth penalty on the water NTR falls from 2.5x to 1.5x (real
    planes).
- **It still loses to the solar dive.** The best Saturn cell, $172 coplanar or $198 on
  the real planes, is above $143 at 9.86 R_sun and $86 at 4 R_sun. Steady cost ($79-85)
  ties.
  - The round (9.5 years) is unchanged, and lofting (lob and plug) is unchanged.
  - Reuse (ADR 0079 addendum) and aerocapture attack different lines and should stack.
    This is not run here.

**Sensitivities (real planes, rNPV; water NTR Iapetus and methalox Phoebe):**

| row | corridor | H2 NTP: water / Phoebe | chemical: water / Phoebe |
|---|---|---|---|
| base | 0.17 deg (33 km) | $198 / $308 | $299 / $296 |
| aeroshell $5000/kg | | $291 / $417 | $419 / $411 |
| carbon-phenolic fit (30.9%) | | $225 / $324 | $353 / $320 |
| structure 15% (29.3%) | | $218 / $321 | $339 / $315 |
| seed aerocaptures too | | $198 / $269 | $299 / $257 |
| L/D 0.4 | 0.27 deg (53 km) | $198 / $308 | $299 / $296 |
| beta 150 (69 W/cm^2) | 0.18 deg | $190 / $303 | $283 / $288 |
| beta 1500 (409 W/cm^2) | 0.16 deg | $206 / $313 | $315 / $304 |

The water NTR's seed keeps its capture burn in every row.

- **The aeroshell's price is the largest lever.** At $5000/kg the gain is about halved.
- **Heating is not the limit.** The pass sheds only about 1.9 km/s, about 400 km above 1
  bar. Peak heat is 70-410 W/cm^2 across the vehicle rows, far below Putnam's 1400 for a
  pass to Titan's orbit.
- **The corridor is the open gate.**
  - At L/D 0.25 it is 33 km of vacuum periapsis; at 0.4 it is 53 km.
  - A factor-two error in density moves the needed periapsis by about 35 km (one scale
    height times ln 2).
  - Approach navigation plus atmospheric knowledge must hold periapsis to roughly
    +/-15 km at L/D 0.25.
  - The seed keeping its burn, and measuring the atmosphere on the way, is the natural
    mitigation. This ADR does not run guidance or Monte Carlo dispersions.

## Consequences

- **Recommendation to the parent.** If the Saturn cycle stays in the paper, its
  recurring ships should aerocapture. It is the largest single improvement found for
  the Saturn cycle:
  - with H2 NTP, $475 -> $198 on the real planes;
  - chemical only, $1,173 -> $299;
  - water NTR at Iapetus becomes the preferred Saturn cell with H2 NTP, and ties
    methalox at Phoebe chemical only.
  It still does not beat the solar dive on the gated rNPV.
- **Labelled assumptions, none sourced here:**
  - the temperature profile and 2.2 g/mol;
  - the vehicle (beta, L/D, 500 t, nose share);
  - 10% structure;
  - $1000/kg aeroshell;
  - the 50 m/s targeting correction;
  - the ring zones' edges (65 000 km below the D ring);
  - HEEET's fit applied to an aerocapture shield;
  - in-plane lift only, no J2, the atmospheric arc taken on the conics for the ring
    check.
- **Not done here:**
  - guidance and dispersions (corridor sufficiency);
  - the apse line's drift through the pass;
  - phasing each apse line to the moon;
  - stacking aerocapture with ADR 0079's reuse rows;
  - the author's queued next cycle: overtake PuffSats to the eccentric orbit, head-on
    PuffSats about 20 days later onto the Saturn arc, aerocapture for all but the seed.
