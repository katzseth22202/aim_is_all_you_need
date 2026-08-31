# A shallower solar dive costs 2.2x the clock, and the nozzle cap is the expansion not the ignition

Status: accepted

Date: 2026-08-31

## Context

ADR 0019 closes the Jovian solar-dive cycle at the paper's own 4 solar radii and
reports the fastest doubling time in this repository: 83.35 kg returned per
impactor kilogram per 3.276 yr cycle, doubling in 0.513 yr. Every number in it
is a trajectory number. None of them ask whether the node it flies through can
be built.

Four objections say it cannot, and none are orbital mechanics:

1. **The node is at 3.9 MW/m^2**, 2890 times Earth's insolation, with an
   isothermal equilibrium temperature of 2041 K. A magnetic nozzle needs a cold
   coil; keeping one cold there is a refrigeration problem, not a shielding one.
2. **The opposing streams meet at 617 km/s.** The impactors are small and
   exposed, and they still have to survive the approach.
3. **Precision scales with closing speed.** 100 m of along-track miss is 162 us
   of relative timing error at that speed.
4. **The energy per impactor kilogram implies a heavy nozzle**, which the growth
   ledger does not carry.

The proposal was to back the perihelion out to **32 solar radii**, take about
**25 km/s** of boost there instead of 35, and come home at half the speed --
with the Earth-side slug ratio capped at **8.5** on the grounds that the closing
speed at the end of the departure burn might fall to "a bit above 60 km/s" and a
magnetic nozzle with no plasma has nothing to grip.

Three of those four inputs turned out to be right to within a few percent. The
fourth -- the reason for the slug-ratio cap -- turned out to be the wrong
mechanism, and the cap it produced was roughly right for a different reason.

## Decision

Add `src/solar_dive_depth_trade.py` (`make dive-depth`), which closes the cycle
at any perihelion and scores it on four ledgers at once, and adopt
**32 solar radii / 75 km/s climb-out / k = 8.5** as the conservative reference
point against ADR 0019's aggressive one.

The module does three things `src/jovian_solar_dive_cycle.py` does not.

**The dive node is derived, not stated.** `jovian_solar_dive_cycle` charges a
constant `DEFAULT_PERIAPSIS_SURVIVAL = 0.60` at the periapsis collision. Here it
comes from the same **impact-angle impulse law** the Earth departure already
uses -- head-on, at the opposing streams' closing speed, with `beta*w/k` as the
effective exhaust speed and the rocket equation on the boost. This is the check
that makes the whole comparison legitimate, and it passes: at 4 solar radii,
`k_p` = 30 and `eta_jet^2` = 0.60 it derives **0.5977** against the stated 0.60.
The shallow rows are therefore not being scored on a device the deep row was
not.

**The slug ratio has two ceilings and they pull opposite ways.** The **plume
ignition window** caps `k` from above because a fixed collision energy shared
over more slug eventually cannot reach the 84.4 MJ/kg ignition bill. The
**launch ledger** also caps `k` from above, because slug is lifted from Earth
and more slug per impactor buys more growth per impactor while returning less
per pad kilogram. They are far apart, and which one binds changes with depth.

**The opposing stream is placed rather than assumed**, because the **dive-placement
floor** is the one quantity in this trade that gets *worse* as the dive gets
shallower.

Search bounds are recorded in the module -- `DEPTH_GRID`, `RETURN_EXCESS_GRID`,
`RETURN_EXCESS_BRACKET`, `SLUG_RATIO_GRID`, `SLUG_RATIO_SEARCH_BRACKET`,
`PLACEMENT_DEPARTURE_BRACKET` -- per CLAUDE.md's rule after ADR 0007's numbers
became unreproducible.

## Consequences

### Three synodic periods still closes, and the margin barely moves

| | 4 R_sun (ADR 0019) | 32 R_sun |
| --- | ---: | ---: |
| Earth departure excess | 10.5386 km/s | **12.4987 km/s** |
| departure aim from prograde | -12.06 deg | **-44.21 deg** |
| speed needed at 200 km (`v_rf`) | 15.2398 km/s | **16.6556 km/s** |
| Jupiter arrival excess | 12.1718 km/s | 10.9456 km/s |
| bend required / available | 74.73 / 133.53 deg | **83.46 / 137.94 deg** |
| bend margin, unpowered | +58.80 deg | **+54.48 deg** |
| legs (out / dive / climb) | 1.429 yr / 1.819 yr / 10.35 d | 1.570 yr / 1.656 yr / 18.48 d |
| Earth re-intercept miss | 0.0000 deg | 0.0000 deg |

The aim root is unique: scanning `DEPARTURE_AIM_BRACKET` at 0.5 degrees produced
exactly one sign change in the intercept residual, so the reported closure is
not a search artefact. **2S is worse than it was** (deficit +18.10 deg prograde,
+9.86 retrograde, against +6.84 at 4 solar radii), and 4S is still unreachable
against the 3.661 yr transfer clock ceiling. So 3S remains bracketed on both
sides at the shallow node, for the same two reasons.

### The proposal's own numbers were right; the convention behind one of them was not

* **"Perihelion speed around 100 km/s"**: 107.66 km/s arriving, 132.37 leaving.
* **"25 km/s of delta-v"**: 24.71 km/s. Essentially exact.
* **"Effective Isp near 25 km/s, propellant fraction below 2/3"**: derived, not
  assumed, the node gives **23.78 km/s** (Isp 2425 s) and **0.646**. Both land
  just inside the guess, and neither was free to choose -- the same `k_p` = 30
  device that reproduces ADR 0019's 0.60 at the deep node produces these here.
* **"Returning at 75 km/s"**: this one is a convention trap. 75 is the *solar*
  hyperbolic-excess speed of the climb-out, the same quantity the paper's 150
  names. The cycle closes at Earth at **85.07 km/s**, not 75, because Earth's
  29.8 km/s of transverse motion is a larger share of a slow return than a fast
  one (150 -> 157.8 is +5 percent; 75 -> 85.1 is +13 percent). Asking for 75
  km/s *Earth-relative* instead is an 18.25 km/s boost, and it grows faster --
  see below.

### What it costs: 10.5x the growth, 2.1x the doubling time

Scored at `k` = 8.5, `k_p` = 30, `eta_jet^2` = 0.60:

```
DIVE NODE at 32 R_sun
  arrive / boost / leave        107.66 / 24.71 / 132.37 km/s
  opposing closing speed        215.3 km/s     (k_p = 30, beta = 3.313)
  effective exhaust speed       23.78 km/s     (Isp 2425 s)
  propellant fraction           0.646  ->  survival 0.3537

EARTH DEPARTURE NOZZLE k = 8.5
  cant (aim separation)         141.07 deg
  impact angle                  130.5 -> 133.1 deg
  closing speed                  91.78 -> 95.55 km/s
  effective exhaust speed       17.64 km/s     (Isp 1799 s)
  burn                           5.6470 km/s
  MASS FRACTION TO JUPITER       0.7261

GROWTH   round trip 0.2568 = departs 0.7261 x survives 0.3537
  per impactor kg               7.97 kg
  doubling / millionfold        1.0939 yr / 21.80 yr
```

Against ADR 0019's 83.35 and 0.5134 yr. The split is **5.2x from the depth and
2.0x from the slug-ratio cap** -- at 32 solar radii and `k` = 30 the cycle
returns 16.15 kg per impactor kilogram and doubles in 0.8164 yr.

The full depth sweep, at a fixed 75 km/s climb-out so the Earth end barely
moves and the variation is all at the node:

| R_sun | v_peri | boost | v_b | bend | cant | v_e node | survive | departs | kg/imp | dbl yr | 1/15 x | T_eq K |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 308.3 | 9.52 | 89.77 | +59.92 | 84.8 | 68.1 | 0.8695 | 0.8353 | 37.48 | 0.627 | 2.72 | 2041 |
| 6 | 251.5 | 11.58 | 89.30 | +59.34 | 94.3 | 55.5 | 0.8118 | 0.8385 | 35.82 | 0.635 | 2.55 | 1666 |
| 8 | 217.6 | 13.28 | 88.88 | +58.85 | 102.0 | 48.1 | 0.7586 | 0.8360 | 32.86 | 0.650 | 2.38 | 1443 |
| 12 | 177.4 | 16.05 | 88.14 | +57.98 | 114.0 | 39.2 | 0.6638 | 0.8226 | 26.17 | 0.696 | 2.05 | 1178 |
| 16 | 153.3 | 18.30 | 87.47 | +57.21 | 122.6 | 33.9 | 0.5824 | 0.8045 | 20.37 | 0.753 | 1.76 | 1020 |
| 24 | 124.8 | 21.89 | 86.23 | +55.79 | 134.0 | 27.6 | 0.4519 | 0.7648 | 12.49 | 0.899 | 1.30 | 833 |
| **32** | **107.7** | **24.71** | **85.07** | **+54.48** | **141.1** | **23.8** | **0.3537** | **0.7261** | **7.97** | **1.094** | **0.96** | **722** |
| 40 | 96.0 | 27.05 | 83.95 | +53.23 | 145.7 | 21.2 | 0.2791 | 0.6899 | 5.28 | 1.365 | 0.72 | 645 |
| 48 | 87.3 | 29.04 | 82.84 | +52.06 | 148.9 | 19.3 | 0.2217 | 0.6561 | 3.60 | 1.775 | 0.55 | 589 |

Everything is monotone except delivered mass, which peaks at 6 solar radii
because the growing cant drags the impact angle through the impulse law's
optimum on the way past. The effect is under half a point and the node's
survival swamps it.

### What it buys, and the four objections were sound

| | 4 R_sun | 32 R_sun | factor |
| --- | ---: | ---: | ---: |
| solar flux | 3.93 MW/m^2 (2890x Earth) | 61.5 kW/m^2 (45x Earth) | 64 |
| isothermal equilibrium temperature | 2041 K | **722 K** | 2.83 |
| opposing closing speed | 616.6 km/s | 215.3 km/s | 2.87 |
| energy thermalised at the node | 5934 MJ/kg | **724 MJ/kg** | 8.2 |
| timing error for a 100 m miss | 162.2 us | **464.4 us** | 2.87 |

2041 K is above the melting point of every structural metal and this is the
*equilibrium* figure for a sphere, before any concentration; 722 K is an
ordinary hot-radiator problem. The refrigeration argument for backing out is
the strongest of the four, and nothing in the trajectory ledger contradicts it.

### The plasma cap was not the binding one, and the first replacement was wrong too

This is the reversal worth recording. The cap was proposed at `k` = 8.5 because
the closing speed "towards the end of our burn might be only a bit above
60 km/s". **The closing speed does not fall through the burn; it rises**, 91.78
to 95.55 km/s. The impact is at 130.5 to 133.1 degrees -- past square, so the
vehicle's own acceleration *adds* to the impactor's approach rather than
subtracting from it. The 60 km/s figure is the *aligned* geometry, which the
model does produce (74.0 -> 68.4 km/s) if the burn is flown down the stream
axis, but this cycle is not in it.

At the true closing speed the plume ignition window is `k` in
**[0.021, 47.88]**. Even under the pessimistic 60 km/s reading it is
[0.052, 19.27]. **Plasma retention does not limit the slug ratio at any depth
swept.**

What does limit it is the **launch ledger**, and it pulls the other way:

| `k` | Isp (s) | departs | kg per impactor kg | doubling | kg per pad kg | vs 1/15 | vs rescaled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2143 | 0.7644 | 2.30 | 2.733 yr | 0.0676 | 1.01 | 1.48 |
| 4 | 2141 | 0.7642 | 4.59 | 1.491 yr | 0.0676 | 1.01 | 1.48 |
| 6 | 1977 | 0.7474 | 6.28 | 1.236 yr | 0.0661 | 0.99 | 1.45 |
| **8.5** | **1799** | **0.7261** | **7.97** | **1.094 yr** | **0.0642** | **0.96** | **1.41** |
| 12 | 1611 | 0.6995 | 9.88 | 0.991 yr | 0.0619 | 0.93 | 1.36 |
| 30 | 1140 | 0.6034 | 16.15 | 0.816 yr | 0.0534 | 0.80 | 1.17 |
| 47 | 944 | 0.5435 | 19.79 | 0.761 yr | 0.0481 | 0.72 | 1.05 |

Growth per impactor kilogram rises with `k` throughout; returned mass per pad
kilogram peaks at **`k` = 2.74** and falls. Against the committed 1/15 **return
floor** the shallow cycle clears only up to **`k` = 5.25**, and only ever by
1.022x. So the answer to "what caps the slug ratio" is **5.25 from the pad**,
not 8.5 from the plasma and not 47.88 from the ignition window. The proposed cap
was within a factor of 1.6 of the right number for entirely the wrong reason.

**This is the margin the depth actually spends.** At 4 solar radii the committed
floor is cleared across the *whole* ignition window -- 2.10x at its peak and
still 1.76x at `k` = 47.9. At 32 it is marginal everywhere and fails above 5.25.
ADR 0019 argued at length that the committed floor's calibration does not
transfer, since it was set against a ~60 km/s return and that cycle closes at
157.8. That argument still holds and still helps -- restated for an 85 km/s
returned kilogram the floor is 1/21.9 and the whole window clears -- but **the
rescale is worth 1.46x here against 2.80x there**, because the rescale is a
function of the closing speed and backing out is what lowered it. The shallow
cycle spends its launch margin twice: once by returning less mass, and once by
returning it slower.

Time-normalised, the ordering is unchanged for the fourth time: 0.0196 kg per
pad kg per year against 0.0375 for ADR 0019's cycle and 0.0497 for the paper's
own single-impulse resonant dive.

### The ignition window is a condition at t = 0, and the nozzle needs one at t = exit

This is the correction that matters most, and the intermediate answer above is
wrong in a way worth recording, because it is the answer an optimiser gives when
handed only the constraint the repository already had.

`plume_thermal.slug_ratio_window` asks whether the merged blob **ignites** --
whether it reaches 15,000 K at the instant of the merge. A magnetic nozzle does
its work by letting that plume *expand*, which cools it. A blob sitting on the
window's upper root therefore lights and then falls out of conduction the moment
the field takes any work out of it: the nozzle decouples partway down and the
quoted `eta_jet^2` is never reached.

Maximising growth at 32 solar radii subject only to the ignition window and the
launch ledger picks a 45 km/s climb-out and **`k` = 23.75**, and reports 23.94 kg
per impactor kilogram doubling in 0.7151 yr. That result is not merely optimistic.
**It is unphysical**, and ADR 0016 is what makes it so.

ADR 0016 established that the dissociation store *defaults* -- the plume crosses
`Da = 1` at the first station past the nozzle lip with 90-100 percent of it still
held -- and that the toll is charged **inside** `eta_jet^2` rather than as a
separate debit. The consequence was never applied to the growth ledgers. A jet
carrying `eta` of the merge energy leaves `1 - eta` behind, and what is left
behind must still cover a bill that does not shrink:

    eta_jet^2  <=  1 - frozen / eps_th          (`maximum_jet_efficiency()`)

The ignition bill splits **53.47 MJ/kg frozen** (vaporise, atomise, seed-ionise)
against **30.94 MJ/kg translational**. At `k` = 23.75 and a 66.09 km/s closing
speed the merge thermalises only **84.7 MJ/kg**, so the largest efficiency
available is **0.369** -- and it was being scored at 0.60. The impulse law was
drawing 52.9 MJ/kg of directed exhaust out of a blob holding 31.2 MJ/kg of
translational energy.

The operating requirement is stricter than that ceiling and subsumes it: the
plume must still clear the ignition bill *after* the jet has been drawn, so the
field is still gripping when the expansion finishes.

    eps_th * (1 - eta)  >=  ignition bill       (`expansion_floor_energy()`)

At `eta_jet^2` = 0.60 that is **211.0 MJ/kg, 2.50x the ignition bill rather than
1x**. Call it the **expansion floor**. It roughly halves the admissible slug
ratio at every closing speed:

| coldest closing speed | ignition `k_max` | **expansion `k_max`** |
| ---: | ---: | ---: |
| 60.00 km/s | 19.27 | **6.37** |
| 66.09 km/s | 23.83 | **8.23** |
| 91.78 km/s | 47.88 | **17.90** |
| 158.17 km/s | 146.19 | **57.26** |

Below **41.1 km/s** the window vanishes outright -- the merge energy peaks at
`k` = 1 with `w^2/8`, and below that speed the peak itself is under the floor, so
no magnetic nozzle of any slug ratio works there at this efficiency.

**The proposed `k` = 8.5 clears the floor, and by the same margin the deep
reference cycle does.** At the 32 solar-radii cycle's 91.78 km/s the merge holds
396.7 MJ/kg, **1.88x** the expansion floor; ADR 0019's `k` = 30 at 158.17 km/s
holds 390.5, **1.85x**. The cap that was proposed to protect the plasma turns out
to be required by the expansion instead, and lands almost exactly where it needed
to be. Its real ceiling at that operating point is **17.90**, not 47.88 and not
5.25.

### The two cycles at their own constrained optima

Scored by `constrained_growth_optimum()`, which maximises growth over the
climb-out excess and the slug ratio together -- they are coupled, since boosting
less lowers the Earth closing speed and *narrows* the expansion window while
spending less of the node's propellant -- subject to the expansion floor at
`DEFAULT_EXPANSION_MARGIN` = 2.0 and the committed 1/15 **return floor**. The
margin is there because an optimiser sits on whatever boundary it is handed, and
a boundary is not an operating point; at unit margin both rows land exactly on
the floor, which is the same mistake as `k` = 23.75 in a milder form.

| | 4 R_sun | 32 R_sun |
| --- | ---: | ---: |
| climb-out excess | 187.5 km/s | 62.5 km/s |
| periapsis boost | 53.00 km/s | 18.06 km/s |
| Earth closing speed `v_b` | 193.76 km/s | 74.68 km/s |
| departure slug ratio | 43.50 | **5.50** |
| coldest closing speed | 196.15 km/s | 80.73 km/s |
| merge energy | 422.6 MJ/kg (2.00x floor) | 424.3 MJ/kg (2.01x floor) |
| node survival | 0.4591 | 0.4679 |
| mass fraction departing | 0.8105 | 0.7480 |
| round trip | 0.3721 | 0.3500 |
| **kg per impactor kg** | **85.44** | **7.64** |
| **doubling** | **0.5106 yr** | **1.1169 yr** |
| millionfold | 10.18 yr | 22.26 yr |
| launch ledger vs 1/15 | 1.40x | 1.31x |
| limited by | expansion | expansion |

**Both are expansion-limited, both clear the pad, and the trade is 11.2x the
growth for 2.19x the clock.** That is the number this ADR exists to produce: a
64x cooler node, an 8.2x gentler collision and 2.9x looser rendezvous timing cost
a factor of 2.19 in doubling time, not the factor of 10 the per-cycle growth
suggests and not the factor of 1 an unconstrained search suggests.

Two footnotes. ADR 0019's own operating point is near-optimal on its own axis --
83.35 against 85.44 -- so nothing here revises it. And the proposed
32 R_sun / 75 / `k` = 8.5 point returns *more* per impactor kilogram than the
constrained optimum (7.97 against 7.64) because it sits just outside the pad
constraint at 0.96x; the ridge is flat enough that the two are the same cycle for
practical purposes, and the optimum is the one to quote because it is feasible.

### At a shallow node the Oberth boost is a pure cost

The least expected result. The growth-optimal climb-out excess falls with depth
and then falls off the bottom of the bracket:

| R_sun | optimal climb-out | boost | `v_b` | kg per impactor kg |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 121.33 km/s | 23.52 | 130.92 | 46.96 |
| 8 | 81.31 km/s | 15.41 | 94.24 | 33.07 |
| 16 | 48.72 km/s | 8.56 | 66.65 | 23.54 |
| 24 | 30.30 km/s | 4.86 | 53.31 | 19.37 |
| 32 | below the 20 km/s bracket floor | -- | -- | -- |

The mechanism is that the boost is charged against the *node's* exhaust speed
and spent on the *Earth departure's*. Deep, the node runs at 68 km/s and the
boost is nearly free, so the optimum sits high -- above the paper's own 150 at
`k` = 30, slightly below it at `k` = 8.5. Shallow, the node runs at 23.8 km/s
while the departure burn stays near 5.65 km/s whatever the stream arrives at, so
a hotter return buys almost nothing at Earth (delivered mass moves only 0.712 to
0.756 across climb-out excesses from 40 to 120 km/s) and the node's propellant
bill is exponential in the boost. Both the growth ledger and the launch ledger
agree, which they did not at 4 solar radii.

Consequence for the proposal: **asking for 75 km/s Earth-relative rather than 75
of solar excess is strictly better** -- an 18.25 km/s boost, survival 0.464,
10.19 kg per impactor kilogram, doubling in 0.978 yr, and 1.25x the committed
launch floor instead of 0.96x. The conservative direction was under-applied, not
over-applied.

### Backing out makes the opposing stream harder to place, not easier

The one quantity in this trade that moves the wrong way. A flyby rotates the
Jupiter-relative excess but never rescales it, so placing a dive means cancelling
Jupiter's 13.0579 km/s of orbital motion -- and a *shallower* perihelion needs
less of it cancelled for a prograde dive and more of it reversed for a retrograde
one. The two **dive-placement floors** spread apart about the unchanged radial
plunge:

| | prograde | radial | retrograde |
| --- | ---: | ---: | ---: |
| 4 R_sun | 11.9557 | 13.0579 | 14.1602 |
| 32 R_sun | **9.9785** | 13.0579 | **16.1374** |

CONTEXT.md's claim that "both signs are reachable from a single arrival speed in
the 14.16-17 km/s overlap" is a statement about the floors, and at 32 solar radii
the overlap moves to 16.14-17. Neither cycle's 3S closure reaches it: the deep
one arrives with 12.1718 km/s and the shallow one with 10.9456, both above their
prograde floor and below their retrograde one. **The opposing stream cannot ride
the payload's cycle at either depth** -- which was already open in ADR 0019's
"both streams must rendezvous at perihelion", now quantified and wider.

It is not a wall. Projectiles are one-way and expendable, so their trajectory
needs to exist and be timed, not to close on a synodic clock, and a tangential
Earth departure of **11.833 km/s** of excess arrives at Jupiter with the 16.1374
the retrograde placement needs -- *below* the payload's own 12.4987. What the
split costs is a second departure energy and a second schedule, not a second
energy budget. Placing the same dive directly from 1 AU without Jupiter is the
expensive alternative and it moves the wrong way too: 35.48 km/s of excess at 4
solar radii against **44.94** at 32, since 1 AU must be the aphelion of a
retrograde ellipse whose perihelion is further out.

The node also eats more of that stream: 2.15 percent of the vehicle arriving at
perihelion against 1.34 percent, since a larger propellant fraction at the same
`k_p` is more arriving impactor. **That mass is uncharged in the growth ledger at
both depths**, so 7.97 and 83.35 are upper bounds in the same way and by a
slightly worse factor here.

### The ranking is unchanged, for the fifth time

The paper's own single-impulse resonant dive still doubles in 0.3054 yr. ADR 0019
sat 1.7x behind it on doubling time; this cycle sits 3.6x behind. Nothing here
recommends the shallow node on growth. What it says is what the trade actually
costs, so the choice between a 0.513 yr doubling through a 2041 K node and a
1.094 yr doubling through a 722 K one can be made on engineering grounds rather
than by assuming the deep node is free.

## What this does not settle

- **The nozzle mass is still uncharged.** Objection 4 -- that the energy per
  impactor kilogram implies a heavy nozzle -- is the one this ADR does not
  answer. The trade shows the shallow node handles 8.2x less specific energy,
  but no mass model turns that into kilograms, so the case for backing out
  remains qualitative on exactly the axis that motivated it.
- **The impactor heat shield is not modelled.** `src/cruise_thermal.py` solves
  ice sublimation for the cruise, not a 2041 K or 722 K terminal approach.
- **`eta_geom` is still unmeasured** (`sec:jet_efficiency`), so `eta_jet^2` =
  0.60 remains an operating point at both nodes -- but it is no longer a free
  one, and this is the second place ADR 0019's reasoning needs revising. The
  **expansion floor** binds the efficiency as well as the slug ratio: at the
  shallow cycle's 396.7 MJ/kg merge, `eps_th*(1 - eta) >= bill` caps `eta_jet^2`
  at **0.787**, and `eta` = 1 is not a limit the architecture can approach at
  all, since the floor diverges there. Within what is reachable the cycle is far
  more efficiency-sensitive than ADR 0019's was: doubling time moves 1.70 to
  1.09 yr between `eta_jet^2` = 0.4 and 0.6, against 0.543 to 0.513 there,
  because the node's propellant bill is now a large exponent rather than a small
  one. **ADR 0019's efficiency-robustness claim does not survive backing the dive
  out.**
- **Circular and coplanar**, as ADR 0019. The +54.48 deg bend margin should
  absorb real-orbit variation as comfortably as +58.80 did; the widened 2S
  deficit is still inside ADR 0011's noise and still must not be called
  infeasible on this model.
- **Both streams still have to rendezvous at perihelion in time as well as
  space.** Two departure energies is a necessary condition, not a schedule.
