# Three synodic periods close the Jovian solar-dive cycle; two do not

Status: accepted

Date: 2026-08-31

## Context

The paper's no-ISRU growth loop (`sec:no_isru_rocket`) sends a payload from Earth
straight to a 4 solar-radii perihelion, boosts it there, and brings it home across
1 AU at about 150 km/s. Crossing 1 AU is not reaching Earth, and `sec:earth_reintercept`
prices the three fixes: a two-impulse phasing loop that needs a second PuffSat boost
node far from Earth, a single-impulse resonant dive that folds the phasing into one
~37.5 km/s heliocentric Earth boost (39.11 km/s at the 200 km interception altitude),
and an unmodelled gravity-assist option.

This ADR takes up that third option in its strongest form. Rather than bending the
*return*, let Jupiter place the *dive*: depart Earth for Jupiter, let an unpowered
Jovian flyby drop perihelion to 4 solar radii, take the Oberth boost there, and cross
1 AU where Earth is waiting. The Earth-side dive injection then disappears entirely —
the collision at Earth only has to buy a Jupiter transfer, not a solar dive. The price
is the clock: the loop has to be timed to an integer multiple of the Earth-Jupiter
synodic period so the next cycle sees the same relative geometry, which is the same
resonance discipline ADR 0011 imposed on the Jupiter-only return.

`sec:jupiter_gravity_initial` already names Jupiter as the way to place the *initial*
low-periapsis orbit, citing the Parker Solar Probe's own original mission design. What
was not known is whether the same assist can be made to *repeat* — to close a cycle
rather than place one payload.

Two questions had to be settled before any of it could be scored:

1. Does the geometry close at all — in particular, does the boosted climb-out from
   4 solar radii come out on the side of the Sun where Earth is, or does reaching Earth
   demand a path through the Sun?
2. Which synodic multiple?

## Decision

Add `src/jovian_solar_dive_cycle.py` (`make jovian-dive`), a circular, coplanar,
float-valued model of the three legs built on `src/conic_kernel.py`, and adopt the
**three-synodic unpowered** closure as the reference cycle for this architecture.

The model solves both closure conditions simultaneously. The cycle must take exactly
`N` synodic periods, and the 1 AU crossing must land where Earth actually is. Two knobs
meet them — the Earth-departure hyperbolic-excess speed and the free aim angle
(CONTEXT.md, "Free-aim departure") — so the solution is discrete and the Jovian bend it
demands is an **output**, not something the search may choose. Whether that demand is
payable at the repository's 4,000 km perijove-altitude floor is the feasibility test.

Search bounds are recorded in the module as `DEPARTURE_EXCESS_BRACKET` (8-40 km/s) and
`DEPARTURE_AIM_BRACKET` (-60 to +85 degrees), per CLAUDE.md's rule after ADR 0007's
numbers became unreproducible.

## Consequences

### The geometry closes; the "through the Sun" worry does not bite

The boosted climb-out from 4 solar radii sweeps 130.4 degrees of heliocentric longitude
before it re-crosses 1 AU — the paper's own whip-around figure — and the three legs
together sweep 415 to 443 degrees. Earth's advance over two synodic periods is
786.3 degrees, or 66.3 degrees modulo a turn. Those are commensurate, and the solve
drives the intercept residual to zero at both 2S and 3S. **The exit direction is never
the problem.** Nothing in this family asks for a path through the Sun.

### Placing the dive from Jupiter has a hard floor, and it is a necessary condition

An unpowered flyby rotates the Jupiter-relative excess velocity but never rescales it,
so reaching a 4 solar-radii perihelion means cancelling nearly all of Jupiter's
13.0579 km/s of orbital motion. The minimum arrival excess speed that can do it:

| placement | minimum arrival `v_inf` |
| --- | ---: |
| prograde dive | 11.9557 km/s |
| radial plunge | 13.0579 km/s |
| retrograde dive | 14.1602 km/s |

Below these the dive does not exist at any perijove. This is the exact analogue of the
13.058 km/s floor on the unpowered retrograde return (CONTEXT.md, flagged ambiguities):
a necessary condition, not a cost. The two dive floors sit symmetrically about the
plunge, differing only in the sign of the same small residual tangential speed.

Note the consequence for the paper's prograde/retrograde split at periapsis: **both
opposing streams are placeable from a single Jupiter arrival speed**, anywhere in the
14.16-17 km/s overlap. The split does not need two different departure energies.

### Three synodic periods is not merely the better choice — it is the only one

| cycle | period | departure `v_inf` | arrival `v_inf` | bend required / available | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 1S | 1.0920 yr | 27.461 km/s | 41.548 km/s | 160.45 / 59.07 deg | short by 101.38 deg |
| 2S | 2.1841 yr | 12.679 km/s | 17.754 km/s | 121.52 / 114.68 deg | short by **6.84 deg** |
| 3S | 3.2761 yr | 10.539 km/s | 12.172 km/s | 74.73 / 133.53 deg | **margin +58.80 deg** |
| 4S+ | 4.3682 yr | — | — | — | longer than any transfer can fly |

The 4S row is a reachability result, not a search failure. The longest cycle a
zero-revolution direct transfer can fly is **3.661 yr**, approached at the departure
excess below which the transfer's aphelion no longer reaches Jupiter's orbit. It
converges to that value as the aim bracket is widened (3.6422 yr at a -60 degree floor,
3.6571 at -75, 3.6608 at -85 and -89), and 4S needs 4.3682 yr. Multi-revolution arcs
are unexplored and are the only route to a longer clock.

So 3S is bracketed on both sides: 2S cannot bend far enough, 4S cannot fly slowly enough.

### The three-synodic reference cycle

```
Earth departure excess (v_inf)     10.539 km/s   (C3 = 111.1 km^2/s^2)
  speed needed at 200 km altitude  15.240 km/s   <- the collision's push target v_rf
departure aim from prograde        -12.06 deg
Jupiter arrival excess             12.172 km/s
Jovian bend required / available   74.73 / 133.53 deg  (margin +58.80, unpowered)
legs  Earth -> Jupiter              1.429 yr
      Jupiter -> perihelion         1.819 yr
      perihelion -> Earth          10.35 d
Earth re-intercept miss             0.0000 deg
periapsis boost at 4 R_sun         35.04 km/s
collision speed v_b at Earth      157.82 km/s
Earth-push payload mass ratio      15.76
```

The departure is *modest*: 10.539 km/s of excess is barely above the 8.793 km/s a
Hohmann transfer to Jupiter needs, and the 15.240 km/s the collision must deliver at
200 km replaces the 39.11 km/s of the paper's single-impulse resonant dive. That is the
whole prize, and it is large: at the same `v_b` the Earth-side payload mass ratio rises
from 5.62 to 15.76.

The 58.80-degree bend margin is the other headline. It means the 3S closure is nowhere
near the perijove floor, so it should survive the real-orbit variation that ADR 0011
found so punishing for the Jupiter-only 2S clock (only 45 of 91 windows cleared the
floor there, the worst by 20,515 km).

### No perijove burn closes the two-synodic cycle

This is the load-bearing negative result. A perijove burn is the obvious second knob —
it is exactly what breaks the `v_b` lottery on the retrograde return — but sweeping the
outgoing excess from 12 to 32 km/s, holding the clock and the Earth intercept at every
point, leaves a deficit that never reaches zero:

| outgoing `v_inf` | arrival `v_inf` | departure `v_inf` | perijove burn | bend deficit |
| ---: | ---: | ---: | ---: | ---: |
| 14.0 km/s | 28.48 | 18.78 | 4,953 m/s | +7.59 deg |
| 15.5 km/s | 22.19 | 14.71 | 2,068 m/s | +6.15 deg |
| **16.0 km/s** | — | — | **1,487 m/s** | **+6.14 deg** (minimum) |
| 17.0 km/s | 18.91 | 13.12 | 565 m/s | +6.44 deg |
| 20.0 km/s | 15.34 | 12.00 | 1,359 m/s | +8.48 deg |
| 26.0 km/s | 12.11 | 11.57 | 4,315 m/s | +13.58 deg |

It is a genuine interior minimum, not a trend running off the edge of the sweep, and the
mechanism is that **the clock is what binds**. Braking lengthens the dive, so holding the
cycle time forces a hotter outbound leg and therefore a hotter Jovian arrival, which
raises the bend demand right back. Accelerating shortens the dive but inflates the
outgoing eccentricity, which eats the outgoing half-angle. Both directions lose, and
1,487 m/s of perijove burn buys 0.70 degrees of the 6.84 needed.

Lowering the perijove floor does not rescue it either: 4,000 km of altitude gives
114.68 degrees of bend, 1,000 km gives 115.82, and 200 km gives 116.13 — the deficit only
falls from 6.84 to 5.39 degrees. Nor does a deeper dive: the deficit runs +3.53 degrees
at 2 R_sun, +5.45 at 3, +6.84 at 4, +8.79 at 6, +11.12 at 9.86, so even abandoning the
shield design of `sec:four_radii_thermal` for a 2 R_sun pass leaves it short.

### Two synodic periods costs about 2 km/s of carried propellant, wherever it is spent

Only a maneuver *off* the flyby closes 2S:

| correction | cost | where |
| --- | ---: | --- |
| SOI-seam rotation after the unpowered bend (ADR 0011's convention) | 2,119 m/s | Jupiter SOI |
| minimum deep-space maneuver | 2,268 m/s | 4.391 AU |

The DSM result is the one worth reading twice, because the textbook intuition is that a
mid-course maneuver leverages cheaply into the arrival direction. It does not here. The
maneuver gets steadily *cheaper* the later it fires — 7,766 m/s at 1.48 AU, 3,353 at
3.04, 2,268 at 4.39 — because it must also hold the 2S clock, and an early burn perturbs
the whole leg. The trend only reaches the seam's price by moving inside Jupiter's
0.322 AU sphere of influence, at which point it is a Jupiter-relative burn wearing a
DSM's name. **So ~2 km/s is the correction scale, not a placement to be optimised away.**

That is roughly 1.5 km/s more than ADR 0011's worst *real-phase* 2S correction for the
Jupiter-only return, and at 380 s of methalox vacuum Isp it costs about 42 percent of the
arriving mass per pass. Against that, 2S buys a 33 percent shorter cycle and a slightly
worse Earth push (mass ratio 14.22 against 3S's 15.76, because the hotter departure
raises `v_rf` from 15.24 to 16.79 km/s). **We take 3S and spend nothing.**

### What this ADR does not settle

- **The Earth-side push axis is unresolved and it is the largest open risk.** The
  returning stream arrives 2.3 degrees off radially outward (heliocentric `v_t` 6.39,
  `v_r` 155.67 km/s at 1 AU), so its push axis sits at 98.55 degrees from Earth's
  prograde direction. The 3S departure needs -12.06 degrees. That is a **110.6 degree
  cant**, where ADR 0012's impulse law gives `beta = sqrt(k + 1 - sin^2 theta) + cos theta`
  = 2.70 against the along-axis 4.16 at `k = 9`. The 15.76 mass ratio above is the
  *uncanted* figure and is therefore an upper bound. The alternative is the closed
  cycle's existing machinery — push into a sub-escape parking orbit, re-aim at apoapsis,
  and depart on methalox — which trades the cant for a rocket-equation charge. Which is
  cheaper is not decided here.
- **The same cant question applies to what this replaces**, and it cuts the other way.
  The paper's single-impulse resonant dive needs a boost at 129.6 degrees against the
  same 98.55 degree axis — only 31 degrees of cant, nearly free on the same law. A push
  *exactly* along the axis cannot make a solar dive at all (it would take 161 km/s and
  produce an escape, not a dive), so the paper's cycle depends on canting too; it just
  depends on it much less. Any growth-rate comparison between the two architectures is
  therefore not settled until both cants are charged.
- **Growth rate is not scored here.** On a first pass the paper's 0.89 yr resonant dive
  still beats this cycle on doubling time, because a 2.5-3.7x longer clock outweighs a
  2.6x better Earth push. That comparison depends on the per-cycle survival fraction at
  the 4 R_sun collision node, which we could derive from `eq:external_reaction_mass`
  (about 0.71 at `eta_jet` = 0.8) but which **contradicts the paper's own "doubling factor
  at two"** for the two-impulse loop. That discrepancy needs resolving before either
  architecture is ranked.
- **Circular and coplanar.** Jupiter's real orbit runs 4.95 to 5.46 AU and is inclined
  1.3 degrees to the ecliptic. ADR 0011 showed that real-orbit variation is what actually
  decides perijove feasibility, and it is far larger than the margins argued over here.
  The 58.80-degree 3S margin should absorb it comfortably; the 6.84-degree 2S deficit is
  well inside the noise, so **2S must not be called infeasible on this model alone** —
  only unpayable at a price this model can see. A real-ephemeris audit on the pattern of
  `src/real_orbit_resonance.py` is the next step if 2S is ever wanted.
- **Both streams must rendezvous at perihelion.** This model flies one trajectory. The
  paper's collision needs the prograde payload and the opposing projectile stream to be
  at 4 solar radii at the same place *and the same time*. From Earth that was automatic
  by symmetry; from Jupiter it is two more constraints on the projectile stream's own
  departure, and they are not modelled.
- **The dive perihelion is virtual for the projectile stream.** A radial plunger needs no
  perihelion at all, only to be at 4 solar radii at the right moment, which floors at
  Jupiter's own 13.0579 km/s and is cheaper to place than the retrograde dive's 14.1602.
  Unexplored.

## Addendum: the departure nozzle, and the fastest doubling time found so far

Date: 2026-08-31

The body above left the Earth-side push as the largest open risk, quoting an
uncanted payload mass ratio of 15.76 and noting the 110.6 degree cant would
reduce it by an unknown amount. That question is now closed, and the answer
reverses its sign: **the cant costs almost nothing, and the cycle grows far
faster than anything previously recorded in this repository.** Two corrections
and one modelling change get there.

Scored by `cycle_growth_ledger()` / `paper_resonant_dive_ledger()`, printed by
`make jovian-dive`. The impulse law itself stays in its one home,
`circular_resonance_impulse.impulse_per_impactor_kg()`, which now takes an
optional `jet_energy_efficiency`.

### Correction 1: 110.6 degrees was the wrong angle, and the right one is friendlier

110.6 degrees is the **aim separation**, a diagnostic between excess vectors at
the patched-conic boundary. CONTEXT.md already warns against feeding it to the
impulse law, and the body of this ADR did exactly that. The law wants the angle
at the 200 km burn point relative to the *moving* vehicle:

- The **departure-hyperbola mirror** is free. At `v_inf` = 10.54 km/s the
  departure hyperbola has `e` = 2.833, so its periapsis velocity sits
  ±20.67 degrees off the outgoing excess and the sign is ours. Taking the sign
  that rotates the thrust axis *toward* the stream puts it at +8.61 degrees.
- The impactor's own Earth hyperbola barely bends: at `v_inf` = 157.42 km/s,
  `e` = 410 and the turn is 0.28 degrees. It arrives on a straight line.
- Subtracting the vehicle's own velocity leaves **93.9 to 95.5 degrees** across
  the burn, at a closing speed of 158.2 to 158.5 km/s.

So the stream arrives essentially perpendicular, only ~5 degrees past square.

### Correction 2: at k = 30 the un-steerable part is a sixth of the impulse

The impulse law splits into a steerable and an un-steerable piece:

    beta(theta, k, eta) = sqrt(eta*(1+k) - sin^2 theta) + cos theta
                          \_______ thermal, aimed ______/   \__ bulk __/

The `cos theta` term is the impactor's own momentum -- the merged blob's centre
of mass drifts along the incoming direction at `w/(1+k)`, and no field turns it.
Its magnitude is **1 regardless of `k`**, while the steerable term grows as
`sqrt(k)`. So dilution drowns a fixed-size wrong-way push in a growing right-way
one:

| `k` | beta along axis | beta at 110.6 deg | kept |
| ---: | ---: | ---: | ---: |
| 3 | 3.000 | 1.415 | 47% |
| 9 | 4.162 | 2.669 | 64% |
| 30 | 6.568 | 5.137 | 78% |
| 100 | 11.050 | 9.654 | 87% |

The efficiency multiplies only the exhaust term. Momentum conservation on the
arriving impactor does not care how good the nozzle is, so a 60 percent *energy*
efficient jet scales exhaust *speed* by `sqrt(0.6)` and leaves `cos theta` alone.

k = 30 is comfortably inside the **plume ignition window** at this cycle's
158 km/s closing speed: `eps_th = w^2 k / 2(1+k)^2` gives 387 MJ/kg against the
84.4 MJ/kg ignition bill, a 4.6x margin, with the upper root near k ~ 150. That
headroom is a gift of the 150 km/s return -- at the Jupiter-only cycle's 75 km/s,
k = 30 sits at 87.8 MJ/kg, right on the edge.

### The 3S departure, k = 30, eta_jet^2 = 0.60

```
impact angle (vehicle frame)     93.9 -> 95.5 deg
closing speed                   158.2 -> 158.5 km/s
beta, canted                      4.114     thermal +4.196, bulk -0.082
effective exhaust speed          21.71 km/s   (Isp 2214 s)
burn                              4.2895 km/s  (parking orbit 10.9503 -> 15.2398)

MASS FRACTION SENT TO JUPITER     0.823
impactors consumed                0.59% of the vehicle -- 1 kg flies 169 kg
```

Cancelling the retro impactor costs **0.6 points** of delivered mass: at 94.7
degrees the cosine has almost nothing left in it. The whole cant is worth **under
5 points** against the same burn flown straight down the axis. The reason is
blunt: `dv/v_e` = 0.198. When Isp is 2214 s and the burn is 4.29 km/s, angle
penalties land in the third decimal.

With 40 percent of the mass at 4 solar radii spent as propellant across both
streams -- conservative, since `eq:external_reaction_mass` at the same efficiency
gives about 31 percent -- the round trip returns **49.4 percent** of departing
mass.

### Growth, with Earth-launched mass treated as free

Counting only impactor kilograms as scarce, one impactor kilogram buys `k`
kilograms of spent slug, which flies `k*f/(1-f)` kilograms of vehicle through the
departure, of which the solar collision keeps `s`:

| cycle | `v_inf` | theta (deg) | w (km/s) | Isp (s) | departs | round trip | per impactor kg | cycle | e-fold/yr | doubling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| this 3S Jovian solar dive | 10.54 | 94-95 | 158-159 | 2214 | 0.823 | 0.494 | **83.7** | 3.276 yr | 1.351 | **0.513 yr** |
| paper single-impulse resonant dive | 37.53 | 31-38 | 148-125 | 2367 | 0.297 | 0.178 | 7.6 | 0.893 yr | **2.274** | **0.305 yr** |

**Both are the fastest doubling times this repository has recorded**, by a wide
margin -- against 3.33-4.04 yr for the Jupiter-only direct flyby (ADR 0008/0009),
1.37-1.51 yr for the unpowered assist chain (ADR 0003), and 1.45-1.74 yr for the
two-wave split (ADR 0013). A millionfold takes 10.3 yr on the 3S cycle and 6.1 yr
on the paper's, against the paper's own published 17 yr.

They are **not** comparable to those earlier numbers, and the reason is the next
section. The paper's row is derived from `single_impulse_resonant_dive()`, not
quoted, so both rows sit on one device and one set of constants.

### The ranking did not flip, and the assumption is doing the work

This cycle wins every per-pass number -- 2.8x the departure survival, 11x the
impactor economy -- and still loses on rate, because 3.7x the clock beats 11x the
payload once the logarithm is taken. Modelling the nozzle properly raised both
architectures together and left the ordering intact.

**Free Earth-launched mass is the load-bearing assumption, and it is a strong
one.** The 83.7x growth is 83.7x *in impactors*; the slug scales one-for-one, at
30 kg lofted per impactor kilogram. Three cycles is a 5.9e5 launch-rate
multiplier. The constraint has moved to the pad, not vanished. CONTEXT.md's
**launch ledger** -- which binds in 56 of 64 cells wherever it is charged -- is
exactly the accounting this deliberately switches off, and no result here
survives turning it back on unchanged.

Two smaller caveats. `eta_geom` remains unmeasured (`sec:jet_efficiency` says
nothing in either repository bounds it), so 0.60 is a stated operating point, not
a result. And the retrograde projectile stream is assumed to pay the same
departure fraction as the prograde payload, which is approximate: its dive needs
a hotter Jovian arrival (14.16 against 11.96 km/s), so its departure differs.

### The stage is not exotic, and k = 30 is not a large propellant load

Recorded here because it is easy to misread and was misread once already. **The
slug ratio is kilograms of slug per kilogram of arriving *impactor*, not per
kilogram of vehicle.** Impactors are well under one percent of the vehicle, so
`k` = 30 does not mean a thirty-to-one propellant fraction. What the stage
actually spends, per kilogram it places on the Jupiter trajectory:

| `eta_jet^2` | Isp (s) | `v_e` (km/s) | slug/kg delivered | impactor/kg delivered | departs | doubling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 1774 | 17.39 | 0.275 | 0.0092 | 0.784 | 0.543 yr |
| 0.50 | 2006 | 19.67 | 0.240 | 0.0080 | 0.806 | 0.526 yr |
| 0.60 | 2214 | 21.71 | 0.215 | 0.0072 | 0.823 | 0.513 yr |
| **0.70** | **2405** | **23.58** | **0.196** | **0.0066** | **0.836** | **0.503 yr** |
| 0.80 | 2582 | 25.32 | 0.182 | 0.0061 | 0.846 | 0.494 yr |
| 1.00 | 2904 | 28.48 | 0.160 | 0.0053 | 0.862 | 0.481 yr |

At a 70 percent efficient nozzle: **about 20 kg of slug and 0.7 kg of impactors
per 100 kg delivered.** A methalox stage flying the same 4.29 km/s burn would
spend **2.11 kg per kilogram delivered** against 0.196 kg of slug -- the nozzle is
an order of magnitude leaner, and it carries no engine. `k` = 30 is not the
extravagance it looks like when the units are read correctly.

**The cycle also barely leans on nozzle efficiency.** Doubling time moves only
0.543 to 0.481 yr across the entire range from `eta_jet^2` = 0.4 to the loss-free
ceiling -- twelve percent for a two-and-a-half-fold change in the input. Two
reasons: `beta` grows as `sqrt(eta*(1+k))` once the bulk term is small, so Isp
rises only as the square root of efficiency; and `dv/v_e` is 0.18, so the rocket
equation is nearly linear and forgives a poor exhaust speed. That matters beyond
this cycle, because `eta_geom` is the largest unmeasured quantity in either
repository (`sec:jet_efficiency`) and it decided the device choice outright in
ADR 0014/0015. **Here it is not load-bearing**, which makes this the most
efficiency-robust result in the repository rather than the most exposed.

### The launch ledger does not rescue this cycle, and the argument that said it might was mine

Charging the launched mass was flagged above as the top open task, on the
expectation that it would demolish the 83.7x growth. It was worked, and it did
something more interesting: **it first appeared to reverse the verdict, and then
the reversal turned out to be an artefact of a mis-transferred calibration.**
Recorded in full because the intermediate step is exactly the kind of result that
gets quoted out of context. Scored by `launch_ledger_verdict()`.

`two_leg_nozzle_sweep.RETURN_FLOOR` demands 1/15 kg returned per kilogram off the
pad (2/3 of liftoff is ground propellant, a quarter of the remainder is launcher
dry mass, so 0.25 kg reaches the intercept point). Read three ways:

| cycle | returned/pad kg | vs 1/15 | vs rescaled 1/42 | per pad kg per yr | vs methalox |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3S Jovian solar dive | 0.1254 | **1.88x** | 5.27x | 0.0383 | 50x |
| paper single-impulse resonant dive | 0.0482 | **0.72x** | 2.03x | **0.0540** | 11,602x |

**Reading 1, as committed.** The 3S cycle clears the floor with 88 percent margin
and the paper's own dive fails it. Taken at face value this is decisive, because a
cycle that cannot pay for its launched mass cannot be run in steady state on
freshly launched payload however fast it compounds.

**Reading 2, and why reading 1 does not stand.** The floor's own calibration
sentence sets it against reusable rockets and says explicitly that the returned
mass is "at ~60 km/s Earth-relative rather than sitting in LEO". Both cycles here
return at **157.8 km/s**. A floor is a statement about what a returned kilogram is
*worth*, so it has to be restated when the kilogram does different work: scored
pushing to a common target, a 157.8 km/s stream is worth **2.80x** a 60 km/s one.
The floor restates as **1/42**, and **both cycles clear it**. The paper's failure
was an artefact of applying a threshold calibrated for a different return.

The rescale deliberately uses the *closing speed alone*, via
`return_value_ratio()`, not each cycle's own leverage. Leverage -- kilograms
placed on the transfer per kilogram of stream consumed, 152.7 here against 14.2
for the paper's dive -- already shows up in `returned/pad kg`, since a cycle that
spends impactors well returns more. Rescaling by it too would credit the same
efficiency twice and manufacture the answer.

**Reading 3, the defect neither rescale fixes.** The floor is a *per-pass*
quantity compared against a per-pass threshold, with no clock in it -- and the
clock is precisely the axis that separates these two cycles. Divide by cycle time
and the ordering flips back: 0.0383 against 0.0540 kg per pad kilogram per year,
the paper's dive ahead by 1.41x, in agreement with the growth rate and the
doubling time. **The un-normalised ledger is the odd reading out, and it is the
only one that favoured the 3S cycle.**

### Consequence

The launch-ledger caveat raised earlier in this ADR is withdrawn in both
directions. It does not demolish the 83.7x growth -- charged honestly, the 3S
cycle returns 0.125 kg per kilogram off the pad and clears every version of the
floor. Nor does it rescue the 3S cycle against the paper's, because the version
of the floor that favoured it is the version whose calibration does not apply,
and time-normalising it reproduces the growth-rate ordering.

**So the ranking is unchanged for the third time.** The paper's single-impulse
resonant dive doubles in 0.289 yr against this cycle's 0.503, and no accounting
tried so far reverses it. What survives is narrower and worth keeping: this cycle
uses a returned kilogram **10.7x more efficiently** (152.7 against 14.2 kg placed
per kilogram of stream), because a 4.29 km/s departure wastes far fewer impactors
than a 28.16 km/s one, and it beats an all-chemical stage flying the same
departure from the same ballistic lob by **50x**. Both are real; neither buys back
3.7x the clock.

The remaining unexamined route to a shorter clock is multi-revolution transfer
arcs, which the **transfer clock ceiling** rules out only for zero-revolution
solutions.
