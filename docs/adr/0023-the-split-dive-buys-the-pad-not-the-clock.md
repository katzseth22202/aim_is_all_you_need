# The split dive buys the pad, not the clock, and its far node is fed by leftovers

Status: accepted

Date: 2026-09-01

## What the paper should take

This ADR grew in four passes and the reasoning below records that. For the paper,
the claims in order, each with the function that produces it. Everything is
circular-coplanar, `k` = 30, `eta_jet^2` = 0.60, node survival 0.60 unless said
otherwise, and every figure is under test in `tests/test_bielliptic_dive_split.py`.

**On the Delta-v, which is the least interesting part.**

1. Injecting a 1 AU payload to a 4 solar-radii perihelion is a **54:1 radius
   ratio**, so the bi-elliptic route always wins: 24.09 km/s direct, 16.94
   through a 3 AU aphelion, floor `(sqrt 2 - 1) * v_earth` = 12.34
   (`bielliptic_injection_cost()`).
2. **The paper's own single-impulse resonant dive is already a raise-then-drop.**
   Aphelion 1.9259 AU; the 37.53 km/s boost is 28.80 radially outward plus 24.07
   retrograde. What is new here is *splitting* the impulse across two nodes, not
   the outbound leg, and the paper should not describe it otherwise.
3. **The split costs clock, and the reason is geometric.** The paper's ellipse
   crosses 1 AU at true anomaly 169.09 deg — eleven degrees short of aphelion —
   so its raise leg is 0.385 yr. A small prograde push puts 1 AU at the outbound
   ellipse's *perihelion*, making the raise leg a half orbit: 0.823 yr to a
   comparable aphelion.

**On growth, where the split loses and one variant does not.**

4. Fully phased through 3 AU: **2.2789 yr cycle, 20.6x per pass, doubling 0.5223
   yr, 0.875 kg launched slug per delivered kg** — against the paper's 0.8925 yr,
   7.61x, **0.3048 yr**, 2.365 kg/kg. Impulse enters growth logarithmically and
   the clock linearly, so 2.7x less slug is 1.3x in e-foldings against 2.55x more
   clock. Same verdict as ADR 0008/0019/0021, by the same mechanism.
5. **The partial split strictly dominates the paper's dive** and is the one
   unqualified result here: outbound perihelion 0.4918 AU on the existing
   resonance, doubling **0.2969 yr against 0.3048** and **1.536 kg/kg against
   2.365** (`partial_split_optimum()`). Better on both axes; it costs nothing but
   splitting a burn already being taken. Its far node is *not* beam-fed — that
   needs a dedicated delivery.
6. The *pure* bi-elliptic end is 6 percent **worse** on doubling than the paper's
   dive. The family punishes "a cheaper injection must be better".

**On phasing, which is the structural result.**

7. **The boosted climb-out is radial** — 2.34 deg off at 1 AU, 0.73 at 3.3 — so a
   beam's 1 AU crossing and its 3 AU crossing are **one ray**, 21.07 days and 1.54
   deg apart. The far node eats what Earth did not intercept (69.5 percent of a
   wave). No second launch, no second aim.
8. Three conditions — Earth re-intercept, outer-node co-location, rational beam
   reuse `j/m` — on three knobs, so **the solutions are discrete**, exactly as the
   Jovian **synodic closure** is. The first two alone never intersect (gap +109.45
   to +121.86 deg at zero extra revolutions).
9. The third knob is an **off-aphelion injection**: out there the vehicle is slow
   and Earth is not, so one degree of burn point buys 3.4 deg of phasing, and every
   closure lands 9.8-11.3 deg past aphelion.
10. Headline pattern, 5/8 reuse: perihelion 0.9193 AU, aphelion 2.9400 AU,
    injection 11.296 deg past aphelion, **2.2789 yr, 8 interleaved chains, a beam
    at Earth every 0.2849 yr** split 30.5 percent caught / 69.5 flying on.

**On depth, which is what the split is actually for.**

11. Backing the dive out weakens the payload's boost *and* the beam together
    (37.53 to 28.43 km/s against 157 to 97), so doubling degrades only 0.305 to
    0.358 yr. **The shallow dive is not unreachable on impulse.**
12. **The Earth departure cools itself.** It aims 31 deg off the arriving stream,
    so closing speed falls through the burn: 148.3 to 125.0 km/s at 4 solar radii,
    95.4 to **76.3** at 23. Against ADR 0022's 79.56 km/s floor, **the direct
    single-impulse departure goes cold at 19.80 solar radii** — 27.40 at margin
    1.25, 40.72 at 1.0 (`direct_departure_conduction_depth()`). Always quote the
    margin.
13. The split's Earth node (3.93 km/s) has **no crossing in 4 to 48 solar radii**,
    and its far node cannot cool itself at all: perpendicular to a radial beam,
    closing speed moves 153.50 to 153.17 across a 10.26 km/s burn.
14. **Window, stated narrowly.** ADR 0022's pad ledger admits (4, 22.93] solar
    radii; the direct departure conducts over (4, 19.80]. They overlap. What the
    split opens is the last **3.1 solar radii** — which is the end ADR 0022
    recommends starting at.

**On the opposing stream, which is the half that was missing.**

15. **The dive node needs two arrivals and the depth dial moves them opposite
    ways.** The payload's injection gets cheaper (24.09 to 14.62 km/s from 4 to 32
    solar radii); the **opposing stream**'s retrograde placement gets dearer
    (35.48 to 44.94), because a shallower perihelion keeps more angular momentum
    and reversing its sign costs more. They cross; past ~8 solar radii the
    opposing stream is the dominant Earth-side cost.
16. It does **not** have the conduction problem — it aims retrograde against a
    radial stream, so its closing speed holds or rises (99.5 to 107.3 km/s at 32
    solar radii). Its problem is cost. Do not conflate the two failures.
17. **The split saves more on it, and the saving grows where the payload's
    shrinks**: 1.70x at 4 solar radii to **1.84x at 32**, against the payload
    leg's 1.42x to 1.08x. At 32 solar radii the split places the opposing stream
    for 24.37 km/s — less than the paper's payload dive costs at 4.

**On the node's own geometry.**

18. `dive_node()` hardcodes 180 degrees, so **ADR 0020/0021/0022's whole depth
    trade silently assumes a head-on node**, and therefore the expensive
    retrograde placement. That assumption was load-bearing and unlabelled.
19. A straight-down plunger arrives at **135 degrees, not 90** — payload and
    plunger reach perihelion within a percent of the same speed — and the closing
    speed falls by exactly 1/root 2 at every depth.
20. **`beta` rises as `w` falls**, because the `cos theta` debit is the impactor's
    own momentum arriving backwards and 135 degrees pays only 0.707 of it. The net
    is a pure **slug-ratio** question: Isp kept is **0.960x at `k` = 1, 0.864 at 3,
    0.802 at 8.5, 0.757 at 30**. Both "it costs root two" and "it costs nothing"
    are wrong.
21. **The collision heat halves at every `k`** (it goes as `w**2`), worth about a
    factor of two in depth: a plunger at 4 solar radii collides as gently as a
    head-on node at **7.86** (`plunger_equivalent_depth()`). Collision term only —
    the node's solar flux still goes as `1/r**2`.
22. The node's own exhaust speed **peaks near `k` = 3** (112.0 km/s) and is falling
    by `k` = 30 (67.6), so the slug ratio at which the plunger looks worst is
    already past the node's Isp optimum.

**Caveats the paper must carry.**

- The model is **circular and coplanar**. The co-location condition is a longitude
  match and has had no real-ephemeris audit (the ADR 0011 treatment).
- The depth table **holds node survival at 0.60 rather than deriving it**, which
  flatters every shallow row — ADR 0020's **derived periapsis survival** shows a
  gentler node is a less efficient one.
- The conduction floor is **imported from ADR 0022's overtaking leg**, not
  re-derived for a departure plume. It carries no slug-ratio dependence, so the
  transfer is defensible, but it is a transfer.
- The **perihelion burn is held at 35.98 km/s across the depth dial**, which is
  the 4 solar-radii tuning; ADR 0020/0022 use the climb-out excess as the free
  knob instead, and that convention could move every depth row.
- The **pad-charged launch ledger is not applied** in its `0.25 * chain *
  survival` form; only slug per delivered kilogram is reported.
- The opposing stream is priced in **placement and node geometry, not in a cycle
  ledger**, and the plunger has not been run through a full cycle.
- The reachable beam-reuse band at fixed depth and burn is roughly **0.61 to
  0.63**, which is why 1/2 and 2/3 return no root. Depth and perihelion burn are
  unexplored knobs that would move it.

## Context

The paper's no-ISRU growth loop pays for its solar dive at Earth. The
**single-impulse resonant dive** (`sec:earth_reintercept`) folds the phasing and
the dive into one 37.53 km/s heliocentric boost, 28.10 km/s of it charged from
the 20-day parking orbit. That is the largest single number in the cycle, and
ADR 0019 built the whole Jovian solar-dive cycle to retire it — successfully in
impactor economy (152.7 kg placed per kilogram of stream against 14.2) and
unsuccessfully on the clock (0.513 yr doubling against 0.305).

This ADR asks the other obvious question. Getting from 1 AU to a 4 solar-radii
perihelion is a **54:1 radius ratio**, far past the ~15.6:1 where a bi-elliptic
transfer always beats a direct one. So: raise aphelion first with a small
prograde push, coast out, and pay the dive injection at the far end where the
vehicle is barely moving. The Delta-v ledger is not close:

| aphelion | raise | drop | total | vs 24.09 direct |
| --- | ---: | ---: | ---: | ---: |
| 1.5 AU | 2.843 | 17.945 | 20.788 | 0.863 |
| 3 AU | 6.694 | 10.250 | 16.944 | **0.703** |
| 10 AU | 10.377 | 3.442 | 13.819 | 0.574 |
| infinity | 12.337 | 0 | 12.337 | 0.512 |

Charged from the parking orbit the Earth side collapses further still: a 1x3 AU
departure needs 6.694 km/s of excess, which is 12.88 km/s at 200 km against the
parking orbit's 10.9503 — a **1.93 km/s** burn where the paper's dive spends
28.10.

Two things had to be established before that could be believed.

**First, the paper's dive is already a raise-then-drop.** Its closing aphelion is
1.9259 AU and its boost decomposes as 28.80 km/s radially outward plus 24.07
km/s retrograde. "Push straight out so the overtake geometry works" is what it
already does — which is why its impact angle is a favourable 31-38 degrees
against the **radial-outward push axis**. What is actually new here is
*splitting* the impulse across two nodes, not the outbound leg.

**Second, the split costs clock, and the reason is where 1 AU sits on the
outbound ellipse.** The paper's ellipse crosses 1 AU at true anomaly 169.09
degrees — eleven degrees short of aphelion — so its raise leg costs 0.385 yr out
of a 0.8925 yr loop. A small prograde push puts 1 AU at the outbound ellipse's
*perihelion*, so the raise leg becomes a half orbit: 0.823 yr to reach a
comparable aphelion, 2.1x the coast, on a loop that had 0.89 yr in it
altogether. Same destination, and the coast has eaten the cycle.

## Decision

Model the split as a one-parameter family and score it on ADR 0019's own
currency, then phase it. `src/bielliptic_dive_split.py`, `make split-dive`.

The family is parametrised by the **outbound perihelion** `q`. At `q` = 4 solar
radii the outbound ellipse *is* the dive ellipse, the outer burn is zero, and the
loop reduces exactly to `single_impulse_resonant_dive()` — same 1.9259 AU
closing aphelion, same 0.8925 yr clock. That degeneracy is the test that makes
every other row comparable to the paper's, and it is pinned
(`test_the_split_reduces_to_the_papers_dive_when_nothing_is_split`).

- `split_dive_geometry()` — one loop, leg by leg, with both closure residuals.
- `dive_injection_impulse()` — the cheapest single impulse onto a given
  perihelion, searched over post-burn tangential speed with both radial signs.
  At an apsis it collapses to the textbook tangential burn, which is pinned.
- `reintercept_closing_aphelion()` — the **Earth re-intercept** closure alone.
- `partial_split_optimum()` — the growth-rate optimum along that curve.
- `two_node_closure()` — all three conditions at once, for a named beam-reuse
  fraction.
- `split_dive_ledger()` — growth, doubling and launched slug per delivered
  kilogram. The Earth node is `jovian_solar_dive_cycle.departure_nozzle_ledger()`
  unchanged, so the rows compare to ADR 0019's line for line.

## Consequences

### The Delta-v saving is real and does not buy growth

Scored at `k` = 30, `eta_jet^2` = 0.60, node survival 0.60, on the **impactor-
scarce accounting**:

| cycle | clock | growth/pass | doubling | slug per delivered kg |
| --- | ---: | ---: | ---: | ---: |
| paper single-impulse resonant dive | 0.8925 yr | 7.61x | **0.3048 yr** | 2.3654 |
| 3S Jovian solar dive (ADR 0019) | 3.276 yr | 83.7x | 0.513 yr | **0.215** |
| split via 3 AU, fully phased (5/8) | 2.2789 yr | 20.6x | 0.5223 yr | 0.8746 |

The arithmetic behind the verdict is the same shape as ADR 0008's and ADR 0019's:
**impulse enters growth logarithmically and the clock enters linearly.** 2.7x
less slug is 1.3x in e-foldings; 2.55x more clock is 2.55x. The ranking is
unchanged for the fourth time, and by the same mechanism.

Time-normalising the pad side does not rescue it either — returned kilograms per
launched-slug kilogram per year run 0.288 for the paper's dive, 0.301 for the
phased split and 0.852 for the 3S Jovian cycle. A 4.5 percent edge, not a
reversal. (Both rows are scored at the 5/8 closure's own 158.50 km/s stream, so
the paper's slug bill reads 2.333 there rather than the 2.365 its own row gives
at 157.4 km/s.)

### The partial split is free, and is the result worth keeping

Stopping short is not the same as going all the way. Along the re-intercept
closure curve the doubling time is nearly flat while the slug bill falls
monotonically:

| `q` (AU) | `r_a` (AU) | clock | dv Earth | dv node | doubling | slug/kg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0186 (paper) | 1.9259 | 0.8925 | 28.101 | 0 | 0.3048 | 2.3654 |
| 0.3 | 1.9725 | 0.9993 | 15.851 | 7.998 | 0.2983 | 1.7646 |
| **0.4918** | **1.9649** | **1.0543** | **10.9** | **10.6** | **0.2969** | **1.5358** |
| 0.7 | 1.9424 | 1.1221 | 6.633 | 12.612 | 0.2983 | 1.3269 |
| 0.99 | 1.8870 | 1.3159 | 0.989 | 14.958 | 0.3235 | 1.0733 |

At `q` = 0.4918 AU the cycle doubles in **0.2969 yr against the paper's 0.3048
and spends 1.536 kg of launched slug per delivered kilogram against 2.365**.
Better on both axes at once — **the only strictly dominating result in this
repository's solar-dive work**, and it costs nothing but splitting a burn that
was already being taken.

Note what it is *not*: the pure bi-elliptic end of the family, `q` = 0.99 AU,
is 6 percent *worse* on doubling than the paper's dive. The intuition that
"cheaper injection must be better" picks the wrong end of its own family.

### The far node can eat what Earth did not catch

The boosted climb-out leaves 4 solar radii with angular momentum only
`r_p * v_p`, so it is 2.34 degrees off radial at 1 AU and 0.73 degrees off at
3.3 AU. Its 1 AU crossing and its 3 AU crossing are therefore **the same ray**:
21.1 days apart and 1.54 degrees apart in longitude. The outer node needs no
launch and no separate aim — it eats the part of a wave that Earth did not
intercept. At the 5/8 closure that is 69.5 percent of the wave.

That is not free in geometry. Three conditions must hold together:

- **Earth re-intercept** — the return crosses 1 AU where Earth is.
- **Outer-node co-location** — `360*(t_coast - t_beam) = lam_coast + lam_beam`
  (mod 360): the outer burn happens on a ray a beam is flying down.
- **Rational beam reuse** — the offset between a beam's Earth crossing and the
  outer burn it feeds is `j/m` of a cycle. Chains are independent and the model
  is epoch-invariant, so the offsets can be chosen freely; but they must close
  into a group under adding that offset, and **`m` is the number of interleaved
  payload chains in flight**.

The first two alone do not intersect. Along the whole re-intercept curve the
co-location gap runs **+109.45 to +121.86 degrees** at zero extra revolutions
and **-55.63 to -81.46** at one, never crossing zero
(`test_the_co_location_gap_never_closes_on_the_curve`). Two conditions, two
knobs, no solution.

The third knob is **taking the injection a few degrees past aphelion**. Out
there the vehicle is slow and Earth is not, so one degree of burn-point shift
moves the phasing by about 3.4 — a 110-degree gap closes in eleven degrees of
true anomaly. With that knob the system is three conditions on three knobs and
the solution set is **discrete**, indexed by the revolution count and the reuse
fraction, exactly as the Jovian cycle's **synodic closure** is discrete:

| reuse | chains | `q` (AU) | `r_a` (AU) | past aphelion | cycle | dv node | doubling | slug/kg | beam cadence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3/5 | 5 | 0.5948 | 3.0387 | 9.830 deg | 2.1340 yr | 8.130 | 0.5696 yr | 1.3408 | 0.4268 yr |
| **5/8** | **8** | **0.9193** | **2.9400** | **11.296 deg** | **2.2789 yr** | **10.256** | **0.5223 yr** | **0.8746** | **0.2849 yr** |
| 8/13 | 13 | 0.8157 | 2.9733 | 11.024 deg | 2.2211 yr | 9.638 | 0.5357 yr | 1.0166 | 0.1709 yr |

The 5/8 cycle in full: depart at 10.09 km/s of excess aimed 57.2 degrees from
prograde (3.926 km/s from the parking orbit), coast 1.482 yr sweeping 151.2
degrees, burn 10.26 km/s at 2.879 AU, fall 0.769 yr to 4 solar radii, take the
Oberth boost, climb out 0.028 yr at 151.1 km/s and arrive where Earth is. Earth
advances 100.41 degrees; the vehicle sweeps 100.41 degrees.

**A repeatable pattern therefore exists, and its cost is chain count, not
propellant.** Eight vehicles in flight, a beam reaching Earth every 0.285 yr,
each beam split 30.5 percent caught at Earth and 69.5 percent flying on 21.1
days to serve the outer node of the chain five cadences ahead.

## Addendum: the depth dial, and why this is the architecture that can fly it

The three results above were all scored at 4 solar radii, and at 4 solar radii
the split is a curiosity — it loses on the clock and wins on the pad. Seth's
objection was that this is the wrong place to score it: the point of the split is
that it **relaxes what the Earth push has to do**, and the cycle we would fly
first is not the 4 solar-radii one. ADR 0022 already put the shallow end at
**23 solar radii**, for thermal survivability.

The objection is right, and the mechanism turns out to be sharper than the one it
was stated as.

### It is not that the shallow return cannot deliver the boost

The first guess — that a slower return beam cannot push a payload to the paper's
~39 km/s — does not survive contact. Backing the dive out weakens the beam *and*
the boost together, because a shallower perihelion is a smaller drop to buy:

| depth | resonant-dive boost | stream at Earth | direct burn from parking | delivered |
| ---: | ---: | ---: | ---: | ---: |
| 4 R☉ | 37.53 km/s | 157.4 km/s | 28.10 | 0.297 |
| 16 R☉ | 32.66 | 114.4 | 23.46 | 0.235 |
| 32 R☉ | 28.43 | 96.6 | 19.48 | 0.235 |

Doubling time degrades from 0.305 yr to 0.358 at its worst (23 R☉) across the
whole dial. That is a tax, not a wall (`test_backing_the_dive_out_weakens_the_stream_and_the_boost_together`).

That table is about the **payload's** injection, and on its own it is a
misleading half of the picture — corrected below under "the other leg". The
payload is not the only thing that has to reach the perihelion.

### It is that the direct departure cools itself below the conduction floor

What does *not* scale down with the depth is ADR 0022's **overtaking-leg
conduction floor** — the closing speed below which the merged blob no longer
expands enough to stay conducting, so the nozzle has nothing to grip. At
`eta_jet^2` = 0.60 and the 1.5 expansion margin that is **79.56 km/s**, fixed.

And the Earth departure is the one burn in this architecture that **cools
itself**. The paper's dive departs 31 degrees off the arriving stream — nearly
*along* it — so the vehicle accelerates away from what is feeding it and the
closing speed falls right through the burn: 148.3 to 125.0 km/s at 4 solar radii,
95.4 to 76.3 at 23. The coldest instant is the end of the burn, and a bigger burn
against a slower beam puts it through the floor.

| depth | direct burn | direct closing speed | conducts | split burn | split closing | conducts | far node | conducts |
| ---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: | :---: |
| 4 R☉ | 28.10 | 148.3 → 125.0 | yes | 3.93 | 156.1 | yes | 153.1 | yes |
| 16 R☉ | 23.46 | 105.0 → 84.5 | yes | 3.93 | 114.0 | yes | 110.7 | yes |
| **23 R☉** | 21.57 | 95.4 → **76.3** | **COLD** | 3.93 | 104.6 | yes | 101.9 | yes |
| 32 R☉ | 19.48 | 87.0 → **69.5** | **COLD** | 3.93 | 96.2 | yes | 94.6 | yes |

**The direct single-impulse dive's departure goes cold at 19.80 solar radii**
(`direct_departure_conduction_depth()`, bisected on the constraint rather than
sampled near it — ADR 0022's lesson). The split's Earth node has no crossing
anywhere in 4 to 48 solar radii, because its burn is 3.93 km/s rather than 21.6
and so barely cools the stream at all.

The crossing moves with the expansion margin and must always be quoted with it:
**19.80 R☉ at margin 1.5, 27.40 at 1.25, 40.72 at 1.0.**

### The far node cannot cool itself, and that is the 90-degree cant paying rent

The far node's burn is the *big* one — 10.26 km/s against the Earth node's 3.93 —
so the obvious worry is that it goes cold instead. It does not, and the reason is
the geometry this ADR already flagged as a cost. The beam arrives within a degree
of radial and the injection thrusts tangentially, so the impact angle is 86.3 to
90.1 degrees: the burn is **perpendicular to what feeds it**. Closing speed
across the whole 10.26 km/s burn moves 153.50 to 153.17 km/s — three tenths of a
percent. A burn cannot run away from a stream it is not travelling along.

So the 86-90 degree cant costs the far node about a fifth of its `beta` and buys
it immunity to the floor that kills the direct departure. **The architecture's
largest liability at one node is its enabling condition at the other** — the same
shape as the **radial-outward push axis** being both the Earth node's cant and
the reason **outer-node co-location** exists at all.

### The other leg, and it runs the other way

The claim above — that backing the dive out shrinks the boost and the beam
together — is true of the payload's injection and **false of the architecture**,
because the dive node needs two arrivals, not one. The **opposing stream** has to
meet the payload at that perihelion, and its placement behaves the opposite way:
a shallower dive keeps more of the orbit's angular momentum, so *reversing* the
sign of that momentum costs more, not less.

| depth | payload injection | opposing stream, retrograde | radial plunge | node closing speed |
| ---: | ---: | ---: | ---: | ---: |
| 4 R☉ | 24.09 km/s | 35.48 km/s | 29.78 | 616.6 km/s |
| 16 R☉ | 18.70 | 40.87 | 29.78 | 306.7 |
| 23 R☉ | 16.69 | 42.88 | 29.78 | 255.0 |
| 32 R☉ | **14.62** | **44.94** | 29.78 | 215.3 |

The two legs cross. At 4 solar radii they are comparable; by 32 the opposing
stream is three times the payload's leg and is the **dominant Earth-side cost**
(`test_the_two_legs_move_in_opposite_directions_under_the_depth_dial`). The
near-radial plunge is a third option — flat at Earth's own orbital speed,
independent of depth — but it arrives *across* the payload's path rather than
into it, so the node's closing speed falls by roughly root two and the **derived
periapsis survival** with it. That trade is not priced here.

**This leg does not have the conduction problem.** It aims retrograde against a
radially outward stream, so it never runs *along* what feeds it: its closing
speed holds or rises through the burn (99.5 → 107.3 km/s at 32 R☉) and it clears
the floor at every depth. Its problem is cost, and the two must not be conflated.

**And the split helps it more, with the saving growing where the payload's
shrinks**, because the raise costs the same either way and only the flip term
differs — while the flip is exactly what the depth dial inflates:

| depth | opposing direct | split raise + flip | saving | payload saving |
| ---: | ---: | ---: | ---: | ---: |
| 4 R☉ | 35.48 | 6.60 + 14.32 = 20.93 | 1.70x | 1.42x |
| 23 R☉ | 42.88 | 6.60 + 16.98 = 23.58 | 1.82x | 1.16x |
| 32 R☉ | 44.94 | 6.60 + 17.77 = 24.37 | **1.84x** | 1.08x |

The flip conducts too, and for the same reason the payload's far node does: it is
perpendicular to a radial beam, so reversing 17.77 km/s of tangential motion moves
the closing speed by less than a km/s.

So **the split's case at a shallow dive rests more on this leg than on the one
scored first.** At 32 solar radii the split places the opposing stream for 24.37
km/s — less than the paper's own single-impulse payload dive costs at 4 solar
radii.

### The plunger is not free, but it is cheap where the node is slug-poor

The near-radial plunge was left "unpriced" above. Pricing it turns out to matter,
because the intuition cuts both ways and the repo's own node model only encodes
one side of it.

`dive_node()` hardcodes 180 degrees, so **the whole depth trade of ADR
0020/0021/0022 is built on a head-on node** — which is precisely the arrival that
needs the retrograde placement, the leg that gets dearer as the dive is backed
out. The alternative deserved a number.

Two corrections to the naive reading, in order.

**The plunger arrives at 135 degrees, not 90.** Payload and plunger reach
perihelion within a percent of the same speed (306.0 against 306.0 km/s at 4 R☉),
so their relative velocity bisects the tangential and radial axes. The closing
speed falls by exactly 1/root 2 — 612.0 to 432.7 km/s — at every depth.

**But `beta` rises, because the `cos theta` debit falls with it.** That term is
the impactor's own momentum arriving backwards, and a 135-degree arrival pays
0.707 of it instead of a full 1. So the geometry that loses closing speed gains
impulse per impactor, and which wins is a question about the **slug ratio** —
the debit is `k`-independent while the useful term grows as `sqrt(k)`:

| `k` | `beta` head-on | `beta` plunger | `beta` gain | `v_e` head-on | `v_e` plunger | Isp kept | heat |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0954 | 0.1295 | 1.357x | 58.4 km/s | 56.1 | **0.960x** | 0.500x |
| 3 | 0.5492 | 0.6713 | 1.222x | 112.0 | 96.8 | 0.864x | 0.500x |
| 8.5 | 1.3875 | 1.5732 | 1.134x | 99.9 | 80.1 | 0.802x | 0.500x |
| 30 | 3.3128 | 3.5473 | 1.071x | 67.6 | 51.2 | **0.757x** | 0.500x |

So "the plunger costs about root two of exhaust speed" is wrong at every `k`, and
"it costs nothing" is wrong at the repo's own `k`. It costs **4 percent at `k` = 1
and 24 percent at `k` = 30**, and the recovery never reaches the full root two
because `beta` cannot gain it.

**What it buys is a clean halving of the collision heat, at every `k`.**
Thermalised energy goes as `w**2` and `w` falls by exactly 1/root 2, so the
factor is 0.500 independent of the slug ratio. Read as depth — which is the axis
ADR 0020 shows is expensive, 2.51x the doubling time from 4 to 32 R☉ — a plunger
at 4 solar radii collides as gently as a head-on node at **7.86**
(`plunger_equivalent_depth()`). Roughly a factor of two in depth, on the
collision term only: the solar flux at the node still goes as `1/r**2` and the
arrival geometry does not touch it.

**One framing point that keeps the trade honest.** The node's own exhaust speed
peaks near `k` = 3 (112.0 km/s) and is falling by `k` = 30 (67.6). So the slug
ratio at which the plunger looks worst is already past the node's Isp optimum,
and a node chosen for its own efficiency rather than inherited from the departure
would sit where the plunger costs 10-15 percent, not 24.

### Consequence: this changes what the split is for

At 4 solar radii the split is a 1.7x worse clock bought with a 2.7x cheaper pad,
and reasonable people would decline it. At the depth this repository already
recommends flying first, **the direct architecture has no departure at all** at
the conservative margin, and the split does.

The window this opens should be stated precisely rather than generously. ADR
0022's **pad-charged launch ledger** admits **(4, 22.93] R☉** — past 22.93 the
cycle stops returning the fifteenth of liftoff it committed to, whichever
architecture flies it. The direct departure conducts over **(4, 19.80]**. Those
overlap, so the direct route is perfectly serviceable over most of the dial; what
the split adds is the last **19.80 to 22.93 R☉**. That is a 3.1 R☉ window — and
it is the *only* part of the dial that buys the shallow dive ADR 0022 recommends,
so its narrowness is not the same as its unimportance.

The right reading of ADR 0023 is therefore not "a cheaper injection that does not
pay" but:

> The split is what makes the *recommended* shallow dive flyable. Its cost is the
> clock and its price of admission is the chain count. It buys two things: the
> last 3.1 solar radii of the admissible depth dial, which the single-impulse
> architecture runs out of at 19.80 R☉ while the pad ledger runs out at 22.93;
> and a 1.7-1.8x cut in the opposing stream's placement, which is the leg that
> gets *dearer* as the dive is backed out and which dominates the Earth-side
> budget past about 8 solar radii.

The phasing survives the move: `two_node_closure()` takes the depth as an
argument and the 5/8 pattern still closes both conditions at 16 R☉, with a
smaller far-node burn (`test_the_split_closure_can_be_flown_at_a_shallower_depth`).

### What this does not settle

- **The opposing stream is priced in placement and node geometry, not in the
  cycle.** Its share of the cycle's mass budget and its own phasing against the
  payload's arrival are still unscored, and the plunger variant has not been run
  through a full cycle ledger — only through the node. ADR 0020's
  **derived periapsis survival** already warns that a gentler node is not a
  cheaper one; nothing here revisits that, and the depth table above holds
  survival fixed at 0.60 rather than deriving it, which flatters every shallow
  row.
- **The two architectures are not yet scored head to head at a shallow depth.**
  This addendum establishes that the direct one has no conducting departure past
  19.80 R☉ and the split does; it does not report the split's growth and pad
  numbers across the depth dial, which is the obvious next table and would sit
  naturally beside ADR 0020's `depth_trade_table()`.
- **The conduction floor is imported, not re-derived.**
  `conduction_threshold_closing_speed()` was built for the **overtaking leg** in
  ADR 0022. It carries no slug-ratio dependence, so applying it to a departure
  burn is defensible, but it has not been re-checked against a departure's own
  plume history.
- **The perihelion burn is held at 35.98 km/s across the dial**, which is the
  4 R☉ tuning. ADR 0020/0022 treat the climb-out excess as the free knob instead,
  and re-running this table on that convention could move every row.
- **The dive depth and the perihelion burn are still fixed at 4 solar radii and
  35.98 km/s in the phasing tables above.** Both are knobs, and either would move
  which `(j, m)` are reachable and could shorten the clock. The reachable reuse
  band with the present knobs is roughly 0.61 to 0.63, which is why 1/2 and 2/3
  return no root.
- **The outer node's impact angle is 86.3 to 90.1 degrees**, right where the
  **impact-angle impulse law** loses its whole `cos theta` bulk term. Closing
  speed there is 153 km/s, so the **plume ignition window** and ADR 0022's
  conduction floor both clear comfortably, but the cant costs about 20 percent
  of `beta` and it is structural, not tunable — the beam is radial and the burn
  is tangential.
- **The pad-charged launch ledger of ADR 0021 is not applied** in its
  `0.25 * chain * survival` form. Only slug per delivered kilogram is reported.
- The model is circular and coplanar, like `jovian_solar_dive_cycle`. The
  co-location condition is a longitude match, and a real-ephemeris audit of it
  (the ADR 0011 treatment) has not been done.
