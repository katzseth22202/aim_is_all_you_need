# The dive node's second arrival is uncharged, and it is a rounding error

Status: accepted

Date: 2026-09-01

## What the paper should take

Everything below is circular-coplanar, node slug ratio `k_peri` = 30,
`eta_jet^2` = 0.60, and every figure is under test in
`tests/test_opposing_stream_ledger.py`. Produced by `make opposing-stream`.

1. **No ledger charges the opposing stream's placement.**
   `cycle_growth_ledger()` takes the payload's departure, the stream it eats,
   the clock and the node's survival. The two fields that know the opposing
   stream exists -- `DiveNode.opposing_impactor_fraction` and
   `OpposingStreamPlacement.retrograde_rides_the_cycle` -- are read only by
   print statements and tests. The omission is real.
2. **It is worth under 1.2 percent of doubling time.** Growth ledger: the
   Jovian 3S cycle pays 1.0014x at 4 solar radii rising to 1.0075x at 32; the
   paper's single-impulse dive pays 1.0063x to 1.0112x. Pad ledger (ADR 0021):
   0.45 to 2.00 percent of returned mass, and it **flips no verdict** -- rows
   that cleared the 1/15 floor still clear it, and the two that already failed
   (32 solar radii Jovian at 0.963, the paper's dive at 0.675) still fail.
3. **The reason is mass, not impulse.** `k_peri` = 30 means one impactor
   kilogram vaporises 30 kg of slug, so the node wants only 0.17-0.49 kg of
   opposing stream per impactor kilogram of payload. The paper's dive places
   its stream at 44.94 km/s through a nozzle that delivers a fifth of the
   vehicle, and *still* only loses 1.1 percent. Quote the mass, not the burn.
4. **The two readings are not ordered.** `pad_charge / growth_charge` is
   exactly `(1 - delivered) / (1 - placed)`. The pad reading bites harder only
   where the placement burn wastes more of its vehicle than the payload's own
   departure does -- true for the paper's dive at every depth and for the
   shallow Jovian rows, false for the deep ones. Do not call either "the
   conservative one".

**On the route, which had to exist before any of that meant anything.**

5. **The Jovian route is flyable at every depth 4-32 solar radii.** A one-way
   tangential launch at 11.06-11.83 km/s of Earth excess reaches the
   **dive-placement floor** at Jupiter; an unpowered flyby bends it retrograde
   needing 72.5-77.9 degrees against 119.6-126.5 available, a margin never
   tighter than 1.5x; it then falls to the node.
6. **Co-location is nearly free.** Making it arrive at the payload's perihelion
   *at the payload's instant* costs **0.12 percent at 4 solar radii rising to
   0.32 at 32** -- 13 to 38 m/s. Arrival excess above the floor sweeps the
   perihelion-longitude gap through a full 360 degrees over roughly 50 m/s
   (the dive time of flight moves 0.25 yr for 0.05 km/s, and the gap carries
   that at 360 deg/yr), so a root always exists and **the solutions are
   discrete**, the same shape as the **synodic closure**.
7. **The schedule is a fixed offset inside six weeks.** The opposing stream
   leaves Earth 6.7 days *before* the payload at 4 solar radii and 40.5 days
   *after* it at 32, passing through simultaneity near 13.
8. **Per kilogram the second arrival is no dearer than the first**: 0.95-1.09x
   the payload's own departure excess, via Jupiter.

**On the architecture that has no Jupiter.**

9. **The bi-elliptic far node phases itself exactly, with no knob.** Injected at
   one far node, payload and opposing stream fly *the same ellipse* in opposite
   senses. Same semi-major axis, so the same half-period; and an ellipse's
   perihelion is antipodal to its aphelion whatever direction it is flown, so
   both perihelia are the same point. They arrive together at 180 degrees.
   There is no residual to drive to zero and nothing to tune.
10. **Neither leg ever goes deeper than the node.** Closest approach is the dive
    depth itself for both vehicles, so the shallow dive's whole reason for
    existing -- avoiding the heating -- survives the placement intact.
11. **The far node pays both legs and the depth dial splits them.** At 1.9649 AU
    the payload's cut falls 14.541 to 9.478 km/s from 4 to 32 solar radii while
    the opposing stream's flip rises 20.362 to 25.425. A 3 AU node makes both
    cheaper (10.250/14.069 to 6.873/17.446) and the coast twice as long.

**Caveats the paper must carry.**

- **The split dive's own charge is not reported, and cannot be.** That
  architecture is Earth-only, so the cheap Jupiter route is not open to it and
  both far-node burns need impactors delivered to ~1.96 AU. That delivery is
  unpriced (worklist S1), so item 2's bound does **not** transfer to it.
- The charge assumes the opposing projectiles are pushed at Earth by the **same
  returning beam** the payload's departure eats, aimed tangentially for the
  Jovian route and retrograde for the direct one.
- It assumes the required mass is set by **node throughput**,
  `(1 - survival) / k_peri`. That is `DiveNode`'s own definition, but it makes
  the whole verdict a statement about `k_peri`: at 8.5 the penalty is still
  under 4 percent, at 3 it is 6-11, at 1 it is 19-34. The result holds because
  ADR 0016 already put the node at 30.
- The Jovian route's co-location has had **no real-ephemeris audit** (the ADR
  0011 treatment); it is a longitude and epoch match in a circular coplanar
  model.
- The **Jupiter-only assist chain is not scored here** and needs no charge: its
  retrograde return is flown only from the Jovian bend to the 1 AU crossing and
  its perihelion is never reached, so it has no dive node and no second
  arrival. Its one arrival carries its full launch cost in `delivered_fraction`.
  Note its currency is kilograms *launched*, not impactor kilograms, so it
  cannot be put on this ADR's axis anyway.

## Context

ADR 0023 established that the dive node needs two arrivals and that the depth
dial moves them opposite ways, and left the second one priced in placement
impulse only. The companion paper then held three claims pending prices, of
which S3 was the blocking one: *does any existing ledger already absorb the
opposing stream's placement, and if not, how much does adding it move the
doubling times?* The concern was specific and reasonable -- placing the stream
costs 35.48 km/s at 4 solar radii rising to 44.94 at 32, which is the same
order as the payload's whole injection and moves the opposite way, so if it
were genuinely uncharged it would move doubling times in three chapters rather
than one subsection.

Two things had to be established before that could be answered, and the first
reframed the question.

**First, the 35.48-44.94 figure is architecture-specific.**
`opposing_stream_placement()` already computed two routes and ADR 0023 quoted
one of them. `direct_from_earth_excess` is the placement from 1 AU *without
Jupiter*, which is right for the paper's single-impulse resonant dive and for
the split dive. `retrograde_tangential_departure` is the one-way launch via
Jupiter, 11.06-11.83 km/s, which is right for the Jovian dive cycle -- and is
0.95-1.09x that cycle's own departure rather than three to four times it. The
paper's own CONTEXT.md already carried the Jovian number; this repository's
carried the Earth-direct one; neither said which architecture it belonged to.

**Second, the cheap route had to be shown to exist.** A price for a flight
nobody has demonstrated is worse than no price. The audit is item 5-8 above.
The one surprise was that it is *phasing* rather than energy that could have
killed it, and phasing turned out to cost tens of metres per second.

## Decision

Price the second arrival in both currencies and record the route that makes it
possible. `src/opposing_stream_ledger.py`, `make opposing-stream`.

- `opposing_placement_route()` -- audits the Jovian route leg by leg and solves
  the co-location root. The only free knob is departure excess; aim is spent on
  being tangential and the dive geometry is spent on the floor.
- `opposing_charge()` -- the charge in both ledgers off one set of inputs, so
  the `(1 - delivered) / (1 - placed)` relation between them is structural
  rather than a coincidence of two code paths.
- `bielliptic_coplacement()` -- the far-node phasing result, which is closed
  form because the symmetry does the work.

### A bug this turned up

`opposing_stream_placement()` solves `retrograde_tangential_departure` to land
*exactly* on the retrograde floor. The floor is a **tangency**: there the
post-flyby state is purely tangential at Jupiter, so `_solve_dive_state()` sits
on the edge of its own bracket with `v_radial = 0` and returns `None` on
floating-point luck. At 23 solar radii `_dive_leg()` therefore fails outright;
at 4, 8 and 32 it happens to survive. Anything building a real leg from that
field gets a depth-dependent coin flip. `TANGENCY_NUDGE` = 1e-4 km/s is the
workaround here and the behaviour is pinned in
`test_the_placement_floor_is_a_tangency_that_defeats_the_dive_solver`. The
field itself is left alone: it is correct as a *floor*, and callers wanting a
flyable leg should ask this module.

### What was considered and rejected

**The straight-down plunger as the second arrival.** Rejected, but not for the
reason first proposed. A true radial drop has zero angular momentum, so its
perihelion is `r = 0` and it does pass through the Sun -- which is why its cost
is quoted flat at Earth's own 29.785 km/s at *every* depth, a depth-independent
price being the tell that it is not aiming at a depth at all. That is
survivable: the collision happens on the way in, the projectile arrives from
1 AU having never been closer than the node, and what continues inward is only
the fraction that missed. Nor is timing the problem: the fall from 1 AU takes
0.172-0.177 yr against the payload's 0.543 from a 1.9649 AU node, so it needs a
launch offset, and because a radial fall sweeps exactly 0 degrees of longitude
it must leave when Earth is at the node's longitude -- once a year, a resonance
condition of the kind the architecture already solves.

It is rejected because of **geometry at high slug ratio**. Payload and plunger
reach perihelion at nearly the same speed, so their relative velocity bisects
the two axes: the impact is at 135 degrees, not 90, and the closing speed falls
by exactly 1/root 2 at every depth. Against that the `cos theta` debit softens
from -1 to -0.707, and since the debit is `k`-independent while the useful term
grows as `sqrt(k)`, the relief is large where the node is slug-poor and
negligible where it is slug-rich. At `k_peri` = 30 the plunger keeps 0.757x the
Isp. It is simply worse, and the reasons are ADR 0023's, not heating.

Also note the trap: forcing the plunger to *bottom out* at the target rather
than pass through means keeping 15.16 km/s of tangential speed at 32 solar
radii, which is a 14.62 km/s burn -- the payload's own prograde injection. It
would then arrive alongside the payload instead of across it and there would be
no collision. "Plunger" and "bottoms out at the target depth" are mutually
exclusive; zero angular momentum is its defining property.

## Addendum, 2026-09-02: the plunger is inadmissible, not merely inferior

The "considered and rejected" section above calls the plunger's Sun-crossing
"survivable, because the collision happens on the way in". That is true of the
*approach* and wrong about the architecture, and this ADR should be read with
**node-depth admissibility** (CONTEXT.md; `docs/paper_changes_owed.md` P1).

The rule adopted since: every trajectory in the architecture -- payload, opposing
stream, and any projectile stream feeding a node -- must have a perihelion no
lower than the node's depth. A zero-angular-momentum drop has perihelion `r = 0`,
and the fraction the node fails to consume is therefore on a Sun-impacting
trajectory; there is always such a fraction, which is why
`rendezvous_timing_tolerance()` exists. Backing a payload out to 23 solar radii
for thermal reasons while aiming its ammunition at `r = 0` moves the exposure
onto the half of the collision nobody was scoring rather than escaping it.

So the plunger is ruled out **before** the 135-degree geometry and the
slug-ratio argument apply. Those stay as recorded analysis -- they are why it
would have lost anyway at `k` = 30 -- but they are no longer the reason.

Nothing numerical in this ADR moves: no function selects the plunger, and the
Jovian placement route it prices targets the node depth as its perihelion at
every arrival excess, so it satisfies the constraint by construction. The same
is true of ADR 0025's bi-elliptic co-placement, which the rule promotes from
convenient to load-bearing.

## Consequences

S3 lifts. The paper can state the omission and its bound in a sentence, and the
restructure S3 was blocking is unblocked -- though the trace argues *against*
the restructure it was protecting. The two-arrival asymmetry is a real result
about impulse and node geometry, but it moves no growth number by more than
1.2 percent, and promoting a sub-1.2-percent effect to a subsection's spine in a
paper whose currency is growth would repeat the overstatement the 2026-09-01
grill corrected twice elsewhere. Recommendation: state it with its bound, keep
the delta-v family leading.

S1 does not lift, and is now load-bearing in a second place: without a far-node
delivery price the split dive can be charged for neither its payload's injection
nor its opposing stream's flip.

S2 is untouched by this ADR.
