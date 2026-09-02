# The shallow dive is affordable for the direct route; its stated node survival is not

Status: accepted

Date: 2026-09-02

## What the paper should take

Circular-coplanar, departure `k` = 30, node `k_peri` = 30, `eta_jet^2` = 0.60,
expansion margin 1.5. Every figure is under test in
`tests/test_shallow_dive_burn_trade.py`. Produced by `make shallow-dive`.

**On the question S2 was asked to decide.**

1. **The direct route can fly shallow.** A perihelion burn of **38.10 km/s**
   (1.059x the paper's 35.98 tuning) holds its Earth departure conducting at ADR
   0022's 22.93 solar-radii pad floor, and the cycle still grows: **2.505 per
   pass, doubling in 0.657 yr**. So the **depth conduction crossing** shows that
   the direct route cannot fly shallow *at the paper's tuning*, **not** that the
   split dive is required for a shallow dive to exist. The weaker claim the
   paper already holds is the true one; ADR 0023's stronger reading is not.
2. **The trade is one-sided in the burn**, so the *cheapest conducting* burn is
   always right: a larger burn buys conduction depth already in hand while
   costing node survival exponentially. Every row here is therefore the direct
   architecture's best case at that depth.
3. **The extra 6 percent of burn is not the expensive part.** It costs 7.4
   percent of node survival (0.2792 to 0.2589 at 22.93 solar radii) and about
   1 percent of clock -- and the clock moves the *helpful* way, because a hotter
   perihelion burn climbs out faster and closes the loop slightly shorter.
4. **Depth itself is the expensive part.** At 19.80 solar radii, with **no extra
   burn at all**, doubling is already 1.93x the 4 solar-radii value. By 22.93 it
   is 2.14x, by 32 it is 3.09x, by 36 it is 3.83x.
5. **The burn search runs out before the physics does.** No burn in (18, 46)
   km/s keeps the departure conducting past about 40 solar radii, which is the
   edge of `bielliptic_dive_split`'s own 4-48 conduction bracket rather than a
   physical wall. Do not quote it as one.

**On what turned out to matter more.**

6. **The paper's ledger holds node survival at a stated 0.60 across the whole
   depth dial**, and that is what really flatters a shallow direct dive.
   Derived from the boost the way `dive_node()` does it, survival is 0.5895 at
   4 solar radii but **0.2589 at 22.93 and 0.1627 at 32** -- the stated value is
   too generous by 1.02x, **2.32x** and **3.68x** respectively.
7. **The cause is the node's exhaust speed collapsing with the arrival speed**:
   68.09 km/s at 4 solar radii falling to 28.20 at 22.93 and 22.38 at 36. The
   same mechanism ADR 0020 recorded for the Jovian cycle -- a gentler node is a
   less efficient one -- had never been applied to the paper's own dive.
8. **On the fixed 0.60 the doubling time is nearly flat across the dial**
   (0.3048 at 4 solar radii, 0.3431 at 22.93, 0.3091 at 32 -- non-monotone, and
   inside 13 percent), which is the illusion that makes backing the dive out
   look almost free. Derived, the same rows are **1.92x and 3.07x worse**.
9. The pad margin falls from 0.669 to 0.237 at 22.93 and 0.147 at 36, but it
   **never cleared ADR 0021's 1/15 floor at any depth including 4**, so this is
   the same failure ADR 0021 already recorded, not a new one.

**Caveats the paper must carry.**

- The stated 0.60 is **defensible where the paper actually uses it**, at 4 solar
  radii, where the derived value is 0.5895. Item 6 is a warning against reusing
  it at depth -- which is exactly what a shallow-dive comparison invites -- not
  a correction to the published headline.
- The conduction floor is **imported from ADR 0022's overtaking leg** and not
  re-derived for a departure plume, as ADR 0023 already noted.
- `dive_node()` hardcodes a head-on 180-degree node (ADR 0023 item 18), so every
  derived survival here inherits that assumption.
- The **opposing stream's placement is not charged here**; it is ADR 0024's, and
  worth under 1.2 percent on top.
- The **partial split** is still unscored on the pad, because its far node's
  delivery is unpriced (worklist S1). The *phased* split is scored in the
  addendum below.

## Context

ADR 0023 found that the direct single-impulse departure aims ~31 degrees off the
arriving stream, so it runs away from what feeds it and cools itself through the
burn, dropping below ADR 0022's 79.56 km/s conduction floor at 19.80 solar
radii. Since ADR 0022 recommends starting at 23, that reads as the split dive
being what makes a shallow dive flyable at all.

The companion paper held the claim, on the grounds that the crossing moves ~1.9
solar radii per km/s of perihelion burn and nobody had priced a larger one:
38.15 km/s reaches 23 solar radii for 6 percent more burn, which would make the
crossing a statement about a tuning rather than about an architecture. The
paper's worklist recorded this as S2 and noted why
`paper_resonant_dive_ledger()` could not settle it: **it takes the stream speed
and the Earth burn and charges the node nothing for a larger boost**, so it
reports the extra burn as a free improvement, which cannot be right.

The charge it was missing already existed. `dive_node()` derives survival as
`exp(-boost / exhaust)` rather than stating it, so a larger perihelion burn pays
for itself in mass at the node. Applying it answers S2 -- and then keeps going,
because the same function says the *stated* survival was the larger error all
along.

## Decision

Score the direct architecture at depth with its node charged for its own boost.
`src/shallow_dive_burn_trade.py`, `make shallow-dive`.

- `conducting_burn()` -- inverts the conduction crossing to the cheapest burn
  that still conducts at a given depth. Returns None both when no burn reaches
  the depth *and* when the constraint is slack (at 4 solar radii every burn in
  the bracket conducts), so callers wanting a specific depth-and-burn pass the
  burn.
- `shallow_dive_row()` -- the cycle end to end on the derived survival, carrying
  the stated-survival scoring alongside so the gap is reported rather than
  inferred.

### Two bugs this turned up

**`resonant_dive_at_depth()` accepted `periapsis_burn` and dropped it.** It
passed only the perihelion radius to `single_impulse_resonant_dive()`, which did
not expose a burn parameter at all and so always used the 35.9807 km/s default.
`depth_conduction()` calls it *with* a burn, so every conduction crossing away
from the default tuning was computed with the Earth boost and closing aphelion
frozen at the paper's tuning while only the returning beam varied -- including
the burn-sensitivity table S2's whole framing rests on.

Both are fixed: `single_impulse_resonant_dive()` now takes `periapsis_burn` and
threads it through the climb-out, and `resonant_dive_at_depth()` passes its
argument on. **The published figures barely move**, because the climb is only
~10 days of a ~326 day cycle: the crossing is unchanged at the default tuning
(19.804), and moves 23.000 to 23.013 at 38.148 km/s and 35.912 to 35.969 at 45.
The largest shift anywhere in 20-45 km/s is 0.06 solar radii. The bug was real
and would have grown with the burn; it changes nothing anyone has quoted.

## Addendum: the split does not pay for its launch either, past 5.58 solar radii

Added the same day, because S2's worklist entry named it as "related and also
unrun": the split dive's own **pad-charged launch ledger** across the depth dial,
in ADR 0021's `0.25 * chain * survival` form.

ADR 0023's headline is that the split **"buys the pad, not the clock"** -- slower,
but far less launched mass. That claim was only ever made at 4 solar radii, and
only in launched-slug-per-delivered-kilogram (0.875 against 2.365). It had never
been put in ADR 0021's *committed* currency: returned mass per kilogram off the
pad, against the 1/15 floor. Scored there, on the phased 5/8 closure whose far
node eats the beam's leftovers and so costs no second launch, and with the node's
survival derived rather than stated:

| depth | split margin | direct margin | split edge | split clears? |
| ---: | ---: | ---: | ---: | :--- |
| 4.00 | **1.179** | 0.657 | 1.80x | **yes** |
| 8.00 | 0.808 | 0.459 | 1.76x | no |
| 16.00 | 0.461 | 0.305 | 1.51x | no |
| 22.93 | 0.306 | 0.237 | 1.29x | no |
| 32.00 | 0.185 | 0.170 | 1.09x | no |

10. **The claim holds, and only just.** The split does clear the committed
    fifteenth where the direct route fails it -- 1.179 against 0.657 at 4 solar
    radii. ADR 0023 was right that this is the scoreboard the split wins on.
11. **It stops paying at 5.58 solar radii** (`split_pad_crossing()`), bisected on
    the constraint. The band over which the split earns its launch is
    **(4, 5.58]** -- against the Jovian dive cycle's (4, ~23] on the same floor.
12. **Its edge shrinks exactly where it would be needed.** 1.80x the direct
    route's margin at 4 solar radii, 1.29x at 22.93, **1.09x at 32**. The
    architecture that buys the pad buys it only where the thermal case is worst.
13. **At ADR 0022's recommended shallow end, neither architecture earns its
    launch** (0.306 and 0.237). The choice there is not between a payer and a
    failer; it is between two failers, one 29 percent less bad.

This does not retire ADR 0023's claim -- it bounds it. "The split buys the pad"
is true at the paper's own 4 solar-radii dive and false by 6 solar radii, so the
sentence must carry its depth or it misleads. And it sharpens item 4 above: the
reason to back the dive out is thermal, and *both* architectures pay for that in
the launch ledger, the split merely more slowly.

The **partial split** is not scored here. Its far node fails outer-node
co-location, so it needs a dedicated impactor delivery that nothing has priced
(worklist S1), and a pad ledger that omits that delivery would flatter it.

## Second addendum: the partial split's dominance does not survive its own delivery

Worklist item S1, which had been recorded as a flag rather than work on the
grounds that nothing the paper claims depends on it. That remains true of the
paper. It is not true of the **partial split**, whose entire case is a claim
about launched mass, and which cannot be scored at all while the cost is missing.

`split_dive_ledger()` charges the impactors *consumed* at both nodes and nothing
for getting them to the far one. The partial split (outbound perihelion 0.4918
AU) fails outer-node co-location by 110.57 degrees, so the returning beam does
not reach its far node at 1.9649 AU and that node must be supplied.

**The trap, and the result.** The far node does not need mass at 1.9649 AU. It
needs mass **moving at 153.35 km/s** there, against a vehicle doing 13.45. A
Hohmann delivery arrives nearly co-moving and is worth nothing as an impactor.
Buying that speed from 1 AU costs:

| arrival | impactor speed | Earth departure | delivered | extra slug | total | beats 2.365? |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| co-linear (lower bound) | 139.91 | **113.20** | 0.0103 | 2.016 | **3.552** | no |
| perpendicular (real) | 152.76 | 125.80 | 0.0063 | 3.327 | 4.863 | no |

14. **The delivery costs about six times the payload's own departure**, 113.20
    km/s against 19.12, and delivers **1 percent** of what is launched.
15. **The partial split's 1.536 kg/kg becomes 3.552**, against the paper's own
    dive at 2.3654 (computed from the degenerate split, not quoted). **It no
    longer beats it.** S1's settling question -- "whether the partial split still
    beats the single-impulse dive's 2.365 kg/kg once the far node's delivery is
    added to its 1.536" -- answers **no**.
16. **The verdict does not rest on the arrival geometry.** Co-linear is the
    cheapest possible arrival and therefore a lower bound; the perpendicular
    arrival the beam actually makes costs more and loses by more.
17. **So the phased split's "leftovers" are not a convenience, they are the only
    affordable source.** The beam carries 153.0 km/s to the far node for nothing,
    because it is a climb-out from the dive. Nothing launched from 1 AU can match
    that at a sane price -- the cheap way to make fast mass is to drop it down the
    Sun's gravity well first, which is what the beam already is.

**But that prices only one way of feeding it, and it is the wrong way.** See the
third addendum, which reverses this verdict.

## Third addendum: a feeder dive is twenty times cheaper, and revives the partial split

The second addendum priced the far node's supply as a **direct high-speed launch
from 1 AU**, and concluded the partial split loses. That priced the obvious route
rather than the architecture's own route, and the difference is a factor of
twenty.

The architecture never makes fast mass by pushing it. It makes fast mass by
**dropping it down the Sun's well and collecting it on the climb-out** -- that is
what the beam is. A **feeder** launched onto the same dive the payload flies
crosses the far node radius carrying the same speed the split's own beam does,
because it is the same dive:

| feeder depth | closing speed at the node | needed | slug per kg placed |
| ---: | ---: | ---: | ---: |
| **4 R_sun** | **153.29** | 153.35 | **4.706** |
| 6 | 139.17 | 153.35 | 5.116 |
| 8 | 129.98 | 153.35 | 5.487 |

18. **A 4 solar-radii feeder meets the requirement essentially exactly** (153.29
    against 153.35), which is not a coincidence: it is the dive that makes the
    split's own beam.
19. **It costs 4.706 kg of spent slug per kilogram placed, against the direct
    launch's 96.1 -- twenty times cheaper.** Slug is counted the way
    `split_dive_ledger` counts it: per kilogram entering the feeder's Earth burn
    it spends `1 - f` there and `f * (1 - s)` at its perihelion, and `f * s`
    arrives, so the cost per kilogram placed is `((1-f) + f(1-s)) / (f*s)`.
20. **The partial split's 1.5358 kg/kg becomes 1.6345, not 3.552.** Against the
    paper's 2.3654, **it still beats it.** S1's settling question answers *yes*
    on this route and *no* on the other, so the honest answer is that **the
    verdict turns entirely on how the far node is fed**, and the spread between
    the two answers is larger than the quantity being decided.
21. **A shallower feeder does not help twice over.** It climbs out too slowly to
    meet the closing speed *and* costs more per kilogram placed, because node
    survival falls faster than the departure gets cheaper. The feeder's depth is
    set by the speed it must carry, not by its own economy.

**What the feeder is not charged for, and this is the open edge.** It is one-way
and expendable, so unlike the payload's dive it carries no **Earth re-intercept**
condition -- it needs only to put its beam ray through the far node when the
vehicle is there. That is two conditions (longitude and epoch) on at least two
free knobs (launch date, departure aim), so discrete solutions should exist by
the same argument as every other closure here. **But no such closure has been
solved.** Until one is, the feeder is a costed route rather than a demonstrated
one, and this addendum's verdict inherits that.

Also uncharged: the feeder's own perihelion node needs an **opposing stream**,
which ADR 0024 prices at under 1.2 percent, and the feeder delivers nothing at
Earth -- the latter is already charged, since its entire cost is attributed here.

**Consequence for the companion paper.** Its held sentence -- "better on both
axes the ledger scores" -- should be **neither lifted nor retired**. It should be
kept, with the reason made explicit: the partial split's case rests on a far-node
supply that is affordable only as a feeder dive, and that feeder's phasing has
not been solved.

## Consequences

S2 lifts, and it lifts *against* the stronger claim. `sec:self_cooling_departure`
can say what it now says with more confidence: the crossing is real, it is a
property of the tuning, and 6 percent more perihelion burn buys past it. The
depth-window claim the paper already withdrew should stay withdrawn -- at 38.10
km/s the conduction crossing (23.01) rises *above* ADR 0022's pad floor (22.93),
so the band the split was said to open is empty at that tuning.

What replaces it is a better argument for the same conclusion, and one the paper
can make without holding anything: **flying shallow costs the direct route
roughly double its clock, and the published ledger hides half of that** by
holding node survival fixed. The split's case at depth should be made against
the derived numbers in this ADR, not against the flat ones.

ADR 0023's item 12 and its addendum should be read with item 1 above: the
crossing stands, the architecture claim built on it does not.
