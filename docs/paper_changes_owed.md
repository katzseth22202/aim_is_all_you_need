# Changes owed to the paper

The mirror of `Balloon-Pulse-Propulsion`'s `docs/deferred_to_companion_repos.md`.
That document lists prices the paper wanted from here; this one lists what came
back, plus what the work changed that the paper does not yet say.

Raised 2026-09-02, working worklist items S1-S3. Target repo:
`katzseth22202/Balloon-Pulse-Propulsion`.

Backing ADRs in this repository: 0024 (`make opposing-stream`) and 0025
(`make shallow-dive`). Every figure below is under test; `make test-all` is the
gate.

---

## P1. Constrain admissible trajectories to the node depth, and retire the straight drop

**The largest change here, and it is a modelling constraint rather than a number.**

**Adopt this, explicitly, wherever the dive node is described:** every trajectory
in the architecture -- the payload, the **opposing stream**, and any projectile
stream feeding a node -- must have a perihelion **no lower than the dive node's
depth**. Nothing in the system may pass inside the node.

**Why.** The **straight-down plunger** has zero angular momentum by definition,
so its perihelion is `r = 0`: it does not skim the Sun, it enters it. It crosses
the node radius on the way down and keeps going. Three consequences make that
disqualifying rather than merely untidy:

1. **The unconsumed fraction impacts the Sun.** A stream is never perfectly
   consumed -- `rendezvous_timing_tolerance()` exists precisely because there is
   an along-track miss budget -- so a plunger architecture continuously puts
   projectiles on a Sun-impacting trajectory.
2. **It contradicts the reason for the depth.** A shallow dive is *chosen* to
   escape the thermal load. An architecture that backs the payload out to 23
   solar radii while aiming its ammunition at `r = 0` has not escaped anything;
   it has moved the exposure onto the half of the collision nobody was scoring.
3. **The depth dial stops meaning one thing.** Under the constraint, "the dive
   is at 23 solar radii" is a statement about the closest approach of *anything*
   in the system. Without it, it is a statement about one of the two arrivals.

**The tell, worth putting in the paper.** The radial placement's cost is quoted
**flat at Earth's own 29.785 km/s at every depth** (`dive_placement_excess_floor`
with `sign = 0`). A depth-independent price is the giveaway that the trajectory
is not aiming at a depth at all. Compare the prograde column, which falls 24.09
to 14.62 km/s from 4 to 32 solar radii because it *is*.

**And the trap that closes the argument.** Forcing the plunger to bottom out at
the target instead of passing through means keeping the tangential speed that a
perihelion there requires -- 15.16 km/s at 32 solar radii, so a 14.62 km/s burn.
But that *is* the payload's own prograde injection. It would then arrive
alongside the payload rather than across it, and there would be no collision at
all. **"Plunger" and "bottoms out at the target depth" are mutually exclusive**;
zero angular momentum is its defining property.

**So retrograde placement is not merely cheaper-per-geometry, it is the only
admissible way to get a head-on arrival.** That promotes ADR 0024's bi-elliptic
result from convenient to load-bearing: injected at one far node, payload and
opposing stream fly the *same ellipse* in opposite senses, so they arrive
together at 180 degrees with no tuning knob, and **neither leg ever goes closer
to the Sun than the node**. The constraint is satisfied by construction there.

**What changes in the paper.** Wherever the near-radial plunge is offered as
"the third option" it should be marked **inadmissible**, not merely worse. The
135-degree geometry and the 1/root-2 closing-speed penalty stay as recorded
analysis -- they are why it would have lost anyway at `k` = 30 -- but they stop
being the reason it is rejected.

**What changes here.** Nothing numerically: no function *selects* the plunger,
so no published figure moves. The rule is now **enforced rather than stated**:
`placement_admissibility()` flags every placement sense, `admissible_placements()`
returns only the two that comply, and `require_admissible()` raises at any site
turning a sense into a trajectory. Worth knowing why that was needed -- the
plunge is the **cheapest-looking** of the three from Earth (13.06 km/s against
the retrograde placement's 15.68 at 23 solar radii), so a rule left to judgement
is a rule left to lose.

**Note against ADR 0024.** Its "considered and rejected" section calls the
Sun-crossing "survivable, because the collision happens on the way in". That is
true of the *approach* and wrong about the architecture, for the reasons above.
ADR 0024 should be read with this item.

---

## P2. S3 is answered: the second arrival is uncharged, and it is a rounding error

Backing: ADR 0024. Lifts `sec:split_dive`'s third hold.

- **No ledger charges the opposing stream's placement.** `cycle_growth_ledger()`
  takes the payload's departure, its stream, the clock and node survival. The two
  fields that know the opposing stream exists are read only by print statements
  and tests. The omission is real.
- **It is worth under 1.2 percent of doubling time.** Growth ledger: Jovian 3S
  pays 1.0014x at 4 solar radii to 1.0075x at 32; the paper's dive 1.0063x to
  1.0112x. Pad ledger: 0.45-2.00 percent of returned mass, **flipping no
  verdict**.
- **The reason is mass, not impulse.** `k_peri` = 30, so the node wants only
  0.17-0.49 kg of opposing stream per impactor kilogram of payload.
- **The placement price is architecture-specific and the paper should say which
  it means.** 35.48 rising to 44.94 km/s is the *Earth-direct* route -- what the
  single-impulse dive and the split dive have. The Jovian dive cycle places the
  same stream for **11.06-11.83 km/s** via a one-way tangential launch and an
  unpowered retrograde bend, which is 0.95-1.09x its own departure. Co-locating
  it with the payload in longitude *and* time costs a further 13-38 m/s.
- **The Jupiter-only chain needs no charge at all**, and for a structural reason
  worth one sentence: its retrograde return is flown only from the Jovian bend to
  the 1 AU crossing and its perihelion is never reached, so it has no dive node
  and no second arrival. Its one arrival carries its full launch cost.

**So the two-arrival asymmetry should be stated with its bound.** It is a real
result about impulse and node geometry; it is not a correction to the growth
numbers. Recommendation: keep the delta-v family leading `sec:split_dive` and
state the asymmetry with "worth under 1.2 percent of doubling", rather than
promoting a sub-1.2-percent effect to the subsection's spine.

---

## P3. S2 is answered: the direct route can fly shallow, and the crossing is about the tuning

Backing: ADR 0025. Lifts `sec:self_cooling_departure`'s hold.

- **38.10 km/s (1.059x the paper's 35.98 tuning) holds the direct departure
  conducting at ADR 0022's 22.93 solar-radii pad floor**, and the cycle still
  grows: 2.505 per pass, doubling 0.657 yr. So the **depth conduction crossing**
  shows the direct route cannot fly shallow *at the paper's tuning*, **not** that
  the split is required. The paper's held weaker claim is the true one; ADR
  0023's stronger reading does not survive.
- The extra 6 percent of burn costs 7.4 percent of node survival and about 1
  percent of clock -- and the clock moves the *helpful* way, since a hotter
  perihelion burn climbs out faster.
- **Depth itself is the expensive part**: at 19.80 solar radii with no extra burn
  at all, doubling is already 1.93x the 4 solar-radii value.
- The withdrawn 3.1 solar-radii window should **stay** withdrawn: at 38.10 km/s
  the conduction crossing (23.01) rises above the pad floor (22.93), so the band
  the split was said to open is empty at that tuning.

---

## P4. The stated node survival flatters every shallow row

Backing: ADR 0025. **This is the larger of S2's two findings and the paper does
not currently say it.**

`paper_resonant_dive_ledger()` holds `periapsis_survival` at a stated **0.60
across the whole depth dial**. Derived from the boost the way `dive_node()` does
it, survival is 0.5895 at 4 solar radii but **0.2589 at 22.93 and 0.1627 at 32**,
because the node's exhaust speed collapses with the arrival speed (68.09 to 22.38
km/s). The stated value is too generous by 2.32x and 3.68x there.

| depth | derived | stated | doubling derived | stated | flattered by |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4.00 | 0.5895 | 0.60 | 0.3075 | 0.3048 | 1.009x |
| 22.93 | 0.2589 | 0.60 | 0.6570 | 0.3431 | 1.915x |
| 32.00 | 0.1629 | 0.60 | 0.9484 | 0.3091 | 3.068x |

**Worst of all, on the fixed 0.60 the doubling time comes out nearly flat across
the dial** (0.3048 / 0.3431 / 0.3091, non-monotone, inside 13 percent). That
flatness is the illusion that makes backing the dive out look almost free.

The stated value is **defensible where the paper actually uses it**, at 4 solar
radii, where the gap is 1.009x. This is a warning against reusing it at depth --
which is exactly what a shallow-dive comparison invites -- not a correction to
the published headline. Same mechanism as ADR 0020's **derived periapsis
survival**, never before applied to the paper's own dive.

---

## P5. Bound "the split buys the pad" by its depth

Backing: ADR 0025 addendum. Answers S2's "related and also unrun" clause.

ADR 0023's headline claim was only ever made at 4 solar radii and only in
launched-slug-per-delivered-kilogram (0.875 against 2.365), never in ADR 0021's
committed currency. Scored there, on the phased 5/8 closure and with survival
derived:

| depth | split margin | direct margin | split edge | clears 1/15? |
| ---: | ---: | ---: | ---: | :--- |
| 4.00 | **1.179** | 0.657 | 1.80x | **yes** |
| 8.00 | 0.808 | 0.459 | 1.76x | no |
| 22.93 | 0.306 | 0.237 | 1.29x | no |
| 32.00 | 0.185 | 0.170 | 1.09x | no |

- **The claim holds, and only just.** The split does clear the floor where the
  direct route fails it. ADR 0023 was right about the scoreboard.
- **It stops paying at 5.58 solar radii.** The band is **(4, 5.58]**, against the
  Jovian dive cycle's (4, ~23] on the same floor. "The split buys the pad" must
  carry its depth or it misleads.
- **Its edge shrinks where it would be needed**: 1.80x at 4 solar radii, 1.29x at
  22.93, 1.09x at 32. The architecture that buys the pad buys it only where the
  thermal case is worst.
- **At ADR 0022's recommended shallow end, neither architecture earns its
  launch.** The choice there is between two failers, one 29 percent less bad.

---

## P5b. Retire the partial split's dominance, do not lift it

Backing: ADR 0025 second addendum. **Answers S1, and answers it the other way
round from what the hold was waiting for.**

`sec:split_dive_growth`'s held sentence says "better on both axes the ledger
scores", pending a price for delivering impactors to the far node. That price is
now in, and it does not lift the hold -- it **retires the claim**.

The trap is that the far node does not need mass at 1.9649 AU. It needs mass
**moving at 153.35 km/s** there, against a vehicle doing 13.45. A Hohmann
delivery arrives nearly co-moving and is worth nothing as an impactor.

| arrival | impactor speed | Earth departure | delivered | total kg/kg | beats 2.365? |
| :--- | ---: | ---: | ---: | ---: | :--- |
| co-linear (lower bound) | 139.91 | **113.20** | 1.0% | **3.552** | no |
| perpendicular (real) | 152.76 | 125.80 | 0.6% | 4.863 | no |

- The delivery costs **nearly six times the payload's own departure** (113.20
  km/s against 19.12) and delivers **1 percent** of what is launched.
- **1.536 kg/kg becomes 3.552**, against the paper's own dive at 2.3654 -- which
  this repository now computes from the degenerate split rather than quoting. The
  partial split **no longer beats the dive it was said to dominate**.
- The verdict does not rest on the arrival geometry: co-linear is the cheapest
  possible arrival and therefore a bound, and the real perpendicular arrival
  loses by more.
- **So the phased split's "leftovers" are not a convenience but the only
  affordable source.** The beam carries 153.0 km/s to the far node for nothing,
  being a climb-out from the dive; nothing launched from 1 AU matches that at a
  sane price, because the cheap way to make fast mass is to drop it down the
  Sun's well first.

**What the paper should do.** Strike the dominance framing rather than
strengthening it, and keep the partial split only as the family's rate optimum
*before* delivery is charged, clearly labelled as such. `CONTEXT.md`'s **Partial
split** entry in this repository has been corrected the same way.

---

## P6. A dangling cross-reference, and the item it points at

`docs/paper_corrections_checklist.md:101` refers to a "One thing left alone"
section of `docs/deferred_to_companion_repos.md`. **That section no longer
exists**: commit `b44907b` ("delete files we're no longer using", 2026-08-26)
deleted the file's 674 lines, and `06ea087` recreated it with only S1-S3.

The orphaned item is still live and reproduces here. `sec:axial_bag` says the
23 m column costs "0.8 kg more film". From `tab:axial_bag`: 23 m minus the sphere
is **1.31 kg** (6.16 - 4.85) and 23 m minus the 16 m row is **0.26 kg**
(6.16 - 5.89). No baseline gives 0.8. A plausible reading is that it predates
commit `3eeaecd` ("Compute the bag's field leak instead of assuming it"), which
resized the bag -- but the intended baseline cannot be recovered from here.

Either fix the reference or restore the item.

---

## Worklist status

All three items the paper deferred are now answered. **S1** (P5b): the far-node
delivery costs 113.20 km/s and reverses the partial split's verdict. **S2** (P3,
P4): the direct route can fly shallow, and the stated node survival is the larger
error. **S3** (P2): the second arrival is uncharged and worth under 1.2 percent.

`deferred_to_companion_repos.md`'s "What landed" section still reads "Nothing
yet" and can now be filled in from P2 to P5b.

## Still open

- The **partial split's pad ledger** is deliberately not reported. Now that its
  far node is priced the arithmetic could be done, but the launched-mass verdict
  (P5b) already settles the architecture, and a pad number would only restate it.
- The Jovian placement route of P2 has had **no real-ephemeris audit** (the ADR
  0011 treatment); it is a longitude and epoch match in a circular coplanar model.
- The far-node delivery of P5b is priced as a **direct high-speed launch from
  1 AU**. A feeder that dives first and is caught on its climb-out would be
  cheaper, but that is a second solar-dive chain rather than a delivery, and
  nothing has costed one.
