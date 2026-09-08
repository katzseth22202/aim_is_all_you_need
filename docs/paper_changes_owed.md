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

## P5b. State the partial split's condition, do not state a verdict

Backing: ADR 0025 second and third addenda. **Answers S1, and the answer is that
the question was underdetermined.**

`sec:split_dive_growth`'s held sentence says "better on both axes the ledger
scores", pending a price for delivering impactors to the far node. That price is
now in, and it comes out **two different ways depending on the route** -- by a
factor larger than the quantity being decided.

The trap is that the far node does not need mass at 1.9649 AU. It needs mass
**moving at 153.35 km/s** there, against a vehicle doing 13.45. A Hohmann
delivery arrives nearly co-moving and is worth nothing as an impactor.

| route | slug per kg placed | partial split | beats 2.3654? |
| :--- | ---: | ---: | :--- |
| push it out from 1 AU | **96.1** | 3.552 | no |
| **feeder dive to 4 R_sun** | **4.706** | **1.6345** | **yes** |

**Pushing.** Buying 140 km/s at 1.9649 AU from 1 AU takes 113.20 km/s of Earth
departure excess -- nearly six times the payload's own 19.12 -- and delivers
1.0 percent of what is launched. On that route the partial split loses.

**Feeding.** But the architecture never makes fast mass by pushing it. It drops
mass down the Sun's well and collects it on the climb-out, which is what the beam
*is*. A feeder launched onto the same 4 solar-radii dive the payload flies
arrives at the far node with **153.29 km/s** of closing speed against the 153.35
needed -- not a coincidence, it is the dive that makes the split's own beam. It
costs **twenty times less**, and on that route the partial split still wins.

**So the honest statement is conditional.** The partial split's case rests on a
far-node supply that is affordable only as a feeder dive. The feeder is one-way
and expendable, so unlike the payload's dive it carries no Earth re-intercept
condition -- it needs only its beam ray through the far node at the right epoch,
two conditions on at least two free knobs, which should give discrete solutions
by the same argument as every other closure here. **But no such closure has been
solved**, so the feeder is costed and not demonstrated.

**What the paper should do.** Keep the held sentence, and make its reason
explicit rather than either lifting or retiring it: the claim rests on a feeder
whose phasing is unsolved. `CONTEXT.md`'s **Partial split** and **Far-node
delivery price** entries in this repository say the same.

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

**Closed 2026-09-05.** The paper's own
`deferred_to_companion_repos.md` records P6 as a stale note rather than a paper
error -- `sec:axial_bag` already says 1.3 kg -- and the 0.8 was never a live
figure. Re-confirmed at the geometry ADR 0029 adopted: the flown column costs
6.162 - 4.844 = **1.32 kg** more film than the sphere, against 1.31 before, so
the printed 1.3 stands through the volume change as well.

---

# Fly-and-park batch, answered 2026-09-08

The paper's `docs/fly_and_park_asks_for_aim_repo.md` (items **S5-S8** of its
register) is answered in full. The handback written to be copied over is
`docs/paper_corrections_synodic_lock_2026-09-08.md`; the backing decision is
`docs/adr/0031-the-synodic-lock-is-the-resonance-we-already-had.md`. Reproduce
with `make fly-park` and `make two-wave`.

## P7. The 2S synodic lock needs the coast, and then it is the resonance (S5)

**The nine-day park is inadmissible**, and the paper asked the question that
settles it. The park *is* the bound near-escape orbit's period, lengthened: the
returning wave pushes the payload up at periapsis and the departure burn lights
at the next, which is the aim reversal. One orbit of a 20-day orbit is not
optional, and `optimize_jovian_cycle_chain()` already encodes it.

**Our constant was wrong and the paper's construction found it.** `MINIMUM_PARK`
was 0.02 S, about eight days, 2.5x under the coast; it is now `COAST_SYNODICS`.

**Charge the coast and the 2.00 S lock is ADR 0011's two-synodic resonance** --
phase 0.8082, flight 1.9497 S, park 20.06 d, `dv` 8.613, `v_b` 63.35 -- which
this repository and the paper's own N4 already publish. Its park is the coast
plus ninety minutes, so fly-and-park has nothing to add to it. The quotable
doubling is **0.873 yr** at Isp 2214 and **1.082** at 1200, +3.5% and +8.1% on
the held figures, over 26 and 11 of 73 phases rather than 28 and 13.

**The chain check is run** (`sustainable_chain_optimum()`), in a new chain,
because the beam search pins departure to arrival plus a fixed coast and so
structurally cannot park. The best repeating policy is a single self-looping
cycle at every exhaust speed -- 3.00 S at methalox (3.641 yr), 2.00 S at Isp 1200
and 2214 (1.082 and 0.873) -- so ADR 0030's N4.2 lookahead caveat is discharged
for this result. The methalox branch dedup was audited and changes nothing.

## P8. The doubling ladder's rungs are not at matched efficiency (S6)

**A different measurement of a different thing, not a refinement.** The paper's
"1.74 yr at `f` = 0.6 and 1.45 at `f` = 0.8" quotes the **nozzle impulse
recovery** `e`, which derates the whole impulse from outside the momentum debit.
The elasticity `f` is `STD_FUDGE_FACTOR` and is 0.8 in *both* rows. Fly-and-park
is at `f` = 0.8 and no `e` at all. `sec:mass_interest`'s "nozzle geometric
efficiency of 0.6" is the same `e` under a third name.

Scope and accounting differ too: circular against real ephemeris, one padded
cycle against eleven flown ones, and -- the load-bearing one -- fly-and-park
**does not charge the projectile stream** the two-wave ledger sources by
splitting 19% of the batch onto the nozzle bend. So the new rungs are
systematically optimistic against 1.45 yr. `doubling_ladder()` emits every rung
carrying its scorer, model, scope and efficiencies.

## P9. The launch window is one arc, and it is pinned (S7)

Reproduced exactly and moved inside `fly_and_park.py`
(`usable_phase_windows()`), so the paper cites `make fly-park`: one contiguous
arc at every exhaust speed, 13/26/36/44/56/69/73 of 73 usable phases and
71/142/197/240/306/377/399 days wide. Measured cyclically, or the Isp 1800 row
wraps through phase 0 and reports two.

## P10. The flown 2S cycles are already exact locks (S8)

**Yes, and more strongly than asked.** The adaptive cadence builds every return
on an exact synodic multiple, so all eleven flown cycles drift under 1e-4 S --
there is no remainder to pad. `TwoWaveCycle` now reports `period_synodics` and
`phase_drift`, and `make two-wave` prints the column. The circular lock's 63.35
km/s arrival lands inside the flown family's 61.83-65.13, and its 8.613 km/s burn
sits about 20% above the dearest of 6.84-7.17. So "lock a cycle we already fly"
is correct.

**But it retires the wrong problem.** What forces the four 3S fallbacks is not
phase drift, it is ADR 0011's perijove floor -- 45 of 91 windows over 200 years.
The honest cadence claim stays 2S with a 3S fallback about half the time.

---

## Worklist status

All three items the paper deferred are now answered. **S1** (P5b): the far-node
supply costs 96.1 kg/kg pushed or 4.706 fed, and the partial split loses on the
first and wins on the second, so the claim is conditional on a feeder whose
phasing is unsolved. **S2** (P3,
P4): the direct route can fly shallow, and the stated node survival is the larger
error. **S3** (P2): the second arrival is uncharged and worth under 1.2 percent.

`deferred_to_companion_repos.md`'s "What landed" section still reads "Nothing
yet" and can now be filled in from P2 to P5b.

**Fly-and-park batch (S5-S8), 2026-09-08.** All four answered; see P7-P10 above.
Three of the paper's holds lift, one with a corrected number: the 2S lock quotes
**0.873 yr** rather than 0.84, the launch-window layout is pinned, and S8's
"lock a cycle we already fly" is confirmed. The S6 hold lifts only into a
*labelled* ladder -- the new rungs must not sit unlabelled beside 1.45 yr.

## Still open

- The **partial split's pad ledger** is deliberately not reported. Now that its
  far node is priced the arithmetic could be done, but the launched-mass verdict
  (P5b) already settles the architecture, and a pad number would only restate it.
- The Jovian placement route of P2 has had **no real-ephemeris audit** (the ADR
  0011 treatment); it is a longitude and epoch match in a circular coplanar model.
- **The mapping between the two-wave chain's recovery `e` and fly-and-park's
  departure Isp is unworked**, so the doubling ladder's rungs can be named but
  not reconciled. Pricing the fly-and-park locks through the two-wave nozzle
  ledger would need a split geometry the circular phased model does not have.
- **The parking orbit's period is treated as fixed at 20 days.** A shorter cycle
  orbit would make a shorter park legal, but it moves `v_rf`, the mass ratio,
  the aim reversal and the split gap together, and nothing prices that.
- **The feeder closure of P5b is unsolved.** Its beam ray must pass through the
  far node at the right epoch: two conditions on at least two knobs, so discrete
  solutions are expected, but none has been found. This is the single item on
  which the partial split's whole case now rests.
