# New material for the paper: the fly-and-park architecture, and the launch cadence it buys

**Written to be copied into
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
and worked there**, on `templateArxiv.tex`. Self-contained: every number needed
to make every edit is stated here, because an agent in the paper repository
cannot run the code that produced them.

Generated 2026-09-07 in `katzseth22202/aim_is_all_you_need`, against
`sec:jupiter_only_growth`. Backing decision:
`docs/adr/0030-the-chain-search-was-reading-its-own-search-box.md`.

Same ground rules as `docs/paper_corrections.md`: **the paper is not the source
of truth**; locate claims by grepping their quoted wording, not by line number;
if a number here looks wrong, say so rather than working around it.

This document carries **one correction the paper owes** (C1) and **one new
subsection** (N1-N4). The correction is not optional: it changes a headline
figure by 3.1x and retires a claim.

---

## Summary

| | edit | what changes |
| --- | --- | --- |
| **C1** | wherever the phased chain's compounded mass is quoted | **x74.8 -> ~x233** and 8 cycles -> 9; the "powered flyby adds a 9th cycle, +20%" claim is **retired** |
| **N1** | new subsection under `sec:jupiter_only_growth` | the **fly-and-park** cycle: fly short and hot, park to rephase |
| **N2** | same subsection | the **exchange rate** that decides whether it pays, and why methalox misses by 5% |
| **N3** | same subsection | the **launch-cadence** consequence: one narrow window per 1.09 yr becomes continuous |
| **N4** | same subsection | what is **not** claimed, stated explicitly |
| **T1** | glossary / first use in `sec:jupiter_only_growth` | **four terms that must be defined in the paper before any of the above is written** |

---

## T1. Terminology the paper must define before using any of this

**Do this first.** Every claim in N1-N3 is stated in these four terms, and three
of them do not currently appear in the paper at all. Define them at first use in
`sec:jupiter_only_growth`, or add them to the glossary. Suggested wording, in
plain language, is given for each.

### Departure phase

> **Departure phase.** Where Earth and Jupiter stand relative to one another at
> the moment the payload leaves Earth, written as a fraction of one Earth-Jupiter
> synodic period (1.0923 years). Phase 0 and phase 1 are the same geometry, so
> the phase runs around a circle and comes back.

This is the independent variable of every table in N2 and N3. Note it is a
*relative* longitude, not a date: the model is circular, coplanar and on a
relative epoch, never a calendar.

### Sweet phase

> **Sweet phase.** The departure phase at which reaching Jupiter is cheapest.
> Because a transfer must arrive where Jupiter actually is, the departure burn
> needed to get there depends strongly on the phase you leave on: across the
> circle the cheapest available burn runs from **4.41 km/s at the sweet phase to
> 38.49 km/s at the worst**, a factor of 8.7. A departure stage that cannot
> afford the expensive phases is therefore pinned to the sweet one -- and being
> pinned to one phase is what forces the loop onto a whole number of synodic
> periods, so that each cycle returns to the phase it can afford.

**This is the single most important definition in the document**, because it
supplies the *mechanism* for the paper's existing three-synodic result. The paper
currently reports the 3S clock; it does not say why. The answer is that the clock
is not chosen, it is the consequence of an 8.7-fold swing in the price of
reaching Jupiter. Computed by `sweet_phase()` in `src/fly_and_park.py`, which
locates it at phase 0.726 rather than assuming it.

### Usable phase

> **Usable phase.** A departure phase from which at least one closing trajectory
> actually *grows* the payload -- the arriving impactor mints more mass than the
> departure burn spent. A phase can be perfectly reachable and still be unusable,
> if every trajectory leaving it costs more than it returns.

The distinction matters because N3's headline is a count of usable phases, not
of reachable ones. Reachability never changes with exhaust speed; usability does.

### Fly-and-park

> **Fly-and-park.** Flying a trajectory *shorter* than the three-synodic window
> and parking in the bound near-escape orbit for the remainder, so that flight
> plus park is exactly 3.00 synodic periods and the next departure falls on the
> same phase. The clock is preserved exactly; nothing drifts.

Note for the writer: the parking orbit is not new hardware. It is the same bound
near-escape orbit the architecture already uses as its phasing buffer and aim
reversal; fly-and-park only lengthens it, from 20 days to about 100.

---

## C1. The phased-chain figures, and the powered-flyby claim

**Correct wherever it appears.** The 30-year compounded launched mass of the
phased Jupiter-only chain is **~x233 across 9 cycles**, not x74.8 across 8. Quote
it as `~x233` with about +/-3%; do not quote four digits.

**Retire entirely** any sentence resembling *"the perijove burn acts as a second
steering knob, tightening return timing enough to fit a ninth cycle, worth about
+20% compounded mass."* At converged search settings the optimizer, handed a free
perijove burn, drives **all nine to exactly 0.0** and reproduces the unpowered
chain bit-for-bit. The 9th cycle is present in both runs.

**Why it moved.** Two coupled search-box constants in
`src/jovian_cycle_phasing.py` were set too coarse, and neither converges alone.
At the old 26-sample outbound grid each generation offered fewer than 48
candidate states, so the 48-wide beam never cut -- widening it to 128 or 320
returned x74.795 unchanged, which made the beam look innocent. Refining the grid
raises the candidate count to 94-403 per generation, at which point the 48-wide
beam cuts in *every* generation. Both had to move (26 -> 200 samples, 48 -> 200
beam).

**This is good news for the paper's own argument.** `sec:jupiter_only_growth`
elsewhere rests on the flyby being unpowered. That claim now has no
counterexample anywhere in the repository.

**Do not change** the self-sustaining verdict, the "mass cannot wait"
constraint, the 20-day coast, or the circular-coplanar caveat. All stand.

---

## N1. The fly-and-park cycle

**Proposed new subsection under `sec:jupiter_only_growth`.** Draft text follows;
adapt freely, but keep the numbers and keep N4.

The growth loop must return the payload to Earth on an Earth-Jupiter relative
phase from which the next departure is affordable. The cost of reaching Jupiter
swings enormously with that phase: across one synodic period the cheapest
available departure burn runs from **4.4 km/s to 38.5 km/s**. A departure stage
that cannot afford the expensive phases is pinned to the cheap one, and the
cycle is then forced onto an integer number of synodic periods so the geometry
repeats. The converged chain does exactly this without being told to: its cycle
lengths are 3.01, 2.99, 3.01, 2.99, 3.01, 3.01, 2.98, 3.01 synodic periods --
**eight consecutive cycles at 3.00 S**, with nothing in the model aware of what a
synodic period is.

That 3S clock is not the constraint it appears to be. **The payload need not
spend the whole window flying.** It may instead fly a shorter, hotter trajectory
and park in its bound Earth orbit for the remainder, so that the *total* is still
exactly 3.00 S and the next departure falls on the same phase. Nothing drifts;
the clock is preserved exactly.

Because every candidate is padded to the same 3.00 S, **cycle time cancels from
the comparison entirely**, and the choice reduces to a single question: is the
faster impactor worth the larger departure burn that buys it?

Concretely, at the phase where the loop runs best:

| | outbound | return | park | departure dv | Earth `v_inf` | arrival `v_b` |
|---|---:|---:|---:|---:|---:|---:|
| exact 3S, repeats | 1.11 yr | 2.08 yr | -- | **5.56 km/s** | 12.30 | 61.3 km/s |
| fly-and-park | 0.98 yr | 2.01 yr | 0.26 S | **6.74 km/s** | 13.84 | **68.7 km/s** |

The parked leg is 0.26 synodic periods, about **104 days**, in the same bound
near-escape orbit the architecture already uses as its phasing buffer and aim
reversal. Lengthening that orbit from 20 days to ~100 raises the push target
`v_rf` from 10.9503 to about 10.99 km/s, worth 0.2% on the mass ratio.

**The arrival speed this reaches, 67-69 km/s, is the perfect-retrograde
boundary** -- the minimum-energy purely tangential retrograde arrival at 1 AU,
which the paper's own catalog rows already assume at 69.27 km/s. It is reached
here with a **zero perijove burn**, and without touching the synodic clock.

---

## N2. The exchange rate, and why methalox misses

Padding to a common cycle length reduces the trade to one number: how much extra
departure burn a hotter arrival is worth before the propellant cost cancels the
gain. Against a 53.5 km/s baseline and the cycle orbit's `v_rf` = 10.9503 km/s:

| arrival `v_b` | mass-ratio gain | **Isp 380 s** | **Isp 1200 s** | **Isp 2214 s** |
|---:|---:|---:|---:|---:|
| 56 km/s | x1.053 | 0.19 | 0.60 | 1.11 |
| **60** | x1.136 | **0.48** | **1.51** | **2.78** |
| 64 | x1.220 | 0.74 | 2.34 | 4.32 |
| **68** | x1.304 | **0.99** | **3.13** | **5.77** |

*(km/s of extra departure burn that exactly cancels the gain.)*

The manoeuvre actually costs **+1.04 km/s** (median across viable phases; range
0.00 to 1.98). Set that against the budget:

| departure stage | budget at `v_b` 68 | cost | verdict |
|---|---:|---:|---|
| **methalox, Isp 380 s** | 0.99 km/s | +1.04 | **fails -- about 5% short** |
| **impactor-driven, Isp 1200 s** | 3.13 km/s | +1.04 | **works, 3x headroom** |
| **impactor-driven, Isp 2214 s** | 5.77 km/s | +1.04 | **works, 5.5x headroom** |

**This is the load-bearing sentence of the whole subsection.** Fly-and-park is
not an idea that methalox nearly supports and richer chemistry would improve; it
is an idea that **methalox misses by about 5%** and that any impactor-driven
departure clears comfortably. Measured across departure phases, fly-and-park
beats the plain 3S cycle at **1 phase of 11** on methalox, **17 of 30** at Isp
1200, and **25 of 30** at Isp 2214.

The Isp 2214 s figure is not optimistic: it is this repository's own
departure-nozzle ledger for the 3S departure at jet efficiency `eta_jet^2` = 0.6.
Isp 1200 s is deliberately conservative against it.

**Size of the prize, stated honestly.** The gain is **5% at the phase where the
loop already runs best**, rising to **27% well off that phase**. It is therefore
a *robustness* mechanism -- it flattens the penalty for departing away from the
ideal phase -- and not a large growth win. Do not quote the 27% without the 5%.

---

## N3. The launch-cadence consequence

This is the operationally interesting result, and it is a statement about the
departure stage's exhaust speed rather than about any trajectory.

For each departure phase across one synodic period, ask whether *any* closing
cycle from that phase grows the payload. The fraction that do:

| departure Isp (s) | 380 | 700 | 1000 | 1200 | 1500 | 1800 | **1900** | 2214 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| effective `v_e` (km/s) | 3.73 | 6.86 | 9.81 | 11.77 | 14.71 | 17.65 | 18.63 | 21.71 |
| **usable departure phases** | **18%** | 36% | 49% | 60% | 77% | 95% | **100%** | **100%** |

**Coverage reaches 100% at Isp 1900 s**, below the 2214 s the nozzle ledger
already claims.

The same result as a launch window. Staggering the departure about a nominal
epoch, and reporting the best growth rate available:

| offset | -96 d | -48 d | -24 d | 0 | +24 d | +48 d | +96 d |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Isp 380** | none | none | 0.127 | **0.205** | 0.073 | none | none |
| **Isp 2214** | 0.537 | 0.684 | 0.750 | **0.796** | 0.797 | 0.721 | 0.429 |

A methalox departure has a usable window of roughly **-36 to +24 days out of a
399-day synodic period**, and loses 65% of its growth rate across it. An
impactor-driven departure has a window spanning at least **+/-96 days**, flat to
within 1% over +/-24 days.

**The claim for the paper:** the architecture's launch schedule changes character.
A methalox departure admits **one ~60-day window every 1.09 years** and requires
everything to go at once. An impactor-driven departure at Isp >= 1900 s admits
**every phase**, so departures may be spread continuously. A 72-hour launch
stagger is negligible in either case; what changes is the existence of a window
at all.

**Where the gain comes from -- do not misattribute it.** At the nominal phase,
per-cycle growth goes from 1.65 to 6.13 between Isp 380 and 2214. Decomposed, the
faster impactor contributes **x1.18** (mass ratio 6.98 -> 8.27 as `v_b` goes
53.5 -> 62.4) and the departure burn's surviving mass fraction contributes
**x3.14** (0.236 -> 0.741). **The exhaust speed does about 85% of the work and
the hotter impactor about 15%.** The headline is not "faster impactors carry more
kinetic energy" -- and note the impulse law is *linear* in closing speed, not
quadratic, so a reader expecting `(68/55)^2` will be disappointed.

---

## N4. What is not claimed

Include this, or the paper will overreach on three points.

1. **This is a departure-opportunity result, not a pad-throughput result.** The
   model compounds mass in space and represents no ground resupply whatsoever.
   "18% -> 100% of phases" is a statement about when the vehicle may leave Earth
   orbit. Whether that relieves ground-launch congestion depends on how the fleet
   is supplied, which is not modelled anywhere in this repository.

2. **Fly-and-park is a single-cycle comparison, not a chain run.** Every
   candidate is padded to the same 3.00 S and returns to the same departure
   phase, so it should chain trivially -- but that has not been run. The backing
   ADR records one case in the same analysis where a single-cycle optimum did
   *not* survive the chain lookahead, so this caveat is not boilerplate.

3. **The higher exhaust speed does not buy a shorter clock.** Running the full
   30-year chain at Isp 2214 still yields ~3.00 S cycles holding phase. A 2.09 S
   cycle has the better instantaneous rate and a worse successor, and a chain
   that compounds mass declines it. What the exhaust speed buys is growth per
   cycle (1.83 -> ~6.3) and the window coverage above -- not a faster loop.

**One open item worth flagging in the text.** Cycles within 1% of exactly 2.00 S
exist (20 of 3,483 enumerated), reaching per-cycle growth 6.021 at 15 of 73
phases. An exactly-2.00 S cycle returns to its own departure phase, so if one
could be *held* it would compound at 0.810 per year against 3S's 0.576 -- roughly
1000x over 30 years. The chain search never sustains one, and reaches only x2.5e6
against the x3.2e7 a repeatable 3.00 S cycle would give, so it is not finding the
best available fixed point either. Whether a true 2S fixed point does not exist,
or the beam cannot hold one, is unresolved and is the largest open question this
analysis raised.

---

## Reproduction

All figures come from `src/jovian_cycle_phasing.py` at the converged settings
committed alongside this document (`_OUTBOUND_TOF_SAMPLES` = 200, `_BEAM_WIDTH`
= 200, `_STATE_BUCKET_DAYS` = 7.0, `_SEED_SAMPLES` = 16, `_OUTBOUND_TOF_MIN` =
1.1 yr except where the phase sweeps free it to 0.70).

The chain figures are printed by `make run` and pinned by the slow tests
`test_the_perijove_burn_converges_to_zero_and_buys_nothing` and
`test_the_converged_chain_settles_onto_the_synodic_clock`.

**The phase sweeps, the exchange rate, and the fly-and-park comparison are
committed as `src/fly_and_park.py`.** Reproduce every table in T1, N1, N2 and N3
with:

```bash
make fly-park
```

It enumerates `_cycle_branches` across 73 departure phases spanning one synodic
period (3,483 closing cycles) and re-scores each with
`M(v_b) * exp(-dv/v_e)`, where `M(v_b) = 2f/ln(v_b/(v_b - v_rf))`, `f` = 0.8 and
`v_rf` = 10.9503 km/s -- re-scored rather than taken from the chain, because the
chain charges methalox and the whole subject here is what happens when it does
not. The search box is recorded in the module docstring. Runtime ~8 s.

Pinned by `tests/test_fly_and_park.py`: the sweet phase and its 8.7x swing, the
usable-phase fractions and their monotonicity in Isp, the methalox-loses /
impactor-wins verdict, the 5%-at-sweet-phase / 27%-off-phase shape, and that
flight plus park is exactly 3.00 S. Run with `pytest tests/test_fly_and_park.py`;
the sweep-backed cases are marked `slow`.
