# The Jovian flyby stays unpowered, and Earth-return phasing is free

Status: accepted — decision 5's trigger has fired; see ADR 0007

Date: 2026-07-15

> **Note (2026-07-15).** Both decisions here **stand**: the Jovian flyby stays
> unpowered, and *return* phasing is free. This ADR drew the right boundary —
> "the remaining phasing question is the inner ladder, not the return" — and ADR
> 0007 answered exactly that question, leaving the return untouched. Two updates:
> (1) the figure "the chain ... reaches end-to-end 5.72, versus the powered
> flyby's 2.02" is stale (now 1.6730 vs 2.0169); it was one of four independent
> lines for the unpowered flyby and **the other three are unaffected**. (2)
> Decision 5 dropped the solar-electric budget with the trigger "Revisit only if
> *ladder* phasing proves to need it." **It does.** Ladder phasing costs 4.6 km/s,
> methalox cannot pay it, and SEP is now the only live path — making this ADR's
> decision 5 the most consequential open item in the program. One input moved in
> SEP's favour: the phased ladder spends 5.632 yr in the inner system, not the
> 1.04 yr assumed when estimating SEP at 1.3-2.0 km/s.

## Context

Two questions were open against the Jupiter-only growth leg, and they turned
out to be the same question.

**Does the Jovian flyby need a burn?** ADR 0002 built the powered flyby with an
impulsive periapsis burn as a free knob, and found the end-to-end optimum drives
it to zero. That could have been an artifact of the objective.

**Will Earth be where the return crosses 1 AU?** Neither the powered flyby (ADR
0002, which explicitly deferred "real phased Earth intercept") nor the assist
chain (ADR 0003) models phasing. `CONTEXT.md` is blunt that crossing 1 AU is not
reaching Earth. The chain charges a 300 m/s deep-space-maneuver budget as a
stand-in, and ADR 0005 flagged the whole timing story as synodic scaffolding
awaiting a Lambert study.

The connection is that both are answered by the *same* degree of freedom: the
unpowered bend at Jupiter. Investigating whether a small burn (~250 m/s), or a
solar-electric budget spent on the way out (500-1500 m/s), would buy phasing
flexibility revealed that the bend already buys more than either, for free.

### What the bend actually does

An unpowered flyby cannot change the magnitude of the Jupiter-relative excess
velocity (Tisserand). The chain arrives with `v_inf` = 15.369 km/s, and the
post-flyby heliocentric state is therefore confined to a circle of that radius
centred on Jupiter's velocity (13.058 km/s). Periapsis is not a second knob:
`e = 1 + r_p v_inf^2 / mu_J` and `sin(delta/2) = 1/e` make periapsis *be* the
bend, and the periapsis floor (ADR 0002's 4000 km perijove) caps the bend at
122.5 deg.

Walking the bend across that limit traces **one connected curve**:

| bend  | `v_b` (km/s) | return TOF (yr) | `v_r` (km/s)              |
|-------|--------------|-----------------|---------------------------|
| 36.1  | 50.12        | 4.373           | +8.10 (outbound branch)   |
| 67.7  | **56.27**    | 2.067           | +0.06 (**full reversal**) |
| 99.3  | 50.28        | 1.257           | -8.01 (inbound branch)    |

The mechanism at the ends is a reflection: flipping the sign of `v_r` changes
neither `|v|` nor `v_t`, hence neither the orbit's energy nor its angular
momentum. The return orbit, its perihelion and its `v_b` are *identical* --
only where Jupiter sits on it moves. Inbound the craft is already falling and
descends directly; outbound it is still climbing, so it coasts to aphelion and
falls back. At `v_t` = 0 the orbit is purely radial (a = 3.22 AU, apoapsis
6.44 AU), and Kepler's equation puts the climb at 1.561 yr -- so the detour
costs 2 x 1.561 = 3.12 yr, matching the measured 4.373 - 1.251 gap exactly.

That span is the phasing knob, and it is enormous: **1130 days = 1113 deg of
Earth phase, 3.09 wraps.** Under ADR 0002's 7 yr cap it is 3.09 wraps; under a
5 yr cap, 1.58. Above one wrap, every launch phase admits a phased intercept.

### The trap this nearly fell into

The span was first measured while holding `v_b` within +/-1 km/s of the chain's,
on the reasoning that the branch flip is `v_b`-neutral. That produced two
disjoint clusters (34 d and 228 d) separated by an 821-day hole, and a "spread"
of 1111 d that was **max-minus-min computed across the hole** -- counting
unreachable time as authority.

Both halves of that were wrong. The band was the error: the bend is *one* knob
with *two* outputs, so `v_b` cannot be held while TOF is swept. Holding `v_b`
selects only the two ends of the curve, because the middle rides up to 56.27.
The clusters were manufactured by the constraint, not by the physics.

## Decision

1. **The Jovian flyby stays unpowered.** Four independent lines agree:
   - The ADR 0002 optimizer drives the flyby burn to 3.4e-11 km/s and holds it
     at zero across the whole trade curve up to `v_b` = 65 km/s.
   - The chain is strictly unpowered and reaches end-to-end 5.72, versus the
     powered flyby's 2.02.
   - Buying the catalog's 69.27 km/s needs `v_inf_out` = 20.47 against 15.369
     available: a 2.62 km/s periapsis burn (v_p 33.58 -> 36.20), which halves
     delivered mass and drops the chain's end-to-end from **5.72 to 3.93**.
     Strictly worse than doing nothing.
   - A small burn (~250 m/s, Oberth-amplified 2.18x to 545 m/s of `v_inf`)
     extends the span from 1130 to ~1400 days -- 22% added to a budget already
     3x oversized.

2. **Earth-return phasing is free; record it as such.** No propellant, no
   powered flyby, no solar-electric budget. `jovian_return_phasing_envelope()`
   computes the reachable span from the chain's own arrival geometry, and
   `main.py` prints it under `sec:jupiter_only_growth`.

3. **Always report `largest_gap` beside the coverage.** The span is authority
   only if connected. The function reports the widest step between sampled
   arrivals, which is resolution-limited (11.9 d at 968 samples, 0.7 d at
   16000, falling as 1/samples). A genuine hole would refuse to shrink. A test
   pins the shrinkage, because that -- not the span -- is the evidence.

4. **State the price honestly: phasing dictates `v_b`, it does not free it.**
   Phasing picks the bend and the bend picks `v_b` somewhere in 50.14-56.27.
   That is a lottery, not a choice, and it is benign *only* because every value
   in the band is an acceptable loop (end-to-end 5.7-6.3, doubling 1.37-1.51
   yr). Quote the band, never just the best case.

5. **Drop the solar-electric phasing budget from consideration.** A 100 kW argon
   Hall thruster on a 100 t craft over the chain's 1.04 yr inside 1 AU buys
   1.3-2.0 km/s (efficiency 0.40-0.60), and the array that powers it is the same
   one that gives ~3.7 kW at Jupiter for thermal survival -- so the concept is
   self-consistent. It is simply not needed: it addresses return phasing, which
   is free. It was not modelled.

6. **Flag for the paper** under `sec:jupiter_only_growth` (PLANNED, like the
   rest of the chain work). This retires "will Earth be there?" for the return
   leg, and the no-powered-flyby result simplifies the architecture.

## Considered options

- **Powered Jovian flyby for phasing (~250 m/s) -- rejected.** Works
  (2.18x Oberth amplification) but adds 22% to an already-3x-oversized span.
  Solves a solved problem.
- **Powered Jovian flyby for `v_b` (2.62 km/s to reach 69.27) -- rejected.**
  Halves delivered mass for end-to-end 3.93 vs 5.72. Worse loop. Separately,
  56 km/s already *loses* to 51 on doubling time (1.51 vs 1.37 yr), so the
  whole direction is wrong even when the propellant is free.
- **Solar-electric phasing budget (500-1500 m/s) -- rejected, not modelled.**
  See decision 5. Would have been ~1/5 the mass of methalox per m/s and worth
  +6.7% end-to-end if the 300 m/s budget were replaced -- a real but modest
  gain against return phasing being free. Revisit only if *ladder* phasing
  proves to need it.
- **Lowering the perijove floor -- rejected as a non-lever.** 4000 km -> 200 km
  raises the bend limit 122.5 -> 123.7 deg (+1.2 deg), consistent with ADR
  0002's ~40 m/s sensitivity. It lifts neither the span nor the `v_b` ceiling.

## Consequences

- **`v_b` = 56.27 km/s is a hard ceiling on the chain's return**, set by full
  reversal of the Tisserand-fixed excess. The catalog's three Jovian rows
  assume 69.27 km/s, which the chain **cannot reach at any phase, periapsis or
  arrival time**. This sharpens ADR 0002's existing discrepancy from "the
  optimizer prefers less" to "the chain is physically incapable of more."
  A test pins it.
- `AssistChainReturn` gains `jovian_arrival_direction`; `_ChainTerminal` gains
  `arrival_direction`. The envelope needs the incoming direction to enforce
  bend reachability, and deriving it beats assuming it.
- The envelope costs ~4 ms and rides on the chain result `main.py` already
  computes, so it is in the default `make run`.
- **The remaining phasing question is the inner ladder, not the return.**
  Venus and Earth must line up for E-V-E-V-V-E, which the bend (acting
  downstream of the whole ladder) cannot touch. That is what the 300 m/s budget
  is for, it remains unverified, and ADR 0005's Lambert-study flag stands.
- `_jovian_terminal` still keeps only the earliest-arriving return. The phasing
  authority is computed and discarded there; using it for a real launch date
  means replacing that tiebreak with a phasing residual. Out of scope until an
  ephemeris makes "the right date" meaningful.
