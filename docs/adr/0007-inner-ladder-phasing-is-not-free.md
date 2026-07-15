# Inner-ladder phasing is not free: it costs ~4.6 km/s, it inverts the pump ladder, and the chain's fate now hinges on the SEP thrust rate

Status: accepted

Date: 2026-07-15

Supersedes in part: ADR 0003 (headline and mass accounting), ADR 0004 (sequence
ranking), ADR 0005 (growth arithmetic). Their reasoning stands *given the model
each assumed*; this ADR replaces the numbers those models produced. See "What
this retires".

## Context

ADR 0003 built the assist chain on an explicit stand-in: the model is
phasing-free (coplanar circular planets, "each body wherever the trajectory
needs it"), so a fixed 300 m/s deep-space-maneuver reserve
(`ASSIST_CHAIN_PHASING_BUDGET`) was charged as spent methalox to keep the mass
accounting from being "misleadingly rosy". ADR 0003 was candid that this was
"an *estimate charged as propellant*, not a phasing solution". ADR 0005 flagged
the Lambert-arc study as "the natural next analysis layer". ADR 0006 narrowed
the open question to exactly one thing:

> **The remaining phasing question is the inner ladder, not the return.** Venus
> and Earth must line up for E-V-E-V-V-E, which the bend (acting downstream of
> the whole ladder) cannot touch. That is what the 300 m/s budget is for, it
> remains unverified.

`_phased_ladder_burn()` is that study. It puts the planets where they actually
are, makes each leg a Lambert arc between true positions, and charges each node
`_flyby_mismatch_burn()` for whatever the unpowered flyby cannot supply.

**The stand-in was wrong by 15x, and it was hiding a second, larger error: the
sequence ranking itself was an artifact of the phasing-free model.**

## What was measured

Global search (pygmo `sade`, archipelago) over leg times, holding the return to
the powered-flyby optimum's collision speed (`v_b >= 51.134 km/s`) exactly as
ADR 0003 does. **Each sequence is optimized at its own best epoch**, swept
across a full Earth-Venus synodic — this matters, see the trap below.

| sequence | legs | epoch | departure | nodes | total dv | end-to-end |
|---|---|---|---|---|---|---|
| **E-V-E-J** (minimal) | 3 | 0.80 | **0.9335** | 3.6342 | **4.5677** | **1.9734** |
| E-V-E-V-V-E-J (ADR 0003) | 6 | 0.40 | 2.6164 | 2.5666 | 5.1830 | 1.6730 |
| E-V-V-E-J (Cassini VVEJGA) | 4 | 0.80 | 0.959 | 4.698 | 5.656 | 1.4735 |
| E-V-E-V-E-J | 5 | 0.30 | 4.338 | 2.585 | 6.923 | 1.0490 |
| E-V-E-V-V-E-V-J | 7 | 0.10 | 5.601 | 4.469 | 10.070 | 0.4508 |
| E-V-E-E-J (Galileo VEEGA) | 4 | 0.20 | 9.294 | 1.949 | 11.243 | 0.3290 |

Reference: the powered flyby scores **2.0169** end-to-end; chain break-even is a
total of **4.4865 km/s**.

The best phased chain is **E-V-E-J at 4.5677 km/s**, converged (identical to four
decimals at 8x40x300x3, 16x64x500x5 and 32x64x700x6; legs 0.414/0.866/1.370 yr =
2.65 yr; `v_b` = 51.1340 against a required 51.134, i.e. binding to the last
digit). It **misses break-even by 0.081 km/s** and loses to the powered flyby by
**2.2%** (1.9734 vs 2.0169).

### The trap: ranking sequences at one epoch

The incumbent ladder was first re-priced at epoch 0.40 and every alternative
scored against it *at that same epoch*, producing "E-V-E-V-V-E-J 5.183, all
others 9.2-20.7" — an apparent 2-4x reconfirmation of ADR 0004. That was an
artifact. 0.40 is the *incumbent's* best epoch; E-V-E-J's is 0.80 and Cassini's
is 0.80. At their own epochs the ordering **inverts**: the 3-leg minimal chain
(4.5677) beats the 6-leg pump ladder (5.1830). Sequence comparisons are only
meaningful with each sequence at its own optimum, because phasing couples
sequence to epoch. This is the phased analogue of ADR 0003's own beam-pruning
lesson: a constraint imposed for convenience manufactured the result.

### Why the pump ladder loses once phasing is real

ADR 0004's case for the ladder is that pumping power is counted by body
alternations, each V<->E hop converting misalignment into heliocentric energy
*for free*. That is true and unchanged. What changed is the bill for arranging
it. The ladder's departure burn rises to 2.6164 km/s, while the minimal E-V-E
departs for 0.9335 — because a 6-rung ladder must leave on an arc that arrives
when Venus is *actually there* for every subsequent rung, not on the cheapest
arc that touches Venus's orbit. The free pumping is real; the phasing needed to
line up six rungs costs more than the pumping saves. **The ladder is a liability
under phasing, and its length is the reason.**

## The SEP question, which now decides everything

ADR 0006 decision 5 dropped the solar-electric budget with an explicit trigger:
"Revisit only if *ladder* phasing proves to need it." It now does. Argon SEP
(`ARGON_SEP_ISP` 2000 s, `v_e` 19.613 km/s) against methalox (380 s, 3.7265)
prices node burns ~5.3x cheaper. Re-solving with departure on methalox and nodes
on SEP — **re-solving, not re-scoring**, since the optimizer then deliberately
shifts cost out of departure and into nodes:

| sequence | end-to-end (SEP) | nodes needed | SEP deliverable @1.92 km/s/yr | @1.25 km/s/yr |
|---|---|---|---|---|
| E-V-E-J | **4.7595** | 4.636 | SHORT 0.08 | SHORT 1.67 |
| E-V-V-E-J (Cassini) | **4.1313** | 4.899 | **OK, margin 2.28** | SHORT 0.22 |
| E-V-E-V-V-E-J (ADR 0003) | 2.9228 | 2.610 | **OK, margin 8.22** | **OK, margin 4.43** |
| E-V-E-V-E-J | 2.8563 | 7.892 | SHORT 1.05 | SHORT 3.44 |
| E-V-E-V-V-E-V-J | 2.3620 | 11.705 | SHORT 2.47 | SHORT 5.71 |
| E-V-E-E-J (Galileo) | 2.3580 | 18.393 | SHORT 12.84 | SHORT 14.78 |

Every sequence clears the powered flyby's 2.0169 on mass alone. But SEP is
thrust-limited, and ADR 0006's own rate (1.3-2.0 km/s per 1.04 yr inside 1 AU =
1.25-1.92 km/s per yr) decides which are deliverable — and it decides the
winner:

- **At the optimistic rate, Cassini's actual flown sequence wins** (4.1313, with
  2.28 km/s of margin), beating the incumbent by 42%.
- **At the pessimistic rate, only the incumbent long ladder survives at all.**
- **E-V-E-J fails at both** despite the best score: at 2.37 yr it is too short to
  give SEP time to deliver 4.636 km/s.

There is a genuine tension with an interior optimum: shorter sequences are
cheaper (less phasing to arrange) but starve SEP of thrusting time; longer ones
feed SEP but pay to line up. Cassini's VVEJGA sits at that sweet spot — simple
enough to be cheap, long enough to thrust — which is a striking, and suspicious,
place for the answer to land.

## Decision

1. **Record that inner-ladder phasing is not free, and is not 300 m/s.** The
   best phased chain costs 4.5677 km/s, 15x the stand-in. Quoting any chain mass
   number against 300 m/s is now an error.

2. **The all-methalox chain is not competitive, but only narrowly.** Best
   end-to-end 1.9734 vs the powered flyby's 2.0169 — a 2.2% loss, missing
   break-even by 0.081 km/s. This is a hair, not a rout; it should not be quoted
   as a decisive defeat.

3. **The V/E pump ladder does not survive phasing as the preferred sequence.**
   E-V-E-J beats it by 12% on total dv in methalox. ADR 0004's ranking is
   retired; its *physics* (Tisserand freeze, single-Venus-pump ceilings, bend
   margins) is untouched.

4. **The chain is conditional, not retired**, and the condition is now a single
   number: **the SEP thrust rate**. Above ~1.9 km/s/yr Cassini VVEJGA wins at
   4.13; below ~1.3 only the long ladder is deliverable at 2.92. Both beat the
   powered flyby. The architecture question has collapsed to a 1-D propulsion
   question, which is the most tractable it has ever been.

5. **State the SEP mass numbers as upper bounds, never as results.** They
   substitute a low-thrust `v_e` into an *impulsive* rocket equation: the node
   burns are impulsive Lambert mismatches, and a Hall thruster cannot deliver
   one. The time-budget filter above is a necessary condition, not a sufficient
   one. **The sequences that score best are exactly those leaning hardest on the
   invalid assumption** — E-V-E-J needs 4.636 km/s of it — so the ranking is
   least trustworthy where it is most favourable. A low-thrust re-solve is
   required before any of this is quotable.

6. **Flag for the paper.** `sec:jupiter_only_growth` is PLANNED and its chain
   numbers derive from the 300 m/s stand-in. Nothing from ADR 0003/0004/0005
   should reach the paper until the low-thrust study resolves; until then the
   powered flyby (ADR 0002) is the defensible architecture.

## What this retires

- **ADR 0003, headline**: "~0.29-0.30 km/s of departure burn replaces the
  powered flyby's 4.45 km/s" and "end-to-end ~5.7 versus the powered flyby's
  ~2.0". Phasing-free artifacts. Its *search* (beam, feasibility-witness,
  time-bucketed pruning) and its floor results (Venus-reach 279.4 m/s, Tisserand
  lock, square-root escape) are untouched and remain correct for that geometry.
- **ADR 0004, ranking**: "The V/E pump ladder stands; Cassini-style VVE only
  opens at ~1 km/s". Under phasing the ladder is beaten by minimal E-V-E in
  methalox, and by Cassini VVE under SEP. Crucially, ADR 0004's rejection of VVE
  rested on "this repo's chain model keeps flybys strictly unpowered, so VVE only
  closes once the departure burn alone carries the pump" — the phased model
  charges node burns *by construction*, so it contains exactly the ~470 m/s DSM
  that ADR 0004 correctly identified as what made the real Cassini work. The
  premise for excluding VVE is gone. Its Tisserand ceilings and bend-margin
  results stand.
- **ADR 0005, growth arithmetic**: "end-to-end ~5.7 per cycle, effective cycle
  3.46-5.06 yr, doubling ~1.4-2.0 yr, millionfold ~27-40 yr". Both inputs moved.
  Its *cadence* claim (windows recur at the ~1.60 yr Earth-Venus synodic) is
  untouched and independently corroborated: the epoch sweep recovers that
  structure, with cost falling again at 1.60-1.65.
- **ADR 0006, one stale figure**: "the chain ... reaches end-to-end 5.72, versus
  the powered flyby's 2.02". This was one of four independent lines for keeping
  the Jovian flyby unpowered; **the other three are untouched and the decision
  stands**. ADR 0006's core results — unpowered Jovian flyby, free *return*
  phasing — concern the return, which this ADR does not touch. ADR 0006 drew that
  boundary correctly and its decision 5 anticipated this ADR exactly.

## Consequences

- **`v_b` = 56.27 km/s remains the hard ceiling** (ADR 0006); this ADR changes
  the chain's cost and preferred sequence, not its reachable speed.
- The phasing-free model retains a role as a **lower bound** and fast feasibility
  filter. It is no longer admissible for mass accounting, for comparison against
  the powered flyby, or **for ranking sequences** — that last is the new lesson.
- `ASSIST_CHAIN_PHASING_BUDGET` should not be silently re-tuned to 4.6. The
  quantity it named — a small residual absorbed by a DSM — does not exist. The
  phased solve produces departure and node burns jointly and cannot be decomposed
  into "chain cost plus a phasing surcharge".
- **Sequence comparisons must optimize each candidate at its own epoch.** A
  shared epoch silently ranks by proximity to the incumbent's optimum. A test
  should pin E-V-E-J below E-V-E-V-V-E-J to keep this from regressing.
- **The open question is a low-thrust ladder solve**, which subsumes ADR 0005's
  Lambert flag. Impulsive-node phasing is now priced; methalox cannot pay it, and
  whether SEP can is a thrust-rate question, not a trajectory question.
- Reproducibility: pygmo `archipelago` scripts must guard module-level work behind
  `if __name__ == '__main__'` and cap `mp_island.init_pool(processes=6)`.
  Unguarded, workers re-import `__main__` and fork-bomb until the OOM killer
  fires; the runs behind this ADR were re-done after that failure. Use
  `archi.wait_check()`, never `archi.wait()` — the latter swallows island
  exceptions and returns an unevolved champion that looks like an answer.
