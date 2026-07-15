# The growth loop is scored on doubling time, which retires VEEGA and VVEJGA unconditionally — but not the minimal chain

Status: accepted

Date: 2026-07-15

Supersedes in part: ADR 0007 (objective and sequence ranking). Its phasing
physics and its SEP caveats stand and are load-bearing here. ADR 0002's
end-to-end mass ratio is retired *as the growth-loop objective* and kept as what
it always was: a per-cycle quantity.

## Context

Every prior ADR scored the chain in mass per cycle — ADR 0002's end-to-end mass
ratio, ADR 0003/0007's total delta-v. That objective is blind to the one thing an
exponential launch loop is actually paid in: **payload per year**. A cycle that
delivers 5.7x per pass is worse than one delivering 2.0x if it takes three times
as long to come round.

The grill's question was whether Galileo's VEEGA (E-V-E-E-J) and Cassini's
VVEJGA (E-V-V-E-J) — sequences with real flight heritage — beat a direct powered
Jovian flyby (ADR 0002). The granted terms were generous to the chain: powered
gravity assists at every flyby including Jupiter, deep-space maneuvers for
alignment, free choice of epoch over a Jovian year, and departure from C3 = 0 at
200 km so the Oberth effect is available.

**The cycle is closed.** The returning PuffSat's collision pushes the mass to
just under Earth escape — a 20-day-period orbit with 200 km periapsis
(`PUFFSAT_CYCLE_ORBIT_PERIOD`) — which falls back to periapsis and departs from
there. So `v_rf` (the collision push target) and the departure burn's starting
speed are **the same number**, 10.9503 km/s. That is what makes a per-year score
well-defined: there is a cycle to divide by.

## Decision

1. **Score the growth loop on doubling time: `cycle x ln2 / ln(net growth)`.**
   Net growth is `M(v_b) x exp(-dv/v_e)` at 380 s methalox, and `cycle` is
   departure-to-departure, not the trip.

2. **Optimize the growth *rate*, `ln(growth)/cycle`, and report doubling time.**
   Doubling has a pole: it diverges as growth -> 1+, so a minimizer sees an
   infinitely tall spike at the edge of the feasible set with a penalty plateau
   beyond. The rate passes smoothly through zero and goes negative for a
   shrinking cycle, so the search can walk in from the losing side. This is not a
   cosmetic reformulation — see "How this was nearly recorded backwards".

3. **Galileo VEEGA and Cassini VVEJGA are retired, and the retirement is
   unconditional** — it survives free node propulsion (below).

4. **The minimal chain E-V-E-J is NOT retired.** It remains conditional on SEP,
   exactly the open question ADR 0007 decision 4 left, unchanged by the new
   objective.

## What was measured

Jupiter is now a ladder body (`_jupiter_assist_body`), so the Jovian leg is a
Lambert arc onto Jupiter's **true position**, and Jupiter's longitude at t=0 is a
free variable — that *is* the granted "any window over a Jovian year". Epoch
spans an Earth-Venus synodic and `jupiter_lon0` spans a full turn, together
covering the configuration torus rather than a curve across it.

Bar — the direct powered flyby (ADR 0002), re-scored on doubling time, `v_b`
free:

| | departure | Jupiter burn | total dv | `v_b` | growth | trip | cycle | **doubling** |
|---|---|---|---|---|---|---|---|---|
| direct powered flyby | 4.5435 | 0.0000 | **4.5435** | 51.46 | 1.9757 | 3.221 | 3.276 | **3.3347 yr** |

Chains, Jupiter phased, best of 3 seeds x 2 bend signs (16x64x700):

| sequence | total dv | `v_b` | growth | trip | cycle | **doubling** | spread |
|---|---|---|---|---|---|---|---|
| E-V-E-J | 3.047 | 49.48 | 2.8239 | 5.57 | 6.39 | **4.2690** | 0.013 |
| E-V-V-E-J (Cassini VVEJGA) | 3.074 | 50.24 | 2.8518 | 6.76 | 7.99 | **5.2861** | 0.025 |
| E-V-E-E-J (Galileo VEEGA) | 3.605 | 52.77 | 2.6149 | 6.80 | 7.99 | **5.7630** | 0.014 |
| E-V-E-V-E-J | 6.423 | 50.68 | 1.1726 | 8.02 | 9.59 | 41.7469 | 0.055 |
| E-V-E-V-V-E-J (ADR 0003) | 8.829 | 49.89 | 0.6041 | 8.22 | 9.59 | **never doubles** | 0.092 |

Every chain loses. The best loses by 28%, against a seed spread of 0.013 — a
margin ~70x the noise. **And the chain is flattered**: it is charged only the
1.5984 yr Earth-Venus synodic as its window, when true E-V-J re-alignments fall
at 23.98 / 59.14 / 83.12 yr (ADR 0005). Closing that hole only widens the gap.

### The mechanism: delta-v saved is bought with time

This is the whole result, and it is visible in one row-pair. E-V-E-J **is**
cheaper than the direct flyby — 3.047 km/s against 4.5435, a 1.50 km/s saving,
exactly what a gravity-assist chain is for. That saving buys real growth:
1.9757 -> 2.8239, **+43% payload per cycle**. And it loses anyway, because the
cycle went 3.276 -> 6.39 yr, **+95%**. Growth compounds per year, so +43% per
pass at half the cadence is a net loss:

```
direct   ln(1.9757)/3.276 = 0.2079 e-foldings/yr  -> 3.33 yr
E-V-E-J  ln(2.8239)/6.39  = 0.1625 e-foldings/yr  -> 4.27 yr
```

**The chain wins the argument it was built to win and loses the one that
matters.** Minimum delta-v is not a proxy for growth; it is closer to an
anti-proxy, because the cheapest arcs are the slowest.

### Why the retirement of VEEGA/VVEJGA is unconditional

ADR 0007 left the chain alive conditional on the SEP thrust rate, which prices
node burns ~5.3x cheaper (`ARGON_SEP_ISP` 2000 s, `v_e` 19.613 vs methalox
3.7265). That hatch must be closed on the new objective, not assumed shut. The
strongest possible form: charge node burns **nothing at all** — more generous
than any thruster — keeping departure and the Jovian burn on methalox, which a
PuffSat cycle cannot thrust electrically:

| sequence | node dv | doubling @ SEP `v_e` | doubling @ nodes FREE | vs bar |
|---|---|---|---|---|
| E-V-E-J | 2.393 | **2.8420** | **2.6356** | **BEATS** |
| E-V-V-E-J (Cassini) | 0.582 | 4.7153 | 4.5991 | loses |
| E-V-E-E-J (Galileo) | 0.406 | 5.2773 | 5.1752 | loses |
| E-V-E-V-E-J | 3.800 | 6.7468 | 5.6381 | loses |
| E-V-E-V-V-E-J (ADR 0003) | 2.947 | 48.6182 | 23.1629 | loses |

**Cassini and Galileo lose with free propellant.** Their node burns total 0.582
and 0.406 km/s — making them free moves Cassini 5.29 -> 4.60, nowhere near
3.3347. Their defeat is not propellant and therefore no propulsion technology
addresses it: **they are beaten by their cycle time**, 7.99 yr against 3.276.
This answers the grill's question in the form it was asked, and it is robust to
the entire SEP question.

**E-V-E-J is a different animal and survives.** At SEP `v_e` it reaches 2.8420
and beats the bar by 15%. But it is a minimal 3-leg chain, not a VEEGA — it has
no extra Earth or Venus rungs — so its survival is not a rehabilitation of
gravity-assist ladders. And ADR 0007 decision 5 applies at full force: these
numbers substitute a low-thrust `v_e` into an **impulsive** rocket equation, and
the node burns are impulsive Lambert mismatches at flyby periapsis that a Hall
thruster cannot deliver. E-V-E-J needs 2.393 km/s of node burn — **the most of
any competitive sequence** — so once again the sequence that scores best is the
one leaning hardest on the invalid assumption. Its 1.25-1.91 yr of required
thrusting inside 1 AU (ADR 0006's 1.25-1.92 km/s/yr) is plausible for a 5.57 yr
trip but unverified.

### A bound that needs no search

`rate = [ln M(v_b) - dv/v_e] / cycle`. Since `dv >= 0`, the numerator is at most
`ln M(v_b)`, and `M` is itself logarithmic in `v_b`, so `ln M` is
*doubly*-logarithmic and nearly immovable: 1.912 at `v_b` 52, 2.072 at 60, and
still only 3.347 at an absurd 200. Against the bar's 0.2079 e-foldings/yr:

| `v_b` | `ln M` | any cycle longer than this loses **at zero delta-v** |
|---|---|---|
| 52 | 1.912 | 9.20 yr |
| 60 | 2.072 | 9.97 yr |
| 200 | 3.347 | 16.10 yr |

This is algebra, not a search result: **a free, perfect chain with a >9.2 yr
cycle cannot beat the direct flyby.** It also explains why the optimizer never
chose a hot return — `v_b` lands at 49-54, never near the grill's 60, because
going 52 -> 60 buys +17% mass ratio for ~1.2 km/s, and `exp(-1.2/3.7265)` = 0.725
gives back more than it earns. Consequently the Jupiter burn goes to ~0 in the
direct flyby: **the powered Jovian flyby is load-bearing only if `v_b` is pinned
high**, which the grill's 60 km/s framing is what created.

## How this was nearly recorded backwards

The first phased run reported "NO PHASED SOLUTION" for all five sequences. That
was a bug in the harness, and it would have made this ADR say something stronger
and false ("phasing kills the chain outright"). Recorded because the failure mode
is silent and repeated:

`PhasedProblem.fitness` returned a flat `1e3` whenever `chain.growth <= 1.0`. A
random point in the box scores growth 0.01-0.6, so **essentially the entire
population received an identical penalty**; sade had no gradient and random-
walked. The comment justifying it ("no graded signal is available here") was
false — `ln(growth)/cycle` is finite and smooth straight through growth = 1, and
being pole-free is the entire reason decision 2 exists. The filter discarded the
gradient the objective was designed to provide. **A losing chain is not an
infeasible one; it is the gradient.**

The diagnosis generalizes: an all-failed table is ambiguous between "empty
feasible set" (physics) and "search never found it" (artifact), and the two must
be separated before anything is recorded. Random-sampling the box settled it in
minutes — 2-11% of points fly a complete phased chain, so geometry was never the
constraint. Two checks now guard the result: blind sampling must find the
feasible manifold at a rate the optimizer should trivially beat, and **phased
doubling must be >= unphased for every sequence** (phasing adds a constraint, so
the phased feasible set is a subset — a theorem the optimizer does not know). All
five obey; before the fix all five reported nothing.

## Consequences

- **Minimum delta-v is retired as an objective for the growth loop**, and ADR
  0007's ranking with it. Any chain number quoted against a mass-per-cycle score
  is now answering the wrong question. ADR 0002's end-to-end mass ratio remains
  correct as a per-cycle quantity and remains the right objective for a
  single-shot mission.
- **The direct powered flyby (ADR 0002) is the defensible architecture**,
  reinforcing ADR 0007 decision 6: nothing from ADR 0003/0004/0005 should reach
  the paper. `sec:jupiter_only_growth` remains PLANNED.
- **The open question is unchanged and has narrowed to one sequence**: a
  low-thrust re-solve of E-V-E-J. Not VEEGA, not VVEJGA, not the pump ladder —
  those are closed by cycle time regardless of propulsion.
- **Launch-window cadence is a first-class term, not a detail.** `cycle` is
  `ceil((trip+coast)/window)*window`, a step function, so within a step extra
  trip time is FREE — the optimizer exploited this exactly, finding a slower,
  cheaper arc that fills the 1.0920 yr window with **zero idle**. The direct
  flyby's cadence is near-annual because Jupiter crawls at 30.33 deg/yr and Earth
  nearly laps it; that is a structural advantage of needing only one body to line
  up, and it is why one free epoch parameter suffices to phase it (charged as
  window quantization, not delta-v) while a chain over-constrains its epoch and
  must pay DSMs.
- **Both models still ignore return-leg Earth phasing** (`_flyby_return_leg`
  never checks where Earth is). This flatters the bar and the chains alike and is
  the next model hole.
- `PUFFSAT_CYCLE_ORBIT_PERIOD = 20 d` is nearly arbitrary: periapsis speed moves
  only 10.8610 -> 10.9806 km/s across 5-60 d. The result does not hang on it.
