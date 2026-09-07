# The growth chain was reading its own search box, and the perijove burn buys nothing

Status: accepted

Amends: ADR `0010-jovian-cycle-phasing-verifier`. Its decisions 1 and 4 stand
unchanged -- the forward chain search is the right instrument, and it does
exhibit a self-sustaining 30-year chain. Its decision 2's **figures** and its
decision 3 **in full** are retired below.

Date: 2026-09-07

## Context

A design question about the Jupiter-only growth loop -- can the returning stream
be made to arrive faster than the ~55 km/s the 3S closure delivers, by relaxing
the synodic clock? -- was put to `optimize_jovian_cycle_chain()`, which is the
only model in the repository that enforces a *real* Earth re-intercept on every
cycle rather than merely crossing 1 AU.

The first symptom was that five of the eight cycles on the reported chain sat at
`outbound_time` = 1.100 yr exactly, the value of `_OUTBOUND_TOF_MIN`. That is the
signature of a binding search bound, so the bound was moved. The objective then
moved by a factor of 20 in both directions depending on the sample count, which
is the signature of something worse.

CLAUDE.md already carries the lesson this ADR is a second instance of: ADR 0007's
numbers became unreproducible when its scratch harness vanished, and ADR 0009 and
ADR 0012 record their search boxes for exactly that reason. This ADR records a
different failure of the same family -- a box that was written down, and was
wrong.

## Decision

**Adopt converged search settings, and retire ADR 0010's perijove-burn verdict.**

    _OUTBOUND_TOF_SAMPLES   26  ->  200
    _BEAM_WIDTH             48  ->  200
    _STATE_BUCKET_DAYS     7.0  ->  7.0   (unchanged; already fine)
    _OUTBOUND_TOF_MIN      1.1  ->  1.1   (unchanged; not binding after all)

`src/main.py` prints only the unpowered run. At converged settings the powered
run returns a bit-identical chain, so printing it costs about ten minutes to
repeat a table verbatim. The powered path stays reachable through the `powered`
argument and is pinned by a slow test.

Convergence evidence. Both series are single-variable, unpowered, 30-year
compounded mass, `_OUTBOUND_TOF_MIN` = 1.1, `_STATE_BUCKET_DAYS` = 7,
`_SEED_SAMPLES` = 16, everything else held fixed:

| samples (beam 200) | 26 | 80 | 120 | 160 | 200 | 260 |
|---|---:|---:|---:|---:|---:|---:|
| 30-yr mass | 74.8 | 186.6 | 239.7 | 227.0 | **233.4** | 233.2 |
| cycles | 8 | 9 | 9 | 9 | 9 | 9 |

| beam (samples 200) | 48 | 100 | 200 |
|---|---:|---:|---:|
| 30-yr mass | 152.1 | **233.355** | **233.355** |
| cycles | 8 | 9 | 9 |

**The samples series is not monotone.** From 120 up it oscillates in a +/-3% band
(239.7, 227.0, 233.4, 233.2) rather than climbing to a limit -- which grid points
land near the optimum is partly luck. What *is* solid is the step from 8 cycles
to 9 and the jump out of the 74.8-152.1 range; the fine value is ~x233 +/- 3%.

Also stable across `_SEED_SAMPLES` 16/32 (233.4 / 231.4), `_PERIAPSIS_SAMPLES`
26/40 (identical), and `_STATE_BUCKET_DAYS` 3/7 (233.355 both, at beam 400).

`_OUTBOUND_TOF_MIN` is left at 1.1 yr because at converged settings it is not
materially binding: 1.1 yr gives x233.36 and 0.70 yr gives x235.45, a 0.9%
difference that sits inside the residual grid scatter (200 vs 260 samples differ
by 0.15%, seeds 16 vs 32 by 0.8%). Every run that *was* free to dive picked a
shortest outbound leg of 1.02-1.19 yr -- never the 0.82-0.98 yr a hot
"past the corner" departure would need.

**Quote the chain as ~x233, not x233.3555.** The four-digit agreement across
beam 100/200/400 is reproducibility at fixed grid, not accuracy; the honest
figure carries about +/-1%.

## Consequences

### The headline figure was 3.1x low, and the two knobs are coupled

**x74.8 -> x233.4** over 30 years, 8 cycles -> 9. Neither knob converges alone,
which is why this was hard to see:

- At the committed 26-sample grid, each generation offered **fewer than 48**
  successors, so the beam never cut. Widening it alone changed *nothing* --
  48, 128 and 320 all returned 74.795 exactly. The beam looked innocent.
- Refining the grid raises the successor count to **94-403 per generation**, at
  which point the 48-wide beam cuts in *every* generation. The grid looked like
  the whole story until the beam was raised too.

The correct rule is that `_BEAM_WIDTH` must exceed the per-generation successor
count, which is itself a function of the grid. It is not a free tuning constant.

### `_STATE_BUCKET_DAYS` is a dedup key, not a clock

Worth recording because it was suspected and cleared. Bucketing keys successors
by `round(next_departure / bucket)` but `add_node` carries the branch's **exact**
`next_departure` forward, so a chain can never slip its clock inside a bucket.
Coarsening it is purely lossy: at a non-binding beam the 30-year mass reads 233.4
at both 3 and 7 days, 214.8 at 14, and 125.9 at 28. The apparent *gain* at 14
days in early testing was the coarse bucket accidentally protecting temporal
diversity from an over-narrow beam.

### The perijove burn buys nothing; ADR 0006 is vindicated

ADR 0010 decision 3 states the burn is a **second steering knob** worth a 9th
cycle and +20% (90.2x against 74.8x). At converged settings:

| run | cycles | 10 yr | 20 yr | 30 yr | perijove burns |
|---|:--:|---:|---:|---:|---|
| unpowered (bend only) | 9 | 6.473 | 41.882 | **233.355** | 0 by construction |
| powered (perijove burn) | 9 | 6.473 | 41.882 | **233.355** | **all nine exactly 0.0** |

The optimizer, handed a free perijove burn, drives all nine to zero and
reproduces the unpowered chain. The 9th cycle exists in *both* runs; it was the
under-searched unpowered run that was missing it, not the powered run that
gained it.

This makes the repository *more* consistent, not less. ADR 0006 decision 1
("the Jovian flyby stays unpowered") rests on four independent lines, and ADR
0002's optimizer drives the flyby burn to 3.4e-11 km/s. ADR 0010 was the single
result that disagreed. It no longer does.

### The chain converges onto the synodic clock without being told to

The converged chain's cycle lengths, in Earth-Jupiter synodic periods:

    3.01, 2.99, 3.01, 2.99, 3.01, 3.01, 2.98, 3.01, 2.17

Eight consecutive cycles at 3.00 S. Nothing in `_cycle_branches` knows what a
synodic period is; the search is free to return any real cycle length. The
trailing 2.17 is the last cycle squeezed under the 30-year horizon.

The mechanism is a **narrow phase-viability window**. Enumerating every closing
cycle from one departure and asking what the *next* departure offers:

| this cycle | 2.09-2.20 S | 2.92 S | 2.95 S | 2.97-3.06 S | 3.08-3.14 S |
|---|:--:|:--:|:--:|:--:|:--:|
| growing cycles available next | **0** | 13/41 | 40/43 | **all** | **0** |

Outside roughly 2.92-3.06 S the mass lands at a phase from which *no* growing
cycle exists -- the best available `net_growth` falls to 0.03-0.9. So faster
round trips are not merely worse, they are one-way doors.

**Parking and waiting for the phase does not rescue them**, and the reason is
exact: waiting until the relative longitude repeats converts *any* cycle into a
3.00 S cycle. Charged with its wait, every fast cycle is dominated by the 3.00 S
trajectory outright (rate 0.1496/yr for 2.19 S and 0.1508 for 2.88 S, against
**0.1895** for 3.00 S). Waiting costs precisely what it buys.

### The constructive result: fly hot, park to rephase

The negative result above -- "parking and waiting does not rescue a fast cycle" --
is true at methalox and **false above about Isp 1200 s**. It deserves stating
positively, because it is what the whole line of inquiry was reaching for.

Hold the departure on one Earth-Jupiter phase so the clock never drifts, but
spend part of the 3S window *parked* rather than flying: take a shorter, hotter
return and pad the remainder to exactly 3.00 S. Because every candidate is padded
to the same total, **cycle time cancels entirely** and the comparison collapses to
per-cycle growth, i.e. to a single exchange rate -- how much extra departure burn
a hotter `v_b` is worth:

| `v_b` (km/s) | mass-ratio gain | Isp 380 | Isp 1200 | Isp 2214 |
|---:|---:|---:|---:|---:|
| 56 | x1.053 | 0.19 | 0.60 | 1.11 |
| **60** | x1.136 | **0.48** | **1.51** | **2.78** |
| 64 | x1.220 | 0.74 | 2.34 | 4.32 |
| 68 | x1.304 | 0.99 | 3.13 | 5.77 |

(km/s of extra departure burn that exactly cancels the gain, against a 53.5 km/s
baseline and the 20-day cycle orbit's `v_rf` = 10.9503 km/s.)

At methalox, going 53.5 -> 60 km/s buys only **0.48 km/s** of Δv headroom, which
is why fly-and-park loses there. At Isp 1200 it is 1.51 km/s, which is enough.

Availability is never the constraint. At *every* departure phase that supports a
viable pure-3S cycle, a sub-3S cycle reaching `v_b` >= 60 also exists:

| | phases with a viable 3S cycle | sub-3S reaching `v_b` >= 60 | fly-and-park beats pure 3S |
|---|---:|---:|---:|
| Isp 380 | 11/73 | 11 | **1** |
| Isp 1200 | 30/73 | 30 | **17** |
| Isp 2214 | 30/73 | 30 | **25** |

At Isp 2214, by departure phase:

| phase | pure 3S: `v_b` / dv / growth | fly-and-park: flight / park / `v_b` / dv / growth | gain |
|---|---|---|---:|
| 0.71 | 62.4 / 5.97 / 6.300 | 2.78 S / 0.22 S / **68.3** / 7.09 / 6.606 | x1.048 |
| 0.75 | 60.7 / 5.60 / 6.213 | 2.75 / 0.25 / **68.1** / 6.64 / 6.727 | x1.083 |
| 0.81 | 56.7 / 7.48 / 5.283 | 2.68 / 0.32 / **68.6** / 8.61 / 6.184 | x1.171 |
| 0.86 | 53.0 / 11.85 / 4.007 | 2.60 / 0.40 / **69.0** / 13.49 / 4.976 | x1.242 |
| 0.89 | 50.9 / 14.57 / 3.378 | 2.62 / 0.38 / **68.0** / 16.55 / 4.254 | x1.267 |

Two things to take from the shape. **The gain is smallest at the sweet phase and
largest off it** -- 5% at phase 0.71, 27% at 0.89 -- so this is a *robustness*
mechanism that flattens the off-phase penalty, not a headline optimizer. And the
`v_b` column sits at **67-69 km/s throughout**: this is the **perfect-retrograde
boundary** of CONTEXT.md, reached without touching the clock. Fly a 2.6-2.8 S
perfect-retrograde return at `v_b` ~ 68, park 0.2-0.4 S (about 80-160 days),
depart again on the same phase.

Parking that long moves `v_rf` from 10.9503 to ~10.99 km/s as the cycle orbit
lengthens, worth 0.2% on the mass ratio -- checked, not assumed, and the table
above is computed at the 20-day value so it is not flattered by it.

**What it costs, against what it may spend.** The exchange-rate table above says
what a hotter `v_b` is *worth*; this is what it actually *costs*. Both cycles
depart the 20-day cycle orbit's periapsis (`v_rf` = 10.9503 km/s at 200 km,
escape there 11.0086):

| phase | | out | ret | park | dep dv | `v_inf` | `v_b` |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.71 | exact 3S, repeats | 1.11 yr | 2.09 yr | -- | **5.974** | 12.85 | 62.43 |
| | perfect retrograde + park | 1.01 | 2.03 | 0.22 S | **7.094** | 14.30 | **68.32** |
| 0.74 | exact 3S, repeats | 1.11 | 2.08 | -- | **5.559** | 12.30 | 61.29 |
| | perfect retrograde + park | 0.98 | 2.01 | 0.26 S | **6.735** | 13.84 | **68.67** |
| 0.75 | exact 3S, repeats | 1.11 | 2.08 | -- | **5.599** | 12.36 | 60.68 |
| | perfect retrograde + park | 0.98 | 2.02 | 0.25 S | **6.636** | 13.71 | **68.13** |

Across all viable phases the extra departure burn runs **+0.000 to +1.978 km/s,
median +1.036** (`perfect_retrograde_premium()`, printed by `make fly-park`;
selection is by *arrival speed* so the premium is a property of the geometry, not
of the exhaust speed being scored). Against the `v_b` 68 budget of 0.99 / 3.13 / 5.77 km/s at Isp
380 / 1200 / 2214, that is:

| | budget at `v_b` 68 | cost | verdict |
|---|---:|---:|---|
| Isp 380 | 0.99 | +1.04 | **just over -- a wash to a slight loss** |
| Isp 1200 | 3.13 | +1.04 | 3x headroom |
| Isp 2214 | 5.77 | +1.04 | 5.5x headroom |

Methalox is not far from working -- it is about 5% short, which is exactly why
fly-and-park wins at 1 phase of 11 rather than at none.

The `+0.000` rows at phases 0.45-0.55 are not a bargain: there the exact-3S cycle
*already is* the perfect-retrograde one (`v_b` 67.6-69.1), but its departure burn
is **15-21 km/s** because the phase is far from Jupiter's cheap position. Hot,
not attractive.

**Cross-check against the free-bend estimate.** A circular-coplanar hand
derivation with a free bend and *no* Earth-intercept constraint (recorded under
"A related correction to CONTEXT.md's Tisserand `v_b` ceiling" below) predicted
Earth departure `v_inf` 13.812 km/s and a 6.712 km/s burn, +1.557 over the 3S
optimum. The Lambert model with Jupiter's true position and Earth intercept
enforced gives `v_inf` 13.71-14.30, burn 6.64-7.09, +1.04 to +1.18. The two agree
to about 0.2 km/s on the burn; what the estimate could not see was the phase
structure, which turned out to be the whole story.

**Caveat: this is a single-cycle exchange-rate comparison, not a chain run.**
Every candidate is padded to the same 3.00 S and returns to the same departure
phase, so it should chain trivially -- but that has not been run, and this ADR
already records one case (the 2.09 S inversion) where a single-cycle result did
not survive the chain.

### The open question this exposed: the two models price different machines

The verdict above holds **only under the chain's accounting**, which charges the
departure burn to **methalox at Isp 380 s, `v_e` = 3.727 km/s**
(`_net_growth`, via `_AssistChainParams.flyby.exhaust_speed`). ADR 0009 and ADR
0012 describe a different machine: the departure burn is driven by the returning
stream through the **head-on nozzle**, `v_e = eta*(sqrt(1+k)-1)*w/k`, which runs
19-22 km/s at `k` = 3.

Recomputing the same enumerated branches with the nozzle exhaust in place of
methalox reverses both findings:

- Every phase becomes live. The 0-growing-cycle phases above go to 38/38, 41/41,
  45/45 -- there are **no dead phases at all**.
- The best *single-cycle* rate inverts, to **2.09 S at 0.781/yr** against
  **3.00 S at 0.546/yr**. **This does not survive the chain**, and the
  distinction matters -- see "the clock survives the accounting change" below.

Sweeping the departure exhaust speed rather than switching between two values
shows this is a **continuum, and the interesting variable is launch-window
coverage**. Enumerating every closing cycle from 73 departure phases spread over
one synodic period (3,483 cycles), and counting how many *phases* admit at least
one growing cycle:

| departure Isp (s) | 380 | 700 | 1000 | 1200 | 1500 | 1800 | **1900** | 2214 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v_e` (km/s) | 3.73 | 6.86 | 9.81 | 11.77 | 14.71 | 17.65 | 18.63 | 21.71 |
| **phases with a growing cycle** | **18%** | 36% | 49% | 60% | 77% | 95% | **100%** | **100%** |
| best single-cycle rate (/yr) | 0.205 | 0.516 | 0.645 | 0.695 | 0.746 | 0.780 | 0.789 | 0.812 |
| its cycle length | 2.18 S | 2.06 S | 2.06 S | 2.06 S | 2.06 S | 2.06 S | 2.06 S | 2.06 S |
| its `v_b` (km/s) | 53.05 | 62.98 | 62.98 | 63.68 | 63.68 | 63.68 | 63.68 | 63.68 |

**Coverage reaches 100% at Isp 1900 s** -- and ADR 0019's own departure-nozzle
ledger already puts the 3S departure at **Isp 2214 s** at `eta_jet^2` = 0.6. So
the repository's own nozzle number sits *above* the threshold at which every
departure phase becomes usable.

The same thing seen as a launch window, staggering the departure about a nominal
epoch and asking what the best available cycle earns:

| offset | -96 d | -48 d | -24 d | 0 | +24 d | +48 d | +96 d |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isp 380 | none | none | 0.127 | **0.205** | 0.073 | none | none |
| Isp 2214 | 0.537 | 0.684 | 0.750 | **0.796** | 0.797 | 0.721 | 0.429 |

At methalox the usable window is about -36 to +24 days out of a 399-day synodic
period, and the rate falls 65% across it. At the nozzle's Isp the window spans at
least +/-96 days and is *flat* within +/-24 days (0.750 to 0.806, under 1% from
the peak). A 72-hour launch stagger is negligible in both cases; what changes is
that one narrow window per 1.09 yr becomes a continuous one.

**Where the gain actually comes from.** At the nominal phase, per-cycle growth
goes 1.65 (Isp 380) to 6.13 (Isp 2214), x3.72. Decomposed: the faster impactor
contributes x1.18 (mass ratio 6.98 -> 8.27 as `v_b` goes 53.5 -> 62.4) and the
departure burn's surviving mass fraction contributes x3.14 (0.236 -> 0.741).
**The exhaust speed does about 85% of the work and the hotter impactor about
15%** -- the headline is not "faster impactors carry more kinetic energy", it is
"the departure burn stops eating four fifths of the payload". Note also that the
impulse law is *linear* in closing speed, not quadratic in it.

### The clock survives the accounting change; only the growth rate moves

The single-cycle inversion above is a trap, and this ADR fell into it before
catching it. Running the *actual* 30-year chain with the departure charged at
Isp 2214 rather than methalox:

| | cycle lengths (synodic periods) | 30-yr mass |
|---|---|---:|
| Isp 380 | 3.00 x8, 2.20 | x235.5 |
| Isp 2214 | 2.97, 3.01, 5.03, 2.97, 3.00, 2.98, 2.98, 3.74 | **x2,504,273** |

**Both hold ~3.00 S, and both hold phase** -- the Isp 2214 chain's departure
phase after each cycle runs 0.97, 0.98, 0.01, 0.98, 0.98, 0.96, 0.94, never
drifting. A 2.09 S cycle has the better instantaneous rate and a worse successor;
the chain optimizes compounded mass, so it declines it. Where the hot returns do
appear they are *long* excursions (5.03 S and 3.74 S, `v_b` 68.8 and 68.1 -- the
perfect-retrograde arrivals) that land back on phase.

So the exhaust speed does **not** buy a shorter clock. It buys growth per cycle,
1.83 -> ~6.3, and the launch-window coverage above. Quote it that way.

**Resolved: the 2S fixed point exists, and declining it on methalox is correct.**
This was recorded here as open; it is not. `fixed_points(grid, multiple=2)` finds
a cycle of **1.9998 synodic periods** at departure phase 0.8082
(Earth-minus-Jupiter -69.04 deg), drifting **-0.0002 S per repetition** -- 130
repeats before it leaves a 0.02 S band, so it repeats for far longer than any
horizon here. It arrives at `v_b` 63.35 km/s on a **1.20 yr inbound return**
(ADR 0006's fast branch) for a **8.613 km/s** departure burn.

| departure Isp | 2S fixed point | best 3S as the chain flies it | winner |
|---|---|---:|---|
| 380 (methalox) | **shrinks** (growth 0.838) | 0.178 /yr | **3S** |
| 1200 | **0.641 /yr** | 0.499 /yr | **2S**, +28% |
| 2214 | **0.794 /yr** | 0.567 /yr | **2S**, +40% |

So the chain never selecting it is **not** a search failure. On methalox the 2S
point does not merely score worse, it *loses mass* -- `exp(-8.613/3.727)` = 0.099
against a mass ratio of 8.44. Declining it is the right answer. Give the
departure a real exhaust speed and it wins by 28-40%, which is the same verdict
as fly-and-park and for the same reason.

**And the real-orbit answer already exists.** ADR 0011 audits this exact
resonance against Astropy ephemerides over 200 years and finds only **45 of 91
windows (49.5%) clear the 4,000 km perijove floor**. Note *why*: not timing drift
-- the period varies only 2.28% and the speeds 8-9% -- but the perijove, because
the required turn maps nonlinearly into perijove radius. That is why
`real_orbit_resonance.py` already carries a chained fall-back that flies 2S when
it closes and 3S when its DSM proxy exceeds a threshold. The eccentric, inclined
Jupiter is what turns a clean 2S fixed point into a 2S-with-3S-fallback cadence.

So "the growth loop wants a 3S clock and a 50-60 km/s stream" is right about the
**clock** under either accounting, and wrong about the **stream and the rate**
under the architecture ADR 0009/0012 propose. **This is not resolved here**, and it
should not be quoted either way until one model prices both legs consistently.
The nozzle figures above are a single-point estimate (`beta` at burn start,
`k` = 3, `eta` = 0.8), not the integrated ledger `circular_resonance_impulse.py`
computes, and they charge nothing for delivering the second, head-on stream --
which has a phasing problem of its own that no model here addresses.

### A related correction to CONTEXT.md's "Tisserand `v_b` ceiling"

Recorded because it was measured in the same session and is independently
checkable. CONTEXT.md says of the catalog's retrograde-Hohmann 69.27 km/s that
it "needs 20.47 km/s of outgoing excess against 15.37 available, so no amount of
timing or periapsis reaches it." That is true of the **assist chain**, whose
arrival excess is Tisserand-locked at 15.369 km/s, and it is *not* true of the
architecture. On a direct departure:

| Earth departure `v_inf` | 13.812 km/s |
|---|---|
| burn above the 10.9503 km/s parking periapsis | 6.712 km/s (+1.557 over the 3S optimum's 5.155) |
| Jupiter arrival `v_inf` | 20.473 km/s |
| **perijove burn** | **0** |
| bend required / available at the 4,000 km floor | 76.8 deg / 106.3 deg |
| resulting `v_b` | 69.24 km/s |

"Perfect retrograde" is a *boundary*, not a point: below 20.473 km/s of arrival
excess the `v_b`-maximizing bend is full reversal and the arrival is partly
radial; at and above it, the maximizing return is purely tangential with
perihelion pinned at 1 AU, and `v_b` climbs to 72.74 at solar escape. Note also
that per-cycle growth peaks at `v_b` ~= 69.4 and *falls* after -- above ~70 km/s
extra `v_b` buys a shorter clock, not a harder push.

These figures come from a circular, coplanar, tangential-departure sweep with a
free bend and **no Earth-intercept constraint enforced**; they agree with the
repository where the two overlap (69.24 against the catalog's 69.27, 20.473
against 20.47, 5.155 km/s at the ADR 0012 3S optimum). They are a statement about
reachability, not a proposed trajectory.

### Cost

`make run` gains about 100 s for the converged unpowered chain, and loses the
~620 s the powered run used to cost. The slow test pins the powered/unpowered
equality at a shorter horizon.

## Considered options

- **Leave the constants and note the sensitivity (rejected).** The committed
  x74.8 is quoted in ADR 0010 and CONTEXT.md as a headline. Leaving a figure that
  is 3.1x low in place, with a footnote, is worse than fixing it.
- **Raise only `_OUTBOUND_TOF_SAMPLES` (rejected).** It converges to x152.1, not
  x233.4 -- still 35% low, because the beam then binds. The knobs must move
  together.
- **Keep printing the powered run (rejected).** Ten minutes of `make run` to
  reproduce the unpowered table verbatim. The equality is a *result*, and results
  belong in a test and an ADR, not in a recomputation on every build.
- **Lower `_OUTBOUND_TOF_MIN` (rejected as immaterial).** This was the original
  hypothesis -- five of eight reported cycles sat at 1.100 yr exactly -- and it
  is wrong. Freeing the floor to 0.70 yr leaves the optimum where it was: the
  chain's shortest outbound leg lands at 1.02-1.19 yr in every run, never near
  the 0.82-0.98 that a hot "past the corner" departure would need, and the
  objective moves 0.9%, inside the residual grid scatter. See the
  measured comparison in the consequences above. The fence only *looked*
  load-bearing because 1.1 was always a grid point sitting near the optimum,
  which accidentally stabilised an under-resolved grid: at the old 48-wide beam,
  fencing at 1.1 gave x152.09 against x152.52 free, while *removing* the fence
  made the objective swing x6.7 to x242 across sample counts, because the
  near-optimal arc was no longer guaranteed to be in the grid.
