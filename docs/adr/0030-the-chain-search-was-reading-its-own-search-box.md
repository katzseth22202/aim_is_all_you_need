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
- The ranking inverts. Best single-cycle rate goes to **2.09 S at 0.781/yr**
  against **3.00 S at 0.546/yr**; the 66.05 km/s cycle at 2.88 S gives 0.595/yr,
  also beating 3S.

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

So "the growth loop wants a 3S clock and a 50-60 km/s stream" is a statement
about a methalox-departure architecture. Under the architecture ADR 0009/0012
actually propose, the fast hot cycles win. **This is not resolved here**, and it
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
