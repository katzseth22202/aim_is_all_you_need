# Assist-chain launch windows recur at the Earth-Venus synodic cadence; flagged for the paper

Status: accepted

Date: 2026-07-14

## Context

With the E-V-E-V-V-E-J chain landed (ADR 0003) and its sequence family
defended (ADR 0004), the remaining timing question was: how often can such a
chain actually fly? The phasing-free model deliberately cannot say — it puts
every planet where the trajectory needs it — but the window *structure*
follows from synodic periods, because the chain only ever encounters two
inner bodies plus Jupiter:

- Every chain starts E->V, so a window can only open when Venus is properly
  phased: the **Earth-Venus synodic period, ~1.60 yr (583.9 d)**, is the base
  cadence of the whole family.
- The final Earth flyby must throw toward where Jupiter will be ~1.1 yr
  later: Jupiter's phase gate recurs every **~1.09 yr** (Earth-Jupiter
  synodic).
- The inner-ladder geometry near-repeats every **~8 yr** (the Earth-Venus
  13:8 cycle, 5 synodics = 7.99 yr); the full Venus-Earth-Jupiter geometry
  re-aligns only every **~24 yr** (15 E-V synodics ~ 22 E-J synodics ~ 2
  Jupiter years).

The mission does not wait 24 years, because the interior ladder has quantized
time knobs: a resonant Venus-Venus revolution adds ~0.62 yr, an Earth-Earth
revolution adds 1.0 yr, and each V<->E hop has inbound/outbound branch
choices differing by tenths of a year. With 0.62 and 1.0 yr increments the
interior duration can be stretched to match Jupiter's phase to within a
couple of months, and the residual is exactly what the 300 m/s DSM phasing
budget (ADR 0003) is charged for. Cassini's own V-V pair was bridged by a
deliberate 2-yr resonant loop for the same reason, and NASA found
VEEGA/VVEJGA-class outer-planet routes in essentially every Venus window of
the late 1980s-90s (Galileo 1989, Cassini 1997, plus studied backups between).

## Decision

- **Adopt the cadence claim**: most ~1.6 yr Venus windows admit some member
  of the E-V ladder family, with quality varying year to year (a bad year
  means a stretched ladder and a trip nearer 4.5-5 yr than the 3.46 yr
  floor). Exact-repeat geometry is an 8 yr / 24 yr curiosity, not the launch
  cadence.
- **Back it with repo evidence**: `assist_chain_window_cadence()` computes
  the synodic periods from Kepler's law and derives the effective growth
  cycle (trip time plus zero to one Venus window of relaunch wait) and the
  doubling times; printed in `main.py`, pinned in a fast test.
- **Flag it for the paper** under `sec:jupiter_only_growth` (PLANNED, like
  the rest of the chain work): the timing story is fairly optimistic and
  completes the Jupiter-only growth argument — the cycle is not gated on rare
  alignments.
- **State the epistemic limit everywhere it is quoted**: this is synodic
  scaffolding, not an ephemeris search. Which specific calendar windows work,
  and their real trip-time spread, need Lambert arcs against actual planet
  positions — the natural next analysis layer, out of scope for the
  phasing-free model.

## Consequences

- **Growth arithmetic at the 300 m/s chain** (end-to-end ~5.7 per cycle,
  ADR 0003): effective cycle 3.46-5.06 yr (trip + 0..1 Venus window), fleet
  doubling every **~1.4-2.0 yr**, millionfold in **~27-40 yr**. Slower than
  the phased solar-dive loop's 0.86 yr doubling cycle (~17 yr millionfold)
  but with no solar dive at all, and well ahead of the apoapsis-raise loop's
  ~2.7 yr doubling (~54 yr millionfold). This is the "doubling times aren't
  that bad" observation that motivated flagging it for the paper.
- **The pipeline is not window-gated once running**: launches stagger every
  ~1.6 yr, so multiple cohorts fly simultaneously and fleet growth is set by
  per-ship cycle time and parallelism, not by waiting for one alignment.
- The paper should quote the cadence as an estimate with the ephemeris caveat
  attached; promoting these numbers to guarantees requires the Lambert-arc
  study first.
