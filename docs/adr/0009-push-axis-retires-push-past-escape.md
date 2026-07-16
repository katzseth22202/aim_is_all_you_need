# The push axis retires push-past-escape; collision-driven departure is head-on or nothing

Status: accepted

Date: 2026-07-16

Amends: ADR 0008 (its quotable 3.6320 yr omits the apoapsis-reversal charge;
the corrected incumbent is 4.038 yr). Everything else in ADR 0008 stands.

## Context

The nozzle investigation (a gitignored scratch note,
`todos/puffsat_nozzle_departure_burn.md`; this ADR is self-sufficient without
it) ended on an open question: the growth collision stops at just-under-escape, and the only
reason on record was that the 20-day parking orbit is a phasing buffer. If a
returning PuffSat's arrival could be phased onto the Jovian departure window,
the collision could push straight past escape — deleting the 4.5-5.4 km/s
methalox departure burn and roughly halving the doubling time ("~1000 s
effective Isp overall"). The grill's question: given that arriving PuffSats
would then set the departure time, is that viable?

## Decision

1. **Push-past-escape is retired, on aim rather than timing.** A collision
   cannot steer: the push axis is the arriving PuffSats' Earth-frame velocity,
   and a high `v_b` *requires* a retrograde heliocentric return, so the push
   axis is always retrograde-and-sunward. At the Earth-phased optimum it is
   148.0° from the required departure aim (−145.7° vs +2.3° from prograde);
   the two Earth hyperbolas can bend ~18.1° of that, and no push magnitude
   along the axis reaches Jupiter at nonzero delivered mass (aphelion 1.04 AU
   at 10 km/s of excess; 5.2 AU only in the limit where delivered mass → 0).
   The note's "overtaking pusher plate" rows (Isp 6609 s, 1.52 yr doubling)
   were 1-D artifacts that never checked where the push points.

2. **The sub-escape push target is forced by aim, not propulsion.** The
   parking orbit is load-bearing twice: phasing buffer *and* aim reversal —
   only a bound orbit lets a ~234 m/s apoapsis burn (2×v_apo at 20 d) turn the
   retrograde push into a prograde departure. That burn is real methalox and
   was uncharged: **ADR 0008's quotable doubling time becomes 4.038 yr**
   (growth ×0.9392), not 3.6320. The ranking against the chains is unaffected
   (they would pay the same charge); only the headline number moves.

3. **Any collision-driven departure toward Jupiter is head-on** (retrograde
   PuffSats meeting a prograde payload), i.e. the Appendix-D nozzle geometry
   the note dismissed. Priced honestly — two-currency accounting (slugs are
   parked-payload mass at 1:1; projectiles are wave mass worth M each, so the
   optimal slug ratio is k* ≈ 6-12, not Appendix D's one-currency k = 3) —
   the nozzle beats the corrected incumbent in both operating modes, and the
   architecture choice between them is **left open** pending nozzle hardware
   reality:

   | | doubling | new machinery |
   |---|---|---|
   | incumbent, corrected | 4.038 yr | — |
   | one-wave parked nozzle (derated f=0.8) | 2.990 yr | nozzle only |
   | two-wave powered split (derated) | 1.69-1.80 yr | nozzle + 0.33-0.90 km/s perijove burn stage |

   Mass fractions at the derated optima (`make nozzle` prints all of them):
   one-wave — each arriving wave splits 85.9% growth push / 14.1% nozzle
   projectiles, and each parked payload departs 67.2% craft / 26.7% slug /
   6.1% reversal methalox (departure-burn delivered fraction 0.716, vs the
   incumbent's methalox 0.236); two-wave at dt = 10 d — the departing batch
   splits at Jupiter 81.0% powered growth bend (burning 8.4% of itself) /
   19.0% unpowered projectile bend, parked payload 63.5% craft / 30.4% slug /
   6.1% reversal (delivered 0.676).

4. **The launch-window constraint itself is benign.** One-wave: each wave both
   pushes the new payload and departs the one parked by the previous wave, so
   the departure epoch is the wave-arrival epoch and the window structure is
   exactly the incumbent's; the cost is a one-cycle parking delay (growth obeys
   `g²(1+σ)/(rM) + (σ/k)g = 1` — a √-law that taxes more than the nozzle's own
   inefficiency, capping even a free departure burn at 2.27 yr). Two-wave: an
   unpowered split is infeasible — the bend's Earth-hit roots are ~1 yr apart
   (−325/+370/+751/+1133 d) — so the 10-20 d split must be bought with a
   perijove burn on the growth-wave portion (0.326-0.626 km/s, which also
   raises its `v_b` to 61.5-63.0). This is the `v_b` lottery broken by a
   powered flyby, as CONTEXT.md predicts: pinning arrival *time* costs a
   Jupiter burn exactly like pinning `v_b` does.

## Consequences

- CONTEXT.md gains **push axis** and **apoapsis reversal** as first-class
  terms; **free-aim departure** and **closed cycle** carry the corrections.
- Any future collision-driven scheme must be scored on where the push points
  before its economics are computed.
- The nozzle rows assume 0.8-to-ideal recovery of 2.4+ GJ/kg impact energy —
  ceilings, not design points. Choosing between one-wave and two-wave (or
  neither) waits on that hardware question; nothing here reaches the paper.
- **Every number reproduces from `src/nozzle_analysis.py`** — run
  `make nozzle` (a separate target, deliberately outside `make run`/`make all`:
  it replays the phased geometry, the bend sweep, and the powered-split
  solves). The pricing algebra and the geometry results are pinned in
  `tests/test_nozzle_analysis.py` (fast pins for the recurrences and mass
  fractions; `slow`-marked pins for the aim geometry, the ~1 yr root spacing,
  and the 0.326 km/s dt = 10 d split). The original scratch derivations remain
  in `todos/aim_direction_check.py`, `todos/nozzle_reprice.py`,
  `todos/two_wave_split.py` (gitignored) and the note's resolution section.
