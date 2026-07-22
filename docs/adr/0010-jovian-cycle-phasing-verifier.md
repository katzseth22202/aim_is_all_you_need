# The phased Jupiter-only loop self-sustains without waiting; the return bend steers it and the perijove burn adds a cycle

Status: accepted

Date: 2026-07-22

## Context

`puffsat_cycle_growth()` scores the Jupiter-only growth loop (Earth -> Jupiter
flyby -> retrograde return -> relaunch) on doubling time, but its clock is
`trip_time + PUFFSAT_CYCLE_ORBIT_PERIOD` -- it assumes the mass can re-depart to
Jupiter *instantly* after the 20-day periapsis coast. ADR 0006 proved the
*return* phases for free (the Jovian bend covers > 3 wraps of Earth phase) but
explicitly deferred the harder question: does the **whole loop** keep closing,
cycle after cycle, or does it drift into a phase where relaunch is no longer
growth-viable and the cycle breaks?

Two facts make this a real question rather than a restatement of ADR 0006:

1. **The mass cannot wait.** It returns when its trajectory says it does, is
   consumed in the collision immediately, and that collision launches the next
   payload one 20-day coast later. So each cycle's departure time is *pinned* to
   the previous cycle's arrival. There is no parking-and-loitering for a good
   Earth-Jupiter alignment.
2. **A direct Earth->Jupiter relaunch is only cheap on-phase.** A retrograde
   return needs a large Jupiter-relative excess (> Jupiter's ~13 km/s orbital
   speed to reverse), so the outbound must be a fast, hot transfer (~1.3 yr, not
   the 2.73 yr Hohmann). Reaching Jupiter's *true* position with that transfer
   costs the incumbent's ~4.45 km/s only when Jupiter is well placed; off-phase
   it runs 10-38 km/s, which shrinks the payload (`exp(-38/3.7) ~ 0`).

The reconciling freedom is that the arrival time is itself a knob: the Jupiter
bend moves the return crossing across ~1130 days (~2.8 Earth-Jupiter synodic
periods, ADR 0006). So the loop does not wait -- it *steers*, choosing each
return so the next departure lands on a growth-viable Jupiter phase. That
couples the cycles into a chain: cycle k's bend sets cycle k+1's departure phase.

## Decision

1. **Model it as a forward chain search, not a per-cycle solve**
   (`src/jovian_cycle_phasing.py`, `optimize_jovian_cycle_chain()`). Each state
   is a departure time; each edge is a closing trajectory -- an `izzo` outbound
   arc onto Jupiter's true position (departure burn from the cycle-orbit
   periapsis speed), a phased perijove bend and, on the powered run, a perijove
   burn (`_phased_jovian_flyby`), and a retrograde return whose Earth-phase
   mismatch is rooted to zero (`_closing_returns` over `_earth_phase_mismatch`).
   Each edge earns a growth factor `delivered_fraction x payload_mass_ratio`
   (`_net_growth`, the `puffsat_cycle_growth` arithmetic inlined). A generational
   beam over cycles, bucketing near-identical departure times, threads the chain
   that **maximizes compounded mass launched off Earth** over the horizon. The
   seed launch is free within the first synodic period; every later departure is
   pinned to the prior arrival.

2. **Both runs self-sustain; record the numbers.** Every cycle grows
   (`all_growth_positive`), so the loop never stalls for want of phasing, and it
   never waits:

   | run                    | cycles / 30 yr | 10 yr | 20 yr | 30 yr | departure burn |
   |------------------------|:--------------:|------:|------:|------:|:--------------:|
   | unpowered (bend only)  | 8              | 4.9x  | 23.4x | 74.8x | 5.0-6.0 km/s   |
   | powered (perijove burn)| 9              | 4.7x  | 22.1x | 90.2x | 4.6-6.1 km/s   |

   The incumbent's ~4.45 km/s stays bounded because the bend places every next
   launch near a good Jupiter phase; the 10-38 km/s off-phase penalty is never
   paid, because the chain is steered away from those phases.

3. **State the perijove burn's value precisely.** It is a *second steering knob*.
   Unpowered, the closing family is one knob (the bend) against one constraint
   (hit Earth), so steering the next phase and staying cheap are two demands on a
   one-parameter set. The burn adds a second parameter, which lets the chain trim
   the return timing (e.g. 2.08 -> 2.00 yr via a 1 km/s perijove burn) enough to
   pack a **9th cycle** into 30 years and reach 90.2x versus 74.8x -- about +20%
   compounded mass for the cost of a few small burns. The unpowered loop already
   self-sustains, so the burn buys throughput, not survival.

4. **This retires the worst reading of the "instant relaunch" caveat.** The
   doubling-time clock's assumption is not that relaunch is free -- it is that a
   *growth-viable* relaunch trajectory exists every cycle without waiting. This
   chain search exhibits one, for 30 years, with and without the burn. What ADR
   0006 spent proving free (the return) is spent here on steering, and it is
   enough.

## Considered options

- **Immediate relaunch at any cost (rejected).** Launch the instant the mass is
  ready and report whatever the departure burn is. Always closes geometrically,
  but at arbitrary phase the burn hits 10-38 km/s and most cycles do not grow --
  a misleading "it keeps working" that hides that the payload is shrinking.
- **Wait for a good launch window (rejected as unphysical).** Picking a
  convenient launch date parallels ADR 0005's windowed cycle, but the mass is
  consumed on arrival; there is nothing to hold. Waiting is available to the
  *assist-chain fleet* (stagger launches from Earth, ADR 0005), not to a single
  returning mass in this loop.
- **Full DP over continuous departure-time buckets (rejected on cost).** A
  time-ordered Dijkstra-style DP over all reachable buckets did not finish a
  30-year run in 120 s; the reachable-bucket set explodes. The generational beam
  (`_BEAM_WIDTH` states per cycle) runs the unpowered chain in ~5 s and the
  powered one in ~25 s.

## Consequences

- **Epistemic limit, stated everywhere it is quoted.** Circular, coplanar,
  no ephemeris; launch times are a *relative epoch*, not calendar dates, and the
  geometry is idealized. Which specific calendar windows work still needs Lambert
  arcs against real planet positions -- the ephemeris study ADR 0005 and ADR 0006
  both flag, still deferred. The chain proves the loop *can* self-sustain in the
  phasing-free model's own idealization; it does not schedule a real mission.
- **Reproducibility (per the ADR 0009 precedent).** The chain is a grid search,
  so its numbers depend on the committed search box. Recorded in
  `src/jovian_cycle_phasing.py`: outbound TOF grid 1.1-5.8 yr x 26; perijove
  radius floor to 50x floor x 26; perijove burn 0-4 km/s x 5 (powered); state
  bucket 7 d; 16 seeds across the first synodic; beam width 48. Pinned in a
  `slow` test (`tests/test_jovian_cycle_phasing.py`).
- **Distinct from ADR 0005's windowed cycle.** ADR 0005 charges "trip + 0..1
  window" because the *chain family* can stagger launches from Earth. This loop
  cannot -- the single returning mass is pinned -- so its cadence is the raw
  cycle time and its phasing is bought by steering, not waiting. The two models
  answer different questions and must not be conflated.
- **`main.py`** prints both 30-year chains under `sec:jupiter_only_growth`, and
  the module rides on constants the growth loop already computes
  (`puffsat_cycle_periapsis_speed()`), so it is in the default `make run`
  (~30 s added).
- **Flagged for the paper** under `sec:jupiter_only_growth` (PLANNED, like the
  rest of the Jupiter-only growth work).
