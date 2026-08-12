# The exact two-synodic resonance needs a real-orbit feasibility audit

Status: accepted

Date: 2026-08-11

## Context

The circular-coplanar Jupiter-return model admits a complete Earth-to-Jupiter-to-
retrograde-Earth trajectory at a fixed Earth-Jupiter phase, repeated after two
Earth-Jupiter synodic periods. Circular orbits make that relative phase sufficient:
rotating the entire geometry produces the same trajectory. Real eccentric and
inclined orbits do not have that symmetry.

## Decision

Add `src/real_orbit_resonance.py` as an opt-in, per-window audit. It uses real
three-dimensional Earth and Jupiter states from Astropy's built-in analytical
ephemeris and joins them with zero-revolution Lambert arcs. Jupiter encounter time
is rooted so incoming and outgoing Jupiter-relative hyperbolic-excess speeds are
equal, enforcing a strictly unpowered flyby. The required turn determines perijove;
a window works only if it clears the repository's 4,000 km Jovian altitude floor.

The repeated phase is Earth-minus-Jupiter = -87.411514 degrees, the circular model's
minimum-outbound-v-infinity solution with the same perijove floor. Every second
recurrence of this phase defines a successive two-synodic window. The analysis
returns pandas per-window and summary tables, including period, Earth departure and
return excess velocities, surface collision speed, Jovian turn, and perijove margin.
It also reports a local signed tangential trim at the 4,000 km-altitude perijove:
positive is forward acceleration and negative is backward braking. The trim changes
the outgoing Jupiter-relative excess speed, so it measures turn authority but is not
silently treated as an Earth-closing powered trajectory; that comparison must
re-optimize the Lambert arcs.

The maneuver comparison uses two chained endpoint schedules over the same 200-year
horizon:

- `real_phase_2S`: successive recurrences of the real Earth-Jupiter relative phase,
  retaining the measured 789-807 day variation;
- `fixed_mean_2S`: the same first epoch followed by the circular-model mean period
  every cycle, testing the simpler exact clock without accumulated timing drift.

For both schedules every return epoch is identically the following cycle's departure
epoch. Three cases are searched: an exact split-hyperbola perijove-only burn, an
unpowered Jovian bend followed by an SOI velocity-correction DSM proxy, and a hybrid
with a perijove trim bounded to +/-50 m/s plus that proxy. The DSM proxy exactly joins
the fixed Lambert return state, but is not yet a finite-location interplanetary DSM
optimization; it is an upper-bound-like architecture comparison at the patched-conic
SOI seam.

Run it explicitly with `make resonance`; it is not imported by `src.main` and is not
part of `make all`. The baseline stays strictly unpowered so the effect of each
correction architecture remains measurable. The full baseline and maneuver tables
can be exported with `--csv` and `--maneuver-csv`, respectively.

## Consequences

- Over the 200-year interval beginning 2026-08-11, 91 complete two-synodic
  trajectories fit inside the horizon. Only **45/91 (49.5%)** clear the 4,000 km
  perijove-altitude floor. The worst exact unpowered closure would require a
  perijove 20,515 km below that floor. The minimum-outbound-velocity resonance
  therefore does **not** repeat unpowered for 200 years.
- The requested timing and speed variations are nevertheless small: period
  789.34-807.49 d (2.28% peak-to-peak / mean), Earth departure `v_inf`
  13.184-14.520 km/s (9.63%), Earth-return `v_inf` 59.609-64.877 km/s (8.44%),
  and surface collision speed 60.649-65.833 km/s (8.18%). Small speed variation
  is not enough to preserve the flyby because the required turn maps
  nonlinearly into perijove radius.
- Small timing or speed variation does not by itself prove repeatability. The
  perijove margin is the binding feasibility test and can vary much more strongly.
- Failed rows remain in the table. They show the closest exact unpowered closure and
  its negative floor margin instead of being hidden by an optimizer penalty.
- The signed perijove trim is zero for an already-feasible unpowered row. For a
  failure it solves the split-hyperbola bend equation at the altitude floor and
  reports the outgoing-minus-incoming perijove speed. It is a local diagnostic, not
  an assertion that the speed-changed return arc still hits Earth.
- In this baseline all 46 failures require a backward/braking trim: 110-1,809 m/s,
  median 1,358 m/s. None is within 50 m/s. Those numbers apply to the selected
  exact-unpowered closure; a powered search is allowed to move the Jupiter encounter
  and must be run before concluding that these are minimum Earth-closing burns.
- The exact-endpoint maneuver search gives the following architecture comparison.
  Correction values are per cycle, not cumulative budgets:

  | Endpoint schedule | Correction case | Feasible windows | Median | Mean | Worst |
  | --- | --- | ---: | ---: | ---: | ---: |
  | Real relative phase | Perijove only | 45/91 | 0 m/s | 0 m/s | 0 m/s |
  | Real relative phase | DSM proxy | 91/91 | 152.82 m/s | 987.36 m/s | 3,046.06 m/s |
  | Real relative phase | Hybrid +/-50 m/s | 91/91 | 152.82 m/s | 987.36 m/s | 3,046.06 m/s |
  | Fixed 797.734430 d | Perijove only | 50/91 | 0.320 m/s | 0.341 m/s | 1.177 m/s |
  | Fixed 797.734430 d | DSM proxy | 91/91 | 2.199 m/s | 620.52 m/s | 2,219.35 m/s |
  | Fixed 797.734430 d | Hybrid +/-50 m/s | 91/91 | 2.199 m/s | 620.52 m/s | 2,219.35 m/s |

- The fixed idealized cadence is therefore the better simple clock in this screen:
  it has no accumulated endpoint drift and lowers both mean and worst correction
  relative to following the variable real-phase recurrence. It is not uniformly
  cheap, however. Only 51/91 fixed-cadence cycles need at most 50 m/s of DSM proxy
  correction, and the upper tail is kilometer-per-second class. A fixed period is
  supportable only if the mission architecture can tolerate that tail or a more
  capable finite-location DSM optimization reduces it substantially.
- In the fixed-cadence DSM-proxy optima, Earth departure `v_inf` spans
  13.847-14.376 km/s (3.76% peak-to-peak relative to the mean), speed at 200 km
  altitude spans 17.690-18.107 km/s (2.33%), Earth-return `v_inf` spans
  58.889-63.927 km/s (8.21%), and surface collision speed spans
  59.941-64.897 km/s (7.95%). Thus fixing the clock does not create a large
  outbound-speed penalty in this search; correction delta-v and return speed are
  the quantities with the important tails.
- The hybrid optimizer selected zero perijove burn in the reported optima at the
  committed grid resolution. A bounded +/-50 m/s perijove trim therefore did not
  reduce the DSM demand in this search. Perijove-only exact powered solutions exist
  for 50 fixed-cadence windows and need just 0.013-1.177 m/s where they exist, but no
  admissible split-hyperbola solution exists for the other 41 windows.
- The worst real-phase DSM-proxy case departs 2181-12-05 and needs 3,046.06 m/s. The
  worst fixed-cadence case departs 2216-11-16 and needs 2,219.35 m/s; its best case
  departs 2177-07-24 and needs 0.0358 m/s.
- These powered rows preserve resonance across the whole study: every return epoch
  is identically the next departure epoch, with zero endpoint timing error by
  construction. This proves repeatability only inside the patched-conic model and
  the stated maneuver bounds; it is not a claim of an indefinitely stable natural
  resonance.
- The reported "DSM" is an exact velocity match at the Jupiter patched-conic/SOI
  seam after the unpowered bend. It establishes the correction scale and endpoint
  continuity, but it is not a finite-location heliocentric DSM trajectory. A true
  deep-space maneuver optimization could move the burn and change the required
  delta-v, so mission-level sizing must not treat these proxy values as final.
- Astropy's built-in ephemeris is an analytical approximation, not a navigation-grade
  JPL DE kernel. It is adequate for this architecture-level screen; any mission claim,
  especially beyond 2100, still requires a DE-kernel/N-body verification.
