# PuffSat Propulsion Analysis

The domain model behind this repo's orbital-mechanics calculations for externally
pulsed (PuffSat) propulsion. This file names the concepts so architecture reviews
and refactors share one vocabulary.

## Language

### Scenarios

**PuffSat scenario**:
A single externally-pulsed propulsion event modelled as one elastic collision —
defined by a collision velocity, a final velocity, and an initial velocity, from
which its mass ratio follows.
_Avoid_: case, row, configuration.

**Mass ratio**:
The payload-to-PuffSat-propulsion mass ratio achievable for a scenario.
_Avoid_: efficiency, ratio (bare), payload fraction.

**Collision velocity** (`v_b`), **final velocity** (`v_rf`), **initial velocity** (`v_ri`):
The three velocities that define a **PuffSat scenario**: the PuffSat's speed at
collision, the payload's speed after, and the payload's speed before.

**Scenario catalog**:
The ordered list of the paper's **PuffSat scenarios** (`paper_scenarios()`). The deep,
typed seam the rest of the system and its tests cross.
_Avoid_: scenario table (that is the projection, not the list).

**Scenario table**:
The DataFrame projection of the **scenario catalog**, produced purely for display.
A one-way adapter at the edge — the only thing in this path that touches pandas.
_Avoid_: catalog (that is the list of scenarios, not the rendered frame).

**Lunar-return optimum**:
The best-burn summary from `find_best_lunar_return()` (a `BurnInfo`): a blended
optimization result, **not a PuffSat scenario**. Presented on its own, not inside
the **scenario table**.
_Avoid_: lunar scenario, lunar row.

### Heliocentric re-intercept (solar-dive return)

The "Sorry, I Don't Need ISRU" cycle sends a payload to a low solar periapsis, boosts it
with PuffSat collisions, and returns it across `1 AU`. The vocabulary below is what the
verification functions in `scenario.py` (`solar_dive_*`, `two_impulse_phasing_loop`,
`single_impulse_resonant_dive`, `earth_reintercept_cycle_floor`, `millionfold_scaling_time`)
name.

**Earth re-intercept**:
The requirement that the boosted return arrive *where Earth actually is*, not merely cross
`1 AU`. The boosted orbit is an escaping hyperbola that crosses 1 AU only once, ~136° from
Earth (`solar_dive_reintercept_gap()`). Crossing 1 AU is not reaching Earth.
_Avoid_: treating "crosses Earth's orbit" as "hits Earth"; the word "interception" (a
near-term LEO terminal-guidance sense in the paper, unrelated to this heliocentric one).

**Whip-around**:
The heliocentric longitude the projectile sweeps from launch to its single 1 AU
re-crossing — 180° falling to periapsis plus the hyperbola's ~115° climb-out, ~295° in all
(`solar_dive_whip_around_angle()`). The miss is set by this whip, not by Earth's drift, and
cannot be re-aimed at periapsis (~5.4 km/s per degree, `periapsis_reaim_cost_per_degree()`).
_Avoid_: "half orbit" (it is more than 3/4 of a turn).

**Phasing loop** (**two-impulse loop**):
A pay-in-time maneuver that delays the deep dive until Earth reaches the fixed crossing
point. The two-impulse form dips shallowly (~0.50 AU) then dives; its two boosts are
colinear and retrograde, so it is free in total impulse (~24 km/s,
`two_impulse_phasing_loop()`) and holds the doubling factor at two.
_Avoid_: calling the phasing a "rocket burn" (every impulse is a PuffSat collision);
re-aiming at periapsis (that is the rejected alternative, not the fix).

**Single-impulse resonant dive**:
The phasing folded into the *one* Earth boost, aimed outbound so the projectile coasts to
a raised aphelion, falls back, dives, and re-crosses `1 AU` where Earth waits. The aphelion
is the free knob that closes the geometry: exactly one value makes Earth's advance equal
the swept longitude, so `single_impulse_resonant_dive()` *solves* for it (~1.9 AU) rather
than hardcoding it, deriving the ~0.85 yr re-cross and the ~37 km/s boost — a ~24 km/s
retrograde component (the direct dive's) plus a ~28 km/s outbound radial one. It needs only
the Earth node, but the heavier boost drops the doubling factor below two.
_Avoid_: reading "aphelion 1.9 AU and periapsis 4 solar radii" as two orbits — it is one
ellipse the 1 AU launch point sits on; treating the boost as free (only the two-impulse
loop is free).

**Re-intercept cycle floor**:
The shortest solar-dive cycle that actually re-intercepts Earth (~0.86 yr,
`earth_reintercept_cycle_floor()`), equal to the whip-around fraction of a year. It is the
payload-doubling interval, so a millionfold scaling takes ~17 yr
(`millionfold_scaling_time()`). Supersedes the paper's earlier implied ~0.5 yr ("6 month")
cycle and its "under a decade" scaling.
_Avoid_: the retired ~0.5 yr / 6-month cycle; "under a decade" for the millionfold.

## Relationships

- A **scenario catalog** holds many **PuffSat scenarios**.
- A **PuffSat scenario** yields exactly one **mass ratio**, computed from its
  **collision**, **final**, and **initial** velocities.
- The **scenario table** is a pure projection of the **scenario catalog** — one row
  per **PuffSat scenario**, no rows of any other kind.
- The **lunar-return optimum** is produced alongside the catalog but lives outside
  the **scenario table**.
- The **re-intercept cycle floor** — not the bare dive time — is the payload-doubling
  interval, so the millionfold scaling time is derived from it, not from a 6-month cycle.

## Example dialogue

> **Dev:** "Where does a scenario's **mass ratio** come from — does the **scenario table** compute it?"
> **Domain expert:** "No. A **PuffSat scenario** knows its own **mass ratio** from its three velocities. The **scenario table** just projects the catalog for display; it computes nothing."
> **Dev:** "Then where does the lunar-return row's ratio come from? It's a ¾/¼ blend, not a single collision."
> **Domain expert:** "That's the **lunar-return optimum** — it isn't a **PuffSat scenario** at all, so it doesn't belong in the **scenario table**. Present it separately."

## Flagged ambiguities

- `scenario_table()` (the old empty-DataFrame factory) and the per-instance
  `append()` conflated *building the list of scenarios* with *rendering the frame*.
  Resolved: **scenario catalog** is the list; **scenario table** is the projection;
  the two are separate, and `scenario_table()`/`append()` are removed.
- The old ninth row forced the **lunar-return optimum** into the **scenario table**
  with a tuple-valued `v_rf`. Resolved: it is not a **PuffSat scenario** and is
  presented separately.
- The solar-dive cycle once doubled payload every ~0.5 yr ("6 months"), implying a
  millionfold in under a decade. Resolved: crossing 1 AU is not **Earth re-intercept**;
  the **re-intercept cycle floor** (~0.86 yr, derived from the **whip-around**) is the
  doubling interval, giving ~17 yr. `main.py` uses the derived floor and the 6-month
  figure is retired. See `docs/adr/0001-earth-reintercept-cycle.md`.
