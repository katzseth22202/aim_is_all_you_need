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

## Relationships

- A **scenario catalog** holds many **PuffSat scenarios**.
- A **PuffSat scenario** yields exactly one **mass ratio**, computed from its
  **collision**, **final**, and **initial** velocities.
- The **scenario table** is a pure projection of the **scenario catalog** — one row
  per **PuffSat scenario**, no rows of any other kind.
- The **lunar-return optimum** is produced alongside the catalog but lives outside
  the **scenario table**.

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
