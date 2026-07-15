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

### Jupiter powered-flyby retrograde return

How a PuffSat actually gets onto the retrograde Earth-crossing trajectory that the
catalog's three Jovian-return rows assume: an Oberth departure burn at 200 km above
Earth (starting from C3 = 0, PuffSat-provided), a coast to Jupiter, and a powered
gravity assist there that bends and pumps the trajectory into a retrograde
heliocentric return. Implemented as `powered_jovian_flyby_return()` /
`jupiter_flyby_vb_trade_curve()` (ADR `0002-jupiter-flyby-objective`); the paper
subsection under `sec:jupiter_only_growth` is still pending, so the catalog rows
keep the published retrograde-Hohmann `v_b` (~69.27 km/s). Headline finding: at the
end-to-end optimum the flyby burn is *zero* — the unpowered bend of a ~13.6 km/s
excess arrival already yields the best retrograde return (`v_b` ~51 km/s,
end-to-end ~2.0 on a flat 49–55 km/s plateau); the Jupiter burn only starts paying
when a `v_b` floor above ~66 km/s is demanded.

**Powered Jovian flyby**:
A gravity assist with an impulsive periapsis burn. The burn splits the flyby into
two hyperbolas sharing one periapsis but with *different* eccentricities, so the
total bend is the sum of two different asymptote half-angles — a burn changes the
turning geometry, it is not "unpowered bend plus a tangent kick".
_Avoid_: applying the unpowered turning-angle formula `sin(δ/2) = 1/e` across a
powered flyby.

**Retrograde-plunge degeneracy**:
Why "minimize propellant subject to any retrograde 1 AU crossing" is ill-posed: the
cheapest trajectories drive the post-flyby tangential velocity to 0⁻ (a near-radial
plunge, `|v∞_out| → v_Jupiter`), so there is no minimizer and every near-optimum is a
barely-retrograde plunge with weak closing speed. The objective must reward `v_b`.
_Avoid_: "minimize total Δv" as the lone objective for retrograde-return legs.

**End-to-end mass ratio**:
The objective for this leg: (delivered mass fraction after both methalox burns, via
the rocket equation) × (payload **mass ratio** at the achieved Earth-closing `v_b`),
scored against the `sec:jupiter_only_growth` push (`v_rf` = lunar-transfer periapsis
speed). Folds propellant and collision strength into one well-posed scalar; the two
Parker rows are reported at the resulting `v_b`, not optimized for.
_Avoid_: optimizing propellant and `v_b` separately.

**Free-aim departure**:
The Earth-departure burn's propellant cost depends only on the hyperbolic-excess
*speed*; aiming the excess-velocity vector at any in-plane angle to Earth's orbital
velocity is free (it just rotates the escape hyperbola). The outbound leg therefore
has two knobs — excess speed and aim angle — at a single propellant price, and the
optimizer searches both rather than assuming a tangential departure.
_Avoid_: assuming departure must be along Earth's velocity; charging propellant for
the aim angle.

**Seven-year cap**:
Hard constraint: outbound Earth→Jupiter plus return Jupiter→1 AU time of flight
≤ 7 yr (time inside Jupiter's sphere of influence is negligible). Excludes extreme
apoapsis-raise trajectories. Baseline Hohmann-out/Hohmann-back is ~5.5 yr.

### The growth loop and its clock

The exponential launch loop the Jovian return exists to close, scored in payload
per *year* rather than per pass (ADR `0008-doubling-time-retires-veega`).

**Closed cycle**:
The returning PuffSat's collision pushes the mass to just under Earth escape — a
20-day orbit at 200 km periapsis (`PUFFSAT_CYCLE_ORBIT_PERIOD`) — which falls back
to periapsis and departs from there. So `v_rf` (the push target) and the departure
burn's starting speed are **the same number**, 10.9503 km/s
(`puffsat_cycle_periapsis_speed()`). _Avoid_: treating the push target and the
departure state as independent — the orbit the PuffSat drives the mass into *is*
the orbit the next cycle departs from.

**Doubling time**:
`cycle x ln2 / ln(net growth)`, where net growth is `M(v_b) x exp(-dv/v_e)` and
`cycle` is departure-to-departure (`puffsat_cycle_growth()`). _Avoid_: scoring the
loop on delta-v or on per-cycle mass ratio — a cheaper, slower chain can save
1.50 km/s, gain 43% payload per pass, and still lose on doubling because its cycle
nearly doubled. Minimum delta-v is closer to an anti-proxy for growth, because the
cheapest arcs are the slowest.

**Growth rate**:
`ln(growth)/cycle`, in e-foldings per year. The *search* objective; doubling time
is the *reported* one. Doubling has a pole (it diverges as growth -> 1+), so an
optimizer sees an infinitely tall spike at the edge of the feasible set; the rate
passes smoothly through zero and goes negative for a shrinking cycle. _Avoid_:
filtering `growth <= 1` points out as infeasible — a losing chain is not
infeasible, it is the gradient.

**Windowed cycle**:
`ceil((trip + coast)/window) * window` — the next launch window at or after the
return, not the trip. A step function, so inside a step extra trip time is FREE
and buys a slower, cheaper arc. _Avoid_: quoting trip time as the cycle.

**Disqualification bound**:
`rate = [ln M(v_b) - dv/v_e]/cycle`, and `dv >= 0`, so the numerator is capped at
`ln M(v_b)` — which is doubly-logarithmic in `v_b` and barely moves (1.912 at 52
km/s, 3.347 at 200). Any cycle beyond ~9.2 yr therefore loses to the direct flyby
**at zero delta-v**. Algebra, not a search result; use it to disqualify a sequence
before optimizing it.

### Unpowered assist chain

The companion question to the powered flyby: can Venus/Earth/Mars gravity assists
replace the 4.45 km/s departure burn? Implemented as `assist_chain_return()` /
`minimum_departure_burn_assist_chain()` / `venus_reach_departure_floor()` (ADR
`0003-assist-chain-search`). Headline finding: yes — ~0.29–0.30 km/s at departure
(barely above the ~0.2794 km/s Venus-reach floor) reaches the same-target
retrograde return in ~3.5 yr at 300 m/s, with a 300 m/s phasing budget charged on
top the end-to-end mass ratio is ~5.7 versus the powered flyby's ~2.0. Sequence
robustness (ADR `0004-pump-ladder-vs-vve`): Cassini-style V-V-E only opens at
~1 km/s of departure burn — same-body legs cannot pump, so the ladder's V↔E
alternations are mandatory at low burn — and the ~15.4 km/s Jovian arrival uses
~96 of ~122 available bend degrees, comfortably inside the unpowered envelope.

**Tisserand invariant**:
An unpowered flyby rotates the planet-relative excess velocity but can never
change its *magnitude*. Growing the heliocentric energy therefore requires
arriving at a body with the excess velocity misaligned from the planet's motion —
which a flyby at *another* body sets up. That is the whole mechanism of the chain.
_Avoid_: "the flyby added energy" without naming which body's frame; expecting a
single body to pump itself (same-body flybys only re-aim).

**Tisserand lock**:
The dead end at the exactly-minimum Venus transfer: the arrival is tangent to
Venus's orbit, so the excess velocity is aligned, every rotation of it is wasted
re-aiming, and the maximum Earth-relative excess is frozen at its launch value
forever. No amount of trip time escapes it.
_Avoid_: treating the Venus-reach floor (`venus_reach_departure_floor()`,
~279.4 m/s) as the chain's minimum burn — *at* the floor the chain is locked;
feasibility starts a few m/s above it (square-root escape: misalignment grows as
the square root of the margin above the floor).

**Pump ladder**:
The alternating-body climb the search finds (e.g. E-V-E-V-V-E-J): each hop
arrives more misaligned, so each flyby converts more of the (fixed) local excess
speed into heliocentric energy, until Jupiter is reachable at ~15 km/s of excess.

**Phasing budget**:
The model is phasing-free — each planet is wherever the trajectory needs it — so
a fixed 300 m/s deep-space-maneuver reserve (`ASSIST_CHAIN_PHASING_BUDGET`) is
charged as spent methalox in the delivered-fraction accounting. The headline mass
numbers already carry the estimated cost of making real ephemerides line up.
_Avoid_: quoting the chain's delivered fraction from the departure burn alone;
calling the reserve a flyby burn (all flybys are strictly unpowered).

**Feasibility witness**:
The beam search proves feasibility by exhibiting a replayed chain; a `None` means
"not found at these beam settings", never "infeasible". Minimum-burn results are
upper bounds on the true minimum (the calibrated production beam closes at
~290 m/s; a finer beam closes at ~285 m/s).
_Avoid_: reading `None` as a proof; global top-by-speed pruning (it made
feasibility non-monotone in the burn — pruning is time-bucketed instead).

**Return-branch knob**:
The free Earth-phasing degree of freedom at Jupiter (`jovian_return_phasing_envelope()`,
ADR `0006-unpowered-jupiter-and-free-return-phasing`). Retuning the unpowered bend walks
the return between a fast **inbound** arrival (the craft is already falling when it
reaches Jupiter, ~1.25 yr) and a slow **outbound** one (it is still climbing, so it
coasts to aphelion and falls back, ~4.37 yr) — one connected curve through full
reversal, spanning ~1130 d = ~1113° of Earth phase, 3.09 wraps. Above one wrap every
launch phase admits a phased intercept, so **Earth re-intercept on the return leg is
free** — no propellant, no powered flyby. Periapsis is not a second knob: it *is* the
bend (`e = 1 + r_p·v∞²/μ`), and dropping the floor 4000→200 km buys +1.2°.
_Avoid_: quoting the span while holding `v_b` fixed (that manufactures two disjoint
clusters and an 821-day phantom hole — the bend is one knob with two outputs, so `v_b`
and arrival time cannot be varied independently); quoting a span without its
`largest_gap` (max-minus-min over a gapped set counts unreachable time as authority);
reading this as solving the **phasing budget**'s problem — that is the inner ladder,
which the bend acts downstream of and cannot touch.

**Tisserand `v_b` ceiling**:
The chain's hard limit on Earth-closing speed, ~56.27 km/s: an unpowered flyby cannot
grow the ~15.369 km/s Jovian excess, so the most retrograde state reachable is full
reversal (`v_t = v_Jupiter − v∞ = −2.31 km/s`). The catalog's three Jovian rows assume
the retrograde-Hohmann 69.27 km/s, which needs 20.47 km/s of outgoing excess and is
therefore **unreachable at any phase, periapsis or arrival time** — not merely
disfavoured. Buying it costs a 2.62 km/s Jovian burn and drops end-to-end 5.72 → 3.93.
The ceiling is a function of *this chain's* arrival excess, not a universal constant:
it is `v_Jupiter + v∞`, so a hotter Jovian arrival lifts it. The direct powered flyby
reaches `v_b` 60 and even 65 with a **zero** Jovian burn, by departing harder
(5.34 / 5.99 km/s) and arriving with more excess. 56.27 bounds the *unpowered chain*,
not the architecture.
_Avoid_: treating 69.27 as a maximum (it is the *minimum-energy* retrograde arrival —
the only purely tangential one); expecting timing to raise it (timing sets position,
energy sets speed, and they are separate currencies); quoting 56.27 as a limit on any
trajectory that did not inherit the chain's 15.369 km/s arrival.

**`v_b` lottery**:
The price of free phasing: the bend that places Earth *dictates* the collision speed
(somewhere in 50.14–56.27), rather than the mission choosing it. Benign only because
every value in the band is an acceptable loop (end-to-end 5.7–6.3, doubling
1.37–1.51 yr) — and note 51 km/s *beats* 56 on doubling time, because the extra `v_b`
is free in propellant but costs trip time.
The lottery is a consequence of the flyby being **unpowered**, not a law. It is a
knob-counting result: one knob (perijove radius *is* the bend) against two demands
(`v_b` and arrival time) is over-determined. A **powered Jovian flyby** has two knobs —
perijove radius *and* periapsis burn — so it meets both demands and **breaks the
lottery**, at the price of propellant. Any mission that pins `v_b` and also requires
**Earth re-intercept** must therefore burn at Jupiter.
_Avoid_: quoting the best-case `v_b` as if it were selectable; treating the lottery as
unbreakable (it is a statement about unpowered flybys); reading a zero Jovian burn from
an optimizer that never enforced Earth re-intercept as evidence the burn is unnecessary.

**Launch-window cadence**:
How often the chain family can fly (`assist_chain_window_cadence()`, ADR
`0005-launch-window-cadence`): windows open every Earth-Venus synodic period
(~1.6 yr), gated secondarily by Jupiter's ~1.1 yr phase; resonant-rev stretching
of the ladder plus the **phasing budget** let most windows fly some family
member. Effective growth cycle = trip time + 0..1 Venus window (~3.5–5 yr),
doubling every ~1.4–2.0 yr at the 300 m/s chain's end-to-end ratio.
_Avoid_: treating the ~8 yr Earth-Venus cycle or ~24 yr V-E-J realignment as the
launch cadence (they are geometry-repeat curiosities); quoting the cadence as an
ephemeris-verified guarantee (it is synodic scaffolding — real calendar windows
need Lambert arcs against actual planet positions).

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
- The **return-branch knob** settles **Earth re-intercept** for the Jovian return leg
  but not for the **pump ladder**: it acts at Jupiter, downstream of every inner-planet
  flyby, so the **phasing budget** still carries the ladder's alignment cost.
- The **return-branch knob** and the **Tisserand `v_b` ceiling** are the same bend read
  two ways: sweeping it phases Earth, and its extreme (full reversal) caps `v_b`. The
  coupling between them is the **`v_b` lottery**.

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
- **An unpowered Jovian flyby needs arrival `v_inf` > 13.058 km/s, full stop.** The
  retrograde return requires the post-flyby heliocentric tangential speed to go
  negative — `v_jupiter + w_out·cos(angle) < 0` — and no bend angle achieves that unless
  `w_out` exceeds Jupiter's own orbital speed. You cannot rotate a 6 km/s excess into
  cancelling 13.058 km/s of Jupiter's motion; the flyby rotates the excess but never
  rescales it. So this is a **necessary condition**, not a cost: below it the unpowered
  return does not exist at any perijove. ADR 0002's direct flyby clears it comfortably
  (arrives at `w_in` ≈ 15.1 and spends **zero** Jovian burn). It is also exactly why a
  cold-arriving chain *must* buy a Jupiter burn — the burn's real job is lifting `w_out`
  over 13.058, not fine-tuning `v_b`.
- **The per-cycle growth budget is bounded, so trip time can disqualify an architecture
  outright.** The growth loop's rate is `[ln M(v_b) - dv/v_e] / cycle`. Because `dv >= 0`
  the numerator can never exceed `ln M(v_b)` — and `M = 2f/ln(v_b/(v_b - v_rf))` is
  *itself* logarithmic in `v_b`, so `ln M` is doubly-logarithmic and crawls: 1.912 at
  `v_b` 52, 2.001 at 56.27, 2.072 at 60, and only 3.347 at `v_b` **200**. No architecture
  earns its way out through a hotter return. Delta-v savings are capped (recovering at
  most the `dv/v_e` term) while cycle time is unbounded in the denominator. Against the
  direct flyby's 0.2259 e-foldings/yr this gives a **hard bound: any cycle over ~8.85 yr
  loses even at zero delta-v**. This is algebra, not a search result — it survives any
  re-optimization.
- **Launch-window cadence is not in the model, and it cuts against the short trip.**
  Cycle is currently set to *trip*, but the next departure must wait for a window. The
  direct loop's window is the **Earth-Jupiter synodic, 1.0920 yr** (Jupiter moves only
  30.33 deg/yr, so Earth nearly laps it annually); the chain needs Venus *and* Jupiter,
  and exact E-V-J re-alignments fall at 23.98 / 59.14 / 83.12 yr. Easy windows help less
  than they appear: quantizing naively, the direct flyby idles 0.522 yr on a 2.754 yr
  trip (**19%**) while Cassini idles 0.727 on 7.265 (**10%**), and the doubling gap
  **collapses from 10% to 2%** (3.651 vs 3.727). A *short* trip is penalised
  proportionally more by window quantization. The resolution is presumably to let the
  optimizer tune the trip to land on a window (a slower, cheaper arc with zero idle),
  which is unbuilt. Unresolved, and load-bearing for any direct-vs-chain ranking.
- **The mass ratio mixes two altitudes.** `payload_mass_ratio(v_rf, v_b)` takes `v_b`
  folded through Earth's well with `v_esc_surface` — escape at the **surface**, 11.1799
  km/s — while `v_rf` is a speed at **200 km** (10.9503 km/s for the cycle orbit, escape
  there 11.0086). So the collision is scored at the surface and the push target 200 km
  up. Worth 0.036 km/s on `v_b` (53.3675 surface vs 53.3319 at 200 km, 0.07%), i.e.
  immaterial to every result so far — but it is a frame mismatch, not a modelling
  choice, and nothing should be built on it. Unresolved (minor).
- **The fast Jovian arc is not a transfer to Jupiter.** The direct flyby's 2.699 yr
  round trip looks impossible against the 2.731 yr *one-way* Hohmann, and the resolution
  is that Hohmann is the minimum-energy case and this is nothing like it: departing at
  `v_inf` 11.425 (Hohmann needs 8.793) puts the craft on an **a = 11.67 AU, e = 0.914,
  aphelion 22.3 AU** ellipse at 41.209 km/s against 42.122 km/s solar escape. Jupiter at
  5.2 AU is crossed in the first quarter of that arc, still moving fast — 1.220 yr out,
  1.478 yr back. Reasoning about these legs with Hohmann intuitions will be wrong by 2x.
- **A return leg's perihelion is virtual.** `_flyby_return_leg()` ends at the first
  *inbound* 1 AU crossing, so the reported perihelion (0.023 AU for the direct flyby —
  ~5 solar radii) is never flown. It is a shape parameter meaning "this return is nearly
  radial", which is what maximises closing speed. Not a solar-dive thermal problem.
- **"Interception" is still overloaded.** The paper's near-term LEO terminal-guidance
  sense, the **Earth re-intercept** requirement (a *position*), and the **collision
  velocity** `v_b` (a *speed*) all get called "interception" in conversation — e.g.
  "phasing for at least 60 km/s interception" is really two constraints. Keep them
  apart; and note "60 km/s" is itself ambiguous by ~1 km/s, since `v_b` = 60 means a
  closing speed of 58.95 while a closing speed of 60 means `v_b` = 61.03.
  `target_collision_speed` is the `v_b` convention.
- **Nothing in the model knows where Jupiter is.** `_jovian_terminal()` takes only
  `(v_t0, v_r0, r0, elapsed, params)` and calls `_conic_radius_crossings()`, which finds
  crossings of Jupiter's orbit *radius* and assumes Jupiter is there; `longitudes0` in
  `_phased_ladder_burn()` is parallel to `ladder`, which only ever holds Venus/Earth/Mars.
  So ADR 0007's "E-V-E-**J**" phased Venus and Earth but let **Jupiter be anywhere** —
  its 4.5677 km/s is a *lower* bound with respect to Jupiter phasing. Unresolved.
- **`_flyby_return_leg()` never checks where Earth is.** It scores `closing_speed` at the
  1 AU crossing, which is the error **Earth re-intercept** exists to name. Both the
  powered flyby's numbers (ADR 0002) and the chain's (ADR 0003/0007) inherit this; they
  lean on the **return-branch knob** making phasing free, which holds only while `v_b`
  floats. Any study that pins `v_b` must re-check it. Unresolved.
- ~~**The collinearity guard in `_phased_ladder_burn()` is dimensionally inconsistent**~~
  Resolved: it now divides by both radii, so it tests `sin(sep) < 1e-6` as intended.
- **ADR 0007's harness is gone and its numbers are not reproducible.** The ADR records
  results (E-V-E-J at 4.5677, epoch 0.80, legs 0.414/0.866) but not the search space
  that produced them, and the scratch script no longer exists. A rebuilt harness on the
  same primitives lands in a *different basin of similar cost* (~4.50 at epoch ~0.83,
  but with departure and node burns inverted: 3.20/1.30 against ADR 0007's 0.93/3.63)
  and, once legs may exceed ~2.5 yr, finds materially cheaper chains still. **The cost
  landscape is multi-modal and the leg-time bound is load-bearing** — so ADR 0007's
  "converged to four decimals" attests convergence *within its own unrecorded bounds*,
  not a physical optimum. Record bounds with results.
- ~~**Minimum-total-dv is a poorly-posed objective for the chain.**~~ **Resolved** by ADR
  `0008-doubling-time-retires-veega`: the growth loop is scored on doubling time, and
  min-dv is retired as its objective. The diagnosis was exactly right — relaxing the leg
  bound from 2.5 to 3.0 yr drops the best E-V-E-J from ~4.50 to ~3.55 km/s by stretching
  one leg to 2.86 yr, pushing the trip from 3.94 to 6.19 yr — and the phased re-score
  confirms the consequence: E-V-E-J beats the direct flyby on delta-v (3.047 vs 4.5435)
  and loses on doubling (4.27 vs 3.33 yr).
- ~~**Return-leg Earth phasing is still unmodelled.**~~ **Resolved** by
  `_ReturnLeg.sweep_angle` / `_earth_phase_mismatch()` (ADR 0008). It cost the direct
  flyby **8.9%** — the quotable figure is **3.6320 yr**, not 3.3347 — and it is paid in
  trajectory *shape*, not time: the trip stays pinned at the launch window (departure
  4.5435 → 5.3751 km/s, `v_b` 51.46 → 59.77). The chains are still scored without it,
  which flatters them deliberately: a constraint can only raise an optimum, so every
  chain is bounded below by its unphased 4.2690 and the ranking holds *a fortiori*.
- **An all-failed optimizer table is not a result.** It is ambiguous between an empty
  feasible set (physics) and a search that never found it (artifact), and the two look
  identical. Random-sample the box first: if blind sampling finds feasible points at a
  rate the optimizer should trivially beat, the harness is broken. This is not
  hypothetical — the first phased run reported "NO PHASED SOLUTION" for all five
  sequences while 2–11% of random points flew a complete chain (ADR 0008, "How this was
  nearly recorded backwards").
