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
verification functions in `heliocentric_reintercept.py` (`solar_dive_*`,
`two_impulse_phasing_loop`, `single_impulse_resonant_dive`,
`earth_reintercept_cycle_floor`, `millionfold_scaling_time`) name.

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

**Conic kernel**:
The float-valued (km, s, km/s, rad) two-body conic geometry shared by this section's
flyby search, the **unpowered assist chain** below, and `nozzle_analysis.py` —
`src/conic_kernel.py`: time-of-flight, true-anomaly-at-radius, radius crossings, the
eccentricity/bend algebra (`half_turn_angle`, `unpowered_bend_angle`,
`powered_bend_angle`), the branch-cut angle wrap (`wrap_pi`), and the well-folding
speed formula (`speed_with_escape_energy`, `sqrt(v_infinity**2 + v_esc**2)` in
quadrature). Plain floats rather than `astropy.units.Quantity` because the
optimizer hot loops (`differential_evolution`, the beam search) evaluate this
geometry many thousands of times per run; `orbit_utils.py`'s Quantity-valued
time-of-flight, true-anomaly, and escape-energy functions delegate to it rather than
duplicating the algebra, so it sits *below* `orbit_utils.py` in the module hierarchy
(CLAUDE.md).
_Avoid_: re-inlining the eccentricity, bend, angle-wrap, or well-folding formulas at
a new call site — the whole point of the kernel is that this algebra has exactly one
home. (Three ad hoc re-implementations of the angle wrap and one of well-folding were
found and fixed when this was written up — check for existing kernel coverage before
adding a new one.)

**Retrograde-return legs**:
The float-valued substrate one layer up from the **conic kernel**, shared by this
section's flyby search, the **unpowered assist chain** below, and
`nozzle_analysis.py` — `src/retrograde_return_legs.py`: the leg-by-leg
body/radius/velocity/time-of-flight state each path assembles differently
(`_ReturnLeg`, `_FlybyLeg`, `_AssistBody`, `_body_state`, `_flyby_return_leg`,
`_powered_flyby_leg`, `_phased_jovian_flyby`, `_phased_ladder_burn`,
`_earth_phase_mismatch`, among others). Named after what it computes, echoing
the codebase's own vocabulary (`_ReturnLeg`, `_FlybyLeg`, `AssistChainStep`).
`jovian_flyby.py` and `assist_chain.py` are siblings built on this one
foundation rather than either depending on the other; `nozzle_analysis.py` is a
third caller reusing the same primitives to score an alternative departure.
_Avoid_: re-deriving any of this leg/state algebra at a new call site, or
treating it as private to whichever of `jovian_flyby.py` / `assist_chain.py`
happens to call it first — the underscore prefix marks internal-to-the-substrate,
not internal-to-one-module.

**Powered Jovian flyby**:
A gravity assist with an impulsive periapsis burn. The burn splits the flyby into
two hyperbolas sharing one periapsis but with *different* eccentricities, so the
total bend is the sum of two different asymptote half-angles (the **conic kernel**'s
`powered_bend_angle`) — a burn changes the turning geometry, it is not "unpowered
bend plus a tangent kick".
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
In the **closed cycle** the "just rotates the hyperbola" justification is subtly
wrong: the parking orbit's orientation is dictated by the **push axis**, ~148° from
the required aim, so free aim is *bought* by the **apoapsis reversal** (~0.23 km/s,
currently uncharged — see flagged ambiguities).
_Avoid_: assuming departure must be along Earth's velocity; charging propellant for
the aim angle; treating free-aim as costless in the closed cycle.

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
(`puffsat_cycle_periapsis_speed()`). The parking orbit is load-bearing twice over:
it is the phasing buffer *and* the aim reversal (the push is along the **push
axis**, retrograde; the departure must be prograde; only a bound orbit lets the
**apoapsis reversal** convert one into the other). _Avoid_: treating the push
target and the departure state as independent — the orbit the PuffSat drives the
mass into *is* the orbit the next cycle departs from; calling the parking orbit
"just a phasing buffer" (pushing past escape departs immediately, aimed at the
Sun).

**Push axis**:
The direction a PuffSat collision can push the payload: along the arriving
PuffSats' Earth-frame velocity, full stop — a collision cannot steer. High `v_b`
*requires* a retrograde heliocentric return, so the push axis is always
retrograde-and-sunward (−145.7° from prograde at the Earth-phased optimum, vs the
+2.3° the Jupiter departure needs; the two Earth hyperbolas can bend only ~18° of
the 148° gap, and no push magnitude along the axis reaches Jupiter at nonzero
delivered mass — `src/nozzle_analysis.py`, `make nozzle`). Consequence: a collision-driven
departure toward Jupiter is **head-on or nothing** — the "overtaking pusher plate"
departure is a 1-D artifact.
_Avoid_: assuming a collision inherits **free-aim departure**; scoring
push-past-escape schemes without checking where the push points.

**Apoapsis reversal**:
The cheap re-aim the parking orbit buys: at the near-escape apoapsis the payload
moves ~117 m/s (20 d orbit), so a ≤ 2×v_apo burn (~234 m/s; ~112 m/s at 60 d)
rotates the periapsis velocity by up to 180°, turning the retrograde push into a
prograde departure at unchanged periapsis speed. This is what makes **free-aim
departure** true-in-effect for the closed cycle — at a small, currently uncharged
methalox cost.
_Avoid_: charging a full plane-change at periapsis speeds for the re-aim; leaving
the reversal out of the growth accounting (it is ×0.94 on growth at 20 d).

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

### Impact direction

Which way the arriving PuffSats point relative to the thrust the vehicle wants, and
what that costs (`src/circular_resonance_impulse.py`, `make resonance-impulse`, ADR
`0012-head-on-impact-penalty-is-bounded`).

**Impact-angle impulse law**:
`beta(theta, k) = sqrt(1 + k - sin^2 theta) + cos theta`, the impulse per arriving
impactor kilogram at impact angle `theta` from the thrust axis, with the exhaust
canted to leave the vehicle no transverse kick. The repo's two impulse laws are its
two endpoints: ADR 0009's head-on nozzle `sqrt(1+k) - 1` at 180°, and the growth
push's along-axis `1 + sqrt(1+k)` at 0°.
_Avoid_: treating head-on and along-axis as separate models; reading the `-1` as a
loss term that can be tuned away (it is `cos 180°`, the full incoming momentum).

**Aim separation**:
The angle between the arriving PuffSats' Earth-relative excess vector and the one the
next Jupiter departure needs. A **diagnostic**, measured at the patched-conic
boundary — *not* the angle the **impact-angle impulse law** consumes, which is taken
at the 200 km burn point relative to the moving vehicle (157.27° becomes 148.5° for
2S once the vehicle's motion and the **departure-hyperbola mirror** are applied).
_Avoid_: feeding the SOI angle into `beta`; scaling impulse by the Earth-inertial
return `v_inf` instead of the vehicle-frame closing speed `w` (that overstates a
canted geometry roughly two-fold and ranks the trade backwards).

**Departure-hyperbola mirror**:
The free sign choice in the ~14° (2S) / ~18° (3S) rotation between the departure
hyperbola's outgoing excess vector and its periapsis velocity, `arcsin(1/e)`. Both
mirror images give the same outgoing excess, so picking the one that rotates the
thrust axis *toward* the incoming stream is worth ×1.012 / ×1.020 at zero
propellant — more than any delta-v purchase on the aim trade curve. Same ~18.1° ADR
0009 counted; now priced.
_Avoid_: leaving it out of an aim comparison (it is larger than the effect being
measured).

**Free-aim ceiling**:
What the impact angle would be worth if it carried *no* delta-v charge: ×1.166 (2S) /
×1.131 (3S) at `k = 3`, ×1.105 / ×1.079 at ADR 0009's `k* = 7.057`, ×1.072 / ×1.050
at `k = 10`. No trajectory, resonance or phasing beats a free angle, so it bounds
every aim-steering scheme; the searched optimum captures under 30% of it.
_Avoid_: quoting an aim result without it — a per-row gain of ×1.216 at the 25 km/s
cap is real and still loses, because it is bought with delivered mass.

**Head-on crossover**:
The slug ratio above which an exactly head-on impact is the *optimum* rather than a
penalty — `k = 18.47` (2S) / `16.43` (3S). `v_e` goes as `beta * w`; canting raises
`beta` but lowers the closing speed `w`, because head-on *adds* the vehicle's own
speed to the impactor's, and the bias term `cos theta` is `k`-independent while
`sqrt(1+k)` grows.
_Avoid_: calling head-on a penalty without naming `k`; searching for aim relief at
`k` near or above the crossover.

### Slug-augmented collisions

Both ends of the cycle can be improved by giving the arriving PuffSat something
to vaporise. The vocabulary below is what `src/nozzle_analysis.py`,
`src/circular_resonance_impulse.py` and `src/two_wave_growth.py` name; the
closed-form derivations live in ADR `0013`, "The algebra, in one place".

**Slug**:
Carried mass the arriving PuffSat vaporises, whose plume is then collimated
onto a pusher. Momentum comes out of energy as `sqrt(2mE)`, so spreading a
fixed impact energy over more mass buys more momentum — that, and only that,
is where the gain comes from. _Avoid_: calling it a pusher-plate improvement
(an inelastic merge followed by an elastic bounce gives `2*mu*w` at every `k`
— a flat plate recovers **nothing** from the merge); calling it reaction mass
(the gain is collimation).

**Slug ratio** (`k`):
Kilograms of slug per kilogram of arriving impactor, a continuous real the
designer sizes, not a count. Trades exhaust speed (falling in `k`) against
impactor economy (rising), so its optimum depends on what is being charged:
~7.06 for bare-dish Isp, 6–12 on ADR 0009's two-currency ledger, 8.5–9.5 for
the **two-wave split**'s chain optimum. _Avoid_: "more mass is better" without
naming the currency; treating `k` as quantized.

**Closing speed** (`w`):
The impactor's speed *relative to the moving vehicle* at impact — the quantity
the **impact-angle impulse law** is linear in. On the departure burn the
vehicle is flying into the stream, so `w = v_b + v` and it *climbs* through the
burn as the vehicle accelerates (74.2 → 80.9 km/s for a 2S closure); on the
growth push the PuffSat overtakes, so `w = v_b − v` and it falls.
_Avoid_: using `v_b`, or the return leg's Earth-relative `closing_speed` at the
1 AU crossing, in place of it — they are 10–20 km/s apart, and ADR 0012 shows
the substitution overstates a canted geometry roughly two-fold.

**Effective exhaust speed**:
`v_e = e * w * (sqrt(1+k) − 1)/k` for the head-on departure nozzle — the
**impact-angle impulse law** at 180°, charged against the slug actually spent.
The `sqrt(1+k)` is the *blob*: impactor plus slug. _Avoid_: `sqrt(k)`, which
omits the impactor and runs 31% high at `k = 3`.

**Recovery** (`e`, also `eta`):
One lumped factor for how much of the ideal collimated impulse a real device
delivers — collimation, geometric capture and plasma coupling together.
Numerically set to `f` = 0.8 in the ideal-ceiling rows so the `k → 0` limit
reproduces the paper, but a bare paraboloid's free-molecular capture is only
0.31 of ideal at `k = 3`. ADR 0013's sweep makes `e ≈ 0.3` the architecture's
survival threshold. _Avoid_: quoting a point estimate without the break-even
`e`; conflating it with `f`, which is an elasticity claim about a bounce, not
a collimation claim about a plume.

**Two-wave split**:
The departing batch divides at Jupiter into a **growth wave** that arrives
early and pushes the next payload, and a **nozzle wave** one parking-orbit
period behind that supplies the head-on departure-burn projectiles for the
payload the growth wave just parked. It is what makes growth *linear*
(`g[(1+sigma)/(rMd1) + sigma/k] = 1`) instead of the one-wave parked
architecture's square-root law (`g^2(1+sigma)/(rM) + (sigma/k)g = 1`), and the
two differ by roughly a square — 3.81 against 1.93 per cycle at `e = 0.6`,
`k = 7`. _Avoid_: pricing a growth loop without saying which of the two it is
(it is the largest single number in the accounting); reading ADR 0009's "the
split must be bought" as general — see **split gap**.

**Split gap**:
How far ahead the **growth wave** arrives, which *is* the parking-orbit period:
the payload is pushed at periapsis, coasts one full orbit, and departs at the
next periapsis. So the same number also sizes the **apoapsis reversal** (372.5
m/s at 10 d, 233.9 at 20 d, 112.1 at 60 d) — a longer gap buys a cheaper
reversal and pays for it in the burn needed to pull the growth wave further
ahead. 10 d and 20 d land within 0.7% of each other. _Avoid_: choosing the gap
and the parking period independently (ADR 0009 priced a 10 d split against a
20-day reversal); assuming the split must be bought at all — in **real orbits**
the Lambert pair's free encounter time makes it nearly free (8 of 11 flown
cycles under 1 m/s, ADR 0013), whereas the circular-coplanar model had only
the bend and found its Earth-hit roots ~1 yr apart.

### Two-legged nozzle and the launch ledger

What happens if the *overtaking* growth push is a magnetic nozzle too, rather
than the paper's pusher plate (`src/plume_thermal.py`,
`src/two_leg_nozzle_sweep.py`, `make two-leg`, ADR `0014`).

**Overtaking nozzle**:
A magnetic nozzle on the growth push, carrying its own slug at ratio `k1`. It
sits at the 0° end of the **impact-angle impulse law**, so the impactor's
momentum *adds*: `beta = 1 + sqrt(1+k)`. Parked mass per arriving impactor
kilogram becomes `k1/sigma1` in place of the plate's `2f/Lambda`, which is the
only change to the **two-wave split**'s ledger. Read as an engine it is the
plate's mirror image — the plate has infinite Isp and a hard impulse cap, the
nozzle finite Isp (~975 s at `k1 = 15`) and impulse growing as `sqrt(k)`.
_Avoid_: assuming "impactors are precious, slug is cheap" makes `k1` unbounded
— `k1/sigma1` peaks and falls, and the **launch ledger** bites first anyway.

**Plume ignition window**:
The closed interval of slug ratios whose post-merge plume reaches 15,000 K, from
`eps_th = w^2 k / (2(1+k)^2)` — only the *dissipated* CM-frame energy heats the
blob, not the bulk drift. It peaks at `k = 1` and falls on **both** sides, so
the condition is a window, not a ceiling: too little slug dissipates nothing and
leaves no plasma to grip. Fleet-wide across the flown chain, `k1` in
[0.098, 10.21] and `k2` in [0.043, 23.32], both ceilings set by the slowest-
closing three-synodic cycles. _Avoid_: reading it as a ceiling only; charging
the blob's *total* energy rather than the thermalised fraction (that roughly
doubles every ceiling).

**Ignition bill**:
84.4 MJ/kg for water with a 1% potassium seed — and 60% of it is **atomising the
water**, not ionising it. The seed carries the conductivity for 0.13% of the
budget. _Avoid_: assuming ionisation dominates (it is the smallest term);
optimising the slug material inside this model, which has no molar-mass
dependence in the impulse and would therefore always pick the heaviest species.

**Launch ledger**:
Kilograms returning from Jupiter per kilogram off the pad: 2/3 of liftoff is
ground-rocket propellant (exactly a 4.09 km/s lob at 380 s), a quarter of the
remainder is launcher dry mass, so 1/4 reaches the intercept point, and
`0.25 * r * d1 / [(1+sigma1)(1+sigma2)]` must clear 1/15. Both legs' slug is
lofted from Earth, so the two nozzles **compete for the same launched
kilograms** — at poor `e2` the departure burn takes the whole budget and `k1`
collapses onto the ignition window's lower root. This is the binding constraint
in 56 of 64 swept cells; the **plume ignition window** binds in one. _Avoid_:
charging the launcher's dry mass against liftoff rather than the non-propellant
remainder (that makes 1/15 unreachable for the plate incumbent too, so the bar
stops discriminating).

**Equivalent plate elasticity**:
The `f` a pusher plate would need to match a given `(e1, e2)` nozzle cell at the
same `e2` — the incumbent's own axis, so the two architectures compare directly.
Above 1.0 means no physically possible plate matches, since `f` is a restitution
coefficient. Crossover against the paper's `f = 0.8`: `e1` between 0.5 and 0.6
at a good head-on leg, `e1 ~ 0.7` at a poor one. Against a perfect plate,
`e1 >= 0.8`. Since ADR 0013 puts the architecture's survival threshold at
`e ~ 0.3`, **the overtaking nozzle must be about twice as good as the leg that
already exists just to replace a device that carries no propellant** — which is
why the plate stays. _Avoid_: comparing a nozzle cell against a plate at a
different `e2` (leg 2 is common to both, and holding it fixed is what makes the
number one-dimensional).

### Phased growth chain (no waiting)

The doubling-time clock above assumes the mass re-departs to Jupiter the instant
it returns; `src/jovian_cycle_phasing.py` (`optimize_jovian_cycle_chain()`,
ADR `0010-jovian-cycle-phasing-verifier`) tests whether that relaunch actually
exists, cycle after cycle, against real Earth and Jupiter phases. The binding
fact is that **the mass cannot wait**: it is consumed on arrival, so each cycle's
departure is pinned to the previous arrival plus the 20-day coast. The only
freedom is to *steer* the arrival — the Jupiter bend moves the return crossing
across ~1130 days (~3 Earth-Jupiter synodics, ADR 0006), so each return is chosen
to land the next launch on a growth-viable Jupiter phase. That couples the cycles
into a **chain** (cycle k's bend sets cycle k+1's departure phase), searched
forward as a generational beam over cycles that maximizes compounded mass
launched off Earth. Result: the loop self-sustains both ways — unpowered (bend
only) 8 cycles / ×74.8 over 30 yr, powered (perijove burn) 9 cycles / ×90.2, the
burn acting as a **second steering knob** that tightens the return timing enough
to fit one more cycle. _Avoid_: the "wait for a good window" model (ADR 0005's
windowed cycle) here — that staggering is available to the assist-chain *fleet*,
not to a single returning mass; and treating a direct off-phase relaunch as cheap
(it is ~4.45 km/s only on-phase, 10–38 km/s off it). Circular-coplanar, relative
epoch, not calendar dates.

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

### Jovian solar-dive cycle

Letting Jupiter place the solar dive instead of Earth: depart for Jupiter, let an
unpowered flyby drop perihelion to 4 solar radii, take the Oberth boost there, and
cross 1 AU where Earth waits -- the whole loop clocked to an integer multiple of the
Earth-Jupiter synodic period so the next cycle repeats the geometry
(`src/jovian_solar_dive_cycle.py`, `make jovian-dive`, ADR
`0019-three-synodic-closes-two-does-not`). It exists to retire the **single-impulse
resonant dive**'s ~37.5 km/s Earth-side boost: the collision at Earth then only has to
buy a Jupiter transfer, 15.24 km/s at 200 km instead of 39.11.

**Dive-placement floor**:
The minimum Jupiter *arrival* excess speed that can put perihelion at 4 solar radii --
11.9557 km/s prograde, 13.0579 km/s for a radial plunge, 14.1602 km/s retrograde
(`dive_placement_excess_floor()`). A flyby rotates the excess but never rescales it, so
placing the dive means cancelling nearly all of Jupiter's 13.06 km/s of orbital motion,
and below these speeds no perijove achieves it. The same shape of statement as the
13.058 km/s floor on the unpowered retrograde return: a **necessary condition, not a
cost**. Both signs are reachable from a single arrival speed in the 14.16-17 km/s
overlap -- but **no 3S closure arrives inside it** (12.17 km/s at 4 solar radii,
10.95 at 32), so the prograde/retrograde split does need two departure energies, and
the floors spread further apart as the dive is backed out (9.98 / 16.14 at 32 solar
radii). The opposing stream is one-way and expendable, so it needs a separate
tangential launch (11.83 km/s of Earth excess at 32 solar radii) rather than a
separate energy *budget*. See the **solar-dive depth trade**.
_Avoid_: reading a floor as a delta-v to be spent; reading the overlap as saying one
departure energy serves both streams on a closed cycle (it does not).

**Synodic closure**:
The two conditions a cycle must meet together -- the loop takes exactly `N` synodic
periods (so the next departure sees the same Earth-Jupiter phase), and the 1 AU crossing
lands where Earth actually is. Two knobs meet them, the departure excess speed and the
**free-aim departure** angle, so the solution is *discrete* and the Jovian bend it
demands is an **output** the search may not choose.
_Avoid_: treating the required bend as a free parameter; scoring a closure that met the
clock but not **Earth re-intercept**.

**Bend deficit**:
Required minus available Jovian turn at the perijove floor, the feasibility test a
**synodic closure** either passes or fails. 1S is short by 101.38 deg, 2S by 6.84,
3S has +58.80 deg of margin. _Avoid_: quoting a deficit without its perijove altitude
floor (4,000 km gives 114.68 deg of authority at the 2S arrival, 200 km only 116.13 --
the floor is nearly free and cannot rescue a failing closure).

**Transfer clock ceiling**:
3.661 yr, the longest cycle a zero-revolution direct transfer can fly -- approached at
the departure excess below which the transfer's aphelion no longer reaches Jupiter's
orbit. It is what rules out 4S (4.3682 yr) and every longer multiple, so with 2S failing
on bend, **3S is the only synodic multiple that closes**, bracketed on both sides.
_Avoid_: reading a missing 4S closure as a search failure; forgetting that
multi-revolution arcs, unexplored, are the only route past the ceiling.

**Departure nozzle ledger**:
What the Earth departure costs once the returning stream drives a magnetic nozzle
rather than a bare collision (`departure_nozzle_ledger()`, `cycle_growth_ledger()`,
ADR 0019 addendum). Two corrections turn the **radial-outward push axis** from the
architecture's largest risk into a rounding error. First, the angle the
**impact-angle impulse law** consumes is *not* the **aim separation**: the free
**departure-hyperbola mirror** (+/-20.67 deg at `v_inf` 10.54) plus the vehicle's
own motion take 110.6 deg down to 94 deg. Second, the un-steerable `cos theta`
term has magnitude 1 whatever the **slug ratio**, while the steerable term grows as
`sqrt(k)`, so at `k` = 30 the bulk is a sixth of the impulse and the whole cant
costs under 5 points of delivered mass. At `eta_jet^2` = 0.6 the 3S departure runs
Isp 2214 s and delivers 0.823, spending 0.59% of the vehicle in impactors.
_Avoid_: feeding the SOI **aim separation** into `beta`; scaling the `cos theta`
term by the jet efficiency (momentum conservation on the arriving impactor does not
care how good the nozzle is -- only the exhaust term scales); reading the cant as
expensive without naming `k`.

**Slug ratio is per impactor, not per vehicle**:
The misreading this repository has already made once. `k` = 30 is kilograms of
slug per kilogram of *arriving impactor*, and impactors run under 1% of the
vehicle, so it is not a thirty-to-one propellant fraction. At `eta_jet^2` = 0.7 the
3S departure spends **0.196 kg of slug and 0.0066 kg of impactor per kilogram
delivered** (Isp 2405 s) against the 2.11 kg a 380 s methalox stage would burn for
the same 4.29 km/s -- an order of magnitude leaner, carrying no engine. The cycle
is also nearly blind to the nozzle: doubling time moves 0.543 -> 0.481 yr across
`eta_jet^2` from 0.4 to 1.0, because `beta` grows as `sqrt(eta*(1+k))` and
`dv/v_e` is only 0.18. So **`eta_geom` is not load-bearing here**, unlike ADR
0014/0015 where it decided a device outright.
_Avoid_: reading the **slug ratio** as a propellant mass fraction; calling `k` = 30
a heavy load without converting to slug-per-delivered-kilogram; treating this
cycle's numbers as exposed to the unmeasured `eta_geom`.

**Impactor-scarce accounting**:
Scoring the loop on returning kilograms per *impactor* kilogram, with Earth-launched
slug treated as free -- one impactor kg buys `k` kg of spent slug, which flies
`k*f/(1-f)` kg of vehicle, of which the solar node keeps `s`. It is what makes the
3S cycle grow 83.7x per pass and double in 0.513 yr, the fastest in this repository.
It is also **an assumption, not a result**: the launched slug scales one-for-one, so
an 83.7x cycle is an 83.7x launch-rate cycle and the constraint has moved to the pad
rather than vanished. Deliberately switches off the **launch ledger**.
_Avoid_: comparing an impactor-scarce doubling time against the **doubling time**
figures of ADR 0003/0008/0009/0013, which all charge launched mass or methalox --
the two are different currencies and the ranking between them is meaningless;
rescaling the **return floor** by a cycle's own leverage (that credits the same
efficiency twice -- use the closing speed alone, `return_value_ratio()`); quoting
the un-normalised **launch ledger** as a verdict, since it has no clock in it and
the clock is what separates these cycles.

**Radial-outward push axis**:
Where the returning 150 km/s stream points when it reaches Earth: 2.3 deg off radially
outward (98.55 deg from prograde), because the boosted climb-out is nearly radial -- its
angular momentum is only `r_p * v_p` at 4 solar radii. It is the **push axis** of this
architecture, and it is badly misaligned with every departure the cycle needs (110.6 deg
for 3S, 87.8 for 2S), so the Earth-side collision is heavily canted on the
**impact-angle impulse law**. Note it also misaligns with the *paper's own* dive
injection by 31 deg: a push exactly along the axis cannot make a solar dive at any
magnitude, since it would take 161 km/s and produce an escape.
_Avoid_: quoting this architecture's Earth-push mass ratio without saying it is the
uncanted upper bound; assuming the paper's solar-dive cycle is free of the same problem.

### Solar-dive depth trade

What backing the dive perihelion out from 4 solar radii costs and buys
(`src/solar_dive_depth_trade.py`, `make dive-depth`, ADR
`0020-the-nozzle-cap-is-the-expansion-not-the-ignition`). The **Jovian solar-dive
cycle** is a trajectory result; this asks whether the node it flies through can be
built, and prices the answer.

**Expansion floor**:
The condition that the plume still be *conducting* when the nozzle has finished
taking its work out: `eps_th * (1 - eta_jet^2 (1+k)/k) >= ` the **ignition bill**
(`expansion_residual()`). The **plume ignition window** is a condition at the instant
of the merge; a magnetic nozzle works by letting the plume *expand*, which cools it,
so a blob on the window's upper root lights and then decouples the moment the field
draws on it. As a window on `k` it sits strictly inside the ignition window on
**both** sides -- [1.93, 16.03] against [0.021, 47.88] at 91.78 km/s -- and vanishes
entirely below 64.96 km/s at `eta_jet^2` = 0.60, where ignition still permits `k` up
to 22 (`expansion_limited_slug_ratio_window()`).
_Avoid_: reading the **plume ignition window** as sufficient (it is necessary only);
using `1 - eta` for the residual instead of `1 - eta(1+k)/k` (that is 18% too
*lenient* at `k` = 8.5 -- see **jet energy fraction**); quoting a slug ratio near
either root as an operating point; letting an optimiser sit on this floor either --
a boundary is not a design point, which is why `constrained_growth_optimum()` takes
a margin, stated at 1.5 because the two reference cycles carry 1.76 and 1.55.

**Split push** (**two-wave departure**):
Charging the payload's whole trip from the pad rather than from a free parking orbit,
and splitting it because the two halves want opposite geometries (`split_push_ledger()`).
The **launch ledger** assumes a 4.09 km/s ballistic lob, about 3.6 km/s at the 200 km
burn point; `cycle_growth_ledger` started the burn at 11.0086 km/s, already at escape.
Nothing charged the ~7.4 km/s between them. The fix: an **overtaking push** (`theta` = 0,
`beta` = 3.39 at `k` = 8.5, the impactor's momentum *adding*) from the lob to a parking
orbit, an **apoapsis reversal** to re-aim, then the short canted departure leg
(`beta` = 1.67). Reference numbers, node survival 1/2 and 1/3: **4 R☉ 23.0x per cycle,
doubling 0.724 yr, millionfold 14.4 yr; 32 R☉ 3.49x, 1.816 yr, 36.2 yr.**
_Avoid_: quoting the free-parking-orbit figures (0.513 / 1.094 yr — both wrong for the
same reason); assuming one canted push is equivalent (it costs 1.4x the growth at 32 R☉).

**Parking-orbit period trade**:
What the **split push** actually trades. The apoapsis re-aim costs `2 v_apo sin(cant/2)`,
and `v_apo` grows fast as the period shortens — 0.207 km/s at 20 days, 4.271 at 6 hours.
But periapsis speed is dominated by `2 mu / r_p`, so even a 6-hour ellipse reaches
9.87 km/s (90% of escape), and growth **saturates**: 5 days captures 95% of what 40 days
offers while Earth advances only 4.93°. Below ~0.5 days the re-aim eats the whole
advantage and one push is better. `DEFAULT_PARKING_PERIOD_DAYS` = 5.
_Avoid_: rejecting the split on phasing grounds by anchoring on the 20-day
`PUFFSAT_CYCLE_ORBIT_PERIOD` default — the coast can be far shorter at little cost.

**Conduction reserve**:
The merge energy the jet may not spend if the plume is to still conduct at nozzle exit
(`conduction_reserve()`). **A bracket, not a number.** The conservative end reserves the
whole **ignition bill** (84.41 MJ/kg, plume still at 15,000 K); the hard end reserves only
the frozen chemistry (53.47), and is defensible because ADR 0016's finding is that
recombination *freezes* past the lip -- if dissociation does not recombine, neither does
the seed's ionisation, so the electrons survive the cooling and Spitzer conductivity falls
only as `T^1.5`. The nozzle floor is itself a design choice: a potassium-seeded plume is
usually taken to stay workable near 6,000 K, which gives 65.85. Every reading clears the
0.60 the solar-dive cycles are scored at, so ADR 0020's numbers do not depend on it; the
two-leg legs at 46-74 km/s do. Closed by the exit magnetic Reynolds number
`Rm = mu0 sigma v L >~ 1`, uncomputed.
_Avoid_: quoting the conservative end as *the* ceiling; assuming the plume must hold
15,000 K all the way down (that is sufficient, not necessary).

**Jet energy fraction**:
`eta_jet^2 (1+k)/k`, the share of the merge's *internal* energy the jet carries off,
as against `eta_jet^2` itself which is defined per `sec:jet_efficiency` against the
ideal one-axis budget `w^2/2` (`jet_energy_fraction()`). 0.671 at `k` = 8.5 rather
than 0.600. It diverges as `k` falls, and below `k = eta/(1 - eta)` the impulse law
asks for more than the collision dissipates -- which is what puts a *lower* root on
the **expansion floor**.
_Avoid_: treating `eta_jet^2` and this as the same number; the error is lenient.

**Available jet efficiency**:
`eta_jet^2 <= (k/(1+k)) (1 - bill/eps_th)`, the ceiling the **expansion floor** puts
on a *stated* efficiency (`maximum_jet_efficiency()`). **Not a new bound** -- ADR 0016
already ceilinged the same parameter with the frozen-chemistry argument, implemented
as `plume_thermal.chemistry_efficiency()`. This one is strictly tighter everywhere
because it asks that the plume still conduct rather than that the jet stay positive:
0.759 against 0.874 at 390.5 MJ/kg, and 0.003 against 0.423 at 84.7. They agree on
every verdict.
_Avoid_: presenting this as new (it is ADR 0016's bound sharpened); treating
`eta_jet^2` as a free sweep parameter at low merge energy.

**Derived periapsis survival**:
Survival at the dive node computed from the **impact-angle impulse law** head-on at
the **opposing-stream closing speed**, rather than stated as a constant
(`dive_node()`). Reproduces `jovian_solar_dive_cycle`'s stated 0.60 at 4 solar radii
to 0.4% (0.5977), which is what licenses comparing depths at all. The node's exhaust
speed is `beta*w/k` and `w` is twice the near-parabolic perihelion speed, so backing
out cuts the exhaust speed *faster* than it cuts the boost -- survival falls even
though less delta-v is being bought.
_Avoid_: comparing a derived-survival row against a stated-survival one; assuming a
gentler node is a cheaper one.

**Opposing-stream closing speed**:
Twice the perihelion speed of a near-parabolic fall from Jupiter's orbit -- 616.6 km/s
at 4 solar radii, 215.3 at 32. Sets the dive node's exhaust speed, its merge energy,
and the rendezvous timing tolerance (which scales as `1/w`: 162 us against 464 us for
a 100 m along-track miss).

**Depth trade**:
A 64x drop in solar flux (3.93 MW/m^2 to 61.5 kW/m^2, 2041 K to 722 K equilibrium)
and an 8.2x gentler collision cost **2.51x the doubling time** -- 0.724 yr at
4 solar radii against 1.816 at 32, on the **split push** with the departure charged from
the pad. Per-*cycle* growth falls 6.6x (23.0 to 3.49 kg per impactor kg), the number
to avoid quoting alone: the clock is unchanged at 3.276 yr either way, so the
logarithm absorbs most of it.
_Avoid_: quoting the per-cycle ratio as the cost; comparing rows at different slug
ratios without saying which ceiling each sits under.

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
- The **push axis** and **free-aim departure** are reconciled only by the
  **apoapsis reversal**, which only a bound parking orbit provides — so the
  **closed cycle**'s sub-escape push target is forced by aim, not by propulsion.
- The **push axis** is why the growth push needs no **aim separation** term: the
  payload goes where it is pushed, so that collision sits at 0° on the
  **impact-angle impulse law** by construction. Only the departure burn is charged
  an angle.
- The **free-aim ceiling** and the **head-on crossover** bound the same question
  from two sides — how much a better angle could ever be worth, and the `k` past
  which "better" means head-on.
- The **impact-angle impulse law** has one endpoint at each end of the cycle:
  the **growth push** sits at 0° (`1 + sqrt(1+k)`, and at `k = 0` the paper's
  elastic plate), the departure nozzle at 180° (`sqrt(1+k) − 1`, and at `k = 0`
  no device at all). **Slug ratio**, **closing speed** and **recovery** are the
  three inputs both ends share.
- The **split gap** and the **apoapsis reversal** are one knob read twice:
  the gap is the parking-orbit period, and the reversal is sized from that
  period. Neither can be chosen without the other.
- The **two-wave split** and the **doubling time** compound: the split decides
  whether growth is linear or square-root in the same per-cycle quantities, so
  it moves doubling more than any propulsion parameter in the model.
- The **overtaking nozzle** and the pusher plate are the same algebra at
  different `k`: as `k1 -> 0`, `k1/sigma1 -> 2 e1 / Lambda`, which *is*
  `payload_mass_ratio()` with **recovery** in the role of the elasticity `f`.
  The identity is arithmetic only — the **plume ignition window**'s lower root
  keeps a real nozzle away from that limit, so the plate remains a device the
  nozzle cannot become.
- The **plume ignition window** and the **launch ledger** bound `k1` from the
  same side but for opposite reasons: one says the plume goes cold when the
  energy is spread too thin, the other that the slug had to be launched. Across
  the swept grid the ledger is what actually binds; the window binds only in the
  high-recovery corner.
- The **equivalent plate elasticity** is independent of `e2` to within a few
  percent, because leg 2 is common to both architectures — so "when does a
  nozzle beat a plate" is a question about `e1` alone.
- The **dive-placement floor** and the **Tisserand `v_b` ceiling** are the same
  invariant read at two targets: a flyby cannot rescale the excess, so one bounds how
  *slow* the post-flyby tangential speed can be made (to fall to the Sun) and the other
  how *retrograde* it can be made (to close fast on Earth).
- A **synodic closure** fixes the clock and **Earth re-intercept**; the **bend deficit**
  then reports whether the flyby can pay for them. The three are one test in three
  parts, and only the third can fail once the first two are solved.
- The **transfer clock ceiling** and the **bend deficit** bracket the **Jovian
  solar-dive cycle** from opposite sides: short clocks demand more bend than Jupiter
  has, long clocks demand a slower transfer than exists.
- The **departure nozzle ledger** is what makes the **radial-outward push axis**
  survivable: the same misalignment that costs 53% at `k` = 3 costs under 5 points
  at `k` = 30, because the **slug ratio** dilutes the un-steerable bulk term.
- **Impactor-scarce accounting** and the **launch ledger** are the same ledger with
  one term switched off. Every headline in the ADR 0019 addendum is stated in the
  first and would have to be restated in the second before it can be ranked against
  anything else in this repository.
- The **radial-outward push axis** is the **push axis** of the near-Sun architecture,
  and it constrains the *paper's* solar-dive cycle as well as the **Jovian solar-dive
  cycle** -- the two differ in how much cant they need, not in whether they need it.

## Example dialogue

> **Dev:** "Where does a scenario's **mass ratio** come from — does the **scenario table** compute it?"
> **Domain expert:** "No. A **PuffSat scenario** knows its own **mass ratio** from its three velocities. The **scenario table** just projects the catalog for display; it computes nothing."
> **Dev:** "Then where does the lunar-return row's ratio come from? It's a ¾/¼ blend, not a single collision."
> **Domain expert:** "That's the **lunar-return optimum** — it isn't a **PuffSat scenario** at all, so it doesn't belong in the **scenario table**. Present it separately."

## Flagged ambiguities
- ~~**The Jovian solar-dive cycle's Earth push is quoted uncanted.**~~ **Resolved**
  by the ADR 0019 addendum, and it reversed sign: the diagnosis that the cant was
  the largest open risk was wrong twice over. The angle was the SOI **aim
  separation** rather than the vehicle-frame one (110.6 deg against 94), and at
  `k` = 30 the un-steerable bulk term is a sixth of the impulse rather than a third.
  The cant costs under 5 points of delivered mass, and cancelling the retro impactor
  costs 0.6. The parking-orbit-plus-methalox alternative is not needed.
- ~~**The Jovian solar-dive cycle's growth is scored only in the currency that
  flatters it.**~~ **Resolved** by ADR 0019's second addendum, and the **launch
  ledger** turned out to cut neither way. Charged as committed, the 3S cycle returns
  0.1254 kg per kilogram off the pad and clears the 1/15 floor 1.88x while the paper's
  own dive fails at 0.72x -- but that floor was calibrated on a ~60 km/s return, and
  these close at 157.8, worth 2.80x more payload pushed to a common target. Restated
  at 1/42, **both clear**, so the failure was an artefact. And the floor carries no
  clock: time-normalised it gives 0.0383 against 0.0540 kg per pad kg per year, the
  paper's dive ahead, agreeing with doubling time. The ranking is unchanged.
- **A monatomic slug may retire the dissociation toll, and the model cannot adjudicate.**
  Argon pays no atomisation, and one kilogram of it is 1.49e25 atoms against water's
  1.00e26, so every per-particle toll shrinks 6.7x -- dissociation, translation and
  ionisation alike. The bill falls 84.41 -> 2.16 MJ/kg at 6,000 K. But argon's own
  ionisation becomes the new frozen toll (37.7 MJ/kg singly, 103.8 to Ar2+) and at leg-1
  merge energies of 145-195 MJ/kg the plume is past Ar1+, so leg 1's ceiling spans 0.36
  to 0.91 depending on the Saha state, which nobody has solved for argon. Water has the
  same problem hidden: the **ignition bill** sets `ionisation_fraction = 0` by design.
  Counterweights the impulse law cannot see: Larmor radius goes as `sqrt(m)`, so argon is
  1.6x harder to magnetise than oxygen; and ADR 0018's ice-plug bag and `cruise_thermal.py`
  are water-specific, though a cryogen is far easier at 32 R_sun's 722 K than 4 R_sun's
  2041 K. See the **ignition bill**'s own _Avoid_: this model always picks the heaviest
  species, so it cannot be used to choose one.
- **The two-legged nozzle sweep has not been re-run against the expansion floor.**
  `two_wave_growth.fleet_ignition_windows` uses the ignition-only test, so ADR 0014/0015
  and `two_leg_nozzle_sweep` allow `k1` to 10.21 and `k2` to 23.32 and sweep `eta` freely.
  On the same 11 flown cycles the **expansion floor** caps the *available* `eta_jet^2` on
  leg 1 at **0.430** (`k1` = 2.5) reserving the whole bill and **0.546** (`k1` = 3.4)
  reserving only the frozen chemistry, and on leg 2 at 0.603 / 0.684. Leg 1 misses 0.60
  at *both* ends of the **conduction reserve**, which is the robust part. Whether
  that moves the plate-versus-nozzle verdict is **not** determined: ADR 0016 split
  `recovery` (which scales `beta` whole, and is what `e1`/`e2` set) from the geometric
  factor inside the root, so the `e1` axis is not `eta_jet^2` and the mapping must be
  done first.
- **ADR 0019's growth figures give the parking orbit away free, and so does its
  comparison against the paper.** `cycle_growth_ledger` starts the departure burn at
  Earth escape while the **launch ledger** in the same module assumes a 4.09 km/s lob;
  nothing charges the ~7.4 km/s between. Charged, the 3S Jovian dive doubles in
  **0.724 yr, not 0.513**, and returns 23.0x per cycle, not 83.35 — see **split push**.
  The paper's own single-impulse resonant dive carries the identical defect (its 0.305 yr
  is scored the same way), so the *ranking* is probably unaffected, but neither figure
  should be quoted until both are re-derived on `split_push_ledger()`. Unresolved for the
  paper's row.
- **The magnetic nozzle's mass is uncharged everywhere it is flown.** ADR 0020 shows
  the shallow node handles 8.2x less specific energy (724 against 5934 MJ/kg) and the
  motivating objection was that the energy per impactor kilogram implies a heavy
  nozzle -- but no mass model turns either figure into kilograms, so the case for
  backing the dive out stays qualitative on exactly the axis that motivated it.
  Unresolved, and it applies to `two_leg_nozzle_sweep` and ADR 0014/0015 too.
- **The opposing projectile stream is uncharged at the dive node.** It costs 1.34% of
  the vehicle at 4 solar radii and 2.15% at 32 (`DiveNode.opposing_impactor_fraction`),
  and neither the **impactor-scarce growth** ledger nor the **launch ledger** counts
  it. Both 97.26 and 8.20 kg per impactor kilogram are upper bounds by roughly that
  factor, slightly worse at the shallow node.
- **The paper's "doubling factor at two" does not reproduce.**
  `eq:external_reaction_mass` at `eta_jet` = 0.8 puts the per-cycle survival at the
  4 R_sun node near 0.71, which makes the **single-impulse resonant dive** grow payload
  by ~3.8 per cycle -- yet `sec:earth_reintercept` says the two-impulse loop "holds the
  doubling factor at two" and that the single-impulse dive falls *below* it. Those
  cannot both be right, and the paper's published 17 yr millionfold rests on the
  smaller one. ADR 0019's addendum sidesteps it by charging a stated 40% propellant at
  that node rather than deriving it. Unresolved, and it belongs to the paper repo.
- **ADR 0019's 6.84 deg 2S bend deficit is inside the real-orbit noise.** ADR 0011 found
  real eccentric Jupiter swings the perijove margin by 20,515 km window to window, far
  more than this. So 2S is *unpayable at a price the circular model can see* -- about
  2 km/s -- not proven infeasible. 3S's +58.80 deg margin is the one that should survive
  a real ephemeris. Unresolved for 2S; a `real_orbit_resonance.py`-style audit settles it.

- "The pusher plate is just the `k = 0` nozzle" is true of the impulse law and
  false of the device. Resolved: the **plume ignition window** is two-sided, so
  a magnetic nozzle cannot be run at vanishing slug; the **overtaking nozzle**
  and the plate are separate devices whose formulas happen to converge.

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
- **ADR 0008's quotable 3.6320 yr omits the apoapsis-reversal charge.** The closed
  cycle's departure aim is bought by the **apoapsis reversal** (~234 m/s methalox at
  the 20 d parking orbit, ~112 m/s at 60 d), which no ADR charges. At 20 d it is
  growth ×0.9392, moving the real doubling to ~4.04 yr (+11%); a 60 d orbit halves
  the charge but forces the trip under 3.11 yr to keep the 3-window cycle, so the
  right period is a re-optimization, not a lookup. The ranking vs the chains is
  unaffected (they'd pay it too); the *published* number is what moves. Unresolved.
- **The capture-fraction derating is a head-on derivation applied to canted
  impacts.** The slug/nozzle capture fraction (`beta_bare/beta_ideal` = 0.31 at
  `k = 3`, 0.92 tamped) is derived for a plume centre-of-mass drifting *along* the
  dish axis. A canted impact drifts the blob off-axis, out of the capture cone, and
  nothing models that. It can only penalize the canted case, so every gain in ADR
  0012 is an upper bound in that respect too. Unresolved, and load-bearing for any
  future aim-steering claim.
- **The 2S aim floor is a clock artifact, not a wall.** The 2S family is pinned to
  149.3-179.8° of **aim separation** across a 721 x 401 grid, but a phasing-free
  envelope (energy and angular momentum only, recipe in ADR 0012, *not* committed
  code) admits ~73° at the same speeds. The lock comes from fitting both legs inside
  two synodic periods on zero-revolution arcs, not from physics; 4S+ with
  multi-revolution arcs is unexplored. Deliberately not pursued — ADR 0012's
  **free-aim ceiling** makes the prize too small — but do not quote 149° as a
  physical limit.
- **`f` and `e` are the same number wearing two hats.** `astro_constants.py`
  defines `STD_FUDGE_FACTOR = 0.8` as "how elastic PuffSat collisions are"; the
  slug models reuse that value as **recovery**, a collimation claim about a
  plume. The `k → 0` limit matches identically, which is what makes the
  substitution defensible, but they are different hardware claims and no source
  calibrates either. ADR 0013 now sweeps both independently (`f` 0.5–0.8, `e`
  0.25–0.9) and finds them worth comparable amounts, so neither should be
  quoted without the other. Unresolved.
  The `f = 0.8` extrapolation is now load-bearing for a *decision*, not just a
  number: ADR 0014 and ADR 0015 reach opposite verdicts purely from whether the
  plate is benchmarked at 0.8 or at the nozzle's own recovery. Any future
  measurement of plate restitution above ~20 km/s settles the leg-1 device choice.
- **The two-wave split's cost is a DSM proxy, not a deep-space maneuver.** ADR
  0013's headline — 8 of 11 flown cycles buy their **split gap** for under
  1 m/s — is priced as an exact velocity match at the Jupiter patched-conic
  seam (ADR 0011's proxy), not a finite-location interplanetary burn. It is the
  claim in that ADR most exposed to a real trajectory optimization, and it is
  load-bearing for the whole two-wave architecture. Unresolved.
- **An all-failed optimizer table is not a result.** It is ambiguous between an empty
  feasible set (physics) and a search that never found it (artifact), and the two look
  identical. Random-sample the box first: if blind sampling finds feasible points at a
  rate the optimizer should trivially beat, the harness is broken. This is not
  hypothetical — the first phased run reported "NO PHASED SOLUTION" for all five
  sequences while 2–11% of random points flew a complete chain (ADR 0008, "How this was
  nearly recorded backwards").
