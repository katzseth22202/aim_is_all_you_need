# The head-on impact penalty is bounded, and above k ~ 17 it is not a penalty

Status: accepted

Date: 2026-08-17

Extends: ADR 0009 (the head-on nozzle's `beta = sqrt(1+k) - 1` is one endpoint of
a single angle-dependent law; its "18.1 degrees of Earth hyperbola bend" is now
priced). Everything in ADR 0009 stands.

## Context

The 2S and 3S Jupiter resonances (ADR 0011) both deliver their departure burn
with ADR 0009's head-on nozzle: PuffSats returning from Jupiter meet the
outbound vehicle nose-to-nose, so the fireball's centre of mass is kicked
*backwards* at `w/(1+k)` before it can do any useful work. That charge is the
whole `-1` in `beta_head(k) = sqrt(1+k) - 1`, and at Appendix D's `k = 3` it
halves the ideal collimated impulse.

The charge is only that large because the impactors arrive almost exactly
opposed to the direction the next departure needs. The design question: can a
small extra departure delta-v buy a resonance whose returning stream arrives at
a *larger* angle to the required thrust axis, so only part of the incoming
momentum is charged?

Two things had to be settled before that could be answered.

**The impulse law is one law, not two.** Balancing momentum and energy for an
impact at angle `theta` to the thrust axis, with the exhaust canted just enough
to leave the vehicle no transverse kick, gives

    beta(theta, k) = sqrt(1 + k - sin^2 theta) + cos theta

per arriving impactor kilogram. At `theta = 180` this is ADR 0009's head-on
nozzle `sqrt(1+k) - 1`; at `theta = 0` it is the growth push's along-axis
`1 + sqrt(1+k)`. The repo's two impulse laws are the two endpoints of this one.

**The angle that matters is not the one at the sphere of influence.** Both
quantities the law consumes -- the closing speed `w` and the bias angle -- are
properties of the impactor's velocity *relative to the vehicle*, which is moving
at 10.95 to ~18 km/s during the burn. Two corrections follow. The vehicle's own
motion rotates the angle (157.3 -> 160.6 degrees for 2S) and sets `w`; and the
departure hyperbola's periapsis velocity is rotated off its outgoing excess
vector by half the Earth turn, `arcsin(1/e)` -- 13.9 degrees (2S), 18.2 degrees
(3S) -- whose mirror sign is a free design choice. That rotation is the same
`~18.1 degrees` ADR 0009 already counted; it had never been priced.

## Decision

**The aim-steering line of inquiry is closed.** It is capped by an argument that
needs no trajectory search, and the cap is small.

`src/circular_resonance_impulse.py` is committed as the reproducible harness
(`make resonance-impulse`; not part of `make all`). It enumerates exact,
strictly unpowered, integer-synodic circular closures by rooting the Jovian
incoming/outgoing excess-speed mismatch to zero, then scores each one on the
**payload fraction surviving the departure burn**, integrating the variable
effective exhaust speed `v_e = eta * beta * w / k` across the burn in the
vehicle frame with the better hyperbola mirror applied. The patched-conic aim
angle is retained as a labelled diagnostic only: it is neither of the two
quantities the impulse law uses.

Search box, recorded per the ADR 0007 lesson: 361 departure phases x 301
encounter times, departure `v_inf <= 25 km/s` (the production powered-flyby
cap), perijove floor 4,000 km altitude, zero-revolution Lambert arcs on both
legs, `k = 3.0`, `eta = 0.8`, 401-point burn quadrature. Continuous SLSQP
refinement from the 8 best coarse starts under the same constraints.

## Consequences

- **Reconfirmed: both families arrive near head-on.** The 2S minimum-departure
  closure sits at Earth-minus-Jupiter `-87.41151`, bit-identical to ADR 0011's
  audited phase, with aim 157.27 degrees at the SOI (92.2% of the incoming
  momentum backward along the wanted thrust axis) and 160.6 degrees in the
  vehicle frame. 3S is at 144.90 / 150.4 degrees, 81.8% backward. Assuming an
  exactly head-on impact is worth **0.5% (2S) / 0.9% (3S)** in delivered mass,
  so ADR 0009's assumption is quantitatively, not merely qualitatively, right.

- **2S is aim-locked; 3S is not.** The least-backward 2S closure is 149.18
  degrees and it costs the full 25 km/s departure cap; the whole family lies
  between 149.3 and 179.8 degrees on a 721 x 401 coarse grid
  (`--phase-samples 721 --encounter-samples 401`), so the lock is not a grid
  artifact. The 3S family walks 144.9 -> 129.6 -> 118.2 -> 102.1 degrees for
  +0.03 / +1.07 / +3.46 km/s of burn, then jumps branch to a *forward-biased*
  66.0 degrees at +4.29 km/s (`v_b` collapsing 61.2 -> 49.0). Reproduce that
  curve by sweeping `--max-departure-vinf` over 11.6, 12.0, 12.5, 13.0, 14.0,
  15.0, 16.0, 17.0, 18.0, 20.0, 25.0 and reading the least-backward row.

- **Scored on mass, the best buy is a few hundred m/s and worth ~3%.** The
  optimum is 3S at phase `-99.707`, departure `v_inf` 11.755 km/s (burn 5.155,
  only +0.139 over the family minimum), delivered-mass gain **x1.0293**. The 2S
  optimum is x1.0140 at +0.011 km/s. Pushing to the 25 km/s cap for the
  55.5-degree forward-biased 3S impact raises the *per-row* gain to x1.216 but
  drops delivered mass to 0.538 against 0.784 -- the extra departure burn costs
  far more than the aim relief returns.

- **The ceiling: if the angle were free, it would be worth 17% at k = 3 and 5%
  at k = 10.** Charging nothing at all for the aim and taking the best possible
  angle -- a perfectly overtaking impact -- gives x1.1656 (2S) / x1.1312 (3S) at
  `k = 3`, x1.1051 / x1.0787 at ADR 0009's `k* = 7.057`, and x1.0721 / x1.0498
  at `k = 10`. No trajectory, resonance, phasing or synodic multiple can beat a
  free angle, so this bounds every scheme of this kind. The searched optimum
  captures under 30% of even that bound.

- **Above `k = 18.47` (2S) / `16.43` (3S), head-on is itself the optimum.**
  `v_e` goes as `beta * w`; canting raises `beta` but lowers `w`, because a
  head-on impact *adds* the vehicle's own speed to the impactor's. The bias term
  `cos theta` is independent of `k` while `sqrt(1+k)` grows with it, so past
  that crossover the closing-speed loss wins outright and canting can only lose.
  ADR 0009's two-currency `k* = 6-12` sits close enough to that crossover that
  the prize is already down to 3-11%.

- **Why the gain is so small even though `beta` triples.** `d(beta)/d(cant) = 0`
  at head-on, so the gain is *quadratic* in the cant -- 30 degrees buys 7% at
  `k = 3`. The closing speed falls linearly against it. And the burn is only
  5-6.7 km/s against `v_e` ~19-21 km/s, so delivered mass sits at 0.73-0.78 and
  is insensitive to even a large `v_e` improvement.

- **The optimum is bang-bang, and the good end is unreachable.** `beta * w` is
  maximized at `theta = 0` or `theta = 180`, never between
  (`free_aim_ceiling()` scans 1,801 angles and always returns an endpoint). The
  forward end needs an *overtaking* stream, and that is capped by solar escape,
  not by search effort: a bound heliocentric arrival at 1 AU has speed at most
  `sqrt(2 mu_sun / 1 AU)` = 42.12 km/s, so a *prograde* arrival has
  `v_inf <= 42.12 - 29.78 = 12.34 km/s` while a *retrograde* one reaches
  `42.12 + 29.78 = 71.90`. High closing speed and forward bias are antagonistic
  by construction. This is the same wall that made ADR 0009's
  overtaking-pusher-plate rows 1-D artifacts.

- **The free mirror is the only piece that pays, and it is bigger than the
  searched angle.** Choosing the departure hyperbola's mirror to rotate the
  thrust axis *toward* the incoming stream is worth **x1.012 (2S) / x1.020 (3S)**
  at zero propellant -- more than any delta-v purchase on the trade curve. It is
  already implied by ADR 0009's 18.1 degrees; it is now priced and applied.

- **The 149-degree 2S floor is a resonance-clock artifact, not a physics wall.**
  A phasing-free envelope admits aim angles near 73 degrees at the 2S baseline
  speeds. That envelope is a **scratch estimate, not committed code**; its
  recipe, recorded here rather than left in a vanished harness: propagate a
  post-flyby state at 5.2 AU with `|v - v_J| = 15.369 km/s`, sweeping the flyby
  direction over the full circle, conserving energy and angular momentum to the
  inbound 1 AU crossing; accept departure directions whose conic apoapsis
  reaches 5.2 AU; take the smallest included angle between the two excess
  vectors. Nothing in it knows where a planet is, so it is a necessary bound
  only -- it cannot exhibit a trajectory. A cleverer resonance therefore may
  exist, plausibly at 4S+ where multi-revolution arcs open up. It is
  deliberately not being looked for: even a *free* 73 degrees is worth x1.099 at
  `k = 3` and x1.040 at `k = 10`, both under the ceiling this ADR closes on.

- **Untested interaction, flagged not modelled.** The handoff's capture-fraction
  derating (`beta_bare/beta_ideal = 0.31` at `k = 3`, 0.92 tamped) is derived for
  a centre-of-mass drift *along* the dish axis. A canted impact drifts the blob
  off-axis, out of the capture cone. The effect is unmodelled and can only
  penalize the canted case, so every gain quoted here is an upper bound in that
  respect too.

- The growth push is untouched. It is along-axis by construction -- the payload
  goes where it is pushed -- so `theta = 0` there already and none of this
  applies to it. This ADR is about the departure burn only.
