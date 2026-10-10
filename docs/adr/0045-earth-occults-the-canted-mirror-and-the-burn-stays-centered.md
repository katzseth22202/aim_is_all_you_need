# Earth occults the canted 3S mirror, and a 20-minute burn stays centered

Status: proposed

Date: 2026-10-09

## Context

ADR 0012 scores the circular three-synodic Jupiter return as an impulse at Earth.
It gives the outgoing departure hyperbola either mirror sign and chooses the sign
that makes the returning PuffSats less nearly head-on.  At `k = 3` that geometry
asks the magnetic exhaust to cant by about 20-22 degrees.  Two facts hidden by the
impulse need checking:

1. a real departure burn takes time, during which Earth turns the vehicle's
   velocity relative to the incoming stream; and
2. the stream has to reach the chosen side of the 600 km burn point without Earth
   lying in the way.

The second is not a small correction.  A patched-conic aim angle says where the
velocity vectors point at infinity.  It does not prove that the corresponding
Earth hyperbola remains above the surface.

## Decision

Add `src/orbit_turn_analysis.py` and `make orbit-turn` as the reproducible harness.
Use the minimum-departure circular 3S closure behind ADR 0012:

```
departure v_inf       11.5642794307 km/s
return v_inf          55.1783413091 km/s
aim separation       144.899950401 deg
parking orbit         20 days, 600 km periapsis
finite burn           1,200 s
reference exhaust     11.0 km/s
angle ledger          k = 3, collimation = 0.8
```

At each candidate burn point, solve both two-body projectile hyperbolae from the
fixed incoming asymptote.  A projectile route is visible if the intercept occurs
while it is still inbound, or, if it is already outbound, its passed periapsis is
above Earth.  This criterion is the hyperbolic version of the line-of-sight test.

For the finite burn, use constant mass flow and constant reference exhaust speed,
which is constant thrust before the vehicle's changing mass is divided out.  Solve
the extra ideal burn required to match the impulsive departure energy, and optimize
the start time.  Report both:

- **fixed direction**, the head-on chamber constraint used by the growth ledger;
- **velocity-steered**, the energy-best sensitivity a chamber cannot necessarily
  fly.

The incoming PuffSats share one asymptotic direction but are individually targeted
onto the moving vehicle.  Earth focusing is recomputed at every sample.  This is
still planar, circular and two-body; it is not a real-ephemeris intercept design.

Also solve the follow-on **uncanted 3000 K chamber** sensitivity.  It differs from
the fixed-`k` angle ledger above:

- the projectile is externally supplied and no plug is carried;
- its impact energy heats the projectile plus hydrogen to 3000 K;
- hot-mixture energy is anchored to the repository's 5500 K hydrogen pairing and
  scaled linearly with temperature, giving 57.328 MJ/kg;
- ideal exhaust is 10.708 km/s and the nozzle supplies 85% of that **velocity**,
  9.102 km/s (72.25% of ideal directed kinetic energy);
- hydrogen loading changes with closing speed as
  `k = w^2 / (2 u_hot) - 1`; and
- the axial nozzle does not cancel the projectile's lateral momentum.

For projectile-relative velocity vector `w_vec` and an axial thrust unit vector
`a`, the impulse per kilogram of onboard hydrogen is

```
((k + 1) / k) v_e a + w_vec / k .
```

The second term retains the projectile's full signed momentum: overtaking helps,
head-on hurts and an oblique hit kicks sideways.  Hold hydrogen mass flow constant
for 1,200 seconds.  Optimize initial-orbit orientation, placement about periapsis
and hydrogen consumption while requiring both the magnitude and direction of the
final 3S excess vector.  On the front-side case, also optimize parking periapsis
altitude subject to projectile clearance throughout the burn.

Keep two safety questions separate.  **Visibility to interception** asks whether a
projectile reaches the vehicle before Earth.  **Miss safety** continues the same
hyperbola after a hypothetical miss.  Report the mass optimum under the first rule,
then re-optimize both mirrors with every missed projectile required to retain a
600 km Earth periapsis.

## Consequences

### The favorable mirror is behind Earth

At 600 km Earth has an angular radius of **66.07 degrees** as seen from the
vehicle.  The two departure mirrors are:

| departure mirror | stream state at intercept | projectile periapsis altitude | impact angle | ideal plume cant | verdict |
|---|---:|---:|---:|---:|---|
| counter-clockwise | inbound, -17.78 km/s radial | +234.9 km | 164.46 deg | 7.70 deg | visible |
| clockwise | outbound, +42.83 km/s radial | **-1,904.6 km** | 137.67 deg | 19.68 deg | **Earth blocks it** |

The negative altitude is not a limb-rounding error.  Rotating the incoming
asymptote in the favorable direction by **27.19 degrees** is required merely to
make that route graze Earth.  Such a rotation changes the Jupiter-return geometry
and is not free steering on the existing closure.

This exposes a sign/visibility gap in ADR 0012's `departure-hyperbola mirror`.
That calculation folds the speed through Earth's gravity but rotates the direction
algebraically, without constructing the incoming Earth hyperbola.  Its quoted free
mirror gain is therefore an upper bound, not a flyable choice.  This proposed ADR
records the correction but does not yet rewrite the circular-family optimizer or
the parent paper's 150.4-degree turnaround figure.

### Earth turn does not keep the burn above 22 degrees

The three angles must not be interchanged:

- **off head-on** is `180 deg - impact angle`, in the vehicle frame;
- **plume cant** is smaller, because only the projectile share's transverse
  momentum has to be carried out in the exhaust;
- **aim separation** is the patched-conic angle at infinity.

For the physically visible mirror:

| 20-minute burn | placement about reference periapsis | finite-burn loss | off head-on, start -> end | time off head-on >22 deg | plume cant range | impulse-mean cant | mass gain vs head-on |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed direction | 599.7 / 600.3 s | 273.3 m/s | 19.81 -> 11.70 deg | **0%** | 5.82-9.76 deg | 7.37 deg | 0.298% |
| velocity-steered | 582.5 / 617.5 s | 211.2 m/s | 3.32 -> 23.16 deg | **11.9%** | 0.01-11.34 deg | 6.74 deg | 0.274% |

So Earth turn is of the expected tens-of-degrees scale, but it swings through
head-on rather than holding the useful large cant.  Even the unbuildably favorable
velocity-steered sensitivity exceeds 22 degrees only in the last 2.4 minutes or
so.  The plume itself never steers beyond 11.34 degrees, and the integrated angle
benefit is under three-tenths of a percent in delivered mass.  It is not a new
performance lever.

### Half before and half after is the right operating rule

Against the periapsis of the **unpowered reference orbit**, fixed-direction thrust
optimizes at **599.7 seconds before and 600.3 seconds after**.  Velocity steering
moves that to **582.5 / 617.5 seconds**.  Re-centering the steered burn from 600/600
to that optimum saves only **0.16 m/s**.  There is no operational reason to schedule
an 18-second asymmetry on this model.

The powered trajectory's actual closest approach is earlier: the fixed burn splits
523/677 seconds about it, and the steered burn 474/726 seconds.  This is not a
contradiction.  Prograde thrust before the reference periapsis changes the orbit and
moves the subsequent closest point.  Burn scheduling should name which periapsis it
means; the usual and useful convention is the osculating unpowered one at ignition.

### Keeping the side kick does not make the front side optimal

At 45 and 73 km/s closing speed, the 3000 K energy rule gives `k = 16.66` and
`45.48`; the optimized burns land inside that bracket.  The vector-closed results
are:

| axial-nozzle solution | parking periapsis | burn before/after periapsis | closing speed | `k` | off head-on | H2 spent / delivered | `m0/mf` | external projectile | total / impulse delta-v | finite penalty | intercept clearance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naturally visible side | **600 km** | 583 / 617 s | 65.82-68.77 km/s | 36.79-40.25 | 4.33 -> 23.65 deg | 0.5008 / **0.4992** | 2.003 | 0.01275 | 5.339 / 5.119 km/s | 220 m/s | 706 km |
| less-head-on front side | **3,866 km** | 0 / 1,200 s | 61.74-66.85 km/s | 32.25-37.97 | 44.82 -> 31.34 deg | 0.5331 / **0.4669** | 2.142 | 0.01524 | 6.056 / 5.796 km/s | 259 m/s | grazing |

The projectile really is a small part of the flow: only 1.28% of initial vehicle
mass arrives externally in the winning case, against 50.08% spent as onboard
hydrogen.  Its uncancelled sideways contribution is nevertheless measurable.  It
integrates to 0.274 km/s in the rotating local frame, changes sign near head-on and
does **not** drive the vehicle into Earth; the powered path bottoms at 706 km.
The finite duration is not hidden: the full thrust-vector integral is 5.339 km/s,
220 m/s above the 5.119 km/s impulse, and the propagated endpoint closes on the
specified 11.56427943 km/s excess vector in both speed and direction.

The front side does benefit from its less-head-on momentum term at a fixed altitude,
but it cannot use the instantaneous 3,037 km grazing altitude for an entire finite
burn.  Joint clearance and 3S-vector closure move its optimum to a 3,866 km parking
periapsis and put the whole burn after periapsis.  Lost Oberth leverage then wins:
it delivers **6.5% less vehicle mass** from the same ignition mass, or consumes
about 14% more hydrogen per delivered kilogram (1.142 versus 1.003).

Thus accepting the side kick is sound on the visible side, but raising periapsis to
recover the front side is not worth it under these assumptions.  A true overtaking
encounter is not among the available directions: the fixed 144.90-degree 3S aim
separation constrains both flyable Earth-hyperbola families to remain broadly
head-on.

### Visibility is not miss safety

The 600 km mass optimum is reachable but not fail-safe.  Later projectiles are
intercepted inbound; if they miss, their unconsumed continuations reach as low as
**3,104 km below the surface**.  Weighted by projectile flow, **44.6%** of the
stream is Earth-intersecting after a miss.  Saying that Earth does not occult the
intercept does not license saying every miss flies by harmlessly.

Requiring every missed projectile to keep at least a 600 km periapsis gives:

| miss-safe solution | parking periapsis | burn before/after | `k` | off head-on | H2 propellant / delivered | `m0/mf` | total / impulse delta-v | finite penalty | vehicle/intercept minimum | missed-shot minimum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| visible-side safe | **2,670 km** | 1,200 / 0 s | 34.62-40.74 | 2.48 -> 21.07 deg | 53.14% / **46.86%** | 2.134 | 5.798 / 5.576 km/s | 222 m/s | 3,082 km | **600 km** |
| front-side safe | **4,535 km** | 0 / 1,200 s | 32.13-37.90 | 43.90 -> 31.42 deg | 53.85% / **46.15%** | 2.167 | 6.144 / 5.908 km/s | 235 m/s | 600 km | **600 km** |

The safety floor costs the winning visible-side design **6.1% of delivered mass**
relative to the intercept-only optimum (0.4686 versus 0.4992), but removes the
Earth-impact miss corridor.  It still beats the safe front-side mirror by 1.5% in
delivered mass.  If harmless misses are an architecture requirement, the 2,670 km,
all-before-periapsis solution is the recommendation from this model.

## Limitations and next decision

- The circular ADR 0012 optimizer still applies its algebraic mirror and therefore
  remains optimistic until it consumes the hyperbolic visibility check here.
- The impactors are allowed to retarget their Earth impact parameter pulse by pulse
  while keeping one incoming asymptote.  The guidance cost of doing that is not
  charged.
- Atmosphere, oblateness, the Moon, real ephemerides and out-of-plane geometry are
  absent.  None can recover a route whose two-body periapsis is 1,905 km underground
  without materially changing the incoming state.
- The uncanted sensitivity uses constant **hydrogen** mass flow.  Projectile rate
  changes as `1/k`; a fixed pulse cadence would be a different throttle law.
- Its 3000 K mixed energy is a transparent scaling of the existing 5500 K pairing,
  not a new equilibrium calculation of hydrogen plus a specified projectile
  material.  The projectile composition, plug/gate hardware and finite chamber
  rate could move the absolute delivered fractions.  The full vector momentum,
  Earth clearance and exact 3S closure are included.

## Reproduction

`make orbit-turn`.  `pytest tests/test_orbit_turn_analysis.py -s`.
