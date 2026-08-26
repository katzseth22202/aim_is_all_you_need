# The slug bag is an axial capsule with an ice plug, and the projectile stays compact

Status: accepted

Date: 2026-08-26

## Context

The bag holds 213 kg of slug spread over 659.6 m^3 so the plume has something to
be a plume *in*. As a sphere that is 5.4 m in radius, and it does not work: a
25 kg ice rod crossing it meets only 3.5 kg/m^2 of areal density against its own
3183, outweighing what it meets by a factor of nine hundred. It surrenders a
tenth of a percent of its momentum, leaves the far side intact, and has swept
27 g of the 213 kg. **A needle through fog.**

There are two ways out, and the paper takes the second: widen the projectile
until it covers the bag, or narrow the bag until it matches the projectile.

## Decision

**Narrow the bag into an axial capsule, keep the projectile compact, and put a
solid ice plug at the end the projectile enters.**

### The bore-for-conductor trade

The standoff volume is set by `PV = nR_gT` and does not care about shape, so
pouring 660 m^3 into a column of length `l` gives a bore
`r = sqrt(V / pi l)`. Bore falls as the inverse square root of length; the
conductor a solenoid needs runs as `B r l`, which grows as its square root.
**Bore and conductor trade inversely, one for one: halving the bore doubles the
tape.** Structure mass does not enter at all, because the virial floor tracks
contained energy and contained energy is the `n R_g T` that shape leaves alone.

`tab:axial_bag` reproduces to every cell (`make bag-state`):

| length | bore | conductor | `F` | film |
| ---: | ---: | ---: | ---: | ---: |
| 10.8 m (sphere) | 5.40 m | 1.00 | 1.50 | 2.8 kg |
| 16 | 3.62 | 0.99 | 1.82 | 3.4 |
| **23** | **3.02** | **1.19** | **1.90** | **3.6** |
| 32 | 2.56 | 1.41 | 1.94 | 3.6 |
| 50 | 2.05 | 1.76 | 1.97 | 3.7 |

**Nothing in the physics picks a row**, so the launch envelope does: a 5.4 m
bore needs coils about 11 m across and orbital assembly, while a 3 m bore is
about 7 m across and flies up built inside an 8 m fairing. Length is the cheap
dimension for a rocket and diameter the expensive one. **23 m at a 3.02 m bore**,
for a fifth more conductor and 0.8 kg more film.

### The plug, and why the projectile stays compact

A narrow bag alone still does not couple: crossing 23 m of 0.32 kg/m^3 mist
meets 7.4 kg/m^2, still nine hundred times thinner than the rod. What couples
them is a **solid ice plug** at the entrance. Hydrodynamic penetration says a
projectile striking a target of comparable density erodes over about its own
length times `sqrt(rho_p/rho_t)`, and ice on ice makes that square root one, so
the plug must be at least as thick as the projectile is long. Sized at 37.5 kg,
half again the projectile's mass, with the excess in width because the crater
runs wider than the rod.

**The plug is close to free twice over.** It adds nothing to the film, because
`eq:bag_film_mass` counts only the vapour fraction and condensed water exerts no
pressure. And it works as a heat sink on the way to being a target, absorbing
0.73 MJ/kg of warming and melting over 37.5 kg -- 27 MJ removed before the
boiling step sees it.

### The aperture argument, restated because the paper's mechanism is wrong

The paper rejects the wide projectile on the ground that a head-on nozzle is a
mirror with a hole where the projectile entered, and "anything left open leaks
it": 0.15 m into a 28 m^2 bore is 0.25% open, against 64% for a front spanning
0.8 of the bore.

**That mechanism does not hold for a magnetic mirror**, whose leak is a
loss-cone property set by the mirror ratio rather than by physical open area.
What genuinely does leave ballistically through a hole is the **un-ionised**
fraction, which the field cannot steer, and at the cold leg that is most of the
plume (`f` = 0.06 at 15 165 K). The conclusion survives; the reason changes.

**And the conclusion does not need the aperture argument at all.** Because the
snowplow front is self-widening (ADR 0016 addendum 2), a compact 0.15 m arrival
still sweeps most of the bag: `k` = **7.21**, which sits inside ADR 0016's
tolled optimum band of 6.75-7.77, where the 8.60 a wide arrival delivers
overshoots it. **Compact wins on delivered chain growth before any argument
about holes is made** -- 99.4% of the achievable optimum against 91.7%.

## Consequences

- **`k = 8.5` is an output of this geometry, not an input.** The full-bore sweep
  gives `rho A L / m` = 8.69, and the delivered value is that times the swept
  fraction. Anything that changes the column changes `k`.
- **The design lever is bag size, not bag shape.** Shape costs 1.6 kg of film
  out of 213 kg of slug across the whole `F` range. Resizing the bag to the
  tolled optimum is worth 10-22% of the slug and up to 9% more growth.
- **The plug position is decided by pressure, not convenience.** It sits at the
  projectile's entrance on both legs -- the far end on the departure burn, the
  throat end on the growth push -- because a throat-end plug lets the fireball
  snowplow the column first and arrives needing **7.6 T** against a ship-end
  plug's **56 T** (ADR 0016; `nozzle_geometry.mirror_stagnation`).
- **The snowplow gives the nozzle the shape it should have anyway**: a strong
  field at the chamber and a weak one at the throat, ~20 T a metre in falling to
  5 T at the exit, which the spherical bag never produced.

### Open

- **The projectile's delivery dispersion is still unstated.** The arrival radius
  itself is nearly irrelevant now that the front is known to self-widen, but the
  aperture must still admit whatever actually arrives, and nothing writes down
  what that is.
- **Whether the widening front stays a coherent plow** or breaks into fingers is
  a 2D question. `puffsat_impact_simulation`'s Study 2 found coupling holds with
  two orders of margin, so it is not positioned to overturn this.

## Reproducing

`make bag-state` for the axial table, `make nozzle-geom` for the sweep and the
plug trade.
