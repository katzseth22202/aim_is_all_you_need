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
| 10.8 m (sphere) | 5.40 m | 1.00 | 1.50 | 4.9 kg |
| 16 | 3.62 | 0.99 | 1.82 | 5.9 |
| **23** | **3.02** | **1.19** | **1.90** | **6.2** |
| 32 | 2.56 | 1.41 | 1.94 | 6.3 |
| 50 | 2.05 | 1.76 | 1.97 | 6.4 |

The film column is the **Earth-storage** pressure vessel, which is the only
column that still boils once the leak is solved; from cold storage it is 0 kg
at every length and the handling floor of the addendum is the whole answer.
(It read 2.8-3.7 kg when this ADR was written, which was the Jupiter column of
the *superseded* 4.4%-leak table. See the 2026-08-28 addendum.)

**Nothing in the physics picks a row**, so the launch envelope does: a 5.4 m
bore needs coils about 11 m across and orbital assembly, while a 3 m bore is
about 7 m across and flies up built inside an 8 m fairing. Length is the cheap
dimension for a rocket and diameter the expensive one. **23 m at a 3.02 m bore**,
for a fifth more conductor and 1.3 kg more film.

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
0.73 MJ/kg of warming and melting over 37.5 kg -- 27.4 MJ removed before the
boiling step sees it, which is **20%** of the pulse's 137.5 MJ waste-heat bill
(it was 13% of the 211 MJ bill the superseded leak implied). The heat sink is
worth something only on the leg that still boils: see the 2026-08-28 addendum.

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

## Addendum, 2026-08-26: the film has a floor, and it is not `F`

**`eq:bag_film_mass` sizes a pressure vessel, and the solved leak leaves nothing
for it to hold.** With the field leak at 0.11-2.54% rather than the assumed 4.4%,
the slug boils nothing from cold storage: the vapour fraction `x` is zero, and a
term linear in `x` returns zero film *wherever it is evaluated*. That is the
right answer to the question the equation asks and the wrong answer to "what does
the bag weigh". A bag still has to be manufactured, folded, packed, deployed and
inflated.

### The decision

**The film that flies is `max(pressure vessel, handling floor)`**, where the
floor is area times gauge times density and nothing else --
`bag_state.handling_film_mass()`, quoted at 12.7 um over a 6-25 um band.

At the flown 23 m column the two are the same size, 5.1 kg of handling floor
against 6.2 kg of pressure vessel from Earth storage, which is why nothing
downstream of the old 2.8 kg breaks. From *cold* storage the pressure term is
0 kg and the floor is the whole answer, which is the case the paper now flies.

### Why the scaling matters more than the number

A capsule's membrane area is exactly `2 pi r L`: the hemispherical caps put back
precisely what shortening the cylinder removed. With the volume fixed,
`r ∝ 1/sqrt(L)`, so **area goes as `sqrt(L)` and never saturates**, while the
shape factor `F = (2L + 2r)/(L + 4r/3)` saturates at 2.0. From 16 m to 50 m,
`F` rises 8% and the area rises 77%.

| length | area | film at 12.7 um | `F` |
| ---: | ---: | ---: | ---: |
| 10.8 m (sphere) | 366 m^2 | 4.3 kg | 1.50 |
| 16.0 m | 364 m^2 | 4.3 kg | 1.82 |
| **23.0 m** | 437 m^2 | **5.1 kg** | 1.90 |
| 32.0 m | 515 m^2 | 6.0 kg | 1.94 |
| 50.0 m | 644 m^2 | 7.5 kg | 1.97 |

**This strengthens the bore-for-conductor trade rather than complicating it.**
The trade above already said bore and conductor swap one for one; the film was
treated as nearly flat across it because `F` is. Once the bag is gauge-limited
the column pays a third cost that rises with length, so the launch envelope's
choice of 23 m over 50 m is favoured on film mass too, not merely tolerated.

### Open

- **Seams, ripstop, metallisation and inflation hardware are all excluded**, so
  the floor is a floor and not a bag design. A real membrane is heavier by
  whatever those cost, and nothing here bounds them.
- **The 12.7 um gauge is anchored on Echo 1**, which flew a 30 m sphere of
  half-mil metallised PET, packed and inflated on orbit, in 1960. The anchor is a
  precedent rather than a derivation; the conclusions hold across the whole
  6-25 um band, which is why the band is carried.

## Reproducing the addendum

`make bag-state`, section "D4: the handling floor under the film".

## Addendum, 2026-08-28: the plug takes vapour away, it does not add it

**The plug is a better heat sink than the paper says, and it works in the
opposite direction to the sentence describing it.** `sec:needle_through_fog`
reads: *"The vapor mass lands at 24.5 kg against 23.4 kg without the plug, and
the saturation curve moves the mist from 306 K to 307 K."* Both halves point
the wrong way. The plug is 37.5 kg of extra condensed mass absorbing heat the
pulse was going to deliver anyway, so at a fixed bill it can only **lower** the
vapour mass and **cool** the mist.

The pair was never two configurations. 24.5 kg is `(211 - 213 x 0.73)/2.26` and
23.4 kg is `0.11 x 213` from the superseded table's Jupiter column -- the same
energy pool divided by two different latent heats, on the pre-solve 211 MJ bill.

### The corrected numbers

On the 137.5 MJ bill of the current `tab:bag_state`, at Earth storage, which is
the only column that still boils:

| | without the plug | with the plug |
| --- | ---: | ---: |
| left to boil | 92.8 MJ | **65.4 MJ** |
| vapour mass | 39.3 kg | **27.7 kg** |
| `x` | 0.185 | **0.130** |
| mist | 316.3 K | **309.3 K** |
| film, 23 m column | 6.2 kg | **4.2 kg** |
| what actually flies | 6.2 kg (pressure) | **5.1 kg (handling)** |

The plug removes 27.4 MJ, which is **20%** of the bill rather than the 13% it
was worth against 211 MJ. Cold storage is where the argument stops applying
entirely: from 122 K the slug never finishes melting with or without the plug,
so on that leg the heat sink buys nothing the bag can spend.

**The consequence is a design one, not a wording one.** At the flown 23 m
column the plug drops the pressure vessel to 4.2 kg, which is below the 5.1 kg
handling floor of the addendum above. **On the leg that still boils, the plug is
what retires the pressure vessel** -- the same role cold storage plays on the
other leg. `tab:axial_bag`'s Pressure column is the no-plug case and should stay
that way, but the flown bag is handling-governed at both ends of the burn.

### The loop the table still cuts

`tab:bag_state`'s warming row is not independent of its mist row. "Warming and
melting the slug up to liquid" warms liquid water up to whatever temperature the
mist settles at, and the published 0.73 and 0.21 MJ/kg were closed against
306 K and 328 K -- the mist of the *superseded* 4.4% table. Read back, they
imply 4.10 and 4.14 kJ/kg/K, which is liquid water; the temperatures are the
part that moved.

Solving the two rows together (`bag_state.melting_fixed_point`, reported by
`make bag-converge` as Cut 4) moves Earth storage to `x` = 0.202 at 318.2 K and
5.35 kg of film, half a kilogram heavier, and gives the Jupiter column a slush
at 275.5 K carrying 3.8 kg of vapour and 0.4 kg of film rather than a dry bag.

**Reported, not applied.** The paper prints the warming row as an input, both
gaps are under a kilogram of film, and neither overturns "cold storage removes
the pressure vessel", which is what the section rests on. It is recorded because
the 0.73 row is the last place in this cascade where a superseded number is
still doing work.

## Reproducing the second addendum

`make bag-state`, sections "sec:needle_through_fog: the plug as a heat sink" and
"The one loop the table still cuts"; `make bag-converge`, Cut 4.
