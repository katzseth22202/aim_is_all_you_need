# The Earth turn, the 3000 K chamber and recovered hardware, for the parent paper

Raised 2026-10-12 in `aim_is_all_you_need` from the October 9-10 `orbit_turn_analysis` work
(ADRs 0045-0049). Reproduce the geometry and the 3000 K burns with `make orbit-turn` (~35 s);
`pytest tests/test_orbit_turn_analysis.py` asserts every number below that comes from it.
The geometry is the **circular** three-synodic (3S) return, two-body and planar; the paper's
chain is the real-orbit mixed 2S/3S one. Treat angles below as the 3S case only, not a
rescore of the chain.

Status of each item for the paper:

| # | item | for the paper? |
|---|---|---|
| 1 | Earth occults the favorable 3S mirror; the arrival angle is 164.5 deg, not 150.4 | **yes, a correction** |
| 2 | Earth turns the vehicle during the burn, but through head-on, not away from it | **yes** |
| 3 | A miss on the mass-optimal 600 km burn can hit Earth | **yes** |
| 4 | An overtake-only plate cannot make the Earth departure | optional, one sentence |
| 5 | The 3000 K hydrogen chamber | **physics yes; it does not close on the chain** |
| 6 | Why a pulsed chamber should beat a pulse detonation engine | **yes, with a caveat** |
| 7 | Recycling spent tanks and chamber into rods (ADR 0050) | **yes, a sensitivity and a "why now"** |
| 8 | Store the hydrogen charge as slush (ADR 0051) | **yes** |
| 9 | Which chamber flies first, and at what pulse rate | **yes, framing** |
| - | Rotating tether (ADR 0048), 3000/3500 K economics (ADRs 0046-0047) | no |

## 1. The favorable 3S mirror is behind Earth (ADR 0045)

The departure hyperbola leaves the same way with either sign of its ~18 deg bend, and ADR 0012
picked the sign that turns the thrust axis toward the incoming stream (x1.020 at 3S). That
choice only works if the stream can reach the burn point. At the 600 km burn it cannot. The
favorable mirror's best projectile hyperbola has its periapsis **1,905 km below the surface**:
the projectile would have to pass through Earth to arrive. The incoming direction would have
to rotate 27.2 deg just to graze. The other mirror is visible: its stream is still inbound at
intercept, with periapsis 235 km above the surface.

**Correction to `fig:earth_turnaround`, panel (a)(3).** "The PuffSats arrive at \ang{150.4} to
the thrust" used the favorable mirror. On the circular 3S return the visible mirror gives
**\ang{164.5}** at the 600 km burn, and the ideal plume cant falls from 19.7 deg to 7.7 deg.
The caption's figure comes from the chain, not the circular return, so run the visibility
check on the chain's 3S cycles before replacing 150.4 with a final number. The 2S figure
(\ang{160.6}) has not been checked for visibility.

Proposed wording, after the inset sentence:

> Only one of the two mirror-image departure hyperbolae is usable. On the other, the incoming
> PuffSats would have to pass through Earth to reach the burn point.

## 2. Earth turns the vehicle during the burn, but through head-on

A 20-minute burn is not an impulse. While it runs, Earth bends the vehicle's velocity by tens
of degrees, so the stream's arrival angle changes. It does not stay favorable:

| 20-minute burn | off head-on, start -> end | time more than 22 deg off | largest plume cant | mass gain over exactly head-on |
|---|---:|---:|---:|---:|
| fixed direction (what a chamber flies) | 19.8 -> 11.7 deg | 0% | 9.8 deg | 0.30% |
| velocity-steered (sensitivity only) | 3.3 -> 23.2 deg | 11.9% | 11.3 deg | 0.27% |

The best placement is half the burn before the periapsis of the unpowered orbit and half after
(599.7/600.3 s). The powered path's own closest approach shifts earlier, so say which periapsis
is meant.

Proposed wording, in the departure discussion:

> Earth's gravity turns the payload through tens of degrees during a twenty-minute burn, but
> it swings the stream through head-on rather than away from it. A fixed-direction burn stays
> within 20 degrees of head-on throughout, and the angle is worth under 0.3 percent of the
> delivered mass. The burn is centered on the periapsis of the unpowered orbit.

## 3. A miss on the mass-optimal burn can hit Earth

Reaching the vehicle is not the same as being safe if missed. On the mass-optimal 600 km burn,
the later projectiles are caught inbound; a projectile that missed would continue to a
periapsis up to 2,983 km below the surface. That is **43% of the stream by mass**. Requiring
every miss to keep a 600 km periapsis raises the parking periapsis to 2,631 km and puts the
whole burn before periapsis. In the 3000 K model of item 5, that costs **7.3% of delivered
mass** (0.4221 -> 0.3914). The same floor, applied to the paper's chamber, has not been run.

Proposed wording:

> A projectile that misses its intercept keeps flying. On the lowest-energy departure, nearly
> half the stream would then strike Earth. Raising the parking periapsis to about 2,600 km
> and firing the whole burn before periapsis keeps every miss above 600 km. That costs about
> 7 percent of the delivered mass, a price we accept.

## 4. An overtake-only plate cannot make the Earth departure (optional)

The paper never proposes a plate departure, so this only answers "why not just use the
plate?" Include it if that question needs answering.

We gave a 4 g plate every advantage: perfect momentum transfer, no carried-mass penalty, a
free ignition point and a stream that overtakes it exactly. It reaches the 3S energy in 147.5 s,
but it leaves in the wrong direction. The plate can push only along the stream it rides, so
the vehicle leaves roughly along the stream's direction, bent only by Earth's gravity. The
next Jupiter departure needs a direction 144.9 deg from the incoming stream. Earth bends the
plate's path 25.1 deg, leaving it **120 deg off**. Even a ballistic flyby at 600 km bends at
most 34.8 deg at this excess speed; a 144.9 deg bend would need a periapsis 6,233 km below the
surface. More thrust makes it worse, because it leaves gravity less time to turn the path.
This is not an occultation problem: every path clears Earth.

This rules out a plate-only Earth departure **from this stream**. It says nothing against the
plate's 400 km push of the growth wave, which does not need the turn.

The growth push is unaffected. It only has to add energy, raising the payload to the
613,000 km apoapsis, where a 0.22-0.46 km/s burn sets the departure orientation. In the
4 g case the collision stayed within 2.3 deg of a perfect overtake, and every projectile
path, missed ones included, cleared Earth by more than 2,000 km. Miss safety for the
paper's own 400 km push geometry has not been run.

Proposed wording, near the two-device figure (`fig:` at line ~107 of `templateArxiv.tex`):

> A pusher plate can only be overtaken, so it leaves along the stream that drives it. The
> next Jupiter departure needs a direction about 145 degrees away, and Earth can bend a
> departing path by at most 35. The departure therefore needs a head-on device.

## 5. The 3000 K hydrogen chamber: the physics holds, the prices are not ready

The idea: run the head-on hydrogen chamber at 3000 K, the temperature NTR fuel and walls were
tested to, instead of 5500 K. What holds up:

- **No recombination store.** At 3000 K and 5 kg/m3 only 0.32% of the hydrogen is atomic
  (Cantera equilibrium). The nozzle need not be long enough to wait for H + H to rejoin, so
  it can open to area ratio 50-100 instead of 300. At a fixed 8.7 m exit that empties the
  chamber **3-6 times faster** (95% out in 27-54 ms, against 162 ms) and still gives 13-26
  acoustic turnovers, inside the ten-turnover mixing screen. (ADR 0049.)
- **Thrust is worth buying.** With the miss floor of item 3, each doubling of burn time costs
  about 8.5% of delivered mass: 39.1% at 1,200 s, 35.7% at 2,500 s, 32.8% at 5,000 s.
  A fast blowdown permits a higher pulse rate, so a shorter burn and more Oberth gain. The
  higher rate itself is not demonstrated: refill, gate reset and wall recovery set it.

What does **not** hold, and must not reach the paper:

- **The energy was overstated.** The October 10 runs scaled the 5500 K energy linearly to
  3000 K (57.3 MJ/kg, 9.10 km/s exhaust). About a third of the 5500 K energy is dissociation
  that 3000 K hydrogen lacks. Equilibrium gives **34.5 MJ/kg and 7.06 km/s** (85% of the
  8.31 km/s ideal). Every 3000 K price in ADRs 0046-0047 is void.
- **A pulsed chamber is not an NTR at the same temperature.** A sealed chamber that blows down
  can turn at most its internal energy `u` into exhaust. A steady-flow NTR collects the
  enthalpy `h = gamma u`. At equal peak temperature the pulsed chamber's ideal exhaust is
  about `1/sqrt(gamma)`, **0.86**, of the NTR's: 8.3 km/s against 9.7 at 3000 K. NTR heritage
  shows what the wall survives, not what the chamber's specific impulse is. A pulsed wall
  sees the peak only briefly, which could allow a hotter peak, but that is not yet credited.

**On the paper's chain it does not grow (ADR 0052).** Charged like the 5500 K
chamber, the 3000 K chamber has `k = 79` at 75 km/s, 1,220 bar peak and a 48 t wall.
It reaches only 606-625 s at 85% of ideal velocity, and about 750 s even at perfect
efficiency. Its growth per cycle is 0.83 as liquid and 1.03 as slush with recovered
tanks. The two-synodic cycles fall below 1. The cap on temperature caps the energy
per kilogram at 34 MJ, against 105 at 5500 K. A third of the 5500 K energy is the
dissociation store the cooler chamber was chosen to avoid. NTR heritage tells us
what a wall survives, not how hot the chamber must run.

Proposed wording, in `sec:ntr_departure`:

> The NTR's fuel and walls have been tested to about 3000 K, which suggests a cooler walled
> chamber at that temperature. Hydrogen at 3000 K is almost all molecular, so the nozzle has
> no recombination to wait for and can be several times shorter, which empties the chamber
> several times faster. It does not inherit the NTR's specific impulse, though. A sealed
> chamber that empties through its nozzle can turn only the gas's internal energy into
> exhaust, while a steady rocket also collects the flow work. At the same temperature that
> costs the pulsed chamber about 14 percent of exhaust speed.

## 6. Why the chamber should beat a pulse detonation engine

Nothing on this was committed in October; this is the argument as it stands. A pulse
detonation engine (PDE) also turns repeated closed-volume energy release into thrust, and
measured PDE performance falls well short of ideal. The chamber differs where those losses
arise:

- **Time to equilibrate.** A PDE tube fires and blows down in a few milliseconds, behind a
  non-uniform rarefaction. Hydrogen at 3000 K crosses the 3.4 m chamber in under a
  millisecond, while blowdown takes 27-162 ms (item 5): 13-77 turnovers by ADR 0049's
  sealed-volume heuristic. The gas leaves well mixed, near thermal equilibrium.
- **Size.** Wall loss scales with surface per volume. A 20 m3 sphere has 1.8 m2 of wall per m3;
  a 5 cm PDE tube has about 80, roughly 45 times more. Our parts are not small.
- **A real nozzle.** A PDE usually exhausts from an open tube end; the chamber expands its gas
  through a converging-diverging nozzle.

The caveat: the chamber shares the PDE's basic limit against a steady rocket. Both are
blowdowns, so both have the internal-energy budget of item 5. The argument places the chamber
above a PDE, not level with an NTR. The PDE loss figures need citations before this goes in.

Proposed wording:

> Pulse detonation engines also convert repeated closed-volume heating into thrust, and
> fall short of their ideal performance. Most of the shortfall comes from small tubes that
> lose heat to their walls and empty before the gas settles. A twenty-cubic-meter chamber
> avoids both. Its gas crosses the chamber dozens of times before it leaves, and it has
> about one-fortieth of a detonation tube's wall per unit volume. It still pays the price
> every blowdown pays against a steady rocket.

## 7. The rocket eats its own spent hardware (ADR 0050)

After the departure burn the empty tanks and the expended chamber ride on toward
Jupiter with the payload anyway. A 5 t processor in the stack cuts and packages them
during the three-year cruise, about 35 kg a day, and they come back as **departure
rods**. They cannot simply become growth PuffSats, which must disperse into gas
before the plate. Rods hit the chamber at 60-75 km/s and vaporize, and a denser rod
needs a shallower, lighter plug, not a heavier one. `make hardware-recovery` runs
this on the paper's chain, behind the spray cup, with the lob climbing (Estimate
book, 10%, 50% odds):

| | growth per cycle | steady $/kg | break-even, cheap / dear seed |
|---|---:|---:|---:|
| methane AR 300, none | 1.905 | 162 | 340 / 810 |
| methane, 90% tanks + 60% steel chamber | **2.178** | **136** | **255 / 528** |
| hydrogen, none | 2.418 | 122 | 192 / 377 |
| hydrogen, 90% tanks (chamber liner is copper) | **2.723** | **110** | **161 / 283** |

If the 44 km/s-plus Jovian-return plate can take recovered pieces as impactors that
spread water PuffSats over it, growth rises further: methane 2.25, hydrogen 2.91.
That is a fallback we may need, not a modeled design.

**Build it to be recycled.**
- Pressure management is carbon fibre in PEEK or PEKK, not aluminium. Recycled
  aluminium would match the meteoric background, 142 t a year, at about five
  departures a year.
- No aluminium-lithium or copper alloys anywhere that ends up as exhaust.
- The methane chamber's steel is recycled. Iron is what meteors already deliver most
  of, 2,477 t a year.
- Nozzle backing is carbon fibre or carbon-carbon.
- Steel tabs on the planned cut lines let robots grip the pieces magnetically.

**Exhaust is the cost.** Every recycled kilogram is eventually fired, and most
departure exhaust is slower than orbital speed and falls back. The natural
comparison is meteoric ablation, about 12,000 t a year, which also recondenses as
nanometre smoke. Volcanic ash is not comparable: it is micron silicate with its
metal bound inside, and almost all of it falls out within days. Per seed's fleet, a
methane program matches nature's iron at about 60 seeds. Hydrogen's carbon-fibre
tanks would match today's rocket soot at about 13 seeds, but only if every carbon
atom became soot, which overstates it. The design's methane and polyethylene rods
already carry far more carbon than the tanks.

Proposed wording:

> The spent tanks and chamber need not be thrown away. They ride on toward Jupiter
> with the payload, and over the three-year cruise a small robotic processor cuts
> them into departure rods. The rocket eats part of itself. Raising growth per cycle
> from 1.9 to 2.2 cuts the methane chamber's delivered cost from \$162 to \$136 per
> kilogram. We build the hardware for it, from carbon fibre and steel rather than
> aluminium. Every recycled kilogram eventually leaves in the exhaust, and most
> exhaust falls back. Meteors already deliver about 2,500 tonnes of iron a year
> to the upper atmosphere, so a steel-recycling fleet stays below that background
> until the program has grown from some sixty seeds.

Cite Schulz et al. 2025 (arXiv 2510.21328) for the meteoric and re-entry tables and
Maloney et al. 2022 for rocket soot. Citations for in-space robotic cutting are still
owed.

## 8. Store the hydrogen charge as slush (ADR 0051)

Carry the hydrogen as 50% solid slush instead of normal-boiling-point liquid. NASA
made, pumped and expelled it at scale for the National Aero-Space Plane (Lewis
K-Site, 1990-94). It is 15% denser and carries 18% more heat sink. The tank shrinks
in proportion, from 0.205 to 0.178 kg per kg. Melting the solid and warming the
liquid absorb about 81 kJ/kg, so the same insulation boils nothing over the 20-day
hold, against 6% for liquid. `make slush-hydrogen`:

| hydrogen chamber | growth per cycle | steady $/kg | break-even, cheap / dear seed |
|---|---:|---:|---:|
| liquid (paper today) | 2.418 | 122 | 192 / 377 |
| slush | 2.560 | 115 | 174 / 323 |
| slush, tanks recycled (item 7) | **2.877** | **105** | **147 / 245** |

Heavier 5% cryostats give back about a seventh of the gain. A $2/kg slush premium
costs $0.4/kg at L1. Solid hydrogen would not lighten the tank further. Liquid
hydrogen tanks are already low pressure; their mass is volume, insulation and
launch loads. Slush's gain is density and heat sink.

Proposed wording, in the hydrogen chamber's departure section:

> We carry the hydrogen as slush, half frozen, as NASA did for the National
> Aero-Space Plane. It is 15 percent denser than liquid, so the tank shrinks, and
> melting its ice soaks up the heat that would otherwise boil 6 percent of the
> charge away over the twenty-day hold. Together the two raise growth per cycle
> from 2.42 to 2.56.

Cite McNelis et al., "A Summary of the Slush Hydrogen Technology Program for the
National Aero-Space Plane" (NTRS 19950017766), and Hardy and Whalen's transfer and
expulsion reports (NASA TM 105596, TM 105191).

## 9. Which chamber flies first, and at what pulse rate

> **Amended 2026-10-13 (ADR 0055).** The hydrogen figures below are the 20 m3 sphere's
> at 4 Hz, whose wall is unresolved. The bulged hydrogen chamber the impact simulation
> solved runs at 2 Hz and leads methane by only $10-25/kg with recovered hardware. The
> methane chamber's impulse and throat also changed. See
> `docs/hydrocode_answers_for_parent.md` items 3, 5 and 7.

**Performance target: 5500 K hydrogen, stored as slush, with recycled tanks.** It
grows 2.88 per cycle at $105/kg. Its advantage over methane is recombination at
the same area-ratio-300 nozzle, not a shorter nozzle. Hydrogen keeps 36% of the
pulse in bonds and the nozzle returns 94% of it, a ceiling of 0.978. Methane at
7000 K keeps 69% in bonds and gets back only 52%, because carbon never re-bonds;
its ceiling is 0.669. The hot, light hydrogen also has a higher sound speed, so the
same geometry empties faster.

**Early version: methane, at 2 Hz.** Until a hydrogen wall passes the bench tests
(`sec:chamber_wall_tests`), the survivable methane chamber flies first; the paper
already calls the hydrogen wall unverified. Be realistic about its rate. The
survivable chamber is bulged to 212 m3 at about 94 bar, ten times the sphere's
volume. It fires 5 kg rods at 2 Hz, because 4 Hz cannot refill it. The paper's
$162/kg already flies it at 2 Hz. The ledger's burns are 24-37 minutes (2,900-4,500
pulses), not the 20-minute reference of the Earth-turn study. Each burn is charged
its integrated finite-burn loss, 7-12% of the burn.

Explain the blowdown problem in the paper. A pulse cannot start until the last one's
gas has left and the next charge is in. Emptying time grows with chamber volume and
falls with throat area and sound speed. A large, low-pressure methane chamber with
heavy products empties slowly. Hydrogen's light gas empties fast, and a cooler,
undissociated charge could use a wider throat (ADR 0049). Fewer pulses per second
mean a longer burn, and a longer burn wastes more of the Oberth gain. In the
Earth-turn study, each doubling of burn time cost about 8.5% of delivered mass.
For the methane chamber, quote that study's 2,500 s case, not the 1,200 s
reference.

**The blowdown, computed (ADR 0053).** The methane chamber reaches its ~1 bar fill
pressure 278-336 ms after the pulse. That leaves 164-222 ms of the 500 ms period
for films, plug, fill and lock, so 2 Hz fits. At 4 Hz the blowdown alone overruns
the 250 ms period. The 20 m3 hydrogen sphere needs 135 ms at 4 Hz, which is tight
but fits. A bulged hydrogen chamber would likely run at 2 Hz. For hydrogen slush
with recycled tanks, that drops growth from 2.88 to 2.60 and raises the cost from
$105 to $116/kg. Shrinking the methane chamber cuts blowdown in proportion: 142 ms
at 106 m3, 73 ms at 53 m3. That would open 4 Hz, if the companion simulation finds
the smaller chamber's wall survives.

Proposed wording:

> The methane chamber is the one we can build first, and it is slow. Its bulged
> 212-cubic-metre shell cannot be emptied and recharged faster than twice a second,
> so a departure takes 25 to 37 minutes instead of 20, and the burn spends more of
> its time away from periapsis. We charge that loss in full. The hydrogen chamber
> empties faster and recovers more of its energy, and stored as slush with its
> tanks recycled it is the better machine. It flies once its wall is shown to
> survive.
