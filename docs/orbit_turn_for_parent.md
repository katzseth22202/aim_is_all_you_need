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
| 4 | An overtake-only plate cannot make the Earth departure | **yes, one sentence** |
| 5 | The 3000 K hydrogen chamber | **physics yes, prices no** |
| 6 | Why a pulsed chamber should beat a pulse detonation engine | **yes, with a caveat** |
| 7 | Recovering tank and chamber metal into PuffSats | **yes, as a 5500 K sensitivity** |
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

## 4. An overtake-only plate cannot make the Earth departure

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

## 7. Recovering tank and chamber metal into PuffSats (ADR 0046)

After departure, the tank and the expended chamber are already-launched mass on its way to
Jupiter. Cutting and packaging them during the three-year trip turns them into PuffSat mass
for the next generation. The scenario recovers 90% of tank mass and 60% of chamber and nozzle
mass. It flies a 5 t cutting machine bought new for $1M every flight, and pays ordinary body
and steering costs on recovered mass. On the 5500 K chamber (unaffected by the energy error):

| | before | after |
|---|---:|---:|
| useful onward mass | 336.0 t | 407.7 t |
| growth per 3S cycle | 2.44 | 2.96 |
| operating cost, $/kg at L1 | 108.4 | 94.6 |
| program break-even, cheap / dear seed, $/kg | 177.7 / 297.6 | 130.4 / 181.9 |

Caveats the paper must keep: the recovery fractions and the machine are assumptions, not
designs. Whether recovered metal behaves as projectile or rod material (copper and aluminum
vapor in the chamber, packaging geometry, steering) is untested. Exhaust and impact
performance are held fixed. The figures come from the common-route 3S ledger, not the paper's
chain, and its generator is not committed; only the re-pricing is. Leave the ozone and
electromagnetic-handling notes in ADR 0046 out of the paper.

Proposed wording:

> The spent tank and chamber need not be discarded. Cut up and repackaged during the
> three-year trip, 90 percent of the tank and 60 percent of the chamber would add about a
> fifth to the next generation's PuffSat mass. That raises growth per cycle from 2.4 to 3.0.
> This is an accounting sensitivity; recovering hardware in flight has not been designed.
