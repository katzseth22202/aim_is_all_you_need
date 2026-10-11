# Plate-only growth with a 100-tonne rotating-tether throw

Exploratory first-pass sizing, not a qualified tether or a revised $/kg growth
ledger. User-selected catch mass: **100 tonnes total**, including the remaining
methalox stage and fuel. The proposed paper conclusion is recorded for review
in [ADR 0048](adr/0048-defer-the-rotating-tether-as-a-paper-baseline.md).
Reproduce with:

```sh
PYTHONPATH=. .venv/bin/python docs/tether_plate_analysis.py
```

## Kinematics: the proposed speed increment is plausible

Assume circular 400 km LEO, a 10 km radial arm, a 3 km/s tip, and an effectively
massive hub. The backward tip matches the payload's velocity; this is **not**
a collision at 3 km/s. Catch and release half a turn apart give an ideal 6 km/s
increase. Tip acceleration is 91.77 g; half-turn time is 10.47 s. At 100 g the
minimum radius is 9.18 km. The hub itself moves 3 km/s faster than circular LEO.

The instantaneous two-body construction puts the hub at 410 km periapsis,
10.6686 km/s, in a bound ellipse with 213,290 km apogee radius and 4.2045-day
period. Ideal release at 420 km gives Earth escape excess speed 8.3404 km/s.
Reaching the ordinary 3S departure excess speed 11.5643 km/s then requires
**2.1744 km/s** of impulsive prograde methalox burn. At 380 s specific impulse,
100 t becomes 55.79 t after propellant expenditure, before subtracting the
stage dry mass. The hotter-return early route instead needs 3.8971 km/s.

These are instantaneous heavy-hub estimates, not a propagated capture/release
trajectory. A finite hub recoils and its rotation changes. An illustrative
two-point-mass elastic reversal gives payload increment 2u M/(M+m): a 20:1 hub
delivers 5.714 km/s and loses 286 m/s; a 100:1 hub delivers 5.941 km/s and loses
59.4 m/s. This analogy is not a flexible-tether simulation.

For growth, the plate must deliver cargo to the appropriate circular LEO for
this same catch geometry. A payload already near escape cannot simply receive
an additional 3 km/s from this bound-hub construction. Circular coasting offers
an opportunity to change departure direction, but the plane, phase, pickup
window, plate delivery, and outgoing Jupiter target remain unclosed.

## Fibre mass: high acceleration shortens the arm, not its mass ratio

For constant allowable tensile stress sigma, density rho, rotation speed u,
and tip payload m, define k = rho u^2/(2 sigma). Ignoring Earth tides, an
optimally tapered radial arm has root/tip area ratio exp(k) and fibre mass

`m_arm/m = sqrt(pi*k) * exp(k) * erf(sqrt(k))`.

Its tip area is `m*u^2/(R*sigma)`. Thus the mass ratio is independent of arm
length at fixed tip speed. This is a static tensile lower bound, not a dynamic
strength assessment. With illustrative density 970 kg/m^3:

| Allowable stress | Root/tip area ratio | One loaded arm | Two equivalent arms |
|---|---:|---:|---:|
| 1.5 GPa | 18.36 | 5,462 t | 10,925 t |
| 1.75 GPa | 12.11 | 3,304 t | 6,609 t |
| 2 GPa | 8.87 | 2,237 t | 4,474 t |
| 3 GPa | 4.28 | 835 t | 1,671 t |

These are **allowable operating stresses**, not ultimate fibre strengths. The
1.75 GPa illustration is half the 3.5 GPa polyethylene strength cited in the
repository's bag material model, not evidence that a complete tether can safely
operate there. The 3 GPa row is an aggressive sensitivity. Creep, defects,
joints, radiation, debris redundancy, capture shock, and fatigue need margins.
The symmetric two-arm benchmark needs an opposing load/counterweight; it does
not optimize a one-arm/ballast architecture. Hub, end hardware, counterweight,
and power/propulsion equipment are additional masses.

At an illustrative $500/kg delivered to LEO, the 1.75 GPa two-arm fibre alone
costs $3.30 billion to launch. That excludes manufacturing and orbital raising;
it is not an amortized cost per delivered kilogram or a program cost estimate.
For scale only, the ordinary-3S 3000 K reference seed carries 133.26 t of
PuffSats/rods and incurs about $1.047 billion of dear-case seed transport.
The proposed plate-only seed has different material needs, so this is not an
equal-service comparison, but it shows why the tether cannot be treated as a
small uncharged addition to the seed.

## SEP restores momentum after every throw

The facility gives the payload orbital energy and momentum. Reboost is needed
after each throw, not only during initial deployment. With 2000 s SEP specific
impulse and 50% electrical-to-jet efficiency, a leading-order 100 t x 6 km/s
momentum replacement consumes **30.59 t propellant and 3.269 GWh electricity**.
This omits steering losses and simultaneous restoration of orbit and spin.

Illustrative battery usable energy is 200 Wh/kg; array specific power 150 W/kg.
One 90-minute burn per orbit, distributing a throw's recovery over N burns:

| N | Thruster electrical power | Battery mass | Nominal N x original period |
|---|---:|---:|---:|
| 1 | 2,179 MW | 16,344 t | 4.2 days |
| 10 | 218 MW | 1,634 t | 42 days |
| 100 | 21.8 MW | 163 t | 420 days |

Array averages would be 32.4/N MW (216/N tonnes). These are energy bookkeeping
scalings, **not a recovery schedule**: recoil changes the orbit, thrust changes
it again, and lunar/solar perturbations and rendezvous constrain the timing.
Thruster/PPU mass, battery power limits, charging losses, eclipse, heat rejection,
life, and reserve energy are not included. Jet inefficiency cannot all be
assigned to spacecraft radiators because some waste energy leaves in the plume.
Battery sizing assumes it supplies the whole burn, without credit for direct
solar input during that burn. This matters more for the shorter-period cases.
More burns reduce peak power and battery capacity, not total energy/propellant.

A 90-minute arc is not all at periapsis speed. Even on the final unthrusted
ellipse, +/-45 minutes reaches 12,076 km altitude: mean speed over the arc is
8.298 km/s rather than 10.669 km/s. The impulse-only reboost estimate therefore
does not prove restoration of the required orbital energy and angular momentum.
At initial LEO, 90 minutes spans almost an entire orbit.

Deployment must also accelerate the battery and the tether. The energy-limited
delta-v per battery discharge is approximately
`2*efficiency*battery_specific_energy/exhaust_speed * battery_mass/total_mass`,
or **36.71 m/s times battery mass fraction** here. Even a physically impossible
100%-battery station needs over 80 such discharges for 3 km/s; a realistic
station needs many more, with changing orbital period and non-impulsive losses.

## Lower tip speed is worth comparing

Holding 100 t catch mass, 10 km radius, 1.75 GPa allowable stress, and ten
recovery burns fixed gives the following heavy-hub sensitivities:

| Tip speed | Two-arm fibre | Chemical top-up | Ideal retained mass | Ten original orbit periods |
|---|---:|---:|---:|---:|
| 1.5 km/s | 384 t | 5.174 km/s | 24.9 t | 1.50 days |
| 2.0 km/s | 977 t | 4.174 km/s | 32.6 t | 2.47 days |
| 2.5 km/s | 2,472 t | 3.174 km/s | 42.7 t | 5.51 days |
| 3.0 km/s | 6,609 t | 2.174 km/s | 55.8 t | 42.05 days |

Battery capacity scales linearly with tip speed at fixed number of burns:
817/1090/1362/1634 tonnes in these rows. Faster cycling requires greater average
array power. Lower station/payload ratios make recoil more important, so the
ideal increments cannot simply be assumed at the lighter end. These are not
equal-throughput or equal-delivered-mass comparisons, nor cost optima.

## What this establishes, and what it does not

The 10 km radius and approximate 2 km/s chemical top-up are credible starting
numbers. The difficult scaling is tether specific strength, station mass, and
recovering momentum at the required cadence. A cheaper seed or $200/kg program
has **not** been established. Charging the station to the seed is essential.
Smaller packets reduce instantaneous capacity requirements but require more
throws; they do not eliminate total reboost energy or propellant.

The next useful design comparison is tip speed versus fibre mass and chemical
top-up, with a specified annual throughput and a fully charged station. It
should precede an integrated growth-ledger claim.

Historical context: NASA studied momentum-exchange/reboost tethers, including
velocity-matched LEO pickup, half-turn release, and restoration of facility
momentum. See [Hyperbolic Injection Issues for MXER Tethers (2003)](https://ntrs.nasa.gov/citations/20040000478)
and [2006 MXER development status](https://ntrs.nasa.gov/citations/20060047739).
Those studies used electrodynamic reboost (coupling to Earth's field), not
this battery/SEP sizing, and do not validate a present-day 3 km/s-tip system.
