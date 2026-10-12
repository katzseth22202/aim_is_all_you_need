# ADR 0049: The 3000 K hydrogen chamber needs its own nozzle

Date: 2026-10-10

Status: Accepted 2026-10-12.  The energy debt it found is paid in
`src/orbit_turn_analysis.py`; the nozzle sweep it asks for is still open.

## Context

The exploratory 3000 K departure calculation inherited hardware from the
5500 K chamber rather than sizing a new nozzle. It carries the 8.7 m exit used
at area ratio 300, assumes an exhaust speed equal to 85% of the ideal speed,
and limits one chamber to 4 Hz. Its hot-mixture energy is the 5500 K reference
scaled linearly by `3000 / 5500`, or 57.328 MJ/kg. The financial comparison
correctly labels the resulting 34.1855 t chamber/nozzle as borrowed hardware,
not a thermal design for the cooler case.

At 3000 K that inheritance is physically mismatched. The high-pressure
hydrogen has scarcely dissociated, so the nozzle does not need a long
expansion to collect an H + H recombination store. A smaller area ratio means
a wider throat at fixed exit area, and a wider throat empties the chamber much
faster. It does not make expansion unnecessary: the nozzle must still convert
sensible and vibrational energy into directed flow, and the chamber must stay
closed long enough for the impact pulse to mix.

The finite burn also needs a precise statement of the returning projectile's
momentum. The vehicle turns substantially during the 1,200 s burn. The
projectile can help the required *sideways turn* during part or all of the
burn, but that must not be described as helping forward thrust.

## Decision for review

Do not treat area ratio 300 as a requirement for the 3000 K chamber. The next
physical design study should sweep area ratios **50, 75 and 100** at the
current 8.7 m exit, with area ratio 30 retained as a diagnostic. The sweep must
solve the following together before changing a production or financial
default:

1. a mixed projectile/hydrogen equation of state at 3000 K instead of the
   linear 5500 K energy scaling;
2. exhaust velocity against area ratio, including ordinary nozzle losses;
3. chamber blowdown and the number of acoustic mixing turnovers;
4. refill, gate-reset and thermal-recovery time; and
5. the resulting pulse-rate ceiling and re-optimized finite burn.

Until that work is done, retain the recorded 4 Hz, 1,200 s comparison as an
exploratory baseline. A shorter blowdown does not by itself authorize a higher
pulse rate or a shorter burn.

Retain the complete signed projectile-momentum term in every rerun. On the
currently selected 1,200 s miss-safe solution, describe the projectile as an
**axial debit that assists the turn laterally**, not as partly forward thrust.

## Hydrogen is almost entirely molecular at 3000 K

As a diagnostic, solve pure hydrogen with Cantera 3.2.0 and `gri30.yaml` at
3000 K and 5.0 kg/m3, approximately 100 kg in the inherited 20 m3 chamber:

```python
gas = ct.Solution("gri30.yaml")
gas.TDX = 3000.0, 5.0, "H2:1"
gas.equilibrate("TV")
```

The ideal-gas equilibrium gives 620.6 bar and mole fractions 0.00636 H and
0.99364 H2. The fraction of hydrogen nuclei carried as free atoms is therefore

```text
X_H / (X_H + 2 X_H2) = 0.00319.
```

Only **0.32% of the hydrogen nuclei are atomic**. Expansion moves equilibrium
farther toward H2 as the gas cools. There is no material dissociation store for
an area-ratio-300 nozzle to recover.

The same diagnostic exposed a modeling debt, now paid.  Relative to molecular
hydrogen at 200 K, the equilibrium gas holds **34.53 MJ/kg** of internal energy,
not the 57.3 MJ/kg the linear scaling gave.  About a third of the 5500 K
pairing's energy sits in dissociation, and scaling by temperature carried that
share down to a gas that has none.  `orbit_turn_analysis.hydrogen_hot_energy`
now solves the equilibrium directly (at 1 kg/m3 it moves under 3%), which sets
the ideal exhaust at 8.31 km/s and the 85%-velocity exhaust at **7.06 km/s**,
not 9.10.  ADR 0045's 3000 K tables are re-solved with it: the selected
miss-safe 1,200 s burn delivers 39.14% of ignition mass, not 46.86%.

Internal energy, not enthalpy, is the right budget.  The chamber is sealed while
the pulse thermalizes and then blows down: the gas left behind does the work of
pushing out what leaves, so a full blowdown to vacuum turns at most `u` into
directed kinetic energy.  A steady-flow nuclear rocket at the same temperature
collects the enthalpy, 46.9 MJ/kg here, `gamma` times more.  At equal wall-limited
peak temperature, a pulsed chamber therefore gives about `1/sqrt(gamma)`, 0.86,
of a steady nuclear rocket's ideal exhaust speed.  NTR materials heritage tells us
what a wall survives, not that a pulsed chamber at that temperature matches NTR
specific impulse.  What can recover some of it is that the wall sees the peak only
briefly in each pulse, which is not yet credited.

## A wider throat shortens blowdown in direct proportion

Use a quasi-steady adiabatic tank model only to size the effect. For chamber
volume `V`, throat area `A*`, initial temperature `T0`, specific gas constant
`R` and constant ratio of specific heats `gamma`, define

```text
Cgamma = (2 / (gamma + 1)) ^ ((gamma + 1) / (2 (gamma - 1)))
lambda = (A* / V) sqrt(gamma R T0) Cgamma
q = (gamma - 1) / 2
m(t) / m0 = (1 + q lambda t) ^ (-1 / q).
```

The table uses `V = 20 m3`, `T0 = 3000 K`, `gamma = 1.2903`,
`R = 4124.48 J/(kg K)`, discharge coefficient one, and the inherited 8.7 m
exit area of 59.4468 m2. It is a scaling estimate, not a transient nozzle
solution.

| Area ratio | Throat area | Throat diameter | 1/e mass time | Time to discharge 95% | Acoustic-turnover heuristic |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.198 m2 | 0.502 m | 46.4 ms | 161.8 ms | 76.7 |
| 100 | 0.594 m2 | 0.870 m | 15.5 ms | 53.9 ms | 25.6 |
| 75 | 0.793 m2 | 1.005 m | 11.6 ms | 40.4 ms | 19.2 |
| 50 | 1.189 m2 | 1.230 m | 7.7 ms | 27.0 ms | 12.8 |
| 30 | 1.982 m2 | 1.588 m | 4.6 ms | 16.2 ms | 7.7 |

At a fixed exit, blowdown time is proportional to area ratio. Area ratio 100
is three times faster than the borrowed geometry; area ratio 50 is six times
faster. The last column applies the existing sealed-volume heuristic
`A_bore / (Cgamma A*)` to the great-circle area of a 20 m3 sphere. The usual
ten-turnover screen passes at 50 and fails at 30. That makes 50--100 the useful
first sweep: it buys most of the timing improvement without assuming that a
very open chamber mixes before it discharges.

The system consequence is conditional. At area ratio 300, discharging 95% of
the mass consumes about 162 ms of the 250 ms period allowed by 4 Hz. Area
ratio 50 reduces that estimate to 27 ms. This creates refill and reset margin,
but the roughly 100 kg hydrogen charge still has to be delivered, the gate has
to cycle, and the wall has to recover. Those operations, not blowdown alone,
set any new cadence.

## The projectile remains a backward axial impulse throughout the turn

The 3000 K vector law per kilogram of onboard hydrogen is

```text
impulse = (1 + 1/k) ve a + w_rel / k,
```

where `a` is the instantaneous velocity/nozzle axis and `w_rel` is the full
projectile-relative velocity vector.  `UncantedThermalBurn` now reports the
projectile's axial share as `integrated_projectile_axial_delta_v`.  For the
selected miss-safe solution at 2,631 km periapsis:

```text
burn duration                         1,200 s, all before reference periapsis
impact angle                          159.01 to 177.54 degrees
projectile axial delta-v             -0.951 km/s
nozzle-only axial delta-v            +6.730 km/s
net axial delta-v                    +5.779 km/s
signed projectile lateral delta-v    +0.226 km/s
integral of thrust-vector magnitude  +5.784 km/s
```

The axial projectile term is negative at every sample.  Integrated with the
constant hydrogen flow and falling vehicle mass, it removes **0.951 km/s**, or
14.1% of the nozzle-only axial impulse, a burn average of **0.79 m/s2
backward**.  It is smaller than the 1.29 km/s recorded before the energy fix
only because the correct, lower energy raises `k` from about 38 to about 63:
each kilogram of hydrogen now meets less projectile.

Earth's turn does provide a credit, but it is transverse.  The projectile adds
0.226 km/s in the sign of the required turn in this all-before-periapsis
solution.  It never becomes forward thrust.  At the unconstrained 600 km mass
optimum (612 s before, 588 s after reference periapsis) the projectile's axial
debit is 0.854 km/s and its net lateral help 0.205 km/s.

Thus Earth's turn barely changes the answer to "how hard does the impactor
knock backward?"  It does not offset the axial debit; it mostly supplies a small
turning impulse that the axial nozzle cannot provide.

## Consequences

- The 3000 K chamber has a plausible 3--6x blowdown improvement available
  before reaching the mixing screen, but no higher cadence has yet been
  demonstrated.
- A wider throat may reduce nozzle and wall dwell as well as emptying time,
  but peak-pressure, local heat-flux and structural consequences remain open.
- The recorded 3000 K economics (ADR 0046, 0047) rest on borrowed hardware and
  the superseded 57.3 MJ/kg energy. They are void until the ledger is rebuilt
  with the equilibrium energy; a nozzle-only geometry edit would give false
  precision.
- Turning helps use the projectile's transverse momentum. It does not reverse
  the head-on momentum debit, which remains roughly 0.85--0.95 km/s across
  the two 1,200 s visible-side solutions.
- Re-solving the equation of state changes `k`, exhaust speed, projectile
  fraction, pulse count and finite-burn trajectory together. None should be
  updated in isolation.

## Reproduction and verification

`make orbit-turn` prints the energy, the re-solved tables and the projectile's
axial share; `pytest tests/test_orbit_turn_analysis.py` asserts them, including
that the 3000 K equilibrium energy is under 61% of the old scaled value.  The
miss-safe result is `uncanted_thermal_burn(False, burn_time=1200 s,
missed_periapsis_floor=600 km)`; the 600 km sensitivity is the same call without
`missed_periapsis_floor`.  The equilibrium diagnostic and blowdown equations
are recorded above, so this ADR does not depend on ignored scratch state.
