# The plume state is solved, and 15 000 K turns out to be the coldest pulse's answer

Status: accepted

Date: 2026-08-26

## Context

`sec:watering_it_down` sizes the bag around a plume at 15 000 K. That number
began as an assumption, and the paper is candid that it did: the collision
delivers 265 MJ/kg per kilogram of merged blob at the hottest pulse against the
85.1 MJ/kg that vaporising, dissociating and heating to 15 000 K costs, so
180 MJ/kg is unaccounted for.

The paper's first instinct was to put it into ionisation by division -- stripping
one electron from each of water's three atoms takes 219 MJ/kg, so 180/219 would
make the plume four fifths ionised. **That is wrong, and the paper says so
itself**: the gas does not choose its ionisation fraction independently of its
temperature. Saha fixes it once temperature and density are given, and at
15 000 K and 0.32 kg/m^3 it allows only **5.9%**. The plume cannot pour
180 MJ/kg into a channel that narrow, so the temperature has to climb until the
channel opens.

That makes the state a root-find rather than an assumption, and it is one this
repository should not be doing: `puffsat_impact_simulation`'s `eos_water`
already solves it properly, with dissociation by law of mass action, the full
`O+ .. O8+` Saha ladder, real degeneracies and real potentials.

## Decision

**Consume the solved state; own the burn envelope and the bag consequence.**

    burn envelope        eps = w^2 k / (2 (1+k)^2)      this repository
    equilibrium solve    (rho, eps) -> (T, f, P)        eos_water, vendored
    bag consequence      P/P0, B/B0, E_B                this repository

`data/plume_state.csv` is vendored from `puffsat_impact_simulation` @ `0216a09`
with its provenance in `data/README.md`, because a consumer in another
repository cannot run its `make`. `src/plume_state.py` reads it; `make
plume-state` prints the envelope.

**The solved states, against the paper's hand figures:**

| `w` [km/s] | dissipated | `T` solved | `T` hand | `f` solved | `f` hand |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 75 | 264.9 MJ/kg | 26 521 K | 26 200 | 0.5805 | 0.573 |
| 65 | 199.0 | 22 685 | 22 400 | 0.3799 | 0.371 |
| 56.53 | 150.5 | 19 708 | 19 400 | 0.2259 | 0.217 |
| 45.58 | 97.8 | 15 165 | 14 700 | 0.0616 | 0.053 |

**The hand solve was right to 1-3%, and the sign of the error is explained**: it
charged 54 MJ/kg for vaporisation plus dissociation where `eos_water`'s bond
energy is 50.9, so ~3 MJ/kg stays in the thermal pool and the solve runs warmer
everywhere. The gap widens toward the cold end because there is least energy
there for it to hide in.

## Consequences

- **15 000 K is the coldest pulse's answer, not the design's.** The state moves
  across the burn -- the craft accelerates into the oncoming wave, so closing
  speed falls from 56.53 to 45.58 km/s on the slowest cycle -- and at the bottom
  the solve lands at 15 165 K with 6% ionisation. **At the one point that
  matters most, the original assumption is the result.** Everywhere else it
  understates.
- **The energy budget closes, which is the real check.** `plume_thermal` now
  carries a water-ionisation term (`water_ionisation_energy`), so this
  repository's own caloric model can price the impact sim's solved state: 262.3
  MJ/kg against 264.9 dissipated at 75 km/s, and 100.0 against 97.8 at 45.58.
  **Two independent models agreeing to 1-2% is what turns 15 000 K from an
  assumption into a result**, and it is not a tautology -- one is a Newton solve
  on a species ladder, the other is arithmetic on enthalpies.
- **Ionisation is where the energy goes, not temperature.** At the hottest pulse
  the split is 125 MJ/kg of stripped electrons against 86 MJ/kg of thermal
  motion. Ionisation absorbs the excess the way boiling absorbs heat from a pot
  without warming it, which is why the temperature rises by a factor of 1.8
  rather than the factor of 3 that dividing energy by heat capacity implies.
- **Pressure counts particles, so the bag consequence is larger than the
  temperature rise.** Water goes from three particles per molecule to `3(1+f)`,
  and `P/P0 = (1+f) T/T0` multiplies the two: 2.79 at the hottest pulse, and
  `E_B` from 4.43 to 12.4 GJ. That is what sizes the nozzle (ADR 0016).
- **The table has to be two-dimensional.** Dissipated energy depends only on `w`
  and `k`, never on density -- but Saha does, so the same budget lands at a
  different temperature in a different bag. At 56.53 km/s, `rho` 0.05 -> 1.0
  moves the plume 16 857 -> 21 795 K. Since this repository sets
  `rho = m_slug/V` and the enclosed volume is a live design variable, a single
  row would not have served.

### What this does not settle

- **Oxygen's second ionisation.** The vendored solve carries the full ladder, so
  this is closed *there*; `plume_thermal`'s own term charges a single 13.598 eV
  to all three atoms and would understate a strongly ionised plume. It exists to
  price a solved fraction, not to find one, and its docstring says so.
- **The solve refuses to extrapolate.** Outside 44-76 km/s and 0.05-2.0 kg/m^3
  `plume_state` raises rather than guessing, because extrapolating a Saha solve
  silently is how the 4 672 K trap in `data/README.md` happens.

## Reproducing

`make plume-state`.
