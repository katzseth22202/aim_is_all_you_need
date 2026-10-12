# Asks for `puffsat_impact_simulation`, 2026-10-12: blowdown, wall forces and the 2 Hz cycle

From `aim_is_all_you_need` (ADRs 0049, 0053, 0054). This is written to be
copied into the impact simulation's todo list, so every input is given in full.

## Why we are asking

The chamber's pulse rate is now the main open question in the departure design.
Each pulse is a burst of a few milliseconds, then a coast: the chamber blows down,
lays its films, seats the next plug, fills, and locks 20 ms before the next rod.
The growth ledger flies the survivable methane chamber at **2 Hz**. A 2 Hz chamber
burns for 24-37 minutes, and that much finite-burn loss is one of the larger lines
in the departure. Doubling the rate is worth about 10% in growth per cycle. For
hydrogen slush with recycled tanks, 2 Hz instead of 4 Hz takes growth from 2.88 to
2.60 and raises the cost from $105 to $116/kg at L1 (ADR 0053).

We can only estimate the cycle with a quasi-steady adiabatic blowdown and a
one-degree-of-freedom wall. We need your transient solves to confirm or replace
those estimates for both chambers.

## What we estimated

**Blowdown** (`chamber_isp.blowdown_time`): a choked throat, isentropic gas
behind it, discharge coefficient one. Peak states come from Cantera `gri30`
equilibrium, extrapolated above 5000 K. Effective gamma is taken from an
equilibrium expansion to a tenth of the density.

| Chamber | V, throat | Peak | Gamma, sound speed | 95% out | To fill pressure | Period left |
|---|---|---:|---|---:|---:|---:|
| CH4 survivable, 7000 K, 5 kg rod, P = 4 | 212 m3, 1.33 m2 | 94 bar | 1.225, 4.48 km/s (your 3.7 km/s as the slow bound) | 215-260 ms | 278-336 ms to 1 bar | 164-222 of 500 ms |
| H2 sphere, 5500 K, 2.5 kg rod, k = 24 | 20 m3, 0.198 m2 (AR 300 at 8.7 m) | 806 bar | 1.194, 5.66 km/s | 105 ms | 135 ms to 9.5 bar | 115 of 250 ms |
| CH4, unbulged | 106 m3, 1.33 m2 | 175 bar | 1.212, 4.29 km/s | 111 ms | 142 ms to 2 bar | - |
| CH4, unbulged | 53 m3, 1.33 m2 | 318 bar | 1.198, 4.07 km/s | 58 ms | 73 ms to 4 bar | - |

The fill pressures are the warm methane charge (97 kg at 420 K, through the
regen passages) and the 77 K hydrogen gas charge. Our e-fold time for the hydrogen
sphere is 31 ms, matching yours.

**Cushioned wall** (`wall_cushion.cushion_response`, ADR 0054): a 10 mm steel tile
(80 kg/m2) on a sealed gas cushion against a backwall. With 6-8 cm of cushion at
60-80 bar and 2-5 kPa s spikes, the tile travels 1-3 cm. The backwall then carries
1.1-1.5 times the 94 bar steady load, against your solved hoop overshoot `D` of
1.8-5.7.

## The asks

**B1. Blowdown history, both chambers.** For the survivable methane chamber (212
m3, AR 100 and 300) and the 20 m3 hydrogen sphere, report:
- pressure, mass, temperature and composition against time, from the pulse until
  the chamber reaches its fill pressure;
- the time to 95% of the mass out and to fill pressure;
- how much wall heat loss and recombination change the time against our adiabatic
  estimate;
- the residual gas state at refill: temperature, composition, and how much residue
  near 4000 K the plug meets.

**B2. Does 2 Hz close?** Lay out the whole 500 ms methane cycle and the 250 ms
hydrogen cycle:
- blowdown;
- films over both openings;
- plug placement, with the frozen-methane plug going in at the last moment;
- the fill, its injector flow rate and what leaks through the throat film;
- the 20 ms lock.

Is there margin, and what limits the rate? What is the fastest rate each chamber
could sustain? Can any stages overlap, for example filling while the last of the
blowdown leaves?

**B3. Wall forces per pulse.** By zone (head, bulge ramp, plateau, taper, cone),
report:
- the spike impulse (kPa s) and its duration;
- the peak and the quasi-static pressure history at the wall;
- the hoop overshoot `D`.

Do this for the bulged 212 m3 chamber and for unbulged 106 and 53 m3 chambers at
the same rod, charge, temperature and throat. We need to know how fast the spike
and `D` grow as the chamber shrinks.

**B4. A cushioned wall (ADR 0054).** Run the wall as 10 mm Cr-Mo tiles on a sealed
gas cushion (6-8 cm, precharged to 60-80 bar, with orifice damping) against a
backwall, at 212, 106 and 53 m3. Report:
- the tile's travel;
- the backwall's peak load against the steady load;
- the tile's own stresses;
- what the seal gaps see.

Does it let a 53-106 m3 chamber survive 4200 pulses? If it does, the chamber can
run at 4 Hz.

**B5. The hydrogen chamber, same questions.** You found the hydrogen sphere
unresolved at this pulse size: about 2.95 GPa centered blast, and a copper liner
bonded to the shell. Report:
- what bulge, wrap or cushion makes it survive;
- its blowdown time at that geometry (B1);
- the rate it can then sustain (B2).

The parent paper's lever for hydrogen is "the same bulge and wrap, a larger
chamber, or a smaller rod per pulse". A bulged hydrogen chamber probably drops to
2 Hz, so this decides whether hydrogen keeps the 4 Hz the ledger credits it.

**B6. Impulse per rod against volume.** The parent says the bulged chamber delivers
about 7% less impulse per rod than a 40 m3 chamber. That is because more energy
stays in broken bonds at low density. Report the impulse per pulse (gate charged,
head-on debit taken) at 212, 106 and 53 m3 for methane, so a faster, smaller
chamber is priced on its own efficiency.

## What we will do with the answers

- Pulse rate, wall mass and efficiency go into `chamber_isp.survivable_methane` and
  the hydrogen pairing.
- `make survivable-chamber`, `make slush-hydrogen` and `make hardware-recovery`
  then reprice growth and $/kg at L1.
- The blowdown and cushion results replace ADR 0053's and ADR 0054's estimates.
- The parent gets the result through `docs/orbit_turn_for_parent.md`, item 9.
