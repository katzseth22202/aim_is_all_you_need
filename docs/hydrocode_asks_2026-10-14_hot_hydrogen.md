# Asks for `puffsat_impact_simulation`, 2026-10-14: the bulged hydrogen chamber at 7000 K and 10 000 K

From `aim_is_all_you_need` (ADR 0063, `src/departure_levers.py`). This is written
to be copied into the impact simulation's todo list, so every input is given in
full.

## Why we are asking

The departure burn is where growth is lost. On the flown chain's first cycle, the
methane chamber turns an 805 t parked stack into 178 t sent onward:
- a 7.30 km/s burn from the 20-day orbit's 600 km periapsis;
- plus 0.84 km/s of fixed-direction loss over a 37-minute, one-chamber burn;
- at about 7 km/s of effective exhaust;
- then 56 t of chamber and 12 t of tanks are expended.

Everything before the departure costs about a third; the departure costs 78%.

The bulged hydrogen chamber (ADR 0055) is the better device:
- efficiency 0.773 against methane's 0.47;
- about 10.3 km/s of effective exhaust;
- doubling in 2.84 yr against 3.36.

It still carries `k = 24` kg of hydrogen per kg of rod to hold its 4884 K, and
0.205 kg of tank per kg of hydrogen. That is about a tenth of the stack in tanks.

A hotter charge carries more energy per kilogram, so a smaller charge holds the
temperature. That means less dead mass per pulse, a faster effective exhaust, and
smaller tanks. But a pulse's energy is fixed by the rod, and momentum goes as
`sqrt(2 m E)`. A smaller charge therefore gives **less impulse per pulse**: about
280 kN s at 10 000 K against 657 today. The chamber then needs a faster pulse
rate to keep its thrust; otherwise the burn stretches or needs more chambers. On
the flown chain, **7000 K beats 10 000 K at every rate and recovery we tried**
(table below), so 7000 K is the first ask and 10 000 K the second.

Whether either pays depends on three things only you can solve:
- how much of the dissociation energy the nozzle gets back;
- whether the wall survives the heat;
- **what pulse rate the smaller charge sustains.**

## What we estimated

**Equilibrium charge** (Cantera `gri30`, ideal gas, at the bulged chamber's
charge density of 0.57 kg/m3; its fits are extrapolated above about 6000 K, and
the 10 000 K energy is checked against 216 MJ/kg of dissociation plus about 124
MJ/kg of atomic thermal energy):

| Temperature | Energy per kg above H2 at 300 K | Pressure at 0.57 kg/m3 | Atomic by mass | Energy in broken bonds | Ionised (Saha) |
|---|---:|---:|---:|---:|---:|
| 4884 K (today) | 111 MJ | 141 bar | 0.23 | 45% | - |
| 7000 K | 242 MJ | 275 bar | 0.67 | 60% | 3e-5 |
| 10 000 K | 340-390 MJ | about 400 bar | 0.75-1.0 | 40-60% | 1e-3 |

**The charge per rod.** We hold `(k + 1 + P) u(T) = w^2 / 2` at w = 75 km/s, with
a 2.5 kg rod and P = 1.76. That gives:
- **k = 9.5 at 7000 K** (about 24 kg of hydrogen a pulse);
- **k = 5.5 at 10 000 K** (about 14 kg a pulse);
- against k = 24 (60 kg) today.

Over the burn, w runs from 74 to 81 km/s and k follows it. At fixed volume, the
smaller charge should roughly offset the higher temperature in pressure, but that
is an estimate.

**The impulse.** We scale the chamber's efficiency by the chemistry ceiling,
`1 - 0.60 (1 - returned)`, keep the 1% gate, take the head-on debit, and fly one
2 Hz chamber through a 7.3 km/s burn with 0.84 km/s of loss:

| Chamber | Bonds returned by the AR 300 nozzle | Effective exhaust | Tanks, share of stack | Sent onward, share of stack |
|---|---|---:|---:|---:|
| 4884 K (today) | - | 10.3 km/s (1047 s) | 10.5% | 0.29 |
| 7000 K | 94% / 70% / 50% | 13.6 / 12.1 / 10.7 km/s | 7.9-9.4% | 0.41 / 0.37 / 0.32 |
| 10 000 K | 94% / 70% / 50% | 15.5 / 13.5 / 11.8 km/s | 6.5-7.9% | 0.47 / 0.42 / 0.36 |

The 94% is your 5500 K sphere's AR 300 recovery at 818 bar. We do not know what
it becomes when the charge starts almost fully atomic and less dense. Three-body
recombination goes as density squared, so it may freeze earlier.

**On the ledger** (`make departure-levers`). Each row is the bulged chamber with
its charge, chemistry ceiling and pulse rate changed and nothing else solved,
flown over the flown chain behind today's spray cup. Doubling time in years,
against today's 2.84 at 2 Hz and 2.60 at the 2.4 Hz top of its cycle:

| Bonds returned | 7000 K at 2 / 3 / 4 Hz | 10 000 K at 2 / 3 / 4 Hz |
|---|---|---|
| 94% | 2.48 / 2.21 / 2.03 | 2.79 / 2.29 / 2.15 |
| 70% | 3.51 / 2.84 / 2.53 | 4.62 / 3.13 / 2.76 |
| 50% | 7.49 / 4.40 / 3.59 | 49.8 / 5.71 / 4.28 |

So 7000 K pays if it returns 94% of its bonds at any rate, or 70% at 3 Hz or
better. 10 000 K needs 94% and 3 Hz or better, or 70% at 4 Hz. At 50% neither
pays at any rate we tried.

## The asks

**H1. Peak state.** For the bulged chamber, give its peak pressure, temperature,
composition and ionisation at 7000 K and 10 000 K, with a real-gas equation of
state. Inputs:
- 106 m3, 0.84 m2 throat, 37.0 t dry Kevlar wrap;
- a 2.5 kg rod at 75 km/s;
- the charge sized to hold each temperature (our k = 9.5 and 5.5; correct them if
  your equation of state disagrees).

Give the energy per kilogram and the share held in dissociation.

**H2. Recombination and impulse.** Fly the same finite-rate H + H + M
recombination you ran for methane (ADR 0056) through the AR 300 nozzle at both
temperatures. Report:
- the share of the bond energy returned;
- the net impulse per pulse, gate charged and head-on debit taken, against
  today's 657 kN s;
- the same at 74 and 81 km/s, the ends of the burn.

**H3. The wall.** Report heat flux to the liner by mode (radiation, convection,
the throat region) per pulse and over a 4500-pulse burn, and the wrap's flung
strain against its 2.4% limit (today's 2.20%). A hydrogen plasma at 10 000 K
radiates up to 17 times today's blackbody limit, 7000 K about 4 times. Is it
optically thin at these densities, and what does the copper or its replacement
reach?

**H4. The cycle (as important as H2).** The smaller charge (about 24 kg at
7000 K, 14 kg at 10 000 K, against 60 kg) should blow down and refill faster.
What is the fastest rate each sustains: blowdown, film, plug, fill and lock, as
in your 2 Hz cycle for methane? We need 3 Hz or better for most of the table to
pay. Can the throat film still seal at 0.3 bar behind a hotter residue? Wall heat
per second also rises with the rate, so H3 should be answered at the rate H4
finds.

**H5. The sweet spot.** Between today's 4884 K and 10 000 K, which temperature
gives the most impulse per kilogram carried *at the rate H4 finds*? On our
estimates it is near 7000 K: hotter thins the charge until the lost thrust
outweighs the faster exhaust. Charge, plug and tank all ride the departure, so
per kilogram carried, not per pulse, is what we optimise.

## What we will do with the answers

- The surviving temperature becomes a `ChamberPairing` in `chamber_isp`:
  - its charge (`reference_slug_ratio`);
  - the chemistry ceiling from H2;
  - its wall mass from H3;
  - its pulse rate from H4.

  Its efficiency comes from `efficiency_for_impulse` at the H2 impulse.
- `src/departure_levers.py`'s unsolved rows are replaced by it, and
  `make survivable-chamber`, `make slush-hydrogen` and `make growth-cost` reprice
  growth and $/kg at L1.
- The parent gets the result as a new row of its departure-device table.
