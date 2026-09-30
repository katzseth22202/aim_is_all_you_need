# The launch unit grows on an argon plate, and both chambers beat methalox

Status: accepted; amended by ADR 0034 (the parking orbit is the split, flown at
20 days). **Its figures below mix a 10-day split with 20-day burns and are 3-8%
optimistic for the chambers; quote ADR 0034's instead.** Its decisions stand.

Builds on: ADR 0032 (the chamber departure), ADR 0009 (the apoapsis reversal),
ADR 0013 (the two-wave split in real orbits).

Date: 2026-09-30

## Context

The child repository's `sec:exponential_mass_growth` needs growth per cycle,
doubling time and 10-year growth for a fixed **1500 t launch unit** at the 400 km
intercept, flown through the real 11-cycle chain (`adaptive_two_wave_cycles`,
7 x 2S + 4 x 3S), pushed by a slug-injected pusher plate on the overtake and
departed by a walled chamber (ADR 0032), with a 380 s methalox departure as the
incumbent to beat. The parent's plate model (`sec:water_injected_overtake`) is an
ideal ceiling at constant loading, `beta = eta_jet sqrt(1 + k) + 1`, which it
leaves at "candidate operating points" k = 9-10, and the chain itself is priced
at 200 km with instantaneous burns.

## Decision

**1. The plate's loading is chosen pulse by pulse, by Pontryagin's principle.**
Water and PuffSats are both costs, so the push minimises `m_in + lambda W`. With
`c` the running price of slug, each pulse takes the `k` minimising
`(1 + c k) / beta(k, w)`, and the price rises through the push as
`dc/dv = (1 + c k) / (beta w)`. One starting price fixes the schedule, so the
search is one-dimensional, and it is re-optimised on every cycle
(`src/water_plate.py`). The schedule sprays hardest early and tapers: **k falls
pulse by pulse**. It beats every constant loading on its own objective (tested
for water and argon).

**2. The loading is capped at k = 10** (decided with the user), the top of the
parent's candidates. Left free, the optimum opens at k = 50-100, where the ideal
ceiling is unvalidated. The per-pulse cost is single-peaked in `k`, so the
clamped rule is still the optimum under the bound. The uncapped schedule is
reported as a sensitivity; it is 3-10% faster.

**3. Plate efficiency is net of chemistry, and the toll is charged per pulse.**
The parent already writes `eta_jet = eta_chem eta_geom` (`sec:jet_efficiency`),
sweeps `eta_geom` and charges `eta_chem`. The plate's 0.5 / 0.7 / 1.0 are
`eta_geom^2`, and `eq:eta_chem` is generalised to a blob of two materials: only
the impactor brings energy, while every kilogram pays its own bonds,

    eta_chem = sqrt(1 - 2 (E_PuffSat + k E_slug) / w^2).

With water on an ice PuffSat this is the parent's form (0.730 at 45.58 km/s and
0.910 at 75 km/s, `k` = 8.52), and it equals `plume_thermal.chemistry_efficiency`.

**4. The plate sprays argon onto ice PuffSats** (decided with the user). Argon has
no bonds, so on an ice PuffSat only the PuffSat's own bonds are paid
(`eta_chem` about 0.98 at any loading). Its drop tank is 14.6/1395 = 0.0105 per
kilogram against water's 0.015 (the parent's tanks all come to about 14.6 kg per
cubic metre of propellant). Water is kept as the comparison (`make plate-slug`).

**5. Argon's ionisation is assumed to recombine while the gas still pushes on the
plate** (stated; the user asked for the physics to be checked). It is not minor at
peak: the merge thermalises about 100 MJ/kg of blob at 50 km/s and k = 10, and
singly ionised argon holds 38.1 MJ/kg, with the second ionisation about 67 more.
The claim is that all of it comes back:

- By the Saha equation argon is 99% recombined by 15 700 K at 1 GPa, 13 500 K at
  100 MPa, 11 800 K at 10 MPa and 10 400 K at 1 MPa (the parent's water plate
  peaks at 5.79 GPa). At 10 000 K only 0.02-0.65% is still ionised, at most
  0.25 MJ/kg.
- At plate densities three-body recombination is fast, so the gas tracks
  equilibrium.
- Gas spilling past the rim freezes ionised, but it is already the capture loss;
  recombination radiation belongs in `eta_geom`, as the parent defines it.

Water's bonds re-form only near 3000-4000 K, later in the expansion, so water
keeps the frozen toll. That toll is pessimistic on a plate (the companion returned
23-31% of the store); water's role as the plate's heat sponge is also unpriced.

**6. The launch unit is carried through the parent's altitudes and every
charge the chain already makes** (`src/growth_ledger.py`):

1. The growth wave pushes it from rest to the 400 km cycle-orbit speed
   (10.786 km/s), its speed moved from the chain's 200 km by energy conservation.
2. At the 613 000 km apoapsis, methalox raises periapsis to 600 km (1.7 m/s) and
   pays the **apoapsis reversal** (ADR 0009; 234 m/s on the 20-day orbit, about
   6% of the stack), which every design needs to turn the retrograde push into a
   prograde departure.
3. The 150 t plate and the slug's empty tank drop; hydrogen's cryostats drop with
   them, and its boil-off over the hold is lost (both solved self-consistently
   with the burn the hydrogen must fly; swept 0-5%).
4. The chamber departs from 600 km into the departure wave, with the chain's burn
   and wave speed moved there too (the parent's 107-110 m/s three-synodic premium
   is reproduced; two-synodic burns pay about 130 m/s).
5. Each wave pays its own correction burn in methalox: the growth wave its
   early-arrival burn, the departure wave's rods the chain's DSM proxy. (The
   chain's own nozzle ledger lumps both onto the growth wave.)

Growth is PuffSats out over PuffSats in, `net x S0 / (m_in + rods x S0)`, both
waves counted as they left Jupiter.

**7. The incumbent flies the same unit on Raptor 3s.** Methalox has no departure
wave, so the whole batch pushes and needs no split. It departs on Raptor 3s at
380 s with a steered loss (engines can follow the velocity), 0.018 tanks, and the
engine count optimised, on a chain forced to three-synodic cycles (it cannot fly
2S). The Raptor 3 is the published sea-level engine (280 tf, 1525 kg; SpaceX,
2024-08-01) paired with the parent's 380 s vacuum Isp, a mixed assumption stated
in the code, because no vacuum Raptor 3 has been published.

## What the numbers say

Doubling time in years, k held to 10 (uncapped in parentheses). Chambers at a
share of their chemistry ceiling (ADR 0032), gated, methane at 5.6 kg of pitch
per pulse, hydrogen with 3% cryostats and 3% boil-off. Plate efficiency net of
chemistry, argon on ice PuffSats.

| Departure | Plate 0.5 | Plate 0.7 | Plate 1.0 |
|---|---|---|---|
| H2 50% | 4.05 (3.81) | 3.17 (2.93) | 2.55 (2.31) |
| H2 70% | 2.05 (1.99) | 1.79 (1.71) | 1.57 (1.47) |
| H2 solved (88%) | 1.66 (1.62) | 1.48 (1.43) | 1.33 (1.25) |
| H2 90% | 1.63 (1.59) | 1.46 (1.41) | 1.31 (1.24) |
| H2 100% | 1.52 (1.49) | 1.37 (1.33) | 1.23 (1.17) |
| CH4 50% | 3.99 (3.66) | 3.14 (2.82) | 2.54 (2.23) |
| CH4 70% | 2.13 (2.04) | 1.85 (1.74) | 1.62 (1.49) |
| CH4 solved (80%) | 1.85 (1.78) | 1.64 (1.55) | 1.45 (1.34) |
| CH4 90% | 1.7 (1.64) | 1.51 (1.44) | 1.35 (1.26) |
| CH4 100% | 1.58 (1.53) | 1.42 (1.35) | 1.27 (1.19) |
| Methalox 380 s (3S only) | 5.27 (4.97) | 3.91 (3.59) | 3.01 (2.68) |

1. **Both chambers beat methalox at every efficiency swept**, even 50% of their
   ceilings: behind a 0.7 plate, 3.17 (H2) and 3.14 (CH4) years against 3.91,
   about 20% faster; at 70% of ceiling, about twice as fast. The chamber here is
   expended every cycle (ADR 0032), so this is the pessimistic case for mass.
2. **Hydrogen and methane tie at 50% of ceiling, and hydrogen leads from 70% up.**
   The solved chambers, hydrogen at 88% of its ceiling and methane at 80%, double
   in 1.48 and 1.64 years behind a 0.7 plate.
3. **Argon beats water on the plate by 8-11% in doubling, and an argon slug on
   ice PuffSats gets nearly all of it** (`make plate-slug`, solved chambers):

   | Plate | Departure | Water on ice | Argon on ice | All argon |
   |---|---|---|---|---|
   | 0.5 | H2 0.858 | 1.80 | 1.66 | 1.64 |
   | 0.7 | H2 0.858 | 1.62 | 1.48 | 1.47 |
   | 1.0 | H2 0.858 | 1.46 | 1.33 | 1.31 |
   | 0.7 | CH4 0.538 | 1.81 | 1.64 | 1.62 |

   Argon PuffSats add only about 1%, so no cryogenic PuffSats or sunshades are
   needed. Water fights its own toll: its ceiling starts at 0.83 on the first pulse
   and recovers only by pumping `k` down, and it gains nothing from uncapping.
4. **Hydrogen's cryostats and boil-off hardly matter.** From 0% to 5% of each, the
   solved hydrogen chamber behind a 0.7 plate moves from 1.46 to 1.50 years.
5. The push ratio is 8.5-14.1 t of departing stack per tonne of growth PuffSat,
   against the 8.43 the parent's `tab:h2_breakdown` used, and a launch unit sends
   about 150-310 t of payload onward.

## Consequences

- The report runs as `make growth-ledger` (about 45 minutes: 30 chamber rows, each
  capped and uncapped, plus the incumbent and the hold sweep) and
  `make plate-slug` (about 20 minutes).
- The plate's module keeps the name `water_plate.py`; argon is now its default.
- **A known conflict with CONTEXT.md, since resolved by ADR 0034.** CONTEXT.md says
  to avoid choosing the split gap and the parking period independently. The chain
  flies a 10-day split, while this ledger prices its burns and reversal on the
  20-day orbit, to match the parent's 600 km figures (10.785 km/s, 1.7 m/s,
  613 000 km). At 10 days the reversal is 372.5 m/s against 233.9. The
  consistent options are the 20-day chain (`split_days=20`; ADR 0013 found 10 and
  20 days within 0.7%) or everything at 10 days.
- Still open: fly-and-park rows (methane only), chamber retrieval as an upside, the
  cost of building a chamber every cycle, offset-ignition burns, re-sizing plate
  area and pulse to these push ratios, and bare-tank boil-off during the burn.

## Considered and rejected

- **A constant plate loading.** The per-pulse optimum is exact and cheap, and the
  user's point that `k` should change pulse by pulse is what it formalises.
- **An uncapped loading as the headline.** It relies on the ideal ceiling far
  outside the parent's validated range; it stays as a sensitivity.
- **Charging argon's ionisation as frozen.** At plate pressures the ions are gone
  by 10 000-16 000 K, while the gas is still hot and pushing (decision 5).
- **Water as the default slug.** Argon is 8-11% faster and lighter to tank; water's
  heat-sponge role is the one reason left to prefer it, and it is unpriced.

## Reproduction

`make growth-ledger` and `make plate-slug` at the committing revision;
`pytest tests/test_water_plate.py tests/test_growth_ledger.py` (the slow tests hold
the schedule searches and the real chain). Chain: `adaptive_two_wave_cycles()` with
its defaults (start 2026-08-11, 30-year horizon, 50 m/s 2S threshold, 10-day
split); the incumbent's chain is the same with `threshold_m_s=0`. Plate schedule:
starting price scanned on `geomspace(1e-3, 1, 31)` then refined by bounded
Brent (`xatol` 1e-6 in log price); per-pulse loading on a 257-node grid over
[0, 10] (or [0, 400] uncapped) refined by a parabola; push ODE `rtol` 1e-10,
`atol` 1e-12. Engine and chamber counts searched up to 200.
