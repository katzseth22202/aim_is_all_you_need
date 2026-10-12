# Recover hydrogen departure hardware into PuffSats

Status: accepted as an exploratory accounting scenario; engineering unverified

Date: 2026-10-10

## Context

> **Superseded in part, 2026-10-12 (ADRs 0049, 0050).** ADR 0050 re-prices recovery
> on the paper's flown chain in production code, confines recovered mass to rods
> (growth PuffSats must disperse), drops aluminium tanks for carbon fibre and
> replaces the Pinatubo comparison with the meteoric background.
>
> **Energy (ADR 0049).** Every 3,000 K figure below was
> computed with a hot energy scaled linearly from 5,500 K (57.3 MJ/kg, 9.10 km/s
> exhaust). Equilibrium gives 34.5 MJ/kg and 7.06 km/s; ADR 0045's re-solved
> miss-safe burn delivers 39.1% of ignition mass instead of 46.9%. The 3,000 K
> column is therefore optimistic and should not be quoted. The error only flatters
> 3,000 K, so the ranking (5,500 K cheaper) stands. The **5,500 K** results did
> not use the scaling and remain the recovery sensitivity of record. The common-route
> ledger that produced the snapshot JSON is described in
> `docs/hydrogen_curvature_cost_2026-10-10.md` but its generator is not committed:
> the replay script re-prices the recorded inputs, it cannot rebuild them.


Hydrogen departure leaves substantial tank and chamber mass after the burn.
Discarding all of it reduces the useful material available to multiply the next
generation of PuffSats. The user proposes cutting and packaging that hardware
during the roughly three-year journey, before the next return.

The [common-route analysis](../hydrogen_curvature_cost_2026-10-10.md) distinguishes
an original-route horizon-only comparison from exploratory curvature-aware
departures on a circular 3S route. The recovery scenario extends the latter.
It does not complete a geometry-only rescore of the original mixed 2S/3S ledger.

## Decision

Record the following selected scenario for both hydrogen temperatures:

- Recover **90% of hydrogen tank mass**.
- Recover **60% of both chamber and nozzle mass**.
- Carry **5 tonnes of cutting and packaging equipment** within the existing
  departure stack, displacing 5 tonnes of original onward payload.
- Purchase that equipment for **$1 million on every growth flight**, without
  a manufacturing-learning discount or recovery credit for the equipment.
- Make recovered material available by the next return, allocated in the
  original growth-PuffSat/departure-rod proportions. Charge ordinary body and
  steering-package costs on that material; do not also claim virgin-material
  purchase savings.
- Keep exhaust and impact performance unchanged for this sensitivity.
- Keep the **15 Earth–Jupiter synodic period** harvest horizon (16.3806 years).
  Continue to distinguish the start of harvesting from an investor payback date.

The initial mass, flight trajectory and launch-push choice are held fixed.
The cooler case retains its 1,200 s burn; the hotter case retains its selected
2,500 s burn. This is not a temperature-only comparison or a reoptimization.
The cooler case retains zero chamber ablation and the previously documented
optimistic removal of the bundled chamber-spray equipment allowance.
No production ledger defaults are changed by this decision.

## Why the recovery fractions differ

The user's rationale for limiting chamber/nozzle recovery to 60% is concern
about possible catalytic ozone effects of iron-oxide-bearing material released
to the atmosphere, while provisionally treating aluminum tank material as
relatively benign. This is a secondary uncertainty motivating the scenario,
**not a measured recovery limit or an established environmental ranking**.
Neither the severity nor the relative importance of iron versus aluminum
chemistry has been evaluated for this system; no ozone cost penalty is applied.

Aluminum also has a potential atmospheric mechanism: oxide particle surfaces
can promote chlorine activation. See Ferreira et al. (2024),
[Potential Ozone Depletion From Satellite Demise During Atmospheric Reentry
in the Era of Mega-Constellations](https://doi.org/10.1029/2024GL109280).
That oxide-formation study does not quantify ozone loss for PuffSats or rank
steel against aluminum. Actual impact depends on atmospheric release rate,
particle properties and altitude; unrecovered material's disposal also matters.

The current mass proxies do not provide a material-by-material bill of materials:
the 32 t hydrogen chamber is described with a copper liner, and the 2.19 t
extension is scaled from an RL10 reference. The unrecovered 40% must not be
interpreted as an established steel fraction.

For scale, [Ukhov et al. (2023)](https://doi.org/10.1029/2022JD038446)
reconstruct 66.5 Mt of Pinatubo fine ash, of which 37.2 Mt was injected into the
troposphere, leaving about 29.3 Mt injected into the stratosphere. An illustrative
8% elemental Al and 3% elemental Fe composition would imply 2.34 Mt Al and
0.88 Mt Fe. Those composition fractions are assumptions here, not measurements
of the stratospheric fraction. Dividing that single event by a chosen 35-year
window gives about 67,000 t Al/year and 25,000 t Fe/year; **this is not a global
annual volcanic flux estimate**. [Guo et al. (2004)](https://doi.org/10.1029/2003GC000655)
estimate the high-altitude cloud's ash mass fell from roughly 50 Mt to below
10 Mt within three days. Large natural injections therefore provide a useful
scale comparison, but silicate-bound metals, particle size and rapid fallout
prevent treating equal injected elemental masses as equal ozone effects.

## Results with the selected $1M equipment price

Dollar figures are per kg delivered to L1. Break-even ranges use the existing
cheap/dear seed-transport scenarios, 10% discount rate, and 50% success weighting.
They include deployment and operation, with post-harvest sales through year 300;
they exclude an explicit program R&D budget.

| Quantity | 3,000 K hydrogen | 5,500 K hydrogen |
|---|---:|---:|
| Recovered tank | 76.45 t | 56.18 t |
| Recovered chamber/nozzle | 20.51 t | 20.51 t |
| Net useful gain after machine mass | 91.96 t | 71.69 t |
| Useful onward mass, before → after | 246.33 → 338.29 t | 336.03 → 407.72 t |
| Growth multiplier per 3S cycle | 1.8485 → 2.5386 | 2.4384 → 2.9587 |
| Operating cost, before → after | $149.86 → $107.07 | $108.43 → $94.61 |
| Program break-even before | $339.72 / $791.98 | $177.68 / $297.55 |
| Program break-even after | **$169.63 / $269.55** | **$130.43 / $181.89** |

The hotter scenario falls below the discussed $200/kg benchmark in both seed
cases under these assumptions. The cooler scenario does so with cheap seed
transport. These are conditional cost estimates, not market-price forecasts.

## Could the nozzle be aluminum?

Do not substitute aluminum or claim a mass saving without structural and thermal
sizing. Internal gas pressure primarily produces tensile hoop stress in the
pressure wall, even though the pressure acts normally on its surface. A spherical
thin-shell estimate is sigma = p*r/(2*t); cylindrical hoop stress is p*r/t.
These static relations are insufficient for localized impulsive loading,
stress-wave reflection, bending, attachment loads and repeated-pulse fatigue.

The existing 5,500 K hydrogen pairing cites an 818 bar peak pressure. The recovery
analysis does not derive a new chamber pressure or a nozzle pressure distribution
for 3,000 K. Gas expansion reduces pressure along the nozzle, so the downstream
extension and chamber/throat must be assessed separately.

Aluminum melts near 933 K (660 degrees C), and structural alloys lose strength
well before melting. A 3,000 or 5,500 K gas temperature does not equal wall
temperature, but it makes wall heat flux, pulse penetration depth, cooling and
thermal fatigue decisive. A sufficiently cooled aluminum outer jacket or cooler
downstream structure is a candidate for study. An exposed aluminum chamber or
throat is not supported by the present model. A protected hot face and different
outer structural material would require accounting for both masses and cooling
equipment. The scaled nozzle-extension proxy does not establish material viability.

## Addendum: candidate electromagnetic handling of aluminum

Recorded 2026-10-10 as a processing concept, not a validated equipment design.

> Cut the chamber or tank into controlled pieces, retain them during cutting,
> and use electromagnetic handling to position them for mechanical capture
> and packaging.

Aluminum's electrical conductivity makes contactless handling through induced
eddy currents plausible. A changing magnetic field induces currents in the
metal; their interaction with the applied field can produce forces and torque.
Suitable coil geometry and feedback could assist translation, orientation, or
braking. This is not ordinary ferromagnetic attraction: a simple oscillating
electromagnet can repel aluminum rather than attract it. Reliable capture
requires a designed field arrangement and control system.

Useful force and working distance depend on coil geometry, field strength and
frequency, and the piece's shape, thickness, orientation, and conductivity.
Eddy-current heating must be included in the power and thermal budgets.
Microgravity removes the need to support weight, but not the need to arrest
translation and rotation; reaction forces and torques act on the processing
spacecraft. Keeping pieces restrained during cutting avoids turning processing
into a free-fragment collection problem.

This is a reason to investigate aluminum-rich recoverable hardware, not proof
that conversion into PuffSats is easy. Cutting, shaping, packaging, and attaching
guidance hardware remain separate tasks. The concept does not validate the
existing five-tonne processing allowance, establish 99% recovery, or qualify
aluminum for the chamber's structural and thermal environment. No recovery
fractions, equipment masses, financial results, or production defaults change
with this addendum.

## Consequences and reproducibility

This accounting credits useful material rather than reducing tank or chamber
purchase costs. Recycled mass is about 28.7% of cool-case onward mass and 18.8%
of hot-case onward mass, so unchanged impact/exhaust behavior is a consequential
assumption. Conversion yield, handling, power, radiator and consumables needs
must fit the assumed 5 t allowance; this has not been demonstrated. The prior
parking-interface and hardware-survival limitations remain open.

Supporting artifacts, all independent of ignored scratch files:

- [Original analysis and provenance](../hydrogen_curvature_cost_2026-10-10.md).
- [Original financial inputs and trajectory snapshot](../../data/hydrogen_curvature_cost_2026-10-10.json).
- [Recovery accounting and sensitivities](../hydrogen_recovery_cost_2026-10-10.md).
- [Replay script](../hydrogen_recovery_analysis.py).
- [Recovery outputs](../../data/hydrogen_recovery_cost_2026-10-10.json).

Replay from the repository root:

```bash
PYTHONPATH=. .venv/bin/python docs/hydrogen_recovery_analysis.py
```

The script reproduces original 15S prices to $1e-8/kg and checks mass accounting
to 1e-6 kg. The selected results have `scope = chamber_and_nozzle`,
`machine_purchase_dollars = 1000000`, and `horizon_synodics = 15`.
