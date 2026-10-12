# Recovering departure hardware into PuffSats

> **2026-10-12:** the 3,000 K and 3,500 K figures here use a hot energy scaled
> linearly from 5,500 K, which ADR 0049 shows overstates it by up to two-thirds.
> Treat them as void; the 5,500 K and methane figures are unaffected.

This is a fixed-flight sensitivity to the [common-route study](hydrogen_curvature_cost_2026-10-10.md),
not a new trajectory optimization. The user selected 90% recovery of hydrogen
tank mass and 60% recovery of **both chamber and nozzle**, with a 5 t cutting
and packaging machine per launch, purchased for **$1 million on every flight**.
Exhaust performance is held unchanged by
assumption. The 3,000 K burn remains 1,200 s; the 5,500 K burn remains 2,500 s.

The machine displaces 5 t of original onward payload within the existing
departure stack. Hardware recovery occurs after departure, is complete before
the next return, and adds useful PuffSat/rod mass in the existing proportions.
The unrecovered hardware and machine receive no useful-mass credit. Tanks and
engines are still purchased anew on each growth launch. Cryostats are excluded
from recovery. All recovered mass receives the normal body and steering-package
costs: no material-purchase saving is credited. The fixed-flight accounting is

```
new_onward = original_onward - 5000 + 0.9*tank + 0.6*(chamber + nozzle)
new_growth = new_onward / original_consumed_puffsats_and_rods
```

## Results at 15 synodic periods (16.3806 years)

| Quantity | 3,000 K | 5,500 K |
|---|---:|---:|
| Tank material recovered | 76.45 t | 56.18 t |
| Chamber and nozzle recovered | 20.51 t | 20.51 t |
| Net addition after 5 t machine | 91.96 t | 71.69 t |
| Usable onward mass, before → after | 246.33 → 338.29 t | 336.03 → 407.72 t |
| Growth per 3S cycle, before → after | 1.8485 → 2.5386 | 2.4384 → 2.9587 |
| Operating cost, before → after | $149.86 → $107.07/kg | $108.43 → $94.61/kg |
| Program break-even before, cheap/dear seed | $339.72 / $791.98 | $177.68 / $297.55 |
| Program break-even after, cheap/dear seed | **$169.63 / $269.55** | **$130.43 / $181.89** |

The main table includes the machine's mass, launch burden and the user's
selected $1M purchase price per growth launch, without learning. The earlier
zero-purchase-price sensitivity gave $168.46/$268.44 per kg (cool) and
$129.56/$181.07 (hot). Charging an illustrative
**$5M machine on every growth launch**, without learning, raises operating
cost to $109.52/$96.46 per kg for cool/hot hydrogen and program break-even to
**$174.29/$274.01** (cool) and **$133.87/$185.20** (hot).
The JSON also records a $20M sensitivity. These are cost assumptions, not quotes.

At 21S (22.9328 years), the zero-purchase-price recovery cases give
$147.69/$177.47 per kg for 3,000 K and $113.59/$124.96 for 5,500 K.
These are later starts to steady harvesting, not deadlines for cash payback.
The original 10% discount, 50% success weighting and post-harvest cash flows
through year 300 remain unchanged. No explicit program R&D budget is added.

## Interpretation and limitations

Recycling turns a portion of already-launched dry mass into the material that
multiplies the next generation. It reduces the mass discarded per flight;
it does not reduce the initial tank or engine purchase bill.

The recovered material constitutes about 28.7% of cool-case onward mass and
18.8% of hot-case onward mass. Its suitability as plate impactors and chamber
rods, packaging geometry, steering, and erosion effects are not established
by this mass ledger. In particular, unchanged exhaust speed is an explicit
scenario assumption, not a demonstrated consequence of hydrogen dominance.
No new processing power, radiator, handling, or consumables mass is added
beyond the user's 5 t total allowance. Recovery equipment life and throughput
have not been engineered. The existing trajectory and parking-interface
limitations of the common-route study remain.

## Reproduction and checks

Run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python docs/hydrogen_recovery_analysis.py
```

The [script](hydrogen_recovery_analysis.py) reads the checked-in original
financial snapshot and writes [the recovery results](../data/hydrogen_recovery_cost_2026-10-10.json).
It also records the narrower nozzle-extension-only interpretation for comparison;
the selected interpretation is `chamber_and_nozzle`.
Equipment purchase cost is added to the existing flat per-chamber cash charge;
there is exactly one chamber and one machine per growth launch. Consequently
that sensitivity appears within `chamber_spray` in the detailed cost lines,
without implying any additional spray or ablation mass.

Checks reproduce both original 15S financial results to $1e-8/kg, verify
growth/consumed-mass consistency and conserve post-burn mass within 1e-6 kg.
`make test-changed` selected no test files; no production source was changed.
