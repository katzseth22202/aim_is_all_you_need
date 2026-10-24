# A 1 kg floor on the Sun-wave disk: ask S59 (for `aim_is_all_you_need`)

Raised 2026-10-24 against aim `edc9d3e`. **Written to be copied verbatim into
`katzseth22202/aim_is_all_you_need`**, so it repeats context the paper repo already has.
Register entry: S59 in `docs/deferred_to_companion_repos.md`.

Same ground rules as your own handbacks. Neither document is the source of truth. If a number here
looks wrong, say so rather than working around it. The figures below come from a parent scratch
harness that calls your `src/solar_dive_growth.py` directly and changes no physics. The harness is
reproduced in full at the end. None of it is a companion result yet.

## The question (Seth, 2026-10-24)

Each Sun-wave disk carries about 85-130 g of guidance hardware (a laser beacon with five steered
outputs, Sun sensor, star camera and gyro, cold-gas valves and tank, radio, cell and capacitor, a
brow-shaded pod) plus 25 g of cold gas. The paper now prints this as `sec:sun_wave_ranging`.
Seth: going below a **1 kg disk** is an implausible level of miniaturization.

Your ledger flies `pulse_disk = disks / (SUN_PULSE_RATE * arc_hours * 3600)`. At the 9.86 R_sun,
X 60 headline (`HEADLINE_STARTS`, k_sun 20, arc 4.97 h) that is **0.20 kg of disk per pulse** per
launch unit. Three ways to raise it were priced. Seth chose the **harder boost** (a shorter arc,
so a higher mean acceleration) as the realistic one:

- **Lower k_sun.** Seth: k stays about 10-20. k_sun 10 gives only 0.35 kg (below your k >= 20 table,
  log-linear extrapolation of `sun_boost` v_e).
- **Stacking** about five launch units into one Sun ship. Ledger-neutral to first order, but the
  seed is one unit, and docking is not designed. Not chosen.
- **The harder boost.** A ~1 h arc, about 0.28 g mean at 9.86 R_sun. Burst-and-coast does not help,
  because the bowl and spring are sized by each pulse's kick, which a burst leaves unchanged.

## What the parent did

`optimize_design` unchanged, with `_BOUNDS["arc_hours"]` capped and the cap iterated until
`pulse_disk >= 1 kg`. Starts from `HEADLINE_STARTS`; dear seed, break-even objective, as your
headline.

| Closure | arc (h) | k_sun | kg/pulse | growth | doubling (y) | steady $/kg | BE cheap / dear | Sun bowl+bag (t/unit) | stroke (m) |
|---|---|---|---|---|---|---|---|---|---|
| 9.86, X 60, no floor (reproduces the headline) | 4.965 | 20.00 | 0.200 | 3.361 | 1.873 | 94.1 | 123.9 / 145.7 | 5.34 | 1.71 |
| 9.86, X 60, 1 kg floor | 0.982 | 20.00 | 1.000 | 3.231 | 1.936 | 96.6 | 130.8 / 158.0 | 15.74 | 2.92 |
| 4, X 90, no floor (reproduces the headline) | 2.188 | 20.00 | 0.547 | 4.995 | 1.412 | 73.5 | 84.3 / 87.2 | 19.12 | 1.81 |
| 4, X 90, 1 kg floor | 1.134 | 20.73 | 1.000 | 4.884 | 1.432 | 74.2 | 85.6 / 88.9 | 28.68 | 2.27 |

Fixed-design arc sweep at 9.86, X 60 (other knobs at `HEADLINE_STARTS`, dear BE): 4.97 h $146,
3 h $147, 2 h $150, 1.5 h $152, 1 h $158, 0.75 h $163, 0.5 h $174 (1.96 kg/pulse).

The paper now prints the floored rows as corrections (parent `sec:sun_wave_ranging`) and says the
rest of the paper still carries the unfloored optimum until you rerun.

## S59. Wanted

1. **A disk-per-pulse floor** in `Design` or the optimizer (default 1 kg, a recorded constant),
   enforced as a constraint rather than by capping the arc, so both headlines re-optimize under it.
2. **Rerun everything the paper prints from the solar-dive ledger** with the floor: both headlines,
   the depth dial (`make solar-dive ARGS="--sweep"`), the faster returns, the Jupiter-only
   comparison, the delivery, seed and best-case ledgers, and the 5-day gap sensitivity.
3. **Check the Sun plate and spring** at 1 kg pulses. `_plate_for` and `gas_spring` were sized
   around 0.2-0.55 kg pulses; say whether 1 kg is still inside what `SUN_PLATE_PULSE` was fit to.
4. **Hardware on the disk.** The 85-130 g of hardware is inside the 1 kg and strikes the charge,
   so it is impactor mass, not dead mass. The 25 g of cold gas is spent. Say whether the ledger
   should carry the gas (2.5%) as a disk-mass loss.
5. **Stacking, as a comparison only.** If cheap, price a Sun ship of N launch units at the long arc
   (N = 5) to confirm it is ledger-neutral to first order. Do not adopt it.

## Harness (parent scratch, venv with numpy scipy astropy cantera pandas boinor tabulate)

```python
"""Re-optimize both headlines with the Sun arc capped so each pulse carries >= 1 kg of disk (scratch)."""
import sys, time
from dataclasses import replace
sys.path.insert(0, sys.argv[1])
from src import solar_dive_growth as g

def per_pulse(dep, push, d):
    L = g.fly_unit(dep, push, dep, d).lines
    return L["disks"] / (g.SUN_PULSE_RATE * d.arc_hours * 3600), L

for key in g.HEADLINES:
    dep, push = g.departure_route(*key), g.push_route(*key)
    start = g.HEADLINE_STARTS[key]
    cap = start.arc_hours
    for it in range(6):
        g._BOUNDS["arc_hours"] = (0.5, cap)
        opt = g.optimize_design(dep, push, replace(start, arc_hours=min(start.arc_hours, cap * 0.999)))
        d = opt.design
        kg, L = per_pulse(dep, push, d)
        print(key, d, kg, opt.valuation.growth, opt.valuation.doubling,
              opt.valuation.steady_cost, opt.valuation.break_even, opt.dear_break_even,
              L["sun_plate"], L["sun_stroke"])
        if kg >= 1.0 - 1e-3:
            break
        cap = d.arc_hours * kg / 1.0
```

Note: `fly_unit(dep, push, dep, d)` flies the departure-wave unit. The parent did not check
whether `pulse_disk` should be taken on the push-wave unit or the steady blend of the two.
