# Straw Way energy from the Ceres loop: ask S58 (for `aim_is_all_you_need`)

Raised 2026-10-22 against aim `0aa5813` (S56's answer, ADR 0083). **Written to be copied verbatim
into `katzseth22202/aim_is_all_you_need`**, so it repeats context the paper repo already has.
Register entry: S58 in `docs/deferred_to_companion_repos.md`.

Same ground rules as your own handbacks. Neither document is the source of truth. If a number here
looks wrong, say so rather than working around it. The figures below come from a parent scratch
harness that calls your `src/ceres_dive.py` directly; the harness is reproduced in full at the end.
None of it is a companion result yet.

**The paper already prints these numbers** (parent `247db94`, `sec:methalox_rebuttal` and
`sec:strawway_economics`), citing ADR 0083's handback for the ledger. Seth chose to fix the
paper's old claim first, because it was wrong by a factor of ten. That claim was $0.22/MWh raw,
"well under a dollar per megawatt-hour". This ask closes the gap. Once you reproduce the figures,
or correct them, the paper will cite your answer instead of the parent scratch.

## The question (Seth, 2026-10-22)

Use the mature Ceres loop to make electricity on Earth instead of delivering payload to L1.

1. **No Earth lob.** The route is Ceres (or a depot of the ring) → the Sun Oberth dive → the Straw
   Way to Heaven, a vacuum tube from the ground to space that the impactors fly down. The L1
   delivery, its lob and the Earth plates are gone from the energy price.
2. **Mass still returns to Ceres to start the next cycle.** The Ceres-bound ships and their stream
   are unchanged from ADR 0083; the Earth-bound ships' share of the Ceres stream is still charged.
3. **The straw has magnetic active support**, and the converter at its bottom is
   magnetohydrodynamic (MHD), 50% efficient from kinetic energy to electricity, fed by plasma pulses
   at 20 Hz. There is no backspray from the previous pulse, so the pulse rate is free.
4. **The mature book**: everything but electronics is made at Ceres. That means S57's book. Plates,
   Sun bowls, bags and tiles are Ceres-made at $10/kg, with 1% of their mass imported at the Earth
   hardware price plus freight. Avionics and steering packages stay Earth-made.
5. **Price the impactors only.** The straw, the MHD plant, the heat sink and the Ceres industry's
   capital are left out.
6. **Open question: what arrival speed?** Is it worth arriving faster than the paper's 137 km/s,
   even though it burns more propellant at the Sun?

## What the parent did

The harness reuses your solver; it changes no physics. Per design:

1. **Freight**: `steady_loop(design)` is called once for its solved freight, then held fixed. This
   is the L1 loop's freight for the same design (it still flies sheets from Earth). See item 4 in
   Wanted.
2. **Both ships** are flown with `fly_ship(design, prices, "ceres" | "earth", freight, ref_ve)`,
   `ref_ve` exactly as `steady_loop` computes it.
3. **Cargo**: the Earth-bound ship's `dollars["cargo"]` is re-priced at `prices.water` (ice
   pellets) instead of `prices.film` (sheets). Steering packages stay charged on the cargo at the
   book's rate (`bound == "earth"` already charges them).
4. **S57's plate book**: `dollars["sun_bowl"]` and `dollars["ceres_plates"]` on both ships are
   multiplied by `(0.99 p + 0.01 H) / H`, where `H = earth_hardware + freight` and `p = $10`.
   `avionics` is left at `H`.
5. **Stream prices** with the freight fixed:
   `c_C = sum(C.dollars) / (C.cargo (1 - misses) - C.stream)`, then
   `c_E = (sum(E.dollars) + E.stream c_C) / (E.cargo (1 - misses))`.
6. **Energy**: speed at the ground `v = sqrt(w_400^2 + 2 mu (1/R_E - 1/(R_E + 400 km)))`, which
   adds only 0.03 km/s at 137. Electricity per kg is `0.5 * (1/2) v^2`, and the price is
   `$/MWh_e = c_E / MWh_e`.
7. **Optimizer**: Powell over `k_sun` and `arc_hours` on `CERES_BOUNDS`, from the best node of
   `SUN_GRID x ARC_GRID`, minimizing $/MWh_e. `X_C` is held at ADR 0083's optima (110 at 4 R_sun,
   90 at 9.86), `k_delivery` at 5, and the plate policy is `recover`.

**Reproduction check.** With film cargo and S56's all-Earth plates, the harness gives c_C
$5.69 and c_E $10.739. Both match ADR 0083's ledger exactly.

## The parent's numbers

### Arrival-speed sweep, 4 R_sun, X_C 110, mature book (Ceres plates $10)

| w_400 km/s | X_E | k_sun | arc h | c_C | c_E $/kg | MWh_e/kg | **$/MWh_e** |
|---|---|---|---|---|---|---|---|
| 100 | 87.0 | 29.9 | 2.6 | 4.38 | 3.18 | 0.695 | 4.58 |
| 137 | 127.9 | 29.9 | 2.4 | 4.35 | 5.83 | 1.304 | **4.47** |
| 175 | 168.0 | 29.9 | 2.4 | 4.34 | 11.57 | 2.127 | 5.44 |
| 200 | 194.0 | 29.9 | 2.4 | 4.34 | 19.28 | 2.778 | 6.94 |
| 250, 300 | | | | | | | no flyable design |

Freight solved at $4.30-5.08 across the sweep. Above about 200 km/s, `fly_ship` raises "the Sun
bowl's survival floor outweighs the ship it pushes".

### The same on S56's all-Earth plates ($30 + freight)

| w_400 | 100 | 120 | 137 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|
| $/MWh_e | 5.95 | 5.74 | 5.88 | 6.16 | 7.19 | 9.22 |

S57's book with Ceres plates at $30 gives $5.63 at 137 and $8.81 at 200.

### Parker's depth, 9.86 R_sun, X_C 90

| w_400 | 80 | 99 | 120 | 150 | 175 |
|---|---|---|---|---|---|
| $/MWh_e, Earth plates | 11.08 | 10.99 | 12.23 | 16.67 | 25.46 |
| $/MWh_e, Ceres plates $10 | | 9.75 | | 14.74 | |

### Line items, 4 R_sun, 137 km/s, mature book ($ per caught kg)

| line | $ | line | $ |
|---|---|---|---|
| ceres_stream | 1.216 | sun_mounts | 0.108 |
| disks | 1.085 | ceres_film | 0.088 |
| sun_charges | 1.082 | water_charges | 0.088 |
| avionics | 0.764 | depot_tug | 0.048 |
| sun_bowl | 0.653 | sun_film | 0.038 |
| charge_mounts | 0.439 | carrier | 0.036 |
| shield | 0.111 | packages | 0.034 |
| cargo (ice) | 0.021 | ceres_plates | 0.017 |
| **total c_E** | **5.83** | | |

### Sensitivities, one at a time, design held at the 137 km/s optimum (k_sun 29.907, arc 2.4)

| lever | $/MWh_e |
|---|---|
| base | 4.47 |
| polyethylene (Sun charges) $0.20 instead of $1 | 3.64 |
| Sun disks $10 instead of $30 | 3.78 |
| Earth hardware $10 instead of $30 | 3.98 |
| Ceres structure $3 instead of $10 | 3.99 |
| `AVIONICS_FRACTION` halved (0.5%) | 4.04 |
| misses 0 instead of 3% | 4.30 |
| packages free | 4.43 |
| all of the above together | 1.78 |

The paper now prints 4.47 (and 5.88 for Earth-made plates, 6.94 at 200 km/s and 9.75 at Parker's
depth) and says that arriving faster does not help.

### 20 Hz sizing (parent arithmetic, not printed)

At 137 km/s a kilogram gives 4.69 GJ of electricity. One TW of electricity needs 213 kg/s, which
is 10.7 kg per pulse at 20 Hz. That is close to ADR 0083's 7.5 kg pellets. At 7.5 kg and 20 Hz,
one straw makes 0.70 TW.

## S58. Wanted

Extend `src/ceres_dive.py`, or add `src/ceres_energy.py` on top of it if that is cleaner. Keep
ADR 0083's headline reproducible beside the new one. Add a `make ceres-energy` target.

1. **An energy objective.** Add a function that prices a caught kilogram of the Earth-bound
   stream as an impactor and returns $/MWh electric. Make the cargo price a parameter (water by
   default for energy, film for L1), and the conversion efficiency too (default 0.5). Use the ground
   speed, not the 400 km speed. Return the line items as in `ceres_lines`.
2. **S57's plate book as real prices.** Add `ceres_plate` (default $10/kg) and `plate_import`
   (default 1%) to `CeresPrices`. They apply to the Ceres plates, the disk plates, the Sun bowl, the
   bore collar and their bags; avionics stays at `earth_hardware` plus freight. This is S57's item 1,
   which was parked for L1 but matters here. At L1 the plates were $0.05 of $1.46. Here the Sun bowl
   line alone moves from $1.91 to $0.65 per caught kg.
3. **The sweep.** Arrival speed w_400 from 80 to 300 km/s at 4 and 9.86 R_sun, re-optimizing
   `k_sun` and `arc_hours`, and **also `X_C`**, which the parent held at ADR 0083's L1 optima.
   Report the $/MWh_e minimum and how flat it is. Report the speed where the Sun bowl's survival
   floor stops the design. If the minimum moves off 120-137 km/s, say why.
4. **Freight, solved for the energy loop.** The parent held the freight at the L1 loop's value
   ($4.30-5.08). In the energy loop, the Earth hardware (avionics, packages, the 1% import) still
   has to reach Ceres. Re-solve the freight as a fixed point on whatever Earth leg now carries it.
   Say whether it still flies the solar dive's lob, and what it costs. Seth's framing is "no Earth
   lob" for the energy itself. The hardware freight may still need one.
5. **Plate policy re-optimized.** ADR 0083 item 7 chose `recover`. With cheaper plates and no
   delivery, does `expend` or `sun_bowl` now win, and at what Hz and push length?
6. **Ice pellets as Earth-bound cargo.** The Ceres-bound stream already flies ice pellets through
   the dive behind the shade. Confirm that the same holds on the Earth-bound dive at each speed of
   the sweep, or say what the cargo needs instead. The pellets must arrive guided, at the
   straw's aim, at 20 Hz.
7. **Sensitivities.** Re-run the table above on the re-optimized design, plus ADR 0083's item 8
   risk rows that apply: dead mass at 15%, Earth hardware at $100, Ceres film at $10, and freight
   at $3 / $10. State the lowest price on the book's most optimistic Ceres prices, and which line
   sets it.
8. **The packages' mass** (S57 item 3). `steady_loop` charges packages per piece, with no mass and
   no freight. For impactors at 7.5-11 kg rather than 60 kg sheets, packages per kg may rise. State
   the package count per kg you use for the energy stream and charge freight on its Earth-made
   mass.
9. **What is left out.** List the uncharged lines, including the straw, the MHD plant, the Ceres
   industry, the seed and the tug reactor. Add anything else that reaches $0.10/MWh_e.

Report as usual. Write a handback doc for the parent, an ADR, and the revision to cite.

## The parent's harness

`ceres_energy.py`, run as `python3 ceres_energy.py <aim checkout> <depth> <w_400 list> [plate $]`.
For example, `python3 ceres_energy.py ./aim 4.0 100,137,175,200 10` gives the mature-book sweep,
and omitting the last argument gives S56's all-Earth plates. Each speed takes about a minute.

```python
"""Price Ceres-grown impactors as Straw Way energy, on aim 0aa5813's Ceres loop.

The Earth-bound ship's cargo is ice pellets at the water price instead of
film sheets. The freight on Earth hardware is held at the L1 loop's solved
value for the same design. Both stream prices are then re-solved:
    c_C = O_C / net,   c_E = (O_E + S_E c_C) / Y_E.
The sweep is over the Earth stream's speed at 400 km, with the Sun's k and
arc re-optimized to minimize $/MWh electric.
"""

import math
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, sys.argv[1])
from src import ceres_dive as cd  # noqa: E402

MU, RE = 398600.4418, 6378.137
ETA = 0.5  # MHD, kinetic to electric


def ground_speed(w_400: float) -> float:
    return math.sqrt(w_400**2 + 2 * MU * (1 / RE - 1 / (RE + 400.0)))


PLATE = None  # Ceres plate price ($/kg); None keeps S56's all-Earth plates
IMPORT = 0.01  # Earth-made share of a Ceres plate (S57)


def _ceres_plates(dollars, hardware):
    """Re-price the plate and bowl lines on S57's book (avionics stay Earth)."""
    if PLATE is None:
        return dollars
    ratio = ((1 - IMPORT) * PLATE + IMPORT * hardware) / hardware
    out = dict(dollars)
    for key in ("sun_bowl", "ceres_plates"):
        out[key] = dollars[key] * ratio
    return out


def energy_price(design, prices=cd.CERES_BOOK, cargo_price=None):
    """(c_C, c_E per caught kg, freight, $/MWh_e, details) for one design."""
    loop = cd.steady_loop(design, prices)  # supplies the freight and reference ve
    flown = cd.routes_for(design)
    laws = cd.StageLaws.load()
    _, w_e = cd.earth_stream(flown.earth)
    earth_design = replace(
        design.spring, k_delivery=design.k_delivery, slow_delivery=True
    )
    ref_ve = laws.push_ve(
        design.k_delivery, w_e, cd.sheet_material(w_e), earth_design.push_law
    )
    caught = 1.0 - prices.misses
    c = cd.fly_ship(design, prices, "ceres", loop.freight, ref_ve)
    e = cd.fly_ship(design, prices, "earth", loop.freight, ref_ve)
    hardware = prices.earth_hardware + loop.freight
    net = c.cargo * caught - c.stream
    c_c = sum(_ceres_plates(c.dollars, hardware).values()) / net
    dollars = _ceres_plates(e.dollars, hardware)
    dollars["cargo"] = (prices.water if cargo_price is None else cargo_price) * e.cargo
    y_e = e.cargo * caught
    lines = {k: v / y_e for k, v in dollars.items()}
    lines["ceres_stream"] = e.stream * c_c / y_e
    c_e = sum(lines.values())
    v = ground_speed(w_e)
    mwh_e = ETA * 0.5 * (v * 1e3) ** 2 / 3.6e9
    return c_c, c_e, loop.freight, c_e / mwh_e, dict(
        lines=lines, w=w_e, v=v, mwh=mwh_e, loop=loop, e=e, x_e=flown.earth_excess
    )


def optimize(design, prices=cd.CERES_BOOK):
    names = ("k_sun", "arc_hours")
    best = []

    def f(x):
        trial = replace(design, **dict(zip(names, map(float, x))))
        try:
            val = energy_price(trial, prices)[3]
        except (ValueError, ZeroDivisionError, RuntimeError, np.linalg.LinAlgError):
            return 1e6
        if not best or val < best[0][0]:
            best[:] = [(val, trial)]
        return val

    grid = [([k, a], f([k, a])) for k in cd.SUN_GRID for a in cd.ARC_GRID]
    start = min(grid, key=lambda r: r[1])[0]
    minimize(f, start, method="Powell",
             bounds=[cd.CERES_BOUNDS[n] for n in names],
             options={"xtol": 1e-3, "ftol": 1e-6})
    if not best:
        return None
    return best[0][1], energy_price(best[0][1], prices)


if __name__ == "__main__":
    speeds = [float(s) for s in sys.argv[3].split(",")]
    depth = float(sys.argv[2])
    if len(sys.argv) > 4:
        PLATE = float(sys.argv[4])
    x_c = 110.0 if depth == 4.0 else 90.0
    print(f"depth {depth} R_sun, X_C {x_c}, Ceres plate {PLATE}")
    print("w400  v_gnd  X_E    k_sun arc   c_C    c_E($/kg)  MWh_e/kg  $/MWh_e")
    for w in speeds:
        d = cd.CeresDesign(depth=depth, ceres_excess=x_c, earth_stream=w)
        try:
            r = optimize(d)
        except Exception as exc:  # noqa: BLE001
            print(f"{w:5.0f}  failed: {exc}")
            continue
        if r is None:
            print(f"{w:5.0f}  no flyable design")
            continue
        dd, (c_c, c_e, fr, pm, det) = r
        print(f"{w:5.0f}  {det['v']:5.1f}  {det['x_e']:5.1f}  {dd.k_sun:5.1f} "
              f"{dd.arc_hours:4.1f}  {c_c:5.2f}  {c_e:7.2f}    {det['mwh']:6.3f}   "
              f"{pm:6.2f}   freight {fr:.2f}")
        top = sorted(det["lines"].items(), key=lambda kv: -kv[1])[:7]
        print("       " + ", ".join(f"{k} {v:.2f}" for k, v in top))
```

The sensitivity rows were run by importing this module, setting `PLATE = 10.0`, fixing the design
at `k_sun=29.907, arc_hours=2.4`, and calling `energy_price` with `replace(CERES_BOOK, ...)`. The
avionics row patched `cd.AVIONICS_FRACTION` to 0.005 for the call.
