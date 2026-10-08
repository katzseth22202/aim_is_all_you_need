# The survivable methane chamber replaces the 20 m^3 sphere in the ledger and cost book

Status: accepted

Answers: the parent's S17 (`docs/survivable_chamber_asks_for_aim_repo.md` in
`Balloon-Pulse-Propulsion`, raised 2026-10-08, applying `puffsat_impact_simulation` @ `69d1f40`).
Adds rows; every earlier chamber figure stands as the record of the sphere.

Date: 2026-10-08

## Context

The impact sim ran the 20 m^3 methane sphere with the rod and the plug modelled as material.
The stopped rod is a line blast, and the bonded carbon overwrap delaminates under it. The
chamber that survives one departure fires a **5 kg rod at 2 Hz** behind a **20 kg frozen-methane
plug** (`P` = 4). It is bulged to 212 m^3 at ~94 bar, on a **48.2 t** dry Kevlar-over-Cr-Mo wall
(68.8 t maraging fallback). Its extension is **4.9 t** at A/A* = 100 or **14.7 t** at A/A* = 300.
It carries ~117 kg of charge and plug per pulse, plus 3.5-24 kg of pitch. The impact sim gives
906 / 971 kN s per pulse. The ledger had priced the sphere: 2.5 kg rods at 4 Hz, `P` = 1.76,
a 19 t wall and a 2.19 t extension.

## Decision

1. **The chamber's rod, plug, rate and extension belong to the pairing.** `ChamberPairing`
   gains `rod_mass`, `plug_ratio`, `pulse_rate` and `extension_mass`. Their defaults are the
   sphere's, so every earlier figure is unchanged. They used to be module constants in
   `chamber_isp` / `chamber_departure`, which `growth_ledger`, `growth_cost_inputs` and
   `seed_cost` all read. `chamber_departure.chamber_unit_mass` adds wall and extension.
   `growth_cost.DesignInputs.rod_mass` sets the packages per kilogram of rod: three ride
   every rod, so a 5 kg rod carries half as many per kilogram.

2. **`eta` is backed out of the impact sim's impulse** (`efficiency_for_impulse`). At 75 km/s,
   with the gate charged and pitch excluded as the sim quotes it, 906 / 971 kN s gives
   eta = 0.488 / 0.539. That is 788 / 845 s on 117 kg. The A/A* = 300 value is the sphere's
   solved 0.538, which shows the sim used the same chamber law. `k + P` = 23.4 at 75 km/s
   is taken as given. Holding the sphere's energy per kilogram would need 29.3. Reported back
   as a number to check, not worked around.

3. **Pitch is per kilogram of the pairing's own rod.** `Design.pitch_per_pulse` sets it, and it
   defaults to the sphere's 5.6 kg on 2.5 kg. The matrix carries the heavy end, 24 kg on 5 kg
   (4.8 per kg of rod), as it carried 5.6 kg on 2.5 kg (2.24).

4. **The finite-burn loss is the integrated fixed-direction one, as since ADR 0032.** The
   impact sim's note said the ledger "does not appear to" charge one, which is wrong. Its
   +3.1-4.8% assumed thrust along the velocity. A head-on chamber cannot fly that, so it is a
   labelled sensitivity here (`steered`). The ledger's burns are also longer than the sim's:
   the departing stack behind the spray cup is 780-820 t, not 500-600 t, so a departure takes
   2900-4500 pulses over 24-37 min, not 2000-2650 over 16-21.

5. **Redundancy is flown, not forced.** Two 2.5 kg chambers share the 49.8 t wall and each has
   half the exit area. Both versions pick their own chamber count. A forced count can burn past
   the loss table's hour at the scan's off-optimum plate schedules, so it is not used.

6. **Plugs ride untanked**, as the sphere's foam did. The frozen methane's tank is about 0.034
   of 20 kg, which is negligible. The cost book still prices plugs at the foam's $5/kg, which
   is conservative for frozen methane in a can.

## Results (`make survivable-chamber`, ~5 min on four workers)

Spray cup, film in the book, 20-day orbit. Sphere behind the same plate (ADR 0043): CH4 solved
2.28 yr, H2 solved 2.02 yr, methalox 7.82 yr.

| departure | doubling | annual | 10-yr | n | chamber t | burn | loss | payload t |
|---|---|---|---|---|---|---|---|---|
| AR 100, 3.5 kg | 2.96 yr | 26.4% | 8.9x | 1 | 53.1 | 28-44 min | 477-1013 m/s (8.7-14.0%) | 229 |
| AR 100, 24 kg | 3.04 yr | 25.6% | 8.3x | 1 | 53.1 | 25-38 min | 403-856 m/s (7.4-11.8%) | 223 |
| AR 300, 3.5 kg | 2.69 yr | 29.5% | 11.6x | 1 | 62.9 | 27-43 min | 455-996 m/s (8.3-13.7%) | 247 |
| **AR 300, 24 kg** | **2.76 yr** | **28.5%** | **10.6x** | 1 | 62.9 | 24-37 min | 385-843 m/s (7.1-11.6%) | 240 |

A/A* = 300 wins at both pitch edges: 0.28 yr against 10 t more extension. Two 2.5 kg chambers
cost 0.04 yr at AR 100 (3.08 yr against 3.04) and nothing measurable at AR 300 (2.76). On
some cycles the AR 300 split flies a single 2.5 kg chamber. The maraging wall costs 0.44-0.60 yr.
The steered loss, which cannot be flown, would be worth 0.11-0.14 yr. Shares of the A/A* = 300
ceiling at 24 kg: 50% does not grow, 70% 3.61 yr, 90% 2.38 yr, 100% 2.13 yr.

Cost (Estimate, 10%, 50% odds, lob x1.065), steady $/kg and BE steady cheap / dear:
AR 300 24 kg $136, $248 / $710; AR 100 24 kg $149, $289 / $886. The sphere was $123,
$202 / $467. H2 solved (wall unverified) is $120, $190 / $372, and methalox $288, $762 / >3000.
The book prices a methane chamber per unit, not per tonne. Priced by mass against the sphere's
21.2 t (x2.5-3.0), the rows become $160-171, $335-370 / $796-966.

## Consequences

- The parent can make methane the headline. AR 300 at 24 kg pitch is the conservative row.
- `make survivable-chamber` (`survivable-ledger`, `survivable-cost`) reproduces every cell.
  The cases run in worker processes, and they are passed by index: a pickled pairing is a
  copy, and the ledger tells hydrogen from methane by identity.
- Open: the hydrogen chamber's wall (S18, impact sim); a per-tonne chamber price; the plug's
  price as frozen methane; the A/A* = 100 chemistry ceiling.
