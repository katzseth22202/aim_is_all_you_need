# Spray cup and plug: every cell the parent prints, for the parent paper

Raised 2026-10-06 in `aim_is_all_you_need` (ADR 0042), answering the parent's asks S10 and S11
(`docs/deferred_to_companion_repos.md` there). Supersedes the cost columns of
`docs/plate_designs_for_parent.md`, whose lob was charged from an apex. Reproduce with
`make plate-grid` (~40 min), `make plate-cost` (~20 min) and `make plate-seed` (~15 min).

## What changed

- **The climbing lob is charged from 0.75 km/s, not from an apex.** The parent's lob already
  passes 400 km at ~0.75 km/s. From there to the 1.1 km/s operating point costs 4.4% of the
  lofted mass (380 s): lob $25 -> $26.1/kg lofted, booster ~1.13-1.28x Super Heavy.
- **Every cell has a target.** Full chamber-share grid, k <= 10 beside the cap, hold, pitch,
  all-in water and 10-day sensitivities; the full cost report; seed amortization; deliveries and
  grow-or-harvest. All on the 20-day orbit.

## Ledger (`make plate-grid`), doubling yr at k <= 8.52 (k <= 10)

| Departure | Spray cup 0.60 | Plug 0.70 |
|---|---|---|
| H2 50% of ceiling | 8.15 (7.89) | 5.53 (5.37) |
| H2 70% | 2.65 (2.62) | 2.29 (2.26) |
| H2 solved | 2.02 (2.00) | 1.80 (1.78) |
| H2 90% | 1.97 (1.96) | 1.76 (1.75) |
| H2 100% | 1.81 (1.80) | 1.63 (1.62) |
| CH4 50% | 7.08 (6.84) | 5.03 (4.86) |
| CH4 70% | 2.73 (2.69) | 2.35 (2.31) |
| CH4 solved | 2.28 (2.25) | 2.01 (1.98) |
| CH4 90% | 2.04 (2.02) | 1.82 (1.80) |
| CH4 100% | 1.87 (1.85) | 1.68 (1.66) |
| Methalox | 7.82 (7.69) | 5.54 (5.44) |

Ten-year multiples behind the spray cup: H2 solved 28.6, CH4 solved 19.0, methalox 2.3. H2 at
half its ceiling shrinks the fleet on its first cycle (G_0 = 0.98).

Sensitivities behind the spray cup: hydrogen's hold 1.94-2.08 yr (0-5% cryostats, 0-0.5%/day);
methane's pitch 1.4-5.6 kg moves 2.26-2.28 yr; water at all-in η_jet 0.50/0.53/0.56 doubles H2
in 2.44/2.35/2.26 and CH4 in 2.84/2.71/2.60, against argon's 2.02/2.28; a 10-day orbit gives
2.01 / 2.30 / 9.51 (H2 / CH4 / methalox).

## Cost (`make plate-cost`), Estimate, 10% a year, 50% odds, lob x1.044

| Plate | Departure | Steady $/kg | BE steady cheap / dear | BE liquidation |
|---|---|---|---|---|
| Spray cup | H2 solved | 119 | 189 / 370 | 225 / 697 |
| | CH4 solved | 122 | 200 / 465 | 237 / 877 |
| | Methalox | 285 | 755 / >3000 | 565 / >3000 |
| Plug | H2 solved | 110 | 160 / 270 | 191 / 495 |
| | CH4 solved | 111 | 167 / 325 | 198 / 611 |
| | Methalox | 215 | 509 / 2536 | 467 / >3000 |
| ADR 0033 (ref.) | H2 solved | 103 | 137 / 197 | 162 / 337 |
| | CH4 solved | 103 | 140 / 225 | 165 / 403 |
| | Methalox | 174 | 360 / 1404 | 382 / 2176 |

Behind the spray cup: Pessimistic book $501 / $555 / $863 steady, solved chambers $692-793 on the
cheap seed; 25% odds puts dear-seed H2 at $575; a solved chamber matches methalox only at
$194-241M a unit or $6300-8300 a package; the Earth loop takes dear-seed H2 to $324.

## Seed and harvest (`make plate-seed`)

Seed 102-112 t per launch unit, 1.11-1.22 stripped ships; seed per fleet kg after ten years
$12-325 (H2 solved), $147-4065 (methalox). At $500 the best delivery carries 866-892 t at P
7.6-9.5, lob $44-45 per cargo kg, $64-66 delivered on the learned plate, $131-135 on the early
one. At 30% cost of capital solved H2 clears 73% of its cycles and methalox none; steady over
liquidation 1.19 (H2), 1.11 (CH4), 0.44 (methalox).

## Not done

Film mass (parent S12), the booster's brake at the faster climb, the plug in the deep bowl.
