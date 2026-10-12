# ADR 0051: Store the hydrogen charge as slush

Date: 2026-10-12

Status: Accepted as the preferred hydrogen storage. No flight-scale slush system
exists.

## Context

The hydrogen chamber's charge has been carried as normal-boiling-point liquid
(NBP, 70.8 kg/m3). The tank is 0.205 kg per kg of hydrogen, scaled from NASA's
Mars drop tank. A 3% cryostat is dropped with the plate. 0.3% boils off each day
of the 20-day hold, about 6% of what is launched.

A liquid hydrogen tank's mass is set mostly by volume, insulation and launch
loads, not pressure. Its pressure is only 1-3 bar. Solid hydrogen would not remove
that structure. At the 13.8 K triple point its vapour pressure is 0.07 bar, so a
thin tank would lose its pressure stabilization and face 1 bar of crush load on
the ground. The gains from colder hydrogen are density and heat sink.

Slush hydrogen at 50% solid keeps most of both and stays pumpable. NASA Lewis made,
transferred, pumped and pressure-expelled it at the K-Site and Small Scale Hydrogen
Test Facility in 1990-94 for the National Aero-Space Plane. The program reports a
15% density gain at 50% solid and 18% more heat capacity than NBP liquid (McNelis
et al., NTRS 19950017766; Hardy and Whalen, NASA TM 105596 and TM 105191).

## Decision

Carry the hydrogen charge as 50% slush, made on the ground by freeze-thaw.
`chamber_isp.HydrogenStorage` now holds each storage's density, cryostat share and
daily boil-off. `ChamberPairing.storage` marks a pairing as hydrogen; that check
replaces `is HYDROGEN_5500K` in the chain path and survives pickling.
`stored_as()` scales the tank by the density ratio:

| Storage | Density, kg/m3 | Tank, kg per kg H2 | Cryostat share | Boil-off per day |
|---|---:|---:|---:|---:|
| NBP liquid (`LIQUID_HYDROGEN`) | 70.8 | 0.205 | 3% | 0.3% |
| 50% slush (`SLUSH_HYDROGEN`) | 81.4 | 0.178 | 3% | **0** |

**Why slush boils nothing over the hold.** Slush absorbs heat before any of it
boils. Melting the solid takes 58.2 kJ/kg on half the mass. Warming the liquid
from 13.8 K to 20.3 K takes about 52 kJ/kg more. Together that is about 81 kJ/kg,
18% of the 446 kJ/kg needed to boil it. That matches NASA's 18% heat-capacity gain.
The insulation that boils 0.3% a day of liquid would take about 60 days to use up
that margin. A 20-day hold therefore loses nothing. The ullage is pressurized from
0.07 bar as it warms, as NASA did with helium. The same insulation and cryostat are
assumed, with no active cooling.

## Results

Run `make slush-hydrogen` (~3 min). It flies the solved hydrogen chamber behind the
spray cup, lob climbing, Estimate book, 10% a year, 50% odds. The seed is $337 /
$9,293 per kg. Tank, hydrogen and cryostat masses are cycle 0, per launch unit.

| Design | Tank, t | H2 launched, t | Growth, geometric mean | Steady $/kg | Break-even, cheap / dear seed |
|---|---:|---:|---:|---:|---:|
| Liquid | 70.6 | 366.6 | 2.418 | 121.6 | 192 / 377 |
| Liquid, tanks recovered (ADR 0050) | 70.6 | 366.6 | 2.723 | 110.4 | 161 / 283 |
| **Slush** | **63.3** | **355.2** | **2.560** | **115.0** | **174 / 323** |
| **Slush, tanks recovered** | 63.3 | 355.2 | **2.877** | **105.3** | **147 / 245** |
| Slush, 5% cryostats | 62.8 | 352.1 | 2.540 | 117.1 | 179 / 332 |
| Slush, 5% cryostats, tanks recovered | 62.8 | 352.1 | 2.853 | 107.1 | 151 / 252 |
| Slush, +$2/kg for hydrogen | 63.3 | 355.2 | 2.560 | 115.4 | 175 / 324 |

- **Growth per cycle:** slush raises it by 5.9% on its own. With recovered tanks the
  gain is 19% over liquid without recovery. The two levers stack.
- **Cost:** the dear-seed break-even falls from $377 to $245/kg.
- **Sensitivities:** heavier cryostats give back about a seventh of the gain. A $2/kg
  slush premium costs $0.4/kg at L1.
- **Recycling interaction:** the smaller tank leaves less to recycle. Hydrogen's
  tanks still cover its rods in most cycles.

## Limitations

- **Holds beyond about 60 days.** Holds longer than about 60 days at the same
  insulation, such as the paper's months-long fly-and-park option, start boiling again. They need either
  a cryostat or the liquid's loss.
- **Ground-to-hold time.** Time on the ground, the lob and the push also draw on the
  heat sink before the hold. Their heat leak is not budgeted separately.
- **Solid fraction.** The solid fraction degrades in transfer. NASA measured density
  loss in its transfer line; the ledger takes the 15% at 50% solid without that
  loss.
- **Feed rate.** The chamber's 200-400 kg/s feed is far beyond K-Site's flow tests.
  Melting the feed against the hot chamber wall is a natural fit, but it is not
  designed.
- **Price.** The production premium is a guess; the sensitivity brackets it.

## Reproduction

`make slush-hydrogen`. `pytest tests/test_chamber_isp.py` checks the tank scaling,
the hydrogen marker and the heat-sink arithmetic.
