# ADR 0050: Recycle spent departure hardware into rods, and build it to be recycled

Date: 2026-10-12

Status: Accepted as an accounting sensitivity; the in-flight processing is not designed.

Supersedes: ADR 0046's allocation of recovered mass and its Pinatubo comparison.
ADR 0046's recovery shares (90% of tanks, 60% of chamber) and its 5 t, $1M
processor are kept as assumptions.

## Context

After the departure burn the empty tanks and the expended chamber ride on toward
Jupiter with the payload anyway. ADR 0046 proposed cutting and packaging them during
the three-year cruise so that they come back as part of the next return. It priced
that on the common circular 3S route from a recorded snapshot whose generator was
never committed. It also let the recovered mass become growth PuffSats and departure
rods in their usual ratio.

That ratio is not available to solid hardware. Growth PuffSats are water, film and
emulsion that disperse into gas before they reach the plate. Departure rods meet the
chamber head-on at 60-75 km/s and vaporize. So recovered pieces can always be rods.
They can be growth-wave mass only as impactors that spread water PuffSats over a
plate met at 44 km/s or faster, the Jovian-return closing speeds. That is a plausible
design, and the impact simulation has not checked it.

A denser rod costs the chamber nothing. The plug must be at least
`L sqrt(rho_rod / rho_plug)` deep. A rod of the same mass and width is shorter by
`rho_rod`, so its plug is shallower and lighter (paper `tab:material_chamber_plug`).
What a metal or composite rod does add is condensed-phase loss in the nozzle, at
about 4% of the flow, and its elements in the exhaust.

## Decision

1. **The cost model carries recovery.** `growth_cost.HardwareRecovery` and
   `recovered()` transform the flown chain. `CycleHardware` now records each
   chamber's wall-plus-extension mass. Each growth launch unit carries a 5 t
   processor, which displaces onward payload and costs `PriceBook.processor`
   ($1M). By default recovered mass stands in only for the rods. The launched
   payload must still supply every growth PuffSat, so onward mass can grow at most
   to `launched / (1 - rod share)`. `plate_takes_metal` lifts that cap.
   `steady_exhaust()` reports launches and recycled tonnes per year for one seed's
   held-level fleet.
2. **Build the hardware to be recycled:**
   - **Tanks:** carbon fibre in PEEK or PEKK, with a thin steel liner against
     hydrogen permeation. No aluminium. No aluminium-lithium (nature supplies
     0.02 t of lithium a year), and no aluminium-copper 2xxx.
   - **Methane chamber:** its Cr-Mo steel is recycled.
   - **Hydrogen chamber:** not recycled. Its liner is copper, which nature supplies
     at 2 t a year. Its tanks alone cover its rods in most cycles.
   - **Nozzle extension backing:** carbon fibre or carbon-carbon. Prefer carbon to
     aramid (Kevlar), which adds nitrogen and is harder to cut.
   - **Never in the exhaust:** brominated resins, fluoropolymers, niobium or
     copper.

   Steel tabs bonded along the planned cut lines give robots something to grip
   magnetically. Carbon fibre has no magnetic grip of its own.
3. **Prefer carbon fibre over aluminium for pressure management throughout.**
   Aluminium was the cheap first choice. But recycled aluminium reaches the meteoric
   aluminium background, 142 t a year, at about five departures a year, close to the
   paper's own harvest scale.

## Results

Run `make hardware-recovery` (~2 min). It uses the paper's flown chain behind the
spray cup, with the lob climbing at 1.1 km/s and the Estimate book. Methane chambers
are priced by mass, as in ADR 0044. Odds and discounting follow ADR 0040: 10% a year
and 50% odds. The seed is $337 / $9,293 per kg.

| Design | Recovery | Growth, geometric mean | Steady $/kg | Break-even, cheap / dear seed |
|---|---|---:|---:|---:|
| CH4 AR 300, 24 kg pitch | none | 1.905 | 161.7 | 340 / 810 |
| | rods only | **2.178** | **136.3** | **255 / 528** |
| | rods + plate | 2.253 | 130.9 | 236 / 472 |
| Hydrogen, solved | none | 2.418 | 121.6 | 192 / 377 |
| | rods only | **2.723** | **110.4** | **161 / 283** |
| | rods + plate | 2.912 | 104.5 | 144 / 236 |

The no-recovery rows reproduce the paper's $162 and $122 per kg. Rods-only recovery
raises growth per cycle by 13-14%. It cuts the steady cost by $11-25/kg and the
dear-seed break-even by about a third. ADR 0046's proportional allocation overstated
the hydrogen gain. On this chain it would give 3.09, not 2.72.

## Exhaust budget

Every recycled kilogram is eventually fired, so it ends up in the exhaust. Most
departure exhaust leaves slower than orbital speed. On the 3000 K miss-safe burn at
2,631 km, the exhaust falls back while the vehicle is below about 13 km/s, roughly
two-thirds of the burn. A 5500 K chamber's faster exhaust falls back almost
throughout. Take the falling share as 0.6-1. That is a hand bracket, not yet
integrated from the burn.

The relevant natural background is meteoric ablation, not volcanic ash.

**Natural injection, tonnes a year** (Schulz et al. 2025, arXiv 2510.21328, Table 5):

| Element | Natural input | Satellite re-entry, 2024 |
|---|---:|---:|
| Iron | 2,477 | 83 |
| Magnesium | 1,404 | 4.7 |
| Aluminium | 142 | 397 |
| Copper | 2 | 52 |
| Lithium | 0.02 | 1.7 |
| All material | 12,325 | 887 |

**Why volcanic ash is the wrong comparison.**
- Meteors vaporize at 80-110 km and recondense as nanometre smoke. Recycled hardware
  in the exhaust would most likely follow the same path.
- Volcanoes inject far more mass, but as micron silicate ash with its metal bound in
  minerals. About 98% of it falls out within days.
- Per kilogram, nanometre particles carry about a thousand times more reactive
  surface than micron particles.

The particle size of condensed exhaust is unknown. It could be nanometre smoke or
micron droplets formed in the plume. That is the main uncertainty here.

**Per seed's held-level fleet, all recycled mass falling back:**

| Design | Recovery | Launches / yr | Tank t / yr | Chamber t / yr | Seeds to equal meteoric Fe | Seeds to equal today's rocket soot |
|---|---|---:|---:|---:|---:|---:|
| CH4 AR 300 | rods only | 1.39 | 11.3 | 42.2 | 61 | 100 |
| Hydrogen | rods only | 2.15 | 84.4 | 0 | - | 13 |

- **Chamber steel** is compared with meteoric iron at 97% Fe. A methane program
  matches nature's iron at about 60 seeds' fleets.
- **Tank carbon** is compared with today's rocket soot, about 1,000 t a year
  (Maloney et al. 2022), at 88% carbon. Counting every atom as soot overstates this.
  The design's methane and polyethylene rods already put far more carbon in the
  exhaust than the tanks add. How much of the exhaust carbon becomes soot, and where
  it falls, is an open question for the whole design, not for recovery alone.

## Why now

The rocket uses robotics to eat part of itself. Laser cutting, robotic handling and
in-space manufacturing make this a credible near-term task. It is also an
unhurried one: about 40 t of hardware per launch over a three-year cruise is
roughly 35 kg a day. The paper can make this its "why now". Citations for the robotics claim are
still owed.

## Limitations

- The recovery shares, processor mass and price are assumptions. The processor's
  power, radiator and consumables are not sized.
- Recovered rods are charged ordinary rod-body and package prices, with no credit for
  virgin material not bought.
- Exhaust performance is held fixed. A composite or steel rod's two-phase nozzle
  loss is not charged.
- Any correction burn after departure now also moves the carried hardware. That is
  small and not yet charged.
- `plate_takes_metal` is an accounting switch. The impactor-blast plate has not been
  simulated, and metal vapour condensing on the plate face is unexamined.
- The falling-back share is a bracket. Integrating it from the burn's exhaust
  states is the next refinement.

## Reproduction

`make hardware-recovery`. `pytest tests/test_growth_cost.py` checks the rod cap,
the plate switch, the processor charge and the yearly tally on hand-built chains.
