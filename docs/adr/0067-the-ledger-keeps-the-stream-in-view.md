# ADR 0067: The ledger keeps every departure's stream in view by default

Date: 2026-10-14

Status: Accepted. Makes ADR 0066's constraint the default and re-runs the
headline reports on it and on ADR 0065's slush. The parent's note is
`docs/slush_and_visibility_for_parent.md`.

## Context

ADR 0066 found that the flown chain's streams arrive 154-175.5 degrees from the
departure, so a burn centred on periapsis loses them behind Earth. It charged
the constraint in the lever ranking only. The cost book, seed and harvest still
flew the centred table, and so did the growth ledger's matrix.

## Decision

1. **`growth_ledger.price_cycle_growth` keeps the stream in view whenever no
   loss model is passed** (`visible_loss`). A real cycle missing from the
   stored tables raises rather than falling back. A synthetic cycle has no
   stream and must pass its loss model, as every test's already did.
2. **The tables are stored** in `data/departure_visibility.csv`, because
   building them takes about 6 minutes a process. It holds the flown and
   three-synodic-only chains on the 20- and 10-day orbits, 40 cycles, keyed by
   dates, onward burn and orbit. Two chains can share a cycle's dates yet feed
   different onward windows. `make departure-visibility` rebuilds it, and a slow
   test regenerates the streams and one node.
3. **`make departure-levers` keeps the stream in view**; `make
   departure-levers-centred` gives the old ranking.
4. **The survivable-chamber ledger now charges hydrogen's cryostats and
   boil-off.** Like `departure_levers` before ADR 0065, it charged neither.
5. **A vehicle exactly on the stream's asymptote line** is reached by the
   radial route the conic solver cannot represent; it counts as visible.

## Results

Estimate book, spray cup, film carried, lob climbing, 10% a year, 50% odds.
"Before" is the repo at `f95033b` (liquid, centred; ADR 0064).

**The flown hydrogen chamber** (bulged, 4884 K, 2 Hz; `make survivable-cost`,
`make hardware-recovery`, `make departure-levers`):

| | Before | Slush, centred | **Slush, in view** |
|---|---|---|---|
| Doubling | 2.54 yr | 2.42 | **2.61** |
| Annual growth, `tab:seed_amortization` | 29.7% | | **30.4%** |
| 10-yr multiple | 11.4x | | **12.8x** |
| Seed | 108 t, 1.18 ships | | **108 t, 1.17 ships** |
| Steady $/kg at L1 | $157 | | **$154** |
| Paper's prices / pessimistic | $150 / $686 | | **$148 / $675** |
| Steady break-even, cheap / dear | 385 / 726 | | **369 / 690** |
| With tanks recovered | | | 2.266 growth, $133, 274 / 469 |
| With tanks and plate recovered | 2.491, $120, 219 / 346 | | **2.468, $121, 222 / 354** |

The annual and multiple columns come from `tab:seed_amortization`, which
compounds the chain's cycles differently from the lever ranking's doubling.

**The methane chamber** (AR100, 24 kg pitch; no storage to change, so this is
visibility alone):

| | Centred | **In view** |
|---|---|---|
| Doubling | 3.36 yr | **3.59** |
| Annual, 10-yr | 22.9%, 6.5x | **21.3%, 5.79x** |
| Steady $/kg (Estimate) | | **$177** ($209 priced by mass) |
| At 3 Hz | 3.02 yr | **3.10 yr, $152** |

**The legacy sphere** ("Hydrogen, solved", `DESIGNS[4]`, the parent's hydrogen
headline; wall unverified):

| | Liquid, centred | Slush, centred | **Slush, in view** |
|---|---|---|---|
| `tab:seed_amortization` seed / annual / 10-yr | 83.25 t / 54.8% / 75.1x | 83.65 / 58.2% / 95.5x | **82.54 / 57.3% / 90.3x** |
| `growth-cost` steady $/kg, break-even | $100, 146 / 194 | $96, 135 / 174 | **$98, 138 / 180** |
| Behind the spray cup (`survivable-cost`) | | | **$118, 214 / 340** |

**The growth ledger's scenario matrix** (`make growth-ledger`; ADR 0033's
legacy pairings at their solved efficiencies, hydrogen now slush, stream in
view):

| Plate | H2 solved (0.858) | CH4 solved (0.538) | Methalox, 3S only |
|---|---|---|---|
| 0.5 | 1.72 yr | 2.03 | 5.57 |
| 0.7 | 1.53 | 1.78 | 4.08 |
| 1.0 | 1.37 | 1.56 | 3.10 |

Its hold sweep, which flies liquid on purpose, spans 1.56-1.64 yr for the
solved hydrogen chamber behind a 0.7 plate (0-5% cryostats, 0-0.5%/day).
Methalox needs no stream and is unchanged.

## Consequences

- **Slush roughly pays for visibility on hydrogen.** The chamber's cost per
  kilogram is about where ADR 0064 left it, $154 against $157, though it now
  doubles in 2.61 yr rather than 2.54.
- **Methane loses about 7% of its doubling** and gains nothing back.
- **Pulse rate matters more.** Under visibility, 2.4 Hz hydrogen gains 0.17 yr
  and 3 Hz methane 0.49 yr.
- **Every pinned test that moved** carries its centred and liquid values in a
  comment.

## Open: the two-wave split's departure is not charged

Found while scouting a split study (author, 2026-10-14). The chain re-solves
each growth wave as its own Earth-to-Jupiter-to-Earth Lambert pair from the
departure epoch, with a free Jupiter encounter time (ADR 0013). Both waves
leave on one departure burn, yet the growth wave's solution needs a different
Earth departure excess vector. On the flown chain at the 20-day split the
difference is **0.52-0.95 km/s** per cycle, and nothing charges it; the ledger
charges only each wave's later correction burn. CONTEXT.md's "divides at
Jupiter" is what the split should be, and is not what the code solves.

- **An upper bound.** Paid in methalox right after departure, it would cost
  the growth wave 13-23% of its mass.
- **The real cost.** It depends on where the split is made: a deep-space
  correction on the shared outbound leg, or a powered flyby at Jupiter after
  a shared encounter.
- **Every growth figure above carries this omission**, as every figure since
  ADR 0013 has. It is the next study's first item, because it decides the
  author's proposed earlier, faster growth wave: the mismatch is 2.6 km/s for
  a 2.8-synodic return at 67 km/s.

## Reproduction

`make departure-visibility` (only if the chain changes), then `make
survivable-cost`, `make hardware-recovery`, `make slush-hydrogen`, `make
growth-cost`, `make departure-levers`, `make departure-levers-centred`, `make
growth-ledger`. Tests: `make test-all`.
