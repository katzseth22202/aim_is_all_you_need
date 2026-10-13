# The impact simulation's chamber answers, for the parent paper

Raised 2026-10-13 in `aim_is_all_you_need` (ADR 0055). It takes
`puffsat_impact_simulation` @ `7afdba3`, which answers this repo's
`docs/hydrocode_asks_2026-10-12.md` (B1-B6). Its handoff is that repo's
`docs/hydrocode_asks_handoff.md`, and its decisions are its ADR-0056 with the 2026-10-13
amendment. Reproduce the ledger with `make survivable-chamber` and the hydrogen rows with
`make slush-hydrogen` and `make hardware-recovery`.

**Items 1, 3 and 4 are superseded by ADR 0056** (`docs/throat_answers_for_parent.md`): the
868 / 899 kN s froze hydrogen recombination suddenly. At the finite rate they are 885 / 941,
the methane rows double in 3.36 / 3.33 yr, and A/A* = 100 no longer leads.

Status of each item for the paper:

| # | item | for the paper? |
|---|---|---|
| 1 | `tab:one_chamber`'s impulse was gross: 906 / 971 → 868 / 899 kN s | **yes, a correction** |
| 2 | The methane throat doubles to 2.66 m2; the film is kilograms a shot | **yes** |
| 3 | The methane rows of the growth and cost tables | **yes, a correction** |
| 4 | A/A* = 100 now beats 300 | **yes** |
| 5 | Burn length and wall life | **yes** |
| 6 | The gas-cushioned wall is dropped | only if the paper mentions it |
| 7 | The bulged hydrogen chamber | **yes; it changes the hydrogen framing** |

## 1. The impulse per pulse

`tab:one_chamber`'s 906 / 971 kN s per 5 kg pulse (A/A* = 100 / 300) charged neither the
1% port gate nor the H + H + M nozzle freeze. With both, the 1.33 m2 chamber gives
**861 / 892 kN s**, and the 2.66 m2 chamber of item 2 gives **868 / 899 kN s**.

- Inverted through `eq:eta_isp` with the gate, these are `eta` 0.459 / 0.483, or 756 / 783 s.
  The parent's figures were 0.488 / 0.539, or 788 / 845 s.
- The A/A* = 300 value is no longer the sphere's solved 0.538.
- The impact simulation's own `eta_net` is 0.484, so the inversion agrees.

## 2. The throat and the cycle (amends ADR 0053's text in the paper)

**The blowdown estimate holds.** The real-gas solve reaches fill pressure in 281 ms through
the 1.33 m2 throat. The paper's 278-336 ms is right at its fast end.

**The cycle did not close.**
- The throat film cannot be laid across a sonic jet, so it seals only once the flow fades.
- That traps ~3 kg of ~4,100 K residue, which joins the next charge.
- A polymer film held under its 400 °C rating must wait until ~0.3 bar. The period then
  runs 548 ms, 48 ms past 2 Hz.

**Doubling the throat to 2.66 m2 closes it.**
- Blowdown time goes as `V / A*`, so the chamber reaches fill pressure in 139 ms and 149 ms
  of the 2 Hz period is left.
- The cone is 1 m shorter, and the dry wrap passes at **46.2 t** (the 1.33 m2 chamber
  re-solved needs 52.4 t).
- The costs:
  - **2.6 kg of aramid film every shot**, about 2.4% of the charge. It leaves in the
    exhaust and is charged like the pitch.
  - **Exits of ~18 m (A/A* = 100) and ~32 m (300)**, against 13.0 / 22.5 m.
- The chamber still cannot reach 4 Hz: about 2.8-3.4 Hz at best.

**Shrinking the chamber is not the alternative.** Without the bulge, the spike beside the
plug grows 5-8x and no dry wrap survives. Stand-off is the only lever on the spike.

## 3. Methane rows (`tab:growth_ledger_doubling`, `tab:growth_ledger_ten_year`, cost tables)

Spray cup, 20-day orbit, gate charged, fixed-direction loss integrated:

| departure | doubling | annual | 10-yr | chamber t | pulses | burn | loss | payload t |
|---|---|---|---|---|---|---|---|---|
| **CH4 AR 100, 24 kg pitch** | **3.57 yr** | **21.4%** | **5.7x** | 56.0 | 3040-4630 | 25-39 min | 7.5-12.1% | 207 |
| CH4 AR 100, 3.5 kg pitch | 3.46 yr | 22.2% | 6.1x | 56.0 | 3420-5350 | 28-45 min | 8.9-14.2% | 213 |
| CH4 AR 300, 24 kg pitch | 3.86 yr | 19.7% | 4.8x | 75.6 | 3080-4880 | 26-41 min | 7.7-12.8% | 211 |
| CH4 AR 300, 3.5 kg pitch | 3.71 yr | 20.6% | 5.3x | 75.6 | 3450-5620 | 29-47 min | 9.0-15.0% | 217 |
| *previous AR 300, 24 kg (ADR 0044)* | *2.76 yr* | *28.5%* | *10.6x* | *62.9* | | | | |

**Where the 1.1 years went** (AR 300, 24 kg):
- 0.70 yr is the impulse correction alone.
- 0.15 yr is the re-solved 1.33 m2 wall and film.
- The doubled throat then gains 0.28 yr, and its doubled extension costs 0.51.

At AR 100 the throat is a net gain of 0.13 yr over the re-solved 1.33 m2 chamber.

**Cost** (Estimate book, lob climbing, 10% a year, 50% odds). Each cell is steady $/kg, then
the cheap / dear seed break-even:

| design | Estimate | Paper's prices | Pessimistic |
|---|---|---|---|
| **CH4 AR 100, 24 kg** | **$172, $363 / $1254** | $174, $393 / $1285 | $808, $1308 / $2200 |
| CH4 AR 100, 3.5 kg | $166, $343 / $1165 | $168, $371 / $1194 | $778, $1246 / $2069 |
| CH4 AR 300, 24 kg | $178, $390 / $1461 | $180, $422 / $1495 | $839, $1387 / $2460 |
| CH4 AR 300, 3.5 kg | $171, $365 / $1334 | $173, $395 / $1365 | $802, $1309 / $2280 |
| H2 sphere, wall unverified | $122, $192 / $377 | $117, $191 / $378 | $508, $702 / $887 |
| H2 bulged, 2 Hz (item 7) | $175, $351 / $925 | $167, $344 / $920 | $772, $1194 / $1770 |
| methalox | $294, $776 / >3000 | $349, $1099 / >3000 | $885, $1755 / >3000 |

- **Priced by chamber mass** against the 21.2 t sphere, the Estimate methane rows become:
  - AR 100: $203 (BE $481 / $1371);
  - AR 300: $229 (BE $589 / $1659).
- **`tab:seed_amortization`**, stripped baseline:
  - CH4 AR 100, 24 kg: 122 t seed, 1.33 ships, 21.4% a year, 5.7x in ten years,
    $59-1633 per fleet kg.
  - AR 300: 132 t, 1.43 ships, 19.7%, 4.8x, $70-1925.

## 4. A/A* = 100 now beats 300

ADR 0044 had A/A* = 300 ahead by 0.28 yr. It is now 0.29 yr *behind* (3.86 against 3.57 yr).
- The doubled throat doubles the exit area, and the extension is scaled by exit area at
  fixed areal density: 29.4 t at 300 against 9.8 t at 100.
- Without that extra mass, 300 would still lead by 0.05 yr.

**Suggested wording:** lead with A/A* = 100. Say that a 32 m extension lighter than a thin
carbon-carbon shell scaled from the RL10B-2 would bring 300 back.

## 5. Burn length and wall life

- **Burn length.** Methane burns run 25-41 min, 3,040-4,880 pulses on one chamber at 2 Hz.
  The longest cycle's burn is about 2,500 s.
- **Wall life.** The impact simulation sized the dry wrap for ~4,200 pulses, "one departure
  of a 500-600 t stack". The ledger's departing stacks are now 790-860 t. The longest
  cycles fire 4,630 (AR 100) to 4,880 (AR 300) pulses on one wall, 10-16% past that.
- **A second chamber is not the fix.**
  - Two 5 kg chambers fire 2,950-3,190 pulses each.
  - They cost doubling: 5.15 yr at AR 100 and 7.42 yr at AR 300.
  - A second 56-76 t chamber outweighs what the shorter burn saves in finite-burn loss.
- **Suggested wording:** the chamber must be rated for ~5,000 pulses. The question is with
  the impact simulation.

## 6. The gas cushion (only if the paper mentions it)

A cushioned wall does not work. The solved spike is 9-77 kPa s, uneven over a few
centimetres, so tiles on a gas foundation bend at 0.9-8 GPa. None of 390 cases lasts 4,200
pulses. Keep the bulge and the dry wrap.

## 7. The bulged hydrogen chamber

The 20 m3 hydrogen sphere's wall is unresolved, at ~2.95 GPa. The impact simulation put the
same 60 kg charge on the 2.5 kg rod into the methane chamber's r 3.0 m bulge.

**The chamber:**
- 106 m3, 4,884 K, 142 bar, 0.84 m2 throat.
- A **37.0 t dry wrap** that passes at 2.20% flung strain against 2.4%, a 1.09x margin. A
  0.25 cm grid check is still owed.
- **657 kN s per pulse** at A/A* = 300, against the sphere's 697.
- **~2-3 Hz, not 4.**

Flown at 2 Hz on liquid hydrogen:
- G_0 is 1.56 and the steady cost $175/kg (Estimate), against the sphere's 2.18 and $122.
- One chamber's long cycles reach the integrated loss table's one-hour edge (3,598 s,
  7,196 pulses). The single-chamber row is therefore bounded by the table, not an optimum,
  and fires 1.7x the wall's rated life.
- Two chambers keep each under ~4,400 pulses and double in **3.28 yr**.

**Stored and recovered** (`make slush-hydrogen`, Estimate book; growth is the geometric
mean per cycle):

| hydrogen chamber | growth | steady $/kg | BE steady |
|---|---:|---:|---:|
| sphere, slush, tanks recovered, 4 Hz (wall unverified) | 2.877 | $105 | $147 / $245 |
| sphere, slush, tanks recovered, 2 Hz | 2.597 | $116 | $176 / $319 |
| **bulged, liquid, 2 Hz** | 1.830 | $175 | $351 / $925 |
| **bulged, slush, tanks recovered, 2 Hz** | **2.236** | **$135** | $230 / $485 |

At the same 2 Hz, the bulge still loses to the sphere (2.597 → 2.236). That gap is its 6%
lower impulse and its larger chamber: a 37 t wall and 9.3 t extension, against 32 t and
2.2 t.

**This changes `orbit_turn_for_parent.md` item 9.**
- That note named 5500 K hydrogen, as slush with recycled tanks, as the performance target.
  It grew 2.88 per cycle at $105/kg.
- Those figures belong to the sphere at 4 Hz, whose wall no solve supports.
- With the wall the impact simulation can support, liquid hydrogen and methane are level:
  about $175 against $172 per kg on the Estimate book.
- **With recovered hardware** (`make hardware-recovery`, Estimate book, methane priced by
  chamber mass), the bulged hydrogen chamber keeps a modest lead:

| | none | rods only | rods + plate |
|---|---:|---:|---:|
| CH4 AR 100, 24 kg (90% tanks, 60% chamber) | $203 | $161 | $152 |
| H2 bulged, liquid, 2 Hz (90% tanks) | $175 | $147 | $126 |
| H2 sphere, 4 Hz (wall unverified) | $122 | $110 | $105 |

  Bulged hydrogen as slush with recovered tanks is $135/kg (growth 2.236).
- **Suggested wording:** with a solved wall, hydrogen leads methane by $10-25/kg once
  hardware is recovered, not by the $50-80 the sphere implied. The sphere's lead is
  conditional on a sphere wall that survives, or on a faster bulged cycle.
