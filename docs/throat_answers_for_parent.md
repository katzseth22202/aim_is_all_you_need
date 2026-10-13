# The doubled throat on the finite-rate impulse, for the parent paper (S24 items 1-4)

Raised 2026-10-13 in `aim_is_all_you_need` (ADR 0056). It answers
`Balloon-Pulse-Propulsion/docs/throat_and_turn_asks_for_aim_repo.md`, S24 items 1-4, and
flies the impact simulation's S26 impulses: `puffsat_impact_simulation` @ `df4bc6e`,
`docs/throat_kinetics_handoff.md` (second issue). **Written to be copied verbatim into the
paper repo.**

**What it replaces.** The paper (as of `e6ffae2`) still prints ADR 0044's chamber, with these
values:
- a 1.33 m2 throat;
- 906 / 971 kN s per pulse;
- a 48.2 t wall;
- 2.78 yr doubling;
- $162/kg.

This note gives every methane cell and sentence that moves, on one chamber. It supersedes
the methane rows of `docs/hydrocode_answers_for_parent.md` (items 1, 3, 4, 5). The rest of
that note still holds for the paper:
- item 2: why the throat doubled, and the film;
- item 6: the cushion;
- item 7: the bulged hydrogen chamber.

**Not answered here:** S24 item 5 (a hydrogen-sphere row) and S25 (the Earth turn).

**Reproduction.** Every number below comes from `make survivable-chamber` and
`make survivable-paper-grid`. The cost cells come from
`python -m src.survivable_chamber --paper-cost`. The figures were checked with
`make test-all`, 908 passed.

## The decisions behind the numbers

1. **The impulse is the finite-rate one,** 885 / 941 kN s (A/A* 100 / 300). Hydrogen
   recombines at its measured rate, and carbon trades atoms freely. The carbon-frozen floor
   is 810 / 849. The 868 / 899 sent with ADR 0055 froze hydrogen suddenly, which tripled
   its loss.
2. **The paper keeps A/A* = 300,** now a 31.9 m exit (S24's option (b)).
   - The three exits double within 0.1 yr of one another: 3.36 / 3.26 / 3.33 yr at
     A/A* 100 / 150 / 300.
   - The impact simulation prefers (b) on chemistry.
   - Every paper table is already built on 300.
   - One cost: priced by chamber mass, as the paper prices it, 300 is $10/kg dearer than
     option (a) ($197 against $187). Say so in a sentence (suggested below).
3. **The chamber fires at 2 Hz.** 3 Hz appears only as a conditional sensitivity (section 6).
4. **Settings are unchanged from the paper's tables:**
   - 24 kg pitch, plus the 2.6 kg throat film;
   - spray cup with its film carried;
   - 20-day orbit, 1500 t unit;
   - fixed-direction loss integrated;
   - methane chamber priced by mass against the 21.2 t sphere;
   - Estimate book, lob climbing at 1.1 km/s, 10% a year, 50% odds.

## 1. `sec:one_chamber`

### `tab:one_chamber`

| row | now | becomes |
|---|---|---|
| Volume, width at the bulge, throat | 212 m3, 7.6 m, 1.33 m2 | 212 m3, 7.6 m, **2.66 m2** |
| Wall | 48.2 t | **46.2 t** |
| Fallback wall | 68.8 t | 68.8 t |
| Nozzle extension | 4.9 / 14.7 t (13.0 / 22.5 m exit) | **9.8 / 29.4 t (18.4 / 31.9 m exit)** |
| Carried per pulse | 115-117 kg, plus 3.5-24 kg pitch | 115-117 kg, plus 3.5-24 kg pitch **and a 2.6 kg throat film** |
| Net impulse per pulse | 906 / 971 kN s | **885 / 941 kN s** |
| Effective specific impulse | 788 / 845 s | **771 / 820 s** |
| Thrust at 2 Hz | 1.81-1.94 MN | **1.77-1.88 MN** |
| Departing stack | 797-815 t | **820-855 t** |
| Pulses and burn per departure | 2900-4500, 24-37 min | **2980-4710, 25-39 min** |
| Finite-burn loss, share of the burn | 7.0-11.6% | **7.3-12.3%** |

**Caption.** The extension is still "the ledger's 2.19 t at 8.7 m scaled by exit area". Add
two sentences: "Net impulse charges the 1% gate and hydrogen's recombination at its measured
rate, with carbon free to exchange atoms. With carbon frozen it is 810 / 849 kN s
\cite{Katz_puffsat_impact_sim_2026}."

### The rate paragraph ("The chamber has to empty and refill between rods...")

This paragraph replaces the 4 Hz argument. Suggested text:

> On the \SI{1.33}{\square\meter} throat the cycle does not close even at \SI{2}{\hertz}. The
> throat film cannot be laid across a sonic jet, so it seals only once the flow fades. A
> polymer film held under its \SI{400}{\celsius} rating must wait for about \SI{0.3}{\bar},
> and the whole cycle then runs \SI{548}{\milli\second} \cite{Katz_puffsat_impact_sim_2026}.
> Blowdown time goes as $V/A^*$, so we double the throat to \SI{2.66}{\square\meter}. The
> chamber then seals in about \SI{195}{\milli\second}. Laying the films, filling at
> \SI{1}{\tonne\per\second} and locking bring the cycle to \SI{352}{\milli\second}, which
> leaves about \SI{150}{\milli\second} of the \SI{2}{\hertz} period. Three hertz would need the
> fill doubled to \SI{2}{\tonne\per\second}. Four hertz does not fit even then: its
> \SI{5}{\milli\second} margin is inside the blowdown clock's own uncertainty
> \cite{Katz_aim_is_all_you_need_2025}.

The film's cost belongs in the same place: "Each shot spends \SI{2.6}{\kilogram} of aramid
film, about 2.4\% of the charge, which leaves with the exhaust and is charged like the
pitch."

### The redundancy sentence

**Now:** "Two 2.5 kg chambers double the fleet 0.04 yr slower at an area ratio of 100 and no
slower at 300."

**Becomes:** "Two 2.5 kg chambers double the fleet \SI{0.05}{\year} slower at an area ratio of
100 and \SI{0.11}{\year} faster at 300, where the ledger flies one half-size chamber on some
cycles."

This weakens the single-chamber choice at 300. The paper's other reasons for it still stand:
- one port, one plug feed and one set of membranes;
- 21% less wall area.

### The hardware paragraph ("The chamber's hardware is about 53 to 63 t...")

| now | becomes |
|---|---|
| 53 to 63 t of hardware | **56 to 76 t** |
| departs 780 to 820 t | **820 to 855 t** |
| 2900 to 4500 pulses over 24 to 37 min | **2980 to 4710 pulses over 25 to 39 min** |
| "run 6% past the 4200 pulses its wall was checked for" | **"run 12% past"** |
| finite-burn loss 7 to 12% | unchanged |

### The area-ratio sentence ("An area ratio of 300 is worth about 5% in impulse...")

Suggested text:

> An area ratio of 300 is worth about 6\% in impulse but needs an exit \SI{31.9}{\meter}
> across, with \SI{19.6}{\tonne} more extension. In the ledger the two come out level, at
> \SI{3.33}{\year} against \SI{3.36}{\year}. Keeping the \SI{22.5}{\meter} exit, an area
> ratio of 150, is \SI{3.26}{\year} and \$10 per kilogram cheaper. The extension is a thin
> shell scaled by exit area, and how much a \SI{32}{\meter} one weighs decides the order.

### Elsewhere

In `sec:growth_ledger`, "The chamber wall (48.2 t for the methane chamber...)" becomes
**46.2 t**.

## 2. `tab:growth_ledger_doubling` (methane rows)

Spray cup and plug at k ≤ 8.52, with k ≤ 10 in parentheses. The earlier plate is at
k ≤ 10.

| departure | spray cup 0.60 | plug 0.70 | earlier 0.84 |
|---|---|---|---|
| Methane, 50% of ceiling | no growth (no growth) | **no growth (no growth)** | **17.0** |
| Methane, 70% | **4.20 (4.19)** | **3.38 (3.36)** | **2.75** |
| Methane, solved (**77%**) | **3.33 (3.32)** | **2.79 (2.77)** | **2.34** |
| Methane, 90% | **2.57 (2.56)** | **2.23 (2.22)** | **1.93** |
| Methane, 100% | **2.27 (2.26)** | **2.00 (1.99)** | **1.75** |

- **The plug no longer grows at 50% either.** It was 18.7 yr.
- **Solved is 77%, not 80%.** That is `eta` 0.515 over the 0.669 ceiling.
- **Caption:** the methane chamber is the 5 kg chamber "on its doubled 2.66 m2 throat, ...
  24 kg of pitch and a 2.6 kg throat film per pulse". Its solved row backs `eta` out of the
  impact simulation's finite-rate 941 kN s.
- **The hydrogen and methalox rows do not change.**

## 3. `tab:growth_ledger_ten_year` (methane rows)

Each cell is annual, then the 10-yr multiple.

| departure | spray: annual, 10 yr, payload t | plug: annual, 10 yr | earlier: annual, 10 yr |
|---|---|---|---|
| Methane, 50% | −4.0%, 0.42, 134 | −0.3%, 0.61 | 4%, 0.95 |
| Methane, 70% | 18%, 4.1, 205 | 23%, 6.2 | 29%, 10.0 |
| Methane, solved | 23%, 6.6, 225 | 28%, 10.1 | 35%, 16.4 |
| Methane, 90% | 31%, 13.1, 257 | 36%, 20.0 | 43%, 33.0 |
| Methane, 100% | 36%, 19.5, 279 | 42%, 29.9 | 49%, 49.6 |

## 4. The growth-ledger prose and `tab:doubling_ladder`

| passage | now | becomes |
|---|---|---|
| `tab:doubling_ladder`, "Launch-unit ledger, methane chamber" | 2.78 | **3.33** |
| "The methane chamber doubles in 2.78 yr and the hydrogen chamber in 2.03 yr, about three and four times as fast" | 2.78; three and four | **3.33; about 2.4 and 3.9 times** as fast |
| "Over ten years that is 10.5 and 28 times the mass" | 10.5 | **6.6** |
| "The earlier 0.84 plate gave 2.04 and 1.59 yr" | 2.04 | **2.34** |
| "At 70% ... the methane chamber in 3.64 yr" | 3.64 | **4.20** |
| "after ten years its fleet is 0.65 of what it launched" | 0.65 | **0.42** |
| "Hydrogen leads methane at every share ... by 17 to 37%" | 17 to 37% | **20 to 37%** |
| "The methane chamber ... with 62.9 t of hardware" | 62.9 t | **75.6 t** |

"The real plate costs the chambers about a quarter of their growth rate" needs re-checking
against the new rows: methane goes from 34.6% a year (earlier plate) to 23.2% (spray cup).

## 5. Cost tables

All methane cells are priced by chamber mass, at ×3.57 the sphere's 21.2 t. The paper's
current $162 already used that basis.

### `tab:delivery_ledger`, CH4 columns

| cost | Estimate | Pessimistic |
|---|---:|---:|
| Lob, per lofted kilogram | 99 | 186 |
| Plate and absorber, left at L1 | 27 | 182 |
| Plate spray system | 7 | 19 |
| Departure hardware | 57 | 1009 |
| Chamber sprayers | 3 | 7 |
| Fleet: PuffSats and rods | 2 | 39 |
| Argon | 1 | 4 |
| Other | 2 | 4 |
| **Total** | **197** | **1449** |

Was 162 and 1073. The H2 and methalox columns do not change. The lob line rises from 89 to
99 because a slower-growing fleet reinvests more launch units per kilogram sold.

### `tab:l1_comparison`

| row | $/kg | cents/kWh, μ = 15 | cents/kWh, μ = 20.5 |
|---|---:|---:|---:|
| PuffSat, methane chamber, estimate | **197** | **6.7** | **9.2** |
| PuffSat, methane chamber, pessimistic | **1449** | **49.6** | **67.8** |

### `tab:seed_amortization`

| departure | seed t | ships | annual | 10 yr | seed per fleet kg | built |
|---|---:|---:|---:|---:|---|---:|
| Methane, 50% | does not grow, **0.42** in ten years | | | | | |
| Methane, solved | **130** | **1.41** | **23.2%** | **6.6** | **$51-1398** | **8.0** |

### `tab:seed_return`, methane rows

| departure | steady | BE 50% | BE 25% | BE 100% | liquidation 50% | steady, Earth loop 50% |
|---|---:|---|---|---|---|---|
| Methane, 50% | -- | >3000 / >3000 | >3000 / >3000 | >3000 / >3000 | **2367** / >3000 | -- |
| Methane, solved | **197** | **472 / 1224** | **604 / 2106** | **405 / 782** | **514 / 1927** | **468 / 1031** |

The Earth-loop column is the EEJ methalox route (ADR 0039), cheap / dear seed.

### `tab:plate_designs_cost`, methane rows

| plate | doubling | 10 yr | steady $/kg | break-even |
|---|---:|---:|---:|---|
| Spray cup, 0.60 | **3.33** | **6.6** | **197** | **472 / 1224** |
| Spray cup, unshielded film | **3.44** | **6.2** | **209** | **508 / 1337** |
| Plug, 0.70 | **2.79** | **10.1** | **168** | **359 / 784** |
| Earlier plate, 0.84 | **2.34** | **16.4** | **146** | **276 / 500** |

### Cost prose

| passage | now | becomes |
|---|---|---|
| "the methane chamber delivers to L1 for $162 per kilogram" | $162 | **$197** |
| "At $500 ... leaves $338 per kilogram behind the methane chamber" | $338 | **$303** |
| "At $200 it leaves $38 behind methane" | $38 | **$3** |
| "the methane chamber by $38 under Suncatcher's" | $38 | **$3** |
| "It breaks even at $340 ... on the cheap seed and $810 on the dear one" | $340 / $810 | **$472 / $1224** |
| "The methane chamber doubles in 2.38 yr behind [the plug] and breaks even at $268 ... and $539" | 2.38; $268 / $539 | **2.79; $359 / $784** |

**What the conclusions still say:**
- The chamber still meets Starcloud's $500 bar on the cheap seed, now by $28. It meets
  Suncatcher's on neither seed.
- It still clears Suncatcher's $200 delivery bar before the seed, by $3. That is too close to
  claim. **Suggested wording:** "meets Suncatcher's bar only at the estimate's edge".

**Not re-run here (they need the cost report's other sweeps):**
- "falling to $107 if the lob costs $10 per kilogram";
- "11 [plates built] behind the methane chamber" by the harvest. The seed table's "built"
  column is 8.0, against 10.2;
- the "860 to 890 t of cargo" in `tab:delivery_ledger`'s caption.

Ask if the paper keeps these.

## 6. 3 Hz, as a sensitivity only

| exit | doubling | floor | steady $/kg (by mass) | BE steady (by mass) |
|---|---:|---:|---:|---|
| A/A* 100 | 3.02 yr | 3.94 yr | $169 | $366 / $954 |
| A/A* 150, option (a) | 2.94 | 3.85 | $167 | $359 / $908 |
| **A/A* 300** | **2.99** | 4.10 | **$174** | $391 / $964 |

**Conditions:**
- the fill is doubled to 2 t/s;
- the polymer film is still sealed at 0.3 bar;
- the 2.6 kg film is charged.

**Unpriced and unchecked:**
- the 2 t/s pump and accumulator;
- the wall's temperature at 1.5x the heat per second.

Pulses per burn are unchanged, so wall life is not relieved.

**Suggested sentence:**

> At \SI{3}{\hertz}, with the fill doubled to \SI{2}{\tonne\per\second}, the burn shortens to
> 17 to \SI{26}{\minute} and the fleet doubles in \SI{2.99}{\year} for \$174 per kilogram.

## 7. Limits to state

- **Wall heat.** Its 2-12% of impulse is not charged in any row, as before.
- **The carbon-frozen floor.** It doubles in 4.94 yr at A/A* 300. The impact simulation
  judges it unlikely but cannot rule it out. It belongs in the `tab:one_chamber` caption, or
  as one sentence.
- **Wall life.** The longest cycles fire 4,708 pulses on one wall, against ~4,200. The ask
  for a ~5,000-pulse rating stands.

## Background: the S24 answers

**Item 1, the cycle.** The cycle model is the impact simulation's own:
- the real-gas seal time on 1.33 m2, scaled as `V / A*`;
- 30 ms to lay the films;
- 108 kg filled;
- a 20 ms lock.

It reproduces its 439 / 548 / 297 / 351 ms within 1.5 ms. On 2.66 m2 the blowdown clock is
uncertain by ±22 ms.

| film sealed at | fill | period | margin at 2 / 3 / 4 Hz |
|---|---:|---:|---|
| 0.3 bar (polymer) | 1 t/s | 352 ms | +148 / −19 / −103 ms |
| 0.3 bar (polymer) | 2 t/s | 298 ms | +202 / +35 / −49 ms |
| fill pressure (~1,100 K film) | 1 t/s | 298 ms | +202 / +35 / −49 ms |
| fill pressure (~1,100 K film) | 2 t/s | 245 ms | +256 / +89 / +5.5 ms |

Options (a) and (b) share the cycle, because the exit does not touch the blowdown.

**Item 2, the impulse.** Both carbon limits are flown and labelled. No equilibrium-only row
is quoted.

**Item 3, the ledger.** Spray cup, 2 Hz:

| exit | finite rate | floor | steady $/kg (by mass) |
|---|---:|---:|---:|
| A/A* 100 | 3.36 yr | 4.66 yr | $191 |
| A/A* 150 | 3.26 | 4.54 | $187 |
| A/A* 300 | 3.33 | 4.94 | $197 |

The impact simulation's 3.36 / 3.33 and 4.66 / 4.94 yr reproduce exactly. Its handoff cited
this repo's `companion_replies_2026-10-13.md` for them. They were not there at `37cc336`; they
are produced here. That reply's "near 2.9 yr" was an interpolation over share rows flown
before the film and the doubled extension, and it is superseded.

**Item 4, the film.**
- **It is charged in every row.** The 2.6 kg aramid film is added to the pitch.
- **Its mass already carries the wider throat.** It holds `p r / (2 t)`, so its mass goes as
  `p r^3`: 2.6 kg on 2.66 m2 against 0.9 kg on 1.33 m2, both at 0.3 bar.
- **The rate does not change it,** because the seal pressure is fixed.
- **Only a fill-pressure seal would change it.** That would need a ~1,100 K film, ~4.2 kg in
  aramid-equivalent, in practice a foil. That route is not flown.

**Recovered hardware** (`make hardware-recovery`, by mass, CH4 at A/A* 100):

| recovery | CH4 | H2 bulged |
|---|---:|---:|
| none | $191 | $175 |
| rods only | $154 | $147 |
| rods + plate | $147 | $126 |

Hydrogen's lead narrows from $10-25/kg to $7-21/kg.
