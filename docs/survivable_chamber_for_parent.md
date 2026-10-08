# The survivable methane chamber, for the parent paper (S17)

Raised 2026-10-08 in `aim_is_all_you_need` (ADR 0044). It answers the parent's S17, asked in
`docs/survivable_chamber_asks_for_aim_repo.md`, which applies `puffsat_impact_simulation` @ `69d1f40`.
Every row flies behind the spray cup with its film left to the cost book, on the 20-day orbit,
lob x1.065. Reproduce with `make survivable-chamber` (~5 min): `survivable-ledger` and
`survivable-cost` are its halves. The sphere's rows (`tab:wall_pairing_doubling`,
`tab:rod_2p5kg_doubling`) and every earlier figure are unchanged.

## Three things to correct before the rows go in

1. **The ledger does charge a finite-burn loss, and has since ADR 0032.** It is integrated (not
   scaled), held in the one fixed direction a head-on chamber can point, and solved together
   with the burn. The impact sim's "+3.1-4.8%" assumes thrust along the velocity, which the
   chamber cannot fly. Charged, the loss is **7-12% of the burn** (385-843 m/s at AR 300, 24 kg).
   Steered, it would be 5.5-9.4%. The steered rows are below as a sensitivity only.
2. **The departing stack is ~800 t, not 500-600 t.** Behind the spray cup the ledger departs
   780-820 t. A departure is then **2900-4500 five-kilogram pulses over 24-37 min**, not
   2000-2650 over 16-21. The longer burn is most of why the loss is larger than the sim's.
3. **117 kg of charge and plug is `k + P` = 23.4 per rod kilogram at 75 km/s.** Holding the
   sphere's energy per kilogram (7000 K) would need 29.3. We took 117 kg as given: with the
   gate, it reproduces 906 / 971 kN s at **eta 0.488 / 0.539**, which is 788 / 845 s. The AR 300
   value equals the sphere's solved 0.538, so the chamber law matches. Please confirm the 117 kg
   is meant, because a hotter chamber per kilogram is what that number implies.

## 1. Methane chamber rows (`tab:growth_ledger_doubling`, `tab:growth_ledger_ten_year`)

| departure | doubling | annual | 10-yr | chambers | chamber t | pulses | burn | loss | payload t |
|---|---|---|---|---|---|---|---|---|---|
| CH4 AR 100, 3.5 kg pitch | 2.96 yr | 26.4% | 8.9x | 1 | 53.1 | 3370-5270 | 28-44 min | 8.7-14.0% | 229 |
| CH4 AR 100, 24 kg pitch | 3.04 yr | 25.6% | 8.3x | 1 | 53.1 | 3000-4550 | 25-38 min | 7.4-11.8% | 223 |
| CH4 AR 300, 3.5 kg pitch | 2.69 yr | 29.5% | 11.6x | 1 | 62.9 | 3260-5190 | 27-43 min | 8.3-13.7% | 247 |
| **CH4 AR 300, 24 kg pitch** | **2.76 yr** | **28.5%** | **10.6x** | 1 | 62.9 | 2900-4490 | 24-37 min | 7.1-11.6% | 240 |
| *sphere, CH4 solved (record)* | *2.28 yr* | | | | *21.2* | | | | |
| *H2 solved, wall unverified* | *2.02 yr* | | | | | | | | |
| *methalox 380 s* | *7.82 yr* | | | | | | | | |

Ranges run over the chain's 11 cycles: the two-synodic cycles have the long burns. Chamber
mass is wall plus extension, expended each cycle. Charge tanks add 12-14 t.

The new chamber costs the sphere 0.4-0.8 yr of doubling. The wall and extension are 53-63 t
against 21 t. And with less charge per kilogram of rod the burn needs more rods, so it runs
longer and loses more.

**Shares of the AR 300 chemistry ceiling** (0.669), at 24 kg pitch: 50% does not grow,
70% 3.61 yr, 80.5% (solved) 2.76 yr, 90% 2.38 yr, 100% 2.13 yr. The AR 100 ceiling is not
sized, so it has no share rows.

## 2. AR 100 against AR 300

**AR 300 wins, by 0.28 yr at both pitch edges** (2.76 against 3.04 yr at 24 kg). Its ~5%
more impulse is worth more than its 9.8 t of extra extension. The 22.5 m exit is not priced
as a packaging cost.

## 3. Redundancy

Two 2.5 kg chambers at 2 Hz each fly the same rod flow as one 5 kg chamber. Each has half the
49.8 t wall and half the exit area, so half the extension. Both versions pick their own
chamber count.

| | one 5 kg chamber | 2.5 kg chambers |
|---|---|---|
| AR 100 | 3.04 yr, 53.1 t | 3.08 yr, two flown, 54.7 t |
| AR 300 | 2.76 yr, 62.9 t | 2.76 yr; flies one 2.5 kg chamber on some cycles |

**The ledger agrees with one chamber, but only just: two cost 0.04 yr (about 1%) at AR 100 and
nothing measurable at AR 300.** The second port, plug feed and membrane set are not priced, so
the true margin is a little wider. Redundancy is nearly free in the ledger.

## 4. Lead with methane: the cost book and the seed

Steady $/kg at L1; break-even (BE) steady, cheap / dear seed; 10% a year, 50% odds.

| design | Estimate | Paper's prices | Pessimistic |
|---|---|---|---|
| **CH4 AR 300, 24 kg** | **$136, $248 / $710** | $137, $269 / $733 | $613, $921 / $1385 |
| CH4 AR 300, 3.5 kg | $132, $236 / $662 | $133, $256 / $684 | $594, $884 / $1311 |
| CH4 AR 100, 24 kg | $149, $289 / $886 | $151, $313 / $912 | $686, $1062 / $1660 |
| CH4 AR 100, 3.5 kg | $145, $275 / $831 | $146, $298 / $855 | $665, $1019 / $1576 |
| *sphere, CH4 solved (record)* | *$123, $202 / $467* | | |
| H2 solved, wall unverified | $120, $190 / $372 | $116, $189 / $373 | |
| methalox | $288, $762 / >3000 | $343, $1078 / >3000 | $870, $1723 / >3000 |

BE liquidation at AR 300, 24 kg is $283 / $1273 (Estimate). Value per seed dollar at $500 is
14.8 / 0.6.

**The chamber's price is per unit, not per tonne.** The book's $40M→$3M curve (Estimate) and
$200M flat (Pessimistic) were set for the 21 t sphere. Scaled by mass (x2.5 at AR 100, x3.0
at AR 300), the Estimate rows become **$160 ($335 / $796)** at AR 300 and $171 ($370 / $966)
at AR 100. The Pessimistic rows become $1061 and $1085, with BE steady $1613 / $2073 and
$1691 / $2286. That is the largest cost question this chamber raises. Say which basis the
paper wants.

**`tab:seed_amortization`** (stripped baseline):

| departure | seed t | ships | annual | 10-yr | seed $/fleet kg |
|---|---|---|---|---|---|
| **CH4 AR 300, 24 kg** | **123** | **1.33** | **28.5%** | **10.6x** | **$32-873** |
| CH4 AR 300, 3.5 kg | 125 | 1.35 | 29.4% | 11.5x | $29-805 |
| CH4 AR 100, 24 kg | 120 | 1.31 | 25.6% | 8.3x | $41-1125 |
| CH4 AR 100, 3.5 kg | 122 | 1.33 | 26.4% | 8.9x | $38-1049 |
| H2 solved, wall unverified | 110 | 1.20 | 41% | 28.6x | $12-325 |
| methalox | 102 | 1.11 | 9.3% | 2.29x | $147-4065 |

`tab:delivery_ledger` and `tab:l1_comparison` do not change: deliveries are plate pushes, and the
chamber never flies them. `tab:plate_designs_cost` gets the methane rows above in place of the
sphere's.

Suggested headline: **methane at AR 300, 24 kg pitch — 2.76 yr doubling, 10.6x in ten years,
$136/kg steady and $248 / $710 to break even**, with hydrogen kept as "wall unverified" (S18).

## 5. Make targets

`make survivable-chamber` prints every cell above: `survivable-ledger` for 1-3 and the
sensitivities, `survivable-cost` for 4. Module: `src/survivable_chamber.py`. Chamber:
`chamber_isp.survivable_methane`.

## The sphere's tables

Keep `tab:wall_pairing_doubling` and `tab:rod_2p5kg_doubling` as the record of the sphere,
labelled superseded. Rerunning them on the 800 t stack would only repeat the ledger rows above.

## Sensitivities (24 kg pitch)

| | AR 100 | AR 300 |
|---|---|---|
| maraging fallback wall (68.8 t) | 3.64 yr | 3.20 yr |
| steered loss (cannot be flown) | 2.90 yr | 2.66 yr |

## Not done

The hydrogen chamber (S18). The AR 100 chemistry ceiling. A per-tonne chamber price. The
plug priced as frozen methane: it is still at the foam's $5/kg, which is conservative. The
plug's tankage, about 0.7 kg a pulse, which is negligible.

## Addendum (parent S21): the rows above leave out the film

Every table above flies `SPRAY_CUP`, which carries no film. The parent's tables fly
`SPRAY_CUP_SHIELDED`, with 6 kg of film per pulse carried as launched mass (parent `353e1a1`).
The "$120 for hydrogen" above therefore does not match the parent's printed $122. The module now
flies the shielded cup, and `make survivable-chamber` reproduces the parent's hydrogen ($122,
$192 / $377), methalox ($294, $776) and seed ($149-4110, $12-329) rows exactly. The methane rows
with the film carried:

| departure | doubling | 10-yr | steady $/kg, BE (per unit) | steady $/kg, BE (by mass) |
|---|---|---|---|---|
| **CH4 AR 300, 24 kg** | **2.78 yr** | **10.5x** | $138, $251 / $723 | **$162, $340 / $810** |
| CH4 AR 300, 3.5 kg | 2.70 yr | 11.4x | $134, $239 / $673 | $156, $322 / $754 |
| CH4 AR 100, 24 kg | 3.06 yr | 8.1x | $151, $293 / $901 | $173, $376 / $983 |
| CH4 AR 100, 3.5 kg | 2.98 yr | 8.7x | $147, $279 / $844 | $168, $357 / $921 |

AR 300 still beats AR 100 by 0.28 yr. Two 2.5 kg chambers cost 0.04 yr at AR 100 and 0.001 yr
at AR 300. The parent prices the chamber by mass (decided 2026-10-08).

One correction to "`tab:delivery_ledger` ... do[es] not change" above: its rows are the cost
book's steady $/kg columns, and its "Departure hardware" line is the chamber that each reinvested
growth unit expends. It changes with the chamber, and `tab:l1_comparison` with it.
