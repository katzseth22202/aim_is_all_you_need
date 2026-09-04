# The 20-day split's tail is worth paying, and low thrust cannot pay it

Status: accepted

Date: 2026-09-02

Builds on ADR 0013 (the two-wave split and its chain), ADR 0011 (the adaptive
2S/3S cadence and the DSM proxy) and ADR 0009 (the two-wave ledger). Amends
nothing: every number in ADR 0013 stands, and `analyze_two_wave_growth`'s
20-day row is reproduced here as the check that this module flies the same
chain.

## Context

ADR 0013 priced the split gap at 10, 20, 30 and 60 days and reported a **chain
mean** growth-wave burn for each — 0.4641 km/s at 20 days, the gap the rest of
the repo carries as `PUFFSAT_CYCLE_ORBIT_PERIOD`. A mean is the wrong summary
for that quantity and it hid three separate questions:

1. **What is the distribution?** It is not a spread around 0.46 km/s. Eight of
   eleven cycles buy their early arrival for under 2 m/s and three pay
   1.4–2.1 km/s. Nothing sits in between.
2. **Should the cadence dodge the expensive ones?** `two_wave_growth`'s policy
   selects a two-synodic return whenever the *nozzle* wave's maneuver proxy
   stays under 50 m/s, and never looks at what the growth wave will then pay.
   That looked like an oversight.
3. **Is `dsm_only` leaving growth on the table?** `_cheapest_return` asks
   `real_orbit_resonance` for the deep-space-burn architecture only, never the
   powered flyby, even though the module implements both.

A fourth question arrived from the design side: the corrections are priced as
**methalox**, and the architecture already carries an argon SEP leg
(`ARGON_SEP_ISP`, ADR's apoapsis-raise Leg 2). Argon's exhaust speed is 5.26×
methalox's. If the corrections could be bought there instead, the whole tail
would become affordable — provided low thrust can actually deliver them.

## Decision

**Keep the adaptive cadence and pay the three expensive splits impulsively as
methalox.** No policy change and no propellant change survives its own costs:
the split-aware cadence loses 9% of the clock, the always-two-synodic cadence
wins only while its 7.72 MW array is free, and argon's remaining 3.7% is gated
behind a 1.61 MW array the repo cannot assume.
`src/sep_split_correction.py` is committed as the reproducible harness
(`make sep-split`; not part of `make all`). It flies three cadence policies
against the same ephemeris, prices each on ADR 0013's ledger in both
propellants, and — separately — integrates how much SEP impulse the array
could actually deliver along each leg.

Search box and inputs, recorded per the ADR 0007 lesson: start 2026-08-11,
30-year horizon, 50 m/s fallback threshold, Astropy built-in analytical
ephemeris, zero-revolution Lambert arcs on both legs, 4,000 km perijove
altitude floor, fixed mean two-synodic lattice of 797.734430 d, **20-day split
gap** with the apoapsis reversal resized to match, priced at `e = 0.6`,
`f = 0.8`. Maneuver architectures searched per wave: `perijove_only`,
`dsm_only`, `hybrid_50mps`. Methalox `v_e` 3.7265 km/s (380 s); argon `v_e`
19.6133 km/s (`ARGON_SEP_ISP` = 2000 s) at thruster efficiency 0.5. Array
100 kW at 1 AU falling as `1/r^2`; SEP system specific mass swept 10/15/20
kg/kW covering array, PPU, thrusters and gimbals; argon tankage 0.15 and
methalox tankage 0.08 kg per kg of propellant. The returning wave's DSM is
confined to `r > 2 AU`, leaving the inner leg for deployment into a spread
PuffSat stream; the 20-day separation burn is unconstrained in place. Legs are
propagated as unperturbed two-body conics, 20,001 samples per leg. Absolute
power and array masses are quoted at a 500 t reference wave.

The three cadence policies:

- **`adaptive`** — `two_wave_growth`'s own, and the one this ADR keeps.
- **`split_aware`** — also gates the 2S choice on the growth wave's cost.
- **`always_2s`** — never falls back, which only becomes thinkable if the
  correction is cheap enough that the fallback's whole purpose disappears.

## Consequences

- **The split's cost is bimodal, and the expensive cycles are bend-limited.**
  At 20 days the flown chain is ADR 0013's — 11 cycles, 7×2S + 4×3S,
  28.3930 yr, mean growth burn 0.46406 km/s. Eight cycles pay under 2 m/s;
  three pay 1584.955, 2089.377 and 1425.619 m/s, departing 2030-02-18,
  2036-09-07 and 2042-02-22. Each of those three has its growth wave pinned at
  **1.05595 R_J, exactly the 4,000 km floor**, and still cannot turn far
  enough. The residue is an *angle*, not a speed mismatch, which is why
  `perijove_only` returns no solution for those windows at all. No perijove
  change buys them; the flyby is already scraping Jupiter.

- **The worst total correction is 2107.86 m/s**, on the cycle departing
  2036-09-07: 18.479 m/s of nozzle-wave DSM plus 2089.377 m/s of separation.
  Chain mean 465.97 m/s.

- **The powered flyby is worth 0.06%, and that retires the suspicion.**
  Enabling `perijove_only` and `hybrid_50mps` cuts the median split burn 2.6×
  (0.717 → 0.275 m/s) and moves the chain from ×76,823 to ×76,866, doubling
  1.7495 → 1.7494 yr. `two_wave_growth` asking only for `dsm_only` costs
  nothing measurable, because the entire ledger effect lives in the three
  cycles the powered flyby cannot reach.

- **The split-aware cadence removes the tail and loses 9.00% of the clock.**
  Gating 2S on both waves gives 9 cycles, 2×2S + 7×3S, 27.3010 yr, and a
  maximum correction of **1.69 m/s** across the whole chain. It doubles in
  1.9069 yr against the adaptive cadence's 1.7494. Two mechanisms, both
  against 3S: each fallback spends **1.092 yr** of clock (2.184 → 3.276) to
  recover a growth wave worth 0.571–0.682, and 3S returns arrive **colder** —
  `v_b` 52.6–55.4 km/s against 60.1–63.3 — so the departure nozzle weakens too.
  A cycle is worth ~3.1× on this chain; a 30–43% mass haircut is much cheaper
  than losing a year of it.

- **The propellant flips the cadence verdict, which is the finding.** Priced
  on the corrections alone:

  | cadence | cycles | span (yr) | methalox | argon |
  | --- | --- | ---: | ---: | ---: |
  | adaptive | 11 (7×2S+4×3S) | 28.3930 | **1.7494** | 1.6288 |
  | split_aware | 9 (2×2S+7×3S) | 27.3010 | 1.9069 | 1.9067 |
  | always_2s | 13 (13×2S) | 28.3930 | 2.6856 | **1.4683** |

  Doubling in years, corrections charged as propellant only — see the array
  table below, which changes the `always_2s` column completely. On methalox
  `always_2s` is the *worst* policy by a wide margin — its 13 cycles carry corrections up to 6135.3 m/s, and at
  `exp(-6.135/3.7265)` almost nothing survives. On argon it is the *best*.
  The 50 m/s fallback threshold exists only because methalox cannot afford a
  kilometre-per-second correction; change the currency and the fallback becomes
  pure cadence loss. Argon does nothing at all for `split_aware`, whose burns
  are already sub-m/s.

- **The array is scale-invariant, so sizing the wave up cannot make it cheap.**
  To buy a fixed delta-v in a fixed time, power scales linearly with wave mass
  and array mass scales with power — so the array is the same *fraction* of a
  5 t wave and a 500 t one. An earlier framing of this as a "break-even wave
  mass", with the array held at 100 kW regardless of what it had to push, is
  wrong and is not in the module. What each cadence demands:

  | cadence | worst correction | kW/tonne | `a` at 1 AU | at 500 t | array @15 kg/kW |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | split_aware | 1.69 m/s | 0.0077 | 3.90e-7 | 3.8 kW | 0.057 t (0.01%) |
  | adaptive | 2107.9 m/s | 3.22 | 1.64e-4 | **1.61 MW** | 24.2 t (4.83%) |
  | always_2s | 6135.3 m/s | 15.44 | 7.87e-4 | **7.72 MW** | 115.8 t (23.2%) |

  `always_2s` is dearest because its binding burn is the *DSM*, and the DSM is
  the one confined outside 2 AU, where the array is weakest.

- **Charging the array eats more than half the argon gain, and kills
  `always_2s` outright.** ADR 0013's ledger charges propellant only — the
  rocket equation has nowhere to put hardware — so the argon rows above are
  optimistic. Carrying the array as mass on the growth wave (doubling, years):

  | array | adaptive | split_aware | always_2s |
  | --- | ---: | ---: | ---: |
  | propellant only | 1.6288 | 1.9067 | **1.4683** |
  | 10 kg/kW | 1.6648 | 1.9068 | 1.6660 |
  | 15 kg/kW | **1.6839** | 1.9069 | 1.8075 |
  | 20 kg/kW | 1.7038 | 1.9069 | 1.9975 |
  | (methalox reference) | 1.7494 | 1.9069 | 2.6856 |

  For the adaptive cadence the honest gain against methalox is **3.7% at
  15 kg/kW**, not 6.9%. For `always_2s` the gain vanishes completely: its
  7.72 MW array is 15–31% of the wave, so it ties the adaptive cadence at
  10 kg/kW (1.6660 vs 1.6648) and is *worse than plain methalox adaptive* at 20
  (1.9975 vs 1.7494). **The always-two-synodic result exists only while the
  array is free.** The *nozzle* wave's array is still uncharged everywhere
  above, so even these flatter argon.

- **At the design's stated characteristic acceleration the km/s corrections
  are unreachable.** This is the table the decision rests on, printed by
  `make sep-split` as *what the SEP system must deliver, against what was
  stated*. Required and available are both per tonne of wave and therefore
  independent of wave size; the separation burn draws on the whole trajectory,
  the returning wave's DSM only on the stretch outside 2 AU.

  | cadence | worst cycle | burn | required | available at 2e-5 m/s² | short by |
  | --- | --- | --- | ---: | ---: | ---: |
  | `split_aware` | 2033-05-29 | separation | 0.6 m/s | 242.1 m/s | — |
  | | | DSM (>2 AU) | 1.1 m/s | 58.1 m/s | — (51× margin) |
  | `adaptive` | 2036-09-07 | separation | **2089.4 m/s** | 254.5 m/s | **8.2×** |
  | | | DSM (>2 AU) | 18.5 m/s | 56.1 m/s | — |
  | `always_2s` | 2050-11-18 | separation | 4112.9 m/s | 249.0 m/s | 16.5× |
  | | | DSM (>2 AU) | **2022.5 m/s** | 51.4 m/s | **39.4×** |

  So SEP buys the 50 m/s-class corrections with room to spare and cannot touch
  the tail. `always_2s` is worst because its binding burn is the DSM, and the
  DSM is the one confined outside 2 AU where the array is weakest.

- **What it would take instead.** Scaling to the acceleration each cadence
  actually demands, with power and thrust quoted at the 500 t reference wave
  and the array at 15 kg/kW:

  | cadence | kW/tonne | `a` at 1 AU | `a` at 5.2 AU | power | thrust at 1 AU | at 5.2 AU | array | × stated |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `split_aware` | 0.008 | 3.90e-7 | 1.44e-8 | 3.8 kW | 0.2 N | 0.01 N | 0.06 t | 0.02× |
  | `adaptive` | 3.220 | 1.64e-4 | 6.07e-6 | **1.610 MW** | 82.1 N | 3.04 N | 24.2 t | **8.2×** |
  | `always_2s` | 15.438 | 7.87e-4 | 2.91e-5 | **7.719 MW** | 393.6 N | 14.55 N | 115.8 t | **39.4×** |

  For scale, a 12.5 kW-class Hall thruster produces about 0.64 N, so the
  adaptive cadence's 1.61 MW is roughly 129 of them and `always_2s`'s 7.72 MW
  about 618.

- **The stated power and acceleration figures do not agree with each other.**
  At Isp 2000 s and η = 0.5, 100 kW on a 500 t wave gives **1.02e-5 m/s²** at
  1 AU, not 2e-5; the stated acceleration implies a **196 kW** array at that
  wave mass, or a 250 t wave at 100 kW. Separately, 2e-5 at 1 AU falls to
  7.4e-7 at 5.2 AU under `1/r^2`, not the stated 1e-6 — that pair implies a
  heliocentric distance of 4.47 AU. Both are pinned in
  `test_the_stated_operating_point_needs_twice_the_stated_power`.

- **The blocker is the power level, not the array mass.** 4.83% of a 500 t wave
  is an affordable array; a 1.61 MW solar array at 1 AU, built and destroyed
  every 2.18 years, is not something this repo can assume. That is the gate the
  argon result sits behind, and it is a manufacturing claim rather than a
  trajectory one.

- **Inherited and new caveats.** The DSM remains ADR 0011's exact velocity
  match at the Jupiter patched-conic seam, not a finite-location interplanetary
  optimum. The impulse integral is a **budget of thrust-seconds, not a
  low-thrust trajectory solution**: whether a burn distributed over 365+ days
  achieves the same endpoint change as the impulsive proxy it is compared
  against is unsolved, and it is the single largest open item here — a real
  low-thrust optimisation could move the required delta-v in either direction.
  The ephemeris is Astropy's analytical model. Thruster efficiency 0.5 and
  specific mass 10–20 kg/kW are asserted engineering ranges, not figures
  calibrated against a flight system, and every power and array figure scales
  linearly in them. The apoapsis reversal is deliberately left on methalox
  throughout: it is a 180-degree turn at the apoapsis of a 20-day parking
  orbit, and low thrust does not perform that maneuver.

## What the paper should say

The paper does not currently mention that solar-electric propulsion was
considered for the correction burns, and it should — the reason it fails is
specific to this architecture rather than to SEP, and a reader who knows the
propellant tables will otherwise ask. Draft text for the section that
introduces the two-wave split's correction burns:

> We considered buying the correction burns with argon solar-electric
> propulsion rather than methalox. Its exhaust speed is 5.26 times higher, and
> on the propellant ledger alone it would cut the worst cycle's 2.108 km/s
> correction from 43% of the wave to 10%.
>
> It does not work here, for two reasons that compound. The thrust
> acceleration a low-thrust stage can develop is `a = 2 eta P / (m v_e)`, and
> both of its unfavourable terms are large in this architecture: the wave is
> massive — a 500 t payload, because the magnetic nozzle has a floor mass and a
> smaller wave would be mostly dry structure — and the array spends most of
> each leg beyond 2 AU, where `1/r^2` has taken all but a few percent of its
> power. Integrating the available impulse along the real Lambert legs, a
> characteristic acceleration of 2e-5 m/s^2 at 1 AU delivers about 255 m/s over
> an entire Earth-Jupiter-Earth trajectory, and about 56 m/s in the stretch
> outside 2 AU that the returning wave must finish correcting within, so as to
> leave the inner leg for deployment into a spread stream. The corrections that
> matter are 1.4 to 2.1 km/s: short by roughly a factor of eight.
>
> | | required | available at 2e-5 m/s^2 | short by |
> | --- | ---: | ---: | ---: |
> | separation burn, worst cycle | 2089.4 m/s | 254.5 m/s | 8.2x |
> | returning wave's DSM, worst cycle | 18.5 m/s | 56.1 m/s | -- |
> | if the cadence never falls back to 3S | 2022.5 m/s (DSM) | 51.4 m/s | 39.4x |
>
> Closing that gap needs a characteristic acceleration of 1.64e-4 m/s^2
> at 1 AU -- 3.22 kW per tonne of wave, hence about 1.6 MW of array for a 500 t
> wave, delivering 82 N near Earth and 3.0 N at Jupiter: on the order of 130
> flight-class 12.5 kW Hall thrusters. The array mass is not the obstacle — at 15 kg/kW that is 24 t, under 5% of the
> wave, and the fraction is independent of wave size because power and mass
> scale together. The obstacle is building a 1.6 MW array and destroying it
> with the wave every cycle. Charged for that array, the remaining advantage
> over methalox is 3.7% in doubling time. We therefore fly the impulsive
> methalox correction, and the adaptive two-or-three-synodic cadence of
> Section [ref], which absorbs the expensive cycles rather than avoiding them.

The paper lives in the separate `Balloon-Pulse-Propulsion` repository (see
CLAUDE.md); this ADR records the result and the text, and landing it there is a
follow-up.

## Open items

1. **The low-thrust trajectory is not solved.** Everything above compares a
   thrust-seconds budget against an impulsive proxy. A genuine finite-thrust
   optimisation is the one piece of work that could overturn the verdict, in
   either direction, and it is not attempted here.
2. **The nozzle wave's array is uncharged** in every priced chain, so the argon
   columns remain optimistic even after the growth wave's array is charged.
3. **The Earth-departure spread between the two waves is uncharged.** Both
   waves leave the same instant but need periapsis speeds 224-559 m/s apart
   (mean 369; ADR 0013 carries one `departure_burn` for both). It is not
   chemical delta-v — the departure burn is the head-on nozzle, so it is extra
   slug on half the batch — but at a 20-day gap it is about twice its size at
   10 days and no longer obviously negligible.
4. **A reusable tug** that carried the array, performed the correction,
   released the wave and was recovered would change the accounting entirely,
   since the array would then be amortised instead of destroyed. It is not
   modelled, and recovering it is its own trajectory problem.

## Addendum, 2026-09-04: the 10-day column, and a label that read the wrong field

Everything above flies the **20-day** split gap, the one the rest of this
repository carries as `PUFFSAT_CYCLE_ORBIT_PERIOD`. **The paper flies 10 days**,
and `sec:split_tail`'s figures are the 10-day ones, so the column above is not
the column the paper quotes. This addendum records the 10-day run beside it and
gives the caption a target to name: **`make sep-split-10d`**, which is exactly
`python -m src.sep_split_correction --split-days 10`. It answers the paper
repository's deferred item **S4**, and the paper-facing write-up is
`docs/paper_corrections_split_tail_2026-09-04.md`.

**Nothing above changes.** The 10-day gap is the same architecture flown with a
smaller head start, and every conclusion survives it.

### The same three cycles, at roughly half the price

| departure | 20-day separation burn | 10-day separation burn | ratio |
| --- | ---: | ---: | ---: |
| 2030-02-18 | 1584.955 | 556.029 | 0.351 |
| 2036-09-07 | 2089.377 | 1011.273 | 0.484 |
| 2042-02-22 | 1425.619 | 398.470 | 0.279 |

Burns in m/s. The tail is the same three departures at both gaps, and the eight
cheap cycles fall from under 2 m/s to under 0.3. The chain mean total
correction drops **465.97 → 181.03 m/s**, and the mean separation burn alone
464.06 → **179.06 m/s** — the 0.179 km/s `sec:jupiter_only_growth` quotes.
Halving the head start roughly halves the bill because what the
burn is buying is a **bend deficit** — 3.34–4.63° short at 20 days,
0.97–2.33° short at 10 — and the deficit scales with the head start (ADR 0028).

### The verdict is unchanged, and the hardware is the same hardware

The adaptive cadence's worst total correction is **1029.752 m/s** (2036-09-07:
18.479 nozzle-wave DSM + 1011.273 separation), against 2107.86 at 20 days. It
still cannot be bought with argon at the stated operating point: the cadence
demands **1.573 kW/t**, i.e. `8.020e-5 m/s²` at 1 AU, which is **4.01× the
stated `2.0e-5`** — where the 20-day gap is over by 8.2×. Sized at the
reference wave that is **0.787 MW**, 40.10 N at 1 AU and 1.483 N at Jupiter
(96% of the thrust taken by the inverse square), about 63 reference Hall
thrusters, and a 2.4% array fraction at 15 kg/kW.

Being over by 4× rather than 8× does not change the answer, and the doubling
ledger says why paying it would be worth it if it could be paid:

| cadence | cycles | 2S + 3S | methalox | argon | argon + array (15 kg/kW) |
| --- | ---: | --- | ---: | ---: | ---: |
| `adaptive` | 11 | 7 + 4 | 1.737 | 1.690 | 1.719 |
| `split_aware` | 10 | 4 + 6 | 1.864 | 1.863 | 1.863 |
| `always_2s` | 13 | 13 + 0 | 2.270 | 1.501 | 1.859 |

Doubling times in years, over a 28.393 yr horizon. Argon is worth **1.1% net of
its own array** on the adaptive cadence, and `adaptive` beats `split_aware` by
**7.3% of the clock** — the same shape as the 20-day table. `always_2s` still
needs 5037.0 m/s on its worst cycle and lands a fifth of the wave (0.1995) on
methalox.

### `bend_limited` was reading the architecture's name

The property tested `growth.case == "dsm_only"`, which is the *name* of the
winning maneuver architecture rather than what that architecture spends. Where
two architectures tie it reads the wrong answer: on **2030-02-18 at the 10-day
gap** `hybrid_50mps` returns the identical trajectory to `dsm_only` and beats
it by **1.437e-6 m/s** — floating-point noise — so it took the label while its
own perijove burn is exactly zero. The paper's "so on two of the three no
change of flyby helps at all" is a faithful report of that flag.

Checked directly on all six cycle-and-gap combinations: `perijove_only` admits
**no solution at all** for any of these windows, and every winning solution —
`dsm_only` or `hybrid_50mps` — spends **exactly 0.0** at the flyby. So it is
all three at both gaps, and the paper's sentence wants "none of the three".

`bend_limited` now tests `growth.perijove_burn <= NO_FLYBY_BURN_KM_S`, pinned
by `test_the_ten_day_gap_is_bend_limited_on_all_three_too`. **No number in this
ADR moves**: at the 20-day gap all three expensive cycles already won on
`dsm_only`, so the old and new tests agree there. The 10-day
`always_2s` cycle departing 2037-10-11 is the other cycle whose label the fix
corrects.

### Reproducing

```bash
make sep-split-10d                                   # this addendum's tables
make sep-split                                       # the 20-day column above
pytest tests/test_sep_split_correction.py -s
```

Same inputs as the parent ADR — start 2026-08-11, 30-year horizon, 50 m/s
fallback threshold, fixed mean two-synodic lattice of 797.734430 d, stated
operating point `2.0e-5 m/s²` at 1 AU — with `--split-days 10` in place of the
default 20.
