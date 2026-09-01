# The launch ledger charged from the pad: 4 solar radii pays for its launch and 32 does not

Status: accepted, with the conduction-bracket figures and the depth crossing
superseded by ADR 0022

Date: 2026-09-01

## Context

This closes the paper's second open problem on the Jovian solar-dive cycle
(`sec:jovian_dive_open`), stated there as an unpaid bill:

> The launch ledger has not been re-scored. Every cycle in this paper is charged
> for its own ground launch and required to return a fifteenth of the mass lifted
> off the pad. That check was run on this cycle before the departure was charged
> from the lob, and charging it changes the mass that survives to departure by
> about a third. No per-pad figure is quoted here because the corrected one has
> not been computed. This is the first thing the companion repository owes the
> cycle, and it bears hardest on the shallow end, whose margin was thin before
> the correction.

ADR 0020's addendum charged the Earth departure from the ballistic lob and split
it into two pushes, and moved the headline doubling times from 0.513 to 0.724 yr
at 4 solar radii and from 1.094 to 1.816 at 32. It did not re-run the **launch
ledger** against that change. Every `kg per pad kg` figure in ADR 0019 and ADR
0020 -- the 1.88x that resolved CONTEXT.md's "scored only in the currency that
flatters it", the `k` = 5.25 ceiling, the 1.36x and 1.02x on the constrained
optima -- comes from `launch_ledger_verdict(cycle_growth_ledger(...))`, which
starts the departure at Earth escape.

The inconsistency is narrow and total. `PAYLOAD_FRACTION_AT_INTERCEPT` = 1/4 is
what a 4.09 km/s lob delivers to the intercept point, and `lob_arrival_speed()`
= 3.596 km/s is the speed it delivers it at. The launch ledger charged the first
and ignored the second, so the payload appeared at the burn point already at
11.0086 km/s and only the departure burn was charged against it.

## Decision

Score the launch ledger on the **split push**
(`split_push_launch_ledger()`), re-derive the caps it puts on the slug ratio
(`split_push_pad_ceilings()`), and scan the one knob the trade leaves free for a
point that both pays and works (`pad_return_frontier()`). All three are in
`src/solar_dive_depth_trade.py`; `make dive-depth` prints the first two and
`--pad-frontier` the third.

The correction is a substitution, not a re-derivation. Where the free-parking
reading had

    returned per pad kg  =  1/4 * departure_fraction * node_survival

the corrected one has

    returned per pad kg  =  1/4 * chain_to_departure * node_survival

with `chain_to_departure` = the overtaking push, times the methalox apoapsis
re-aim, times the canted departure leg. The free-parking reading set that
product to `departure_fraction` alone, which is to say it set the first two
factors to 1.

Search bounds are recorded in the module per CLAUDE.md's rule after ADR 0007 --
`_PAD_PEAK_BRACKET`, `_PAD_CEILING_BRACKET`, `PAD_FRONTIER_EXCESS_GRID`,
`PAD_FLOOR_DEPTH_BRACKET`.

## Consequences

### The paper's own estimate of the correction was right, and it decides the shallow cycle

Charging the lob costs **32.0 percent** of the pad return at 4 solar radii and
**31.3 percent** at 32 -- the paper's "about a third", unprompted by it.

| | 4 R_sun, `k` = 30 | 4 R_sun, `k` = 8.5 | 32 R_sun, `k` = 8.5 |
| --- | ---: | ---: | ---: |
| overtaking leg `f1` | 0.7615 | 0.8860 | 0.7911 |
| apoapsis re-aim | 0.8934 | 0.8934 | 0.8682 |
| departure leg `f2` | 0.8174 | 0.8935 | 0.7201 |
| **chain to the transfer** | **0.5561** | 0.7073 | **0.4945** |
| node survival | 1/2 | 1/2 | 1/3 |
| **returned per pad kg** | **0.0695** | 0.0884 | **0.0412** |
| **vs the committed 1/15** | **1.043x** | 1.326x | **0.618x** |
| free-parking reading was | 0.1022 (1.533x) | 0.1117 (1.675x) | 0.0600 (0.900x) |
| returned kg's value ratio | 2.80x | 2.80x | 1.46x |
| vs the rescaled floor | 2.923x (1/42.0) | 3.717x | **0.904x** (1/21.9) |
| time-normalised, kg/pad/yr | 0.0212 | 0.0270 | 0.0126 |

**The deep cycle survives the correction with 4 percent to spare. The shallow one
does not survive it at all**, and this is where the verdict changes rather than
the digits. ADR 0019's argument that the committed floor was calibrated on a
~60 km/s return, and that a faster return deserves a lower bar, is still correct
and still applies. It is simply no longer enough: the rescale is worth 1.46x at
85 km/s where it was worth 2.80x at 158, and the correction is worth more than
the rescale. The shallow cycle now fails **both** readings, where before it
failed the committed one at 0.90x and cleared the rescaled one at 1.32x.

### And it fails at every slug ratio, which is a different statement

Pad return peaks in `k` and falls away on both sides, so "fails at 8.5" would
leave open that a different loading fixes it. It does not:

| `k` | `f1` | `f2` | chain | kg/imp | doubling | kg/pad kg | vs 1/15 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9233 | 0.7591 | 0.6085 | 1.50 | 5.566 yr | 0.0507 | 0.76 |
| 4 | 0.8722 | 0.7588 | 0.5746 | 2.47 | 2.514 | 0.0479 | 0.72 |
| 6 | 0.8321 | 0.7417 | 0.5358 | 3.02 | 2.053 | 0.0447 | 0.67 |
| **8.5** | **0.7911** | **0.7201** | **0.4945** | **3.49** | **1.816** | **0.0412** | **0.62** |
| 12 | 0.7441 | 0.6930 | 0.4477 | 3.94 | 1.655 | 0.0373 | 0.56 |
| 30 | 0.5901 | 0.5954 | 0.3051 | 4.94 | 1.421 | 0.0254 | 0.38 |
| 47 | 0.5018 | 0.5348 | 0.2330 | 5.21 | 1.376 | 0.0194 | 0.29 |

The maximum is **0.761x the floor at `k` = 1.93**. Against the rescaled floor a
window does open, `k` in [1.06, 5.40], and the next section is why it does not
help.

At 4 solar radii the committed floor is cleared from `k` = 0.68 to **34.97**, so
ADR 0019's `k` = 30 clears it by 1.043x -- thin, but the correct side of the
line.

### The split push added a colder leg, and nothing had asked it to conduct

The second thing the correction changed, and it was not anticipated. The
overtaking push runs at `theta` = 0, where the vehicle's own speed subtracts
from the closing speed directly, so that leg **ends colder than the canted leg
ever gets**:

| | overtaking leg, coldest | departure leg, coldest |
| --- | ---: | ---: |
| 4 R_sun, 150 km/s climb-out | 146.96 km/s | 158.18 km/s |
| 32 R_sun, 75 km/s climb-out | **74.21 km/s** | 91.71 km/s |

`slug_ratio_ceilings()` reads the **expansion floor** off the single-push
ledger's coldest instant, which is the *departure* leg's -- the only leg that
ledger had. At the shallow cycle's true coldest instant, 74.21 km/s, the
expansion floor admits **no slug ratio at all** at the 1.5x margin ADR 0020
adopted, against [2.339, 8.947] on the warmer leg. So the shallow cycle fails
twice over, independently: its plume will not stay conducting on the first push,
and it does not pay for its launch on either.

At 4 solar radii both legs are comfortable -- [1.716, 30.397] and
[1.682, 35.840] -- and the intersection is what binds. **ADR 0019's `k` = 30
sits at 98.7 percent of that ceiling**: admissible, and with no headroom worth
quoting.

### No climb-out rescues 32 solar radii, because the two constraints want opposite things

The dive depth is fixed by the objection that motivated it, but the climb-out
excess is free, and the two caps pull opposite ways on it. Boosting less at the
node spends less of the node's propellant, so the pad return rises; but a
smaller boost lowers the Earth closing speed, and the expansion floor needs that
speed to leave the plume conducting once the jet is drawn.

Scanning `PAD_FRONTIER_EXCESS_GRID` at 32 solar radii, expansion floor enforced
on **both** legs at 1.5x, node survival derived at each point:

| climb-out | `v_b` | coldest | survival | expansion window | best `k` | kg/pad kg | vs 1/15 |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 40 | 57.93 | 47.07 | 0.6985 | **none** | -- | -- | -- |
| 60 | 72.68 | 61.82 | 0.4926 | **none** | -- | -- | -- |
| 75 | 85.07 | 74.21 | 0.3537 | **none** | -- | -- | -- |
| **85** | 93.73 | 82.87 | 0.2759 | [2.91, 5.94] | 2.91 | **0.0417** | **0.63** |
| 100 | 107.12 | 96.25 | 0.1836 | [2.20, 10.44] | 2.20 | 0.0286 | 0.43 |
| 120 | 125.49 | 114.63 | 0.1011 | [1.91, 16.85] | 2.13 | 0.0161 | 0.24 |
| 150 | 153.76 | 142.90 | 0.0380 | [1.73, 28.52] | 2.20 | 0.0063 | 0.09 |

**Every admissible point fails, and every point that would have paid is
inadmissible.** At a 40 km/s climb-out the shallow cycle returns 0.0753 per pad
kilogram, 1.13x the floor, and doubles in 1.363 yr -- better on both counts than
the reference -- but its overtaking leg closes at 47.07 km/s, far below the
**79.56 km/s** at which the expansion floor vanishes outright at
`eta_jet^2` = 0.60 and a 1.5x margin (64.96 km/s at unit margin). The gap
between "pays" and "conducts" is not a tuning problem; it is the whole width of
the grid.

That threshold is a property of the **split push**, not of the depth, so it
applies at 4 solar radii too: climb-outs of 75 km/s and below are barred there
as well. The deep cycle is unaffected only because its own reference climb-out
is 150.

The same scan at 4 solar radii clears the floor at **every** admissible point,
1.25x to 2.42x.

### How much of this rests on the number 1/15

The floor is one round number, so it is worth asking what it is actually
carrying. Three answers, and they point the same way. All rows are at the 1.5x
expansion margin and the conservative **conduction reserve**; the next section
sweeps both.

| | returned per pad kg | as a fraction |
| --- | ---: | ---: |
| the committed floor | 0.0667 | 1/15 |
| the floor restated at 85 km/s | 0.0457 | 1/21.9 |
| 4 R_sun, `k` = 30 | 0.0695 | **1/14.4** |
| 4 R_sun, `k` = 8.5 | 0.0884 | **1/11.3** |
| 32 R_sun, reference `k` = 8.5 | 0.0412 | 1/24.3 |
| 32 R_sun, best over `k` ignoring conduction | 0.0507 | 1/19.7 |
| 32 R_sun, **best admissible anywhere** | 0.0417 | **1/24.0** |

> **Superseded by ADR 0022.** The last row is a grid reading. Scored at the
> conduction threshold itself the best admissible point at 32 solar radii
> returns **0.0443, or 1/22.5**, and doubles in **2.852 yr** rather than 3.993.
> Substitute those two figures in the two paragraphs below; the argument they
> carry is unchanged, since 2.852 yr still loses to the Jupiter-only chain's
> 1.74 and 1/22.5 is still a 50 percent relaxation of the committed floor.

**The deep cycle's 4 percent is thin but it is not the whole story**, because
`k` is a free choice and the pad return rises as it falls: at `k` = 8.5 the same
cycle returns 1/11.3, comfortably clear, at the cost of doubling in 0.835 yr
rather than 0.724. The deep cycle has somewhere to go; the margin is thin only
at the growth-optimal end of a dial it owns.

**The shallow cycle has nowhere to go, and loosening the floor does not give it
anywhere worth going.** At the operating margin and the conservative reserve,
passing it at an admissible point means moving the floor from 1/15 to 1/24 -- a
60 percent relaxation of the one quantity that ties this architecture to the
reusable rocket it has to beat -- and the cycle that then passes doubles in
**3.993 yr**, losing to the Jupiter-only chain's 1.74 and to every other cycle
in the paper.

**And the second failure does not involve the floor at all.** Dropping the
**expansion floor**'s margin from 1.5 to 1.0 restores an admissible window at
the reference point -- the overtaking leg at 74.21 km/s then allows `k` in
[2.371, 8.678], so 8.5 scrapes in -- and the pad return there is still 0.618x.
At the reference climb-out, relaxing either constraint alone leaves the other
one biting. What that argument does *not* cover is relaxing the margin and the
**conduction reserve** together, because the reserve moves the admissible
climb-out rather than the pad return, and the next section works it.

### Does the verdict survive the conduction reserve bracket? At 1.5x, yes

ADR 0020 left the **conduction reserve** an explicit bracket, because the
magnetic Reynolds number at nozzle exit has never been computed, and recorded
that nothing in it depended on where the truth sat -- every reading cleared the
`eta_jet^2` = 0.60 the cycles are scored at. Charging the launch ledger from the
pad ends that, and the coupling is worth stating carefully because it is not the
obvious one.

**The reserve does not change the pad return at any slug ratio.** It is a
plume-thermodynamics quantity and the ledger is a mass one; nothing in
`split_push_launch_ledger` takes a reserve. What it changes is the *coldest
closing speed at which any slug ratio still conducts* -- and therefore the
lowest climb-out excess the cycle may be flown at. Since the pad return rises as
the node boost falls, the reserve decides whether the paying end of the dial is
reachable at all. `conduction_bracket_frontier()` sweeps it.

At 32 solar radii:

| conduction reserve | margin | plume dies below | best admissible point | kg/pad kg | vs 1/15 | doubling | |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 15000 K, all reserved (84.41) | 1.5 | 79.57 km/s | 85 km/s, `k` = 2.91 | 0.0417 | 0.625 | 3.993 yr | fails |
| 6000 K (65.85) | 1.5 | 70.28 | 75 km/s, `k` = 2.78 | 0.0529 | 0.794 | 3.086 | fails |
| frozen chemistry only (53.47) | 1.5 | 63.33 | 65 km/s, `k` = 2.92 | 0.0653 | **0.979** | 2.401 | fails |
| 15000 K, all reserved | 1.0 | 64.97 | 65 km/s, `k` = 3.30 | 0.0644 | **0.966** | 2.225 | fails |
| 6000 K | 1.0 | 57.38 | 55 km/s, `k` = 3.41 | 0.0773 | **1.159** | 1.917 | **CLEARS** |
| frozen chemistry only | 1.0 | 51.71 | 50 km/s, `k` = 2.86 | 0.0861 | **1.291** | 1.981 | **CLEARS** |

> **Superseded by ADR 0022.** Every cell here was scored off
> `PAD_FRONTIER_EXCESS_GRID`, and each cell's optimum sits within about a
> kilometre per second of its *own* conduction threshold — which moves 46.7 to
> 81.2 km/s across the six cells while the grid steps 5 to 15. So all six were
> understated: 0.665 / 0.849 / **1.009** at the 1.5x margin and 0.970 / 1.159 /
> 1.312 at unit margin. **The frozen-chemistry cell clears at the operating
> margin**, and the paragraph immediately below is wrong. What survives, and is
> ADR 0022's replacement for it: every cell that pays doubles in 1.81 to 2.07 yr
> against the Jupiter-only chain's 1.74. The rest of this section is kept as
> written.

**At the 1.5x operating margin the shallow cycle fails at every reading of the
bracket**, so the headline verdict does not rest on plume physics nobody has
computed. The hard end reaches 0.979 and that is worth quoting rather than
rounding to "fails".

**It does not survive unit margin combined with a 6000 K or looser outlet.**
There the shallow cycle pays, at 1.159x and 1.291x. This is the honest limit of
the result and it is recorded rather than buried. Two things bound what it is
worth. Unit margin means sitting exactly on the **expansion floor**, which is
the practice ADR 0020 introduced the margin to prevent -- the two reference
cycles carry 1.76 and 1.55, and a boundary is not an operating point. And the
cycles that clear are *slow*: 1.917 and 1.981 yr, against the 1.816 yr this very
ADR's reference quotes and the Jupiter-only chain's 1.74. **So at the loosest
corner of the bracket the shallow node becomes marginally admissible and is then
beaten by a cycle the paper already has.** It never becomes attractive.

At 4 solar radii the question does not arise -- every cell clears, 2.42x to
2.59x, and the doubling moves only 0.953 to 1.057 yr across the whole bracket:

| conduction reserve | margin | best admissible point | vs 1/15 | doubling |
| --- | ---: | --- | ---: | ---: |
| 15000 K, all reserved | 1.5 | 85 km/s, `k` = 2.54 | 2.421 | 0.953 yr |
| 6000 K | 1.5 | 75 km/s, `k` = 2.44 | 2.487 | 0.987 |
| frozen chemistry only | 1.5 | 70 km/s, `k` = 2.26 | 2.529 | 1.015 |
| 15000 K, all reserved | 1.0 | 70 km/s, `k` = 2.34 | 2.519 | 1.011 |
| 6000 K | 1.0 | 65 km/s, `k` = 2.15 | 2.560 | 1.044 |
| frozen chemistry only | 1.0 | 65 km/s, `k` = 1.97 | 2.587 | 1.057 |

**The deep cycle's pad verdict is robust across the entire bracket and both
margins.** That asymmetry is itself the finding: the same uncertainty that is
irrelevant at 4 solar radii is decisive at 32, because only the shallow cycle is
close enough to the floor for it to matter.

### The corrected constrained optimum, replacing ADR 0020's

Growth per impactor kilogram rises with `k` throughout, so the fastest
admissible cycle sits on whichever cap arrives first:

| climb-out | expansion window | fastest admissible `k` | doubling | its pad margin | stopped by |
| ---: | --- | ---: | ---: | ---: | --- |
| 85 | [2.54, 7.53] | 7.53 | 0.843 yr | 2.05x | expansion |
| 100 | [2.10, 11.89] | 11.89 | 0.778 yr | 1.82x | expansion |
| 120 | [1.87, 18.44] | 18.44 | 0.726 yr | 1.57x | expansion |
| **150** | **[1.72, 30.40]** | **30.40** | **0.684 yr** | **1.24x** | **expansion** |
| 187.5 | [1.63, 49.21] | 32.48 | 0.693 yr | 1.00x | **launch** |
| 200 | [1.61, 56.45] | 21.23 | 0.737 yr | 1.00x | **launch** |

**0.684 yr at 4 solar radii, `k` = 30.40, a 150 km/s climb-out**, against ADR
0020's 0.4961 yr at `k` = 57.11 and a 187.5 km/s climb-out. That row is retired:
it was found by a search whose pad constraint used the free-parking ledger, and
`k` = 57.11 is past both caps once the lob is charged. **At 32 solar radii there
is no such row at all.**

Note where the crossover sits. Below 187.5 km/s the expansion floor binds and
the pad has margin; at and above it the pad floor binds. Backing the climb-out
*out* to buy pad margin costs growth faster than it buys margin, which is why
the optimum sits at the expansion boundary rather than between the two.

### How deep the dive has to stay

On the depth dial at a fixed 75 km/s climb-out and `k` = 8.5, node survival
derived at each depth:

| R_sun | survival | cant | `f1` | `f2` | chain | kg/imp | doubling | kg/pad kg | vs 1/15 | vs rescaled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.8695 | 65.0 | 0.8017 | 0.8305 | 0.6111 | 13.98 | 0.861 | 0.1328 | 1.99 | 3.09 |
| 8 | 0.7586 | 80.6 | 0.7998 | 0.8306 | 0.5992 | 11.98 | 0.914 | 0.1136 | 1.70 | 2.61 |
| 12 | 0.6638 | 92.7 | 0.7981 | 0.8169 | 0.5809 | 9.87 | 0.992 | 0.0964 | 1.45 | 2.20 |
| 16 | 0.5824 | 102.3 | 0.7966 | 0.7985 | 0.5619 | 8.06 | 1.088 | 0.0818 | 1.23 | 1.85 |
| 24 | 0.4519 | 115.8 | 0.7938 | 0.7587 | 0.5261 | 5.41 | 1.345 | 0.0594 | 0.89 | 1.32 |
| 32 | 0.3537 | 124.8 | 0.7911 | 0.7201 | 0.4945 | 3.71 | 1.734 | 0.0437 | 0.66 | 0.96 |
| 48 | 0.2217 | 135.7 | 0.7856 | 0.6503 | 0.4407 | 1.84 | 3.725 | 0.0244 | 0.37 | 0.52 |

The committed floor is lost at **21.09 solar radii** (`pad_floor_depth()`).
Almost all of the fall is the node: the chain to the transfer moves only 0.611
to 0.441 across the whole dial while survival falls 0.870 to 0.222.

> **Superseded by ADR 0022.** This dial holds the climb-out at 75 km/s, and the
> section above shows that 75 km/s leaves the **overtaking leg** below the
> **expansion floor** at *both* depths — so every row here is inadmissible and
> 21.09 is a crossing between two operating points nothing may fly. Flying each
> depth at its own best conducting climb-out puts the crossing at **22.9 solar
> radii** (`admissible_pad_floor_depth()`), and the cycle there doubles in
> 1.75 yr. The paragraph below still holds with 22.9 in place of 21.09.

**The dial is not a free parameter down its whole length.** ADR 0020 framed the
depth as a dial where every step inward is bought with thermal engineering and
paid back in growth. That still holds between 4 and 21 solar radii. Past 21 the
cycle stops paying for the rocket that launched it, and the thing being traded
away is no longer growth rate but whether the architecture is worth flying at
all.

### The paper's own dive carried the identical defect, and the ranking survives it

The comparison row was scored the same way, so neither figure could be quoted
until both were re-derived. `paper_resonant_dive_split_push()` charges it from
the same lob. It has far more of the burn to pay for -- its single impulse asks
for **37.53 km/s** of Earth-relative excess, because it must cancel most of
Earth's orbital motion in one go, so it flies **35.51 km/s** from the lob
against the Jovian cycles' 11.6 and 13.1:

| | free parking | charged from the pad |
| --- | ---: | ---: |
| chain to the transfer | 1.0 x 0.2955 | 0.7615 x 0.9609 x 0.2955 = **0.2162** |
| growth per cycle | 7.61 | **5.16** |
| doubling | 0.3048 yr | **0.3770 yr** |
| returned per pad kg | 0.0446 (0.669x) | **0.0324 (0.486x)** |
| vs the rescaled 1/42 floor | 1.87x | **1.36x** |
| time-normalised | 0.0499 | **0.0363** kg/pad/yr |

Its cant is a favourable 28.98 degrees, so the split costs it less
proportionally than either Jovian cycle -- 26.8 percent against 32.0 and 31.3.
**Nothing reverses.** The paper's dive still grows faster (0.377 yr against
0.724), still fails the committed floor while the deep Jovian cycle clears it,
and is still ahead time-normalised. That was ADR 0019's verdict on the launch
ledger and it survives the correction intact. **The ranking is unchanged for the
sixth time.**

## What this does not settle

- **The 1/15 floor's own calibration is still the weakest link in the chain,
  and it now decides something.** While every cycle cleared it, arguing about
  its provenance was academic. It is now the reason a depth is ruled out, and
  it is a single round number set against "roughly 150 t to LEO from a ~5000 t
  liftoff", with a value rescale that this ADR applies as a check rather than
  as the number. The sensitivity is worked above: 1/24 would pass the shallow
  cycle at an admissible point and nothing here says 1/15 is right to within
  that, but the cycle it would pass doubles in 3.993 yr. So the *verdict* does
  not rest on the floor's exact value; the deep cycle's 4 percent at `k` = 30
  does, and that is why the same cycle's 1/11.3 at `k` = 8.5 is worth having on
  the record beside it.
- **The expansion floor's 1.5x margin is doing part of the work.** At unit
  margin the overtaking leg at 74.21 km/s admits `k` in [2.371, 8.678], so the
  reference `k` = 8.5 scrapes in and the "no admissible point" verdict at 32
  solar radii softens to "one with no thermal headroom". The margin is ADR
  0020's, chosen because the two reference cycles carried 1.76 and 1.55, and
  the pad failure at 0.62x does not depend on it either way.
- **The conduction reserve is now load-bearing at the shallow node, and it is
  still a bracket.** Worked above rather than asserted. An earlier draft of this
  ADR claimed the hard end "would not restore a paying one, since the pad ledger
  fails there at every `k` regardless" -- true of the reference 75 km/s
  climb-out and false of the frontier, because the reserve moves which
  climb-outs are admissible and the pad return is won at low climb-out. The
  correct statement is narrower: the failure is robust across the whole bracket
  at the 1.5x margin, and not robust to unit margin plus a 6000 K or looser
  outlet, where the shallow cycle pays at 1.159x while doubling in 1.917 yr.
  What closes it is the exit magnetic Reynolds number, still uncomputed.
  **ADR 0022 narrows this further**: scored at each cell's own threshold rather
  than off the climb-out grid, the hard end of the bracket clears at the 1.5x
  margin too, by 0.9 percent. The failure is not robust across the bracket at
  either margin; what is robust is that every corner which pays doubles slower
  than the Jupiter-only chain.
- **The parking-orbit period is held at 5 days throughout.** ADR 0020 showed the
  growth saturates there, but the pad ledger was not swept against it. A longer
  coast makes the re-aim cheaper and would move `chain_to_departure` up by at
  most the re-aim's own 11-13 percent, which does not reach the shallow cycle's
  38 percent shortfall.
- **The nozzle mass is still uncharged**, and the opposing impactors consumed at
  the node (1.34 percent of the vehicle at 4 solar radii, 2.15 at 32) are still
  outside both ledgers. Both make the pad figures upper bounds, slightly worse
  at the shallow node -- which is the direction that hurts the row already
  failing.
- **`constrained_growth_optimum()` and `slug_ratio_ceilings()` still read the
  free-parking ledger.** They are kept as the superseded reading rather than
  rewritten, because ADR 0019 and ADR 0020 quote their numbers and the size of
  the correction should stay checkable. `make dive-depth` labels them as
  superseded in its output. Anything quoting a per-pad figure should use
  `split_push_launch_ledger()`.
- **Circular and coplanar**, as ADR 0019 and ADR 0020.
