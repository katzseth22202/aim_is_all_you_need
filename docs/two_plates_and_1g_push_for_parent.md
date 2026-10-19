# Two plates, the visible sign and the 1 g push: the cycle re-charged (for the parent)

Raised 2026-10-19 in `aim_is_all_you_need` (ADR 0073). **Written to be copied into
`Balloon-Pulse-Propulsion`.** It answers S33 items 1-3 and S35 for the solar-dive charge
pusher, and S38 items 2, 3 and 5. It takes `puffsat_impact_simulation` @ `bfea495`: the 2 Hz,
1 g push on ~100 um sheets with the 12.2 t plate (its ADR-0064 amendment), and the
departure bowl with a hole on the visible sign (its ADR-0065). Reproduce with
`make solar-dive` (~30 s).

## The design flown

- **Launch.** Each 1500 t launch unit carries both Earth plates.
- **Push** at 400 km, 2 Hz, 1 g on the starting unit (7.35 MN s a pulse), on a ~11-12 t flat
  plate with a 2 m skirt. The unit is lofted climbing at **2.43 km/s**. The push plate is
  left in the 10-day orbit at apoapsis, before the methalox turn.
- **Departure** from the 600 km periapsis on the **visible sign** (117-140 deg). Compact
  impactors go through the hole in a 13-22 t paraboloid bowl, at 1 Hz and a 0.25 g mean
  (a 32-34 minute burn). The bowl then diverts into Jupiter.
- **Sun, Jupiter, seed, valuation:** as ADR 0072. Gated rNPV at 10% a year, 15 synodic
  periods of growth, disks at $1000/kg, plates priced at half the spray cup's per-kg
  curve, the Sun's at the full curve.
- **Slug ratios:** every leg's is optimized for the dear-seed break-even.

## The answer

| closure | Earth wave at Earth | slug ratio: push / departure / Sun / delivery | growth per 3S | doubling | steady $/kg at L1 | break-even, cheap / **dear** seed |
|---|---|---|---|---|---|---|
| **Parker, 9.86 R_sun** (X 90) | 100.8 km/s | 7.9 / 13.9 / 20 / 7.0 | **3.27x** | **1.92 yr** | $83 | $119 / **$151** |
| **4 R_sun** (X 150) | 157 km/s | 8.2 / 13.7 / 20 / 7.5 | **3.85x** | **1.68 yr** | $70 | $93 / **$107** |
| 9.86 R_sun, X 60 (the paper's current closure) | 75 km/s | 7.1 / 12.7 / 20 / 5.8 | 3.81x (was 3.60) | 1.70 yr (1.77) | $78 ($69) | $101 / **$113** ($104) |
| 4 R_sun, X 90 (the paper's current closure) | 101 km/s | 6.6 / 12.1 / 20.3 / 5.4 | 5.72x (4.92) | 1.30 yr (1.43) | $60 ($55) | $69 / **$71** ($67) |

The bulged hydrogen chamber is $430 dear on the same footing, so the ranking is unchanged.

- **The bowl is what wins.** Its exhaust speed is 10.7-21.7 km/s, against the canted flat
  plate's ~5-6. At the paper's closures that more than pays for the faster lob, the
  larger apoapsis turn and the half-hour burn's loss. Growth rises from 3.60x to 3.81x and
  from 4.92x to 5.72x.
- **Cost rises a little** ($104 to $113 and $67 to $71 dear), because growth lobs now cost
  1.48 times as much.
- **The ~100 and ~150 km/s Earth waves are dearer than the paper's slower closures.** A faster
  return needs a bigger Sun boost, which means more $1000/kg disks, more carrier and a
  heavier Sun plate. The 157 km/s wave at 4 R_sun falls from 5.72x to 3.85x.
- **Optimal slug ratios.** The push wants 7-8 and the departure 12-14, below the simulation's
  20-30. That range maximized departing mass per kilogram of impactor, but the charge is
  launched mass, and a lower k buys Isp. The Sun sits at k = 20, the bottom of the simulated
  table, so its true optimum is lower.

## S33: the 1 g push and its lob

1. **The least climb is 2.43 km/s at 400 km** (`lob_rise.push_track`, thrust along the
   stream). It is the least climb that keeps the unit at or above 400 km through the push
   and ends on a periapsis above 400 km. Your scratch rows reproduce: 2.25 km/s ends at a
   265 km periapsis (yours: 268) and 2.5 km/s at 455 km (478). The push lasts ~1000 s. The
   unit is orbital after ~700 s and holds itself up from then on. The mass ratio is
   unchanged (your 1.330 against 1.325), so no gravity loss is charged.
2. **The booster lofts 1/1.48 of the priced lob's mass**: 869 t at 2.5 km/s against 1326 t at
   1.1 km/s, on the same return. **Each growth lob costs x1.48**, about $7-12 more per
   kilogram at L1.
3. **The return closes with an entry burn.** At this climb no brake at separation meets the
   1.5 km/s / ~9 g rule (your NaN). Instead the booster coasts over its ~190 km apex and
   brakes just above 60 km, from ~3.5 to 1.5 km/s, as Falcon 9's entry burn does. The
   reserve is ~2.0 km/s. Its heating is not checked.
4. **This is not "a slightly faster lob".** It is more than twice the climb, but it works
   and is priced.

**The unit sits up to ~2000 km below any single stream path**, so each sheet must be aimed at
the unit's predicted position on its own hyperbola, with its perigee above 400 km. The
steering budget for that is not priced beyond guidance mass.

## S38: the visible sign

- **Item 2 (fly it on the bowl): done.** The geometry reproduces your table exactly
  (129.9 deg / 84.0 km/s at 9.86 R_sun, X 60; 117.0 / 107.9 at 4 R_sun, X 90), and a
  periapsis occultation test (66.1 deg cone at 600 km) confirms the other sign is hidden.
  The requested closures fly at 140.5 deg / 111.2 km/s (9.86 R_sun, X 90) and 134.7 deg /
  166.8 km/s (4 R_sun, X 150).
- **Item 3 (the apoapsis turn): 0.33-0.35 km/s, not 0.24.** The visible sign's thrust axis
  lies ~42 deg further from the push wave's direction. That is 93-113 t of methalox, ~9%
  of the stack.
- **Item 5 (the plates):**
  - The push plate is 10.8-12.5 t.
  - The bowl is 13.1-22.1 t, with its 1 t collar. That is heavier than the simulation's
    10-12 t, because the dollar optimum flies k ~13 rather than 20-30, and the 167 km/s
    stream at 4 R_sun carries more energy per pulse.
  - Both together are 25.6-32.9 t, against the single 39.9-49.6 t flat plate. In dollars
    the plates were and remain ~$0.3 of each L1 kilogram.
- **Correction to ADR 0072:** every ship is departed by the departure wave. ADR 0072
  departed push-wave ships on their own faster return.

## What moves it (9.86 R_sun, X 90; dear break-even $151)

| change | growth | dear break-even |
|---|---|---|
| lob climb not charged (the simulation's assumption) | 3.27 | $139 |
| **deliveries also at 2 Hz / 1 g** (if their sheets cannot survive 4 Hz) | 3.27 | **$171** |
| push at the child's 3.26 g instead (24 MN s pulses, a heavier plate) | 3.00 | $173 |
| departure at 0.5 g | 3.37 | $144 |
| Earth plates at 25% / 100% of the spray cup | 3.27 | $148 / $157 |
| disks at $500 / $2000 per kg | 3.27 | $144 / $164 |
| lob at $50/kg | 3.27 | $229 |

The 1 g push beats the child's 3.26 g. Its plate is ~12 t against ~126 t for 24 MN s
pulses on the same 2 m, 100 g absorber, and that outweighs the dearer lob. **The largest open line is the delivery.** It still flies the child's 4 Hz
push from the 1.1 km/s lob. If the sheets cannot survive that plume (S37), deliveries must
fly 1 g too, which adds ~$20/kg.

## Suggested changes to the paper

- **`sec:vertical_lob`:** the growth lob reaches 400 km climbing at about 2.4 km/s, not
  1.1, so that a 1 g push keeps the unit above 400 km for the ~700 s until it is orbital.
  The booster then brakes on its way down rather than at separation, and lofts about two
  thirds as much (x1.48 per kilogram). Deliveries keep the 1.1 km/s lob while they fly
  the faster push.
- **`sec:charge_pusher_cycle`:**
  - Replace the canted plate with the departure bowl on the visible sign.
  - Two plates fly up in the launch unit. The ~12 t push plate is left in the parking
    orbit, and the 13-22 t bowl diverts into Jupiter after the departure.
  - The apoapsis turn is 0.33-0.35 km/s.
  - The departure is a 32-34 minute burn that loses ~0.5 km/s.
- **Growth and cost headlines:**
  - At the paper's closures: 3.81x per 3S (doubling 1.70 yr, $113 dear) at 9.86 R_sun,
    and 5.72x (1.30 yr, $71) at 4 R_sun.
  - At the ~100 / ~150 km/s waves: 3.27x (1.92 yr, $151) and 3.85x (1.68 yr, $107).
- **Register:** S33 items 1-3 answered (item 4 dropped, since 2 Hz at 1 g is adopted); S35
  answered for this cycle; S38 items 2, 3 and 5 answered.

## Not modelled

- Drag, Earth rotation and the entry burn's heating.
- The bowl's real collision at 130-140 deg. Its capture is extrapolated in cos(theta)
  beyond the 129.9 deg runs, and its speed factor recomputed out to 167 km/s.
- Aiming compact impactors through the 1 m hole, priced only as guidance mass and
  attrition.
- Development cost, as in ADR 0072.
