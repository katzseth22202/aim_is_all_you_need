# ADR 0073: Two plates, the visible sign and the 1 g push

Date: 2026-10-19

Status: Accepted (author, 2026-10-19). Supersedes ADR 0072's canted plate, its single
flat plate, its 4 Hz push and its free departure pulse. ADR 0072's valuation, gates,
growth phase, Sun stage and seed are unchanged. Takes `puffsat_impact_simulation` @
`bfea495` (its ADR-0064 amendment and ADR-0065). Answers the parent's S33 items 1-3, S35
and S38 items 2, 3 and 5 for this cycle. Parent note:
`docs/two_plates_and_1g_push_for_parent.md`.

## Context

ADR 0072 departed the ship on the flat canted plate at 76-101 degrees. The parent's S38
found that this sign's stream passes through Earth. The simulation then proposed (its
Result 14, ADR-0065) a second plate. On the visible sign, compact impactors thread a hole
in a paraboloid bowl and stop in a charge at its focus, at 1 Hz and a 0.25 g mean. The
time between pulses goes to aiming through the hole. Separately (its Result 13, ADR-0064
amendment), the growth push moved to 2 Hz at 1 g, so that ~100 um Kapton sheets survive
the earlier pulses' gas. That push uses a 12.2 t plate sized by its absorber.

The author's design (2026-10-19): both plates fly up in the launch unit, and the push plate
is left behind before the departure burn. The slug ratios of every leg are optimized for
dollars, and growth is reported at Parker's 9.86 R_sun with a ~100 km/s Earth wave and at
4 R_sun with a ~150 km/s one.

## Decision

1. **The push plate** is the absorber bound `I^2/(2 s M a)` at the 7.35 MN s pulse, 100 g and
   2 m, with `M` the unit after the push: 10.8-12.5 t. Its 9.5 m skirted plate keeps 95.5% of
   the overtake law's impulse (the simulation's 94-97%). It is left at apoapsis before the
   methalox burn.
2. **The departure bowl** is the paraboloid's survival floor (30 kg/m^2 over 533 m^2, rated
   for 1 kg at 400 km/s, scaled as pulse energy^(2/3)), checked for the g cap, plus a 1 t
   hole collar: 13-22 t. It diverts into Jupiter after the burn, as the flat plate did.
3. **The bowl's law is vendored** (`data/departure_hole_law.csv`): the simulation's
   `hole_ve` factors exactly into a capture (angle, k) times a speed factor (k, w), so it can
   be evaluated at this repository's routes. It reproduces the simulation's figures to
   within 0.5%.
4. **The visible sign** (`visible_departure`): of the two departure-hyperbola signs, the one
   whose upstream direction lies outside the 66.1 degree occultation cone at the 600 km
   periapsis. It reproduces S38 exactly: 129.9 deg and 84.0 km/s at 9.86 R_sun, X 60 (117.0 and
   107.9 at 4 R_sun, X 90), and the other sign is hidden. **Every ship is departed by the
   departure wave's stream.** ADR 0072 departed push-wave ships on their own faster return.
5. **The 1 g push needs a 2.43 km/s lob** (`push_climb`, on `lob_rise.push_track`). At 1 g the
   unit takes ~700 s to become orbital, so the lob must hold it above 400 km until then, and
   the push must end on a periapsis above 400 km (ADR 0062's growth-wave floor). This
   reproduces the parent's S33 (2.25 km/s fails, 2.5 holds). Above ~2.4 km/s the booster's
   brake at separation cannot close, so it brakes on the way down instead
   (`lob_rise.entry_burn_dv`, as Falcon 9's entry burn does). Measured on that return, the
   growth lob lofts 1/1.48 of the child's 1.1 km/s lob, so **growth lobs cost x1.48**.
   Deliveries keep the child's 4 Hz plate and 1.1 km/s lob (a sensitivity flies them at
   1 g too). The push's own gravity loss is not charged (S33: mass ratio 1.330 against 1.325).
6. **Pulses are set, not optimized**: the push at 1 g on the starting unit at 2 Hz
   (7.35 MN s), and the departure at a 0.25 g mean at 1 Hz (1.97-2.51 MN s, a 32-34 minute
   burn). The burn's fixed-direction loss is charged: 0.46-0.52 km/s.
7. **Plates are priced** on the spray cup's per-kg curve at half (the author: two lighter
   plates should cost about half the heavy one), each on its own learning count. The Sun
   paraboloid keeps the full curve.
8. **The optimizer** turns the four slug ratios (push, departure, Sun, delivery) and the Sun
   arc for the dear-seed break-even, from the recorded starts (`HEADLINE_STARTS`).

## Results (`make solar-dive`, ~30 s)

| closure | Earth wave | k push / departure / Sun / delivery | growth per 3S | doubling | steady $/kg | break-even cheap / **dear** |
|---|---|---|---|---|---|---|
| **9.86 R_sun, X 90** | 100.8 km/s | 7.9 / 13.9 / 20.0 / 7.0 | **3.27** | 1.92 yr | $83 | $119 / **$151** |
| **4 R_sun, X 150** | 157 km/s | 8.2 / 13.7 / 20.0 / 7.5 | **3.85** | 1.68 yr | $70 | $93 / **$107** |
| 9.86 R_sun, X 60 (ADR 0072's) | 75 km/s | 7.1 / 12.7 / 20.0 / 5.8 | 3.81 (3.60) | 1.70 (1.77) | $78 ($69) | $101 / **$113** ($104) |
| 4 R_sun, X 90 (ADR 0072's) | 101 km/s | 6.6 / 12.1 / 20.3 / 5.4 | 5.72 (4.92) | 1.30 (1.43) | $60 ($55) | $69 / **$71** ($67) |

ADR 0072's figures are in brackets. The bulged hydrogen chamber, on the same footing, is
$430 dear (ADR 0072).

- **The bowl pays for everything else.** It departs at v_e 10.7-21.7 km/s against the
  canted plate's ~5-6, so growth rises at ADR 0072's closures despite the x1.48 growth
  lob, the larger apoapsis turn and the 0.5 km/s finite-burn loss.
- **Cost rises a little**: the dearer growth lob adds ~$7-12/kg at L1, and growth units
  are a third of the lob bill.
- **A faster Earth wave does not pay.** X 90 and X 150 need a bigger Sun boost, so more
  $1000/kg disks, more carrier and a heavier Sun plate per ship. ADR 0072's slower
  returns stay cheaper at both depths.
- **The slug ratios**: the push 7-8, the departure 12-14, the delivery 5-8. The Sun sits at
  20, the bottom of the simulated table, so its true optimum is lower. The departure's
  dollar optimum lies below the simulation's k 20-30, which maximized departing mass per
  kg of impactor: charge is launched mass, and lower k buys Isp.
- **The visible sign's apoapsis turn** is 0.33-0.35 km/s, not 0.24 (S38 item 3): its thrust
  axis lies ~42 degrees further from the push wave's direction. That is ~9% of the stack
  in methalox (93-113 t).
- **Plates**: 25.6 t (9.86 R_sun) and 27.5 t (4 R_sun) against ADR 0072's single 39.9 t
  and 49.6 t. In dollars they were and remain ~$0.3 of each L1 kg.

### Sensitivities (9.86 R_sun, X 90, design held; dear break-even)

| case | growth | dear |
|---|---|---|
| headline | 3.27 | $151 |
| lob climb not charged (the simulation's assumption) | 3.27 | $139 |
| deliveries at 2 Hz / 1 g too (lob x1.48) | 3.27 | $171 |
| push at 3.26 g (the child's 4 Hz; 24 MN s pulse, a 126 t plate) | 3.00 | $173 |
| departure at 0.5 g | 3.37 | $144 |
| Earth plates at 25% / 100% of the spray cup | 3.27 | $148 / $157 |
| disks $500 / $2000 per kg | 3.27 | $144 / $164 |
| lob $50/kg | 3.27 | $229 |
| g cap 20 | 3.16 | $164 |

At 4 R_sun, X 150 the same rows move it $99 (no climb charge), $125 (deliveries at 1 g),
$115 (3.26 g), $106 (0.5 g) and $152 (shade scaling with sunlight).

## Consequences

- The parent's 1 g push stands. The climb it needs is 2.43 km/s, more than twice the
  child's 1.1 km/s, and the booster must brake on the way down. That is priced here, but
  the entry burn's heating at ~3.5 km/s is not checked.
- If the sheets cannot survive the 4 Hz plume, deliveries must fly 1 g as well, which adds
  ~$20/kg. That is the largest open line.
- **Ask the simulation for the Sun table below k 20** (again), and the bowl's real
  collision at 117-140 deg.

## Limits

- The 1 g push is the parent's 2D model: no drag, no Earth rotation, thrust along the
  stream. The unit sits up to ~2000 km below a single stream path, so every sheet must be
  aimed on its own (ADR 0062's per-sheet aim, unpriced beyond guidance mass).
- The entry burn is impulsive at 60 km.
- Compact departure impactors are priced as sheets. Aiming them through the 1 m hole is
  charged only as guidance mass and attrition.
- The bowl at 134-140 deg extrapolates the simulation's capture linearly in cos(theta)
  beyond its 129.9 deg runs. Its closing speeds (111-167 km/s) lie beyond the runs'
  84-108, where only the speed factor (rebound, bond toll) is recomputed.
- Push and departure plate film use ADR 0072's spray-cup rate per unit impulse.

## Reproduction

`make solar-dive` prints the headline, ADR 0072's closures re-flown and the
sensitivities. Tests: `tests/test_solar_dive_growth.py` (the bowl law against the
simulation's, the visible sign against S38, the 1 g climb) and `tests/test_lob_rise.py`
(S33's climb, the entry burn closing where the separation brake cannot).
