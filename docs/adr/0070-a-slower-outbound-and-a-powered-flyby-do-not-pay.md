# ADR 0070: A slower outbound leg and a powered Jupiter flyby do not pay

Date: 2026-10-14

Status: Rejected (author, 2026-10-14). The circular, impulsive first pass
fails, so no realistic-ephemeris version is built.

## Context

The author asked whether the 3S cycle could fly out to Jupiter slower, on a
cheaper Earth departure, pass deeper, and burn at perijove to make up the time.
It would still intercept Earth after exactly three synodic periods, perhaps
returning faster. The Earth departure is a chamber burn at about 7 km/s of
effective exhaust (methane) or 10.6 (hydrogen). If the Oberth effect kept the
Jupiter burn small, the cheaper departure looked like a major win.

The author asked for ideal circular, coplanar orbits and instantaneous burns
first, with realism added only if the first-order gain was real.

## Method

`src/powered_resonance.py` (`make powered-resonance`) reuses ADR 0012's
circular Lambert pair (`circular_resonance_impulse._evaluate_geometry`): a
prograde Earth-to-Jupiter leg and a retrograde Jupiter-to-Earth leg closing on
Earth after exactly 3S.

- **Powered flybys are admitted.** ADR 0012 kept only unpowered flybys, where
  the excess speed into Jupiter equals the speed out. Here the mismatch is paid
  by a tangential perijove burn.
- **The perijove is solved, not chosen.** It is the radius at which the
  incoming and outgoing hyperbolae bend through the turn the two legs need
  (`perijove_for_turn`), no lower than the 4,000 km floor.
- **The grid** is 361 departure phases by 301 encounter times. 101,942
  closures can be flown.
- **The valuation.** Each closure's Earth departure burn (from the cycle
  orbit's 200 km periapsis) is valued at a chamber's exhaust. Its Jupiter burn
  is paid in methalox by the whole batch.

## Results

Least Jupiter burn per Earth departure excess speed, 3S:

| Departure v_inf | Departure burn | Jupiter burn | Excess at Jupiter, in / out | Perijove | Outbound | Return v_inf |
|---|---|---|---|---|---|---|
| 11.60 km/s (unpowered) | 5.04 km/s | 0 | 15.57 / 15.57 | 5.4 R_J | 1.20 yr | 55.7 km/s |
| 11.48 | 4.95 | 0.15 km/s | 15.04 / 15.33 | 5.7 | 1.22 | 55.1 |
| 10.98 | 4.60 | 1.01 | 13.77 / 15.70 | 6.1 | 1.29 | 56.1 |
| 10.48 | 4.25 | 1.99 | 12.38 / 16.16 | 6.7 | 1.39 | 57.2 |
| 10.00 | 3.92 | 3.18 | 10.85 / 16.82 | 7.4 | 1.52 | 58.7 |
| 9.50 | 3.59 | 4.93 | 9.12 / 18.13 | 8.4 | 1.70 | 61.0 |
| 9.00 | 3.27 | 9.03 | 6.84 / 21.74 | 11.2 | 2.07 | 65.2 |

- **The unpowered closure is best for every chamber.** Methane at 7 km/s,
  hydrogen at 10.6, and even a hypothetical 20 km/s chamber all pick the
  departure at 11.60 km/s with no Jupiter burn.
- **A methalox burn at Jupiter pays only if it costs less than the Earth
  saving.** The break-even is about half the saving for methane and a third for
  hydrogen. Beyond the first ~0.1 km/s saved it costs 1.5 to 5 times the
  saving.
- **A faster return is available but dear.** The return rises from 55.7 to 57
  km/s for a 2 km/s Jupiter burn, and to 61 km/s for 4.9 km/s.

## Why

1. **The energy needed at Jupiter climbs steeply.** The cycle is pinned at 3S,
   so every month added to the outbound comes off the retrograde return. A
   shorter return from 5.2 AU needs a much faster departure from Jupiter: from
   15.6 km/s at the unpowered closure to 16.2 at a 10.5 km/s Earth departure,
   and to 21.7 at 9.0. Meanwhile the slower outbound arrives with less: 15.6
   down to 12.4 and 6.8 km/s.
2. **Geometry pins the perijove where the leverage is only moderate.** The two
   legs fix the turn at Jupiter, here 65-70 degrees, and a deeper pass bends
   more. Arriving slower costs little leverage in itself: at 6.1 R_J, 13.8
   against 15.6 km/s lowers the perijove speed by about 3%. Falling closer is
   what would help, and the geometry forbids it.
   - **The worked example.** 13.77 km/s in and 15.70 out, 69.9 degrees of turn,
     pins the perijove at 6.1 R_J, where the burn is 1.0 km/s.
   - **The deep pass would turn the wrong way.** At the 4,000 km floor the same
     speed change costs only 0.48 km/s, but the flyby bends through about 124
     degrees, 54 more than the legs allow.
   - **A tangential burn cannot fix it.** It changes speed, not direction, and
     redirecting a 15.7 km/s excess by 54 degrees costs on the order of 14 km/s.
   - **No geometry escapes it.** The scan includes every phase and encounter
     time, and the cheapest closures still pass at 5.7-11 R_J.

## Decision

Reject the slower outbound leg with a powered flyby. Do not build the
realistic version. The 3S cycle stays on its unpowered, minimum-departure
closure.

## Limits

- **The grid is coarse,** with no local refinement around the knee. Refinement
  could move the first ~0.1 km/s slightly, not a 1.5-5 times gap.
- **The Earth burn is priced from the 200 km periapsis,** not the ledger's 600
  km. At 600 km the Earth saving is slightly smaller, which only strengthens
  the rejection.
- **Only tangential, impulsive perijove burns are considered.** A non-tangential
  burn that also turns the trajectory costs more, not less.

## Reproduction

`make powered-resonance` (about 30 s). Tests:
`tests/test_powered_resonance.py` (the worked example is fast; the verdict and
the 10.5-11 km/s frontier point are slow).
