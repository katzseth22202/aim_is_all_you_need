# The doubling cycle is the derived Earth re-intercept floor, not a 6-month dive

Status: accepted

Date: 2026-07-08

## Context

`main.py` used to estimate the "Sorry, I Don't Need ISRU" growth timeline with
`launch_capacity_time(2, 0.5 * u.year)` — payload doubling every ~0.5 yr, giving a
millionfold in under a decade. The paper's Appendix `sec:earth_reintercept` (paper repo
commit `580250e`, ADR `0001-earth-reintercept-phasing`) corrected this: a boosted solar-dive
return leaves periapsis on an escaping hyperbola and crosses `1 AU` only once, ~136° from
where Earth actually is. Crossing 1 AU is not reaching Earth. The miss is set by the ~295°
whip-around, cannot be re-aimed at periapsis (~5.4 km/s per degree against a ~24 km/s dive
boost), and is fixed by **phasing** the return to an Earth resonance. The shortest
re-intercepting cycle is ~0.82 yr, so a millionfold takes ~16 yr, not under a decade.

This repo is the companion calculations repo the paper-repo ADR explicitly flagged as still
carrying the stale 6-month figure. We had to decide how to represent the corrected cycle.

## Decision

1. **Derive the 0.82 yr floor from geometry, do not hardcode it.**
   `earth_reintercept_cycle_floor()` returns `whip_around / 360° × 1 yr`, where the
   whip-around is `180° + (hyperbolic true anomaly at 1 AU)`. The floor falls out of the
   same geometry that produces the 136° miss — literally "matching the phase of the Earth" —
   so it cannot silently drift from a magic constant.

2. **Keep the doubling factor at 2.** `millionfold_scaling_time()` defaults to factor 2 at
   the derived floor, reproducing the appendix's ~16 yr. Only the two-impulse phasing loop
   holds the factor at two (`two_impulse_phasing_loop()` proves its two boosts are colinear
   and free in total impulse), so ~16 yr is itself a floor; the single-impulse and
   gravity-assist routes are slower.

3. **Prove the appendix chain in code.** Each cited figure (309/233 km/s, 66 d, 295° whip,
   136° gap, 5.4 km/s/°, 24 km/s two-impulse boost, 0.82 yr floor, 16 yr) has a
   verification function in `scenario.py` and a test pinning it to the paper's value.

## Considered options

- **Hardcode `SOLAR_DIVE_REINTERCEPT_CYCLE = 0.82 * u.year` — rejected.** Simple, but 0.82
  becomes a magic number disconnected from the whip-around it comes from; a future change to
  the periapsis depth would not flow through.
- **Model the doubling-factor degradation too — deferred.** The single-impulse resonant dive
  grows the boost 24 → ~41 km/s and drops the factor below two. Faithful, but it goes beyond
  the paper's headline (the factor-2 floor); left out to keep the scope to the ~16 yr figure.
- **Re-aim at periapsis instead of phasing — rejected (physics).** Turning the velocity near
  the 309 km/s periapsis costs ~5.4 km/s per degree; a tens-of-degrees correction runs to
  hundreds of km/s. Encoded as the `periapsis_reaim_cost_per_degree()` check, not a fix.

## Consequences

- The repo's growth headline moves from "under a decade" to ~16 yr, matching the paper.
  `test_millionfold_scaling_time_is_about_16_years` also asserts the retired 6-month cycle
  gave under a decade, pinning the direction of the correction.
- The whip-around needs an escaping-hyperbola true anomaly and time-of-flight, added as
  reusable primitives in `orbit_utils.py` (`hyperbolic_eccentricity`,
  `true_anomaly_at_radius`, `hyperbolic_time_of_flight`).
- The domain language (Earth re-intercept, whip-around, phasing loop, re-intercept cycle
  floor) is recorded in `CONTEXT.md`.
