# Fly the seed direct for now: methalox assists never pay, SEP is unverified

Status: accepted

Builds on: ADR 0035/0036 (the expended, refuelled seed ship and its price),
ADR 0037 (the stepped cost of capital), ADR 0026 (the argon SEP stage's mass
and efficiency), ADR 0007/0008 (phased gravity-assist ladders).

Date: 2026-10-03

## Context

The seed (ADR 0035/0036) flies once, direct to Jupiter, on an expended Starship
refuelled by twelve tankers. Under the bank flight prices the tankers are most
of its bill. The parent's draft (`docs/growth_cost_for_parent.md` §6) asked
whether a slower route sends more PuffSats per ship, so the seed costs less per
kilogram, by enough to pay for arriving later. The test is exact: a route that
makes the seed `k` times cheaper and returns `dt` years later pays when
`k (1 + r)^-dt > 1`, with `r` = 30% before the cycle is proven
(`seed_route.seed_route_worth`). At 30% a two-year delay needs `k` above 1.69.

Two ways to buy a lower departure burn were priced: methalox burns between
flybys, and an argon solar-electric (SEP) stage that rides with the seed. SEP is
charged twice (author, 2026-10-01): its hardware, argon and tankage displace
PuffSats, and the hardware is bought.

## Decision

**Fly the seed direct (EJ) until an SEP route is verified.** No methalox
gravity-assist route pays. SEP routes pay on the dear ship only in an
impulsive model whose treatment of low thrust is optimistic, and they have not
yet been checked with continuous thrust.

1. **Methalox flyby routes never pay.** The burns between flybys eat the mass
   the lower departure saves, and the route returns years later. Best: EEJ at
   0.92x direct on the dear ship and 0.81x on the cheap one. No low thrust is
   involved, so this verdict stands as found.
2. **On the $31M ship nothing beats direct**, even in the impulsive model at
   $10/W. That model is optimistic about SEP (item 3), so a loss in it is a
   loss: the array is a large share of a cheap ship's bill.
3. **SEP on the $670M ship is unverified.** The impulsive search (below) finds
   SEP with Earth or Venus flybys beating direct by up to 1.5x at near-commercial
   prices (1.8x in the cheap-and-light bound) and 1.1-1.2x at $200/W. But the
   model credits each leg's SEP thrust as one burn at the next flyby, and asks
   only whether the leg supplies enough velocity change in total. Real thrust
   spread over a leg also moves the ship, which must still meet the planet at a
   fixed time and place, so the model is optimistic in principle. By how much
   is not measured: the first low-thrust check was itself unreliable
   (§ Low-thrust check). These wins are not adopted until a continuous-thrust
   check confirms them.
4. `growth_cost_report.SEED_ROUTES` stays empty and section 9 of
   `make growth-cost` says why. `growth_cost.RouteSeed` and
   `route_break_even` stay, tested, to price a route once it is verified.
5. `SepStage.capacity` says what it checks: a necessary condition for a route
   to be flyable, not a sufficient one.

**Not decided here, and owed to the paper** (author, 2026-10-02): without a
refuelled Starship, no expendable chemical launch sends a seed-sized mass past
Jupiter, so a few conventional launches flown together, each with an SEP stage,
may be the only provider-independent path. That is a different comparison
(against expendable chemical launch, not a refuelled Starship) and needs a
low-thrust trajectory shown to fly, which this ADR does not yet provide. It goes
to the paper as a direction with that caveat
(`docs/growth_cost_for_parent.md` §5 item 14).

## The impulsive search (`src/seed_route.py`)

Patched conics on circular, coplanar planet orbits phased at their real
ecliptic longitudes (astropy's built-in ephemeris). Lambert legs between
flybys; the Jupiter perijove is solved inside each evaluation so the return
crosses 1 AU on Earth exactly, at no less than `RETURN_FLOOR` = 51.134 km/s.

- **Window:** launch within 6 years of JD 2461353.5 (2026-11-09). Leg bounds
  (yr): inner hops 0.08-1.6, Earth-Earth 0.9-3.2, to Jupiter 0.8-5.0. Perijove
  scanned over log10(r/R_J) in (0, 2.5) at 24 points, then bracketed.
- **Sequences:** EJ, EEJ, EVEJ, EVVEJ, EVEEJ.
- **Ships:** dear = stripped 60 t ship, 13 flights at $50M (Morgan Stanley's
  $500/kg over 100 t) plus a $20M hull = $670M; cheap = stripped 40 t ship, 13
  flights at $2M plus a $5M hull = $31M. Stack per ship from the rocket
  equation at 380 s from 200 km with a 117 m/s steering loss
  (`stack_per_ship`); champions re-priced with the integrated finite-burn
  loss (`seed_payload`).
- **Score:** seed per dollar, discounted at 30% from purchase (window opening)
  to the return. Fitness `-ln(value) + 10 x SEP shortfall + 10 x return-speed
  shortfall`.
- **Optimiser:** pygmo SADE, 8 independent islands per flyby-bend side (16
  total), 40 individuals, 150 generations, 3 evolutions; island `i` on side
  `s` seeded `seed + 2i + s`; best of seeds 7, 101 and 2027. Workers capped by
  free memory (`_island_workers`), which changes wall time only.
- **Methalox nodes:** burned at periapsis, with an 8% stage fraction.
- **SEP stage:** argon at 2000 s, thruster efficiency 0.5, tankage 0.15 kg per
  kg of argon (ADR 0026). Thrust `2 eta P / (m v_e)` scales as `1/r^2`
  outside 1 AU and is **capped at its 1 AU rating inside it** (author,
  2026-10-02: the electronics are sized at 1 AU, and the array is angled away
  from the Sun near Venus to stay cool). A leg's capacity is
  `a_1AU x duration x mean(min(1, 1/r^2))`; burns may be paid on any leg
  before their node (cumulative check).
- **Price tiers** (author, 2026-10-02; unsourced): $10/W near-commercial,
  $50/W mass-produced argon SEP, $200/W conventional. Stage masses paired with
  them are Claude's assumption: 40, 25 and 15 kg/kW, plus $10/W at 15 kg/kW
  as a cheap-and-light bound. Each tier was searched afresh at 2, 4 and 8 W/kg
  on EEJ, EVEJ and EVVEJ (EEJ dropped on the cheap ship, where it cannot tie
  direct even with a free array).

The runs were scratch scripts driving `best_route`; every setting above is what
they passed. Rerunning one case is a single call, e.g.
`best_route("EVEJ", Propulsion.SEP, SepStage(4.0, 40.0, 10.0),
ship=STRIPPED_SHIPS[1], ship_dollars=670e6, seed=7)`, about 5 minutes on 5
cores.

## Impulsive results

Seed per dollar against direct methalox on the same ship. "Ties direct at" is
the array price at which the route, as found at $200/W, matches direct
(re-priced, so a lower bound); "never" means it loses even with a free array.
The dear ship's methalox flyby rows come from the first, uncapped run (the cap
does not touch methalox): EEJ 0.92x, EVEJ 0.29x, EVVEJ 0.29x; its EVEEJ
methalox case was not rerun.

### $670M ship, SEP power capped at 1 AU, $200/W and 15 kg/kW (re-priced at other prices for the tie)

Direct: 58.5 kg per $M.

| Route | Seed (t) | Cost ($M) | Return (yr) | kg per $M | vs direct | Ties direct at |
|---|---|---|---|---|---|---|
| EJ direct | 93.9 | 670 | 3.33 | 58.5 | 1.00× | — |
| EEJ sep 0.5 W/kg | 119.3 | 682 | 6.61 | 30.9 | 0.53× | never |
| EEJ sep 1 W/kg | 150.8 | 703 | 6.61 | 37.9 | 0.65× | never |
| EEJ sep 2 W/kg | 183.0 | 757 | 5.52 | 56.8 | 0.97× | $152/W |
| EEJ sep 4 W/kg | 306.9 | 1,050 | 5.52 | 68.7 | 1.18× | $297/W |
| EVEJ sep 0.5 W/kg | — | — | — | none found | — | — |
| EVEJ sep 1 W/kg | 97.3 | 692 | 7.70 | 18.6 | 0.32× | never |
| EVEJ sep 2 W/kg | 453.8 | 907 | 7.70 | 66.4 | 1.14× | $304/W |
| EVEJ sep 4 W/kg | 264.2 | 998 | 5.52 | 62.3 | 1.07× | $240/W |
| EVVEJ sep 0.5 W/kg | 185.2 | 691 | 7.70 | 35.6 | 0.61× | never |
| EVVEJ sep 1 W/kg | 206.2 | 722 | 7.70 | 37.9 | 0.65× | never |
| EVVEJ sep 2 W/kg | 323.7 | 850 | 7.70 | 50.5 | 0.86× | $72/W |
| EVVEJ sep 4 W/kg | 401.6 | 1,190 | 6.61 | 59.6 | 1.02× | $209/W |
| EVEEJ sep 0.5 W/kg | 229.4 | 694 | 7.62 | 44.7 | 0.77× | never |
| EVEEJ sep 1 W/kg | 201.2 | 716 | 6.60 | 49.7 | 0.85× | never |
| EVEEJ sep 2 W/kg | 176.0 | 763 | 5.50 | 54.5 | 0.93× | $88/W |
| EVEEJ sep 4 W/kg | 182.2 | 868 | 5.50 | 49.6 | 0.85× | $67/W |

### $31M ship, the same settings, methalox routes included

Direct: 1532.3 kg per $M.

| Route | Seed (t) | Cost ($M) | Return (yr) | kg per $M | vs direct | Ties direct at |
|---|---|---|---|---|---|---|
| EJ direct | 113.9 | 31 | 3.33 | 1532.3 | 1.00× | — |
| EEJ methalox | 162.9 | 31 | 5.52 | 1235.9 | 0.81× | — |
| EEJ sep 0.5 W/kg | 138.4 | 45 | 6.61 | 537.2 | 0.35× | never |
| EEJ sep 1 W/kg | 169.0 | 68 | 6.61 | 437.8 | 0.29× | never |
| EEJ sep 2 W/kg | 199.8 | 126 | 5.52 | 372.1 | 0.24× | never |
| EEJ sep 4 W/kg | 125.2 | 167 | 4.42 | 235.8 | 0.15× | never |
| EVEJ methalox | 88.5 | 31 | 7.70 | 379.2 | 0.25× | — |
| EVEJ sep 0.5 W/kg | — | — | — | none found | — | — |
| EVEJ sep 1 W/kg | 113.5 | 56 | 7.70 | 266.9 | 0.17× | never |
| EVEJ sep 2 W/kg | 469.1 | 276 | 7.70 | 225.4 | 0.15× | $8/W |
| EVEJ sep 4 W/kg | 275.1 | 370 | 5.52 | 174.6 | 0.11× | $7/W |
| EVVEJ methalox | 55.6 | 31 | 7.68 | 239.1 | 0.16× | — |
| EVVEJ sep 0.5 W/kg | 203.9 | 54 | 7.70 | 503.3 | 0.33× | never |
| EVVEJ sep 1 W/kg | 202.7 | 77 | 7.69 | 350.3 | 0.23× | never |
| EVVEJ sep 2 W/kg | 178.9 | 125 | 6.59 | 254.4 | 0.17× | never |
| EVVEJ sep 4 W/kg | 169.3 | 215 | 6.59 | 139.8 | 0.09× | never |
| EVEEJ methalox | 165.6 | 31 | 7.64 | 719.2 | 0.47× | — |
| EVEEJ sep 0.5 W/kg | 247.0 | 56 | 7.67 | 590.7 | 0.39× | never |
| EVEEJ sep 1 W/kg | 212.9 | 78 | 6.59 | 484.7 | 0.32× | never |
| EVEEJ sep 2 W/kg | 190.5 | 132 | 5.49 | 342.3 | 0.22× | never |
| EVEEJ sep 4 W/kg | 182.9 | 224 | 5.50 | 193.1 | 0.13× | never |

### $670M ship, price tiers, each re-searched; seed per dollar as a multiple of direct

Direct: 58.5 kg per $M.

| Tier | EEJ 2 W/kg | EEJ 4 W/kg | EEJ 8 W/kg | EVEJ 2 W/kg | EVEJ 4 W/kg | EVEJ 8 W/kg | EVVEJ 2 W/kg | EVVEJ 4 W/kg | EVVEJ 8 W/kg |
|---|---|---|---|---|---|---|---|---|---|
| near-commercial | 1.03 | 1.51 | 1.12 | 1.41 | 1.52 | 1.02 | 1.10 | 1.45 | 0.97 |
| best case | 1.09 | 1.79 | 2.29 | 1.51 | 1.81 | 1.59 | 1.12 | 1.73 | 1.60 |
| mass-produced | 1.04 | 1.51 | 1.35 | 1.38 | 1.43 | 1.11 | 1.01 | 1.42 | 1.01 |
| conventional | 0.97 | 1.18 | 0.86 | 1.14 | 1.07 | 0.83 | 0.86 | 1.02 | 0.64 |

Tiers: near-commercial $10/W at 40 kg/kW, best case $10/W at 15 kg/kW,
mass-produced $50/W at 25 kg/kW, conventional $200/W at 15 kg/kW. The EEJ
8 W/kg rows at $10 and $50 per watt leave Earth with no excess speed and let the
array supply 10.86 km/s while the ship sits on Earth's orbit, which the
impulsive model cannot represent at all; they are not results. All of these
SEP rows are impulsive-model figures and fail the low-thrust check below.

### $31M ship, price tiers (EEJ dropped)

Direct: 1532.3 kg per $M.

| Tier | EVEJ 2 W/kg | EVEJ 4 W/kg | EVEJ 8 W/kg | EVVEJ 2 W/kg | EVVEJ 4 W/kg | EVVEJ 8 W/kg |
|---|---|---|---|---|---|---|
| near-commercial | 0.88 | 0.75 | 0.39 | 0.76 | 0.68 | 0.33 |
| best case | 0.94 | 0.88 | 0.67 | 0.72 | 0.78 | 0.53 |
| mass-produced | 0.43 | 0.34 | 0.18 | 0.37 | 0.27 | 0.13 |
| conventional | 0.15 | 0.11 | 0.06 | 0.17 | 0.09 | 0.05 |

## Low-thrust check (first attempt, inconclusive)

A first check re-flew the best SEP routes with pykep 3.0.1's Sims-Flanagan legs
(planar, 16 impulses per powered leg), keeping the last flyby and everything
after it, freeing the launch date, departure, intermediate flyby dates and
thrust, and solving with scipy SLSQP under a thrust homotopy (50x down to 1x)
with basin hopping. Its results are recorded because each failure taught
something, but **none of them stands**:

- **The conservative thrust setting made routes infeasible by construction.**
  Holding thrust at each leg's farthest point from the Sun (2.2 AU on EEJ)
  left the stage about 1.2 km/s of capacity against the 2.38 km/s EEJ needs.
  A positive control (targets reached by flying known thrust patterns
  forward, 6 of 6 recovered) exposed this. At the optimistic setting (1 AU
  thrust throughout, capped) EEJ at 2 W/kg still needed 1.8x its thrust, and
  EEJ at 4 W/kg "flew".
- **Sixteen impulses are far too coarse at this thrust.** Each stands for
  about 43 days of thrusting, an impulse of 0.8-1.5 km/s. Integrating the
  "flyable" EEJ 4 W/kg solution's throttle history continuously, it missed
  Earth by 15 million km and 2.1 km/s; the optimiser had exploited the
  discretisation. The coarse model's failures are no more reliable than its
  successes.
- **Local optima.** The same launch date converged to 109 t of seed in one run
  and 240 t in another, depending only on the basin-hopping random sequence.

**What a valid check needs:** pykep's zero-order-hold leg (`pk.leg.zoh` with
`pk.ta.zoh_kep`), which integrates continuous thrust exactly through each
segment and supplies exact gradients; thrust bracketed between the near and
far settings or, better, varying as `min(1, 1/r^2)` along the arc; several
launch dates and random restarts; positive controls built with the same
dynamics; and every accepted solution confirmed by an independent continuous
integration that meets each planet (position to about 150 km, velocity to a
few cm/s, in the patched-conic sense).

## Consequences

- The seed stays as ADR 0035/0036 price it. `tab:seed_amortization`,
  `tab:seed_return` and the growth-cost break-evens do not move. The preview in
  which SEP routes brought solved hydrogen under $200/kg on the dear seed is
  not adopted; it returns only if a verified SEP route supports it.
- §6 of `docs/growth_cost_for_parent.md`: the route paragraphs say methalox
  flybys do not pay and SEP is unverified, an upside on a dear seed if a
  continuous-thrust check confirms it.
- The continuous-thrust check is the next step (§ Low-thrust check).

## Considered and rejected

- **Quoting the impulsive SEP results as wins.** The model is optimistic in
  principle about spread-out thrust; until measured, the wins are upside, not
  results.
- **Treating the first low-thrust check's failures as a verdict.** It was too
  coarse to support either a failure or a success (§ Low-thrust check).
- **A burn-to-approach-speed ratio as the screen.** It compares burn sizes,
  not whether thrust can be delivered in time; it is no substitute for flying
  the trajectory.
- **pykep for the impulsive search.** It can abort at interpreter exit
  (pykep 3.0.1 with pygmo 2.19.8 on aarch64); the search uses pygmo and
  `conic_kernel` only. Low-thrust scripts save their results and exit with
  `os._exit`.

## Reproduction

Fast: `tests/test_seed_route.py` (the 1 AU cap, the route-worth test, the
stage charges), `tests/test_growth_cost.py` (`route_break_even`). Slow:
`test_the_best_direct_seed_closes_on_earth_at_the_return_floor`.
