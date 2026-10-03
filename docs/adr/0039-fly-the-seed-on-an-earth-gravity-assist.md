# Fly the seed on an Earth gravity assist; SEP is out of scope

Status: accepted

Builds on: ADR 0035/0036 (the expended, refuelled seed ship and its price),
ADR 0040 (the valuation: 10% a year, 50% odds the cycle works), ADR 0007/0008
(phased gravity-assist ladders). ADR 0026 for the SEP stage's figures.

Date: 2026-10-03 (revised the same day, see "History")

## Context

The seed (ADR 0035/0036) flies once to Jupiter on an expended Starship
refuelled by twelve tankers. Under the bank flight prices the tankers are most
of its bill. The parent's draft (`docs/growth_cost_for_parent.md` §6) asked
whether a slower route sends more PuffSats per ship, so the seed costs less per
kilogram, by enough to pay for arriving later. A route that makes the seed `k`
times cheaper and returns `dt` years later pays when `k (1 + r)^-dt > 1`
(`seed_route.seed_route_worth`). Under ADR 0040 the chance the cycle works is
the same whichever way the seed travels, so it cancels, and `r` is the time
value of money, 10% a year.

## Decision

1. **Fly the seed on one Earth loop (EEJ) with methalox.** It sends 1.64 times
   the direct route's seed on the $670M ship and 1.43 times on the $31M ship,
   at no extra cost, and returns 2.18 years later: **1.33x** direct seed per
   dollar on the dear ship, **1.16x** on the cheap one. Galileo's
   Venus-Earth-Earth route (EVEEJ) does slightly better on the cheap ship
   (1.19x) and barely pays on the dear one (1.04x). Routes needing several km/s
   of burns between flybys (EVEJ, EVVEJ) lose.
2. **`growth_cost_report.SEED_ROUTES`** carries EEJ and EVEEJ on both ships
   (price ratio = the route's dollars per seed kilogram over direct's, delay =
   its later return), and section 9 of `make growth-cost` prices the program
   with the seed on each (`growth_cost.route_break_even`, the proof sliding
   with the delay).
3. **SEP is out of scope.** Deciding it needs a trajectory designed for low
   thrust from end to end; see "SEP: what was learned". `SepStage` stays in
   `seed_route.py`, tested, with its capacity check documented as necessary but
   not sufficient.
4. **Without Starship, gravity assists are the hedge** (author, 2026-10-03).
   The direct route leaves at 11.3 km/s of excess speed, beyond what
   conventional expendable launchers deliver usefully; EEJ leaves at 6.8 km/s,
   Venus routes at 2.8-4.4. Without a refuelled Starship, whether through
   competition, technical setbacks or access, the choice is assisted routes or
   none. Owed to the paper as a direction (`docs/growth_cost_for_parent.md` §5
   item 14); not priced, since no other launcher is modelled.

## Results

Seed per dollar against direct on the same ship, delay charged at 10% a year
(ADR 0040; the odds cancel). Launch and return in years from the window's
opening; v_inf is the departure excess speed (km/s); node burns are methalox,
total km/s between flybys. The last column is the same search scored at 30% a
year (ADR 0037's rate), where every route lost.

| Ship / route | Launch | Return | v_inf | Node burns | Stack (t) | Seed (t) | vs direct, ADR 0040 | at 30%/yr |
|---|---|---|---|---|---|---|---|---|
| $670M (stripped 60 t): EJ (direct) | 0.00 | 3.33 | 11.30 | 0.00 | 93.9 | 93.9 | 1.00x | 1.00x |
| EEJ | 0.33 | 5.52 | 6.81 | 2.57 | 332.4 | 153.7 | **1.33x** | 0.92x |
| EVEJ | 2.79 | 7.70 | 4.42 | 5.61 | 533.6 | 85.3 | **0.60x** | 0.29x |
| EVVEJ | 0.81 | 7.70 | 8.39 | 3.24 | 226.8 | 84.5 | **0.59x** | 0.29x |
| EVEEJ | 0.69 | 7.67 | 9.09 | 0.83 | 187.9 | 147.3 | **1.04x** | — |
| $31M (stripped 40 t): EJ (direct) | 0.00 | 3.33 | 11.30 | 0.00 | 113.9 | 113.9 | 1.00x | 1.00x |
| EEJ | 0.33 | 5.52 | 6.90 | 2.51 | 345.7 | 162.9 | **1.16x** | 0.81x |
| EVEJ | 2.79 | 7.70 | 4.42 | 5.61 | 553.5 | 88.5 | **0.51x** | 0.25x |
| EVVEJ | 4.23 | 10.96 | 5.70 | 4.15 | 441.3 | 121.1 | **0.51x** | 0.16x |
| EVEEJ | 0.73 | 7.69 | 8.88 | 0.22 | 219.2 | 205.5 | **1.19x** | 0.47x |

Steady-state break-even ($/kg, Estimate, ADR 0040), seed direct and on each
route (`make growth-cost` §9):

| Design | Direct, cheap / dear | EEJ, cheap | EEJ, dear | EVEEJ, cheap | EVEEJ, dear |
|---|---|---|---|---|---|
| Methalox | 352 / 1396 | 346 | 1129 | 346 | 1357 |
| Methane, solved | 137 / 222 | 136 | 201 | 136 | 219 |
| Hydrogen, solved | 134 / 194 | 134 | **179** | 134 | 192 |

On the cheap seed the route hardly moves the break-evens: the seed is a small
share of the bill. On the dear seed it is worth $15-21/kg to the solved
chambers.

## The search (`src/seed_route.py`)

Patched conics on circular, coplanar planet orbits phased at their real
ecliptic longitudes (astropy's built-in ephemeris). Lambert legs between flybys;
the Jupiter perijove is solved inside each evaluation so the return crosses
1 AU on Earth exactly, at no less than `RETURN_FLOOR` = 51.134 km/s. Chemical
burns last minutes, so treating them as impulses at the flyby is accurate.

- **Window:** launch within 6 years of JD 2461353.5 (2026-11-09). Leg bounds
  (yr): inner hops 0.08-1.6, Earth-Earth 0.9-3.2, to Jupiter 0.8-5.0. Perijove
  scanned over log10(r/R_J) in (0, 2.5) at 24 points, then bracketed.
- **Ships:** dear = stripped 60 t ship, 13 flights at $50M (Morgan Stanley's
  $500/kg over 100 t) plus a $20M hull = $670M; cheap = stripped 40 t ship, 13
  flights at $2M plus a $5M hull = $31M. Stack from the rocket equation at
  380 s from 200 km with a 117 m/s steering loss (`stack_per_ship`); champions
  re-priced with the integrated finite-burn loss (`seed_payload`).
- **Node burns:** methalox at periapsis, with an 8% stage fraction.
- **Score:** `discounted_seed_per_dollar(..., rate=0.30, late_rate=0.10,
  switch_years=3.33)`: 30% until the direct route's return (3.33 yr on both
  ships), 10% after. Every route returns later than direct, so its value
  relative to direct is exactly the ADR 0040 test `k x 1.1^-dt`.
- **Optimiser:** pygmo SADE, 8 independent islands per flyby-bend side (16
  total), 40 individuals, 150 generations, 3 evolutions; island `i` on side
  `s` seeded `seed + 2i + s`; best of seeds 7, 101 and 2027. Workers capped by
  free memory (`_island_workers`), which changes wall time only.

The runs were a scratch script calling `best_route` with those settings; one
case is e.g. `best_route("EEJ", Propulsion.METHALOX, ship=STRIPPED_SHIPS[1],
ship_dollars=670e6, seed=7, late_rate=0.10, switch_years=3.33)`, a few minutes
on 5 cores.

## SEP: what was learned (out of scope)

An argon SEP stage (2000 s, efficiency 0.5, tankage 0.15 kg/kg, power capped
at its 1 AU rating inside 1 AU; author's price tiers $10/$50/$200 per watt) was
searched with the same tool. The impulsive model credited SEP routes with wins
on the dear ship of up to 1.5x at $10/W and 1.1-1.2x at $200/W (at 30% a year;
full tables in `docs/adr/0039-fly-the-seed-direct.md` as committed at
`d6dcfff`). On the cheap ship no SEP
route beat direct even there. Three lessons, so nobody repeats them:

1. **The impulsive model is optimistic about SEP.** It credits a leg's thrust
   as one burn at the next flyby and asks only whether the leg supplies enough
   velocity change in total, not whether thrust spread over the leg can still
   meet the planet on time.
2. **Sixteen Sims-Flanagan impulses per leg are far too coarse at this
   thrust.** A first check used them; integrated continuously, its "flyable"
   solution missed Earth by 15 million km. Its failures were no more reliable
   than its successes. A conservative thrust bound (thrust at the leg's farthest
   point) also made routes infeasible by construction.
3. **With continuous thrust** (pykep 3.0.1 `pk.leg.zoh` with `pk.ta.zoh_kep`,
   24 segments per leg, agreeing with an independent scipy integration to the
   kilometre), keeping the sweep's last flyby and everything after it, freeing
   the launch date, departure and thrust, with a thrust homotopy, basin
   hopping and the `1/r^2` cap phased in: **EEJ at 2 W/kg** needed twice its
   array's thrust at every one of 6 launch dates; **EEJ at 4 W/kg** flew at its
   full 1 AU thrust but failed once thrust fell with distance from the Sun (its
   leg reaches 1.69 AU), at all 6 dates. EVEJ runs were stopped when SEP was
   ruled out of scope.

A real SEP case needs a low-thrust design from the start: every flyby, the
Jupiter leg and the return free. Scratch code for the continuous-thrust check
is in the gitignored `todos/seed_routes/` (not load-bearing).

## History

The first version of this ADR (2026-10-03, then named
`0039-fly-the-seed-direct.md`) concluded "fly the seed direct",
because at ADR 0037's 30% a year every gravity-assist route lost, and it
claimed SEP routes could not be flown, from the 16-impulse check. The second
(`d6dcfff`) withdrew the SEP claim as unverified. ADR 0040 then replaced the
30% rate; the routes were re-searched under it and the decision is now the
Earth gravity assist.

## Consequences

- `tab:seed_amortization` stays as ADR 0035/0036 price the seed (direct). The
  paper's seed section should add the route and its break-evens (above).
- §6 of `docs/growth_cost_for_parent.md` carries the draft text.

## Considered and rejected

- **Scoring at 30% a year.** It charged venture risk for every year of
  waiting, which decided the question by assumption (ADR 0040).
- **Quoting the impulsive SEP wins.** Optimistic in principle and not
  confirmed by continuous thrust.
- **pykep for the chemical search.** It can abort at interpreter exit (pykep
  3.0.1 with pygmo 2.19.8 on aarch64); the search uses pygmo and
  `conic_kernel` only.

## Reproduction

Fast: `tests/test_seed_route.py` (the stepped discount, the route-worth test,
the SEP stage charges and 1 AU cap), `tests/test_growth_cost.py`
(`route_break_even`, the risked schedule). Slow:
`test_the_best_direct_seed_closes_on_earth_at_the_return_floor`. Route
break-evens: `make growth-cost` §9.
