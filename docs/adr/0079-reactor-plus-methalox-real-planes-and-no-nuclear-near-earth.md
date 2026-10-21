# ADR 0079: The reactor plus methalox, the real planes, and no nuclear near Earth

Date: 2026-10-21

Status: Accepted. Answers the parent's S51 item 5 and the rest of item 6
(`docs/iapetus_cycle_ledger_asks_for_aim_repo.md` @ `ec5995d`). Extends ADR 0078 with a
second Earth-departure scenario the author added on 2026-10-21.

## Context

ADR 0078 priced the single-use water NTR at Iapetus. S51 item 5 asks for the core
comparison: the nuclear rocket against a reactor that stays at Saturn and makes
methalox for many single-use chemical stages. Item 6 asks for the sites to be scored on
growth rate, with Phoebe's outbound leg and Iapetus's inclination flown on a propagator.

The author added three points in conversation (2026-10-21):

- **No nuclear near Earth.** A nuclear thermal rocket would likely not get approval to
  run inside Earth's magnetosphere. After startup an accident scatters fission
  products; before startup a core is far less of a hazard. So a scenario with chemical
  Earth departures only is compared on everything, beside the hydrogen NTP.
  - Its seed leaves LEO in one burn.
  - Recurring departures are the bags' push to the eccentric orbit, then one methalox
    burn at perigee.
- **Carbon, not necessarily CO2.** The hypothesis is that the dark material carries
  carbon. It is gasified with steam, shifted and methanated. The ceria cycle, or an
  ordinary electric plant with electrolysis, splits water for the extra hydrogen and
  all the LOX.
- **Hydrogen NTP** means the outbound nuclear stage at Earth (900 s), not chemical.

## Decision

1. **`src/saturn_moon_legs.py`, `make saturn-legs` (~20 s).**
   - Earth-Saturn (3.0 yr) and Saturn-Earth (3.7 yr, reaching 51.1 km/s at 200 km)
     Lambert arcs run on astropy's built-in ephemeris, in 13 windows from 2035.
   - Each capture or departure apse line is placed on its moon's real plane, using
     Horizons `sat441l` elements at 2042-07-01: Iapetus i 16.78 deg, node 138.2 deg;
     Phoebe i 172.77 deg, node 277.1 deg. Over 2035-2050 they move under 0.2 deg and
     5 deg.
   - The apse is free on a cone about the asymptote (half-angle `pi - nu_inf`), so it
     meets the moon's plane in at most two places. The plane turn is paid at apoapsis
     as a vector difference.
2. **`src/saturn_methalox.py`, `make saturn-methalox` (~3 min).**
   - Plants: ceria at 10% (5% pessimistic), or electric (Brayton plus SOEC) at 20%. The
     plant is priced per unit, a fixed $50M plus $15M/MWt (ceria) or $30M/MWt
     (electric). Mass is 1.5 / 4.5 t per MWt. Life is 1 / 2 rounds. All of these are
     assumptions.
   - The stage is methalox at 370 s and 10% dry, with a 50 m/s divert. Its engine
     count is set by a 1 h periapsis burn.
   - **Carbon feed as carbon by default:** 0.163 kg per kg of methalox, 1.2x CO2's
     energy (1068 against 890 kJ/mol), regolith taken to about 1000 K. CO2 ice is
     a row.
   - Tanks are made at Saturn by default. Their 75 MJ/kg and their carbon are charged
     to the plant. Shipped tanks are a row.
   - Plant capacity and per-stage consumables are tracked apart:
     - a plant fuels a stage for the overlap of its run with that stage's fuelling
       window;
     - plants are shipped a round ahead of need;
     - the first stage may wait at the moon while the seed plant stockpiles (0-9.5
       years, chosen on the break-even).
3. **The chemical-only scenario in both modules** (`--scenario chemical`; both are
   printed by default).
   - The seed is the repository's refuelled, expended Starship in one burn from LEO
     (`seed_cost`): 112 t sent to 11.36 km/s, at $1,785/kg (cheap end, the base) to
     $7,281 (dear end).
   - Recurring departures are one methalox stage from the eccentric orbit's perigee.
   - Every reactor crosses the magnetosphere cold, as cargo.
4. **Real-plane rows for every Saturn option, the water NTR included.** These rows use
   the median arcs: Earth departure 5.19 km/s (not 4.90), Saturn arrival 10.3-11.9
   km/s, return 10.49 km/s.

## Results ($/kg at the eccentric orbit, Ledger A, compact plug k 4, gated rNPV 10% / 4% loan)

**The headline matrix.** Methalox uses carbon at 1%, tanks made at Saturn, and the
better of ceria and electric (electric everywhere).

| Saturn side | planes | hydrogen NTP Earth: steady / rNPV / loan | chemical only: steady / rNPV / loan |
|---|---|---|---|
| water NTR, Iapetus | coplanar | $90 / **$309** / $131 | $95 / **$532** / $165 |
| water NTR, Iapetus | real | $121 / $475 / $188 | $119 / $1,173 / $270 |
| methalox, Iapetus | coplanar | $86 / $478 / $148 | $91 / $666 / $167 |
| methalox, Iapetus | real | $93 / $603 / $167 | $101 / $947 / $206 |
| **methalox, Phoebe** | coplanar | $81 / **$308** / $129 | $84 / **$338** / $129 |
| **methalox, Phoebe** | real | $87 / **$357** / $140 | $91 / **$426** / $147 |
| solar dive 9.86 / 4 R_sun (dear seed) | | $93 / $143 · $72 / $86 | (no nuclear) |

- **The restriction on nuclear near Earth costs the water NTR 1.7x on the rNPV.** Its
  growth falls from 5.95 to 3.31 a round: the outbound stack is 28.9 kg per kg landed,
  not 16.1. On the real planes it costs 2.5x ($475 to $1,173). Steady cost barely
  moves, and the seed gets cheaper ($9.7B against $16.2B for ten engines).
- **Under it, methalox at Phoebe wins clearly:** $338 against $532 coplanar, and $426
  against $1,173 on the real planes. With hydrogen NTP allowed it ties the water NTR
  coplanar ($308 / $309) and wins on the real planes ($357 / $475).
- **The parent's premise, that the single-use uranium is the expensive line, does not
  hold.** The water engine is $8 of $90/kg; the plant line is $5-9/kg. Methalox wins
  where it does on **Earth mass per kg of PuffSat**: 0.068-0.072 kg with carbon and
  Saturn-made tanks, against the water stage's 0.076. It also wins on **Phoebe's
  shorter round** (7.7 yr against 9.5).
- **Phoebe beats Iapetus for methalox at every carbon grade and ice fraction, on rate and
  on rNPV.** At 1% carbon:
  - ln G / T is 0.253 at Phoebe against 0.200 at Iapetus, with hydrogen NTP; 0.197
    against 0.138 chemical only;
  - doubling is 2.7 years against 3.5;
  - the flip table finds no flip down to 0.1% carbon or 25% ice.
- **Carbon is the open gate, and its form matters.**
  - As carbon, at the base kit (0.3 kg/t): 1% gives G 6.7, 0.3% gives 4.9, and 0.1%
    gives 2.6 at Iapetus with hydrogen NTP. The kit stops paying near 0.1%.
  - As CO2 ice, 3.7x the mass must be dug: 1% CO2 gives G 4.7, and 0.1% does not grow.
  - Chemical only, 0.1% carbon does not repay its seed.
- **Tanks must be made at Saturn.** Shipped, at 10% dry, they are 0.22-0.25 kg per kg
  of PuffSat. Growth falls to 2.1-2.3 with hydrogen NTP, and chemical only the cycle
  does not grow.
- **Plant sizing.**
  - At 10 MWt a ceria plant makes ~1,320 t of methalox a year with carbon (2,300 t
    before its other work): ~7 stages of 826 t a round at Iapetus, 5 at Phoebe.
  - Electric doubles that. 5-10 MWt fuels 3.5-14 stages a round.
  - One plant fuels 5-14 stage-equivalents over its life, against the water NTR's one.
  - Larger units only dilute the fixed $50M: at Phoebe, $424 at 1 MWt, $308 at 10 MWt,
    $297 at 100 MWt (electric).
  - The plant is almost never what limits growth: its mass is ~4% of the Earth mass
    it enables each round.
- **Ceria wins nowhere.** Electric is within $0-25 of it everywhere, and better at
  Phoebe and in small units. Ceria at 5% loses $340 at Iapetus.
- **Shorter stays help a little.** Stockpiling lets the stage load and leave: a 0.1 yr
  stay gives T = 7.34 yr at Phoebe and $312 against $331 (ceria). The Iapetus dog-leg
  stays (ADR 0078); a whole-round count fits two rounds in 20 years either way.
- **The real planes.**
  - The plane turns cost under 0.02 km/s, because the apse is free on its cone.
  - The real arcs cost more than the coplanar ones: capture is 2.81 km/s (median), not
    2.55.
  - Saturn-side outbound is 5.10 km/s at Iapetus, not 4.78, and 4.47 at Phoebe, not
    ~4.15.
  - Phoebe's eccentricity (0.164) is worth 0.22 km/s each way: a 1.23 km/s match near
    its 15.1 million km apoapsis, 1.45 at the mean, 1.71 at periapsis.
- **Ring plane.** No 150 000 km orbit crosses the equator inside the F ring. Several
  cross between 150 000 and 175 000 km, and two Phoebe windows cross inside the G ring.
  A real arc should place its nodes outside 175 000 km or accept the G ring's dust.
- **The $1-2 target.** The cheapest cell is Phoebe, electric, on the loan: $129 ($81
  steady, $49 of it lofting). As in ADR 0078, the target needs the lofted kilogram at
  about $1.

## Consequences

- **Recommendation to the parent.** If nuclear thermal rockets are barred near Earth,
  the Saturn cycle should be **methalox from a kept reactor at Phoebe** with carbon feed
  and Saturn-made tanks. The water NTR at Iapetus remains the simpler, carbon-free
  fallback at about 1.6x the rNPV ($532 against $338 coplanar). Neither beats the solar
  dive's $86-143.
- The plant prices, masses and lives are assumptions, as are the regolith heat, the
  kit's 0.3 kg/t, the 90% plant gate and the carbon grades. Each is a printed row.
- Not done here:
  - Phoebe's anchoring (unpriced);
  - phasing each apse line to the moon's position;
  - a multi-perigee Earth departure;
  - using the carbon route's surplus hydrogen;
  - the most power a heat-only core reaches.
