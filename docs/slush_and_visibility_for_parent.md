# Slush hydrogen, the 4884 K chamber and a departure kept in view, for the parent paper

Raised 2026-10-14 in `aim_is_all_you_need`, ADRs 0065-0067; the split's
charge, ADRs 0068-0071. Target repo:
`katzseth22202/Balloon-Pulse-Propulsion`. Passages are located by quoted
wording in `templateArxiv.tex`, as of its commit `5e7edbf`.

| # | item | for the paper? |
|---|---|---|
| 1 | Store the hydrogen charge as slush everywhere | **yes** |
| 2 | Fly hydrogen at 4884 K; explain why 5500 K does not pay | **yes; changes `tab:wall_pairings`' framing** |
| 3 | The departure must keep its stream in view of Earth | **yes, a correction** |
| 4 | The headline figures that move | **yes** |
| 5 | What Earth's bending of the impact angle is worth | already right; one number to refresh |

## 1. Slush everywhere

Every hydrogen charge is stored as 50% slush: 15% denser than boiling-point
liquid, so the tank is 0.178 kg per kg instead of 0.205, and 18% more heat sink.

- **The hold boils nothing.** Melting the solid and warming the liquid from the
  13.8 K triple point absorbs about 81 kJ/kg before any of it boils, 18% of
  the 446 kJ/kg that boils it. At the insulation that boils liquid at 0.3% a
  day, that is about 60 days with nothing lost, against a 20-day hold.
- **Cryostats drop just before ignition**, not with the plate. They ride the
  reversal and the coast, so the insulation is on for the whole hold. Nothing
  burns in between, so the mass charge is unchanged.
- **Passages:**
  - Cycle step 3, "A hydrogen stack also drops its cryostats, and it loses its
    boil-off over the hold." Now: the stack drops its cryostats just before
    the burn, and its slush loses nothing over the hold.
  - "Hydrogen's cryostats and boil-off, swept from none to 5% cryostats and
    0.5% a day". This is now a liquid-storage sensitivity. Slush boils none;
    5% cryostats instead of 3% cost under 1% of growth (`make slush-hydrogen`:
    2.498 against 2.516).
- **Basis:** NASA Lewis K-Site and Small Scale Hydrogen Test Facility, 1990-94
  (McNelis et al., NTRS 19950017766; Hardy and Whalen, NASA TM 105596 and TM
  105191). No flight-scale slush system exists.

## 2. Fly hydrogen at 4884 K, not 5500 K

The flown hydrogen chamber is the bulged one: the sphere's 60 kg charge in the
methane chamber's r 3.0 m bulge, 106 m3, which lands at **4884 K and 142 bar**.
The impact simulation also solved it at 5500, 6000 and 7000 K. None pays.

**Why a hotter charge loses** (cycle 0, both as slush, at the solved rows):

| | 4884 K, 2.27 Hz | 5500 K, 2.66 Hz | 5500 K at 3.09 Hz (thrust matched) |
|---|---|---|---|
| Thrust per chamber | 1.50 MN | 1.29 MN | 1.50 MN |
| Chambers flown | 1 | 2 | 1 |
| Rods per tonne of stack | 26.0 kg | 35.3 kg | 36.4 kg |
| Growth | **1.786** | 1.672 | 1.811 |

- **The exhaust gains only 6%** per kilogram carried (8.51 to 9.02 km/s). The
  hotter gas holds more of its energy in broken bonds, and the nozzle returns
  only 60-71% of them, so the efficiency falls from 0.78 to 0.69.
- **Each rod buys 27% less impulse** (661 to 485 kN s). A pulse's energy is
  fixed by the rod and its momentum goes as `sqrt(2 m E)`, so a smaller, hotter
  charge gives less. The departure then takes 36% more rods out of the
  returning fleet.
- **Those two nearly cancel**, and thrust decides. At its solved rate, 5500 K
  needs a second 46 t chamber. It would pay only at about 3.1 Hz, 17% above
  the rate its cycle solves.

Flown through the chain, doubling time with the stream kept in view: 4884 K
2.68 yr, 5500 K 2.77, 6000 K 2.88, 7000 K 2.86 (`make departure-levers`, the
"solved" rows at the wall-heat-debited impulses).

**Passages:** `tab:wall_pairings` and "Two pairings of gas and wall survive. The
first is hydrogen at 5500 K on a bare copper liner." The sphere's 5500 K is the
pairing table's; the chamber flown is the bulged one at 4884 K. "Hydrogen at
5500 K can use a bare copper liner" stands for the sphere.

## 3. The departure must keep its stream in view (a correction)

The departure-wave PuffSats must reach the vehicle without passing through
Earth, for the whole burn.

- **What the paper says.** "A chamber held in one direction, with its burn
  centered on periapsis, sees the stream 19.8 off head-on at ignition and 11.7
  at cutoff." That is a 20-minute burn on the circular 3S closure, whose
  stream arrives 144.9 degrees from the departure, and there Earth never
  blocks it.
- **What the flown chain does.** Its burns run 25-38 min (methane) and 41-60
  min (hydrogen). Its streams arrive 154-175.5 degrees from the departure
  (3S cycles 167-168; 2S cycles 154-175.5). A burn centred on periapsis loses
  the stream behind Earth on its approach leg: at 175 degrees, 27% of even a
  20-minute burn.
- **The fix is to start later.** Blocking always falls at the start of the
  burn, so the burn shifts after periapsis. Every flown stream has a fully
  visible placement up to two-hour burns. It costs finite-burn loss: the
  bulged hydrogen chamber's mean loss rises from 1.17 to 1.47 km/s.
- **The other departure mirror does not help.** It is marginal or blocked on
  every cycle.
- **Passages:**
  - "Every burn is centered on periapsis, and an earlier ignition might lose
    less on the longest burns." Now: a *later* ignition is required, and it is
    priced.
  - The turnaround paragraph's 20-minute numbers. Keep them as the reference
    closure's, and add that the flown burns are longer and must shift after
    periapsis.

**Misses are not priced.** About half the stream would enter the atmosphere if
it missed. A polyethylene rod arriving at 55 km/s relative to Earth ablates.

**Thrust.** A second chamber halves the loss but costs more than it recovers
(2.72 against 2.61 yr). Pulse rate pays more than before: 2.4 Hz hydrogen
doubles in 2.44 yr, 3 Hz methane in 3.10.

## 4. Headline figures

Estimate book, spray cup, film carried, lob climbing, 10% a year, 50% odds. Each
is the repo's figure before ADRs 0065-0067 against after.

| Figure | Before | After |
|---|---|---|
| Bulged hydrogen chamber: doubling (`make departure-levers`) | 2.54 yr | **2.61 yr** |
| Its annual growth, 10-yr multiple (`tab:seed_amortization`) | 29.7%, 11.4x | **30.4%, 12.8x** |
| Its steady $/kg at L1 | $157 | **$154** |
| Its paper's prices / pessimistic | $150 / $686 | **$148 / $675** |
| Its steady break-even, cheap / dear seed | 385 / 726 | **369 / 690** |
| Its seed | 108 t, 1.18 ships | **108 t, 1.17 ships** |
| Methane AR100, 24 kg: doubling | 3.36 yr | **3.59 yr** |
| Its annual, 10-yr | 22.9%, 6.5x | **21.3%, 5.79x** |
| Its steady $/kg | | **$177** ($209 priced by mass) |
| Methane AR100 at 3 Hz: doubling, $/kg | 3.02 yr | **3.10 yr, $152** |
| Sphere ("hydrogen chamber, whose wall is unverified"), `tab:seed_amortization` | 83.25 t, 54.8%, 75.1x | **82.54 t, 57.3%, 90.3x** |
| Sphere behind the spray cup, steady $/kg, break-even | | **$118, 214 / 340** |

The paper's own figures predate several earlier repo changes (its methane
chamber at $197, its hydrogen at $122 and 2.03 yr). Take the "after" column as
the current state rather than applying the differences to the paper's numbers.

## The two-wave split, now charged (ADRs 0068-0071)

The growth figures in section 4 left the two-wave split's departure
uncharged: the growth wave's trajectory leaves Earth 0.52-0.95 km/s away from
the nozzle wave's, though both left on one burn (ADR 0068). The ledger now
dedicates launch units by role, growth carriers and rod carriers, each
departing on its own trajectory and burn (ADR 0071). With it the best split
gap, which is also the parking orbit, falls from 20 to 10 days.

- **Passages.** "The departing batch divides at Jupiter": the batch no longer
  divides; about 85% of departing units carry growth and the rest carry rods,
  each to its own window. The parking orbit and split gap are 10 days, not 20;
  the 600 km finite-burn figures quoted on the 20-day orbit stand as that
  orbit's.
- **The early, faster 3S growth wave** (68 km/s at 2.8 synodic periods) gains
  about 2% and is declined (ADR 0069).
- **A slower outbound with a powered Jupiter flyby** does not pay at first
  order (ADR 0070).

Figures, the split by role on the 10-day orbit against section 4's "after":

| Figure | Section 4 | Now |
|---|---|---|
| Bulged hydrogen, 2 Hz: doubling | 2.61 yr | **2.77 yr** |
| At 2.4 Hz | 2.44 yr | **2.58 yr** |
| Its steady $/kg at L1 | $154 | **$165** |
| Its steady break-even, cheap / dear seed | 369 / 690 | **404 / 765** |
| Methane AR100, 24 kg: doubling, $/kg | 3.59 yr, $177 | **3.98 yr, $194** |
| Methane AR100 at 3 Hz | 3.10 yr, $152 | **3.38 yr, $163** |
| Sphere, `tab:seed_amortization` | 82.54 t, 57.3%, 90.3x | **84.94 t, 55.0%, 74.8x** |

Take the "Now" column as current. The rankings in sections 2 and 3 are
unchanged: 4884 K still beats 5500 K (2.86 against 2.94 yr, solved), and pulse
rate still beats a second chamber.

## 5. The impact angle

The paper says the head-on assumption is pessimistic "by a fraction of a
percent". That holds on the flown chain (`make impact-angle`). Each departure,
at its ledger length and placed to keep the stream in view, sees impacts 0-17
degrees off head-on. That is worth -0.01% to +0.5% of delivered mass for the
bulged hydrogen chamber and up to 0.8% for methane.

For the record, the bulged chamber's jet leaves at about 1,340 s. Per kilogram
carried that is about 1,380-1,396 s gross and 1,050-1,078 s net of the head-on
debit; the debit takes about 24% of the impulse.

## Reproduction

`make departure-levers`, `make departure-levers-centred`, `make survivable-cost`,
`make hardware-recovery`, `make slush-hydrogen`, `make growth-cost`; `make
test-all`. The stored visibility tables: `make departure-visibility`.
