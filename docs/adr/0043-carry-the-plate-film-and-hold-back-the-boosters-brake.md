# Carry the plate film as launched mass, and hold back the booster's brake at the faster climb

Status: accepted

Amends: ADR 0042 §1 (the lob charge), and the cost book's film line wherever a plate carries
its film. Answers the parent's S12 and the brake it names as unrecomputed in `sec:vertical_lob`.

Date: 2026-10-06

## Context

ADR 0042 left two items open.

**The film (parent S12).** The impact sim's P5 puts the spray cup's film at 4-6 kg of pitch per
12 MN s pulse when the film's own vapor shields the face, and 28-33 kg when it does not. Neither
the ledger nor the cost book carried that mass. The book's film line was 0.1% (Estimate) or
4% (Pessimistic) of the PuffSat mass on the plate. Against the cup's 74-92 kg of PuffSat per
pulse, the sim's figures are 4-8% and 30-45%.

**The brake.** The parent's booster brakes 1.33-1.38 km/s right after separation. Without the
brake it would fall from its ~430 km top and cross 60 km at ~2.6 km/s, a ~21 g entry. With it,
it crosses 60 km at 1.5 km/s (~9 g) and lands on a 0.3 km/s burn. The 114-185 t of propellant
this takes is why the lob lofts 1250-1430 t and not 9-19% more. ADR 0041-0042 charged the
1.1 km/s climb with an integration that held nothing back. A faster climb separates the booster
higher and faster, so its brake grows, and that growth was not charged.

## Decision

1. **The film is burned per unit of impulse inside the push.** A pulse is a fixed 12 MN s
   (`PULSE_IMPULSE`, the parent's `eq:plate_pulse_size`), so film leaves at
   `phi = film per pulse / 12 MN s` kilograms per newton-second. In `water_plate`, `dM/dv`
   gains `-phi M`. The loss does not depend on `k`, so the Pontryagin rule per pulse is
   unchanged. The price picks up `c phi` (`dc/dv = (1 + c k)/(beta w) + c phi`), since a
   kilogram carried further burns film on the way. `PlatePush.film_fraction` records what
   burned. The film is part of the 1500 t unit, so the lob pays for it, and it never reaches
   the departure.

   The push integrates to about 1050 pulses, against the parent's 1200-1500. That figure was a
   ceiling: the unit's mass falls to ~0.65 through the push. So 6 kg a pulse is 6.4 t a push,
   not 7-9 t.

2. **`PlateDesign.film_per_pulse_kg`**, zero by default, so every earlier figure stands.
   `SPRAY_CUP_SHIELDED` (6 kg) and `SPRAY_CUP_UNSHIELDED` (33 kg) take the heavy end of each
   band. `--plate spray-cup-shielded` / `spray-cup-unshielded` run them through the cost model.
   When the design carries its film, `CycleHardware.film` and `DeliveryOption.film` hold its
   mass, and the cost book prices that mass at `film` $/kg in place of `film_fraction`.

3. **The booster holds back its brake (`lob_rise.braked_lob`).** The brake is impulsive at
   separation, and it is sized so the booster crosses 60 km at 1.5 km/s. A vertical entry's
   peak deceleration depends on its entry speed alone (Allen and Eggers), so the same ~9 g
   criterion holds at any climb. The reserve, `dry * (exp((brake + 0.3 km/s)/v_e) - 1)`, is
   solved as a fixed point with where the booster separates. `booster_growth` now takes the
   braked ratio by default; `brake=False` reproduces ADR 0041-0042.

   The 0.75 km/s lob it reproduces: a brake of 1.35-1.39 km/s, a 143-155 t reserve, and 10.5%
   more lofted without the brake. The parent has 1.33-1.38 km/s, 114-185 t and 9-19%.

   | climb at 400 km | brake (380 s) | reserve | charge 380 s | 350 s | unbraked (ADR 0042) |
   |---|---|---|---|---|---|
   | 0.75 km/s | 1.39 km/s | 144 t | 1 | 1 | 1 |
   | 1.0 km/s | 1.52 km/s | 157 t | 1.043 | 1.049 | 1.029 |
   | **1.1 km/s** | **1.58 km/s** | **164 t** | **1.065** | 1.074 | 1.044 |
   | 1.2 km/s | 1.64 km/s | 171 t | 1.089 | 1.101 | 1.059 |

   The booster separates at 133-143 km. Without the brake it would cross 60 km at
   2.60-2.77 km/s. The lob becomes $26.6/kg lofted at 1.1 km/s ($26.1 in ADR 0042). The lifting
   rocket becomes ~1.15-1.32x Super Heavy (1.1-1.2x the 0.75 km/s booster times 1.043-1.101).
   The finite brake is not integrated. Under 4 g it lasts ~40 s. Gravity helps it, and spending
   its thrust at falling speed hurts it, and the impulse is taken as the estimate.

4. **Targets**: `make plate-film` (ledger at each band's ends and both cost headlines), and
   `make lob-brake`.

## Results (spray cup, η_jet 0.60, k <= 8.52, 20-day orbit)

Ledger (`make plate-film`), doubling in years. The lob charge does not enter the ledger.

| film | t / push | H2 solved | CH4 solved | methalox |
|---|---|---|---|---|
| not carried | 0 | 2.019 | 2.277 | 7.82 |
| shielded 4 kg | 4.2 | 2.024 | 2.283 | 7.89 |
| shielded 6 kg | 6.4 | 2.026 | 2.286 | 7.92 |
| unshielded 28 kg | 29.5 | 2.054 | 2.319 | 8.31 |
| unshielded 33 kg | 34.7 | 2.061 | 2.327 | 8.41 |

Cost, Estimate book, 10% a year, 50% odds, lob x1.065. Steady $/kg; break-even steady,
cheap / dear seed:

| plate | H2 solved | CH4 solved | methalox |
|---|---|---|---|
| spray cup, film in book (ADR 0042: 119, 189 / 370) | 120, 190 / 372 | 123, 202 / 467 | 288, 762 / >3000 |
| spray cup, film 6 kg | 122, 192 / 377 | 124, 205 / 473 | 294, 776 / >3000 |
| spray cup, film 33 kg | 126, 202 / 400 | 129, 216 / 503 | 319, 849 / >3000 |
| plug (film in book) | 111, 162 / 272 | 113, 169 / 327 | 218, 514 / 2541 |
| ADR 0033 plate (film in book) | 104, 138 / 198 | 105, 141 / 227 | 176, 363 / 1408 |

The brake adds about $1/kg steady and $1-3 to break-even. The shielded film adds about $2 and
$2-6 more. Unshielded, it adds $6 and $12-36, most of it the lob for 35 t of pitch a push.
Methalox suffers most, since it has fewer cargo kilograms to spread the push over. Pessimistic
book, spray cup: $504 / $559 / $870 steady with the film in the book, and $529 / $588 / $962
with 33 kg carried. The book's 4% was lighter than the sim's unshielded film.

Deliveries (`make plate-seed`): the lob is $44.8-46.1 per cargo kg at $500, against $44-45.
Seed amortization is unchanged, since it does not price the lob.

## Not done

- The plug's film. The sim gives no figure. `PLUG` carries none, and its cost rows keep the
  book's line.
- The brake as a finite burn. Drag and the 4 g cap are left out, as in the rest of `lob_rise`.
- The plug in the deep bowl (impact sim, parent S13).
