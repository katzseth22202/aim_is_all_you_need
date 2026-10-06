# Spray-cup film and the booster's brake, for the parent paper

Raised 2026-10-06 in `aim_is_all_you_need` (ADR 0043). It answers the parent's S12 and the
brake `sec:vertical_lob` names as unrecomputed. It supersedes the lob charge in
`docs/plate_grid_for_parent.md` (x1.044 becomes x1.065) and moves its cost cells by $1-3.
Reproduce with `make lob-brake` (seconds), `make plate-film` (~45 min) and `make plate-cost`
(~20 min).

## The brake at the faster climb (`sec:vertical_lob`)

The brake now comes out of the lob. The booster brakes after separation so that it crosses
60 km at 1.5 km/s (~9 g), then lands on 0.3 km/s. A vertical entry's peak g depends on its
entry speed alone, so the criterion is the same at any climb. At 0.75 km/s this gives the
paper's numbers: a 1.35-1.39 km/s brake, 143-155 t reserved, and 10.5% more lofted without it.

| climb at 400 km | separates at | brake | reserve | lob charge (380 / 350 s) |
|---|---|---|---|---|
| 0.75 km/s | 133 km, 2.32 km/s | 1.39 km/s | 144 t | 1 |
| 1.0 km/s | 138 km | 1.52 km/s | 157 t | 1.043 / 1.049 |
| **1.1 km/s** | 140 km, 2.43 km/s | **1.58 km/s** | **164 t** | **1.065** / 1.074 |
| 1.2 km/s | 143 km | 1.64 km/s | 171 t | 1.089 / 1.101 |

Unbraked, the booster would cross 60 km at 2.72 km/s, against 2.60 at 0.75 km/s.

Suggested replacement for the last sentences of the faster-climb paragraph: reaching 400 km at
1.1 km/s costs 6.5% of the lofted mass at 380 s, and 4 to 10% across 1.0 to 1.2 km/s and
350-380 s, brake included. The booster separates higher and faster, near 140 km at 2.4 km/s,
so its brake grows from 1.39 to 1.58 km/s and its reserve from 144 to 164 t. The lifting rocket
grows to about 1.15 to 1.32 times Super Heavy, and the $25 becomes $26.6.

## The film as launched mass (S12)

The film burns pulse by pulse out of the 1500 t unit. A push is ~1050 pulses of 12 MN s, not
1200-1500, because the unit's mass falls to ~0.65 through it.

| film per pulse | t per push | H2 solved yr | CH4 solved yr | methalox yr |
|---|---|---|---|---|
| not carried | 0 | 2.02 | 2.28 | 7.82 |
| 4-6 kg (vapor-shielded) | 4.2-6.4 | 2.02-2.03 | 2.28-2.29 | 7.89-7.92 |
| 28-33 kg (unshielded) | 29-35 | 2.05-2.06 | 2.32-2.33 | 8.31-8.41 |

## Cost behind the spray cup (Estimate, 10%, 50% odds, lob x1.065)

| film | H2 solved: steady, BE cheap / dear | CH4 solved | methalox |
|---|---|---|---|
| in the book (0.1%) | $120, $190 / $372 | $123, $202 / $467 | $288, $762 / >3000 |
| 6 kg carried | $122, $192 / $377 | $124, $205 / $473 | $294, $776 / >3000 |
| 33 kg carried | $126, $202 / $400 | $129, $216 / $503 | $319, $849 / >3000 |

Plug (film in the book): $111, $162 / $272 (H2); $113, $169 / $327 (CH4); $218, $514 / $2541
(methalox). Pessimistic book behind the spray cup: $504 / $559 / $870, or $529 / $588 / $962
with 33 kg carried. Delivery lob: $44.8-46.1 per cargo kg at $500.

The cost book's film line now holds the film's mass where the plate carries it. It should read
4-8% / 30-45% of PuffSat mass (6.4 t / 35 t a push), not 0.1% / 4%.

## Not done

The plug's film (no sim figure yet), the brake as a finite burn with drag, the plug in the deep
bowl (S13).
