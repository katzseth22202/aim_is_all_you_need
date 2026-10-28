# ADR 0087: The thin 1 kg disk flies on a 1.2x charge, at $156 here and $89 at 4 R_sun

Date: 2026-10-28

Status: Accepted. It answers the parent's S62 on the aim side, using the impact simulation's
Result 24 (`puffsat_impact_simulation` @ `1eb4115`, `docs/thin_disk_wide_charge_answers_for_parent.md`,
ADR-0075 there). It replaces ADR 0086's assumed x1.3 inputs and its proposed $129 / $155.
Reproduce with `make disk-train-aim` (~70 s; `ARGS="--no-deep"` skips the 4 R_sun rows).

## Context

ADR 0086 proposed the 1 kg, 2 Hz disk on a charge 1.3x its radius at $129 / $155 (9.86 R_sun,
X 60, cheap / dear seed). Three of its inputs were not measured for this disk:

- the coupling, 1.08x, from the 0.025 m grid only;
- the bore clearance, taken as 5 cm;
- the side-impulse slope, borrowed from the 2.5 kg disk on x1.3 (13.5% per metre).

The parent held the figure and asked the simulation for all three, the best width, and the
same at 4 R_sun (S62). Result 24 answered:

- **A run resolves a charge only to whole cells.** Result 23's "x1.3, 1.08x" was a x1.25 charge
  on both grids. On the 0.0125 m grid, x1.3 (resolved x1.31) couples **1.047x**.
- **The best width is x1.2** (resolved x1.19), at **1.074x**. It keeps the 0.25 m bore, so the
  clearance is **5.05 cm**. The disk is caught whole to 4.0 cm. At the bore's edge the lone-lens
  bound keeps 1.057x.
- **The side impulse belongs to the bowl.** It is 0.136 per metre on any wide charge, for both
  disks, linear to 3% from 3 cm out. On the matched charge it is 0.072 per metre, since the burst
  sits at the struck lens's centroid.
- **4 R_sun changes no ratio.** The 2-D gas has no speed scale.

## Decision

1. **Result 24's inputs replace the assumed ones** (`THIN_MATCHED`, `THIN_X12`, `THIN_X13`).
   - The matched charge's clearance is `0.25 - 0.1995` m.
   - The matched slope is the measured 0.0717 per metre, not the 2.5 kg disk's 0.070.
   - ADR 0086's x1.3 stays as `ASSUMED_X13`, a comparison row that still reproduces $155.0.
2. **The overhang is charged.** On x1.2 the disk is whole only to `R_c - R` = 4.0 cm, short of
   the 5.05 cm clip.
   - Past 4.0 cm the unstruck overhang is the circle overlap (`struck_share`). It reproduces
     the simulation's 0.98489 at 5 cm.
   - The loss is averaged over the 2-D Gaussian out to the bore (`mean_coupling_loss`).
   - At the required 1.36 cm it is 5e-5, so it is negligible.
3. **The closure is a parameter.** The 1 kg trains are priced again at ADR 0085's deep closure,
   `DEEP_CLOSURE` = (4 R_sun, X 90), with the same charges.
   - Drift, the CME budget and the thrusters stay at 9.86 R_sun only, because the simulation's
     CME cases were run there.
4. **The headline is the 1 kg disk on x1.2, flown at centimetres with the ship holding its
   track.**

| 9.86 R_sun, X 60 | arc (h) | Sun plate + bag dep / push (t) | hold (t) | growth | BE cheap / dear |
|---|---|---|---|---|---|
| ADR 0085, matched, centred, no hold | 0.986 | 15.9 / 22.8 | 0 / 0 | 3.226 | 130.7 / 158.2 |
| matched, flown at 1.36 cm | 1.032 | 15.9 / 22.6 | 4.2 / 8.1 | 3.035 | 142.6 / 181.2 |
| **x1.2 (Result 24)** | 0.928 | 16.0 / 23.1 | 6.0 / 11.7 | **3.243** | **129.5 / 156.1** |
| x1.3 (Result 24) | 0.950 | 16.0 / 23.0 | 7.1 / 13.8 | 3.172 | 133.3 / 163.4 |
| x1.3 assumed (ADR 0086) | 0.923 | 16.0 / 23.1 | 5.9 / 11.5 | 3.255 | 128.9 / 155.0 |

| 4 R_sun, X 90 | arc (h) | Sun plate + bag dep / push (t) | hold (t) | growth | BE cheap / dear |
|---|---|---|---|---|---|
| ADR 0085, matched, centred, no hold | 1.171 | 28.8 / 40.9 | 0 / 0 | 4.880 | 85.5 / 88.7 |
| matched, flown at 1.36 cm | 1.223 | 28.8 / 40.6 | 6.9 / 13.1 | 4.562 | 89.4 / 94.1 |
| **x1.2 (Result 24)** | 1.101 | 28.9 / 41.3 | 9.9 / 18.8 | **4.844** | **85.6 / 89.1** |
| x1.3 (Result 24) | 1.126 | 28.9 / 41.1 | 11.7 / 22.2 | 4.729 | 86.9 / 90.8 |

The 2.5 kg rows of ADR 0086 are unchanged ($150 / $195 on x1.3).

- **The measured x1.2 lands $1.1 dear above ADR 0086's assumed x1.3.** That is within the
  simulation's "about a dollar".
- **The honest headline is the old printed figure, give or take a dollar.** The x1.2 charge buys
  back the whole cost of aiming: the matched charge's 5.4% lens loss and the hold.
  - 9.86 R_sun: $158.2 → $156.1.
  - 4 R_sun: $88.7 → $89.1.
- **Measured x1.3 does not pay.** It couples 2.6% less than x1.2. Its wider clearance lets the
  dispersion grow to 1.61 cm, and the hold, charged at the required dispersion, grows with it.
  Flown at x1.2's dispersion it would still lose on coupling.
- **The aim figures for x1.2** (9.86 R_sun):
  - tolerance: 1.36 cm an axis;
  - side impulse at that dispersion: 0.18%;
  - hold: 23 + 13 m/s on the departure ship (31 + 18 m/s at 4 R_sun);
  - kick knowledge: 0.055 mm/s, still the open requirement;
  - disk thrusters: 1.56 mN, set by the final move;
  - the 1 Hz short-loop alternative would need 41 mN disks.

## Consequences

- `src/disk_train_aim.py` gains `THIN_X12`, `THIN_X13` (measured), `ASSUMED_X13`, `struck_share`,
  the overhang in `mean_coupling_loss`, a `closure` argument on `ledger_design`, `fly_train`,
  `price` and `priced_trains`, and `DEEP_CLOSURE` / `DEEP_DISKS`.
- ADR 0086's matched-charge rows move by under $0.5: $142.3 / $180.8 → $142.6 / $181.2, from the
  measured 5.05 cm clearance and 0.0717 per metre slope.
- `HEADLINE_STARTS` and the ledger's defaults are unchanged. The x1.2 headline is
  `Design(sun_coupling=1.074, sun_hold=...)` on top of them, as `disk_train_aim.price` builds it.

## Assumptions to revisit

- **4 R_sun reuses 9.86 R_sun inputs.** The network's 1.3 cm sideways error is the parent's
  figure for the 0.99 h train, while the 4 R_sun train is 1.10 h. The CME budget at 4 R_sun is
  not run.
- **The design width is nominal.** The finest grid resolved x1.2 as x1.19 and x1.3 as x1.31.
- **Off-centre strikes are axisymmetric geometry plus a tracer,** not 3-D runs (Result 24's
  caveat). Past 4.0 cm the lone-lens bound understates the struck impulse, so the overhang loss
  here is an upper bound.
