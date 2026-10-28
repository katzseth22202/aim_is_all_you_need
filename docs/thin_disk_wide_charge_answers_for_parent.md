# The thin 1 kg disk on a wider charge, and 4 R_sun: aim's rerun for S62 (for the parent)

Raised 2026-10-28 in `aim_is_all_you_need` (ADR 0087). **Written to be copied into
`Balloon-Pulse-Propulsion`.** It closes the aim side of S62. The simulation's side is Result 24
(`puffsat_impact_simulation` @ `1eb4115`, `docs/thin_disk_wide_charge_answers_for_parent.md`
there). Reproduce with `make disk-train-aim` (~70 s).

## The short answer

**Adopt the 1 kg disk on a charge 1.2x its radius. It breaks even at $129 / $156 at Parker's
depth and $86 / $89 at 4 R_sun, flown at centimetres with the ship holding its track.**

That is the printed $131 / $158 and $85 / $89 within about two dollars. The difference is that
the new figures now carry the aim: the 1.36 cm dispersion, the track hold and the overhang loss.

| Train (dear-seed optimum) | 9.86 R_sun, X 60: BE cheap / dear | 4 R_sun, X 90: BE cheap / dear |
|---|---|---|
| Printed today (ADR 0085: matched, centred, no hold) | $130.7 / $158.2 | $85.5 / $88.7 |
| Matched, flown at the real aim | $142.6 / $181.2 | $89.4 / $94.1 |
| **1.2x charge, flown at the real aim** | **$129.5 / $156.1** | **$85.6 / $89.1** |
| 1.3x charge, measured (Result 24) | $133.3 / $163.4 | $86.9 / $90.8 |
| 1.3x charge, ADR 0086's assumed inputs | $128.9 / $155.0 | — |

## What changed from ADR 0086

- **x1.3 was not what it seemed.** Result 23's 1.08x for x1.3 was really a x1.25 charge.
  Resolved, x1.3 couples 1.047x and costs $163 dear, not $155.
- **x1.2 is the best width.** It couples 1.074x through the unchanged 0.25 m bore.
  - Clearance: 5.05 cm. Required dispersion: 1.36 cm an axis, at one clip in a thousand.
  - The disk is caught whole to 4.0 cm. Beyond that the overhang goes unstruck, which aim now
    charges (5e-5 of the coupling at 1.36 cm).
- **The side-impulse slope was right.** It is 0.136 per metre on any wide charge, the 13.5% ADR
  0086 borrowed. The matched charge's slope is 0.072 per metre, measured.
- **4 R_sun carries every ratio unchanged** (Result 24). With the same charges, x1.2 costs $89.1
  dear against the printed $88.7.

## Aim figures on x1.2 (for `sec:sun_wave_ranging`)

| | 9.86 R_sun | 4 R_sun |
|---|---|---|
| tolerance, one axis | 1.36 cm | 1.36 cm |
| side impulse at that dispersion | 0.18% of the axial | same |
| hold on the departure ship, side + torque | 23 + 13 m/s, 6.0 t | 31 + 18 m/s, 9.9 t |
| hold on the push-wave ship | 11.7 t | 18.8 t |
| loop gain at a 60 s blind window | 75 (unstable) | 87 |
| kick knowledge the ship needs | 0.055 mm/s per pulse | same |
| disk thrust (final move) | 1.56 mN | — |

The CME budget is unchanged from ADR 0086 for the 1 kg disk. An extreme 23° sheath still makes
those disks deliberate misses.

## Suggested text

For the harder-boost paragraph, replacing the sentence the simulation quoted ("A charge 1.3 times
the disk's radius ... not been simulated closely enough to adopt the figure."):

> A charge 1.2 times the disk's radius couples 7\% more and catches the disk whole to
> \SI{4}{\centi\meter} off center, through the same \SI{0.25}{\meter} bore
> \cite{Katz_puffsat_impact_sim_2026}. Flown at that aim, with the ship cancelling each pulse's
> side kick, it breaks even at \$129 and \$156 here and \$86 and \$89 at \SI{4}{R_\odot}.

Places the $158 → $156 change reaches: the 8 places at Parker's depth and the sensitivities that
hang off $158, as S62 lists them ($216/$146 stroke, $178 carbon fabric, $143 stacking, the Saturn
caption). Aim has not rerun those sensitivities on the x1.2 charge. Each is a delta on $158, so
shifting them by -$2 is a reasonable first reading, not a rerun. $85 / $89 at 4 R_sun stands
($85.6 / $89.1).

## Caveats

- At 4 R_sun the network's 1.3 cm error is the parent's figure for the 0.99 h train, applied to
  a 1.10 h train. The CME budget there is not run.
- Off-centre strikes are axisymmetric geometry plus a tracer, not 3-D runs. Past 4.0 cm the
  overhang loss is an upper bound.
