# ADR 0047: Prefer ordinary 3S among the tested hydrogen follow-ups

Date: 2026-10-10

Status: Accepted as the working comparison baseline, not a global optimum.

## Decision

Keep the ordinary 3S architecture with recovered departure hardware (ADR 0046)
as the preferred tested baseline. Raising hydrogen temperature to 3500 K,
advancing the growth wave, or selecting 2S whenever the correction is below
50 m/s did not establish a cheaper, closed 3000 K architecture.

The recovery assumption is **90% of tanks and 60% of chamber/nozzle**, not 75%.
The dear-seed figures below are program break-even L1 sale prices, not seed
transport prices or operating costs.

| Tested case | Dear-seed break-even, $/kg |
|---|---:|
| Ordinary 3S, 3000 K, recovered hardware | 269.55 |
| Ordinary 3S, 3500 K, retuned launch setting | 251.27 |
| Early growth wave, actual seed routes, existing storage loss | 821.02 |
| Early growth wave, actual seed routes, optimistic 6% storage loss | 455.15 |
| Adaptive 2S/3S: later-wave correction below 50 m/s | 489.13 |
| Adaptive 2S/3S: both corrections below 50 m/s | 401.30 |

The earlier 5500 K ordinary-3S recovery result remains $181.89/kg (ADR 0046).
The only early-wave sensitivity below $200 grants free seed transport and
optimistic storage: $195.47/kg. That is not an achieved design.

## Limitations and evidence

The early 2.736S return is 68.617 km/s at 200 km, requiring departure excess
speed 13.831 km/s. Its 105-day two-body parking ellipse extends outside Earth's
Hill sphere. Independent split-stack departures also do not establish a shared
parking orientation. Adaptive studies optimize a common launch setting, not
every cycle independently; the sampled ledger is repeated for valuation.

The [follow-up record](../hydrogen_followups_2026-10-10.md) documents assumptions,
search settings, and reproducible selected-design financial/trajectory checks.
No production defaults change. A tether/plate-only concept is a new architecture
to investigate separately, not evidence reversing this comparison.
