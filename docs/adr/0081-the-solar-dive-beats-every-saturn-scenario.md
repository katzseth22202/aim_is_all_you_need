# ADR 0081: The solar dive beats every Saturn scenario, with no nuclear approvals

Date: 2026-10-21

Status: Accepted. Records the author's headline for the parent paper (2026-10-21): "on
every scenario solar dive is better, and no need for nuclear approvals." It closes the S51
work (ADRs 0078-0080 and the ADR 0079 addendum), subject to the queued head-on-impactor
Saturn cycle, which has not been run.

## Context

S51 asked for a Saturn water cycle to be priced against the paper's current route. Three
ADRs and an addendum priced it in every variant raised:

- **ADR 0078:** the single-use water NTR at Iapetus.
- **ADR 0079:**
  - a kept reactor making methalox at Iapetus or Phoebe;
  - the real arcs and planes;
  - a second Earth-departure scenario, with no nuclear near Earth.
- **ADR 0079 addendum:** the reactor and the Earth plates reused until nearly free.
- **ADR 0080:** aerocapture at Saturn, rings cleared.

Nuclear electric propulsion was raised and dropped before it was modelled (ADR 0080,
Context).

The solar dive's figures to the eccentric orbit were available only at 10% (ADR 0078). The
Saturn ledgers also print a 4% loan. So `eccentric_report` (`make solar-dive
ARGS=--eccentric`, ~30 s) now prints the dear-seed design at 4% as well: $105 at 9.86
R_sun, $76 at 4 R_sun.

## Decision

The paper's headline is the solar dive. The best Saturn cell in each scenario, $/kg to the
eccentric orbit:

| scenario | rNPV 10% | 4% loan | steady |
|---|---|---|---|
| base, H2 NTP (methalox at Phoebe, coplanar) | $308 | $129 | $81 |
| base, no nuclear near Earth | $338 | $129 | $84 |
| reuse, H2 NTP (Phoebe, 0.1 yr stay) | $206 | $99 | $72 |
| reuse, no nuclear near Earth | $235 | $103 | $73 |
| aerocapture, H2 NTP (water NTR at Iapetus, coplanar) | $172 | $100 | $79 |
| aerocapture, no nuclear near Earth | $223 | $108 | $80 |
| **solar dive 9.86 / 4 R_sun (dear seed)** | **$143 / $86** | $105 / $76 | $93 / $72 |

- **On the gated rNPV at 10%, the solar dive wins every scenario at both depths.** The
  closest Saturn cell is 1.2x the shallow dive and 2.0x the deep one.
- **The deep dive wins on every metric.** The shallow dive does not:
  - On the 4% loan the best coplanar or reuse cells tie or edge it ($99-100 against
    $105). On the real planes, aerocapture is $106.
  - On steady cost the Saturn cells undercut it ($72-80 against $93).
  - The paper should state the comparison on the rNPV, or against the deep dive.
- **No nuclear approvals.** Every Saturn variant flies a fission reactor: the water NTR,
  the outbound hydrogen NTP, or the methalox plant. The no-nuclear-near-Earth scenario
  still launches reactors cold as cargo. The solar dive flies PuffSats and methalox only.
- **Why: distance, not hardware.** The 7.7-9.5 year round puts first cargo at year 24-30.
  Every kilogram also rides the same lofting floor (lob plus plug, about $49 of the
  steady cost). Reuse cuts the hardware lines and aerocapture cuts the outbound stack;
  neither touches the round or the floor.

## Consequences

- The parent-facing record is `docs/iapetus_cycle_answers_for_parent.md`. Its first
  section is the headline, the table and suggested paper text.
- Not scenarios:
  - **Bounds that zero every price but the lob** ($71-95 rNPV with H2 NTP). They need the
    single-use nuclear engines free.
  - **The queued head-on-impactor Saturn cycle** (overtake push to the eccentric orbit,
    head-on push onto the Saturn arc about 20 days later, aerocapture for all but the
    seed). The author expects it to be the cheapest long-term architecture. If it is run,
    this headline must be re-checked against it.
