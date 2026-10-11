# ADR 0048: Defer the rotating tether as a paper baseline

Date: 2026-10-10

Status: Proposed for review.

## Context

A plate-only growth architecture could replace the head-on departure chamber
with a rotating tether that catches payloads in circular LEO and releases them
toward Jupiter, followed by a conventional methalox burn. The proposed tether
is launched to LEO and raised using solar-electric propulsion (SEP), with solar
arrays charging batteries for concentrated periapsis burns. Momentum must also
be restored after each throw, not just during initial deployment.

The selected example catches 100 tonnes **including stage and propellant**,
with a 3 km/s tip speed and 10 km arm radius. The ideal heavy-hub approximation
gives a 6 km/s payload increment, about 92 g tip acceleration, and a further
2.17 km/s methalox burn for the ordinary 3S departure. These kinematics do not
constitute a validated capture, release, or rendezvous trajectory.

## Proposed paper wording

> A rotating tether could eliminate the head-on departure chamber. However,
> preliminary sizing for 100-tonne payloads and a 3 km/s tip speed indicates
> substantial tether and reboost-system masses. We therefore retain the head-on
> architecture as the modeling baseline, without claiming that its engineering
> feasibility is established or that tether alternatives are ruled out.

## Rationale and limits

The [sizing analysis](../tether_plate_feasibility_2026-10-10.md) records the
derivation, assumptions, sensitivities, and historical tether references.
Its [standalone calculation](../tether_plate_analysis.py) requires no scratch
files or ignored caches.

- At density 970 kg/m^3 and an illustrative **allowable** stress of 1.75 GPa,
  the constant-stress tapered model gives about 6,609 tonnes of fibre for two
  equivalent loaded arms. Hub, counterweight, capture equipment, propulsion,
  and additional engineering margins are excluded. This is a symmetric
  benchmark, not a minimum over all tether architectures.
- Replacing the ideal throw momentum with 2000 s SEP at 50% efficiency takes
  about 3.269 GWh and 30.59 tonnes of propellant. At 200 Wh/kg usable battery
  energy, distributing recovery across ten 90-minute burns still implies
  about 1,634 tonnes of batteries and 218 MW thruster electrical power.
  These are leading-order momentum/energy estimates, not a solved schedule
  restoring both the orbit and spin. Thruster and power-processing mass,
  charging losses, heat rejection, and lifetime remain unpriced.
- Lower tip speed materially reduces fibre mass but increases chemical
  propellant expenditure. Thus this example does not rule out a better tether
  configuration; no equal-throughput cost optimization has been performed.
- The head-on architecture retains unresolved targeting, impact loading,
  chamber-survival, and mission-interface risks. The tether mass penalty is
  **not evidence that those risks are solved**, or proof of comparative
  engineering practicality.

## Decision for review

Retain ADR 0047's preferred tested 3S configuration as the modeling baseline.
Discuss the tether as an alternative with a substantial, explicitly charged
seed-infrastructure burden. Do not claim a cheaper tether seed, a demonstrated
$200/kg program, universal tether impracticality, or a validated head-on rocket.

This record proposes wording for the paper; it does not modify the separate
paper repository or change production model defaults.

## Verification

The full repository suite completed immediately before this documentation
commit: **877 passed, 1 skipped**, with 23 warnings. Production source, tests,
and dependency configuration are unchanged since that run. The standalone
tether calculation is rerun for this record; formatting and `make test-changed`
are checked separately. The latter selects no affected tests for these docs
files. Passing calculations and repository tests do not validate tether or
chamber engineering.
