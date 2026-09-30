# The walled chamber departs with its hardware charged and expended

Status: accepted

Date: 2026-09-30

## Context

The parent paper (`Balloon-Pulse-Propulsion`) prices two walled head-on
chambers for the departure burn: hydrogen at 5500 K on bare copper, and methane
at 7000 K on steel lined with pitch (`tab:wall_pairings`), each fired by a 2.5 kg
rod at 75 km/s. Its figures come from `eq:eta_isp`,

    v_e = w sqrt(eta / (k + 1 + P)),   I_eff = ((k + 1 + P) v_e - w) / ((k + P) g0),

and its doubling times (`tab:wall_pairing_doubling`) from the circular
synodic-lock model with tanks charged and the chamber recovered for free. The
child repository's growth ledger needs the same chambers on the **real 11-cycle
chain**, with the hardware charged. Nothing in this repository implemented
`eq:eta_isp`, and the parent's own scratch notes said its gated methane range
(762-777 s) could not be reproduced from the paper text.

## Decision

**1. `eq:eta_isp` as written, plus two readings that reproduce every gated row.**
`src/chamber_isp.py` reproduces each cell of `tab:nozzle_area_ratio`, and all of
the gated `I_eff` in `tab:wall_pairing_doubling`, once:

- the port gate's 1% comes off the **exhaust momentum** `(k+1+P) v_e`, before the
  head-on debit `-w` (about 1.3% of Isp). That gives hydrogen 1112 s and 922 s;
  taking 1% off the finished Isp gives 1114/924, which the tests reject.
- methane's lost pitch (1.4-5.6 kg per pulse) is **carried propellant at the same
  eta**, joining `k + 1 + P` in the exhaust and `k + P` in the carried mass. That
  gives 762-777 s (A/A* 300, carbon in equilibrium), 711-725 (frozen) and 612-623
  (A/A* 14).

Cells are checked against the band their own rounded inputs span (eta printed to
3 dp, k to 1 dp), plus the half second of output rounding; raw residuals reach
about 1 s.

**2. The slug ratio is per pulse.** The closing speed changes pulse to pulse as
the craft accelerates into the departure wave. The rod and plug stay fixed and
the gas load varies, holding the chamber's energy per kilogram `u(T)`:
`k(w) = (k_ref + 1 + P)(w / 75 km/s)^2 - 1 - P`. A pulse too slow to heat rod and
plug to temperature at any charge is refused (hydrogen needs about 24 km/s).
Eta is held at its 75 km/s value across speeds, an assumption: the companion
simulation solved it only there. The departure burn is integrated with `I_eff(w)`
recomputed every pulse (`chamber_departure_burn`), and returns the rod count.

**3. Efficiency is quoted as a share of the chemistry ceiling.** The ceiling is
the most of the pulse any chamber of that pairing could turn into directed
exhaust: one minus the energy held in bonds at peak that the A/A* = 300 nozzle
never returns, even in equilibrium (`ChamberPairing.chemistry_ceiling`,
`absolute_efficiency`):

| Pairing | Bonds at peak | Returned at A/A* 300 | Ceiling | Solved eta | Share |
|---|---|---|---|---|---|
| H2 5500 K | 36% | 94% | 0.978 | 0.858 | 88% |
| CH4 7000 K | 69% | 52% (carbon never re-bonds) | 0.669 | 0.538 | 80% |

The A/A* = 1000 stretch nozzle lifts methane to 0.683. Quoting absolute eta was
too generous to methane: "50%" is 0.33 for methane but 0.49 for hydrogen.

**4. The finite-burn loss is integrated, not scaled.** `src/finite_burn_loss.py`
integrates a planar two-body burn centred on the 600 km periapsis of the 20-day
orbit, at constant mass flow, either steered along the velocity or held in the
one fixed direction that loses least. The loss is the extra ideal burn needed to
reach the impulsive burn's orbital energy. The chamber must point along the
arriving stream, so it pays the fixed-direction loss. Cached tables cover burns
of 4-10 km/s over 0-3600 s.

**5. The hardware is charged, and the departure wave's dry mass is expended.**
`src/chamber_departure.py` charges tanks on the gas only (0.205 for hydrogen,
0.034 for methane; plug and pitch ride untanked), the chamber wall (32 t
hydrogen, 19 t methane; `sec:steel_chamber_service`), and a nozzle extension of
2.19 t (the RL10B-2's 131 kg scaled by (8.7 m / 2.13 m)^2). These ride the burn
and are never payload. Each launch unit brings a fresh set, and the set leaves
with the stack: **expended every cycle** (decided with the user). Retrieving the
chambers for reuse -- the parent's plan, PuffSats braking them back into the
parking orbit -- would free that mass at the cost of the braking PuffSats. It is
upside the ledger does not take, not a hidden cost.

**6. The chamber count is solved, not assumed.** More chambers shorten the burn
and its loss but add 21-34 t each. `best_departure` walks up from one chamber
until the net delivered mass has fallen twice.

## What the numbers say

- The integrated loss reproduces the parent: steered, 21.3/223.8/482.7 m/s
  against 22/226/486 (320/1200/2100 s, the reactor's 5.39 km/s at 906 s); fixed,
  26.4/98.0 against 27/101 (320/640 s, 5.4 km/s).
- **The loss follows the burn, not the exhaust speed.** Under 1% separates 800 s
  from 1431 s, while a 7.0 km/s two-synodic burn loses about 30% more than a
  5.4 km/s three-synodic one at any length. The parent's single 27 m/s is a 3S
  figure. Past 640 s the parent's `t^2` scaling overstates 3S losses (1000 s:
  213 against 264 m/s).
- **Lost efficiency bites the head-on chamber harder than an overtaking plate.**
  The rod's momentum `w` is debited in full at any eta, while the exhaust's falls
  as `sqrt(eta)`, so the debit's share of gross momentum is `1/sqrt(eta(k+1+P))`:
  19% at eta 1 and 27% at 0.5 for hydrogen. Isp at eta 0.5 is 0.637 (H2) / 0.642
  (CH4) of its eta-1 value, below `sqrt(0.5)` = 0.707; the plate keeps 0.779. This
  is the parent's `eq:isp_coupling` point, now a tested quantity
  (`momentum_debit_share`).
- **At equal absolute eta the gas barely matters**, because the rod supplies the
  energy: gated, at 70 km/s and eta 1, hydrogen 1209 s against methane 1149 s. Then the
  tanks (0.205 against 0.034) and wall (32 against 19 t) decide, and methane wins.
  At an equal *share of ceiling* hydrogen's higher ceiling restores its lead (ADR
  0033), which is why decision 3 matters.
- A chamber is 21% (CH4) / 34% (H2) of a 100 t stack: it wants a big one. At the
  1000 t placeholder stack the best count is 1-2, with burns of about
  800-2000 s losing 150-900 m/s -- the same trade the parent's reactor makes
  (42-53 minute burns, 600-900 m/s). On the growth ledger's ~500-600 t departing
  stacks the best count is one.

## Consequences

- `ChamberPairing` carries its hardware and ceiling, so every caller prices the
  same chamber. The parent's `tab:h2_breakdown` growth equation
  (`push x share x net`) is pinned in `growth_per_cycle` (2.51 per cycle, 1.646 yr).
- `make chamber-departure` prices the chambers over the flown chain at a 1000 t
  placeholder stack and the parent's 8.43 push ratio. It is superseded for
  headline figures by `make growth-ledger` (ADR 0033), which derives both.

## Considered and rejected

- **Scaling the parent's 27 m/s as `t^2`.** Right to about 640 s on a 5.4 km/s
  burn, but it ignores the burn's size, and the optimum burns run past it.
- **One mean Isp for the burn.** The closing speed rises through the burn and `k`
  follows it; the integral is cheap.
- **Retrieving the chamber for free**, as the parent's lock model does. The ledger
  must charge the chamber somewhere; expending it is the pessimistic, simple case.

## Reproduction

`pytest tests/test_chamber_isp.py tests/test_finite_burn_loss.py
tests/test_chamber_departure.py` (the slow tests hold the table-accuracy checks
and the real chain). `make chamber-departure`. Loss-table nodes: burns
`arange(4, 10.01, 1)` km/s by times `arange(0, 3600.01, 200)` s, built at an
exhaust speed of 11 km/s (the loss is insensitive to it), cubic interpolation.
Trapezoid burn integral on 401 nodes. Chamber-count search ceiling 200.
