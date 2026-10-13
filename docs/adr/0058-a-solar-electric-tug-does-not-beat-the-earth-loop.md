# ADR 0058: A solar-electric tug for the seed ship does not beat the Earth loop

Date: 2026-10-13

Status: Accepted (negative result). Builds on ADR 0057 (the seed's prices) and
ADR 0039 (the Earth loop, and why SEP was out of scope there).

## Context

The author proposed (2026-10-13) a way to cut the seed's cost by accepting a
slower start:
- spiral out of low orbit on solar-electric propulsion;
- drop perigee back to Earth;
- burn about 5.5 km/s at perigee instead of 8.50 km/s from a 200 km circle.

The arrays would survive the radiation belts over a few months, or be shielded
as they are at Jupiter. A ship starting from a high ellipse's perigee sends
three to four times the seed.

**The catch.** The ship's 1200 t of methalox has to be up there too. If the ship
climbs chemically, the apogee raise costs about 3.2 km/s. That burn plus the
perigee burn add up to the 8.5 km/s it replaces, because the burn's Oberth
leverage depends on where it fires, not on how it is split. So the tug must lift
the **fuelled** ship: about 1500 t plus its own argon and hardware.

## Decision

**Keep EEJ as the headline seed. Do not fly a tug.** The trade is recorded in
`src/seed_sep_tug.py` (`make seed-sep-tug`).

## The model

Each case is tested by ADR 0039's rule: the tug's seed per dollar over the
direct route's (`k`), discounted by the wait for the next Jupiter window,
`k × 1.1^-dt`.

- **Circular spiral (the first proposal).** Edelbaum's `v_LEO - v_circ` out to
  the 20-day orbit's apogee is 6.98 km/s, and dropping perigee there costs
  0.69 km/s more. That is 7.67 km/s of argon: the slow spiral spends its thrust
  where the speed is low and gives up the Oberth effect. On the 40 t ship it
  sends 328 t, but it needs about 900 t of argon and 12.6 MW for a one-year climb.
  At $50/W it loses outright.
- **Perigee pump (the better version).** Thrusting tangentially within ±θ of
  perigee raises apogee and keeps perigee low. A 30° arc costs only 2% above the
  impulsive apogee raise. But the duty cycle collapses as the orbit grows: near
  the 20-day orbit a perigee arc is about an hour out of every 20 days. So the
  tug hands the ship over on an ellipse with `a_f` = 15 000-40 000 km, and the
  ship's Oberth burn (6.3-7.1 km/s) does the rest.
- **The bill.**
  - Hardware: ADR 0026's 15 kg/kW, bought by the watt.
  - Argon: 2000 s at efficiency 0.5, with 15% tankage.
  - Launch: everything lofted at $150/kg.
  - Power: sized for the climb time, with thrust only in sunlight (0.87, the
    spiral's noon-midnight eclipse average) and the array at 0.9 of its rating
    in the belts.
  - Waiting methalox boils off.
  - Afterwards the tug goes on to the Sun-Earth L1 and **its array is sold
    there**. The credit is half the hardware price, at 0.8 of rating, plus
    $500/kg shipping on 7 kg/kW, received 0.25 yr after the climb and discounted
    at 10%.

## Results (40 t ship, $5M hull, $150/kg; best on the grid θ ∈ 30-120°, `a_f` ∈ 15 000-160 000 km, climb 0.5-3 yr)

| Boil-off | $/W | Array sold at L1 | `k` | Delay (yr) | **k × 1.1^-dt** | Seed (t) | MW | Argon (t) |
|---|---|---|---|---|---|---|---|---|
| none | 10 | no | 1.24 | 2.18 | 1.01 | 204 | 7.4 | 255 |
| none | 10 | yes | 1.42 | 2.18 | **1.15** | 204 | 7.4 | 255 |
| none | 25 | yes | 1.17 | 2.18 | 0.95 | 168 | 4.1 | 180 |
| none | 50 | yes | 1.08 | 3.28 | 0.79 | 168 | 2.7 | 177 |
| none | 200 | yes | 0.60 | 3.28 | 0.44 | 134 | 1.7 | 190 |
| 0.05%/day | 10 | yes | 1.02 | 1.09 | 0.92 | 164 | 15.5 | 269 |
| 0.05%/day | 50 | yes | 0.58 | 1.09 | 0.53 | 135 | 8.3 | 183 |

The direct ship sends 92 t. EEJ gives **1.16×** on the same ship (ADR 0039) with
no new hardware. ADR 0059's two-year loop, measured on the chain's return floor
with the hull carried, gives 1.22×, so the comparison only gets harder for the
tug.

## Consequences

- **The tug pays only if three things hold together:** hardware at about
  $10/W, no boil-off, and a buyer at L1 for the array. Even then it merely
  matches EEJ. It could also fly EEJ afterwards, but that compounds an
  unvalidated trajectory model with an unsourced price.
- **Boil-off, not radiation, is the binding constraint.** A one- to two-year
  climb holds 1200 t of methalox in Earth orbit. At 0.05% a day (a
  well-insulated tank, assumed) it loses 17-31% of the load, and every row
  loses.
- **Reopen it if** argon Hall hardware is shown at about $10/W or less
  (Starlink-class buses are the candidate; no source is in hand), *and*
  methalox can be held for a year at under about 0.01% a day.

## Limits

The model is orbit-averaged and coplanar. It has no drag, no lunar or solar
perturbation and no plane change. The eclipse factor is a constant. The prices
per watt below $50, the L1 credit's terms and the derating are assumptions, not
sources. The grid's best rows sit at its 15 000 km edge for the dearer prices,
where the tug does little and the ship does almost everything. The answer there
is "do not fly a tug", which a finer grid would not change.

## Reproduction

`make seed-sep-tug` (about 40 s). Tests: `tests/test_seed_sep_tug.py`. The fast
tests pin the spiral, the pump and the best case. The slow test asserts that
every case above $10/W or with boil-off loses.
