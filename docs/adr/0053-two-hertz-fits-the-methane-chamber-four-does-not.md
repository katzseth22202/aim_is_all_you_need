# ADR 0053: Two hertz fits the methane chamber's cycle; four does not

Date: 2026-10-12

Status: Accepted.

## Context

The survivable methane chamber (ADR 0044, parent `tab:one_chamber`) fires a 5 kg rod
at 2 Hz. Its chamber is bulged to 212 m3 at about 94 bar, with a 1.33 m2 throat.
The parent says blowdown fits at 4 Hz and refill does not. It quotes an e-fold time
near 60 ms from `V / (0.6 A* a)` at a = 3.7 km/s. Nothing in this repository checked
that the whole cycle fits in 500 ms. Each pulse has to:

1. blow down the hot gas;
2. lay films over both openings;
3. seat the next plug;
4. fill the warm-gas charge through the liner passages (`sec:methane_regen`);
5. lock 20 ms before the next rod arrives.

A rate also has to be argued, not just assumed, before the paper credits hydrogen with
4 Hz.

## Decision

`chamber_isp.blowdown_time()` implements ADR 0049's quasi-steady adiabatic blowdown.
It assumes a choked throat, an isentropic gas behind it and a discharge coefficient of
one. The peak gas states come from Cantera equilibrium (`gri30`, extrapolated above
5000 K):

| Chamber | Peak | Effective gamma | Sound speed at peak |
|---|---:|---:|---:|
| CH4, 7000 K, 212 m3, 1.33 m2 throat | 94 bar | 1.225 | 4.48 km/s (the parent uses 3.7) |
| H2, 5500 K, 20 m3 sphere, 0.198 m2 throat | 806 bar | 1.194 | 5.66 km/s |

**Effective gamma** is the isentropic exponent of an equilibrium expansion to a
tenth of the density. It counts the heat that recombination returns. The hydrogen
sphere's e-fold time comes out at 31 ms, matching the parent.

**Refill pressure.** Refill must wait until the chamber is near its fill pressure.
For methane that is the ~1 bar of 97 kg of 420 K gas in 212 m3. For hydrogen it is
the 9.5 bar of 60 kg of 77 K gas in 20 m3. Getting there takes about 97.5% of the
peak mass out.

| Chamber | Period | 95% out | To fill pressure | Left for films, plug, fill and lock |
|---|---:|---:|---:|---:|
| CH4 survivable, 2 Hz | 500 ms | 215-260 ms | **278-336 ms** | 164-222 ms |
| CH4 survivable, 4 Hz | 250 ms | 215-260 ms | 278-336 ms | **none** |
| H2 sphere, 4 Hz | 250 ms | 105 ms | **135 ms** | 115 ms |

- **Methane at 2 Hz fits.** 164-222 ms is left once the chamber reaches fill
  pressure. Filling 97 kg in 120 ms needs about 800 kg/s at the injectors. An
  accumulator brings the pump's average down to 194 kg/s.
- **Methane at 4 Hz does not fit.** Blowdown alone overruns the period. That
  explains the parent's statement more firmly than "the refill does not fit".
- **The hydrogen sphere at 4 Hz is tight but plausible.** It starts at 806 bar, so it
  takes nearly as long as methane to reach fill pressure in relative terms.

**Would removing the bulge help?** Blowdown time goes as `V / (A* a)`. Shrinking the
methane chamber at the same throat, charge and temperature:

| Volume | Peak | 95% out | To fill pressure |
|---:|---:|---:|---:|
| 212 m3 (as flown) | 94 bar | 215 ms | 278 ms (1.0 bar) |
| 106 m3 | 175 bar | 111 ms | 142 ms (2.0 bar) |
| 53 m3 | 318 bar | 58 ms | 73 ms (4.0 bar) |

So yes, timing roughly halves with the volume. At 106 m3 the blowdown fits 4 Hz with
about 110 ms to spare, as tight as the hydrogen sphere. At 53 m3, 4 Hz is
comfortable. The wall's mass hardly moves, because `pV` is fixed by the charge (it
falls about 15%). The denser gas also dissociates less, which should help efficiency.

The bulge was not chosen for timing. It is what the companion impact simulation
needed for the wall to survive the blast: the hoop overshoot, the hot taper and the
crack life. Un-bulging is therefore a question for that simulation, not for this
repo.

**What 4 Hz is worth.** For hydrogen slush, 2 Hz instead of 4 Hz drops growth from
2.560 to 2.295, and from 2.877 to 2.597 with recycled tanks (`make slush-hydrogen`).
The steady cost rises from $105 to $116/kg, and the dear-seed break-even from $245 to
$319. A bulged hydrogen chamber, which the parent lists as hydrogen's lever, would
likely land at 2 Hz. Methane's 4 Hz would be worth a similar ~10% in growth.

## Consequences

- The paper can state that methane at 2 Hz is consistent with a blowdown estimate,
  and that 4 Hz is ruled out by blowdown alone, with the table above.
- Quote hydrogen's 4 Hz as the sphere's. A bulged hydrogen chamber should be assumed
  at 2 Hz until its own blowdown is solved.
- Ask the companion simulation whether a 53-106 m3 methane chamber survives. If it
  does, 4 Hz becomes available.

## Limitations

- The blowdown is quasi-steady and adiabatic, with discharge coefficient one.
  Wall heat loss would shorten it, and real flow would lengthen it.
- Gas properties at 7000 K are extrapolated. The parent's 3.7 km/s is kept as the
  slow bound.
- Laying the films and seating the plug are not timed. The fill's injector flow is
  sized, not designed.

## Reproduction

`pytest tests/test_chamber_isp.py` pins the 31 ms hydrogen e-fold time and the
278-336 ms methane blowdown. `make slush-hydrogen` prints the 2 Hz rows.
