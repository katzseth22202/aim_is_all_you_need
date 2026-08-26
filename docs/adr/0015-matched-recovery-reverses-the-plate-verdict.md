# Compared at matched recovery, the two-leg nozzle wins, and f = 0.8 is why 0014 said otherwise

Status: accepted. Its **verdict survives on new grounds**; its recombination
escape and its matched-pair method are both retired by ADR 0016. See the two
amendments below -- the conclusion was right for the wrong reason.

Date: 2026-08-20

Supersedes the **verdict** of ADR 0014, not its arithmetic. Every number in 0014
reproduces; the sweep was re-run and matched to all published digits. What changes
is the baseline the comparison is made against.

## Context

ADR 0014 concluded "the plate stays" because a two-leg nozzle needs `e1 ~ 0.6` to
draw level with a pusher plate at `f = 0.8`. That is true and remains true. The
problem is `f = 0.8` itself.

`STD_FUDGE_FACTOR = 0.8` comes from a hydrodynamic sweep across **3.2-16 km/s**
(`puffsat_impact_simulation`). The growth push runs from **56.53 km/s down to
45.58 km/s**. That is roughly four times the top of the validated envelope and
about nineteen times the specific energy. `f` is a restitution coefficient, an
elasticity claim about gas rebounding off a solid surface, and nothing rebounds
elastically at 46 km/s. ADR 0014 therefore benchmarked a candidate device against
an incumbent holding the least defensible number in the repo.

The honest comparison holds device quality equal on both sides and asks which
architecture does more with it.

## Decision

**Compare the two-leg nozzle against a plate at matched quality (`f = e1`), and
report the `f = 0.8` crossover alongside it rather than as the verdict.**

On that basis the nozzle wins, and the margin compounds:

| matched `e = f` | nozzle | plate | ratio |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.000133 | 0.000309 | **0.4x (nozzle loses)** |
| 0.30 | 0.0295 | 0.0291 | 1.0x (tie) |
| 0.40 | 13.19 | 6.96 | 1.9x |
| 0.50 | 1,440 | 362.6 | 4.0x |
| 0.60 | 6.632e4 | 7,827 | **8.5x** |
| 0.70 | 1.698e6 | 9.548e4 | 17.8x |
| 0.80 | 2.833e7 | 7.826e5 | 36.2x |
| 0.90 | 3.353e8 | 4.79e6 | 70.0x |

(`total_growth` over the 11-cycle, 28.3930 yr chain, 10-day split.)

**State the currency.** As e-foldings/yr the same cells differ by only 23-26%
(0.3910 against 0.3158 at 0.6). Compounded over the chain that is 8.5x the
delivered mass. Both are correct and they sound completely different, so neither
should be quoted without naming which one it is.

**The win is not universal.** It switches on near `e = 0.3`. Below that the
ground-launch floor crushes `k1` onto the ignition window's lower root (~0.1) and
the nozzle degenerates toward a plate while still paying to launch slug.

## Consequences

- **ADR 0014's equivalent-plate table is the load-bearing artifact**, not its
  conclusion. `f_equiv > e1` in every cell at `e2 >= 0.4`, which is this ADR
  restated. The table needs no change.
- **The decision now rests on one unmeasured number**, the plate's true
  restitution at 45-65 km/s. If it is near 0.8 the plate holds; if it is near
  0.4-0.5, which is what the shock regime suggests, the nozzle wins outright. A
  field touches nothing and cannot ablate.
- **The paper leads with the nozzle** and states the `e1 ~ 0.6` crossover against
  `f = 0.8` explicitly, so a reader who believes the incumbent's number can reach
  the opposite conclusion from the same tables.

### What would reverse this

**Plume re-ingestion, still unmodelled.** The overtake puts impactor entry and
exhaust exit at the *same* end of the vehicle, so each arriving PuffSat flies up
the previous shot's plume. This argues `e1 < e2`, possibly strongly, and matched
recovery assumes `e1 = e2`. The head-on leg has no equivalent, since there the
impactor enters the front and the exhaust leaves the back. ADR 0014 already names
this as probably the largest missing effect; this ADR makes it the single
assumption the verdict depends on.

Pointing the other way, the wrong-way bulk drift the overtaking nozzle must
reverse carries only `1/(1+k1)` of the blob energy, 8.9% at the fleet ceiling.

### Recombination is assumed recovered

Not in ADR 0014 and not in `src/`. `plume_thermal` charges 50.4 MJ/kg to atomise
the water, 60% of the 84.4 MJ/kg bill, and nothing asks whether that energy comes
back. If it does not, the ideal impulse law overstates by `sqrt(1-phi)` where
`phi` is the locked fraction: a ceiling of 0.675 on `e1` at `k = 10.21` and 0.870
at `k = 4.02`, which would make the `e1 >= 0.8` cells unreachable.

It does come back. Three-body recombination time scales as `1/n^2`: 0.01 us at
1 kg/m^3 against a ~200 us expansion. It freezes only below ~0.01 kg/m^3. The
alkali seed is what makes the energy collectable, since potassium (4.34 eV) keeps
supplying electrons at 3,000-6,000 K where the water has already re-formed, giving
`Rm` of 9 to 400. Dissociation energy also carries no pressure, so a dissociated
plume stores 84 MJ/kg while pressing with only its 31 MJ/kg of translation.

**The fireball density decides this and is computed nowhere.** Same gap the paper
defers to a radiation-hydrodynamic calculation.

> **Amendment, 2026-08-26. The density was computed, and this section's escape does
> not hold. Its arithmetic, however, was right.**
>
> `puffsat_impact_simulation` computed the fireball density (`make analysis-fireball`,
> `make analysis-toll`). The plume freezes at **1.1e-2 to 2.4e-2 kg/m^3**, at the first
> station past the nozzle lip, with **90-100% of the store still held**. The rate check
> above is not wrong, it is evaluated at the wrong station: at 1 kg/m^3 recombination
> really is ~0.01 us, but there the plume is fully atomised and has nothing to give back
> yet. By the time it does, it is past the lip and the expansion clock has stepped down
> 8x.
>
> **This section predicted the consequence almost exactly and then dismissed it.** Its
> "ceiling of 0.675 on `e1` at `k = 10.21` and 0.870 at `k = 4.02`" reproduces from
> `plume_thermal.chemistry_efficiency` at the growth push's cold end as **0.6713** and
> **0.8683** -- agreement to 0.004. So the numbers here were never the weak part; the
> escape sentence was. **The `e1 >= 0.8` cells are unreachable, exactly as this section
> said they would be if the energy stayed locked.**
>
> ADR 0016 carries the consequence for the growth tables. Two things follow for *this*
> ADR specifically:
>
> - **`f` and `e1` can no longer be swept as a matched pair**, which is the method the
>   Decision rests on. They now carry different physical ceilings: `f` is a plate
>   restitution carrying the impact sim's ADR-0026 frozen bracket, `e1` is a nozzle
>   recovery carrying the `eta_chem` ceiling above. Matching them numerically compares
>   the plate at a value it can reach against a nozzle at one it may not.
> - **The margin the Decision won is much smaller.** Charging the toll on both legs, the
>   nozzle beats the plate by **12.3x** at the matched 0.80 rather than the 36.2x in the
>   table above, and below `eta_geom = 0.7` the two-leg chain is inadmissible outright.
>
> **The Context paragraph is also stale for an unrelated reason**, recorded here so a
> reader does not have to find it elsewhere: its premise that `f = 0.8` sits outside the
> validated envelope was already false when this ADR was written. The impact sim extended
> `f(v)` to 63 km/s on 2026-08-17, three days earlier, and measures 0.817-0.820 on the
> heavy plate. Seth confirmed 2026-08-26 that **the growth push flies the heavy plate**,
> so `STD_FUDGE_FACTOR = 0.8` stands unchanged and "nothing rebounds elastically at
> 46 km/s" is simply wrong.
>
> **The verdict is deliberately not revisited here.** Retiring an escape and weakening a
> method is not the same as reversing a decision, and which architecture the paper leads
> with is a call for Seth, not a consequence this amendment can draw on its own.

## Reproducing

`make two-leg`, plus per-cell `price_chain_two_leg(cycles, e, e).total_growth` and
`price_chain_two_leg(cycles, e, None, f).total_growth` for the matched diagonal.


---

## Amendment 2, 2026-08-26: the verdict survives, and this ADR's argument is not why

Amendment 1 retired this ADR's recombination escape and left its Decision resting
on a comparison method -- sweep `f` and `e1` as a matched pair -- that ADR 0016
then showed is unsound, because the two carry different physical ceilings. That
left the verdict standing on nothing in particular.

**It does not need the method.** Comparing the two architectures at the *same*
`eta_geom`, charging the frozen-dissociation toll on whichever legs carry a
nozzle, and granting the plate its **full measured** `f = 0.818` -- the heavy
plate value `puffsat_impact_simulation` measured across 45-63 km/s, which Seth
confirmed 2026-08-26 is the plate the growth push flies:

| `eta_geom` | nozzle on both legs | plate on the growth push, `f` = 0.818 | ratio |
| ---: | ---: | ---: | ---: |
| 1.0 | 6.931e7 | 1.464e6 | **47.4x** |
| 0.9 | 9.520e6 | 4.244e5 | **22.4x** |
| 0.8 | 7.723e5 | 7.486e4 | **10.3x** |

**The nozzle wins outright, against the incumbent's best measured number, with
the chemistry charged against the nozzle and not against the plate.** No matched
diagonal, no claim that `f = 0.8` is indefensible, no methodological argument at
all -- the comparison is now like-for-like and one-sided.

**This is a stronger position than the original.** The old argument required a
reader to accept that benchmarking against an unmeasured incumbent value was
unfair. The new one requires nothing: it hands the plate its measurement and
still loses to it by an order of magnitude.

**What the toll did take away** is the *size* of the margin, and that is worth
stating rather than burying: it was 36.2x at matched 0.80 before the chemistry
was charged and is 10.3x after. The direction is unchanged; the headroom is not.

**Both of this ADR's original factual claims remain wrong** and Amendment 1 stands
on that. "Nothing rebounds elastically at 46 km/s" is false (`e_eff` ~ 0.66-0.67
there), and `f = 0.8` was inside the validated envelope three days before this ADR
was written. A reader should take the verdict from this amendment and the
reasoning from nowhere in the Context section.

### What is still Seth's call, and it is not the verdict

**How the paper frames it.** `sec:two_leg_nozzle` currently states the `e1 ~ 0.6`
crossover against `f = 0.8`, which invites a reader who believes the incumbent's
number to reach the opposite conclusion. That framing was built for the old
argument and no longer matches this one. Whether to rewrite it, and whether to
lead with the nozzle at all given `eta_geom` is entirely unmeasured, is an
editorial decision this ADR does not make.

## Reproducing amendment 2

`price_chain_two_leg(cycles, 1.0, 1.0, geometric_efficiency=g)` against
`price_chain_two_leg(cycles, 1.0, None, fudge=0.818, geometric_efficiency=g)`
on the 11-cycle, 28.3930 yr chain at a 10-day split.
