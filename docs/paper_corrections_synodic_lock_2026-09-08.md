# Answering S5-S8: the synodic lock, the coast that decides it, and the ladder

**Written to be copied into
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
and worked there**, on `templateArxiv.tex`. Self-contained: every number needed
to make every edit is stated here, because an agent in the paper repository
cannot run the code that produced them.

Generated 2026-09-08 in `katzseth22202/aim_is_all_you_need`, answering
`docs/fly_and_park_asks_for_aim_repo.md` (asks **S5-S8** of the paper's
`docs/deferred_to_companion_repos.md`). Backing decision:
`docs/adr/0031-the-synodic-lock-is-the-resonance-we-already-had.md`.

Same ground rules: **neither document is the source of truth**; locate claims by
grepping their quoted wording, not by line number; if a number here looks wrong,
say so rather than working around it.

---

## Summary

| ask | verdict | what the paper may now write |
| --- | --- | --- |
| **S5** | **answered, with one figure corrected.** The 9-day park is inadmissible; the admissible 2.00 S lock exists, and it is ADR 0011's own two-synodic resonance | **0.873 yr**, not 0.84 -- and worded as "lock the resonance we already have", not "fly a new cycle" |
| **S6** | **answered: a different measurement of a different thing.** 1.19 and 0.87 are not at matched efficiency with 1.45, and not at matched scope either | the ladder below, every rung tagged. Do not present the new rungs as refinements |
| **S7** | **absorbed and pinned.** One contiguous arc at every exhaust speed | "one window, 3.4x wider at Isp 1200" -- cite `make fly-park` |
| **S8** | **yes, and more strongly than expected.** The flown 2S cycles are already exact locks, needing no padding | "lock a cycle we already fly" is correct |

**The single most important sentence.** Fly-and-park is not what produces the 2S
operating point. The admissible 2S lock's park *is* the mandatory 20-day coast,
so the construction has nothing to add to it. Fly-and-park stays what ADR 0030
said it was -- a way to let a *shorter, hotter* flight hold the 3S clock, worth
5% at the sweet phase and 27% off it. It does not buy the faster clock.

---

## S5. The nine-day park, and what is on the other side of it

**You asked the right question, and it decides the item.** A nine-day coast is
not admissible, and the reason is structural rather than a tolerance.

**The park is the coast, lengthened -- it cannot be shorter than it.** Every
cycle already ends with one full period of the bound near-escape orbit: the
returning wave pushes the payload up at periapsis, and the departure burn lights
at the *next* periapsis. That is the aim reversal, the prograde-retrograde trick
that turns a retrograde push into a prograde departure. It is not an optional
wait; it is one orbit of a 20-day orbit. `optimize_jovian_cycle_chain()` encodes
exactly this, pinning each departure to the previous arrival plus
`PUFFSAT_CYCLE_ORBIT_PERIOD`.

**Our constant was wrong, and your construction found it.** `fly_and_park.py`
shipped with `MINIMUM_PARK` = 0.02 synodic periods, about eight days -- 2.5x
under the coast it stood for. Nothing ADR 0030 published turned on the
difference, because the winning 3S cycles park for 83 to 106 days. Your 2.00 S
lock was constructed precisely on the difference. `MINIMUM_PARK` is now the
coast.

### The admissible lock exists, and you already have it

Charge the coast and the best 2.00 S lock moves:

| | phase | flight | park | `dv` | `v_b` | growth (2214) | doubling (2214) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| your construction | 0.781 | 1.978 S | 0.022 S (**8.7 d**) | 7.25 | 63.2 | 6.021 | 0.843 yr |
| **admissible** | **0.8082** | **1.9497 S** | **0.0503 S (20.06 d)** | **8.613** | **63.35** | **5.670** | **0.873 yr** |

**That second row is not new.** It is, to every digit, the two-synodic fixed
point our `fixed_points()` already reports and that **your own N4 already
publishes** -- "1.9998 synodic periods at Earth-minus-Jupiter -69.04 deg, `v_b`
63.35 km/s, 8.613 km/s departure burn". The test asserts cycle identity, not
resemblance. Its park is the coast plus about ninety minutes, which is why
fly-and-park adds nothing to it: there is no remainder left to pad.

**So the wording changes, and in your favour.** "Pad a short flight up to 2.00 S
and the cycle becomes a fixed point" becomes "the two-synodic resonance the paper
already discusses *is* a synodic lock, and holds its own departure phase." That
is a much easier sentence to defend.

### What the correction costs

| | Isp 1200 | Isp 2214 |
| --- | ---: | ---: |
| doubling, 9-day park | 1.001 yr | 0.843 yr |
| **doubling, coast charged** | **1.082 yr** | **0.873 yr** |
| phases offering a 2S lock | 13 -> **11** of 73 | 28 -> **26** of 73 |

**+3.5% on the clock and two phases.** Small -- but the figure the paper quotes
should be **0.873 yr**, not 0.84.

### The chain check, run rather than argued

Your N4.2 caveat is the reason this needed running, so it was run. One thing had
to be built first: **the existing beam search structurally cannot park.** It pins
each departure to the arrival plus a fixed coast, so asking it about fly-and-park
is asking a model that has no park. `sustainable_chain_optimum()` searches the
same enumerated branches over policies that may park *any* length, as a
maximum-ratio-cycle over the departure-phase graph -- the objective is
`sum(log growth) / sum(time)`, which is the long-run rate a grower actually
holds.

| Isp | best repeating policy | length | phase | doubling |
| ---: | :--- | ---: | ---: | ---: |
| 380 (methalox) | one cycle, self-looping | 3.0000 S | 0.7397 | 3.641 yr |
| 1200 | one cycle, self-looping | 2.0000 S | 0.8082 | 1.082 yr |
| 2214 | one cycle, self-looping | 2.0000 S | 0.8082 | 0.873 yr |

**A one-cycle loop is a synodic lock**, so the chain's own optimum is the fixed
point and the single-cycle answer survives the lookahead here. The methalox row
is ADR 0030's verdict reached by a second route: the 2S point costs an 8.613 km/s
departure burn, which at 3.727 km/s of exhaust delivers 9.9% of the vehicle, so
declining it is correct.

**One caveat, stated because it is real.** This is a chain over the circular
phased grid, not the real-ephemeris cadence. It says nothing about whether the
2S window is *available* -- see S8, where that turns out to be the binding
constraint.

### Also checked, so you need not wonder

**The branch dedup is innocent.** Our enumerator collapses branches landing in
one 7-day bucket, keeping the highest growth *as scored on methalox*, which could
in principle hide a branch that only wins at Isp 2214. Re-enumerating the entire
grid with the dedup scored at Isp 2214 instead returns every figure above
unchanged to four digits.

**"Does a longer sub-2.00 S flight exist that leaves a 20-day remainder?"** Yes,
and it is the row above: flight 1.9497 S leaves 20.06 days. It is the only one
that both clears the coast and grows, which is why the 2S lock and the resonance
are the same object rather than two nearby ones.

---

## S6. The ladder, and why the rungs do not compare

**Verdict: a different measurement of a different thing.** Not a refinement of
1.45 yr. Three things separate them, and the first is a term collision.

### They are not at matched `f`

The paper's *"doubles every 1.74 years when the nozzle delivers 60% of its ideal
impulse, and every 1.45 years at 80%"* is correct as written. But your ask
restates those as **"1.74 yr at `f` = 0.6 and 1.45 yr at `f` = 0.8"**, and that
is a conflation. Those are the **nozzle impulse recovery** `e` of
`two_wave_growth.price_chain`, which derates the whole impulse from *outside* the
momentum debit. The elasticity `f` -- `STD_FUDGE_FACTOR`, the collision
elasticity in `eq:PuffSat_ratio` -- is held at **0.8 in both rows**. Our
fly-and-park figures are also at `f` = 0.8, and at **no `e` at all**.

**So "at that same `f` = 0.8" is true and misleading in the same breath.** The
efficiency that separates the rungs is `e`, and the new rungs do not have one.

There is a second collision waiting. `sec:mass_interest` calls the two-wave rung
*"a nozzle geometric efficiency of 0.6"*. That is `e` as well, not `eta_geom`,
which is a different sweep again (`analysis.tolled`). Worth fixing while the
ladder is being written -- and we had the same slip on our side: `make nozzle`
labelled its 2.990 yr row `derated f=0.8` when the knob is `recovery=0.8`. Fixed
here, mentioned so you know the ADR 0009 table carries the old label too.

**The paper has already made this exact distinction once**, in `sec:depth_cost`:
*"The nozzle efficiencies are not the same parameter and should not be read as
one... the mapping between the two has not been worked."* That sentence applies
here unchanged, and citing it is the cheapest way to write the caveat.

### They are not at matched scope or accounting either

| | fly-and-park lock | two-wave chain |
| --- | --- | --- |
| model | circular coplanar, relative epoch | real ephemeris, calendar epochs |
| scope | best single cycle at its best phase | 11 flown cycles, including the dear ones |
| departure charged as | `exp(-dv/v_e)` at Isp 2214 | two-currency nozzle ledger, recovery `e` |
| projectile stream | **not charged at all** | 20-29% of the batch takes the nozzle bend |

**The last row is the one that matters.** `fly_and_park` charges the slug against
the payload and mints `M(v_b)` from the *whole* arriving wave; the two-wave ledger
sources the projectiles explicitly by splitting the batch at Jupiter. Across the
eleven flown cycles that split puts **19.5-23.6% on the nozzle bend at `e` = 0.8**
and 24.2-29.0% at `e` = 0.6 -- mass that mints no payload. Our `CONTEXT.md`
already records the omission under the **departure-burn accounting seam**:
nothing is charged for delivering the head-on stream, whose own phasing is
unmodelled.

**Therefore 0.873 yr is systematically optimistic against 1.45 yr.** Which
direction the bias runs is knowable even though its size is not: the new rungs
omit a charge the old ones pay.

### The ladder

| rung | doubling | scorer | model | scope | efficiencies | reproduce |
| --- | ---: | --- | --- | --- | --- | --- |
| chemical direct flyby | 4.038 yr | growth / cycle time | circular, unphased | single cycle | methalox Isp 380, `f` = 0.8 | `make nozzle` |
| one-wave parked nozzle | 2.990 yr | two-currency nozzle ledger | circular | single cycle | **`e` = 0.8**, `f` = 0.8, `k*` = 6.0 | `make nozzle` |
| **two-wave chain** | **1.737 yr** | `prod(growth) / sum(cycle yr)` | **real ephemeris** | **11 cycles, 28.39 yr** | **`e` = 0.6**, `f` = 0.8, `k*` = 8.53 | `make two-wave` |
| **two-wave chain** | **1.450 yr** | same | same | same | **`e` = 0.8**, `f` = 0.8, `k*` = 9.25 | `make two-wave` |
| fly-and-park **3S lock**, Isp 1200 | 1.377 yr | `M(v_b) exp(-dv/v_e)` | circular phased grid | **one cycle repeated** | `f` = 0.8, **no `e`**, no stream cost | `make fly-park` |
| fly-and-park **3S lock**, Isp 2214 | 1.189 yr | same | same | same | same | `make fly-park` |
| fly-and-park **2S lock**, Isp 1200 | 1.082 yr | same | same | same | same | `make fly-park` |
| fly-and-park **2S lock**, Isp 2214 | **0.873 yr** | same | same | same | same | `make fly-park` |

**What the paper should write.** Either present the fly-and-park rungs in their
own labelled block, stating that they are single-cycle figures in a circular
model that do not charge the projectile stream -- or leave them out. What they
must not do is sit unlabelled in the same list as 1.45, where a reader will read
0.873 as an improvement on it.

**Also worth saying plainly, because it is the honest reading:** the fastest rung
in the table is 0.873, and the fastest rung anyone could *defend* today is 1.45,
because it is the only one on real ephemerides with its projectiles paid for.

---

## S7. The launch-window layout -- absorbed, pinned, reproduces

Confirmed exactly, and now computed inside `fly_and_park.py`
(`usable_phase_windows()`) rather than a scratch script, so the paper can cite
`make fly-park`:

| Isp (s) | 380 | 700 | 1000 | **1200** | 1500 | 1800 | **1900** | 2214 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| usable phases | 13/73 | 26/73 | 36/73 | **44/73** | 56/73 | 69/73 | **73/73** | 73/73 |
| **separate windows** | **1** | **1** | **1** | **1** | **1** | **1** | **1** | **1** |
| widest (d of 399) | **71** | 142 | 197 | **240** | 306 | 377 | **399** | 399 |

Your two cautions both hold. The arc must be measured **cyclically** or the Isp
1800 row, which wraps through phase 0, reports two windows -- a test pins that
directly on a synthetic wrapping grid. And the methalox 71 days agrees with your
finer -36/+24 day read to within one grid cell.

**So the claim for the paper is "one window, 3.4x wider at Isp 1200"**, not a set
of windows. Pinned by `test_the_usable_phases_form_one_window_at_every_exhaust_speed`.

---

## S8. Yes -- and the flown cycles are already locks

**Stronger than the ask expected.** You asked whether any 2-synodic cycle in the
cadence run pads to an exact 2.00 S total and holds its departure phase. They do
not need to pad: **they are already exact.**

The adaptive cadence builds every return on an exact synodic multiple of the
circular synodic period, so all eleven flown cycles have a departure-phase drift
below 1e-4 S. Seven are 2S at 2.1841 yr, four are 3S at 3.2761 yr. There is no
remainder to park.

Against the circular lock:

| | departure `dv` | growth-wave `v_b` |
| --- | ---: | ---: |
| flown 2S cycles (7 of them) | 6.841 - 7.171 km/s | 61.83 - 65.13 km/s |
| circular 2.00 S lock | **8.613 km/s** | **63.35 km/s** |

The arrival lands **inside** the flown family. The burn sits about **20% above
the dearest** of them, which is what you would expect of a circular coplanar
model that cannot choose when it meets Jupiter -- the same second steering knob
ADR 0013 credits with making the split nearly free. Different models, so this is
a family resemblance and not an identity.

**So "lock a cycle we already fly" is the correct claim**, and it may be worded
that strongly.

**But it retires the wrong problem, and the paper should say so.** The reason the
flown chain still takes four 3S returns is not phase drift -- there is none. It
is ADR 0011's perijove floor: only **45 of 91 windows over 200 years** clear
4,000 km. The 2S lock removes a difficulty the architecture did not have. The
honest cadence claim stays what your N4 already says: **2S with a 3S fallback
about half the time.**

---

## What this changes on our side, which the paper does not need to restate

- **One published fly-and-park figure moves.** Padding a shorter, hotter cycle
  beats the plain 3S cycle at **16** of 30 phases at Isp 1200, not 17. Methalox
  (1 of 11) and Isp 2214 (25 of 30) are unchanged, as are the exchange rate, the
  sweet phase, the usable-phase fractions and the 5%-at-sweet / 27%-off-phase
  shape.
- The perfect-retrograde premium's median moves **1.036 -> 1.037 km/s** over 29
  phases rather than 30: charging the coast shrinks the parkable set enough that
  one phase's hottest survivor was colder than its own pure-3S cycle, and that
  row is now excluded rather than priced as a premium.
- Nothing in N1, N2 or N3 of `docs/paper_corrections_fly_and_park_2026-09-07.md`
  is retired. The 3S fly-and-park result stands as published.

---

## Reproduction

```bash
make fly-park    # ~18 s: windows, locks, the chain check, the ladder
make two-wave    # the flown chain, now with a period_synodics column
pytest tests/test_fly_and_park.py tests/test_two_wave_growth.py
```

Search box unchanged from ADR 0030: 73 departure phases over one synodic period,
`_cycle_branches` at the committed settings with `_OUTBOUND_TOF_MIN` freed to
0.70 yr, growth re-scored as `M(v_b) * exp(-dv/v_e)` with `f` = 0.8 and `v_rf` =
10.9503 km/s. The chain search adds one parameter of its own: the maximum-ratio
bisection is bracketed on [0, 3] e-foldings per synodic period and run 60 steps.
