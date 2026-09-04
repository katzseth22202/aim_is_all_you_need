# Corrections owed to the paper: `sec:split_tail`'s low-thrust caveat is answered

**Written to be copied into
[`katzseth22202/Balloon-Pulse-Propulsion`](https://github.com/katzseth22202/Balloon-Pulse-Propulsion)
and worked there**, on `templateArxiv.tex`. Self-contained: every number needed
to make every edit is stated here, because an agent in the paper repository
cannot run the code that produced them.

Generated 2026-09-04 in `katzseth22202/aim_is_all_you_need` at the commit that
carries this file, against the paper at commit `1d42c91`. It answers the item
`sec:split_tail` left open in its own closing paragraph:

> Two things this calculation does not do. It compares a budget of
> thrust-seconds against an impulsive burn rather than solving the low-thrust
> trajectory, and a push spread over a year does not in general move the
> endpoint as far as one kick placed well. A finite-thrust optimization could
> shift the requirement either way, and it is the one piece of work that could
> overturn the verdict.

and `docs/deferred_to_companion_repos.md`'s **S4**, which asked this repository
to record the 10-day column alongside ADR 0026's 20-day one and to name a
`make` invocation the caption could use.

Same ground rules as `docs/paper_corrections.md`: **the paper is not the source
of truth**; locate claims by grepping their quoted wording, not by line number;
if a number here looks wrong, say so rather than working around it.

The decisions behind it are
`docs/adr/0028-freeing-the-burn-does-not-move-the-splits-price.md` and the
2026-09-04 addendum to
`docs/adr/0026-the-splits-tail-is-worth-paying-and-sep-cannot-pay-it.md`.

---

## Summary

| | edit | what changes |
| --- | --- | --- |
| **C1** | `sec:split_tail`'s closing paragraph, "Two things this calculation does not do" | the first of the two is now **bounded rather than open**; full replacement text below |
| **C2** | `sec:split_tail`, "so on two of the three no change of flyby helps at all" | it is **all three**; the exception was a floating-point tie in a label, now fixed here |
| **C3** | `sec:split_tail`, "What is left over is an angle rather than a speed mismatch" | the angle can now carry its number: **107.6–108.6° demanded, 105.3–107.2° available, 0.97–2.33° short** |
| **C4** | `tab:cadence_propellant`'s caption, "Reproduce with `make sep-split`" | **`make sep-split-10d`** — the paper's figures are the 10-day ones and the plain target runs 20 |
| **C5** | `docs/deferred_to_companion_repos.md`, S4 | answered; the 10-day column is recorded and verified here |

**Confirmed, no edit needed.** Every 10-day figure `sec:split_tail` quotes
reproduces exactly: the three separation burns 398.5 / 556.0 / 1011.3 m/s and
their departure months; the worst total correction 1029.8 m/s; the eight cheap
cycles under 0.3 m/s and the chain mean of 0.179 km/s that
`sec:jupiter_only_growth` quotes; the 1.056 R_J perijove; 1.737 against 1.863 years and the
7.3% of clock; every cell of `tab:cadence_propellant` (1.737 / 1.690 / 1.719,
1.863 / 1.863 / 1.863, 2.270 / 1.501 / 1.859) and its 7+4, 4+6, 13+0 cadence
counts; 5.04 km/s as the never-fall-back worst and a fifth of the wave
surviving it; 1.57 kW/t, 0.79 MW, 40 N at 1 AU and 1.5 N at Jupiter, 96% taken
by the inverse square, ~60 Hall thrusters, 2.4% array fraction, and argon worth
1.1% net of its array.

**Do not change:** the array sizing, `tab:cadence_propellant`'s numbers, the
cadence verdict, or the second half of the closing paragraph — the uncharged
departure spread between the two waves is untouched and still open.

---

## A. What the new work is, in one paragraph

The gap the paper names is real but it is bounded, and bounding it does not
need a low-thrust trajectory solver. Deleting the thrust-magnitude limit turns
the finite-thrust problem into an impulsive one; that is a *relaxation*, so its
optimum cannot cost more than the problem it relaxes. The cheapest impulsive
trajectory between the same two fixed Earth dates is therefore a floor under
any finite-thrust solution. `src/free_dsm_bound.py` solves that relaxation with
the burns free in time, direction and magnitude — up to five on each leg — over
a free Jupiter encounter date, a free aim point and a free departure excess,
with the flyby left unpowered because an electric stage cannot fire an Oberth
burn. Forty free parameters. **The answer is the trajectory the chain already
flies**: a single burn just after Jupiter, at the perijove floor, at the seam's
own encounter date, with nothing at all spent outbound, at the same price to
within 0.09%. Nothing was found below it, at either split gap.

---

## C1. The closing paragraph

**Find:** "Two things this calculation does not do." — the paragraph after
`tab:cadence_propellant`'s discussion, beginning `Two things this calculation
does not do. It compares a budget of thrust-seconds`.

**Why it changes:** the first of its two items is answered. The second (the
uncharged departure spread) is not, and is preserved verbatim below.

**Replacement** (keeping the paper's `\SI{}` conventions; the second half is
unchanged from the current text):

> One of the two gaps in this calculation is now closed. It compares a budget
> of thrust-seconds against an impulsive burn rather than solving the low-thrust
> trajectory, and a push spread over a year does not in general move the
> endpoint as far as one kick placed well. That comparison is bounded rather
> than open. Deleting the thrust limit turns the finite-thrust problem into an
> impulsive one, and a relaxation cannot cost more than the problem it relaxes,
> so the cheapest impulsive trajectory between the same two Earth dates is a
> floor under any low-thrust solution. Solved with the burns free in time,
> direction and magnitude, up to five on each leg, over a free flyby date, a
> free aim point and a free departure speed, the cheapest trajectory found is
> the one the chain already flies: a single burn just after the flyby, at the
> same price to within a tenth of a percent, with nothing spent on the way out
> and the flyby still scraping the altitude floor. Freeing the burn buys
> nothing because the residue is an angle, and no placement of a burn buys an
> angle. The \SI{0.79}{\mega\watt} above is therefore a floor on what an
> electric stage would need rather than an estimate of it. The search is a
> heuristic one, so this is a strong floor rather than a proof: a cheaper
> optimum would have to have hidden from a forty-parameter multi-start search
> and from two thousand blind draws of the same box, of which the cheapest came
> in at a hundred times the price \cite{Katz_aim_is_all_you_need_2025}.
>
> Separately, no ledger charges the difference between what the two waves'
> departures require. Both leave Earth at the same instant but need different
> periapsis speeds, and that gap runs \SIrange{224}{559}{\meter\per\second} at a
> twenty-day split and about half that at the ten days flown here. It is not
> carried propellant, since the departure is the head-on nozzle, so it lands as
> extra slug on half the batch \cite{Katz_aim_is_all_you_need_2025}.

**The numbers behind each claim in it,** at the 10-day gap the paper flies:

| departure | separation burn | free-burn optimum | spent outbound | perijove | encounter moved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2030-02-18 | 556.029 m/s | 556.112 m/s | 0.0005 m/s | 1.05595 R_J | +0.0004 d |
| 2036-09-07 | 1011.273 m/s | 1011.429 m/s | 0.005 m/s | 1.05595 R_J | +0.02 d |
| 2042-02-22 | 398.470 m/s | 398.828 m/s | 0.004 m/s | 1.05595 R_J | +0.10 d |

The optimum lands 0.015–0.09% *above* the burn the chain already pays, not
below it. That residual is the bound's own model rebuilding the flown
trajectory rather than importing it, and it is the wrong sign to be a saving.
The blind draws: 221, 265 and 219 of 2000 uniform samples of the search box fly
a complete Earth–Jupiter–Earth trajectory at all, and the cheapest of them
costs 92.40, 110.47 and 94.43 km/s — 109 to 237 times the price. "A hundred
times" in the draft text is the safe rounding of the smallest of those ratios.

---

## C2. "On two of the three"

**Find:** `What is left over is an angle rather than a speed mismatch, so on two
of the three no change of flyby helps at all.`

**Change "two of the three" to "none of the three".** Suggested wording:

> What is left over is an angle rather than a speed mismatch, so on none of the
> three does a change of flyby help at all.

**Why.** The sentence is a faithful report of a flag in the companion code that
was reading the wrong thing. `CorrectionCycle.bend_limited` tested the *name* of
the winning maneuver architecture rather than what that architecture spends. On
2030-02-18 the hybrid architecture returns the identical trajectory to the pure
deep-space one and beats it by **1.4e-6 m/s** — floating-point noise — and took
the label with it, while its own flyby burn is exactly zero. The property now
tests the burn, and reads bend-limited on all three.

The underlying fact, checked directly on all six cycle-and-gap combinations:
the pure powered-flyby architecture (`perijove_only`) returns **no solution at
all** for any of these windows, and every winning hybrid solution spends
**exactly zero** at the flyby. Fixed in `src/sep_split_correction.py` and
pinned by `test_the_ten_day_gap_is_bend_limited_on_all_three_too`. No number in
`tab:cadence_propellant` or anywhere else in `sec:split_tail` moves.

---

## C3. The angle, with its number

**Find:** the same sentence as C2, `What is left over is an angle rather than a
speed mismatch`.

The paper states this qualitatively and can now state it. Optional addition,
after the sentence C2 amends:

> The trajectory asks Jupiter for \SIrange{107.6}{108.6}{\degree} of turn and
> the flyby supplies \SIrange{105.3}{107.2}{\degree} at the altitude floor, so
> the burn is paying off a shortfall of \SIrange{0.97}{2.33}{\degree} at a
> \SIrange{20.2}{20.8}{\kilo\meter\per\second} arrival. The chord that closes
> that shortfall, $2 v_\infty \sin(\delta/2)$, runs about
> \SI{0.36}{\kilo\meter\per\second} per degree and accounts for 84 to 86\% of
> the bill; the remainder is the speed error the exact match also pays.

**The numbers,** per cycle at the 10-day gap. Turn authority is
$\delta = 2\arcsin(1/e)$ with $e = 1 + r_p v_\infty^2/\mu_J$ at the
\SI{4000}{\kilo\meter} altitude floor:

| departure | $v_\infty$ in | turn needed | available | deficit | chord | chord / burn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2030-02-18 | 20.167 km/s | 108.570° | 107.211° | 1.359° | 478.3 m/s | 0.860 |
| 2036-09-07 | 20.802 km/s | 107.650° | 105.318° | 2.332° | 846.6 m/s | 0.837 |
| 2042-02-22 | 20.190 km/s | 108.117° | 107.143° | 0.974° | 343.2 m/s | 0.861 |

If it is ever useful, the same three cycles at the 20-day gap demand
108.83–109.56° against 104.20–105.85° available, so 3.34–4.63° short at
20.6–21.2 km/s. **Halving the head start halves the deficit and so halves the
bill** — that is the whole difference between the two gaps, and it is worth one
clause if the ten-versus-twenty question ever comes up in review.

---

## C4. The caption names a target that runs the other gap

**Find:** `tab:cadence_propellant`'s caption, ending `Reproduce with
\texttt{make sep-split}.`

**Change to** `Reproduce with \texttt{make sep-split-10d}.`

**Why.** `make sep-split` runs the companion repository's default 20-day split
gap and prints ADR 0026's column, not the paper's. The table's numbers are the
10-day ones. `make sep-split-10d` is new in this commit and is exactly
`python -m src.sep_split_correction --split-days 10`.

While there: `sec:split_tail`'s prose figures come from the same run, so if any
other sentence in that subsection names `make sep-split`, it wants the same
change. (At `1d42c91` the caption is the only place.)

---

## C5. S4 is answered

`docs/deferred_to_companion_repos.md`'s S4 asked for "a short amendment to ADR
0026, or a new ADR, recording the 10-day column alongside the 20-day one, and a
`make sep-split` invocation the paper's caption can name". Both are done: the
addendum dated 2026-09-04 at the end of ADR 0026, and `make sep-split-10d`.

S4 can be marked answered, with this row for the "What landed" table:

| item | verdict | where it landed |
| --- | --- | --- |
| **S4** | **Answered, no figure moves.** The 10-day column reproduces every figure `sec:split_tail` quotes; `make sep-split-10d` names it. One label read a cycle wrong (C2) | ADR 0026 addendum 2026-09-04; `sec:split_tail`'s "two of the three"; `tab:cadence_propellant`'s caption |

S4's closing note — that `STATED_ACCELERATION_1AU = 2.0e-5` is described as the
design's operating point while nothing in `templateArxiv.tex` states it — still
stands, and **the C1 replacement text is written so as not to depend on it**.
It quotes the required power (0.79 MW), which is a property of the trajectory,
rather than a shortfall ratio against an acceleration the paper never asserts.
The companion repository continues to carry 254.5 m/s as its own reference
figure; the paper does not need it and should keep not quoting it.

---

## D. What this still does not settle

- **The bound is empirical, not proved.** The relaxation argument is exact, but
  a heuristic global search returns an *upper* bound on the relaxation's own
  optimum. Nothing certifies that a cheaper impulsive trajectory does not exist;
  what is established is that none was found in a forty-parameter box by a
  seeded multi-start search, and that the cheapest of 2000 blind draws was 109
  times dearer. The C1 text says so in its last sentence, and that sentence
  should survive editing.
- **It bounds the delta-v, not the trajectory.** A finite-thrust solution that
  actually delivered 1.0 km/s over a 440-day leg would still have to meet
  Jupiter and reach Earth on the day it is wanted, and nothing here says it can.
  The bound is useful precisely because it points the wrong way for the
  optimist: the tail is ruled out without that question needing an answer.
- **The arc topology is fixed.** Both legs are zero-revolution Lambert arcs with
  a single Jupiter encounter. Multi-revolution arcs and repeated flybys are
  outside the search box, and primer-vector theory covers the number of impulses
  rather than the topology.
- **The departure spread is still uncharged**, exactly as the paper's own
  second item says.
