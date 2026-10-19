# The Jupiter-only chain on plates: answers to S41 (for the parent)

Raised 2026-10-19 in `aim_is_all_you_need` (ADR 0075). **Written to be copied into
`Balloon-Pulse-Propulsion`.** Answers S41 items 1-5 (`docs/plate_jupiter_loop_asks.md` @
`988a9e3`). S42 is for the impact simulation and is not answered here. Reproduce with
`make plate-jupiter-loop ARGS="--family"` (~10 min).

## The answer: no, the Sun dive is needed

The flown chain is real ephemerides, 2S/3S, split by role on the 10-day orbit. It is flown
on the solar dive's own plate code:

- the push at 2 Hz and 1 g from the 2.43 km/s lob;
- the visible-sign bowl at 1 Hz;
- the stream-in-view finite-burn loss;
- each carrier's own return burn;
- the 2 Hz delivery on its 25 g plate;
- the gated rNPV with no Sun gate (95 / 85 / 90), and k optimized for the dear break-even.

| chain, departure | growth per 3S | doubling | steady | break-even cheap / **dear** |
|---|---|---|---|---|
| flown, 0.25 g | 2.01 | 3.26 yr | $263 | $315 / **$573** |
| flown, 0.5 g | 2.45 | 2.54 yr | $167 | $197 / **$279** |
| flown, 1 g | 2.59 | 2.39 yr | $153 | $179 / **$238** |
| 3S only, 0.25 / 0.5 / 1 g | 1.74 / 2.00 / 2.07 | 4.11 / 3.28 / 3.12 yr | $172 / $141 / $134 | dear $1026 / $560 / $499 |
| **Sun dive, 9.86 R_sun, X 60** | **3.80** | **1.70 yr** | $87 | $107 / **$118** |

`tab:no_sun_loop`'s ballpark (2.49x, $292 dear at the best exact-3S phase) is too kind.
There are two reasons:

1. **The loss.** The ballpark used a fixed-direction loss. With every incoming route kept in
   view (ADR 0066, the chain's own tables) the 0.25 g departure loses **0.8-2.0 km/s**.
   Swapping in the fixed-direction loss lifts the 3S chain from 1.74x / $1026 to
   1.85x / $802.
2. **The real 3S windows.** They return at 52-56 km/s excess, not the best phase's 61.3, and
   depart at 11.9-13.2 km/s. Per-window growth is 1.5-2.0x at 0.25 g.

The flown chain beats 3S-only on plates. Its 2S windows (14.0-14.6 km/s out, 1.2-1.8x a
window) cycle faster, and that also pays the seed sooner.

## Item 2: slug ratios

The dollar optimum is k **6.5-6.8** on the push, **12.3-12.7** on the departure and
**6.0-6.6** on the delivery. "Relatively high" k (10 / 20 / 10) barely grows: 1.06x and
$29 660 dear on the flown chain. Your ballpark's direction is confirmed.

## Item 3: the departure rate

Rate is the strongest lever. At 0.25 g the burns last 43-62 minutes. The 2S burns (8.5-9.4 km/s
at 600 km) stay inside the visibility table's 7200 s. 0.5 g cuts the flown chain's dear
break-even $573 -> $279, and 1 g cuts it to $238. The bowl's heavier pulse is charged, and it
costs far less than the loss saves.

## Item 4: the seed

The seed is the child's: the same chain, leaving at 11.99 km/s. A stripped Starship gives
$2172 / $9293 per kg. ADR 0059's two-year Earth loop (x0.6687, +2.18 yr) cuts the flown
0.25 g break-even $315 / $573 to $300 / $513, 5% cheap and 10% dear. ADR 0060 stands: fly
direct, quote the loop as a sensitivity.

## Item 5: a faster unpowered return

At each window's fixed epochs, every sampled encounter and path pair with at most a
0.5 km/s correction was scored on the plate ledger. In both a 3S and a 2S window the
chain's least-correction return is also the plate ledger's best. The fastest member is only
0.1 km/s quicker (54.42 vs 54.31 km/s excess; 60.50 vs 60.37). It pays 0.28-0.42 km/s of
correction and grows less. At fixed epochs the Jupiter-only geometry pins the return. A
69 km/s return at a 12.1 km/s departure is not on this chain.

## Still owed (S42, impact simulation)

The bowl at 160-180 deg and 66-76 km/s is interpolated between the 129.9 and 180 deg capture
runs. The push at 54-65 km/s uses the 100 km/s law scaled by `w / 100`. These figures stand on
both.
