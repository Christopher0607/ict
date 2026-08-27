# 04 · Strategy search: 10,384 configurations, no survivors

**Date:** 2026-08-22
**Reproduce:** `python run_strategy_search.py && python run_search_report.py`
**Data:** NQ 1-minute, 2016-01 → 2024-08, 3,014,976 bars. Holdout never opened.
**Pre-registration:** `research/search/registry.py` — seven families committed at `5f44f1a`, the ICT family added at `4421171`, each before its results existed.

> ## ⚠ Corrections, 2026-08-26
>
> Two numbers in this document were wrong. **Every conclusion below survives —
> all four move in the same direction, and that direction is worse.** Details
> and full recomputed tables in `findings/05`.
>
> **1. Commission was estimated, not measured — overstated by 1.8×.** The gross
> expectancies here were computed by converting the $4 round turn into R with an
> all-session median ATR of 2.77. But every family in this document trades RTH
> only, where ATR is larger, and the mean of a ratio is not the ratio of the
> medians. Measured from the trades themselves across the same 10,336 evaluable
> configurations:
>
> | | published | measured |
> |---|---|---|
> | commission | 0.0481 R | **0.0265 R** |
> | gross expectancy (median) | −0.0080 R | **−0.0259 R** |
>
> **2. "The entire deficit is commission" is false.** Commission explains **48%**
> of the median loss, not 100%. The other half is the signal itself losing money
> before a single fee is charged. Every family's gross expectancy in the table
> below is overstated by 0.011 to 0.025 R.
>
> **3. The stop-width table's punchline is gone.** At a 4-ATR stop the gross
> expectancy is **−0.0061 R**, not +0.0018 R. It is still small; it is no longer
> "zero to three decimal places", and it is no longer positive.
>
> **4. ICT was additionally flattered by a slippage bug.** Its London killzone is
> 03:00–04:00 ET — entirely outside RTH — and was charged one tick of stop
> slippage like everything else. Overnight liquidity is twelve times thinner
> (median 1-minute volume 57 against 707), so it now pays two. ICT's own code is
> unchanged. Corrected: median net **−0.0975 R** (published −0.0868), median gross
> **−0.0640 R** (published −0.0394). It finishes last of eight by a wider margin
> than reported.
>
> The direction of every error is the same: this document was too kind to these
> strategies.

---

## Result

**Nothing clears the bar.** The best configuration in the grid reaches
**+0.128R** after costs against a required **+0.185R**, and that best result is
statistically indistinguishable from the best you would expect to find by
searching 10,384 coin flips.

The holdout stays sealed. There is nothing to validate on it.

| Gate | Passing |
|---|---|
| ≥ 200 trades | 10,336 / 10,384 |
| ≥ 200 trades/year | 8,952 |
| BH-FDR significant, q=0.10 | 5,898 |
| **expectancy ≥ +0.185R** | **0** |

The grid grew from 8,656 to 10,384 when the ICT family was added. That raises
the noise ceiling — expected largest |t| under the null goes from 3.855 to
3.900 — for every family, including the seven already run, so all seven were
re-run rather than carried over. Searching more is not free.

---

## The 5,684 "significant" results are significantly *losing*

| | count |
|---|---|
| significant, positive expectancy | ~350 |
| **significant, negative expectancy** | **~5,550** |

The FDR gate is detecting transaction costs, not edges. This is worth stating
plainly because "5,684 of 8,656 configurations were statistically significant"
is the kind of sentence that could be quoted as encouraging.

---

## Why they lose: about half commission, about half the signal

> **Corrected 2026-08-26.** The original version of this section estimated
> commission from a median ATR and concluded the entire deficit was commission.
> Measured per trade, it is roughly half. The table below is recomputed.

Adding the measured round-turn commission back:

| | median across 10,336 evaluable configs |
|---|---|
| Net expectancy | **−0.0546 R** |
| Commission (measured per trade) | +0.0265 R |
| **Gross expectancy (before costs)** | **−0.0259 R** |

Before costs these rules lose money, and after costs they lose roughly twice as
much. The original reading — "worth approximately nothing before costs" — was
too generous by a factor of three.

The stop-width breakdown still confirms the mechanism. A fixed $4 fee is a
larger fraction of a smaller risk, so widening the stop shrinks the net loss
toward the gross — and it does, monotonically:

| Stop (× ATR) | Net | Commission | Gross | *(published gross)* |
|---|---|---|---|---|
| 1.0 | −0.1049 | 0.0492 | **−0.0513** | *−0.0256* |
| 1.5 | −0.0634 | 0.0323 | **−0.0286** | *−0.0088* |
| 2.5 | −0.0352 | 0.0193 | **−0.0137** | *−0.0002* |
| 4.0 | −0.0187 | 0.0127 | **−0.0061** | *+0.0018* |

The gross expectancy is negative at every stop width. The widest stop does not
reach zero from below; it approaches it and stays underneath.

---

## The best result sits exactly on the noise ceiling

With 10,384 two-sided trials, the expected largest |t| under a pure-noise null
is **3.90**.

The largest positive t in the entire search is **3.99**.

That is not a near miss — it is the definition of finding nothing. The Deflated
Sharpe Ratio says the same thing about the top config directly:

| Best config | `orb[entry 300–390, or=15, long] stop=4.0 target=3R` |
|---|---|
| Trades | 1,845 |
| Expectancy | +0.1281 R |
| Block-bootstrap 95% CI (by session) | **[+0.0632, +0.1929]** |
| Sharpe | 0.0926 |
| Noise benchmark SR₀ at 8,656 trials | **0.0892** |
| **Deflated Sharpe** | **0.5603** (needs > 0.95) |

Its Sharpe exceeds the noise benchmark by 0.0034. After accounting for how hard
we looked, the probability its true Sharpe beats noise is 56% — a coin flip.
Its bootstrap CI does not exclude values well below the target either.

---

## What positive signal exists is the bull market, not an edge

| Side | configs | median expectancy | share positive |
|---|---|---|---|
| long only | 2,877 | −0.0236 R | 31.9% |
| both | 2,782 | −0.0478 R | 19.2% |
| short only | 2,840 | −0.0724 R | **1.9%** |

**19 of the top 20 configurations are long-only.** Holding side and every other
parameter fixed, switching long to short costs **0.0385R** per trade.

NQ went from roughly 4,500 to 20,000 across this sample. A long-biased intraday
rule inherits that drift. It is beta, and a prop-firm account that must be flat
by the close cannot hold it anyway.

---

## What it does to an actual account

The best configuration, run through the Lucid 50k Flex simulator over the full
8.6-year development window (risking ~$467/trade, 23% of the $2,000 drawdown):

> Passes the evaluation. Takes **one** payout. **Net $617.** Then breaches the
> drawdown and the account is dead.

Identical under both excursion orderings, so bar ambiguity is not what decided
it. Eight and a half years of trading the single best rule out of 8,656, for
$617 and a blown account.

---

## Families tested

Opening-range breakout, momentum, mean reversion, range breakout, VWAP
reversion, prior-day break, and time-of-day — 8,656 configurations over stop
width (1–4 ATR), target (1–3R), entry window, and side.

> **Corrected 2026-08-26.** The gross column below is now measured per trade.
> The published version, shown in italics, was estimated from a median ATR and
> was too generous for every family by 0.011 to 0.025 R.

| Family | configs | median net | commission | **median gross** | *published gross* | best |
|---|---|---|---|---|---|---|
| prior_day_break | 240 | −0.0199 | 0.0243 | **+0.0065** | *+0.0229* | +0.1161 |
| orb | 672 | −0.0201 | 0.0246 | **+0.0046** | *+0.0196* | +0.1464 |
| momentum | 2,880 | −0.0397 | 0.0266 | **−0.0120** | *+0.0031* | +0.0912 |
| range_breakout | 960 | −0.0445 | 0.0263 | **−0.0155** | *−0.0016* | +0.1167 |
| time_of_day (null control) | 256 | −0.0533 | 0.0272 | **−0.0252** | *−0.0134* | +0.0696 |
| vwap_reversion | 720 | −0.0597 | 0.0270 | **−0.0321** | *−0.0191* | +0.0086 |
| mean_reversion | 2,880 | −0.0599 | 0.0265 | **−0.0323** | *−0.0212* | +0.0830 |
| **ict_silver_bullet** | 1,728 | **−0.0975** | 0.0295 | **−0.0640** | *−0.0394* | +0.0885 |

Only two families have positive gross expectancy, and both are barely positive
rather than the comfortable margin originally reported.

`time_of_day` reads no price at all and was included as a near-null control. It
is not the worst family — **four** price-reading families do worse than a rule
that enters at a fixed clock time, which is the clearest single sign that
nothing here is reading a signal.

---

## ICT Silver Bullet: worst of the eight, and its own filters are why

ICT gets its own section because it is the family this project was originally
built around, and because the result is specific rather than merely negative.

`ict_lab` implements this strategy correctly — one of its trades was verified
bar by bar against raw NQ. It is simply too slow to sweep: its own Phase 4
frequency diagnostic ran a full 60-minute timeout on the development window and
produced no output at all. So the detectors were re-expressed vectorized, tested
against `ict_lab`'s own swing detector on real bars, and put through the
identical grid, cost model and correction as everything else.

**It finishes last of eight on gross expectancy** — **−0.0640R** before costs
(corrected; −0.0394R as published), two and a half times worse than the null
control that reads no price. This is not a transaction-cost problem; the signal
is actively harmful.

Part of the correction is ICT-specific and independent of the commission error:
its London killzone is 03:00–04:00 ET, entirely outside RTH, so every entry it
produces there was charged one tick of stop slippage when overnight liquidity is
twelve times thinner. It now pays two. ICT's own code is unchanged.

**Its two distinctive filters both make it significantly worse.** Paired
comparisons hold killzone, sweep lookback, side, stop and target fixed and flip
one gate:

| Gate turned on | paired configs | mean change | t | share made worse |
|---|---|---|---|---|
| `require_mss` | 55 | **−0.0178 R** | **−3.78** | 73% |
| `require_displacement` | 122 | **−0.0134 R** | **−6.84** | 74% |

These t-statistics are not subject to the max-of-search inflation that governs
the headline result: they measure a consistent directional effect across many
paired configurations, not the best of many draws.

Consistent with that, **the best-performing ICT configurations are the ones with
ICT's distinctive machinery switched off**. All gates off gives a median of
−0.0749R; all gates on gives −0.1042R. The market-structure-shift and
displacement filters — the parts that make Silver Bullet *Silver Bullet* rather
than "buy a gap after a sweep" — subtract value monotonically.

Frequency is the other problem. Only **26% of ICT configurations trade often
enough to qualify** (≥200/year); the median is 142 trades/year. A prop-firm
evaluation cannot be completed at that rate with a per-trade edge anywhere near
zero.

The best ICT configuration reaches t = 2.11 against a noise ceiling of 3.90.
Not close.

**What this does not test.** The reimplementation covers the core sequence —
killzone, liquidity sweep, optional structure shift, optional displacement,
entry on the first fair-value gap. It does not cover `ict_lab`'s bias methods,
its sweep-universe presets, its entry-level and stop-type choices, or
next-liquidity targeting. A fuller sweep of those would need the original
engine to be fast enough to run, which it is not. What can be said is that the
core sequence is worse than no signal at all, and that adding more of ICT's own
gating makes it worse still.

---

## What this does and does not establish

**Does:** simple, well-known intraday rules on NQ **lose money before costs**
and roughly twice that after, across 8.6 years and 8,656 parameterizations, with
multiple-testing corrections applied and the holdout untouched.

**Does not:** prove no intraday edge exists. This grid covers price-derived
rules on 1-minute bars for one instrument. It says nothing about order-flow
data, cross-asset signals, options-implied information, event-driven trading,
or execution-quality edges — none of which are reachable with what we have.

**A caution about the search itself:** searching harder does not help. Every
configuration added raises the noise ceiling for all of them. The best result
here already sits at that ceiling, so a grid ten times larger would need a
proportionally better result just to stand still.

---

## Recommendation

**Do not buy a prop-firm account on the strength of anything in this grid.**

The bar from `findings/02` is +0.185R after costs, and the honest reading of
this search — after the corrections above — is that these rule families sit at
**−0.026R** before costs. The gap is not a tuning problem, and it is wider than
this document originally reported.

The cheapest remaining question is whether the ICT engine — the one strategy
family this search does not cover — does any better. Its own frequency
diagnostic has been running 54 minutes without producing output, and a
verified single trade in March 2019 suggests roughly 12 trades a year, far
below the 200/year this search required of everything else. That is worth
resolving before spending anything further.
