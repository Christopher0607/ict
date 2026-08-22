# 04 · Strategy search: 8,656 configurations, no survivors

**Date:** 2026-08-22
**Reproduce:** `python run_strategy_search.py && python run_search_report.py`
**Data:** NQ 1-minute, 2016-01 → 2024-08, 3,014,976 bars. Holdout never opened.
**Pre-registration:** `research/search/registry.py`, committed at `5f44f1a` before the search ran.

---

## Result

**Nothing clears the bar.** The best configuration in the grid reaches
**+0.128R** after costs against a required **+0.185R**, and that best result is
statistically indistinguishable from the best you would expect to find by
searching 8,656 coin flips.

The holdout stays sealed. There is nothing to validate on it.

| Gate | Passing |
|---|---|
| ≥ 200 trades | 8,608 / 8,656 |
| ≥ 200 trades/year | 8,499 |
| BH-FDR significant, q=0.10 | 5,684 |
| **expectancy ≥ +0.185R** | **0** |

---

## The 5,684 "significant" results are significantly *losing*

| | count |
|---|---|
| significant, positive expectancy | 341 |
| **significant, negative expectancy** | **5,343** |

The FDR gate is detecting transaction costs, not edges. This is worth stating
plainly because "5,684 of 8,656 configurations were statistically significant"
is the kind of sentence that could be quoted as encouraging.

---

## Why they lose: the entire deficit is commission

Adding the $4 round-turn commission back:

| | median across evaluable configs |
|---|---|
| Net expectancy | **−0.0502 R** |
| Commission | +0.0481 R |
| **Gross expectancy (before costs)** | **−0.0080 R** |

Before costs these rules are worth approximately nothing, and after costs they
lose approximately the commission. That is what an efficient market looks like
from the inside.

The stop-width breakdown confirms the mechanism exactly. A fixed $4 fee is a
larger fraction of a smaller risk, so widening the stop should shrink the net
loss toward the gross — and it does, monotonically:

| Stop (× ATR) | Net | Commission | Gross |
|---|---|---|---|
| 1.0 | −0.0978 | 0.0722 | −0.0256 |
| 1.5 | −0.0569 | 0.0481 | −0.0088 |
| 2.5 | −0.0291 | 0.0289 | −0.0002 |
| 4.0 | −0.0163 | 0.0181 | **+0.0018** |

At a 4-ATR stop the gross expectancy is +0.0018R — zero to three decimal places.

---

## The best result sits exactly on the noise ceiling

With 8,656 two-sided trials, the expected largest |t| under a pure-noise null is
**3.86**.

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

| Family | configs | median expectancy | best |
|---|---|---|---|
| prior_day_break | 240 | −0.0197 | +0.1161 |
| orb | 672 | −0.0200 | +0.1464 |
| momentum | 2,880 | −0.0397 | +0.0911 |
| range_breakout | 960 | −0.0445 | +0.1164 |
| time_of_day | 256 | −0.0533 | +0.0696 |
| vwap_reversion | 720 | −0.0597 | +0.0086 |
| mean_reversion | 2,880 | −0.0598 | +0.0830 |

`time_of_day` reads no price at all and was included as a near-null control. It
is not the worst family. Two price-reading families do worse than a rule that
enters at a fixed clock time — which is the clearest single sign that nothing
here is reading a signal.

---

## What this does and does not establish

**Does:** simple, well-known intraday rules on NQ have no edge before costs and
lose the commission after, across 8.6 years and 8,656 parameterizations, with
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
this search is that these rule families sit at approximately 0.000R before
costs. The gap is not a tuning problem.

The cheapest remaining question is whether the ICT engine — the one strategy
family this search does not cover — does any better. Its own frequency
diagnostic has been running 54 minutes without producing output, and a
verified single trade in March 2019 suggests roughly 12 trades a year, far
below the 200/year this search required of everything else. That is worth
resolving before spending anything further.
