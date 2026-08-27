# 07 · The New York open, on the micro contract

**Date:** 2026-08-27
**Reproduce:** `python run_strategy_search.py && python run_opening_report.py`
**Data:** NQ 1-minute, 2016-01 → 2024-08 development window. Prices are the micro's too — MNQ is the same index at a tenth the multiplier and the same 0.25 tick, so no new data was bought.
**Pre-registration:** `research/search/registry.py`, `OPEN_GRID`, committed at `73cc257` before the run.
**Cumulative trials:** 19,596. Noise ceiling **4.051**.

---

## The question

"A 1:1 strategy that profits steadily on MNQ, traded 09:30–10:00 or 09:30–10:30."

The account owner then chose to search **any** payoff ratio and filter for a
**win rate above 50%** at report time, which is what "1:1" was actually for.
Commission was run at **three broker tiers** ($0.74 / $1.04 / $1.34 round turn)
rather than guessed at.

Every configuration counts as a trial whether or not it passes the win-rate
filter. 1,776 were genuinely new: 720 of the 2,496 combinations already existed
at the 09:30–10:30 window and were counted the first time.

---

## Result: 850 configurations clear the win-rate filter, none clear the bar

| Gate | Passing |
|---|---|
| ≥ 200 trades (opening windows) | 3,888 |
| **win rate > 50% and ≥ 200 trades/year** | **850** |
| expectancy ≥ +0.185 R, any contract | **0** |
| Deflated Sharpe > 0.95 | **0** |

The best configuration at any commission tier reaches **+0.1059 R** on MNQ
against a **+0.185 R** bar. Its Deflated Sharpe is 0.0072.

The shape the request asked for does exist. A 60% win rate at 204 trades a year
is real and repeatable. It is simply worth about a twentieth of an R.

---

## What the micro costs

Commission does not scale with point value and slippage does, so per dollar of
risk the micro is strictly worse:

| | median expectancy | share positive | best | ≥ +0.185 R |
|---|---|---|---|---|
| NQ | −0.0333 | **22.8%** | +0.1537 | 0 |
| MNQ $0.74 | −0.0507 | 14.4% | +0.1250 | 0 |
| MNQ $1.04 | −0.0661 | **9.7%** | +0.1059 | 0 |
| MNQ $1.34 | −0.0815 | 6.4% | +0.1003 | 0 |

**Moving from NQ to MNQ at $1.04 more than halves the share of
positive-expectancy configurations, 22.8% to 9.7%.** The three broker tiers
differ among themselves by a factor of 2.25 (14.4% against 6.4%), so the rate
you actually pay is not a detail.

The micro's advantage is granularity, not cost: one contract risks about $135 in
this window against $1,350 for the full-size, which is the difference between
tradeable and untradeable on a $2,000 drawdown. It buys access at a worse price.

**Win rate is unchanged between contracts** — 53.37% on both — because with a
4-ATR stop no trade lands close enough to zero for the commission gap to flip
it. That had to be checked by re-simulating rather than re-scoring.

---

## 09:30–10:00 against 09:30–10:30

| window | configs | median | share positive | **median win rate** | best |
|---|---|---|---|---|---|
| 09:30–10:00 | 1,248 | −0.0665 | 7.5% | **44.3%** | +0.1059 |
| 09:30–10:30 | 2,640 | −0.0660 | 10.8% | 40.1% | +0.0996 |

The half-hour window buys a higher win rate and gives back the same amount in
frequency. On expectancy the two are indistinguishable. **The 09:30–10:00 window
had never been tested before this round** — the grid's shortest window was the
opening hour — and the answer is that it is not different.

Testing it required 5- and 10-minute opening ranges. A 30-minute range first
becomes tradeable at minute 30, exactly when a 30-minute entry window closes, so
the old `or_minutes` would have produced silence and looked like a null result.
That is now a test.

---

## By family, priced on MNQ at $1.04

| family | configs | median NQ | median MNQ | median gross | median win rate | best MNQ |
|---|---|---|---|---|---|---|
| orb | 444 | −0.0057 | −0.0260 | +0.0061 | 44.8% | **+0.1059** |
| **opening_drive** | 384 | −0.0118 | −0.0370 | +0.0063 | **47.3%** | +0.0678 |
| prior_day_break | 156 | −0.0186 | −0.0490 | +0.0017 | 42.1% | +0.0996 |
| range_breakout | 372 | −0.0183 | −0.0512 | +0.0039 | 41.4% | +0.0903 |
| momentum | 816 | −0.0185 | −0.0518 | +0.0054 | 40.8% | +0.0595 |
| vwap_reversion | 324 | −0.0509 | −0.0869 | −0.0281 | 40.6% | −0.0059 |
| mean_reversion | 816 | −0.0622 | −0.0984 | −0.0408 | 38.9% | −0.0118 |
| gap_trade | 528 | −0.0598 | −0.1125 | −0.0299 | 40.9% | +0.0535 |
| overnight_level_break | 48 | −0.4067 | −0.4347 | −0.3898 | 23.5% | −0.2021 |

`opening_drive` — follow the direction of the first five minutes — is new this
round and carries the highest median win rate of any family here. Five of the
nine have positive median gross expectancy; none survive the micro's commission.

---

## The six best, re-simulated trade by trade on MNQ

| configuration | trades | win rate | payoff | expectancy | t | bootstrap 95% CI | DSR |
|---|---|---|---|---|---|---|---|
| `opening_drive` drive=5 thr=1.0 stop=4.0 **1:1** | 1,767 | 53.37% | 0.94:1 | +0.0369 | +1.50 | [−0.0090, +0.0855] | 0.0072 |
| `opening_drive` drive=5 thr=0.5 stop=4.0 1:0.75, 09:30–10:00 | 1,764 | **59.98%** | 0.71:1 | +0.0269 | +1.35 | [−0.0127, +0.0661] | 0.0037 |
| `opening_drive` drive=5 thr=1.0 stop=2.5 1:1, 09:30–10:00 | 1,784 | 52.91% | 0.92:1 | +0.0154 | +0.63 | [−0.0357, +0.0630] | 0.0004 |
| `opening_drive` drive=5 thr=0.5 stop=4.0 1:0.75 | 2,369 | 59.31% | 0.71:1 | +0.0143 | +0.81 | [−0.0179, +0.0474] | 0.0007 |

**Every bootstrap interval contains zero.** The largest t is +1.50 against a
noise ceiling of 4.051 and a conventional bar of 2. Deflated Sharpe peaks at
0.0072 where 0.95 is required.

A 60% win rate at 0.71:1 is a genuine, stable shape. It is also arithmetically
almost exactly break-even, which is what a 60% win rate at that payoff means.

---

## The one thing here that is different from every earlier round

`findings/06` established that the two best price rules from the first three
rounds inverted in 2024–2026, when NQ's cash-session drift fell from +2.72
points a day to +0.09. Every high-win-rate configuration in this round is also
long-only, so the same collapse should have taken them.

It did not.

| configuration | period | trades | win rate | expectancy | t |
|---|---|---|---|---|---|
| `opening_drive` stop=4.0 1:1 | development | 1,767 | 53.4% | +0.0369 | +1.50 |
| | **2024–26** | 399 | **53.9%** | **+0.0701** | +1.51 |
| `opening_drive` stop=4.0 1:0.75 | development | 1,764 | 60.0% | +0.0269 | +1.35 |
| | **2024–26** | 385 | **60.5%** | **+0.0561** | +1.26 |
| `orb` or=5 stop=4.0 1:1 | development | 2,161 | 52.3% | +0.0117 | +0.54 |
| | **2024–26** | 506 | **54.3%** | **+0.0793** | +1.92 |

**All three roughly doubled their expectancy in the window that killed S and A**,
and win rates held or improved. Whatever the opening drive is collecting, it is
not the intraday drift that disappeared.

**This is a weak check, not a clean test.** The NQ holdout was spent on three
other configurations in `findings/06`; that window has now been looked at more
than once. It is reported because the question is well posed and the contrast
with S and A is sharp, not because it validates anything.

---

## What an account actually does with it

The strict 1:1 configuration, MNQ at $1.04, three years to 2026-08 in
chronological order, withdrawing at the first legal opportunity, buying a new
account whenever one dies. 605 trades, 53.6% win rate, 0.98:1 payoff, +0.0577 R,
t=+1.48.

| contracts | risk/trade | accounts | passed | payouts | withdrawn | fees | **net** | **per year** |
|---|---|---|---|---|---|---|---|---|
| 1 | $135 (6.7% of drawdown) | 3 | **2** | 4 | $1,991 | $315 | +$1,676 | **$559** |
| 2 | $270 (13.5%) | 15 | 4 | 7 | $3,926 | $1,575 | +$2,351 | **$784** |
| 3 | $405 (20.2%) | 32 | 7 | 10 | $5,738 | $3,360 | +$2,378 | **$793** |

**Size does not buy income, it buys account churn.** Tripling the contract count
adds $234 a year and takes the accounts bought from 3 to 32 and fees from $315
to $3,360. Gross withdrawals rise 2.9× while net rises 1.4×; the difference is
evaluation fees. One contract is the only size with a pass rate above half.

At three contracts, 2026 is a losing year: 15 accounts bought, 2 passed,
−$396.

---

## Conclusion

**No.** Nothing here profits steadily on MNQ, and the reasons are specific
rather than general.

The requested shape is achievable — 60% win rate, 204 trades a year, in the
half-hour after the open. It clears +0.185 R by a factor of seven in the wrong
direction, every confidence interval spans zero, and the best Deflated Sharpe in
19,596 trials is 0.0072.

Three things this round did establish:

1. **The micro is the expensive way to trade this.** 2.6× the commission per
   dollar of risk at a typical rate, which more than halves the share of
   positive-expectancy configurations. It is worth paying only because $135 of
   risk fits a $2,000 drawdown and $1,350 does not.
2. **09:30–10:00 is not different from 09:30–10:30.** Higher win rate, same
   expectancy. The shorter window is not the edge people assume it is.
3. **`opening_drive` is the first family in four rounds that did not break when
   the intraday drift stopped.** That is worth remembering, and it is not
   evidence of an edge — it is one weak check on a spent holdout, at t=+1.50.

**ES remains sealed.** It is the only clean test this project still owns, and it
was not opened for this round.
