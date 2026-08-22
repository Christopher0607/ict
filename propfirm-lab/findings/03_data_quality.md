# 03 · What the data actually is

**Date:** 2026-08-22
**Reproduce:** `python run_data_report.py`
**Spent:** ~$36 of the Databento credit (see "The budget ran out" below)

---

## Headline

Three things about this dataset differ from what its advertised availability
window implies, and all three would have changed results downstream:

1. **16 years of history is really 8.6 years of usable history.**
2. **The 1-minute stop-versus-target ambiguity is a non-issue** — 0.02%–0.14%,
   measured, not assumed. The 1-second data bought to quantify it was never
   needed.
3. **A back-adjustment I diagnosed as broken was correct.** Recorded here
   because the wrong diagnosis is more instructive than the right answer.

---

## 1 — Usable history is 2016+, not 2010+

`GLBX.MDP3` advertises 2010-06-06 onward. Its early minute coverage does not
match that. Measured on NQ.v.0:

| Period | Sessions with a full RTH day | Verdict |
|---|---|---|
| 2010–2012 | 22% / 28% / 43% | unusable |
| 2013–2015 | 83% / 83% / 89% | **seasonally biased** |
| 2016–2026 | 98.8%–99.6% | usable |

Two properties decide what to do about it:

**It is the dataset, not the symbology.** The raw contracts are equally sparse
— `NQ.FUT` across *all* contracts averages 627 bars/day in June 2010 against
~1,380 for full coverage. CME's MDP 3.0 feed post-dates this period, so the
early history is reconstructed from a thinner source. In June 2010 only
*Mondays* have a complete cash session.

**The shortfall is always whole missing days, never partial ones.** In every
year, a session that exists carries its full 390 RTH minutes at both the median
and the 10th percentile. A thin session could be filtered at runtime; a
systematically absent set of sessions has to be cut by date.

And they are systematically absent: of the 116 sessions missing in 2013–2015,
**92 fall in November–February**. Research on that window would under-sample
exactly the months that carry the most volatility.

`MIN_USABLE_DATE = 2016-01-01`. Development data is **2016-01 → 2024-08, 8.6
years, 3.01M bars**. Holdout is 2024-08 → 2026-08, sealed. 2013–2015 remains
loadable for robustness checks, labelled seasonally incomplete.

---

## 2 — Bar ambiguity is negligible, and now measured

A bracket asks one question a 1-minute bar cannot answer: stop or target first,
when the bar's range spans both? Backtests settle it by convention — normally
"assume the stop won" — and then report the result as if the convention were a
measurement.

Measured over the full development window, 3.01M bars, entries every 15 minutes:

| Stop / target (pts) | Long | Short |
|---|---|---|
| 15 / 30 | 0.12% | 0.10% |
| 25 / 50 | 0.14% | 0.11% |
| 40 / 80 | 0.11% | 0.08% |
| 60 / 120 | 0.02% | 0.02% |

**At most 1 trade in 700 rests on the convention.** Even if every ambiguous case
resolved the wrong way, results move by ~0.1%. The rate falls as brackets widen,
as it must: spanning a 180-point bracket needs a 180-point minute.

This is measurable from 1-minute data alone, so it cost nothing. The 6 months of
1-second data budgeted to refine it would have been buying precision on a 0.1%
effect. It never downloaded — and that turned out not to matter.

It also retires a worry from `findings/01`: the `excursion_order` worst/best band
in `paths/parametric.py` is a much smaller source of uncertainty than assumed.

---

## 3 — A wrong diagnosis, recorded

Running the full 16 years, roll gaps came out at mean **+53.5 points**, peaking
at **+317.8**, and the June 2010 close back-adjusted from 1,826 to **5,275**.

I called this a bug and had a mechanism ready: Databento switches contracts at a
session boundary every time, so "first bar of the new contract minus last bar of
the old" is structurally vulnerable to swallowing an overnight move — and
overnight moves scale with the index, which fit the evidence that the estimator
had looked fine on 2010–2017 (gaps of −10 to +8) and wrong by 2026.

So I bought daily bars for every contract ($0.19) to re-derive each gap from days
where both contracts printed a close, removing the overnight term entirely.

**The corrected gaps were mean +53.1 against the estimate's +53.5.** Agreement to
within 1%. There was no bug.

The gaps are cost of carry. Annualised as `gap / price × 4`:

| Era | Policy rate | Annualised gap |
|---|---|---|
| 2010–2016 | ~0% | −0.27% to −0.74% |
| 2017–2019 | rising to 2.4% | +0.65% to +1.47% |
| 2020–2021 | back to ~0% | −0.39%, −0.20% |
| 2023–2026 | 4–5% | **+3.89% to +5.34%** |

That is the carry relationship, and it tracks the rate cycle across sixteen
years. A several-hundred-point quarterly spread on a 29,000 index is exactly
right, and a back-adjusted series drifting thousands of points from traded
prices is the correct behaviour of additive adjustment under positive carry.

**What actually was wrong was my plausibility check.** It flagged gaps beyond 1%
of the index — a threshold that is regime-dependent and therefore useless: the
same +300 points is absurd at zero rates and routine at 5%. It now checks implied
annualised carry against 10%, which no rate/dividend combination reaches.

The $0.19 was well spent. It did not find an error; it demonstrated there wasn't
one, which is worth more than the assumption I would otherwise have carried.

This is also the sharpest illustration of why the points-and-ticks rule exists:
the back-adjusted *level* is a price nobody ever paid, while every point
*distance* is exact.

---

## The budget ran out

The account had far less than the $125 the free-credit documentation describes.
`NQ ohlcv-1s` failed with `402 account_insufficient_funds` on a ~$3.17 monthly
chunk, after roughly $36 of downloads.

| Bought | Status |
|---|---|
| NQ `ohlcv-1m` 2010-06 → 2026-08 | ✅ complete, 4.80M bars |
| ES `ohlcv-1m` 2010-06 → 2026-08 | ✅ complete, 4.91M bars, sealed |
| NQ `ohlcv-1d` parent (all contracts) | ✅ 2010-06 → 2026-02, covers the whole dev window |
| ES `ohlcv-1d` parent | ❌ not fetched |
| NQ `ohlcv-1s` 6 months | ❌ not fetched |

**Nothing blocking is missing.** The research instrument is complete, the
out-of-sample instrument is complete and sealed, and exact roll spreads cover
the entire development window. ES daily is only needed when the holdout is
unsealed, and costs $0.30 then. The 1-second data is not needed at all, per
section 2.

Check the actual balance in the Databento portal before the next purchase — the
figure I planned against was wrong.
