# 05 · Round 3: 17,124 configurations, and what the model's confidence was actually selecting

**Date:** 2026-08-26
**Reproduce:**
```
python run_strategy_search.py          # resumable; findings/shards/
python run_search_report.py
python run_model_cache.py              # walk-forward predictions
python run_model_cache.py --shuffled   # the control
python run_confidence_economics.py
python run_confidence_economics.py --shuffled
```
**Data:** NQ 1-minute, 2016-01 → 2024-08, 3,014,976 bars. **Holdout never opened.**
**Cumulative trials:** 17,124. Noise ceiling (expected largest |t| under the null) **4.019**.

---

## What this round was for

`findings/04` searched 10,384 configurations of simple intraday rules and found
nothing. The predictability audit then asked the prior question — is there
*anything* forecastable in the return series — and found a real directional
edge worth 1/22 of what it costs to collect.

Three things were left open, and this round closes them:

1. **The second-round grid never finished.** 5,936 pre-registered non-ICT
   configurations — overnight momentum and reversion, gap trades, turn of
   month, day of week, compression breakouts, overnight levels — died at
   2,512/16,320 when the process was reaped at a session boundary.
2. **The audit's one economically positive result was a fixed-time hold**, and
   the backtester had no time exit. The only direction worth testing was the
   one the engine could not express.
3. **The audit's edge concentrated sharply in the model's own confidence** —
   +$48.75 per trade in the top 19% of |prediction| against +$10.10 overall.
   That was the single most promising number this project had produced.

---

## The search: 17,820 configurations, no survivors

The 5,936 pre-registered non-ICT configurations finished, plus 1,500 added this
round. The decision rule was committed before any of them ran.

| Gate | Passing |
|---|---|
| ≥ 200 trades | 17,646 / 17,820 |
| ≥ 200 trades/year | 14,872 |
| BH-FDR significant, q=0.10 | 10,799 |
| **expectancy ≥ +0.185 R** | **23** |
| ...and frequent enough | **1** |
| **Deflated Sharpe > 0.95** | **0** |

The one configuration to reach the final gate was
`model_confidence[horizon=120,quantile=0.8,regime=rth,side=long]stop=1.5R=99.0t=120`
at **+0.1973 R** over 2,472 trades. Its Sharpe is **0.0650 against a noise
benchmark SR₀ of 0.0805** — searching 17,820 times, a pure-noise process is
expected to produce a *better* Sharpe than this. Deflated Sharpe 0.196, needing
0.95. Rejected.

The other 22 that cleared +0.185 R trade 78 to 108 times a year against a
required 200. A prop-firm evaluation cannot be completed at that rate.

**10,271 of the 10,799 "significant" results are significantly losing**, against
342 significantly positive. As in round one, the FDR gate is detecting
transaction costs.

### Every family, with commission measured rather than estimated

| Family | configs | median net | commission | **median gross** | best net | share positive |
|---|---|---|---|---|---|---|
| **model_confidence** | 330 | **+0.0236** | 0.0132 | **+0.0443** | +0.2815 | **80.6%** |
| orb | 1,344 | −0.0014 | 0.0191 | +0.0087 | +0.1464 | 47.8% |
| prior_day_break | 480 | −0.0046 | 0.0184 | +0.0072 | +0.1537 | 40.6% |
| low_vol_long | 180 | −0.0170 | 0.0279 | +0.0068 | +0.1865 | 38.9% |
| momentum | 2,880 | −0.0397 | 0.0266 | −0.0120 | +0.0912 | 24.9% |
| day_of_week | 320 | −0.0387 | 0.0228 | −0.0137 | +0.1462 | 20.0% |
| range_breakout | 960 | −0.0445 | 0.0263 | −0.0155 | +0.1167 | 23.8% |
| gap_trade | 576 | −0.0571 | 0.0342 | −0.0234 | +0.1281 | 18.4% |
| *time_of_day (null control)* | 256 | −0.0533 | 0.0272 | −0.0252 | +0.0696 | 12.5% |
| vwap_reversion | 720 | −0.0597 | 0.0270 | −0.0321 | +0.0086 | 1.0% |
| mean_reversion | 2,880 | −0.0599 | 0.0265 | −0.0323 | +0.0830 | 8.9% |
| compression_breakout | 528 | −0.0687 | 0.0330 | −0.0326 | +0.1737 | 18.4% |
| turn_of_month | 288 | −0.0675 | 0.0214 | −0.0469 | +0.0136 | 0.4% |
| **ict_silver_bullet** | 1,728 | −0.0975 | 0.0295 | **−0.0640** | +0.0885 | 7.9% |
| eth_reversion | 1,728 | −0.1844 | 0.0764 | −0.1081 | +0.0003 | 0.1% |
| eth_momentum | 1,728 | −0.2034 | 0.0762 | −0.1250 | +0.0111 | 0.2% |
| eth_range_breakout | 576 | −0.2096 | 0.0746 | −0.1342 | −0.0168 | 0.0% |
| overnight_level_break | 144 | −0.4123 | 0.0218 | −0.4002 | −0.1387 | 0.0% |

**The overnight round is a rout.** The four ETH families are the four worst,
by a wide margin, and three of them have literally zero configurations with
positive expectancy out of 4,032. Commission alone is 0.076 R there against
0.027 R in RTH — the same $4 against a smaller ATR-scaled stop, plus two ticks
of slippage instead of one. The 71.6% of bars nobody had tested turn out to be
untradeable, which is a useful thing to have established rather than assumed.

`time_of_day` reads no price at all. **Nine of the eighteen price-reading
families do worse than it**, up from four in round one.

---

## The confidence result, taken apart

This is the part worth reading. The headline from the audit was:

| RTH, h=60 | trades | net $/trade | t |
|---|---|---|---|
| all | 641,832 | +$10.10 | +1.30 |
| top 50% | 320,916 | +$27.48 | +2.58 |
| **top 19%** | 128,367 | **+$48.75** | **+3.50** |
| top 9% | 64,184 | +$50.31 | +2.96 |
| top 1% | 6,419 | +$44.09 | +0.95 |

Rising then flattening is what a real signal looks like. Three things had to be
checked before it could mean anything, and each changed the reading.

### 1. The threshold had to stop seeing the future

Ranking |prediction| over the whole sample is fine for a diagnostic and wrong
for a strategy: knowing today's forecast lands in the top 19% of 2016–2024
requires 2024. Rebuilt so each test month is ranked against **earlier test
months only** (`model_signal.causal_threshold`):

| RTH h=60, top 19% | sign acc | mean $/trade | median | P(win) | t |
|---|---|---|---|---|---|
| full-sample threshold | 0.5619 | +$49.44 | +$51 | 55.4% | +3.51 |
| **causal threshold** | 0.5633 | **+$52.28** | +$56 | 55.6% | **+3.48** |

It survives. The hindsight in the threshold was not what produced it.

Entry timing was also wrong in the original diagnostic: the regression target
is close-to-close, so scoring it directly has the account transacting at a
price it only knows once the bar is finished. Moving entry to the next bar's
open — the rule the backtester enforces — costs about $0.40 a trade at these
horizons. Small, and it was there.

### 2. The level is drift. The slope is not.

The control: same bars, same features, whole **sessions of the target
permuted**. Everything real about the feature-to-target link is destroyed;
everything structural survives.

| RTH h=60 | real | shuffled |
|---|---|---|
| all bars | 0.5242 acc, +$9.84 | **0.5274 acc, +$7.15** |
| top 50% | 0.5409, +$27.10 | 0.5269, +$4.38 |
| top 19% | 0.5619, +$49.44 | 0.5183, +$2.77 |
| top 9% | 0.5687, +$50.71 | 0.5168, +$16.08 |

| RTH h=120 | real | shuffled |
|---|---|---|
| all bars | 0.5343 acc, +$24.71 | **0.5321 acc, +$33.46** |
| top 50% | 0.5605, +$56.17 | 0.5372, +$7.38 |
| top 19% | 0.5936, +$97.29 | 0.5257, −$2.76 |
| top 9% | 0.6107, +$110.23 | 0.5103, −$7.67 |

Two separate readings, and they point in opposite directions:

**The unconditional number is entirely structural.** Shuffling reproduces it
exactly — 0.5274 against 0.5242, +$7.15 against +$9.84. Sign accuracy over all
bars carries the same t-statistic in the control as in the real data (+7.36 vs
+6.54). A model whose weights contain no information still collects this,
through its intercept: NQ drifted from 4,500 to 20,000 across the sample, the
target has a positive mean, and predicting that mean is enough to be long most
of the time and right slightly more than half of it.

**The sorting is real.** In the real data accuracy climbs monotonically with
confidence; in the control it *falls*, decaying toward 50% as the filter
tightens. The shuffle cannot manufacture the slope. Something in the features
genuinely orders these bars.

### 3. What the sorting selects is "the market is calm, be long"

| RTH h=60, top 19% | value | sample |
|---|---|---|
| share of predictions long | **90.5%** | 67.0% |
| median ATR at selected bars | **5.61** | 7.43 |
| median minutes into RTH | 265 | 188 |

Split by side, the short leg is not carrying anything:

| RTH h=60, top 19% | n | sign acc | mean $ | median $ | P(win) | t |
|---|---|---|---|---|---|---|
| long | 98,743 | 0.5656 | +$52.41 | +$56 | 56.0% | **+3.38** |
| short | 10,397 | 0.5162 | +$51.01 | +$26 | 51.3% | +0.94 |

So the model has learned that when volatility is low the drift-to-volatility
ratio is high, and it should be long. Which raises the obvious question.

### 4. A one-line rule with nothing fitted in it gets most of the way there

Same bars, same costs, no execution model:

| RTH h=60 | n | mean $/trade | median | P(win) | t |
|---|---|---|---|---|---|
| model, all bars | 641,832 | +$9.84 | +$21 | 51.8% | +1.27 |
| always long, all bars | 641,832 | +$1.12 | +$36 | 53.0% | +0.12 |
| **model, top-19% confidence** | 109,140 | **+$52.28** | +$56 | 55.6% | **+3.48** |
| always long, *those same bars* | 109,140 | +$40.84 | +$51 | 55.2% | +2.70 |
| **long when ATR is below its own trailing average** | 121,911 | **+$36.08** | +$41 | 55.2% | **+3.29** |

| RTH h=120 | n | mean $/trade | median | P(win) | t |
|---|---|---|---|---|---|
| model, top-19% confidence | 99,536 | +$100.91 | +$131 | 58.9% | +3.35 |
| always long, those same bars | 99,536 | +$63.67 | +$116 | 58.2% | +2.08 |
| long when ATR is calm | 102,382 | +$70.74 | +$96 | 58.3% | +2.95 |

Read the second row of each block first. **Being long on the model's own
confident bars, ignoring its direction entirely, captures 78% of its return at
h=60 and 63% at h=120.** The directional call is worth about $11 and $37 a
trade respectively; everything else is bar selection.

And the bar selection is not exotic. A rule with no features, no fit and no
walk-forward — *long whenever ATR sits below its trailing 1440-bar average* —
earns $36.08 at t=+3.29 against the model's $52.28 at t=+3.48. The two
selections overlap on only 44% of bars, so this is not the same trades by
another name; it is a different, simpler route to the same place.

That effect has a name outside this project. Low realised volatility predicting
higher risk-adjusted equity returns is the volatility-managed-portfolio result,
and finding it intraday in NQ is a rediscovery, not a discovery.

### 5. The overnight session is decisively negative

| ETH h=30 | n | mean $/trade | median | P(win) | t |
|---|---|---|---|---|---|
| all bars | 1,607,704 | −$12.61 | −$9 | 48.1% | **−11.68** |
| top 19% | 236,115 | −$11.47 | −$4 | 48.7% | −5.02 |

Sign accuracy overnight is *above* 50% at every horizon and the money is still
gone: two ticks of slippage instead of one, on moves that are no larger. This
is the clearest illustration in the project of the difference between a
statistical edge and a tradeable one.

---

## Under a real execution model, the confidence family is the only one that holds up

`model_confidence` is **the only family of nineteen with a positive median**
(+0.0236 R, 80.6% of configurations positive). Every other family is negative at
the median. Whatever the confidence sorting is doing, it survives non-overlapping
entries, a stop, and flat-by-the-close — which the diagnostic tables above did
not have to.

It also beats the one-line rival it was tested against. `low_vol_long` —
long whenever ATR sits below its own trailing average, nothing fitted — has a
median of **−0.0170 R** and only 38.9% positive. On the raw diagnostic the two
looked close ($36.08 against $52.28 a trade). Through the backtester they
separate: the model's bar selection is worth something the volatility filter
alone is not.

That is the strongest positive statement this project can make, and it is still
not enough. The best `model_confidence` configuration was rejected by Deflated
Sharpe, and the family was designed after seeing the results it was designed to
exploit.

### Time exits, paired

Adding a time exit to the two families with the best round-one gross expectancy,
holding the signal parameters, stop and target fixed:

| Family | hold | pairs | mean change | t | share improved |
|---|---|---|---|---|---|
| orb | 30 bars | 84 | **−0.0122 R** | **−6.22** | 21% |
| orb | 60 bars | 84 | **−0.0047 R** | **−5.42** | 18% |
| prior_day_break | 30 bars | 30 | **−0.0093 R** | **−3.38** | 23% |
| prior_day_break | 60 bars | 30 | −0.0024 R | −1.41 | 50% |

These t-statistics are not subject to max-of-search inflation — they measure one
consistent effect across many pairs, not the best of many draws. **A time exit
makes a bracket strategy worse**, and consistently so.

The pairing only covers the stop widths present in both sets (1.5 and 2.5 ATR),
which is exactly the regime where a time exit should hurt: `f.atr` is a
*one-minute* ATR, a random walk covers about √N of them over an N-bar hold, and
a 1.5-ATR stop on a 60-bar hold is a coin flip on the first few minutes rather
than a risk limit. The 8.0 and 16.0 ATR stops added this round exist only in the
time-exit families, so they are not in this comparison — and they are where the
time-exit configurations that ranked well actually live.

### A stop wide enough to hold is too wide for the account

At 16 ATR and a median RTH ATR near 5.6 points, one NQ contract risks
**$1,792 a trade — 90% of Lucid's entire $2,000 drawdown**. Any configuration
surviving at that stop width has to be traded in micros. MNQ pays roughly $1.20
round turn against $2 of point value, versus $4 against $20: **about 50% more
commission per unit of risk**. A result that clears +0.185 R on NQ does not
clear it on the instrument you would actually have to trade it on.

---

## Corrections to `findings/04`

Four numbers in `findings/04` were wrong, computed from the completed run
rather than estimated. **Every conclusion in that document survives — all four
errors run the same way, and that way is "too kind".**

### 1. Commission was estimated from a median ATR, and overstated by 1.8×

The $4 round turn was converted to R using an all-session median ATR of 2.77.
Every family in that document trades RTH only, where ATR is larger, and the mean
of a ratio is not the ratio of the medians. Measured per trade across the same
10,336 evaluable configurations:

| | published | **measured** |
|---|---|---|
| commission | 0.0481 R | **0.0265 R** |
| gross expectancy (median) | −0.0080 R | **−0.0259 R** |

### 2. "The entire deficit is commission" is false

Commission is **48%** of the median loss. The other half is the signal losing
money before a fee is charged. The original sentence — "before costs these rules
are worth approximately nothing" — was too generous by a factor of three.

### 3. The stop-width table's punchline is gone

| Stop (× ATR) | net | commission | **gross** | *published gross* |
|---|---|---|---|---|
| 1.0 | −0.1049 | 0.0492 | **−0.0513** | *−0.0256* |
| 1.5 | −0.0634 | 0.0323 | **−0.0286** | *−0.0088* |
| 2.5 | −0.0352 | 0.0193 | **−0.0137** | *−0.0002* |
| 4.0 | −0.0187 | 0.0127 | **−0.0061** | *+0.0018* |

Gross expectancy is negative at every stop width. The widest stop approaches
zero from below and stays there; it was reported as reaching +0.0018 R, "zero to
three decimal places".

### 4. ICT was flattered a second, independent way

Its London killzone is 03:00–04:00 ET — entirely outside RTH — and every entry
it produces lands there. Until regime-dependent slippage was introduced, those
trades were charged one tick like everything else, though overnight liquidity is
twelve times thinner (median 1-minute volume 57 against 707). ICT's own code is
unchanged.

| | published | **measured** |
|---|---|---|
| median net | −0.0868 R | **−0.0975 R** |
| median gross | −0.0394 R | **−0.0640 R** |

It still finishes last of the original eight, by a wider margin than reported.

---

## Method notes

**Everything above is on the development window.** `research/data/holdout.py`
keeps 2024-08-22 onward and the whole ES symbol sealed behind `unseal(reason)`.
Neither was touched in this round.

**`model_confidence` and `low_vol_long` were not pre-registered.** They were
designed after the confidence table existed — the horizon, the regime and the
quantile range were all chosen with the result in view. That is stated in
`registry.py`, in `model_signal.py` and here, because it is the reason a
development-set pass for either would be a candidate for the holdout rather
than a finding. Everything else in the 17,124 was committed before it ran.

**Trials are counted cumulatively and never reset.** 8,656 → 10,384 → 16,320 →
17,124, ceiling 3.855 → 3.900 → 4.008 → 4.019. Adding a family raises the bar
for every family, including the ones already run, which is why each round
re-runs everything rather than carrying results forward.

**Standard errors are clustered on the session.** Trades inside one day are not
independent, and the naive standard error inflates every t-statistic in the
search. At h=120 in RTH there are roughly 248 overlapping observations per
session; the clustered SE is why 539,052 observations produce t=+1.61 rather
than something enormous and meaningless.

**Commission is measured per trade, not estimated from a median ATR.** See the
corrections section.

**The search is resumable.** Results checkpoint to `findings/shards/` every 250
configurations, and the merged output is reindexed to enumeration order so it
cannot depend on where a run was interrupted — verified by comparing 408
configurations run straight through against the same 408 across four
interrupted runs, row for row.

---

## The holdout was opened, and it settled it

At the account owner's explicit direction, the sealed NQ window
(2024-08-22 → 2026-08-21, 707,113 bars) was opened once, on three configurations
chosen before it was opened. **No parameter was changed after reading the
result.** Full month-by-month tables are in `findings/06_holdout.md`.

| | development | **holdout** | verdict |
|---|---|---|---|
| **S** `orb` late-session, stop 8.0, time exit 60 | +0.0644 R, 54.4% win | **−0.0508 R, 46.5% win**, t=−1.71 | CI excludes the dev estimate — **broken** |
| **A** `orb` all-day, stop 4.0, target 3R | +0.1159 R, 41.5% win | **−0.0777 R, 35.4% win**, t=−1.51 | CI excludes the dev estimate — **broken** |
| **M** `model_confidence` h=120 q=0.8 | +0.1973 R, 19.9% win | +0.0955 R, 20.1% win, t=+0.83 | CI contains zero — **not established** |

S and A were the two best pre-registered price rules. Both went negative, and in
both cases the bootstrap CI on the holdout excludes the development estimate.
That is not sampling noise; the edge is not there.

### What happened, and it is the most useful thing in this document

| | NQ annualised | **RTH intraday drift** |
|---|---|---|
| development 2016–2024 | +18.5% | **+2.72 points/day** |
| holdout 2024–2026 | **+21.4%** | **+0.09 points/day** |

**The index rose faster while intraday drift fell thirty-fold.** The gains moved
overnight. This is the overnight-return anomaly, and the development window
happens to be a period when NQ also drifted during the cash session.

The arithmetic closes exactly. S holds 60 minutes, takes 1.79 trades a day, so
its intraday exposure is roughly 35% of the RTH session. Losing 2.63 points a
day of tailwind at 35% exposure and $20 a point is **$18.4 a day**. S's
development edge was $17.3 a day. **The tailwind it lost is its entire edge.**

Every one of the top fourteen configurations by profit-to-drawdown is long-only.
That was never a coincidence, and a prop-firm account that must be flat by the
close cannot hold the part of the drift that survived.

### The pre-registered rule was right

Zero of 17,820 survived, so the rule said trade none of them. The holdout agrees.
Had the rule been relaxed — had +0.185 R been softened, or Deflated Sharpe
waived for the one configuration that reached it — the result would have been an
account funded on a strategy that proceeded to lose money for two years.

### A flaw in my own account modelling, found the same way

The account economics reported to the owner before the holdout was opened used a
**session-level bootstrap**: resample whole trading days with replacement, build
synthetic account paths, run the Lucid lifecycle. Against the same five years
replayed in **actual chronological order**:

| S, $150 a trade | session bootstrap | **real chronological path** |
|---|---|---|
| evaluation pass rate | 76.9% | **25%** (1 of 4 accounts) |
| net per account | $881 | **$422** |
| P(profitable over 10 years) | 100% | — |
| 5-year net | — | **+$1,689**, all of it earned in 2022–2023 |

Accounts bought in 2024, 2025 and 2026 **each failed the evaluation**.

The bootstrap destroys serial structure. Bad periods cluster, and for a strategy
whose edge depends on a market regime, every resampled path mixes good and bad
sessions evenly and no path ever contains "two straight years where nothing
works" — which is what actually happened. **A session-level bootstrap
systematically overstates account survival for any regime-dependent strategy.**
Report the chronological replay alongside it, or do not report it.

---

## What this round establishes

**Does:** across 17,820 configurations, nineteen families, 8.6 years of
development data and a sealed two-year holdout, no simple price rule, no ICT
sequence, no overnight family, and no walk-forward model signal on NQ 1-minute
bars produces a tradeable edge at the +0.185 R this account requires. The
positive results that did appear were intraday drift, and that drift stopped.

**Does not:** prove no intraday edge exists. This covers price-derived rules and
a linear model on 1-minute bars for one instrument. Order flow, cross-asset
signals, options-implied information, event-driven trading and execution-quality
edges remain untested and unreachable with what we have.

**The holdout is spent.** ES remains sealed, and it is now the only clean test
this project owns. Nothing above may be re-tuned against the NQ holdout.

## Recommendation

**Do not buy a prop-firm account on the strength of anything in this document.**

The single most useful sentence in three rounds of searching: every strategy
that looked profitable here was long-only, and was collecting intraday drift
rather than reading a signal. When that drift went to zero — while the index
kept rising — all of them went negative together. That is what to check first
about any future candidate.
