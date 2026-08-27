# 06 · The holdout, opened once

**Date:** 2026-08-26
**Reproduce:** `python run_holdout_validation.py`
**Window:** NQ 2024-08-22 → 2026-08-21, 707,113 one-minute bars, sealed since Phase B.
**Status:** **SPENT.** ES remains sealed and is now the only clean test this project owns.

---

## Why it was opened

The pre-registered rule was: open the holdout only if a configuration survives
the decision gates on the development set. **Zero of 17,820 survived**
(`findings/05`), so under that rule it should have stayed closed.

It was opened anyway, at the account owner's explicit and repeated direction,
after being told what the rule said. The reason recorded in the run log:

> Forward test of three pre-selected NQ configurations at the account owner's
> explicit direction. Zero configurations survived the pre-registered decision
> rule, so this cannot confirm a survivor; it reports how S, A and M behave on
> data none of them were fitted to. No parameter may be changed after reading
> the result.

**No parameter was changed after reading the result.** That commitment is the
only thing that makes a spent holdout worth anything, and it was kept.

Three configurations were tested, so read the t-statistics against three trials,
not one. All three were fixed before the unseal.

---

## Result

| | development | **holdout** | Δ | verdict |
|---|---|---|---|---|
| **S** `orb` 14:30–16:00, or=30, stop 8.0, target 2R, time exit 60 | +0.0644 R, 54.4% win | **−0.0508 R, 46.5% win** | −0.1152 R | t=−1.71, CI **excludes** +0.0644 — **broken** |
| **A** `orb` all day, or=60, stop 4.0, target 3R | +0.1159 R, 41.5% win | **−0.0777 R, 35.4% win** | −0.1936 R | t=−1.51, CI **excludes** +0.1159 — **broken** |
| **M** `model_confidence` h=120 q=0.8 stop 1.5 time exit 120 | +0.1973 R, 19.9% win | +0.0955 R, 20.1% win | −0.1018 R | t=+0.83, CI contains 0 — **not established** |

At $150 risk per trade over the two years: S **−$3,608**, A **−$7,051**,
M +$8,978. Maximum drawdown: S $4,117, A $9,281, M $6,129 — **every one of
them larger than Lucid's entire $2,000 limit.**

Profitable months: S 7 of 25, A 10 of 25, M 14 of 25.

S and A were the two best pre-registered price rules in the entire search. Both
inverted. The bootstrap confidence interval on the holdout excludes the
development estimate in both cases, which is what distinguishes a broken edge
from an unlucky sample.

| 月份 | S 笔数 | S 胜率 | S 净额 | A 净额 | M 净额 |
|---|---|---|---|---|---|
| 2024-08 | 4 | 50.0% | $-31 | $-206 | $+657 |
| 2024-09 | 18 | 38.9% | $-342 | $-1,383 | $-528 |
| 2024-10 | 19 | 57.9% | $+22 | $+311 | $-1,819 |
| 2024-11 | 17 | 35.3% | $-252 | $-930 | $-2,450 |
| 2024-12 | 20 | 45.0% | $-228 | $-226 | $+3,324 |
| 2025-01 | 16 | 50.0% | $-63 | $-257 | $-746 |
| 2025-02 | 20 | 55.0% | $-14 | $-1,111 | $-1,222 |
| 2025-03 | 22 | 45.5% | $+44 | $-917 | $+145 |
| 2025-04 | 20 | 50.0% | $+267 | $+1,797 | $+3,602 |
| 2025-05 | 31 | 32.3% | $-1,324 | $-3,162 | $+1,293 |
| 2025-06 | 22 | 36.4% | $-418 | $-1,198 | $+1,394 |
| 2025-07 | 18 | 27.8% | $-674 | $-1,902 | $-1,676 |
| 2025-08 | 19 | 47.4% | $+51 | $+244 | $+2,764 |
| 2025-09 | 24 | 70.8% | $+960 | $+491 | $+768 |
| 2025-10 | 27 | 40.7% | $-387 | $+474 | $+3,147 |
| 2025-11 | 18 | 44.4% | $-318 | $+1,235 | $-1,676 |
| 2025-12 | 17 | 58.8% | $+155 | $-144 | $-476 |
| 2026-01 | 15 | 33.3% | $-615 | $-834 | $+254 |
| 2026-02 | 17 | 35.3% | $-418 | $-165 | $-1,290 |
| 2026-03 | 15 | 46.7% | $-218 | $+0 | $+2,382 |
| 2026-04 | 24 | 75.0% | $+1,386 | $+2,128 | $-2,804 |
| 2026-05 | 23 | 52.2% | $-185 | $-127 | $+920 |
| 2026-06 | 22 | 45.5% | $-292 | $+282 | $+2,100 |
| 2026-07 | 17 | 29.4% | $-481 | $-1,578 | $-2,679 |
| 2026-08 | 8 | 62.5% | $-235 | $+126 | $+3,595 |

---

## Why: the intraday drift stopped

| | NQ annualised | **RTH intraday, cumulative** |
|---|---|---|
| development 2016–2024 | +18.5% | **+2.72 points/day** (+6,051 pts over 2,228 days) |
| holdout 2024–2026 | **+21.4%** | **+0.09 points/day** (+47 pts over 515 days) |

**The index rose faster while the cash-session drift fell thirty-fold.** The
gains moved overnight — the overnight-return anomaly, well documented in equity
indices, and the development window happens to be a stretch when NQ also drifted
during the day.

The arithmetic closes almost exactly. S holds 60 minutes and takes 1.79 trades a
day, so it is exposed for roughly 35% of the 390-minute session. Losing 2.63
points a day of tailwind, at 35% exposure and $20 a point, is **$18.4 a day**.
S's development edge was **$17.3 a day**.

**The tailwind it lost is its entire edge.** It was never reading a signal.

Every one of the top fourteen configurations by profit-to-drawdown across the
whole search is long-only. That was never a coincidence, and an account that
must be flat by the close cannot hold the part of the drift that survived.

---

## Five years, replayed in the order it happened

S over 2021-08-22 → 2026-08-21, chronological, no resampling. Three of the five
years are the development window S was selected from.

| | all 5 years | first 3 (in-sample) | last 2 (holdout) |
|---|---|---|---|
| trades | 1,180 (237/yr) | 707 | 473 |
| **win rate** | **51.02%** | 54.0% | **46.5%** |
| **payoff ratio** | **1.05 : 1** | 1.13 : 1 | **0.93 : 1** |
| profit factor | 1.098 | — | — |
| expectancy | +0.0235 R (t=**+1.23**) | +0.0733 R | −0.0508 R |
| net at $150/trade | **+$4,167** | +$7,775 | **−$3,608** |
| max drawdown | $4,589 | — | — |

Average win $77.52, average loss $73.53. Over five years t=+1.23 — not
significant — and every dollar of it was earned in the three years S was
selected from.

**The nominal 2R target is a fiction.** Exit distribution on the development
window: stop 12.5%, **target 1.9%**, time or close **85.6%**. The strategy never
reaches its target; it is a 54%-win-rate, 1.05:1 coin flip whose edge is the
drift accumulated during a 60-minute hold. Out of sample the win rate fell to
46.5% and the payoff to 0.93:1 — both at once, which is why expectancy inverted.

### The account chain: buy, blow, re-buy

Lucid 50k Flex, $150 a trade, withdraw at the first legal opportunity, buy a new
account whenever one dies. Real chronological order.

| year | accounts bought | passed eval | payouts | withdrawn | fees | net |
|---|---|---|---|---|---|---|
| 2021 | 1 | **1** | 0 | $0 | $105 | −$105 |
| 2022 | 0 | 0 | 1 | $585 | $0 | **+$585** |
| 2023 | 0 | 0 | 3 | $1,523 | $0 | **+$1,523** |
| 2024 | 1 | **0** | 0 | $0 | $105 | −$105 |
| 2025 | 1 | **0** | 0 | $0 | $105 | −$105 |
| 2026 | 1 | **0** | 0 | $0 | $105 | −$105 |
| **total** | **4** | **1** | **4** | **$2,109** | **$420** | **+$1,689** |

Payouts: mean $527, median $514, max $585 — the cap is
min(50% of cycle profit, $2,000) and requesting one snaps the loss limit to
$50,100, so a payout leaves roughly $350 of room. That is 2.3 losing trades.

**Accounts bought in 2024, 2025 and 2026 each failed the evaluation.** Every
dollar came from one account bought in 2021.

---

## A flaw in the account modelling, caught the same way

Before the holdout was opened, the account economics reported to the owner used a
**session-level bootstrap**: resample whole trading days with replacement, build
synthetic paths, run the lifecycle. Against the chronological replay above:

| S, $150 a trade | session bootstrap | **chronological** |
|---|---|---|
| evaluation pass rate | 76.9% | **25%** (1 of 4) |
| net per account | $881 | **$422** |
| P(profitable over 10 years) | **100%** | — |
| 5-year net | — | **+$1,689** |

The bootstrap destroys serial structure. Bad periods cluster; for a strategy
whose edge depends on a market regime, every resampled path mixes good and bad
sessions evenly, and **no path ever contains "two straight years where nothing
works"** — which is exactly what happened. That 100% was an artefact of the
resampling method, not a property of the strategy.

**A session-level bootstrap systematically overstates account survival for any
regime-dependent strategy.** Report the chronological replay alongside it, or do
not report the bootstrap at all.

---

## What the holdout settled

**The pre-registered rule was right.** It said trade none of the 17,820. The
holdout agrees: the two best price rules lost money, and the one configuration
that reached the final gate was rejected there for a reason — its Sharpe was
below the noise benchmark — and could not be established on new data either.

Had the rule been relaxed at any of the three points where it was tempting to —
softening +0.185 R, waiving Deflated Sharpe for the single finalist, or trusting
the bootstrap's 100% — the outcome would have been an account funded on a
strategy that then lost money for two straight years.

**The holdout is spent and cannot be reused.** ES is the only clean test left.
Any future candidate should be checked first against the one thing this document
actually established: **is it long-only, and is it collecting intraday drift?**
