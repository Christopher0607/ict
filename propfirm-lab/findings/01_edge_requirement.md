# 01 · Edge requirement: how much edge before an account is worth buying

**Date:** 2026-08-22
**Reproduce:** `python run_edge_curve.py` → `findings/edge_curve.json`, `findings/figures/edge_curve.png`
**Market data used:** none. This is a property of the rules and of path shape.

---

## Method

Monte Carlo over a trader described only by the statistics that affect survival:
$250 risked per trade, 2R targets, 3 trades/day, and a swept **after-cost
expectancy** from −0.20R to +0.40R. Each trade emits four equity marks
(entry → excursion → excursion → exit) so a trailing floor sees the excursion,
not just the settlement.

6,000 eval paths and 4,000 full account lifecycles per grid point. Every account
is run cradle to grave: eval → activation → funded stage → payout cycles →
death, with fees charged as they actually accrue.

**Validation.** With a static floor and zero edge the simulator must reproduce
the gambler's-ruin closed form `D/(D+T)` = 2500/5500 = **45.45%**. It returns
**44.2%** (n=6,000, CI contains the analytic value). That test
(`tests/test_sim_validation.py::test_gamblers_ruin_closed_form`) is a hard gate.

---

## Result 1 — the rule costs more than the skill

Identical trader, zero edge, one 50k account. Only the drawdown rule changes:

| Drawdown rule | P(pass eval) |
|---|---|
| Static | **44.2%** |
| EOD trailing | **34.6%** |
| Intraday trailing (Apex default) | **31.9%** |

**12.3 points of pass rate, decided entirely by which account you bought.**

The mechanism is not subtle and is worth stating plainly: an intraday trailing
threshold follows *unrealized* equity. A trade that runs +$900 in your favour
and comes back to breakeven costs nothing on your P&L statement and permanently
raises your floor by $900. You are charged for excursions you never booked.

This is also why the simulator consumes an equity path rather than a daily P&L
series. Summarize first and this effect disappears from the model while
remaining in your account.

(The eval pass rates quoted in Result 2 differ by a few tenths of a point from
this table: these come from the 6,000-path eval study over 300 days, those from
the 4,000-account lifecycle study over a 200-day eval window.)

---

## Result 2 — the mean is a liar

At **zero after-cost expectancy** (a trader who exactly covers commission and
slippage, and is otherwise a coin flip):

| | Apex 50k | TopStep 50k | MFFU 50k Core |
|---|---|---|---|
| P(pass eval) | 31.6% | 28.1% | 28.1% |
| **Mean net / account** | **+$134** | **+$254** | **+$238** |
| 95% CI | ($92, $175) | ($213, $296) | ($196, $279) |
| **Median net / account** | **−$131** | **−$49** | **−$77** |
| **P(account ever profits)** | **8.3%** | 8.6% | 7.6% |

The mean is positive. **More than 91% of accounts still lose their entry fee.**

Where the positive mean comes from, at zero edge on Apex (n=4,000):

| Payouts taken | Accounts |
|---|---|
| 0 | 3,666 (91.6%) |
| 1 | 179 (4.5%) |
| 2 | 94 (2.4%) |
| 3–6 | 61 (1.5%) |

The mean rides on that 1.5% tail. The mechanism is Apex's threshold lock: once
the trailing floor reaches start+$100 it stops moving, so a funded account that
has cleared the safety net sits permanently ~$2,500 above its floor and can
grind repeat payouts. That is real, and it is also why the distribution is so
skewed — almost all of the expected value lives in accounts that get there.

**Any report of a mean in this repo must carry the median and P(profit)
alongside it.** A positive mean here is entirely compatible with a >90% chance
that this particular account loses its fee.

---

## Result 3 — two break-even edges, five times apart

| What you want | Apex 50k | TopStep 50k | MFFU 50k |
|---|---|---|---|
| Positive **mean** over many accounts | −0.036R | −0.107R | −0.090R |
| Positive **median** — P(profit) > 50% | **+0.201R** | **+0.225R** | **+0.230R** |
| Ratio | 5.5× | 2.1× | 2.6× |

The first row is a lottery threshold: buy enough accounts and the average comes
out ahead. The second row is what "this account will probably make money"
requires.

> **The project's stated goal is *stable* long-term profit. That is the second
> row.** Approximately **+0.20R per trade after all costs** is the bar. Any
> strategy proposed in `research/` gets measured against that number, not
> against the mean-break-even number.

For scale, +0.20R at 2R targets means a **40% win rate after costs** — versus
33.3% to break even. Every point of win rate between those two is the entire
business.

---

## What this does not say

- **It is not a claim that a zero-edge trader should buy accounts.** The mean is
  positive by roughly one fee; the median is negative by exactly one fee. Acting
  on the mean requires buying many accounts and surviving the variance, and the
  model grants perfect discipline throughout — fixed sizing, no tilt, no rule
  violations, no time pressure. Real accounts do not get that.
- **The rule numbers could be wrong.** `rules/ruleset.py` was compiled from
  public pages in August 2026; Apex overhauled everything in March 2026 and
  terms move. The load-bearing assumption here is the post-payout retained
  balance (safety net = drawdown + $100, retained after each withdrawal), which
  was verified against public documentation but should be checked against the
  account dashboard before any money is committed.
- **The trader model is parametric, not empirical.** Fixed R, IID trades, no
  serial correlation, no volatility clustering, no behavioural response to being
  down. Every one of those omissions flatters the trader.

## Next

The single highest-value input available is **the trade history from the
accounts that were already blown.** Replaying those through `sim/` and checking
that it reproduces the real death dates and death causes would convert this
from a parametric model into a calibrated one — and `sim/adapters/trade_log.py`
already accepts that schema. That is worth more than any amount of further
simulation.
