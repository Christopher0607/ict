"""The gate: if these fail, nothing downstream is trustworthy.

The headline test reproduces the closed-form gambler's-ruin probability. A
symmetric +/-R walk with a static floor D below and a target T above has an
exact absorption probability D/(D+T) when both barriers are integer multiples
of the step. Apex's 50k numbers make that 2500/5500 = 0.4545..., and the
simulator has to land on it without being tuned to.
"""

from __future__ import annotations

import numpy as np
import pytest

from paths import TradeModel, generate_path
from rules import DrawdownType, get_ruleset
from sim import Outcome, simulate_stage
from sim.study import study_eval

# A pure coin flip: +/-1R, no excursions, so the path is an exact random walk
# on multiples of R and the closed form applies with no overshoot.
COIN_FLIP = TradeModel(
    r_dollars=250.0,
    expectancy_r=0.0,
    payoff_r=1.0,
    trades_per_day=3,
    mae_frac=0.0,
    mfe_frac=0.0,
)


def test_coin_flip_win_rate_is_half():
    assert COIN_FLIP.win_rate == pytest.approx(0.5)


def test_gamblers_ruin_closed_form():
    """Static floor, zero edge -> P(pass) must equal D/(D+T)."""
    rs = get_ruleset("apex_50k_intraday").with_drawdown_type(DrawdownType.STATIC)
    analytic = rs.max_drawdown / (rs.max_drawdown + rs.profit_target)
    assert analytic == pytest.approx(2500 / 5500)

    study = study_eval(COIN_FLIP, rs, n_paths=20_000, n_days=400, seed=42)

    # Essentially every path must resolve; an unresolved path would silently
    # bias the estimate downward.
    assert study.outcomes["ran_out_of_path"] == 0
    lo, hi = study.pass_rate_ci
    assert lo <= analytic <= hi, (
        f"pass rate {study.pass_rate:.4f} CI ({lo:.4f}, {hi:.4f}) "
        f"excludes analytic {analytic:.4f}"
    )
    assert study.pass_rate == pytest.approx(analytic, abs=0.01)


def test_rule_severity_ordering():
    """Same trader, three floors: intraday < EOD < static pass rates."""
    base = get_ruleset("apex_50k_intraday")
    rates = {}
    for dd in (DrawdownType.STATIC, DrawdownType.EOD_TRAILING, DrawdownType.INTRADAY_TRAILING):
        rs = base.with_drawdown_type(dd)
        rates[dd] = study_eval(COIN_FLIP, rs, n_paths=6_000, n_days=400, seed=7).pass_rate

    assert rates[DrawdownType.INTRADAY_TRAILING] < rates[DrawdownType.EOD_TRAILING]
    assert rates[DrawdownType.EOD_TRAILING] < rates[DrawdownType.STATIC]


def test_edge_monotonically_improves_pass_rate():
    rs = get_ruleset("apex_50k_intraday")
    rates = []
    for e in (-0.2, -0.1, 0.0, 0.1, 0.2):
        m = TradeModel(r_dollars=250.0, expectancy_r=e, payoff_r=2.0,
                       trades_per_day=3, mae_frac=0.0, mfe_frac=0.0)
        rates.append(study_eval(m, rs, n_paths=3_000, n_days=300, seed=3).pass_rate)
    assert rates == sorted(rates), rates


# ---------------------------------------------------------------------------
# hand-built paths: exact boundary behaviour, one rule at a time
# ---------------------------------------------------------------------------


def _path(values, day=0):
    eq = np.asarray(values, dtype=float)
    return eq, np.full(eq.shape, day, dtype=np.int64)


def test_intraday_floor_ratchets_on_unrealized_peak():
    """Up to +2000 unrealized, back to flat -> dead under intraday, fine under static.

    This is the effect the whole project exists to measure: the round trip
    costs nothing on the P&L statement and $2000 of floor.
    """
    rs = get_ruleset("apex_50k_intraday")

    # Peak 52,000 lifts the floor from 47,500 to 49,500 (the 50,100 lock is
    # not binding yet -- that needs a 52,600 peak). Check both sides of it.
    survives, days = _path([50_000, 52_000, 50_000, 49_600])
    assert simulate_stage(survives, days, rs, target_equity=rs.target_equity).outcome \
        is Outcome.RAN_OUT_OF_PATH

    breaches, days = _path([50_000, 52_000, 50_000, 49_500])
    intraday = simulate_stage(breaches, days, rs, target_equity=rs.target_equity)
    assert intraday.outcome is Outcome.BREACH_DRAWDOWN
    assert intraday.stop_index == 3
    assert intraday.final_floor == pytest.approx(49_500)

    # The same 49,500 path is untouched by a static floor at 47,500: the
    # entire loss came from the ratchet, not from the P&L.
    static = simulate_stage(
        breaches, days, rs.with_drawdown_type(DrawdownType.STATIC),
        target_equity=rs.target_equity,
    )
    assert static.outcome is Outcome.RAN_OUT_OF_PATH


def test_trailing_lock_freezes_the_floor():
    """Once the threshold reaches start+$100 it stops following the peak."""
    rs = get_ruleset("apex_50k_intraday")
    assert rs.trailing_lock_at == 50_100

    # Peak 60,000 would imply a floor of 57,500 if it kept trailing.
    eq, days = _path([50_000, 60_000, 50_200])
    res = simulate_stage(eq, days, rs, target_equity=None)
    assert res.outcome is Outcome.RAN_OUT_OF_PATH
    assert res.final_floor == pytest.approx(50_100)

    eq, days = _path([50_000, 60_000, 50_100])
    res = simulate_stage(eq, days, rs, target_equity=None)
    assert res.outcome is Outcome.BREACH_DRAWDOWN


def test_eod_trailing_ignores_intraday_spike():
    """A spike that is given back before the close must not raise the floor."""
    rs = get_ruleset("apex_50k_intraday").with_drawdown_type(DrawdownType.EOD_TRAILING)
    eq = np.array([50_000, 52_000, 50_000,   # day 0: spike, closes flat
                   50_000, 49_600], dtype=float)
    days = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    res = simulate_stage(eq, days, rs, target_equity=None)
    # Day 0 closed at 50,000 so day 1's floor is 47,500, not 49,500.
    assert res.outcome is Outcome.RAN_OUT_OF_PATH
    assert res.final_floor == pytest.approx(47_500)


def test_daily_loss_limit_boundary():
    """Just above the limit survives; exactly at it breaches."""
    rs = get_ruleset("topstep_50k")
    assert rs.daily_loss_limit == 1_000

    eq, days = _path([50_000, 49_001])
    assert simulate_stage(eq, days, rs, target_equity=None).outcome is Outcome.RAN_OUT_OF_PATH

    eq, days = _path([50_000, 49_000])
    assert simulate_stage(eq, days, rs, target_equity=None).outcome is Outcome.BREACH_DAILY_LOSS


def test_daily_loss_limit_resets_each_day():
    rs = get_ruleset("topstep_50k")
    eq = np.array([50_000, 49_100, 49_100, 48_300], dtype=float)
    days = np.array([0, 0, 1, 1], dtype=np.int64)
    # -900 on day 0, then -800 from day 1's open: neither day breaches alone.
    assert simulate_stage(eq, days, rs, target_equity=None).outcome is Outcome.RAN_OUT_OF_PATH


def test_target_reached_is_a_pass():
    rs = get_ruleset("apex_50k_intraday")
    eq, days = _path([50_000, 53_000])
    res = simulate_stage(eq, days, rs, target_equity=rs.target_equity)
    assert res.outcome is Outcome.PASSED


def test_earliest_event_wins():
    """Breaching on the way to the target still counts as a breach."""
    rs = get_ruleset("apex_50k_intraday").with_drawdown_type(DrawdownType.STATIC)
    eq, days = _path([50_000, 47_500, 53_000])
    res = simulate_stage(eq, days, rs, target_equity=rs.target_equity)
    assert res.outcome is Outcome.BREACH_DRAWDOWN
    assert res.stop_index == 1
