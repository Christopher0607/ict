"""Trade-log adapter: schema, and the excursion-ordering band.

The point of the adapter is not just format conversion -- it is refusing to
silently pick an excursion ordering that the source data does not actually
determine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rules import get_ruleset
from sim import Outcome, simulate_stage
from sim.adapters import ExcursionOrder, path_from_trade_log

APEX = get_ruleset("apex_50k_intraday")


def _log(rows):
    return pd.DataFrame(rows, columns=["session_date", "net_pnl", "mae_points", "mfe_points"])


def test_missing_columns_are_rejected():
    bad = pd.DataFrame({"session_date": ["2026-01-02"], "net_pnl": [100.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        path_from_trade_log(bad, 50_000.0)


def test_empty_log_is_rejected():
    with pytest.raises(ValueError, match="empty trade log"):
        path_from_trade_log(_log([]), 50_000.0)


def test_settled_pnl_is_preserved_under_every_ordering():
    df = _log([
        ("2026-01-02", 500.0, 100.0, 800.0),
        ("2026-01-02", -250.0, 250.0, 400.0),
        ("2026-01-05", 750.0, 300.0, 900.0),
    ])
    for order in ExcursionOrder:
        equity, _ = path_from_trade_log(df, 50_000.0, order=order)
        assert equity[-1] == pytest.approx(51_000.0)


def test_days_are_indexed_per_session():
    df = _log([
        ("2026-01-02", 100.0, 0.0, 0.0),
        ("2026-01-02", 100.0, 0.0, 0.0),
        ("2026-01-05", 100.0, 0.0, 0.0),
    ])
    _, days = path_from_trade_log(df, 50_000.0)
    assert days.max() == 1
    assert list(np.unique(days)) == [0, 1]


def test_excursion_order_changes_survival_not_pnl():
    """One flat trade, +/-$2000 either way. Ordering alone decides life or death.

    Worst case: +2000 first lifts the floor from 47,500 to 49,500, so the
    -2000 leg lands at 48,000 and kills the account. Best case: the -2000 leg
    comes first while the floor is still 47,500, so 48,000 survives, and the
    account finishes flat. Identical net P&L, opposite outcomes.
    """
    df = _log([("2026-01-02", 0.0, 2_000.0, 2_000.0)])

    worst, wdays = path_from_trade_log(df, 50_000.0, order=ExcursionOrder.WORST)
    best, bdays = path_from_trade_log(df, 50_000.0, order=ExcursionOrder.BEST)

    assert worst[-1] == best[-1] == pytest.approx(50_000.0)

    w = simulate_stage(worst, wdays, APEX, target_equity=APEX.target_equity)
    b = simulate_stage(best, bdays, APEX, target_equity=APEX.target_equity)

    assert w.outcome is Outcome.BREACH_DRAWDOWN
    assert b.outcome is Outcome.RAN_OUT_OF_PATH


def test_settled_ordering_is_the_optimistic_error():
    """Dropping excursions entirely hides the breach the account really took."""
    df = _log([("2026-01-02", 0.0, 2_000.0, 2_000.0)])
    settled, days = path_from_trade_log(df, 50_000.0, order=ExcursionOrder.SETTLED)
    res = simulate_stage(settled, days, APEX, target_equity=APEX.target_equity)
    assert res.outcome is Outcome.RAN_OUT_OF_PATH  # looks like a survivor
    assert res.peak_equity == pytest.approx(50_000.0)


def test_point_value_scales_excursions():
    df = _log([("2026-01-02", 0.0, 100.0, 100.0)])
    eq_1, _ = path_from_trade_log(df, 50_000.0, order=ExcursionOrder.WORST, point_value=1.0)
    eq_20, _ = path_from_trade_log(df, 50_000.0, order=ExcursionOrder.WORST, point_value=20.0)
    assert eq_1.max() == pytest.approx(50_100.0)
    assert eq_20.max() == pytest.approx(52_000.0)
