from __future__ import annotations

import pandas as pd
import pytest

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.discretion_premium import (
    discretion_premium_report,
    every_setup_result,
    format_discretion_premium,
    hindsight_perfect_result,
    minimum_skip_fraction,
    run_discretion_premium,
)

D1 = pd.Timestamp("2020-06-01")
D2 = pd.Timestamp("2020-06-02")


def _trade(session_date, r_multiple, net_pnl):
    return {"session_date": session_date, "r_multiple": r_multiple, "net_pnl": net_pnl}


def _config(**overrides):
    kwargs = dict(name="t", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    kwargs.update(overrides)
    return StrategyConfig(**kwargs)


# ---------- every_setup_result / hindsight_perfect_result ----------


def test_every_setup_result_basic():
    trades = pd.DataFrame([_trade(D1, 2.0, 100.0), _trade(D2, -1.0, -50.0)])
    result = every_setup_result(trades, pd.DatetimeIndex([D1, D2]))
    assert result["trades"] == 2
    assert result["net_pnl"] == pytest.approx(50.0)


def test_every_setup_result_empty_trades():
    result = every_setup_result(pd.DataFrame(columns=["session_date", "r_multiple", "net_pnl"]), pd.DatetimeIndex([D1]))
    assert result["trades"] == 0
    assert result["net_pnl"] == 0.0
    assert pd.isna(result["net_sharpe"])


def test_hindsight_perfect_result_keeps_only_winners():
    trades = pd.DataFrame(
        [_trade(D1, 2.0, 100.0), _trade(D1, -1.0, -50.0), _trade(D2, 1.0, 40.0), _trade(D2, -0.5, -10.0)]
    )
    result = hindsight_perfect_result(trades, pd.DatetimeIndex([D1, D2]))
    assert result["trades"] == 2  # only the two winners
    assert result["net_pnl"] == pytest.approx(140.0)


def test_hindsight_perfect_result_no_winners_is_empty():
    trades = pd.DataFrame([_trade(D1, -1.0, -50.0), _trade(D2, -0.5, -10.0)])
    result = hindsight_perfect_result(trades, pd.DatetimeIndex([D1, D2]))
    assert result["trades"] == 0
    assert result["net_pnl"] == 0.0


# ---------- minimum_skip_fraction ----------


def test_minimum_skip_fraction_worst_loser_first_reaches_breakeven():
    trades = pd.DataFrame(
        [
            _trade(D1, 2.0, 100.0),
            _trade(D1, -1.0, -80.0),
            _trade(D2, -3.0, -200.0),  # the worst loss
            _trade(D2, 1.0, 50.0),
        ]
    )
    # total = 100-80-200+50 = -130 (net negative overall)
    # skip the single worst loser (-200) -> 100-80+50 = 70 >= 0 -> 1 of 2 losers = 0.5
    result = minimum_skip_fraction(trades, pd.DatetimeIndex([D1, D2]))
    assert result["n_losers"] == 2
    assert result["breakeven_fraction"] == pytest.approx(0.5)


def test_minimum_skip_fraction_no_losers_is_zero():
    trades = pd.DataFrame([_trade(D1, 2.0, 100.0), _trade(D2, 1.0, 50.0)])
    result = minimum_skip_fraction(trades, pd.DatetimeIndex([D1, D2]))
    assert result["n_losers"] == 0
    assert result["breakeven_fraction"] == 0.0
    assert result["sharpe_fraction"] == 0.0


def test_minimum_skip_fraction_already_profitable_needs_zero_skips():
    trades = pd.DataFrame([_trade(D1, 2.0, 100.0), _trade(D1, -1.0, -10.0), _trade(D2, 1.0, 50.0)])
    result = minimum_skip_fraction(trades, pd.DatetimeIndex([D1, D2]))
    assert result["breakeven_fraction"] == pytest.approx(0.0)  # already net positive, skip nothing


def test_minimum_skip_fraction_unreachable_even_skipping_every_loser():
    # "Winner" by r_multiple but net-negative after commission; skipping the
    # one real loser still leaves a net-negative remainder.
    trades = pd.DataFrame([_trade(D1, 0.1, -2.0), _trade(D2, -1.0, -50.0)])
    result = minimum_skip_fraction(trades, pd.DatetimeIndex([D1, D2]))
    assert result["breakeven_fraction"] is None


def test_minimum_skip_fraction_empty_trades():
    result = minimum_skip_fraction(pd.DataFrame(columns=["session_date", "r_multiple", "net_pnl"]), pd.DatetimeIndex([D1]))
    assert result["breakeven_fraction"] is None
    assert result["sharpe_fraction"] is None
    assert result["n_losers"] == 0


# ---------- discretion_premium_report / format_discretion_premium ----------


def test_discretion_premium_report_structure():
    trades = pd.DataFrame([_trade(D1, 2.0, 100.0), _trade(D2, -1.0, -50.0)])
    report = discretion_premium_report("my_config", trades, pd.DatetimeIndex([D1, D2]))
    assert report["config_name"] == "my_config"
    assert set(report) == {"config_name", "every_setup", "hindsight_perfect", "minimum_skip"}


def test_format_discretion_premium_produces_four_lines():
    trades = pd.DataFrame([_trade(D1, 2.0, 100.0), _trade(D2, -1.0, -50.0)])
    report = discretion_premium_report("my_config", trades, pd.DatetimeIndex([D1, D2]))
    lines = format_discretion_premium(report)
    assert len(lines) == 4
    assert all("my_config" in line for line in lines)
    assert all(isinstance(line, str) and "\n" not in line for line in lines)


def test_format_discretion_premium_handles_unreachable_fraction():
    trades = pd.DataFrame([_trade(D1, 0.1, -2.0), _trade(D2, -1.0, -50.0)])
    report = discretion_premium_report("x", trades, pd.DatetimeIndex([D1, D2]))
    lines = format_discretion_premium(report)
    assert any("not achievable" in line for line in lines)


# ---------- run_discretion_premium (integration smoke test) ----------


def test_run_discretion_premium_smoke(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")
    refs = {"t": _config(entry_level="50%", stop_type="gap_distal", displacement_required=False, mss_required=False)}
    results = run_discretion_premium(df, "NQ", refs)
    assert set(results) == {"t"}
    assert results["t"]["config_name"] == "t"
