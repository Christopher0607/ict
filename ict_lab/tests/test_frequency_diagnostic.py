from __future__ import annotations

import pandas as pd

from ict_lab.configs.grid import load_named_configs
from ict_lab.engine.frequency_diagnostic import (
    _drop_pnl_columns,
    atr_mult_sensitivity,
    compute_frequency_diagnostic,
    pct_days_with_trade,
    pct_windows_with_setup,
    raw_setup_counts,
    reason_mix,
    trades_per_day_distribution,
    trades_per_year,
)

Y1 = pd.Timestamp("2020-01-06")  # Monday
Y2 = pd.Timestamp("2021-01-04")  # Monday


def _signals(rows):
    cols = ["session_date", "window"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _no_signals(rows):
    cols = ["session_date", "window", "reason"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _trades(rows):
    cols = ["session_date"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ---------- pct_windows_with_setup / raw_setup_counts ----------


def test_pct_windows_with_setup_basic():
    signals = _signals(
        [
            {"session_date": Y1, "window": "killzone_ny_am"},
            {"session_date": Y1 + pd.Timedelta(days=1), "window": "killzone_ny_am"},
        ]
    )
    no_signals = _no_signals(
        [
            {"session_date": Y1 + pd.Timedelta(days=2), "window": "killzone_ny_am", "reason": "no_sweep"},
            {"session_date": Y1 + pd.Timedelta(days=3), "window": "killzone_ny_am", "reason": "no_fvg"},
        ]
    )
    out = pct_windows_with_setup(signals, no_signals)
    row = out[(out["window"] == "killzone_ny_am") & (out["year"] == 2020)].iloc[0]
    assert row["windows_total"] == 4
    assert row["windows_with_setup"] == 2
    assert row["pct_with_setup"] == 50.0


def test_pct_windows_with_setup_separates_by_year():
    signals = _signals([{"session_date": Y1, "window": "killzone_ny_am"}])
    no_signals = _no_signals([{"session_date": Y2, "window": "killzone_ny_am", "reason": "no_sweep"}])
    out = pct_windows_with_setup(signals, no_signals)
    assert set(out["year"]) == {2020, 2021}
    y2020 = out[out["year"] == 2020].iloc[0]
    y2021 = out[out["year"] == 2021].iloc[0]
    assert y2020["pct_with_setup"] == 100.0
    assert y2021["pct_with_setup"] == 0.0


def test_raw_setup_counts_counts_every_signal_row_not_just_presence():
    # Three uncapped signals on the same session+window (Phase 4: signals.py
    # emits every matching FVG) must count as 3, unlike pct_windows_with_setup
    # which only asks "did this window have >=1".
    signals = _signals(
        [
            {"session_date": Y1, "window": "killzone_ny_am"},
            {"session_date": Y1, "window": "killzone_ny_am"},
            {"session_date": Y1, "window": "killzone_ny_am"},
        ]
    )
    out = raw_setup_counts(signals)
    row = out[(out["window"] == "killzone_ny_am") & (out["year"] == 2020)].iloc[0]
    assert row["raw_setup_count"] == 3


# ---------- pct_days_with_trade / trades_per_year / distribution ----------


def test_pct_days_with_trade_counts_zero_trade_days_in_denominator():
    all_days = pd.DatetimeIndex([Y1 + pd.Timedelta(days=i) for i in range(4)])
    trades = _trades([{"session_date": Y1}])  # only 1 of 4 days has a trade
    out = pct_days_with_trade(trades, all_days)
    row = out[out["year"] == 2020].iloc[0]
    assert row["trading_days"] == 4
    assert row["days_with_trade"] == 1
    assert row["pct_days_with_trade"] == 25.0


def test_pct_days_with_trade_multiple_trades_same_day_count_once():
    all_days = pd.DatetimeIndex([Y1])
    trades = _trades([{"session_date": Y1}, {"session_date": Y1}])
    out = pct_days_with_trade(trades, all_days)
    row = out[out["year"] == 2020].iloc[0]
    assert row["days_with_trade"] == 1
    assert row["pct_days_with_trade"] == 100.0


def test_trades_per_year_groups_by_year():
    trades = _trades([{"session_date": Y1}, {"session_date": Y1 + pd.Timedelta(days=1)}, {"session_date": Y2}])
    out = trades_per_year(trades)
    assert dict(zip(out["year"], out["trades"])) == {2020: 2, 2021: 1}


def test_trades_per_day_distribution_includes_zero_trade_days():
    all_days = pd.DatetimeIndex([Y1 + pd.Timedelta(days=i) for i in range(3)])
    # day0: 2 trades, day1: 0 trades, day2: 1 trade
    trades = _trades([{"session_date": Y1}, {"session_date": Y1}, {"session_date": Y1 + pd.Timedelta(days=2)}])
    out = trades_per_day_distribution(trades, all_days)
    counts = dict(zip(out["trades_per_day"], out["days"]))
    assert counts == {0: 1, 1: 1, 2: 1}


# ---------- reason_mix ----------


def test_reason_mix_counts_by_year_window_reason():
    no_trades = _no_signals(
        [
            {"session_date": Y1, "window": "killzone_ny_am", "reason": "no_sweep"},
            {"session_date": Y1 + pd.Timedelta(days=1), "window": "killzone_ny_am", "reason": "no_sweep"},
            {"session_date": Y1 + pd.Timedelta(days=1), "window": "killzone_ny_am", "reason": "no_fvg"},
        ]
    )
    out = reason_mix(no_trades)
    row = out[(out["window"] == "killzone_ny_am") & (out["reason"] == "no_sweep")].iloc[0]
    assert row["count"] == 2
    assert row["year"] == 2020


# ---------- _drop_pnl_columns ----------


def test_drop_pnl_columns_removes_every_pnl_field():
    trades = pd.DataFrame(
        [
            {
                "session_date": Y1, "symbol": "NQ", "window": "killzone_ny_am", "direction": "bullish",
                "setup_at": Y1, "entry_at": Y1, "exit_at": Y1, "exit_reason": "target", "bars_held": 3,
                "gross_pnl": 100.0, "net_pnl": 96.0, "r_multiple": 2.0, "mae_points": 1.0, "mfe_points": 3.0,
                "ambiguous_bar": False, "is_roll_day": False, "config": {},
            }
        ]
    )
    out = _drop_pnl_columns(trades)
    assert not ({"gross_pnl", "net_pnl", "r_multiple", "mae_points", "mfe_points", "config"} & set(out.columns))
    assert "entry_at" in out.columns


# ---------- integration smoke tests (synthetic data -- NOT the real CHECKPOINT) ----------


def test_compute_frequency_diagnostic_smoke_with_synthetic_data(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-17 00:00:00")  # 2 weeks
    config = load_named_configs()["as_taught_5m"]
    diag = compute_frequency_diagnostic(df, config, "NQ")

    assert diag["config_name"] == "as_taught_5m"
    assert diag["placeholder_note"] is None
    for key in ("pct_windows_with_setup", "raw_setup_counts", "pct_days_with_trade", "trades_per_year", "reason_mix"):
        assert isinstance(diag[key], pd.DataFrame)
    if not diag["pct_days_with_trade"].empty:
        assert diag["pct_days_with_trade"]["pct_days_with_trade"].between(0, 100).all()


def test_compute_frequency_diagnostic_flags_as_traded_placeholder(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")
    config = load_named_configs()["as_traded"]
    diag = compute_frequency_diagnostic(df, config, "NQ", include_day_distribution=True)
    assert diag["placeholder_note"] == "as_traded (default placeholder, not yet user-specified)"
    assert "trades_per_day_distribution" in diag


def test_atr_mult_sensitivity_includes_configs_own_value_first(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")
    config = load_named_configs()["as_taught_5m"]
    out = atr_mult_sensitivity(df, config, "NQ", alt_mults=(0.25,))
    assert list(out["fvg_min_size_atr_mult"]) == [config.fvg_min_size_atr_mult, 0.25]
    assert out["pct_days_with_trade_overall"].between(0, 100).all()
