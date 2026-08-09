from __future__ import annotations

import pandas as pd
import pytest

from ict_lab.data.sessions import add_session_columns
from ict_lab.features.bias import (
    bias_daily_ma_slope,
    bias_none,
    bias_perfect,
    bias_prior_day,
    bias_swing_structure,
    project_bias,
)


def test_bias_none_and_project_defaults_to_none():
    updates = bias_none(pd.DataFrame())
    assert updates.empty
    idx = pd.date_range("2024-06-03", periods=3, freq="1min", tz="UTC")
    assert (project_bias(updates, idx) == "none").all()


def test_project_bias_forward_fills_from_last_update():
    updates = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2024-06-03 10:00", "2024-06-03 12:00"], utc=True),
            "bias": ["bullish", "bearish"],
        }
    )
    query = pd.to_datetime(
        ["2024-06-03 09:00", "2024-06-03 10:00", "2024-06-03 11:00", "2024-06-03 13:00"], utc=True
    )
    projected = project_bias(updates, query)
    assert list(projected) == ["none", "bullish", "bullish", "bearish"]


def test_bias_prior_day_reflects_previous_session_close_vs_open():
    rows = {
        "2024-06-02 22:00:00": {"open": 100, "high": 101, "low": 99, "close": 100},
        "2024-06-03 20:00:00": {"open": 104, "high": 106, "low": 103, "close": 105},  # session A: bullish
        "2024-06-03 22:00:00": {"open": 110, "high": 111, "low": 109, "close": 110},
        "2024-06-04 20:00:00": {"open": 109, "high": 110, "low": 107, "close": 108},  # session B: bearish
        "2024-06-04 22:00:00": {"open": 120, "high": 121, "low": 119, "close": 120},
        "2024-06-05 20:00:00": {"open": 119, "high": 122, "low": 118, "close": 121},  # session C
    }
    idx = pd.to_datetime(list(rows.keys()), utc=True)
    df = pd.DataFrame(list(rows.values()), index=idx)
    df["volume"] = 10
    df["contract"] = "X"
    df = df.sort_index()

    updates = bias_prior_day(df)
    assert len(updates) == 2

    with_sessions = add_session_columns(df)
    b_start = with_sessions[with_sessions["session_date"] == pd.Timestamp("2024-06-04")].index.min()
    c_start = with_sessions[with_sessions["session_date"] == pd.Timestamp("2024-06-05")].index.min()

    assert project_bias(updates, pd.DatetimeIndex([b_start])).iloc[0] == "bullish"
    assert project_bias(updates, pd.DatetimeIndex([c_start])).iloc[0] == "bearish"


def test_bias_swing_structure_bullish_then_bearish():
    # A rising zigzag (higher highs + higher lows -> bullish) followed by one
    # lower-high/lower-low pair (-> bearish). Traced by hand against the
    # exact swing_n=1 fractal rule -- see the module docstring's HH/HL logic.
    v = [90, 100, 95, 110, 105, 120, 80, 90, 70]
    idx = pd.date_range("2024-06-03 09:30", periods=len(v), freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": v,
            "close": v,
            "high": [x + 1 for x in v],
            "low": [x - 1 for x in v],
            "volume": 10,
            "contract": "X",
        },
        index=idx,
    )
    updates = bias_swing_structure(df, timeframe="1m", swing_n=1)
    assert list(updates["bias"]) == ["none", "none", "none", "bullish", "bullish", "none", "bearish"]


def test_bias_daily_ma_slope_rising_then_falling():
    # A 7th, otherwise-irrelevant session is appended so the 6th's close is
    # provably final -- bias_daily_ma_slope always excludes the very last
    # session_date in the data, since nothing proves it's actually over.
    closes = [100, 102, 104, 110, 108, 104, 999]
    dates = pd.date_range("2024-06-03", periods=len(closes), freq="1D")
    rows, idx = [], []
    for d, c in zip(dates, closes):
        ts = pd.Timestamp(f"{d.date()} 16:00:00", tz="America/New_York").tz_convert("UTC")
        idx.append(ts)
        rows.append({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 10, "contract": "X"})
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))

    updates = bias_daily_ma_slope(df, period=3)
    assert list(updates["bias"]) == ["none", "none", "none", "bullish", "bullish", "none"]


def test_bias_perfect_uses_future_session_close_and_flags_lookahead():
    idx = [
        pd.Timestamp("2024-06-03 14:00:00", tz="UTC"),  # 10:00 ET -> killzone_ny_am start
        pd.Timestamp("2024-06-03 20:00:00", tz="UTC"),  # 16:00 ET -> session close
    ]
    df = pd.DataFrame(
        {
            "open": [100, 104],
            "high": [101, 106],
            "low": [99, 103],
            "close": [100.5, 105],
            "volume": 10,
            "contract": "X",
        },
        index=pd.DatetimeIndex(idx),
    )
    updates = bias_perfect(df, "killzone_ny_am")
    assert len(updates) == 1
    row = updates.iloc[0]
    assert row["bias"] == "bullish"
    assert row["as_of"] == idx[0]
    assert bool(row["look_ahead"]) is True


def test_bias_perfect_rejects_unknown_window():
    with pytest.raises(ValueError):
        bias_perfect(pd.DataFrame({"a": [1]}), "bogus_window")
