from __future__ import annotations

import pandas as pd

from ict_lab.features.displacement import displacement_atr, displacement_percentile


def _bars_with_range(ranges, base=100.0, start="2024-06-03 09:30"):
    idx = pd.date_range(start, periods=len(ranges), freq="1min", tz="UTC")
    rows = [
        {
            "open": base,
            "high": base + r / 2,
            "low": base - r / 2,
            "close": base,
            "volume": 10,
            "contract": "X",
        }
        for r in ranges
    ]
    return pd.DataFrame(rows, index=idx)


def test_displacement_atr_flags_large_bar_relative_to_recent_range():
    df = _bars_with_range([1.0] * 20 + [5.0])
    flags = displacement_atr(df, atr_mult=2.0)
    assert flags.iloc[-1]
    assert not flags.iloc[13:-1].any()  # constant range never clears 2x its own ATR


def test_displacement_atr_causal(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-03 10:00:00")
    full = displacement_atr(df, atr_mult=1.5)
    truncated = displacement_atr(df.iloc[:40], atr_mult=1.5)
    pd.testing.assert_series_equal(full.iloc[:40], truncated, check_names=False)


def test_displacement_percentile_flags_top_bracket():
    df = _bars_with_range(list(range(1, 11)))  # ranges 1..10
    flags = displacement_percentile(df, lookback=10, percentile=20)
    assert not flags.iloc[:9].any()  # no full lookback window yet
    assert flags.iloc[9]  # range=10 is comfortably top 20% of [1..10]


def test_displacement_percentile_does_not_flag_small_bar_in_full_window():
    df = _bars_with_range(list(range(1, 11)) + [2])  # trailing 10 window for last bar: 2..10,2
    flags = displacement_percentile(df, lookback=10, percentile=20)
    assert not flags.iloc[-1]


def test_displacement_percentile_rejects_bad_percentile():
    df = _bars_with_range([1.0] * 5)
    try:
        displacement_percentile(df, lookback=5, percentile=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_displacement_percentile_causal(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-03 10:00:00")
    full = displacement_percentile(df, lookback=20, percentile=10)
    truncated = displacement_percentile(df.iloc[:40], lookback=20, percentile=10)
    pd.testing.assert_series_equal(full.iloc[:40], truncated, check_names=False)
