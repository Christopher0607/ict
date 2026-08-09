from __future__ import annotations

import pandas as pd

from ict_lab.features.resample import resample_ohlcv


def test_5m_bar_aggregates_ohlcv_correctly(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:30:00", "2024-06-03 09:40:00")
    out = resample_ohlcv(df, "5m")

    assert len(out) == 2
    first_five = df.iloc[0:5]
    row = out.iloc[0]
    assert row["open"] == first_five["open"].iloc[0]
    assert row["close"] == first_five["close"].iloc[-1]
    assert row["high"] == first_five["high"].max()
    assert row["low"] == first_five["low"].min()
    assert row["volume"] == first_five["volume"].sum()


def test_incomplete_trailing_bar_is_dropped(synthetic_bars):
    # 12 minutes of 1m data -> two full 5m bars and a partial third (2 bars).
    df = synthetic_bars("2024-06-03 09:30:00", "2024-06-03 09:42:00")
    out = resample_ohlcv(df, "5m")
    assert len(out) == 2
    assert out.index[-1] == pd.Timestamp("2024-06-03 09:35:00", tz="UTC")


def test_mid_period_gap_does_not_cause_false_incompleteness(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:30:00", "2024-06-03 09:35:00")
    df = df.drop(df.index[2])  # remove one bar from the middle of the only 5m period
    out = resample_ohlcv(df, "5m")
    assert len(out) == 1  # still counted complete: data reaches the period's last minute


def test_1m_passthrough_returns_same_rows(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:30:00", "2024-06-03 09:40:00")
    out = resample_ohlcv(df, "1m")
    assert len(out) == len(df)


def test_knowable_at_is_the_last_constituent_1m_bar(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:30:00", "2024-06-03 09:40:00")
    out = resample_ohlcv(df, "5m")
    assert out["knowable_at"].iloc[0] == pd.Timestamp("2024-06-03 09:34:00", tz="UTC")
    assert out["knowable_at"].iloc[1] == pd.Timestamp("2024-06-03 09:39:00", tz="UTC")


def test_1m_knowable_at_equals_own_timestamp(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:30:00", "2024-06-03 09:35:00")
    out = resample_ohlcv(df, "1m")
    assert (out["knowable_at"] == out.index).all()


def test_truncating_input_never_changes_already_complete_bars(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:30:00", "2024-06-03 10:00:00")
    full = resample_ohlcv(df, "5m")
    truncated = resample_ohlcv(df.iloc[:17], "5m")  # cuts partway through the 4th 5m bar
    common = truncated.index
    pd.testing.assert_frame_equal(full.loc[common], truncated)
