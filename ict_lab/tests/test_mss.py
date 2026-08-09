from __future__ import annotations

import pandas as pd

from ict_lab.features.mss import detect_mss, detect_mss_after_sweeps


def _bars(rows, start="2024-06-03 10:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    df["volume"] = 100
    df["contract"] = "X"
    return df


def _swing(kind, price, timestamp, confirmed_at):
    return {
        "kind": kind,
        "bar_index": 0,
        "timestamp": timestamp,
        "price": price,
        "confirmed_at_index": 0,
        "confirmed_at": confirmed_at,
    }


def test_bullish_mss_close_through():
    rows = [
        {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
        {"open": 99.2, "high": 99.8, "low": 99, "close": 99.5},
        {"open": 99.6, "high": 100.6, "low": 99.4, "close": 100.3},  # closes above 100 -> MSS
    ]
    df = _bars(rows)
    swings = pd.DataFrame([_swing("high", 100.0, df.index[0] - pd.Timedelta(minutes=10), df.index[0])])

    hit = detect_mss(df, swings, after=df.index[0], direction="bullish", break_style="close")
    assert hit is not None
    assert hit["broken_at"] == df.index[2]
    assert hit["reference_swing_price"] == 100.0


def test_bearish_mss_wick_through():
    rows = [
        {"open": 101, "high": 101.5, "low": 100.5, "close": 101},
        {"open": 100.8, "high": 101, "low": 100.2, "close": 100.5},  # no wick below 100 yet
        {"open": 100.4, "high": 100.5, "low": 99.8, "close": 100.1},  # wicks below 100
    ]
    df = _bars(rows)
    swings = pd.DataFrame([_swing("low", 100.0, df.index[0] - pd.Timedelta(minutes=10), df.index[0])])

    hit = detect_mss(df, swings, after=df.index[0], direction="bearish", break_style="wick")
    assert hit is not None
    assert hit["broken_at"] == df.index[2]


def test_close_through_requires_more_than_wick():
    rows = [
        {"open": 101, "high": 101.5, "low": 100.5, "close": 101},
        {"open": 100.4, "high": 100.5, "low": 99.8, "close": 100.1},  # wicks below, closes above
        {"open": 99.9, "high": 100.0, "low": 99.5, "close": 99.7},  # closes below -> confirmed here
    ]
    df = _bars(rows)
    swings = pd.DataFrame([_swing("low", 100.0, df.index[0] - pd.Timedelta(minutes=10), df.index[0])])

    hit = detect_mss(df, swings, after=df.index[0], direction="bearish", break_style="close")
    assert hit["broken_at"] == df.index[2]


def test_unconfirmed_swing_is_ignored():
    rows = [
        {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
        {"open": 99.6, "high": 100.6, "low": 99.4, "close": 100.3},
    ]
    df = _bars(rows)
    after = df.index[0]
    # An older CONFIRMED swing high at 100, and a newer swing high at 105
    # that only confirms *after* our search point -- must be ignored.
    swings = pd.DataFrame(
        [
            _swing("high", 100.0, after - pd.Timedelta(minutes=20), after - pd.Timedelta(minutes=5)),
            _swing("high", 105.0, after - pd.Timedelta(minutes=2), after + pd.Timedelta(minutes=30)),
        ]
    )
    hit = detect_mss(df, swings, after=after, direction="bullish", break_style="close")
    assert hit is not None
    assert hit["reference_swing_price"] == 100.0


def test_most_recent_confirmed_swing_is_used_when_multiple_qualify():
    rows = [
        {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
        {"open": 99.6, "high": 103.6, "low": 99.4, "close": 103.3},
    ]
    df = _bars(rows)
    after = df.index[0]
    swings = pd.DataFrame(
        [
            _swing("high", 100.0, after - pd.Timedelta(minutes=20), after - pd.Timedelta(minutes=15)),
            _swing("high", 102.0, after - pd.Timedelta(minutes=5), after - pd.Timedelta(minutes=1)),
        ]
    )
    hit = detect_mss(df, swings, after=after, direction="bullish", break_style="close")
    assert hit["reference_swing_price"] == 102.0


def test_no_eligible_swing_returns_none():
    df = _bars([{"open": 99, "high": 100, "low": 98, "close": 99.5}])
    swings = pd.DataFrame(
        columns=["kind", "bar_index", "timestamp", "price", "confirmed_at_index", "confirmed_at"]
    )
    assert detect_mss(df, swings, after=df.index[0], direction="bullish") is None


def test_detect_mss_after_sweeps_uses_opposite_direction():
    rows = [
        {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
        {"open": 99.6, "high": 100.6, "low": 99.4, "close": 100.3},  # bullish break of swing high 100
    ]
    df = _bars(rows)
    swings = pd.DataFrame([_swing("high", 100.0, df.index[0] - pd.Timedelta(minutes=10), df.index[0])])
    sweeps = pd.DataFrame([{"level_type": "swing_low", "swept_direction": "low", "confirmed_at": df.index[0]}])

    out = detect_mss_after_sweeps(df, swings, sweeps, break_style="close")
    assert len(out) == 1
    assert out.iloc[0]["direction"] == "bullish"
    assert out.iloc[0]["broken_at"] == df.index[1]
