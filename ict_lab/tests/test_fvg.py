from __future__ import annotations

import pandas as pd

from ict_lab.features.fvg import detect_fvg

_EVENT_COLS = [
    "direction",
    "timeframe",
    "bar_index",
    "timestamp",
    "knowable_at",
    "gap_top",
    "gap_bottom",
    "midpoint",
    "size_points",
    "size_atr_mult",
]


def _bars(rows):
    idx = pd.date_range("2024-06-03 09:30", periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    df["volume"] = 100
    df["contract"] = "X"
    return df


def test_bullish_fvg_detected_with_correct_extents():
    rows = [
        {"open": 9.5, "high": 10, "low": 9, "close": 9.8},
        {"open": 10.5, "high": 11, "low": 10, "close": 10.8},
        {"open": 12.5, "high": 13, "low": 12, "close": 12.8},
    ]
    df = _bars(rows)
    out = detect_fvg(df, timeframe="1m")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["direction"] == "bullish"
    assert row["gap_top"] == 12
    assert row["gap_bottom"] == 10
    assert row["midpoint"] == 11
    assert row["size_points"] == 2
    assert row["bar_index"] == 2
    assert row["knowable_at"] == df.index[2]


def test_bearish_fvg_detected_with_correct_extents():
    rows = [
        {"open": 12.5, "high": 13, "low": 12, "close": 12.2},
        {"open": 11, "high": 11.5, "low": 10.5, "close": 10.8},
        {"open": 9.5, "high": 10, "low": 9, "close": 9.2},
    ]
    df = _bars(rows)
    out = detect_fvg(df, timeframe="1m")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["direction"] == "bearish"
    assert row["gap_top"] == 12
    assert row["gap_bottom"] == 10
    assert row["size_points"] == 2


def test_no_gap_when_bars_overlap():
    rows = [
        {"open": 10, "high": 11, "low": 9, "close": 10.5},
        {"open": 10.2, "high": 10.8, "low": 9.8, "close": 10.1},
        {"open": 10.3, "high": 10.9, "low": 9.9, "close": 10.4},
    ]
    out = detect_fvg(_bars(rows), timeframe="1m")
    assert len(out) == 0


def test_min_size_points_filters_small_gaps():
    rows = [
        {"open": 9.5, "high": 10, "low": 9, "close": 9.8},
        {"open": 10.1, "high": 10.3, "low": 10, "close": 10.2},
        {"open": 10.6, "high": 10.7, "low": 10.5, "close": 10.65},
    ]
    df = _bars(rows)
    assert len(detect_fvg(df, timeframe="1m")) == 1
    assert len(detect_fvg(df, timeframe="1m", min_size_points=1.0)) == 0


def _the_gap(out: pd.DataFrame, bar_index: int, direction: str) -> pd.Series:
    # A sliding 3-bar window can legitimately find more than one FVG in a
    # longer sequence (e.g. an up move followed by a down move creates gaps
    # in both directions) -- pick out the specific one under test rather
    # than assuming it's the only one detected.
    match = out[(out["bar_index"] == bar_index) & (out["direction"] == direction)]
    assert len(match) == 1, f"expected exactly one {direction} gap at bar {bar_index}, got {len(match)}"
    return match.iloc[0]


def test_mitigation_and_fill_tracked_at_1m_resolution():
    rows = [
        {"open": 9.5, "high": 10, "low": 9, "close": 9.8},
        {"open": 10.5, "high": 11, "low": 10, "close": 10.8},
        {"open": 12.5, "high": 13, "low": 12, "close": 12.8},  # bar3 -> gap [10, 12]
        {"open": 12.7, "high": 12.9, "low": 12.6, "close": 12.7},  # no touch
        {"open": 12.0, "high": 12.1, "low": 11.5, "close": 11.8},  # touches top edge -> mitigated
        {"open": 11.0, "high": 11.1, "low": 9.9, "close": 10.5},  # trades through bottom -> filled
    ]
    df = _bars(rows)
    out = detect_fvg(df, timeframe="1m")
    row = _the_gap(out, bar_index=2, direction="bullish")
    assert row["gap_top"] == 12
    assert row["gap_bottom"] == 10
    assert row["mitigated_at"] == df.index[4]
    assert row["filled_at"] == df.index[5]


def test_unmitigated_gap_has_nat():
    rows = [
        {"open": 9.5, "high": 10, "low": 9, "close": 9.8},
        {"open": 10.5, "high": 11, "low": 10, "close": 10.8},
        {"open": 12.5, "high": 13, "low": 12, "close": 12.8},
        {"open": 12.7, "high": 12.9, "low": 12.6, "close": 12.7},
    ]
    out = detect_fvg(_bars(rows), timeframe="1m")
    row = _the_gap(out, bar_index=2, direction="bullish")
    assert pd.isna(row["mitigated_at"])
    assert pd.isna(row["filled_at"])


def test_no_lookahead_gap_identification_matches_up_to_knowable_at(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-03 11:00:00")
    full = detect_fvg(df, timeframe="5m")

    cutoff_ts = df.index[80]
    truncated = detect_fvg(df.iloc[:81], timeframe="5m")

    expected = full[full["knowable_at"] <= cutoff_ts][_EVENT_COLS].reset_index(drop=True)
    actual = truncated[_EVENT_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
