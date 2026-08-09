from __future__ import annotations

import pandas as pd

from ict_lab.features.sweep import detect_sweeps


def _bars(rows, start="2024-06-03 10:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    df["volume"] = 100
    df["contract"] = "X"
    return df


def _level(level_type, price, knowable_at, session_date="2024-06-03"):
    return pd.DataFrame(
        [
            {
                "level_type": level_type,
                "session_date": pd.Timestamp(session_date),
                "price": price,
                "knowable_at": knowable_at,
            }
        ]
    )


def test_high_type_level_swept_when_penetrated_then_closes_back():
    rows = [
        {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
        {"open": 99.5, "high": 101.5, "low": 99, "close": 101},  # penetrates above 100, closes above
        {"open": 100.8, "high": 100.9, "low": 99.5, "close": 99.7},  # closes back below -> confirms
    ]
    df = _bars(rows)
    level = _level("prior_session_high", 100.0, df.index[0])

    out = detect_sweeps(df, level, k=3, min_penetration_ticks=1, tick_size=0.25)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["swept_direction"] == "high"
    assert row["penetration_at"] == df.index[1]
    assert row["confirmed_at"] == df.index[2]


def test_low_type_level_swept_when_penetrated_then_closes_back():
    rows = [
        {"open": 101, "high": 101.5, "low": 100.5, "close": 101},
        {"open": 100.5, "high": 101, "low": 98.5, "close": 99},  # penetrates below 100, closes below
        {"open": 99.3, "high": 100.5, "low": 99, "close": 100.3},  # closes back above -> confirms
    ]
    df = _bars(rows)
    level = _level("swing_low", 100.0, df.index[0])

    out = detect_sweeps(df, level, k=3, min_penetration_ticks=1, tick_size=0.25)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["swept_direction"] == "low"
    assert row["penetration_at"] == df.index[1]
    assert row["confirmed_at"] == df.index[2]


def test_no_sweep_when_close_back_is_outside_k_bars():
    rows = [
        {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
        {"open": 99.5, "high": 101.5, "low": 99, "close": 101},  # penetration, closes above
        {"open": 100.9, "high": 101, "low": 100.7, "close": 100.9},  # still above
    ]
    df = _bars(rows)
    level = _level("prior_session_high", 100.0, df.index[0])
    out = detect_sweeps(df, level, k=1, min_penetration_ticks=1, tick_size=0.25)
    assert len(out) == 0


def test_min_penetration_filters_shallow_pokes():
    rows = [
        {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
        {"open": 99.9, "high": 100.1, "low": 99.5, "close": 99.8},  # only 0.1 above -- shallow
        {"open": 99.8, "high": 99.9, "low": 99.6, "close": 99.7},
    ]
    df = _bars(rows)
    level = _level("prior_session_high", 100.0, df.index[0])
    out = detect_sweeps(df, level, k=3, min_penetration_ticks=4, tick_size=0.25)  # requires 1.0 pt
    assert len(out) == 0


def test_level_types_filter():
    df = _bars(
        [
            {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
            {"open": 99.5, "high": 101.5, "low": 99, "close": 101},
            {"open": 100.8, "high": 100.9, "low": 99.5, "close": 99.7},
        ]
    )
    levels = pd.concat(
        [
            _level("prior_session_high", 100.0, df.index[0]),
            _level("swing_high", 100.0, df.index[0]),
        ],
        ignore_index=True,
    )

    out_all = detect_sweeps(df, levels, k=3, min_penetration_ticks=1, tick_size=0.25)
    assert len(out_all) == 2

    out_filtered = detect_sweeps(
        df, levels, level_types=["swing_high"], k=3, min_penetration_ticks=1, tick_size=0.25
    )
    assert len(out_filtered) == 1
    assert out_filtered.iloc[0]["level_type"] == "swing_high"


def test_no_penetration_returns_empty():
    df = _bars(
        [
            {"open": 99, "high": 99.5, "low": 98.5, "close": 99},
            {"open": 99.2, "high": 99.6, "low": 98.9, "close": 99.3},
        ]
    )
    level = _level("prior_session_high", 100.0, df.index[0])
    out = detect_sweeps(df, level, k=3, min_penetration_ticks=1, tick_size=0.25)
    assert len(out) == 0


def test_failed_penetration_leaves_level_active_for_a_later_one():
    rows = [
        {"open": 99, "high": 99.5, "low": 98.5, "close": 99},  # level knowable here
        {"open": 99.5, "high": 101.0, "low": 99, "close": 100.8},  # penetration attempt #1
        {"open": 100.1, "high": 100.2, "low": 99.8, "close": 100.0},  # no close-back: #1's window fails
        {"open": 100.5, "high": 101.2, "low": 100.3, "close": 100.9},  # penetration attempt #2
        {"open": 100.5, "high": 100.6, "low": 99.4, "close": 99.5},  # closes back -> #2 confirms
    ]
    df = _bars(rows)
    level = _level("prior_session_high", 100.0, df.index[0])
    out = detect_sweeps(df, level, k=2, min_penetration_ticks=1, tick_size=0.25)
    assert len(out) == 1
    assert out.iloc[0]["penetration_at"] == df.index[3]
    assert out.iloc[0]["confirmed_at"] == df.index[4]
