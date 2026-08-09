from __future__ import annotations

import pandas as pd

from ict_lab.features.swings import swing_points


def _hand_crafted_df() -> pd.DataFrame:
    idx = pd.date_range("2024-06-03 09:30", periods=11, freq="1min", tz="UTC")
    high = [1, 2, 3, 4, 10, 4, 3, 2, 1, 2, 3]
    low = [9, 8, 7, 6, 5, 4, 0, 4, 5, 6, 7]
    return pd.DataFrame(
        {"open": high, "high": high, "low": low, "close": low, "volume": 1, "contract": "X"},
        index=idx,
    )


def test_swing_high_and_low_hand_crafted_example():
    df = _hand_crafted_df()
    out = swing_points(df, n=2)

    swing_highs = out[out["kind"] == "high"]
    swing_lows = out[out["kind"] == "low"]

    assert list(swing_highs["bar_index"]) == [4]
    assert swing_highs.iloc[0]["price"] == 10
    assert swing_highs.iloc[0]["confirmed_at_index"] == 6
    assert swing_highs.iloc[0]["confirmed_at"] == df.index[6]

    assert list(swing_lows["bar_index"]) == [6]
    assert swing_lows.iloc[0]["price"] == 0
    assert swing_lows.iloc[0]["confirmed_at_index"] == 8


def test_too_short_series_returns_empty():
    df = _hand_crafted_df().iloc[:3]
    out = swing_points(df, n=5)
    assert len(out) == 0


def test_no_lookahead_confirmed_swings_match_between_full_and_truncated_runs(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-03 11:00:00")
    n = 5
    full = swing_points(df, n=n)

    for cutoff in (30, 60, 90):
        truncated = swing_points(df.iloc[: cutoff + 1], n=n)
        # Everything the truncated run finds must already be knowable by
        # `cutoff` -- confirmed_at_index can never exceed the data it had.
        assert (truncated["confirmed_at_index"] <= cutoff).all()

        expected = full[full["confirmed_at_index"] <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            truncated.reset_index(drop=True), expected, check_dtype=False
        )
