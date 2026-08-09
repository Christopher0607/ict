"""Phase 2's explicit no-lookahead check: "Write a test that runs each
detector twice, once on full data and once on data truncated at bar N, and
asserts the outputs match up to bar N." Every real detector is covered here;
`bias_perfect` is the sole, clearly-flagged exception (tested separately,
below, for the opposite property).
"""
from __future__ import annotations

import pandas as pd

from ict_lab.features.bias import bias_daily_ma_slope, bias_perfect, bias_prior_day, bias_swing_structure
from ict_lab.features.displacement import displacement_atr, displacement_percentile
from ict_lab.features.fvg import detect_fvg
from ict_lab.features.liquidity import all_liquidity_levels
from ict_lab.features.mss import detect_mss_after_sweeps
from ict_lab.features.sweep import detect_sweeps
from ict_lab.features.swings import swing_points


def _prefix_by(full: pd.DataFrame, cutoff_col: str, cutoff) -> pd.DataFrame:
    return full[full[cutoff_col] <= cutoff].reset_index(drop=True)


def test_fvg_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-05 09:00:00")
    full = detect_fvg(df, timeframe="5m")
    cols = [c for c in full.columns if c not in ("mitigated_at", "filled_at")]
    for cutoff_pos in (200, 600, 1500):
        cutoff = df.index[cutoff_pos]
        truncated = detect_fvg(df.iloc[: cutoff_pos + 1], timeframe="5m")
        expected = _prefix_by(full, "knowable_at", cutoff)[cols]
        pd.testing.assert_frame_equal(truncated[cols].reset_index(drop=True), expected, check_dtype=False)


def test_swings_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-05 09:00:00")
    full = swing_points(df, n=5)
    for cutoff_pos in (200, 600, 1500):
        truncated = swing_points(df.iloc[: cutoff_pos + 1], n=5)
        expected = _prefix_by(full, "confirmed_at_index", cutoff_pos)
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected, check_dtype=False)


def test_liquidity_levels_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-06 09:00:00")
    full = all_liquidity_levels(df)
    for cutoff_pos in (500, 1500, 3000):
        cutoff = df.index[cutoff_pos]
        truncated = all_liquidity_levels(df.iloc[: cutoff_pos + 1])
        expected = _prefix_by(full, "knowable_at", cutoff)
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected, check_dtype=False)


def test_sweep_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-06 09:00:00")
    levels = all_liquidity_levels(df)  # held fixed -- isolates the sweep detector's own behavior
    full = detect_sweeps(df, levels, k=3, min_penetration_ticks=1, tick_size=0.25)
    for cutoff_pos in (1500, 3000):
        cutoff = df.index[cutoff_pos]
        truncated = detect_sweeps(
            df.iloc[: cutoff_pos + 1], levels, k=3, min_penetration_ticks=1, tick_size=0.25
        )
        expected = _prefix_by(full, "confirmed_at", cutoff)
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected, check_dtype=False)


def test_mss_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-06 09:00:00")
    swings = swing_points(df, n=5)  # held fixed, same isolation strategy as the sweep test
    levels = all_liquidity_levels(df)
    sweeps = detect_sweeps(df, levels, k=3, min_penetration_ticks=1, tick_size=0.25)
    full = detect_mss_after_sweeps(df, swings, sweeps, break_style="close")
    for cutoff_pos in (1500, 3000):
        cutoff = df.index[cutoff_pos]
        truncated = detect_mss_after_sweeps(df.iloc[: cutoff_pos + 1], swings, sweeps, break_style="close")
        expected = _prefix_by(full, "broken_at", cutoff)
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected, check_dtype=False)


def test_displacement_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-04 09:00:00")
    cutoff_pos = 500

    full_atr = displacement_atr(df, atr_mult=1.5)
    truncated_atr = displacement_atr(df.iloc[: cutoff_pos + 1], atr_mult=1.5)
    pd.testing.assert_series_equal(full_atr.iloc[: cutoff_pos + 1], truncated_atr, check_names=False)

    full_pct = displacement_percentile(df, lookback=20, percentile=10)
    truncated_pct = displacement_percentile(df.iloc[: cutoff_pos + 1], lookback=20, percentile=10)
    pd.testing.assert_series_equal(full_pct.iloc[: cutoff_pos + 1], truncated_pct, check_names=False)


def test_bias_prior_day_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-06 09:00:00")
    full = bias_prior_day(df)
    for cutoff_pos in (1500, 3000):
        cutoff = df.index[cutoff_pos]
        truncated = bias_prior_day(df.iloc[: cutoff_pos + 1])
        expected = _prefix_by(full, "as_of", cutoff)
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected, check_dtype=False)


def test_bias_swing_structure_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-05 09:00:00")
    full = bias_swing_structure(df, timeframe="5m", swing_n=5)
    for cutoff_pos in (600, 1500):
        cutoff = df.index[cutoff_pos]
        truncated = bias_swing_structure(df.iloc[: cutoff_pos + 1], timeframe="5m", swing_n=5)
        expected = _prefix_by(full, "as_of", cutoff)
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected, check_dtype=False)


def test_bias_daily_ma_slope_no_lookahead(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-10 09:00:00")
    full = bias_daily_ma_slope(df, period=3)
    for cutoff_pos in (1500, 4000):
        cutoff = df.index[cutoff_pos]
        truncated = bias_daily_ma_slope(df.iloc[: cutoff_pos + 1], period=3)
        expected = _prefix_by(full, "as_of", cutoff)
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected, check_dtype=False)


def test_bias_perfect_is_flagged_and_genuinely_uses_future_data():
    """The one detector allowed to use future information -- confirm it's
    unmistakably marked as such, and that it actually does depend on data
    after its own as_of label (unlike every detector above)."""
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-03 14:00:00", tz="UTC"),  # 10:00 ET -> killzone_ny_am start
            pd.Timestamp("2024-06-03 20:00:00", tz="UTC"),  # 16:00 ET -> session close
        ]
    )
    base = {"open": [100, 104], "high": [101, 106], "low": [99, 103], "volume": 10, "contract": "X"}
    bullish_close = pd.DataFrame({**base, "close": [100.5, 105]}, index=idx)
    bearish_close = pd.DataFrame({**base, "close": [100.5, 95]}, index=idx)

    up = bias_perfect(bullish_close, "killzone_ny_am")
    down = bias_perfect(bearish_close, "killzone_ny_am")

    assert up["look_ahead"].all() and down["look_ahead"].all()
    assert up.iloc[0]["as_of"] == down.iloc[0]["as_of"] == idx[0]  # identical label timestamp
    assert up.iloc[0]["bias"] == "bullish"
    assert down.iloc[0]["bias"] == "bearish"  # differs solely due to a later, different session close
