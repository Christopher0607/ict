from __future__ import annotations

import pandas as pd

from ict_lab.features.atr import atr
from ict_lab.features.resample import resample_ohlcv

FVG_COLUMNS = [
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
    "mitigated_at",
    "filled_at",
]


def detect_fvg(
    df_1m: pd.DataFrame,
    timeframe: str = "1m",
    min_size_points: float = 0.0,
    min_size_atr_mult: float = 0.0,
) -> pd.DataFrame:
    """3-bar Fair Value Gap detector. Runs on `timeframe` bars (resampled
    from df_1m). Bullish: bar1.high < bar3.low. Bearish: bar1.low > bar3.high.

    Mitigation/fill are tracked at 1-minute resolution against df_1m, scanning
    strictly after knowable_at -- the moment bar3 is actually fully formed --
    never from bar3's own (possibly left-labeled, pre-close) timestamp.
    """
    bars = resample_ohlcv(df_1m, timeframe)
    if len(bars) < 3:
        return pd.DataFrame(columns=FVG_COLUMNS)

    atr_series = atr(bars, period=14)
    bar1_high = bars["high"].shift(2)
    bar1_low = bars["low"].shift(2)
    bar3_high = bars["high"]
    bar3_low = bars["low"]

    is_bullish = bar1_high < bar3_low
    is_bearish = bar1_low > bar3_high

    gap_top = bar3_low.where(is_bullish, bar1_low.where(is_bearish))
    gap_bottom = bar1_high.where(is_bullish, bar3_high.where(is_bearish))
    size_points = gap_top - gap_bottom
    size_atr_mult = size_points / atr_series  # ATR is a volatility measure, not a price ratio

    qualifies = (is_bullish | is_bearish) & (size_points >= min_size_points)
    if min_size_atr_mult > 0:
        qualifies = qualifies & (size_atr_mult >= min_size_atr_mult)

    positions = qualifies.to_numpy().nonzero()[0]
    if len(positions) == 0:
        return pd.DataFrame(columns=FVG_COLUMNS)

    events = []
    for pos in positions:
        direction = "bullish" if is_bullish.iloc[pos] else "bearish"
        top = gap_top.iloc[pos]
        bottom = gap_bottom.iloc[pos]
        knowable_at = bars["knowable_at"].iloc[pos]
        mitigated_at, filled_at = _track_mitigation(df_1m, knowable_at, direction, top, bottom)
        events.append(
            {
                "direction": direction,
                "timeframe": timeframe,
                "bar_index": pos,
                "timestamp": bars.index[pos],
                "knowable_at": knowable_at,
                "gap_top": top,
                "gap_bottom": bottom,
                "midpoint": (top + bottom) / 2,
                "size_points": top - bottom,
                "size_atr_mult": size_atr_mult.iloc[pos],
                "mitigated_at": mitigated_at,
                "filled_at": filled_at,
            }
        )
    return pd.DataFrame(events, columns=FVG_COLUMNS)


def _track_mitigation(
    df_1m: pd.DataFrame, knowable_at: pd.Timestamp, direction: str, top: float, bottom: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    future = df_1m[df_1m.index > knowable_at]
    if direction == "bullish":
        touch = future["low"] <= top
        fill = future["low"] <= bottom
    else:
        touch = future["high"] >= bottom
        fill = future["high"] >= top
    return _first_true_timestamp(touch), _first_true_timestamp(fill)


def _first_true_timestamp(mask: pd.Series) -> pd.Timestamp:
    hits = mask.to_numpy().nonzero()[0]
    return mask.index[hits[0]] if len(hits) else pd.NaT
