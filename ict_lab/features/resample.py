from __future__ import annotations

import pandas as pd

TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15}
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "contract"]


def resample_ohlcv(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resamples 1-minute bars to a coarser timeframe.

    Drops any trailing bar whose period end hasn't been reached by the input
    data yet -- a still-forming higher-timeframe bar must never be visible to
    a detector, since every multi-timeframe feature's no-lookahead guarantee
    depends on this. A period is "complete" once df_1m's data reaches its
    last minute; this tolerates ordinary gaps mid-period (e.g. a quiet
    overnight minute with no printed bar) without mistaking them for
    truncation.

    Every row also carries `knowable_at`: the 1-minute timestamp at which
    this bar's OHLC is actually fully formed (its own timestamp for 1m bars;
    the timestamp of its last constituent 1m bar for resampled ones, since
    label="left" stamps a resampled bar at its *start*). Any feature built on
    top of these bars must gate on knowable_at, not on the bar's own index,
    or it's look-ahead.
    """
    if timeframe == "1m":
        out = df_1m[_OHLCV_COLUMNS].copy()
        out["knowable_at"] = out.index
        return out
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")

    minutes = TIMEFRAME_MINUTES[timeframe]
    freq = f"{minutes}min"
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "contract": "last",
    }

    resampled = df_1m[_OHLCV_COLUMNS].resample(freq, label="left", closed="left").agg(agg)
    resampled = resampled.dropna(subset=["open"])

    period_ends = resampled.index + pd.Timedelta(minutes=minutes - 1)
    complete = period_ends <= df_1m.index.max()
    resampled = resampled[complete].copy()
    resampled["knowable_at"] = period_ends[complete]
    return resampled
