from __future__ import annotations

import pandas as pd

from ict_lab.features.atr import atr


def displacement_atr(df: pd.DataFrame, atr_mult: float = 1.5, atr_period: int = 14) -> pd.Series:
    """Boolean per-bar flag: bar range (high-low) >= atr_mult * ATR(atr_period).
    Causal: ATR at bar i only uses bars <= i, and range uses only bar i."""
    bar_range = df["high"] - df["low"]
    atr_series = atr(df, period=atr_period)
    return (bar_range >= atr_mult * atr_series).fillna(False)


def displacement_percentile(df: pd.DataFrame, lookback: int, percentile: float) -> pd.Series:
    """Boolean per-bar flag: bar range is in the top `percentile` (0-100] of
    bar ranges over the trailing `lookback` bars, inclusive of the bar
    itself. Causal: rolling quantile only looks backward; a bar without a
    full lookback window yet is never flagged."""
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    bar_range = df["high"] - df["low"]
    threshold = bar_range.rolling(lookback, min_periods=lookback).quantile((100 - percentile) / 100)
    return (bar_range >= threshold).fillna(False)
