from __future__ import annotations

import pandas as pd

from ict_lab.features.atr import atr


def test_atr_is_nan_before_warmup_then_positive(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-03 10:00:00")
    out = atr(df, period=14)
    assert out.iloc[:13].isna().all()
    assert out.iloc[13:].notna().all()
    assert (out.iloc[13:] > 0).all()


def test_atr_is_causal(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-03 10:00:00")
    full = atr(df, period=14)
    truncated = atr(df.iloc[:40], period=14)
    pd.testing.assert_series_equal(full.iloc[:40], truncated, check_names=False)
