from __future__ import annotations

import pandas as pd

SWING_COLUMNS = ["kind", "bar_index", "timestamp", "price", "confirmed_at_index", "confirmed_at"]


def swing_points(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """N-bar fractal swing highs/lows: bar i is a swing high if its high is
    the max over the centered window [i-n, i+n] (swing low: min of lows).

    A swing point is NOT knowable at bar i -- confirming it requires seeing
    the n bars that follow. Every row carries both where it occurred
    (bar_index/timestamp) and when it became knowable
    (confirmed_at_index/confirmed_at, always == bar_index + n). Any caller
    using these as a reference level must gate on confirmed_at, never on
    timestamp, or it's look-ahead.
    """
    if len(df) < 2 * n + 1:
        return pd.DataFrame(columns=SWING_COLUMNS)

    window = 2 * n + 1
    rolling_max = df["high"].rolling(window, center=True).max()
    rolling_min = df["low"].rolling(window, center=True).min()

    is_high = (df["high"] == rolling_max) & rolling_max.notna()
    is_low = (df["low"] == rolling_min) & rolling_min.notna()

    bar_index = pd.Series(range(len(df)), index=df.index)
    confirmed_at_index = bar_index + n
    confirmed_at = pd.Series(df.index, index=df.index).shift(-n)

    def _build(mask: pd.Series, kind: str, price: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "kind": kind,
                "bar_index": bar_index[mask].astype(int).to_numpy(),
                "timestamp": df.index[mask.to_numpy()],
                "price": price[mask].to_numpy(),
                "confirmed_at_index": confirmed_at_index[mask].astype(int).to_numpy(),
                "confirmed_at": confirmed_at[mask].to_numpy(),
            }
        )

    highs = _build(is_high, "high", df["high"])
    lows = _build(is_low, "low", df["low"])

    out = pd.concat([highs, lows], ignore_index=True)
    return out.sort_values("bar_index", kind="mergesort").reset_index(drop=True)[SWING_COLUMNS]
