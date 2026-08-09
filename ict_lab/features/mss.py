from __future__ import annotations

import pandas as pd

MSS_COLUMNS = [
    "direction",
    "reference_swing_price",
    "reference_swing_timestamp",
    "broken_at",
    "break_style",
]


def detect_mss(
    df_1m: pd.DataFrame,
    swings: pd.DataFrame,
    after: pd.Timestamp,
    direction: str,
    break_style: str = "close",
) -> dict | None:
    """Looks for the first Market Structure Shift after `after` (typically a
    sweep's confirmed_at): direction="bullish" breaks the most recent
    CONFIRMED swing high (as of `after`); direction="bearish" breaks the
    most recent confirmed swing low. break_style="close" requires the bar's
    close beyond the level; "wick" only requires the high/low to reach it.

    Only swings with confirmed_at <= after are eligible as the reference --
    an unconfirmed swing isn't knowable as "the most recent swing" yet, and
    using it would be look-ahead.
    """
    if break_style not in ("close", "wick"):
        raise ValueError('break_style must be "close" or "wick"')
    if direction not in ("bullish", "bearish"):
        raise ValueError('direction must be "bullish" or "bearish"')

    kind = "high" if direction == "bullish" else "low"
    eligible = swings[(swings["kind"] == kind) & (swings["confirmed_at"] <= after)]
    if eligible.empty:
        return None
    reference = eligible.sort_values("timestamp").iloc[-1]
    level_price = reference["price"]

    future = df_1m[df_1m.index > after]
    if future.empty:
        return None

    if direction == "bullish":
        breaks = future["close"] > level_price if break_style == "close" else future["high"] > level_price
    else:
        breaks = future["close"] < level_price if break_style == "close" else future["low"] < level_price

    hits = breaks.to_numpy().nonzero()[0]
    if len(hits) == 0:
        return None
    pos = hits[0]

    return {
        "direction": direction,
        "reference_swing_price": level_price,
        "reference_swing_timestamp": reference["timestamp"],
        "broken_at": future.index[pos],
        "break_style": break_style,
    }


def detect_mss_after_sweeps(
    df_1m: pd.DataFrame,
    swings: pd.DataFrame,
    sweeps: pd.DataFrame,
    break_style: str = "close",
) -> pd.DataFrame:
    """For every sweep event, looks for the opposite-direction MSS after it
    confirms: a swept low (sell-side liquidity taken) expects a bullish MSS;
    a swept high expects a bearish MSS."""
    columns = MSS_COLUMNS + ["sweep_confirmed_at", "sweep_level_type"]
    events = []
    for _, sweep in sweeps.iterrows():
        direction = "bullish" if sweep["swept_direction"] == "low" else "bearish"
        hit = detect_mss(df_1m, swings, sweep["confirmed_at"], direction, break_style)
        if hit is None:
            continue
        events.append(
            {**hit, "sweep_confirmed_at": sweep["confirmed_at"], "sweep_level_type": sweep["level_type"]}
        )

    if not events:
        return pd.DataFrame(columns=columns)
    # kind="mergesort": stable, so ties in broken_at keep a deterministic
    # relative order between a full run and any prefix of it.
    return (
        pd.DataFrame(events, columns=columns)
        .sort_values("broken_at", kind="mergesort")
        .reset_index(drop=True)
    )
