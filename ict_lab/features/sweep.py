from __future__ import annotations

import pandas as pd

SWEEP_COLUMNS = [
    "level_type",
    "session_date",
    "level_price",
    "swept_direction",
    "penetration_at",
    "penetration_price",
    "confirmed_at",
]


def _is_high_type(level_type: str) -> bool:
    return level_type.endswith("_high")


def detect_sweeps(
    df_1m: pd.DataFrame,
    levels: pd.DataFrame,
    level_types: list[str] | None = None,
    k: int = 3,
    min_penetration_ticks: float = 1.0,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """For each liquidity level, finds the first bar (at or after the level
    becomes knowable) that penetrates it by >= min_penetration_ticks AND
    closes back on the original side within k bars (the window is [that bar,
    that bar + k - 1], i.e. k bars inclusive of the penetration itself).

    A level not yet penetrated, or penetrated without a qualifying
    close-back, stays active: a later penetration attempt on the same level
    is still checked, matching "levels remain active until swept" -- only a
    fully-confirmed sweep consumes a level.

    NOTE on cost: scans forward from every level's knowable_at independently
    (O(levels x remaining bars) worst case). Fine for Phase 2's one-time
    precompute at the sizes tested here; revisit if it's too slow on the
    real 15-year history (e.g. chunk the scan by session).
    """
    min_penetration = min_penetration_ticks * tick_size
    scoped = levels if level_types is None else levels[levels["level_type"].isin(level_types)]

    events = []
    for _, level in scoped.iterrows():
        hit = _first_confirmed_sweep(
            df_1m,
            level["knowable_at"],
            _is_high_type(level["level_type"]),
            level["price"],
            min_penetration,
            k,
        )
        if hit is None:
            continue
        events.append(
            {
                "level_type": level["level_type"],
                "session_date": level["session_date"],
                "level_price": level["price"],
                "swept_direction": "high" if _is_high_type(level["level_type"]) else "low",
                **hit,
            }
        )

    if not events:
        return pd.DataFrame(columns=SWEEP_COLUMNS)
    return (
        pd.DataFrame(events, columns=SWEEP_COLUMNS)
        # Different levels can share an identical confirmed_at (e.g. two
        # levels swept by the same bar) -- a stable sort keeps that tie's
        # relative order deterministic between a full run and any prefix of
        # it, which the default quicksort does not guarantee.
        .sort_values("confirmed_at", kind="mergesort")
        .reset_index(drop=True)
    )


def _first_confirmed_sweep(
    df_1m: pd.DataFrame,
    knowable_at: pd.Timestamp,
    is_high_type: bool,
    level_price: float,
    min_penetration: float,
    k: int,
) -> dict | None:
    future = df_1m[df_1m.index > knowable_at]
    if future.empty:
        return None

    if is_high_type:
        penetrates = (future["high"] > level_price + min_penetration).to_numpy()
        closes_back = (future["close"] < level_price).to_numpy()
        penetration_price = future["high"]
    else:
        penetrates = (future["low"] < level_price - min_penetration).to_numpy()
        closes_back = (future["close"] > level_price).to_numpy()
        penetration_price = future["low"]

    closes_back_ahead = pd.Series(closes_back[::-1]).rolling(k, min_periods=1).max().to_numpy()[::-1] > 0
    qualifies = penetrates & closes_back_ahead

    hits = qualifies.nonzero()[0]
    if len(hits) == 0:
        return None
    pen_pos = hits[0]

    window = closes_back[pen_pos : pen_pos + k]
    confirm_pos = pen_pos + window.nonzero()[0][0]

    return {
        "penetration_at": future.index[pen_pos],
        "penetration_price": penetration_price.iloc[pen_pos],
        "confirmed_at": future.index[confirm_pos],
    }
