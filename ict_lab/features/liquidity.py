from __future__ import annotations

import pandas as pd

from ict_lab.data.sessions import SESSION_END_ET, WINDOWS, add_session_columns, complete_sessions, to_eastern
from ict_lab.features.resample import resample_ohlcv
from ict_lab.features.swings import swing_points

LEVEL_COLUMNS = ["level_type", "session_date", "price", "knowable_at"]


def _prior_aggregate_levels(
    scoped: pd.DataFrame, high_type: str, low_type: str, end_time: str, data_max: pd.Timestamp
) -> pd.DataFrame:
    """Shared shape for "prior X high/low": one row per session that's
    provably complete (data observed at or past `end_time` ET on that
    session's own calendar date) -- session_date names the SOURCE session
    the aggregate was computed from, knowable right when that session's
    last (in-scope) bar printed. A still-in-progress session is silently
    skipped rather than guessed at from a partial aggregate.
    """
    if scoped.empty:
        return pd.DataFrame(columns=LEVEL_COLUMNS)

    grouped = scoped.groupby("session_date")
    high = grouped["high"].max()
    low = grouped["low"].min()
    last_ts = pd.Series(scoped.index, index=scoped.index).groupby(scoped["session_date"]).max()

    done = complete_sessions(high.index, end_time, data_max)
    if len(done) == 0:
        return pd.DataFrame(columns=LEVEL_COLUMNS)

    return pd.DataFrame(
        {
            "level_type": [high_type] * len(done) + [low_type] * len(done),
            "session_date": list(done) + list(done),
            "price": list(high.loc[done]) + list(low.loc[done]),
            "knowable_at": list(last_ts.loc[done]) + list(last_ts.loc[done]),
        }
    )[LEVEL_COLUMNS]


def prior_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Prior full-session (18:00 ET -> 17:00 ET) high/low, one per session."""
    with_sessions = add_session_columns(df)
    return _prior_aggregate_levels(
        with_sessions, "prior_session_high", "prior_session_low", SESSION_END_ET, df.index.max()
    )


def prior_rth_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Prior day's RTH (09:30-16:00 ET) high/low."""
    with_sessions = add_session_columns(df)
    return _prior_aggregate_levels(
        with_sessions[with_sessions["rth"]],
        "prior_rth_high",
        "prior_rth_low",
        WINDOWS["rth"][1],
        df.index.max(),
    )


def pre_window_levels(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """High/low from session open up to `window`'s start clock time, one per
    session that's provably complete (data observed at or past that start
    time on the session's own calendar date). Knowable at the last
    pre-window bar.

    Session open (18:00 ET) falls on the calendar day *before* the killzone
    clock times (all of which land on the session's own session_date, per
    add_session_columns' rollover rule) -- so the overnight 18:00-23:59 ET
    portion is unconditionally part of every pre-window, not just whatever
    bars happen to be numerically "less than" the killzone's minute-of-day.
    """
    if window not in WINDOWS:
        raise ValueError(f"Unknown window: {window!r}. Choose from {sorted(WINDOWS)}.")
    start, _ = WINDOWS[window]
    start_minute = 60 * int(start.split(":")[0]) + int(start.split(":")[1])

    with_sessions = add_session_columns(df)
    et_index = to_eastern(with_sessions.index)
    minute_of_day = et_index.hour.to_numpy() * 60 + et_index.minute.to_numpy()
    is_evening = et_index.hour.to_numpy() >= 18
    pre = with_sessions[is_evening | (minute_of_day < start_minute)]

    if pre.empty:
        return pd.DataFrame(columns=LEVEL_COLUMNS)

    grouped = pre.groupby("session_date")
    high = grouped["high"].max()
    low = grouped["low"].min()
    last_ts = pd.Series(pre.index, index=pre.index).groupby(pre["session_date"]).max()

    done = complete_sessions(high.index, start, df.index.max())
    if len(done) == 0:
        return pd.DataFrame(columns=LEVEL_COLUMNS)

    return pd.DataFrame(
        {
            "level_type": [f"pre_{window}_high"] * len(done) + [f"pre_{window}_low"] * len(done),
            "session_date": list(done) + list(done),
            "price": list(high.loc[done]) + list(low.loc[done]),
            "knowable_at": list(last_ts.loc[done]) + list(last_ts.loc[done]),
        }
    )[LEVEL_COLUMNS]


def swing_levels(df: pd.DataFrame, n: int = 5, timeframe: str = "1m") -> pd.DataFrame:
    """Swing highs/lows as liquidity levels, knowable once the fractal
    confirms. timeframe="1m" uses raw bars directly; any other timeframe
    resamples first and remaps confirmed_at through the resampled bars' own
    knowable_at column -- a resampled bar's left-labeled timestamp is not
    when it's actually knowable, same reasoning as bias_swing_structure.
    Non-1m level types get a `_{timeframe}` suffix (e.g. swing_high_15m) so
    they're independently selectable by a sweep-universe preset.
    """
    if timeframe == "1m":
        swings = swing_points(df, n=n)
        confirmed_at = swings["confirmed_at"] if not swings.empty else None
    else:
        bars = resample_ohlcv(df, timeframe)
        swings = swing_points(bars, n=n)
        confirmed_at = (
            bars["knowable_at"].to_numpy()[swings["confirmed_at_index"].to_numpy()]
            if not swings.empty
            else None
        )

    if swings.empty:
        return pd.DataFrame(columns=LEVEL_COLUMNS)

    suffix = "" if timeframe == "1m" else f"_{timeframe}"
    session_of = add_session_columns(df)["session_date"]
    return pd.DataFrame(
        {
            "level_type": swings["kind"].map({"high": f"swing_high{suffix}", "low": f"swing_low{suffix}"}),
            "session_date": session_of.reindex(swings["timestamp"].to_numpy()).to_numpy(),
            "price": swings["price"],
            "knowable_at": confirmed_at,
        }
    )[LEVEL_COLUMNS]


def all_liquidity_levels(
    df: pd.DataFrame,
    pre_windows: list[str] = ("killzone_london", "killzone_ny_am", "killzone_ny_pm"),
    swing_n: int = 5,
    swing_15m_n: int = 5,
) -> pd.DataFrame:
    """Every level type combined, sorted by when each became knowable. Levels
    don't carry an "active" flag here -- that's sweep.py's job: a level stays
    a valid reference until sweep detection finds the first bar that trades
    through it.
    """
    parts = [
        prior_session_levels(df),
        prior_rth_levels(df),
        swing_levels(df, n=swing_n, timeframe="1m"),
        swing_levels(df, n=swing_15m_n, timeframe="15m"),
    ]
    parts += [pre_window_levels(df, w) for w in pre_windows]
    combined = pd.concat(parts, ignore_index=True)
    # kind="mergesort": stable. prior_session_high/low always share an
    # identical knowable_at, and other ties are plausible too -- an unstable
    # sort would make row order (and thus any prefix comparison) nondeterministic.
    return combined.sort_values("knowable_at", kind="mergesort").reset_index(drop=True)
