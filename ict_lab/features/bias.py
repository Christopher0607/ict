from __future__ import annotations

import pandas as pd

from ict_lab.data.sessions import SESSION_END_ET, WINDOWS, add_session_columns, complete_sessions
from ict_lab.features.resample import resample_ohlcv
from ict_lab.features.swings import swing_points

BIAS_VALUES = ("bullish", "bearish", "none")
UPDATE_COLUMNS = ["as_of", "bias"]


def project_bias(updates: pd.DataFrame, query_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fills sparse (as_of, bias) updates onto arbitrary query
    timestamps: the bias as of any query time is whatever the last update at
    or before it said, or "none" before the first update / with no updates
    at all."""
    if updates.empty:
        return pd.Series("none", index=query_index)
    updates = updates[UPDATE_COLUMNS].sort_values("as_of", kind="mergesort")
    query = pd.DataFrame({"as_of": pd.DatetimeIndex(query_index).sort_values()})
    merged = pd.merge_asof(query, updates, on="as_of", direction="backward")
    merged["bias"] = merged["bias"].fillna("none")
    return pd.Series(merged["bias"].to_numpy(), index=merged["as_of"].to_numpy()).reindex(query_index)


def bias_none(df_1m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=UPDATE_COLUMNS)


def bias_prior_day(df_1m: pd.DataFrame) -> pd.DataFrame:
    """bullish if a (provably complete) session's close > its open, bearish
    otherwise; knowable at that session's last bar. "Prior day" is which
    session the number describes -- a consumer reads this as of any later
    timestamp via project_bias, same as every other bias update here."""
    with_sessions = add_session_columns(df_1m)
    grouped = with_sessions.groupby("session_date")
    day_open = grouped["open"].first()
    day_close = grouped["close"].last()
    last_ts = pd.Series(with_sessions.index, index=with_sessions.index).groupby(
        with_sessions["session_date"]
    ).max()

    done = complete_sessions(day_open.index, SESSION_END_ET, df_1m.index.max())
    if len(done) == 0:
        return pd.DataFrame(columns=UPDATE_COLUMNS)

    bias = pd.Series("bearish", index=done)
    bias[(day_close.loc[done] > day_open.loc[done]).to_numpy()] = "bullish"

    return pd.DataFrame({"as_of": last_ts.loc[done].to_numpy(), "bias": bias.to_numpy()})


def bias_swing_structure(df_1m: pd.DataFrame, timeframe: str = "15m", swing_n: int = 5) -> pd.DataFrame:
    """bullish on higher-high+higher-low, bearish on lower-high+lower-low,
    else "none", re-evaluated every time a new swing confirms on
    `timeframe` bars."""
    bars = resample_ohlcv(df_1m, timeframe)
    raw_swings = swing_points(bars, n=swing_n)
    if raw_swings.empty:
        return pd.DataFrame(columns=UPDATE_COLUMNS)

    # swing_points' confirmed_at is bars' own (possibly left-labeled) index;
    # remap to bars['knowable_at'] so a resampled swing's confirmation lines
    # up with when it's *actually* knowable, not its bar's period start.
    confirmed_at = bars["knowable_at"].to_numpy()[raw_swings["confirmed_at_index"].to_numpy()]
    # kind="mergesort": stable, so a tie between a high and a low confirming
    # at the same instant always processes in the same relative order --
    # otherwise the HH/HL bookkeeping below could genuinely disagree between
    # a full run and a truncated one, not just reorder rows.
    swings = raw_swings.assign(confirmed_at=confirmed_at).sort_values("confirmed_at", kind="mergesort")

    rows = []
    last_high = prev_high = last_low = prev_low = None
    for _, s in swings.iterrows():
        if s["kind"] == "high":
            prev_high, last_high = last_high, s["price"]
        else:
            prev_low, last_low = last_low, s["price"]

        if None not in (prev_high, prev_low, last_high, last_low):
            if last_high > prev_high and last_low > prev_low:
                bias = "bullish"
            elif last_high < prev_high and last_low < prev_low:
                bias = "bearish"
            else:
                bias = "none"
        else:
            bias = "none"
        rows.append({"as_of": s["confirmed_at"], "bias": bias})

    return pd.DataFrame(rows, columns=UPDATE_COLUMNS)


def bias_daily_ma_slope(df_1m: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """bullish if today's `period`-session MA of daily closes > yesterday's,
    bearish if lower, else "none". Knowable at each session's last bar.

    Only provably complete sessions (data observed at or past 17:00 ET on
    that session's own calendar date) feed the MA -- an in-progress
    session's close-so-far isn't final, and treating it as such would make
    the MA (and therefore the slope) shift retroactively once more of that
    session's bars arrive.
    """
    with_sessions = add_session_columns(df_1m)
    grouped = with_sessions.groupby("session_date")
    daily_close_all = grouped["close"].last().sort_index()
    last_ts = pd.Series(with_sessions.index, index=with_sessions.index).groupby(
        with_sessions["session_date"]
    ).max()

    settled = complete_sessions(daily_close_all.index, SESSION_END_ET, df_1m.index.max())
    if len(settled) == 0:
        return pd.DataFrame(columns=UPDATE_COLUMNS)
    daily_close = daily_close_all.loc[settled]

    ma = daily_close.rolling(period, min_periods=period).mean()
    slope = ma.diff()

    bias = pd.Series("none", index=daily_close.index)
    bias[(slope > 0).fillna(False)] = "bullish"
    bias[(slope < 0).fillna(False)] = "bearish"

    return pd.DataFrame({"as_of": last_ts.loc[settled].to_numpy(), "bias": bias.to_numpy()})


def bias_perfect(df_1m: pd.DataFrame, window: str) -> pd.DataFrame:
    """LOOK-AHEAD. Labels bias using the session's actual close (future
    information relative to the window's own start): bullish if the session
    closed above the price at the window's start, bearish if below. Exists
    only as a controlled, explicitly-flagged comparison point (Phase 5's
    ablation ladder) -- the `look_ahead` column is always True and this must
    never be wired into a real signal path.
    """
    if window not in WINDOWS:
        raise ValueError(f"Unknown window: {window!r}. Choose from {sorted(WINDOWS)}.")

    with_sessions = add_session_columns(df_1m)
    session_close = with_sessions.groupby("session_date")["close"].last()

    window_bars = with_sessions[with_sessions[window]]
    if window_bars.empty:
        return pd.DataFrame(columns=[*UPDATE_COLUMNS, "look_ahead"])
    window_start_price = window_bars.groupby("session_date")["open"].first()
    window_start_ts = pd.Series(window_bars.index, index=window_bars.index).groupby(
        window_bars["session_date"]
    ).min()

    common = window_start_price.index.intersection(session_close.index)
    bias = pd.Series("none", index=common)
    bias[(session_close.loc[common] > window_start_price.loc[common]).to_numpy()] = "bullish"
    bias[(session_close.loc[common] < window_start_price.loc[common]).to_numpy()] = "bearish"

    return pd.DataFrame(
        {
            "as_of": window_start_ts.loc[common].to_numpy(),
            "bias": bias.to_numpy(),
            "look_ahead": True,
        }
    )
