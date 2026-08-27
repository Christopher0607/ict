"""Causal feature matrix for the strategy search.

Every column here is computable from bars up to and including its own bar. A
feature that peeks one bar ahead produces a backtest that cannot be traded and
looks wonderful, so the rule is enforced structurally: anything derived from a
rolling window uses only trailing windows, and anything derived from a session
aggregate (opening range, prior-day levels) is masked until the moment it is
genuinely known.

Entry timing is not this module's problem -- ``backtest.simulate`` always enters
on the bar *after* the signal, so a signal computed at bar t's close transacts
at t+1's open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.data.sessions import add_session_columns


@dataclass
class FeatureSet:
    """Aligned arrays, one entry per bar."""

    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    session_id: np.ndarray        # dense integer per trade session
    session_end_idx: np.ndarray   # last bar index of this bar's session
    minutes_into_rth: np.ndarray  # -1 outside RTH
    is_rth: np.ndarray
    et_minute: np.ndarray         # minutes since ET midnight; London killzone
                                  # sits outside RTH and is unreachable otherwise

    atr: np.ndarray               # trailing ATR in points
    ret: dict[int, np.ndarray]    # lookback -> trailing return in points
    range_hi: dict[int, np.ndarray]  # trailing N-bar high, EXCLUDING this bar
    range_lo: dict[int, np.ndarray]
    vwap: np.ndarray              # session VWAP up to and including this bar
    rel_volume: np.ndarray        # volume / trailing average

    prior_high: np.ndarray        # previous session's RTH high, known at open
    prior_low: np.ndarray
    prior_close: np.ndarray
    or_high: dict[int, np.ndarray]   # opening-range high, NaN until it is set
    or_low: dict[int, np.ndarray]

    # Second-round features. All causal: the overnight range is only published
    # once the RTH open has passed it, calendar fields are known in advance,
    # and compression looks strictly backwards.
    overnight_high: np.ndarray
    overnight_low: np.ndarray
    session_gap: np.ndarray          # this session's open minus last close
    day_of_week: np.ndarray
    day_of_month: np.ndarray
    trading_day_of_month: np.ndarray  # 1-based; turn-of-month effects need this
    days_to_month_end: np.ndarray
    compression: dict[int, np.ndarray]  # N-bar range / trailing average range

    def __len__(self) -> int:
        return self.close.size


DEFAULT_LOOKBACKS = (5, 15, 30, 60)
DEFAULT_OR_MINUTES = (15, 30, 60)


def build(
    df: pd.DataFrame,
    *,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    or_minutes: tuple[int, ...] = DEFAULT_OR_MINUTES,
    atr_period: int = 14,
) -> FeatureSet:
    d = add_session_columns(df.reset_index(drop=True))
    o = d["open"].to_numpy(float)
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)
    v = d["volume"].to_numpy(float)

    session_id = pd.factorize(d["session_date"], sort=False)[0].astype(np.int64)
    session_end_idx = _session_end_index(session_id)
    is_rth = d["is_rth"].to_numpy(bool)
    minutes_into_rth = _minutes_into_rth(d, is_rth)
    et = d["et"]
    et_minute = (et.dt.hour * 60 + et.dt.minute).to_numpy(np.int64)

    # True range, then a trailing simple ATR. shift(1) so the current bar's own
    # range never informs the stop distance used to enter on it.
    prev_close = pd.Series(c).shift(1)
    tr = pd.concat([
        pd.Series(h - l),
        (pd.Series(h) - prev_close).abs(),
        (pd.Series(l) - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean().to_numpy()

    ret = {n: c - pd.Series(c).shift(n).to_numpy() for n in lookbacks}
    # Trailing extremes EXCLUDE the current bar: a breakout above "the last 30
    # bars' high" must not be compared against a high that includes itself.
    range_hi = {n: pd.Series(h).shift(1).rolling(n, min_periods=n).max().to_numpy()
                for n in lookbacks}
    range_lo = {n: pd.Series(l).shift(1).rolling(n, min_periods=n).min().to_numpy()
                for n in lookbacks}

    vwap = _session_vwap(c, v, session_id)
    rel_volume = v / pd.Series(v).shift(1).rolling(60, min_periods=20).mean().to_numpy()

    prior_high, prior_low, prior_close = _prior_session_levels(h, l, c, session_id, is_rth)
    or_high, or_low = {}, {}
    for m in or_minutes:
        oh, ol = _opening_range(h, l, session_id, minutes_into_rth, m)
        or_high[m], or_low[m] = oh, ol

    on_hi, on_lo = _overnight_range(h, l, session_id, is_rth)
    gap = _session_gap_arr(o, c, session_id)
    dow, dom, tdom, dtme = _calendar(d, session_id)
    compression = {n: _compression(h, l, n) for n in lookbacks}

    return FeatureSet(
        overnight_high=on_hi, overnight_low=on_lo, session_gap=gap,
        day_of_week=dow, day_of_month=dom, trading_day_of_month=tdom,
        days_to_month_end=dtme, compression=compression,
        ts=d["ts_open"].to_numpy(), open=o, high=h, low=l, close=c, volume=v,
        session_id=session_id, session_end_idx=session_end_idx,
        minutes_into_rth=minutes_into_rth, is_rth=is_rth, et_minute=et_minute,
        atr=atr, ret=ret, range_hi=range_hi, range_lo=range_lo,
        vwap=vwap, rel_volume=rel_volume,
        prior_high=prior_high, prior_low=prior_low, prior_close=prior_close,
        or_high=or_high, or_low=or_low,
    )


def _session_end_index(session_id: np.ndarray) -> np.ndarray:
    """Last bar index of each bar's OWN session.

    A running maximum over reversed session-end marks gives the last session's
    end for every bar, not each bar's own -- which would let a trade run to the
    end of the dataset instead of to its session close. It has to be a running
    minimum of the marks at or after each bar.
    """
    n = session_id.size
    idx = np.arange(n)
    is_last = np.concatenate([session_id[1:] != session_id[:-1], [True]])
    marks = np.where(is_last, idx, n)  # sentinel above any real index
    return np.minimum.accumulate(marks[::-1])[::-1]


def _minutes_into_rth(d: pd.DataFrame, is_rth: np.ndarray) -> np.ndarray:
    out = np.full(len(d), -1, dtype=np.int64)
    if not is_rth.any():
        return out
    grp = d.loc[is_rth].groupby("session_date").cumcount().to_numpy()
    out[is_rth] = grp
    return out


def _session_vwap(c: np.ndarray, v: np.ndarray, session_id: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"pv": c * v, "v": v, "s": session_id})
    cum_pv = df.groupby("s")["pv"].cumsum().to_numpy()
    cum_v = df.groupby("s")["v"].cumsum().to_numpy()
    return np.divide(cum_pv, cum_v, out=np.full_like(cum_pv, np.nan), where=cum_v > 0)


def _prior_session_levels(h, l, c, session_id, is_rth):
    """Previous session's RTH high/low/close, broadcast to every bar.

    Known at the moment the next session opens, so no masking is needed beyond
    the shift -- but a session with no RTH bars yields NaN rather than silently
    reusing an older session's levels.
    """
    df = pd.DataFrame({"h": h, "l": l, "c": c, "s": session_id, "rth": is_rth})
    rth = df.loc[df["rth"]]
    agg = rth.groupby("s").agg(hi=("h", "max"), lo=("l", "min"), cl=("c", "last"))
    agg = agg.reindex(np.unique(session_id))
    prev = agg.shift(1)
    mapper_hi = prev["hi"].to_dict()
    mapper_lo = prev["lo"].to_dict()
    mapper_cl = prev["cl"].to_dict()
    return (
        pd.Series(session_id).map(mapper_hi).to_numpy(float),
        pd.Series(session_id).map(mapper_lo).to_numpy(float),
        pd.Series(session_id).map(mapper_cl).to_numpy(float),
    )


def _opening_range(h, l, session_id, minutes_into_rth, window):
    """Opening-range extremes, NaN until the range is complete.

    The mask matters: an opening range is not knowable while it is still
    forming, and a strategy that trades the 30-minute range at minute 5 is
    reading the future.
    """
    in_or = (minutes_into_rth >= 0) & (minutes_into_rth < window)
    df = pd.DataFrame({"h": h, "l": l, "s": session_id, "in_or": in_or})
    sub = df.loc[df["in_or"]]
    agg = sub.groupby("s").agg(hi=("h", "max"), lo=("l", "min"))
    # copy=True: a mapped Series can hand back a read-only view, and the mask
    # below writes into it.
    hi = pd.Series(session_id).map(agg["hi"].to_dict()).to_numpy(float, copy=True)
    lo = pd.Series(session_id).map(agg["lo"].to_dict()).to_numpy(float, copy=True)
    not_yet = minutes_into_rth < window
    hi[not_yet] = np.nan
    lo[not_yet] = np.nan
    return hi, lo


def _overnight_range(h, l, session_id, is_rth):
    """High and low of the pre-RTH portion of each session.

    NaN until RTH begins: the overnight range is not a level you can trade
    against while it is still forming.
    """
    df = pd.DataFrame({"h": h, "l": l, "s": session_id, "rth": is_rth})
    pre = df.loc[~df["rth"]]
    agg = pre.groupby("s").agg(hi=("h", "max"), lo=("l", "min"))
    hi = pd.Series(session_id).map(agg["hi"].to_dict()).to_numpy(float, copy=True)
    lo = pd.Series(session_id).map(agg["lo"].to_dict()).to_numpy(float, copy=True)
    hi[~is_rth] = np.nan
    lo[~is_rth] = np.nan
    return hi, lo


def _session_gap_arr(o, c, session_id):
    """This session's first price minus the previous session's last."""
    df = pd.DataFrame({"o": o, "c": c, "s": session_id})
    first_open = df.groupby("s")["o"].transform("first").to_numpy()
    prev_close = df.groupby("s")["c"].last().shift(1)
    mapped = pd.Series(session_id).map(prev_close.to_dict()).to_numpy(float)
    return first_open - mapped


def _calendar(d: pd.DataFrame, session_id: np.ndarray):
    """Calendar position, counted in trading days rather than dates.

    Turn-of-month effects are defined on trading days: the last trading day of
    a month is the tradeable one, whatever date it falls on.
    """
    sess = pd.Series(d["session_date"].to_numpy())
    ts = pd.to_datetime(sess)
    dow = ts.dt.dayofweek.to_numpy(float)
    dom = ts.dt.day.to_numpy(float)

    uniq = pd.Series(pd.to_datetime(pd.unique(sess))).sort_values()
    ym = uniq.dt.year * 12 + uniq.dt.month
    tdom_map = dict(zip(uniq, uniq.groupby(ym).cumcount() + 1))
    total = ym.map(ym.value_counts())
    dtme_map = dict(zip(uniq, total.to_numpy() - (uniq.groupby(ym).cumcount() + 1)))

    tdom = ts.map(tdom_map).to_numpy(float)
    dtme = ts.map(dtme_map).to_numpy(float)
    return dow, dom, tdom, dtme


def _compression(h, l, n):
    """This bar's N-bar range against its own trailing average.

    Below 1 means the market has gone quiet relative to itself, which is the
    setup half of every range-expansion idea.
    """
    rng = pd.Series(h).rolling(n, min_periods=n).max() - pd.Series(l).rolling(n, min_periods=n).min()
    baseline = rng.shift(1).rolling(n * 5, min_periods=n * 2).mean()
    return (rng / baseline).to_numpy()
