"""UTC to Eastern, CME trade dates, and session labels.

Two things here are easy to get wrong and expensive when you do.

**``ts_open`` is the bar's opening timestamp.** Databento's ``ts_event`` on an
OHLCV bar marks when the bar *started*, not when it closed. A 14:30 bar covers
14:30:00-14:30:59, so it is not knowable until 14:31. Anything that treats it
as a closing stamp reads the future by one bar and backtests wonderfully.
``knowable_at`` below makes that explicit rather than leaving it to memory.

**The CME trade date is not the calendar date.** The session opens at 18:00 ET
and runs to 17:00 ET the following day, so bars printed on Sunday evening
belong to Monday's trade date. Grouping by calendar date splits every overnight
session in half.

Conversions go through ``America/New_York`` rather than a fixed offset, so DST
is handled by the tz database instead of by arithmetic that is wrong twice a
year.
"""

from __future__ import annotations

import pandas as pd

ET = "America/New_York"

# CME equity index futures: 18:00 ET open, 17:00 ET close, one hour halt.
SESSION_OPEN_HOUR = 18
SESSION_CLOSE_HOUR = 17

# Regular US cash-session hours for index futures.
RTH_START = (9, 30)
RTH_END = (16, 0)


def to_eastern(ts: pd.Series) -> pd.Series:
    """UTC timestamps to tz-aware Eastern, DST-safe."""
    out = pd.to_datetime(ts, utc=True)
    return out.dt.tz_convert(ET)


def knowable_at(df: pd.DataFrame, bar_seconds: int = 60) -> pd.Series:
    """When each bar became known: its open plus its own duration.

    Every look-ahead check in this project should compare against this, not
    against ``ts_open``.
    """
    return pd.to_datetime(df["ts_open"], utc=True) + pd.Timedelta(seconds=bar_seconds)


def add_session_columns(df: pd.DataFrame, bar_seconds: int = 60) -> pd.DataFrame:
    """Add ``et``, ``session_date``, ``is_rth``, ``session`` and ``knowable_at``."""
    if "ts_open" not in df.columns:
        raise ValueError("frame needs a ts_open column")

    df = df.copy()
    et = to_eastern(df["ts_open"])
    df["et"] = et
    df["knowable_at"] = knowable_at(df, bar_seconds)

    # Bars at or after 18:00 ET belong to the next day's trade date.
    rolls_forward = et.dt.hour >= SESSION_OPEN_HOUR
    df["session_date"] = (
        et.dt.normalize() + pd.to_timedelta(rolls_forward.astype(int), unit="D")
    ).dt.date

    mins = et.dt.hour * 60 + et.dt.minute
    rth_start = RTH_START[0] * 60 + RTH_START[1]
    rth_end = RTH_END[0] * 60 + RTH_END[1]
    df["is_rth"] = (mins >= rth_start) & (mins < rth_end)

    df["session"] = "eth"
    df.loc[df["is_rth"], "session"] = "rth"
    # The daily maintenance halt: 17:00-18:00 ET.
    df.loc[(et.dt.hour >= SESSION_CLOSE_HOUR) & (et.dt.hour < SESSION_OPEN_HOUR),
           "session"] = "halt"
    return df
