from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

WINDOWS = {
    "rth": ("09:30", "16:00"),
    "killzone_london": ("03:00", "04:00"),
    "killzone_ny_am": ("10:00", "11:00"),
    "killzone_ny_pm": ("14:00", "15:00"),
    # start==end means "every bar" (see _window_mask's wrap handling) --
    # Phase 5 Part 4's ablation ladder rung 1 needs a genuine full-session
    # window (no window restriction at all, added only at rung 4).
    "full_session": ("00:00", "00:00"),
}

# The completeness boundary for "is this session over" is 18:00 ET -- where
# add_session_columns' rollover rule actually flips session_date -- not the
# nominal 17:00 ET trading close. A bar can still print for this session
# anywhere in 17:00-17:59 ET (real CME data usually won't have one there,
# since that's the maintenance break, but nothing here should assume that
# gap exists), and it still belongs to this session_date, not the next.
SESSION_END_ET = "18:00"


def to_eastern(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        raise ValueError("Index must be tz-aware UTC before converting to US/Eastern.")
    return index.tz_convert(ET)


def _minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy() * 60 + index.minute.to_numpy()


def _window_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """Half-open [start, end). end <= start means the window wraps past
    midnight (e.g. 23:00-00:00) -- none of WINDOWS' own entries do this,
    but Phase 5's null-model "other hours" enumeration needs it."""
    start_h, start_m = (int(x) for x in start.split(":"))
    end_h, end_m = (int(x) for x in end.split(":"))
    minute_of_day = _minute_of_day(index)
    start_minute = start_h * 60 + start_m
    end_minute = end_h * 60 + end_m
    if end_minute <= start_minute:
        return (minute_of_day >= start_minute) | (minute_of_day < end_minute)
    return (minute_of_day >= start_minute) & (minute_of_day < end_minute)


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds session_date (CME session: 18:00 ET prior day -> 17:00 ET) plus
    rth/killzone_* booleans. All windows are half-open [start, end) so a bar
    stamped exactly at a boundary (e.g. 16:00) falls in the *next* window,
    never both.
    """
    out = df.copy()
    et_index = to_eastern(out.index)

    # Do day-rollover arithmetic on naive ET wall-clock dates, not on the
    # tz-aware index, so DST transition days (where +24h != +1 calendar day)
    # can't shift the session boundary.
    et_naive = et_index.tz_localize(None)
    calendar_day = et_naive.normalize()
    is_evening = et_index.hour >= 18
    session_date = calendar_day.where(~is_evening, calendar_day + pd.Timedelta(days=1))
    out["session_date"] = session_date

    for name, (start, end) in WINDOWS.items():
        out[name] = _window_mask(et_index, start, end)

    return out


def complete_sessions(
    session_dates: pd.DatetimeIndex, end_time: str, data_max: pd.Timestamp
) -> pd.DatetimeIndex:
    """Which of `session_dates` are provably over: data observed at or past
    `end_time` ET on that session's own calendar date. Pass SESSION_END_ET
    for a full session, or one of WINDOWS[...]'s end times for a sub-window
    (e.g. WINDOWS["rth"][1]).

    This is deliberately time-based rather than "does a later session/bar
    exist yet": the gap between one RTH close and the next RTH open is ~17.5
    hours, so "is there a later grouping in the data" and "has this session
    actually ended" are NOT interchangeable checks -- a session can be fully
    over long before any evidence of the next one shows up.
    """
    end_h, end_m = (int(x) for x in end_time.split(":"))
    end_utc = (
        pd.DatetimeIndex([d + pd.Timedelta(hours=end_h, minutes=end_m) for d in session_dates])
        .tz_localize(ET, ambiguous="infer", nonexistent="shift_forward")
        .tz_convert("UTC")
    )
    return session_dates[end_utc <= data_max]
