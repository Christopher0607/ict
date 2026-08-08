from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

_WINDOWS = {
    "rth": ("09:30", "16:00"),
    "killzone_london": ("03:00", "04:00"),
    "killzone_ny_am": ("10:00", "11:00"),
    "killzone_ny_pm": ("14:00", "15:00"),
}


def to_eastern(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        raise ValueError("Index must be tz-aware UTC before converting to US/Eastern.")
    return index.tz_convert(ET)


def _minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy() * 60 + index.minute.to_numpy()


def _window_mask(index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    start_h, start_m = (int(x) for x in start.split(":"))
    end_h, end_m = (int(x) for x in end.split(":"))
    minute_of_day = _minute_of_day(index)
    return (minute_of_day >= start_h * 60 + start_m) & (minute_of_day < end_h * 60 + end_m)


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

    for name, (start, end) in _WINDOWS.items():
        out[name] = _window_mask(et_index, start, end)

    return out
