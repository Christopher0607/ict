"""Session labelling: trade dates, DST, and the bar-timestamp convention.

Each of these encodes a mistake that is easy to make and silent when made.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.data.sessions import add_session_columns, knowable_at, to_eastern


def _df(stamps):
    return pd.DataFrame({"ts_open": pd.to_datetime(list(stamps), utc=True)})


def test_bar_is_knowable_only_after_it_closes():
    """ts_open marks the START of the bar, so a 1m bar is known a minute later."""
    df = _df(["2026-03-10T14:30:00Z"])
    assert knowable_at(df).iloc[0] == pd.Timestamp("2026-03-10T14:31:00Z")
    assert knowable_at(df, bar_seconds=1).iloc[0] == pd.Timestamp("2026-03-10T14:30:01Z")


def test_evening_bars_belong_to_the_next_trade_date():
    """18:00 ET Sunday opens Monday's session, not Sunday's."""
    # 2026-03-08 is a Sunday and DST has just begun, so 23:00Z is 19:00 ET.
    out = add_session_columns(_df(["2026-03-08T23:00:00Z"]))
    assert str(out.loc[0, "session_date"]) == "2026-03-09"


def test_afternoon_bars_keep_the_calendar_date():
    out = add_session_columns(_df(["2026-03-09T18:00:00Z"]))  # 14:00 ET Monday
    assert str(out.loc[0, "session_date"]) == "2026-03-09"


def test_one_session_spans_two_calendar_days():
    """The overnight session must not be split in half by calendar date."""
    out = add_session_columns(_df([
        "2026-03-08T23:00:00Z",  # Sun 18:00 ET
        "2026-03-09T04:00:00Z",  # Mon 00:00 ET
        "2026-03-09T18:00:00Z",  # Mon 14:00 ET
    ]))
    assert out["session_date"].nunique() == 1


def test_dst_is_handled_by_the_timezone_not_by_arithmetic():
    """US DST began 2026-03-08. The same UTC hour is a different ET hour after."""
    before = to_eastern(pd.Series(pd.to_datetime(["2026-03-06T18:00:00Z"], utc=True)))
    after = to_eastern(pd.Series(pd.to_datetime(["2026-03-16T18:00:00Z"], utc=True)))
    assert before.iloc[0].hour == 13   # EST, UTC-5
    assert after.iloc[0].hour == 14    # EDT, UTC-4


def test_rth_window_boundaries():
    # All four stamps are EDT (UTC-4): 2026-03-09 falls after DST began on the
    # 8th, which is exactly the arithmetic this module exists to not do by hand.
    out = add_session_columns(_df([
        "2026-03-09T13:29:00Z",  # 09:29 ET -- before the open
        "2026-03-09T13:30:00Z",  # 09:30 ET -- first RTH bar
        "2026-03-09T19:59:00Z",  # 15:59 ET -- last RTH bar
        "2026-03-09T20:00:00Z",  # 16:00 ET -- closed
    ]))
    assert list(out["is_rth"]) == [False, True, True, False]


def test_maintenance_halt_is_labelled():
    out = add_session_columns(_df([
        "2026-03-09T21:30:00Z",  # 17:30 ET -- inside the halt
        "2026-03-09T23:30:00Z",  # 19:30 ET -- back open
    ]))
    assert list(out["session"]) == ["halt", "eth"]


def test_missing_ts_open_is_rejected():
    with pytest.raises(ValueError, match="ts_open"):
        add_session_columns(pd.DataFrame({"close": [1.0]}))


# ---------------------------------------------------------------------------
# coverage accounting
# ---------------------------------------------------------------------------


def test_coverage_separates_missing_sessions_from_thin_ones():
    """The distinction that decides whether data can be filtered or must be cut.

    Two sessions: one with a full RTH day, one with only overnight bars. The
    complete session must show a full median, and the incomplete one must be
    counted as absent rather than dragging the median down -- a thin session
    can be filtered at runtime, a missing one biases the sample by date.
    """
    import numpy as np
    from research.data.quality import coverage_by_year

    rth = pd.date_range("2026-03-09T13:30:00Z", periods=390, freq="1min", tz="UTC")
    overnight = pd.date_range("2026-03-09T23:00:00Z", periods=60, freq="1min", tz="UTC")
    ts = rth.append(overnight)
    df = pd.DataFrame({
        "ts_open": ts,
        "close": np.arange(len(ts), dtype=float),
    })
    cov = coverage_by_year(df)
    assert cov.loc[2026, "sessions"] == 2          # 03-09 RTH, plus 03-10 evening
    assert cov.loc[2026, "with_rth"] == 1
    assert cov.loc[2026, "median_rth_bars"] == 390.0
    assert cov.loc[2026, "pct_complete"] == 0.5


def test_min_usable_date_is_documented_and_early():
    from research.data.quality import MIN_USABLE_DATE

    assert MIN_USABLE_DATE.year == 2016
