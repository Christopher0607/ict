from __future__ import annotations

import pandas as pd
import pytest

from ict_lab.data.sessions import add_session_columns, to_eastern
from ict_lab.features.liquidity import (
    all_liquidity_levels,
    pre_window_levels,
    prior_rth_levels,
    prior_session_levels,
    swing_levels,
)


def test_prior_session_levels_match_actual_previous_session_extremes(synthetic_bars):
    df = synthetic_bars("2024-06-03 12:00:00", "2024-06-06 12:00:00")
    with_sessions = add_session_columns(df)
    actual_high = with_sessions.groupby("session_date")["high"].max()
    actual_low = with_sessions.groupby("session_date")["low"].min()

    levels = prior_session_levels(df)
    highs = levels[levels["level_type"] == "prior_session_high"].set_index("session_date")["price"]
    lows = levels[levels["level_type"] == "prior_session_low"].set_index("session_date")["price"]

    # session_date on a level row names the SOURCE session it was computed
    # from (not which later session it applies to) -- the last session in
    # the data is excluded since nothing yet proves it's actually over.
    ordered_sessions = sorted(with_sessions["session_date"].unique())
    for source in ordered_sessions[:-1]:
        assert highs.loc[source] == actual_high.loc[source]
        assert lows.loc[source] == actual_low.loc[source]


def test_prior_session_knowable_at_is_last_bar_of_source_session(synthetic_bars):
    df = synthetic_bars("2024-06-03 12:00:00", "2024-06-06 12:00:00")
    with_sessions = add_session_columns(df)
    last_bar_by_session = (
        pd.Series(with_sessions.index, index=with_sessions.index)
        .groupby(with_sessions["session_date"])
        .max()
    )

    levels = prior_session_levels(df)
    ordered_sessions = sorted(with_sessions["session_date"].unique())
    for source in ordered_sessions[:-1]:
        row = levels[
            (levels["session_date"] == source) & (levels["level_type"] == "prior_session_high")
        ].iloc[0]
        assert row["knowable_at"] == last_bar_by_session.loc[source]


def test_prior_rth_levels_scoped_to_rth_only(synthetic_bars):
    df = synthetic_bars("2024-06-03 12:00:00", "2024-06-06 12:00:00")
    with_sessions = add_session_columns(df)
    rth = with_sessions[with_sessions["rth"]]
    actual_high = rth.groupby("session_date")["high"].max()

    levels = prior_rth_levels(df)
    highs = levels[levels["level_type"] == "prior_rth_high"].set_index("session_date")["price"]

    ordered_sessions = sorted(rth["session_date"].unique())
    for source in ordered_sessions[:-1]:
        assert highs.loc[source] == actual_high.loc[source]


def test_pre_window_includes_overnight_portion(synthetic_bars):
    # A session's 18:00-23:59 ET bars are on the *prior* calendar date but
    # must still count toward the next morning's pre-window. Extends past
    # 14:00 UTC (10:00 ET, the killzone_ny_am start) so the window actually
    # opens and the pre-window is provably complete.
    df = synthetic_bars("2024-06-03 19:00:00", "2024-06-04 15:00:00")
    with_sessions = add_session_columns(df)
    session = with_sessions["session_date"].iloc[-1]  # the single session this all belongs to

    evening_high = with_sessions[to_eastern(with_sessions.index).hour >= 18]["high"].max()

    levels = pre_window_levels(df, "killzone_ny_am")
    row = levels[(levels["level_type"] == "pre_killzone_ny_am_high") & (levels["session_date"] == session)]
    assert len(row) == 1
    # The pre-window high must be at least the overnight high -- it can only
    # be missing if the (buggy) version excluded the evening bars entirely.
    assert row.iloc[0]["price"] >= evening_high


def test_pre_window_levels_knowable_before_window_start(synthetic_bars):
    df = synthetic_bars("2024-06-03 12:00:00", "2024-06-04 12:00:00")
    levels = pre_window_levels(df, "killzone_ny_am")
    assert not levels.empty
    knowable_et = to_eastern(pd.DatetimeIndex(levels["knowable_at"]))
    # Every pre-window knowable_at is either the overnight portion (>=18) or
    # strictly before the 10:00 killzone start.
    assert ((knowable_et.hour >= 18) | (knowable_et.hour < 10)).all()


def test_pre_window_rejects_unknown_window():
    with pytest.raises(ValueError):
        pre_window_levels(pd.DataFrame(), "bogus_window")


def test_swing_levels_reuses_swing_points(synthetic_bars):
    df = synthetic_bars("2024-06-03 09:00:00", "2024-06-03 11:00:00")
    levels = swing_levels(df, n=5)
    assert set(levels["level_type"].unique()) <= {"swing_high", "swing_low"}
    assert levels["knowable_at"].notna().all()


def test_all_liquidity_levels_sorted_by_knowable_at(synthetic_bars):
    df = synthetic_bars("2024-06-03 12:00:00", "2024-06-05 12:00:00")
    combined = all_liquidity_levels(df)
    assert combined["knowable_at"].is_monotonic_increasing
    assert set(combined["level_type"].unique()) >= {
        "prior_session_high",
        "prior_session_low",
        "prior_rth_high",
        "prior_rth_low",
        "swing_high",
        "swing_low",
    }
