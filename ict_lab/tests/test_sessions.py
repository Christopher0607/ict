from __future__ import annotations

import pandas as pd
import pytest

from ict_lab.data.sessions import add_session_columns, to_eastern


def _bars_at_et_times(et_time_strings: list[str]) -> pd.DataFrame:
    et_index = pd.DatetimeIndex(et_time_strings, tz="America/New_York")
    n = len(et_index)
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1, "contract": "X"},
        index=et_index.tz_convert("UTC"),
    )


def test_to_eastern_handles_spring_forward():
    # US spring-forward: 2024-03-10, clocks jump 02:00 -> 03:00 ET (EST -> EDT).
    before = pd.Timestamp("2024-03-09 12:00:00", tz="UTC")  # EST = UTC-5 -> 07:00 ET
    after = pd.Timestamp("2024-03-11 12:00:00", tz="UTC")  # EDT = UTC-4 -> 08:00 ET
    et = to_eastern(pd.DatetimeIndex([before, after]))
    assert et[0].hour == 7
    assert et[1].hour == 8


def test_to_eastern_handles_fall_back():
    # US fall-back: 2024-11-03, clocks fall 02:00 -> 01:00 ET (EDT -> EST).
    before = pd.Timestamp("2024-11-02 12:00:00", tz="UTC")  # EDT = UTC-4 -> 08:00 ET
    after = pd.Timestamp("2024-11-04 12:00:00", tz="UTC")  # EST = UTC-5 -> 07:00 ET
    et = to_eastern(pd.DatetimeIndex([before, after]))
    assert et[0].hour == 8
    assert et[1].hour == 7


def test_to_eastern_rejects_naive_index():
    with pytest.raises(ValueError):
        to_eastern(pd.DatetimeIndex(["2024-01-01 12:00:00"]))


def test_session_date_rolls_over_at_1800_et():
    df = _bars_at_et_times(["2024-06-03 17:59", "2024-06-03 18:00", "2024-06-03 18:01"])
    out = add_session_columns(df)
    assert out["session_date"].iloc[0].date() == pd.Timestamp("2024-06-03").date()
    assert out["session_date"].iloc[1].date() == pd.Timestamp("2024-06-04").date()
    assert out["session_date"].iloc[2].date() == pd.Timestamp("2024-06-04").date()


def test_session_date_rollover_holds_across_spring_forward_weekend():
    # 2024-03-10 is the spring-forward Sunday, but the 02:00->03:00 ET jump
    # happens hours before the 18:00 rollover, so the boundary itself must
    # behave exactly like any other day.
    df = _bars_at_et_times(["2024-03-10 17:59", "2024-03-10 18:00"])
    out = add_session_columns(df)
    assert out["session_date"].iloc[0].date() == pd.Timestamp("2024-03-10").date()
    assert out["session_date"].iloc[1].date() == pd.Timestamp("2024-03-11").date()


def test_session_date_rollover_holds_across_fall_back_weekend():
    # 2024-11-03 is the fall-back Sunday; same logic, opposite direction.
    df = _bars_at_et_times(["2024-11-03 17:59", "2024-11-03 18:00"])
    out = add_session_columns(df)
    assert out["session_date"].iloc[0].date() == pd.Timestamp("2024-11-03").date()
    assert out["session_date"].iloc[1].date() == pd.Timestamp("2024-11-04").date()


def test_fall_back_repeated_hour_does_not_duplicate_or_drop_bars(synthetic_bars):
    # The 01:00-02:00 ET hour occurs twice in wall-clock terms on fall-back
    # day, but each occurrence is still a distinct UTC instant.
    df = synthetic_bars("2024-11-03 00:00:00", "2024-11-04 00:00:00")
    out = add_session_columns(df)
    assert not out.index.duplicated().any()
    assert len(out) == len(df)


def test_killzones_do_not_overlap_and_match_expected_clock_times(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-04 00:00:00")
    out = add_session_columns(df)

    london_hours = set(to_eastern(out[out["killzone_london"]].index).hour)
    ny_am_hours = set(to_eastern(out[out["killzone_ny_am"]].index).hour)
    ny_pm_hours = set(to_eastern(out[out["killzone_ny_pm"]].index).hour)
    rth_hours = set(to_eastern(out[out["rth"]].index).hour)

    assert london_hours == {3}
    assert ny_am_hours == {10}
    assert ny_pm_hours == {14}
    assert 16 not in rth_hours  # 16:00 bar is the boundary, excluded (half-open window)
    assert not (london_hours & ny_am_hours & ny_pm_hours)
