"""Causality of the feature matrix.

The headline is the truncation harness, copied in spirit from the ICT lab's
tests/test_no_lookahead.py: build features on the full series, build them again
on the series cut at bar k, and require that everything knowable by k agrees.
A feature that reads one bar ahead backtests beautifully and cannot be traded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.search.features import build


def _synthetic(n_sessions: int = 6, bars_per_session: int = 400, seed: int = 3):
    """Bars spanning RTH so opening ranges and prior-day levels are exercised."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        day = pd.Timestamp("2026-03-02", tz="UTC") + pd.Timedelta(days=s)
        start = day + pd.Timedelta(hours=13, minutes=30)  # 09:30 ET (EDT)
        price = 20_000 + s * 10
        for m in range(bars_per_session):
            step = rng.normal(0, 3)
            o = price
            c = price + step
            hi = max(o, c) + abs(rng.normal(0, 1))
            lo = min(o, c) - abs(rng.normal(0, 1))
            rows.append((start + pd.Timedelta(minutes=m), o, hi, lo, c,
                         float(rng.integers(50, 2000))))
            price = c
    return pd.DataFrame(rows, columns=["ts_open", "open", "high", "low", "close", "volume"])


def test_no_lookahead_under_truncation():
    df = _synthetic()
    cut = 900

    full = build(df)
    trunc = build(df.iloc[:cut].reset_index(drop=True))

    # Compare only bars the truncated run could have completed.
    k = cut - 1

    def agree(a, b, name):
        a, b = a[:k], b[:k]
        both = ~np.isnan(a) & ~np.isnan(b)
        assert np.allclose(a[both], b[both]), f"{name} disagrees under truncation"
        # A value known in the truncated run must also be known in the full run.
        assert not (np.isnan(a) & ~np.isnan(b)).any(), f"{name} lost values in full run"

    agree(full.atr, trunc.atr, "atr")
    agree(full.vwap, trunc.vwap, "vwap")
    agree(full.rel_volume, trunc.rel_volume, "rel_volume")
    for n in full.ret:
        agree(full.ret[n], trunc.ret[n], f"ret[{n}]")
        agree(full.range_hi[n], trunc.range_hi[n], f"range_hi[{n}]")
        agree(full.range_lo[n], trunc.range_lo[n], f"range_lo[{n}]")


def test_trailing_extremes_exclude_the_current_bar():
    """A breakout must not be measured against a high that includes itself."""
    df = _synthetic(n_sessions=1, bars_per_session=120)
    f = build(df, lookbacks=(10,))
    i = 60
    manual = df["high"].iloc[i - 10:i].max()
    assert f.range_hi[10][i] == pytest.approx(manual)
    assert f.range_hi[10][i] != pytest.approx(max(manual, df["high"].iloc[i]))  # unless equal by luck


def test_opening_range_is_nan_until_it_is_complete():
    """Trading the 30-minute range at minute 5 is reading the future."""
    df = _synthetic(n_sessions=2, bars_per_session=200)
    f = build(df, or_minutes=(30,))
    early = (f.minutes_into_rth >= 0) & (f.minutes_into_rth < 30)
    assert np.isnan(f.or_high[30][early]).all()
    later = f.minutes_into_rth >= 30
    assert not np.isnan(f.or_high[30][later]).all()


def test_opening_range_matches_a_manual_computation():
    df = _synthetic(n_sessions=1, bars_per_session=200)
    f = build(df, or_minutes=(15,))
    first_session = f.session_id == f.session_id[0]
    window = first_session & (f.minutes_into_rth >= 0) & (f.minutes_into_rth < 15)
    expected = f.high[window].max()
    done = first_session & (f.minutes_into_rth >= 15)
    assert f.or_high[15][done][0] == pytest.approx(expected)


def test_prior_session_levels_come_from_the_previous_session():
    df = _synthetic(n_sessions=3, bars_per_session=200)
    f = build(df)
    sessions = np.unique(f.session_id)
    s0, s1 = sessions[0], sessions[1]
    rth0 = (f.session_id == s0) & f.is_rth
    expected_high = f.high[rth0].max()
    in_s1 = f.session_id == s1
    assert f.prior_high[in_s1][0] == pytest.approx(expected_high)


def test_first_session_has_no_prior_levels():
    df = _synthetic(n_sessions=2, bars_per_session=200)
    f = build(df)
    first = f.session_id == f.session_id[0]
    assert np.isnan(f.prior_high[first]).all()


def test_session_end_index_points_at_each_session_last_bar():
    df = _synthetic(n_sessions=3, bars_per_session=100)
    f = build(df)
    for s in np.unique(f.session_id):
        in_s = f.session_id == s
        last = np.flatnonzero(in_s)[-1]
        assert (f.session_end_idx[in_s] == last).all()


def test_vwap_is_cumulative_within_a_session_and_resets():
    df = _synthetic(n_sessions=2, bars_per_session=100)
    f = build(df)
    s1 = np.flatnonzero(f.session_id == f.session_id[-1])
    first_of_s1 = s1[0]
    # The first bar of a session has VWAP equal to its own close.
    assert f.vwap[first_of_s1] == pytest.approx(f.close[first_of_s1])


def test_a_thirty_minute_range_cannot_be_traded_in_a_thirty_minute_window():
    """The reason 5- and 10-minute opening ranges had to be added.

    An opening range is NaN until it is complete, so a 30-minute range first
    becomes tradeable at minute 30 -- exactly when a 09:30-10:00 entry window
    closes. Searching the short window with the old or_minutes would have
    silently produced nothing and looked like a null result.
    """
    import numpy as np

    from research.search.features import build
    from research.search.rules import FAMILIES
    from tests.test_predictability import _series

    f = build(_series(n_sessions=30, bars=120, seed=5))
    orb = FAMILIES["orb"]

    late = orb(f, or_minutes=30, entry_from=0, entry_to=30, side="both")
    assert late.idx.size == 0, "a 30-minute range fired inside a 30-minute window"

    early = orb(f, or_minutes=5, entry_from=0, entry_to=30, side="both")
    assert early.idx.size > 0
    assert (f.minutes_into_rth[early.idx] >= 5).all()
    assert (f.minutes_into_rth[early.idx] < 30).all()


def test_opening_drive_is_not_knowable_before_the_drive_completes():
    import numpy as np

    from research.search.features import build
    from research.search.rules import FAMILIES
    from tests.test_predictability import _series

    f = build(_series(n_sessions=30, bars=120, seed=6))
    sig = FAMILIES["opening_drive"](
        f, drive_minutes=10, threshold_atr=0.5, entry_from=0, entry_to=60,
        side="both",
    )
    assert sig.idx.size > 0
    assert (f.minutes_into_rth[sig.idx] >= 10).all(), "fired before the drive closed"
