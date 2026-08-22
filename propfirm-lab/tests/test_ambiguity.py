"""Bracket resolution and the ambiguity measurement.

The whole point of this module is refusing to silently decide an undecidable
bar, so the tests are mostly about that refusal being real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.data.ambiguity import (
    AMBIGUOUS,
    STOP_FIRST,
    TARGET_FIRST,
    UNRESOLVED,
    compare_resolutions,
    resolve_bracket,
)


def test_clean_target_hit():
    high = np.array([101.0, 105.0])
    low = np.array([99.5, 100.0])
    assert resolve_bracket(high, low, stop=95.0, target=104.0) == (TARGET_FIRST, 1)


def test_clean_stop_hit():
    high = np.array([101.0, 101.0])
    low = np.array([99.5, 94.0])
    assert resolve_bracket(high, low, stop=95.0, target=110.0) == (STOP_FIRST, 1)


def test_bar_touching_both_is_ambiguous_not_guessed():
    """The bar that motivates this module: it spans both barriers."""
    high = np.array([110.0])
    low = np.array([94.0])
    verdict, idx = resolve_bracket(high, low, stop=95.0, target=104.0)
    assert verdict is AMBIGUOUS or verdict == AMBIGUOUS
    assert idx == 0


def test_nothing_touched_is_unresolved():
    high = np.array([101.0, 102.0])
    low = np.array([99.0, 98.0])
    assert resolve_bracket(high, low, stop=90.0, target=110.0) == (UNRESOLVED, -1)


def test_short_side_inverts_the_barriers():
    # Short from 100: stop above at 105, target below at 96.
    high = np.array([101.0, 106.0])
    low = np.array([99.0, 100.0])
    assert resolve_bracket(high, low, stop=105.0, target=96.0, long=False) == (STOP_FIRST, 1)

    high = np.array([101.0, 101.0])
    low = np.array([99.0, 95.0])
    assert resolve_bracket(high, low, stop=105.0, target=96.0, long=False) == (TARGET_FIRST, 1)


def test_earlier_bar_wins_over_later():
    high = np.array([104.5, 110.0])
    low = np.array([99.0, 90.0])
    # Target is reached on bar 0, long before the stop on bar 1.
    assert resolve_bracket(high, low, stop=95.0, target=104.0) == (TARGET_FIRST, 0)


def _bars(ts, o, h, l):
    return pd.DataFrame({"ts_open": pd.to_datetime(ts, utc=True),
                         "open": o, "high": h, "low": l})


def test_second_resolution_decides_what_the_minute_could_not():
    """One ambiguous minute, resolved by the seconds inside it.

    The minute spans both barriers. The seconds show price reaching the target
    before ever coming near the stop, so the stop-wins convention would have
    booked a winner as a loser.
    """
    minute = _bars(["2024-03-01T14:30:00Z", "2024-03-01T14:31:00Z"],
                   [100.0, 100.0], [110.0, 101.0], [94.0, 99.0])
    seconds = _bars(
        [f"2024-03-01T14:30:{s:02d}Z" for s in range(6)],
        [100.0] * 6,
        [101.0, 105.0, 110.0, 110.0, 100.0, 96.0],   # target (104) hit at s=2
        [100.0, 100.0, 103.0, 100.0, 96.0, 94.0],    # stop (95) only at s=5
    )
    rep = compare_resolutions(
        minute, seconds, stop_points=5.0, target_points=4.0,
        sample_every_minutes=1, horizon_minutes=1,
    )
    assert rep.minute_counts["ambiguous"] == 1
    assert rep.truth_in_ambiguous["target_first"] == 1
    assert rep.stop_wins_convention_error == pytest.approx(1.0)


def test_empty_input_raises_rather_than_reporting_nothing():
    one = _bars(["2024-03-01T14:30:00Z"], [100.0], [110.0], [94.0])
    empty = one.iloc[0:0]
    with pytest.raises(ValueError, match="non-empty bar frames"):
        compare_resolutions(one, empty, stop_points=5.0, target_points=4.0)
    with pytest.raises(ValueError, match="non-empty bar frames"):
        compare_resolutions(empty, one, stop_points=5.0, target_points=4.0)


def test_a_single_bar_still_forms_one_bracket():
    one = _bars(["2024-03-01T14:30:00Z"], [100.0], [110.0], [94.0])
    rep = compare_resolutions(one, one, stop_points=5.0, target_points=4.0,
                              sample_every_minutes=60)
    assert rep.n_brackets == 1


def test_missing_columns_are_rejected():
    good = _bars(["2024-03-01T14:30:00Z"], [100.0], [101.0], [99.0])
    bad = pd.DataFrame({"ts_open": pd.to_datetime(["2024-03-01T14:30:00Z"], utc=True)})
    with pytest.raises(ValueError, match="missing column"):
        compare_resolutions(good, bad, stop_points=5.0, target_points=4.0)
