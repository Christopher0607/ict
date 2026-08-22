"""ICT detectors: definitions, the look-ahead trap, and fidelity to ict_lab.

The reimplementation exists for speed, so the thing that matters is that it
agrees with the engine it replaces. The last test compares both on real NQ
bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.search.ict_features import (
    displacement,
    fair_value_gaps,
    fvg_edges,
    market_structure_shift,
    sweeps,
    swing_points,
)


def test_bullish_fvg_is_a_three_bar_imbalance():
    # bar 2's low (105) sits above bar 0's high (101): untraded band 101..105.
    high = np.array([101.0, 104.0, 108.0])
    low = np.array([99.0, 102.0, 105.0])
    bull, bear = fair_value_gaps(high, low)
    assert bull[2] and not bear[2]
    top, bottom = fvg_edges(high, low)
    assert top[2] == pytest.approx(105.0)
    assert bottom[2] == pytest.approx(101.0)


def test_bearish_fvg_is_the_mirror():
    high = np.array([108.0, 104.0, 99.0])
    low = np.array([105.0, 102.0, 97.0])
    bull, bear = fair_value_gaps(high, low)
    assert bear[2] and not bull[2]
    top, bottom = fvg_edges(high, low)
    assert top[2] == pytest.approx(105.0)
    assert bottom[2] == pytest.approx(99.0)


def test_overlapping_bars_are_not_a_gap():
    high = np.array([101.0, 102.0, 103.0])
    low = np.array([99.0, 100.0, 100.5])
    bull, bear = fair_value_gaps(high, low)
    assert not bull.any() and not bear.any()


def test_swings_are_reported_only_when_confirmable():
    """The look-ahead trap: a fractal high at i needs n bars after i to exist."""
    n = 2
    high = np.array([1, 2, 9, 2, 1, 1, 1, 1, 1], float)
    low = np.full(9, 0.0)
    hi_price, _, hi_idx, _ = swing_points(high, low, n=n)

    # The swing sits at bar 2 and cannot be known before bar 4.
    assert hi_idx[3] != 2
    assert hi_idx[4] == 2
    assert hi_price[4] == pytest.approx(9.0)


def test_swing_price_is_the_most_recent_confirmed_one():
    n = 1
    high = np.array([1, 5, 1, 1, 9, 1, 1], float)
    low = np.zeros(7)
    hi_price, _, _, _ = swing_points(high, low, n=n)
    assert hi_price[2] == pytest.approx(5.0)
    assert hi_price[5] == pytest.approx(9.0)


def test_sweep_needs_penetration_and_rejection():
    level_high = np.array([100.0, 100.0, 100.0])
    level_low = np.array([90.0, 90.0, 90.0])
    high = np.array([101.0, 101.0, 99.0])
    low = np.array([95.0, 95.0, 95.0])
    close = np.array([99.0, 100.5, 98.0])   # rejected, held above, never traded through
    swept_high, _ = sweeps(high, low, close, level_high, level_low)
    assert swept_high[0]          # pierced and closed back below
    assert not swept_high[1]      # pierced but closed above -- a break, not a sweep
    assert not swept_high[2]      # never reached the level


def test_displacement_is_relative_to_atr():
    high = np.array([110.0, 101.0])
    low = np.array([100.0, 100.0])
    atr = np.array([4.0, 4.0])
    d = displacement(high, low, atr, mult=1.5)
    assert d[0] and not d[1]


def test_mss_requires_closing_beyond_a_confirmed_swing():
    close = np.array([100.0, 106.0])
    swing_hi = np.array([105.0, 105.0])
    swing_lo = np.array([95.0, 95.0])
    up, down = market_structure_shift(close, swing_hi, swing_lo)
    assert not up[0] and up[1]
    assert not down.any()


def test_matches_ict_lab_on_real_bars():
    """Fidelity check against the engine this replaces, on real NQ data."""
    ict = pytest.importorskip("ict_lab.features.swings")
    from research.data.export_dataset import load_exported

    df = load_exported("NQ").iloc[200_000:206_000].reset_index(drop=True)
    frame = df.set_index("ts_open")[["open", "high", "low", "close", "volume"]]

    theirs = ict.swing_points(frame, n=5)
    their_highs = set(theirs.loc[theirs["kind"] == "high", "bar_index"])

    _, _, hi_idx, _ = swing_points(
        frame["high"].to_numpy(float), frame["low"].to_numpy(float), n=5
    )
    mine = set(int(i) for i in np.unique(hi_idx) if i >= 0)

    # Every swing ict_lab found, this implementation found too.
    assert their_highs.issubset(mine | {-1}), (
        f"missed {len(their_highs - mine)} of {len(their_highs)} ict_lab swing highs"
    )
