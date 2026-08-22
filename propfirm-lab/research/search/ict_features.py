"""ICT detectors, vectorized.

ict_lab implements all of this already and is correct -- one of its trades was
hand-verified bar by bar against raw NQ (findings notes, 2019-03-26). It is
also too slow to use: its own Phase 4 frequency diagnostic ran 60 minutes on
the real development window and produced no output before being killed, and its
64,512-configuration sweep is unreachable at that speed.

So the detectors are re-expressed here in the same vectorized style as the rest
of the search, which lets the ICT family face the identical pre-registered
grid, cost model and multiple-testing correction as everything else. Faithful
to ict_lab's semantics is the requirement; ``tests/test_ict_features.py``
checks the reimplementation against the engine's own output.

The look-ahead trap in this material is swing points. An N-bar fractal high at
bar i is not knowable until bar i+N, because it needs N bars on its right to be
a fractal at all. Marking it at i -- which is where it visually belongs on a
chart -- reads the future by N bars and is the single easiest way to make an
ICT backtest look profitable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fair_value_gaps(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Three-bar imbalance, knowable at the third bar.

    Bullish: bar t's low sits above bar t-2's high, leaving an untraded band.
    Bearish: the mirror. Returns (bullish, bearish) boolean masks aligned to t.
    """
    n = high.size
    bull = np.zeros(n, bool)
    bear = np.zeros(n, bool)
    if n < 3:
        return bull, bear
    bull[2:] = low[2:] > high[:-2]
    bear[2:] = high[2:] < low[:-2]
    return bull, bear


def fvg_edges(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The gap boundaries, for entry placement. NaN where no gap exists."""
    n = high.size
    top = np.full(n, np.nan)
    bottom = np.full(n, np.nan)
    if n < 3:
        return top, bottom
    bull, bear = fair_value_gaps(high, low)
    # Bullish gap spans high[t-2] .. low[t]; bearish spans high[t] .. low[t-2].
    top[2:] = np.where(bull[2:], low[2:], np.where(bear[2:], low[:-2], np.nan))
    bottom[2:] = np.where(bull[2:], high[:-2], np.where(bear[2:], high[2:], np.nan))
    return top, bottom


def swing_points(high: np.ndarray, low: np.ndarray, n: int = 5):
    """N-bar fractal swings, reported at the bar they become KNOWABLE.

    ``swing_high_price[t]`` is the price of the most recent swing high that is
    confirmed as of bar t -- never one that needs bars after t to exist. The
    index it came from is returned alongside so callers can require a break to
    happen strictly after the swing itself.
    """
    size = high.size
    is_high = np.zeros(size, bool)
    is_low = np.zeros(size, bool)
    if size < 2 * n + 1:
        return (np.full(size, np.nan), np.full(size, np.nan),
                np.full(size, -1, np.int64), np.full(size, -1, np.int64))

    win_h = np.lib.stride_tricks.sliding_window_view(high, 2 * n + 1)
    win_l = np.lib.stride_tricks.sliding_window_view(low, 2 * n + 1)
    centre = slice(n, size - n)
    # Equality with the window extreme, not argmax: ict_lab marks every bar
    # that ties the extreme, and argmax would keep only the first of them.
    is_high[centre] = high[centre] == win_h.max(axis=1)
    is_low[centre] = low[centre] == win_l.min(axis=1)

    # Shift right by n: a fractal centred at i is only confirmed at i+n.
    conf_high = np.zeros(size, bool)
    conf_low = np.zeros(size, bool)
    conf_high[n:] = is_high[:-n]
    conf_low[n:] = is_low[:-n]

    idx = np.arange(size)
    hi_idx = np.where(conf_high, idx - n, -1)
    lo_idx = np.where(conf_low, idx - n, -1)
    hi_idx = np.maximum.accumulate(hi_idx)
    lo_idx = np.maximum.accumulate(lo_idx)

    hi_price = np.where(hi_idx >= 0, high[np.maximum(hi_idx, 0)], np.nan)
    lo_price = np.where(lo_idx >= 0, low[np.maximum(lo_idx, 0)], np.nan)
    return hi_price, lo_price, hi_idx, lo_idx


def sweeps(high, low, close, level_high, level_low):
    """Liquidity taken and rejected: trade through a level, close back inside.

    Returns (swept_high, swept_low). A swept high is bearish -- buy-side
    liquidity above was taken and rejected -- and vice versa.
    """
    swept_high = (high > level_high) & (close < level_high) & ~np.isnan(level_high)
    swept_low = (low < level_low) & (close > level_low) & ~np.isnan(level_low)
    return swept_high, swept_low


def displacement(high, low, atr, mult: float = 1.5) -> np.ndarray:
    """A bar whose range is large relative to recent volatility."""
    with np.errstate(invalid="ignore"):
        return (high - low) > mult * atr


def market_structure_shift(close, swing_high_price, swing_low_price):
    """Close beyond the most recently confirmed swing, in each direction."""
    up = close > swing_high_price
    down = close < swing_low_price
    return np.nan_to_num(up, nan=0).astype(bool), np.nan_to_num(down, nan=0).astype(bool)


def first_true_after(mask: np.ndarray, group: np.ndarray, after_idx: np.ndarray) -> np.ndarray:
    """Per group, the first index where ``mask`` holds strictly after ``after_idx``.

    ``after_idx`` is one value per group. Returns -1 where nothing qualifies.
    Used to chain the setup: sweep, then structure shift, then gap, each
    strictly later than the last.
    """
    out = {}
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.full(np.unique(group).size, -1, np.int64)
    g = group[idx]
    order = np.lexsort((idx, g))
    idx, g = idx[order], g[order]
    lookup = dict(zip(np.unique(group), after_idx))
    for gi, ii in zip(g, idx):
        if gi in out:
            continue
        if ii > lookup.get(gi, -1) >= 0:
            out[gi] = ii
    return np.array([out.get(k, -1) for k in np.unique(group)], np.int64)
