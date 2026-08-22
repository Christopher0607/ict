"""Strategy families for the search.

Each family turns the feature matrix into (bar index, direction) pairs. They
are deliberately simple and well-known -- opening-range breakouts, momentum,
mean reversion, VWAP deviation, prior-day levels, time-of-day. That is the
point: if a simple rule on the most heavily traded index future in the world
carried a tradeable edge, it would be arbitraged. Testing them is how you find
out what the honest baseline is, not a hope that one works.

Every family reads only causal features (see features.py) and every entry is
executed on the following bar by backtest.simulate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.search.features import FeatureSet


@dataclass(frozen=True)
class Signal:
    idx: np.ndarray        # bar indices where the rule fires
    direction: np.ndarray  # +1 long, -1 short


def _tradeable(f: FeatureSet, entry_from: int, entry_to: int) -> np.ndarray:
    """Bars eligible for entry: inside RTH, inside the allowed window, ATR known."""
    return (
        f.is_rth
        & (f.minutes_into_rth >= entry_from)
        & (f.minutes_into_rth < entry_to)
        & ~np.isnan(f.atr)
        & (f.atr > 0)
    )


def _emit(mask_long: np.ndarray, mask_short: np.ndarray, side: str) -> Signal:
    if side == "long":
        mask_short = np.zeros_like(mask_short)
    elif side == "short":
        mask_long = np.zeros_like(mask_long)
    idx = np.flatnonzero(mask_long | mask_short)
    direction = np.where(mask_long[idx], 1, -1).astype(np.int64)
    return Signal(idx, direction)


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------


def opening_range_breakout(f, *, or_minutes, entry_from, entry_to, side, **_):
    hi, lo = f.or_high[or_minutes], f.or_low[or_minutes]
    ok = _tradeable(f, entry_from, entry_to) & ~np.isnan(hi)
    return _emit(ok & (f.close > hi), ok & (f.close < lo), side)


def momentum(f, *, lookback, threshold_atr, entry_from, entry_to, side, **_):
    r = f.ret[lookback]
    ok = _tradeable(f, entry_from, entry_to) & ~np.isnan(r)
    thr = threshold_atr * f.atr
    return _emit(ok & (r > thr), ok & (r < -thr), side)


def mean_reversion(f, *, lookback, threshold_atr, entry_from, entry_to, side, **_):
    """Momentum with the direction reversed -- fading the move instead."""
    r = f.ret[lookback]
    ok = _tradeable(f, entry_from, entry_to) & ~np.isnan(r)
    thr = threshold_atr * f.atr
    return _emit(ok & (r < -thr), ok & (r > thr), side)


def range_breakout(f, *, lookback, entry_from, entry_to, side, **_):
    hi, lo = f.range_hi[lookback], f.range_lo[lookback]
    ok = _tradeable(f, entry_from, entry_to) & ~np.isnan(hi)
    return _emit(ok & (f.close > hi), ok & (f.close < lo), side)


def vwap_reversion(f, *, threshold_atr, entry_from, entry_to, side, **_):
    dev = f.close - f.vwap
    ok = _tradeable(f, entry_from, entry_to) & ~np.isnan(dev)
    thr = threshold_atr * f.atr
    return _emit(ok & (dev < -thr), ok & (dev > thr), side)


def prior_day_break(f, *, entry_from, entry_to, side, **_):
    ok = _tradeable(f, entry_from, entry_to) & ~np.isnan(f.prior_high)
    return _emit(ok & (f.close > f.prior_high), ok & (f.close < f.prior_low), side)


def time_of_day(f, *, entry_minute, side, **_):
    """Enter at a fixed minute into RTH regardless of price.

    A deliberate near-null: if this shows an edge, it is a calendar effect, and
    if the search turns up nothing better than this then nothing in the search
    is reading price at all.
    """
    ok = _tradeable(f, entry_minute, entry_minute + 1)
    return _emit(ok, ok, side if side != "both" else "long")


FAMILIES = {
    "orb": opening_range_breakout,
    "momentum": momentum,
    "mean_reversion": mean_reversion,
    "range_breakout": range_breakout,
    "vwap_reversion": vwap_reversion,
    "prior_day_break": prior_day_break,
    "time_of_day": time_of_day,
}
