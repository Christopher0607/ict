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


# ---------------------------------------------------------------------------
# ICT Silver Bullet
# ---------------------------------------------------------------------------
#
# The setup sequence ict_lab/engine/signals.py implements: inside a killzone,
# liquidity is swept, structure shifts against the sweep, and entry is the
# first fair-value gap in the new direction. Each stage must occur strictly
# after the last -- that ordering is the strategy.
#
# Killzones are ET clock windows, not RTH offsets: London runs 03:00-04:00 ET,
# well before the cash open.
KILLZONES = {
    "london": (3 * 60, 4 * 60),
    "ny_am": (10 * 60, 11 * 60),
    "ny_pm": (14 * 60, 15 * 60),
}


def ict_silver_bullet(
    f,
    *,
    killzone,
    sweep_lookback,
    require_mss,
    require_displacement,
    swing_n=5,
    displacement_atr=1.5,
    side="both",
    **_,
):
    """Sweep, then structure shift, then the first gap in the new direction."""
    import pandas as pd

    from research.search.ict_features import (
        displacement as _disp,
        fair_value_gaps,
        market_structure_shift,
        swing_points,
        sweeps as _sweeps,
    )

    lo_min, hi_min = KILLZONES[killzone]
    in_kz = (f.et_minute >= lo_min) & (f.et_minute < hi_min) & ~np.isnan(f.atr)
    if not in_kz.any():
        return Signal(np.array([], np.int64), np.array([], np.int64))

    # Liquidity taken is measured against the trailing extremes the sweep
    # lookback defines -- levels that exist before the bar that sweeps them.
    lvl_hi = f.range_hi.get(sweep_lookback)
    lvl_lo = f.range_lo.get(sweep_lookback)
    if lvl_hi is None:
        return Signal(np.array([], np.int64), np.array([], np.int64))

    swept_hi, swept_lo = _sweeps(f.high, f.low, f.close, lvl_hi, lvl_lo)
    bull_fvg, bear_fvg = fair_value_gaps(f.high, f.low)
    sh, sl, _, _ = swing_points(f.high, f.low, n=swing_n)
    mss_up, mss_down = market_structure_shift(f.close, sh, sl)
    disp = _disp(f.high, f.low, f.atr, displacement_atr)

    # One group per session-killzone occurrence.
    group = np.where(in_kz, f.session_id, -1)
    idx = np.flatnonzero(in_kz)
    if idx.size == 0:
        return Signal(np.array([], np.int64), np.array([], np.int64))

    tab = pd.DataFrame({
        "i": idx,
        "g": group[idx],
        # A swept high is bearish: buy-side liquidity was taken and rejected.
        "sweep_short": swept_hi[idx],
        "sweep_long": swept_lo[idx],
        "mss_up": mss_up[idx],
        "mss_down": mss_down[idx],
        "fvg_up": bull_fvg[idx],
        "fvg_dn": bear_fvg[idx],
        "disp": disp[idx],
    })

    out_idx, out_dir = [], []
    for _, grp in tab.groupby("g", sort=False):
        for direction, sweep_col, mss_col, fvg_col in (
            (1, "sweep_long", "mss_up", "fvg_up"),
            (-1, "sweep_short", "mss_down", "fvg_dn"),
        ):
            if side == "long" and direction == -1:
                continue
            if side == "short" and direction == 1:
                continue

            sw = grp.loc[grp[sweep_col]]
            if sw.empty:
                continue
            cursor = int(sw["i"].iloc[0])

            if require_displacement:
                d = grp.loc[(grp["i"] > cursor) & grp["disp"]]
                if d.empty:
                    continue
                cursor = int(d["i"].iloc[0])

            if require_mss:
                m = grp.loc[(grp["i"] > cursor) & grp[mss_col]]
                if m.empty:
                    continue
                cursor = int(m["i"].iloc[0])

            g = grp.loc[(grp["i"] > cursor) & grp[fvg_col]]
            if g.empty:
                continue
            out_idx.append(int(g["i"].iloc[0]))
            out_dir.append(direction)

    if not out_idx:
        return Signal(np.array([], np.int64), np.array([], np.int64))
    order = np.argsort(out_idx)
    return Signal(np.array(out_idx, np.int64)[order], np.array(out_dir, np.int64)[order])


FAMILIES["ict_silver_bullet"] = ict_silver_bullet
