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
import pandas as pd

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


# ---------------------------------------------------------------------------
# Second round: the untested 71%, and hypotheses with documented priors
# ---------------------------------------------------------------------------
#
# The first eight families all required is_rth, so 2,158,676 bars -- 71.6% of
# the development window -- were never tested at all. These reach them, and add
# four ideas chosen for their standing in the literature rather than to widen
# coverage: overnight gap behaviour, turn-of-month, range compression, and the
# overnight/weekly levels that extend the only two families whose gross
# expectancy was not negative.

ETH_WINDOWS = {
    "asia": (18 * 60, 24 * 60),
    "europe": (0, 3 * 60),
    "london": (3 * 60, 9 * 60 + 30),
    "all_eth": (-1, -1),      # any bar outside RTH
}


def _eth_tradeable(f, window: str) -> np.ndarray:
    ok = ~f.is_rth & ~np.isnan(f.atr) & (f.atr > 0)
    if window == "all_eth":
        return ok
    lo, hi = ETH_WINDOWS[window]
    return ok & (f.et_minute >= lo) & (f.et_minute < hi)


def eth_momentum(f, *, eth_window, lookback, threshold_atr, side, **_):
    r = f.ret[lookback]
    ok = _eth_tradeable(f, eth_window) & ~np.isnan(r)
    thr = threshold_atr * f.atr
    return _emit(ok & (r > thr), ok & (r < -thr), side)


def eth_reversion(f, *, eth_window, lookback, threshold_atr, side, **_):
    r = f.ret[lookback]
    ok = _eth_tradeable(f, eth_window) & ~np.isnan(r)
    thr = threshold_atr * f.atr
    return _emit(ok & (r < -thr), ok & (r > thr), side)


def eth_range_breakout(f, *, eth_window, lookback, side, **_):
    hi, lo = f.range_hi[lookback], f.range_lo[lookback]
    ok = _eth_tradeable(f, eth_window) & ~np.isnan(hi)
    return _emit(ok & (f.close > hi), ok & (f.close < lo), side)


def gap_trade(f, *, min_gap_atr, mode, entry_from, entry_to, side, **_):
    """Overnight gap: fade it back toward the prior close, or follow it.

    ``mode='fade'`` buys a gap down and sells a gap up; ``mode='follow'`` does
    the opposite. Both are documented in the literature, with the sign of the
    effect depending on gap size, which is why min_gap_atr is a grid axis.
    """
    ok = _tradeable(f, entry_from, entry_to) & ~np.isnan(f.session_gap)
    thr = min_gap_atr * f.atr
    gap_up = ok & (f.session_gap > thr)
    gap_dn = ok & (f.session_gap < -thr)
    if mode == "fade":
        return _emit(gap_dn, gap_up, side)
    return _emit(gap_up, gap_dn, side)


def turn_of_month(f, *, window_days, entry_from, entry_to, side, **_):
    """Long into the turn of the month, one of the more durable calendar
    effects in equity indices. Counted in trading days, not dates."""
    ok = _tradeable(f, entry_from, entry_to)
    near_end = f.days_to_month_end <= window_days
    near_start = f.trading_day_of_month <= window_days
    inside = ok & (near_end | near_start)
    return _emit(inside, ok & ~(near_end | near_start), side)


def day_of_week(f, *, weekday, entry_from, entry_to, side, **_):
    ok = _tradeable(f, entry_from, entry_to) & (f.day_of_week == weekday)
    return _emit(ok, ok, side if side != "both" else "long")


def compression_breakout(f, *, lookback, max_compression, side, entry_from, entry_to, **_):
    """Break out of a range that has gone quiet relative to its own history."""
    comp = f.compression[lookback]
    hi, lo = f.range_hi[lookback], f.range_lo[lookback]
    ok = (
        _tradeable(f, entry_from, entry_to)
        & ~np.isnan(comp) & ~np.isnan(hi)
        & (comp <= max_compression)
    )
    return _emit(ok & (f.close > hi), ok & (f.close < lo), side)


def overnight_level_break(f, *, entry_from, entry_to, side, **_):
    """Break of the overnight range, published the moment RTH opens.

    Extends prior_day_break and orb -- the only two first-round families whose
    gross expectancy was not negative.
    """
    ok = _tradeable(f, entry_from, entry_to) & ~np.isnan(f.overnight_high)
    return _emit(ok & (f.close > f.overnight_high), ok & (f.close < f.overnight_low), side)


FAMILIES.update({
    "eth_momentum": eth_momentum,
    "eth_reversion": eth_reversion,
    "eth_range_breakout": eth_range_breakout,
    "gap_trade": gap_trade,
    "turn_of_month": turn_of_month,
    "day_of_week": day_of_week,
    "compression_breakout": compression_breakout,
    "overnight_level_break": overnight_level_break,
})


# ---------------------------------------------------------------------------
# The one-line rival
# ---------------------------------------------------------------------------
#
# Registered after the confidence diagnostic, and it is a control rather than a
# hope. Sorting the model by its own confidence selects bars that are 90% long
# and sit at a median ATR of 5.6 against 7.4 for the sample -- so the question
# is whether seventeen features, a monthly refit and a purged walk-forward beat
# "be long when the market is calm".
#
# Measured over the same bars with no execution model: at h=60 the model's
# confident set earns $52.28 a trade, always-long on those same bars earns
# $40.84, and this rule earns $36.08 at the same t-statistic. That is the
# comparison this family puts through the real backtester -- non-overlapping
# entries, a stop, flat by the close -- where it can be judged against +0.185R
# like everything else.
#
# The threshold is a fixed ratio to the ATR's own trailing average, not a
# sample quantile. A quantile of the whole sample would need the whole sample.


def low_vol_long(f, *, max_rel_atr, entry_from, entry_to, side, **_):
    """Take a position whenever ATR sits below its own trailing average."""
    trailing = pd.Series(f.atr).rolling(1440, min_periods=200).mean().to_numpy()
    rel = np.divide(f.atr, trailing, out=np.full(f.atr.shape, np.nan),
                    where=np.isfinite(trailing) & (trailing > 0))
    ok = _tradeable(f, entry_from, entry_to) & np.isfinite(rel) & (rel <= max_rel_atr)
    return _emit(ok, ok, side if side != "both" else "long")


FAMILIES["low_vol_long"] = low_vol_long


# ---------------------------------------------------------------------------
# The New York open
# ---------------------------------------------------------------------------
#
# Registered for round four, which targets 09:30-10:00 and 09:30-10:30
# specifically. The window is worth its own families for a measurable reason:
# median ATR in the first thirty minutes is 8.91 points against 5.75 across the
# whole session, and since stops are ATR-scaled, commission is a 35% smaller
# fraction of risk there than anywhere else in the day. That is the only
# structural advantage this round has, and it is working against a micro
# contract whose commission per dollar of risk is 1.8-3.4x the full-size one.


def opening_drive(f, *, drive_minutes, threshold_atr, entry_from, entry_to, side, **_):
    """Follow the direction of the first N minutes of the cash session.

    The drive is measured from the session's opening price to the close of bar
    N-1, and is only knowable from bar N onward -- the same rule the opening
    range obeys. Below the threshold the session has no drive and nothing fires.
    """
    sess = pd.DataFrame({"s": f.session_id, "o": f.open, "c": f.close,
                         "m": f.minutes_into_rth})
    first_open = sess.loc[sess["m"] == 0].groupby("s")["o"].first()
    drive_close = (sess.loc[sess["m"] == drive_minutes - 1]
                   .groupby("s")["c"].first())
    drive = (drive_close - first_open).reindex(
        pd.Index(f.session_id).unique()).to_dict()
    move = pd.Series(f.session_id).map(drive).to_numpy(float, copy=True)
    # Not knowable until the drive window has closed.
    move[f.minutes_into_rth < drive_minutes] = np.nan

    ok = _tradeable(f, entry_from, entry_to) & np.isfinite(move)
    thr = threshold_atr * f.atr
    return _emit(ok & (move > thr), ok & (move < -thr), side)


FAMILIES["opening_drive"] = opening_drive
