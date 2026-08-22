"""Vectorized bracket backtester.

Fast enough to test tens of thousands of rule configurations over 3M bars,
which is the point: a search that can only afford a few hundred configurations
tempts you into picking them after glancing at results.

Speed comes from resolving every bracket at once. For N signals and a horizon
of H bars, ``sliding_window_view`` gives an (N, H) view of the forward highs
and lows with no copying, and the first stop-touch and first target-touch
become two argmax calls. No Python loop over trades.

Three rules keep it honest:

* **A signal computed on bar t enters at bar t+1's open.** Never at t's close,
  which is a price you only know once the bar is over.
* **Every position is flat by session end.** Prop-firm accounts with trailing
  drawdown are not held overnight, so a backtest that holds is measuring a
  strategy nobody could run here.
* **A bar spanning both barriers is charged as a stop.** Measured at 0.02-0.14%
  of trades (findings/03), so the convention costs almost nothing -- but it is
  applied in the pessimistic direction rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# NQ: $20 per point, $4.00 commission per round turn, 1 tick (0.25pt) of
# slippage on stop exits. Matches ict_lab/configs/strategy_config.py.
TICK_SIZE = 0.25
POINT_VALUE = 20.0
COMMISSION_RT = 4.00
STOP_SLIPPAGE_TICKS = 1.0


@dataclass
class TradeResult:
    """Per-trade outcomes, aligned with the signal array that produced them."""

    entry_idx: np.ndarray
    exit_idx: np.ndarray
    direction: np.ndarray      # +1 long, -1 short
    entry_price: np.ndarray
    exit_price: np.ndarray
    stop_price: np.ndarray
    target_price: np.ndarray
    hit_stop: np.ndarray
    hit_target: np.ndarray
    r_multiple: np.ndarray
    net_pnl: np.ndarray
    mae_points: np.ndarray
    mfe_points: np.ndarray
    ambiguous: np.ndarray

    def __len__(self) -> int:
        return self.entry_idx.size

    @property
    def expectancy_r(self) -> float:
        """Mean R per trade after costs -- the number the whole project targets."""
        return float(np.mean(self.r_multiple)) if len(self) else 0.0

    @property
    def expectancy_se(self) -> float:
        """Standard error of that mean. Never report the mean without it."""
        n = len(self)
        if n < 2:
            return float("inf")
        return float(np.std(self.r_multiple, ddof=1) / np.sqrt(n))

    @property
    def win_rate(self) -> float:
        return float(np.mean(self.r_multiple > 0)) if len(self) else 0.0


def simulate(
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    signal_idx: np.ndarray,
    direction: np.ndarray,
    stop_points: np.ndarray,
    target_points: np.ndarray,
    session_end_idx: np.ndarray,
    *,
    horizon: int = 240,
) -> TradeResult:
    """Resolve every bracket at once.

    ``signal_idx`` holds bar indices where a signal fired; entry happens at
    ``signal_idx + 1``'s open. ``session_end_idx`` is the last bar index of each
    signal's own session, so positions close there if neither barrier is hit.
    """
    n_bars = high.size
    entry_idx = signal_idx + 1
    keep = entry_idx < n_bars
    entry_idx, direction = entry_idx[keep], direction[keep]
    stop_points, target_points = stop_points[keep], target_points[keep]
    session_end_idx = session_end_idx[keep]

    if entry_idx.size == 0:
        return _empty()

    entry_price = open_[entry_idx]
    stop_price = entry_price - direction * stop_points
    target_price = entry_price + direction * target_points

    # Forward windows, one row per trade. sliding_window_view does not copy.
    pad = horizon
    hi = np.concatenate([high, np.full(pad, np.nan)])
    lo = np.concatenate([low, np.full(pad, np.nan)])
    win_hi = np.lib.stride_tricks.sliding_window_view(hi, horizon)[entry_idx]
    win_lo = np.lib.stride_tricks.sliding_window_view(lo, horizon)[entry_idx]

    offsets = np.arange(horizon)[None, :]
    # Bars past this trade's session end are not tradeable.
    max_off = np.minimum(session_end_idx - entry_idx, horizon - 1)[:, None]
    valid = (offsets <= max_off) & ~np.isnan(win_hi)

    long_mask = (direction == 1)[:, None]
    touch_stop = np.where(long_mask, win_lo <= stop_price[:, None],
                          win_hi >= stop_price[:, None]) & valid
    touch_target = np.where(long_mask, win_hi >= target_price[:, None],
                            win_lo <= target_price[:, None]) & valid

    first_stop = _first_true_offset(touch_stop, horizon)
    first_target = _first_true_offset(touch_target, horizon)

    hit_stop = first_stop <= first_target
    hit_target = first_target < first_stop
    resolved = np.minimum(first_stop, first_target)
    timed_out = resolved >= horizon

    # Anything unresolved exits at its session's last bar.
    exit_off = np.where(timed_out, max_off[:, 0], resolved)
    exit_idx = entry_idx + exit_off

    ambiguous = (first_stop == first_target) & ~timed_out
    hit_stop = hit_stop & ~timed_out
    hit_target = hit_target & ~timed_out

    exit_price = np.where(
        hit_stop,
        stop_price - direction * STOP_SLIPPAGE_TICKS * TICK_SIZE,
        np.where(hit_target, target_price, open_[np.minimum(exit_idx, n_bars - 1)]),
    )

    gross = direction * (exit_price - entry_price) * POINT_VALUE
    net = gross - COMMISSION_RT
    risk_dollars = stop_points * POINT_VALUE
    r_multiple = np.divide(net, risk_dollars, out=np.zeros_like(net),
                           where=risk_dollars > 0)

    excursion_hi = np.where(valid, win_hi, -np.inf).max(axis=1)
    excursion_lo = np.where(valid, win_lo, np.inf).min(axis=1)
    mae = np.where(direction == 1, entry_price - excursion_lo, excursion_hi - entry_price)
    mfe = np.where(direction == 1, excursion_hi - entry_price, entry_price - excursion_lo)

    return TradeResult(
        entry_idx=entry_idx, exit_idx=exit_idx, direction=direction,
        entry_price=entry_price, exit_price=exit_price,
        stop_price=stop_price, target_price=target_price,
        hit_stop=hit_stop, hit_target=hit_target,
        r_multiple=r_multiple, net_pnl=net,
        mae_points=np.maximum(mae, 0.0), mfe_points=np.maximum(mfe, 0.0),
        ambiguous=ambiguous,
    )


def _first_true_offset(mask: np.ndarray, horizon: int) -> np.ndarray:
    """Column of the first True per row, or ``horizon`` if the row is all False."""
    any_true = mask.any(axis=1)
    return np.where(any_true, mask.argmax(axis=1), horizon)


def _empty() -> TradeResult:
    z = np.array([], dtype=float)
    zi = np.array([], dtype=np.int64)
    zb = np.array([], dtype=bool)
    return TradeResult(zi, zi, zi, z, z, z, z, zb, zb, z, z, z, z, zb)


# Bounded, non-overlapping entries. Fixed before any result was computed.
#
# A raw signal mask fires on consecutive bars -- momentum stays true for as long
# as the move lasts -- and taking all of them would book dozens of simultaneous
# entries on one move, which is neither tradeable nor a meaningful test. Real
# intraday strategies take a position and wait for it to resolve.
#
# So: at most MAX_ENTRIES_PER_SESSION trades per session, each starting no
# earlier than the previous one's exit. Implemented as a few rounds of the
# vectorized simulator rather than a Python loop over signals, which keeps the
# whole search affordable.
MAX_ENTRIES_PER_SESSION = 3


def simulate_sequential(
    f,
    signal_idx: np.ndarray,
    direction: np.ndarray,
    stop_points: np.ndarray,
    target_points: np.ndarray,
    *,
    horizon: int = 240,
    max_entries: int = MAX_ENTRIES_PER_SESSION,
) -> TradeResult:
    """Non-overlapping entries, at most ``max_entries`` per session."""
    if signal_idx.size == 0:
        return _empty()

    # A signal on a session's last bar would enter on the first bar of the NEXT
    # session, carrying a setup across the overnight break. Drop those: the
    # entry has to belong to the session that produced the signal.
    entry = signal_idx + 1
    same_session = (entry < f.session_id.size) & (
        f.session_id[np.minimum(entry, f.session_id.size - 1)] == f.session_id[signal_idx]
    )
    signal_idx, direction = signal_idx[same_session], direction[same_session]
    stop_points, target_points = stop_points[same_session], target_points[same_session]
    if signal_idx.size == 0:
        return _empty()

    order = np.argsort(signal_idx, kind="stable")
    signal_idx, direction = signal_idx[order], direction[order]
    stop_points, target_points = stop_points[order], target_points[order]

    session_of_signal = f.session_id[signal_idx]
    # Earliest bar each session is allowed to open a new position on.
    free_from = {int(s): -1 for s in np.unique(session_of_signal)}
    taken_count = {int(s): 0 for s in free_from}

    rounds: list[TradeResult] = []
    for _ in range(max_entries):
        pick = np.zeros(signal_idx.size, dtype=bool)
        seen: set[int] = set()
        for pos in range(signal_idx.size):
            s = int(session_of_signal[pos])
            if s in seen or taken_count[s] >= max_entries:
                continue
            if signal_idx[pos] <= free_from[s]:
                continue
            pick[pos] = True
            seen.add(s)
        if not pick.any():
            break

        res = simulate(
            f.high, f.low, f.open,
            signal_idx[pick], direction[pick],
            stop_points[pick], target_points[pick],
            f.session_end_idx[signal_idx[pick]],
            horizon=horizon,
        )
        rounds.append(res)
        for ent, ex in zip(res.entry_idx, res.exit_idx):
            s = int(f.session_id[ent])
            free_from[s] = int(ex)
            taken_count[s] += 1

    return _concat(rounds) if rounds else _empty()


def _concat(parts: list[TradeResult]) -> TradeResult:
    if len(parts) == 1:
        return parts[0]
    cat = np.concatenate
    order = np.argsort(cat([p.entry_idx for p in parts]), kind="stable")
    return TradeResult(
        **{
            field: cat([getattr(p, field) for p in parts])[order]
            for field in (
                "entry_idx", "exit_idx", "direction", "entry_price", "exit_price",
                "stop_price", "target_price", "hit_stop", "hit_target",
                "r_multiple", "net_pnl", "mae_points", "mfe_points", "ambiguous",
            )
        }
    )
