"""Build an equity path from a real trade log.

This is how a real strategy -- or a real blown account -- gets fed into the
simulator. The input schema is deliberately the one
``ict_lab/engine/execution.py`` already emits, so the sibling ICT lab's output
drops straight in with no translation layer:

    session_date, net_pnl, mae_points, mfe_points, ambiguous_bar, ...

Only five of those columns are actually required (see ``TRADE_LOG_COLUMNS``);
anything else is carried along and ignored.

**The excursion ordering problem.** A trailing floor reacts to the peak, so it
matters whether a trade's favourable excursion happened before or after its
adverse one. A trade log built from 1-minute bars usually cannot tell you --
that is exactly what ``ambiguous_bar`` flags. So this adapter does not guess.
Ask it for ``worst`` and ``best`` and simulate both: the spread between the two
survival estimates is the part your data genuinely cannot resolve, and
reporting a single number in its place is how backtests come to overstate
survival.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

TRADE_LOG_COLUMNS = ("session_date", "net_pnl", "mae_points", "mfe_points")

MARKS_PER_TRADE = 4


class ExcursionOrder(str, Enum):
    WORST = "worst"    # favourable excursion first: ratchets the floor up early
    BEST = "best"      # adverse excursion first
    SETTLED = "settled"  # ignore excursions entirely -- the optimistic error


def path_from_trade_log(
    trades: pd.DataFrame,
    start_equity: float,
    *,
    order: ExcursionOrder | str = ExcursionOrder.WORST,
    point_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(equity, days)`` marks for the simulator.

    ``mae_points``/``mfe_points`` are direction-agnostic magnitudes (as the ICT
    engine emits them), converted to dollars with ``point_value``. ``net_pnl``
    is assumed to already be in dollars and already net of commission.
    """
    order = ExcursionOrder(order)

    missing = [c for c in TRADE_LOG_COLUMNS if c not in trades.columns]
    if missing:
        raise ValueError(f"trade log is missing required columns: {missing}")
    if trades.empty:
        raise ValueError("empty trade log")

    df = trades.sort_values("session_date", kind="stable").reset_index(drop=True)

    settled = df["net_pnl"].to_numpy(dtype=float)
    mae = np.abs(df["mae_points"].to_numpy(dtype=float)) * point_value
    mfe = np.abs(df["mfe_points"].to_numpy(dtype=float)) * point_value

    equity_before = start_equity + np.concatenate(([0.0], np.cumsum(settled)[:-1]))
    equity_after = start_equity + np.cumsum(settled)

    if order is ExcursionOrder.SETTLED:
        first = np.zeros_like(settled)
        second = np.zeros_like(settled)
    elif order is ExcursionOrder.WORST:
        first, second = mfe, -mae
    else:  # BEST
        first, second = -mae, mfe

    marks = np.empty(len(df) * MARKS_PER_TRADE, dtype=float)
    marks[0::MARKS_PER_TRADE] = equity_before
    marks[1::MARKS_PER_TRADE] = equity_before + first
    marks[2::MARKS_PER_TRADE] = equity_before + second
    marks[3::MARKS_PER_TRADE] = equity_after

    # Day index: consecutive integers per distinct session_date, in order.
    codes = pd.factorize(df["session_date"], sort=False)[0]
    days = np.repeat(codes.astype(np.int64), MARKS_PER_TRADE)

    return marks, days
