"""Turns signals into simulated trades: entry, exits, fills, costs, and the
trade / no-trade logs. Only ever scans a single session's own bars per
trade (grouped once up front), never the full multi-year series, matching
the "fills iterate over killzone bars only" performance requirement.
"""
from __future__ import annotations

import pandas as pd

from ict_lab.configs.strategy_config import COST_MODELS, StrategyConfig
from ict_lab.data.sessions import add_session_columns

TRADE_COLUMNS = [
    "session_date",
    "symbol",
    "window",
    "direction",
    "setup_at",
    "entry_at",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_at",
    "exit_price",
    "exit_reason",
    "bars_held",
    "gross_pnl",
    "net_pnl",
    "r_multiple",
    "mae_points",
    "mfe_points",
    "ambiguous_bar",
    "is_roll_day",
    "config",
]
NO_TRADE_COLUMNS = ["session_date", "window", "reason"]


def _snap_to_tick(price: float, tick_size: float) -> float:
    return round(round(price / tick_size) * tick_size, 10)


def _entry_price(direction: str, entry_level: str, fvg_top: float, fvg_bottom: float, tick_size: float) -> float:
    proximal, distal = (fvg_top, fvg_bottom) if direction == "bullish" else (fvg_bottom, fvg_top)
    raw = {"proximal": proximal, "distal": distal, "50%": (fvg_top + fvg_bottom) / 2}[entry_level]
    return _snap_to_tick(raw, tick_size)


def _simulate_entry(
    direction: str, entry_price: float, setup_at: pd.Timestamp, window_end: pd.Timestamp, session_bars: pd.DataFrame
) -> pd.Timestamp | None:
    """Limit order active from the bar after setup confirmation; fills only
    on a strict through-trade (touching exactly does not fill); cancelled
    if unfilled by window end."""
    scan = session_bars[(session_bars.index > setup_at) & (session_bars.index <= window_end)]
    fills = scan["low"] < entry_price if direction == "bullish" else scan["high"] > entry_price
    hits = fills.to_numpy().nonzero()[0]
    return scan.index[hits[0]] if len(hits) else None


def _stop_price(
    direction: str,
    entry_price: float,
    config: StrategyConfig,
    tick_size: float,
    fvg_top: float,
    fvg_bottom: float,
    swept_level_price: float | None,
) -> float:
    buffer = config.stop_buffer_ticks * tick_size
    if config.stop_type == "swing":
        if swept_level_price is None:
            raise ValueError(
                "stop_type='swing' needs a swept level on the signal -- was sweep_required set?"
            )
        raw = swept_level_price - buffer if direction == "bullish" else swept_level_price + buffer
    elif config.stop_type == "gap_distal":
        distal = fvg_bottom if direction == "bullish" else fvg_top
        raw = distal - buffer if direction == "bullish" else distal + buffer
    else:  # fixed_points
        raw = (
            entry_price - config.stop_fixed_points
            if direction == "bullish"
            else entry_price + config.stop_fixed_points
        )
    return _snap_to_tick(raw, tick_size)


def _target_price(
    direction: str,
    entry_price: float,
    stop_price: float,
    config: StrategyConfig,
    tick_size: float,
    levels: pd.DataFrame,
    sweeps: pd.DataFrame,
    entry_at: pd.Timestamp,
) -> tuple[float | None, int | None]:
    """Returns (target_price, target_time_bars). Exactly one is non-None:
    a price target, or a bar-count for a time-based exit."""
    stop_distance = abs(entry_price - stop_price)

    if config.target_type == "fixed_r":
        sign = 1 if direction == "bullish" else -1
        raw = entry_price + sign * config.target_r_multiple * stop_distance
        return _snap_to_tick(raw, tick_size), None

    if config.target_type == "time":
        return None, config.target_time_bars

    # next_liquidity: nearest still-active opposing-type level beyond entry.
    opposing_suffix = "_high" if direction == "bullish" else "_low"
    candidates = levels[levels["level_type"].str.endswith(opposing_suffix) & (levels["knowable_at"] <= entry_at)]
    if not sweeps.empty:
        already_swept = sweeps.loc[sweeps["confirmed_at"] <= entry_at, ["level_type", "level_price"]]
        if not already_swept.empty:
            swept_keys = set(zip(already_swept["level_type"], already_swept["level_price"]))
            candidates = candidates[
                ~candidates.apply(lambda r: (r["level_type"], r["price"]) in swept_keys, axis=1)
            ]
    if direction == "bullish":
        candidates = candidates[candidates["price"] > entry_price]
        chosen = candidates["price"].min() if not candidates.empty else None
    else:
        candidates = candidates[candidates["price"] < entry_price]
        chosen = candidates["price"].max() if not candidates.empty else None

    if chosen is None:
        sign = 1 if direction == "bullish" else -1
        raw = entry_price + sign * config.target_fallback_r_multiple * stop_distance
        return _snap_to_tick(raw, tick_size), None
    return _snap_to_tick(chosen, tick_size), None


def _simulate_exit(
    direction: str,
    entry_at: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    target_price: float | None,
    target_time_bars: int | None,
    hard_exit_at: pd.Timestamp,
    session_bars: pd.DataFrame,
    tick_size: float,
    stop_slippage_ticks: float,
) -> dict:
    scan = session_bars[(session_bars.index > entry_at) & (session_bars.index <= hard_exit_at)]
    if scan.empty:
        return {"exit_at": entry_at, "exit_price": entry_price, "exit_reason": "hard_exit", "ambiguous_bar": False}

    if direction == "bullish":
        stop_hit = (scan["low"] <= stop_price).to_numpy()
        target_hit = (scan["high"] >= target_price).to_numpy() if target_price is not None else None
    else:
        stop_hit = (scan["high"] >= stop_price).to_numpy()
        target_hit = (scan["low"] <= target_price).to_numpy() if target_price is not None else None

    candidates = []
    stop_pos = stop_hit.nonzero()[0]
    if len(stop_pos):
        candidates.append((stop_pos[0], "stop"))
    if target_hit is not None:
        target_pos = target_hit.nonzero()[0]
        if len(target_pos):
            candidates.append((target_pos[0], "target"))
    if target_time_bars:
        time_pos = target_time_bars - 1
        if time_pos < len(scan):
            candidates.append((time_pos, "target_time"))

    if not candidates:
        return {
            "exit_at": scan.index[-1],
            "exit_price": scan["close"].iloc[-1],
            "exit_reason": "hard_exit",
            "ambiguous_bar": False,
        }

    min_pos = min(pos for pos, _ in candidates)
    reasons = {reason for pos, reason in candidates if pos == min_pos}
    # "If a bar's high and low both cross the stop and the target, the STOP
    # fills first. Always." -- extended the same way against a coincident
    # time-exit: a real price-based fill always outranks a timer.
    if "stop" in reasons:
        exit_reason = "stop"
    elif "target" in reasons:
        exit_reason = "target"
    else:
        exit_reason = "target_time"
    ambiguous_bar = "stop" in reasons and "target" in reasons

    exit_at = scan.index[min_pos]
    if exit_reason == "stop":
        slippage = stop_slippage_ticks * tick_size
        raw = stop_price - slippage if direction == "bullish" else stop_price + slippage
        exit_price = _snap_to_tick(raw, tick_size)
    elif exit_reason == "target":
        exit_price = target_price
    else:
        exit_price = scan["close"].iloc[min_pos]

    return {"exit_at": exit_at, "exit_price": exit_price, "exit_reason": exit_reason, "ambiguous_bar": ambiguous_bar}


def _mae_mfe(
    direction: str, entry_price: float, entry_at: pd.Timestamp, exit_at: pd.Timestamp, session_bars: pd.DataFrame
) -> tuple[float, float]:
    scan = session_bars[(session_bars.index >= entry_at) & (session_bars.index <= exit_at)]
    if direction == "bullish":
        mae = entry_price - scan["low"].min()
        mfe = scan["high"].max() - entry_price
    else:
        mae = scan["high"].max() - entry_price
        mfe = entry_price - scan["low"].min()
    return max(mae, 0.0), max(mfe, 0.0)


def simulate_trades(
    signals: pd.DataFrame,
    no_signals: pd.DataFrame,
    config: StrategyConfig,
    symbol: str,
    df_1m: pd.DataFrame,
    levels: pd.DataFrame,
    sweeps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost_model = COST_MODELS[symbol]
    tick_size, tick_value = cost_model.tick_size, cost_model.tick_value

    with_sessions = add_session_columns(df_1m)
    session_groups = {date: group for date, group in with_sessions.groupby("session_date")}

    trades = []
    no_trades = [
        {"session_date": r["session_date"], "window": r["window"], "reason": r["reason"]}
        for _, r in no_signals.iterrows()
    ]

    for _, sig in signals.iterrows():
        session_bars = session_groups[sig["session_date"]]
        window_bars = session_bars[session_bars[sig["window"]]]
        window_end = window_bars.index.max()

        if config.hard_exit == "window_end":
            hard_exit_at = window_end
        else:
            rth_bars = session_bars[session_bars["rth"]]
            hard_exit_at = rth_bars.index.max() if not rth_bars.empty else window_end

        direction = sig["direction"]
        entry_price = _entry_price(direction, config.entry_level, sig["fvg_top"], sig["fvg_bottom"], tick_size)
        entry_at = _simulate_entry(direction, entry_price, sig["setup_at"], window_end, session_bars)
        if entry_at is None:
            no_trades.append(
                {"session_date": sig["session_date"], "window": sig["window"], "reason": "limit_unfilled"}
            )
            continue

        stop_price = _stop_price(
            direction, entry_price, config, tick_size, sig["fvg_top"], sig["fvg_bottom"], sig["sweep_level_price"]
        )
        target_price, target_time_bars = _target_price(
            direction, entry_price, stop_price, config, tick_size, levels, sweeps, entry_at
        )
        exit_result = _simulate_exit(
            direction,
            entry_at,
            entry_price,
            stop_price,
            target_price,
            target_time_bars,
            hard_exit_at,
            session_bars,
            tick_size,
            cost_model.stop_slippage_ticks,
        )
        mae_points, mfe_points = _mae_mfe(
            direction, entry_price, entry_at, exit_result["exit_at"], session_bars
        )

        price_diff = (
            exit_result["exit_price"] - entry_price
            if direction == "bullish"
            else entry_price - exit_result["exit_price"]
        )
        ticks = price_diff / tick_size
        gross_pnl = ticks * tick_value
        net_pnl = gross_pnl - cost_model.commission_round_turn
        stop_distance = abs(entry_price - stop_price)
        r_multiple = price_diff / stop_distance if stop_distance > 0 else float("nan")

        entry_pos = session_bars.index.get_indexer([entry_at])[0]
        exit_pos = session_bars.index.get_indexer([exit_result["exit_at"]])[0]
        bars_held = exit_pos - entry_pos + 1
        is_roll_day = bool(session_bars.loc[entry_at, "is_roll_day"]) if "is_roll_day" in session_bars.columns else False

        trades.append(
            {
                "session_date": sig["session_date"],
                "symbol": symbol,
                "window": sig["window"],
                "direction": direction,
                "setup_at": sig["setup_at"],
                "entry_at": entry_at,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "exit_at": exit_result["exit_at"],
                "exit_price": exit_result["exit_price"],
                "exit_reason": exit_result["exit_reason"],
                "bars_held": bars_held,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "r_multiple": r_multiple,
                "mae_points": mae_points,
                "mfe_points": mfe_points,
                "ambiguous_bar": exit_result["ambiguous_bar"],
                "is_roll_day": is_roll_day,
                "config": config.to_dict(),
            }
        )

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS) if trades else pd.DataFrame(columns=TRADE_COLUMNS)
    no_trades_df = (
        pd.DataFrame(no_trades, columns=NO_TRADE_COLUMNS) if no_trades else pd.DataFrame(columns=NO_TRADE_COLUMNS)
    )
    return trades_df, no_trades_df
