"""Phase 5 Part 5: discretion premium. For as_taught_5m, as_traded, and
the best realistic survivor, from the (real, actual) trade log:
- the result taking every setup (no discretion)
- the result with hindsight-perfect skipping of every losing trade
- the minimum fraction of losing trades a trader must correctly skip IN
  ADVANCE to (a) break even net, (b) reach 1.0 net Sharpe

"Minimum fraction" is computed via the optimal skip order -- worst loss
(most negative net_pnl) first -- since that's what actually gives the
smallest count needed; skipping losers in any other order would require
skipping more of them for the same effect. This is a search, not a closed
form, because net Sharpe depends on the whole daily PnL distribution, not
just a running total: for k = 0..n_losers, the k worst losers are removed
and the result re-evaluated, stopping at the first k where each target is
met. That answers "minimum needed" literally, regardless of whether Sharpe
moves in a perfectly straight line as more losers are removed.
"""
from __future__ import annotations

import pandas as pd

from ict_lab.engine.pipeline import all_session_dates, run_config
from ict_lab.engine.sweep_runner import annualized_sharpe

TARGET_SHARPE = 1.0


def _net_sharpe(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> float:
    if trades.empty:
        return float("nan")
    daily = trades.groupby("session_date")["net_pnl"].sum().reindex(session_dates, fill_value=0.0)
    return annualized_sharpe(daily)


def every_setup_result(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> dict:
    return {
        "trades": len(trades),
        "net_pnl": float(trades["net_pnl"].sum()) if len(trades) else 0.0,
        "net_sharpe": _net_sharpe(trades, session_dates),
    }


def hindsight_perfect_result(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> dict:
    winners = trades[trades["r_multiple"] > 0] if len(trades) else trades
    return {
        "trades": len(winners),
        "net_pnl": float(winners["net_pnl"].sum()) if len(winners) else 0.0,
        "net_sharpe": _net_sharpe(winners, session_dates),
    }


def minimum_skip_fraction(
    trades: pd.DataFrame, session_dates: pd.DatetimeIndex, target_sharpe: float = TARGET_SHARPE
) -> dict:
    """Returns {"breakeven_fraction", "sharpe_fraction", "n_losers"}. A
    fraction is None if the target isn't reached even skipping every
    losing trade."""
    if trades.empty:
        return {"breakeven_fraction": None, "sharpe_fraction": None, "n_losers": 0}

    losers = trades[trades["r_multiple"] <= 0].sort_values("net_pnl", kind="mergesort")
    n_losers = len(losers)
    if n_losers == 0:
        return {"breakeven_fraction": 0.0, "sharpe_fraction": 0.0, "n_losers": 0}

    breakeven_fraction = None
    sharpe_fraction = None
    for k in range(n_losers + 1):
        skip_ids = losers.index[:k]
        remaining = trades.drop(index=skip_ids)
        net_pnl = float(remaining["net_pnl"].sum()) if len(remaining) else 0.0
        sharpe = _net_sharpe(remaining, session_dates)

        if breakeven_fraction is None and net_pnl >= 0:
            breakeven_fraction = k / n_losers
        if sharpe_fraction is None and pd.notna(sharpe) and sharpe >= target_sharpe:
            sharpe_fraction = k / n_losers
        if breakeven_fraction is not None and sharpe_fraction is not None:
            break

    return {"breakeven_fraction": breakeven_fraction, "sharpe_fraction": sharpe_fraction, "n_losers": n_losers}


def discretion_premium_report(config_name: str, trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> dict:
    return {
        "config_name": config_name,
        "every_setup": every_setup_result(trades, session_dates),
        "hindsight_perfect": hindsight_perfect_result(trades, session_dates),
        "minimum_skip": minimum_skip_fraction(trades, session_dates),
    }


def run_discretion_premium(df_1m: pd.DataFrame, symbol: str, reference_configs: dict) -> dict[str, dict]:
    session_dates = all_session_dates(df_1m)
    results = {}
    for name, config in reference_configs.items():
        _, _, trades, _ = run_config(df_1m, config, symbol)
        results[name] = discretion_premium_report(name, trades, session_dates)
    return results


def _fmt_fraction(fraction: float | None) -> str:
    return f"{fraction * 100:.1f}%" if fraction is not None else "not achievable even skipping every loser"


def format_discretion_premium(report: dict) -> list[str]:
    """"Single clear numbers with one-line captions" -- one line per
    reported figure, plain text, safe to paste into discretion_premium.txt."""
    name = report["config_name"]
    every, perfect, skip = report["every_setup"], report["hindsight_perfect"], report["minimum_skip"]
    return [
        f"{name}: every setup -> {every['trades']} trades, net PnL ${every['net_pnl']:.0f}, net Sharpe {every['net_sharpe']:.2f}",
        f"{name}: hindsight-perfect (every loser skipped) -> {perfect['trades']} trades, net PnL ${perfect['net_pnl']:.0f}, net Sharpe {perfect['net_sharpe']:.2f}",
        f"{name}: minimum fraction of losers to skip for breakeven net PnL -> {_fmt_fraction(skip['breakeven_fraction'])}",
        f"{name}: minimum fraction of losers to skip for net Sharpe >= {TARGET_SHARPE} -> {_fmt_fraction(skip['sharpe_fraction'])}",
    ]
