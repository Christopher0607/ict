"""Phase 5 Part 4: the ablation ladder. as_taught_5m's parameters
throughout, both symbols, one table, 6 rungs each adding exactly one more
gate on top of the previous rung:
1. FVG entry only: full session, no sweep, no displacement, no bias
2. + sweep required (bsl_ssl_15m universe)
3. + displacement required
4. + window restriction (the three Silver Bullet hours)
5. + 15m bias (= the full as_taught_5m)
6. + perfect bias instead (LOOKAHEAD, labeled)

Two things had to be resolved to make "as_taught_5m parameters throughout"
constructible, since as_taught_5m's real stop_type="swing" needs
sweep_required=True (StrategyConfig's own hard rule -- no swept level to
reference otherwise) and rung 1 has no sweep at all:
- stop_type is "gap_distal" ONLY at rung 1 (the one rung where "swing" is
  literally impossible), then "swing" from rung 2 onward. This isn't
  really a second independent variable sneaking into the comparison: "stop
  beyond the swept level" is inseparable from having a sweep in the first
  place, so switching stop_type the moment a sweep exists is what "as
  taught" naturally means, not a confound. It's also required for the
  spec's own claim to hold: rung 5 must equal the actual as_taught_5m
  config, which uses stop_type="swing" -- verified by a dedicated test.
- rung 1 needs a genuine "no window restriction at all" window, added
  to sessions.WINDOWS as "full_session" (see sessions.py) rather than a
  one-off hack, since this concept is reusable and rungs 1-3 all need it.

Every OTHER as_taught_5m parameter (fvg_timeframe, entry_level, target
type/fallback, sweep_universe, max_trades_per_window) is held fixed across
all 6 rungs, so the table isolates exactly the 4 gates the spec names --
nothing else varies rung to rung.
"""
from __future__ import annotations

import pandas as pd

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.pipeline import all_session_dates, run_config
from ict_lab.engine.sweep_runner import stats_from_trades

_BASE_KWARGS = dict(
    sweep_universe="bsl_ssl_15m",
    fvg_timeframe="5m",
    entry_level="50%",
    target_type="next_liquidity",
    target_fallback_r_multiple=2.0,
    max_trades_per_window=1,
    mss_required=False,
)

_KILLZONES = ("killzone_london", "killzone_ny_am", "killzone_ny_pm")

RUNG_DEFINITIONS: list[tuple[str, dict]] = [
    (
        "1_fvg_entry_only",
        dict(windows=("full_session",), bias_method="none", sweep_required=False, displacement_required=False, stop_type="gap_distal"),
    ),
    (
        "2_plus_sweep_required",
        dict(windows=("full_session",), bias_method="none", sweep_required=True, displacement_required=False, stop_type="swing"),
    ),
    (
        "3_plus_displacement_required",
        dict(windows=("full_session",), bias_method="none", sweep_required=True, displacement_required=True, stop_type="swing"),
    ),
    (
        "4_plus_window_restriction",
        dict(windows=_KILLZONES, bias_method="none", sweep_required=True, displacement_required=True, stop_type="swing"),
    ),
    (
        "5_plus_15m_bias_full_as_taught_5m",
        dict(
            windows=_KILLZONES, bias_method="swing_structure", bias_timeframe="15m", sweep_required=True,
            displacement_required=True, stop_type="swing",
        ),
    ),
    (
        "6_perfect_bias_lookahead",
        dict(windows=_KILLZONES, bias_method="perfect", sweep_required=True, displacement_required=True, stop_type="swing"),
    ),
]

RUNG_COLUMNS = ["rung", "symbol", "trades", "win_rate", "avg_r", "net_sharpe", "max_drawdown_r"]


def build_rung_configs() -> dict[str, StrategyConfig]:
    return {
        name: StrategyConfig(name=f"ablation_{name}", **{**_BASE_KWARGS, **overrides})
        for name, overrides in RUNG_DEFINITIONS
    }


def run_ablation_ladder(df_nq_1m: pd.DataFrame, df_es_1m: pd.DataFrame) -> pd.DataFrame:
    """One row per (rung, symbol): trades, win rate, avg R, net Sharpe, max
    drawdown in R (points -- the project-wide no-price-ratio rule), reusing
    Part 1's exact statistics formulas via stats_from_trades."""
    rungs = build_rung_configs()
    data_by_symbol = {"NQ": df_nq_1m, "ES": df_es_1m}

    rows = []
    for rung_name, config in rungs.items():
        for symbol, df_1m in data_by_symbol.items():
            session_dates = all_session_dates(df_1m)
            _, _, trades, _ = run_config(df_1m, config, symbol)
            stats = stats_from_trades(config, trades, session_dates)
            rows.append(
                {
                    "rung": rung_name,
                    "symbol": symbol,
                    "trades": stats["trade_count"],
                    "win_rate": stats["win_rate"],
                    "avg_r": stats["avg_r"],
                    "net_sharpe": stats["net_sharpe"],
                    "max_drawdown_r": stats["max_drawdown_r"],
                }
            )
    return pd.DataFrame(rows, columns=RUNG_COLUMNS)
