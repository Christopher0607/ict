"""Shared orchestration: FeatureStore -> signals -> trades for one config.
The sweep-universe level-type resolution (needed both for the sweep gate's
own levels/sweeps frame and, per Phase 4 item 3, for next_liquidity target
selection) is done once here so every caller -- the verification script,
the frequency diagnostic, and any future sweep runner -- does it the same
way instead of each keeping its own copy.
"""
from __future__ import annotations

import pandas as pd

from ict_lab.configs.strategy_config import COST_MODELS, StrategyConfig
from ict_lab.configs.sweep_universe import resolve_sweep_universe_for_windows
from ict_lab.engine.execution import simulate_trades
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.engine.signals import generate_signals


def sweep_level_types(config: StrategyConfig) -> tuple[str, ...]:
    """The level types simulate_trades' levels/sweeps arguments need to
    cover: explicit sweep_level_types if set, else the union of
    resolve_sweep_universe(config.sweep_universe, w) across every w in
    config.windows. Empty when sweep_required=False -- nothing to resolve."""
    if not config.sweep_required:
        return ()
    return config.sweep_level_types or resolve_sweep_universe_for_windows(config.sweep_universe, config.windows)


def run_config(
    df_1m: pd.DataFrame, config: StrategyConfig, symbol: str, store: FeatureStore | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (signals, no_signals, trades, no_trades). Pass a shared
    `store` across multiple configs on the same df_1m to reuse cached
    detector computations -- FeatureStore's whole reason for existing."""
    tick_size = COST_MODELS[symbol].tick_size
    store = store if store is not None else FeatureStore(df_1m)
    signals, no_signals = generate_signals(store, config, tick_size=tick_size)

    level_types = sweep_level_types(config)
    levels = store.liquidity_levels(config.swing_n, config.swing_15m_n)
    sweeps = (
        store.sweeps(
            config.swing_n, level_types, config.sweep_k, config.sweep_min_penetration_ticks, tick_size,
            swing_15m_n=config.swing_15m_n,
        )
        if config.sweep_required
        else pd.DataFrame()
    )
    trades, no_trades = simulate_trades(signals, no_signals, config, symbol, df_1m, levels, sweeps)
    return signals, no_signals, trades, no_trades
