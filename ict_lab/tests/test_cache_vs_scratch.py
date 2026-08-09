"""Phase 3 verification item 2: "Extend the cache-vs-scratch equality check
end to end: it must now assert identical TRADE logs, not just signal logs,
for 3 random configs on 1 month."

"Cache" here means one FeatureStore instance reused across all 3 configs
(as a real multi-config run would do); "scratch" means a fresh FeatureStore
per config, so nothing is ever shared. If FeatureStore's cache keys were
missing a parameter (or two different parameter combinations collided onto
the same key), an earlier config's cached entry could leak into a later
config's results under "cache" but not "scratch" -- this is exactly the bug
class this check exists to catch.

A 4th config (multi-window, sweep_universe, max_trades_per_window>1) was
added on top of the spec's original 3 to cover Phase 4's new knobs the same
way.
"""
from __future__ import annotations

import pandas as pd

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.engine.pipeline import run_config
from ict_lab.engine.signals import generate_signals

CONFIGS = [
    StrategyConfig(
        name="a",
        windows=("killzone_ny_am",),
        fvg_timeframe="5m",
        swing_n=5,
        sweep_required=True,
        sweep_level_types=("prior_session_high", "prior_session_low"),
        displacement_required=True,
        displacement_atr_mult=1.5,
        bias_method="none",
        stop_type="swing",
        target_type="fixed_r",
        target_r_multiple=2,
    ),
    StrategyConfig(
        name="b",
        windows=("killzone_ny_pm",),
        fvg_timeframe="1m",
        swing_n=3,
        sweep_required=True,
        sweep_level_types=("swing_high", "swing_low"),
        displacement_required=True,
        displacement_atr_mult=2.0,
        bias_method="prior_day",
        stop_type="swing",
        target_type="next_liquidity",
    ),
    StrategyConfig(
        name="c",
        windows=("killzone_london",),
        fvg_timeframe="15m",
        swing_n=7,
        sweep_required=True,
        sweep_level_types=("pre_killzone_london_high", "pre_killzone_london_low"),
        displacement_required=False,
        mss_required=True,
        bias_method="swing_structure",
        stop_type="swing",
        target_type="fixed_r",
        target_r_multiple=1,
    ),
    StrategyConfig(
        name="d",
        windows=("killzone_ny_am", "killzone_ny_pm"),
        fvg_timeframe="5m",
        swing_n=5,
        sweep_required=True,
        sweep_universe="session_refs_plus_swings",
        displacement_required=True,
        displacement_atr_mult=1.5,
        bias_method="none",
        stop_type="swing",
        target_type="fixed_r",
        target_r_multiple=2,
        max_trades_per_window=3,
    ),
]


def _run(store: FeatureStore, config: StrategyConfig):
    _, _, trades, no_trades = run_config(store.df_1m, config, "NQ", store=store)
    return trades, no_trades


def test_cache_vs_scratch_identical_trade_and_no_trade_logs(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-07-03 00:00:00")  # ~1 month

    shared_store = FeatureStore(df)
    for config in CONFIGS:
        cached_trades, cached_no_trades = _run(shared_store, config)
        scratch_trades, scratch_no_trades = _run(FeatureStore(df), config)

        pd.testing.assert_frame_equal(
            cached_trades.drop(columns=["config"]), scratch_trades.drop(columns=["config"])
        )
        assert list(cached_trades["config"]) == list(scratch_trades["config"])
        pd.testing.assert_frame_equal(cached_no_trades, scratch_no_trades)


def test_cache_vs_scratch_signals_identical_across_shared_store(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-07-03 00:00:00")

    shared_store = FeatureStore(df)
    for config in CONFIGS:
        cached_signals, cached_no_signals = generate_signals(shared_store, config, tick_size=0.25)
        scratch_signals, scratch_no_signals = generate_signals(FeatureStore(df), config, tick_size=0.25)

        pd.testing.assert_frame_equal(cached_signals, scratch_signals)
        pd.testing.assert_frame_equal(cached_no_signals, scratch_no_signals)
