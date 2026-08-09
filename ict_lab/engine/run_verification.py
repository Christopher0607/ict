"""Phase 3 VERIFICATION item 1: run the consensus config on one month of
NQ, printing every trade with its surrounding bars so entries and exits can
be hand-checked by eye.

    python -m ict_lab.engine.run_verification --start 2015-03-01 --end 2015-04-01

Needs the real purchased NQ data in ict_lab/data/raw/ -- there is nothing
meaningful to hand-verify against synthetic random-walk bars.
"""
from __future__ import annotations

import argparse

import pandas as pd

from ict_lab.configs.strategy_config import COST_MODELS, CONSENSUS_CONFIG, StrategyConfig
from ict_lab.data.loader import load_symbol
from ict_lab.engine.execution import simulate_trades
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.engine.signals import generate_signals


def _as_utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def run(symbol: str, start: str, end: str, config: StrategyConfig = CONSENSUS_CONFIG, context_bars: int = 5) -> None:
    df = load_symbol(symbol, price_series="backadjusted")
    df = df.loc[_as_utc(start) : _as_utc(end)]
    if df.empty:
        raise ValueError(f"No {symbol} data in [{start}, {end}] outside the holdout -- check the range.")

    tick_size = COST_MODELS[symbol].tick_size
    store = FeatureStore(df)
    signals, no_signals = generate_signals(store, config, tick_size=tick_size)
    levels = store.liquidity_levels(config.swing_n)
    sweeps = (
        store.sweeps(config.swing_n, config.sweep_level_types, config.sweep_k, config.sweep_min_penetration_ticks, tick_size)
        if config.sweep_required
        else pd.DataFrame()
    )
    trades, no_trades = simulate_trades(signals, no_signals, config, symbol, df, levels, sweeps)

    print(f"=== {symbol} {start} .. {end}, config={config.name!r} ===")
    print(f"signals: {len(signals)}  no_signals: {len(no_signals)}")
    print(f"trades: {len(trades)}  no_trades: {len(no_trades)}")
    if not no_trades.empty:
        print("\nno-trade reasons:")
        print(no_trades["reason"].value_counts().to_string())

    trade_fields = [
        "setup_at", "entry_at", "entry_price", "stop_price", "target_price",
        "exit_at", "exit_price", "exit_reason", "bars_held", "gross_pnl", "net_pnl",
        "r_multiple", "mae_points", "mfe_points", "ambiguous_bar", "is_roll_day",
    ]
    for i, (_, trade) in enumerate(trades.iterrows(), start=1):
        print(f"\n--- trade {i}/{len(trades)}: {trade['session_date'].date()} {trade['window']} {trade['direction']} ---")
        for col in trade_fields:
            print(f"  {col}: {trade[col]}")

        window_start = trade["setup_at"] - pd.Timedelta(minutes=context_bars)
        window_end = trade["exit_at"] + pd.Timedelta(minutes=context_bars)
        surrounding = df.loc[window_start:window_end, ["open", "high", "low", "close", "volume"]]
        print("  surrounding bars:")
        print(surrounding.to_string())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NQ", choices=["ES", "NQ"])
    parser.add_argument("--start", required=True, help="e.g. 2015-03-01")
    parser.add_argument("--end", required=True, help="e.g. 2015-04-01")
    parser.add_argument("--context-bars", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.symbol, args.start, args.end, context_bars=args.context_bars)
