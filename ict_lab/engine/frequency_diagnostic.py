"""Phase 4 item 5: VERIFICATION (frequency and coverage, still blind to
PnL). Every function here reads only session_date/window/direction/
setup_at/entry_at-type columns -- never gross_pnl, net_pnl, r_multiple,
mae_points, or mfe_points -- because "frequency tuning stays blind to
performance" (spec: "COUNTS AND REASONS ONLY"). compute_frequency_diagnostic
drops the PnL columns from `trades` immediately after running the pipeline,
before any metric function sees it, so that guarantee is structural rather
than just a matter of discipline in what gets printed.

c) is the spec's required diagnostic: day/window coverage, trades/year,
reason mix, all three named configs, NQ, full non-holdout span. a) and b)
are ours to design ("design any additional blind, PnL-free coverage/
consistency checks you think are useful here") -- we implemented the spec's
own two parenthetical suggestions directly rather than inventing a third:

  a) raw_setup_counts: total signal COUNT (not just window/day presence) by
     window/year -- how many qualifying setups existed before
     max_trades_per_window and position-sequencing trim them down to actual
     trades. Read next to pct_windows_with_setup, this tells apart "this
     window rarely qualifies at all" from "it qualifies often but the cap/
     sequencing discards most of it."
  b) atr_mult_sensitivity: reruns pct_days_with_trade -- the CHECKPOINT's
     "number that matters" -- at a couple of alternative
     fvg_min_size_atr_mult values, since the named configs currently use
     0.0 (no minimum). Shows how sensitive day coverage would be to adding
     a minimum-gap-size filter, purely as a blind coverage check.

IMPORTANT -- this module is code-complete and tested against synthetic
fixtures, but the spec's actual CHECKPOINT ("entire non-holdout span" of
real NQ data) cannot be satisfied in this environment: ict_lab/data/raw/
has no purchased data yet (see README). A synthetic random walk has no
genuine session structure, liquidity behavior, or FVG frequency to measure,
so running this against synthetic bars produces syntactically valid output
that is NOT evidence about the real CHECKPOINT question. run_and_print
prints a loud disclaimer whenever data_is_real is not explicitly set True,
and the __main__ CLI only sets it True because reaching that line requires
load_symbol to have actually found real data on disk.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from ict_lab.configs.grid import load_named_configs, placeholder_note
from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.data.loader import load_symbol
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.engine.pipeline import all_session_dates, run_config

_NON_PNL_TRADE_COLUMNS = [
    "session_date", "symbol", "window", "direction", "setup_at",
    "entry_at", "exit_at", "exit_reason", "bars_held", "ambiguous_bar", "is_roll_day",
]


def _drop_pnl_columns(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[[c for c in _NON_PNL_TRADE_COLUMNS if c in trades.columns]]


def pct_windows_with_setup(signals: pd.DataFrame, no_signals: pd.DataFrame) -> pd.DataFrame:
    """% of (session_date, window) occurrences with >=1 qualified setup
    (a signal, pre max_trades_per_window/sequencing), by window and year."""
    all_pairs = pd.concat(
        [signals[["session_date", "window"]], no_signals[["session_date", "window"]]], ignore_index=True
    ).drop_duplicates()
    all_pairs["year"] = pd.DatetimeIndex(all_pairs["session_date"]).year

    setup_pairs = signals[["session_date", "window"]].drop_duplicates().copy()
    setup_pairs["year"] = pd.DatetimeIndex(setup_pairs["session_date"]).year

    total = all_pairs.groupby(["window", "year"]).size().rename("windows_total")
    with_setup = setup_pairs.groupby(["window", "year"]).size().rename("windows_with_setup")
    out = pd.concat([total, with_setup], axis=1).fillna(0)
    out["windows_with_setup"] = out["windows_with_setup"].astype(int)
    out["pct_with_setup"] = (out["windows_with_setup"] / out["windows_total"] * 100).round(1)
    return out.reset_index().sort_values(["window", "year"], kind="mergesort").reset_index(drop=True)


def raw_setup_counts(signals: pd.DataFrame) -> pd.DataFrame:
    """Item (a): total signal-row count (every uncapped qualified setup,
    not just window/day presence) by window and year."""
    df = signals.copy()
    df["year"] = pd.DatetimeIndex(df["session_date"]).year
    return (
        df.groupby(["window", "year"]).size().rename("raw_setup_count")
        .reset_index().sort_values(["window", "year"], kind="mergesort").reset_index(drop=True)
    )


def pct_days_with_trade(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """% of trading days (any window) with >=1 filled trade, by year. This
    is the CHECKPOINT's "number that matters"."""
    all_days = pd.DataFrame({"session_date": session_dates})
    all_days["year"] = pd.DatetimeIndex(all_days["session_date"]).year

    traded_days = trades[["session_date"]].drop_duplicates().copy()
    traded_days["year"] = pd.DatetimeIndex(traded_days["session_date"]).year

    total = all_days.groupby("year").size().rename("trading_days")
    with_trade = traded_days.groupby("year").size().rename("days_with_trade")
    out = pd.concat([total, with_trade], axis=1).fillna(0)
    out["days_with_trade"] = out["days_with_trade"].astype(int)
    out["pct_days_with_trade"] = (out["days_with_trade"] / out["trading_days"] * 100).round(1)
    return out.reset_index().sort_values("year", kind="mergesort").reset_index(drop=True)


def trades_per_year(trades: pd.DataFrame) -> pd.DataFrame:
    years = pd.DatetimeIndex(trades["session_date"]).year
    out = trades.groupby(years).size().rename("trades").rename_axis("year")
    return out.reset_index().sort_values("year", kind="mergesort").reset_index(drop=True)


def trades_per_day_distribution(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Histogram of trades-per-day counts (0 included), for as_traded per
    the spec's explicit callout."""
    counts_by_day = trades.groupby("session_date").size()
    full = counts_by_day.reindex(session_dates, fill_value=0)
    out = full.value_counts().rename("days").rename_axis("trades_per_day")
    return out.reset_index().sort_values("trades_per_day", kind="mergesort").reset_index(drop=True)


def reason_mix(no_trades: pd.DataFrame) -> pd.DataFrame:
    """Gate/no-trade reason counts by year and window (covers both
    signals.py's gate reasons and execution.py's limit_unfilled/
    position_open/max_trades_reached)."""
    df = no_trades.copy()
    df["year"] = pd.DatetimeIndex(df["session_date"]).year
    return (
        df.groupby(["year", "window", "reason"]).size().rename("count").reset_index()
        .sort_values(["year", "window", "count"], ascending=[True, True, False], kind="mergesort")
        .reset_index(drop=True)
    )


def atr_mult_sensitivity(
    df_1m: pd.DataFrame, config: StrategyConfig, symbol: str, alt_mults: tuple[float, ...] = (0.25, 0.5),
) -> pd.DataFrame:
    """Item (b): pct_days_with_trade (overall) at config's own
    fvg_min_size_atr_mult plus a couple of alternatives."""
    all_days = all_session_dates(df_1m)
    mults = (config.fvg_min_size_atr_mult,) + tuple(m for m in alt_mults if m != config.fvg_min_size_atr_mult)
    store = FeatureStore(df_1m)  # shared: only fvg_min_size_atr_mult varies across these runs
    rows = []
    for mult in mults:
        variant = replace(config, fvg_min_size_atr_mult=mult)
        _, _, trades, _ = run_config(df_1m, variant, symbol, store=store)
        coverage = pct_days_with_trade(_drop_pnl_columns(trades), all_days)
        overall_pct = (
            round(coverage["days_with_trade"].sum() / coverage["trading_days"].sum() * 100, 1)
            if len(coverage) and coverage["trading_days"].sum() > 0
            else 0.0
        )
        rows.append({"fvg_min_size_atr_mult": mult, "pct_days_with_trade_overall": overall_pct})
    return pd.DataFrame(rows)


def compute_frequency_diagnostic(
    df_1m: pd.DataFrame, config: StrategyConfig, symbol: str, include_day_distribution: bool = False,
) -> dict:
    signals, no_signals, trades, no_trades = run_config(df_1m, config, symbol)
    trades = _drop_pnl_columns(trades)
    all_days = all_session_dates(df_1m)

    result = {
        "config_name": config.name,
        "symbol": symbol,
        "placeholder_note": placeholder_note(config.name),
        "pct_windows_with_setup": pct_windows_with_setup(signals, no_signals),
        "raw_setup_counts": raw_setup_counts(signals),
        "pct_days_with_trade": pct_days_with_trade(trades, all_days),
        "trades_per_year": trades_per_year(trades),
        "reason_mix": reason_mix(no_trades),
    }
    if include_day_distribution:
        result["trades_per_day_distribution"] = trades_per_day_distribution(trades, all_days)
    return result


def _print_diagnostic(diag: dict) -> None:
    print(f"\n{'=' * 70}\n{diag['config_name']} ({diag['symbol']})\n{'=' * 70}")
    if diag["placeholder_note"]:
        print(f"*** {diag['placeholder_note']} ***")

    print("\npct of windows with >=1 qualified setup, by window/year:")
    print(diag["pct_windows_with_setup"].to_string(index=False))

    print("\nraw setup counts (pre window/trade-cap), by window/year:")
    print(diag["raw_setup_counts"].to_string(index=False))

    coverage = diag["pct_days_with_trade"]
    print("\npct of trading days with >=1 trade, by year:")
    print(coverage.to_string(index=False))
    if len(coverage) and coverage["trading_days"].sum() > 0:
        overall = round(coverage["days_with_trade"].sum() / coverage["trading_days"].sum() * 100, 1)
        print(f"  overall (all years): {overall}%")

    print("\ntrades per year:")
    print(diag["trades_per_year"].to_string(index=False))

    if "trades_per_day_distribution" in diag:
        print("\ntrades-per-day distribution:")
        print(diag["trades_per_day_distribution"].to_string(index=False))

    print("\ngate/no-trade reason mix, by year/window:")
    print(diag["reason_mix"].to_string(index=False))


def run_and_print(df_1m: pd.DataFrame, symbol: str = "NQ", data_is_real: bool = False) -> dict[str, dict]:
    if not data_is_real:
        print(
            "*** NOT THE CHECKPOINT DIAGNOSTIC. data_is_real=False.\n"
            "*** The spec requires the entire non-holdout span of REAL NQ data; ict_lab/data/raw/\n"
            "*** has no purchased data yet. A synthetic random walk has no genuine session\n"
            "*** structure, liquidity behavior, or FVG frequency -- this run is a pipeline smoke\n"
            "*** test only. Do not present it as satisfying the CHECKPOINT.\n"
        )

    configs = load_named_configs()
    diagnostics = {
        name: compute_frequency_diagnostic(df_1m, config, symbol, include_day_distribution=(name == "as_traded"))
        for name, config in configs.items()
    }
    for name in ("as_taught_5m", "as_taught_1m", "as_traded"):
        _print_diagnostic(diagnostics[name])

    print(f"\n{'=' * 70}\nitem (b): pct_days_with_trade sensitivity to fvg_min_size_atr_mult (as_taught_5m)\n{'=' * 70}")
    print(atr_mult_sensitivity(df_1m, configs["as_taught_5m"], symbol).to_string(index=False))

    return diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NQ", choices=["ES", "NQ"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = load_symbol(args.symbol, price_series="backadjusted")  # default include_holdout=False
    run_and_print(df, args.symbol, data_is_real=True)
