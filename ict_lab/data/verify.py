"""Manual verification tool (Phase 1 item 8): print raw OHLCV for a date/time
window so it can be eyeballed against a live chart.

Usage:
    python -m ict_lab.data.verify NQ "2024-06-03 09:30" "2024-06-03 10:30"
    python -m ict_lab.data.verify NQ "2024-06-03 09:30" "2024-06-03 10:30" --series backadjusted
"""
from __future__ import annotations

import argparse

import pandas as pd

from ict_lab.data.loader import load_symbol


def show_window(
    symbol: str,
    start: str,
    end: str,
    price_series: str = "unadjusted",
    include_holdout: bool = False,
) -> pd.DataFrame:
    """start/end are interpreted as UTC (matching the raw file timestamps)
    unless they already carry explicit tz info."""
    df = load_symbol(symbol, price_series=price_series, include_holdout=include_holdout)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    window = df.loc[start_ts:end_ts]
    print(f"\n{symbol} {price_series} — {start_ts} to {end_ts} (UTC index)")
    print(window[["open", "high", "low", "close", "volume", "contract"]].to_string())
    return window


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", choices=["ES", "NQ"])
    parser.add_argument("start", help='e.g. "2024-06-03 09:30"')
    parser.add_argument("end", help='e.g. "2024-06-03 10:30"')
    parser.add_argument("--series", choices=["unadjusted", "backadjusted"], default="unadjusted")
    parser.add_argument("--include-holdout", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    show_window(
        args.symbol,
        args.start,
        args.end,
        price_series=args.series,
        include_holdout=args.include_holdout,
    )
