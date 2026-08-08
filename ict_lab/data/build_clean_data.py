"""Phase 1 entry point. Run this once real files exist in ict_lab/data/raw/:

    python -m ict_lab.data.build_clean_data

For each symbol/series it prints the yearly close range, runs and saves the
data quality report, and writes the partitioned parquet cache to
ict_lab/data/clean/. This includes the holdout years -- Phase 8 needs them
cached too -- but this is the only place in the pipeline that's allowed to
touch them; every downstream feature/signal/backtest call must keep loading
through load_symbol()'s default include_holdout=False.
"""
from __future__ import annotations

from ict_lab.data.cache import write_clean_cache
from ict_lab.data.loader import load_symbol, print_yearly_close_range
from ict_lab.data.quality import print_and_save_report

SYMBOLS = ["ES", "NQ"]
SERIES = ["unadjusted", "backadjusted"]


def main() -> None:
    for symbol in SYMBOLS:
        print_yearly_close_range(symbol)
        for series in SERIES:
            df = load_symbol(symbol, price_series=series, include_holdout=True)
            print_and_save_report(df, symbol, series)
            written = write_clean_cache(df, symbol, series)
            print(f"Wrote {len(written)} partitioned parquet file(s) for {symbol}/{series}.")


if __name__ == "__main__":
    main()
