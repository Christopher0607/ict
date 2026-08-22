"""Export the usable window as a durable, committable dataset.

This exists because the downloaded cache is the only copy of data that cost
real money, and the Databento account can no longer afford to fetch it again.
The raw cache is ~340MB across two directories and is gitignored; trimmed to
the usable window and stored as zstd parquet it is about 58MB per symbol,
which is small enough to live in the repository.

Only ``MIN_USABLE_DATE`` onward is exported. Everything before it has whole
sessions missing, seasonally, and no downstream consumer should be able to
reach it by accident -- see ``research/data/quality.py`` for the measurement.

The holdout is *included* in the export and enforced at load time instead.
Sealing by absence would mean re-buying data to run the out-of-sample test,
which is exactly the pressure the holdout exists to resist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.data.contracts import exact_gaps_from_daily, find_rolls
from research.data.databento_fetch import load_symbol
from research.data.quality import MIN_USABLE_DATE

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PRICE_COLS = ("open", "high", "low", "close")


def export_symbol(symbol: str, *, out_dir: Path = DATA_DIR) -> dict:
    """Write ``{root}_1m_2016plus.parquet`` and ``{root}_rolls.csv``."""
    root = symbol.split(".")[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_symbol(symbol, "ohlcv-1m")
    ts = pd.to_datetime(df["ts_open"], utc=True)
    df = df.loc[ts >= MIN_USABLE_DATE].reset_index(drop=True)

    # Rolls are computed on the *trimmed* series so the roll table and the bars
    # always agree; a roll before the usable window is not a roll in this file.
    rolls = find_rolls(df)
    gap_source = "boundary_estimate"
    try:
        daily = load_symbol(f"{root}.FUT", "ohlcv-1d")
        rolls = exact_gaps_from_daily(rolls, daily)
        gap_source = "daily_overlap"
    except FileNotFoundError:
        rolls["gap_source"] = gap_source

    out = df[["ts_open", "instrument_id", *PRICE_COLS, "volume"]].copy()
    for c in PRICE_COLS:
        out[c] = out[c].astype("float32")
    out["volume"] = out["volume"].astype("uint32")
    out["instrument_id"] = out["instrument_id"].astype("uint32")

    bars_path = out_dir / f"{root}_1m_2016plus.parquet"
    rolls_path = out_dir / f"{root}_rolls.csv"
    out.to_parquet(bars_path, compression="zstd", index=False)
    rolls.to_csv(rolls_path, index=False)

    out_ts = pd.to_datetime(out["ts_open"], utc=True)
    return {
        "symbol": symbol,
        "rows": len(out),
        "first": out_ts.min(),
        "last": out_ts.max(),
        "rolls": len(rolls),
        "gap_source": gap_source,
        "bars_mb": bars_path.stat().st_size / 1e6,
        "bars_path": bars_path,
        "rolls_path": rolls_path,
    }


def load_exported(root: str, *, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read an exported dataset back. The entry point for everything downstream."""
    path = data_dir / f"{root}_1m_2016plus.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is the only copy of paid data -- see "
            f"data/README.md for what it costs to rebuild."
        )
    df = pd.read_parquet(path)
    df["ts_open"] = pd.to_datetime(df["ts_open"], utc=True)
    return df
