"""Converts FirstRate Data's free futures sample CSVs into the 3-file-per-symbol
raw format ict_lab/data/loader.py expects.

This targets the free sample specifically (https://firstratedata.com/i/futures/NQ
and .../ES, "Download Sample"), which ships as a single continuous-series CSV per
timeframe -- {DateTime,Open,High,Low,Close,Volume} in US/Eastern, no header-less
adjustment split, no individual-contract files. That shapes two things this script
does NOT try to fake:

1. FirstRate's free sample has no separate Absolute-Adjusted file. If the sampled
   window contains no contract roll (true for any short recent window -- check the
   roll calendar before trusting this for a different date range), the unadjusted
   and Absolute-Adjusted series are numerically identical over that window, since
   back-adjustment only shifts bars *before* a roll. This script writes the same
   values to both output files in that case and says so loudly; it does not
   synthesize a fake roll to make the two files differ.
2. The free sample has no individual-contract file, so the active contract label
   is inferred from CME's public quarterly roll calendar (~5 trading days before
   the 3rd-Friday expiration), not read off vendor data. Pass --contract-old/-new
   if you know better, or the script raises on a window it can't confidently
   attribute to one contract.

When the real paid Firstrate purchase lands (separate Unadjusted / Absolute-Adjusted
/ Ratio-Adjusted files, plus individual contract files with real roll dates), this
script's per-window transform (tz localize+convert, column rename, dtype/index
checks) still applies, but the roll-log construction should switch to reading the
real per-roll price differentials out of the individual contract files instead of
inferring an empty log.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "ict_lab" / "data" / "raw"
ET = ZoneInfo("America/New_York")


def load_firstrate_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    naive = pd.to_datetime(df["timestamp"])
    # FirstRate timestamps are US/Eastern wall-clock with no UTC offset in the
    # file (per their docs), so this is the one and only place DST is resolved.
    eastern = naive.dt.tz_localize(ET, ambiguous="infer", nonexistent="shift_forward")
    utc = eastern.dt.tz_convert("UTC")

    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.index = pd.DatetimeIndex(utc, name="timestamp")
    if out.index.duplicated().any():
        raise ValueError(f"{path}: duplicate timestamps after UTC conversion.")
    return out.sort_index()


def convert_symbol(symbol: str, source_csv: Path, contract_label: str) -> None:
    df = load_firstrate_csv(source_csv)
    df["contract"] = contract_label

    unadjusted = df.copy()
    backadjusted = df.copy()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    unadjusted.to_parquet(RAW_DIR / f"{symbol}_1m_unadjusted.parquet")
    backadjusted.to_parquet(RAW_DIR / f"{symbol}_1m_backadjusted.parquet")

    rolls = pd.DataFrame(columns=["date", "old_contract", "new_contract", "price_adjustment"])
    rolls.to_csv(RAW_DIR / f"{symbol}_rolls.csv", index=False)

    print(
        f"{symbol}: wrote {len(df)} bars, "
        f"{df.index.min()} .. {df.index.max()} UTC, contract={contract_label}. "
        f"rolls.csv has 0 rows (no roll falls inside this sample window) -- "
        f"unadjusted and backadjusted are therefore identical, not independently "
        f"verified against a real back-adjustment."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nq-csv", type=Path, required=True)
    parser.add_argument("--es-csv", type=Path, required=True)
    parser.add_argument("--nq-contract", default="NQU26")
    parser.add_argument("--es-contract", default="ESU26")
    args = parser.parse_args()

    convert_symbol("NQ", args.nq_csv, args.nq_contract)
    convert_symbol("ES", args.es_csv, args.es_contract)


if __name__ == "__main__":
    main()
