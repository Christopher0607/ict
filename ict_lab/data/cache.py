from __future__ import annotations

from pathlib import Path

import pandas as pd

CLEAN_DIR = Path(__file__).resolve().parents[1] / "data" / "clean"


def write_clean_cache(df: pd.DataFrame, symbol: str, series: str) -> list[Path]:
    """Writes the FULL series (including the holdout years) — this is a
    storage layer, not a strategy computation, and Phase 8 needs the holdout
    years cached too. Callers doing strategy development must still load
    through load_symbol()'s default include_holdout=False.
    """
    written = []
    for year, group in df.groupby(df.index.year):
        out_dir = CLEAN_DIR / f"symbol={symbol}" / f"series={series}" / f"year={year}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "part.parquet"
        group.to_parquet(path)
        written.append(path)
    return written


def read_clean_cache(symbol: str, series: str, years: list[int] | None = None) -> pd.DataFrame:
    base = CLEAN_DIR / f"symbol={symbol}" / f"series={series}"
    if not base.exists():
        raise FileNotFoundError(f"No cached clean data at {base}. Run the cache build first.")

    year_dirs = sorted(base.glob("year=*"))
    if years is not None:
        wanted = {f"year={y}" for y in years}
        year_dirs = [d for d in year_dirs if d.name in wanted]

    frames = [pd.read_parquet(d / "part.parquet") for d in year_dirs]
    if not frames:
        raise FileNotFoundError(f"No cached parquet parts found under {base} for years={years}.")
    return pd.concat(frames).sort_index()
