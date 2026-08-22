"""Adapter: the exported dataset -> the format ict_lab/data/loader.py expects.

The ICT lab in the parent repository is code-complete through Phase 6 with 212
passing tests and has never run on real data, because every phase from 2 onward
was gated on purchased history landing in ``ict_lab/data/raw/``. This writes
exactly that, from the data now in ``data/``.

Three details the ICT loader is strict about, and one it does not know:

* It requires a tz-aware UTC ``DatetimeIndex`` and fails loudly otherwise
  (``loader.py``), while the exported dataset carries ``ts_open`` as a column.
* It asserts the unadjusted and backadjusted files share an *identical* index,
  so both are written from one frame rather than assembled separately.
* It requires a ``contract`` column, but only ever carries it through -- for
  display and as ``last`` when resampling. The Databento instrument id, as a
  string, satisfies that without paying to resolve real contract codes.

And the one it does not know: **the ICT lab has no idea the vendor's early
history is unusable.** Left alone it would happily consume 2010-2012 data where
RTH coverage is 22-43%. The export is already trimmed at ``MIN_USABLE_DATE``,
and this module re-asserts that rather than trusting it, because the failure
mode is silent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.data.contracts import add_backadjusted
from research.data.export_dataset import load_exported
from research.data.quality import MIN_USABLE_DATE

def _find_ict_raw_dir() -> Path:
    """Locate ict_lab/data/raw by walking up, not by counting directories.

    This package lives inside the ICT repository, but it is also developed as a
    standalone tree, so a fixed parent depth resolves correctly in one layout
    and silently to nonsense in the other.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "ict_lab" / "data" / "raw"
        if candidate.parent.parent.is_dir():
            return candidate
    raise FileNotFoundError(
        "could not locate ict_lab/ above this file; the adapter expects to run "
        "inside (or alongside) the ICT repository"
    )

OHLCV = ("open", "high", "low", "close", "volume")


def write_ict_raw(root: str, *, out_dir: Path | None = None) -> dict:
    """Write ``{root}_1m_{unadjusted,backadjusted}.parquet`` and ``{root}_rolls.csv``."""
    out_dir = _find_ict_raw_dir() if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_exported(root)
    if df["ts_open"].min() < MIN_USABLE_DATE:
        raise ValueError(
            f"{root} export contains bars before {MIN_USABLE_DATE.date()}, where "
            f"whole sessions are missing seasonally. Refusing to hand that to a "
            f"backtest engine that has no way to know."
        )

    rolls_csv = _read_rolls(root)
    adj = add_backadjusted(df, rolls=rolls_csv if not rolls_csv.empty else None)

    index = pd.DatetimeIndex(adj["ts_open"], name="ts_event")
    contract = adj["instrument_id"].astype(str).to_numpy()

    # .to_numpy() is load-bearing. Handing pandas a dict of Series together
    # with a different index makes it ALIGN on the Series' own RangeIndex,
    # which shares no labels with a DatetimeIndex, and every value silently
    # becomes NaN -- with the row count, column names and dtypes all still
    # looking correct.
    unadjusted = pd.DataFrame(
        {c: adj[c].to_numpy() for c in OHLCV} | {"contract": contract},
        index=index,
    )
    backadjusted = pd.DataFrame(
        {c: adj[f"{c}_adj"].to_numpy() for c in OHLCV[:4]}
        | {"volume": adj["volume"].to_numpy(), "contract": contract},
        index=index,
    )

    for name, frame in (("unadjusted", unadjusted), ("backadjusted", backadjusted)):
        na = frame[list(OHLCV)].isna().sum().sum()
        if na:
            raise ValueError(f"{root} {name}: {na:,} NaN values after assembly")

    # The loader compares these two indices for exact equality.
    assert unadjusted.index.equals(backadjusted.index)

    un_path = out_dir / f"{root}_1m_unadjusted.parquet"
    ba_path = out_dir / f"{root}_1m_backadjusted.parquet"
    ro_path = out_dir / f"{root}_rolls.csv"

    unadjusted.to_parquet(un_path, compression="zstd")
    backadjusted.to_parquet(ba_path, compression="zstd")
    _write_rolls(rolls_csv, ro_path)

    return {
        "root": root,
        "rows": len(unadjusted),
        "first": index.min(),
        "last": index.max(),
        "rolls": len(rolls_csv),
        "paths": [un_path, ba_path, ro_path],
    }


def _read_rolls(root: str) -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "data" / f"{root}_rolls.csv"
    if not path.exists():
        return pd.DataFrame()
    r = pd.read_csv(path)
    r["roll_date"] = pd.to_datetime(r["roll_date"]).dt.date
    return r


def _write_rolls(rolls: pd.DataFrame, path: Path) -> None:
    """Emit the ICT roll-log schema: date, old_contract, new_contract, price_adjustment."""
    if rolls.empty:
        pd.DataFrame(
            columns=["date", "old_contract", "new_contract", "price_adjustment"]
        ).to_csv(path, index=False)
        return
    out = pd.DataFrame({
        "date": rolls["roll_date"],
        "old_contract": rolls["old_instrument_id"].astype(str),
        "new_contract": rolls["new_instrument_id"].astype(str),
        "price_adjustment": rolls["gap"],
    })
    out.to_csv(path, index=False)
