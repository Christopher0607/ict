"""Checkpointed result storage, so an interrupted search resumes instead of restarting.

The second-round search died at 2,512 of 16,320 configurations with no error in
the log -- the process was reaped at a session boundary -- and thirty minutes of
compute went with it. Nothing about that is unusual for a long sweep; the fix is
to stop treating the run as atomic.

Two properties matter, and both are tested:

* **A completed configuration is never recomputed.** Shards are keyed by config
  name, and a restart skips every name already present in one.
* **The merged result does not depend on how the run was chopped up.** Rows are
  reindexed to the enumeration order and columns to a canonical order, so a run
  killed and resumed four times produces a file identical to one that ran
  straight through. Without this the shard boundaries would leak into the
  output's row order, and "resumable" would quietly mean "almost the same".

Shard files are written to a temporary name and renamed into place. A killed
process then leaves either a complete shard or nothing -- never a truncated
parquet that the resume logic would read as done. (This project has already
been bitten once by non-atomic chunk writes in the data layer; same fix.)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd

# Columns every row carries, in the order they appear in the output. Family
# parameter columns are inserted between the two blocks, sorted by name.
HEAD_COLS = ["name", "family", "stop_atr", "target_r", "time_exit_bars"]
METRIC_COLS = [
    "trades", "trades_per_year", "expectancy_r",
    "commission_r", "gross_expectancy_r",
    "expectancy_se", "expectancy_se_clustered", "t_stat", "t_stat_naive",
    "win_rate", "net_pnl", "hit_stop_rate", "hit_target_rate",
    "ambiguous_rate", "sessions_traded", "r_std", "r_skew",
]


def canonical_columns(configs) -> list[str]:
    """Column order for the merged output, independent of processing order."""
    param_keys = sorted({k for c in configs for k in c.params})
    return HEAD_COLS + param_keys + METRIC_COLS


def _shard_files(shard_dir: Path) -> list[Path]:
    """Completed shards in write order.

    The sequence prefix is what makes that order real. Sorting bare UUIDs would
    order shards at random, so "last write wins" would silently mean "whichever
    filename sorted highest" -- fine while a rerun reproduces the same numbers,
    wrong the moment the scoring code changes underneath an existing shard dir.
    A `.part` file left by a killed writer does not match the glob, so the
    configs it held are simply recomputed.
    """
    return sorted(Path(shard_dir).glob("shard_*.parquet"))


def flush_shard(shard_dir: Path, rows: list[dict]) -> Path:
    """Write one shard atomically: temp name, then rename into place."""
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    seq = len(_shard_files(shard_dir))
    final = shard_dir / f"shard_{seq:06d}_{uuid.uuid4().hex}.parquet"
    part = final.with_suffix(".parquet.part")
    pd.DataFrame(rows).to_parquet(part, index=False)
    os.replace(part, final)
    return final


def load_shards(shard_dir: Path) -> pd.DataFrame:
    """Every shard merged, deduped by config name, last write winning."""
    files = _shard_files(shard_dir)
    if not files:
        return pd.DataFrame(columns=["name"])
    merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return merged.drop_duplicates(subset="name", keep="last").reset_index(drop=True)


def done_names(shard_dir: Path) -> set[str]:
    df = load_shards(shard_dir)
    return set(df["name"]) if len(df) else set()


def merge(shard_dir: Path, configs) -> pd.DataFrame:
    """The finished result: one row per config, in enumeration order.

    Configs still missing from the shards appear as all-NaN rows rather than
    being silently dropped -- an incomplete run should look incomplete.
    """
    df = load_shards(shard_dir)
    order = [c.name for c in configs]
    out = (
        df.set_index("name")
        .reindex(order)
        .reset_index()
        .reindex(columns=canonical_columns(configs))
    )
    return out
