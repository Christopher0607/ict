"""Phase 5 Part 1: the parallel sweep runner. Each worker computes one
config's full-pipeline summary row; workers accumulate rows and flush them
to their own shard parquet file every `flush_every` configs, so a restart
can skip config hashes already present in a completed shard instead of
recomputing them (checkpointing) and so a crash never loses more than one
flush interval of work.

Interpretive choices, since the spec states the row schema and mechanics
but not every formula:
- Sharpe (gross and net): annualized from the DAILY pnl series with
  no-trade days included as zero, per the spec's explicit project-wide
  convention. 252 trading days/year.
- win_rate: fraction of trades with r_multiple > 0. R multiple is this
  project's own PnL-neutral unit, so this sidesteps any gross-vs-net
  ambiguity the spec doesn't resolve.
- profit_factor: sum(gross wins) / abs(sum(gross losses)) -- the
  traditional convention describes the strategy's raw rules, with cost
  overlay reported separately via gross vs net Sharpe/PnL.
- max_drawdown: computed on the trade-exit-ordered cumulative series (R
  and dollars/net_pnl separately), reported as a non-negative magnitude
  (peak-to-trough decline size, not a signed delta).
- trades_per_year divides by the number of distinct years the DATA spans
  (all_session_dates), not just years that happened to have a trade --
  otherwise a config active in only 1 of 5 years would misleadingly look
  as frequent as one active throughout.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.configs.sweep_grid import canonical_hash
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.engine.pipeline import all_session_dates, run_config

# Parquet round-trips tuples as numpy arrays and None as NaN (verified
# empirically, not assumed) -- both would silently corrupt a
# reconstructed StrategyConfig (e.g. `not windows_array` raises on a
# multi-element ndarray). windows/sweep_level_types are JSON-encoded to
# plain strings; sweep_universe's None is written as "" (a real empty
# string round-trips through parquet reliably; None does not).
_TUPLE_FIELDS = ("windows", "sweep_level_types")

TRADING_DAYS_PER_YEAR = 252
PROGRESS_EVERY = 500

_PARAM_FIELDS = [f for f in StrategyConfig.__dataclass_fields__ if f != "name"]

SUMMARY_COLUMNS = (
    ["config_hash", "config_name"]
    + _PARAM_FIELDS
    + [
        "trade_count", "trades_per_year", "pct_days_with_trade", "win_rate", "avg_r", "total_r",
        "gross_pnl", "net_pnl", "gross_sharpe", "net_sharpe", "max_drawdown_r", "max_drawdown_dollars",
        "profit_factor", "pct_ambiguous_bar", "pct_roll_day", "first_trade_date", "last_trade_date",
    ]
)


def annualized_sharpe(daily_pnl: pd.Series) -> float:
    if len(daily_pnl) < 2:
        return float("nan")
    std = daily_pnl.std(ddof=1)
    if not std or np.isnan(std):
        return float("nan")
    return float(daily_pnl.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(ordered_cumulative: pd.Series) -> float:
    """Non-negative magnitude of the worst peak-to-trough decline."""
    if ordered_cumulative.empty:
        return 0.0
    running_max = ordered_cumulative.cummax()
    drawdown = running_max - ordered_cumulative
    return float(drawdown.max())


def stats_from_trades(config: StrategyConfig, trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> dict:
    """The pure statistics half of summarize_config: turns an already-
    computed trade log into one Part 1 summary row. Split out from
    summarize_config so the formulas (Sharpe, drawdown, profit factor, ...)
    can be unit-tested against a hand-crafted trade log directly, without
    needing bars that happen to produce it through the full pipeline.
    Always returns a row, even for trade_count=0 (mostly NaN stats) -- the
    funnel (Part 2), not this step, is what filters on insufficient
    sample."""
    row = {f: getattr(config, f) for f in _PARAM_FIELDS}
    for f in _TUPLE_FIELDS:
        row[f] = json.dumps(list(row[f]))
    row["sweep_universe"] = row["sweep_universe"] or ""
    row["config_hash"] = canonical_hash(config)
    row["config_name"] = config.name

    n = len(trades)
    row["trade_count"] = n
    n_years = max(len(set(session_dates.year)), 1)
    row["trades_per_year"] = n / n_years

    total_days = len(session_dates)
    days_with_trade = trades["session_date"].nunique() if n else 0
    row["pct_days_with_trade"] = (days_with_trade / total_days * 100) if total_days else float("nan")

    if n == 0:
        row.update(
            {
                "win_rate": float("nan"), "avg_r": float("nan"), "total_r": 0.0,
                "gross_pnl": 0.0, "net_pnl": 0.0, "gross_sharpe": float("nan"), "net_sharpe": float("nan"),
                "max_drawdown_r": 0.0, "max_drawdown_dollars": 0.0, "profit_factor": float("nan"),
                "pct_ambiguous_bar": float("nan"), "pct_roll_day": float("nan"),
                "first_trade_date": pd.NaT, "last_trade_date": pd.NaT,
            }
        )
        return row

    ordered = trades.sort_values("exit_at", kind="mergesort")
    row["win_rate"] = float((ordered["r_multiple"] > 0).mean() * 100)
    row["avg_r"] = float(ordered["r_multiple"].mean())
    row["total_r"] = float(ordered["r_multiple"].sum())
    row["gross_pnl"] = float(ordered["gross_pnl"].sum())
    row["net_pnl"] = float(ordered["net_pnl"].sum())

    daily_gross = ordered.groupby("session_date")["gross_pnl"].sum().reindex(session_dates, fill_value=0.0)
    daily_net = ordered.groupby("session_date")["net_pnl"].sum().reindex(session_dates, fill_value=0.0)
    row["gross_sharpe"] = annualized_sharpe(daily_gross)
    row["net_sharpe"] = annualized_sharpe(daily_net)

    row["max_drawdown_r"] = _max_drawdown(ordered["r_multiple"].cumsum())
    row["max_drawdown_dollars"] = _max_drawdown(ordered["net_pnl"].cumsum())

    gains = ordered.loc[ordered["gross_pnl"] > 0, "gross_pnl"].sum()
    losses = -ordered.loc[ordered["gross_pnl"] < 0, "gross_pnl"].sum()
    row["profit_factor"] = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else float("nan"))

    row["pct_ambiguous_bar"] = float(ordered["ambiguous_bar"].mean() * 100)
    row["pct_roll_day"] = float(ordered["is_roll_day"].mean() * 100)
    row["first_trade_date"] = ordered["session_date"].min()
    row["last_trade_date"] = ordered["session_date"].max()
    return row


def _to_native(value):
    """A parquet round-trip hands back numpy scalars (np.int64, np.bool_,
    ...) for numeric/bool columns. They compare equal to the Python
    originals but aren't JSON-serializable, which would silently change
    canonical_hash's output for a row-reconstructed config vs. the config
    that produced it -- normalize back to plain int/float/bool."""
    return value.item() if hasattr(value, "item") else value


def config_from_row(row: pd.Series | dict) -> StrategyConfig:
    """Inverse of the serialization stats_from_trades applies: reconstructs
    a real StrategyConfig from a summary row (freshly built or loaded back
    from a shard parquet file), undoing the JSON-string/"" encoding used to
    survive the parquet round-trip and the numpy-scalar widening it causes."""
    kwargs = {f: _to_native(row[f]) for f in _PARAM_FIELDS}
    for f in _TUPLE_FIELDS:
        kwargs[f] = tuple(json.loads(row[f]))
    kwargs["sweep_universe"] = row["sweep_universe"] or None
    return StrategyConfig(name=row["config_name"], **kwargs)


R_MULTIPLES_COLUMNS = ["config_hash", "r_multiples"]


def _r_multiples_row(config_hash: str, trades: pd.DataFrame) -> dict:
    """A companion row (config_hash -> JSON-encoded R-multiple sequence),
    kept separate from the spec-pure Part 1 summary schema. Part 2's
    Benjamini-Hochberg stage needs every config's trade-level R sequence to
    bootstrap on -- correcting only across a pre-filtered shortlist would
    defeat the purpose of the correction -- so this is captured for free
    during the same pipeline run rather than re-running the sweep later."""
    values = trades["r_multiple"].tolist() if len(trades) else []
    return {"config_hash": config_hash, "r_multiples": json.dumps(values)}


def summarize_config(
    df_1m: pd.DataFrame,
    config: StrategyConfig,
    symbol: str,
    session_dates: pd.DatetimeIndex,
    store: FeatureStore | None = None,
) -> tuple[dict, dict]:
    """Runs the full pipeline for `config` once and returns (summary_row,
    r_multiples_row). `store` lets a sequential caller running a handful of
    configs against the same df_1m (e.g. ES validation) share cached
    detector computations; the parallel sweep never passes one in, since
    each process-pool worker builds its own."""
    _, _, trades, _ = run_config(df_1m, config, symbol, store=store)
    row = stats_from_trades(config, trades, session_dates)
    return row, _r_multiples_row(row["config_hash"], trades)


# ---------- process pool plumbing ----------
# Module-level so it's picklable for ProcessPoolExecutor; populated once per
# worker process by _init_worker, never mutated after that.
_worker_state: dict = {}


def _init_worker(df_1m: pd.DataFrame, symbol: str, session_dates: pd.DatetimeIndex) -> None:
    _worker_state["df_1m"] = df_1m
    _worker_state["symbol"] = symbol
    _worker_state["session_dates"] = session_dates


def _run_one(config: StrategyConfig) -> tuple[dict, dict]:
    return summarize_config(
        _worker_state["df_1m"], config, _worker_state["symbol"], _worker_state["session_dates"]
    )


def _flush_shard(shard_dir: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    path = shard_dir / f"shard_{uuid.uuid4().hex}.parquet"
    df.to_parquet(path, index=False)
    return path


def _flush_r_multiples_shard(shard_dir: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows, columns=R_MULTIPLES_COLUMNS)
    path = shard_dir / f"rmult_{uuid.uuid4().hex}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_shards(shard_dir: Path) -> pd.DataFrame:
    """Merges every summary shard in shard_dir, deduped by config_hash
    (checkpoint restart safety: a config finished in an earlier run's
    shard is never recomputed). This is the authoritative "done" set --
    run_sweep flushes the r_multiples companion shard first and the
    summary shard second, so a crash between the two just makes one config
    look not-yet-done and safely recomputed (and re-deduped) on restart."""
    files = sorted(Path(shard_dir).glob("shard_*.parquet"))
    if not files:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return merged.drop_duplicates(subset="config_hash", keep="last").reset_index(drop=True)


def load_r_multiples_shards(shard_dir: Path) -> pd.DataFrame:
    """Merges every r_multiples companion shard, deduped by config_hash."""
    files = sorted(Path(shard_dir).glob("rmult_*.parquet"))
    if not files:
        return pd.DataFrame(columns=R_MULTIPLES_COLUMNS)
    merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return merged.drop_duplicates(subset="config_hash", keep="last").reset_index(drop=True)


def r_multiples_for_hash(shard_dir: Path, config_hash: str) -> np.ndarray:
    r = load_r_multiples_shards(shard_dir)
    match = r[r["config_hash"] == config_hash]
    if match.empty:
        return np.array([])
    return np.array(json.loads(match.iloc[0]["r_multiples"]))


def run_sweep(
    df_1m: pd.DataFrame,
    configs: list[StrategyConfig],
    symbol: str,
    shard_dir: Path,
    workers: int | None = None,
    flush_every: int = PROGRESS_EVERY,
    progress: bool = True,
) -> pd.DataFrame:
    """Runs `configs` through summarize_config across a process pool.
    Skips any config whose canonical hash is already present in an
    existing shard under shard_dir (checkpointing). Flushes completed rows
    to a new shard file, and prints a progress line with ETA, every
    `flush_every` configs. Returns the merged, deduped summary DataFrame
    across all shards (pre-existing and newly written)."""
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    already_done = load_shards(shard_dir)
    done_hashes = set(already_done["config_hash"]) if not already_done.empty else set()
    pending = [c for c in configs if canonical_hash(c) not in done_hashes]
    if not pending:
        return already_done

    workers = workers or (os.cpu_count() or 1)
    session_dates = all_session_dates(df_1m)
    total = len(pending)
    start = time.perf_counter()
    completed = 0
    buffer: list[dict] = []
    r_buffer: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(df_1m, symbol, session_dates)
    ) as pool:
        futures = {pool.submit(_run_one, c): c for c in pending}
        for future in as_completed(futures):
            row, r_row = future.result()
            buffer.append(row)
            r_buffer.append(r_row)
            completed += 1
            if progress and completed % flush_every == 0:
                elapsed = time.perf_counter() - start
                eta = elapsed / completed * (total - completed)
                print(
                    f"[sweep] {completed}/{total} configs, elapsed={elapsed:.0f}s, ETA={eta:.0f}s",
                    file=sys.stderr,
                )
            if len(buffer) >= flush_every:
                _flush_r_multiples_shard(shard_dir, r_buffer)
                _flush_shard(shard_dir, buffer)
                buffer, r_buffer = [], []

    if buffer:
        _flush_r_multiples_shard(shard_dir, r_buffer)
        _flush_shard(shard_dir, buffer)

    return load_shards(shard_dir)
