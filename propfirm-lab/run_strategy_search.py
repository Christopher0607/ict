"""Run the pre-registered strategy search over the development window.

    python run_strategy_search.py --smoke   # a few hundred configs, for timing
    python run_strategy_search.py           # the full pre-registered grid

Writes findings/search_results.parquet. Applying the decision gates is a
separate step (run_search_report.py) so that the raw results exist on disk
before anything is judged against a threshold.

Results are checkpointed to findings/shards/ as they are produced, and a
restart picks up where the last one stopped. `--no-resume` starts over, which
means deleting the shard directory first.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search import registry, shards
from research.search.backtest import simulate_sequential
from research.search.features import build
from research.search.stats import clustered_se
from research.search.rules import FAMILIES
# Importing this registers the model_confidence family in FAMILIES.
from research.search import model_signal  # noqa: F401

OUT = Path("findings")


def _empty_row(cfg: registry.Config) -> dict:
    return {
        "name": cfg.name, "family": cfg.family, "stop_atr": cfg.stop_atr,
        "target_r": cfg.target_r, "time_exit_bars": cfg.time_exit_bars,
        **cfg.params, "trades": 0,
    }


def run_config(f, cfg: registry.Config, sig=None) -> dict:
    # Signals depend only on the family parameters. Stop width and target
    # multiple change how a trade is managed, never whether it is taken, so one
    # signal set serves all 16 stop/target variants -- which is where most of
    # the search's cost would otherwise go.
    if sig is None:
        sig = FAMILIES[cfg.family](f, **cfg.params)
    if sig.idx.size == 0:
        return _empty_row(cfg)

    atr = f.atr[sig.idx]
    stop_points = cfg.stop_atr * atr
    target_points = stop_points * cfg.target_r

    res = simulate_sequential(
        f, sig.idx, sig.direction, stop_points, target_points,
        time_exit_bars=cfg.time_exit_bars,
    )
    if len(res) == 0:
        return _empty_row(cfg)

    sessions = f.session_id[res.entry_idx]
    cl_se = clustered_se(res.r_multiple, sessions)
    n_years = (f.ts[-1] - f.ts[0]) / np.timedelta64(365, "D")

    return {
        "name": cfg.name,
        "family": cfg.family,
        "stop_atr": cfg.stop_atr,
        "target_r": cfg.target_r,
        "time_exit_bars": cfg.time_exit_bars,
        **{k: v for k, v in cfg.params.items()},
        "trades": len(res),
        "trades_per_year": len(res) / max(n_years, 1e-9),
        "expectancy_r": res.expectancy_r,
        # Measured from the trades, never estimated from a median ATR.
        "commission_r": res.commission_r,
        "gross_expectancy_r": res.gross_expectancy_r,
        "expectancy_se": res.expectancy_se,
        # Clustered on the session: trades within a day are not independent,
        # and the naive SE inflates every t-stat in the search.
        "expectancy_se_clustered": cl_se,
        "t_stat": res.expectancy_r / cl_se if np.isfinite(cl_se) and cl_se > 0 else 0.0,
        "t_stat_naive": res.expectancy_r / res.expectancy_se if res.expectancy_se > 0 else 0.0,
        "win_rate": res.win_rate,
        "net_pnl": float(res.net_pnl.sum()),
        "hit_stop_rate": float(np.mean(res.hit_stop)),
        "hit_target_rate": float(np.mean(res.hit_target)),
        "ambiguous_rate": float(np.mean(res.ambiguous)),
        "sessions_traded": int(np.unique(sessions).size),
        "r_std": float(np.std(res.r_multiple, ddof=1)) if len(res) > 1 else 0.0,
        "r_skew": float(pd.Series(res.r_multiple).skew()) if len(res) > 2 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="ignore existing shards and recompute everything")
    ap.add_argument("--flush-every", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many configs (for restart testing)")
    args = ap.parse_args()

    configs = registry.enumerate_configs()
    if args.smoke:
        configs = configs[::40]
    tag = "smoke" if args.smoke else "full"
    shard_dir = OUT / "shards" / f"{args.symbol}_{tag}"

    done = shards.done_names(shard_dir) if args.resume else set()
    pending = [c for c in configs if c.name not in done]
    print(f"grid total {registry.grid_size()['TOTAL']:,} | this run {len(configs):,} configs")
    if done:
        print(f"  resuming: {len(done):,} already in shards, {len(pending):,} to go")

    if pending:
        print("loading development window (holdout stays sealed) ...")
        df = holdout.dev_slice(load_exported(args.symbol))
        print(f"  {len(df):,} bars  {df['ts_open'].min()} -> {df['ts_open'].max()}")
        t0 = time.time()
        f = build(df)
        print(f"  features built in {time.time() - t0:.1f}s")

        # Signals depend only on the family parameters, so one signal set serves
        # every stop/target/time-exit variant built on it.
        groups: dict[tuple, list] = defaultdict(list)
        for cfg in pending:
            groups[(cfg.family, tuple(sorted(cfg.params.items())))].append(cfg)
        print(f"  {len(groups):,} distinct signal sets to compute")

        buf: list[dict] = []
        t0, i, total = time.time(), 0, len(pending)
        if args.limit:
            total = min(total, args.limit)
        stop = False
        for (family, _), cfgs in groups.items():
            sig = FAMILIES[family](f, **cfgs[0].params)
            for cfg in cfgs:
                buf.append(run_config(f, cfg, sig=sig))
                i += 1
                if len(buf) >= args.flush_every:
                    shards.flush_shard(shard_dir, buf)
                    buf = []
                    rate = i / (time.time() - t0)
                    print(f"  {i:,}/{total:,}  {rate:.1f} cfg/s  "
                          f"eta {(total - i) / rate / 60:.1f} min", flush=True)
                if args.limit and i >= args.limit:
                    stop = True
                    break
            if stop:
                break
        if buf:
            shards.flush_shard(shard_dir, buf)
        print(f"  computed {i:,} configs in {time.time() - t0:.0f}s")

    out = shards.merge(shard_dir, configs)
    missing = int(out["trades"].isna().sum())
    OUT.mkdir(exist_ok=True)
    path = OUT / ("search_results_smoke.parquet" if args.smoke else "search_results.parquet")
    out.to_parquet(path, index=False)
    print(f"\nwrote {path}  ({len(out):,} rows)")
    if missing:
        print(f"  INCOMPLETE: {missing:,} configs still unrun -- rerun to resume")
    print(f"  configs with >= {registry.MIN_TRADES} trades: "
          f"{int((out['trades'] >= registry.MIN_TRADES).sum()):,}")


if __name__ == "__main__":
    main()
