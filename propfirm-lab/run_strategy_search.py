"""Run the pre-registered strategy search over the development window.

    python run_strategy_search.py --smoke   # a few hundred configs, for timing
    python run_strategy_search.py           # the full pre-registered grid

Writes findings/search_results.parquet. Applying the decision gates is a
separate step (run_search_report.py) so that the raw results exist on disk
before anything is judged against a threshold.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search import registry
from research.search.backtest import simulate_sequential
from research.search.features import build
from research.search.stats import clustered_se
from research.search.rules import FAMILIES

OUT = Path("findings")


def run_config(f, cfg: registry.Config) -> dict:
    sig = FAMILIES[cfg.family](f, **cfg.params)
    if sig.idx.size == 0:
        return {"name": cfg.name, "family": cfg.family, "trades": 0}

    atr = f.atr[sig.idx]
    stop_points = cfg.stop_atr * atr
    target_points = stop_points * cfg.target_r

    res = simulate_sequential(
        f, sig.idx, sig.direction, stop_points, target_points
    )
    if len(res) == 0:
        return {"name": cfg.name, "family": cfg.family, "trades": 0}

    sessions = f.session_id[res.entry_idx]
    cl_se = clustered_se(res.r_multiple, sessions)
    n_years = (f.ts[-1] - f.ts[0]) / np.timedelta64(365, "D")

    return {
        "name": cfg.name,
        "family": cfg.family,
        "stop_atr": cfg.stop_atr,
        "target_r": cfg.target_r,
        **{k: v for k, v in cfg.params.items()},
        "trades": len(res),
        "trades_per_year": len(res) / max(n_years, 1e-9),
        "expectancy_r": res.expectancy_r,
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
    args = ap.parse_args()

    print("loading development window (holdout stays sealed) ...")
    df = holdout.dev_slice(load_exported(args.symbol))
    print(f"  {len(df):,} bars  {df['ts_open'].min()} -> {df['ts_open'].max()}")

    t0 = time.time()
    f = build(df)
    print(f"  features built in {time.time() - t0:.1f}s")

    configs = registry.enumerate_configs()
    if args.smoke:
        configs = configs[::40]
    print(f"\nrunning {len(configs):,} configurations "
          f"(grid total {registry.grid_size()['TOTAL']:,})")

    rows, t0 = [], time.time()
    for i, cfg in enumerate(configs, 1):
        rows.append(run_config(f, cfg))
        if i % 250 == 0 or i == len(configs):
            rate = i / (time.time() - t0)
            eta = (len(configs) - i) / rate
            print(f"  {i:,}/{len(configs):,}  {rate:.1f} cfg/s  eta {eta/60:.1f} min")

    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    path = OUT / ("search_results_smoke.parquet" if args.smoke else "search_results.parquet")
    out.to_parquet(path, index=False)
    print(f"\nwrote {path}  ({len(out):,} rows, {time.time() - t0:.0f}s)")
    print(f"  configs with >= {registry.MIN_TRADES} trades: "
          f"{int((out['trades'] >= registry.MIN_TRADES).sum()):,}")


if __name__ == "__main__":
    main()
