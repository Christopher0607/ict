"""Apply the pre-registered decision gates to the search results.

Deliberately a separate script from the search itself, so the raw results exist
on disk before anything is judged. Every threshold comes from
research/search/registry.py, which was committed before the search ran.

    python run_search_report.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search import registry
from research.search.backtest import simulate_sequential
from research.search.features import build
from research.search.rules import FAMILIES
from research.search.stats import bh_fdr, block_bootstrap_ci, deflated_sharpe

OUT = Path("findings")


def main() -> None:
    res = pd.read_parquet(OUT / "search_results.parquet")
    n_grid = registry.grid_size()["TOTAL"]
    print(f"=== {len(res):,} configurations, grid total {n_grid:,} ===\n")

    # Gate 1-2: enough trades to say anything, and often enough to be tradeable.
    res["gate_trades"] = res["trades"] >= registry.MIN_TRADES
    res["gate_frequency"] = res["trades_per_year"].fillna(0) >= registry.MIN_TRADES_PER_YEAR
    print(f"gate 1  trades >= {registry.MIN_TRADES}: "
          f"{int(res['gate_trades'].sum()):,} pass")
    print(f"gate 2  trades/year >= {registry.MIN_TRADES_PER_YEAR}: "
          f"{int((res['gate_trades'] & res['gate_frequency']).sum()):,} pass")

    # Gate 3: BH-FDR over the WHOLE grid. A config that could not be evaluated
    # still counts as a trial -- it was searched.
    t = res["t_stat"].fillna(0.0).to_numpy()
    df_eff = np.maximum(res["sessions_traded"].fillna(2).to_numpy() - 1, 1)
    pvals = np.where(
        res["gate_trades"] & res["gate_frequency"],
        2 * (1 - stats.t.cdf(np.abs(t), df_eff)),
        1.0,
    )
    res["p_value"] = pvals
    rejected, adj = bh_fdr(pvals, q=registry.FDR_Q)
    res["p_adj"] = adj
    res["gate_fdr"] = rejected
    print(f"gate 3  BH-FDR q={registry.FDR_Q} over all {n_grid:,} trials: "
          f"{int(rejected.sum()):,} pass")

    # Gate 5 (screen): the economic bar. Applied before the expensive
    # per-trade recomputation, since it needs no extra data.
    res["gate_economic"] = res["expectancy_r"].fillna(-1) >= registry.TARGET_EXPECTANCY_R
    print(f"gate 5  expectancy >= {registry.TARGET_EXPECTANCY_R}R: "
          f"{int(res['gate_economic'].sum()):,} pass")

    finalists = res[
        res["gate_trades"] & res["gate_frequency"] & res["gate_fdr"] & res["gate_economic"]
    ].copy()
    print(f"\n=== {len(finalists)} configuration(s) reach the Deflated Sharpe stage ===")

    if len(finalists):
        finalists = _deflate(finalists, n_grid)
        survivors = finalists[finalists["dsr"] > 0.95]
        print(f"=== {len(survivors)} survivor(s) after Deflated Sharpe ===")
        finalists.to_parquet(OUT / "search_finalists.parquet", index=False)
    else:
        survivors = finalists

    _describe(res)

    print("\n" + "=" * 72)
    if len(survivors):
        print("SURVIVORS -- these, and only these, may go to the holdout:")
        cols = ["name", "trades", "trades_per_year", "expectancy_r",
                "expectancy_se_clustered", "p_adj", "dsr", "boot_lo", "boot_hi"]
        print(survivors[cols].to_string(index=False))
    else:
        print("NO SURVIVORS.")
        print("Nothing in the pre-registered grid clears the bar. The holdout")
        print("stays sealed -- there is nothing to validate on it.")
    print("=" * 72)


def _deflate(finalists: pd.DataFrame, n_grid: int) -> pd.DataFrame:
    """Recompute per-trade series for finalists only, then deflate."""
    print("  recomputing per-trade returns for the finalists ...")
    df = holdout.dev_slice(load_exported("NQ"))
    f = build(df)

    by_name = {c.name: c for c in registry.enumerate_configs()}
    rows = []
    for _, row in finalists.iterrows():
        cfg = by_name[row["name"]]
        sig = FAMILIES[cfg.family](f, **cfg.params)
        atr = f.atr[sig.idx]
        res = simulate_sequential(
            f, sig.idx, sig.direction, cfg.stop_atr * atr,
            cfg.stop_atr * atr * cfg.target_r,
        )
        sessions = f.session_id[res.entry_idx]
        d = deflated_sharpe(res.r_multiple, n_grid)
        lo, hi = block_bootstrap_ci(res.r_multiple, sessions, n_boot=2000)
        rows.append({**row.to_dict(), **{f"{k}": v for k, v in d.items()},
                     "boot_lo": lo, "boot_hi": hi})
    return pd.DataFrame(rows)


def _describe(res: pd.DataFrame) -> None:
    ok = res[res["trades"] >= registry.MIN_TRADES]
    if ok.empty:
        return
    print("\n=== expectancy distribution across evaluable configs ===")
    q = ok["expectancy_r"].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    for k, v in q.items():
        print(f"  p{int(k*100):>2d}  {v:+.4f} R")
    print(f"  best {ok['expectancy_r'].max():+.4f} R   "
          f"worst {ok['expectancy_r'].min():+.4f} R")
    print(f"  share with positive expectancy: "
          f"{float((ok['expectancy_r'] > 0).mean()):.1%}")

    print("\n=== by family (median expectancy, after costs) ===")
    fam = ok.groupby("family").agg(
        configs=("expectancy_r", "size"),
        median_r=("expectancy_r", "median"),
        best_r=("expectancy_r", "max"),
        median_trades=("trades", "median"),
    ).sort_values("median_r", ascending=False)
    print(fam.to_string())


if __name__ == "__main__":
    main()
