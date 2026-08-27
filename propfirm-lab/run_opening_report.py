"""Round four: the New York open, priced on the micro contract.

Two things this does that the main report cannot.

**Re-scores every configuration on MNQ, exactly.** Gross expectancy in R is
instrument-independent -- point value appears in both the P&L and the risk and
cancels -- so the only term that moves is commission, and it is measured per
trade rather than estimated. Three broker tiers are carried because commission
varies $0.74-$1.34 round turn, an 80% swing in commission per dollar of risk.

**Recomputes win rate rather than re-scoring it.** Shifting every trade's R down
by the commission gap flips any trade that landed in the band between them, so
win rate cannot be derived from summary statistics. Finalists are re-simulated
on the actual micro spec.

    python run_opening_report.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search import model_signal, registry, shards  # noqa: F401
from research.search.backtest import (
    MNQ, MNQ_CHEAP, MNQ_DEAR, NQ, simulate_sequential,
)
from research.search.features import build
from research.search.rules import FAMILIES
from research.search.stats import bh_fdr, block_bootstrap_ci, clustered_se, deflated_sharpe

SHARDS = "findings/shards/NQ_full"
TIERS = {"NQ": NQ, "MNQ $0.74": MNQ_CHEAP, "MNQ $1.04": MNQ, "MNQ $1.34": MNQ_DEAR}
MIN_WIN_RATE = 0.50


def rescore(d: pd.DataFrame, inst) -> pd.Series:
    """Exact: only commission moves, and it is measured."""
    ratio = (inst.commission_rt / inst.point_value) / (NQ.commission_rt / NQ.point_value)
    return d["gross_expectancy_r"] - d["commission_r"] * ratio


def main() -> None:
    d = shards.load_shards(SHARDS)
    n_grid = registry.grid_size()["TOTAL"]
    ceiling = float(stats.norm.ppf(1 - 1 / (2 * n_grid)))
    opening = d[(d.entry_from == 0) & (d.entry_to.isin([30, 60])) & (d.trades >= 200)]
    print(f"=== 累计试验 {n_grid:,}，噪声天花板 {ceiling:.3f} ===")
    print(f"开盘时段可评估配置: {len(opening):,}"
          f"  (09:30-10:00 {int((opening.entry_to==30).sum()):,}"
          f" / 09:30-10:30 {int((opening.entry_to==60).sum()):,})\n")

    for name, inst in TIERS.items():
        opening = opening.copy()
        opening[f"exp_{name}"] = rescore(opening, inst)

    print("=== 三档手续费下的开盘时段配置 ===")
    print(f"{'合约':<12}{'中位期望':>12}{'正期望':>10}{'最好':>11}{'>=+0.185R':>12}")
    for name in TIERS:
        e = opening[f"exp_{name}"]
        print(f"{name:<12}{e.median():>+12.4f}{float((e>0).mean()):>10.1%}"
              f"{e.max():>+11.4f}{int((e>=0.185).sum()):>12,}")

    print(f"\n=== 09:30-10:00 vs 09:30-10:30 (MNQ $1.04) ===")
    for to, lab in ((30, "09:30-10:00"), (60, "09:30-10:30")):
        w = opening[opening.entry_to == to]
        e = w["exp_MNQ $1.04"]
        print(f"  {lab}  {len(w):>4} 个  中位 {e.median():+.4f}R  "
              f"正期望 {float((e>0).mean()):>5.1%}  最好 {e.max():+.4f}R  "
              f"中位胜率 {w.win_rate.median():.1%}")

    print(f"\n=== 按族 (MNQ $1.04) ===")
    fam = opening.groupby("family").agg(
        配置数=("expectancy_r", "size"),
        中位NQ=("expectancy_r", "median"),
        中位MNQ=("exp_MNQ $1.04", "median"),
        中位毛=("gross_expectancy_r", "median"),
        中位胜率=("win_rate", "median"),
        最好MNQ=("exp_MNQ $1.04", "max"),
    ).sort_values("中位MNQ", ascending=False)
    print(fam.round(4).to_string())

    # The filter the account owner asked for. Applied after the fact; every
    # configuration above still counted as a trial.
    cand = opening[(opening.win_rate > MIN_WIN_RATE)
                   & (opening.trades_per_year >= registry.MIN_TRADES_PER_YEAR)].copy()
    print(f"\n=== 胜率 > {MIN_WIN_RATE:.0%} 且 >=200 笔/年: {len(cand):,} 个 ===")
    if cand.empty:
        print("  没有。")
        return

    cand = cand.sort_values("exp_MNQ $1.04", ascending=False)
    cols = ["name", "trades_per_year", "win_rate", "gross_expectancy_r",
            "expectancy_r", "exp_MNQ $0.74", "exp_MNQ $1.04", "exp_MNQ $1.34"]
    with pd.option_context("display.width", 260, "display.max_colwidth", 62):
        print(cand.head(10)[cols].round(4).to_string(index=False))

    _verify(cand.head(6), n_grid, ceiling)


def _verify(top: pd.DataFrame, n_grid: int, ceiling: float) -> None:
    """Re-simulate on the real micro spec: win rate cannot be re-scored."""
    print(f"\n=== 前 {len(top)} 个在 MNQ $1.04 上重新逐笔模拟 ===")
    f = build(holdout.dev_slice(load_exported("NQ")))
    by = {c.name: c for c in registry.enumerate_configs()}
    for _, row in top.iterrows():
        cfg = by[row["name"]]
        sig = FAMILIES[cfg.family](f, **cfg.params)
        atr = f.atr[sig.idx]
        r = simulate_sequential(
            f, sig.idx, sig.direction, cfg.stop_atr * atr,
            cfg.stop_atr * atr * cfg.target_r,
            time_exit_bars=cfg.time_exit_bars, instrument=MNQ,
        )
        R, sess = r.r_multiple, f.session_id[r.entry_idx]
        se = clustered_se(R, sess)
        dsr = deflated_sharpe(R, n_grid)
        lo, hi = block_bootstrap_ci(R, sess, n_boot=2000)
        wins, losses = R[R > 0], R[R <= 0]
        payoff = wins.mean() / abs(losses.mean()) if len(losses) else float("inf")
        print(f"\n  {row['name']}")
        print(f"    {len(R):,} 笔  胜率 {np.mean(R>0):.2%} (NQ 上 {row.win_rate:.2%})  "
              f"盈亏比 {payoff:.2f}:1")
        print(f"    期望 {R.mean():+.4f}R  t={R.mean()/se if se>0 else 0:+.2f} "
              f"(天花板 {ceiling:.2f})  自助CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"    DSR {dsr['dsr']:.4f} (需 >0.95)   门槛 +0.185R "
              f"-> {'通过' if R.mean() >= 0.185 else '不通过'}")


if __name__ == "__main__":
    main()
