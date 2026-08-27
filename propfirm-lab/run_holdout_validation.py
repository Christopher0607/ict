"""The exam paper. Run once, on strategies chosen before it was opened.

Sealed since Phase B: NQ from 2024-08-22 onward, and all of ES. This script
opens the NQ window. It is deliberately separate from the search so that
unsealing is an act someone has to perform on purpose.

What is being tested, and why each one is here:

  S  orb[entry_from=300,entry_to=390,or_minutes=30,side=long]stop=8.0R=2.0t=60
     Best profit-to-drawdown of the 17,820 searched. +0.0644 R and 54.4% win
     rate on the development window. Selected AFTER results existed, on a
     ranking chosen after results existed, and it never passed the
     pre-registered economic gate of +0.185 R.

  A  orb[entry_from=0,entry_to=390,or_minutes=60,side=long]stop=4.0R=3.0
     Highest t-statistic of the pre-registered families, +0.1159 R, t=+4.01.

  M  model_confidence[horizon=120,quantile=0.8,regime=rth,side=long]stop=1.5R=99.0t=120
     The only configuration to clear +0.185 R with enough trades. Deflated
     Sharpe rejected it: Sharpe 0.0650 against a noise benchmark of 0.0805.

Three tests, so read the t-statistics against three, not one. None of these
survived the pre-registered decision rule -- zero of 17,820 did -- so this run
cannot confirm a survivor. It can only show what these look like on data none
of them were fitted to, which is the question actually being asked.

NOTHING may be re-tuned after reading this output. That is the entire point of
having sealed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search import registry, model_signal  # noqa: F401
from research.search.backtest import simulate_sequential, POINT_VALUE
from research.search.features import build
from research.search.rules import FAMILIES
from research.search.stats import clustered_se, block_bootstrap_ci

RISK = 150.0
CANDIDATES = {
    "S  ORB 收盘时段 or=30 stop=8.0 target=2R 时间出场60":
        "orb[entry_from=300,entry_to=390,or_minutes=30,side=long]stop=8.0R=2.0t=60",
    "A  ORB 全天 or=60 stop=4.0 target=3R":
        "orb[entry_from=0,entry_to=390,or_minutes=60,side=long]stop=4.0R=3.0",
    "M  模型置信 h=120 q=0.8 stop=1.5 时间出场120":
        "model_confidence[horizon=120,quantile=0.8,regime=rth,side=long]stop=1.5R=99.0t=120",
}
DEV = {"S": 0.0644, "A": 0.1159, "M": 0.1973}


def main() -> None:
    holdout.unseal(
        "Forward test of three pre-selected NQ configurations at the account "
        "owner's explicit direction. Zero configurations survived the "
        "pre-registered decision rule, so this cannot confirm a survivor; it "
        "reports how S, A and M behave on data none of them were fitted to. "
        "No parameter may be changed after reading the result."
    )

    df = load_exported("NQ")   # the seal is on dev_slice; unseal() above is the audit record
    f = build(df)
    ts = pd.to_datetime(pd.Series(f.ts), utc=True)
    start = pd.Timestamp(holdout.HOLDOUT_START, tz="UTC")
    in_holdout = (ts >= start).to_numpy()
    print(f"\n开发集: {ts.min()} -> {start}")
    print(f"holdout: {start} -> {ts.max()}   ({in_holdout.sum():,} 根 1 分钟 K 线)\n")

    by = {c.name: c for c in registry.enumerate_configs()}
    month = ts.dt.to_period("M")

    for label, name in CANDIDATES.items():
        cfg = by[name]
        sig = FAMILIES[cfg.family](f, **cfg.params)
        # Features are built on the full series so ATR and opening ranges are
        # warm; only the signals inside the sealed window are traded.
        keep = in_holdout[sig.idx]
        idx, direction = sig.idx[keep], sig.direction[keep]
        if idx.size == 0:
            print(f"{label}: holdout 内无信号")
            continue

        atr = f.atr[idx]
        r = simulate_sequential(f, idx, direction, cfg.stop_atr * atr,
                                cfg.stop_atr * atr * cfg.target_r,
                                time_exit_bars=cfg.time_exit_bars)
        R = r.r_multiple
        sess = f.session_id[r.entry_idx]
        se = clustered_se(R, sess)
        lo, hi = block_bootstrap_ci(R, sess, n_boot=2000)
        dev = DEV[label[0]]

        print("=" * 96)
        print(label)
        print("=" * 96)
        print(f"  开发集期望 {dev:+.4f}R   ->   holdout 期望 {R.mean():+.4f}R"
              f"   (差 {R.mean()-dev:+.4f}R)")
        print(f"  {len(R):,} 笔   胜率 {np.mean(R>0):.1%}   "
              f"按交易日聚类 SE {se:.4f}   t={R.mean()/se if se>0 else 0:+.2f}")
        print(f"  自助 95% CI [{lo:+.4f}, {hi:+.4f}]R"
              f"   -- {'包含' if lo <= dev <= hi else '不包含'}开发集的 {dev:+.4f}R"
              f"，{'包含' if lo <= 0 <= hi else '不包含'} 0")
        print(f"  每笔风险 ${RISK:.0f} 时:总盈亏 ${R.sum()*RISK:+,.0f}   "
              f"每笔 ${R.mean()*RISK:+,.2f}")
        eq = np.cumsum(R*RISK); dd = (np.maximum.accumulate(eq)-eq).max()
        print(f"  最大回撤 ${dd:,.0f}   (Lucid 上限 $2,000)")

        m = month.to_numpy()[r.entry_idx]
        g = pd.DataFrame({"m": m, "R": R}).groupby("m").agg(
            笔数=("R", "size"), 胜率=("R", lambda s: float((s > 0).mean())),
            期望R=("R", "mean"), 净额=("R", lambda s: float(s.sum()*RISK)))
        print(f"\n  {'月份':<10}{'笔数':>6}{'胜率':>9}{'期望R':>10}{'净额':>12}{'累计':>12}")
        cum = 0.0
        for mm, row in g.iterrows():
            cum += row.净额
            print(f"  {str(mm):<10}{int(row.笔数):>6}{row.胜率:>9.1%}"
                  f"{row.期望R:>+10.4f}{'$'+format(row.净额,'+,.0f'):>12}"
                  f"{'$'+format(cum,'+,.0f'):>12}")
        print(f"  {'盈利月份':<10}{int((g.净额>0).sum())}/{len(g)}\n")


if __name__ == "__main__":
    main()
