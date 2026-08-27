"""Is there forecastable structure in NQ 1-minute returns at all?

    python run_predictability_audit.py

Runs before any strategy is built, because it answers the prior question: a
strategy search pays commission, slippage and entry timing before it can detect
anything, so it can fail while an edge exists. This pays none of that. If the
out-of-sample R-squared is zero here, no strategy on these features can work.

Every row is accompanied by its shuffled-target control. A non-zero score on a
shuffled target would mean the walk-forward reads its own future and nothing
else in the table could be believed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search.features import build
from research.search.predictability import walk_forward_audit

OUT = Path("findings")
HORIZONS = (1, 5, 15, 30, 60)
REGIMES = ("all", "rth", "eth", "high_vol", "low_vol")


def main() -> None:
    print("loading development window (holdout sealed) ...")
    f = build(holdout.dev_slice(load_exported("NQ")))
    print(f"  {len(f):,} bars\n")

    rows = []
    for regime in REGIMES:
        for h in HORIZONS:
            real = walk_forward_audit(f, h, regime=regime)
            ctrl = walk_forward_audit(f, h, regime=regime, shuffle_target=True)
            rows.append({
                "regime": regime, "horizon": h, "n_test": real.n_test,
                "oos_r2": real.oos_r2, "in_sample_r2": real.in_sample_r2,
                "sign_acc": real.sign_accuracy, "sign_t": real.sign_edge_t,
                "shuffled_oos_r2": ctrl.oos_r2,
            })
            print(f"  {regime:9s} h={h:>3d}  n={real.n_test:>9,}  "
                  f"OOS R2={real.oos_r2:>+9.6f}  (in-sample {real.in_sample_r2:>+.6f})  "
                  f"sign={real.sign_accuracy:.4f} t={real.sign_edge_t:>+5.2f}  "
                  f"shuffled R2={ctrl.oos_r2:>+9.6f}")

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_json(OUT / "predictability_audit.json", orient="records", indent=2)

    print("\n" + "=" * 78)
    pos = df[df["oos_r2"] > 0]
    print(f"cells with positive out-of-sample R2: {len(pos)} / {len(df)}")
    if len(pos):
        print(pos[["regime", "horizon", "oos_r2", "sign_acc", "sign_t"]].to_string(index=False))
    print(f"largest OOS R2 anywhere : {df['oos_r2'].max():+.6f}")
    print(f"largest |sign t| anywhere: {df['sign_t'].abs().max():.2f}")
    print(f"shuffled control, max R2 : {df['shuffled_oos_r2'].max():+.6f}  (must be ~0)")
    print("=" * 78)


if __name__ == "__main__":
    main()
