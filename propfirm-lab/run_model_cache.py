"""Precompute the walk-forward predictions the model_confidence family trades on.

One expanding-window pass over 3M bars per (horizon, regime), cached to
findings/cache/. The predictions do not depend on stop, target, side or
quantile, so this pays for all 216 configurations at once -- and the shuffled
controls alongside them, which is the point of building them here rather than
inside the search.

    python run_model_cache.py                # the six real caches
    python run_model_cache.py --shuffled     # the six controls

Each cache is keyed on a hash of the bars it was built from, so it can never be
served to a different slice of data.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search import model_signal
from research.search.features import build
from research.search.stats import clustered_se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NQ")
    ap.add_argument("--shuffled", action="store_true",
                    help="build the label-shuffled controls instead")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print("loading development window (holdout stays sealed) ...")
    df = holdout.dev_slice(load_exported(args.symbol))
    print(f"  {len(df):,} bars  {df['ts_open'].min()} -> {df['ts_open'].max()}")
    f = build(df)
    print(f"  fingerprint {model_signal.fingerprint(f)}")

    for horizon in model_signal.HORIZONS:
        for regime in model_signal.REGIMES:
            t0 = time.time()
            wf = model_signal.predictions(
                f, horizon, regime, symbol=args.symbol,
                shuffle_target=args.shuffled, seed=args.seed,
            )
            live = wf.actual != 0
            if wf.prediction.size == 0 or not live.any():
                print(f"  h={horizon:<4} {regime:<4} no predictions")
                continue
            hit = (np.sign(wf.prediction[live]) == np.sign(wf.actual[live])).astype(float)
            se = clustered_se(hit, wf.session[live])
            tag = "SHUFFLED " if args.shuffled else ""
            print(f"  {tag}h={horizon:<4} {regime:<4} "
                  f"n={wf.prediction.size:>9,}  "
                  f"sign acc {hit.mean():.4f} +/- {se:.4f}  "
                  f"t={(hit.mean() - 0.5) / se if se > 0 else 0:+.2f}  "
                  f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
