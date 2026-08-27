"""Does the confidence-stratified edge survive an honest threshold?

The audit's table -- +$48.75 per trade in the top 19% of |prediction| against
+$10.10 overall -- ranked confidence over the whole sample. For a diagnostic
that is fine. For a strategy it is not: knowing that today's prediction lands
in the top 19% of 2016-2024 requires 2024.

This recomputes the same table two ways. `full_sample` reproduces the audit.
`causal` ranks each test month against earlier test months only, which is what
a live account could actually do. The gap between them is the part of that
+$48.75 that came from hindsight.

Entry is the bar after the signal, not the signal bar's close. The regression
target is close-to-close, and scoring it as written would let the account
transact at a price it only knows once the bar is finished.

Costs are the same as the search: $4.00 round turn plus one tick of slippage
inside RTH, two outside. This is still an upper bound on what a strategy would
earn -- it takes every bar above the threshold with no non-overlap rule and no
stop -- so a number that is negative here is decisively negative.

    python run_confidence_economics.py        # needs run_model_cache.py first
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search import model_signal
from research.search.backtest import COMMISSION_RT, POINT_VALUE, TICK_SIZE
from research.search.features import build
from research.search.predictability import WalkForward
from research.search.stats import clustered_se

OUT = Path("findings")
QUANTILES = (0.0, 0.5, 0.81, 0.91, 0.99)


def _stratum(net_usd: np.ndarray, sess: np.ndarray, pred: np.ndarray,
             actual_pts: np.ndarray, label: str, q: float) -> dict:
    se = clustered_se(net_usd, sess)
    nz = actual_pts != 0
    return {
        "threshold": label,
        "quantile": q,
        "n": int(net_usd.size),
        "sessions": int(np.unique(sess).size),
        "sign_accuracy": float((np.sign(pred[nz]) == np.sign(actual_pts[nz])).mean())
                         if nz.any() else float("nan"),
        "mean_net_usd": float(net_usd.mean()),
        # The mean alone is a lie on a distribution this skewed: a handful of
        # large winners can carry a strategy that loses on most trades.
        "median_net_usd": float(np.median(net_usd)),
        "share_profitable": float((net_usd > 0).mean()),
        "net_se_clustered": float(se),
        "t": float(net_usd.mean() / se) if se > 0 else 0.0,
    }


def analyse(f, horizon: int, regime: str, shuffled: bool = False) -> list[dict]:
    wf = model_signal.predictions(f, horizon, regime,
                                  shuffle_target=shuffled, seed=7)
    if wf.prediction.size == 0:
        return []

    atr = f.atr[wf.bar_idx]
    # The regression target is close[t+h] - close[t], so scoring it directly
    # would have the account transacting at a close it only knows once the bar
    # is over. Entry moves to the next bar's open, the same rule the backtester
    # enforces, and the difference comes straight off the captured move.
    nxt = wf.bar_idx + 1
    tradeable = (nxt < f.close.size) & (f.session_id[np.minimum(nxt, f.close.size - 1)]
                                        == f.session_id[wf.bar_idx])
    wf = WalkForward(
        bar_idx=wf.bar_idx[tradeable], prediction=wf.prediction[tradeable],
        actual=wf.actual[tradeable], session=wf.session[tradeable],
        test_month=wf.test_month[tradeable], in_sample_r2=[], n_features=0,
    )
    atr = atr[tradeable]
    entry_lag = f.open[wf.bar_idx + 1] - f.close[wf.bar_idx]
    actual_pts = wf.actual * atr - entry_lag

    is_rth = f.is_rth[wf.bar_idx]
    captured = np.sign(wf.prediction) * actual_pts * POINT_VALUE
    slip = np.where(is_rth, 1.0, 2.0) * TICK_SIZE * POINT_VALUE
    net = captured - COMMISSION_RT - slip

    conf = np.abs(wf.prediction)
    rows = []
    for q in QUANTILES:
        # As the audit did it: rank against the whole sample.
        keep = conf >= np.quantile(conf, q) if q > 0 else np.ones(conf.size, bool)
        rows.append({"regime": regime, "horizon": horizon,
                     **_stratum(net[keep], wf.session[keep], wf.prediction[keep],
                                actual_pts[keep], "full_sample", q)})

        # As an account could do it: rank against earlier months only.
        if q > 0:
            thr = model_signal.causal_threshold(wf, q)
            keep = np.isfinite(thr) & (conf >= thr)
        else:
            thr = model_signal.causal_threshold(wf, 0.5)
            keep = np.isfinite(thr)      # same month coverage, no confidence filter
        if keep.sum() < 200:
            continue
        rows.append({"regime": regime, "horizon": horizon,
                     **_stratum(net[keep], wf.session[keep], wf.prediction[keep],
                                actual_pts[keep], "causal", q)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffled", action="store_true",
                    help="the control: same bars and features, target sessions permuted")
    args = ap.parse_args()

    print("loading development window (holdout stays sealed) ...")
    df = holdout.dev_slice(load_exported("NQ"))
    f = build(df)

    rows: list[dict] = []
    for horizon in model_signal.HORIZONS:
        for regime in model_signal.REGIMES:
            rows.extend(analyse(f, horizon, regime, shuffled=args.shuffled))

    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    name = "confidence_economics_shuffled.json" if args.shuffled else "confidence_economics.json"
    out.to_json(OUT / name, orient="records", indent=2)

    for (regime, horizon), g in out.groupby(["regime", "horizon"]):
        print(f"\n=== {regime} h={horizon} ===")
        print(f"{'threshold':<12}{'q':>6}{'n':>10}{'acc':>8}"
              f"{'mean $':>10}{'median $':>10}{'P(win)':>9}{'t':>8}")
        for _, r in g.iterrows():
            print(f"{r['threshold']:<12}{r['quantile']:>6.2f}{r['n']:>10,}"
                  f"{r['sign_accuracy']:>8.4f}{r['mean_net_usd']:>10.2f}"
                  f"{r['median_net_usd']:>10.2f}{r['share_profitable']:>9.1%}"
                  f"{r['t']:>+8.2f}")
    print(f"\nwrote {OUT / name}")


if __name__ == "__main__":
    main()
