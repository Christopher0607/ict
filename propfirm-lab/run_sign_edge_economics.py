"""Is the directional edge the audit found worth anything after costs?

The audit reports sign accuracy above 50% with very large t-statistics -- up to
+54 in the overnight session -- while out-of-sample R-squared sits at zero.
Both can be true: the model calls direction slightly better than a coin while
predicting magnitude not at all.

Statistical significance is not the question. With 2.3 million observations a
1.9-point accuracy edge is overwhelming evidence of *something*, and still
worth nothing if the something is smaller than the commission. This measures
the edge in dollars per trade and puts it next to what a trade costs.

The number that matters is the mean signed return captured: sum(sign(pred) *
actual) / n. That is what a trader following the model actually collects, and
it does not assume the edge is uniform across move sizes -- if the accuracy
lives in the smallest moves, this reports it as the small number it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.data import holdout
from research.data.export_dataset import load_exported
from research.search.backtest import COMMISSION_RT, POINT_VALUE, TICK_SIZE
from research.search.features import build
from research.search.predictability import (
    build_design,
    forward_return,
)
from research.search.stats import clustered_se

OUT = Path("findings")
CELLS = [("eth", 1), ("all", 1), ("rth", 60), ("rth", 30), ("eth", 5)]


def capture(f, horizon: int, regime: str, min_train_months: int = 24) -> dict:
    """Walk-forward again, but score dollars captured rather than R-squared."""
    X, _ = build_design(f)
    y_pts = forward_return(f.close, f.session_id, horizon)
    atr = np.where(np.isnan(f.atr) | (f.atr <= 0), np.nan, f.atr)
    y_norm = y_pts / atr

    mask = np.isfinite(y_norm) & np.isfinite(X).all(axis=1)
    if regime == "rth":
        mask &= f.is_rth
    elif regime == "eth":
        mask &= ~f.is_rth

    idx = np.flatnonzero(mask)
    Xm, yn, yp = X[idx], y_norm[idx], y_pts[idx]
    sess = f.session_id[idx]
    rth = f.is_rth[idx]
    ts = pd.to_datetime(pd.Series(f.ts), utc=True)
    mm = (ts.dt.year * 12 + ts.dt.month).to_numpy()[idx]

    months = np.unique(mm)
    k = Xm.shape[1]
    S1, S2, Sy, sy, n_seen = np.zeros(k), np.zeros((k, k)), np.zeros(k), 0.0, 0
    preds, act_pts, keep_sess, keep_rth = [], [], [], []

    for i, test_m in enumerate(months):
        if i > 0:
            prev = np.flatnonzero(mm == months[i - 1])
            if prev.size > horizon:
                prev = prev[:-horizon]
            if prev.size:
                Xp = Xm[prev]
                S1 += Xp.sum(0); S2 += Xp.T @ Xp
                Sy += Xp.T @ yn[prev]; sy += float(yn[prev].sum()); n_seen += prev.size
        if i < min_train_months or n_seen < 5_000:
            continue
        test = mm == test_m
        if test.sum() < 200:
            continue
        mu = S1 / n_seen
        sd = np.sqrt(np.maximum(S2.diagonal() / n_seen - mu ** 2, 1e-12))
        ybar = sy / n_seen
        Di = 1.0 / sd
        ZtZ = (S2 - n_seen * np.outer(mu, mu)) * np.outer(Di, Di)
        Zty = (Sy - n_seen * mu * ybar) * Di
        try:
            w = np.linalg.solve(ZtZ + np.eye(k), Zty)
        except np.linalg.LinAlgError:
            continue
        preds.append((Xm[test] - mu) * Di @ w + ybar)
        act_pts.append(yp[test])
        keep_sess.append(sess[test])
        keep_rth.append(rth[test])

    p = np.concatenate(preds)
    a = np.concatenate(act_pts)
    s = np.concatenate(keep_sess)
    is_rth = np.concatenate(keep_rth)

    captured_pts = np.sign(p) * a
    captured_usd = captured_pts * POINT_VALUE
    # One entry and one exit per prediction; slippage by the session it sits in.
    slip_usd = np.where(is_rth, 1.0, 2.0) * TICK_SIZE * POINT_VALUE
    net_usd = captured_usd - COMMISSION_RT - slip_usd

    nz = a != 0
    return {
        "regime": regime,
        "horizon": horizon,
        "n": int(p.size),
        "sign_accuracy": float((np.sign(p[nz]) == np.sign(a[nz])).mean()),
        "gross_usd_per_trade": float(captured_usd.mean()),
        "gross_se": float(clustered_se(captured_usd, s)),
        "commission_usd": COMMISSION_RT,
        "slippage_usd": float(slip_usd.mean()),
        "net_usd_per_trade": float(net_usd.mean()),
        "net_se": float(clustered_se(net_usd, s)),
        "breakeven_gross_needed": float(COMMISSION_RT + slip_usd.mean()),
    }


def main() -> None:
    f = build(holdout.dev_slice(load_exported("NQ")))
    rows = [capture(f, h, r) for r, h in CELLS]
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_json(OUT / "sign_edge_economics.json", orient="records", indent=2)

    print(f"{'regime':>7s} {'h':>3s} {'n':>10s} {'sign':>7s} "
          f"{'gross $':>9s} {'need $':>8s} {'net $':>9s} {'net t':>7s}")
    for r in rows:
        t = r["net_usd_per_trade"] / r["net_se"] if r["net_se"] > 0 else 0.0
        print(f"{r['regime']:>7s} {r['horizon']:>3d} {r['n']:>10,} "
              f"{r['sign_accuracy']:>7.4f} {r['gross_usd_per_trade']:>+9.3f} "
              f"{r['breakeven_gross_needed']:>8.2f} {r['net_usd_per_trade']:>+9.3f} "
              f"{t:>+7.2f}")
    print()
    print("gross $ = mean signed return captured per prediction, before costs")
    print("need $  = commission + slippage that must be cleared to break even")


if __name__ == "__main__":
    main()
