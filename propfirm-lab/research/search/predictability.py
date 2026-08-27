"""Is there anything to predict at all?

The first search asked "does any of these 10,384 strategies win" and answered
no. This asks the prior question directly: **is there forecastable structure in
the return series** — before costs, before execution, before any strategy?

That is a far more powerful question for the same data. A strategy search pays
commission, slippage, discrete stops and entry timing before it can detect
anything, so it can fail while a real edge exists. A predictability audit pays
none of that. If the out-of-sample R-squared is zero here, no strategy built on
these features can work, and further searching is wasted effort.

It is also nearly free in multiple-testing terms: about twenty regressions,
against the 10,384 trials already spent.

Two things make the answer trustworthy:

**Out-of-sample only.** In-sample R-squared is positive by construction --
enough parameters fit any noise. Fits run on an expanding window and predict
the next month, and only the accumulated out-of-sample predictions are scored.

**Purged.** A target spanning h bars at the end of the training window overlaps
the start of the test window, so the last h bars of each training set are
dropped. Without that the audit leaks the future into itself and reports
predictability that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.search.features import FeatureSet
from research.search.stats import clustered_se


@dataclass
class AuditResult:
    horizon: int
    regime: str
    n_test: int
    oos_r2: float
    sign_accuracy: float
    sign_se: float
    in_sample_r2: float
    n_features: int
    top_features: list[tuple[str, float]] = field(default_factory=list)

    @property
    def sign_edge_t(self) -> float:
        """Standard errors above a coin flip."""
        if self.sign_se <= 0:
            return 0.0
        return (self.sign_accuracy - 0.5) / self.sign_se


def build_design(f: FeatureSet) -> tuple[np.ndarray, list[str]]:
    """Causal features only. Everything here is knowable at its own bar."""
    cols: dict[str, np.ndarray] = {}
    atr = np.where(np.isnan(f.atr) | (f.atr <= 0), np.nan, f.atr)

    # Returns scaled by volatility, so the regression is not dominated by
    # whichever era had the biggest point moves.
    for n, r in f.ret.items():
        cols[f"ret{n}"] = r / atr
    cols["vwap_dev"] = (f.close - f.vwap) / atr
    cols["rel_volume"] = np.clip(f.rel_volume, 0, 10)
    cols["atr_level"] = atr / pd.Series(atr).rolling(1440, min_periods=200).mean().to_numpy()
    rng_ = f.high - f.low
    cols["range_pos"] = np.divide(
        f.close - f.low, rng_, out=np.full(rng_.shape, 0.5), where=rng_ > 0
    )

    ts = pd.to_datetime(pd.Series(f.ts), utc=True)
    cols["dow"] = ts.dt.dayofweek.to_numpy(float)
    cols["dom"] = ts.dt.day.to_numpy(float)
    cols["minute_of_day"] = f.et_minute.astype(float)
    cols["is_rth"] = f.is_rth.astype(float)

    # Overnight gap: the previous session's close against this session's open,
    # constant within a session and known at its first bar.
    cols["gap"] = _session_gap(f) / atr

    names = sorted(cols)
    X = np.column_stack([cols[k] for k in names])
    return X, names


def _session_gap(f: FeatureSet) -> np.ndarray:
    df = pd.DataFrame({"s": f.session_id, "o": f.open, "c": f.close})
    first_open = df.groupby("s")["o"].transform("first").to_numpy()
    last_close = df.groupby("s")["c"].last().shift(1)
    prev_close = pd.Series(f.session_id).map(last_close.to_dict()).to_numpy(float)
    return first_open - prev_close


def forward_return(close: np.ndarray, session_id: np.ndarray, horizon: int) -> np.ndarray:
    """Return over the next ``horizon`` bars, NaN where it would cross a session."""
    n = close.size
    out = np.full(n, np.nan)
    if horizon >= n:
        return out
    out[:-horizon] = close[horizon:] - close[:-horizon]
    same = np.full(n, False)
    same[:-horizon] = session_id[horizon:] == session_id[:-horizon]
    out[~same] = np.nan
    return out


@dataclass
class WalkForward:
    """Raw out-of-sample output of the walk-forward, before it is scored.

    `bar_idx` indexes back into the original bar array, which is what makes
    these predictions usable as a trading signal rather than only as a
    diagnostic. `test_month` is carried so a downstream threshold can be built
    from prior months only.
    """

    bar_idx: np.ndarray
    prediction: np.ndarray
    actual: np.ndarray
    session: np.ndarray
    test_month: np.ndarray
    in_sample_r2: list[float]
    n_features: int
    feature_names: list[str] = field(default_factory=list)


def walk_forward(
    f: FeatureSet,
    horizon: int,
    *,
    regime: str = "all",
    min_train_months: int = 24,
    shuffle_target: bool = False,
    seed: int = 0,
    ridge_alpha: float = 1.0,
) -> WalkForward:
    """Expanding-window monthly walk-forward, purged by the target horizon."""
    X, names = build_design(f)
    y = forward_return(f.close, f.session_id, horizon)
    atr = np.where(np.isnan(f.atr) | (f.atr <= 0), np.nan, f.atr)
    y = y / atr  # volatility-normalized target

    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if regime == "rth":
        mask &= f.is_rth
    elif regime == "eth":
        mask &= ~f.is_rth
    elif regime == "high_vol":
        mask &= atr > np.nanmedian(atr)
    elif regime == "low_vol":
        mask &= atr <= np.nanmedian(atr)

    ts = pd.to_datetime(pd.Series(f.ts), utc=True)
    month = (ts.dt.year * 12 + ts.dt.month).to_numpy()

    idx = np.flatnonzero(mask)
    if idx.size < 10_000:
        return _empty_wf(len(names), names)

    Xm, ym, mm = X[idx], y[idx], month[idx]
    sess = f.session_id[idx]
    if shuffle_target:
        # Shuffle whole sessions, preserving intraday structure but destroying
        # any real feature/target link. A working audit must score ~0 here.
        rng = np.random.default_rng(seed)
        uniq = np.unique(sess)
        perm = dict(zip(uniq, rng.permutation(uniq)))
        order = np.argsort([perm[s] for s in sess], kind="stable")
        ym = ym[order]

    months = np.unique(mm)
    preds, actuals, pred_sess, pred_bar, pred_month = [], [], [], [], []
    in_sample = []

    # Running sufficient statistics for the expanding training window. The
    # standardized normal equations are recoverable from raw sums, so each
    # month costs O(features^2) instead of restandardizing millions of rows:
    #   Z'Z       = D^-1 (X'X - n mu mu') D^-1
    #   Z'(y-ybar) = D^-1 (X'y - n mu ybar)
    k = Xm.shape[1]
    S1 = np.zeros(k)
    S2 = np.zeros((k, k))
    Sy = np.zeros(k)
    sy = 0.0
    sy2 = 0.0
    n_seen = 0

    for i, test_m in enumerate(months):
        if i > 0:
            # Fold in the previous month, purged by the target horizon.
            prev = np.flatnonzero(mm == months[i - 1])
            if prev.size > horizon:
                prev = prev[:-horizon]
            if prev.size:
                Xp, yp = Xm[prev], ym[prev]
                S1 += Xp.sum(0)
                S2 += Xp.T @ Xp
                Sy += Xp.T @ yp
                sy += float(yp.sum())
                sy2 += float(yp @ yp)
                n_seen += prev.size

        if i < min_train_months or n_seen < 5_000:
            continue
        test = mm == test_m
        if test.sum() < 200:
            continue

        mu = S1 / n_seen
        var = np.maximum(S2.diagonal() / n_seen - mu ** 2, 1e-12)
        sd = np.sqrt(var)
        ybar = sy / n_seen

        D_inv = 1.0 / sd
        ZtZ = (S2 - n_seen * np.outer(mu, mu)) * np.outer(D_inv, D_inv)
        Zty = (Sy - n_seen * mu * ybar) * D_inv

        try:
            w = np.linalg.solve(ZtZ + ridge_alpha * np.eye(k), Zty)
        except np.linalg.LinAlgError:
            continue

        # In-sample R^2, also from the sufficient statistics: the explained
        # sum of squares over the total. Reported only to show it is the
        # larger, flattering number that the out-of-sample figure replaces.
        ss_tot = sy2 - n_seen * ybar ** 2
        if ss_tot > 0:
            in_sample.append(float(w @ Zty) / ss_tot)

        Zte = (Xm[test] - mu) * D_inv
        preds.append(Zte @ w + ybar)
        actuals.append(ym[test])
        pred_sess.append(sess[test])
        pred_bar.append(idx[test])
        pred_month.append(np.full(int(test.sum()), test_m))

    if not preds:
        return _empty_wf(len(names), names)

    return WalkForward(
        bar_idx=np.concatenate(pred_bar),
        prediction=np.concatenate(preds),
        actual=np.concatenate(actuals),
        session=np.concatenate(pred_sess),
        test_month=np.concatenate(pred_month),
        in_sample_r2=in_sample,
        n_features=len(names),
        feature_names=names,
    )


def _empty_wf(k: int, names: list[str]) -> WalkForward:
    z = np.array([], float)
    zi = np.array([], np.int64)
    return WalkForward(zi, z, z, zi, zi, [], k, names)


def walk_forward_audit(
    f: FeatureSet,
    horizon: int,
    *,
    regime: str = "all",
    min_train_months: int = 24,
    shuffle_target: bool = False,
    seed: int = 0,
    ridge_alpha: float = 1.0,
) -> AuditResult:
    """Score the walk-forward. Unchanged in behaviour; the loop moved out so
    the predictions themselves can be reused as a trading signal."""
    wf = walk_forward(
        f, horizon, regime=regime, min_train_months=min_train_months,
        shuffle_target=shuffle_target, seed=seed, ridge_alpha=ridge_alpha,
    )
    if wf.prediction.size == 0:
        return AuditResult(horizon, regime, 0, 0.0, 0.5, 0.0, 0.0, wf.n_features)

    p, a, s = wf.prediction, wf.actual, wf.session
    in_sample = wf.in_sample_r2
    names = wf.feature_names

    # Bars whose forward return is exactly zero have no sign to predict, and
    # np.sign(0) == 0 would score every one of them as wrong. At a 1-minute
    # horizon 8.8% of bars are flat, which alone drags accuracy from 51% to
    # 49% and makes a real edge look like an anti-edge.
    nonzero = a != 0
    correct = (np.sign(p[nonzero]) == np.sign(a[nonzero])).astype(float)
    return AuditResult(
        horizon=horizon,
        regime=regime,
        n_test=p.size,
        oos_r2=_r2(a, p),
        sign_accuracy=float(correct.mean()) if correct.size else 0.5,
        sign_se=clustered_se(correct, s[nonzero]) if correct.size else 0.0,
        in_sample_r2=float(np.mean(in_sample)) if in_sample else 0.0,
        n_features=len(names),
    )


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Out-of-sample R-squared against the sample mean. Negative is normal and
    means the model predicts worse than a constant."""
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
