"""Trade the model only when it is confident.

The predictability audit found a real directional edge and then showed it was
worthless: sign accuracy 51-53%, t up to +54, out-of-sample R-squared ~0, and
$0.63 per trade against $14 of costs. Averaged over every bar, that is the end
of it.

Sorted by the model's own confidence it stops looking flat:

    RTH, h=60      trades     net $/trade     net t
    all           641,832         +$10.10     +1.30
    top 50%       320,916         +$27.48     +2.58
    top 19%       128,367         +$48.75     +3.50
    top 9%         64,184         +$50.31     +2.96
    top 1%          6,419         +$44.09     +0.95

Rising then flattening is what a real signal does; noise does not sort itself
that cleanly by a number computed before the outcome. h=30 shows the same shape
independently.

**This family was designed after seeing that table.** It is not pre-registered
in the sense the rest of the grid is -- the horizon, the regime and the quantile
were all chosen with the result in view. That is stated here, in findings/05,
and in the run log, because it is the whole reason a development-set pass here
would not be a conclusion. t=3.50 also sits under the 4.008 noise ceiling that
16,320 trials imply.

Two things keep it from being circular anyway:

**The predictions are walk-forward.** Each month is predicted by a model fit
only on earlier months, purged by the target horizon. Nothing here sees its own
future.

**The confidence threshold is also walk-forward.** The audit ranked |prediction|
over the whole sample, which is fine for a diagnostic and not fine for a
strategy: knowing that today's prediction lands in the top 19% of 2016-2024
requires 2024. Here the quantile at each test month is computed from previous
test months only, so the first predicted month is unusable and dropped.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from research.search.features import FeatureSet
from research.search.predictability import WalkForward, walk_forward
from research.search.rules import Signal

CACHE = Path("findings/cache")

# The audit's grid. Predictions are expensive and independent of stop, target
# and side, so one pass per (horizon, regime) feeds every variant built on it.
HORIZONS = (30, 60, 120)
REGIMES = ("rth", "eth")


def _cache_path(symbol: str, horizon: int, regime: str, shuffled: bool,
                fp: str) -> Path:
    """The bar-array fingerprint is part of the filename, not just the contents.

    `bar_idx` is a position, so a cache built on the development window and
    reused against a different slice would aim every signal at the wrong bar and
    look completely normal doing it. Keying the file on the fingerprint makes
    that impossible rather than detectable: a different slice simply misses the
    cache and recomputes.
    """
    tag = "_shuffled" if shuffled else ""
    digest = hashlib.sha256(fp.encode()).hexdigest()[:8]
    return CACHE / f"wf_pred_{symbol}_{regime}_h{horizon}{tag}_{digest}.parquet"


def fingerprint(f: FeatureSet) -> str:
    """Identifies the exact bar array the predictions were computed against.

    The prices are in the hash, not just the time axis. Two different series
    over the same window are a real case -- a re-fetch, a roll-adjustment
    change, a synthetic fixture -- and keying on timestamps alone would hand
    the second one the first one's predictions without a word.
    """
    first, last = pd.Timestamp(f.ts[0]), pd.Timestamp(f.ts[-1])
    h = hashlib.sha256()
    for arr in (f.open, f.high, f.low, f.close):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return f"{f.ts.size}:{first.value}:{last.value}:{h.hexdigest()[:16]}"


def predictions(
    f: FeatureSet,
    horizon: int,
    regime: str,
    *,
    symbol: str = "NQ",
    shuffle_target: bool = False,
    seed: int = 0,
    min_train_months: int = 24,
    cache: bool = True,
) -> WalkForward:
    """Walk-forward out-of-sample predictions, cached to disk.

    One pass over 3M bars per (horizon, regime); the cache is what makes a few
    hundred strategy variants on top of them affordable.
    """
    fp = fingerprint(f)
    path = _cache_path(symbol, horizon, regime, shuffle_target, fp)
    if cache and path.exists():
        d = pd.read_parquet(path)
        if len(d) and d["fingerprint"].iloc[0] != fp:
            raise ValueError(
                f"{path} holds predictions for bars {d['fingerprint'].iloc[0]!r}, "
                f"not {fp!r} -- delete it."
            )
        return WalkForward(
            bar_idx=d["bar_idx"].to_numpy(np.int64),
            prediction=d["prediction"].to_numpy(float),
            actual=d["actual"].to_numpy(float),
            session=d["session"].to_numpy(np.int64),
            test_month=d["test_month"].to_numpy(np.int64),
            in_sample_r2=[],
            n_features=0,
        )

    wf = walk_forward(f, horizon, regime=regime, shuffle_target=shuffle_target,
                      seed=seed, min_train_months=min_train_months)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.part")
        pd.DataFrame({
            "bar_idx": wf.bar_idx, "prediction": wf.prediction,
            "actual": wf.actual, "session": wf.session,
            "test_month": wf.test_month,
            "fingerprint": fp,
        }).to_parquet(tmp, index=False)
        tmp.replace(path)
    return wf


def causal_threshold(wf: WalkForward, quantile: float) -> np.ndarray:
    """Per-bar confidence threshold built from earlier test months only.

    Returns one threshold per prediction, or NaN where there is no history to
    build one from (the first test month). Using the full-sample quantile here
    instead would let a 2024 prediction be judged against a distribution that
    includes 2024 -- a small leak, but the kind that manufactures exactly the
    result this family is trying to check.
    """
    order = np.argsort(wf.test_month, kind="stable")
    months = wf.test_month[order]
    conf = np.abs(wf.prediction[order])

    out = np.full(conf.size, np.nan)
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    bounds = np.r_[starts, conf.size]
    for i in range(1, starts.size):
        history = conf[: bounds[i]]
        out[bounds[i]: bounds[i + 1]] = np.quantile(history, quantile)

    unsorted = np.empty_like(out)
    unsorted[order] = out
    return unsorted


def model_confidence(
    f: FeatureSet,
    *,
    horizon: int,
    regime: str,
    quantile: float,
    side: str,
    symbol: str = "NQ",
    shuffle_target: bool = False,
    min_train_months: int = 24,
    **_,
) -> Signal:
    """Enter in the model's predicted direction when its confidence clears the
    walk-forward quantile."""
    wf = predictions(f, horizon, regime, symbol=symbol,
                     shuffle_target=shuffle_target,
                     min_train_months=min_train_months)
    if wf.prediction.size == 0:
        return Signal(np.array([], np.int64), np.array([], np.int64))

    thr = causal_threshold(wf, quantile)
    fire = np.isfinite(thr) & (np.abs(wf.prediction) >= thr)

    # A prediction is made on the bar it is indexed to and traded on the next
    # one, which backtest.simulate handles. Bars with no usable ATR cannot be
    # sized and are dropped.
    idx = wf.bar_idx[fire]
    atr_ok = np.isfinite(f.atr[idx]) & (f.atr[idx] > 0)
    idx = idx[atr_ok]
    direction = np.sign(wf.prediction[fire][atr_ok]).astype(np.int64)

    keep = direction != 0
    idx, direction = idx[keep], direction[keep]
    if side == "long":
        keep = direction == 1
    elif side == "short":
        keep = direction == -1
    else:
        keep = np.ones(direction.size, bool)

    order = np.argsort(idx[keep], kind="stable")
    return Signal(idx[keep][order], direction[keep][order])


# Registered here rather than in rules.py: this module imports Signal from
# rules, and the dependency has to point one way.
from research.search import rules as _rules  # noqa: E402

_rules.FAMILIES["model_confidence"] = model_confidence
