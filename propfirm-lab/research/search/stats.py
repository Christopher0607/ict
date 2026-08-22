"""Multiple-testing corrections for a search of this size.

Running 8,656 configurations and reporting the best one is not research, it is
sampling the maximum of 8,656 noise draws. Two corrections keep that honest.

**Benjamini-Hochberg** controls the false discovery rate across the whole grid:
of the configs declared significant, at most q are expected to be noise.

**The Deflated Sharpe Ratio** (Bailey & Lopez de Prado) asks a sharper
question: given that N strategies were tried, and given the skew and kurtosis
of this one's returns, what is the probability its true Sharpe exceeds zero?
The expected maximum Sharpe under the null grows with N -- with thousands of
trials a Sharpe near 1 is unremarkable -- so this is the number that stops a
large search from manufacturing a winner.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def bh_fdr(pvalues: np.ndarray, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns (rejected, adjusted p-values)."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], bool), np.array([])

    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out_adj = np.empty(n)
    out_adj[order] = adjusted
    return out_adj <= q, out_adj


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent null strategies.

    This is the benchmark a candidate has to beat: with enough trials, an
    impressive-looking Sharpe is simply the top of the noise distribution.
    """
    if n_trials < 2:
        return 0.0
    sd = np.sqrt(max(sharpe_variance, 0.0))
    a = stats.norm.ppf(1 - 1.0 / n_trials)
    b = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sd * ((1 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    *,
    sharpe_variance: float | None = None,
) -> dict:
    """Probability the true Sharpe exceeds the trial-adjusted benchmark.

    ``returns`` is the per-trade R series. ``sharpe_variance`` is the variance
    of Sharpe estimates across the trials that were run; when omitted it is
    approximated from this strategy's own sample, which is the conservative
    fallback the paper allows.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return {"sharpe": 0.0, "sr0": 0.0, "dsr": 0.0, "n": n}

    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    if sd == 0:
        return {"sharpe": 0.0, "sr0": 0.0, "dsr": 0.0, "n": n}

    sharpe = mu / sd
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    if sharpe_variance is None:
        sharpe_variance = (1 + 0.5 * sharpe**2) / n
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)

    denom = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denom <= 0:
        return {"sharpe": sharpe, "sr0": sr0, "dsr": 0.0, "n": n}

    z = (sharpe - sr0) * np.sqrt(n - 1) / np.sqrt(denom)
    return {
        "sharpe": sharpe,
        "sr0": sr0,
        "dsr": float(stats.norm.cdf(z)),
        "skew": skew,
        "kurtosis": kurt,
        "n": n,
    }


def block_bootstrap_ci(
    returns: np.ndarray,
    session_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI for mean R, resampling whole sessions.

    Intraday trades cluster hard within a day -- one trend day produces a run
    of winners that are anything but independent -- so resampling individual
    trades manufactures precision that is not there.
    """
    r = np.asarray(returns, float)
    s = np.asarray(session_ids)
    if r.size < 2:
        return (float("nan"), float("nan"))

    sessions = np.unique(s)
    by_session = [r[s == sess] for sess in sessions]
    rng = np.random.default_rng(seed)

    means = np.empty(n_boot)
    k = len(by_session)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        means[b] = np.concatenate([by_session[i] for i in pick]).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def clustered_se(returns: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Standard error of the mean, robust to clustering within sessions.

    Intraday trades are not independent draws: one trend day produces a run of
    winners together. Treating them as independent inflates every t-statistic
    in the search, and across thousands of trials that manufactures survivors.
    Standard cluster-robust estimator, clustering on the session.
    """
    r = np.asarray(returns, float)
    g = np.asarray(cluster_ids)
    n = r.size
    if n < 2:
        return float("inf")

    resid = r - r.mean()
    order = np.argsort(g, kind="stable")
    gs, rs = g[order], resid[order]
    boundaries = np.flatnonzero(np.concatenate([[True], gs[1:] != gs[:-1], [True]]))
    cluster_sums = np.add.reduceat(rs, boundaries[:-1])
    n_clusters = cluster_sums.size
    if n_clusters < 2:
        return float("inf")

    meat = float(np.sum(cluster_sums ** 2))
    correction = n_clusters / (n_clusters - 1)
    return float(np.sqrt(correction * meat) / n)
