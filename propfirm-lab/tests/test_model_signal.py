"""The model-confidence family, and the leaks it would be easy to build into it.

This family was designed after seeing the audit's confidence table, so it gets
more scrutiny than the pre-registered ones, not less. Two things could
manufacture the result on their own -- a prediction that saw its own future, and
a confidence threshold that saw the whole sample -- and both are tested here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.search.features import build
from research.search.model_signal import (
    causal_threshold,
    fingerprint,
    model_confidence,
    predictions,
)
from research.search.predictability import WalkForward

from tests.test_predictability import _series


def _wf(pred, months, bar_idx=None):
    pred = np.asarray(pred, float)
    months = np.asarray(months, np.int64)
    if bar_idx is None:
        bar_idx = np.arange(pred.size, dtype=np.int64)
    return WalkForward(
        bar_idx=np.asarray(bar_idx, np.int64), prediction=pred,
        actual=np.zeros_like(pred), session=months.copy(),
        test_month=months, in_sample_r2=[], n_features=0,
    )


# ---------------------------------------------------------------------------
# the threshold cannot see forward
# ---------------------------------------------------------------------------


def test_first_month_has_no_threshold_and_trades_nothing():
    """There is no history to rank the first month's predictions against."""
    thr = causal_threshold(_wf([1, 2, 3, 4], [1, 1, 2, 2]), 0.5)
    assert np.isnan(thr[0]) and np.isnan(thr[1])
    assert np.isfinite(thr[2]) and np.isfinite(thr[3])


def test_month_two_is_ranked_against_month_one_only():
    wf = _wf([1.0, 3.0, 100.0, 200.0], [1, 1, 2, 2])
    thr = causal_threshold(wf, 0.5)
    assert thr[2] == pytest.approx(2.0)   # median of {1, 3}, not of all four
    assert thr[3] == pytest.approx(2.0)


def test_a_late_spike_cannot_move_an_earlier_threshold():
    """The tell for full-sample ranking: change the last month, watch the past."""
    base = _wf([1.0, 3.0, 2.0, 4.0, 5.0, 6.0], [1, 1, 2, 2, 3, 3])
    spiked = _wf([1.0, 3.0, 2.0, 4.0, 5_000.0, 6_000.0], [1, 1, 2, 2, 3, 3])
    a, b = causal_threshold(base, 0.9), causal_threshold(spiked, 0.9)
    assert a[2] == b[2] and a[3] == b[3]
    assert a[:4].tolist()[2:] == b[:4].tolist()[2:]


def test_the_threshold_expands_with_history():
    """Month 3 is ranked against months 1 and 2 together, not month 2 alone."""
    wf = _wf([0.0, 0.0, 10.0, 10.0, 1.0, 1.0], [1, 1, 2, 2, 3, 3])
    thr = causal_threshold(wf, 0.5)
    assert thr[4] == pytest.approx(np.quantile([0, 0, 10, 10], 0.5))


def test_a_higher_quantile_never_trades_more():
    wf = _wf(np.arange(-50, 50, dtype=float), np.repeat([1, 2, 3, 4], 25))
    counts = []
    for q in (0.5, 0.8, 0.9, 0.95):
        thr = causal_threshold(wf, q)
        counts.append(int((np.isfinite(thr) & (np.abs(wf.prediction) >= thr)).sum()))
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# the signal itself
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feat():
    return build(_series(n_sessions=140, bars=200, seed=3, signal_strength=1.5))


def test_signal_direction_follows_the_prediction(feat, tmp_path, monkeypatch):
    monkeypatch.setattr("research.search.model_signal.CACHE", tmp_path)
    sig = model_confidence(feat, horizon=30, regime="rth", quantile=0.8,
                           side="both", min_train_months=2)
    assert sig.idx.size > 0
    assert set(np.unique(sig.direction)) <= {-1, 1}
    assert (np.diff(sig.idx) > 0).all()          # sorted, no duplicates
    assert sig.idx.max() < feat.ts.size


def test_side_filters_are_subsets_of_both(feat, tmp_path, monkeypatch):
    monkeypatch.setattr("research.search.model_signal.CACHE", tmp_path)
    kw = dict(horizon=30, regime="rth", quantile=0.8, min_train_months=2)
    both = model_confidence(feat, side="both", **kw)
    lng = model_confidence(feat, side="long", **kw)
    sht = model_confidence(feat, side="short", **kw)
    assert set(lng.idx) <= set(both.idx) and set(sht.idx) <= set(both.idx)
    assert set(lng.idx) & set(sht.idx) == set()
    assert lng.idx.size + sht.idx.size == both.idx.size
    assert (lng.direction == 1).all() and (sht.direction == -1).all()


def test_a_tighter_quantile_is_a_subset_of_a_looser_one(feat, tmp_path, monkeypatch):
    monkeypatch.setattr("research.search.model_signal.CACHE", tmp_path)
    kw = dict(horizon=30, regime="rth", side="both", min_train_months=2)
    loose = model_confidence(feat, quantile=0.5, **kw)
    tight = model_confidence(feat, quantile=0.95, **kw)
    assert set(tight.idx) <= set(loose.idx)
    assert tight.idx.size < loose.idx.size


def test_rth_regime_only_fires_inside_rth(feat, tmp_path, monkeypatch):
    monkeypatch.setattr("research.search.model_signal.CACHE", tmp_path)
    sig = model_confidence(feat, horizon=30, regime="rth", quantile=0.8,
                           side="both", min_train_months=2)
    assert feat.is_rth[sig.idx].all()


# ---------------------------------------------------------------------------
# the cache cannot be pointed at the wrong bars
# ---------------------------------------------------------------------------


def test_a_different_bar_array_cannot_hit_the_cache(feat, tmp_path, monkeypatch):
    """bar_idx is a position. Reusing a cache across slices aims every signal
    at the wrong bar, and nothing about the output would look wrong -- so the
    fingerprint is in the filename and a different slice simply misses."""
    monkeypatch.setattr("research.search.model_signal.CACHE", tmp_path)
    predictions(feat, 30, "rth", min_train_months=2)
    other = build(_series(n_sessions=100, bars=200, seed=9))
    assert fingerprint(other) != fingerprint(feat)

    files_before = set(p.name for p in tmp_path.glob("*.parquet"))
    predictions(other, 30, "rth", min_train_months=2)
    files_after = set(p.name for p in tmp_path.glob("*.parquet"))
    assert len(files_after - files_before) == 1   # recomputed, not reused


def test_cache_round_trips(feat, tmp_path, monkeypatch):
    monkeypatch.setattr("research.search.model_signal.CACHE", tmp_path)
    a = predictions(feat, 30, "rth", min_train_months=2)
    assert a.prediction.size > 0
    b = predictions(feat, 30, "rth", min_train_months=2)   # reads the parquet
    np.testing.assert_array_equal(a.bar_idx, b.bar_idx)
    np.testing.assert_allclose(a.prediction, b.prediction)


# ---------------------------------------------------------------------------
# power and control
# ---------------------------------------------------------------------------
#
# The pair that makes a null result mean something: the family has to find a
# planted signal, and has to find nothing when the link is destroyed.


def _accuracy(f, *, shuffle_target=False, seed=0):
    """Sign accuracy over all predicted bars, and over the confident subset.

    The planted link in the fixture is one bar deep -- this bar's position in
    its range moves the next bar -- so horizon=1 is where it lives. At h=30 it
    is diluted across twenty-nine bars of noise and undetectable, which is a
    fact about the fixture, not about the family.
    """
    wf = predictions(f, 1, "rth", shuffle_target=shuffle_target, seed=seed,
                     min_train_months=2)
    live = wf.actual != 0
    overall = float((np.sign(wf.prediction[live]) == np.sign(wf.actual[live])).mean())

    thr = causal_threshold(wf, 0.8)
    fire = np.isfinite(thr) & (np.abs(wf.prediction) >= thr) & live
    conf = (float((np.sign(wf.prediction[fire]) == np.sign(wf.actual[fire])).mean())
            if fire.sum() >= 100 else None)
    return overall, conf, int(live.sum()), int(fire.sum())


def test_confident_trades_find_a_planted_signal(tmp_path, monkeypatch):
    """Power check. Without this, the family scoring nothing means nothing."""
    monkeypatch.setattr("research.search.model_signal.CACHE", tmp_path)
    f = build(_series(n_sessions=140, bars=200, seed=11, signal_strength=1.5))
    overall, conf, n, n_conf = _accuracy(f)
    assert overall > 0.55, f"planted signal not recovered: {overall:.3f} on {n} bars"
    assert conf is not None and conf > overall, (
        f"confidence should sort it: {conf:.3f} confident vs {overall:.3f} overall"
    )


def test_shuffling_the_target_destroys_it(tmp_path, monkeypatch):
    """Same bars, same features, sessions of the target permuted.

    One-sided on purpose. A leak shows up as accuracy above chance; a shuffled
    run landing below chance on a few hundred correlated bars is noise, and
    asserting on that would make the test flaky for no reason.
    """
    monkeypatch.setattr("research.search.model_signal.CACHE", tmp_path)
    f = build(_series(n_sessions=140, bars=200, seed=11, signal_strength=1.5))
    overall, conf, n, n_conf = _accuracy(f, shuffle_target=True, seed=7)
    assert overall < 0.52, f"shuffled control scored {overall:.3f} on {n} bars"
    if conf is not None:
        assert conf < 0.56, f"shuffled confident subset scored {conf:.3f} on {n_conf}"
