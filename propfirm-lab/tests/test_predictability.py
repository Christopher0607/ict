"""The predictability audit, and the two controls that make it mean anything.

An audit that reports "no signal" is worthless unless it can be shown to
(a) find a signal that is really there, and (b) not find one that is not.
Both are tested here on synthetic data with a known answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.search.features import build
from research.search.predictability import (
    forward_return,
    walk_forward_audit,
)


def _series(n_sessions=140, bars=200, seed=0, signal_strength=0.0):
    """Random-walk bars, optionally with a genuinely predictable component.

    When ``signal_strength`` > 0 the next bar's move is partly determined by
    the current bar's position in its own range -- a real, learnable link.
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = 20_000.0
    carry = 0.0
    for s in range(n_sessions):
        day = pd.Timestamp("2020-01-06", tz="UTC") + pd.Timedelta(days=s)
        start = day + pd.Timedelta(hours=14, minutes=30)
        for m in range(bars):
            step = rng.normal(0, 3) + signal_strength * carry
            o = price
            c = price + step
            hi = max(o, c) + abs(rng.normal(0, 1))
            lo = min(o, c) - abs(rng.normal(0, 1))
            rows.append((start + pd.Timedelta(minutes=m), o, hi, lo, c,
                         float(rng.integers(100, 2000))))
            carry = (c - lo) / (hi - lo) - 0.5 if hi > lo else 0.0
            price = c
    return pd.DataFrame(rows, columns=["ts_open", "open", "high", "low", "close", "volume"])


def test_forward_return_does_not_cross_sessions():
    close = np.array([10.0, 11.0, 12.0, 100.0, 101.0])
    sess = np.array([0, 0, 0, 1, 1])
    fr = forward_return(close, sess, horizon=2)
    assert fr[0] == pytest.approx(2.0)
    assert np.isnan(fr[1])   # would land in session 1
    assert np.isnan(fr[3])   # not enough bars left
    assert np.isnan(fr[4])


def test_audit_finds_a_planted_signal():
    """Power check. If this fails, a null result from the audit means nothing."""
    f = build(_series(seed=1, signal_strength=1.5))
    res = walk_forward_audit(f, horizon=1, min_train_months=2)
    assert res.n_test > 1000
    assert res.oos_r2 > 0.01, f"planted signal not detected (R2={res.oos_r2:.5f})"
    assert res.sign_edge_t > 3


def test_audit_reports_nothing_on_a_random_walk():
    """Specificity check on data with no learnable structure."""
    f = build(_series(seed=2, signal_strength=0.0))
    res = walk_forward_audit(f, horizon=1, min_train_months=2)
    assert res.oos_r2 < 0.005, f"found structure in a random walk (R2={res.oos_r2:.5f})"


def test_shuffled_target_scores_zero():
    """The leakage control: destroy the link, the audit must find nothing.

    A positive result here would mean the walk-forward is reading its own
    future, and every other number the audit produces would be worthless.
    """
    f = build(_series(seed=3, signal_strength=1.5))
    real = walk_forward_audit(f, horizon=1, min_train_months=2)
    shuffled = walk_forward_audit(f, horizon=1, min_train_months=2, shuffle_target=True)

    assert real.oos_r2 > 0.01
    assert shuffled.oos_r2 < 0.005, (
        f"shuffled target still scores R2={shuffled.oos_r2:.5f} -- the audit leaks"
    )


def test_in_sample_r2_exceeds_out_of_sample_on_noise():
    """Why only the out-of-sample number is reported: the in-sample one always
    looks better, because enough parameters fit any noise."""
    f = build(_series(seed=4, signal_strength=0.0))
    res = walk_forward_audit(f, horizon=5, min_train_months=2)
    assert res.in_sample_r2 > res.oos_r2


def test_regime_filter_reduces_the_sample():
    f = build(_series(seed=5))
    all_ = walk_forward_audit(f, horizon=1, min_train_months=2, regime="all")
    rth = walk_forward_audit(f, horizon=1, min_train_months=2, regime="rth")
    assert rth.n_test <= all_.n_test


def test_flat_bars_do_not_count_against_sign_accuracy():
    """np.sign(0) == 0 would score every flat bar as a wrong prediction.

    At a 1-minute horizon 8.8% of real NQ bars are exactly flat, which alone
    drags measured accuracy from 51% to 49% and turns a real edge into an
    apparent anti-edge.
    """
    f = build(_series(seed=6, signal_strength=1.5))
    res = walk_forward_audit(f, horizon=1, min_train_months=2)
    # A detectable planted signal must read as better than a coin flip, not worse.
    assert res.sign_accuracy > 0.5
    assert res.sign_edge_t > 0
