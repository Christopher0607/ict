"""Multiple-testing corrections."""

from __future__ import annotations

import numpy as np
import pytest

from research.search.stats import (
    bh_fdr,
    block_bootstrap_ci,
    deflated_sharpe,
    expected_max_sharpe,
)


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(0)
    p = rng.random(500)
    mine_rej, mine_adj = bh_fdr(p, q=0.10)
    ref_rej, ref_adj, _, _ = multipletests(p, alpha=0.10, method="fdr_bh")
    assert np.array_equal(mine_rej, ref_rej)
    assert np.allclose(mine_adj, ref_adj)


def test_bh_rejects_a_clear_signal_among_noise():
    p = np.concatenate([[1e-8], np.linspace(0.2, 0.99, 199)])
    rejected, _ = bh_fdr(p, q=0.10)
    assert rejected[0]
    assert rejected.sum() == 1


def test_bh_rejects_nothing_in_pure_noise():
    rng = np.random.default_rng(1)
    rejected, _ = bh_fdr(rng.random(1000), q=0.10)
    assert rejected.sum() == 0


def test_expected_max_sharpe_grows_with_trial_count():
    """The reason a big search raises the bar for everything in it."""
    v = 1.0 / 500
    assert expected_max_sharpe(10, v) < expected_max_sharpe(1_000, v)
    assert expected_max_sharpe(1_000, v) < expected_max_sharpe(100_000, v)


def test_deflated_sharpe_punishes_a_large_search():
    """The same return series, judged after 10 trials and after 50,000."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.05, 1.0, 1000)
    few = deflated_sharpe(r, n_trials=10)
    many = deflated_sharpe(r, n_trials=50_000)
    assert few["sharpe"] == pytest.approx(many["sharpe"])
    assert many["sr0"] > few["sr0"]
    assert many["dsr"] < few["dsr"]


def test_deflated_sharpe_near_zero_for_pure_noise_in_a_big_search():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 1.0, 800)
    assert deflated_sharpe(r, n_trials=8_656)["dsr"] < 0.5


def test_block_bootstrap_is_wider_than_naive_when_sessions_cluster():
    """Clustered trades carry less information than their count suggests."""
    rng = np.random.default_rng(5)
    # 50 sessions, each with 20 near-identical trades: 1000 trades, 50 facts.
    per_session = rng.normal(0.1, 1.0, 50)
    r = np.repeat(per_session, 20) + rng.normal(0, 0.01, 1000)
    sess = np.repeat(np.arange(50), 20)

    lo, hi = block_bootstrap_ci(r, sess, n_boot=500, seed=0)
    naive_half = 1.96 * np.std(r, ddof=1) / np.sqrt(r.size)
    assert (hi - lo) / 2 > naive_half


def test_block_bootstrap_brackets_the_mean():
    rng = np.random.default_rng(11)
    r = rng.normal(0.2, 1.0, 600)
    sess = np.repeat(np.arange(60), 10)
    lo, hi = block_bootstrap_ci(r, sess, n_boot=500, seed=1)
    assert lo < r.mean() < hi


def test_clustered_se_exceeds_naive_when_trades_cluster():
    """The correction that stops a big search manufacturing survivors."""
    from research.search.stats import clustered_se

    rng = np.random.default_rng(2)
    per_session = rng.normal(0.1, 1.0, 60)
    r = np.repeat(per_session, 15) + rng.normal(0, 0.01, 900)
    sess = np.repeat(np.arange(60), 15)

    naive = np.std(r, ddof=1) / np.sqrt(r.size)
    assert clustered_se(r, sess) > 2 * naive


def test_clustered_se_matches_naive_when_each_trade_is_its_own_session():
    from research.search.stats import clustered_se

    rng = np.random.default_rng(4)
    r = rng.normal(0.1, 1.0, 400)
    naive = np.std(r, ddof=1) / np.sqrt(r.size)
    assert clustered_se(r, np.arange(400)) == pytest.approx(naive, rel=0.02)
