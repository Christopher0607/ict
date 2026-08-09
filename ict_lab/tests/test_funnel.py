from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.funnel import (
    FUNNEL_STAGE_NAMES,
    apply_bh_correction,
    block_bootstrap_p_value,
    compute_p_values,
    es_validation,
    estimate_bootstrap_seconds,
    run_funnel,
    ttest_p_value,
)
from ict_lab.engine.sweep_runner import _flush_r_multiples_shard, stats_from_trades

D1 = pd.Timestamp("2020-06-01")


def _write_r_multiples(shard_dir, config_hash: str, values: list[float]) -> None:
    _flush_r_multiples_shard(shard_dir, [{"config_hash": config_hash, "r_multiples": json.dumps(values)}])


# ---------- block_bootstrap_p_value ----------


def test_block_bootstrap_all_positive_gives_minimal_pvalue():
    r = np.full(100, 2.0)  # every resample mean is exactly 2.0 -- never <= 0
    p = block_bootstrap_p_value(r, n_resamples=1000, seed=1)
    assert p == pytest.approx(1 / 1001)


def test_block_bootstrap_all_negative_gives_pvalue_near_one():
    r = np.full(100, -2.0)
    p = block_bootstrap_p_value(r, n_resamples=1000, seed=1)
    assert p == pytest.approx(1.0)


def test_block_bootstrap_empty_is_nan():
    assert pd.isna(block_bootstrap_p_value(np.array([]), seed=1))


def test_block_bootstrap_deterministic_with_fixed_seed():
    r = np.array([1.0, -0.5, 2.0, -1.0, 0.5] * 10)
    a = block_bootstrap_p_value(r, n_resamples=200, seed=42)
    b = block_bootstrap_p_value(r, n_resamples=200, seed=42)
    assert a == b


# ---------- ttest_p_value ----------


def test_ttest_p_value_matches_manual_one_sided_halving():
    r = np.array([1.0, -0.5, 2.0, -1.0, 0.5, 3.0, -0.2])
    t_stat, two_sided = scipy_stats.ttest_1samp(r, popmean=0.0)
    expected = two_sided / 2 if t_stat > 0 else 1 - two_sided / 2
    assert ttest_p_value(r) == pytest.approx(expected)


def test_ttest_p_value_all_positive_is_small():
    r = np.random.default_rng(3).normal(loc=2.0, scale=0.5, size=50)
    assert ttest_p_value(r) < 0.01


def test_ttest_p_value_needs_at_least_two_observations():
    assert pd.isna(ttest_p_value(np.array([1.0])))
    assert pd.isna(ttest_p_value(np.array([])))


# ---------- apply_bh_correction ----------


def test_apply_bh_correction_rejects_small_pvalues_keeps_large():
    p_values = pd.Series({"a": 0.001, "b": 0.01, "c": 0.5, "d": 0.9}, name="p_value")
    result = apply_bh_correction(p_values, alpha=0.10)
    assert result["a"] and result["b"]
    assert not result["c"] and not result["d"]


def test_apply_bh_correction_all_nan_is_all_false():
    p_values = pd.Series({"a": float("nan"), "b": float("nan")})
    result = apply_bh_correction(p_values)
    assert not result.any()


def test_apply_bh_correction_empty_series():
    result = apply_bh_correction(pd.Series([], dtype=float))
    assert result.empty


# ---------- estimate_bootstrap_seconds ----------


def test_estimate_bootstrap_seconds_is_nonnegative(tmp_path):
    population = pd.DataFrame({"config_hash": ["h1", "h2", "h3"]})
    for h in population["config_hash"]:
        _write_r_multiples(tmp_path, h, [1.0, -0.5, 2.0] * 10)
    projected = estimate_bootstrap_seconds(population, tmp_path, sample_size=3, seed=1)
    assert projected >= 0


def test_estimate_bootstrap_seconds_empty_population_is_zero(tmp_path):
    assert estimate_bootstrap_seconds(pd.DataFrame(columns=["config_hash"]), tmp_path) == 0.0


# ---------- compute_p_values ----------


def test_compute_p_values_generous_budget_uses_bootstrap(tmp_path):
    population = pd.DataFrame({"config_hash": ["h1", "h2"]})
    _write_r_multiples(tmp_path, "h1", list(np.random.default_rng(1).normal(1.0, 0.3, 50)))
    _write_r_multiples(tmp_path, "h2", list(np.random.default_rng(2).normal(-1.0, 0.3, 50)))
    p_values, method = compute_p_values(population, tmp_path, time_budget_seconds=1e9, seed=1)
    assert method == "bootstrap"
    assert set(p_values.index) == {"h1", "h2"}
    assert p_values["h1"] < p_values["h2"]


def test_compute_p_values_tiny_budget_forces_ttest_fallback(tmp_path):
    population = pd.DataFrame({"config_hash": ["h1", "h2"]})
    _write_r_multiples(tmp_path, "h1", list(np.random.default_rng(1).normal(1.0, 0.3, 50)))
    _write_r_multiples(tmp_path, "h2", list(np.random.default_rng(2).normal(-1.0, 0.3, 50)))
    p_values, method = compute_p_values(population, tmp_path, time_budget_seconds=0.0, seed=1)
    assert method == "ttest_fallback"
    assert set(p_values.index) == {"h1", "h2"}


# ---------- run_funnel ----------


def _population_row(config_hash, trade_count, net_pnl, net_sharpe):
    return {"config_hash": config_hash, "trade_count": trade_count, "net_pnl": net_pnl, "net_sharpe": net_sharpe}


def test_run_funnel_stage_counts_decrease_monotonically_and_match_expected_survivors(tmp_path):
    rows = [
        _population_row("fails_trades", 50, 100.0, 1.0),  # fails stage 2
        _population_row("fails_pnl", 150, -10.0, 1.0),  # fails stage 3
        _population_row("fails_sharpe", 150, 100.0, 0.2),  # fails stage 4
        _population_row("survives_bh", 150, 100.0, 1.0),  # obviously significant R
        _population_row("fails_bh", 150, 100.0, 1.0),  # positive Sharpe but noisy/insignificant R
    ]
    population = pd.DataFrame(rows)

    _write_r_multiples(tmp_path, "fails_trades", [1.0] * 50)
    _write_r_multiples(tmp_path, "fails_pnl", [1.0] * 150)
    _write_r_multiples(tmp_path, "fails_sharpe", [1.0] * 150)
    _write_r_multiples(tmp_path, "survives_bh", [1.0] * 150)  # constant +1R -> p ~= 1/1001
    _write_r_multiples(tmp_path, "fails_bh", [0.01, -0.01] * 75)  # mean ~0 -> not significant

    result = run_funnel(population, tmp_path, seed=1)
    counts = result["counts"]

    assert list(counts) == list(FUNNEL_STAGE_NAMES)
    assert counts["population"] == 5
    assert counts["min_100_trades"] == 4  # everyone but fails_trades
    assert counts["net_pnl_positive"] == 3  # everyone but fails_trades, fails_pnl
    assert counts["net_sharpe_0_5"] == 2  # everyone but fails_trades, fails_pnl, fails_sharpe
    assert counts["bh_fdr_10pct"] == 1  # only survives_bh

    by_hash = result["population"].set_index("config_hash")
    assert by_hash.loc["survives_bh", "survivor"]
    assert not by_hash.loc["fails_bh", "survivor"]
    assert not by_hash.loc["fails_sharpe", "survivor"]


def test_run_funnel_reports_bh_method_used(tmp_path):
    population = pd.DataFrame([_population_row("h1", 150, 100.0, 1.0)])
    _write_r_multiples(tmp_path, "h1", [1.0] * 150)
    result = run_funnel(population, tmp_path, seed=1)
    assert result["bh_method"] in ("bootstrap", "ttest_fallback")


# ---------- es_validation ----------


def test_es_validation_runs_survivors_and_named_configs(synthetic_bars, tmp_path):
    df_es = synthetic_bars("2024-06-03 00:00:00", "2024-06-07 00:00:00")

    survivor_config = StrategyConfig(
        name="sweep_abc", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal",
    )
    survivor_row = stats_from_trades(survivor_config, pd.DataFrame(columns=["session_date"]), pd.DatetimeIndex([D1]))
    survivors = pd.DataFrame([survivor_row])

    named = {
        "fake_named": StrategyConfig(
            name="fake_named", windows=("killzone_ny_pm",), sweep_required=False, stop_type="gap_distal",
        )
    }

    result = es_validation(survivors, df_es, named_configs=named)

    assert len(result) == 2
    assert set(result["source"]) == {"survivor", "named"}
    assert set(result["config_name"]) == {"sweep_abc", "fake_named"}
    assert result["cross_symbol_survivor"].dtype == bool
