from __future__ import annotations

import pandas as pd
import pytest

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.sweep_benchmark import (
    BenchmarkResult,
    MAX_SAMPLE,
    MIN_VIABLE_SAMPLE,
    benchmark_configs,
    decide_sizing,
    sample_benchmark_configs,
    sample_sweep_population,
    sizing_rule,
)


def _configs(n):
    return [
        StrategyConfig(name=f"c{i}", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
        for i in range(n)
    ]


def test_sample_benchmark_configs_returns_distinct_configs_without_replacement():
    configs = _configs(20)
    sample = sample_benchmark_configs(configs, n=10, seed=1)
    assert len(sample) == 10
    assert len({c.name for c in sample}) == 10


def test_sample_benchmark_configs_deterministic_with_fixed_seed():
    configs = _configs(20)
    a = sample_benchmark_configs(configs, n=10, seed=7)
    b = sample_benchmark_configs(configs, n=10, seed=7)
    assert [c.name for c in a] == [c.name for c in b]


def test_sample_benchmark_configs_caps_at_population_size():
    configs = _configs(5)
    sample = sample_benchmark_configs(configs, n=50, seed=1)
    assert len(sample) == 5


def test_benchmark_configs_runs_each_config_and_reports_median(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    configs = _configs(3)
    result = benchmark_configs(df, "NQ", configs, workers=4)
    assert isinstance(result, BenchmarkResult)
    assert result.n_benchmarked == 3
    assert len(result.per_config_seconds) == 3
    assert all(t >= 0 for t in result.per_config_seconds)
    assert result.median_seconds_per_config >= 0
    assert result.workers == 4


def test_sizing_rule_formula():
    # floor(90 * 60 * 4 / 1.0) = 21600
    assert sizing_rule(median_seconds_per_config=1.0, workers=4, budget_minutes=90) == 21600


def test_sizing_rule_caps_at_max_sample():
    assert sizing_rule(median_seconds_per_config=0.001, workers=64, budget_minutes=90) == MAX_SAMPLE


def test_sizing_rule_rejects_nonpositive_median():
    with pytest.raises(ValueError):
        sizing_rule(median_seconds_per_config=0.0, workers=4)


def test_sizing_rule_rejects_nonpositive_workers():
    with pytest.raises(ValueError):
        sizing_rule(median_seconds_per_config=1.0, workers=0)


def test_decide_sizing_viable_when_n_above_threshold():
    bench = BenchmarkResult(n_benchmarked=50, median_seconds_per_config=0.01, workers=8, per_config_seconds=tuple([0.01] * 50))
    decision = decide_sizing(bench)
    assert decision.viable is True
    assert decision.n >= MIN_VIABLE_SAMPLE
    assert "STOP for go-ahead" in decision.message


def test_decide_sizing_not_viable_when_n_below_threshold():
    # Deliberately slow median so N < 10000: floor(90*60*1/60) = 90.
    bench = BenchmarkResult(n_benchmarked=50, median_seconds_per_config=60.0, workers=1, per_config_seconds=tuple([60.0] * 50))
    decision = decide_sizing(bench)
    assert decision.viable is False
    assert decision.n < MIN_VIABLE_SAMPLE
    assert "do not run" in decision.message


def test_sample_sweep_population_always_includes_three_named_configs():
    configs = _configs(20)
    population = sample_sweep_population(configs, n=5, seed=1)
    names = [c.name for c in population]
    assert {"as_taught_5m", "as_taught_1m", "as_traded"} <= set(names)


def test_sample_sweep_population_size_is_n_plus_three_named():
    configs = _configs(20)
    population = sample_sweep_population(configs, n=5, seed=1)
    assert len(population) == 5 + 3


def test_sample_sweep_population_caps_n_at_canonical_population_size():
    configs = _configs(4)
    population = sample_sweep_population(configs, n=100, seed=1)
    assert len(population) == 4 + 3
