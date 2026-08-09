from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.sweep_benchmark import SizingDecision, BenchmarkResult
from ict_lab.engine.sweep_orchestrator import Timer, run_part0, run_parts_1_through_7
from ict_lab.engine.summary_pack import SUMMARY_FILES


def _make_bars(base_price, start="2024-06-03 00:00:00", end="2024-06-24 00:00:00"):
    rng = np.random.default_rng(int(base_price))
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    n = len(idx)
    close = base_price + np.cumsum(rng.normal(0, 0.1, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + rng.uniform(0, 0.1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.1, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 10, "contract": "X"}, index=idx
    )


def _small_population(n=8):
    configs = []
    entry_levels = ["proximal", "50%", "distal"]
    windows_options = [("killzone_ny_am",), ("killzone_ny_pm",), ("killzone_london", "killzone_ny_am")]
    for i in range(n):
        configs.append(
            StrategyConfig(
                name=f"pop_{i}",
                windows=windows_options[i % len(windows_options)],
                sweep_required=False,
                stop_type="gap_distal",
                displacement_required=False,
                mss_required=False,
                entry_level=entry_levels[i % len(entry_levels)],
                target_type="fixed_r",
                target_r_multiple=2.0,
            )
        )
    return configs


def test_run_part0_returns_population_and_decision(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    population, canon_report, bench, decision = run_part0(df, "NQ", timer=Timer(), benchmark_sample_size=5)
    assert canon_report["canonical"] == len(population)
    assert bench.n_benchmarked == 5
    assert isinstance(decision, SizingDecision)


def test_run_parts_1_through_7_end_to_end_writes_all_summary_files(tmp_path):
    df_nq = _make_bars(100.0, end="2024-06-10 00:00:00")
    df_es = _make_bars(5000.0, end="2024-06-10 00:00:00")
    population = _small_population(8)

    # Force a tiny, fully viable sample size so the test stays fast --
    # bypasses Part 0's benchmark, which is tested separately.
    fake_bench = BenchmarkResult(n_benchmarked=5, median_seconds_per_config=0.01, workers=2, per_config_seconds=(0.01,) * 5)
    decision = SizingDecision(benchmark=fake_bench, n=5, viable=True, message="test")

    shard_dir = tmp_path / "shards"
    output_dir = tmp_path / "summary"

    result = run_parts_1_through_7(
        df_nq, df_es, population, decision, shard_dir, output_dir, n_null_iterations=5, workers=2,
    )

    for filename in SUMMARY_FILES:
        path = output_dir / filename
        assert path.exists(), f"{filename} was not written"
        assert path.stat().st_size > 0, f"{filename} is empty"

    assert "summary" in result and len(result["summary"]) > 0
    assert "funnel_result" in result
    assert "null_results" in result
    assert set(result["null_results"]) >= {"as_taught_5m", "as_taught_1m", "as_traded"}
    assert "es_null_results" in result
    assert set(result["es_null_results"]) == {"as_taught_5m", "as_traded"}
    assert "ablation" in result and len(result["ablation"]) == 12
    assert "timings" in result and len(result["timings"]) > 0


def test_run_parts_1_through_7_never_imports_holdout_module():
    import ict_lab.engine.sweep_orchestrator as orch_mod
    import inspect

    source = inspect.getsource(orch_mod)
    assert "import ict_lab.data.holdout" not in source
    assert "from ict_lab.data.holdout" not in source
    assert "from ict_lab.data import holdout" not in source
    assert "include_holdout=True" not in source
