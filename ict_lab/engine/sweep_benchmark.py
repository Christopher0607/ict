"""Phase 5 Part 0 items 3-4: benchmark the full pipeline against a random
sample of canonical configs, then apply the sizing rule to determine how
many configs the 90-minute sweep budget can actually afford.

Benchmark methodology note: each config is timed with its OWN fresh
FeatureStore (no cross-config cache sharing), because Part 1's real
worker-process architecture can't transparently share one Python object's
in-memory cache across process-pool workers -- this is the conservative,
worst-case-per-config number the sizing rule should be built on. If Part 1
ends up batching configs that share detector parameters through one worker
with a shared store, the real run will be faster than this projects, which
is the safe direction to be wrong in.

This module only computes the sizing DECISION (N, viable or not, why). The
actual "STOP for my go-ahead before launching" gate -- refusing to proceed
to Part 1 without an explicit human confirmation -- is enforced by the
Part 7 orchestrator CLI, not here.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ict_lab.configs.grid import load_named_configs
from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.pipeline import run_config

BENCHMARK_SAMPLE_SIZE = 50
SWEEP_BUDGET_MINUTES = 90
MAX_SAMPLE = 25000
MIN_VIABLE_SAMPLE = 10000
BENCHMARK_SEED = 20260809  # arbitrary fixed seed, documented for reproducibility -- not tuned to any result


@dataclass(frozen=True)
class BenchmarkResult:
    n_benchmarked: int
    median_seconds_per_config: float
    workers: int
    per_config_seconds: tuple[float, ...]


def sample_benchmark_configs(
    canonical_configs: list[StrategyConfig], n: int = BENCHMARK_SAMPLE_SIZE, seed: int = BENCHMARK_SEED
) -> list[StrategyConfig]:
    rng = np.random.default_rng(seed)
    n = min(n, len(canonical_configs))
    idx = rng.choice(len(canonical_configs), size=n, replace=False)
    return [canonical_configs[i] for i in idx]


def benchmark_configs(
    df_1m: pd.DataFrame, symbol: str, configs: list[StrategyConfig], workers: int | None = None
) -> BenchmarkResult:
    """Runs each config once through the full pipeline (signals, fills,
    PnL, and trade/no-trade log construction -- run_config already covers
    all four; there is no separate disk-logging step at this stage, that's
    Part 1's shard-parquet writer), timing each independently. Sequential
    by design -- this measures single-config cost; the sizing rule divides
    it across `workers`. It is not itself the parallel sweep."""
    workers = workers or (os.cpu_count() or 1)
    timings = []
    for config in configs:
        start = time.perf_counter()
        run_config(df_1m, config, symbol)
        timings.append(time.perf_counter() - start)

    return BenchmarkResult(
        n_benchmarked=len(configs),
        median_seconds_per_config=float(np.median(timings)),
        workers=workers,
        per_config_seconds=tuple(timings),
    )


def sizing_rule(median_seconds_per_config: float, workers: int, budget_minutes: int = SWEEP_BUDGET_MINUTES) -> int:
    """N = min(25000, floor(90min * workers / median_sec_per_config))."""
    if median_seconds_per_config <= 0:
        raise ValueError("median_seconds_per_config must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    n = math.floor(budget_minutes * 60 * workers / median_seconds_per_config)
    return min(MAX_SAMPLE, n)


def sample_sweep_population(
    canonical_configs: list[StrategyConfig], n: int, seed: int = BENCHMARK_SEED
) -> list[StrategyConfig]:
    """Uniform random, one draw, fixed seed, n from the canonical
    population. The three named configs are always included on top of the
    sample (not counted against n)."""
    rng = np.random.default_rng(seed)
    n = min(n, len(canonical_configs))
    idx = rng.choice(len(canonical_configs), size=n, replace=False)
    sample = [canonical_configs[i] for i in idx]
    named = list(load_named_configs().values())
    return named + sample


@dataclass(frozen=True)
class SizingDecision:
    benchmark: BenchmarkResult
    n: int
    viable: bool
    message: str


def decide_sizing(benchmark: BenchmarkResult) -> SizingDecision:
    n = sizing_rule(benchmark.median_seconds_per_config, benchmark.workers)
    if n < MIN_VIABLE_SAMPLE:
        return SizingDecision(
            benchmark=benchmark,
            n=n,
            viable=False,
            message=(
                f"N={n} < {MIN_VIABLE_SAMPLE}: do not run. Optimize first -- batch configs sharing "
                f"detector streams through a single pass over the sessions, vectorize threshold "
                f"filters/sequencing/fills across the batch. Target N>={MIN_VIABLE_SAMPLE} inside "
                f"{SWEEP_BUDGET_MINUTES} minutes, re-benchmark, then run."
            ),
        )
    return SizingDecision(
        benchmark=benchmark,
        n=n,
        viable=True,
        message=(
            f"N={n} canonical configs sampled (plus the 3 named configs on top). "
            f"STOP for go-ahead before launching Part 1."
        ),
    )
