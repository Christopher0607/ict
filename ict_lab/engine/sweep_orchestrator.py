"""Phase 5 orchestrator: wires Parts 0-7 together.

    python -m ict_lab.engine.sweep_orchestrator --shard-dir runs/sweep_nq --output-dir analysis/summary

PRECONDITION (spec's own first line): all engine and feature tests must
pass before this runs. This script does not invoke pytest itself -- that
would mix a test-runner concern into a production data pipeline -- run the
suite by hand first.

Part 0 (grid enumeration, canonicalization, benchmark, sizing) always runs
and always STOPS after printing its sizing decision: launching Part 1 (the
real sweep) and beyond needs an explicit --confirm-launch flag on a
SECOND invocation, after the sizing decision has actually been read. This
is the spec's own gate ("STOP for my go-ahead before launching") and nothing
here weakens it, even once real data exists -- --confirm-launch is a
human's typed decision, never inferred or defaulted to true.

Never touches the holdout: every load_symbol call here relies on its
default include_holdout=False, and nothing in this module imports
ict_lab.data.holdout. Never changes the grid in response to results (the
grid is fixed in configs/grid.json, read once at Part 0 and never mutated
by anything this script does after seeing a result).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from ict_lab.configs.grid import load_named_configs
from ict_lab.configs.sweep_grid import build_canonical_population
from ict_lab.data.loader import load_symbol
from ict_lab.engine.ablation import run_ablation_ladder
from ict_lab.engine.discretion_premium import run_discretion_premium
from ict_lab.engine.funnel import es_validation, run_funnel
from ict_lab.engine.null_models import run_null_tests, select_reference_configs
from ict_lab.engine.pipeline import all_session_dates, run_config
from ict_lab.engine.statistics_slices import (
    deflated_sharpe_ratio_for_config,
    era_split_for_configs,
    per_year_tables_for_survivors,
    roll_day_sensitivity_for_survivors,
    slippage_sensitivity_for_survivors,
)
from ict_lab.engine.summary_pack import (
    write_ablation_ladder,
    write_discretion_premium,
    write_era_split,
    write_es_validation,
    write_frequency_report,
    write_funnel_counts,
    write_named_configs_report,
    write_null_summary,
    write_results_configs,
    write_timing_report,
)
from ict_lab.engine.sweep_benchmark import benchmark_configs, decide_sizing, sample_benchmark_configs, sample_sweep_population
from ict_lab.engine.sweep_runner import config_from_row, run_sweep, stats_from_trades

DISCRETION_PREMIUM_REFS = ("as_taught_5m", "as_traded", "best_realistic_survivor")
ES_NULL_TEST_REFS = ("as_taught_5m", "as_traded")


class Timer:
    """Records elapsed wall-clock time per named stage, for Part 7's
    timing_report.txt -- lets the actual run account for its own time
    against the spec's 2-hour budget."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def stage(self, name: str) -> "_StageTimer":
        return _StageTimer(self, name)


class _StageTimer:
    def __init__(self, timer: Timer, name: str) -> None:
        self._timer, self._name = timer, name

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self._timer.timings[self._name] = time.perf_counter() - self._start


def run_part0(df_nq_1m: pd.DataFrame, symbol: str, timer: Timer, benchmark_sample_size: int | None = None):
    """Enumeration + canonicalization + dedup + benchmark + sizing. Always
    safe to run (read-only against the grid and a benchmark sample of the
    already-loaded NQ data) -- this is the part that decides whether
    Part 1 should even be attempted. benchmark_sample_size overrides the
    spec's 50-config default -- exists for fast integration testing of the
    wiring, never for a real run (the spec names 50 explicitly)."""
    print("=== Part 0: canonicalization, benchmark, sizing ===", file=sys.stderr)
    with timer.stage("part0_canonicalization"):
        population, canon_report = build_canonical_population()
    print(f"canonical population: {canon_report}", file=sys.stderr)

    with timer.stage("part0_benchmark"):
        sample_kwargs = {"n": benchmark_sample_size} if benchmark_sample_size is not None else {}
        sample = sample_benchmark_configs(population, **sample_kwargs)
        bench = benchmark_configs(df_nq_1m, symbol, sample)
    decision = decide_sizing(bench)
    print(
        f"benchmark: median {bench.median_seconds_per_config:.3f}s/config over "
        f"{bench.n_benchmarked} configs, {bench.workers} workers",
        file=sys.stderr,
    )
    print(decision.message, file=sys.stderr)
    return population, canon_report, bench, decision


def _config_result_rows(configs: dict, data_by_symbol: dict[str, pd.DataFrame]) -> list[dict]:
    """Runs each config against each (symbol, data) pair once; returns one
    light-summary row per (config, symbol), reused by both the named-configs
    report and the frequency report so nothing gets run twice."""
    rows = []
    for name, config in configs.items():
        for symbol, df_1m in data_by_symbol.items():
            session_dates = all_session_dates(df_1m)
            _, _, trades, _ = run_config(df_1m, config, symbol)
            stats = stats_from_trades(config, trades, session_dates)
            rows.append(
                {
                    "config_name": name,
                    "symbol": symbol,
                    "trades": stats["trade_count"],
                    "win_rate": stats["win_rate"],
                    "avg_r": stats["avg_r"],
                    "net_sharpe": stats["net_sharpe"],
                    "net_pnl": stats["net_pnl"],
                    "trades_per_year": stats["trades_per_year"],
                    "pct_days_with_trade": stats["pct_days_with_trade"],
                }
            )
    return rows


def run_parts_1_through_7(
    df_nq_1m: pd.DataFrame,
    df_es_1m: pd.DataFrame,
    population: list,
    decision,
    shard_dir: Path,
    output_dir: Path,
    n_null_iterations: int = 1000,
    workers: int | None = None,
) -> dict:
    """Only ever called after an explicit human --confirm-launch. Runs
    Parts 1-7 in order, never skipping a stage, and writes the summary
    pack. Returns the full set of intermediate results (useful for tests
    and for anyone driving this interactively rather than via the CLI)."""
    timer = Timer()
    symbol_nq, symbol_es = "NQ", "ES"
    shard_dir, output_dir = Path(shard_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Part 1: the sweep (NQ) ===", file=sys.stderr)
    with timer.stage("part1_sweep"):
        sweep_sample = sample_sweep_population(population, decision.n)
        summary = run_sweep(df_nq_1m, sweep_sample, symbol_nq, shard_dir, workers=workers)
    print(f"sweep complete: {len(summary)} configs", file=sys.stderr)

    print("=== Part 2: the funnel + ES validation ===", file=sys.stderr)
    with timer.stage("part2_funnel"):
        funnel_result = run_funnel(summary, shard_dir)
    survivors = funnel_result["population"][funnel_result["population"]["survivor"]]
    print(f"funnel counts: {funnel_result['counts']}", file=sys.stderr)

    with timer.stage("part2_es_validation"):
        es_df = es_validation(survivors, df_es_1m)
    es_counts = {
        "survivors_tested": int((es_df["source"] == "survivor").sum()),
        "cross_symbol_survivors": int(es_df.loc[es_df["source"] == "survivor", "cross_symbol_survivor"].sum()),
    }
    print(f"ES validation: {es_counts}", file=sys.stderr)

    named_configs = load_named_configs()
    reference_configs = select_reference_configs(summary, survivors, named_configs)

    print("=== Part 3: null tests ===", file=sys.stderr)
    with timer.stage("part3_null_tests_nq"):
        null_results = run_null_tests(df_nq_1m, symbol_nq, reference_configs, n_iterations=n_null_iterations)
    with timer.stage("part3_null_tests_es"):
        es_ref_subset = {k: v for k, v in named_configs.items() if k in ES_NULL_TEST_REFS}
        es_null_results = run_null_tests(
            df_es_1m, symbol_es, es_ref_subset, n_iterations=n_null_iterations, include_shuffled=False
        )

    print("=== Part 4: ablation ladder ===", file=sys.stderr)
    with timer.stage("part4_ablation"):
        ablation_df = run_ablation_ladder(df_nq_1m, df_es_1m)

    print("=== Part 5: discretion premium ===", file=sys.stderr)
    with timer.stage("part5_discretion_premium"):
        discretion_refs = {k: v for k, v in reference_configs.items() if k in DISCRETION_PREMIUM_REFS}
        discretion_reports = run_discretion_premium(df_nq_1m, symbol_nq, discretion_refs)

    print("=== Part 6: statistics and slices ===", file=sys.stderr)
    with timer.stage("part6_statistics"):
        dsr = None
        if not survivors.empty:
            best_hash = survivors.loc[survivors["net_sharpe"].idxmax(), "config_hash"]
            dsr = deflated_sharpe_ratio_for_config(summary, best_hash, shard_dir)

        per_year = per_year_tables_for_survivors(df_nq_1m, symbol_nq, survivors)

        era_configs = {row["config_name"]: config_from_row(row) for _, row in survivors.iterrows()}
        era_configs.update(named_configs)
        era_results = era_split_for_configs(df_nq_1m, symbol_nq, era_configs)

        roll_day = roll_day_sensitivity_for_survivors(df_nq_1m, symbol_nq, survivors)
        slippage = slippage_sensitivity_for_survivors(df_nq_1m, symbol_nq, survivors)

    print("=== Part 7: summary pack ===", file=sys.stderr)
    with timer.stage("part7_summary_pack"):
        named_rows = _config_result_rows(named_configs, {symbol_nq: df_nq_1m, symbol_es: df_es_1m})

        write_results_configs(output_dir, summary)
        write_funnel_counts(output_dir, funnel_result, es_counts)
        write_named_configs_report(output_dir, named_rows)
        write_es_validation(output_dir, es_df)
        write_ablation_ladder(output_dir, ablation_df)
        write_null_summary(output_dir, {**null_results, **{f"{k}_ES": v for k, v in es_null_results.items()}})
        write_discretion_premium(output_dir, discretion_reports)
        write_era_split(output_dir, era_results)
        write_frequency_report(output_dir, summary, [r for r in named_rows if r["symbol"] == symbol_nq])
        write_timing_report(output_dir, timer.timings)

    return {
        "summary": summary,
        "funnel_result": funnel_result,
        "es_validation": es_df,
        "null_results": null_results,
        "es_null_results": es_null_results,
        "ablation": ablation_df,
        "discretion_premium": discretion_reports,
        "dsr": dsr,
        "per_year": per_year,
        "era_split": era_results,
        "roll_day_sensitivity": roll_day,
        "slippage_sensitivity": slippage,
        "timings": timer.timings,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", default="runs/sweep_nq", help="Sweep checkpoint/shard directory")
    parser.add_argument("--output-dir", default="analysis/summary", help="Where to write the summary pack")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-null-iterations", type=int, default=1000)
    parser.add_argument(
        "--confirm-launch", action="store_true",
        help="Required to run Part 1 onward. Omit this on the first run to see Part 0's sizing decision only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df_nq = load_symbol("NQ", price_series="backadjusted")  # default include_holdout=False
    timer = Timer()
    population, canon_report, bench, decision = run_part0(df_nq, "NQ", timer)

    if not decision.viable:
        print("Sizing not viable -- stopping. See the message above for what to optimize.", file=sys.stderr)
        sys.exit(1)

    if not args.confirm_launch:
        print(
            "\nPart 0 complete. Re-run with --confirm-launch to launch Part 1 (the real sweep) and beyond, "
            "now that you've read the sizing decision above.",
            file=sys.stderr,
        )
        sys.exit(0)

    df_es = load_symbol("ES", price_series="backadjusted")  # default include_holdout=False
    run_parts_1_through_7(
        df_nq, df_es, population, decision, Path(args.shard_dir), Path(args.output_dir),
        n_null_iterations=args.n_null_iterations, workers=args.workers,
    )
