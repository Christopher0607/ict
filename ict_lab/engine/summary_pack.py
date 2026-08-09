"""Phase 5 Part 7: summary pack. Writes analysis/summary/'s 10 files.
results_configs.csv is Part 1's full per-config table by design (the spec
names it as the sweep's raw output) and is not itself "small enough to
paste into a chat" for a full-size sweep -- that qualifier is read as
applying to the other 9, genuinely summary-shaped files.

Every writer takes already-computed results (from Parts 1-6) and a target
directory; nothing in this module runs the pipeline itself, so it can be
tested against hand-built inputs without needing real market data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ict_lab.configs.grid import placeholder_note
from ict_lab.engine.discretion_premium import format_discretion_premium

SUMMARY_FILES = (
    "results_configs.csv",
    "funnel_counts.json",
    "named_configs_report.md",
    "es_validation.csv",
    "ablation_ladder.csv",
    "null_summary.md",
    "discretion_premium.txt",
    "era_split.csv",
    "frequency_report.csv",
    "timing_report.txt",
)

FREQUENCY_PERCENTILES = (10, 25, 50, 75, 90)


def write_results_configs(output_dir: Path, population: pd.DataFrame) -> Path:
    path = Path(output_dir) / "results_configs.csv"
    population.to_csv(path, index=False)
    return path


def write_funnel_counts(output_dir: Path, funnel_result: dict, es_counts: dict | None = None) -> Path:
    path = Path(output_dir) / "funnel_counts.json"
    payload = {"stage_counts": funnel_result["counts"], "bh_method": funnel_result["bh_method"]}
    if es_counts is not None:
        payload["es_validation"] = es_counts
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def write_named_configs_report(output_dir: Path, rows: list[dict]) -> Path:
    """rows: one dict per (named config, symbol) with config_name, symbol,
    trades, win_rate, avg_r, net_sharpe, and net_pnl."""
    path = Path(output_dir) / "named_configs_report.md"
    lines = [
        "# Named configs report",
        "",
        "Win rate is reported next to the commonly-claimed 70-80% Silver Bullet band for direct comparison.",
        "",
        "| config | symbol | trades | win rate | vs. 70-80% claim | avg R | net Sharpe | net PnL |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        note = placeholder_note(row["config_name"])
        name = f"{row['config_name']} ({note})" if note else row["config_name"]
        win_rate = row["win_rate"]
        if pd.isna(win_rate):
            vs_claim = "n/a"
            win_rate_str = "n/a"
        else:
            vs_claim = "within band" if 70.0 <= win_rate <= 80.0 else ("below band" if win_rate < 70.0 else "above band")
            win_rate_str = f"{win_rate:.1f}%"
        lines.append(
            f"| {name} | {row['symbol']} | {row['trades']} | {win_rate_str} | {vs_claim} | "
            f"{row['avg_r']:.2f} | {row['net_sharpe']:.2f} | {row['net_pnl']:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_es_validation(output_dir: Path, es_validation_df: pd.DataFrame) -> Path:
    path = Path(output_dir) / "es_validation.csv"
    es_validation_df.to_csv(path, index=False)
    return path


def write_ablation_ladder(output_dir: Path, ablation_df: pd.DataFrame) -> Path:
    path = Path(output_dir) / "ablation_ladder.csv"
    ablation_df.to_csv(path, index=False)
    return path


def write_null_summary(output_dir: Path, null_results: dict[str, dict]) -> Path:
    path = Path(output_dir) / "null_summary.md"
    lines = ["# Null test summary", ""]
    for config_name, entry in null_results.items():
        lines.append(f"## {config_name}")
        lines.append(f"- real net Sharpe: {entry['real_net_sharpe']:.2f}")
        for null_name, null_entry in entry.items():
            if null_name == "real_net_sharpe":
                continue
            pct = null_entry["percentile"]
            pct_str = f"{pct:.1f}th percentile" if pd.notna(pct) else "n/a"
            flag = " -- REAL RESULT SITS INSIDE THIS NULL DISTRIBUTION" if pd.notna(pct) and 5.0 <= pct <= 95.0 else ""
            lines.append(f"- {null_name}: real result at the {pct_str} of the null distribution{flag}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_discretion_premium(output_dir: Path, reports: dict[str, dict]) -> Path:
    path = Path(output_dir) / "discretion_premium.txt"
    lines = []
    for report in reports.values():
        lines.extend(format_discretion_premium(report))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_era_split(output_dir: Path, era_results: dict[str, dict]) -> Path:
    path = Path(output_dir) / "era_split.csv"
    rows = []
    for config_name, eras in era_results.items():
        for era_name, stats in eras.items():
            rows.append({"config_name": config_name, "era": era_name, **stats})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_frequency_report(output_dir: Path, population: pd.DataFrame, named_rows: list[dict]) -> Path:
    """Trades/year and day-coverage percentiles across the population,
    plus the named configs' own values for comparison."""
    path = Path(output_dir) / "frequency_report.csv"
    rows = []
    for p in FREQUENCY_PERCENTILES:
        rows.append(
            {
                "row": f"population_p{p}",
                "trades_per_year": population["trades_per_year"].quantile(p / 100) if not population.empty else float("nan"),
                "pct_days_with_trade": population["pct_days_with_trade"].quantile(p / 100) if not population.empty else float("nan"),
            }
        )
    for row in named_rows:
        rows.append(
            {
                "row": row["config_name"],
                "trades_per_year": row["trades_per_year"],
                "pct_days_with_trade": row["pct_days_with_trade"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_timing_report(output_dir: Path, timings: dict[str, float]) -> Path:
    path = Path(output_dir) / "timing_report.txt"
    lines = [f"{stage}: {seconds:.1f}s" for stage, seconds in timings.items()]
    total = sum(timings.values())
    lines.append(f"total: {total:.1f}s")
    path.write_text("\n".join(lines) + "\n")
    return path
