from __future__ import annotations

import json

import pandas as pd

from ict_lab.engine.summary_pack import (
    SUMMARY_FILES,
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


def test_write_results_configs(tmp_path):
    population = pd.DataFrame([{"config_hash": "a", "net_sharpe": 1.0}])
    path = write_results_configs(tmp_path, population)
    assert path.name == "results_configs.csv"
    assert pd.read_csv(path)["config_hash"].tolist() == ["a"]


def test_write_funnel_counts(tmp_path):
    funnel_result = {"counts": {"population": 100, "min_100_trades": 40}, "bh_method": "bootstrap"}
    path = write_funnel_counts(tmp_path, funnel_result, es_counts={"survivors_tested": 3, "cross_symbol_survivors": 1})
    payload = json.loads(path.read_text())
    assert payload["stage_counts"]["population"] == 100
    assert payload["bh_method"] == "bootstrap"
    assert payload["es_validation"]["cross_symbol_survivors"] == 1


def test_write_named_configs_report_flags_placeholder_and_win_rate_band(tmp_path):
    rows = [
        {"config_name": "as_taught_5m", "symbol": "NQ", "trades": 50, "win_rate": 75.0, "avg_r": 0.3, "net_sharpe": 1.2, "net_pnl": 1000.0},
        {"config_name": "as_traded", "symbol": "NQ", "trades": 40, "win_rate": 60.0, "avg_r": 0.1, "net_sharpe": 0.5, "net_pnl": 200.0},
    ]
    path = write_named_configs_report(tmp_path, rows)
    text = path.read_text()
    assert "as_traded (default placeholder, not yet user-specified)" in text
    assert "within band" in text  # 75% is within 70-80%
    assert "below band" in text  # 60% is below 70-80%


def test_write_named_configs_report_handles_nan_win_rate(tmp_path):
    rows = [{"config_name": "x", "symbol": "NQ", "trades": 0, "win_rate": float("nan"), "avg_r": float("nan"), "net_sharpe": float("nan"), "net_pnl": 0.0}]
    path = write_named_configs_report(tmp_path, rows)
    assert "n/a" in path.read_text()


def test_write_es_validation(tmp_path):
    df = pd.DataFrame([{"config_name": "x", "cross_symbol_survivor": True}])
    path = write_es_validation(tmp_path, df)
    assert pd.read_csv(path)["cross_symbol_survivor"].iloc[0] == True  # noqa: E712


def test_write_ablation_ladder(tmp_path):
    df = pd.DataFrame([{"rung": "1_fvg_entry_only", "symbol": "NQ", "trades": 10}])
    path = write_ablation_ladder(tmp_path, df)
    assert pd.read_csv(path)["rung"].iloc[0] == "1_fvg_entry_only"


def test_write_null_summary_flags_result_inside_null_distribution(tmp_path):
    null_results = {
        "as_taught_5m": {
            "real_net_sharpe": 1.0,
            "random_entry_same_windows": {"distribution": [0.0], "percentile": 50.0},
            "other_hours_same_logic": {"distribution": [0.0], "percentile": 99.0},
        }
    }
    path = write_null_summary(tmp_path, null_results)
    text = path.read_text()
    assert "REAL RESULT SITS INSIDE THIS NULL DISTRIBUTION" in text  # 50th percentile -> inside
    assert text.count("REAL RESULT SITS INSIDE") == 1  # 99th percentile -> not flagged


def test_write_discretion_premium(tmp_path):
    reports = {
        "x": {
            "config_name": "x",
            "every_setup": {"trades": 10, "net_pnl": 100.0, "net_sharpe": 1.0},
            "hindsight_perfect": {"trades": 6, "net_pnl": 500.0, "net_sharpe": 3.0},
            "minimum_skip": {"breakeven_fraction": 0.2, "sharpe_fraction": None, "n_losers": 4},
        }
    }
    path = write_discretion_premium(tmp_path, reports)
    text = path.read_text()
    assert "x: every setup" in text
    assert "not achievable" in text


def test_write_era_split(tmp_path):
    era_results = {
        "x": {"2010_2021": {"trades": 5, "win_rate": 60.0, "avg_r": 0.5, "net_sharpe": 1.0}, "2022_to_holdout_cutoff": {"trades": 3, "win_rate": 66.0, "avg_r": 0.6, "net_sharpe": 1.1}},
    }
    path = write_era_split(tmp_path, era_results)
    df = pd.read_csv(path)
    assert len(df) == 2
    assert set(df["era"]) == {"2010_2021", "2022_to_holdout_cutoff"}


def test_write_frequency_report_includes_percentiles_and_named_rows(tmp_path):
    population = pd.DataFrame({"trades_per_year": [10, 20, 30, 40, 50], "pct_days_with_trade": [1, 2, 3, 4, 5]})
    named_rows = [{"config_name": "as_taught_5m", "trades_per_year": 25.0, "pct_days_with_trade": 12.0}]
    path = write_frequency_report(tmp_path, population, named_rows)
    df = pd.read_csv(path)
    assert "population_p50" in set(df["row"])
    assert "as_taught_5m" in set(df["row"])


def test_write_timing_report_includes_total(tmp_path):
    path = write_timing_report(tmp_path, {"sweep": 100.0, "funnel": 20.0})
    text = path.read_text()
    assert "sweep: 100.0s" in text
    assert "total: 120.0s" in text


def test_summary_files_constant_has_ten_entries():
    assert len(SUMMARY_FILES) == 10
