from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from ict_lab.engine.statistics_slices import (
    EULER_MASCHERONI,
    _apply_slippage_delta,
    _light_summary,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_for_config,
    era_split,
    per_year_table,
    roll_day_sensitivity,
    sharpe_estimator_std,
    slippage_sensitivity,
    sr_0,
)
from ict_lab.engine.sweep_runner import _flush_r_multiples_shard
import json

D1 = pd.Timestamp("2020-06-01")
D2020 = pd.Timestamp("2020-06-01")
D2022 = pd.Timestamp("2022-06-01")


def _trade(session_date, r_multiple, net_pnl, is_roll_day=False):
    return {"session_date": session_date, "r_multiple": r_multiple, "net_pnl": net_pnl, "is_roll_day": is_roll_day}


def _slippage_trade(direction, entry_price, stop_price, exit_price, exit_reason):
    return {
        "direction": direction, "entry_price": entry_price, "stop_price": stop_price, "exit_price": exit_price,
        "exit_reason": exit_reason, "session_date": D1, "gross_pnl": 0.0, "net_pnl": 0.0, "r_multiple": 0.0,
    }


# ---------- sr_0 ----------


def test_sr_0_matches_manual_evt_formula():
    n_trials, std = 1000, 0.5
    z1 = scipy_stats.norm.ppf(1 - 1 / n_trials)
    z2 = scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
    expected = std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    assert sr_0(n_trials, std) == pytest.approx(expected)


def test_sr_0_single_trial_is_nan():
    assert np.isnan(sr_0(1, 0.5))


def test_sr_0_nan_std_is_nan():
    assert np.isnan(sr_0(100, float("nan")))


def test_sr_0_increases_with_more_trials():
    # More trials -> higher expected max Sharpe under the null, for fixed spread.
    assert sr_0(10000, 0.5) > sr_0(100, 0.5)


# ---------- sharpe_estimator_std ----------


def test_sharpe_estimator_std_normal_returns_matches_known_asymptotic_formula():
    # skew=0, kurtosis=3 (normal) -> sqrt((1 + 0.5*SR^2) / (T-1))
    sharpe, T = 1.2, 101
    expected = np.sqrt((1 + 0.5 * sharpe**2) / (T - 1))
    assert sharpe_estimator_std(sharpe, skew=0.0, kurtosis=3.0, n_observations=T) == pytest.approx(expected)


def test_sharpe_estimator_std_single_observation_is_nan():
    assert np.isnan(sharpe_estimator_std(1.0, 0.0, 3.0, 1))


def test_sharpe_estimator_std_never_negative_under_extreme_inputs():
    # Even with pathological skew/kurtosis, the variance is clamped at 0.
    result = sharpe_estimator_std(sharpe=5.0, skew=10.0, kurtosis=1.0, n_observations=10)
    assert result >= 0.0


# ---------- deflated_sharpe_ratio ----------


def test_deflated_sharpe_ratio_high_sharpe_beats_noise_floor():
    result = deflated_sharpe_ratio(
        observed_sharpe=3.0, n_trials=100, cross_config_sharpe_std=0.3, trade_skew=0.0, trade_kurtosis=3.0, n_trades=500,
    )
    assert result["dsr"] > 0.99


def test_deflated_sharpe_ratio_sharpe_at_noise_floor_is_middling():
    sr0 = sr_0(100, 0.3)
    result = deflated_sharpe_ratio(
        observed_sharpe=sr0, n_trials=100, cross_config_sharpe_std=0.3, trade_skew=0.0, trade_kurtosis=3.0, n_trades=500,
    )
    assert result["dsr"] == pytest.approx(0.5, abs=0.01)  # Z-score ~0 -> CDF ~0.5


def test_deflated_sharpe_ratio_zero_sigma_is_nan():
    result = deflated_sharpe_ratio(1.0, 100, 0.3, 0.0, 3.0, n_trades=1)  # n_trades=1 -> sigma_sr NaN
    assert np.isnan(result["dsr"])


# ---------- deflated_sharpe_ratio_for_config (integration) ----------


def test_deflated_sharpe_ratio_for_config_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    hashes = []
    for i in range(20):
        h = f"h{i}"
        hashes.append(h)
        r = list(rng.normal(0.0, 1.0, 60))  # mediocre, noisy configs
        _flush_r_multiples_shard(tmp_path, [{"config_hash": h, "r_multiples": json.dumps(r)}])

    best_hash = "best"
    best_r = list(rng.normal(1.0, 0.5, 60))  # obviously strong: high positive mean, real variance
    _flush_r_multiples_shard(tmp_path, [{"config_hash": best_hash, "r_multiples": json.dumps(best_r)}])
    hashes.append(best_hash)

    population = pd.DataFrame({"config_hash": hashes})
    result = deflated_sharpe_ratio_for_config(population, best_hash, tmp_path)

    assert result["n_trials"] == 21
    assert result["n_trades"] == 60
    assert not np.isnan(result["observed_sharpe"])
    assert 0.0 <= result["dsr"] <= 1.0


# ---------- per_year_table ----------


def test_per_year_table_groups_by_year():
    trades = pd.DataFrame(
        [
            _trade(pd.Timestamp("2020-03-01"), 2.0, 100.0),
            _trade(pd.Timestamp("2020-08-01"), -1.0, -50.0),
            _trade(pd.Timestamp("2021-01-01"), 1.0, 50.0),
        ]
    )
    table = per_year_table(trades)
    row_2020 = table[table["year"] == 2020].iloc[0]
    row_2021 = table[table["year"] == 2021].iloc[0]
    assert row_2020["trades"] == 2
    assert row_2020["avg_r"] == pytest.approx(0.5)  # (2.0-1.0)/2
    assert row_2021["trades"] == 1
    assert row_2021["avg_r"] == pytest.approx(1.0)


def test_per_year_table_empty_trades():
    table = per_year_table(pd.DataFrame(columns=["session_date", "r_multiple"]))
    assert table.empty
    assert list(table.columns) == ["year", "avg_r", "trades"]


# ---------- era_split ----------


def test_era_split_separates_pre_and_post_2022():
    trades = pd.DataFrame(
        [_trade(D2020, 2.0, 100.0), _trade(pd.Timestamp("2021-12-31"), -1.0, -50.0), _trade(D2022, 1.0, 50.0)]
    )
    session_dates = pd.DatetimeIndex([D2020, pd.Timestamp("2021-12-31"), D2022])
    result = era_split(trades, session_dates)
    assert result["2010_2021"]["trades"] == 2
    assert result["2022_to_holdout_cutoff"]["trades"] == 1


def test_era_split_empty_trades():
    result = era_split(pd.DataFrame(columns=["session_date", "r_multiple", "net_pnl"]), pd.DatetimeIndex([D2020]))
    assert result["2010_2021"]["trades"] == 0
    assert result["2022_to_holdout_cutoff"]["trades"] == 0


# ---------- roll_day_sensitivity ----------


def test_roll_day_sensitivity_excludes_roll_days_in_without_variant():
    trades = pd.DataFrame(
        [_trade(D1, 2.0, 100.0, is_roll_day=False), _trade(D1, -5.0, -500.0, is_roll_day=True)]
    )
    result = roll_day_sensitivity(trades, pd.DatetimeIndex([D1]))
    assert result["with_roll_days"]["trades"] == 2
    assert result["without_roll_days"]["trades"] == 1
    assert result["without_roll_days"]["avg_r"] == pytest.approx(2.0)


# ---------- _apply_slippage_delta ----------


def test_apply_slippage_delta_stop_bullish_matches_manual_calc():
    trades = pd.DataFrame([_slippage_trade("bullish", 100.0, 98.0, 97.75, "stop")])  # baseline 1 tick @ 0.25
    adjusted = _apply_slippage_delta(trades, "stop", new_ticks=3, baseline_ticks=1, tick_size=0.25, tick_value=5.0, commission=4.0)
    row = adjusted.iloc[0]
    assert row["exit_price"] == pytest.approx(97.25)
    assert row["r_multiple"] == pytest.approx(-1.375)
    assert row["net_pnl"] == pytest.approx(-59.0)


def test_apply_slippage_delta_stop_bearish_matches_manual_calc():
    trades = pd.DataFrame([_slippage_trade("bearish", 100.0, 102.0, 102.25, "stop")])
    adjusted = _apply_slippage_delta(trades, "stop", new_ticks=3, baseline_ticks=1, tick_size=0.25, tick_value=5.0, commission=4.0)
    row = adjusted.iloc[0]
    assert row["exit_price"] == pytest.approx(102.75)
    assert row["r_multiple"] == pytest.approx(-1.375)
    assert row["net_pnl"] == pytest.approx(-59.0)


def test_apply_slippage_delta_same_ticks_is_noop():
    trades = pd.DataFrame([_slippage_trade("bullish", 100.0, 98.0, 97.75, "stop")])
    adjusted = _apply_slippage_delta(trades, "stop", new_ticks=1, baseline_ticks=1, tick_size=0.25, tick_value=5.0, commission=4.0)
    assert adjusted.iloc[0]["exit_price"] == pytest.approx(97.75)


def test_apply_slippage_delta_only_touches_matching_exit_reason():
    trades = pd.DataFrame(
        [_slippage_trade("bullish", 100.0, 98.0, 97.75, "stop"), _slippage_trade("bullish", 100.0, 98.0, 104.0, "target")]
    )
    adjusted = _apply_slippage_delta(trades, "stop", new_ticks=3, baseline_ticks=1, tick_size=0.25, tick_value=5.0, commission=4.0)
    target_row = adjusted[adjusted["exit_reason"] == "target"].iloc[0]
    assert target_row["exit_price"] == pytest.approx(104.0)  # untouched


def test_apply_slippage_delta_empty_trades():
    empty = pd.DataFrame(columns=["direction", "entry_price", "stop_price", "exit_price", "exit_reason", "session_date", "gross_pnl", "net_pnl", "r_multiple"])
    result = _apply_slippage_delta(empty, "stop", 3, 1, 0.25, 5.0, 4.0)
    assert result.empty


# ---------- slippage_sensitivity ----------


def test_slippage_sensitivity_covers_all_documented_tick_options():
    trades = pd.DataFrame(
        [
            _slippage_trade("bullish", 100.0, 98.0, 97.75, "stop"),
            _slippage_trade("bullish", 100.0, 98.0, 100.5, "target_time"),
        ]
    )
    result = slippage_sensitivity(trades, pd.DatetimeIndex([D1]), "NQ")
    assert set(result["stop_slippage_ticks"]) == {0, 1, 2, 3}
    assert set(result["time_exit_slippage_ticks"]) == {0, 1, 2}
