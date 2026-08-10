from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ict_lab.data import sessions as sessions_module
from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.null_models import (
    N_ITERATIONS,
    _sharpe_from_daily,
    _simulate_null_trade_net_pnl,
    _temporary_window,
    other_hour_windows,
    other_hours_null_distribution,
    percentile_of_real_result,
    random_entry_null_distribution,
    run_null_tests,
    select_reference_configs,
    shuffled_direction_null_distribution,
)
from ict_lab.engine.pipeline import run_config
from ict_lab.engine.sweep_runner import annualized_sharpe, stats_from_trades

D1 = pd.Timestamp("2020-06-01")
D2 = pd.Timestamp("2020-06-02")


def _flat_bars(start="2024-06-03 00:00:00", end="2024-06-10 00:00:00"):
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    return pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10, "contract": "X"}, index=idx
    )


def _config(**overrides):
    kwargs = dict(name="t", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    kwargs.update(overrides)
    return StrategyConfig(**kwargs)


# ---------- percentile_of_real_result ----------


def test_percentile_of_real_result_basic():
    dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile_of_real_result(3.0, dist) == pytest.approx(60.0)  # 3 of 5 values <= 3.0


def test_percentile_of_real_result_real_value_nan():
    assert pd.isna(percentile_of_real_result(float("nan"), np.array([1.0, 2.0])))


def test_percentile_of_real_result_all_nan_distribution():
    assert pd.isna(percentile_of_real_result(1.0, np.array([float("nan"), float("nan")])))


def test_percentile_of_real_result_drops_nan_before_computing():
    dist = np.array([1.0, 2.0, float("nan"), 3.0])
    assert percentile_of_real_result(2.0, dist) == pytest.approx(2 / 3 * 100)


# ---------- _sharpe_from_daily ----------


def test_sharpe_from_daily_matches_annualized_sharpe_after_reindex():
    values = np.array([100.0])
    dates = [D1]
    all_days = pd.DatetimeIndex([D1, D2])
    expected = annualized_sharpe(pd.Series([100.0, 0.0]))
    assert _sharpe_from_daily(values, dates, all_days) == pytest.approx(expected)


def test_sharpe_from_daily_empty_is_nan():
    assert pd.isna(_sharpe_from_daily(np.array([]), [], pd.DatetimeIndex([D1])))


# ---------- _simulate_null_trade_net_pnl ----------


def test_simulate_null_trade_net_pnl_bullish_hits_target():
    idx = pd.date_range("2024-06-03 14:00:00", periods=5, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.5, "high": 100.5, "low": 100.5, "close": 100.5, "volume": 10, "contract": "X"}, index=idx
    )
    bars.loc[idx[2], "high"] = 106.0  # target hit at bar 2
    net = _simulate_null_trade_net_pnl(
        entry_at=idx[0], entry_price=100.5, stop_distance=1.5, target_distance=3.0, direction="bullish",
        session_bars=bars, hard_exit_at=idx[-1], tick_size=0.25, tick_value=5.0, commission=4.0,
        stop_slippage_ticks=1.0,
    )
    # target = 100.5+3.0=103.5, hit exactly -> price_diff=3.0 -> gross=3.0/0.25*5=60 -> net=56
    assert net == pytest.approx(56.0)


def test_simulate_null_trade_net_pnl_bearish_hits_stop():
    idx = pd.date_range("2024-06-03 14:00:00", periods=5, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.5, "high": 100.5, "low": 100.5, "close": 100.5, "volume": 10, "contract": "X"}, index=idx
    )
    bars.loc[idx[1], "high"] = 103.0  # stop (at 102.0) hit at bar 1
    net = _simulate_null_trade_net_pnl(
        entry_at=idx[0], entry_price=100.5, stop_distance=1.5, target_distance=None, direction="bearish",
        session_bars=bars, hard_exit_at=idx[-1], tick_size=0.25, tick_value=5.0, commission=4.0,
        stop_slippage_ticks=1.0,
    )
    # stop = 100.5+1.5=102.0, exit with 1 tick slippage -> exit_price=102.25
    # price_diff (bearish) = entry - exit = 100.5 - 102.25 = -1.75 -> gross=-1.75/0.25*5=-35 -> net=-39
    assert net == pytest.approx(-39.0)


# ---------- other_hour_windows ----------


def test_other_hour_windows_no_exclusions_gives_23():
    assert len(other_hour_windows(())) == 23


def test_other_hour_windows_excludes_the_named_killzones():
    windows = other_hour_windows(("killzone_london", "killzone_ny_am", "killzone_ny_pm"))
    assert len(windows) == 20
    starts = {start for _, start, _ in windows}
    assert "03:00" not in starts  # killzone_london
    assert "10:00" not in starts  # killzone_ny_am
    assert "14:00" not in starts  # killzone_ny_pm


def test_other_hour_windows_names_are_unique():
    windows = other_hour_windows(())
    names = [n for n, _, _ in windows]
    assert len(names) == len(set(names))


def test_other_hour_windows_cover_the_session_without_gaps_or_overlap():
    windows = other_hour_windows(())
    hours = sorted(int(start.split(":")[0]) for _, start, _ in windows)
    expected = sorted((18 + i) % 24 for i in range(23))
    assert hours == expected


# ---------- _temporary_window ----------


def test_temporary_window_adds_then_restores():
    before = dict(sessions_module.WINDOWS)
    with _temporary_window("null_test_window", "05:00", "06:00"):
        assert sessions_module.WINDOWS["null_test_window"] == ("05:00", "06:00")
    assert sessions_module.WINDOWS == before
    assert "null_test_window" not in sessions_module.WINDOWS


def test_temporary_window_restores_even_on_exception():
    before = dict(sessions_module.WINDOWS)
    with pytest.raises(ValueError):
        with _temporary_window("null_test_window2", "05:00", "06:00"):
            raise ValueError("boom")
    assert sessions_module.WINDOWS == before


# ---------- other_hours_null_distribution ----------


def test_other_hours_null_distribution_length_matches_other_hour_windows(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    config = _config(windows=("killzone_ny_am",))
    result = other_hours_null_distribution(df, config, "NQ")
    assert len(result) == len(other_hour_windows(config.windows))


def test_other_hours_null_distribution_does_not_leak_windows_state(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-04 00:00:00")
    before = dict(sessions_module.WINDOWS)
    other_hours_null_distribution(df, _config(), "NQ")
    assert sessions_module.WINDOWS == before


# ---------- random_entry_null_distribution / shuffled_direction_null_distribution ----------


def _real_trades(df, config, symbol="NQ"):
    _, _, trades, _ = run_config(df, config, symbol)
    return trades


def test_random_entry_null_distribution_empty_trades_is_all_nan():
    result = random_entry_null_distribution(_flat_bars(), _config(), "NQ", pd.DataFrame(), n_iterations=10)
    assert len(result) == 10
    assert np.isnan(result).all()


def test_random_entry_null_distribution_deterministic_with_fixed_seed(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")
    config = _config(entry_level="50%", stop_type="gap_distal", displacement_required=False, mss_required=False)
    trades = _real_trades(df, config)
    if trades.empty:
        pytest.skip("no real trades generated on this synthetic window -- nothing to build a null for")
    a = random_entry_null_distribution(df, config, "NQ", trades, n_iterations=15, seed=5)
    b = random_entry_null_distribution(df, config, "NQ", trades, n_iterations=15, seed=5)
    np.testing.assert_array_equal(a, b)


def test_random_entry_null_distribution_return_paths_false_is_unchanged(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")
    config = _config(entry_level="50%", stop_type="gap_distal", displacement_required=False, mss_required=False)
    trades = _real_trades(df, config)
    if trades.empty:
        pytest.skip("no real trades generated on this synthetic window -- nothing to build a null for")
    without = random_entry_null_distribution(df, config, "NQ", trades, n_iterations=10, seed=7)
    with_ = random_entry_null_distribution(df, config, "NQ", trades, n_iterations=10, seed=7, return_paths=False)
    np.testing.assert_array_equal(without, with_)


def test_random_entry_null_distribution_return_paths_shape_and_consistency(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")
    config = _config(entry_level="50%", stop_type="gap_distal", displacement_required=False, mss_required=False)
    trades = _real_trades(df, config)
    if trades.empty:
        pytest.skip("no real trades generated on this synthetic window -- nothing to build a null for")
    from ict_lab.engine.pipeline import all_session_dates

    n_days = len(all_session_dates(df))
    sharpes, paths = random_entry_null_distribution(df, config, "NQ", trades, n_iterations=10, seed=7, return_paths=True)
    assert paths.shape == (10, n_days)
    # The path's own final cumulative value implies the same daily series
    # annualized_sharpe was computed from -- not an independent recomputation,
    # a direct algebraic identity of what the function already built.
    sharpe_only = random_entry_null_distribution(df, config, "NQ", trades, n_iterations=10, seed=7)
    np.testing.assert_array_equal(sharpes, sharpe_only)


def test_random_entry_null_distribution_return_paths_empty_trades(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    from ict_lab.engine.pipeline import all_session_dates

    n_days = len(all_session_dates(df))
    sharpes, paths = random_entry_null_distribution(
        df, _config(), "NQ", pd.DataFrame(), n_iterations=6, return_paths=True
    )
    assert len(sharpes) == 6
    assert np.isnan(sharpes).all()
    assert paths.shape == (6, n_days)
    assert np.isnan(paths).all()


def test_shuffled_direction_null_distribution_empty_trades_is_all_nan():
    result = shuffled_direction_null_distribution(_flat_bars(), _config(), "NQ", pd.DataFrame(), n_iterations=10)
    assert len(result) == 10
    assert np.isnan(result).all()


def test_shuffled_direction_null_distribution_deterministic_with_fixed_seed(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")
    config = _config(entry_level="50%", stop_type="gap_distal", displacement_required=False, mss_required=False)
    trades = _real_trades(df, config)
    if trades.empty:
        pytest.skip("no real trades generated on this synthetic window -- nothing to build a null for")
    a = shuffled_direction_null_distribution(df, config, "NQ", trades, n_iterations=15, seed=5)
    b = shuffled_direction_null_distribution(df, config, "NQ", trades, n_iterations=15, seed=5)
    np.testing.assert_array_equal(a, b)


# ---------- select_reference_configs ----------


def test_select_reference_configs_always_includes_named():
    named = {"as_taught_5m": _config(name="as_taught_5m")}
    refs = select_reference_configs(pd.DataFrame(), pd.DataFrame(), named)
    assert refs == named


def test_select_reference_configs_picks_best_survivor_and_median_config():
    configs = [
        _config(name="low", entry_level="proximal"),
        _config(name="mid", entry_level="50%"),
        _config(name="high", entry_level="distal"),
    ]
    rows = [stats_from_trades(c, pd.DataFrame(columns=["session_date"]), pd.DatetimeIndex([D1])) for c in configs]
    for row, sharpe in zip(rows, (0.5, 1.0, 3.0)):
        row["net_sharpe"] = sharpe
    population = pd.DataFrame(rows)
    survivors = population[population["config_name"].isin(["low", "high"])]

    refs = select_reference_configs(population, survivors, {})
    assert refs["best_realistic_survivor"].name == survivors.loc[survivors["net_sharpe"].idxmax(), "config_name"]
    assert refs["median_sharpe_config"].name == "mid"  # 1.0 is the population median of [0.5, 1.0, 3.0]


def test_select_reference_configs_handles_empty_survivors_and_population():
    refs = select_reference_configs(pd.DataFrame(), pd.DataFrame(), {"x": _config(name="x")})
    assert "best_realistic_survivor" not in refs
    assert "median_sharpe_config" not in refs
    assert refs["x"].name == "x"


# ---------- run_null_tests ----------


def test_run_null_tests_structure(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-06 00:00:00")
    refs = {"t": _config(entry_level="50%", stop_type="gap_distal", displacement_required=False, mss_required=False)}
    results = run_null_tests(df, "NQ", refs, n_iterations=10)

    assert set(results) == {"t"}
    entry = results["t"]
    assert "real_net_sharpe" in entry
    for key in ("random_entry_same_windows", "other_hours_same_logic", "shuffled_direction"):
        assert key in entry
        assert "distribution" in entry[key]
        assert "percentile" in entry[key]
        pct = entry[key]["percentile"]
        assert pd.isna(pct) or 0.0 <= pct <= 100.0


def test_run_null_tests_excludes_shuffled_when_disabled(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    refs = {"t": _config()}
    results = run_null_tests(df, "NQ", refs, n_iterations=5, include_shuffled=False)
    assert "shuffled_direction" not in results["t"]
    assert "random_entry_same_windows" in results["t"]
