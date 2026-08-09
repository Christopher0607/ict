from __future__ import annotations

import json

import pandas as pd
import pytest

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.configs.sweep_grid import canonical_hash
from ict_lab.engine.sweep_runner import (
    R_MULTIPLES_COLUMNS,
    SUMMARY_COLUMNS,
    annualized_sharpe,
    _max_drawdown,
    _flush_r_multiples_shard,
    _flush_shard,
    config_from_row,
    load_r_multiples_shards,
    load_shards,
    r_multiples_for_hash,
    run_sweep,
    stats_from_trades,
    summarize_config,
)

D1 = pd.Timestamp("2020-06-01")
D2 = pd.Timestamp("2020-06-02")
D3 = pd.Timestamp("2020-06-03")


def _config(**overrides):
    kwargs = dict(name="t", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    kwargs.update(overrides)
    return StrategyConfig(**kwargs)


def _trade_row(session_date, exit_at, r_multiple, gross_pnl, net_pnl, ambiguous_bar=False, is_roll_day=False):
    return {
        "session_date": session_date, "exit_at": exit_at, "r_multiple": r_multiple,
        "gross_pnl": gross_pnl, "net_pnl": net_pnl, "ambiguous_bar": ambiguous_bar, "is_roll_day": is_roll_day,
    }


def _empty_trades():
    return pd.DataFrame(columns=["session_date", "exit_at", "r_multiple", "gross_pnl", "net_pnl", "ambiguous_bar", "is_roll_day"])


# ---------- annualized_sharpe ----------


def test_annualized_sharpe_matches_independent_manual_calculation():
    values = [10.0, -5.0, 20.0, 0.0, -3.0]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    expected = mean / (variance ** 0.5) * (252 ** 0.5)
    assert annualized_sharpe(pd.Series(values)) == pytest.approx(expected)


def test_annualized_sharpe_zero_std_is_nan():
    assert pd.isna(annualized_sharpe(pd.Series([5.0, 5.0, 5.0])))


def test_annualized_sharpe_single_value_is_nan():
    assert pd.isna(annualized_sharpe(pd.Series([5.0])))


# ---------- _max_drawdown ----------


def test_max_drawdown_matches_manual_peak_to_trough():
    cum = pd.Series([10.0, 15.0, 5.0, 8.0, 20.0, 12.0])
    # running peak: 10,15,15,15,20,20 ; decline: 0,0,10,7,0,8 -> worst is 10
    assert _max_drawdown(cum) == pytest.approx(10.0)


def test_max_drawdown_monotonic_increase_is_zero():
    assert _max_drawdown(pd.Series([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_max_drawdown_empty_is_zero():
    assert _max_drawdown(pd.Series([], dtype=float)) == 0.0


# ---------- stats_from_trades ----------


def test_stats_from_trades_basic_aggregates():
    trades = pd.DataFrame(
        [
            _trade_row(D1, D1, 2.0, 100.0, 96.0, ambiguous_bar=False, is_roll_day=False),
            _trade_row(D2, D2, -1.0, -50.0, -54.0, ambiguous_bar=True, is_roll_day=False),
            _trade_row(D3, D3, 1.0, 50.0, 46.0, ambiguous_bar=False, is_roll_day=True),
        ]
    )
    row = stats_from_trades(_config(), trades, pd.DatetimeIndex([D1, D2, D3]))

    assert row["trade_count"] == 3
    assert row["win_rate"] == pytest.approx(200 / 3)  # 2 of 3 trades have r_multiple > 0
    assert row["avg_r"] == pytest.approx((2.0 - 1.0 + 1.0) / 3)
    assert row["total_r"] == pytest.approx(2.0)
    assert row["gross_pnl"] == pytest.approx(100.0)
    assert row["net_pnl"] == pytest.approx(88.0)
    assert row["profit_factor"] == pytest.approx(150.0 / 50.0)  # gains 150, losses 50
    assert row["pct_ambiguous_bar"] == pytest.approx(100 / 3)
    assert row["pct_roll_day"] == pytest.approx(100 / 3)
    assert row["first_trade_date"] == D1
    assert row["last_trade_date"] == D3
    assert row["pct_days_with_trade"] == pytest.approx(100.0)
    assert row["trades_per_year"] == pytest.approx(3.0)  # all 3 dates in the same year


def test_stats_from_trades_zero_trades_returns_nan_stats_not_crash():
    row = stats_from_trades(_config(), _empty_trades(), pd.DatetimeIndex([D1, D2]))
    assert row["trade_count"] == 0
    assert row["trades_per_year"] == 0
    assert row["pct_days_with_trade"] == 0
    assert row["gross_pnl"] == 0.0
    assert row["net_pnl"] == 0.0
    assert pd.isna(row["win_rate"])
    assert pd.isna(row["gross_sharpe"])
    assert pd.isna(row["net_sharpe"])
    assert pd.isna(row["first_trade_date"])


def test_stats_from_trades_sharpe_treats_no_trade_days_as_zero():
    # Only D1 has a trade; D2 and D3 are genuine no-trade days that must
    # still enter the daily pnl series as zero, per the project-wide
    # Sharpe convention.
    trades = pd.DataFrame([_trade_row(D1, D1, 1.0, 100.0, 100.0)])
    row = stats_from_trades(_config(), trades, pd.DatetimeIndex([D1, D2, D3]))

    values = [100.0, 0.0, 0.0]
    mean = sum(values) / 3
    variance = sum((v - mean) ** 2 for v in values) / (3 - 1)
    expected = mean / (variance ** 0.5) * (252 ** 0.5)
    assert row["gross_sharpe"] == pytest.approx(expected)
    assert row["pct_days_with_trade"] == pytest.approx(100 / 3)


def test_stats_from_trades_max_drawdown_r_and_dollars():
    trades = pd.DataFrame(
        [
            _trade_row(D1, D1, 2.0, 200.0, 196.0),
            _trade_row(D2, D2, -3.0, -300.0, -304.0),
            _trade_row(D3, D3, 1.0, 100.0, 96.0),
        ]
    )
    row = stats_from_trades(_config(), trades, pd.DatetimeIndex([D1, D2, D3]))
    # cumsum R: 2, -1, 0 -> peak 2,2,2 -> decline 0,3,2 -> max 3
    assert row["max_drawdown_r"] == pytest.approx(3.0)
    # cumsum net_pnl: 196, -108, -12 -> peak 196,196,196 -> decline 0,304,208 -> max 304
    assert row["max_drawdown_dollars"] == pytest.approx(304.0)


def test_stats_from_trades_profit_factor_all_wins_is_inf():
    trades = pd.DataFrame([_trade_row(D1, D1, 1.0, 100.0, 96.0)])
    row = stats_from_trades(_config(), trades, pd.DatetimeIndex([D1]))
    assert row["profit_factor"] == float("inf")


def test_stats_from_trades_trades_per_year_uses_full_data_span_not_active_years():
    y2020 = pd.Timestamp("2020-01-06")
    y2021 = pd.Timestamp("2021-01-06")
    y2022 = pd.Timestamp("2022-01-06")
    trades = pd.DataFrame([_trade_row(y2020, y2020, 1.0, 100.0, 96.0)])  # only 1 trade, only in 2020
    row = stats_from_trades(_config(), trades, pd.DatetimeIndex([y2020, y2021, y2022]))  # data spans 3 years
    assert row["trades_per_year"] == pytest.approx(1 / 3)


def test_stats_from_trades_includes_all_param_fields_and_matches_summary_columns():
    row = stats_from_trades(_config(name="checkme"), _empty_trades(), pd.DatetimeIndex([D1]))
    assert set(row) == set(SUMMARY_COLUMNS)
    assert row["config_name"] == "checkme"
    assert row["stop_type"] == "gap_distal"


# ---------- summarize_config (thin integration wrapper) ----------


def test_summarize_config_smoke(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    config = _config()
    session_dates = pd.DatetimeIndex([pd.Timestamp("2024-06-03"), pd.Timestamp("2024-06-04")])
    row, r_row = summarize_config(df, config, "NQ", session_dates)
    assert row["config_name"] == "t"
    assert isinstance(row["trade_count"], int)
    assert r_row["config_hash"] == row["config_hash"]
    assert isinstance(json.loads(r_row["r_multiples"]), list)


# ---------- config_from_row round-trip ----------


def test_config_from_row_round_trips_from_a_plain_dict():
    config = StrategyConfig(
        name="orig", windows=("killzone_london", "killzone_ny_am"), sweep_required=True,
        sweep_universe="session_refs", stop_type="swing",
    )
    row = stats_from_trades(config, _empty_trades(), pd.DatetimeIndex([D1]))
    rebuilt = config_from_row(row)
    assert canonical_hash(rebuilt) == canonical_hash(config)
    assert rebuilt.windows == config.windows
    assert rebuilt.sweep_universe == config.sweep_universe


def test_config_from_row_round_trips_none_sweep_universe():
    config = StrategyConfig(name="orig", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    row = stats_from_trades(config, _empty_trades(), pd.DatetimeIndex([D1]))
    rebuilt = config_from_row(row)
    assert rebuilt.sweep_universe is None


def test_config_from_row_round_trips_through_parquet(tmp_path):
    # The real risk: parquet turns tuples into numpy arrays and None into
    # NaN, and numeric columns into numpy scalar dtypes. Both would corrupt
    # a reconstructed StrategyConfig (or silently change its canonical hash)
    # if stats_from_trades/config_from_row didn't guard against it.
    config = StrategyConfig(
        name="orig", windows=("killzone_london", "killzone_ny_am"), sweep_required=True,
        sweep_universe="session_refs", stop_type="swing",
    )
    row = stats_from_trades(config, _empty_trades(), pd.DatetimeIndex([D1]))
    _flush_shard(tmp_path, [row])
    loaded = load_shards(tmp_path)
    rebuilt = config_from_row(loaded.iloc[0])

    assert canonical_hash(rebuilt) == canonical_hash(config)
    assert rebuilt.windows == config.windows
    assert isinstance(rebuilt.sweep_k, int)
    assert isinstance(rebuilt.sweep_required, bool)


# ---------- shard load/merge/dedup ----------


def test_load_shards_empty_dir_returns_empty_with_summary_columns(tmp_path):
    out = load_shards(tmp_path)
    assert out.empty
    assert list(out.columns) == SUMMARY_COLUMNS


def test_load_shards_dedups_duplicate_hash_across_shards(tmp_path):
    config = StrategyConfig(name="dup", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    row_a = stats_from_trades(config, _empty_trades(), pd.DatetimeIndex([D1]))
    row_a["trade_count"] = 111
    row_b = dict(row_a)
    row_b["trade_count"] = 222

    _flush_shard(tmp_path, [row_a])
    _flush_shard(tmp_path, [row_b])
    merged = load_shards(tmp_path)

    assert len(merged) == 1
    assert merged.iloc[0]["trade_count"] in (111, 222)


# ---------- r_multiples companion shard ----------


def test_r_multiples_round_trip_through_shard(tmp_path):
    config = StrategyConfig(name="t", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    trades = pd.DataFrame(
        [
            _trade_row(D1, D1, 2.0, 100.0, 96.0),
            _trade_row(D2, D2, -1.0, -50.0, -54.0),
        ]
    )
    row = stats_from_trades(config, trades, pd.DatetimeIndex([D1, D2]))
    r_row = {"config_hash": row["config_hash"], "r_multiples": json.dumps(trades["r_multiple"].tolist())}

    _flush_r_multiples_shard(tmp_path, [r_row])
    loaded = load_r_multiples_shards(tmp_path)
    assert list(loaded.columns) == R_MULTIPLES_COLUMNS
    assert json.loads(loaded.iloc[0]["r_multiples"]) == [2.0, -1.0]

    fetched = r_multiples_for_hash(tmp_path, row["config_hash"])
    assert list(fetched) == [2.0, -1.0]


def test_r_multiples_for_hash_missing_returns_empty_array(tmp_path):
    assert len(r_multiples_for_hash(tmp_path, "does-not-exist")) == 0


def test_load_r_multiples_shards_empty_dir_returns_empty_with_columns(tmp_path):
    out = load_r_multiples_shards(tmp_path)
    assert out.empty
    assert list(out.columns) == R_MULTIPLES_COLUMNS


# ---------- run_sweep (process pool integration) ----------


def test_run_sweep_writes_shards_and_returns_merged_summary(tmp_path, synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    configs = [
        StrategyConfig(name="a", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal"),
        StrategyConfig(name="b", windows=("killzone_ny_pm",), sweep_required=False, stop_type="gap_distal"),
    ]
    shard_dir = tmp_path / "shards"
    result = run_sweep(df, configs, "NQ", shard_dir, workers=2, flush_every=10, progress=False)

    assert len(result) == 2
    assert set(result["config_name"]) == {"a", "b"}
    assert list(shard_dir.glob("shard_*.parquet"))
    assert list(shard_dir.glob("rmult_*.parquet"))

    # every summary row has a matching r_multiples companion entry
    r_hashes = set(load_r_multiples_shards(shard_dir)["config_hash"])
    assert set(result["config_hash"]) <= r_hashes


def test_run_sweep_checkpointing_skips_already_completed_configs(tmp_path, synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    config_a = StrategyConfig(name="a", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    config_b = StrategyConfig(name="b", windows=("killzone_ny_pm",), sweep_required=False, stop_type="gap_distal")
    config_c = StrategyConfig(name="c", windows=("killzone_london",), sweep_required=False, stop_type="gap_distal")

    shard_dir = tmp_path / "shards"
    run_sweep(df, [config_a, config_b], "NQ", shard_dir, workers=1, flush_every=10, progress=False)

    # Plant a detectable marker in the already-completed shard: if a second
    # run recomputes config_a/config_b instead of skipping them (as
    # checkpointing should), this marker gets overwritten.
    marker = -999999
    merged = load_shards(shard_dir)
    merged["trade_count"] = marker
    for f in shard_dir.glob("shard_*.parquet"):
        f.unlink()
    _flush_shard(shard_dir, merged.to_dict("records"))

    result = run_sweep(df, [config_a, config_b, config_c], "NQ", shard_dir, workers=1, flush_every=10, progress=False)

    assert len(result) == 3
    by_name = result.set_index("config_name")
    assert by_name.loc["a", "trade_count"] == marker
    assert by_name.loc["b", "trade_count"] == marker
    assert by_name.loc["c", "trade_count"] != marker


def test_run_sweep_returns_existing_shards_when_nothing_pending(tmp_path, synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-05 00:00:00")
    config_a = StrategyConfig(name="a", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    shard_dir = tmp_path / "shards"
    run_sweep(df, [config_a], "NQ", shard_dir, workers=1, flush_every=10, progress=False)

    result = run_sweep(df, [config_a], "NQ", shard_dir, workers=1, flush_every=10, progress=False)
    assert len(result) == 1
