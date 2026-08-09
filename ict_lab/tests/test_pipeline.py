from __future__ import annotations

import pandas as pd

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.pipeline import run_config, sweep_level_types


def _flat_bars(start="2024-06-03 00:00:00", end="2024-06-05 00:00:00"):
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    return pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 10, "contract": "X"}, index=idx
    )


def _config(**overrides):
    kwargs = dict(
        name="t", windows=("killzone_ny_am",), sweep_required=True,
        sweep_level_types=("prior_session_high", "prior_session_low"), stop_type="swing",
    )
    kwargs.update(overrides)
    return StrategyConfig(**kwargs)


def test_sweep_level_types_empty_when_sweep_not_required():
    config = _config(sweep_required=False, stop_type="gap_distal", sweep_level_types=())
    assert sweep_level_types(config) == ()


def test_sweep_level_types_explicit_takes_precedence_over_universe():
    config = _config(sweep_level_types=("prior_session_high",))
    assert sweep_level_types(config) == ("prior_session_high",)


def test_sweep_level_types_resolves_universe_across_all_windows():
    config = _config(
        windows=("killzone_ny_am", "killzone_london"), sweep_level_types=(), sweep_universe="session_refs",
    )
    result = sweep_level_types(config)
    assert set(result) == {
        "prior_session_high", "prior_session_low",
        "pre_killzone_ny_am_high", "pre_killzone_ny_am_low",
        "pre_killzone_london_high", "pre_killzone_london_low",
    }


def test_run_config_returns_four_dataframes_with_expected_columns():
    config = _config(mss_required=False, displacement_required=False)
    signals, no_signals, trades, no_trades = run_config(_flat_bars(), config, "NQ")
    for frame in (signals, no_signals, trades, no_trades):
        assert isinstance(frame, pd.DataFrame)
    assert "window" in signals.columns
    assert "reason" in no_trades.columns
