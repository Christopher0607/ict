from __future__ import annotations

from ict_lab.configs.grid import load_named_configs, placeholder_note
from ict_lab.configs.strategy_config import StrategyConfig


def test_load_named_configs_returns_the_three_taught_configs():
    configs = load_named_configs()
    assert set(configs) == {"as_taught_5m", "as_taught_1m", "as_traded"}
    for name, config in configs.items():
        assert isinstance(config, StrategyConfig)
        assert config.name == name


def test_windows_and_level_types_come_back_as_tuples():
    configs = load_named_configs()
    for config in configs.values():
        assert isinstance(config.windows, tuple)
        assert isinstance(config.sweep_level_types, tuple)


def test_as_taught_5m_matches_the_spec_definition():
    c = load_named_configs()["as_taught_5m"]
    assert c.windows == ("killzone_london", "killzone_ny_am", "killzone_ny_pm")
    assert c.bias_method == "swing_structure"
    assert c.bias_timeframe == "15m"
    assert c.sweep_required is True
    assert c.sweep_universe == "bsl_ssl_15m"
    assert c.sweep_level_types == ()
    assert c.sweep_k == 3
    assert c.sweep_min_penetration_ticks == 1.0
    assert c.mss_required is False
    assert c.displacement_required is True
    assert c.displacement_atr_mult == 1.5
    assert c.fvg_timeframe == "5m"
    assert c.fvg_min_size_points == 0.0
    assert c.entry_level == "50%"
    assert c.stop_type == "swing"
    assert c.target_type == "next_liquidity"
    assert c.target_fallback_r_multiple == 2.0
    assert c.max_trades_per_window == 1


def test_as_taught_1m_differs_from_5m_only_in_fvg_timeframe():
    configs = load_named_configs()
    five, one = configs["as_taught_5m"], configs["as_taught_1m"]
    assert one.fvg_timeframe == "1m"
    five_dict = {k: v for k, v in five.to_dict().items() if k not in ("name", "fvg_timeframe")}
    one_dict = {k: v for k, v in one.to_dict().items() if k not in ("name", "fvg_timeframe")}
    assert five_dict == one_dict


def test_as_traded_differs_from_as_taught_5m_only_in_mss_required():
    configs = load_named_configs()
    five, traded = configs["as_taught_5m"], configs["as_traded"]
    assert traded.mss_required is True
    five_dict = {k: v for k, v in five.to_dict().items() if k not in ("name", "mss_required")}
    traded_dict = {k: v for k, v in traded.to_dict().items() if k not in ("name", "mss_required")}
    assert five_dict == traded_dict


def test_placeholder_note_flags_as_traded_only():
    assert placeholder_note("as_traded") == "as_traded (default placeholder, not yet user-specified)"
    assert placeholder_note("as_taught_5m") is None
    assert placeholder_note("as_taught_1m") is None
