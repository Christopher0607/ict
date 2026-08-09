from __future__ import annotations

import pytest

from ict_lab.configs.strategy_config import COST_MODELS, CONSENSUS_CONFIG, StrategyConfig


def _base_kwargs(**overrides):
    kwargs = dict(
        name="t",
        window="killzone_ny_am",
        sweep_required=True,
        sweep_level_types=("prior_session_high",),
    )
    kwargs.update(overrides)
    return kwargs


def test_consensus_config_matches_spec_verification_baseline():
    c = CONSENSUS_CONFIG
    assert c.window == "killzone_ny_am"
    assert c.bias_method == "none"
    assert c.sweep_required is True
    assert c.sweep_k == 3
    assert c.sweep_min_penetration_ticks == 1.0
    assert c.mss_required is False
    assert c.displacement_required is True
    assert c.displacement_atr_mult == 1.5
    assert c.fvg_timeframe == "5m"
    assert c.fvg_min_size_points == 0.0
    assert c.entry_level == "50%"
    assert c.stop_type == "swing"
    assert c.target_type == "fixed_r"
    assert c.target_r_multiple == 2.0


def test_cost_models_match_spec_defaults():
    assert COST_MODELS["NQ"].tick_size == 0.25
    assert COST_MODELS["NQ"].tick_value == 5.00
    assert COST_MODELS["ES"].tick_size == 0.25
    assert COST_MODELS["ES"].tick_value == 12.50
    for m in COST_MODELS.values():
        assert m.commission_round_turn == 4.00
        assert m.stop_slippage_ticks == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"entry_level": "bogus"},
        {"stop_type": "bogus"},
        {"stop_type": "fixed_points", "stop_fixed_points": 0},
        {"target_type": "bogus"},
        {"target_type": "time", "target_time_bars": 0},
        {"hard_exit": "bogus"},
        {"bias_method": "bogus"},
        {"mss_break_style": "bogus"},
        {"max_trades_per_window": 2},
    ],
)
def test_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        StrategyConfig(**_base_kwargs(**overrides))


def test_swing_stop_requires_sweep_required():
    with pytest.raises(ValueError, match="sweep_required"):
        StrategyConfig(
            name="t", window="killzone_ny_am", stop_type="swing", sweep_required=False
        )


def test_sweep_required_needs_level_types():
    with pytest.raises(ValueError, match="sweep_level_types"):
        StrategyConfig(
            name="t", window="killzone_ny_am", sweep_required=True, sweep_level_types=()
        )


def test_to_dict_round_trips_key_fields():
    d = CONSENSUS_CONFIG.to_dict()
    assert d["name"] == "consensus"
    assert d["window"] == "killzone_ny_am"
    assert d["target_r_multiple"] == 2.0
    assert isinstance(d["sweep_level_types"], tuple)
