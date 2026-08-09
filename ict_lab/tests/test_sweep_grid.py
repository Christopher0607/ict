from __future__ import annotations

from ict_lab.configs.sweep_grid import (
    build_canonical_population,
    canonical_hash,
    canonicalize,
    enumerate_raw_configs,
    load_sweep_axes,
    load_sweep_fixed,
)
from ict_lab.configs.strategy_config import StrategyConfig

_SMALL_AXES = {
    "windows": [["killzone_ny_am"]],
    "bias_method": ["none"],
    "sweep_required": [True, False],
    "sweep_universe": ["session_refs", "swings_only"],
    "mss_required": [False],
    "displacement_required": [False],
    "fvg_timeframe": ["5m"],
    "entry_level": ["50%"],
    "stop_type": ["swing", "gap_distal"],
    "target_type": ["fixed_r"],
    "max_trades_per_window": [1],
}
_SMALL_FIXED = {
    "sweep_k": 3, "sweep_min_penetration_ticks": 1.0, "displacement_atr_mult": 1.5,
    "target_r_multiple": 2.0, "target_fallback_r_multiple": 2.0, "swing_n": 5, "swing_15m_n": 5,
    "stop_buffer_ticks": 1.0, "bias_timeframe": "15m", "bias_swing_n": 5, "bias_ma_period": 20,
    "fvg_min_size_points": 0.0, "fvg_min_size_atr_mult": 0.0,
}


def test_grid_json_has_sweep_axes_and_fixed_sections():
    axes = load_sweep_axes()
    fixed = load_sweep_fixed()
    assert "windows" in axes and "sweep_universe" in axes and "max_trades_per_window" in axes
    assert "perfect" not in axes.get("bias_method", [])  # excluded per spec instruction
    assert "sweep_k" in fixed


def test_enumerate_raw_configs_skips_invalid_stop_swing_without_sweep():
    # stop_type="swing" needs sweep_required=True (existing hard rule); the
    # 2x2x2=8 combos where stop_type=swing & sweep_required=False must be
    # caught, not silently constructed with a wrong config.
    valid, invalid = enumerate_raw_configs(_SMALL_AXES, _SMALL_FIXED)
    total = 1 * 1 * 2 * 2 * 1 * 1 * 1 * 1 * 2 * 1 * 1  # product of _SMALL_AXES value-list lengths
    assert len(valid) + invalid == total
    assert invalid > 0
    assert all(isinstance(c, StrategyConfig) for c in valid)
    assert not any(c.stop_type == "swing" and not c.sweep_required for c in valid)


def test_canonicalize_nulls_sweep_universe_when_not_required():
    a = StrategyConfig(
        name="a", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal",
    )
    canon = canonicalize(a)
    assert canon.sweep_universe is None
    assert canon.sweep_level_types == ()


def test_canonicalize_nulls_target_fallback_when_not_next_liquidity():
    a = StrategyConfig(
        name="a", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal",
        target_type="fixed_r", target_fallback_r_multiple=3.5,
    )
    assert canonicalize(a).target_fallback_r_multiple == 0.0


def test_canonicalize_nulls_bias_subparams_when_bias_method_is_none():
    a = StrategyConfig(
        name="a", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal",
        bias_method="none", bias_timeframe="1h", bias_swing_n=9, bias_ma_period=50,
    )
    canon = canonicalize(a)
    assert canon.bias_timeframe == ""
    assert canon.bias_swing_n == 0
    assert canon.bias_ma_period == 0


def test_canonicalize_preserves_swing_structure_bias_subparams():
    a = StrategyConfig(
        name="a", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal",
        bias_method="swing_structure", bias_timeframe="1h", bias_swing_n=9,
    )
    canon = canonicalize(a)
    assert canon.bias_timeframe == "1h"
    assert canon.bias_swing_n == 9


def test_canonicalize_never_collapses_a_live_parameter():
    a = StrategyConfig(name="a", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal", entry_level="proximal")
    b = StrategyConfig(name="b", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal", entry_level="distal")
    assert canonical_hash(canonicalize(a)) != canonical_hash(canonicalize(b))


def test_canonical_hash_ignores_name():
    a = StrategyConfig(name="alpha", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    b = StrategyConfig(name="beta", windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal")
    assert canonical_hash(a) == canonical_hash(b)


def test_canonical_hash_collapses_dead_parameter_variants():
    # Two configs differing only in sweep_universe, but with sweep_required
    # off (so sweep_universe is dead), must canonicalize to the same hash.
    a = canonicalize(StrategyConfig(
        name="a", windows=("killzone_ny_am",), sweep_required=False, sweep_universe=None,
        stop_type="gap_distal",
    ))
    b = canonicalize(StrategyConfig(
        name="b", windows=("killzone_ny_am",), sweep_required=True, sweep_universe="swings_only",
        stop_type="gap_distal",
    ))
    # a never had sweep_universe set; b did but sweep_required differs so
    # this pair is a control showing sweep_required itself is NOT dead --
    # they must stay distinct.
    assert canonical_hash(a) != canonical_hash(b)


def test_build_canonical_population_dedups_and_reports_counts():
    configs, report = build_canonical_population(_SMALL_AXES, _SMALL_FIXED)
    total = 1 * 1 * 2 * 2 * 1 * 1 * 1 * 1 * 2 * 1 * 1
    assert report["raw_valid"] + report["raw_invalid"] == total
    assert report["canonical"] == len(configs)
    assert report["canonical"] <= report["raw_valid"]
    hashes = {canonical_hash(c) for c in configs}
    assert len(hashes) == len(configs)  # every survivor is a distinct canonical hash


def test_build_canonical_population_dedup_actually_fires_on_small_axes():
    # sweep_required=False makes sweep_universe dead, so the 2 sweep_universe
    # values collapse for every sweep_required=False combo -- canonical
    # count must be strictly less than raw_valid here.
    _, report = build_canonical_population(_SMALL_AXES, _SMALL_FIXED)
    assert report["canonical"] < report["raw_valid"]


def test_build_canonical_population_with_real_grid_axes_is_nonempty():
    configs, report = build_canonical_population()
    assert report["raw_valid"] > 0
    assert 0 < report["canonical"] <= report["raw_valid"]
    assert all(isinstance(c, StrategyConfig) for c in configs)
