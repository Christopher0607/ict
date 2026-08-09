from __future__ import annotations

import pandas as pd

from ict_lab.configs.grid import load_named_configs
from ict_lab.configs.sweep_grid import canonical_hash
from ict_lab.engine.ablation import RUNG_COLUMNS, RUNG_DEFINITIONS, build_rung_configs, run_ablation_ladder


def test_build_rung_configs_has_six_rungs_in_spec_order():
    rungs = build_rung_configs()
    assert list(rungs) == [name for name, _ in RUNG_DEFINITIONS]
    assert len(rungs) == 6


def test_rung1_has_no_gates_and_uses_full_session_window():
    rung1 = build_rung_configs()["1_fvg_entry_only"]
    assert rung1.windows == ("full_session",)
    assert rung1.bias_method == "none"
    assert rung1.sweep_required is False
    assert rung1.displacement_required is False
    assert rung1.stop_type == "gap_distal"  # "swing" is impossible without sweep_required=True


def test_rung2_adds_sweep_and_unlocks_swing_stop():
    rung2 = build_rung_configs()["2_plus_sweep_required"]
    assert rung2.sweep_required is True
    assert rung2.sweep_universe == "bsl_ssl_15m"
    assert rung2.displacement_required is False
    assert rung2.windows == ("full_session",)
    assert rung2.stop_type == "swing"


def test_rung3_adds_displacement_only():
    rung2 = build_rung_configs()["2_plus_sweep_required"]
    rung3 = build_rung_configs()["3_plus_displacement_required"]
    assert rung3.displacement_required is True
    # everything else matches rung2 except displacement_required
    d2, d3 = rung2.to_dict(), rung3.to_dict()
    d2.pop("name"), d2.pop("displacement_required")
    d3.pop("name"), d3.pop("displacement_required")
    assert d2 == d3


def test_rung4_adds_window_restriction_only():
    rung3 = build_rung_configs()["3_plus_displacement_required"]
    rung4 = build_rung_configs()["4_plus_window_restriction"]
    assert rung4.windows == ("killzone_london", "killzone_ny_am", "killzone_ny_pm")
    d3, d4 = rung3.to_dict(), rung4.to_dict()
    d3.pop("name"), d3.pop("windows")
    d4.pop("name"), d4.pop("windows")
    assert d3 == d4


def test_rung5_equals_the_real_as_taught_5m_config():
    rung5 = build_rung_configs()["5_plus_15m_bias_full_as_taught_5m"]
    as_taught_5m = load_named_configs()["as_taught_5m"]
    assert canonical_hash(rung5) == canonical_hash(as_taught_5m)


def test_rung6_uses_perfect_bias_otherwise_matches_rung5():
    rung5 = build_rung_configs()["5_plus_15m_bias_full_as_taught_5m"]
    rung6 = build_rung_configs()["6_perfect_bias_lookahead"]
    assert rung6.bias_method == "perfect"
    d5, d6 = rung5.to_dict(), rung6.to_dict()
    d5.pop("name"), d5.pop("bias_method"), d5.pop("bias_timeframe")
    d6.pop("name"), d6.pop("bias_method"), d6.pop("bias_timeframe")
    assert d5 == d6


def test_run_ablation_ladder_produces_one_row_per_rung_per_symbol(synthetic_bars):
    df_nq = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00")
    df_es = synthetic_bars("2024-06-03 00:00:00", "2024-06-10 00:00:00", base_price=5000.0)
    result = run_ablation_ladder(df_nq, df_es)

    assert list(result.columns) == RUNG_COLUMNS
    assert len(result) == 12  # 6 rungs x 2 symbols
    assert set(result["symbol"]) == {"NQ", "ES"}
    assert set(result["rung"]) == {name for name, _ in RUNG_DEFINITIONS}
    assert (result["trades"] >= 0).all()
