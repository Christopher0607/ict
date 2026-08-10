from __future__ import annotations

import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine import dashboard as d
from ict_lab.engine.sweep_benchmark import BenchmarkResult, SizingDecision
from ict_lab.engine.sweep_orchestrator import run_parts_1_through_7


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


def _rng(seed=0):
    return np.random.default_rng(seed)


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Reads width/height straight from the PNG IHDR chunk -- no Pillow
    dependency needed just to confirm save_panel_png's exact-pixel-size
    contract (the spec requires standalone 1920x1080 PNGs)."""
    with open(path, "rb") as f:
        header = f.read(24)
    width, height = struct.unpack(">II", header[16:24])
    return width, height


# ---------- save_panel_png ----------


def test_save_panel_png_is_exactly_1920x1080(tmp_path):
    fig = d.panel_funnel({"population": 10}, None)
    path = d.save_panel_png(fig, tmp_path / "out.png")
    assert path.exists() and path.stat().st_size > 0
    assert _png_dimensions(path) == (1920, 1080)


# ---------- panel 1: funnel ----------


def test_panel_funnel_bar_count_and_es_label():
    counts = {"population": 1000, "min_100_trades": 400, "net_pnl_positive": 150, "net_sharpe_0_5": 60, "bh_fdr_10pct": 12}
    fig = d.panel_funnel(counts, es_cross_symbol_survivors=5)
    ax = fig.axes[0]
    assert len(ax.patches) == 6
    assert "survived ES validation" in [t.get_text() for t in ax.get_yticklabels()]
    assert ax.get_title() == "The funnel"


def test_panel_funnel_without_es_is_five_bars():
    counts = {"population": 10, "min_100_trades": 5, "net_pnl_positive": 3, "net_sharpe_0_5": 2, "bh_fdr_10pct": 1}
    fig = d.panel_funnel(counts, es_cross_symbol_survivors=None)
    assert len(fig.axes[0].patches) == 5


def test_panel_funnel_missing_stage_keys_default_to_zero():
    fig = d.panel_funnel({"population": 5}, None)
    ax = fig.axes[0]
    assert len(ax.patches) == 5


# ---------- panel 2: sharpe distribution ----------


def test_panel_sharpe_distribution_empty_shows_placeholder():
    fig = d.panel_sharpe_distribution(np.array([np.nan, np.nan]), None)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("no configs" in t for t in texts)


def test_panel_sharpe_distribution_with_data_draws_bars_no_placeholder():
    fig = d.panel_sharpe_distribution(_rng().normal(0, 1, 100), _rng(1).normal(0, 1, 50), "some null")
    ax = fig.axes[0]
    assert len(ax.patches) > 0
    assert not any("no configs" in t.get_text() for t in ax.texts)
    assert any("some null" in h.get_label() for h in ax.get_legend().legend_handles)


# ---------- panel 3: frequency spectrum ----------


def test_panel_frequency_spectrum_marks_named_configs():
    fig = d.panel_frequency_spectrum(_rng().uniform(1, 500, 200), {"as_taught_5m": 40.0, "as_traded": 30.0})
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert len(ax.lines) == 2  # one axvline per named point


def test_panel_frequency_spectrum_empty_population_still_marks_named_points():
    fig = d.panel_frequency_spectrum(np.array([]), {"as_taught_5m": 10.0})
    ax = fig.axes[0]
    assert any("no configs" in t.get_text() for t in ax.texts)
    assert len(ax.lines) == 1


# ---------- panel 4: ablation ladder ----------


def _ablation_df():
    return pd.DataFrame(
        {
            "rung": [f"{i}_x" for i in range(1, 6)] + ["6_perfect_bias_lookahead"],
            "symbol": ["NQ"] * 6,
            "net_sharpe": [0.1, 0.3, 0.5, 0.8, 1.1, 1.4],
        }
    )


def test_panel_ablation_ladder_six_bars():
    fig = d.panel_ablation_ladder(_ablation_df(), "NQ")
    assert len(fig.axes[0].patches) == 6


def test_panel_ablation_ladder_lookahead_rung_uses_red_others_blue():
    fig = d.panel_ablation_ladder(_ablation_df(), "NQ")
    ax = fig.axes[0]
    from matplotlib.colors import to_hex

    colors = [to_hex(p.get_facecolor()) for p in ax.patches]
    assert colors[-1] == d.RED.lower()
    assert colors[0] == d.BLUE.lower()


def test_panel_ablation_ladder_missing_symbol_shows_placeholder():
    fig = d.panel_ablation_ladder(_ablation_df(), "ES")
    assert any("no ablation rows" in t.get_text() for t in fig.axes[0].texts)


# ---------- panel 5: null comparison ----------


def test_panel_null_comparison_three_subplots_with_real_line():
    null_dists = {
        "random_entry_same_windows": _rng().normal(0, 0.5, 100),
        "other_hours_same_logic": _rng(1).normal(0, 0.4, 100),
        "shuffled_direction": _rng(2).normal(0, 0.6, 100),
    }
    fig = d.panel_null_comparison(1.2, null_dists, "as_taught_5m")
    assert len(fig.axes) == 3
    for ax in fig.axes:
        assert len(ax.lines) == 1  # the real-result axvline


def test_panel_null_comparison_no_distributions_is_single_placeholder_axes():
    fig = d.panel_null_comparison(float("nan"), {}, "")
    assert len(fig.axes) == 1
    assert any("no null distributions" in t.get_text() for t in fig.axes[0].texts)


# ---------- panel 6: NQ vs ES scatter ----------


def test_panel_nq_es_scatter_plots_points():
    merged = pd.DataFrame(
        {"config_hash": ["a", "b"], "config_name": ["x", "y"], "nq_net_sharpe": [1.0, 0.6], "es_net_sharpe": [0.8, 0.2]}
    )
    fig = d.panel_nq_es_scatter(merged)
    ax = fig.axes[0]
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape[0] == 2


def test_panel_nq_es_scatter_empty_shows_placeholder():
    fig = d.panel_nq_es_scatter(pd.DataFrame(columns=["config_hash", "config_name", "nq_net_sharpe", "es_net_sharpe"]))
    assert any("no NQ survivors" in t.get_text() for t in fig.axes[0].texts)


# ---------- panel 7: per-year heatmap ----------


def test_panel_per_year_heatmap_draws_image_for_all_configs_and_years():
    per_year = {
        "cfg_a": pd.DataFrame({"year": [2019, 2020, 2021], "avg_r": [0.2, -0.1, 0.4], "trades": [50, 60, 70]}),
        "cfg_b": pd.DataFrame({"year": [2020, 2021], "avg_r": [0.1, 0.3], "trades": [40, 55]}),
    }
    fig = d.panel_per_year_heatmap(per_year)
    ax = fig.axes[0]
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (2, 3)  # 2 configs x 3 distinct years


def test_panel_per_year_heatmap_empty_shows_placeholder():
    fig = d.panel_per_year_heatmap({})
    assert any("no survivors" in t.get_text() for t in fig.axes[0].texts)


# ---------- panel 8: era split ----------


def test_panel_era_split_bar_count_is_configs_times_eras():
    era_results = {
        "cfg_a": {"2010_2021": {"net_sharpe": 0.8}, "2022_to_holdout_cutoff": {"net_sharpe": 0.3}},
        "as_taught_5m": {"2010_2021": {"net_sharpe": 1.0}, "2022_to_holdout_cutoff": {"net_sharpe": -0.2}},
    }
    fig = d.panel_era_split(era_results)
    ax = fig.axes[0]
    assert len(ax.patches) == 4  # 2 configs x 2 eras


def test_panel_era_split_empty_shows_placeholder():
    fig = d.panel_era_split({})
    assert any("no configs to era-split" in t.get_text() for t in fig.axes[0].texts)


# ---------- panel 9: discretion premium ----------


def _discretion_reports():
    return {
        "as_taught_5m": {
            "config_name": "as_taught_5m",
            "minimum_skip": {"breakeven_fraction": 0.1, "sharpe_fraction": 0.35, "n_losers": 40},
        },
        "as_traded": {
            "config_name": "as_traded",
            "minimum_skip": {"breakeven_fraction": None, "sharpe_fraction": None, "n_losers": 30},
        },
    }


def test_panel_discretion_premium_big_number_matches_headline_fraction():
    fig = d.panel_discretion_premium(_discretion_reports(), "as_taught_5m")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "35%" in texts


def test_panel_discretion_premium_not_achievable_shows_na():
    fig = d.panel_discretion_premium(_discretion_reports(), "as_traded")
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "N/A" in texts


def test_panel_discretion_premium_empty_shows_placeholder():
    fig = d.panel_discretion_premium({}, None)
    assert any("no discretion-premium" in t.get_text() for t in fig.axes[0].texts)


# ---------- panel 10: win rate ----------


def test_panel_win_rate_colors_by_claim_band():
    named_rows = [
        {"config_name": "as_taught_5m", "symbol": "NQ", "win_rate": 74.0},
        {"config_name": "as_taught_1m", "symbol": "NQ", "win_rate": 65.0},
        {"config_name": "as_traded", "symbol": "NQ", "win_rate": 90.0},
    ]
    fig = d.panel_win_rate(named_rows)
    ax = fig.axes[0]
    from matplotlib.colors import to_hex

    bar_colors = [to_hex(p.get_facecolor()) for p in ax.patches if p.get_height() > 0]
    assert to_hex(d.GREEN) in bar_colors
    assert to_hex(d.ORANGE) in bar_colors
    assert to_hex(d.BLUE) in bar_colors


def test_panel_win_rate_empty_shows_placeholder():
    fig = d.panel_win_rate([])
    assert any("no named-config win rates" in t.get_text() for t in fig.axes[0].texts)


# ---------- panel 11: cost sensitivity ----------


def _slippage_result(base):
    return {
        "stop_slippage_ticks": {t: {"net_sharpe": base - t * 0.1} for t in (0, 1, 2, 3)},
        "time_exit_slippage_ticks": {t: {"net_sharpe": base - t * 0.05} for t in (0, 1, 2)},
    }


def test_panel_cost_sensitivity_two_axes_one_line_per_config():
    fig = d.panel_cost_sensitivity({"cfg_a": _slippage_result(0.8), "cfg_b": _slippage_result(0.5)})
    assert len(fig.axes) == 2
    assert len(fig.axes[0].lines) == 3  # 2 config lines + the zero-line axhline
    assert len(fig.axes[1].lines) == 3


def test_panel_cost_sensitivity_empty_shows_placeholder_on_both_axes():
    fig = d.panel_cost_sensitivity({})
    for ax in fig.axes:
        assert any("no survivors" in t.get_text() for t in ax.texts)


# ---------- panel 12: ambiguous-bar pct ----------


def test_panel_ambiguous_bar_pct_bar_count():
    rows = [{"config_name": "as_taught_5m", "pct_ambiguous_bar": 12.0}, {"config_name": "as_traded", "pct_ambiguous_bar": 30.0}]
    fig = d.panel_ambiguous_bar_pct(rows)
    assert len(fig.axes[0].patches) == 2


def test_panel_ambiguous_bar_pct_empty_shows_placeholder():
    fig = d.panel_ambiguous_bar_pct([])
    assert any("no configs with ambiguous-bar" in t.get_text() for t in fig.axes[0].texts)


# ---------- panel 13: equity curves ----------


def _equity_trades(all_days, seed, n=40):
    rng = _rng(seed)
    dates = rng.choice(all_days, size=n)
    return pd.DataFrame({"session_date": dates, "net_pnl": rng.normal(20, 100, n)})


def test_panel_equity_curves_plots_three_lines_and_null_band():
    all_days = pd.date_range("2024-01-01", periods=20, freq="D")
    trades_by_config = {
        "best_realistic_survivor": _equity_trades(all_days, 1),
        "as_taught_5m": _equity_trades(all_days, 2),
        "as_traded": _equity_trades(all_days, 3),
    }
    null_paths = np.cumsum(_rng(9).normal(0, 50, (30, len(all_days))), axis=1)
    fig = d.panel_equity_curves(trades_by_config, all_days, "best_realistic_survivor", null_paths)
    ax = fig.axes[0]
    assert len(ax.lines) == 4  # 3 config lines + the zero-line axhline
    assert len(ax.collections) == 1  # the fill_between null band


def test_panel_equity_curves_no_null_paths_still_plots_lines():
    all_days = pd.date_range("2024-01-01", periods=10, freq="D")
    trades_by_config = {"as_taught_5m": _equity_trades(all_days, 2), "as_traded": _equity_trades(all_days, 3)}
    fig = d.panel_equity_curves(trades_by_config, all_days, "as_taught_5m", None)
    ax = fig.axes[0]
    assert len(ax.lines) == 3  # 2 config lines + the zero-line axhline
    assert len(ax.collections) == 0


def test_panel_equity_curves_empty_shows_placeholder():
    fig = d.panel_equity_curves({}, pd.DatetimeIndex([]), "x", None)
    assert any("no trade logs" in t.get_text() for t in fig.axes[0].texts)


# ---------- _select_headline_name ----------


def test_select_headline_name_prefers_best_realistic_survivor():
    assert d._select_headline_name({"as_taught_5m": 1, "best_realistic_survivor": 2}) == "best_realistic_survivor"


def test_select_headline_name_falls_back_to_as_taught_5m():
    assert d._select_headline_name({"as_taught_5m": 1, "as_traded": 2}) == "as_taught_5m"


def test_select_headline_name_falls_back_to_first_when_neither_present():
    assert d._select_headline_name({"only_one": 1}) == "only_one"


def test_select_headline_name_empty_is_none():
    assert d._select_headline_name({}) is None


# ---------- compute_equity_curve_data ----------


def _small_config(name, **overrides):
    kwargs = dict(
        name=name, windows=("killzone_ny_am",), sweep_required=False, stop_type="gap_distal",
        displacement_required=False, mss_required=False, entry_level="50%", target_type="fixed_r",
        target_r_multiple=2.0,
    )
    kwargs.update(overrides)
    return StrategyConfig(**kwargs)


def test_compute_equity_curve_data_shapes(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-08 00:00:00")
    named_configs = {"as_taught_5m": _small_config("as_taught_5m"), "as_traded": _small_config("as_traded")}
    data = d.compute_equity_curve_data(df, "NQ", {}, named_configs, n_null_iterations=5, seed=1)
    assert data["headline_name"] == "as_taught_5m"
    assert set(data["trades_by_config"]) == {"as_taught_5m", "as_traded"}
    assert data["null_paths"].shape == (5, len(data["all_days"]))


def test_compute_equity_curve_data_uses_best_realistic_survivor_when_present(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-08 00:00:00")
    named_configs = {"as_taught_5m": _small_config("as_taught_5m"), "as_traded": _small_config("as_traded")}
    reference_configs = {"best_realistic_survivor": _small_config("best_realistic_survivor", entry_level="distal")}
    data = d.compute_equity_curve_data(df, "NQ", reference_configs, named_configs, n_null_iterations=5, seed=1)
    assert data["headline_name"] == "best_realistic_survivor"
    assert set(data["trades_by_config"]) == {"as_taught_5m", "as_traded", "best_realistic_survivor"}


# ---------- build_dashboard end-to-end ----------


def _make_bars(base_price, start="2024-06-03 00:00:00", end="2024-06-10 00:00:00"):
    rng = np.random.default_rng(int(base_price))
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    n = len(idx)
    close = base_price + np.cumsum(rng.normal(0, 0.1, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + rng.uniform(0, 0.1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.1, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 10, "contract": "X"}, index=idx
    )


def _small_population(n=8):
    configs = []
    entry_levels = ["proximal", "50%", "distal"]
    windows_options = [("killzone_ny_am",), ("killzone_ny_pm",), ("killzone_london", "killzone_ny_am")]
    for i in range(n):
        configs.append(
            StrategyConfig(
                name=f"pop_{i}",
                windows=windows_options[i % len(windows_options)],
                sweep_required=False,
                stop_type="gap_distal",
                displacement_required=False,
                mss_required=False,
                entry_level=entry_levels[i % len(entry_levels)],
                target_type="fixed_r",
                target_r_multiple=2.0,
            )
        )
    return configs


def test_build_dashboard_end_to_end_writes_all_panel_files_and_html(tmp_path):
    df_nq = _make_bars(100.0)
    df_es = _make_bars(5000.0)
    population = _small_population(8)

    fake_bench = BenchmarkResult(n_benchmarked=5, median_seconds_per_config=0.01, workers=2, per_config_seconds=(0.01,) * 5)
    decision = SizingDecision(benchmark=fake_bench, n=5, viable=True, message="test")

    result = run_parts_1_through_7(
        df_nq, df_es, population, decision, tmp_path / "shards", tmp_path / "summary", n_null_iterations=5, workers=2,
    )
    equity_data = d.compute_equity_curve_data(
        df_nq, "NQ", result["reference_configs"], result["named_configs"], n_null_iterations=5, seed=1
    )

    dashboard_dir = tmp_path / "dashboard"
    written = d.build_dashboard(result, equity_data, output_dir=dashboard_dir, caveat="synthetic fixture, not real data")

    assert {p.name for p in written} == set(d.DASHBOARD_FILES)
    for filename in d.DASHBOARD_FILES:
        path = dashboard_dir / filename
        assert path.exists(), f"{filename} was not written"
        assert path.stat().st_size > 0, f"{filename} is empty"

    for path in written:
        if path.suffix == ".png":
            assert _png_dimensions(path) == (1920, 1080)

    html = (dashboard_dir / "index.html").read_text()
    assert html.count("data:image/png;base64,") == 13
    for title in d.PANEL_TITLES.values():
        assert title in html
    assert "synthetic fixture, not real data" in html
