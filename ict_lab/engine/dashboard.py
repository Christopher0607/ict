"""Phase 6: results dashboard. A single self-contained dark-theme HTML page
(analysis/dashboard/index.html) visualizing every Phase 5 statistical
result, plus each of its 13 panels exported standalone as a 1920x1080 PNG
(analysis/dashboard/panel_XX_*.png), per the spec.

Every panel_* function is a pure function over already-computed data (a
DataFrame slice, a dict, a numpy array) -- mirrors summary_pack.py's
writers, and keeps each panel unit-testable against hand-built fixtures
without needing a real sweep. build_dashboard() is the only place that
reaches into a full Phase 5 `result` dict (as returned by
sweep_orchestrator.run_parts_1_through_7) and wires it into the 13 panels.

Panel 13 (equity curves) needs data Part 1-7's summary schema does not
retain: real per-config trade logs, and, for an honest null "band" (not a
statistic-derived illustration), the null model's own per-iteration
cumulative-PnL paths. compute_equity_curve_data() builds both from the
already-tested pipeline (run_config, random_entry_null_distribution(...,
return_paths=True)) -- it is Phase 6's own preparation step, run once
after run_parts_1_through_7 returns, not a change to Part 1-7's contract.

Every panel renders whatever data it is given, honestly, including "no
survivors" / "no data" placeholders where a stage produced nothing --
never fabricated numbers to fill a panel. In this sandbox, with no real
NQ/ES data, every panel a caller actually renders will show exactly that
placeholder or a shape built from synthetic fixtures -- see README.
"""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.figure import Figure

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.null_models import random_entry_null_distribution
from ict_lab.engine.pipeline import all_session_dates, run_config

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "analysis" / "dashboard"

PANEL_WIDTH_PX, PANEL_HEIGHT_PX = 1920, 1080
DPI = 100
FIGSIZE = (PANEL_WIDTH_PX / DPI, PANEL_HEIGHT_PX / DPI)

BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#30363d"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
GREEN = "#3fb950"
RED = "#f85149"
PURPLE = "#bc8cff"
YELLOW = "#d29922"

PANEL_TITLES = {
    1: "The funnel",
    2: "Net Sharpe: population vs. null",
    3: "Trade-frequency spectrum",
    4: "Ablation ladder",
    5: "Real result vs. null distributions",
    6: "NQ vs. ES net Sharpe (survivors)",
    7: "Per-year average R (survivors)",
    8: "Era split: 2010-2021 vs. 2022+",
    9: "The discretion premium",
    10: "Named configs: win rate vs. the 70-80% claim",
    11: "Cost sensitivity: net Sharpe vs. slippage",
    12: "Percent of trades on the ambiguous-bar assumption",
    13: "Equity curves",
}

PANEL_SLUGS = {
    1: "funnel",
    2: "sharpe_distribution",
    3: "frequency_spectrum",
    4: "ablation_ladder",
    5: "real_vs_null",
    6: "nq_es_scatter",
    7: "per_year_heatmap",
    8: "era_split",
    9: "discretion_premium",
    10: "win_rate",
    11: "cost_sensitivity",
    12: "ambiguous_bar_pct",
    13: "equity_curves",
}

DASHBOARD_PANEL_FILES = tuple(f"panel_{i:02d}_{PANEL_SLUGS[i]}.png" for i in sorted(PANEL_SLUGS))
DASHBOARD_FILES = DASHBOARD_PANEL_FILES + ("index.html",)

_NULL_LABELS = {
    "random_entry_same_windows": "random entry, same windows",
    "other_hours_same_logic": "other hours, same logic",
    "shuffled_direction": "shuffled direction",
}


# ---------- shared plotting helpers ----------


def _new_figure(nrows: int = 1, ncols: int = 1) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.35, linewidth=0.6)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
    return fig, axes_arr


def _style_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    leg = ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT, fontsize=9)
    leg.get_frame().set_alpha(0.9)


def _empty_panel(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=14, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def save_panel_png(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _select_headline_name(candidates, preferred=("best_realistic_survivor", "as_taught_5m")) -> str | None:
    for name in preferred:
        if name in candidates:
            return name
    return next(iter(candidates), None)


# ---------- panel 1: the funnel ----------

_FUNNEL_STAGE_LABELS = [
    ("population", "configs tested"),
    ("min_100_trades", "min. 100 trades"),
    ("net_pnl_positive", "net PnL > 0"),
    ("net_sharpe_0_5", "net Sharpe >= 0.5"),
    ("bh_fdr_10pct", "survived BH-FDR 10%"),
]


def panel_funnel(counts: dict, es_cross_symbol_survivors: int | None = None) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    values = [counts.get(key, 0) for key, _ in _FUNNEL_STAGE_LABELS]
    labels = [label for _, label in _FUNNEL_STAGE_LABELS]
    if es_cross_symbol_survivors is not None:
        values.append(es_cross_symbol_survivors)
        labels.append("survived ES validation")

    n = len(values)
    y = np.arange(n)[::-1]
    colors = colormaps["Blues"](np.linspace(0.9, 0.4, n))
    ax.barh(y, values, color=colors, edgecolor=GRID_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=12)
    max_v = max(values) if values else 0
    for yi, v in zip(y, values):
        ax.text(v + max_v * 0.01 + 0.01, yi, f"{v:,}", va="center", color=TEXT, fontsize=11)
    ax.set_xlim(0, max_v * 1.15 if max_v else 1)
    ax.set_xlabel("configs remaining")
    ax.set_title("The funnel", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------- panel 2: net Sharpe distribution vs. null ----------


def panel_sharpe_distribution(
    population_sharpe: np.ndarray, null_distribution: np.ndarray | None = None, null_label: str = "null"
) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    pop = np.asarray(population_sharpe, dtype=float)
    pop = pop[~np.isnan(pop)]
    null = np.asarray(null_distribution, dtype=float) if null_distribution is not None else np.array([])
    null = null[~np.isnan(null)]

    if len(pop) == 0 and len(null) == 0:
        _empty_panel(ax, "no configs with a computable net Sharpe")
        fig.tight_layout()
        return fig

    if len(pop):
        ax.hist(pop, bins=30, color=BLUE, alpha=0.75, density=True, label=f"population (n={len(pop)})")
    if len(null):
        ax.hist(null, bins=30, color=MUTED, alpha=0.55, density=True, label=f"{null_label} (n={len(null)})")
    ax.axvline(0, color=GRID_COLOR, linewidth=1)
    ax.set_xlabel("net Sharpe")
    ax.set_ylabel("density")
    ax.set_title("Net Sharpe: population vs. null", color=TEXT, fontsize=16, fontweight="bold")
    _style_legend(ax)
    fig.tight_layout()
    return fig


# ---------- panel 3: trade-frequency spectrum ----------


def panel_frequency_spectrum(population_trades_per_year: np.ndarray, named_points: dict[str, float] | None = None) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    pop = np.asarray(population_trades_per_year, dtype=float)
    pop = pop[(~np.isnan(pop)) & (pop > 0)]
    named_points = named_points or {}

    if len(pop):
        lo, hi = float(pop.min()), float(pop.max())
        if lo == hi:
            hi = lo * 1.5 + 1
        bins = np.logspace(np.log10(lo), np.log10(hi), 25)
        ax.hist(pop, bins=bins, color=BLUE, alpha=0.8)
        ax.set_xscale("log")
    else:
        _empty_panel(ax, "no configs with trades/year > 0")

    ax.figure.canvas.draw()
    ymax = ax.get_ylim()[1] or 1
    colors = [ORANGE, GREEN, PURPLE, YELLOW]
    for (name, value), color in zip(named_points.items(), colors):
        if value and np.isfinite(value) and value > 0:
            ax.axvline(value, color=color, linestyle="--", linewidth=1.5)
            ax.text(value, ymax * 0.97, f" {name}", rotation=90, va="top", ha="left", color=color, fontsize=9)

    ax.set_xlabel("trades / year (log scale)")
    ax.set_ylabel("configs")
    ax.set_title("Trade-frequency spectrum", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------- panel 4: ablation ladder ----------


def panel_ablation_ladder(ablation_df: pd.DataFrame, symbol: str = "NQ") -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    subset = ablation_df[ablation_df["symbol"] == symbol].sort_values("rung") if not ablation_df.empty else ablation_df
    if subset.empty:
        _empty_panel(ax, f"no ablation rows for {symbol}")
        fig.tight_layout()
        return fig

    x = np.arange(len(subset))
    colors = [RED if ("perfect" in str(r) or "lookahead" in str(r)) else BLUE for r in subset["rung"]]
    values = subset["net_sharpe"].fillna(0.0)
    ax.bar(x, values, color=colors, edgecolor=GRID_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(subset["rung"], rotation=20, ha="right", fontsize=9, color=TEXT)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
    ax.set_ylabel("net Sharpe")
    ax.set_title(f"Ablation ladder ({symbol})", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------- panel 5: real result vs. the three null distributions ----------


def panel_null_comparison(real_sharpe: float, null_dists: dict[str, np.ndarray], config_name: str = "") -> Figure:
    keys = [k for k in _NULL_LABELS if k in null_dists]
    fig, axes = _new_figure(1, max(len(keys), 1))
    if not keys:
        _empty_panel(axes[0], "no null distributions available")
        fig.tight_layout()
        return fig

    for ax, key in zip(axes, keys):
        dist = np.asarray(null_dists[key], dtype=float)
        dist = dist[~np.isnan(dist)]
        if len(dist):
            ax.hist(dist, bins=30, color=MUTED, alpha=0.8)
        else:
            _empty_panel(ax, "no draws")
        if real_sharpe is not None and np.isfinite(real_sharpe):
            ax.axvline(real_sharpe, color=GREEN, linewidth=2.0, label=f"real ({real_sharpe:.2f})")
            _style_legend(ax)
        ax.set_title(_NULL_LABELS[key], color=TEXT, fontsize=12)
        ax.set_xlabel("net Sharpe")

    fig.suptitle(f"Real result vs. null distributions -- {config_name}", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------- panel 6: NQ vs. ES net Sharpe scatter (survivors) ----------


def panel_nq_es_scatter(merged: pd.DataFrame) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    if merged.empty:
        _empty_panel(ax, "no NQ survivors to compare against ES")
        fig.tight_layout()
        return fig

    x = merged["nq_net_sharpe"].to_numpy(dtype=float)
    y = merged["es_net_sharpe"].to_numpy(dtype=float)
    colors = np.where(y >= 0.5, GREEN, RED)
    ax.scatter(x, y, c=colors, s=70, edgecolor=GRID_COLOR, linewidth=0.6, alpha=0.9)

    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)], np.array([0.0, 1.0])])
    lo, hi = float(finite.min()) - 0.2, float(finite.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", linewidth=1, label="NQ = ES")
    ax.axhline(0.5, color=YELLOW, linestyle=":", linewidth=1, label="ES net Sharpe = 0.5")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("NQ net Sharpe")
    ax.set_ylabel("ES net Sharpe")
    ax.set_title("NQ vs. ES net Sharpe (survivors)", color=TEXT, fontsize=16, fontweight="bold")
    _style_legend(ax)
    fig.tight_layout()
    return fig


# ---------- panel 7: per-year average R heatmap (survivors) ----------


def panel_per_year_heatmap(per_year: dict[str, pd.DataFrame]) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    ax.grid(False)
    non_empty = {name: df for name, df in (per_year or {}).items() if df is not None and not df.empty}
    if not non_empty:
        _empty_panel(ax, "no survivors with per-year data")
        fig.tight_layout()
        return fig

    all_years = sorted(set().union(*[set(df["year"]) for df in non_empty.values()]))
    names = sorted(non_empty)
    matrix = np.full((len(names), len(all_years)), np.nan)
    for i, name in enumerate(names):
        by_year = non_empty[name].set_index("year")["avg_r"]
        for j, year in enumerate(all_years):
            if year in by_year.index:
                matrix[i, j] = by_year.loc[year]

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.abs(finite).max()) if len(finite) else 1.0
    vmax = vmax or 1.0
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels(all_years, color=TEXT)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=TEXT, fontsize=9)
    for i in range(len(names)):
        for j in range(len(all_years)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT)
    ax.set_title("Per-year average R (survivors)", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------- panel 8: era split ----------


def panel_era_split(era_results: dict[str, dict]) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    if not era_results:
        _empty_panel(ax, "no configs to era-split")
        fig.tight_layout()
        return fig

    names = sorted(era_results)
    era_keys = sorted({k for eras in era_results.values() for k in eras})
    x = np.arange(len(names))
    width = 0.8 / max(len(era_keys), 1)
    colors = [BLUE, ORANGE, PURPLE, YELLOW]
    for i, era_key in enumerate(era_keys):
        values = [era_results[n].get(era_key, {}).get("net_sharpe", np.nan) for n in names]
        offset = i * width - width * (len(era_keys) - 1) / 2
        ax.bar(x + offset, values, width=width, label=era_key, color=colors[i % len(colors)])
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9, color=TEXT)
    ax.set_ylabel("net Sharpe")
    ax.set_title("Era split: 2010-2021 vs. 2022+", color=TEXT, fontsize=16, fontweight="bold")
    _style_legend(ax)
    fig.tight_layout()
    return fig


# ---------- panel 9: the discretion premium ----------


def panel_discretion_premium(reports: dict[str, dict], headline_name: str | None = None) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    ax.axis("off")
    ax.grid(False)
    if not reports:
        _empty_panel(ax, "no discretion-premium reports available")
        fig.tight_layout()
        return fig

    headline_name = headline_name if headline_name in reports else next(iter(reports))
    headline = reports[headline_name]
    fraction = headline["minimum_skip"]["sharpe_fraction"]
    big_text = f"{fraction * 100:.0f}%" if fraction is not None else "N/A"
    ax.text(0.5, 0.62, big_text, ha="center", va="center", color=GREEN, fontsize=110, fontweight="bold", transform=ax.transAxes)
    caption = (
        f"{headline_name}: minimum fraction of losing trades that must be correctly skipped in advance, "
        "to reach net Sharpe >= 1.0"
        if fraction is not None
        else f"{headline_name}: net Sharpe >= 1.0 is not achievable even skipping every losing trade"
    )
    ax.text(0.5, 0.40, caption, ha="center", va="center", color=TEXT, fontsize=15, transform=ax.transAxes)

    others = [n for n in reports if n != headline_name]
    detail_lines = []
    for name in [headline_name] + others:
        skip = reports[name]["minimum_skip"]
        be = skip["breakeven_fraction"]
        sh = skip["sharpe_fraction"]
        be_s = f"{be * 100:.0f}%" if be is not None else "n/a"
        sh_s = f"{sh * 100:.0f}%" if sh is not None else "n/a"
        detail_lines.append(f"{name}: breakeven skip {be_s}   |   Sharpe>=1.0 skip {sh_s}")
    ax.text(0.5, 0.14, "\n".join(detail_lines), ha="center", va="center", color=MUTED, fontsize=11, transform=ax.transAxes)

    ax.set_title("The discretion premium", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------- panel 10: named configs' win rate vs. the 70-80% claim ----------


def panel_win_rate(named_rows: list[dict]) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    rows = [r for r in (named_rows or []) if pd.notna(r.get("win_rate"))]
    if not rows:
        _empty_panel(ax, "no named-config win rates available")
        fig.tight_layout()
        return fig

    labels = [f"{r['config_name']} ({r['symbol']})" for r in rows]
    values = [r["win_rate"] for r in rows]
    x = np.arange(len(rows))
    colors = [GREEN if 70.0 <= v <= 80.0 else (ORANGE if v < 70.0 else BLUE) for v in values]
    ax.axhspan(70, 80, color=MUTED, alpha=0.18, label="commonly claimed 70-80% band")
    ax.bar(x, values, color=colors, edgecolor=GRID_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9, color=TEXT)
    ax.set_ylabel("win rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Named configs: win rate vs. the 70-80% claim", color=TEXT, fontsize=16, fontweight="bold")
    _style_legend(ax)
    fig.tight_layout()
    return fig


# ---------- panel 11: cost / slippage sensitivity ----------


def panel_cost_sensitivity(slippage_results: dict[str, dict]) -> Figure:
    fig, axes = _new_figure(1, 2)
    if not slippage_results:
        for ax in axes:
            _empty_panel(ax, "no survivors to report cost sensitivity for")
        fig.tight_layout()
        return fig

    colors = [BLUE, ORANGE, PURPLE, YELLOW, GREEN]
    for i, (name, data) in enumerate(slippage_results.items()):
        color = colors[i % len(colors)]
        stop = data.get("stop_slippage_ticks", {})
        ticks = sorted(stop)
        values = [stop[t]["net_sharpe"] for t in ticks]
        axes[0].plot(ticks, values, marker="o", color=color, label=name)

        time_exit = data.get("time_exit_slippage_ticks", {})
        ticks2 = sorted(time_exit)
        values2 = [time_exit[t]["net_sharpe"] for t in ticks2]
        axes[1].plot(ticks2, values2, marker="o", color=color, label=name)

    axes[0].set_title("Stop-slippage sensitivity", color=TEXT, fontsize=13)
    axes[0].set_xlabel("stop slippage (ticks)")
    axes[0].set_ylabel("net Sharpe")
    axes[1].set_title("Time-exit slippage sensitivity", color=TEXT, fontsize=13)
    axes[1].set_xlabel("time-exit slippage (ticks)")
    for ax in axes:
        ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
        _style_legend(ax)
    fig.suptitle("Cost sensitivity: net Sharpe vs. slippage assumptions", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------- panel 12: percent of trades on the ambiguous-bar assumption ----------


def panel_ambiguous_bar_pct(rows: list[dict]) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    valid = [r for r in (rows or []) if pd.notna(r.get("pct_ambiguous_bar"))]
    if not valid:
        _empty_panel(ax, "no configs with ambiguous-bar data")
        fig.tight_layout()
        return fig

    labels = [r["config_name"] for r in valid]
    values = [r["pct_ambiguous_bar"] for r in valid]
    x = np.arange(len(valid))
    ax.bar(x, values, color=YELLOW, edgecolor=GRID_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9, color=TEXT)
    ax.set_ylabel("% of trades")
    ax.set_title("Percent of trades on the ambiguous-bar assumption", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------- panel 13: equity curves ----------


def _cumulative_daily_pnl(trades: pd.DataFrame | None, all_days: pd.DatetimeIndex) -> pd.Series:
    if trades is None or trades.empty:
        return pd.Series(0.0, index=all_days).cumsum()
    daily = trades.groupby("session_date")["net_pnl"].sum().reindex(all_days, fill_value=0.0)
    return daily.cumsum()


def panel_equity_curves(
    trades_by_config: dict[str, pd.DataFrame],
    all_days: pd.DatetimeIndex,
    headline_name: str,
    null_paths: np.ndarray | None = None,
) -> Figure:
    fig, axes = _new_figure()
    ax = axes[0]
    if not trades_by_config or len(all_days) == 0:
        _empty_panel(ax, "no trade logs available for an equity curve")
        fig.tight_layout()
        return fig

    x = np.arange(len(all_days))

    if null_paths is not None and getattr(null_paths, "ndim", 0) == 2 and len(null_paths):
        valid_paths = null_paths[~np.isnan(null_paths).any(axis=1)]
        if len(valid_paths):
            p10 = np.percentile(valid_paths, 10, axis=0)
            p90 = np.percentile(valid_paths, 90, axis=0)
            ax.fill_between(x, p10, p90, color=MUTED, alpha=0.25, label="random-entry null (p10-p90)")

    colors = {headline_name: GREEN, "as_taught_5m": BLUE, "as_traded": PURPLE}
    order, seen = [], set()
    for name in [headline_name, "as_taught_5m", "as_traded"]:
        if name not in seen:
            seen.add(name)
            order.append(name)
    for name in order:
        if name not in trades_by_config:
            continue
        path = _cumulative_daily_pnl(trades_by_config[name], all_days)
        ax.plot(x, path.to_numpy(), color=colors.get(name, ORANGE), linewidth=2.0, label=name)

    ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
    tick_stride = max(len(all_days) // 8, 1)
    ax.set_xticks(x[::tick_stride])
    ax.set_xticklabels([d.strftime("%Y-%m") for d in all_days[::tick_stride]], rotation=20, ha="right", fontsize=9, color=TEXT)
    ax.set_ylabel("cumulative net PnL ($)")
    ax.set_title("Equity curves", color=TEXT, fontsize=16, fontweight="bold")
    _style_legend(ax)
    fig.tight_layout()
    return fig


# ---------- Phase 6's own prep step: real trade logs + null paths ----------


def compute_equity_curve_data(
    df_1m: pd.DataFrame,
    symbol: str,
    reference_configs: dict[str, StrategyConfig],
    named_configs: dict[str, StrategyConfig],
    n_null_iterations: int = 1000,
    seed: int | None = None,
) -> dict:
    """Real trade logs for the headline config (best_realistic_survivor if
    the funnel produced one, else as_taught_5m) plus as_taught_5m and
    as_traded, and the headline config's own random-entry null equity
    paths (real per-iteration paths, not a stat-derived band). Run once,
    after run_parts_1_through_7 returns."""
    headline_name = _select_headline_name(reference_configs)
    if headline_name is None:
        headline_name = "as_taught_5m"

    configs = {"as_taught_5m": named_configs["as_taught_5m"], "as_traded": named_configs["as_traded"]}
    if headline_name not in configs:
        configs[headline_name] = reference_configs.get(headline_name, named_configs["as_taught_5m"])

    all_days = all_session_dates(df_1m)
    trades_by_config = {}
    for name, config in configs.items():
        _, _, trades, _ = run_config(df_1m, config, symbol)
        trades_by_config[name] = trades

    _, null_paths = random_entry_null_distribution(
        df_1m, configs[headline_name], symbol, trades_by_config[headline_name],
        n_iterations=n_null_iterations, seed=seed, return_paths=True,
    )

    return {
        "headline_name": headline_name,
        "trades_by_config": trades_by_config,
        "all_days": all_days,
        "null_paths": null_paths,
    }


# ---------- HTML assembly ----------

_DASHBOARD_CSS = """
:root { color-scheme: dark; }
body { background:#0d1117; color:#c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; margin:0; padding:32px 16px; }
h1 { text-align:center; font-size:28px; margin-bottom:4px; }
.subtitle { text-align:center; color:#8b949e; margin-bottom:12px; font-size:14px; }
.caveat { text-align:center; color:#f85149; margin-bottom:24px; font-size:13px; }
.panel { max-width:1400px; margin:0 auto 40px auto; background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; }
.panel h2 { font-size:16px; margin:0 0 12px 4px; color:#c9d1d9; }
.panel img { width:100%; height:auto; border-radius:6px; display:block; }
footer { text-align:center; color:#8b949e; font-size:12px; margin-top:24px; }
"""


def _b64_image(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _write_index_html(output_dir: Path, panel_paths: list[Path], caveat: str = "") -> Path:
    by_name = {p.name: p for p in panel_paths}
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>ICT Silver Bullet -- Results Dashboard</title>",
        f"<style>{_DASHBOARD_CSS}</style></head><body>",
        "<h1>ICT Silver Bullet -- Results Dashboard</h1>",
        "<div class='subtitle'>NQ is the headline; ES is the validation story.</div>",
    ]
    if caveat:
        parts.append(f"<div class='caveat'>{caveat}</div>")
    for idx in sorted(PANEL_TITLES):
        filename = f"panel_{idx:02d}_{PANEL_SLUGS[idx]}.png"
        path = by_name.get(filename)
        if path is None:
            continue
        b64 = _b64_image(path)
        parts.append(
            f"<div class='panel'><h2>{idx}. {PANEL_TITLES[idx]}</h2>"
            f"<img src='data:image/png;base64,{b64}' alt='{PANEL_TITLES[idx]}'></div>"
        )
    parts.append("<footer>Generated by ict_lab.engine.dashboard (Phase 6)</footer>")
    parts.append("</body></html>")
    html_path = Path(output_dir) / "index.html"
    html_path.write_text("".join(parts))
    return html_path


# ---------- orchestration ----------


def build_dashboard(
    result: dict, equity_data: dict, output_dir: Path = DASHBOARD_DIR, ablation_symbol: str = "NQ", caveat: str = ""
) -> list[Path]:
    """Wires a Phase 5 `result` dict (as returned by
    sweep_orchestrator.run_parts_1_through_7) plus Phase 6's own
    equity_data (compute_equity_curve_data) into all 13 panels: each
    exported standalone as a 1920x1080 PNG, plus a single self-contained
    index.html embedding all of them."""
    output_dir = Path(output_dir)
    written: list[Path] = []

    def _save(idx: int, fig: Figure) -> None:
        written.append(save_panel_png(fig, output_dir / f"panel_{idx:02d}_{PANEL_SLUGS[idx]}.png"))

    summary = result["summary"]
    funnel_result = result["funnel_result"]
    es_df = result["es_validation"]
    null_results = result["null_results"]
    ablation_df = result["ablation"]
    per_year = result["per_year"]
    era_results = result["era_split"]
    discretion_reports = result["discretion_premium"]
    slippage = result["slippage_sensitivity"]
    named_rows = result["named_rows"]

    es_cross = None
    if not es_df.empty and "source" in es_df.columns:
        es_cross = int(es_df.loc[es_df["source"] == "survivor", "cross_symbol_survivor"].sum())
    _save(1, panel_funnel(funnel_result["counts"], es_cross))

    headline_null_name = _select_headline_name(null_results)
    headline_null_dist = (
        null_results[headline_null_name]["random_entry_same_windows"]["distribution"]
        if headline_null_name is not None
        else None
    )
    _save(2, panel_sharpe_distribution(summary["net_sharpe"].to_numpy(dtype=float), headline_null_dist, "random-entry null"))

    named_points = {r["config_name"]: r["trades_per_year"] for r in named_rows if r["symbol"] == "NQ"}
    _save(3, panel_frequency_spectrum(summary["trades_per_year"].to_numpy(dtype=float), named_points))

    _save(4, panel_ablation_ladder(ablation_df, ablation_symbol))

    if headline_null_name is not None:
        entry = null_results[headline_null_name]
        null_dists = {k: v["distribution"] for k, v in entry.items() if isinstance(v, dict) and "distribution" in v}
        _save(5, panel_null_comparison(entry["real_net_sharpe"], null_dists, headline_null_name))
    else:
        _save(5, panel_null_comparison(float("nan"), {}, ""))

    population = funnel_result["population"]
    survivors = population[population["survivor"]] if "survivor" in population.columns else population.iloc[0:0]
    if not survivors.empty and not es_df.empty:
        es_survivor_rows = es_df[es_df["source"] == "survivor"][["config_hash", "net_sharpe"]].rename(
            columns={"net_sharpe": "es_net_sharpe"}
        )
        merged = survivors[["config_hash", "config_name", "net_sharpe"]].rename(
            columns={"net_sharpe": "nq_net_sharpe"}
        ).merge(es_survivor_rows, on="config_hash", how="inner")
    else:
        merged = pd.DataFrame(columns=["config_hash", "config_name", "nq_net_sharpe", "es_net_sharpe"])
    _save(6, panel_nq_es_scatter(merged))

    _save(7, panel_per_year_heatmap(per_year))
    _save(8, panel_era_split(era_results))

    discretion_headline = _select_headline_name(discretion_reports)
    _save(9, panel_discretion_premium(discretion_reports, discretion_headline))

    _save(10, panel_win_rate(named_rows))
    _save(11, panel_cost_sensitivity(slippage))

    ambiguous_rows = [r for r in named_rows if r["symbol"] == "NQ"]
    _save(12, panel_ambiguous_bar_pct(ambiguous_rows))

    _save(
        13,
        panel_equity_curves(
            equity_data["trades_by_config"], equity_data["all_days"], equity_data["headline_name"], equity_data.get("null_paths")
        ),
    )

    html_path = _write_index_html(output_dir, written, caveat=caveat)
    written.append(html_path)
    return written
