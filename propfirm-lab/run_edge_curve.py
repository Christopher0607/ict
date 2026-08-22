"""Produce the edge-requirement curve -- this project's north star.

Answers, in dollars: *how much edge do you need before buying a prop-firm
account is a positive-expectancy purchase?*

Needs no market data. It is a question about rules and about the shape of a
P&L path, and both of those are fully specified before any price series is
involved. Run it, read the break-even column, and you have the bar that any
strategy must clear to be worth funding.

    python run_edge_curve.py --quick     # ~1 min, coarse
    python run_edge_curve.py             # the real run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paths import TradeModel
from rules import DrawdownType, get_ruleset, list_rulesets
from sim.study import study_eval, study_lifecycle

OUT_DIR = Path("findings")
FIG_DIR = OUT_DIR / "figures"

# dataviz categorical slots 1-4, light mode. Validated:
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light
# All checks pass; the contrast WARN on aqua/yellow is discharged by direct
# labels on every line plus the table in the findings document.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"


def base_model(expectancy: float, args) -> TradeModel:
    return TradeModel(
        r_dollars=args.r_dollars,
        expectancy_r=expectancy,
        payoff_r=args.payoff_r,
        trades_per_day=args.trades_per_day,
        mae_frac=0.5,
        mfe_frac=0.5,
    )


def breakeven(xs: list[float], ys: list[float]) -> float | None:
    """Linear interpolation of the first upward zero crossing."""
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 <= 0 <= y1 and y1 != y0:
            t = -y0 / (y1 - y0)
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--r-dollars", type=float, default=250.0)
    ap.add_argument("--payoff-r", type=float, default=2.0)
    ap.add_argument("--trades-per-day", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    n_eval = 1_500 if args.quick else 6_000
    n_life = 800 if args.quick else 4_000
    grid = np.round(np.arange(-0.20, 0.41, 0.05 if not args.quick else 0.10), 3)

    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # -- Panel A: the rule effect, holding the trader constant ---------------
    apex = get_ruleset("apex_50k_intraday")
    dd_labels = {
        DrawdownType.STATIC: "Static",
        DrawdownType.EOD_TRAILING: "EOD trailing",
        DrawdownType.INTRADAY_TRAILING: "Intraday trailing",
    }
    panel_a: dict[str, list[float]] = {}
    for dd, label in dd_labels.items():
        rs = apex.with_drawdown_type(dd)
        panel_a[label] = [
            study_eval(base_model(e, args), rs, n_paths=n_eval, n_days=300,
                       seed=args.seed).pass_rate
            for e in grid
        ]
        print(f"[A] {label:18s} done")

    # -- Panel B: the money question, per real firm --------------------------
    panel_b: dict[str, dict] = {}
    for key in ["apex_50k_intraday", "topstep_50k", "mffu_50k"]:
        rs = get_ruleset(key)
        means, los, his, passes, pprof, meds = [], [], [], [], [], []
        for e in grid:
            st = study_lifecycle(
                base_model(e, args), rs,
                n_accounts=n_life, eval_days=200, funded_days=250, seed=args.seed,
            )
            lo, hi = st.mean_net_ci
            means.append(st.mean_net)
            los.append(lo)
            his.append(hi)
            passes.append(st.pass_rate)
            pprof.append(st.prob_profitable)
            meds.append(st.median_net)
        panel_b[str(rs)] = {
            "key": key, "mean_net": means, "ci_lo": los, "ci_hi": his,
            "pass_rate": passes, "prob_profitable": pprof, "median_net": meds,
            "breakeven": breakeven(list(grid), means),
            "breakeven_median": breakeven(list(grid), meds),
            "edge_for_coinflip": breakeven(
                list(grid), [p - 0.5 for p in pprof]
            ),
        }
        print(f"[B] {rs} done  breakeven={panel_b[str(rs)]['breakeven']}")

    payload = {
        "grid": [float(g) for g in grid],
        "panel_a": panel_a,
        "panel_b": panel_b,
        "params": {
            "r_dollars": args.r_dollars, "payoff_r": args.payoff_r,
            "trades_per_day": args.trades_per_day,
            "n_eval_paths": n_eval, "n_lifecycle_accounts": n_life,
            "seed": args.seed,
        },
    }
    (OUT_DIR / "edge_curve.json").write_text(json.dumps(payload, indent=2))
    plot(payload)
    print(f"\nwrote {OUT_DIR/'edge_curve.json'} and {FIG_DIR/'edge_curve.png'}")


def _place_labels(ax, entries, x, *, min_gap_frac=0.062):
    """Direct-label line ends, nudged apart so they never overlap.

    Direct labels are not decoration here: two of the light-mode series sit
    below 3:1 contrast on this surface, so the labels are what discharge that
    (together with the tables in the findings doc).
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    items = sorted(entries, key=lambda t: t[0])
    placed: list[float] = []
    for y, _text, _c in items:
        frac = (y - lo) / span
        if placed and frac - placed[-1] < min_gap_frac:
            frac = placed[-1] + min_gap_frac
        placed.append(frac)

    # Pushing labels apart can drive the top one off the axes and into the
    # title. If it does, slide the whole stack back down.
    overflow = placed[-1] - (1.0 - min_gap_frac * 0.5)
    if overflow > 0:
        placed = [f - overflow for f in placed]
    for (y, text, c), frac in zip(items, placed):
        ax.annotate(
            text, xy=(x, y), xycoords="data",
            xytext=(8, 0), textcoords="offset points",
            color=c, fontsize=9, weight="bold", va="center",
            annotation_clip=False,
        )
        # Re-anchor to the de-collided position when it had to move.
        target = lo + frac * span
        if abs(target - y) > 1e-9:
            ax.annotate(
                "", xy=(x, y), xytext=(x, target), xycoords="data",
                textcoords="data", arrowprops=dict(arrowstyle="-", color=c,
                                                   lw=0.8, alpha=0.5),
                annotation_clip=False,
            )
            ax.texts[-2].set_position((8, 0))
            ax.texts[-2].xy = (x, target)


def plot(p: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    grid = p["grid"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.6), facecolor=SURFACE)

    def style(ax, title, subtitle, ylab):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, color=INK, fontsize=12.5, weight="bold", loc="left", pad=18)
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK_2,
                fontsize=9.5, va="bottom")
        ax.set_xlabel("Expectancy per trade, after costs (R)", color=INK_2, fontsize=9.5)
        ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)

    # Panel A -- pass rate by drawdown rule
    labels_a = []
    for i, (label, ys) in enumerate(p["panel_a"].items()):
        ax1.plot(grid, ys, lw=2, color=SERIES[i], solid_capstyle="round")
        labels_a.append((ys[-1], label, SERIES[i]))
    ax1.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax1.annotate("zero edge", (0, 0.02), xytext=(4, 0), textcoords="offset points",
                 color=MUTED, fontsize=8.5)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_ylim(0, 1.16)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xlim(grid[0], grid[-1] + 0.16)
    style(ax1, "The rule costs more than the skill",
          "P(pass eval) for one 50k account, same trader, three drawdown rules",
          "P(pass evaluation)")
    _place_labels(ax1, labels_a, grid[-1])

    # Panel B -- net dollars per account bought
    labels_b = []
    for i, (label, d) in enumerate(p["panel_b"].items()):
        c = SERIES[i]
        ax2.fill_between(grid, d["ci_lo"], d["ci_hi"], color=c, alpha=0.13, lw=0)
        ax2.plot(grid, d["mean_net"], lw=2, color=c, solid_capstyle="round")
        labels_b.append((d["mean_net"][-1], label.split(" (")[0], c))
        be = d["breakeven"]
        if be is not None:
            ax2.plot([be], [0], "o", ms=7, color=c, mec=SURFACE, mew=2, zorder=5)
    for i, (label, d) in enumerate(p["panel_b"].items()):
        ax2.plot(grid, d["median_net"], lw=1.4, color=SERIES[i], ls=(0, (4, 3)),
                 alpha=0.85)
    ax2.annotate("dashed = median account", (grid[0], 0), xytext=(2, -16),
                 textcoords="offset points", color=MUTED, fontsize=8.5)
    ax2.axhline(0, color=INK_2, lw=1.2)
    ax2.annotate("break-even", (grid[0], 0), xytext=(2, 5),
                 textcoords="offset points", color=INK_2, fontsize=8.5)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.set_xlim(grid[0], grid[-1] + 0.22)
    style(ax2, "What one account is actually worth",
          "Mean net $ per account bought (payouts - fees), 95% CI band",
          "Net $ per account bought")
    _place_labels(ax2, labels_b, grid[-1])

    # Panel C -- the skew. A positive mean is not a positive outcome.
    labels_c = []
    for i, (label, d) in enumerate(p["panel_b"].items()):
        c = SERIES[i]
        ax3.plot(grid, d["prob_profitable"], lw=2, color=c, solid_capstyle="round")
        labels_c.append((d["prob_profitable"][-1], label.split(" (")[0], c))
    ax3.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax3.annotate("coin flip", (grid[0], 0.5), xytext=(2, 5),
                 textcoords="offset points", color=MUTED, fontsize=8.5)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax3.set_ylim(0, 1.02)
    ax3.set_xlim(grid[0], grid[-1] + 0.22)
    style(ax3, "Why the mean lies",
          "P(this account ever ends up ahead) -- the mean rides on a thin tail",
          "P(account is profitable)")
    _place_labels(ax3, labels_c, grid[-1])

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.text(0.005, 0.012,
             f"Monte Carlo: {p['params']['n_eval_paths']:,} eval paths / "
             f"{p['params']['n_lifecycle_accounts']:,} accounts per point. "
             f"{p['params']['payoff_r']}R target, "
             f"{p['params']['trades_per_day']} trades/day, "
             f"${p['params']['r_dollars']:,.0f} risked per trade. "
             "No market data involved -- this is a property of the rules.",
             color=MUTED, fontsize=8)
    fig.savefig(FIG_DIR / "edge_curve.png", dpi=170, facecolor=SURFACE)


if __name__ == "__main__":
    main()
