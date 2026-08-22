"""When should you take the first payout?

On Lucid Flex a payout request snaps the max loss limit up to $50,100 no
matter where the trailing floor had reached. Taking $500 out at $51,000 of
equity leaves $400 of room instead of $1,900. So "withdraw as soon as you are
allowed to" is not obviously right, and the answer is worth computing.

Apex is the control: it has no floor snap, so if the scan shows a strong
optimum there too, the effect is something other than the snap.

    python run_payout_timing.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paths import TradeModel
from rules import get_ruleset
from sim.study import study_lifecycle
from viz import GRID, INK_2, MUTED, SERIES, SURFACE, place_labels, style_axes

OUT = Path("findings")
FIGS = OUT / "figures"

FIRMS = ["lucid_50k_flex", "apex_50k_intraday"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--expectancy", type=float, default=0.10,
                    help="after-cost R per trade to run the scan at")
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    n = 1_000 if args.quick else 6_000
    # $1,000 is the floor: at a 50% payout rule that is the smallest profit
    # that yields the $500 minimum payout.
    grid = [1_000, 1_500, 2_000, 2_500, 3_000, 4_000, 5_000, 6_000,
            8_000, 10_000, 13_000, 16_000, 20_000]

    OUT.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    model = TradeModel(expectancy_r=args.expectancy)
    out: dict[str, dict] = {}
    for key in FIRMS:
        rs = get_ruleset(key)
        means, pprof, los, his, npay = [], [], [], [], []
        for thr in grid:
            st = study_lifecycle(model, rs, n_accounts=n, eval_days=200,
                                 funded_days=250, seed=args.seed,
                                 withdraw_at_profit=float(thr))
            lo, hi = st.mean_net_ci
            means.append(st.mean_net)
            los.append(lo)
            his.append(hi)
            pprof.append(st.prob_profitable)
            npay.append(float(np.mean(st.payout_count)))
        best = int(np.argmax(means))
        out[str(rs)] = {
            "key": key, "mean_net": means, "ci_lo": los, "ci_hi": his,
            "prob_profitable": pprof, "mean_payouts": npay,
            "best_threshold": grid[best], "best_mean": means[best],
        }
        print(f"{rs}: best threshold ${grid[best]:,} -> ${means[best]:,.0f}")

    payload = {"grid": grid, "expectancy_r": args.expectancy,
               "n_accounts": n, "seed": args.seed, "firms": out}
    (OUT / "payout_timing.json").write_text(json.dumps(payload, indent=2))
    plot(payload)
    print(f"\nwrote {OUT/'payout_timing.json'} and {FIGS/'payout_timing.png'}")


def plot(p: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    grid = p["grid"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor=SURFACE)

    lab1, lab2 = [], []
    for i, (label, d) in enumerate(p["firms"].items()):
        c = SERIES[i]
        name = label.split(" (")[0]
        ax1.fill_between(grid, d["ci_lo"], d["ci_hi"], color=c, alpha=0.13, lw=0)
        ax1.plot(grid, d["mean_net"], lw=2, color=c, solid_capstyle="round")
        lab1.append((d["mean_net"][-1], name, c))
        ax1.plot([d["best_threshold"]], [d["best_mean"]], "o", ms=8, color=c,
                 mec=SURFACE, mew=2, zorder=5)

        ax2.plot(grid, d["prob_profitable"], lw=2, color=c, solid_capstyle="round")
        lab2.append((d["prob_profitable"][-1], name, c))

    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax1.set_xlim(grid[0], grid[-1] * 1.28)
    style_axes(ax1, "When to take the first payout",
               f"Mean net $ per account, 95% CI band, at "
               f"{p['expectancy_r']:+.2f}R after costs",
               "Hold the payout until cycle profit reaches...",
               "Net $ per account bought")
    place_labels(ax1, lab1, grid[-1])

    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax2.set_xlim(grid[0], grid[-1] * 1.28)
    style_axes(ax2, "...and what it does to your odds",
               "P(this account ever ends up ahead)",
               "Hold the payout until cycle profit reaches...",
               "P(account is profitable)")
    place_labels(ax2, lab2, grid[-1])

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.005, 0.014,
             f"{p['n_accounts']:,} accounts per point. Dots mark the best "
             "threshold. Lucid snaps the max loss limit to $50,100 on a payout "
             "request; Apex has no such rule and is the control.",
             color=MUTED, fontsize=8)
    fig.savefig(FIGS / "payout_timing.png", dpi=170, facecolor=SURFACE)


if __name__ == "__main__":
    main()
