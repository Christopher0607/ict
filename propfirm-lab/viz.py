"""Shared chart styling.

Palette is dataviz categorical slots 1-4, light mode. Validated with:
    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light
All checks pass. The contrast WARN on the aqua and yellow slots is discharged
the way the method requires: every line carries a direct label, and every
figure has a table of the same numbers in its findings document.
"""

from __future__ import annotations

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"


def style_axes(ax, title, subtitle, xlabel, ylabel):
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
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)


def place_labels(ax, entries, x, *, min_gap_frac=0.062):
    """Direct-label line ends, nudged apart so they never overlap.

    Direct labels are not decoration: two of the light-mode series sit below
    3:1 contrast on this surface, and the labels are what discharge that.
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

    for (_y, text, c), frac in zip(items, placed):
        ax.annotate(text, xy=(x, lo + frac * span), xycoords="data",
                    xytext=(8, 0), textcoords="offset points",
                    color=c, fontsize=9, weight="bold", va="center",
                    annotation_clip=False)
