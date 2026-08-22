"""The pre-registered search grid, and the trial accounting that goes with it.

**This file is committed before the search runs.** That is the whole point. A
grid chosen after glancing at results is not a grid, it is a selection, and
every significance threshold computed from it is meaningless. If the grid
changes, it changes in a commit with a stated reason, before the next run.

The counting matters as much as the content. Every configuration enumerated
here is a hypothesis test, and the Benjamini-Hochberg and Deflated-Sharpe
corrections downstream take their `n` from this file. Searching harder does not
make an edge easier to find -- it raises the bar for everything, which is the
correct behaviour and the reason a "comprehensive search" is not free.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

# Shared execution parameters, applied to every family.
STOP_ATR_MULTS = (1.0, 1.5, 2.5, 4.0)
TARGET_R_MULTS = (1.0, 1.5, 2.0, 3.0)
SIDES = ("long", "short", "both")

# RTH is 390 minutes. Windows are (from, to) in minutes after the open.
ENTRY_WINDOWS = (
    (0, 390),     # all day
    (0, 60),      # opening hour
    (30, 180),    # morning after the open settles
    (180, 330),   # midday
    (300, 390),   # closing stretch
)

GRID: dict[str, dict[str, tuple]] = {
    "orb": {
        "or_minutes": (15, 30, 60),
        "entry_window": ENTRY_WINDOWS,
        "side": SIDES,
    },
    "momentum": {
        "lookback": (5, 15, 30, 60),
        "threshold_atr": (0.5, 1.0, 2.0),
        "entry_window": ENTRY_WINDOWS,
        "side": SIDES,
    },
    "mean_reversion": {
        "lookback": (5, 15, 30, 60),
        "threshold_atr": (0.5, 1.0, 2.0),
        "entry_window": ENTRY_WINDOWS,
        "side": SIDES,
    },
    "range_breakout": {
        "lookback": (5, 15, 30, 60),
        "entry_window": ENTRY_WINDOWS,
        "side": SIDES,
    },
    "vwap_reversion": {
        "threshold_atr": (0.5, 1.0, 2.0),
        "entry_window": ENTRY_WINDOWS,
        "side": SIDES,
    },
    "prior_day_break": {
        "entry_window": ENTRY_WINDOWS,
        "side": SIDES,
    },
    "time_of_day": {
        "entry_minute": (0, 30, 60, 120, 180, 240, 300, 360),
        "side": ("long", "short"),
    },
}


@dataclass(frozen=True)
class Config:
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    stop_atr: float = 1.5
    target_r: float = 2.0

    @property
    def name(self) -> str:
        bits = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.family}[{bits}]stop={self.stop_atr}R={self.target_r}"


def enumerate_configs() -> list[Config]:
    """Every configuration in the grid, in a deterministic order."""
    out: list[Config] = []
    for family, axes in GRID.items():
        keys = sorted(axes)
        for combo in itertools.product(*(axes[k] for k in keys)):
            params = dict(zip(keys, combo))
            window = params.pop("entry_window", (0, 390))
            params["entry_from"], params["entry_to"] = window
            for stop_atr in STOP_ATR_MULTS:
                for target_r in TARGET_R_MULTS:
                    out.append(Config(family, params, stop_atr, target_r))
    return out


def grid_size() -> dict[str, int]:
    """Per-family and total trial counts. This is the `n` for every correction."""
    counts: dict[str, int] = {}
    for family, axes in GRID.items():
        n = 1
        for values in axes.values():
            n *= len(values)
        counts[family] = n * len(STOP_ATR_MULTS) * len(TARGET_R_MULTS)
    counts["TOTAL"] = sum(counts.values())
    return counts


# ---------------------------------------------------------------------------
# The decision rule, fixed before any result exists
# ---------------------------------------------------------------------------
#
# Written down in advance so it cannot be relaxed once the numbers come back
# looking disappointing -- which, given the prior on simple rules over a liquid
# index future, is the expected outcome. Changing any of these after seeing
# results invalidates every significance figure computed from them.

# A config below this many trades has a standard error too wide to say
# anything, however good its point estimate looks.
MIN_TRADES = 200

# Benjamini-Hochberg false discovery rate across ALL trials in the grid.
FDR_Q = 0.10

# The economic bar, from findings/02: the after-cost expectancy at which the
# MEDIAN Lucid 50k Flex account turns profitable. Beating zero is not the
# target; beating this is.
TARGET_EXPECTANCY_R = 0.185

# A strategy also has to trade often enough to reach a profit target inside an
# evaluation. Roughly one trade per session over the development window.
MIN_TRADES_PER_YEAR = 200

# Survivors must clear every gate below, in order, before the holdout is opened.
DECISION_GATES = (
    "trades >= MIN_TRADES",
    "trades_per_year >= MIN_TRADES_PER_YEAR",
    "BH-FDR significant at q=0.10 across all grid trials",
    "Deflated Sharpe > 0 accounting for the full trial count",
    "point estimate expectancy_r >= TARGET_EXPECTANCY_R",
)
