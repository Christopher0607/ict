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

# Overnight sub-sessions, as ET clock windows.
ETH_WINDOW_NAMES = ("asia", "europe", "london", "all_eth")

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
    # Added 2026-08-22, before any ICT result existed, after the first seven
    # families returned no survivors. ict_lab implements this strategy already
    # and is correct, but is too slow to sweep -- its own frequency diagnostic
    # ran 60 minutes on the development window and produced no output. Giving
    # it the same grid, cost model and correction as everything else is the
    # only way its result is comparable.
    #
    # Adding it raises the trial count, and therefore the bar, for every family
    # including the seven already run. That is the correct behaviour and the
    # corrections are recomputed against the combined total.
    # --- second round, pre-registered 2026-08-22 before any of its results
    # existed. Chosen for standing in the literature, not to widen coverage:
    # the first round showed that adding parameterizations of the same idea
    # only raises the noise ceiling. The ETH families exist because the first
    # eight required is_rth and therefore never touched 71.6% of the bars.
    "eth_momentum": {
        "eth_window": ETH_WINDOW_NAMES,
        "lookback": (15, 30, 60),
        "threshold_atr": (0.5, 1.0, 2.0),
        "side": SIDES,
    },
    "eth_reversion": {
        "eth_window": ETH_WINDOW_NAMES,
        "lookback": (15, 30, 60),
        "threshold_atr": (0.5, 1.0, 2.0),
        "side": SIDES,
    },
    "eth_range_breakout": {
        "eth_window": ETH_WINDOW_NAMES,
        "lookback": (15, 30, 60),
        "side": SIDES,
    },
    "gap_trade": {
        "min_gap_atr": (0.5, 1.0, 2.0),
        "mode": ("fade", "follow"),
        "entry_window": ((0, 60), (0, 390)),
        "side": SIDES,
    },
    "turn_of_month": {
        "window_days": (1, 2, 3),
        "entry_window": ((0, 390), (30, 180)),
        "side": SIDES,
    },
    "day_of_week": {
        "weekday": (0, 1, 2, 3, 4),
        "entry_window": ((0, 390), (30, 180)),
        "side": ("long", "short"),
    },
    "compression_breakout": {
        "lookback": (15, 30, 60),
        "max_compression": (0.6, 0.8),
        "entry_window": ((0, 390), (30, 180)),
        "side": SIDES,
    },
    "overnight_level_break": {
        "entry_window": ((0, 60), (0, 390), (30, 180)),
        "side": SIDES,
    },
    "ict_silver_bullet": {
        "killzone": ("london", "ny_am", "ny_pm"),
        "sweep_lookback": (15, 30, 60),
        "require_mss": (True, False),
        "require_displacement": (True, False),
        "side": SIDES,
    },
}


@dataclass(frozen=True)
class Config:
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    stop_atr: float = 1.5
    target_r: float = 2.0
    # None means "hold until a barrier or the close", which is what every
    # round-1 and round-2 config did. The suffix is omitted in that case so
    # those names stay byte-identical to the ones already on disk.
    time_exit_bars: int | None = None

    @property
    def name(self) -> str:
        bits = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        tail = "" if self.time_exit_bars is None else f"t={self.time_exit_bars}"
        return f"{self.family}[{bits}]stop={self.stop_atr}R={self.target_r}{tail}"


# ---------------------------------------------------------------------------
# Round 3: time exits, and the model-confidence family
# ---------------------------------------------------------------------------
#
# Everything above was registered blind. This block was not, and saying so is
# the point of writing it down separately.
#
# The predictability audit found the model's edge concentrates sharply in its
# own confidence -- +$48.75 per trade in the top 19% against +$10.10 overall --
# and its only economically positive cell was a fixed 60-minute hold. Both the
# family below and the choice to add time exits come from having seen that.
# The horizon, the regime and the quantile range were picked with the result in
# view, so a pass here is a candidate for the holdout and not a finding. See
# research/search/model_signal.py and findings/05.
#
# A target 99x the stop is never reached in an intraday hold; it is how a
# stop-and-time-exit design is expressed without a special case in the engine.
NO_TARGET_R = 99.0

# Stop widths for the families that hold for a fixed time.
#
# The rest of the grid stops at 4.0, which is right for a bracket that resolves
# in minutes and wrong for a hold that lasts hours. `f.atr` is a ONE-MINUTE ATR,
# and a random walk covers about sqrt(N) of them over an N-bar hold -- 5.5 at
# 30 bars, 7.7 at 60, 11.0 at 120. A 1.5x stop on a 120-bar hold is not a risk
# limit, it is a coin flip on the first few minutes, and testing only those
# would answer "does a stop one-eighth the size of the move work" rather than
# anything about time exits.
#
# 8.0 and 16.0 are roughly one and two standard deviations of a 60-bar hold.
# This is arithmetic about the holding period, fixed before these families ran;
# no result was consulted to choose it.
HOLD_STOPS = (8.0, 16.0)

MODEL_GRID: dict[str, tuple] = {
    "horizon": (30, 60, 120),
    "regime": ("rth", "eth"),
    "quantile": (0.5, 0.8, 0.9, 0.95),
    "side": SIDES,
}
MODEL_STOPS = (1.5, 2.5, 4.0) + HOLD_STOPS

# Time-exit variants of the two families with the best round-1 gross
# expectancy. Not a new hypothesis about those families -- a check on whether
# the bracket exit was what buried them.
TIME_EXIT_FAMILIES = ("orb", "prior_day_break")
TIME_EXIT_BARS = (30, 60)
TIME_EXIT_STOPS = (1.5, 2.5) + HOLD_STOPS
TIME_EXIT_TARGETS = (2.0, NO_TARGET_R)


def _family_configs(family: str, axes: dict[str, tuple]) -> list[Config]:
    out: list[Config] = []
    keys = sorted(axes)
    for combo in itertools.product(*(axes[k] for k in keys)):
        params = dict(zip(keys, combo))
        window = params.pop("entry_window", (0, 390))
        params["entry_from"], params["entry_to"] = window
        for stop_atr in STOP_ATR_MULTS:
            for target_r in TARGET_R_MULTS:
                out.append(Config(family, params, stop_atr, target_r))
    return out


# The control for the model family: same execution, no model. Declared with the
# same caveat -- it was added after the confidence diagnostic, to answer whether
# the model beats a rule with nothing fitted in it at all.
LOW_VOL_GRID: dict[str, tuple] = {
    "max_rel_atr": (0.7, 0.85, 1.0),
    "side": SIDES,
}
LOW_VOL_STOPS = (1.5, 2.5, 4.0) + HOLD_STOPS
LOW_VOL_TARGETS = (2.0, NO_TARGET_R)
LOW_VOL_BARS = (60, 120)


def _low_vol_configs() -> list[Config]:
    out: list[Config] = []
    keys = sorted(LOW_VOL_GRID)
    for combo in itertools.product(*(LOW_VOL_GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        params["entry_from"], params["entry_to"] = 0, 390
        for bars in LOW_VOL_BARS:
            for stop_atr in LOW_VOL_STOPS:
                for target_r in LOW_VOL_TARGETS:
                    out.append(Config("low_vol_long", params, stop_atr, target_r,
                                      time_exit_bars=bars))
    return out


def _model_configs() -> list[Config]:
    out: list[Config] = []
    keys = sorted(MODEL_GRID)
    for combo in itertools.product(*(MODEL_GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        for stop_atr in MODEL_STOPS:
            # The hold is the horizon the model was fit to predict. Pairing a
            # 60-bar forecast with a 30-bar hold would test something else.
            out.append(Config("model_confidence", params, stop_atr,
                              NO_TARGET_R, time_exit_bars=params["horizon"]))
    return out


def _time_exit_configs() -> list[Config]:
    out: list[Config] = []
    for family in TIME_EXIT_FAMILIES:
        axes = GRID[family]
        keys = sorted(axes)
        for combo in itertools.product(*(axes[k] for k in keys)):
            params = dict(zip(keys, combo))
            window = params.pop("entry_window", (0, 390))
            params["entry_from"], params["entry_to"] = window
            for bars in TIME_EXIT_BARS:
                for stop_atr in TIME_EXIT_STOPS:
                    for target_r in TIME_EXIT_TARGETS:
                        out.append(Config(family, params, stop_atr, target_r,
                                          time_exit_bars=bars))
    return out


def enumerate_configs() -> list[Config]:
    """Every configuration in the grid, in a deterministic order."""
    out: list[Config] = []
    for family, axes in GRID.items():
        out.extend(_family_configs(family, axes))
    out.extend(_model_configs())
    out.extend(_time_exit_configs())
    out.extend(_low_vol_configs())
    return out


def grid_size() -> dict[str, int]:
    """Per-family and total trial counts. This is the `n` for every correction.

    Round 3's additions are counted here like everything else. They raise the
    bar for every result in the file, including the ones from rounds 1 and 2 --
    which is the correct behaviour and the reason searching harder is not free.
    """
    counts: dict[str, int] = {}
    for family, axes in GRID.items():
        n = 1
        for values in axes.values():
            n *= len(values)
        counts[family] = n * len(STOP_ATR_MULTS) * len(TARGET_R_MULTS)

    n_model = len(MODEL_STOPS)
    for values in MODEL_GRID.values():
        n_model *= len(values)
    counts["model_confidence"] = n_model

    n_te = len(TIME_EXIT_BARS) * len(TIME_EXIT_STOPS) * len(TIME_EXIT_TARGETS)
    for family in TIME_EXIT_FAMILIES:
        n = 1
        for values in GRID[family].values():
            n *= len(values)
        counts[f"{family}+time_exit"] = n * n_te

    n_lv = len(LOW_VOL_BARS) * len(LOW_VOL_STOPS) * len(LOW_VOL_TARGETS)
    for values in LOW_VOL_GRID.values():
        n_lv *= len(values)
    counts["low_vol_long"] = n_lv

    counts["TOTAL"] = sum(v for k, v in counts.items() if k != "TOTAL")
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
