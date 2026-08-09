"""Turns Phase 2's detectors into the canonical setup sequence: eligibility
-> bias gate -> sweep -> MSS -> optional displacement -> every
direction-matching FVG after that chain (Phase 4 item 1: not just the
first). Each window in config.windows is evaluated independently within a
session (Phase 4 item 2).

Interpretive choices worth being explicit about, since the source spec
describes this stage list in a couple of sentences without pinning down
the exact mechanics:
- Each stage that resolves a direction narrows the candidate set (starting
  at {bullish, bearish}) and advances a reference_point that the next stage
  must search strictly after -- this keeps the whole chain naturally
  ordered in time and prevents e.g. an MSS "confirming" a sweep that hasn't
  happened yet.
- displacement is a pure gate (existence of a qualifying bar after
  reference_point, direction-agnostic) rather than a check that the
  specific FVG's own formation bar is a displacement bar -- the spec lists
  it as a step in a sequence of filters, not as a property of the FVG
  itself.
- The bias/sweep/MSS/displacement chain is resolved ONCE per session+window
  (establishing a direction and a reference_point); every direction-matching
  FVG after that single chain becomes its own signal, rather than
  re-running sweep/MSS detection per FVG -- matching "every direction-
  matching FVG following any qualifying sweep/displacement chain."
- Signals here are candidates only, emitted without any cap: "signals must
  emit EVERY valid setup." Enforcing max_trades_per_window and skipping
  setups that land while a position is still open is execution.py's job,
  since only it knows which setups actually filled.
- window bounds are inclusive; stage-to-stage progression is exclusive
  (strictly after the previous stage's reference point).
"""
from __future__ import annotations

import pandas as pd

from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.configs.sweep_universe import resolve_sweep_universe
from ict_lab.data.sessions import add_session_columns
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.features.bias import project_bias
from ict_lab.features.mss import detect_mss

SIGNAL_COLUMNS = [
    "session_date",
    "window",
    "direction",
    "setup_at",
    "fvg_timeframe",
    "fvg_top",
    "fvg_bottom",
    "fvg_midpoint",
    "sweep_level_type",
    "sweep_level_price",
    "sweep_confirmed_at",
    "mss_broken_at",
    "displacement_at",
    "bias_value",
]
NO_SIGNAL_COLUMNS = ["session_date", "window", "reason"]
_DIRECTIONS = ("bullish", "bearish")


def generate_signals(
    store: FeatureStore, config: StrategyConfig, tick_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (signals, no_signals). Every session+window with bars gets
    exactly one no_signals row (if the chain failed) or one-or-more signals
    rows (every direction-matching FVG found)."""
    with_sessions = add_session_columns(store.df_1m)

    signals, no_signals = [], []
    for window in config.windows:
        scoped = with_sessions[with_sessions[window]]
        session_dates = sorted(scoped["session_date"].unique())
        for session_date in session_dates:
            bars = scoped[scoped["session_date"] == session_date]
            if bars.empty:
                continue
            kind, payload = _evaluate_window(
                store, config, tick_size, window, session_date, bars.index.min(), bars.index.max()
            )
            if kind == "signal":
                signals.extend(payload)
            else:
                no_signals.append({"session_date": session_date, "window": window, "reason": payload})

    signals_df = (
        pd.DataFrame(signals, columns=SIGNAL_COLUMNS) if signals else pd.DataFrame(columns=SIGNAL_COLUMNS)
    )
    no_signals_df = (
        pd.DataFrame(no_signals, columns=NO_SIGNAL_COLUMNS)
        if no_signals
        else pd.DataFrame(columns=NO_SIGNAL_COLUMNS)
    )
    return signals_df, no_signals_df


def _evaluate_window(
    store: FeatureStore,
    config: StrategyConfig,
    tick_size: float,
    window: str,
    session_date: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[str, list[dict] | str]:
    candidate = set(_DIRECTIONS)
    reference_point = window_start
    bias_value = None
    sweep_row = None
    mss_row = None
    displacement_at = None

    if config.bias_method != "none":
        updates = store.bias_updates(
            config.bias_method,
            timeframe=config.bias_timeframe,
            swing_n=config.bias_swing_n,
            ma_period=config.bias_ma_period,
            window=window,
        )
        bias_value = project_bias(updates, pd.DatetimeIndex([window_start])).iloc[0]
        if bias_value == "none":
            return "no_signal", "bias_gate"
        candidate &= {bias_value}

    if config.sweep_required:
        level_types = config.sweep_level_types or resolve_sweep_universe(config.sweep_universe, window)
        sweeps = store.sweeps(
            config.swing_n,
            level_types,
            config.sweep_k,
            config.sweep_min_penetration_ticks,
            tick_size,
            swing_15m_n=config.swing_15m_n,
        )
        in_window = sweeps[(sweeps["confirmed_at"] >= window_start) & (sweeps["confirmed_at"] <= window_end)]
        implied = in_window["swept_direction"].map({"low": "bullish", "high": "bearish"})
        in_window = in_window[implied.isin(candidate)]
        if in_window.empty:
            return "no_signal", "no_sweep"
        sweep_row = in_window.sort_values("confirmed_at", kind="mergesort").iloc[0]
        candidate &= {"bullish" if sweep_row["swept_direction"] == "low" else "bearish"}
        reference_point = sweep_row["confirmed_at"]

    if config.mss_required:
        swings = store.swings(config.swing_n)
        hits = []
        for direction in sorted(candidate):
            hit = detect_mss(store.df_1m, swings, reference_point, direction, config.mss_break_style)
            if hit is not None and window_start <= hit["broken_at"] <= window_end:
                hits.append(hit)
        if not hits:
            return "no_signal", "no_mss"
        mss_row = min(hits, key=lambda h: h["broken_at"])
        candidate &= {mss_row["direction"]}
        reference_point = mss_row["broken_at"]

    if config.displacement_required:
        disp = store.displacement(config.displacement_atr_mult)
        in_window = disp[(disp.index > reference_point) & (disp.index <= window_end)]
        flagged = in_window[in_window]
        if flagged.empty:
            return "no_signal", "no_displacement"
        displacement_at = flagged.index[0]
        reference_point = displacement_at

    fvgs = store.fvgs(config.fvg_timeframe, config.fvg_min_size_points, config.fvg_min_size_atr_mult)
    in_window = fvgs[
        (fvgs["knowable_at"] > reference_point)
        & (fvgs["knowable_at"] <= window_end)
        & (fvgs["direction"].isin(candidate))
    ]
    if in_window.empty:
        return "no_signal", "no_fvg"
    matching = in_window.sort_values("knowable_at", kind="mergesort")

    rows = [
        {
            "session_date": session_date,
            "window": window,
            "direction": fvg_row["direction"],
            "setup_at": fvg_row["knowable_at"],
            "fvg_timeframe": fvg_row["timeframe"],
            "fvg_top": fvg_row["gap_top"],
            "fvg_bottom": fvg_row["gap_bottom"],
            "fvg_midpoint": fvg_row["midpoint"],
            "sweep_level_type": sweep_row["level_type"] if sweep_row is not None else None,
            "sweep_level_price": sweep_row["level_price"] if sweep_row is not None else None,
            "sweep_confirmed_at": sweep_row["confirmed_at"] if sweep_row is not None else None,
            "mss_broken_at": mss_row["broken_at"] if mss_row is not None else None,
            "displacement_at": displacement_at,
            "bias_value": bias_value,
        }
        for _, fvg_row in matching.iterrows()
    ]
    return "signal", rows
