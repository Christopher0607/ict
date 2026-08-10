"""Phase 5 Part 3: null tests.

Reference set: as_taught_5m, as_taught_1m, as_traded, the best realistic NQ
survivor (highest net_sharpe among Part 2's survivors), and one
median-Sharpe NQ config -- selected by median net_sharpe across the FULL
sampled population, not just survivors, as a deliberate "typical, not
necessarily significant" contrast point against the best survivor.

a) RANDOM ENTRY, SAME WINDOWS: for each of the config's real trades, 1000
   independent substitutes -- a coin-flip direction at a uniform random
   minute inside that trade's own (window, session_date), with the SAME
   stop/target point-DISTANCES from entry as the real trade had (fills
   re-simulated via execution.py's own simulate_exit). "Stop distance and
   target structure copied from the matched config's trades" is read as
   "same absolute point distances": a random minute has no swept level or
   liquidity target of its own to re-derive, so the only thing that can be
   faithfully "copied" is how far away the real trade placed its stop and
   target. "Same trades-per-day count": exactly one substitute per real
   trade, preserving the real config's day-by-day trade cadence.
b) OTHER HOURS, SAME LOGIC: the config's actual bias/sweep/MSS/
   displacement/entry/stop/target logic, entirely unchanged, re-run one
   hour at a time across every non-overlapping 60-minute window spanning
   the full session OTHER than the config's own window(s) -- "other
   hours" wouldn't be a null contrast if it re-tested the config's own
   hour. ~20-22 windows survive exclusion depending on the config
   (spec: "about 21"). Multi-window configs get their one-window
   restriction here, per the spec, since each hour is tested alone.
c) SHUFFLED DIRECTION: the config's real trade log, each trade's direction
   flipped with p=0.5 (entry_at/entry_price unchanged; stop/target
   distances preserved but reflected to the new direction), fills
   re-simulated only. 1000 iterations.

All three report net Sharpe (dollar-based, the project's headline
risk-adjusted metric, computed the same "no-trade days as zero" way as
Part 1) per iteration/hour, forming a null distribution; the real config's
own net Sharpe is reported as a percentile of it.
"""
from __future__ import annotations

import contextlib
from dataclasses import replace

import numpy as np
import pandas as pd

from ict_lab.configs.strategy_config import COST_MODELS, StrategyConfig
from ict_lab.data import sessions as sessions_module
from ict_lab.data.sessions import WINDOWS as REAL_WINDOWS
from ict_lab.data.sessions import add_session_columns
from ict_lab.engine.execution import simulate_exit
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.engine.pipeline import all_session_dates, run_config
from ict_lab.engine.sweep_runner import annualized_sharpe, config_from_row, stats_from_trades

N_ITERATIONS = 1000
NULL_SEED = 20260809  # arbitrary fixed seed, documented for reproducibility -- not tuned to any result


def percentile_of_real_result(real_value: float, null_distribution: np.ndarray) -> float:
    """% of the null distribution AT OR BELOW the real value -- e.g. 97.0
    means the real result beat 97% of null draws. A null draw's NaN (zero
    tradeable substitutes that iteration) is dropped first; it carries no
    information either way."""
    if real_value is None or (isinstance(real_value, float) and np.isnan(real_value)):
        return float("nan")
    valid = null_distribution[~np.isnan(null_distribution)]
    if len(valid) == 0:
        return float("nan")
    return float((valid <= real_value).mean() * 100)


def _daily_pnl_series(values: np.ndarray, dates: list, all_days: pd.DatetimeIndex) -> pd.Series:
    """Per-day summed PnL reindexed to all_days (0 on no-trade days) --
    shared by _sharpe_from_daily (Sharpe of this series) and, with
    return_paths=True, random_entry_null_distribution's real per-iteration
    equity path (cumsum of this series)."""
    daily = pd.Series(values, index=pd.DatetimeIndex(dates)).groupby(level=0).sum()
    return daily.reindex(all_days, fill_value=0.0)


def _sharpe_from_daily(values: np.ndarray, dates: list, all_days: pd.DatetimeIndex) -> float:
    if len(values) == 0:
        return float("nan")
    return annualized_sharpe(_daily_pnl_series(values, dates, all_days))


def _simulate_null_trade_net_pnl(
    entry_at: pd.Timestamp,
    entry_price: float,
    stop_distance: float,
    target_distance: float | None,
    direction: str,
    session_bars: pd.DataFrame,
    hard_exit_at: pd.Timestamp,
    tick_size: float,
    tick_value: float,
    commission: float,
    stop_slippage_ticks: float,
) -> float:
    """Shared by (a) and (c): given an entry point/direction and stop/
    target point-distances (not levels -- a null substitute has none),
    places the stop/target on the correct side of `direction`, re-uses
    execution.py's own exit simulation, and returns the resulting net
    PnL (dollars, this symbol's real cost model)."""
    stop_price = entry_price - stop_distance if direction == "bullish" else entry_price + stop_distance
    target_price = (
        (entry_price + target_distance if direction == "bullish" else entry_price - target_distance)
        if target_distance is not None
        else None
    )
    exit_result = simulate_exit(
        direction, entry_at, entry_price, stop_price, target_price, None, hard_exit_at,
        session_bars, tick_size, stop_slippage_ticks,
    )
    price_diff = (
        exit_result["exit_price"] - entry_price if direction == "bullish" else entry_price - exit_result["exit_price"]
    )
    gross_pnl = (price_diff / tick_size) * tick_value
    return gross_pnl - commission


def _hard_exit_at(config: StrategyConfig, session_bars: pd.DataFrame, window_bars: pd.DataFrame) -> pd.Timestamp:
    if config.hard_exit == "window_end":
        return window_bars.index.max()
    rth_bars = session_bars[session_bars["rth"]]
    return rth_bars.index.max() if not rth_bars.empty else window_bars.index.max()


def _null_trial_inputs(df_1m: pd.DataFrame, config: StrategyConfig, trades: pd.DataFrame) -> list[dict]:
    """One entry per real trade: everything (a)/(c) need to build a null
    substitute for it, precomputed once and shared across every
    iteration."""
    with_sessions = add_session_columns(df_1m)
    session_groups = {d: g for d, g in with_sessions.groupby("session_date")}

    inputs = []
    for _, t in trades.iterrows():
        session_bars = session_groups.get(t["session_date"])
        if session_bars is None:
            continue
        window_bars = session_bars[session_bars[t["window"]]]
        if window_bars.empty:
            continue
        stop_distance = abs(t["entry_price"] - t["stop_price"])
        if stop_distance <= 0:
            continue
        target_distance = abs(t["target_price"] - t["entry_price"]) if pd.notna(t["target_price"]) else None
        inputs.append(
            {
                "session_bars": session_bars,
                "window_bars": window_bars,
                "hard_exit_at": _hard_exit_at(config, session_bars, window_bars),
                "stop_distance": stop_distance,
                "target_distance": target_distance,
                "session_date": t["session_date"],
                "real_entry_at": t["entry_at"],
                "real_entry_price": float(t["entry_price"]),
                "real_direction": t["direction"],
            }
        )
    return inputs


def random_entry_null_distribution(
    df_1m: pd.DataFrame,
    config: StrategyConfig,
    symbol: str,
    trades: pd.DataFrame,
    n_iterations: int = N_ITERATIONS,
    seed: int | None = NULL_SEED,
    return_paths: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """return_paths=True additionally returns a (n_iterations, len(all_days))
    array of each iteration's own cumulative net-PnL path, reindexed to
    all_session_dates(df_1m) -- real per-substitute-trade paths (not a
    band derived after the fact from the Sharpe distribution's summary
    stats), used for Phase 6's equity-curve null shading. Default False
    is unchanged from before this option existed: every existing caller
    still gets back exactly one array."""
    all_days = all_session_dates(df_1m)
    if trades.empty:
        sharpes = np.full(n_iterations, np.nan)
        return (sharpes, np.full((n_iterations, len(all_days)), np.nan)) if return_paths else sharpes

    cost_model = COST_MODELS[symbol]
    rng = np.random.default_rng(seed)
    trial_inputs = _null_trial_inputs(df_1m, config, trades)
    if not trial_inputs:
        sharpes = np.full(n_iterations, np.nan)
        return (sharpes, np.full((n_iterations, len(all_days)), np.nan)) if return_paths else sharpes

    results = np.empty(n_iterations)
    paths = np.empty((n_iterations, len(all_days))) if return_paths else None
    for i in range(n_iterations):
        net_pnls, dates = [], []
        for trial in trial_inputs:
            window_bars = trial["window_bars"]
            entry_at = window_bars.index[rng.integers(0, len(window_bars))]
            entry_price = float(window_bars.loc[entry_at, "close"])
            direction = "bullish" if rng.random() < 0.5 else "bearish"
            net = _simulate_null_trade_net_pnl(
                entry_at, entry_price, trial["stop_distance"], trial["target_distance"], direction,
                trial["session_bars"], trial["hard_exit_at"], cost_model.tick_size, cost_model.tick_value,
                cost_model.commission_round_turn, cost_model.stop_slippage_ticks,
            )
            net_pnls.append(net)
            dates.append(trial["session_date"])
        daily = _daily_pnl_series(np.array(net_pnls), dates, all_days)
        results[i] = annualized_sharpe(daily)
        if return_paths:
            paths[i] = daily.cumsum().to_numpy()
    return (results, paths) if return_paths else results


def shuffled_direction_null_distribution(
    df_1m: pd.DataFrame,
    config: StrategyConfig,
    symbol: str,
    trades: pd.DataFrame,
    n_iterations: int = N_ITERATIONS,
    seed: int | None = NULL_SEED,
) -> np.ndarray:
    if trades.empty:
        return np.full(n_iterations, np.nan)

    cost_model = COST_MODELS[symbol]
    all_days = all_session_dates(df_1m)
    rng = np.random.default_rng(seed)
    trial_inputs = _null_trial_inputs(df_1m, config, trades)
    if not trial_inputs:
        return np.full(n_iterations, np.nan)

    flips = rng.random((n_iterations, len(trial_inputs))) < 0.5

    results = np.empty(n_iterations)
    for i in range(n_iterations):
        net_pnls, dates = [], []
        for j, trial in enumerate(trial_inputs):
            real_direction = trial["real_direction"]
            direction = ("bearish" if real_direction == "bullish" else "bullish") if flips[i, j] else real_direction
            net = _simulate_null_trade_net_pnl(
                trial["real_entry_at"], trial["real_entry_price"], trial["stop_distance"], trial["target_distance"],
                direction, trial["session_bars"], trial["hard_exit_at"], cost_model.tick_size, cost_model.tick_value,
                cost_model.commission_round_turn, cost_model.stop_slippage_ticks,
            )
            net_pnls.append(net)
            dates.append(trial["session_date"])
        results[i] = _sharpe_from_daily(np.array(net_pnls), dates, all_days)
    return results


@contextlib.contextmanager
def _temporary_window(name: str, start: str, end: str):
    """Registers one extra (start,end) window in sessions.WINDOWS for the
    duration of the context, restoring exactly the prior state afterward
    (even on exception). generate_signals/simulate_trades call
    add_session_columns internally with no way to pass window definitions
    through them, so this is the narrowest way to let Part 3b's "other
    hour" windows flow through the existing, already-tested pipeline
    without threading a new parameter through several modules. Not safe
    for concurrent use -- Part 3 runs sequentially over a handful of
    reference configs, never through the sweep's process pool."""
    original = dict(sessions_module.WINDOWS)
    sessions_module.WINDOWS[name] = (start, end)
    try:
        yield
    finally:
        sessions_module.WINDOWS.clear()
        sessions_module.WINDOWS.update(original)


def other_hour_windows(exclude_windows: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Every non-overlapping 60-minute window across the full session
    (18:00 ET -> 17:00 ET next day, wrapping past midnight), excluding
    whichever of the standard killzone/rth clock ranges are in
    exclude_windows. Named null_hour_00..null_hour_22; returns (name,
    start, end) triples."""
    excluded_ranges = {REAL_WINDOWS[w] for w in exclude_windows if w in REAL_WINDOWS}
    result = []
    for i in range(23):
        h = (18 + i) % 24
        start = f"{h:02d}:00"
        end = f"{(h + 1) % 24:02d}:00"
        if (start, end) in excluded_ranges:
            continue
        result.append((f"null_hour_{i:02d}", start, end))
    return result


def other_hours_null_distribution(df_1m: pd.DataFrame, config: StrategyConfig, symbol: str) -> np.ndarray:
    session_dates = all_session_dates(df_1m)
    # FeatureStore's caches (fvgs/swings/liquidity_levels/bias_updates) key
    # on detector parameters, never on window boundaries -- window scoping
    # happens later, inside generate_signals' per-window loop -- so one
    # store safely serves all ~20 "other hour" variants below instead of
    # rebuilding FVGs/swings/bias from scratch ~20 times over.
    store = FeatureStore(df_1m)
    results = []
    for name, start, end in other_hour_windows(config.windows):
        with _temporary_window(name, start, end):
            variant = replace(config, windows=(name,))
            _, _, trades, _ = run_config(df_1m, variant, symbol, store=store)
            row = stats_from_trades(variant, trades, session_dates)
        results.append(row["net_sharpe"])
    return np.array(results, dtype=float)


def select_reference_configs(
    population: pd.DataFrame, survivors: pd.DataFrame, named_configs: dict[str, StrategyConfig]
) -> dict[str, StrategyConfig]:
    refs = dict(named_configs)

    if not survivors.empty:
        best_row = survivors.loc[survivors["net_sharpe"].idxmax()]
        refs["best_realistic_survivor"] = config_from_row(best_row)

    if not population.empty:
        finite = population[population["net_sharpe"].notna()]
        if not finite.empty:
            median_value = finite["net_sharpe"].median()
            closest_idx = (finite["net_sharpe"] - median_value).abs().idxmin()
            refs["median_sharpe_config"] = config_from_row(population.loc[closest_idx])

    return refs


def run_null_tests(
    df_1m: pd.DataFrame,
    symbol: str,
    reference_configs: dict[str, StrategyConfig],
    n_iterations: int = N_ITERATIONS,
    seed: int | None = NULL_SEED,
    include_shuffled: bool = True,
) -> dict[str, dict]:
    """For every reference config: runs it for real, builds the null
    distribution(s), and reports the real net Sharpe as a percentile of
    each. include_shuffled=False supports "repeat (a) and (b) ... on ES"
    (shuffled direction is NQ-population-only in the spec)."""
    session_dates = all_session_dates(df_1m)
    results: dict[str, dict] = {}
    for name, config in reference_configs.items():
        _, _, trades, _ = run_config(df_1m, config, symbol)
        real_sharpe = stats_from_trades(config, trades, session_dates)["net_sharpe"]

        random_entry = random_entry_null_distribution(df_1m, config, symbol, trades, n_iterations, seed)
        other_hours = other_hours_null_distribution(df_1m, config, symbol)

        entry = {
            "real_net_sharpe": real_sharpe,
            "random_entry_same_windows": {
                "distribution": random_entry,
                "percentile": percentile_of_real_result(real_sharpe, random_entry),
            },
            "other_hours_same_logic": {
                "distribution": other_hours,
                "percentile": percentile_of_real_result(real_sharpe, other_hours),
            },
        }
        if include_shuffled:
            shuffled = shuffled_direction_null_distribution(df_1m, config, symbol, trades, n_iterations, seed)
            entry["shuffled_direction"] = {
                "distribution": shuffled,
                "percentile": percentile_of_real_result(real_sharpe, shuffled),
            }
        results[name] = entry
    return results
