"""Phase 5 Part 6: statistics and slices.

- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) for the best
  realistic config.
- Per-year table of average R and trade count for every survivor.
- Era split: 2010-2021 vs 2022-to-holdout-cutoff, for survivors and all
  three named configs.
- Roll-day sensitivity: survivor results with and without roll-day trades.
- Slippage sensitivity, analytic from trade logs (no re-simulation): stop
  slippage at 0/1/2/3 ticks, time-exit slippage at 0/1/2 ticks.

DSR granularity, since the spec names two different "Sharpe" roles that
need to agree on scale for the formula's Z-score to mean anything:
everything here -- the observed Sharpe being tested, the cross-config
Sharpe spread, and the skew/kurtosis -- is computed at the TRADE level
(mean/std of each config's own r_multiple sequence), not the project's
usual annualized-from-daily-PnL net_sharpe. The spec explicitly asks for
"trade-level skew and kurtosis"; using a daily-annualized Sharpe for SR_hat
against a trade-level sigma_hat_SR would compare two different scales.
Kurtosis is RAW (Pearson, normal=3), matching the paper's asymptotic
variance formula for Gaussian returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ict_lab.configs.strategy_config import COST_MODELS
from ict_lab.engine.pipeline import all_session_dates, run_config
from ict_lab.engine.sweep_runner import annualized_sharpe, config_from_row, r_multiples_for_hash

EULER_MASCHERONI = 0.5772156649015329
STOP_SLIPPAGE_TICKS_OPTIONS = (0, 1, 2, 3)
TIME_EXIT_SLIPPAGE_TICKS_OPTIONS = (0, 1, 2)
ERA_SPLIT_YEAR = 2022  # 2010-2021 vs 2022-to-holdout-cutoff


# ---------- Deflated Sharpe Ratio ----------


def sr_0(n_trials: int, cross_config_sharpe_std: float) -> float:
    """Expected maximum Sharpe ratio under the null (no skill) across
    n_trials independent trials, via Extreme Value Theory."""
    if n_trials <= 1 or np.isnan(cross_config_sharpe_std):
        return float("nan")
    z1 = scipy_stats.norm.ppf(1 - 1 / n_trials)
    z2 = scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(cross_config_sharpe_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def sharpe_estimator_std(sharpe: float, skew: float, kurtosis: float, n_observations: int) -> float:
    """Standard deviation of the Sharpe-ratio estimator itself, adjusted
    for skewness and RAW (Pearson) kurtosis of the underlying return
    sequence (Bailey & Lopez de Prado 2014, eq. 10)."""
    if n_observations <= 1:
        return float("nan")
    variance = (1 - skew * sharpe + ((kurtosis - 1) / 4) * sharpe**2) / (n_observations - 1)
    return float(np.sqrt(max(variance, 0.0)))


def deflated_sharpe_ratio(
    observed_sharpe: float, n_trials: int, cross_config_sharpe_std: float,
    trade_skew: float, trade_kurtosis: float, n_trades: int,
) -> dict:
    sr0 = sr_0(n_trials, cross_config_sharpe_std)
    sigma_sr = sharpe_estimator_std(observed_sharpe, trade_skew, trade_kurtosis, n_trades)
    if not sigma_sr or np.isnan(sigma_sr) or np.isnan(sr0):
        return {"sr_0": sr0, "sigma_sr": sigma_sr, "dsr": float("nan")}
    z = (observed_sharpe - sr0) / sigma_sr
    return {"sr_0": sr0, "sigma_sr": sigma_sr, "dsr": float(scipy_stats.norm.cdf(z))}


def _trade_level_sharpe(r_multiples: np.ndarray) -> float:
    if len(r_multiples) < 2:
        return float("nan")
    std = np.std(r_multiples, ddof=1)
    return float(np.mean(r_multiples) / std) if std > 0 else float("nan")


def deflated_sharpe_ratio_for_config(population: pd.DataFrame, best_config_hash: str, shard_dir) -> dict:
    """Runs the full DSR calculation for one config (intended: the best
    realistic survivor) against the population it was drawn from."""
    trade_level_sharpes = []
    for h in population["config_hash"]:
        r = r_multiples_for_hash(shard_dir, h)
        sharpe = _trade_level_sharpe(r)
        if not np.isnan(sharpe):
            trade_level_sharpes.append(sharpe)

    n_trials = len(trade_level_sharpes)
    cross_config_std = float(np.std(trade_level_sharpes, ddof=1)) if n_trials >= 2 else float("nan")

    best_r = r_multiples_for_hash(shard_dir, best_config_hash)
    n_trades = len(best_r)
    observed_sharpe = _trade_level_sharpe(best_r)
    skew = float(scipy_stats.skew(best_r)) if n_trades >= 2 else float("nan")
    kurtosis = float(scipy_stats.kurtosis(best_r, fisher=False)) if n_trades >= 2 else float("nan")

    result = {
        "n_trials": n_trials, "cross_config_sharpe_std": cross_config_std, "observed_sharpe": observed_sharpe,
        "skew": skew, "kurtosis": kurtosis, "n_trades": n_trades,
    }
    if np.isnan(observed_sharpe):
        result.update({"sr_0": float("nan"), "sigma_sr": float("nan"), "dsr": float("nan")})
        return result
    result.update(deflated_sharpe_ratio(observed_sharpe, n_trials, cross_config_std, skew, kurtosis, n_trades))
    return result


# ---------- shared light summary ----------


def _light_summary(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": float("nan"), "avg_r": float("nan"), "net_sharpe": float("nan")}
    daily = trades.groupby("session_date")["net_pnl"].sum().reindex(session_dates, fill_value=0.0)
    return {
        "trades": n,
        "win_rate": float((trades["r_multiple"] > 0).mean() * 100),
        "avg_r": float(trades["r_multiple"].mean()),
        "net_sharpe": annualized_sharpe(daily),
    }


# ---------- per-year table ----------


def per_year_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["year", "avg_r", "trades"])
    df = trades.copy()
    df["year"] = pd.DatetimeIndex(df["session_date"]).year
    out = df.groupby("year")["r_multiple"].agg(avg_r="mean", trades="count")
    return out.reset_index().sort_values("year", kind="mergesort").reset_index(drop=True)


def per_year_tables_for_survivors(df_1m: pd.DataFrame, symbol: str, survivors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {}
    for _, row in survivors.iterrows():
        config = config_from_row(row)
        _, _, trades, _ = run_config(df_1m, config, symbol)
        results[row["config_name"]] = per_year_table(trades)
    return results


# ---------- era split ----------


def era_split(trades: pd.DataFrame, session_dates: pd.DatetimeIndex, split_year: int = ERA_SPLIT_YEAR) -> dict:
    trade_years = pd.DatetimeIndex(trades["session_date"]).year if len(trades) else pd.Index([], dtype=int)
    era1_trades = trades[trade_years < split_year]
    era2_trades = trades[trade_years >= split_year]
    era1_days = session_dates[session_dates.year < split_year]
    era2_days = session_dates[session_dates.year >= split_year]
    return {
        f"2010_{split_year - 1}": _light_summary(era1_trades, era1_days),
        f"{split_year}_to_holdout_cutoff": _light_summary(era2_trades, era2_days),
    }


def era_split_for_configs(df_1m: pd.DataFrame, symbol: str, configs: dict) -> dict[str, dict]:
    session_dates = all_session_dates(df_1m)
    results = {}
    for name, config in configs.items():
        _, _, trades, _ = run_config(df_1m, config, symbol)
        results[name] = era_split(trades, session_dates)
    return results


# ---------- roll-day sensitivity ----------


def roll_day_sensitivity(trades: pd.DataFrame, session_dates: pd.DatetimeIndex) -> dict:
    without_roll = trades[~trades["is_roll_day"]] if len(trades) else trades
    return {"with_roll_days": _light_summary(trades, session_dates), "without_roll_days": _light_summary(without_roll, session_dates)}


def roll_day_sensitivity_for_survivors(df_1m: pd.DataFrame, symbol: str, survivors: pd.DataFrame) -> dict[str, dict]:
    session_dates = all_session_dates(df_1m)
    results = {}
    for _, row in survivors.iterrows():
        config = config_from_row(row)
        _, _, trades, _ = run_config(df_1m, config, symbol)
        results[row["config_name"]] = roll_day_sensitivity(trades, session_dates)
    return results


# ---------- slippage sensitivity (analytic, no re-simulation) ----------


def _apply_slippage_delta(
    trades: pd.DataFrame, exit_reason: str, new_ticks: float, baseline_ticks: float,
    tick_size: float, tick_value: float, commission: float,
) -> pd.DataFrame:
    """Analytically re-derives exit_price/gross_pnl/net_pnl/r_multiple for
    trades with the given exit_reason under a different slippage-tick
    assumption than what was actually simulated -- algebra on the
    already-computed trade log, no re-simulation."""
    if trades.empty:
        return trades.copy()
    mask = trades["exit_reason"] == exit_reason
    subset = trades[mask].copy()
    other = trades[~mask]
    if subset.empty:
        return trades.copy()

    delta = (new_ticks - baseline_ticks) * tick_size
    sign = subset["direction"].map({"bullish": 1.0, "bearish": -1.0})
    subset["exit_price"] = subset["exit_price"] - sign * delta

    price_diff = np.where(
        subset["direction"] == "bullish",
        subset["exit_price"] - subset["entry_price"],
        subset["entry_price"] - subset["exit_price"],
    )
    subset["gross_pnl"] = (price_diff / tick_size) * tick_value
    subset["net_pnl"] = subset["gross_pnl"] - commission
    stop_distance = (subset["entry_price"] - subset["stop_price"]).abs()
    subset["r_multiple"] = np.where(stop_distance > 0, price_diff / stop_distance, np.nan)

    return pd.concat([other, subset], ignore_index=True)


def slippage_sensitivity(trades: pd.DataFrame, session_dates: pd.DatetimeIndex, symbol: str) -> dict:
    cost_model = COST_MODELS[symbol]
    baseline_stop_ticks = cost_model.stop_slippage_ticks

    stop_results = {}
    for ticks in STOP_SLIPPAGE_TICKS_OPTIONS:
        adjusted = _apply_slippage_delta(
            trades, "stop", ticks, baseline_stop_ticks, cost_model.tick_size, cost_model.tick_value,
            cost_model.commission_round_turn,
        )
        stop_results[ticks] = _light_summary(adjusted, session_dates)

    time_results = {}
    for ticks in TIME_EXIT_SLIPPAGE_TICKS_OPTIONS:
        # Baseline (Part 1's actual simulation) applies zero time-exit
        # slippage -- target_time exits use the bar's raw close.
        adjusted = _apply_slippage_delta(
            trades, "target_time", ticks, 0.0, cost_model.tick_size, cost_model.tick_value,
            cost_model.commission_round_turn,
        )
        time_results[ticks] = _light_summary(adjusted, session_dates)

    return {"stop_slippage_ticks": stop_results, "time_exit_slippage_ticks": time_results}


def slippage_sensitivity_for_survivors(df_1m: pd.DataFrame, symbol: str, survivors: pd.DataFrame) -> dict[str, dict]:
    session_dates = all_session_dates(df_1m)
    results = {}
    for _, row in survivors.iterrows():
        config = config_from_row(row)
        _, _, trades, _ = run_config(df_1m, config, symbol)
        results[row["config_name"]] = slippage_sensitivity(trades, session_dates, symbol)
    return results
