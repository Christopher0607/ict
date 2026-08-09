"""Phase 5 Part 2: the funnel (NQ population) and ES validation.

Five stages, applied in order, with a count reported after each:
1. Population: every sampled canonical config's summary row.
2. Minimum 100 trades. Below that: insufficient_sample. The row stays in
   the table but can never survive, whatever its later-stage numbers say.
3. Net PnL > 0.
4. Net Sharpe >= 0.5.
5. Benjamini-Hochberg FDR at 10%, computed across the FULL population from
   stage 1 -- not just the stage 2-4 survivors. Correcting only against a
   pre-filtered shortlist would understate how many hypotheses were
   actually tried and defeat the point of FDR control; a config that fails
   an earlier stage is still "one of the M things we tried" and its
   (typically unremarkable) p-value still belongs in the correction. The
   only configs excluded from the correction itself are ones with 0-1
   trades, where no p-value is even computable, not merely underpowered.
   Per-config p-value: one-sided test of mean trade R > 0 via the
   stationary block bootstrap on that config's own trade R sequence (1000
   resamples). If a timed sample projects the full-population bootstrap
   past its 15-minute budget, the WHOLE population falls back to a
   one-sided t-test instead -- one consistent method throughout, since
   mixing methods within a single BH correction would make the p-values
   incomparable -- and the method actually used is reported alongside the
   result.

Survivors = configs passing all five stages.

ES VALIDATION reruns every survivor plus the three named configs on ES,
cold (a fresh FeatureStore built on the ES bars -- detector caching is
per-symbol-data, never shared with anything NQ-derived). Cross-symbol
survival = net PnL > 0 AND net Sharpe >= 0.5 on ES (the NQ-funnel passage
is already guaranteed for survivor rows by construction).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from ict_lab.configs.grid import load_named_configs
from ict_lab.configs.strategy_config import StrategyConfig
from ict_lab.engine.feature_store import FeatureStore
from ict_lab.engine.pipeline import all_session_dates
from ict_lab.engine.sweep_runner import SUMMARY_COLUMNS, config_from_row, r_multiples_for_hash, summarize_config

MIN_TRADES = 100
MIN_NET_SHARPE = 0.5
BH_ALPHA = 0.10
N_RESAMPLES = 1000
BOOTSTRAP_TIME_BUDGET_SECONDS = 15 * 60
BOOTSTRAP_PROJECTION_SAMPLE = 50
DEFAULT_BLOCK_LENGTH = 10  # expected block length, in trades -- a moderate,
# documented default (the spec doesn't pin one): long enough to capture
# short-run serial dependence between consecutive trade outcomes, short
# enough not to collapse into resampling the whole sequence as one block.
FUNNEL_SEED = 20260809  # arbitrary fixed seed, documented for reproducibility -- not tuned to any result

FUNNEL_STAGE_NAMES = ("population", "min_100_trades", "net_pnl_positive", "net_sharpe_0_5", "bh_fdr_10pct")


# ---------- per-config p-values ----------


def _stationary_bootstrap_indices(
    n: int, n_resamples: int, expected_block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Stationary bootstrap (Politis & Romano 1994): each resample is a
    chain of consecutive (circularly-wrapped) indices into the original
    sequence, restarting at a fresh uniform index with probability
    1/expected_block_length at every step (always at step 0). Vectorized
    across the n_resamples axis; sequential across the n-trades axis,
    since a block's continuation inherently depends on its own prior step."""
    p = 1.0 / expected_block_length
    restarts = rng.random((n_resamples, n)) < p
    restarts[:, 0] = True
    fresh_draws = rng.integers(0, n, size=(n_resamples, n))

    idx = np.empty((n_resamples, n), dtype=int)
    idx[:, 0] = fresh_draws[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restarts[:, t], fresh_draws[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def block_bootstrap_p_value(
    r_multiples: np.ndarray,
    n_resamples: int = N_RESAMPLES,
    expected_block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int | None = None,
) -> float:
    """One-sided p-value for H0: mean(r_multiples) <= 0, via the stationary
    block bootstrap. p = (#resample means <= 0 + 1) / (n_resamples + 1):
    the "+1" add-one correction (Davison & Hinkley) keeps a finite resample
    count from ever reporting an unearned p=0."""
    n = len(r_multiples)
    if n == 0:
        return float("nan")
    r = np.asarray(r_multiples, dtype=float)
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(n, n_resamples, expected_block_length, rng)
    resample_means = r[idx].mean(axis=1)
    count_le_zero = int((resample_means <= 0).sum())
    return (count_le_zero + 1) / (n_resamples + 1)


def ttest_p_value(r_multiples: np.ndarray) -> float:
    """One-sided p-value for H0: mean(r_multiples) <= 0, via a one-sample
    t-test -- the spec's documented fallback when the bootstrap can't
    finish inside its time budget."""
    n = len(r_multiples)
    if n < 2:
        return float("nan")
    r = np.asarray(r_multiples, dtype=float)
    t_stat, two_sided_p = scipy_stats.ttest_1samp(r, popmean=0.0)
    if np.isnan(t_stat):
        return float("nan")
    return float(two_sided_p / 2) if t_stat > 0 else float(1 - two_sided_p / 2)


def estimate_bootstrap_seconds(
    population: pd.DataFrame,
    shard_dir,
    n_resamples: int = N_RESAMPLES,
    expected_block_length: int = DEFAULT_BLOCK_LENGTH,
    sample_size: int = BOOTSTRAP_PROJECTION_SAMPLE,
    seed: int | None = None,
) -> float:
    """Projects the full population's bootstrap wall-clock time from a
    timed random sample -- the same benchmark-then-project pattern Part 0
    uses for the sweep itself, applied here to decide bootstrap vs.
    t-test up front rather than abandoning a partially-completed bootstrap
    mid-population (which would leave some configs on one method and some
    on the other, making the p-values incomparable under one correction)."""
    if population.empty:
        return 0.0
    n = min(sample_size, len(population))
    sample = population.sample(n=n, random_state=seed)
    rng = np.random.default_rng(seed)

    start = time.perf_counter()
    for _, row in sample.iterrows():
        r = r_multiples_for_hash(shard_dir, row["config_hash"])
        block_bootstrap_p_value(r, n_resamples, expected_block_length, seed=int(rng.integers(0, 2**32 - 1)))
    elapsed = time.perf_counter() - start

    per_config = elapsed / n
    return per_config * len(population)


def compute_p_values(
    population: pd.DataFrame,
    shard_dir,
    n_resamples: int = N_RESAMPLES,
    expected_block_length: int = DEFAULT_BLOCK_LENGTH,
    seed: int | None = FUNNEL_SEED,
    time_budget_seconds: float = BOOTSTRAP_TIME_BUDGET_SECONDS,
) -> tuple[pd.Series, str]:
    """Returns (p_values indexed by config_hash, method), method being
    "bootstrap" or "ttest_fallback". One method is used for the entire
    population."""
    projected = estimate_bootstrap_seconds(population, shard_dir, n_resamples, expected_block_length, seed=seed)
    method = "bootstrap" if projected <= time_budget_seconds else "ttest_fallback"

    rng = np.random.default_rng(seed)
    p_values: dict[str, float] = {}
    for _, row in population.iterrows():
        r = r_multiples_for_hash(shard_dir, row["config_hash"])
        if method == "bootstrap":
            p_values[row["config_hash"]] = block_bootstrap_p_value(
                r, n_resamples, expected_block_length, seed=int(rng.integers(0, 2**32 - 1))
            )
        else:
            p_values[row["config_hash"]] = ttest_p_value(r)
    return pd.Series(p_values, name="p_value"), method


def apply_bh_correction(p_values: pd.Series, alpha: float = BH_ALPHA) -> pd.Series:
    """Boolean Series (same index as p_values) of which hypotheses are
    BH-significant at `alpha`. NaN p-values (0-1 trade configs -- no test
    could even run) are excluded from the correction and marked False."""
    valid = p_values.dropna()
    result = pd.Series(False, index=p_values.index)
    if valid.empty:
        return result
    reject, _, _, _ = multipletests(valid.to_numpy(), alpha=alpha, method="fdr_bh")
    result.loc[valid.index] = reject
    return result


# ---------- the funnel ----------


def run_funnel(population: pd.DataFrame, shard_dir, alpha: float = BH_ALPHA, seed: int | None = FUNNEL_SEED) -> dict:
    """Applies the 5-stage funnel in order. Returns {"counts": {stage ->
    configs still alive}, "population": population with pass_*/p_value/
    bh_significant/survivor columns added, "bh_method": "bootstrap" or
    "ttest_fallback"}."""
    df = population.copy()
    counts = {"population": len(df)}

    df["pass_min_trades"] = df["trade_count"] >= MIN_TRADES
    counts["min_100_trades"] = int(df["pass_min_trades"].sum())

    df["pass_net_pnl"] = df["net_pnl"] > 0
    counts["net_pnl_positive"] = int((df["pass_min_trades"] & df["pass_net_pnl"]).sum())

    df["pass_sharpe"] = df["net_sharpe"] >= MIN_NET_SHARPE
    counts["net_sharpe_0_5"] = int((df["pass_min_trades"] & df["pass_net_pnl"] & df["pass_sharpe"]).sum())

    p_values, bh_method = compute_p_values(df, shard_dir, seed=seed)
    df["p_value"] = df["config_hash"].map(p_values)
    df["bh_significant"] = apply_bh_correction(df["p_value"], alpha=alpha).reindex(df.index, fill_value=False)

    df["survivor"] = df["pass_min_trades"] & df["pass_net_pnl"] & df["pass_sharpe"] & df["bh_significant"]
    counts["bh_fdr_10pct"] = int(df["survivor"].sum())

    return {"counts": counts, "population": df, "bh_method": bh_method}


# ---------- ES validation ----------


def es_validation(survivors: pd.DataFrame, df_es_1m: pd.DataFrame, named_configs: dict[str, StrategyConfig] | None = None) -> pd.DataFrame:
    """Runs every survivor plus the three named configs on ES, cold.
    Returns a summary-schema DataFrame (Part 1's SUMMARY_COLUMNS) plus a
    cross_symbol_survivor column (net_pnl > 0 and net_sharpe >= 0.5 on ES)
    and a source column ("survivor" or "named") so a report can tell them
    apart -- named configs never went through the NQ funnel, so their
    cross_symbol_survivor flag describes ES performance on its own terms,
    not literal "cross-symbol SURVIVAL" in the funnel sense."""
    named_configs = named_configs if named_configs is not None else load_named_configs()

    configs_and_source = [(config_from_row(row), "survivor") for _, row in survivors.iterrows()]
    configs_and_source += [(c, "named") for c in named_configs.values()]

    session_dates = all_session_dates(df_es_1m)
    store = FeatureStore(df_es_1m)  # fresh, ES-only -- never shares state with any NQ-built store

    rows = []
    for config, source in configs_and_source:
        row, _ = summarize_config(df_es_1m, config, "ES", session_dates, store=store)
        row["cross_symbol_survivor"] = bool(row["net_pnl"] > 0 and row["net_sharpe"] >= MIN_NET_SHARPE)
        row["source"] = source
        rows.append(row)

    columns = SUMMARY_COLUMNS + ["cross_symbol_survivor", "source"]
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
