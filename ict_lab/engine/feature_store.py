"""Phase 3 assumes "engine/signals.py builds the canonical setup sequence...
from the FeatureStore" as pre-existing infrastructure -- it isn't, in this
build, so this is that piece: Phase 2's detectors, memoized per (detector,
parameters) so repeated signal-generation calls sharing a config don't redo
the same work.

In-memory only, and deliberately not Phase 5's heavier precompute/caching
layer (sharded parquet, checkpointing across a multi-config sweep) -- that's
a separate concern for when it's actually needed. This just has to be
correct, and to make the "cache vs scratch" consistency check meaningful.
"""
from __future__ import annotations

import pandas as pd

from ict_lab.features.bias import (
    bias_daily_ma_slope,
    bias_none,
    bias_perfect,
    bias_prior_day,
    bias_swing_structure,
)
from ict_lab.features.displacement import displacement_atr
from ict_lab.features.fvg import detect_fvg
from ict_lab.features.liquidity import all_liquidity_levels
from ict_lab.features.sweep import detect_sweeps
from ict_lab.features.swings import swing_points


class FeatureStore:
    def __init__(self, df_1m: pd.DataFrame):
        self.df_1m = df_1m
        self._cache: dict[tuple, object] = {}

    def _get(self, key: tuple, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def fvgs(
        self, timeframe: str, min_size_points: float = 0.0, min_size_atr_mult: float = 0.0
    ) -> pd.DataFrame:
        key = ("fvg", timeframe, min_size_points, min_size_atr_mult)
        return self._get(
            key, lambda: detect_fvg(self.df_1m, timeframe, min_size_points, min_size_atr_mult)
        )

    def swings(self, n: int) -> pd.DataFrame:
        return self._get(("swings", n), lambda: swing_points(self.df_1m, n))

    def liquidity_levels(self, swing_n: int) -> pd.DataFrame:
        return self._get(
            ("liquidity_levels", swing_n), lambda: all_liquidity_levels(self.df_1m, swing_n=swing_n)
        )

    def sweeps(
        self,
        swing_n: int,
        level_types: tuple[str, ...],
        k: int,
        min_penetration_ticks: float,
        tick_size: float,
    ) -> pd.DataFrame:
        level_types = tuple(sorted(level_types))
        key = ("sweeps", swing_n, level_types, k, min_penetration_ticks, tick_size)

        def compute():
            levels = self.liquidity_levels(swing_n)
            return detect_sweeps(
                self.df_1m, levels, list(level_types), k, min_penetration_ticks, tick_size
            )

        return self._get(key, compute)

    def displacement(self, atr_mult: float) -> pd.Series:
        return self._get(("displacement_atr", atr_mult), lambda: displacement_atr(self.df_1m, atr_mult))

    def bias_updates(
        self,
        method: str,
        timeframe: str = "15m",
        swing_n: int = 5,
        ma_period: int = 20,
        window: str | None = None,
    ) -> pd.DataFrame:
        key = ("bias", method, timeframe, swing_n, ma_period, window)

        def compute():
            if method == "none":
                return bias_none(self.df_1m)
            if method == "prior_day":
                return bias_prior_day(self.df_1m)
            if method == "swing_structure":
                return bias_swing_structure(self.df_1m, timeframe=timeframe, swing_n=swing_n)
            if method == "daily_ma_slope":
                return bias_daily_ma_slope(self.df_1m, period=ma_period)
            if method == "perfect":
                if window is None:
                    raise ValueError("bias_method='perfect' needs a window")
                return bias_perfect(self.df_1m, window)
            raise ValueError(f"unknown bias method {method!r}")

        return self._get(key, compute)
