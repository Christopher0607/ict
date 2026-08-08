from __future__ import annotations

import pandas as pd

from ict_lab.configs.data_config import DataConfig


def resolve_cutoff(index: pd.DatetimeIndex, config: DataConfig) -> pd.Timestamp:
    if config.holdout_cutoff is not None:
        return config.holdout_cutoff
    return index.max() - pd.DateOffset(years=config.holdout_years)


def split_holdout(
    df: pd.DataFrame, config: DataConfig, include_holdout: bool = False
) -> pd.DataFrame:
    """Every data-loading function must route through this and default to
    include_holdout=False. The holdout stays untouched until Phase 8."""
    if include_holdout:
        return df
    cutoff = resolve_cutoff(df.index, config)
    return df[df.index < cutoff]
