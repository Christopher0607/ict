from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    """holdout_cutoff overrides the auto-derived (max timestamp - holdout_years)
    cutoff when set. Leave it None until you know the real data's date range."""

    holdout_years: int = 2
    holdout_cutoff: pd.Timestamp | None = None


DEFAULT_DATA_CONFIG = DataConfig()
