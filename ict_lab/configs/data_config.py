from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    """holdout_cutoff overrides the auto-derived (max timestamp - holdout_years)
    cutoff when set. Leave it None until you know the real data's date range."""

    holdout_years: int = 2
    holdout_cutoff: pd.Timestamp | None = None


# Pinned rather than auto-derived. The data now in data/raw/ ends 2026-08-21,
# so `max - 2 years` would resolve to 2024-08-21 -- one day off from the
# boundary propfirm-lab/research/data/holdout.py enforces, and a one-day
# disagreement between two holdout definitions is the kind of thing that goes
# unnoticed until it has already leaked.
DEFAULT_DATA_CONFIG = DataConfig(
    holdout_cutoff=pd.Timestamp("2024-08-22", tz="UTC"),
)
