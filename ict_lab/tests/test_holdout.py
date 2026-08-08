from __future__ import annotations

import pandas as pd

from ict_lab.configs.data_config import DataConfig
from ict_lab.data.holdout import resolve_cutoff, split_holdout


def test_resolve_cutoff_defaults_to_two_years_before_max():
    index = pd.date_range("2020-01-01", "2026-01-01", freq="180D", tz="UTC")
    cutoff = resolve_cutoff(index, DataConfig())
    assert cutoff == index.max() - pd.DateOffset(years=2)


def test_resolve_cutoff_honors_explicit_override():
    index = pd.date_range("2020-01-01", "2026-01-01", freq="180D", tz="UTC")
    override = pd.Timestamp("2023-01-01", tz="UTC")
    cutoff = resolve_cutoff(index, DataConfig(holdout_cutoff=override))
    assert cutoff == override


def test_split_holdout_excludes_by_default():
    index = pd.date_range("2020-01-01", "2026-01-01", freq="30D", tz="UTC")
    df = pd.DataFrame({"close": range(len(index))}, index=index)
    config = DataConfig()
    included = split_holdout(df, config, include_holdout=False)
    assert included.index.max() < resolve_cutoff(index, config)
    assert len(included) < len(df)


def test_split_holdout_includes_with_explicit_override():
    index = pd.date_range("2020-01-01", "2026-01-01", freq="30D", tz="UTC")
    df = pd.DataFrame({"close": range(len(index))}, index=index)
    full = split_holdout(df, DataConfig(), include_holdout=True)
    assert len(full) == len(df)
