from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ict_lab.data import loader as loader_mod


def _write_symbol_files(raw_dir: Path, symbol: str, n: int = 100, mismatched: bool = False) -> pd.DatetimeIndex:
    index = pd.date_range("2024-06-03", periods=n, freq="1min", tz="UTC")
    base = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10,
            "contract": "TESTZ24",
        },
        index=index,
    )
    base.to_parquet(raw_dir / f"{symbol}_1m_backadjusted.parquet")
    unadjusted = base.iloc[:-1] if mismatched else base
    unadjusted.to_parquet(raw_dir / f"{symbol}_1m_unadjusted.parquet")

    rolls = pd.DataFrame(
        {
            "date": [index[0].normalize()],
            "old_contract": ["TESTH24"],
            "new_contract": ["TESTZ24"],
            "price_adjustment": [1.5],
        }
    )
    rolls.to_csv(raw_dir / f"{symbol}_rolls.csv", index=False)
    return index


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_mod, "RAW_DIR", tmp_path)
    return tmp_path


def test_mismatched_indices_fail_loudly(raw_dir):
    _write_symbol_files(raw_dir, "NQ", mismatched=True)
    with pytest.raises(ValueError, match="do not match exactly"):
        loader_mod.load_symbol("NQ", price_series="backadjusted")


def test_is_roll_day_flag_set_on_the_roll_date(raw_dir):
    _write_symbol_files(raw_dir, "NQ")
    df = loader_mod.load_symbol("NQ", price_series="backadjusted", include_holdout=True)
    assert "is_roll_day" in df.columns
    assert bool(df["is_roll_day"].iloc[0])
    assert df["is_roll_day"].sum() == (df.index.date == df.index[0].date()).sum()


def test_unadjusted_series_selectable(raw_dir):
    _write_symbol_files(raw_dir, "NQ")
    df = loader_mod.load_symbol("NQ", price_series="unadjusted", include_holdout=True)
    assert "is_roll_day" in df.columns


def test_invalid_price_series_rejected(raw_dir):
    _write_symbol_files(raw_dir, "NQ")
    with pytest.raises(ValueError, match="price_series"):
        loader_mod.load_symbol("NQ", price_series="bogus")


def test_missing_file_raises_helpful_error(raw_dir):
    with pytest.raises(FileNotFoundError, match="Phase 1 needs"):
        loader_mod.load_symbol("NQ", price_series="backadjusted")


def test_missing_required_columns_raises(raw_dir):
    index = pd.date_range("2024-06-03", periods=10, freq="1min", tz="UTC")
    incomplete = pd.DataFrame({"close": 100.0}, index=index)
    incomplete.to_parquet(raw_dir / "NQ_1m_backadjusted.parquet")
    incomplete.to_parquet(raw_dir / "NQ_1m_unadjusted.parquet")
    pd.DataFrame(
        {
            "date": [index[0]],
            "old_contract": ["A"],
            "new_contract": ["B"],
            "price_adjustment": [1.0],
        }
    ).to_csv(raw_dir / "NQ_rolls.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        loader_mod.load_symbol("NQ", price_series="backadjusted")


# ---------------------------------------------------------------------------
# out-of-sample instrument guard
# ---------------------------------------------------------------------------


def test_es_is_not_loadable_by_default(raw_dir):
    """ES validates configs chosen without it, so reaching it must be deliberate."""
    _write_symbol_files(raw_dir, "ES")
    with pytest.raises(loader_mod.OutOfSampleAccess, match="out-of-sample instrument"):
        loader_mod.load_symbol("ES", price_series="backadjusted")


def test_es_loads_with_an_explicit_opt_in(raw_dir):
    _write_symbol_files(raw_dir, "ES")
    df = loader_mod.load_symbol(
        "ES", price_series="backadjusted", allow_out_of_sample=True, include_holdout=True
    )
    assert not df.empty


def test_nq_is_unaffected_by_the_guard(raw_dir):
    _write_symbol_files(raw_dir, "NQ")
    df = loader_mod.load_symbol("NQ", price_series="backadjusted", include_holdout=True)
    assert not df.empty
