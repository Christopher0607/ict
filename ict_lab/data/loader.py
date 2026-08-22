from __future__ import annotations

from pathlib import Path

import pandas as pd

from ict_lab.configs.data_config import DEFAULT_DATA_CONFIG, DataConfig
from ict_lab.data.holdout import split_holdout

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "contract"]
REQUIRED_ROLL_COLUMNS = ["date", "old_contract", "new_contract", "price_adjustment"]


def _raw_path(symbol: str, series: str) -> Path:
    return RAW_DIR / f"{symbol}_1m_{series}.parquet"


def _rolls_path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol}_rolls.csv"


def load_rolls(symbol: str) -> pd.DataFrame:
    path = _rolls_path(symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Phase 1 needs the purchased roll log for {symbol} "
            f"in ict_lab/data/raw/ before the loader can run."
        )
    rolls = pd.read_csv(path, parse_dates=["date"])
    missing = set(REQUIRED_ROLL_COLUMNS) - set(rolls.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rolls.sort_values("date").reset_index(drop=True)


def _read_raw_parquet(symbol: str, series: str) -> pd.DataFrame:
    path = _raw_path(symbol, series)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Phase 1 needs the purchased ES/NQ 1-minute files "
            f"in ict_lab/data/raw/ — see docs/ for the expected filenames "
            f"({symbol}_1m_unadjusted.parquet, {symbol}_1m_backadjusted.parquet, "
            f"{symbol}_rolls.csv)."
        )
    df = pd.read_parquet(path)
    missing = set(REQUIRED_OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError(f"{path} must have a tz-aware (UTC) DatetimeIndex.")
    return df.sort_index()


def _add_is_roll_day(df: pd.DataFrame, rolls: pd.DataFrame) -> pd.DataFrame:
    roll_dates = set(pd.DatetimeIndex(rolls["date"]).date)
    out = df.copy()
    out["is_roll_day"] = pd.Index(out.index.date).isin(roll_dates)
    return out


# ES is the out-of-sample INSTRUMENT: it exists to test configurations that
# were chosen without it. The pipeline already enforces that by ordering --
# es_validation runs after the funnel -- but ordering is not a barrier against
# someone loading ES while exploring and letting what they see inform a choice.
# Reaching it requires saying so.
OUT_OF_SAMPLE_SYMBOLS = frozenset({"ES"})


class OutOfSampleAccess(RuntimeError):
    """Raised when ES is loaded without an explicit opt-in."""


def load_symbol(
    symbol: str,
    price_series: str = "backadjusted",
    include_holdout: bool = False,
    config: DataConfig = DEFAULT_DATA_CONFIG,
    allow_out_of_sample: bool = False,
) -> pd.DataFrame:
    """Loads one symbol's OHLCV series. Backadjusted drives all strategy
    logic; pass price_series="unadjusted" only for chart rendering and manual
    verification. Asserts the two raw parquet files share an identical
    timestamp index and fails loudly if they don't — downstream session and
    killzone logic assumes that.
    """
    if price_series not in ("backadjusted", "unadjusted"):
        raise ValueError('price_series must be "backadjusted" or "unadjusted"')

    if symbol in OUT_OF_SAMPLE_SYMBOLS and not allow_out_of_sample:
        raise OutOfSampleAccess(
            f"{symbol} is the out-of-sample instrument and is not loadable by "
            f"default. It validates configurations selected without it, so "
            f"looking at it during selection destroys what it is for. Pass "
            f"allow_out_of_sample=True only from a validation step."
        )

    backadjusted = _read_raw_parquet(symbol, "backadjusted")
    unadjusted = _read_raw_parquet(symbol, "unadjusted")

    if not backadjusted.index.equals(unadjusted.index):
        raise ValueError(
            f"{symbol}: backadjusted and unadjusted timestamp indices do not "
            f"match exactly. Reconcile the raw files before proceeding."
        )

    rolls = load_rolls(symbol)
    chosen = backadjusted if price_series == "backadjusted" else unadjusted
    chosen = _add_is_roll_day(chosen, rolls)

    return split_holdout(chosen, config, include_holdout=include_holdout)


def print_yearly_close_range(symbol: str, config: DataConfig = DEFAULT_DATA_CONFIG) -> pd.DataFrame:
    df = load_symbol(symbol, price_series="backadjusted", config=config)
    yearly = df.groupby(df.index.year)["close"].agg(["min", "max"])
    yearly.index.name = "year"
    print(f"\n{symbol} backadjusted close range by year (holdout excluded):")
    print(yearly.to_string())
    return yearly
