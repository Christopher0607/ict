from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ict_lab.data.sessions import add_session_columns

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"

# 18:00 ET prior day -> 17:00 ET, minus the 17:00-18:00 ET maintenance break.
EXPECTED_SESSION_MINUTES = 23 * 60
EXPECTED_NY_AM_MINUTES = 60


def build_quality_report(df: pd.DataFrame, symbol: str, series: str) -> dict:
    with_sessions = add_session_columns(df)

    per_session_bars = with_sessions.groupby("session_date").size()
    missing_minutes = (EXPECTED_SESSION_MINUTES - per_session_bars).clip(lower=0)
    missing_minutes_by_year = missing_minutes.groupby(per_session_bars.index.year).sum()

    ny_am = with_sessions[with_sessions["killzone_ny_am"]]
    ny_am_bars_per_session = ny_am.groupby("session_date").size()
    short_ny_am_days = ny_am_bars_per_session[ny_am_bars_per_session < EXPECTED_NY_AM_MINUTES]

    roll_days_by_year: dict[int, int] = {}
    if "is_roll_day" in df.columns:
        roll_sessions = with_sessions.loc[with_sessions["is_roll_day"], "session_date"].drop_duplicates()
        roll_days_by_year = {int(y): int(v) for y, v in roll_sessions.dt.year.value_counts().items()}

    all_session_dates = pd.DatetimeIndex(sorted(with_sessions["session_date"].unique()))
    expected_business_days = pd.bdate_range(all_session_dates.min(), all_session_dates.max())
    missing_full_days = sorted(
        str(d.date()) for d in expected_business_days.difference(all_session_dates)
    )

    bad_high_low = df["high"] < df["low"]
    ohlc_out_of_range = (
        (df["open"] > df["high"]) | (df["open"] < df["low"])
        | (df["close"] > df["high"]) | (df["close"] < df["low"])
    )

    return {
        "symbol": symbol,
        "series": series,
        "date_range": [str(df.index.min()), str(df.index.max())],
        "duplicate_timestamps": int(df.index.duplicated(keep=False).sum()),
        "zero_volume_bars": int((df["volume"] == 0).sum()),
        "bad_high_low_bars": int(bad_high_low.sum()),
        "ohlc_out_of_range_bars": int(ohlc_out_of_range.sum()),
        "missing_minutes_by_year": {int(y): int(v) for y, v in missing_minutes_by_year.items()},
        "short_ny_am_window_days": int(len(short_ny_am_days)),
        "missing_full_trading_days": missing_full_days,
        "missing_full_trading_days_note": (
            "Computed against a plain Mon-Fri business-day calendar, not a "
            "real CME holiday calendar, so US market holidays will show up "
            "here as false positives -- cross-check before treating any of "
            "these as an actual data gap."
        ),
        "roll_days_by_year": roll_days_by_year,
    }


def print_and_save_report(df: pd.DataFrame, symbol: str, series: str) -> Path:
    report = build_quality_report(df, symbol, series)
    print(f"\n=== Data quality report: {symbol} ({series}) ===")
    print(json.dumps(report, indent=2))

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / f"data_quality_{symbol}_{series}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
