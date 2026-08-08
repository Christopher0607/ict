from __future__ import annotations

import pandas as pd

from ict_lab.data.quality import build_quality_report


def test_quality_report_flags_known_defects(synthetic_bars, tmp_path):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-06 00:00:00")

    # Inject one of each known defect.
    df.iloc[0, df.columns.get_loc("high")] = df.iloc[0]["low"] - 1.0  # high < low
    df.iloc[1, df.columns.get_loc("open")] = df.iloc[1]["high"] + 1.0  # open > high
    df.iloc[2, df.columns.get_loc("volume")] = 0
    dup_row = df.iloc[[3]]
    df = pd.concat([df, dup_row])

    report = build_quality_report(df, "NQ", "backadjusted")

    assert report["bad_high_low_bars"] >= 1
    assert report["ohlc_out_of_range_bars"] >= 1
    assert report["zero_volume_bars"] >= 1
    assert report["duplicate_timestamps"] == 2  # the original row + its duplicate
    assert report["symbol"] == "NQ"
    assert report["series"] == "backadjusted"


def test_quality_report_clean_data_has_zero_defects(synthetic_bars):
    df = synthetic_bars("2024-06-03 00:00:00", "2024-06-04 00:00:00")
    report = build_quality_report(df, "NQ", "backadjusted")

    assert report["duplicate_timestamps"] == 0
    assert report["zero_volume_bars"] == 0
    assert report["bad_high_low_bars"] == 0
    assert report["ohlc_out_of_range_bars"] == 0
