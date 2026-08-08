from __future__ import annotations

import pandas as pd

from ict_lab.data import cache as cache_mod


def test_write_then_read_clean_cache_roundtrips(synthetic_bars, tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CLEAN_DIR", tmp_path)
    df = pd.concat(
        [
            synthetic_bars("2023-12-31 23:00:00", "2024-01-01 01:00:00"),
            synthetic_bars("2024-06-01 00:00:00", "2024-06-01 01:00:00"),
        ]
    )

    written = cache_mod.write_clean_cache(df, "NQ", "backadjusted")
    assert len(written) == 2  # spans 2023 and 2024

    roundtrip = cache_mod.read_clean_cache("NQ", "backadjusted")
    assert len(roundtrip) == len(df)
    assert roundtrip.index.is_monotonic_increasing

    only_2024 = cache_mod.read_clean_cache("NQ", "backadjusted", years=[2024])
    assert (only_2024.index.year == 2024).all()
