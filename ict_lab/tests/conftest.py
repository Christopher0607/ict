from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_minute_bars(
    start_utc: str, end_utc: str, base_price: float = 100.0, contract: str = "TESTZ24"
) -> pd.DataFrame:
    index = pd.date_range(start_utc, end_utc, freq="1min", tz="UTC", inclusive="left")
    n = len(index)
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(0, 0.1, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(open_, close) + rng.uniform(0, 0.1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.1, n)
    volume = rng.integers(1, 500, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "contract": contract,
        },
        index=index,
    )


@pytest.fixture
def synthetic_bars():
    return _make_minute_bars
