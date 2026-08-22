"""Market data acquisition and preparation.

Downloading costs real money, so everything here is built around never paying
twice: month-sized chunks, a raw cache that is never re-fetched, and a hard
spend ceiling that refuses to run rather than overspend.
"""

from research.data.databento_fetch import (
    CHUNK_DIR_RAW,
    CHUNK_DIR_CLEAN,
    SpendLimitExceeded,
    estimate_cost,
    fetch_range,
    load_symbol,
)

__all__ = [
    "CHUNK_DIR_RAW",
    "CHUNK_DIR_CLEAN",
    "SpendLimitExceeded",
    "estimate_cost",
    "fetch_range",
    "load_symbol",
]
