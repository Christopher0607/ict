"""Fetch CME futures bars from Databento, once and only once.

Three rules shape this module, all of them about not wasting money:

1. **The raw payload is cached and never re-fetched.** Every chunk is written
   to disk as the exact ``.dbn.zst`` Databento returned, alongside the parsed
   parquet. Changing the parsing logic later re-reads the local ``.dbn.zst``;
   it never re-downloads. Delete the parquet cache freely -- it costs nothing
   to rebuild.
2. **A spend ceiling that refuses rather than warns.** ``fetch_range`` prices
   the request through the free ``metadata.get_cost`` endpoint first and
   raises if it exceeds the limit. There is no flag to bypass it silently.
3. **No retry loops.** A failed chunk stops the run and reports. A loop that
   retries on failure is how a bug turns into an empty account balance.

The API key comes from ``DATABENTO_API_KEY`` and nowhere else -- no default,
no config file, no argument, so it cannot end up in a commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATASET = "GLBX.MDP3"
CACHE = Path(__file__).resolve().parents[1] / "cache"
CHUNK_DIR_RAW = CACHE / "raw"
CHUNK_DIR_CLEAN = CACHE / "clean"

# Refuse any single call that prices above this. Deliberately not overridable
# from the command line.
DEFAULT_SPEND_LIMIT_USD = 70.0


class SpendLimitExceeded(RuntimeError):
    """Raised instead of downloading when the quote is over the ceiling."""


def _client():
    import databento as db

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY is not set. Put it in .env (which is gitignored) "
            "and export it; it must never be hard-coded."
        )
    return db.Historical(key)


def _months(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end) into month-aligned chunks."""
    out: list[tuple[date, date]] = []
    cur = start
    while cur < end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def estimate_cost(
    symbol: str,
    schema: str,
    start: date,
    end: date,
    *,
    stype_in: str = "continuous",
) -> float:
    """Price a request without downloading anything. This call is free."""
    return float(
        _client().metadata.get_cost(
            dataset=DATASET,
            symbols=[symbol],
            schema=schema,
            start=start.isoformat(),
            end=end.isoformat(),
            stype_in=stype_in,
        )
    )


def record_count(
    symbol: str,
    schema: str,
    start: date,
    end: date,
    *,
    stype_in: str = "continuous",
) -> int:
    """How many records the request will return. Also free -- used to reconcile."""
    return int(
        _client().metadata.get_record_count(
            dataset=DATASET,
            symbols=[symbol],
            schema=schema,
            start=start.isoformat(),
            end=end.isoformat(),
            stype_in=stype_in,
        )
    )


@dataclass
class FetchReport:
    symbol: str
    schema: str
    quoted_usd: float
    spent_usd: float
    chunks_downloaded: int
    chunks_cached: int
    rows: int

    def __str__(self) -> str:
        return (
            f"{self.symbol} {self.schema}: {self.rows:,} rows | "
            f"{self.chunks_downloaded} downloaded, {self.chunks_cached} from cache | "
            f"quoted ${self.quoted_usd:.2f}, spent ${self.spent_usd:.2f}"
        )


def _paths(symbol: str, schema: str, chunk_start: date) -> tuple[Path, Path]:
    tag = f"{chunk_start:%Y-%m}"
    stem = f"{symbol.replace('.', '_')}_{schema}_{tag}"
    return (
        CHUNK_DIR_RAW / symbol.replace(".", "_") / f"{stem}.dbn.zst",
        CHUNK_DIR_CLEAN / symbol.replace(".", "_") / f"{stem}.parquet",
    )


def _decode_to_parquet(raw_path: Path, parquet_path: Path) -> int:
    """Parse a cached .dbn.zst into parquet. Costs nothing -- purely local."""
    import databento as db

    store = db.DBNStore.from_file(raw_path)
    df = store.to_df()
    if df.empty:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)
        return 0

    df = df.reset_index()
    # ts_event is the bar's OPENING timestamp, not its close. Everything
    # downstream that asks "when was this bar knowable?" depends on that, and
    # getting it backwards is a look-ahead bug that backtests beautifully.
    if "ts_event" in df.columns:
        df = df.rename(columns={"ts_event": "ts_open"})
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return len(df)


def fetch_range(
    symbol: str,
    schema: str,
    start: date,
    end: date,
    *,
    stype_in: str = "continuous",
    spend_limit_usd: float = DEFAULT_SPEND_LIMIT_USD,
    dry_run: bool = False,
) -> FetchReport:
    """Download ``symbol`` over ``[start, end)``, skipping anything cached.

    Set ``dry_run`` to price the job and list what would be fetched without
    spending anything.
    """
    chunks = _months(start, end)
    todo = [(a, b) for a, b in chunks if not _paths(symbol, schema, a)[0].exists()]
    cached = len(chunks) - len(todo)

    quoted = 0.0
    if todo:
        # One quote for the whole outstanding span rather than one per chunk:
        # get_cost is free but not instant, and ~200 sequential calls per symbol
        # turns a dry run into a coffee break. Chunks are contiguous, so with an
        # empty cache this is exactly the same number; with a partial cache it
        # over-quotes slightly, which is the safe direction for a spend ceiling.
        quoted = estimate_cost(symbol, schema, todo[0][0], todo[-1][1],
                               stype_in=stype_in)

    if quoted > spend_limit_usd:
        raise SpendLimitExceeded(
            f"{symbol} {schema} {start}..{end} quotes at ${quoted:.2f}, over the "
            f"${spend_limit_usd:.2f} ceiling. Narrow the range or raise the limit "
            f"deliberately -- this guard exists so a bug cannot drain the account."
        )

    print(f"  {symbol} {schema}: {len(chunks)} chunks, {cached} cached, "
          f"{len(todo)} to fetch, quoted ${quoted:.2f}")

    if dry_run:
        return FetchReport(symbol, schema, quoted, 0.0, 0, cached, 0)

    client = _client()
    downloaded = 0
    for a, b in todo:
        raw_path, pq_path = _paths(symbol, schema, a)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a .part file and rename only on success. A chunk killed
        # mid-write would otherwise be left truncated at its final path, where
        # the cache check sees a complete file and silently serves corrupt
        # data forever. Rename within one directory is atomic.
        part = raw_path.with_suffix(raw_path.suffix + ".part")
        part.unlink(missing_ok=True)
        # No try/except: a failure must stop the run loudly, not retry.
        store = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[symbol],
            schema=schema,
            start=a.isoformat(),
            end=b.isoformat(),
            stype_in=stype_in,
            path=part,
        )
        del store
        part.rename(raw_path)
        downloaded += 1
        print(f"    {a:%Y-%m} ok  ({downloaded}/{len(todo)})")

    # Everything quoted was fetched, so the quote is the spend.
    spent = quoted if downloaded else 0.0

    rows = 0
    for a, _ in chunks:
        raw_path, pq_path = _paths(symbol, schema, a)
        if not raw_path.exists():
            continue
        if not pq_path.exists():
            _decode_to_parquet(raw_path, pq_path)
        rows += len(pd.read_parquet(pq_path, columns=["ts_open"]))

    return FetchReport(symbol, schema, quoted, spent, downloaded, cached, rows)


def rebuild_parquet(symbol: str, schema: str) -> int:
    """Re-parse every cached raw chunk. Costs nothing; use after a parser change."""
    raw_dir = CHUNK_DIR_RAW / symbol.replace(".", "_")
    total = 0
    for raw_path in sorted(raw_dir.glob(f"*_{schema}_*.dbn.zst")):
        pq = CHUNK_DIR_CLEAN / symbol.replace(".", "_") / (raw_path.name.replace(".dbn.zst", ".parquet"))
        total += _decode_to_parquet(raw_path, pq)
    return total


def load_symbol(symbol: str, schema: str = "ohlcv-1m") -> pd.DataFrame:
    """Concatenate every cached chunk for a symbol into one frame."""
    pq_dir = CHUNK_DIR_CLEAN / symbol.replace(".", "_")
    files = sorted(pq_dir.glob(f"*_{schema}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cached parquet for {symbol} {schema} in {pq_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("ts_open", kind="stable").reset_index(drop=True)
