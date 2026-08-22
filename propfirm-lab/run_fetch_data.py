"""Download the Phase B data set. Run with --dry-run first.

    python run_fetch_data.py --dry-run     # price it, download nothing
    python run_fetch_data.py               # actually buy it
"""

from __future__ import annotations

import argparse
from datetime import date

from research.data.databento_fetch import fetch_range, record_count

# NQ and ES continuous front month, rolled on volume (.v.0) rather than the
# calendar, so the series follows where the liquidity actually is.
CORE_START, CORE_END = date(2010, 6, 6), date(2026, 8, 22)

# Six months of 1-second bars, bought only to measure how wrong 1-minute bars
# are about whether a stop or a target was hit first. See findings/03.
#
# The window ends exactly on the holdout boundary rather than running to the
# present. Measuring bar-resolution error is blind to P&L and selects nothing,
# so it would arguably have been harmless inside the holdout -- but "arguably
# harmless" is how holdouts leak, and the pre-holdout window is $3 cheaper
# anyway ($22.17 against $25.03).
SEC_START, SEC_END = date(2024, 2, 22), date(2024, 8, 22)

JOBS = [
    ("NQ.v.0", "ohlcv-1m", CORE_START, CORE_END),
    ("ES.v.0", "ohlcv-1m", CORE_START, CORE_END),
    ("NQ.v.0", "ohlcv-1s", SEC_START, SEC_END),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="substring filter on symbol/schema")
    args = ap.parse_args()

    jobs = JOBS
    if args.only:
        jobs = [j for j in JOBS if args.only in f"{j[0]} {j[1]}"]

    total_quoted = total_spent = 0.0
    reports = []
    for symbol, schema, start, end in jobs:
        rep = fetch_range(symbol, schema, start, end, dry_run=args.dry_run)
        reports.append(rep)
        total_quoted += rep.quoted_usd
        total_spent += rep.spent_usd

    print("\n" + "=" * 72)
    for rep in reports:
        print(" ", rep)
    print("=" * 72)
    print(f"  quoted total ${total_quoted:.2f}   spent total ${total_spent:.2f}")

    if not args.dry_run:
        print("\n  reconciling row counts against Databento's own metadata:")
        for (symbol, schema, start, end), rep in zip(jobs, reports):
            expected = record_count(symbol, schema, start, end)
            flag = "OK " if expected == rep.rows else "MISMATCH"
            print(f"    [{flag}] {symbol} {schema}: got {rep.rows:,}, "
                  f"expected {expected:,}")


if __name__ == "__main__":
    main()
