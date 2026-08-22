"""Data quality report and the 1-minute bar ambiguity study.

    python run_data_report.py            # quality only (NQ)
    python run_data_report.py --ambiguity  # also run the 1s comparison

Writes findings/03_bar_ambiguity.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research.data import holdout
from research.data.ambiguity import compare_resolutions
from research.data.contracts import (
    add_backadjusted,
    check_gaps_plausible,
    exact_gaps_from_daily,
    find_rolls,
)
from research.data.databento_fetch import load_symbol
from research.data.quality import report

OUT = Path("findings")

# NQ points. A 2R bracket at each stop distance, spanning the range an
# intraday futures strategy would plausibly use.
STOPS = [15.0, 25.0, 40.0, 60.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambiguity", action="store_true")
    ap.add_argument("--symbol", default="NQ.v.0")
    args = ap.parse_args()

    holdout.check_symbol(args.symbol)

    print(f"loading {args.symbol} ohlcv-1m ...")
    df = load_symbol(args.symbol, "ohlcv-1m")
    print(f"  {len(df):,} rows")

    print("\n" + report(df, args.symbol).render())

    rolls = find_rolls(df)
    print(f"\n=== rolls ===\n  {len(rolls)} contract changes detected")
    print(f"  boundary estimate: mean {rolls['gap'].mean():+.1f}, "
          f"min {rolls['gap'].min():+.1f}, max {rolls['gap'].max():+.1f} pts")

    # Re-derive the spread from days where both contracts printed a close,
    # which removes the overnight term the boundary estimate can absorb. On NQ
    # the two agree to within 1% -- the large gaps are real cost of carry.
    parent = args.symbol.split(".")[0] + ".FUT"
    try:
        daily = load_symbol(parent, "ohlcv-1d")
        rolls = exact_gaps_from_daily(rolls, daily)
        exact = int((rolls["gap_source"] == "daily_overlap").sum())
        print(f"  exact spread from daily overlap: {exact}/{len(rolls)} rolls priced")
        print(f"  corrected: mean {rolls['gap'].mean():+.1f}, "
              f"min {rolls['gap'].min():+.1f}, max {rolls['gap'].max():+.1f} pts")
    except FileNotFoundError:
        print(f"  !! {parent} ohlcv-1d not cached -- falling back to the "
              f"unreliable boundary estimate")

    warnings = check_gaps_plausible(rolls, float(df["close"].iloc[-1]))
    if warnings:
        print(f"  !! {len(warnings)} implausible gap(s):")
        for w in warnings[:5]:
            print("     ", w)
    else:
        print("  all gaps within a plausible cost-of-carry range")

    adj = add_backadjusted(df, rolls=rolls)
    print(f"\n=== back-adjusted series ===")
    print(f"  first close: raw {adj['close'].iloc[0]:>10,.2f}  "
          f"adj {adj['close_adj'].iloc[0]:>10,.2f}")
    print(f"  last  close: raw {adj['close'].iloc[-1]:>10,.2f}  "
          f"adj {adj['close_adj'].iloc[-1]:>10,.2f}  "
          f"(must match -- newest contract is never shifted)")
    print(f"  roll-day bars: {int(adj['is_roll_day'].sum()):,}")

    dev = holdout.dev_slice(adj)
    print(f"\n=== holdout ===")
    print(f"  development rows {len(dev):,} (through {holdout.HOLDOUT_START})")
    print(f"  sealed rows      {len(adj) - len(dev):,}")

    if args.ambiguity:
        run_ambiguity(dev)


def run_ambiguity(bars_1m: pd.DataFrame) -> None:
    print("\nloading NQ.v.0 ohlcv-1s ...")
    s = load_symbol("NQ.v.0", "ohlcv-1s")
    print(f"  {len(s):,} rows  {s['ts_open'].min()} -> {s['ts_open'].max()}")

    # Restrict the minute bars to the window the 1s data covers.
    lo, hi = pd.to_datetime(s["ts_open"].min(), utc=True), pd.to_datetime(s["ts_open"].max(), utc=True)
    m_ts = pd.to_datetime(bars_1m["ts_open"], utc=True)
    m = bars_1m.loc[(m_ts >= lo) & (m_ts <= hi)].reset_index(drop=True)
    print(f"  overlapping minute bars: {len(m):,}")

    rows = []
    for stop in STOPS:
        target = stop * 2.0
        for long in (True, False):
            rep = compare_resolutions(
                m, s, stop_points=stop, target_points=target,
                sample_every_minutes=15, horizon_minutes=240, long=long,
            )
            rows.append({
                "side": "long" if long else "short",
                "stop_pts": stop,
                "target_pts": target,
                "brackets": rep.n_brackets,
                "min_ambig_rate": rep.minute_ambiguity_rate,
                "sec_ambig_rate": rep.second_ambiguity_rate,
                "truth_stop": rep.truth_in_ambiguous["stop_first"],
                "truth_target": rep.truth_in_ambiguous["target_first"],
                "truth_still_ambig": rep.truth_in_ambiguous["ambiguous"],
                "convention_error": rep.stop_wins_convention_error,
            })
            print(f"  {'long ' if long else 'short'} stop={stop:>5.0f} "
                  f"target={target:>5.0f}: 1m ambiguous {rep.minute_ambiguity_rate:6.2%}, "
                  f"1s residual {rep.second_ambiguity_rate:6.2%}, "
                  f"convention wrong {rep.stop_wins_convention_error:6.2%}")

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_json(OUT / "bar_ambiguity.json", orient="records", indent=2)
    print(f"\nwrote {OUT/'bar_ambiguity.json'}")


if __name__ == "__main__":
    main()
