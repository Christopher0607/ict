"""Data quality report.

Run this before any research touches the data. Every problem it looks for has
a specific way of corrupting results downstream: missing minutes make a gap
look like a flat market, zero-volume bars produce prices nobody could trade,
and an unexplained jump is usually an unhandled roll rather than a real move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from research.data.sessions import add_session_columns

# GLBX.MDP3 reaches back to 2010-06-06, but its early minute coverage is not
# usable and the shortfall is not random. Measured on NQ.v.0:
#
#   2010-2012   ~20-27% of RTH minutes present; in June 2010 only Mondays have
#               a full cash session at all. Not a symbology artifact -- the raw
#               contracts are equally sparse (NQ.FUT, all contracts together,
#               averages 627 bars/day against ~1,380 for full coverage), so it
#               is the dataset itself. CME's MDP 3.0 feed post-dates this
#               period; earlier history is reconstructed from a thinner source.
#   2013-2015   85% of sessions complete, 15% absent entirely -- and the
#               absences cluster hard in winter: 92 of 116 fall in Nov-Feb.
#               Research on this window would under-sample exactly the months
#               that carry the most volatility.
#   2016+       99.4% of sessions present, and a session that is present has
#               its full 390 RTH minutes (median and 10th percentile both 390).
#
# So the shortfall is whole missing days, never partial ones. Anything before
# MIN_USABLE_DATE is available for robustness checks but must be labelled as
# seasonally incomplete, never treated as a clean extension of the sample.
MIN_USABLE_DATE = pd.Timestamp("2016-01-01", tz="UTC")

RTH_MINUTES = 390


@dataclass
class CoverageRow:
    period: str
    sessions: int
    sessions_with_rth: int
    median_rth_bars: float
    pct_sessions_complete: float


@dataclass
class QualityReport:
    symbol: str
    rows: int
    first: pd.Timestamp
    last: pd.Timestamp
    sessions: int
    zero_volume_bars: int
    duplicate_timestamps: int
    largest_gaps: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    big_jumps: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)

    def render(self) -> str:
        lines = [
            f"=== {self.symbol} ===",
            f"  rows              {self.rows:,}",
            f"  range             {self.first}  ->  {self.last}",
            f"  trade sessions    {self.sessions:,}",
            f"  zero-volume bars  {self.zero_volume_bars:,}",
            f"  duplicate stamps  {self.duplicate_timestamps:,}",
            "",
            "  largest time gaps between consecutive bars:",
            self.largest_gaps.to_string(index=False) if not self.largest_gaps.empty else "    (none)",
            "",
            "  per-year price range and coverage:",
            self.per_year.to_string() if not self.per_year.empty else "    (none)",
            "",
            "  session coverage (missing days, not partial ones -- see MIN_USABLE_DATE):",
            self.coverage.to_string() if not self.coverage.empty else "    (none)",
            "",
            "  largest bar-to-bar jumps (points, unadjusted). For NQ these are"
            " session-open gaps, not rolls: roll gaps run about +/-10 points.",
            self.big_jumps.to_string(index=False) if not self.big_jumps.empty else "    (none)",
        ]
        return "\n".join(lines)


def report(df: pd.DataFrame, symbol: str, *, top: int = 8) -> QualityReport:
    df = add_session_columns(df)
    ts = pd.to_datetime(df["ts_open"], utc=True)

    deltas = ts.diff()
    gap_idx = deltas.nlargest(top).index
    gaps = pd.DataFrame({
        "after": ts.loc[gap_idx - 1].to_numpy(),
        "gap": deltas.loc[gap_idx].to_numpy(),
    }).sort_values("gap", ascending=False)

    year = ts.dt.year
    per_year = df.groupby(year).agg(
        bars=("close", "size"),
        low=("low", "min"),
        high=("high", "max"),
        sessions=("session_date", "nunique"),
    )
    per_year.index.name = "year"

    jump = df["close"].diff().abs()
    jidx = jump.nlargest(top).index
    big_jumps = pd.DataFrame({
        "ts": ts.loc[jidx].to_numpy(),
        "jump_points": jump.loc[jidx].to_numpy(),
        "instrument_id": df["instrument_id"].loc[jidx].to_numpy()
        if "instrument_id" in df.columns else None,
    }).sort_values("jump_points", ascending=False)

    return QualityReport(
        coverage=coverage_by_year(df),
        symbol=symbol,
        rows=len(df),
        first=ts.iloc[0],
        last=ts.iloc[-1],
        sessions=df["session_date"].nunique(),
        zero_volume_bars=int((df["volume"] == 0).sum()) if "volume" in df.columns else 0,
        duplicate_timestamps=int(ts.duplicated().sum()),
        largest_gaps=gaps,
        per_year=per_year,
        big_jumps=big_jumps,
    )


def coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Per year: how many sessions exist, and how many have a full RTH day.

    Reported separately from bar counts because the two failure modes need
    different responses. A partially covered session can be filtered on the
    fly; a systematically absent set of sessions biases the sample and has to
    be excluded by date range instead.
    """
    if "session_date" not in df.columns:
        df = add_session_columns(df)
    ts = pd.to_datetime(df["ts_open"], utc=True)
    year = ts.dt.year

    rth = df.loc[df["is_rth"]]
    per_session = rth.groupby(["session_date"]).size()
    sess_year = pd.to_datetime(pd.Series(per_session.index)).dt.year

    out = []
    for y in sorted(year.unique()):
        sessions = df.loc[year == y, "session_date"].nunique()
        with_rth = int((sess_year == y).sum())
        med = float(per_session[(sess_year == y).to_numpy()].median()) if with_rth else 0.0
        out.append({
            "year": int(y),
            "sessions": sessions,
            "with_rth": with_rth,
            "median_rth_bars": med,
            "pct_complete": round(with_rth / sessions, 3) if sessions else 0.0,
        })
    return pd.DataFrame(out).set_index("year")
