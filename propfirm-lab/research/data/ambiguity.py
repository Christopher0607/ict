"""How wrong are 1-minute bars about which barrier was hit first?

A bracket order has two exits and one question: did price reach the stop or the
target first? On a 1-minute bar whose range spans both, the bar itself cannot
answer -- it records a high and a low but not their order. Backtests paper over
this with a convention, almost always "assume the stop won", and then report a
number as if the convention were a measurement.

This module replaces the convention with a measurement. It resolves the same
brackets twice, once at 1-minute resolution and once at 1-second, and reports:

* how often a 1-minute bar is genuinely ambiguous,
* what actually happened in those cases, and
* what the stop-wins convention costs or overstates as a result.

1-second bars are not ground truth either -- a second can also contain both
barriers -- so the residual ambiguity at 1s is reported alongside, as the
remaining uncertainty rather than as zero.

The output calibrates ``paths/parametric.py``'s excursion band, which is
currently set by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STOP_FIRST = 0
TARGET_FIRST = 1
UNRESOLVED = -1
AMBIGUOUS = 2


@dataclass
class BracketOutcome:
    """Vectorized verdicts for a batch of brackets."""

    verdict: np.ndarray      # STOP_FIRST / TARGET_FIRST / AMBIGUOUS / UNRESOLVED
    bars_to_resolve: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            "stop_first": int((self.verdict == STOP_FIRST).sum()),
            "target_first": int((self.verdict == TARGET_FIRST).sum()),
            "ambiguous": int((self.verdict == AMBIGUOUS).sum()),
            "unresolved": int((self.verdict == UNRESOLVED).sum()),
        }


def resolve_bracket(
    high: np.ndarray,
    low: np.ndarray,
    stop: float,
    target: float,
    *,
    long: bool = True,
) -> tuple[int, int]:
    """Resolve one bracket over a forward window of bars.

    Returns ``(verdict, bar_index)``. A bar that touches both barriers is
    reported as AMBIGUOUS rather than silently resolved -- deciding what to do
    about it is the caller's business, and pretending it is decided is the
    thing this module exists to stop.
    """
    if long:
        hit_stop = low <= stop
        hit_target = high >= target
    else:
        hit_stop = high >= stop
        hit_target = low <= target

    either = hit_stop | hit_target
    idx = np.flatnonzero(either)
    if idx.size == 0:
        return UNRESOLVED, -1

    i = int(idx[0])
    if hit_stop[i] and hit_target[i]:
        return AMBIGUOUS, i
    return (STOP_FIRST if hit_stop[i] else TARGET_FIRST), i


def resolve_batch(
    high: np.ndarray,
    low: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    targets: np.ndarray,
    horizon: int,
    *,
    long: bool = True,
) -> BracketOutcome:
    """Resolve many brackets, each starting at its own index."""
    n = starts.size
    verdict = np.full(n, UNRESOLVED, dtype=np.int8)
    bars = np.full(n, -1, dtype=np.int32)

    for k in range(n):
        s = int(starts[k])
        e = min(s + horizon, high.size)
        if e <= s:
            continue
        v, i = resolve_bracket(high[s:e], low[s:e], stops[k], targets[k], long=long)
        verdict[k] = v
        bars[k] = i
    return BracketOutcome(verdict, bars)


def ambiguity_rate(
    bars_1m: pd.DataFrame,
    *,
    stop_points: float,
    target_points: float,
    sample_every_minutes: int = 15,
    horizon_minutes: int = 240,
    long: bool = True,
) -> dict:
    """How often a bracket resolves on a bar that spans both barriers.

    This needs only 1-minute data, so it answers half the question for free:
    what share of a backtest's trades rest on the stop-wins convention rather
    than on something the bars actually show. It cannot say which way those
    trades truly went -- that needs finer data -- but the rate alone bounds how
    much the convention can matter. At 2% it is a rounding error; at 30% the
    backtest is mostly reporting the convention back to you.
    """
    if bars_1m.empty:
        raise ValueError("ambiguity_rate needs a non-empty bar frame")

    m = bars_1m.reset_index(drop=True)
    starts = np.arange(0, max(len(m) - 1, 1), sample_every_minutes, dtype=np.int64)
    entry = m["open"].to_numpy()[starts]
    if long:
        stops, targets = entry - stop_points, entry + target_points
    else:
        stops, targets = entry + stop_points, entry - target_points

    out = resolve_batch(
        m["high"].to_numpy(), m["low"].to_numpy(),
        starts, stops, targets, horizon_minutes, long=long,
    )
    c = out.counts()
    resolved = c["stop_first"] + c["target_first"] + c["ambiguous"]
    return {
        "stop_points": stop_points,
        "target_points": target_points,
        "side": "long" if long else "short",
        "brackets": int(starts.size),
        "resolved": resolved,
        **c,
        "ambiguity_rate": c["ambiguous"] / resolved if resolved else 0.0,
    }


def _second_index(ts_1s: pd.Series) -> dict:
    """Map a UTC timestamp to its row in the 1-second series."""
    return {t: i for i, t in enumerate(ts_1s.to_numpy())}


@dataclass
class AmbiguityReport:
    n_brackets: int
    stop_points: float
    target_points: float
    minute_counts: dict[str, int]
    second_counts: dict[str, int]
    # What the 1s data says about the brackets the 1m data called ambiguous.
    truth_in_ambiguous: dict[str, int]

    @property
    def minute_ambiguity_rate(self) -> float:
        resolved = self.n_brackets - self.minute_counts["unresolved"]
        return self.minute_counts["ambiguous"] / resolved if resolved else 0.0

    @property
    def second_ambiguity_rate(self) -> float:
        resolved = self.n_brackets - self.second_counts["unresolved"]
        return self.second_counts["ambiguous"] / resolved if resolved else 0.0

    @property
    def stop_wins_convention_error(self) -> float:
        """Share of ambiguous bars where 'the stop won' is simply wrong.

        Every one of those is a winning trade the backtest booked as a loss, so
        the convention is conservative -- but this says by how much.
        """
        t = self.truth_in_ambiguous
        decided = t["stop_first"] + t["target_first"]
        return t["target_first"] / decided if decided else 0.0


def compare_resolutions(
    bars_1m: pd.DataFrame,
    bars_1s: pd.DataFrame,
    *,
    stop_points: float,
    target_points: float,
    sample_every_minutes: int = 15,
    horizon_minutes: int = 240,
    long: bool = True,
) -> AmbiguityReport:
    """Resolve the same brackets at both resolutions and compare.

    Entries are taken at the open of every ``sample_every_minutes``-th minute
    bar; the stop and target sit a fixed number of points either side.
    """
    for name, df in (("bars_1m", bars_1m), ("bars_1s", bars_1s)):
        for col in ("ts_open", "open", "high", "low"):
            if col not in df.columns:
                raise ValueError(f"{name} is missing column {col!r}")

    if bars_1m.empty or bars_1s.empty:
        raise ValueError(
            "compare_resolutions needs non-empty bar frames; returning an "
            "empty report would read like a clean result."
        )

    m = bars_1m.reset_index(drop=True)
    s = bars_1s.reset_index(drop=True)

    m_ts = pd.to_datetime(m["ts_open"], utc=True)
    s_ts = pd.to_datetime(s["ts_open"], utc=True)

    # max(..., 1) so a single bar still forms one bracket against itself
    # rather than yielding an empty, innocuous-looking report.
    starts_m = np.arange(0, max(len(m) - 1, 1), sample_every_minutes, dtype=np.int64)
    entry = m["open"].to_numpy()[starts_m]

    if long:
        stops = entry - stop_points
        targets = entry + target_points
    else:
        stops = entry + stop_points
        targets = entry - target_points

    minute = resolve_batch(
        m["high"].to_numpy(), m["low"].to_numpy(),
        starts_m, stops, targets, horizon_minutes, long=long,
    )

    # Line the same entries up in the 1-second series.
    pos = np.searchsorted(s_ts.to_numpy(), m_ts.to_numpy()[starts_m])
    valid = pos < len(s)
    second = resolve_batch(
        s["high"].to_numpy(), s["low"].to_numpy(),
        pos[valid], stops[valid], targets[valid],
        horizon_minutes * 60, long=long,
    )

    # For the brackets the minute data could not decide, what does 1s say?
    amb_mask = minute.verdict[valid] == AMBIGUOUS
    truth = second.verdict[amb_mask]
    truth_counts = {
        "stop_first": int((truth == STOP_FIRST).sum()),
        "target_first": int((truth == TARGET_FIRST).sum()),
        "ambiguous": int((truth == AMBIGUOUS).sum()),
        "unresolved": int((truth == UNRESOLVED).sum()),
    }

    return AmbiguityReport(
        n_brackets=int(valid.sum()),
        stop_points=stop_points,
        target_points=target_points,
        minute_counts=minute.counts(),
        second_counts=second.counts(),
        truth_in_ambiguous=truth_counts,
    )
