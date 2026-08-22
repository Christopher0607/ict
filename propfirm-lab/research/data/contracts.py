"""Roll detection and back-adjustment for continuous futures series.

Databento's continuous symbology (``NQ.v.0``) returns one series that hops
from contract to contract, with ``instrument_id`` changing at each roll and the
``symbol`` column staying ``"NQ.v.0"`` throughout. So rolls are found by
watching ``instrument_id``, not the symbol.

**Back-adjustment is additive, never proportional.** This project prices
everything in points, ticks and R multiples, and an additive shift leaves point
distances untouched while a ratio adjustment silently rescales every stop and
target in the history. That rule comes from the sibling ICT lab and it is not
negotiable here.

**On roll gaps, and a wrong diagnosis worth recording.** Over 16 years of NQ
the boundary estimate -- first bar of the new contract minus last bar of the
old -- produces gaps averaging +53 points and peaking above +320, and
back-adjusts the June 2010 close from 1,826 up to about 5,275. That looked
obviously broken: a quarterly spread should be small, and the estimate is
structurally vulnerable to swallowing an overnight move, since Databento
switches contracts at a session boundary every time.

It was not broken. ``exact_gaps_from_daily`` re-derives every gap from days
where both contracts printed a close, removing the overnight term completely,
and lands on mean +53.1 against the estimate's +53.5 -- agreement to within
1%. The gaps are real, and they are cost of carry: annualised as
``gap / price * 4`` they track the policy rate across the whole sample.

    2010-2016  ZIRP          -0.27% to -0.74%   (dividend yield exceeds rates)
    2017-2019  hiking        +0.65% to +1.47%
    2020-2021  back to zero  -0.39%, -0.20%
    2023-2026  4-5% rates    +3.89% to +5.34%

So a several-hundred-point quarterly spread on a 29,000 index is exactly what
it should be, and a back-adjusted series that drifts thousands of points from
the traded price over 16 years is the correct behaviour of additive
adjustment under positive carry, not a defect. This is precisely why the
points-and-ticks rule exists: the adjusted *level* is not a price anyone paid,
while every point *distance* is exact.

Both estimators are kept. ``exact_gaps_from_daily`` is preferred because it
is unconditionally correct and costs about $0.19 for 16 years of NQ; the
boundary estimate is a sound fallback when parent daily data is unavailable.

Whichever gap is used, the additive rule holds: this project prices everything
in points, ticks and R multiples, and an additive shift leaves point distances
untouched while a ratio adjustment silently rescales every stop and target in
the history.
"""

from __future__ import annotations

import pandas as pd


# A quarterly spread is cost of carry, so the meaningful check is on the
# implied *annualised rate*, not on a fraction of the index. A flat fraction
# is regime-dependent and gets this wrong: 1% of the index is absurd at zero
# rates and entirely normal at 5%. No combination of policy rate and dividend
# yield plausibly implies carry beyond this.
MAX_PLAUSIBLE_ANNUALISED_CARRY = 0.10
QUARTERS_PER_YEAR = 4


def find_rolls(df: pd.DataFrame) -> pd.DataFrame:
    """One row per contract roll.

    Returns columns: ``ts_open`` (first bar of the new contract),
    ``old_instrument_id``, ``new_instrument_id``, ``gap`` (new price minus old,
    in points) and ``roll_date``.
    """
    _require(df, ["ts_open", "instrument_id", "open", "close"])
    ids = df["instrument_id"].to_numpy()
    changed = ids[1:] != ids[:-1]
    idx = changed.nonzero()[0] + 1  # first bar of each new contract

    if idx.size == 0:
        return pd.DataFrame(
            columns=["ts_open", "old_instrument_id", "new_instrument_id",
                     "gap", "roll_date"]
        )

    prev_close = df["close"].to_numpy()[idx - 1]
    new_open = df["open"].to_numpy()[idx]
    out = pd.DataFrame({
        "ts_open": df["ts_open"].to_numpy()[idx],
        "old_instrument_id": ids[idx - 1],
        "new_instrument_id": ids[idx],
        "gap": new_open - prev_close,
    })
    out["roll_date"] = pd.to_datetime(out["ts_open"]).dt.date
    return out.reset_index(drop=True)


def add_backadjusted(
    df: pd.DataFrame,
    rolls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add ``open_adj``/``high_adj``/``low_adj``/``close_adj`` and ``is_roll_day``.

    The most recent contract is left at its true price and history is shifted
    to meet it, so the right-hand edge of the adjusted series equals the real
    traded price -- which is what you want when translating a backtest into
    live orders.

    Pass ``rolls`` from ``exact_gaps_from_daily`` to adjust with true contract
    spreads. Omitting it falls back to the boundary estimate, which the module
    docstring explains is unreliable at scale.
    """
    _require(df, ["ts_open", "instrument_id", "open", "high", "low", "close"])
    df = df.sort_values("ts_open", kind="stable").reset_index(drop=True)
    if rolls is None:
        rolls = find_rolls(df)

    # Cumulative shift applied to each bar: everything before a roll moves by
    # that roll's gap, and gaps accumulate backwards through history.
    # Every bar before a roll takes that roll's gap, so gaps accumulate
    # backwards through history and the final segment keeps a shift of zero.
    shift = pd.Series(0.0, index=df.index)
    if not rolls.empty:
        ids = df["instrument_id"].to_numpy()
        boundaries = (ids[1:] != ids[:-1]).nonzero()[0] + 1
        for b, gap in zip(boundaries, rolls["gap"].to_numpy()):
            shift.iloc[:b] += gap

    for col in ("open", "high", "low", "close"):
        df[f"{col}_adj"] = df[col] + shift

    roll_dates = set(rolls["roll_date"]) if not rolls.empty else set()
    df["is_roll_day"] = pd.to_datetime(df["ts_open"]).dt.date.isin(roll_dates)
    return df


def _require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")


def exact_gaps_from_daily(rolls: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Replace boundary-estimated gaps with the true contract spread.

    ``daily`` is parent-symbology ``ohlcv-1d``: one row per contract per day,
    so on any given day both the outgoing and incoming contract have a close.
    Taking the difference there removes the overnight term entirely.

    Returns ``rolls`` with ``gap`` replaced and a ``gap_source`` column marking
    which rolls could be priced exactly. Any roll with no overlapping day keeps
    its estimate and is flagged, rather than being silently trusted.
    """
    if rolls.empty:
        return rolls.assign(gap_source=pd.Series(dtype=str))

    d = daily.copy()
    d["date"] = pd.to_datetime(d["ts_open"], utc=True).dt.date
    # close per (date, instrument_id)
    lookup = d.set_index(["date", "instrument_id"])["close"]

    gaps, sources = [], []
    for _, r in rolls.iterrows():
        old_id, new_id = int(r["old_instrument_id"]), int(r["new_instrument_id"])
        roll_day = r["roll_date"]

        # Walk back from the roll date for the most recent day both traded.
        found = None
        for back in range(0, 10):
            day = roll_day - pd.Timedelta(days=back)
            day = day.date() if hasattr(day, "date") else day
            try:
                old_c = lookup.loc[(day, old_id)]
                new_c = lookup.loc[(day, new_id)]
            except KeyError:
                continue
            old_c = float(old_c.iloc[0] if hasattr(old_c, "iloc") else old_c)
            new_c = float(new_c.iloc[0] if hasattr(new_c, "iloc") else new_c)
            found = new_c - old_c
            break

        if found is None:
            gaps.append(float(r["gap"]))
            sources.append("boundary_estimate")
        else:
            gaps.append(found)
            sources.append("daily_overlap")

    out = rolls.copy()
    out["gap"] = gaps
    out["gap_source"] = sources
    return out


def check_gaps_plausible(rolls: pd.DataFrame, reference_price: float) -> list[str]:
    """Flag roll gaps that no plausible carry could produce.

    Checks the implied annualised rate rather than a fraction of the index,
    because the fraction is regime-dependent: the same +300 points on a 29,000
    index is a bug at zero rates and correct at 5%. Getting this backwards is
    what made a correct back-adjustment look broken.
    """
    if rolls.empty:
        return []
    annualised = (rolls["gap"] / reference_price) * QUARTERS_PER_YEAR
    bad = rolls.loc[annualised.abs() > MAX_PLAUSIBLE_ANNUALISED_CARRY]
    out = []
    for r in bad.itertuples():
        rate = (r.gap / reference_price) * QUARTERS_PER_YEAR
        out.append(
            f"roll {r.roll_date}: gap {r.gap:+.1f} pts implies {rate:+.1%} "
            f"annualised carry, beyond any plausible rate"
        )
    return out
