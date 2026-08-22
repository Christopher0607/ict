"""Roll detection and additive back-adjustment.

The back-adjusted series is what multi-day features read, and an error here is
invisible until a strategy quietly trades a price that never existed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.data.contracts import (
    add_backadjusted,
    check_gaps_plausible,
    exact_gaps_from_daily,
    find_rolls,
)


def _bars(rows):
    """rows: (ts, instrument_id, open, high, low, close)"""
    return pd.DataFrame(rows, columns=[
        "ts_open", "instrument_id", "open", "high", "low", "close"])


def test_no_rolls_means_no_adjustment():
    df = _bars([
        ("2026-01-02T14:30Z", 1, 100.0, 101.0, 99.0, 100.5),
        ("2026-01-02T14:31Z", 1, 100.5, 102.0, 100.0, 101.5),
    ])
    out = add_backadjusted(df)
    assert find_rolls(df).empty
    assert (out["close_adj"] == out["close"]).all()
    assert not out["is_roll_day"].any()


def test_single_roll_gap_and_direction():
    # Old contract closes at 100, new one opens at 130 -> a +30 gap.
    df = _bars([
        ("2026-03-10T14:30Z", 1, 99.0, 101.0, 98.0, 100.0),
        ("2026-03-10T14:31Z", 2, 130.0, 131.0, 129.0, 130.5),
    ])
    rolls = find_rolls(df)
    assert len(rolls) == 1
    assert rolls.loc[0, "gap"] == pytest.approx(30.0)
    assert rolls.loc[0, "old_instrument_id"] == 1
    assert rolls.loc[0, "new_instrument_id"] == 2


def test_backadjustment_makes_the_series_continuous():
    df = _bars([
        ("2026-03-10T14:30Z", 1, 99.0, 101.0, 98.0, 100.0),
        ("2026-03-10T14:31Z", 2, 130.0, 131.0, 129.0, 130.5),
    ])
    out = add_backadjusted(df)
    # History is lifted by the gap; the newest contract keeps its true price.
    assert out.loc[0, "close_adj"] == pytest.approx(130.0)
    assert out.loc[1, "close_adj"] == pytest.approx(130.5)
    assert out.loc[1, "close_adj"] == out.loc[1, "close"]


def test_newest_contract_is_never_shifted():
    """The right edge must equal the real traded price, or live orders are wrong."""
    df = _bars([
        ("2026-01-05T14:30Z", 1, 10.0, 10.0, 10.0, 10.0),
        ("2026-03-10T14:30Z", 2, 50.0, 50.0, 50.0, 50.0),
        ("2026-06-09T14:30Z", 3, 90.0, 90.0, 90.0, 90.0),
    ])
    out = add_backadjusted(df)
    last = out.iloc[-1]
    for col in ("open", "high", "low", "close"):
        assert last[f"{col}_adj"] == pytest.approx(last[col])


def test_gaps_accumulate_backwards_through_multiple_rolls():
    df = _bars([
        ("2026-01-05T14:30Z", 1, 10.0, 10.0, 10.0, 10.0),
        ("2026-03-10T14:30Z", 2, 50.0, 50.0, 50.0, 50.0),   # gap +40
        ("2026-06-09T14:30Z", 3, 90.0, 90.0, 90.0, 90.0),   # gap +40
    ])
    out = add_backadjusted(df)
    assert out.loc[0, "close_adj"] == pytest.approx(90.0)   # 10 + 40 + 40
    assert out.loc[1, "close_adj"] == pytest.approx(90.0)   # 50 + 40
    assert out.loc[2, "close_adj"] == pytest.approx(90.0)


def test_intraday_point_distances_survive_adjustment():
    """The property the whole additive rule exists to protect.

    A ratio adjustment would rescale this distance and silently move every
    stop and target in the backtest.
    """
    df = _bars([
        ("2026-01-05T14:30Z", 1, 100.0, 120.0, 95.0, 110.0),
        ("2026-01-05T14:31Z", 1, 110.0, 115.0, 105.0, 108.0),
        ("2026-03-10T14:30Z", 2, 500.0, 505.0, 495.0, 500.0),
    ])
    out = add_backadjusted(df)
    raw_range = df.loc[0, "high"] - df.loc[0, "low"]
    adj_range = out.loc[0, "high_adj"] - out.loc[0, "low_adj"]
    assert adj_range == pytest.approx(raw_range)

    raw_move = df.loc[1, "close"] - df.loc[0, "close"]
    adj_move = out.loc[1, "close_adj"] - out.loc[0, "close_adj"]
    assert adj_move == pytest.approx(raw_move)


def test_roll_day_is_flagged():
    df = _bars([
        ("2026-03-09T14:30Z", 1, 99.0, 101.0, 98.0, 100.0),
        ("2026-03-10T14:30Z", 2, 130.0, 131.0, 129.0, 130.5),
        ("2026-03-10T14:31Z", 2, 130.5, 131.0, 130.0, 130.8),
        ("2026-03-11T14:30Z", 2, 131.0, 132.0, 130.0, 131.5),
    ])
    out = add_backadjusted(df)
    assert list(out["is_roll_day"]) == [False, True, True, False]


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        find_rolls(pd.DataFrame({"ts_open": [1], "open": [1.0]}))


# ---------------------------------------------------------------------------
# exact roll gaps from daily parent data
# ---------------------------------------------------------------------------


def _daily(rows):
    """rows: (ts, instrument_id, close)"""
    df = pd.DataFrame(rows, columns=["ts_open", "instrument_id", "close"])
    df["ts_open"] = pd.to_datetime(df["ts_open"], utc=True)
    return df


def test_daily_overlap_overrides_the_boundary_estimate():
    """Daily overlap removes the overnight term from the gap.

    On real NQ the two methods agree to within 1%, so the boundary estimate is
    a sound fallback -- but only the overlap is correct by construction, which
    is what this pins down. Here the boundary is deliberately contaminated by a
    300-point overnight move while the true spread is -5.
    """
    bars = _bars([
        ("2026-03-12T20:59:00Z", 1, 20_000.0, 20_010.0, 19_990.0, 20_000.0),
        ("2026-03-13T00:00:00Z", 2, 20_295.0, 20_300.0, 20_290.0, 20_295.0),
    ])
    rolls = find_rolls(bars)
    assert rolls.loc[0, "gap"] == pytest.approx(295.0)  # contaminated

    daily = _daily([
        ("2026-03-12T00:00:00Z", 1, 20_000.0),
        ("2026-03-12T00:00:00Z", 2, 19_995.0),   # both trade the same day
    ])
    fixed = exact_gaps_from_daily(rolls, daily)
    assert fixed.loc[0, "gap"] == pytest.approx(-5.0)
    assert fixed.loc[0, "gap_source"] == "daily_overlap"


def test_roll_without_overlap_keeps_its_estimate_but_is_flagged():
    bars = _bars([
        ("2026-03-12T20:59:00Z", 1, 20_000.0, 20_010.0, 19_990.0, 20_000.0),
        ("2026-03-13T00:00:00Z", 2, 20_295.0, 20_300.0, 20_290.0, 20_295.0),
    ])
    rolls = find_rolls(bars)
    daily = _daily([("2026-03-12T00:00:00Z", 1, 20_000.0)])  # new contract absent
    fixed = exact_gaps_from_daily(rolls, daily)
    assert fixed.loc[0, "gap_source"] == "boundary_estimate"
    assert fixed.loc[0, "gap"] == pytest.approx(295.0)


def test_plausibility_is_judged_on_annualised_carry_not_index_fraction():
    """A real quarterly spread at 5% rates must not be flagged.

    This is the check that a naive "1% of the index" rule got wrong, making a
    correct back-adjustment look like a bug: +250 points on a 20,000 index is
    1.25% of the index but only 5% annualised, which is just the policy rate.
    """
    bars = _bars([
        ("2026-03-12T20:59:00Z", 1, 20_000.0, 20_010.0, 19_990.0, 20_000.0),
        ("2026-03-13T00:00:00Z", 2, 20_250.0, 20_260.0, 20_240.0, 20_250.0),
    ])
    rolls = find_rolls(bars)
    assert rolls.loc[0, "gap"] == pytest.approx(250.0)
    assert check_gaps_plausible(rolls, reference_price=20_000.0) == []

    # 15% annualised is beyond any rate/dividend combination.
    absurd = rolls.copy()
    absurd["gap"] = [750.0]
    warnings = check_gaps_plausible(absurd, reference_price=20_000.0)
    assert len(warnings) == 1
    assert "annualised carry" in warnings[0]


def test_negative_carry_is_plausible_at_zero_rates():
    """Dividend yield above the policy rate makes the spread negative."""
    bars = _bars([
        ("2012-03-12T20:59:00Z", 1, 2_670.0, 2_671.0, 2_669.0, 2_670.0),
        ("2012-03-13T00:00:00Z", 2, 2_665.0, 2_666.0, 2_664.0, 2_665.0),
    ])
    rolls = find_rolls(bars)
    assert rolls.loc[0, "gap"] == pytest.approx(-5.0)
    assert check_gaps_plausible(rolls, reference_price=2_670.0) == []


def test_backadjustment_uses_supplied_gaps():
    bars = _bars([
        ("2026-03-12T20:59:00Z", 1, 20_000.0, 20_010.0, 19_990.0, 20_000.0),
        ("2026-03-13T00:00:00Z", 2, 20_295.0, 20_300.0, 20_290.0, 20_295.0),
    ])
    rolls = find_rolls(bars)
    rolls["gap"] = [-5.0]   # the true spread
    out = add_backadjusted(bars, rolls=rolls)
    # History shifts by -5, not by the contaminated +295.
    assert out.loc[0, "close_adj"] == pytest.approx(19_995.0)
    assert out.loc[1, "close_adj"] == pytest.approx(20_295.0)
