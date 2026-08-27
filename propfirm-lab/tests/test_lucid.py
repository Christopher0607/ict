"""LucidFlex 50k: the two mechanics no other ruleset here has.

Lucid caps a payout at min(50% of cycle profit, $2,000) and snaps the max loss
limit up to $50,100 the moment you request one. Both change what the optimal
play is, so both need to be pinned down rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rules import get_ruleset
from sim import Outcome, simulate_lifecycle, simulate_stage
from sim.account import _withdrawable, payout_gate

LUCID = get_ruleset("lucid_50k_flex")
APEX = get_ruleset("apex_50k_intraday")


def _path(values, day=0):
    eq = np.asarray(values, dtype=float)
    return eq, np.full(eq.shape, day, dtype=np.int64)


# ---------------------------------------------------------------------------
# payout cap: min(50% of profit, $2,000)
# ---------------------------------------------------------------------------


def test_payout_cap_absolute_ceiling_binds():
    """$4,000 profit -> 50% is $2,000, which is also the cap. Both bind."""
    assert _withdrawable(54_000.0, LUCID) == pytest.approx(2_000.0)


def test_payout_cap_percentage_binds_below_the_ceiling():
    """$2,000 profit -> 50% is $1,000, under the $2,000 cap."""
    assert _withdrawable(52_000.0, LUCID) == pytest.approx(1_000.0)


def test_payout_cap_ceiling_holds_as_profit_grows():
    """Past $4,000 of profit the cap stops the payout growing at all."""
    assert _withdrawable(60_000.0, LUCID) == pytest.approx(2_000.0)
    assert _withdrawable(100_000.0, LUCID) == pytest.approx(2_000.0)


def test_apex_has_no_such_cap():
    """Same equity, no percentage and no ceiling -> the whole surplus."""
    assert _withdrawable(60_000.0, APEX) == pytest.approx(60_000.0 - 52_600.0)


def test_min_payout_boundary_under_the_percentage_rule():
    """$500 minimum against a 50% rule means $1,000 of profit is the real bar."""
    days = np.array([300.0] * 4)

    ok, _, why = payout_gate(days, 50_999.0, LUCID)   # 50% of 999 = 499.50
    assert not ok and why == "min_payout"

    ok, amt, why = payout_gate(days, 51_000.0, LUCID)  # 50% of 1,000 = 500
    assert ok, why
    assert amt == pytest.approx(500.0 * LUCID.profit_split)


def test_profit_split_is_ninety_ten():
    assert LUCID.profit_split == 0.90
    days = np.array([1_000.0] * 4)
    _, amt, _ = payout_gate(days, 54_000.0, LUCID)
    assert amt == pytest.approx(2_000.0 * 0.90)


# ---------------------------------------------------------------------------
# consistency: 50% to pass, nothing once funded
# ---------------------------------------------------------------------------


def test_funded_consistency_is_absent_on_lucid_but_present_on_apex():
    """One day carrying all the profit: Lucid pays, Apex refuses.

    This is the same daily-profit vector through both rulesets, so it also
    proves the eval and funded consistency fields are genuinely separate
    rather than one value read twice.
    """
    one_big_day = np.array([3_000.0, 100.0, 100.0, 100.0, 100.0])
    equity = 53_400.0

    ok_lucid, amt, _ = payout_gate(one_big_day, equity, LUCID)
    assert ok_lucid
    assert amt > 0

    ok_apex, _, why = payout_gate(one_big_day, equity, APEX)
    assert not ok_apex and why == "consistency"


def test_eval_consistency_still_applies_on_lucid():
    """Funded has no consistency rule, but the evaluation does."""
    assert LUCID.consistency_pct_eval == 0.50
    assert LUCID.consistency_pct is None

    eq, days = _path([50_000, 53_000])
    assert simulate_stage(eq, days, LUCID, target_equity=LUCID.target_equity).outcome \
        is Outcome.RAN_OUT_OF_PATH


# ---------------------------------------------------------------------------
# the floor snap
# ---------------------------------------------------------------------------


def test_min_floor_raises_the_kill_floor():
    """The snap only ever applies post-payout, so start above it.

    On the untouched EOD trail this path's floor is 48,000 and both marks are
    safe. Snapped to 50,100 they are still safe; snapped to 50,400 the dip to
    50,300 is a breach.
    """
    eq, days = _path([50_500, 50_300])

    assert simulate_stage(eq, days, LUCID, target_equity=None).outcome \
        is Outcome.RAN_OUT_OF_PATH
    assert simulate_stage(eq, days, LUCID, target_equity=None,
                          min_floor=50_100.0).outcome is Outcome.RAN_OUT_OF_PATH
    assert simulate_stage(eq, days, LUCID, target_equity=None,
                          min_floor=50_400.0).outcome is Outcome.BREACH_DRAWDOWN


def test_payout_snaps_the_floor_and_it_bites():
    """An early payout trades $500 of cash for $1,100 of floor.

    Eval passed over four even days. Funded: grind to $51,000 (floor still
    $49,000 on the EOD trail), take the $500 payout, and the floor snaps to
    $50,100. The account then dies on a dip that it would otherwise have
    survived comfortably.
    """
    ev_eq = np.array([50_000, 50_750, 51_500, 52_250, 53_000], dtype=float)
    ev_days = np.array([0, 0, 1, 2, 3], dtype=np.int64)

    # Four days up to 51,000, then a dip to 50,400.
    fn_eq = np.array([50_000, 50_250, 50_500, 50_750, 51_000, 50_400], dtype=float)
    fn_days = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

    res = simulate_lifecycle(ev_eq, ev_days, fn_eq, fn_days, LUCID)
    assert res.passed_eval
    assert len(res.payouts) == 1
    assert res.payouts[0] == pytest.approx(500.0 * LUCID.profit_split)
    # 50,400 minus the 500 withdrawn = 49,900, under the snapped 50,100 floor.
    assert res.death_reason is Outcome.BREACH_DRAWDOWN


def test_without_the_payout_the_same_path_survives():
    """Control for the test above: no payout, no snap, no death."""
    ev_eq = np.array([50_000, 50_750, 51_500, 52_250, 53_000], dtype=float)
    ev_days = np.array([0, 0, 1, 2, 3], dtype=np.int64)

    # Never reaches $1,000 of profit, so no payout is possible.
    fn_eq = np.array([50_000, 50_200, 50_400, 50_600, 50_800, 50_400], dtype=float)
    fn_days = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)

    res = simulate_lifecycle(ev_eq, ev_days, fn_eq, fn_days, LUCID)
    assert res.passed_eval
    assert res.payouts == []
    assert res.death_reason is None


# ---------------------------------------------------------------------------
# cost structure
# ---------------------------------------------------------------------------


def test_lucid_has_no_activation_or_monthly_fee():
    from sim.account import _monthly_fees

    assert LUCID.activation_fee == 0.0
    assert LUCID.upfront_cost == LUCID.eval_fee
    assert _monthly_fees(LUCID, days_used=500, days_per_month=21) == 0.0
    assert LUCID.upfront_cost < APEX.upfront_cost


def test_payout_days_land_inside_the_account_life():
    """Every payout is dated, and the dates are ordered and in range.

    Needed to attribute fees and withdrawals to calendar years when accounts are
    bought and blown in a chain rather than studied one at a time.
    """
    import numpy as np

    from rules.ruleset import get_ruleset
    from sim.account import simulate_lifecycle

    rs = get_ruleset("lucid_50k_flex")
    # A steady climb: passes the eval, then earns payout after payout.
    days = np.repeat(np.arange(400), 4).astype(np.int64)
    equity = rs.starting_balance + np.linspace(0, 24_000, days.size)
    res = simulate_lifecycle(equity, days, equity, days, rs)

    assert res.passed_eval
    assert len(res.payout_days) == len(res.payouts)
    assert res.payout_days == sorted(res.payout_days)
    assert all(0 <= d < res.days_used for d in res.payout_days)
    # The eval has to finish before any payout can land.
    assert min(res.payout_days) > res.eval_result.stop_day


def test_no_payouts_means_no_payout_days():
    import numpy as np

    from rules.ruleset import get_ruleset
    from sim.account import simulate_lifecycle

    rs = get_ruleset("lucid_50k_flex")
    days = np.repeat(np.arange(60), 4).astype(np.int64)
    flat = np.full(days.size, rs.starting_balance)
    res = simulate_lifecycle(flat, days, flat, days, rs)
    assert res.payouts == [] and res.payout_days == []
