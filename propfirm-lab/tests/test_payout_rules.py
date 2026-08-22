"""Payout gates: each rule gets a just-passes and a just-fails case.

The funded stage is where the money is, and it is gated by four rules that
are easy to state and easy to model wrong. Each test isolates one gate by
satisfying all the others.
"""

from __future__ import annotations

import numpy as np
import pytest

from rules import get_ruleset
from sim.account import payout_gate

APEX = get_ruleset("apex_50k_intraday")


def test_no_profit_blocks_everything():
    ok, amt, why = payout_gate(np.array([0.0, 0.0]), 50_000.0, APEX)
    assert not ok and amt == 0.0 and why == "no_profit"


def test_min_trading_days_boundary():
    """Apex wants 5 qualifying days; a qualifying day needs >= $50 profit."""
    assert APEX.min_trading_days == 5

    four = np.array([800.0, 800.0, 800.0, 800.0])
    ok, _, why = payout_gate(four, 53_200.0, APEX)
    assert not ok and why == "min_trading_days"

    five = np.array([640.0, 640.0, 640.0, 640.0, 640.0])
    ok, amt, why = payout_gate(five, 53_200.0, APEX)
    assert ok, why
    assert amt > 0


def test_sub_threshold_days_do_not_qualify():
    """Six days, but two are worth less than $50 -> only four count."""
    days = np.array([800.0, 800.0, 800.0, 800.0, 49.0, 49.0])
    ok, _, why = payout_gate(days, 53_298.0, APEX)
    assert not ok and why == "min_trading_days"


def test_consistency_boundary():
    """No single day may exceed 50% of total profit since the last payout."""
    assert APEX.consistency_pct == 0.50

    # Total 3,200; best day 1,600 == exactly 50% -> allowed.
    at_limit = np.array([1_600.0, 400.0, 400.0, 400.0, 400.0])
    ok, _, why = payout_gate(at_limit, 53_200.0, APEX)
    assert ok, why

    # Total 3,200; best day 1,700 > 50% -> blocked.
    over = np.array([1_700.0, 375.0, 375.0, 375.0, 375.0])
    ok, _, why = payout_gate(over, 53_200.0, APEX)
    assert not ok and why == "consistency"


def test_consistency_is_why_one_big_bet_does_not_work():
    """The single-lucky-day route clears the target and still cannot withdraw.

    This is the rule that stops 'take one huge trade' from being the optimal
    play against a trailing floor.
    """
    one_day = np.array([3_200.0, 60.0, 60.0, 60.0, 60.0])
    ok, _, why = payout_gate(one_day, 53_440.0, APEX)
    assert not ok and why == "consistency"


def test_safety_net_boundary():
    assert APEX.safety_net_equity == 52_600.0
    days = np.array([500.0] * 5)

    ok, _, why = payout_gate(days, 52_599.0, APEX)
    assert not ok and why == "safety_net"

    ok, _, why = payout_gate(days, 52_600.0, APEX)
    # At exactly the safety net there is nothing withdrawable above it, so the
    # next gate down is what blocks -- but it must not be the safety net.
    assert why != "safety_net"


def test_min_payout_boundary():
    """Apex will not process a request under $500."""
    assert APEX.min_payout == 500.0
    days = np.array([700.0] * 5)

    # Equity 53,099 -> withdrawable above the 52,600 safety net is $499.
    ok, _, why = payout_gate(days, 53_099.0, APEX)
    assert not ok and why == "min_payout"

    ok, amt, why = payout_gate(days, 53_100.0, APEX)
    assert ok, why
    assert amt == pytest.approx(500.0)


def test_withdrawal_keeps_the_safety_net_intact():
    days = np.array([1_000.0] * 5)
    ok, amt, _ = payout_gate(days, 55_000.0, APEX)
    assert ok
    assert amt == pytest.approx(55_000.0 - 52_600.0)


def test_profit_split_is_applied():
    half = APEX.__class__(**{**APEX.__dict__, "profit_split": 0.5})
    days = np.array([1_000.0] * 5)
    _, full_amt, _ = payout_gate(days, 55_000.0, APEX)
    ok, half_amt, _ = payout_gate(days, 55_000.0, half)
    assert ok
    assert half_amt == pytest.approx(full_amt * 0.5)


def test_topstep_gates_differ_from_apex():
    """Sanity that rulesets are actually distinct, not silently sharing state."""
    ts = get_ruleset("topstep_50k")
    assert ts.safety_net_equity is None
    assert ts.qualifying_day_min_profit == 200.0

    # Five days of $150 qualify at Apex ($50 bar) but not at Topstep ($200 bar).
    days = np.array([150.0] * 5)
    _, _, apex_why = payout_gate(days, 50_750.0, APEX)
    _, _, ts_why = payout_gate(days, 50_750.0, ts)
    assert apex_why == "safety_net"
    assert ts_why == "min_trading_days"


# ---------------------------------------------------------------------------
# payout cadence and fees
# ---------------------------------------------------------------------------


def test_first_payout_day_finds_the_earliest_qualifying_day():
    from sim.account import first_payout_day

    # Day 4 is the first with 5 qualifying days, but it is NOT the answer:
    # equity is 53,000, only $400 sits above the 52,600 safety net, and Apex
    # will not process a request under $500. Day 5 is the first that clears
    # every gate at once.
    prof = np.array([600.0] * 8)
    day, amount = first_payout_day(prof, 50_000.0, APEX)
    assert day == 5
    assert amount == pytest.approx(53_600.0 - 52_600.0)


def test_first_payout_day_waits_out_a_consistency_breach():
    from sim.account import first_payout_day

    # A huge first day blocks the payout until later days dilute its share
    # below 50% of the running total.
    # A $3,000 opening day needs the running total to reach $6,000 before it
    # is back under half of it -- eight more $400 days, not the five the
    # minimum-days rule alone would ask for.
    prof = np.array([3_000.0] + [400.0] * 10)
    day, _ = first_payout_day(prof, 50_000.0, APEX)
    assert day == 8
    cum = prof[: day + 1].sum()
    assert prof[: day + 1].max() <= 0.5 * cum + 1e-9


def test_first_payout_day_returns_minus_one_when_never_eligible():
    from sim.account import first_payout_day

    prof = np.array([10.0] * 30)  # never clears the safety net
    day, amount = first_payout_day(prof, 50_000.0, APEX)
    assert day == -1 and amount == 0.0


def test_monthly_fees_accrue_with_holding_time():
    from sim.account import _monthly_fees

    ts = get_ruleset("topstep_50k")
    assert ts.monthly_fee == 49.0
    assert _monthly_fees(ts, days_used=1, days_per_month=21) == 0.0    # month 1 is the eval fee
    assert _monthly_fees(ts, days_used=21, days_per_month=21) == 0.0
    assert _monthly_fees(ts, days_used=22, days_per_month=21) == 49.0  # into month 2
    assert _monthly_fees(ts, days_used=63, days_per_month=21) == 98.0  # month 3

    # Apex is one-time-payment since March 2026, so holding costs nothing extra.
    assert _monthly_fees(APEX, days_used=500, days_per_month=21) == 0.0
