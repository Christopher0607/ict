"""Run an equity path through a prop-firm ruleset and see what happens to it.

The input is an *equity path*, not a daily P&L summary. That distinction is the
whole point of this module: an intraday trailing threshold reacts to unrealized
peaks, so a day that opens +$900, gives it all back, and closes flat is not a
flat day -- it permanently raised your floor by $900. Summarize to daily P&L
first and you will systematically overestimate your survival odds.

Everything here is vectorized over a single path with numpy cumulative ops, so
a 40k-path Monte Carlo is seconds rather than minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from rules import DrawdownType, Ruleset


class Outcome(str, Enum):
    PASSED = "passed"                        # reached the profit target
    BREACH_DRAWDOWN = "breach_drawdown"      # hit the trailing/static floor
    BREACH_DAILY_LOSS = "breach_daily_loss"  # hit the daily loss limit
    RAN_OUT_OF_PATH = "ran_out_of_path"      # neither, before the path ended


@dataclass
class StageResult:
    """What one stage (eval, or one funded payout cycle) did."""

    outcome: Outcome
    stop_index: int          # index into the path where it ended
    stop_day: int            # day index where it ended
    final_equity: float
    peak_equity: float
    final_floor: float


@dataclass
class LifecycleResult:
    """One account, cradle to grave: eval, funding, payouts, and the bill."""

    passed_eval: bool
    eval_result: StageResult | None
    funded_result: StageResult | None
    payouts: list[float] = field(default_factory=list)
    # Day index within this account's life at which each payout landed. Needed
    # to attribute fees and withdrawals to calendar years when an account is
    # one link in a chain rather than the unit of analysis.
    payout_days: list[int] = field(default_factory=list)
    death_reason: Outcome | None = None
    days_used: int = 0
    fees_paid: float = 0.0
    # Why a payout never happened, when the account survived but stayed dry.
    payout_blocked_by: str | None = None

    @property
    def gross_payout(self) -> float:
        return float(sum(self.payouts))

    @property
    def net(self) -> float:
        """The only number that matters: dollars in minus dollars out."""
        return self.gross_payout - self.fees_paid


# ---------------------------------------------------------------------------
# floor construction
# ---------------------------------------------------------------------------


def _day_boundaries(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (first_index_of_each_day, last_index_of_each_day)."""
    change = np.flatnonzero(np.diff(days)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(days) - 1]))
    return starts, ends


def _floor_series(equity: np.ndarray, days: np.ndarray, rs: Ruleset) -> np.ndarray:
    """The kill-floor applicable at each mark in the path."""
    if rs.drawdown_type is DrawdownType.STATIC:
        return np.full(equity.shape, rs.initial_floor, dtype=float)

    if rs.drawdown_type is DrawdownType.INTRADAY_TRAILING:
        # Trails the running max of *unrealized* equity.
        peak = np.maximum.accumulate(equity)
        np.maximum(peak, rs.starting_balance, out=peak)
        floor = peak - rs.max_drawdown

    elif rs.drawdown_type is DrawdownType.EOD_TRAILING:
        # Trails the highest *completed day's* close. A day's own close cannot
        # raise the floor until that day is over, so the series is shifted by
        # one day.
        starts, ends = _day_boundaries(days)
        closes = equity[ends]
        peak_after_day = np.maximum.accumulate(closes)
        peak_before_day = np.concatenate(([rs.starting_balance], peak_after_day[:-1]))
        np.maximum(peak_before_day, rs.starting_balance, out=peak_before_day)
        counts = ends - starts + 1
        floor = np.repeat(peak_before_day, counts) - rs.max_drawdown

    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unhandled drawdown type {rs.drawdown_type}")

    if rs.trailing_lock_at is not None:
        # The threshold stops trailing once it reaches this level; it never
        # moves down, so a running min is not what we want -- a plain cap is.
        np.minimum(floor, rs.trailing_lock_at, out=floor)
    return floor


def _daily_loss_floor(equity: np.ndarray, days: np.ndarray, rs: Ruleset) -> np.ndarray | None:
    """Per-mark floor implied by the daily loss limit, measured from day open."""
    if rs.daily_loss_limit is None:
        return None
    starts, ends = _day_boundaries(days)
    day_open = equity[starts]
    counts = ends - starts + 1
    return np.repeat(day_open, counts) - rs.daily_loss_limit


def _first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1


# ---------------------------------------------------------------------------
# one stage
# ---------------------------------------------------------------------------


def _consistent_pass_index(
    equity: np.ndarray,
    days: np.ndarray,
    rs: Ruleset,
    *,
    target_equity: float,
    opening_equity: float,
) -> int:
    """First mark at which the evaluation is genuinely passed.

    Evaluated at day closes, because the consistency rule is stated over
    closed days. The account must be at or above the target *and* have no
    single day worth more than ``consistency_pct_eval`` of its total profit.
    A trader who clears the whole target in one session has hit the number but
    has not passed; they have to keep trading until the rest of the days
    dilute that day's share.
    """
    starts, ends = _day_boundaries(days)
    closes = equity[ends]
    profits = closes - np.concatenate(([opening_equity], closes[:-1]))
    cum = closes - opening_equity

    ok = closes >= target_equity
    ok &= cum > 0
    ok &= np.maximum.accumulate(profits) <= rs.consistency_pct_eval * cum + 1e-9

    idx = np.flatnonzero(ok)
    return int(ends[idx[0]]) if idx.size else -1


def simulate_stage(
    equity: np.ndarray,
    days: np.ndarray,
    rs: Ruleset,
    *,
    target_equity: float | None = None,
    min_floor: float | None = None,
    opening_equity: float | None = None,
) -> StageResult:
    """Walk one equity path until it passes, dies, or the path runs out.

    ``equity`` is the account equity at each mark, ``days`` the (monotonic
    non-decreasing) day index of each mark. Pass ``target_equity=None`` for a
    funded stage, which has no profit target to reach -- it just has to stay
    alive.
    """
    equity = np.asarray(equity, dtype=float)
    days = np.asarray(days, dtype=np.int64)
    if equity.shape != days.shape:
        raise ValueError("equity and days must be the same length")
    if equity.size == 0:
        raise ValueError("empty path")
    if np.any(np.diff(days) < 0):
        raise ValueError("days must be non-decreasing")

    floor = _floor_series(equity, days, rs)
    if min_floor is not None:
        np.maximum(floor, min_floor, out=floor)
    breach_dd = _first_true(equity <= floor)

    daily_floor = _daily_loss_floor(equity, days, rs)
    breach_daily = _first_true(equity <= daily_floor) if daily_floor is not None else -1

    if target_equity is None:
        hit_target = -1
    elif rs.consistency_pct_eval is None:
        hit_target = _first_true(equity >= target_equity)
    else:
        hit_target = _consistent_pass_index(
            equity, days, rs,
            target_equity=target_equity,
            opening_equity=rs.starting_balance if opening_equity is None else opening_equity,
        )

    # Earliest event wins. A drawdown breach and a target hit cannot land on
    # the same mark (the target is always above the floor), so ties are not
    # possible between those two; between the two breach types, the drawdown
    # floor is reported first since it is the account-ending one.
    candidates = [
        (breach_dd, Outcome.BREACH_DRAWDOWN),
        (breach_daily, Outcome.BREACH_DAILY_LOSS),
        (hit_target, Outcome.PASSED),
    ]
    live = [(i, o) for i, o in candidates if i >= 0]
    if live:
        stop_index, outcome = min(live, key=lambda t: t[0])
    else:
        stop_index, outcome = len(equity) - 1, Outcome.RAN_OUT_OF_PATH

    return StageResult(
        outcome=outcome,
        stop_index=stop_index,
        stop_day=int(days[stop_index]),
        final_equity=float(equity[stop_index]),
        peak_equity=float(np.max(equity[: stop_index + 1])),
        final_floor=float(floor[stop_index]),
    )


# ---------------------------------------------------------------------------
# payout gates
# ---------------------------------------------------------------------------


def _daily_profit(equity: np.ndarray, days: np.ndarray, opening_equity: float) -> np.ndarray:
    """Profit booked on each day of the path."""
    starts, ends = _day_boundaries(days)
    closes = equity[ends]
    prev_closes = np.concatenate(([opening_equity], closes[:-1]))
    return closes - prev_closes


def _withdrawable(equity_now: float, rs: Ruleset) -> float:
    """Cash that can actually leave the account right now, before the split.

    Three ceilings, any of which may be absent: what sits above the balance
    you must retain, a percentage of the cycle's profit, and a hard per-payout
    cap. Lucid applies all three; Apex applies only the first.
    """
    keep = rs.starting_balance
    if rs.safety_net_equity is not None:
        keep = max(keep, rs.safety_net_equity)

    amount = equity_now - keep
    profit = equity_now - rs.starting_balance
    if rs.payout_pct_of_profit is not None:
        amount = min(amount, rs.payout_pct_of_profit * profit)
    if rs.payout_cap is not None:
        amount = min(amount, rs.payout_cap)
    return max(0.0, amount)


def payout_gate(
    daily_profit: np.ndarray,
    equity_now: float,
    rs: Ruleset,
) -> tuple[bool, float, str | None]:
    """Can this account withdraw right now, and how much?

    Returns ``(eligible, amount, blocked_by)``. The gates are checked in the
    order a firm would check them, and the *first* failure is reported so the
    caller can tell "not enough qualifying days" apart from "one day was too
    big a share of the profit".
    """
    total_profit = equity_now - rs.starting_balance
    if total_profit <= 0:
        return False, 0.0, "no_profit"

    qualifying = int(np.sum(daily_profit >= max(rs.qualifying_day_min_profit, 1e-9)))
    if qualifying < rs.min_trading_days:
        return False, 0.0, "min_trading_days"

    if rs.safety_net_equity is not None and equity_now < rs.safety_net_equity:
        return False, 0.0, "safety_net"

    if rs.consistency_pct is not None:
        best_day = float(np.max(daily_profit)) if daily_profit.size else 0.0
        if best_day > rs.consistency_pct * total_profit + 1e-9:
            return False, 0.0, "consistency"

    amount = _withdrawable(equity_now, rs)

    if amount < rs.min_payout:
        return False, 0.0, "min_payout"

    return True, amount * rs.profit_split, None


# ---------------------------------------------------------------------------
# full lifecycle
# ---------------------------------------------------------------------------


def first_payout_day(
    daily_profit: np.ndarray,
    opening_equity: float,
    rs: Ruleset,
    *,
    withdraw_at_profit: float = 0.0,
) -> tuple[int, float]:
    """The earliest day this cycle clears every payout gate.

    ``withdraw_at_profit`` holds the payout back until the cycle has made at
    least that much, which is a real decision on rulesets that snap the
    drawdown floor up on withdrawal: taking the money early buys cash at the
    price of trading room.

    Returns ``(day_offset, amount)``, or ``(-1, 0.0)`` if the cycle never
    qualifies. Traders withdraw as soon as they are allowed to, so modelling a
    funded account as "compound for a year, then withdraw the lot" both
    overstates the payout and understates the risk of dying before you ever
    touch the money. This finds the real skim point.
    """
    if daily_profit.size == 0:
        return -1, 0.0

    cum = np.cumsum(daily_profit)
    equity = opening_equity + cum

    qualifying = np.cumsum(daily_profit >= max(rs.qualifying_day_min_profit, 1e-9))
    ok = qualifying >= rs.min_trading_days
    ok &= cum > 0
    if withdraw_at_profit > 0:
        ok &= cum >= withdraw_at_profit

    if rs.safety_net_equity is not None:
        ok &= equity >= rs.safety_net_equity

    if rs.consistency_pct is not None:
        best = np.maximum.accumulate(daily_profit)
        ok &= best <= rs.consistency_pct * cum + 1e-9

    keep = rs.starting_balance
    if rs.safety_net_equity is not None:
        keep = max(keep, rs.safety_net_equity)
    withdrawable = equity - keep
    profit = equity - rs.starting_balance
    if rs.payout_pct_of_profit is not None:
        withdrawable = np.minimum(withdrawable, rs.payout_pct_of_profit * profit)
    if rs.payout_cap is not None:
        withdrawable = np.minimum(withdrawable, rs.payout_cap)
    withdrawable = np.maximum(0.0, withdrawable)
    ok &= withdrawable >= rs.min_payout

    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return -1, 0.0
    d = int(idx[0])
    return d, float(withdrawable[d] * rs.profit_split)


def _why_blocked(daily_profit: np.ndarray, equity_now: float, rs: Ruleset) -> str | None:
    """Which gate stopped a surviving-but-dry account. Diagnostics only."""
    _, _, why = payout_gate(daily_profit, equity_now, rs)
    return why


def _monthly_fees(rs: Ruleset, days_used: int, days_per_month: int) -> float:
    """Recurring subscription cost for however long the account was held.

    ``eval_fee`` already covers the first month on the subscription firms
    (Topstep, MFFU), so only the months after the first are added. Apex has
    charged one-time since the March 2026 overhaul and has monthly_fee == 0,
    which makes this a no-op there.
    """
    if rs.monthly_fee <= 0:
        return 0.0
    months = max(1, -(-days_used // days_per_month))  # ceil
    return rs.monthly_fee * (months - 1)


def simulate_lifecycle(
    eval_equity: np.ndarray,
    eval_days: np.ndarray,
    funded_equity: np.ndarray,
    funded_days: np.ndarray,
    rs: Ruleset,
    *,
    trading_days_per_month: int = 21,
    withdraw_at_profit: float = 0.0,
) -> LifecycleResult:
    """Buy one account and run it until it dies or the path runs out.

    Two paths go in because the eval and the funded account are two separate
    accounts with separate equity curves -- passing the eval does not carry
    your eval profit over.

    The funded stage runs as a sequence of payout cycles: trade until the
    payout gates open, withdraw, and start the next cycle from the reduced
    equity. Withdrawing is not free under a trailing threshold -- it moves you
    back down toward your floor -- which is exactly why it has to be modelled
    rather than assumed away.
    """
    fees = rs.eval_fee
    ev = simulate_stage(
        eval_equity, eval_days, rs,
        target_equity=rs.target_equity,
        opening_equity=rs.starting_balance,
    )

    if ev.outcome is not Outcome.PASSED:
        days = ev.stop_day + 1
        fees += _monthly_fees(rs, days, trading_days_per_month)
        return LifecycleResult(
            passed_eval=False,
            eval_result=ev,
            funded_result=None,
            death_reason=ev.outcome,
            days_used=days,
            fees_paid=fees,
        )

    fees += rs.activation_fee
    funded_equity = np.asarray(funded_equity, dtype=float)
    funded_days = np.asarray(funded_days, dtype=np.int64)

    payouts: list[float] = []
    payout_days: list[int] = []
    blocked_by: str | None = None
    cursor = 0
    offset = 0.0        # everything withdrawn so far
    opening = rs.starting_balance
    fn: StageResult | None = None
    death: Outcome | None = None
    funded_days_used = 0
    # Some firms snap the drawdown floor up once you have taken a payout.
    min_floor: float | None = None

    while cursor < funded_equity.size:
        seg_equity = funded_equity[cursor:] - offset
        seg_days = funded_days[cursor:]

        fn = simulate_stage(
            seg_equity, seg_days, rs, target_equity=None, min_floor=min_floor
        )
        died = fn.outcome is not Outcome.RAN_OUT_OF_PATH

        starts, ends = _day_boundaries(seg_days)
        prof = _daily_profit(seg_equity, seg_days, opening)
        pay_day, amount = first_payout_day(
            prof, opening, rs, withdraw_at_profit=withdraw_at_profit
        )

        # A payout only lands if the account was still alive at that day's close.
        pay_ok = pay_day >= 0 and ends[pay_day] <= fn.stop_index

        if not pay_ok:
            # Survived to here but never qualified -- record why, then stop.
            upto = fn.stop_index
            day_of_stop = int(np.searchsorted(ends, upto))
            blocked_by = _why_blocked(
                prof[: day_of_stop + 1], float(seg_equity[upto]), rs
            )
            funded_days_used += fn.stop_day + 1
            death = fn.outcome if died else None
            break

        payouts.append(amount)
        # Recorded before the counter advances: funded_days_used still holds
        # what earlier payout cycles consumed, and pay_day is the offset into
        # this one.
        payout_days.append(ev.stop_day + 1 + funded_days_used + pay_day)
        blocked_by = None
        funded_days_used += pay_day + 1
        if rs.payout_resets_floor_to is not None:
            min_floor = rs.payout_resets_floor_to

        if rs.max_payouts is not None and len(payouts) >= rs.max_payouts:
            break

        end_idx = ends[pay_day]
        offset += amount / rs.profit_split
        opening = float(seg_equity[end_idx]) - amount / rs.profit_split
        cursor += end_idx + 1

    total_days = ev.stop_day + 1 + funded_days_used
    fees += _monthly_fees(rs, total_days, trading_days_per_month)

    return LifecycleResult(
        passed_eval=True,
        eval_result=ev,
        funded_result=fn,
        payouts=payouts,
        payout_days=payout_days,
        death_reason=death,
        days_used=total_days,
        fees_paid=fees,
        payout_blocked_by=blocked_by,
    )
