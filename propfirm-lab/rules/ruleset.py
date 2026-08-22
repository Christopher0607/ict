"""Prop-firm account rules, encoded so a simulator can execute them.

Every number in here is a *rule*, not a preference. They are grouped into the
two stages an account actually passes through:

  eval stage   -- buy it, hit a profit target without breaching a floor
  funded stage -- keep it alive long enough, and consistently enough, to be
                  allowed to withdraw

Most retail modelling stops at the eval stage. That is the cheap half. The
funded stage is where the money actually is and where the extra rules
(consistency, qualifying days, safety net, payout caps) bite.

Sources are dated because these rules move. Confirm against your own account
dashboard before trusting any of it with real money -- see README.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class DrawdownType(str, Enum):
    """How the account's kill-floor is computed.

    STATIC            floor is fixed at ``starting_balance - max_drawdown``
    EOD_TRAILING      floor trails the highest *end-of-day* equity
    INTRADAY_TRAILING floor trails the highest *intraday, unrealized* equity

    The third one is the dangerous one and the one people model wrong: a trade
    that runs +$800 in your favour and comes back to breakeven has permanently
    raised your floor by $800. You paid for that excursion without booking it.
    """

    STATIC = "static"
    EOD_TRAILING = "eod_trailing"
    INTRADAY_TRAILING = "intraday_trailing"


@dataclass(frozen=True)
class Ruleset:
    """One firm's rules for one account size, at one point in time."""

    firm: str
    account: str
    effective_date: date
    source: str

    # --- sizing -----------------------------------------------------------
    starting_balance: float
    profit_target: float
    max_drawdown: float
    drawdown_type: DrawdownType

    # Absolute equity level at which a trailing floor stops trailing.
    # Apex locks the threshold once it reaches starting_balance + $100.
    # None means "trails forever".
    trailing_lock_at: float | None = None

    # Hard intraday loss cap measured from the day's starting equity.
    daily_loss_limit: float | None = None

    # --- funded-stage payout gates ---------------------------------------
    # No single day's profit may exceed this fraction of total profit since
    # the last payout. 0.50 => a day worth more than half your profit blocks
    # the withdrawal until you trade more days.
    consistency_pct: float | None = None
    # Days with at least `qualifying_day_min_profit` of profit.
    min_trading_days: int = 0
    qualifying_day_min_profit: float = 0.0
    # Equity you must hold *above* the starting balance to withdraw at all.
    safety_net: float | None = None
    min_payout: float = 0.0
    max_payouts: int | None = None
    profit_split: float = 1.0

    # --- costs ------------------------------------------------------------
    eval_fee: float = 0.0
    activation_fee: float = 0.0
    monthly_fee: float = 0.0

    # --- notes ------------------------------------------------------------
    notes: str = ""

    def __post_init__(self) -> None:
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        if self.profit_target <= 0:
            raise ValueError("profit_target must be positive")
        if self.max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive")
        if self.consistency_pct is not None and not 0 < self.consistency_pct <= 1:
            raise ValueError("consistency_pct must be in (0, 1]")
        if not 0 < self.profit_split <= 1:
            raise ValueError("profit_split must be in (0, 1]")
        if self.drawdown_type is DrawdownType.STATIC and self.trailing_lock_at is not None:
            raise ValueError("a static drawdown cannot have a trailing lock")

    # -- derived levels ----------------------------------------------------

    @property
    def target_equity(self) -> float:
        """Equity that clears the eval stage."""
        return self.starting_balance + self.profit_target

    @property
    def initial_floor(self) -> float:
        """The kill-floor before any profit has been made."""
        return self.starting_balance - self.max_drawdown

    @property
    def safety_net_equity(self) -> float | None:
        if self.safety_net is None:
            return None
        return self.starting_balance + self.safety_net

    @property
    def upfront_cost(self) -> float:
        """What it costs to get one account to the starting line."""
        return self.eval_fee + self.activation_fee

    def with_drawdown_type(self, dd: DrawdownType) -> "Ruleset":
        """Same account, different floor rule -- for apples-to-apples comparison.

        Used by ``findings/01`` to isolate how much of the pass rate is the
        rule and how much is the trader.
        """
        lock = None if dd is DrawdownType.STATIC else self.trailing_lock_at
        return replace(self, drawdown_type=dd, trailing_lock_at=lock)

    def __str__(self) -> str:
        return f"{self.firm} {self.account} ({self.drawdown_type.value})"


# ---------------------------------------------------------------------------
# The rulesets themselves.
#
# VERIFY THESE AGAINST YOUR ACCOUNT DASHBOARD. They were compiled from public
# pages in August 2026 and prop-firm terms change without much warning.
# ---------------------------------------------------------------------------

_APEX_50K_INTRADAY = Ruleset(
    firm="Apex",
    account="50k",
    effective_date=date(2026, 3, 1),
    source="Apex 4.0 overhaul, March 2026; payout rules as of Aug 2026",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_drawdown=2_500.0,
    drawdown_type=DrawdownType.INTRADAY_TRAILING,
    trailing_lock_at=50_100.0,  # start + $100, then frozen
    daily_loss_limit=None,      # Apex has no daily loss limit
    consistency_pct=0.50,       # tightened from 30% to 50% in the 4.0 overhaul
    min_trading_days=5,
    qualifying_day_min_profit=50.0,
    safety_net=2_600.0,
    min_payout=500.0,
    max_payouts=6,
    profit_split=1.0,
    eval_fee=131.0,             # list price; routinely discounted 60-90%
    activation_fee=79.0,
    monthly_fee=0.0,            # one-time-payment model since March 2026
    notes=(
        "Trailing threshold follows unrealized intraday peak, then locks at "
        "start+$100. No daily loss limit, which makes the trailing floor the "
        "only thing standing between you and a blown account."
    ),
)

_TOPSTEP_50K = Ruleset(
    firm="TopStep",
    account="50k",
    effective_date=date(2026, 8, 1),
    source="Topstep public rules + TopstepX API docs, Aug 2026",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_drawdown=2_000.0,
    drawdown_type=DrawdownType.EOD_TRAILING,
    trailing_lock_at=50_000.0,  # stops trailing once it reaches the start balance
    daily_loss_limit=1_000.0,
    consistency_pct=0.50,
    min_trading_days=5,
    qualifying_day_min_profit=200.0,
    safety_net=None,
    min_payout=0.0,
    max_payouts=None,
    profit_split=1.0,
    eval_fee=49.0,              # per month
    activation_fee=149.0,
    monthly_fee=49.0,
    notes=(
        "The automation-friendly one: official TopstepX API on ProjectX "
        "Gateway. EOD trailing plus a hard $1k daily loss limit -- a gentler "
        "floor but an extra way to die."
    ),
)

_MFFU_50K = Ruleset(
    firm="MyFundedFutures",
    account="50k Core",
    effective_date=date(2026, 8, 1),
    source="MFFU Core plan public pricing, Aug 2026",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_drawdown=2_000.0,
    drawdown_type=DrawdownType.EOD_TRAILING,
    trailing_lock_at=50_000.0,
    daily_loss_limit=1_000.0,
    consistency_pct=0.40,
    min_trading_days=5,
    qualifying_day_min_profit=0.0,
    safety_net=None,
    min_payout=0.0,
    max_payouts=None,
    profit_split=1.0,
    eval_fee=77.0,              # per month
    activation_fee=0.0,
    monthly_fee=77.0,
    notes="Allowed algorithmic trading from July 2025. No activation fee.",
)

RULESETS: dict[str, Ruleset] = {
    "apex_50k_intraday": _APEX_50K_INTRADAY,
    "apex_50k_eod": _APEX_50K_INTRADAY.with_drawdown_type(DrawdownType.EOD_TRAILING),
    "topstep_50k": _TOPSTEP_50K,
    "mffu_50k": _MFFU_50K,
}


def get_ruleset(key: str) -> Ruleset:
    try:
        return RULESETS[key]
    except KeyError:
        raise KeyError(f"unknown ruleset {key!r}; known: {sorted(RULESETS)}") from None


def list_rulesets() -> list[str]:
    return sorted(RULESETS)
