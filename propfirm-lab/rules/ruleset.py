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

    # --- consistency ------------------------------------------------------
    # No single day's profit may exceed this fraction of the total. Firms
    # apply it at two different points and not always with the same value:
    # `consistency_pct_eval` gates *passing the evaluation*, while
    # `consistency_pct` gates *withdrawing from a funded account*. Lucid Flex
    # is the case that forces them apart -- 50% to pass, nothing at all once
    # funded.
    consistency_pct_eval: float | None = None
    consistency_pct: float | None = None

    # --- funded-stage payout gates ----------------------------------------
    # Ceiling on a single withdrawal. Lucid pays min(50% of cycle profit,
    # $2,000); the half it keeps back stays in the account and pushes equity
    # further above the locked floor, so withdrawing actually makes the
    # account safer over time.
    payout_pct_of_profit: float | None = None
    payout_cap: float | None = None

    # Requesting a payout snaps the drawdown floor up to this absolute level.
    # On Lucid that is $50,100 regardless of where the trailing floor had got
    # to, which makes an early first payout genuinely expensive: take $500 at
    # $51,000 and you are left with $400 of room instead of $1,900.
    payout_resets_floor_to: float | None = None
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
        for name in ("consistency_pct", "consistency_pct_eval", "payout_pct_of_profit"):
            v = getattr(self, name)
            if v is not None and not 0 < v <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.payout_cap is not None and self.payout_cap <= 0:
            raise ValueError("payout_cap must be positive")
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
    consistency_pct_eval=0.50,
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
    consistency_pct_eval=0.50,
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
    consistency_pct_eval=0.40,
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


_LUCID_50K_FLEX = Ruleset(
    firm="Lucid",
    account="50k Flex",
    effective_date=date(2026, 8, 1),
    source="Lucid public rules + LucidFlex account pages, Aug 2026",
    starting_balance=50_000.0,
    profit_target=3_000.0,
    max_drawdown=2_000.0,
    drawdown_type=DrawdownType.EOD_TRAILING,
    trailing_lock_at=50_100.0,   # locks once a close clears 52,100
    daily_loss_limit=None,       # Flex is the Lucid account with no DLL anywhere
    consistency_pct_eval=0.50,   # to pass
    consistency_pct=None,        # ...and nothing at all once funded
    min_trading_days=0,          # a one-day pass is possible in principle
    qualifying_day_min_profit=0.0,
    safety_net=None,             # no buffer balance required
    min_payout=500.0,
    payout_pct_of_profit=0.50,
    payout_cap=2_000.0,
    payout_resets_floor_to=50_100.0,
    max_payouts=None,
    profit_split=0.90,
    eval_fee=105.0,              # promo price; see notes
    activation_fee=0.0,
    monthly_fee=0.0,
    notes=(
        "List price ~$149 (some sources ~$136); 30-40% promo codes are "
        "routine, so ~$85-105 is the realistic paid price. One-time fee: no "
        "activation, no rebills. Bots, EAs and trade copiers are explicitly "
        "permitted; HFT and hedging are not. ProjectX/LucidX support was "
        "discontinued in December 2025 -- automation goes through Rithmic or "
        "Tradovate. Two mechanics make this account unlike the others: a "
        "payout is capped at min(50% of cycle profit, $2,000), and requesting "
        "one snaps the max loss limit up to $50,100."
    ),
)

RULESETS: dict[str, Ruleset] = {
    "apex_50k_intraday": _APEX_50K_INTRADAY,
    "apex_50k_eod": _APEX_50K_INTRADAY.with_drawdown_type(DrawdownType.EOD_TRAILING),
    "topstep_50k": _TOPSTEP_50K,
    "mffu_50k": _MFFU_50K,
    "lucid_50k_flex": _LUCID_50K_FLEX,
}


def get_ruleset(key: str) -> Ruleset:
    try:
        return RULESETS[key]
    except KeyError:
        raise KeyError(f"unknown ruleset {key!r}; known: {sorted(RULESETS)}") from None


def list_rulesets() -> list[str]:
    return sorted(RULESETS)
