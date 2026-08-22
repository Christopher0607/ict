"""The sealed data. Enforced in code, not by good intentions.

Two things are off limits until the strategy search is finished and its
thresholds are locked:

* **The last two years of NQ** -- the out-of-sample window, run exactly once.
* **All of ES** -- the out-of-sample *instrument*. The sibling ICT lab's
  methodology requires surviving configurations to be re-validated on a symbol
  that played no part in selecting them, with no going back to re-pick if the
  result disappoints.

ES was bought at the same time as NQ specifically so that this check could
exist from day one. Buying it later would have meant arriving at the
validation step with a purchase decision standing between us and an
inconvenient answer.

Unsealing is a deliberate, reviewable act: call ``unseal`` with an explicit
reason, in a script that is not part of the search.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# Data runs to 2026-08-22; the last two years are the exam paper.
HOLDOUT_START = date(2024, 8, 22)

SEALED_SYMBOLS = {"ES.v.0"}

_unsealed = False
_unseal_reason: str | None = None


class HoldoutViolation(RuntimeError):
    """Raised when development code reaches for data it must not see."""


def unseal(reason: str) -> None:
    """Open the holdout. Use once, from a dedicated validation script."""
    global _unsealed, _unseal_reason
    if not reason or len(reason.strip()) < 20:
        raise ValueError(
            "unseal() needs a real written reason (>= 20 chars) -- it goes in "
            "the run log so the decision is auditable afterwards."
        )
    _unsealed = True
    _unseal_reason = reason.strip()
    print(f"[HOLDOUT UNSEALED] {_unseal_reason}")


def reseal() -> None:
    global _unsealed, _unseal_reason
    _unsealed = False
    _unseal_reason = None


def is_sealed() -> bool:
    return not _unsealed


def check_symbol(symbol: str) -> None:
    if symbol in SEALED_SYMBOLS and not _unsealed:
        raise HoldoutViolation(
            f"{symbol} is the out-of-sample instrument and is sealed. It exists "
            f"to validate configurations that were chosen without it. Call "
            f"unseal(reason) from a validation script if that step has been "
            f"reached."
        )


def check_range(start: date, end: date) -> None:
    if end > HOLDOUT_START and not _unsealed:
        raise HoldoutViolation(
            f"requested data through {end}, which crosses into the holdout "
            f"beginning {HOLDOUT_START}. Development must stop at that date."
        )


def dev_slice(df: pd.DataFrame, ts_col: str = "ts_open") -> pd.DataFrame:
    """Trim a frame to the development window. Safe to call unconditionally."""
    if _unsealed:
        return df
    ts = pd.to_datetime(df[ts_col], utc=True)
    cutoff = pd.Timestamp(HOLDOUT_START, tz="UTC")
    return df.loc[ts < cutoff].reset_index(drop=True)
