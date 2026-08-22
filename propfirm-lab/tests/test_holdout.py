"""The holdout guard has to actually stop things, or it is decoration."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research.data import holdout
from research.data.holdout import HOLDOUT_START, HoldoutViolation


@pytest.fixture(autouse=True)
def _always_resealed():
    holdout.reseal()
    yield
    holdout.reseal()


def test_es_is_sealed():
    with pytest.raises(HoldoutViolation, match="out-of-sample instrument"):
        holdout.check_symbol("ES.v.0")


def test_nq_is_not_sealed():
    holdout.check_symbol("NQ.v.0")  # must not raise


def test_range_crossing_the_cutoff_is_refused():
    with pytest.raises(HoldoutViolation, match="crosses into the holdout"):
        holdout.check_range(date(2020, 1, 1), date(2026, 8, 22))


def test_range_ending_on_the_cutoff_is_allowed():
    holdout.check_range(date(2010, 6, 6), HOLDOUT_START)


def test_dev_slice_trims_to_the_cutoff():
    df = pd.DataFrame({"ts_open": pd.to_datetime(
        ["2024-08-21T00:00Z", "2024-08-22T00:00Z", "2025-01-01T00:00Z"], utc=True)})
    out = holdout.dev_slice(df)
    assert len(out) == 1
    assert out.loc[0, "ts_open"] == pd.Timestamp("2024-08-21T00:00Z")


def test_unseal_requires_a_written_reason():
    with pytest.raises(ValueError, match="real written reason"):
        holdout.unseal("because")


def test_unseal_opens_everything_and_reseal_closes_it():
    holdout.unseal("Phase C search is complete and its thresholds are locked; "
                   "running the single out-of-sample validation now.")
    assert not holdout.is_sealed()
    holdout.check_symbol("ES.v.0")
    holdout.check_range(date(2010, 1, 1), date(2026, 8, 22))

    holdout.reseal()
    assert holdout.is_sealed()
    with pytest.raises(HoldoutViolation):
        holdout.check_symbol("ES.v.0")
