# This directory is the only copy of paid market data

Do not delete these files. The Databento account that paid for them can no
longer afford to fetch them again — it returned `402 account_insufficient_funds`
partway through the original download.

| File | What it is |
|---|---|
| `NQ_1m_2016plus.parquet` | NQ continuous front month (volume roll), 1-minute bars |
| `ES_1m_2016plus.parquet` | ES equivalent — the sealed out-of-sample instrument |
| `NQ_rolls.csv` | Contract rolls with exact spreads from daily overlap |
| `ES_rolls.csv` | Contract rolls, boundary-estimated (see below) |

## What is and is not in here

**2016-01-01 onward only.** Earlier data exists at the vendor but is not usable:
RTH coverage runs 22–43% in 2010–2012 and 83–89% in 2013–2015, and the missing
sessions are seasonal — 92 of the 116 absent in 2013–2015 fall in November to
February. Measured in `findings/03_data_quality.md`.

**The holdout is included in the file and enforced at load time**
(`research/data/holdout.py`), not by leaving it out. Sealing by absence would
mean re-buying data to run the out-of-sample test, which is precisely the
pressure a holdout is supposed to resist.

**ES roll gaps are boundary estimates, not exact.** The daily parent data that
prices them exactly ran out of budget for ES. On NQ, where both methods are
available, they agree to within 1% (mean +53.1 vs +53.5), so the estimate is
sound — but if ES is ever unsealed for validation, `ES.FUT ohlcv-1d` costs
about $0.30 and is worth buying first.

## Rebuilding, if it ever comes to that

`research/data/databento_fetch.py` will rebuild everything from a funded
account. Costs at the time of writing:

| | |
|---|---|
| NQ `ohlcv-1m` 2010-06 → 2026-08 | $17.52 |
| ES `ohlcv-1m` same range | $17.93 |
| NQ `ohlcv-1d` parent (exact roll spreads) | $0.19 |
| ES `ohlcv-1d` parent | $0.30 |

Check the real balance in the Databento portal first. The documented $125 free
credit did not match what the account actually held.
