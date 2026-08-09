# ict

Backtesting lab for the ICT Silver Bullet strategy on CME futures (ES, NQ), gated
toward a monitored live-trading deployment on Gate's `NAS100_USDT` perpetual
contract if — and only if — it survives a holdout-tested statistical study.

The full methodology and phase-by-phase build plan lives in
[`docs/ICT_SilverBullet_Gate_NAS100USDT_Prompt.md`](docs/ICT_SilverBullet_Gate_NAS100USDT_Prompt.md).
Read it before touching this repo. The short version:

- Phases are built **one at a time**, in order, each one reviewed before the next starts.
- Several phases end in a **STOP** or **CHECKPOINT** that requires a human read of the
  output before continuing.
- The last 2 years of data are a **holdout**: untouched until Phase 8, run exactly once.
- Live order placement (Phase 9) stays behind an explicit, manually-typed confirmation
  gate — no code path may flip `LIVE_TRADING_ENABLED` or place a real order on its own.
- No calculation anywhere uses a percentage of price or a price ratio — only points,
  ticks, or R multiples.

## Status

- [x] Phase 0 — project structure, git, venv, requirements
- [~] Phase 1 — data and sessions: code-complete, 24 tests passing, verified
      end-to-end against synthetic fixtures **and now also against real
      market data** — a session with working internet access (2026-08-09)
      downloaded FirstRate Data's free ~2.5-week ES/NQ futures sample,
      converted it into the exact raw-file format `ict_lab/data/loader.py`
      expects, and ran `python -m ict_lab.data.build_clean_data` for real:
      DST-safe UTC→ET conversion, session/killzone labeling, the data-quality
      report, and partitioned parquet caching all ran clean against real
      1-minute CME bars with zero errors. See "Getting the Phase 1 data"
      below for exactly what that did and did not prove. **Still blocked on
      the actual purchase** — Phases 2 onward (and any real statistical work
      in Phase 5+) need the full 15+ year, purchased ES/NQ history in
      `ict_lab/data/raw/`, which needs a human to buy and hand back; a free
      sample can validate the pipeline but is far too short and too recent to
      be real research data (it's entirely inside the Phase 1 holdout window
      by definition).
- [~] Phase 2 — ICT feature detectors: code-complete, 94 tests passing
      (up from 24), including a dedicated no-lookahead harness
      (`tests/test_no_lookahead.py`) that runs every detector on full data
      vs. data truncated at a cutoff bar and asserts the outputs agree on
      everything knowable by that cutoff. This harness caught and drove the
      fix for several real bugs — most notably, multiple places treating
      "the last bar we've seen for a session" as proof that session had
      ended, when a session can still have bars left to print. Covers: FVG
      (`features/fvg.py`, mitigation/fill tracked at 1m resolution),
      N-bar fractal swings (`features/swings.py`), liquidity levels
      (`features/liquidity.py`: prior session/RTH, pre-window, swing-based,
      each gated on a time-based "is this period provably over" check, not
      just "does a later grouping happen to exist yet"), sweeps
      (`features/sweep.py`), displacement (`features/displacement.py`:
      ATR-multiple and percentile versions), MSS
      (`features/mss.py`, gated on already-confirmed swings only), HTF bias
      (`features/bias.py`: none / prior-day / 15m-or-1h swing structure /
      daily-MA-slope / perfect — the last one deliberately look-ahead and
      flagged as such everywhere it appears), and a candlestick +
      feature-overlay plotting function (`features/plotting.py`). The
      "generate 15 random-day charts for visual review" step
      (`generate_review_charts`) is wired up and tested structurally, but
      there's nothing a human would recognize in synthetic random-walk
      bars — running it for real still needs the Phase 1 data.
- [ ] Phase 3 — execution engine
- [ ] Phase 4 — taught configs, frequency, verification
- [ ] Phase 5 — parameter sweep, nulls, statistics
- [ ] Phase 6 — results dashboard
- [ ] Phase 7 — reporting pass
- [ ] Phase 8 — holdout (run once)
- [ ] Phase 9 — Gate `NAS100_USDT` deployment (blocked until Phase 8 passes its
      go/no-go gate)

## Getting the Phase 1 data

15+ years of clean, back-adjusted, roll-logged 1-minute CME futures data is a
licensed commercial product, not something scrapable off the open web — that's
why the spec lists it as an "already purchased" prerequisite. Two earlier
sessions couldn't get past this because their sandboxes had no real outbound
network access at all (blocked even on inert hosts like example.com). A
session on 2026-08-09 did have real access, confirmed it (`curl` to
example.com, wikipedia.org, and every vendor below all returned 200), and
used it to actually browse each vendor site, verify current pricing/terms
directly, and — where a genuinely free, no-payment sample existed — download
and validate it against this repo's pipeline. **No purchase was made or
attempted; buying the full history is still a human action.**

- **Kibot** (kibot.com) — "All Futures Continuous Contracts 1-Minute
  Intraday Data": **$520 one-time, no subscription**, 83 futures contracts,
  history back to 2009 (all confirmed directly on kibot.com/buy.html and the
  free-sample page). Two corrections to earlier notes:
  - **No free futures/forex sample exists.** Kibot's free-sample program
    (kibot.com/free_historical_data.aspx) is US equities/ETFs only (IBM, OIH,
    IVE, WDC) — the site says so explicitly ("Futures and forex samples are
    available on request through the custom-order builder"), and that
    builder is a paid quote-and-checkout tool (FastSpring), not a free
    download.
  - **The futures product is unadjusted-only.** Per
    kibot.com/futures/continuous-futures.html, Kibot's continuous series has
    "no back-adjustment: prices are the actual traded prices from each
    contract" — every rollover date is published, and the customer is
    expected to build their own adjusted series on top. Kibot's own docs name
    this the **"Panama Canal (Constant) Adjustment"** — add/subtract the
    absolute roll-date price gap from all prior bars — which is exactly this
    project's additive, no-ratio rule, so it's usable, but it means the $520
    purchase gives you the unadjusted file + published roll dates, not a
    ready-made `_backadjusted.parquet`; that still has to be derived.
- **Firstrate Data** (firstratedata.com) — confirmed directly on the ES and
  NQ product pages: continuous history for both starts **2008-01-02**,
  individual contracts from the Dec-2008 contract. Each continuous dataset
  ships in **three pre-built series: Unadjusted, Absolute-Adjusted
  (points-based, additive — exactly this project's rule, delivered ready to
  use, no derivation needed), and Ratio-Adjusted.** Pricing: ES alone is
  $215.95 one-time; the "Futures - Most Active (130 Most Active Futures)"
  bundle (includes both ES and NQ plus 128 others) is $539.95 one-time, 1
  month of free daily updates then $59.95/mo optional. **A genuine 2-week
  free sample exists per ticker, no login or payment** — "Download Sample" on
  each product page links straight to a public S3 file
  (`frd001.s3.us-east-2.amazonaws.com/frd_sample_futures_{ES,NQ}.zip`); this
  is what got downloaded and validated below.
- **Databento** (databento.com) — official CME data vendor (GLBX.MDP3),
  confirmed available **from 2010-06-06**. Pricing confirmed on
  databento.com/pricing: usage-based pay-per-GB with $125 in free credits
  (expire after 6 months), or a **Standard subscription at $199/mo** covering
  unlimited OHLCV-1m (and 1s/1h/1d) for CME's entire history — the schema
  this project needs — plus a $1,750/mo "Plus" and $4,500/mo "Unlimited" tier
  for order-book depth this project doesn't need. No free sample checked
  (their model is usage-based credits, not a static sample file).
- **CME DataMine** (cmegroup.com/datamine) — the exchange's own platform,
  50+ dataset categories / 5,000+ products, earliest dataset back to 1972
  (that figure is platform-wide, not specific to ES/NQ 1-minute bars).
  Confirmed: self-service catalog at datamine.new.cmegroup.com requires
  account creation and a per-dataset license agreement; one-time vs.
  subscription order options both exist. Pricing is still not posted
  publicly on any page reached this session — the FAQ page URL from earlier
  notes now 404s, so the dataminesales@cmegroup.com contact could not be
  re-confirmed directly; treat that detail as unverified rather than
  corrected.

Whichever you pick, the 6 files need to land in `ict_lab/data/raw/` matching
the exact schema in Phase 1 of the spec (index = tz-aware UTC timestamps;
columns `open, high, low, close, volume, contract`; roll log columns `date,
old_contract, new_contract, price_adjustment`) — upload them here (or drop
them in directly if running locally) and Phase 1 is ready to run for real.

### Free-sample pipeline validation (2026-08-09)

FirstRate Data's free NQ and ES samples were downloaded, converted with
[`scripts/convert_firstrate_sample.py`](scripts/convert_firstrate_sample.py)
into the loader's exact 3-file-per-symbol format, dropped into
`ict_lab/data/raw/` (never committed — still gitignored), and run for real
through `python -m ict_lab.data.build_clean_data`. Result: all 24 existing
tests still pass, and the full pipeline — DST-safe UTC→US/Eastern conversion,
CME session/killzone labeling, the data-quality report, and partitioned
parquet caching — completed with zero errors against real 1-minute CME bars.
A spot-check of raw bars during a real NY AM killzone window
(`python -m ict_lab.data.verify NQ "2026-07-27 14:00" "2026-07-27 14:10"
--include-holdout`) showed a plausible volume spike at the 10:00 ET open with
sane OHLC values.

**What this does and does not prove.** This is a pipeline-plumbing check, not
research data, for three compounding reasons:
1. The free sample is only ~2.5 weeks (2026-07-23 to 2026-08-07), nowhere
   close to the thousands of trades and multi-year holdout Phases 5–8 need.
2. It's *entirely inside* the last-2-years holdout window by construction
   (it's this week's data) — `load_symbol()`'s default `include_holdout=False`
   correctly returns empty on it, which is itself a real, positive
   confirmation that the holdout guard works on real timestamps, but means
   every command above needed an explicit `--include-holdout` /
   `include_holdout=True` override to see anything at all.
   `build_clean_data.py` already does this by design (it's documented as the
   one authorized place allowed to touch holdout years); nothing else in the
   pipeline was changed to bypass the guard, and no actual research must ever
   use this override.
3. The sample window contains no contract roll (next NQ/ES roll is ~mid
   September 2026), so the converted `_backadjusted.parquet` is byte-identical
   to `_unadjusted.parquet` — the additive-adjustment math itself was not
   exercised, only the file-format/loader contract. The `contract` column
   value (`NQU26`/`ESU26`) is inferred from CME's public quarterly roll
   calendar, not read off vendor data, since the free sample has no
   individual-contract file to confirm it against.

In short: the raw-file schema, loader assertions, timezone/session/killzone
logic, quality report, and parquet cache all now have a real-data check
behind them, not just synthetic fixtures — but this in no way substitutes for
the actual purchase, which is still required before Phase 1 can be trusted
for anything past this plumbing check.

## Project layout

```
ict_lab/
  data/raw/        purchased 1-minute ES/NQ bars (gitignored — not committed)
  data/clean/      processed parquet cache (gitignored — regenerated, not committed)
  features/        ICT concept detectors (FVG, sweeps, displacement, MSS, bias)
  engine/          signal pipeline + execution/fill simulation
  configs/         grid.json and named strategy configs
  runs/            backtest run outputs (gitignored)
  analysis/        sweep results, dashboards, summary pack
  live/            Gate NAS100_USDT deployment code — intentionally empty until
                   Phase 9 clears the Phase 8 holdout gate
  logs/            runtime logs (gitignored)
docs/              the full phase-by-phase spec this project follows
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rules that hold across every phase

- No price percentages or price ratios anywhere — points, ticks, or R multiples only.
- No look-ahead bias, except the explicitly-labeled `perfect` bias control.
- Frequency/coverage tuning happens blind to PnL.
- The holdout is off-limits until Phase 8, run once, and the result stands as-is.
- Survivor configs get validated out-of-sample on ES with no reselection.
- Gate API keys live only in environment variables (`GATE_API_KEY`, `GATE_API_SECRET`)
  — never in a config file, log, or commit.
