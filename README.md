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
      end-to-end against synthetic fixtures. **Blocked on real data** — needs
      the purchased ES/NQ 1-minute files in `ict_lab/data/raw/` (see
      prerequisites in the doc above) before `python -m
      ict_lab.data.build_clean_data` can produce a real quality report or
      cache. Two separate sessions have now confirmed this from the inside:
      neither a general-purpose sandbox nor a session explicitly provisioned
      for outbound network access could reach data vendor sites or download
      anything (see "Getting the Phase 1 data" below) — this step needs a
      human, outside any agent session, to buy/download the data and hand it
      back.
- [ ] Phase 2 — ICT feature detectors
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
why the spec lists it as an "already purchased" prerequisite. A coding session
can't complete this step on your behalf: buying data needs your own payment
action regardless, and (as of 2026-08-08) even a session specifically
provisioned with "trusted network access" found its actual egress fully
blocked by policy — outbound HTTPS was denied to every external host tried,
including inert ones like example.com and wikipedia.org, not just the data
vendors. Only a server-side web-search tool (which returns summarized
snippets, not raw pages, and can't download files) worked at all. So beyond
what search snippets could confirm below, none of this has been verified by
actually browsing the vendor sites, and no free sample was downloaded or
validated against the pipeline — if a future session has real outbound
network access, that's the first thing to check before attempting a
download.

Vendor details below reflect what could be confirmed via search as of
2026-08-08; treat pricing/terms as needing a final human check at purchase
time since they weren't loaded and read directly:

- **Kibot** — "All Futures Continuous Contracts 1-Minute Intraday Data": 83
  symbols back to 2009, adjusted + raw unadjusted columns in one CSV schema,
  **$520 one-time, no subscription** (confirmed). Kibot's general free-sample
  program (kibot.com/free_historical_data.aspx) is documented as covering the
  most recent ~3 months of 1-minute data for a couple of named equity/ETF
  symbols (IBM, OIH) plus daily EOD for all US stocks/ETFs — whether the
  *continuous futures* product page offers its own free ES/NQ sample distinct
  from that, as previously assumed, was **not confirmed**; check the product
  page directly before relying on a free futures sample existing. Simplest,
  most concrete paid option either way. kibot.com
- **Firstrate Data** — NQ continuous history confirmed back to **2008-01-02**
  (~15+ years), with an **"Absolute-Adjusted" (points-based)** adjustment
  option confirmed to exist — the additive adjustment this project's own
  no-percentage rule requires, as opposed to their ratio-adjusted variant.
  Pricing wasn't confirmed via search. firstratedata.com
- **Databento** — official CME data vendor (GLBX.MDP3). Search results were
  inconsistent/unclear on both pricing (mentions of a ~$179/mo "Standard"
  plan alongside usage-based pricing) and historical depth (one blog result
  referenced coverage "beginning September 25, 2025," which looks like it's
  about a specific newly-added event-contract dataset rather than the general
  OHLCV history — GLBX.MDP3 is understood to go back much further, but this
  needs a direct page read to confirm). Don't trust either figure without
  checking databento.com/datasets/GLBX.MDP3 directly. databento.com
- **CME DataMine** — the exchange's own first-party historical data platform.
  Confirmed: purchased online by credit card at cme.com/datamine, one-time or
  1/12-month subscription per dataset, API-based delivery requiring a CME
  Group Login + API ID; pricing varies by dataset/duration and isn't posted
  publicly (contact dataminesales@cmegroup.com for a quote). More
  institutional/higher setup overhead than the others. cmegroup.com/datamine

Whichever you pick, the 6 files need to land in `ict_lab/data/raw/` matching
the exact schema in Phase 1 of the spec (index = tz-aware UTC timestamps;
columns `open, high, low, close, volume, contract`; roll log columns `date,
old_contract, new_contract, price_adjustment`) — upload them here (or drop
them in directly if running locally) and Phase 1 is ready to run for real.

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
